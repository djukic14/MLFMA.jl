# #121: is λ/4 the right leaf size, now that I2O translation is 70 % of the far matvec?
#
# Translation cost is (translating pairs) x (grid size). Grid size is closed (#114-#119: ~16 %
# available, spent on the interpolation penalty) and antipodal symmetry is closed (#110, it cannot
# reach translation at all). What is left is the PAIR COUNT, which is a function of tree shape, and
# `minhalfsize` is the exposed knob -- pinned at λ/4 for every number measured so far.
#
# THE FIXTURE HAS A NEAR FIELD, and that is the whole design point. Every #120 number used
# `DummyNearInteractionsStrategy`, which prices only one side of the trade: larger leaves look free
# and the sweep runs away to a trivial tree. Near and far are reported separately AND totalled.
#
# IT SWEEPS λ/4 UPWARD, NOT DOWNWARD. Below ~λ/4 HALFSIZE MLFMA suffers low-frequency breakdown --
# the addition theorem's propagating-mode expansion stops representing the interaction and the error
# stops tracking ϵ at any sampling rate. `MINHALFSIZERECOM = 0.25` is that floor and
# `MLFMA.checkboxsizes` warns at it. Sub-floor points do not measure the trade; they measure the
# method failing, which is what an earlier version of this study would have done.
#
# IT RUNS BOTH WAYS ON PROTRUSION (#126). `protrusionaware=true` sizes grids from the box that
# actually contains the basis functions rather than the nominal one. Protrusion grows as boxes
# shrink, so a protrusion-blind sweep penalises small leaves for an under-sampling artifact instead
# of measuring tree shape. Running both settles #126's default from the same data, entirely inside
# the valid regime -- which no measurement of it so far has been.
#
# ACCURACY IS PART OF THE ANSWER, NOT A FOOTNOTE. Same geometry and same ϵ, differing only in tree
# shape, has already produced far-only 9.4e-5 against 2.8e-3. A sweep reporting only time will pick
# an operating point that does not meet ϵ. The error column is also the data #127 needs to re-derive
# the suite's tolerances at valid box sizes.
#
# OUTPUT IS FLUSHED PER ROW on purpose: this runs for hours, and a buffered log that only appears at
# the end is useless for deciding whether to keep waiting.

using MLFMA
using BEAST, CompScienceMeshes, H2Trees, H2Factory
using LinearAlgebra, Printf, Random, SparseArrays

const HT = H2Trees

function physicalconstants(f::U) where {U}
    μ0 = U(4 * π * 1e-7)
    ε0 = U(8.854187812e-12)
    c = U(1) / sqrt(ε0 * μ0)
    λ = c / f
    κ = U(2 * π / λ)
    return (; ω=κ * c, μ=μ0, λ, κ, γ=im * κ)
end

best(f; samples=3) = (f(); minimum(@elapsed(f()) for _ in 1:samples))

function sweep(;
    frequency=5e7,
    radius=2.0,
    meshfactor=6,
    halfsizes=(0.25, 0.375, 0.5, 0.75, 1.0),   # in λ; the FLOOR is 0.25, so this runs upward
    ϵ=1e-4,
)
    p = physicalconstants(frequency)
    λ, γ = p.λ, p.γ

    X = raviartthomas(meshsphere(radius * λ, λ / meshfactor))
    efie = MLFMA.efie(; wavenumber=p.κ, alpha=(-im * p.ω * p.μ))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)

    @printf(
        "N = %d   sphere %.1fλ   mesh λ/%d   ϵ = %.0e\n",
        numfunctions(X),
        radius,
        meshfactor,
        ϵ
    )
    println("assembling the dense reference once (this is the slow part)...")
    flush(stdout)
    t0 = time()
    Adense = BEAST.assemble(efie, X, X; quadstrat=quadstrat)
    @printf("dense assembled in %.1f s\n\n", time() - t0)
    flush(stdout)

    Random.seed!(23)
    x = randn(ComplexF64, numfunctions(X))
    y = similar(x)

    println(
        "  hs/λ  prot  levels  leaves     nnz(near)   build     near mv    far mv   total mv    far err    full err    map MB",
    )
    flush(stdout)

    for hs in halfsizes, aware in (false, true)
        tree = HT.TwoNTree(X; builder=HT.TwoNTreeBuilder(; minhalfsize=hs * λ, minvalues=0))
        leaves = collect(HT.leaves(tree))
        leaflevels = sort(unique(HT.level(tree, l) for l in leaves))

        strategy = MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ, protrusionaware=aware),
            nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
        )

        local A
        buildtime = try
            @elapsed A = MLFMA.GalerkinMLFMA(
                efie, X; tree=tree, strategy=strategy, verbose=false
            )
        catch e
            @printf(
                "%6.3f  %5s  -- construction failed: %s\n",
                hs,
                aware,
                first(sprint(showerror, e), 70)
            )
            flush(stdout)
            continue
        end

        near = H2Factory.nearinteractions(A)
        nnznear = nnz(SparseArrays.sparse(near))

        tnear = best(() -> H2Factory.nearmul!(y, A, x))
        tfar = best(() -> H2Factory.farmul!(y, A, x))
        ttotal = best(() -> LinearAlgebra.mul!(y, A, x))

        reference = Adense * x
        farreference = reference - near * x
        H2Factory.farmul!(y, A, x)
        farerror = norm(y - farreference) / norm(farreference)
        fullerror = norm(A * x - reference) / norm(reference)

        mapbytes =
            (
                Base.summarysize(H2Factory.trialmomentcollections(A)) +
                Base.summarysize(MLFMA.translationcollection(A))
            ) / 2^20

        @printf(
            "%6.3f  %5s  %6s  %6d  %11d  %7.1f  %8.4f  %8.4f  %9.4f  %.3e  %.3e  %8.1f\n",
            hs,
            aware,
            string(leaflevels),
            length(leaves),
            nnznear,
            buildtime,
            tnear,
            tfar,
            ttotal,
            farerror,
            fullerror,
            mapbytes
        )
        flush(stdout)

        A = nothing
        GC.gc()
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    sweep()
end
