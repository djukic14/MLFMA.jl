# What sets the SHE order P? Calibrating a RULE, across box size and tolerance.
#
# `sheaccuracy.jl` establishes P = 10 for one box (λ/4) at one tolerance (1e-4). That is an operating
# point, not a rule, and a constant would be wrong for two reasons: `minhalfsize` is an exposed knob
# (#121 sweeps it), and a non-uniform tree has different box sizes per level. This file measures P
# across both axes and calibrates a rule that predicts it.
#
# ============================== P AND L ARE DIFFERENT QUANTITIES ==============================
# L truncates the TRANSLATION OPERATOR and is set by the box DIAGONAL:
#     diameter(halfsize) = 2√3·halfsize                              (utilsmaps.jl:378)
#     L = floor(κd + 1.8·digits^(2/3)·(κd)^(1/3))                    (utilsmaps.jl:357)
# and it fixes the grid, Nθ x Nφ = 2(L+1) x 4(L+1).
#
# P is the harmonic content of ONE BASIS FUNCTION's pattern about its own box centre --
# `exp(-γ k̂·(r-c))` over that function's support -- so it depends on a RADIUS, `max |r-c|`, not on
# box-to-box geometry. Bounded by the box, not determined by it: a function at the corner reaches
# √3·halfsize, one near the centre far less, and under `NoProtrusionCheck()` a support may protrude
# past the box entirely.
#
# SO P IS NOT A FUNCTION OF L -- but both scale with κ·(box size), so they move together, and the
# compression has a closed form:
#
#     today  2 tangential components x 8(L+1)² samples      = 16(L+1)²
#     SHE    3 Cartesian components  x  (P+1)² coefficients =  3(P+1)²
#     compression = (16/3)·((L+1)/(P+1))²
#
# The important consequence: EVEN AT P = L THE SCHEME WINS 16/3 = 5.33x, because a band-limited
# function on the sphere needs (L+1)² coefficients while the product grid spends 8(L+1)² samples. The
# P < L margin is a bonus on top (1.62x at L=13, P=10, giving the measured 8.64x). SHE therefore does
# not depend on basis functions being small -- which is what makes it robust where adaptive grouping
# (#122 item c) was not.
#
# ============================== METHOD ==============================
# For each (box size, ε): build the map, overwrite every leaf collection with its SHE round trip at
# order P (see `sheaccuracy.jl` -- same eager simulation of the wired-in pipeline), and find the
# SMALLEST P whose far-only error stays within `tolerance` of the untouched baseline. Then check what
# the two candidate rules would have predicted:
#
#   GEOMETRIC  P = cutoffparameter(γ, ε, 2·max|r-c|) -- needs the per-box support extent
#   SPECTRAL   P = truncationorder(coefficients, τ)  -- reads the MEASURED coefficient tail, is free
#              (the coefficients exist at setup anyway), and handles protrusion and partly-filled
#              boxes with no geometric bookkeeping. `τ` is what this file calibrates.
#
# The spectral rule must be applied as a MAX OVER THE LEVEL's boxes, not per box: the batched
# synthesis needs one P per level or the GEMM becomes ragged (Eibert §7).
#
# WHY τ IS NOT SIMPLY ε: the ε = 1e-4 tail criterion picks P = 9 at λ/4, and `sheaccuracy.jl` shows
# P = 9 costs 15 % of the far error. Per-pattern errors do not take their maximum when summed into a
# group moment, so the tail threshold has to be tighter than the map's tolerance. How much tighter is
# the number this file exists to produce.
#
# ****************************** CORRECTION, 2026-08-25 ******************************
# SIX OF THE NINE CONFIGURATIONS BELOW ARE OUTSIDE MLFMA'S VALID REGIME and are not evidence.
# MLFMA suffers LOW-FREQUENCY BREAKDOWN below roughly λ/4 HALFSIZE -- the addition theorem's
# propagating-mode expansion stops representing the interaction and the far-field error stops
# tracking ε at any sampling rate. `MINHALFSIZERECOM = 0.25` is that floor and
# `MLFMA.checkboxsizes` now warns at construction. The `0.2λ` and `0.1λ` rows are below it.
#
# RE-RUN RESTRICTED TO VALID BOX SIZES, the rule still holds exactly:
#
#   halfsize/λ    ε     L    NΩ  leaves  baseline    P*   compression   spectral P at τ/ε = 1, .1, .01
#       0.4     1e-03   16  2312     32  9.296e-06   12       9.12x         9   10   12
#       0.4     1e-04   18  2888     32  6.763e-06   13       9.82x        10   12   13
#       0.4     1e-06   20  3528     32  6.718e-06   13      12.00x        13   14   26
#
# τ = max(ε/100, 1e-7) predicts 12, 13 and (floored to ε/10 at 1e-6) 14 against P* of 12, 13, 13 --
# never under-predicting, over-predicting by at most 1. So the RULE is unharmed. What is harmed is
# the CLAIM ABOUT ITS SUPPORT: it is calibrated on three configurations at ONE box size, not nine
# across three. Widening the box-size axis needs a larger geometry -- a 2λ sphere reaches halfsizes
# 1.0λ / 0.5λ / 0.25λ -- and a dense reference at that size is expensive. Until that is run, treat
# the box-size dependence of the rule as UNMEASURED rather than as measured over 5.3x-12x.
#
# The sub-floor rows are left in place below because they are still the record of what was run, and
# because the P*/L and 16/3-floor observations drawn from them are geometry, not MLFMA accuracy.
# Anything about ERROR read off them is not usable.
# ***********************************************************************************
#
# ================== RESULTS, sphere 1λ mesh λ/10, P* within 1.02x of baseline ==================
#
#  halfsize/λ    ε     L    NΩ  leaves   baseline    P*   compression   spectral P at τ/ε = 1, .1, .01
#      0.4     1e-03   16  2312     32   9.296e-06   12       9.12x         9   10   12
#      0.4     1e-04   18  2888     32   6.763e-06   13       9.82x        10   12   13
#      0.4     1e-06   20  3528     32   6.718e-06   13      12.00x        13   14   32
#      0.2     1e-03   10   968    113   1.252e-04    9       6.45x         7    8   10
#      0.2     1e-04   11  1152    113   5.957e-05    9       7.68x         8   10   11
#      0.2     1e-06   14  1800    113   2.186e-05   10       9.92x        11   12   32
#      0.1     1e-03    7   512    395   1.653e-03    7       5.33x         5    6    7
#      0.1     1e-04    8   648    395   8.108e-04    7       6.75x         6    7    8
#      0.1     1e-06    9   800    395   4.204e-04    8       6.58x         8    9   32
#
# 1. THE CALIBRATED RULE IS  τ = max(ε/100, 1e-7).
#    Across all nine configurations it NEVER UNDER-PREDICTS P* and over-predicts by at most 2. The
#    floor is not cosmetic: at ε = 1e-6, τ = ε/100 = 1e-8 asks for a relative tail below τ² = 1e-16,
#    which is round-off, and `truncationorder` correctly runs to the analysis ceiling instead (the
#    three `32`s in the last column). τ = ε/10 takes over there and predicts 14 / 12 / 9 against
#    P* = 13 / 10 / 8.
#    τ = ε ITSELF IS UNUSABLE: it under-predicts by up to 3 orders (5 against P* = 7), which is the
#    P = 9-at-λ/4 failure from `sheaccuracy.jl` generalised. Under-prediction is the dangerous
#    direction -- it silently spends the map's error budget.
#    COST OF THE MARGIN: over-predicting by 2 costs ((P+3)/(P+1))² ~ 1.4x of the leaf storage at
#    P = 9. Real, and the price of never having to check.
#
# 2. P* TRACKS L BUT IS NOT A FIXED FRACTION OF IT: P*/L runs 0.65 to 1.00, tending to 1 as boxes get
#    SMALL (0.1λ: 7/7, 7/8, 8/9) and to ~0.7 as they get large (0.4λ: 12/16, 13/18, 13/20). That is
#    the excess-bandwidth term: it dominates both P and L at small κd, so they converge, while at
#    large κd, L follows the diagonal and P a radius. No single P/L constant would be right.
#
# 3. THE 16/3 FLOOR IS REACHED, EXACTLY. At halfsize 0.1λ, ε = 1e-3 the measurement gives P* = L = 7
#    and compression 5.33x -- (16/3)·(8/8)² to three digits. So the closed form above is not a
#    heuristic; small boxes really do sit on its floor, and the scheme still wins there.
#
# 4. COMPRESSION IS 5.3x TO 12x ACROSS THE SWEEP. Quoting the λ/4 figure (8.64x) as "the" compression
#    is fixture-specific. It improves with box size and with tighter ε -- both because L grows faster
#    than P does.
#
# 5. FOR #121: bigger leaves are both MORE COMPRESSIBLE and MORE ACCURATE here -- baseline far error
#    runs 9.3e-06 at halfsize 0.4λ against 1.65e-03 at 0.1λ, a 178x spread at fixed ε. That
#    corroborates the accuracy-degrades-with-depth note already on #121 from #123, on a second
#    fixture family and across three tolerances.

using MLFMA
using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics:
    SphericalHarmonicTransform, analyze, gausslegendregrid, numcoefficients, truncationorder
using BEAST, CompScienceMeshes, H2Trees, H2Factory
using LinearAlgebra, StaticArrays, Printf, Random

const HT = H2Trees

function sphericalframe(sampling)
    k̂, _ = MLFMA.sphericalsamplepoints(sampling)
    θs = [p[2] for p in vec(k̂)]
    φs = [p[3] for p in vec(k̂)]
    return (
        hcat(cos.(θs) .* cos.(φs), cos.(θs) .* sin.(φs), -sin.(θs)),
        hcat(-sin.(φs), cos.(φs), zeros(length(φs))),
    )
end

# Overwrite every leaf collection with the order-`P` reconstruction, then measure the far matvec.
function farerror!(
    collections, leaves, coefficients, sampling, θ̂, φ̂, A, x, farreference, y, P
)
    S = SphericalHarmonicTransform(ComplexF64, sampling, P).synthesis
    nc = numcoefficients(P)
    NΩ = size(S, 1)
    for (i, leaf) in enumerate(leaves)
        collection = collections[leaf]
        nj = size(collection[1], 2)
        F = S * @view coefficients[i][1:nc, :]
        @inbounds for j in 1:nj, q in 1:NΩ
            a = F[q, j]
            b = F[q, nj + j]
            d = F[q, 2nj + j]
            collection[1][q, j] = θ̂[q, 1] * a + θ̂[q, 2] * b + θ̂[q, 3] * d
            collection[2][q, j] = φ̂[q, 1] * a + φ̂[q, 2] * b + φ̂[q, 3] * d
        end
    end
    H2Factory.farmul!(y, A, x)
    return norm(y - farreference) / norm(farreference)
end

function shecalibration(;
    frequency=5e7,
    radius=1.0,
    meshfactor=10,
    minhalfsizes=(0.4, 0.2, 0.1),      # -> leaf halfsize 0.5λ, 0.25λ, 0.125λ on this geometry
    tolerances=(1e-3, 1e-4, 1e-6),
    analysisorder=32,                   # must exceed every P any configuration could need
    tolerancefactor=1.02,               # "indistinguishable from baseline"
    taus=(1.0, 0.1, 0.01),              # τ = tau * ε, for the spectral rule
)
    μ0 = 4π * 1e-7
    ε0 = 8.854187812e-12
    c = 1 / sqrt(ε0 * μ0)
    λ = c / frequency
    κ = 2π / λ
    ω = κ * c
    γ = im * κ

    X = raviartthomas(meshsphere(radius * λ, λ / meshfactor))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ0))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)

    # ONE dense assembly for the whole sweep: it depends only on the mesh, not on the tree or on ε.
    println("N = ", numfunctions(X), " -- assembling the dense reference once")
    Adense = BEAST.assemble(efie, X, X; quadstrat=quadstrat)
    Random.seed!(11)
    x = randn(ComplexF64, numfunctions(X))
    y = similar(x)

    assembler = MLFMA.BEASTFarFieldAssembler(
        BEAST.MWFarField3D(; gamma=γ),
        X,
        MLFMA.FARFIELDQUADSTRATEGYRECOM,   # MUST match production's far-field quadrature
        MLFMA.StoreAsFarFieldValue(),
        MLFMA.CartesianVectorFarField,
    )
    θa, φa, wa = gausslegendregrid(analysisorder)
    glpoints = [SVector(sin(t) * cos(p), sin(t) * sin(p), cos(t)) for (t, p) in zip(θa, φa)]
    tanalysis = SphericalHarmonicTransform(ComplexF64, θa, φa, wa, analysisorder)

    println(
        "\n halfsize/λ  ε        L    NΩ   leaves   baseline      P*   compression   " *
        "spectral P at τ/ε = " *
        join(taus, ", "),
    )
    for mh in minhalfsizes, ϵ in tolerances
        tree = HT.TwoNTree(
            X;
            builder=HT.TwoNTreeBuilder(;
                minhalfsize=mh * λ, minvalues=0, protrusion=HT.NoProtrusionCheck()
            ),
        )
        leaves = collect(HT.leaves(tree))
        halfsize = HT.halfsize(tree, leaves[begin])

        A = try
            MLFMA.GalerkinMLFMA(
                efie,
                X;
                tree=tree,
                strategy=MLFMA.MLFMAStrategy(
                    efie,
                    X;
                    samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                    nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(
                        quadstrat
                    ),
                ),
                verbose=false,
            )
        catch e
            @printf(
                "%10.4f  %.0e   -- map construction failed: %s\n",
                halfsize / λ,
                ϵ,
                first(sprint(showerror, e), 90)
            )
            continue
        end

        sampling, _ = MLFMA.sampling(
            γ, ϵ, MLFMA.diameter(halfsize, 3), MLFMA.SphericalSamplingBuilder{Float64}()
        )
        NΩ = length(MLFMA.cartesiansamplepoints(sampling)[1])
        L = MLFMA.cutoffparameter(γ, ϵ, MLFMA.diameter(halfsize, 3))
        θ̂, φ̂ = sphericalframe(sampling)

        far = H2Factory.farinteractions(Adense, A)
        farreference = far * x
        H2Factory.farmul!(y, A, x)
        baseline = norm(y - farreference) / norm(farreference)

        Fs = assembler(
            tree,
            [
                MLFMA.BoxFarFieldSampling(sampling, glpoints, HT.center(tree, leaf), leaf)
                for leaf in leaves
            ];
            verbose=false,
        )
        coefficients = map(Fs) do F
            nj = size(F[1], 2)
            G = Matrix{ComplexF64}(undef, size(F[1], 1), 3nj)
            for component in 1:3
                @views G[:, ((component - 1) * nj + 1):(component * nj)] .= F[component]
            end
            return analyze(tanalysis, G)
        end

        # smallest P within `tolerancefactor` of baseline
        Pstar = -1
        for P in 3:analysisorder
            e = farerror!(
                collections(A),
                leaves,
                coefficients,
                sampling,
                θ̂,
                φ̂,
                A,
                x,
                farreference,
                y,
                P,
            )
            if e <= tolerancefactor * baseline
                Pstar = P
                break
            end
        end

        # what the SPECTRAL rule would have said, as a max over the level's boxes
        spectral = [
            maximum(truncationorder(a, tau * ϵ) for a in coefficients) for tau in taus
        ]

        compression = Pstar > 0 ? 2NΩ / (3 * numcoefficients(Pstar)) : NaN
        @printf(
            "%10.4f  %.0e  %3d %5d %6d   %.3e  %4d  %9.2fx     %s\n",
            halfsize / λ,
            ϵ,
            L,
            NΩ,
            length(leaves),
            baseline,
            Pstar,
            compression,
            join(lpad.(spectral, 4), " ")
        )
        A = nothing
        GC.gc()
    end
    return nothing
end

collections(A) = H2Factory.trialmomentcollections(A)

if abspath(PROGRAM_FILE) == @__FILE__
    shecalibration()
end
