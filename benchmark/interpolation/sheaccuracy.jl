# SE-MLFMA end to end: what a P-order leaf representation costs the FAR MATVEC.
#
# `shepatterns.jl` bounds the leaf representation IN ISOLATION -- how well coefficients reproduce one
# basis function's pattern. That is not the quantity anyone cares about. This file asks how that error
# composes with interpolation, translation and addition-theorem error once it is inside a matvec, and
# it is the last accuracy question SE-MLFMA had open.
#
# NO API CHANGE IS NEEDED TO ANSWER IT. Build an ordinary `GalerkinMLFMA`, then OVERWRITE each leaf
# collection's patterns with their SHE round trip -- assemble Cartesian on a Gauss-Legendre grid,
# analyse, truncate to `P`, synthesise onto the leaf grid, project to tangential. That is exactly the
# wired-in pipeline, run eagerly at setup instead of lazily per matvec, so it is a functional
# simulation of the real thing rather than a model of it. It buys none of the speed or the memory --
# only the numbers.
#
# ============================ RESULTS, sphere 1λ, mesh λ/10, leaf λ/4 ============================
# N = 4830, 56 leaves at level 3, production grid 28x56 = 1568 directions, ϵ = 1e-4.
# Far-only, against `farinteractions` extracted from a dense BEAST reference.
#
#      P   coeffs/bf   storage    far-only rel err   vs baseline
#      -        3136     1.00x       1.4962e-05         1.00x     <- untouched baseline
#      7         192    16.33x       2.5966e-04        17.36x
#      8         243    12.91x       4.3298e-05         2.89x
#      9         300    10.45x       1.7194e-05         1.15x
#     10         363     8.64x       1.5104e-05         1.01x
#     12         507     6.19x       1.4960e-05         1.00x
#     14         675     4.65x       1.4962e-05         1.00x
#     16         867     3.62x       1.4962e-05         1.00x
#     20        1323     2.37x       1.4962e-05         1.00x
#
# P = 10 IS THE OPERATING POINT, not P = 9. At P = 10 the far matvec is indistinguishable from the
# sampled baseline (1.01x) at 8.64x compression. P = 9 costs 15 % of the far error for 10.45x, which
# is a defensible trade if the extra 21 % of memory matters, but it is a trade and P = 10 is not.
#
# THE SYSTEM IS SLIGHTLY MORE DEMANDING THAN THE PATTERNS. `shepatterns.jl` found P = 9 meets 1e-4 at
# the worst single basis function; end to end, P = 9 still moves the far error by 15 %. Per-pattern
# errors do not simply take their maximum when summed into a group moment -- so the isolated bound is
# necessary but not sufficient, and this measurement is the one to quote.
#
# THE P >= 12 ROWS ARE THE CONTROL, and they are why the table can be trusted. At an order where
# truncation is at round-off the far error must return to the untouched baseline EXACTLY. It does --
# 1.4962e-05 to five digits, flat across four orders. A first run read 1.02x there instead, and the
# 2 % was not SHE: this file's Cartesian reassembly was using `SingleNumQStrat(3)` while production's
# far-field assembler uses `FARFIELDQUADSTRATEGYRECOM = SingleNumQStrat(5)`. A flat, non-unity control
# is what exposed it. Keep the quadratures tied together.

using MLFMA
using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics:
    SphericalHarmonicTransform, analyze, gausslegendregrid, numcoefficients
using BEAST, CompScienceMeshes, H2Trees, H2Factory
using LinearAlgebra, StaticArrays, Printf, Random

const HT = H2Trees

# θ̂ and φ̂ per direction -- the frame `StoreCartesianAsTangentialSpherical` projects onto.
function sphericalframe(sampling)
    k̂, _ = MLFMA.sphericalsamplepoints(sampling)
    θs = [p[2] for p in vec(k̂)]
    φs = [p[3] for p in vec(k̂)]
    return (
        hcat(cos.(θs) .* cos.(φs), cos.(θs) .* sin.(φs), -sin.(θs)),
        hcat(-sin.(φs), cos.(φs), zeros(length(φs))),
    )
end

function sheaccuracy(;
    frequency=5e7,
    radius=1.0,
    meshfactor=10,
    leaffactor=4,
    ϵ=1e-4,
    orders=(7, 8, 9, 10, 12, 14, 16, 20),
    analysisorder=22,        # must EXCEED every swept order, or the control cannot be formed
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
    tree = HT.TwoNTree(
        X;
        builder=HT.TwoNTreeBuilder(;
            minhalfsize=λ / leaffactor, minvalues=0, protrusion=HT.NoProtrusionCheck()
        ),
    )
    leaves = collect(HT.leaves(tree))

    A = MLFMA.GalerkinMLFMA(
        efie,
        X;
        tree=tree,
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
            nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
        ),
        verbose=false,
    )

    sampling, _ = MLFMA.sampling(
        γ,
        ϵ,
        MLFMA.diameter(HT.halfsize(tree, leaves[begin]), 3),
        MLFMA.SphericalSamplingBuilder{Float64}(),
    )
    NΩ = length(MLFMA.cartesiansamplepoints(sampling)[1])
    θ̂, φ̂ = sphericalframe(sampling)
    println(
        "N = ",
        numfunctions(X),
        "  leaves = ",
        length(leaves),
        "  leaf level = ",
        HT.level(tree, leaves[begin]),
        "  production grid = ",
        NΩ,
        " directions",
    )

    far = H2Factory.farinteractions(BEAST.assemble(efie, X, X; quadstrat=quadstrat), A)
    Random.seed!(11)
    x = randn(ComplexF64, numfunctions(X))
    farreference = far * x
    y = similar(x)
    H2Factory.farmul!(y, A, x)
    baseline = norm(y - farreference) / norm(farreference)
    @printf("BASELINE far-only rel err = %.4e\n", baseline)

    # THE QUADRATURE MUST MATCH PRODUCTION'S far-field assembler, or the control row lands off unity
    # and every other row inherits the offset. See this file's header.
    assembler = MLFMA.BEASTFarFieldAssembler(
        BEAST.MWFarField3D(; gamma=γ),
        X,
        MLFMA.FARFIELDQUADSTRATEGYRECOM,
        MLFMA.StoreAsFarFieldValue(),
        MLFMA.CartesianVectorFarField,
    )
    θs, φs, w = gausslegendregrid(analysisorder)
    points = [SVector(sin(t) * cos(p), sin(t) * sin(p), cos(t)) for (t, p) in zip(θs, φs)]
    t = SphericalHarmonicTransform(ComplexF64, θs, φs, w, analysisorder)
    Fs = assembler(
        tree,
        [
            MLFMA.BoxFarFieldSampling(sampling, points, HT.center(tree, leaf), leaf) for
            leaf in leaves
        ];
        verbose=false,
    )

    coefficients = map(Fs) do F
        nj = size(F[1], 2)
        G = Matrix{ComplexF64}(undef, size(F[1], 1), 3nj)
        for component in 1:3
            @views G[:, ((component - 1) * nj + 1):(component * nj)] .= F[component]
        end
        return analyze(t, G)
    end

    collections = H2Factory.trialmomentcollections(A)

    println("\n   P   coeffs/bf   storage    far-only rel err   vs baseline")
    for P in orders
        S = SphericalHarmonicTransform(ComplexF64, sampling, P).synthesis
        nc = numcoefficients(P)
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
        e = norm(y - farreference) / norm(farreference)
        @printf(
            "%4d %10d %8.2fx       %.4e        %6.2fx\n", P, 3nc, 2NΩ / 3nc, e, e / baseline
        )
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    sheaccuracy()
end
