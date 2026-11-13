# SE-MLFMA on REAL basis-function patterns: what order they need, what they cost to produce.
#
# Everything in `shesynthesis.jl` -- the kernel timings, the batching verdict -- was measured on
# SYNTHETIC band-limited patterns, i.e. patterns built to be exactly representable in the basis they
# were then compressed into. That is fine for timing a GEMM and useless for answering whether real
# RWG far-field patterns compress at all. This file assembles one real EFIE problem's leaves TWICE:
#
#     (a) on the leaf's PRODUCTION direction grid                     -> the reference
#     (b) on a Gauss-Legendre grid, analysed into coefficients        -> the SHE representation
#
# and compares (b) synthesised back onto (a)'s grid against (a). Section 1 sweeps the order kept;
# section 2 prices the setup, which is the one cost in the scheme that had never been measured.
#
# CARTESIAN, NOT TANGENTIAL, and that is not an implementation detail. The θ̂/φ̂ components carry the
# spherical frame's pole singularity, so their harmonic content is the FRAME's, not the pattern's, and
# they do not compress. EFIE normally stores tangential via `StoreCartesianAsTangentialSpherical`;
# here the same assembler is driven with `StoreAsFarFieldValue` + `CartesianVectorFarField` to keep
# all three Cartesian components -- the first consumer that type has ever had. Comparing in Cartesian
# is also the CONSERVATIVE choice: the tangential projection has orthonormal rows, so it can only
# shrink an error, never grow one. (The assembled field is transverse to 3.6e-17, so the two norms
# agree anyway and the "conservative" caveat costs nothing.)
#
# ============================== RESULTS, sphere 1λ, mesh λ/10, leaf λ/4 ==============================
# N = 4830, 56 leaves, N_j median 92.5, production grid 28x56 = 1568 directions, L = 13.
#
# 1. ORDER REQUIRED, over 894 real basis functions across 12 leaves:
#
#      P     median         p90          max
#      6     1.078e-03    3.683e-03    1.540e-02
#      7     1.400e-04    6.160e-04    3.299e-03
#      8     1.582e-05    8.786e-05    6.097e-04
#      9     1.591e-06    1.122e-05    9.772e-05   <- meets 1e-4 at the WORST basis function
#     10     1.403e-07    1.250e-06    1.381e-05
#     12     7.948e-10    1.168e-08    2.053e-07
#     16     1.081e-14    3.816e-13    1.634e-11
#
#    P = 9 on real patterns, which is exactly what the synthetic estimate predicted (median 9). The
#    distribution matters as much as the median: at P = 8 the median is already 1.6e-5 but the worst
#    basis function is 6.1e-4, so a median-driven choice would miss the tolerance by 6x on the
#    patterns that actually set the error.
#
# 2. STORAGE, in the terms production actually stores in -- TANGENTIAL samples today (2 components),
#    CARTESIAN coefficients under SHE (3):
#        2 x 1568 = 3136 complex  ->  3 x (9+1)^2 = 300 complex  =  10.45x
#    Note the asymmetry is real and must not be quietly dropped: SHE pays a third component to avoid
#    the pole singularity, and still wins by an order of magnitude. Quoting 3-vs-3 (15.7x) would
#    compare SHE against a representation nothing stores.
#
# 3. SETUP IS CHEAPER THAN TODAY'S ASSEMBLY, which inverts the risk this file was written to check.
#    `Pa` is the analysis order, i.e. the grid `gausslegendregrid(Pa)`; the kept order is 9 throughout.
#    Analysis is ONE GEMM per leaf over all 3*N_j component columns.
#
#     Pa   GL grid   assembly   analysis   setup total   vs today   err(P=9) med / max
#      9      200     0.435 s    0.309 s     0.744 s       0.32x    1.10e-06 / 9.86e-05
#     10      242     0.362 s    0.285 s     0.648 s       0.28x    1.10e-06 / 9.79e-05
#     12      338     0.480 s    0.309 s     0.789 s       0.34x    1.10e-06 / 9.79e-05
#     16      578     0.836 s    0.369 s     1.205 s       0.52x    1.10e-06 / 9.79e-05
#     24     1250     1.822 s    0.786 s     2.608 s       1.12x    1.10e-06 / 9.79e-05
#                                          (today: 2.335 s)
#
#    The three cheap rows are within each other's noise -- this is a shared machine and the analysis
#    column in particular moved by 3x between runs. Read the SHAPE, which is stable: roughly a third
#    of today's cost while the analysis grid stays well below the production grid, crossing 1x only
#    once the analysis grid approaches it.
#
#    WHY, and it is the whole thesis of the method rather than an accident: the production grid must
#    resolve the TRANSLATION bandwidth L = 13 plus its excess-bandwidth term, while the SHE grid need
#    only resolve the PATTERN's own bandwidth P = 9 -- and `gausslegendregrid` is exact in cos θ where
#    `MLFMA.sphericalsamplepoints` merely converges, which buys a further factor in θ. 1568 directions
#    become 200.
#
#    THE CHEAPEST GRID IS THE RIGHT ONE. Content above `Pa` aliases DOWN into the retained
#    coefficients, so a small analysis grid is exactly where a scheme like this quietly loses accuracy.
#    It does not here: the P = 9 error is the same to three digits whether analysed on 200 directions
#    or on 1250. Oversampling the setup buys nothing and costs up to 7x.
#
# WHAT THIS FILE DOES NOT ESTABLISH. It measures the leaf REPRESENTATION in isolation. It does not
# run a far matvec, so it says nothing about how SHE error composes with interpolation, translation
# and addition-theorem error -- that is the A/B still outstanding on #122.

using MLFMA
using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics:
    SphericalHarmonicTransform, analyze, gausslegendregrid, numcoefficients, synthesize
using BEAST, CompScienceMeshes, H2Trees
using LinearAlgebra, StaticArrays, Statistics, Printf

const HT = H2Trees

# The fixture is the one every other SHE measurement used, so the numbers compose: λ/4 leaves, which
# is `MINHALFSIZERECOM`, and a forced-uniform tree so that "the leaf grid" is a single thing.
function shefixture(; frequency=5e7, radius=1.0, meshfactor=10, leaffactor=4, ϵ=1e-4)
    μ0 = 4π * 1e-7
    ε0 = 8.854187812e-12
    c = 1 / sqrt(ε0 * μ0)
    λ = c / frequency
    κ = 2π / λ
    γ = im * κ

    X = raviartthomas(meshsphere(radius * λ, λ / meshfactor))
    tree = HT.TwoNTree(
        X;
        builder=HT.TwoNTreeBuilder(;
            minhalfsize=λ / leaffactor, minvalues=0, protrusion=HT.NoProtrusionCheck()
        ),
    )
    leaves = collect(HT.leaves(tree))
    sampling, _ = MLFMA.sampling(
        γ,
        ϵ,
        MLFMA.diameter(HT.halfsize(tree, leaves[begin]), 3),
        MLFMA.SphericalSamplingBuilder{Float64}(),
    )
    points, _ = MLFMA.cartesiansamplepoints(sampling)

    # `StoreAsFarFieldValue` + `CartesianVectorFarField`, NOT the EFIE default -- see this file's
    # header. The rule needs no `SphericalFrameCache`, so the `sampling` carried by each
    # `BoxFarFieldSampling` below is never read and any grid may be passed as `points`.
    assembler = MLFMA.BEASTFarFieldAssembler(
        BEAST.MWFarField3D(; gamma=γ),
        X,
        BEAST.SingleNumQStrat(3),
        MLFMA.StoreAsFarFieldValue(),
        MLFMA.CartesianVectorFarField,
    )
    return (; X, tree, leaves, sampling, points, assembler, λ, γ)
end

boxes(f, points, leaves) = [
    MLFMA.BoxFarFieldSampling(f.sampling, points, HT.center(f.tree, leaf), leaf) for
    leaf in leaves
]

function gausslegendrepoints(P)
    θs, φs, _ = gausslegendregrid(P)
    return [SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)) for (θ, φ) in zip(θs, φs)]
end

# `k̂ · F`, which must vanish for a MW far field. A check on the Cartesian store path itself, since
# nothing has ever exercised it.
function transversality(F, points)
    radial = 0.0
    total = 0.0
    for j in axes(F[1], 2), q in eachindex(points)
        v = SVector(F[1][q, j], F[2][q, j], F[3][q, j])
        radial += abs2(dot(points[q], v))
        total += abs2(norm(v))
    end
    return sqrt(radial / total)
end

# One GEMM per leaf over all `3 * N_j` component columns -- the shape a real setup would use, and
# roughly 3x faster than a GEMV per basis function.
function analyselevel(t, Fs)
    return map(Fs) do F
        nj = size(F[1], 2)
        G = Matrix{ComplexF64}(undef, size(F[1], 1), 3nj)
        for component in 1:3
            @views G[:, ((component - 1) * nj + 1):(component * nj)] .= F[component]
        end
        return analyze(t, G)
    end
end

# Error of the `keep`-order reconstruction against the directly assembled production patterns, one
# entry per basis function.
function reconstructionerrors(f, coefficients, reference, keep)
    t = SphericalHarmonicTransform(ComplexF64, f.sampling, keep)
    errors = Float64[]
    for i in eachindex(coefficients)
        nj = size(reference[i][1], 2)
        S = synthesize(t, coefficients[i][1:numcoefficients(keep), :])
        for j in 1:nj
            R = hcat(reference[i][1][:, j], reference[i][2][:, j], reference[i][3][:, j])
            E = hcat(S[:, j], S[:, nj + j], S[:, 2nj + j])
            push!(errors, norm(E - R) / norm(R))
        end
    end
    return errors
end

# --- section 1: what order do real patterns need? ------------------------------------------------
#
# Analysed ONCE at `analysisorder`, then truncated. Truncation is exact row-slicing: with
# `lmindex(l, m) = l² + l + m + 1`, every coefficient of degree <= P occupies the first (P+1)² rows,
# so one analysis serves the whole sweep.
function ordersweep(
    f=shefixture();
    orders=[4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20],
    analysisorder=24,
    nleaves=12,
)
    NΩ = length(f.points)
    selection = f.leaves[round.(
        Int, range(1, length(f.leaves); length=min(nleaves, length(f.leaves)))
    )]
    println(
        "N = ",
        numfunctions(f.X),
        "  leaves ",
        length(selection),
        "/",
        length(f.leaves),
        "  production grid = ",
        NΩ,
        " directions",
    )

    reference = f.assembler(f.tree, boxes(f, f.points, selection); verbose=false)
    println(
        @sprintf(
            "transversality: |k̂·F| / |F| = %.2e", transversality(reference[1], f.points)
        )
    )

    points = gausslegendrepoints(analysisorder)
    θs, φs, w = gausslegendregrid(analysisorder)
    t = SphericalHarmonicTransform(ComplexF64, θs, φs, w, analysisorder)
    coefficients = analyselevel(
        t, f.assembler(f.tree, boxes(f, points, selection); verbose=false)
    )

    println("\n   P   coeffs/bf   vs today       median          p90            max")
    for P in orders
        e = reconstructionerrors(f, coefficients, reference, P)
        nc = 3 * numcoefficients(P)
        println(
            @sprintf(
                "%4d %10d %9.2fx  %13.3e  %13.3e  %13.3e",
                P,
                nc,
                2NΩ / nc,
                median(e),
                quantile(e, 0.9),
                maximum(e)
            )
        )
    end
    return nothing
end

# --- section 2: what does the setup cost, against the assembly it replaces? ----------------------
#
# `analysisorder` sweeps the GRID; `keep` is fixed, so every row reconstructs to the same order and
# the error column isolates ALIASING from truncation.
function setupcost(f=shefixture(); analysisorders=(9, 10, 12, 16, 24), keep=9)
    println(
        "N = ",
        numfunctions(f.X),
        "  leaves = ",
        length(f.leaves),
        "  production grid = ",
        length(f.points),
        " directions   keeping P = ",
        keep,
    )

    f.assembler(f.tree, boxes(f, f.points, f.leaves[1:1]); verbose=false)   # warm up
    todays = @elapsed reference = f.assembler(
        f.tree, boxes(f, f.points, f.leaves); verbose=false
    )
    println(@sprintf("\nproduction assembly (today's cost): %.3f s", todays))

    println(
        "\n Pa   GL grid   assembly   analysis   setup total   vs today   err med / max"
    )
    for Pa in analysisorders
        points = gausslegendrepoints(Pa)
        θs, φs, w = gausslegendregrid(Pa)
        t = SphericalHarmonicTransform(ComplexF64, θs, φs, w, Pa)

        f.assembler(f.tree, boxes(f, points, f.leaves[1:1]); verbose=false)
        tassemble = @elapsed Fs = f.assembler(
            f.tree, boxes(f, points, f.leaves); verbose=false
        )
        analyselevel(t, Fs[1:1])
        tanalyse = @elapsed coefficients = analyselevel(t, Fs)

        e = reconstructionerrors(f, coefficients, reference, keep)
        println(
            @sprintf(
                "%3d %8d %9.3f s %9.3f s %11.3f s %9.2fx   %.2e / %.2e",
                Pa,
                length(points),
                tassemble,
                tanalyse,
                tassemble + tanalyse,
                (tassemble + tanalyse) / todays,
                median(e),
                maximum(e)
            )
        )
    end
    println(
        @sprintf(
            "\nstorage per basis function: today 2*%d = %d complex; SHE 3*(%d+1)^2 = %d  ->  %.2fx",
            length(f.points),
            2 * length(f.points),
            keep,
            3 * numcoefficients(keep),
            2 * length(f.points) / (3 * numcoefficients(keep))
        )
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    f = shefixture()
    ordersweep(f)
    println()
    setupcost(f)
end
