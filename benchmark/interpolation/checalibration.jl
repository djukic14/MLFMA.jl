# What sets the circular-harmonic order M? Calibrating the 2-D truncation rule on REAL patterns.
#
# `CircularHarmonics.leveltruncationorder` reads a coefficient tail and returns the order to keep,
# with `τ = max(ε/ratio, tailfloor)`. The 3-D rule uses `ratio = 100`, calibrated in
# `shecalibration.jl` against spherical-harmonic expansions of RWG patterns on a sphere. THAT NUMBER
# IS NOT TRANSFERABLE and this file exists to say what the 2-D one is instead, rather than inheriting
# a constant fitted on a different geometry, a different basis and a different leaf occupancy.
#
# ============================== WHAT IS MEASURED, AND WHAT IS NOT ==============================
# MEASURED: the GROUP MOMENT error at one leaf. For a leaf holding `N_j` basis functions and a random
# excitation `x`, the sampled leaf multiply produces `F_g = Σ_j x_j F_j` on the production grid; the
# compressed path produces `S_M (Σ_j x_j a_j)` truncated to order `M`. The relative difference of the
# two, in the production grid's 2-norm, is what the compressed leaf multiply actually gets wrong.
#
# The group moment and not the per-pattern error, because that is the quantity the scheme produces.
# Per-pattern reconstruction errors do not simply carry over: they neither take their maximum nor
# average when summed against `x`, and the gap between the two is exactly what the `ratio` absorbs.
#
# NOT MEASURED HERE: the far-field error of a full matvec. The compressed leaf multiply is only the
# first stage; interpolation, translation and the addition theorem all contribute after it, and how
# the leaf error composes with those is slice 3's test (d). If that measurement disagrees with this
# one, THIS CONSTANT IS WHAT SHOULD MOVE -- it is calibrated against a well-defined intermediate, not
# against the end of the chain. Stated up front because #120 and #135 both produced numbers that had
# to be retracted for being about a different quantity than the one they were quoted for.
#
# ============================== M AND L ARE DIFFERENT QUANTITIES ==============================
# Exactly as in 3-D. `L` truncates the TRANSLATION operator and is set by the box diameter, and it
# fixes the grid at `Nφ = NφCircular(L) = 2(L+1)`. `M` is the harmonic content of ONE BASIS FUNCTION's
# pattern about its own box centre -- `exp(-γ k̂·(r-c))` over that function's support -- so it depends
# on a RADIUS, `max|r-c|`, not on box-to-box geometry.
#
# The compression has a closed form, and it is where 2-D differs from 3-D most sharply:
#
#     today  1 component x 2(L+1) samples        = 2(L+1)
#     CHE    1 component x (2M+1) coefficients   = 2M+1
#     compression = 2(L+1)/(2M+1)                       -- LINEAR in L/M
#
# against 3-D's `(16/3)·((L+1)/(P+1))²`. The 3-D scheme wins 5.33x even at `P = L`, purely because a
# band-limited function on the sphere needs `(L+1)²` coefficients while the product grid spends
# `8(L+1)²` samples. THE CIRCLE HAS NO SUCH SLACK: at `M = L` the ratio is `2(L+1)/(2L+1) ≈ 1`. Every
# bit of 2-D compression comes from `M < L` and nothing comes from the grid being redundant -- which
# is the same O(L) versus O(L²) asymmetry, seen from the storage side.
#
# ============================== METHOD ==============================
# Each leaf's patterns are assembled TWICE, both times referenced to that leaf's OWN centre via
# `BoxFarFieldSampling` (which is what makes the coefficients summable at all):
#
#     (a) on the leaf's PRODUCTION grid, `Nφ = NφCircular(L)`     -> the reference
#     (b) on an OVERSAMPLED circular grid, analysed to order M₀   -> the CHE representation
#
# Oversampling (b) is about ALIASING, not quadrature: the circle's trapezoidal rule is exact for every
# mode it resolves (there is no Jacobian -- see `transform.jl`'s header), but a real basis pattern is
# not band-limited, so content above `N/2` folds back onto the modes being kept. The sweep over
# `oversampling` below is what shows where that floor stops moving.
#
# ============================== RESULTS ==============================
#
# 1. ANALYSIS OVERSAMPLING BUYS NOTHING. `meshcircle(4.0, 1/20)`, 60 leaves, production Nφ = 28:
#
#      over   Nφ(ana)   M₀    err M=6       err M=8       err M=10
#         1        28   13    4.30757e-04   7.07681e-06   6.81650e-08
#         2        56   27    4.30757e-04   7.07681e-06   6.81650e-08
#         4       112   55    4.30757e-04   7.07681e-06   6.81650e-08
#         8       224  111    4.30757e-04   7.07681e-06   6.81650e-08
#        16       448  223    4.30757e-04   7.07681e-06   6.81650e-08
#
#    Identical to six digits. Aliasing is a real mechanism here and it simply never reaches the
#    retained modes: the pattern's own spectrum is at ~1e-14 by |m| = 16, so what folds back past the
#    leaf grid's Nyquist bin at 14 is far below the error being measured. Analysing on the leaf's own
#    grid is therefore correct AND cheapest. (The 3-D side reached the same verdict from the opposite
#    starting point -- see `shepatterns.jl` §3, "the cheapest grid is the right one".)
#
# 2. THE RULE. `M*` is the smallest kept order meeting ε; `r1`/`r10`/`r100` are what
#    `leveltruncationorder` predicts at `τ = ε`, `ε/10`, `ε/100`. A `*` marks an UNDER-prediction.
#
#      box   N_j    ε=1e-3          ε=1e-4          ε=1e-5          ε=1e-6
#                  M*  r1 r10 r100  M*  r1 r10 r100  M*  r1 r10 r100  M*  r1 r10 r100
#      λ/4    10    6   6   7   8    7   7   8   9    8   8   9  10    9   9  10  10
#      λ/4    20    6   6   7   8    8   7*  8   9    9   8*  9  10   10   9* 10  10
#      λ/4    40    6   6   7   8    8   7*  8   9    9   8*  9  10   10   9* 10  10
#      λ/2    40    9   8* 10  11   10  10  11  13   12  11* 13  14   13  13  14  14
#      λ/2    81    9   8* 10  11   10  10  11  13   12  11* 13  14   13  13  14  14
#
#    `τ = ε` under-predicts in 8 of 20 rows, always by exactly 1. `τ = ε/10` NEVER under-predicts and
#    over-predicts by at most 1 -- so `CHETAILRATIO = 10`. `τ = ε/100` is also safe but over-predicts
#    by up to 3 and costs up to 17 % more coefficients than `ε/10` does -- at ε = 1e-4 on the λ/2
#    box, 27 coefficients where 23 suffice, i.e. compression 1.48x instead of 1.74x.
#
# 3. WHAT `M*` DEPENDS ON: box size and ε, NOT leaf occupancy. The λ/4 rows at N_j = 20 and 40 are
#    identical, as are the λ/2 rows at 40 and 81; only the sparsest leaf (10) differs, by one order.
#    `M*` SATURATES -- once a leaf holds enough functions to cover its box, the group moment's
#    bandwidth is the BOX's, not any individual function's. That also gives this rule support the 3-D
#    one explicitly lacks: `shecalibration.jl`'s correction note records that all three of its valid
#    configurations shared one box size, leaving its box-size dependence unmeasured. This one spans
#    two.
#
# 4. COMPRESSION AT THE CALIBRATED ORDER, which is LOWER than the estimate this task was specified
#    with. That estimate quoted 1.75x from a per-leaf MEDIAN order and a `2M+2` count; the scheme runs
#    at the level MAXIMUM (one M per level or the batched GEMM goes ragged) with a `2M+1` count. At
#    λ/4, ε = 1e-5 that is `28/19 = 1.47x`, not 1.75x. Range over the calibrated orders in the
#    table: 1.33x (λ/4, ε = 1e-6) to 1.90x (λ/2, ε = 1e-3).
#    `checrossover.jl` carries the same correction into the timing.

using MLFMA
using MLFMA.CircularHarmonics
using MLFMA.CircularHarmonics:
    CircularHarmonicTransform,
    analyze,
    leveltruncationorder,
    numcoefficients,
    synthesize,
    truncatecoefficients,
    truncationorder
using BEAST, CompScienceMeshes, H2Trees
using LinearAlgebra, Random, StaticArrays, Statistics, Printf

const HT = H2Trees
const CH = MLFMA.CircularHarmonics

# κ = 2π, i.e. λ = 1, so `minhalfsize` reads directly as a fraction of a wavelength and 0.25 IS
# `MINHALFSIZERECOM` -- the λ/4 low-frequency-breakdown floor `MLFMA.checkboxsizes` warns below.
function chefixture(; radius=4.0, h=1 / 20, minhalfsize=0.25, ϵ=1e-5)
    κ = 2π
    γ = im * κ
    operator = Helmholtz2D.singlelayer(; wavenumber=κ)
    X = lagrangec0d1(CompScienceMeshes.meshcircle(radius, h))
    tree = HT.TwoNTree(
        X;
        builder=HT.TwoNTreeBuilder(;
            minhalfsize=minhalfsize, minvalues=0, protrusion=HT.NoProtrusionCheck()
        ),
    )
    leaves = collect(HT.leaves(tree))
    leaflevel = maximum(HT.level(tree, leaf) for leaf in leaves)
    sampling, _ = MLFMA.sampling(
        γ,
        ϵ,
        tree;
        level=leaflevel,
        samplingbuilder=MLFMA.CircularSamplingBuilder{Float64}(),
    )
    assembler = MLFMA.BEASTFarFieldAssembler(
        MLFMA.radiationoperator(operator),
        X,
        BEAST.SingleNumQStrat(3),
        MLFMA.farfieldstorerule(operator),
        MLFMA.farfieldtype(operator),
    )
    return (; X, tree, leaves, sampling, assembler, operator, γ, ϵ)
end

# Every leaf gets its own centre and the SAME grid: a forced-uniform tree, so "the leaf grid" is one
# thing and the level order the rule returns is meaningful.
boxes(f, sampling, leaves) = [
    MLFMA.BoxFarFieldSampling(
        sampling,
        first(MLFMA.cartesiansamplepoints(sampling)),
        HT.center(f.tree, leaf),
        leaf,
    ) for leaf in leaves
]

# `Fs[i]` is an `SVector{1,Matrix}` for a scalar 2-D operator: `(ndirections × N_j)`.
patternsat(f, sampling, leaves) =
    f.assembler(f.tree, boxes(f, sampling, leaves); verbose=false)

"""
Coefficients of every leaf, analysed at order `M₀` on a grid oversampled by `oversampling`.
Returns `(coefficients, M₀)` with `coefficients[i]` a `(2M₀+1) × N_j` matrix.
"""
function analyseleaves(f, leaves; oversampling=8)
    Nφ = length(f.sampling)
    analysissampling = MLFMA.CircularSampling(oversampling * Nφ)
    M₀ = MLFMA.cutoffparameter(analysissampling)
    t = CircularHarmonicTransform(ComplexF64, analysissampling, M₀)
    Fs = patternsat(f, analysissampling, leaves)
    return [analyze(t, F[1]) for F in Fs], M₀
end

"""
Worst relative GROUP MOMENT error over `leaves` at kept order `M`, over `ntrials` excitations.

`reference[i]` is the leaf's production-grid pattern matrix; `coefficients[i]` its expansion. The
compressed path aggregates in coefficient space FIRST and synthesises once, which is the operation
order the scheme actually uses -- synthesising per basis function and summing afterwards would be a
different (and pointlessly expensive) computation.
"""
function groupmomenterror(f, reference, coefficients, M₀, M; ntrials=4, seed=0)
    t = CircularHarmonicTransform(ComplexF64, f.sampling, M)
    rng = MersenneTwister(seed)
    worst = 0.0
    for i in eachindex(reference)
        P = reference[i][1]
        a = truncatecoefficients(coefficients[i], M)
        for _ in 1:ntrials
            x = randn(rng, ComplexF64, size(P, 2))
            exact = P * x
            approx = synthesize(t, a * x)
            worst = max(worst, norm(approx - exact) / norm(exact))
        end
    end
    return worst
end

# --- section 1: how much oversampling does the ANALYSIS need? -------------------------------------
#
# The aliasing floor, isolated. If the answer were "more is always better" the setup cost would be
# unbounded; the useful answer is where it stops moving.
function oversamplingsweep(f=chefixture(); factors=[1, 2, 4, 8, 16], leaves=nothing)
    ls = leaves === nothing ? f.leaves : leaves
    reference = patternsat(f, f.sampling, ls)
    L = MLFMA.cutoffparameter(f.sampling)
    @printf(
        "\n=== analysis oversampling, %d leaves, production Nφ = %d (L = %d) ===\n",
        length(ls),
        length(f.sampling),
        L
    )
    @printf(
        "%5s %8s %6s   %-11s %-11s %-11s\n",
        "over",
        "Nφ(ana)",
        "M₀",
        "err M=6",
        "err M=8",
        "err M=10"
    )
    for over in factors
        coefficients, M₀ = analyseleaves(f, ls; oversampling=over)
        errs = [groupmomenterror(f, reference, coefficients, M₀, M) for M in (6, 8, 10)]
        @printf(
            "%5d %8d %6d   %.5e %.5e %.5e\n",
            over,
            over * length(f.sampling),
            M₀,
            errs[1],
            errs[2],
            errs[3]
        )
        flush(stdout)
    end
end

# --- section 2: the rule ---------------------------------------------------------------------------
#
# For each ε: find `M*`, the smallest kept order whose worst group-moment error is at or below ε, then
# report what `leveltruncationorder` predicts at each candidate `ratio`. A ratio is ACCEPTABLE if it
# never predicts below `M*` (under-prediction silently spends the map's error budget) and WASTEFUL to
# the extent it predicts above.
function rulesweep(
    f=chefixture();
    tolerances=[1e-3, 1e-4, 1e-5, 1e-6],
    ratios=[1, 10, 100],
    orders=0:20,
    oversampling=8,
    leaves=nothing,
)
    ls = leaves === nothing ? f.leaves : leaves
    reference = patternsat(f, f.sampling, ls)
    coefficients, M₀ = analyseleaves(f, ls; oversampling=oversampling)
    L = MLFMA.cutoffparameter(f.sampling)

    errors = Dict(M => groupmomenterror(f, reference, coefficients, M₀, M) for M in orders)

    @printf(
        "\n=== truncation rule, %d leaves, Nφ = %d (L = %d), N_j median %.1f ===\n",
        length(ls),
        length(f.sampling),
        L,
        median([size(R[1], 2) for R in reference])
    )
    @printf(
        "%9s %5s %12s %10s   %s\n",
        "ε",
        "M*",
        "err(M*)",
        "compress",
        join([@sprintf("ratio %d", r) for r in ratios], "  ")
    )
    for ε in tolerances
        candidates = collect(orders)
        Mstar = candidates[something(
            findfirst(M -> errors[M] <= ε, candidates), length(candidates)
        )]
        predicted = [leveltruncationorder(coefficients, ε; ratio=r) for r in ratios]
        @printf(
            "%9.0e %5d %12.4e %9.2fx   %s\n",
            ε,
            Mstar,
            errors[Mstar],
            MLFMA.NφCircular(L) / numcoefficients(Mstar),
            join([@sprintf("%9d", p) for p in predicted], "  ")
        )
        flush(stdout)
    end

    @printf("\nper-order group-moment error:\n")
    for M in orders
        @printf(
            "  M = %2d   %.5e   compression %.2fx\n",
            M,
            errors[M],
            MLFMA.NφCircular(L) / numcoefficients(M)
        )
    end

    @printf("\nper-leaf spectral order at τ = ε (no tightening), for reference:\n")
    for ε in tolerances
        os = [truncationorder(a, ε) for a in coefficients]
        @printf(
            "  ε = %.0e   min %d  median %.1f  max %d\n",
            ε,
            minimum(os),
            median(os),
            maximum(os)
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    f = chefixture()
    @printf(
        "fixture: %d functions, %d leaves, Nφ = %d\n",
        numfunctions(f.X),
        length(f.leaves),
        length(f.sampling)
    )
    oversamplingsweep(f)
    rulesweep(f)
    @printf("\n--- finer mesh (h = 1/40), a second point on the occupancy axis ---\n")
    f2 = chefixture(; radius=4.0, h=1 / 40)
    @printf(
        "fixture: %d functions, %d leaves, Nφ = %d\n",
        numfunctions(f2.X),
        length(f2.leaves),
        length(f2.sampling)
    )
    rulesweep(f2)
end
