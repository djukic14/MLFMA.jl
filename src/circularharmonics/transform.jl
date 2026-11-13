# The circular-harmonic transform pair over a fixed set of angles.
#
# Synthesis is `S a`; analysis is `Sᴴ W F`. Both use the same stored `S` and `W`, so analysis
# applies the adjoint of synthesis to weighted samples without a separate reverse kernel.
#
# ------------------------------------------------------------------------------------------------
# The circular basis simplifies storage and transformation.
#
# The spherical transform's header spends most of its length on quadrature accuracy: `sphericalsamplepoints`
# carries `sin θ` as an explicit weight, so its integrand is not a polynomial in `θ`, no exactness
# theorem applies, and setup analysis needs about 4x the θ nodes (measured: 1.2e0 at `Nθ = P+1`,
# 1.3e-15 only at `4P+4`). `gausslegendregrid` exists to substitute `x = cos θ` and remove the
# Jacobian.
#
# The unit circle has line element `dφ`. The trapezoidal rule on
# `[0, 2π)` with `N` points is EXACT for every mode `e^{imφ}` with `|m| < N` -- not spectrally
# convergent, exact -- so the analysis integrand `f · conj(ψ_m)`, of bandwidth `2M` for a pattern
# band-limited to `M`, is integrated with no error at all by `N = 2M + 2 = NφCircular(M)` points. The
# grid MLFMA already builds is the exact one, and there is no substitution to make and no oversampling
# theorem to respect.
#
# `oversampling` on [`circulargrid`](@ref) therefore exists for ONE reason only, and it is a different
# reason than the spherical `oversampling`: a real basis function's pattern is `∫exp(-γ k̂·(r-c))`
# over its support, which is an entire function of `φ` and NOT band-limited. Its coefficients decay
# fast but never terminate, so content above `N/2` ALIASES back onto the retained modes. Oversampling
# pushes the aliasing floor down; it is not correcting a quadrature that would otherwise be inexact.
#
# ------------------------------------------------------------------------------------------------
# Dense synthesis versus an FFT.
#
# Circular synthesis is a zero-pad plus an inverse DFT, so an FFT is available here where no such thing
# is available on the sphere -- `CircularInterpolator` uses exactly that, with PRE-PLANNED IN-PLACE
# transforms in its per-(level, chunk) scratch. The question of whether this interface should too was
# Measurements against a batched in-place `plan_ifft!` over the same buffer (so both
# sides are planned and batched), with the zero-fill and the centered scatter the FFT path genuinely
# needs included:
#
#     Nφ    ncoeff   FFT / GEMM        what it is
#     28      19     1.14x - 1.23x     λ/4 leaf at ε = 1e-5   <- the grid this actually runs on
#     28      15     1.33x - 1.44x     λ/4 leaf at ε ≈ 1e-3
#     40      27     0.90x - 0.96x     λ/2 leaf at ε = 1e-5
#     46      27     2.20x - 2.38x     L = 22, so Nφ = 2·23
#    200     121     0.27x - 0.39x     a box far larger than any leaf
#
# At the λ/4 leaf the GEMM wins by 1.14x-1.23x; at a λ/2 leaf the planned FFT is marginally faster.
# Arithmetic volume alone does not explain the choice.
#
# GEMM cost does not depend on factorization. `Nφ = 2(L+1)` comes from the rate rule and
# it can land on `2·prime` -- at `L = 22`, `Nφ = 46 = 2·23` sends FFTW to Rader/Bluestein and the FFT
# becomes 2.38x SLOWER. A dense `S` costs `Nφ·(2M+1)` regardless. The FFT only wins decisively at
# `Nφ = 200`, which needs `L ≈ 99`, far above what a λ/4 leaf box produces -- and leaf storage is by
# definition a leaf-grid operation.
#
# The GEMM form is also what lets the whole LEVEL share one `S` in a single batched multiply, which is
# mandatory rather than optional here (see `storage.jl`'s header for the 2-D measurement and for why
# its mechanism differs from 3-D's).
#
# So: `CircularInterpolator` resamples with an FFT because there the transform is the whole operation
# and `N` is both the input and the output size; here the transform lands on `2M+1 << N` coefficients,
# a dense `S` is shared across a level, and the cost is insensitive to how `N` factorizes. The two are
# not in tension -- they are different shapes.
#
# ------------------------------------------------------------------------------------------------
# LAYOUT: `m` RUNS `-M:M`, CENTERED, and truncation is a CONTIGUOUS CENTER VIEW.
#
# The spherical side orders `(l, m)` as `l² + l + m + 1`, which puts every coefficient of degree `≤ P`
# in the LEADING `(P+1)²` rows, so `truncatecoefficients` is a leading slice. The centered circular
# ordering gives up "leading" but keeps "contiguous": degree `≤ M′` occupies rows
# `(M-M′+1):(M+M′+1)`, still a `view` with no gather. An ordering of `0, +1, -1, +2, -2, …` would
# restore the leading slice at the cost of making every other index expression unreadable, and buys
# nothing since a centered range is equally free.

"""
    mindex(m, M) -> Int

Row index of mode `m` in a `-M:M` centered expansion: `m + M + 1`.

Takes `M` because the ordering is centered rather than leading -- see this file's header. The inverse
is [`orderof`](@ref).
"""
@inline mindex(m::Integer, M::Integer) = Int(m) + Int(M) + 1

"""
    orderof(index, M) -> Int

The signed mode `m` at 1-based `index` of a `-M:M` centered expansion. Inverse of [`mindex`](@ref).
"""
@inline orderof(index::Integer, M::Integer) = Int(index) - Int(M) - 1

"""
    numcoefficients(M) -> Int

`2M + 1`, the number of circular harmonics of order `|m| ≤ M`.

The 2-D counterpart of `(P+1)²`, and the whole O(L) versus O(L²) story in one line: a degree-`L`
pattern on a circle needs `2L+1` numbers where one on a sphere needs `(L+1)²`.
"""
numcoefficients(M::Integer) = 2 * Int(M) + 1

"""
    CircularHarmonicTransform(C, φs, weights, M)
    CircularHarmonicTransform(C, sampling::MLFMA.CircularSampling, M)
    CircularHarmonicTransform(C, M; oversampling=1)

The synthesis matrix `S[q, m] = ψ_m(φ_q)` together with the quadrature weights `w_q` that define its
adjoint, for maximum order `M` and element type `C <: Complex`.

The harmonics are ORTHONORMAL on the circle,

    ψ_m(φ) = exp(i m φ) / √(2π)        ∫ψ_m conj(ψ_m′) dφ = δ_{m m′}

so Parseval holds as `Σ_m |a_m|² = ∫|f|² dφ` and [`modeenergy`](@ref) is an honest `L²` measure. That
costs a `1/√(2π)` that a bare DFT does not carry, and buys the truncation rule its meaning.

`φs` is any vector of angles in the same linear order the sampled fields use; the `sampling` form
reads both angles and weights from the grid, whose order is the one 2-D far fields are stored in. The
third form builds the transform on [`circulargrid`](@ref) for analysis during setup.

For band-limited patterns, the circle's trapezoidal rule is exact when the grid resolves the
analysis integrand. General patterns can still alias. See this file's header.
"""
struct CircularHarmonicTransform{T<:Real,C<:Complex}
    order::Int
    synthesis::Matrix{C}
    weights::Vector{T}

    function CircularHarmonicTransform(
        order::Integer, synthesis::Matrix{C}, weights::Vector{T}
    ) where {T<:Real,C<:Complex}
        order >= 0 || throw(ArgumentError("order must be non-negative, got $order"))
        size(synthesis, 1) == length(weights) || throw(
            DimensionMismatch(
                "synthesis has $(size(synthesis, 1)) rows but there are $(length(weights)) weights",
            ),
        )
        size(synthesis, 2) == numcoefficients(order) || throw(
            DimensionMismatch(
                "synthesis has $(size(synthesis, 2)) columns, expected " *
                "$(numcoefficients(order)) = 2M+1 for order $order",
            ),
        )
        return new{T,C}(Int(order), synthesis, weights)
    end
end

order(t::CircularHarmonicTransform) = t.order
numcoefficients(t::CircularHarmonicTransform) = numcoefficients(t.order)
numdirections(t::CircularHarmonicTransform) = length(t.weights)

function CircularHarmonicTransform(
    ::Type{C}, φs::AbstractVector, weights::AbstractVector{T}, M::Integer
) where {T<:Real,C<:Complex}
    length(φs) == length(weights) ||
        throw(DimensionMismatch("φs and weights must have equal length"))
    nc = numcoefficients(M)
    S = Matrix{C}(undef, length(φs), nc)
    norm = one(real(C)) / sqrt(2 * real(C)(π))
    @inbounds for q in eachindex(φs)
        φ = φs[q]
        for k in 1:nc
            S[q, k] = norm * cis(orderof(k, M) * φ)
        end
    end
    return CircularHarmonicTransform(M, S, collect(T, weights))
end

function CircularHarmonicTransform(
    ::Type{C}, sampling::Sampling.CircularSampling, M::Integer
) where {C<:Complex}
    # Read weights from their owner rather than rebuilding `2π/Nφ` here.
    # `MLFMA.φs` has the same order as stored 2-D far fields; see `Base.enumerate(::CircularSampling)`.
    _, w = Sampling.cartesiansamplepoints(sampling)
    return CircularHarmonicTransform(C, Sampling.φs(sampling), w, M)
end

"""
    circulargrid(M; oversampling=1) -> (φs, weights)

The angles and weights on which analysis up to order `M` is EXACT: `NφCircular(M) = 2M+2` equidistant
points with the uniform trapezoidal weight `2π/N`.

Exact rather than merely convergent, because the circle carries no Jacobian: the trapezoidal rule
integrates every mode `|m| < N` with no error, and the analysis integrand of a pattern band-limited to
`M` reaches only `2M < N`. This is `MLFMA.NφCircular` -- the SAME rate rule the MLFMA level grids use,
which is not a coincidence but the same exactness argument applied to a different product.

`oversampling` multiplies `N` to reduce aliasing. A real basis
function's pattern is not band-limited at all, so its content above `N/2` folds back onto the retained
modes; more points push that floor down. `SphericalHarmonics.gausslegendregrid` is also exact
for its stated band limit; oversampling helps when a pattern has content beyond that limit.
"""
function circulargrid(M::Integer; oversampling::Integer=1)
    M >= 0 || throw(ArgumentError("order must be non-negative, got $M"))
    oversampling >= 1 ||
        throw(ArgumentError("oversampling must be >= 1, got $oversampling"))
    N = oversampling * Sampling.NφCircular(Int(M))
    φs = [2π * (j - 1) / N for j in 1:N]
    weights = fill(2π / N, N)
    return φs, weights
end

"""
    CircularHarmonicTransform(C, M; oversampling=1)

The transform on [`circulargrid`](@ref), for computing coefficients during setup.
"""
function CircularHarmonicTransform(
    ::Type{C}, M::Integer; oversampling::Integer=1
) where {C<:Complex}
    φs, w = circulargrid(M; oversampling=oversampling)
    return CircularHarmonicTransform(C, φs, w, M)
end

"""
    synthesize(t, a) -> F

Evaluate the expansion `a` at the transform's angles: `F = S a`.

`a` may be a vector of `2M+1` coefficients or a matrix whose columns are independent components, in
which case the result carries the same columns.
"""
synthesize(t::CircularHarmonicTransform, a::AbstractVecOrMat) = t.synthesis * a

"""
    synthesize!(F, t, a) -> F

In-place [`synthesize`](@ref).
"""
function synthesize!(F::AbstractVecOrMat, t::CircularHarmonicTransform, a::AbstractVecOrMat)
    return LinearAlgebra.mul!(F, t.synthesis, a)
end

"""
    analyze(t, F) -> a

Project sampled values onto the harmonics: `a = Sᴴ (w .* F)`.

This is the exact adjoint of [`synthesize`](@ref) with respect to the weighted inner product.
It also recovers coefficients exactly when the angles and weights resolve the analysis integrand.
Custom grids need not satisfy that condition.
"""
function analyze(t::CircularHarmonicTransform, F::AbstractVecOrMat)
    size(F, 1) == numdirections(t) || throw(
        DimensionMismatch(
            "F has $(size(F, 1)) rows but the transform has $(numdirections(t)) directions",
        ),
    )
    return t.synthesis' * (t.weights .* F)
end

"""
    modeparity(M) -> Vector{Int}

`(-1)^m` for `m = -M:M`, in the centered row order -- the diagonal of the ANTIPODAL map in coefficient
space.

`ψ_m(φ + π) = e^{im(φ+π)}/√(2π) = (-1)^m ψ_m(φ)`, so the mirror `k̂ → -k̂` that the sampled
representation realises as a permutation of `Nφ` samples is, here, one sign per mode. The 2-D
counterpart of `SphericalHarmonics.parity`'s `(-1)^l` -- and that is NOT a cosmetic substitution: on
the sphere the sign is per DEGREE and every order within a degree shares it, while on the circle
degree and order are the same index.
"""
modeparity(M::Integer) = [iseven(orderof(k, M)) ? 1 : -1 for k in 1:numcoefficients(M)]

"""
    mirrorsynthesis(t) -> Matrix

`S · diag((-1)^m)` -- the synthesis matrix with the antipodal parity folded in, so synthesising with
it evaluates the expansion at `-k̂` instead of at `k̂`.

What the Galerkin receive side uses in place of a mirror-index permutation: the mirror is DIAGONAL in
coefficient space (see [`modeparity`](@ref)), so it costs one matrix built once per level rather than
a gather per leaf per matvec.
"""
function mirrorsynthesis(t::CircularHarmonicTransform)
    signs = modeparity(order(t))
    S = copy(t.synthesis)
    @inbounds for k in axes(S, 2)
        signs[k] == 1 && continue
        for q in axes(S, 1)
            S[q, k] = -S[q, k]
        end
    end
    return S
end

"""
    modeenergy(a) -> Vector

Total `|coefficient|²` per `|m|`, summed over the two signs and over the columns of `a`, returned as a
`0:M`-indexed-by-position vector (entry `|m|+1` is mode `|m|`).

Both signs together, because `±m` are one physical angular rate and a truncation keeps or drops them
as a pair -- an expansion truncated to `M′` is `-M′:M′`. Meaningful as an error measure because the
harmonics are orthonormal: by Parseval this is the `L²([0,2π))` energy of the corresponding piece of
the field.
"""
function modeenergy(a::AbstractVecOrMat)
    n = size(a, 1)
    isodd(n) || throw(
        ArgumentError(
            "$n coefficients is not 2M+1 for any integer M -- expected an odd count"
        ),
    )
    M = (n - 1) ÷ 2
    energy = zeros(real(eltype(a)), M + 1)
    for q in axes(a, 1)
        acc = zero(real(eltype(a)))
        for c in axes(a, 2)
            acc += abs2(a[q, c])
        end
        energy[abs(orderof(q, M)) + 1] += acc
    end
    return energy
end

"""
    truncationorder(a, ε) -> Int

Smallest `M` whose discarded tail holds less than `ε²` of the total energy:
`Σ_{|m|>M} energy / Σ_m energy < ε²`.

The squaring is deliberate -- `ε` is a relative error on the FIELD, and energy is quadratic in it, so
this returns the order at which the reconstruction is accurate to about `ε`. Returns the full order
present in `a` if no smaller one qualifies, and `0` for an all-zero expansion.
"""
function truncationorder(a::AbstractVecOrMat, ε::Real)
    energy = modeenergy(a)
    total = sum(energy)
    total > 0 || return 0
    tail = total
    for M in 0:(length(energy) - 1)
        tail -= energy[M + 1]
        tail / total < ε^2 && return M
    end
    return length(energy) - 1
end

"""
    CHETAILFLOOR

The smallest relative tail [`leveltruncationorder`](@ref) will ask for, `1e-7`.

This matches `SphericalHarmonics.SHETAILFLOOR` because the numerical limit does not depend on geometry.
The tail criterion compares against `τ²`, so `τ = 1e-8` demands a relative energy below `1e-16`,
near round-off in `Float64`. Cancellation can keep the computed tail above that threshold, causing
[`truncationorder`](@ref) to return the full order.
"""
const CHETAILFLOOR = 1e-7

"""
    CHETAILRATIO

The factor by which [`leveltruncationorder`](@ref) tightens the map's `ε` before reading the tail:
`τ = max(ε/CHETAILRATIO, CHETAILFLOOR)`, with `CHETAILRATIO = 10`.

The 2-D calibration differs from the spherical rule, which uses `ε/100`.
`benchmark/interpolation/checalibration.jl` measures 20 rows: five `meshcircle`
configurations spanning two box sizes (λ/4 and λ/2) and leaf occupancies 10, 20, 40, 40 and 81, each
at ε = 1e-3, 1e-4, 1e-5 and 1e-6. `M*` is the smallest kept order whose worst GROUP MOMENT error
meets ε:

    box   N_j    ε=1e-3          ε=1e-4          ε=1e-5          ε=1e-6
                M*  r1 r10 r100  M*  r1 r10 r100  M*  r1 r10 r100  M*  r1 r10 r100
    λ/4    10    6   6   7   8    7   7   8   9    8   8   9  10    9   9  10  10
    λ/4    20    6   6   7   8    8   7*  8   9    9   8*  9  10   10   9* 10  10
    λ/4    40    6   6   7   8    8   7*  8   9    9   8*  9  10   10   9* 10  10
    λ/2    40    9   8* 10  11   10  10  11  13   12  11* 13  14   13  13  14  14
    λ/2    81    9   8* 10  11   10  10  11  13   12  11* 13  14   13  13  14  14
                                                          (* = UNDER-predicts M*)

`τ = ε` under-predicts in 8 of the 20 rows, always by exactly 1. `τ = ε/10` never under-predicts and
over-predicts by at most 1. `τ = ε/100` never under-predicts either but over-predicts `M*` by up
to 3, which costs up to 17 % more coefficients than `ε/10` does: at ε = 1e-4 on the λ/2 box it asks
for 27 where 23 suffice, i.e. compression 1.48x instead of 1.74x.

In these measurements, the order depends mainly on box size and ε, not leaf occupancy.
The λ/4 rows at 20 and 40 basis functions per leaf are identical,
as are the λ/2 rows at 40 and 81; only the sparsest leaf (10) differs, and by one order. `M*`
saturates -- once a leaf holds enough functions to cover its box, the group moment's bandwidth is the
BOX's, not any individual function's. That is also why this rule has support the 3-D one explicitly
lacks: `SHETAILFLOOR`'s calibration note records that all three of its valid configurations shared a
single box size, so its dependence on box size is unmeasured. This one is measured across two.

The calibration measures group-moment error at the leaf, not the error of a full matvec.
End-to-end measurements are needed to check whether this constant remains suitable.
Under-prediction is the dangerous direction -- it silently spends the map's error budget -- so the
rule is biased against it here exactly as it is in 3-D.
"""
const CHETAILRATIO = 10

"""
    leveltruncationorder(coefficients, ε; tailfloor=CHETAILFLOOR, ratio=CHETAILRATIO) -> Int

The expansion order to keep for a whole LEVEL: the largest order any of its boxes needs.

`coefficients` is an iterable of per-box coefficient matrices, each `(2M+1) × ncolumns`.

All boxes in a level must use one order because the leaf multiply synthesises a whole chunk
against one matrix `S`. The maximum supplies enough coefficients for every box in the batch.

See [`CHETAILRATIO`](@ref) for what `τ = max(ε/ratio, tailfloor)` was calibrated against and where it
differs from the spherical rule.
"""
function leveltruncationorder(
    coefficients, ε::Real; tailfloor::Real=CHETAILFLOOR, ratio::Real=CHETAILRATIO
)
    ε > 0 || throw(ArgumentError("ε must be positive, got $ε"))
    ratio > 0 || throw(ArgumentError("ratio must be positive, got $ratio"))
    isempty(coefficients) && throw(ArgumentError("need at least one box's coefficients"))
    τ = max(ε / ratio, tailfloor)
    return maximum(truncationorder(a, τ) for a in coefficients)
end

"""
    truncatecoefficients(a, M) -> AbstractVecOrMat

The order-`≤ M` part of an expansion: the CENTERED rows `(M₀-M+1):(M₀+M+1)`, where `M₀` is the order
`a` was analysed at.

A contiguous `view` and not a gather, but a center slice rather than the spherical side's leading one,
because `m` runs `-M:M` here -- see this file's header for why that ordering was kept. One analysis at
a generous order therefore still serves every candidate order below it.
"""
function truncatecoefficients(a::AbstractVecOrMat, M::Integer)
    n = size(a, 1)
    isodd(n) || throw(
        ArgumentError(
            "$n coefficients is not 2M+1 for any integer M -- expected an odd count"
        ),
    )
    M₀ = (n - 1) ÷ 2
    M >= 0 || throw(ArgumentError("cannot truncate to a negative order, got $M"))
    M <= M₀ || throw(
        ArgumentError(
            "cannot truncate to order $M ($(numcoefficients(M)) coefficients): the expansion " *
            "only has $n rows, i.e. order $M₀",
        ),
    )
    rows = (M₀ - M + 1):(M₀ + M + 1)
    return a isa AbstractVector ? view(a, rows) : view(a, rows, :)
end
