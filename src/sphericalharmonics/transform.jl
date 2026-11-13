# The spherical-harmonic transform pair over a fixed set of directions.
#
# Synthesis is `S a`; analysis is `Sᴴ W F`. Both use the same stored `S` and `W`, so analysis
# applies the adjoint of synthesis to weighted samples without a separate reverse kernel.
#
# The leaf pair relies on exact reconstruction. In the intended use, `V_j = Σ_lm
# conj(a_lm) B_lm` reproduces the conventional leaf pairing `Σ_q w_q conj(F_j[q]) G[q]` EXACTLY --
# as algebra, not as an approximated integral -- PROVIDED `B` is formed with the same `w` the
# conventional pairing uses:
#
#     Σ_q w_q (Σ_lm conj(a_lm) conj(Y_lm[q])) G[q] = Σ_lm conj(a_lm) (Σ_q w_q conj(Y_lm[q]) G[q])
#
# It is a reordering of a finite sum. In particular it does NOT require the quadrature to integrate
# `G·conj(Y_lm)` exactly -- that product has bandwidth `L + P`, which the leaf grid does not resolve,
# and needing it to would sink the whole scheme.
#
# Computing coefficients during setup requires quadrature that resolves the integrand's bandwidth,
# about `2P`. Use `gausslegendregrid` below. The grids from `MLFMA.sphericalsamplepoints` differ:
#
#   * On a `SphericalSampling` it places Gauss-Legendre nodes in `θ` (`θs = π/2 (x+1)`) and carries
#     `sin θ` as an explicit weight factor. The integrand is then `f(cos θ)·sin θ`, which is NOT a
#     polynomial in `θ`, so no finite-`Nθ` exactness theorem applies. It still CONVERGES -- the
#     integrand is analytic, so convergence is spectral -- but it needs about 4x the θ nodes to get
#     there. Measured round-trip error at degree 6: `Nθ = P+1` gives 1.2e0, `2P+2` gives 1.6e-3,
#     `3P+3` gives 1.6e-10, `4P+4` finally reaches 1.3e-15. `gausslegendregrid` is at 5.5e-15 from
#     `Nθ = P+1`. So this grid is usable for setup analysis if heavily oversampled, and simply
#     wasteful compared to substituting `x = cos θ`.
#   * On an equidistant grid it gives the plain midpoint rule, only O(h²) here -- extending θ to the
#     torus turns `sin θ` into `|sin θ|`, whose kink at the poles costs the rule its spectral accuracy
#     (0.29 % on `∮dΩ = 4π` at `Nθ = 12`). `MLFMA.Equidistant.exactquadratureweights` fixes that one,
#     via the explicit-weights constructor below.
#
# Substituting `x = cos θ` removes the Jacobian instead of carrying it, which is what makes
# `gausslegendregrid` exact -- see its docstring.

"""
    SphericalHarmonicTransform(C, directions, weights, P)
    SphericalHarmonicTransform(C, sampling, P)

The synthesis matrix `S[q, lm] = Y_lm(direction q)` together with the quadrature weights `w_q` that
define its adjoint, for maximum degree `P` and element type `C <: Complex`.

`directions` is any iterable of `(θ, φ)`-bearing points in the SAME linear order the sampled fields
use; `weights` is one weight per direction. The `sampling` form reads both from
`MLFMA.sphericalsamplepoints`, whose `k̂[q] = (1, θ, φ)` and whose linear ordering is the one far
fields are stored in.

**Read this file's header before choosing `sampling`**: on an equidistant grid those weights are the
O(h²) midpoint rule, not an exact one.
"""
struct SphericalHarmonicTransform{T<:Real,C<:Complex}
    order::Int
    synthesis::Matrix{C}
    weights::Vector{T}

    function SphericalHarmonicTransform(
        order::Integer, synthesis::Matrix{C}, weights::Vector{T}
    ) where {T<:Real,C<:Complex}
        size(synthesis, 1) == length(weights) || throw(
            DimensionMismatch(
                "synthesis has $(size(synthesis, 1)) rows but there are $(length(weights)) weights",
            ),
        )
        size(synthesis, 2) == numcoefficients(order) || throw(
            DimensionMismatch(
                "synthesis has $(size(synthesis, 2)) columns, expected " *
                "$(numcoefficients(order)) for degree $order",
            ),
        )
        return new{T,C}(Int(order), synthesis, weights)
    end
end

order(t::SphericalHarmonicTransform) = t.order
numcoefficients(t::SphericalHarmonicTransform) = numcoefficients(t.order)
numdirections(t::SphericalHarmonicTransform) = length(t.weights)

function SphericalHarmonicTransform(
    ::Type{C},
    θs::AbstractVector,
    φs::AbstractVector,
    weights::AbstractVector{T},
    P::Integer,
) where {T<:Real,C<:Complex}
    length(θs) == length(φs) == length(weights) ||
        throw(DimensionMismatch("θs, φs and weights must have equal length"))
    S = Matrix{C}(undef, length(θs), numcoefficients(P))
    row = Vector{C}(undef, numcoefficients(P))
    for q in eachindex(θs)
        ylm!(row, P, θs[q], φs[q])
        @inbounds S[q, :] .= row
    end
    return SphericalHarmonicTransform(P, S, collect(T, weights))
end

function SphericalHarmonicTransform(
    ::Type{C}, sampling::Sampling.AbstractSphericalSampling, P::Integer
) where {C<:Complex}
    k̂, w = Sampling.sphericalsamplepoints(sampling)
    # Preserve the linear order of `k̂` so `analyze` accepts stored far-field components
    # without a permutation.
    θs = [p[2] for p in vec(k̂)]
    φs = [p[3] for p in vec(k̂)]
    return SphericalHarmonicTransform(C, θs, φs, vec(w), P)
end

"""
    gausslegendregrid(P; oversampling=1) -> (θs, φs, weights)

A spherical grid on which analysis up to degree `P` is EXACT: Gauss-Legendre in `cos θ` with `P+1`
nodes, times a `2P+2`-point trapezoidal rule in `φ`, flattened φ-fastest to match the linear order
`MLFMA.sphericalsamplepoints` uses.

Exact because the substitution `x = cos θ` removes the Jacobian entirely -- `∮f dΩ = ∫∫f dx dφ` -- so
a product of two harmonics of degree `≤ P` is a polynomial of degree `≤ 2P` in `x` and `P+1`
Gauss-Legendre nodes integrate it with no error at all.

**This is NOT what `MLFMA.sphericalsamplepoints` gives you, and the difference is a factor of four in
θ nodes.** That one places Gauss-Legendre nodes in `θ` and carries `sin θ` in the weight, so its
integrand is not polynomial and it has no exactness property -- only spectral convergence. It reaches
round-off around `Nθ = 4(P+1)` where this grid is there at `Nθ = P+1`. See this file's header for the
measured numbers.

`oversampling` multiplies both counts, for the `2P`-bandwidth setup integrand of a pattern whose own
degree is only approximately known.
"""
function gausslegendregrid(P::Integer; oversampling::Integer=1)
    P >= 0 || throw(ArgumentError("degree must be non-negative, got $P"))
    oversampling >= 1 ||
        throw(ArgumentError("oversampling must be >= 1, got $oversampling"))
    Nθ = oversampling * (P + 1)
    Nφ = oversampling * (2P + 2)
    x, wx = FastGaussQuadrature.gausslegendre(Nθ)
    θs = Vector{Float64}(undef, Nθ * Nφ)
    φs = similar(θs)
    weights = similar(θs)
    for i in 1:Nθ, j in 1:Nφ
        q = (i - 1) * Nφ + j
        θs[q] = acos(x[i])
        φs[q] = 2π * (j - 1) / Nφ
        weights[q] = wx[i] * 2π / Nφ
    end
    return θs, φs, weights
end

"""
    SphericalHarmonicTransform(C, P; oversampling=1)

The transform on [`gausslegendregrid`](@ref). Use it to compute coefficients during setup,
where quadrature accuracy is required as well as adjoint consistency.
"""
function SphericalHarmonicTransform(
    ::Type{C}, P::Integer; oversampling::Integer=1
) where {C<:Complex}
    θs, φs, w = gausslegendregrid(P; oversampling=oversampling)
    return SphericalHarmonicTransform(C, θs, φs, w, P)
end

"""
    synthesize(t, a) -> F

Evaluate the expansion `a` at the transform's directions: `F = S a`.

`a` may be a vector of `(P+1)²` coefficients or a matrix whose columns are independent components
(the Cartesian `x, y, z` of a vector pattern, say), in which case the result carries the same columns.
"""
synthesize(t::SphericalHarmonicTransform, a::AbstractVecOrMat) = t.synthesis * a

"""
    synthesize!(F, t, a) -> F

In-place [`synthesize`](@ref).
"""
function synthesize!(
    F::AbstractVecOrMat, t::SphericalHarmonicTransform, a::AbstractVecOrMat
)
    return LinearAlgebra.mul!(F, t.synthesis, a)
end

"""
    analyze(t, F) -> a

Project sampled values onto the harmonics: `a = Sᴴ (w .* F)`.

This is the exact adjoint of [`synthesize`](@ref) with respect to the weighted inner product, by
construction rather than by a matching implementation. See this file's header for when that adjointness
is all that is needed and when genuine quadrature exactness is required as well.
"""
function analyze(t::SphericalHarmonicTransform, F::AbstractVecOrMat)
    size(F, 1) == numdirections(t) || throw(
        DimensionMismatch(
            "F has $(size(F, 1)) rows but the transform has $(numdirections(t)) directions",
        ),
    )
    return t.synthesis' * (t.weights .* F)
end

"""
    degreeenergy(a) -> Vector

Total `|coefficient|²` per degree, summed over orders `m` and over the columns of `a`, returned as a
`0:P`-indexed-by-position vector (entry `l+1` is degree `l`).

Meaningful as an error measure only because the harmonics are orthonormal: by Parseval it is the
`L²(S²)` energy of the corresponding piece of the field.
"""
function degreeenergy(a::AbstractVecOrMat)
    P = isqrt(size(a, 1)) - 1
    numcoefficients(P) == size(a, 1) ||
        throw(ArgumentError("$(size(a, 1)) coefficients is not (P+1)² for any integer P"))
    energy = zeros(real(eltype(a)), P + 1)
    for q in axes(a, 1)
        acc = zero(real(eltype(a)))
        for c in axes(a, 2)
            acc += abs2(a[q, c])
        end
        energy[degreeof(q) + 1] += acc
    end
    return energy
end

"""
    truncationorder(a, ε) -> Int

Smallest `P` whose discarded tail holds less than `ε²` of the total energy:
`Σ_{l>P} energy / Σ_l energy < ε²`.

The squaring is deliberate -- `ε` is a relative error on the FIELD, and energy is quadratic in it, so
this returns the order at which the reconstruction is accurate to about `ε`. Returns the full degree
present in `a` if no smaller order qualifies, and `0` for an all-zero expansion.
"""
function truncationorder(a::AbstractVecOrMat, ε::Real)
    energy = degreeenergy(a)
    total = sum(energy)
    total > 0 || return 0
    tail = total
    for P in 0:(length(energy) - 1)
        tail -= energy[P + 1]
        tail / total < ε^2 && return P
    end
    return length(energy) - 1
end

"""
    SHETAILFLOOR

The smallest relative tail [`leveltruncationorder`](@ref) will ask for, `1e-7`.

This is a numerical limit, not a safety margin. The tail criterion compares against `τ²`, so
`τ = 1e-8` demands a relative energy below `1e-16`, near round-off. Cancellation can keep the
computed tail above that threshold, causing [`truncationorder`](@ref) to return the full degree.
See the three ceiling values at `ε = 1e-6` in the last column of
`benchmark/interpolation/shecalibration.jl`.
"""
const SHETAILFLOOR = 1e-7

"""
    leveltruncationorder(coefficients, ε; tailfloor=SHETAILFLOOR) -> Int

The expansion order to keep for a whole LEVEL: the largest order any of its boxes needs.

`coefficients` is an iterable of per-box coefficient matrices, each `(P+1)² × ncolumns`.

All boxes in a level must use one order because the leaf multiply synthesises a whole chunk
against one matrix `S`. The maximum supplies enough coefficients for every box in the batch.

`τ = max(ε/100, tailfloor)` uses a calibrated factor of one hundredth. Only three of the nine
calibration configurations are valid: the other six have boxes below λ/4 halfsize, within MLFMA's
low-frequency breakdown (see `benchmark/interpolation/shecalibration.jl`). On the valid cases,
the rule never under-predicts and over-predicts by at most 1. All three share one box size, so
dependence on box size remains unmeasured. Using `τ = ε` instead
under-predicts by up to 3 orders, which silently spends the map's error budget: at λ/4 it picks P = 9,
which `benchmark/interpolation/sheaccuracy.jl` measures at 15 % worse far-field error than the sampled
path, where P = 10 is indistinguishable from it. Under-prediction is the dangerous direction, so the
rule is deliberately biased against it. Full table in `benchmark/interpolation/shecalibration.jl`.
"""
function leveltruncationorder(coefficients, ε::Real; tailfloor::Real=SHETAILFLOOR)
    ε > 0 || throw(ArgumentError("ε must be positive, got $ε"))
    isempty(coefficients) && throw(ArgumentError("need at least one box's coefficients"))
    τ = max(ε / 100, tailfloor)
    return maximum(truncationorder(a, τ) for a in coefficients)
end

"""
    truncatecoefficients(a, P) -> AbstractMatrix

The degree-`≤ P` part of an expansion: the first `(P+1)²` rows.

A plain row slice because of how [`lmindex`](@ref) orders things -- `l² + l + m + 1` puts every
coefficient of degree `≤ P` in the leading `(P+1)²` entries -- so truncating needs no gather, and one
analysis at a generous order serves every candidate order below it.
"""
function truncatecoefficients(a::AbstractVecOrMat, P::Integer)
    nc = numcoefficients(P)
    nc <= size(a, 1) || throw(
        ArgumentError(
            "cannot truncate to order $P ($(nc) coefficients): the expansion only has " *
            "$(size(a, 1)) rows, i.e. degree $(isqrt(size(a, 1)) - 1)",
        ),
    )
    return a isa AbstractVector ? view(a, 1:nc) : view(a, 1:nc, :)
end
