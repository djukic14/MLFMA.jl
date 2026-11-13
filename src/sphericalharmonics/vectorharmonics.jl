# Tangential VECTOR spherical harmonics, and the transform pair over a set of directions.
#
# Eibert's SE-MLFMA expands the three Cartesian components of a leaf pattern in scalar harmonics. The
# tangential components θ̂/φ̂ carry the frame's pole singularity, so expanding `F_θ` and `F_φ`
# independently in scalar harmonics converges badly. A tangential vector basis avoids that problem:
# its basis functions carry the angular structure. A tangential field can be smooth as a vector
# field even when its θ/φ components are singular at the poles.
#
# The coefficient count predicts a 33 % saving:
#
#     N_VSH = 2 P(P+2)          against          N_Cartesian = 3 (P+1)²
#
# The measured saving is TWICE that, because VSH also needs a LOWER DEGREE. On real RWG patterns
# (0.8λ sphere, mesh λ/8, λ/4 leaves, both bases analysed from the SAME assembled Cartesian samples on
# the SAME grid, so only the basis differs):
#
#     ε        P_C   3(P_C+1)²    P_V   2P_V(P_V+2)   VSH/Eibert
#     1e-3       7         192      6            96       0.500x
#     1e-4       8         243      7           126       0.519x
#     1e-5       9         300      9           198       0.660x
#
# The structural reason is that MLFMA's far field is TANGENTIAL BY CONSTRUCTION. Three Cartesian
# scalars encode two degrees of freedom plus the constraint that ties them, and the constraint costs
# bandwidth of its own; the tangential basis encodes exactly the free ones. Its degree spectrum sits
# 5-10x below the Cartesian one at every degree. NOTE the advantage narrows as ε tightens -- at 1e-5
# the two bases want the same degree -- so the figure must be quoted with its tolerance.
#
# Two matrices suffice. Writing out the components,
#
#     X_θ = -i m Y /(√(l(l+1)) sinθ)      X_φ =  ∂_θY / √(l(l+1))
#     Ψ_θ =  ∂_θY / √(l(l+1))             Ψ_φ =  i m Y /(√(l(l+1)) sinθ)
#
# so `X_φ == Ψ_θ` and `Ψ_φ == -X_θ`. Two distinct blocks serve all four:
#
#     A = X_θ,   D = ∂_θY/√(l(l+1))
#     F_θ = A a + D b            F_φ = D a - A b
#
# which is the block operator `[[A, D], [D, -A]]`. Analysis applies its adjoint to weighted samples,
# using the same blocks as synthesis. See `transform.jl`.
#
# Sampling must exclude the poles: `1/sinθ` appears in `A` by definition, so
# `θ = 0` and `θ = π` are excluded outright -- both `gausslegendregrid` and the midpoint-equidistant
# grid avoid them, so no MLFMA grid reaches one. What that leaves is an accuracy question, and it
# answers itself: `Y_lm ~ (sinθ)^|m|` cancels the `1/sinθ` exactly, so nothing is amplified. Measured
# on the grids this is used with, `cond([[A,D],[D,-A]])` is 2.4-2.6 and a synthesis/analysis round
# trip returns 2e-15 at analysis orders P+2 through P+8 (min sinθ 0.12-0.18).

"""
    vshindex(l, m) -> Int

1-based contiguous index of the tangential mode `(l, m)`: `l² + l + m`.

The scalar [`lmindex`](@ref) shifted down by one, because there is no `l = 0` tangential mode -- a
constant on the sphere has no surface gradient. Modes stay ordered by increasing degree, so
truncating an expansion to degree `P` is still a row slice rather than a gather.
"""
vshindex(l::Integer, m::Integer) = l^2 + l + m

"""
    numvshcoefficients(P) -> Int

Coefficients per FAMILY in a tangential expansion of maximum degree `P`: `(P+1)² - 1 = P(P+2)`.

A full expansion carries two of these, one for `X` and one for `Ψ`, so `2P(P+2)` in total -- against
`3(P+1)²` for the Cartesian scalar expansion of the same field. See this file's header for what that
is worth in practice, which is more than the ratio suggests.
"""
numvshcoefficients(P::Integer) = (P + 1)^2 - 1

"""
    vshdegreeof(v) -> Int

Degree `l` of the tangential mode at 1-based index `v` -- the inverse of [`vshindex`](@ref) in its
first argument, and `isqrt(v)` because dropping the `l = 0` mode shifts the scalar `isqrt(q-1)`.
"""
vshdegreeof(v::Integer) = isqrt(v)

"""
    vshauxiliaries!(Y, D, A, P, θ, φ) -> (Y, D, A)

Fill the scalar harmonics `Y_lm(θ, φ)` and the two tangential building blocks at that direction,
every buffer of length `(P+1)²` and indexed by [`lmindex`](@ref):

  - `D[q] = ∂_θ Y_lm / √(l(l+1))`
  - `A[q] = -i m Y_lm / (√(l(l+1)) sinθ)`

`θ` must not be a pole; see this file's header.

The θ-derivative comes from a recurrence on the ALREADY-NORMALIZED Legendre functions, for the same
reason `ylm!` does -- forming `P_l^m` and its normalization separately overflows and underflows at the
same factorial rate. Starting from `(1-x²)dP/dx = (l+m)P_{l-1}^m - l x P_l^m` and the ratio
`N_lm/N_{l-1,m} = √((2l+1)(l-m)/((2l-1)(l+m)))`:

    sinθ ∂_θ P̄_l^m = l cosθ P̄_l^m - √((2l+1)(l²-m²)/(2l-1)) P̄_{l-1}^m

Both terms carry `e^{imφ}` unchanged, so the recurrence applies to `Y` directly and no separate
Legendre table is needed.
"""
function vshauxiliaries!(
    Y::AbstractVector{C},
    D::AbstractVector{C},
    A::AbstractVector{C},
    P::Integer,
    θ::Real,
    φ::Real,
) where {C<:Complex}
    n = numcoefficients(P)
    (length(Y) >= n && length(D) >= n && length(A) >= n) ||
        throw(ArgumentError("Y, D and A must each hold at least $n entries for degree $P"))
    T = real(C)
    s = T(sin(θ))
    # Use `s <= eps(T)` rather than `iszero(s)`: `sin(Float64(π))` is 1.22e-16, not zero.
    # Treat directions whose sine is at or below `eps` as poles to avoid dividing by round-off.
    #
    # Nowhere near any real grid: the smallest `sinθ` these are built on is ~0.12, and even a
    # thousand-node θ grid stays around 3e-3. So this separates poles from directions; it is not a
    # conditioning threshold, and there are no ill-conditioned rows here -- see this file's header.
    s <= eps(T) && throw(
        DomainError(
            θ,
            "the tangential vector harmonics carry 1/sinθ and are not evaluated at a pole " *
            "(sinθ = $s); both `gausslegendregrid` and the midpoint-equidistant grid avoid " *
            "θ = 0 and θ = π",
        ),
    )
    x = T(cos(θ))
    ylm!(Y, P, θ, φ)

    # The `l = 0` slot has no tangential mode. Zeroed rather than left undefined so a caller that
    # slices by `lmindex` cannot read garbage out of it.
    D[lmindex(0, 0)] = zero(C)
    A[lmindex(0, 0)] = zero(C)

    @inbounds for l in 1:P
        r = one(T) / sqrt(T(l) * (l + 1))
        for m in (-l):l
            q = lmindex(l, m)
            k = sqrt((2l + one(T)) * (l^2 - m^2) / (2l - one(T)))
            previous = abs(m) <= l - 1 ? Y[lmindex(l - 1, m)] : zero(C)
            D[q] = r * (l * x * Y[q] - k * previous) / s
            A[q] = (-im * m * r / s) * Y[q]
        end
    end
    return Y, D, A
end

"""
    VectorSphericalHarmonicTransform(C, θs, φs, weights, P)
    VectorSphericalHarmonicTransform(C, sampling, P)

The tangential synthesis blocks `A` and `D` over a fixed set of directions, together with the
quadrature weights that define the adjoint.

Synthesis writes the two TANGENTIAL components directly:

    F_θ = A a + D b        F_φ = D a - A b

with `a` the `X`-family coefficients and `b` the `Ψ`-family ones, each of length
[`numvshcoefficients`](@ref)`(P)`. There is no Cartesian stage and no frame projection -- unlike the
scalar transform, whose output has to be projected onto θ̂/φ̂ after synthesis.

Read this file's header before choosing `sampling`, and `transform.jl`'s before trusting its weights:
on an equidistant grid they are the O(h²) midpoint rule.
"""
struct VectorSphericalHarmonicTransform{T<:Real,C<:Complex}
    order::Int
    # `A = X_θ = -Ψ_φ` and `D = Ψ_θ = X_φ`; see this file's header for why four component blocks
    # collapse to two.
    A::Matrix{C}
    D::Matrix{C}
    weights::Vector{T}

    function VectorSphericalHarmonicTransform(
        order::Integer, A::Matrix{C}, D::Matrix{C}, weights::Vector{T}
    ) where {T<:Real,C<:Complex}
        size(A) == size(D) || throw(
            DimensionMismatch(
                "A is $(size(A)) but D is $(size(D)); they index the same modes"
            ),
        )
        size(A, 1) == length(weights) || throw(
            DimensionMismatch(
                "the synthesis blocks have $(size(A, 1)) rows but there are " *
                "$(length(weights)) weights",
            ),
        )
        size(A, 2) == numvshcoefficients(order) || throw(
            DimensionMismatch(
                "the synthesis blocks have $(size(A, 2)) columns, expected " *
                "$(numvshcoefficients(order)) for degree $order",
            ),
        )
        return new{T,C}(Int(order), A, D, weights)
    end
end

order(t::VectorSphericalHarmonicTransform) = t.order
numvshcoefficients(t::VectorSphericalHarmonicTransform) = numvshcoefficients(t.order)
numdirections(t::VectorSphericalHarmonicTransform) = length(t.weights)

function VectorSphericalHarmonicTransform(
    ::Type{C},
    θs::AbstractVector,
    φs::AbstractVector,
    weights::AbstractVector{T},
    P::Integer,
) where {T<:Real,C<:Complex}
    length(θs) == length(φs) == length(weights) ||
        throw(DimensionMismatch("θs, φs and weights must have equal length"))
    P >= 1 || throw(ArgumentError("a tangential expansion needs degree >= 1, got $P"))
    n = numvshcoefficients(P)
    Amatrix = Matrix{C}(undef, length(θs), n)
    Dmatrix = Matrix{C}(undef, length(θs), n)
    Y = Vector{C}(undef, numcoefficients(P))
    Drow = Vector{C}(undef, numcoefficients(P))
    Arow = Vector{C}(undef, numcoefficients(P))
    for q in eachindex(θs)
        vshauxiliaries!(Y, Drow, Arow, P, θs[q], φs[q])
        # `lmindex` -> `vshindex` is a shift by one, so the tangential row is the scalar row with its
        # first entry dropped -- which is exactly what makes degree truncation a slice in both.
        @inbounds Amatrix[q, :] .= @view Arow[2:(n + 1)]
        @inbounds Dmatrix[q, :] .= @view Drow[2:(n + 1)]
    end
    return VectorSphericalHarmonicTransform(P, Amatrix, Dmatrix, collect(T, weights))
end

function VectorSphericalHarmonicTransform(
    ::Type{C}, sampling::Sampling.AbstractSphericalSampling, P::Integer
) where {C<:Complex}
    k̂, w = Sampling.sphericalsamplepoints(sampling)
    θs = [p[2] for p in vec(k̂)]
    φs = [p[3] for p in vec(k̂)]
    return VectorSphericalHarmonicTransform(C, θs, φs, vec(w), P)
end

"""
    vshsynthesize!(Fθ, Fφ, t, a, b; mirrored=false) -> (Fθ, Fφ)

`F_θ = A a + D b`, `F_φ = D a - A b` -- the tangential field of an expansion, written straight into
its two component arrays.

`a` and `b` may be vectors (one expansion) or matrices (one per column, which is the batched form the
leaf multiply uses).

With `mirrored=true` the ANTIPODAL field is produced instead, `F(-k̂)` with the tangential mirror rule
already applied. That is a per-degree sign and not a permutation:

    a' = (-1)^l a        b' = (-1)^(l+1) b

so folding it into the blocks flips the sign pattern to `F_θ = Aᵖ a - Dᵖ b`, `F_φ = Dᵖ a + Aᵖ b` with
`Aᵖ = A·diag((-1)^l)`. See [`vshmirrorblocks`](@ref) for the folded blocks themselves, which is what
the receive collection actually uses -- this keyword exists for tests that want the identity stated
in one place.
"""
function vshsynthesize!(
    Fθ, Fφ, t::VectorSphericalHarmonicTransform, a, b; mirrored::Bool=false
)
    A, D = mirrored ? vshmirrorblocks(t) : (t.A, t.D)
    sign = mirrored ? -one(eltype(t.A)) : one(eltype(t.A))
    LinearAlgebra.mul!(Fθ, A, a)
    LinearAlgebra.mul!(Fθ, D, b, sign, true)
    LinearAlgebra.mul!(Fφ, D, a)
    LinearAlgebra.mul!(Fφ, A, b, -sign, true)
    return Fθ, Fφ
end

"""
    vshsynthesize(t, a, b; mirrored=false) -> (Fθ, Fφ)

Allocating [`vshsynthesize!`](@ref).
"""
function vshsynthesize(t::VectorSphericalHarmonicTransform, a, b; mirrored::Bool=false)
    Fθ = similar(t.A, size(t.A, 1), size(a)[2:end]...)
    Fφ = similar(Fθ)
    return vshsynthesize!(Fθ, Fφ, t, a, b; mirrored=mirrored)
end

"""
    vshanalyze(t, Fθ, Fφ) -> (a, b)

The adjoint of [`vshsynthesize!`](@ref) against `t`'s weights:

    a = Aᴴ W F_θ + Dᴴ W F_φ        b = Dᴴ W F_θ - Aᴴ W F_φ

The tangential basis is orthonormal, so this recovers the coefficients exactly when the grid's
quadrature resolves degree `2P`. On other grids it remains the weighted adjoint of synthesis,
not a separately implemented inverse.
"""
function vshanalyze(t::VectorSphericalHarmonicTransform, Fθ, Fφ)
    Wθ = t.weights .* Fθ
    Wφ = t.weights .* Fφ
    a = t.A' * Wθ .+ t.D' * Wφ
    b = t.D' * Wθ .- t.A' * Wφ
    return a, b
end

"""
    vshparity(P) -> Vector{Int}

`(-1)^l` per tangential coefficient index, the antipodal map's action on the `X` family. The `Ψ`
family takes the negative of this.

DERIVED, and then checked against `MLFMA.mirrorindices`/`mirrorrule` in
`test/SphericalHarmonics/`. Under `θ -> π-θ, φ -> φ+π`: `Y -> (-1)^l Y`, `∂_θY -> -(-1)^l ∂_θY`, and
`sinθ` is unchanged. Feeding that through the component formulas together with the tangential mirror
rule (θ̂ even, φ̂ odd) gives `a' = (-1)^l a`, `b' = (-1)^(l+1) b`.
"""
function vshparity(P::Integer)
    p = Vector{Int}(undef, numvshcoefficients(P))
    @inbounds for v in eachindex(p)
        p[v] = iseven(vshdegreeof(v)) ? 1 : -1
    end
    return p
end

"""
    vshmirrorblocks(t) -> (Aᵖ, Dᵖ)

`t`'s synthesis blocks with the antipodal parity folded in, `Aᵖ = A·diag((-1)^l)` and likewise `D`.

The tangential counterpart of [`mirrorsynthesis`](@ref) and used the same way: the Galerkin receive
collection is the radiation one with these blocks substituted, so it stores no data of its own. The
sign PATTERN differs from the plain synthesis as well -- see [`vshsynthesize!`](@ref) -- because the
two families pick up opposite signs.
"""
function vshmirrorblocks(t::VectorSphericalHarmonicTransform)
    p = vshparity(order(t))
    return t.A .* transpose(p), t.D .* transpose(p)
end

"""
    vshdegreeenergy(a, b) -> Vector

Total `|a|² + |b|²` per degree `l = 1..P`, summed over orders and over every column.

Parseval on an orthonormal basis, so this IS the energy of the field's degree-`l` content and needs no
normalization of its own. [`vshtruncationorder`](@ref) reads a tail through it.
"""
function vshdegreeenergy(a::AbstractMatrix, b::AbstractMatrix)
    size(a) == size(b) || throw(
        DimensionMismatch(
            "the two families must have equal size, got $(size(a)) and $(size(b))"
        ),
    )
    P = vshdegreeof(size(a, 1))
    energy = zeros(real(eltype(a)), P)
    @inbounds for v in axes(a, 1)
        l = vshdegreeof(v)
        for j in axes(a, 2)
            energy[l] += abs2(a[v, j]) + abs2(b[v, j])
        end
    end
    return energy
end

vshdegreeenergy(a::AbstractVector, b::AbstractVector) =
    vshdegreeenergy(reshape(a, :, 1), reshape(b, :, 1))

"""
    vshtruncationorder(a, b, ε) -> Int

Smallest degree `P` whose relative error, measured by the square root of the tail energy fraction,
is at most `ε`:

    ε_P² = Σ_{l>P} (|a_lm|² + |b_lm|²) / Σ_l (|a_lm|² + |b_lm|²)

The tangential analogue of [`truncationorder`](@ref), and the same measurement Borries uses for the
scalar case. Returns the full degree if no truncation meets `ε`.
"""
function vshtruncationorder(a, b, ε::Real)
    energy = vshdegreeenergy(a, b)
    total = sum(energy)
    total > 0 || return 1
    for P in 1:length(energy)
        tail = sum(@view energy[(P + 1):end])
        sqrt(tail / total) <= ε && return P
    end
    return length(energy)
end

"""
    vshleveltruncationorder(pairs, ε; tailfloor=SHETAILFLOOR) -> Int

The order a LEVEL needs: the maximum over its leaves of [`vshtruncationorder`](@ref), read at
`τ = max(ε/100, tailfloor)`.

`pairs` is one `(a, b)` tuple per leaf, which is exactly what `vshanalyze` returns.

The factor `ε/100` and the floor come from the scalar rule, not an independent VSH calibration.
See [`leveltruncationorder`](@ref) for the calibration's limits. A per-leaf tail bound alone does
not bound the full map's error. The floor avoids asking for a tail below round-off, which can
force the analysis order to be retained.

ONE ORDER PER LEVEL, not per leaf, because the batched leaf multiply synthesises a whole chunk against
one pair of blocks -- a ragged order has no shared `B`.
"""
function vshleveltruncationorder(pairs, ε::Real; tailfloor::Real=SHETAILFLOOR)
    τ = max(ε / 100, tailfloor)
    return maximum(vshtruncationorder(a, b, τ) for (a, b) in pairs)
end
