# Orthonormal complex spherical harmonics, evaluated by a recurrence on the ALREADY-NORMALIZED
# associated Legendre functions.
#
# The unnormalized sectoral function grows like `(2l-1)!!`, while the normalization factor
# `sqrt((l-m)!/(l+m)!)` shrinks at the same factorial rate. Computing them separately can produce
# `Inf` and `0`, whose product is `NaN`. The recurrence on `P̄_l^m` avoids those factorial
# intermediates. The oracle in `test/SphericalHarmonics/` uses separate normalization at low orders,
# where it remains valid, to provide an independent check.
#
# Use the Condon-Shortley convention: `Y_{l,-m} = (-1)^m conj(Y_{l,m})`. The sign is in the sectoral
# step below and nowhere else.

"""
    lmindex(l, m) -> Int

1-based contiguous index of mode `(l, m)`: `l² + l + m + 1`.

Modes are ordered by increasing degree, and within a degree by increasing order, so every mode of
degree `≤ P` occupies `1:(P+1)²` exactly and truncating an expansion is a `view` rather than a gather.
"""
lmindex(l::Integer, m::Integer) = l^2 + l + m + 1

"""
    numcoefficients(P) -> Int

Number of coefficients in an expansion of maximum degree `P`, i.e. `(P+1)²`.
"""
numcoefficients(P::Integer) = (P + 1)^2

"""
    degreeof(q) -> Int

Degree `l` of the mode stored at 1-based index `q` -- the inverse of [`lmindex`](@ref) in its first
argument. Used to bin coefficient energy by degree without materializing an `(l, m)` table.
"""
degreeof(q::Integer) = isqrt(q - 1)

"""
    ylm!(out, P, θ, φ) -> out

Fill `out[1:(P+1)²]` with the orthonormal complex spherical harmonics `Y_lm(θ, φ)` for every
`0 ≤ l ≤ P`, `-l ≤ m ≤ l`, indexed by [`lmindex`](@ref).
"""
function ylm!(out::AbstractVector{C}, P::Integer, θ::Real, φ::Real) where {C<:Complex}
    T = real(C)
    length(out) >= numcoefficients(P) || throw(
        ArgumentError(
            "out has length $(length(out)), need at least $(numcoefficients(P)) for degree $P",
        ),
    )
    x = T(cos(θ))
    s = T(sin(θ))

    pmm = sqrt(one(T) / (4 * T(π)))                      # P̄_0^0
    for m in 0:P
        # Sectoral step. The minus sign IS the Condon-Shortley phase.
        m > 0 && (pmm = -sqrt((2m + one(T)) / (2m)) * s * pmm)
        _storeylm!(out, m, m, pmm, φ)
        m == P && break

        plm2 = pmm
        plm1 = sqrt(2m + T(3)) * x * pmm                 # P̄_{m+1}^m
        _storeylm!(out, m + 1, m, plm1, φ)

        for l in (m + 2):P
            a = sqrt((4 * l^2 - one(T)) / (l^2 - m^2))
            b = sqrt(((l - 1)^2 - m^2) / (4 * (l - 1)^2 - one(T)))
            pl = a * (x * plm1 - b * plm2)
            _storeylm!(out, l, m, pl, φ)
            plm2, plm1 = plm1, pl
        end
    end
    return out
end

# Writes BOTH signs of the order from one real `P̄_l^m`, via Y_{l,-m} = (-1)^m conj(Y_{l,m}).
# Doing it here rather than in a second pass is what keeps the recurrence over |m| only.
@inline function _storeylm!(out, l::Integer, m::Integer, p::Real, φ::Real)
    e = cis(m * φ)
    out[lmindex(l, m)] = p * e
    m > 0 && (out[lmindex(l, -m)] = (isodd(m) ? -p : p) * conj(e))
    return nothing
end

"""
    ylm(l, m, θ, φ) -> Complex

A single orthonormal `Y_lm(θ, φ)`.

Convenience only: it runs the full recurrence up to degree `l` and keeps one value, so it is `O(l²)`
per call. Anything evaluating a whole expansion should use [`ylm!`](@ref), and anything evaluating one
over many directions should build a [`SphericalHarmonicTransform`](@ref).
"""
function ylm(l::Integer, m::Integer, θ::Real, φ::Real)
    abs(m) <= l || throw(ArgumentError("need |m| <= l, got l=$l, m=$m"))
    buf = Vector{ComplexF64}(undef, numcoefficients(l))
    ylm!(buf, l, θ, φ)
    return buf[lmindex(l, m)]
end
