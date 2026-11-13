# The diagonal MLFMA translation operator, sampled on the equidistant grid.
#
# T_L(k̂·d̂) = Σ_{l=0}^{L} (2l+1) iˡ h_l⁽¹⁾(k d) P_l(k̂·d̂)
#
# and the identity it serves, valid for |u| < |d| (Gegenbauer / plane-wave addition theorem):
#
#     e^{ik|d+u|} / |d+u| = (ik/4π) ∮ dΩ(k̂) e^{ik k̂·u} T_L(k̂·d̂)
#
# Filtering controls which modes are retained. T_L is exactly band-limited
# to degree L: it is a finite Legendre sum, so its θ-content on the grid is L and no more. Sampling it
# on a grid with Nθ ≥ L+1 is therefore exact, and no post-hoc spectral filtering is needed -- the
# truncation IS the filter, applied analytically at construction. Sampling it on a COARSER grid
# aliases the top modes and silently corrupts the operator, which is why `translationoperator`
# refuses rather than returning a plausible-looking wrong answer.
#
# The integrand is the product of T_L (bandwidth L) with the pattern e^{ik k̂·u} (bandwidth ≈ k|u|),
# so the QUADRATURE has to resolve L + k|u|, not just L. That is a strictly stronger requirement than
# representing either factor, and it is why the grid used for the integral is oversampled relative to
# L in the tests. The quadrature must also be the spectrally exact rule from `quadrature.jl`; the
# plain midpoint rule converges only algebraically as the grid is refined.

"""
    sphericalhankel1(L, x) -> Vector

`h_l⁽¹⁾(x) = j_l(x) + i y_l(x)` for `l = 0:L`, by upward recurrence.

Upward recurrence is unstable for `j_l` but STABLE for `h_l⁽¹⁾`, whose magnitude grows with `l` --
the direction of growth is what makes the recurrence well conditioned here.
"""
function sphericalhankel1(L::Int, x::Real)
    T = float(typeof(x))
    h = Vector{Complex{T}}(undef, L + 1)
    e = cis(x) / x
    h[1] = -im * e
    L >= 1 && (h[2] = -(one(T) + im / x) * e)
    for l in 1:(L - 1)
        h[l + 2] = (2l + 1) / x * h[l + 1] - h[l]
    end
    return h
end

"""
    legendrep(L, x) -> Vector

`P_l(x)` for `l = 0:L` by the standard three-term recurrence.
"""
function legendrep(L::Int, x::T) where {T<:Real}
    p = Vector{T}(undef, L + 1)
    p[1] = one(T)
    L >= 1 && (p[2] = x)
    for l in 1:(L - 1)
        p[l + 2] = ((2l + 1) * x * p[l + 1] - l * p[l]) / (l + 1)
    end
    return p
end

"""
    directions(sampling) -> Matrix

The unit vector `k̂(θᵢ, φⱼ)` at every grid point, as an `Nφ × Nθ` matrix of 3-tuples.
"""
function directions(s::EquidistantSampling{T}) where {T}
    return [(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)) for φ in s.φs, θ in s.θs]
end

"""
    translationoperator(L, kd, dhat, sampling) -> Matrix

`T_L(k̂·d̂)` sampled on `sampling`, with `kd = k|d|` and `dhat` the unit separation vector.

Errors when `Nθ < L + 1`: `T_L` carries content up to degree `L` exactly, so a coarser grid aliases
it. See this file's header on why truncation at `L` is itself the correct filtering.
"""
function translationoperator(L::Int, kd::Real, dhat, s::EquidistantSampling)
    Nθ(s) >= L + 1 || error(
        "translationoperator: Nθ = $(Nθ(s)) cannot represent degree L = $L without aliasing; " *
        "need Nθ >= L + 1.",
    )
    h = sphericalhankel1(L, kd)
    coeff = [(2l + 1) * im^l * h[l + 1] for l in 0:L]
    return map(directions(s)) do k̂
        c = k̂[1] * dhat[1] + k̂[2] * dhat[2] + k̂[3] * dhat[3]
        p = legendrep(L, clamp(c, -one(c), one(c)))
        return sum(coeff[l + 1] * p[l + 1] for l in 0:L)
    end
end

"""
    excessbandwidth(ku, digits) -> Int

The usual MLFMA truncation `L ≈ ku + d₀^{2/3} (ku)^{1/3}` for `digits` of accuracy.
"""
excessbandwidth(ku::Real, digits::Real) = ceil(Int, ku + digits^(2 / 3) * cbrt(ku))
