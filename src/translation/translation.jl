"""
    translationoperator!(
        translationoperator::AbstractArray{T1},
        γ::T1,
        cartesiansamplepoints::AbstractArray{SVector{D,T}},
        originalcenter::SVector{D,T},
        newcenter::SVector{D,T},
        cutoffparameter::Int,
    )

Compute the translation function between two points in space.

# Arguments

  - `translationoperator`: The destination array for the translation operator.
  - `γ`: `im * wavenumber`.
  - `cartesiansamplepoints`: Cartesian sample directions as an array of `SVector{D,T}`.
  - `originalcenter`: The original center point.
  - `newcenter`: The new center point.
  - `cutoffparameter`: The requested truncation order.

# Returns

  - The truncation order used. The operator values are written into `translationoperator`.
"""
function translationoperator!(
    translationoperator::AbstractArray{T1},
    γ::T1,
    cartesiansamplepoints::AbstractArray{SVector{D,T}},
    originalcenter::SVector{D,T},
    newcenter::SVector{D,T},
    cutoffparameter::Int,
) where {D,T1,T}
    return translationoperator!(
        translationoperator,
        γ,
        cartesiansamplepoints,
        originalcenter - newcenter,
        cutoffparameter,
    )
end

# The 3-D addition theorem uses spherical Hankel functions and Legendre polynomials.
# The direction dimension selects this method or the 2-D cylindrical expansion below.
function translationoperator!(
    translationoperator::AbstractArray{T1},
    γ::T1,
    cartesiansamplepoints::AbstractArray{SVector{3,T}},
    R::SVector{3,T},
    cutoffparameter::Int,
) where {T1,T}
    Literations = cutoffparameter

    κ = abs(γ)
    #TODO Optimization: use more refined formula for this: https://onlinelibrary.wiley.com/doi/10.1002/mop.10862
    Literations = min(cutoffparameter, ceil(Int, κ * norm(R)))
    Literations = cutoffparameter

    sphericalhankels = collectsphericalhankelsecond(Literations, κ * norm(R))

    for l in range(0, Literations)
        sphericalhankels[l + 1] *= κ / (4 * pi)^2 * (-im)^(l + 1) * (2 * l + 1)
    end

    R̂ = R / norm(R)

    legendrebefore = T(0)
    legendrebeforebefore = T(0)
    sum = T1(0)

    for (i, k̂) in enumerate(cartesiansamplepoints)
        legendreargument = dot(R̂, k̂)
        legendrebeforebefore = legendre0(legendreargument)
        legendrebefore = legendre1(legendreargument)

        sum =
            legendrebeforebefore * sphericalhankels[1] +
            legendrebefore * sphericalhankels[2]

        for L in 2:Literations
            invL = 1 / L
            legendre =
                (2 * L - 1) * invL * legendreargument * legendrebefore -
                (L - 1) * invL * legendrebeforebefore
            legendrebeforebefore = legendrebefore
            legendrebefore = legendre
            sum += legendre * sphericalhankels[L + 1]
        end
        translationoperator[i] = sum
    end

    return Literations
end

# --- the 2-D translation operator ------------------------------------------------------------------
#
# Translation is normalized to the Green's function in the outgoing `exp(-iκr)` /
# Hankel-second-kind convention. Integrating the 3-D operator with the sampling weights and
# `phasefactor` reproduces `exp(-iκr)/(4πr)`. The 2-D analogue uses circle weights `2π/Nφ`:
#
#     Σ_q w_q T(φ_q) exp(-γ k̂_q · d)  =  G₂ᴰ(|R + d|)  =  -(i/4) H₀⁽²⁾(κ|R + d|)
#
# The constant matches BEAST's `HH2DSingleLayerFDBIO` kernel (`green = -im/4 * hankelh2(0,kR)`),
# which uses the same `exp(+iωt)`, `γ = iκ` convention.
#
# Writing `T(φ) = Σ_m t_m e^{imφ}` and expanding the radiation phase by
# Jacobi-Anger, the φ integral picks `n = -m` and Graf's addition theorem matches term by term to give
# `t_m ∝ (-i)^m H_m⁽²⁾(κ|R|) e^{-imφ_R}`. Pairing `+m` with `-m` and using the identity
# `H₋ₘ⁽²⁾ = (-1)^m Hₘ⁽²⁾` collapses the pair to a real cosine:
#
#     (-i)^m Hₘ e^{imΔ} + (-i)^{-m} H₋ₘ e^{-imΔ} = (-i)^m Hₘ (e^{imΔ} + e^{-imΔ}) = 2(-i)^m Hₘ cos(mΔ)
#
# Only non-negative orders need evaluation. That halves the Bessel work, so the loop
# runs `1:L` rather than `-L:L`.
#
# The `R + d` orientation in Graf's theorem requires `(-i)^m`, not `i^m`.
# In the sign sweep, `(-i)^m` reproduced the identity to 3e-15; `i^m` gave 5.1e-01 at every order.
#
# Truncation limits cancellation: `Hₘ⁽²⁾(x)` grows like `(m-1)!(2/x)^m`, so high-order terms can
# overwhelm the final sum in finite precision. At `L = 40` with `κ|R| ≈ 11.7`, the error was ~100x.
# `_cancellationlimitedorder` selects the usable order from the Hankel magnitudes, rather than a
# distance-only bound such as `min(cutoffparameter, ceil(κ|R|) + margin)`.
function translationoperator!(
    translationoperator::AbstractArray{T1},
    γ::T1,
    cartesiansamplepoints::AbstractArray{SVector{2,T}},
    R::SVector{2,T},
    cutoffparameter::Int,
) where {T1,T}
    κ = abs(γ)
    Rnorm = norm(R)
    x = κ * Rnorm

    hankels = collectcylindricalhankelsecond(cutoffparameter, x)
    Literations = _cancellationlimitedorder(hankels, cutoffparameter)

    φR = atan(R[2], R[1])
    # `-im/4` is G₂ᴰ's constant, `1/(2π)` the series'. Folded into one factor applied per term.
    prefactor = T1(-im / 4) / (2 * T(π))

    @inbounds for (i, k̂) in enumerate(cartesiansamplepoints)
        Δ = atan(k̂[2], k̂[1]) - φR

        sum = prefactor * hankels[1]
        for m in 1:Literations
            sum += 2 * prefactor * (-im)^m * hankels[m + 1] * cos(m * Δ)
        end
        translationoperator[i] = sum
    end

    return Literations
end

# Limit the order by the cancellation budget, not just `κ|R|`.
# Each term has the magnitude of its `Hₘ⁽²⁾`, while the total is of order `H₀⁽²⁾`. Thus
# a term `ρ` times larger than `H₀` costs about `log₁₀ ρ` of the 16 available digits. Once `ρ` reaches
# `1/eps` the result is pure noise.
#
# A cap at `κ|R|`, where `Hₘ⁽²⁾(x)` begins growing, is too restrictive for small boxes.
# For a λ/4 box, `κd ≈ 1.6`, and the excess-bandwidth rule asks for
# `L ≈ κd + 1.8·digits^(2/3)·(κd)^(1/3)`, dominated by the margin:
# `L ≈ 9` against a `κ|R| ≈ 3` for the nearest interaction. Capping at `κ|R|` would reduce
# leaf-level accuracy. A cancellation budget adapts instead:
# at small `x` the Hankels grow fast and it clamps early, at large `x` it barely binds.
#
# The budget allows a loss of 6 of the ~16 digits, leaving ~1e-10, beyond the usual 1e-4 to 1e-6
# target. Measurements check this accuracy in
# `test/Translations/translation2d.jl`.
#
# As in 3-D, the operator returns the chosen order so callers can detect clamping.
const TRANSLATIONCANCELLATIONBUDGET = 1.0e6

function _cancellationlimitedorder(hankels::AbstractVector, cutoffparameter::Int)
    reference = abs(hankels[1])
    # A zero-order Hankel of zero magnitude means `x` is degenerate; nothing sensible to scale against.
    iszero(reference) && return 0
    limit = TRANSLATIONCANCELLATIONBUDGET * reference

    for m in 1:cutoffparameter
        abs(hankels[m + 1]) > limit && return m - 1
    end
    return cutoffparameter
end

# Cylindrical Hankel functions of the second kind, orders `0:order`.
#
# Spherical Hankel functions have closed forms at orders 0 and 1 (`cis(-x)` times a rational).
# Cylindrical functions do not, and upward recurrence in `J` is unstable. Use
# `SpecialFunctions.hankelh2` rather than implementing Miller's downward algorithm here.
function collectcylindricalhankelsecond(order::Int, x::T) where {T}
    v = Vector{complex(float(T))}(undef, order + 1)
    collectcylindricalhankelsecond!(v, order, x)
    return v
end

function collectcylindricalhankelsecond!(v::AbstractArray, order::Int, x::T) where {T}
    @inbounds for m in 0:order
        v[m + 1] = SpecialFunctions.hankelh2(m, x)
    end
    return nothing
end

# calculate the spherical hankel function of the second kind of order zero
function sphericalhankelsecond0(x::T) where {T}
    return im / x * cis(-x)
end

# calculate the spherical hankel function of the second kind of order one
function sphericalhankelsecond1(x::T) where {T}
    return (im - x) / (x^2) * cis(-x)
end

# calculate the spherical hankel function of the second kind of order n
function sphericalhankelsecond(order::Int, x::T) where {T}
    order == 0 && return sphericalhankelsecond0(x)
    order == 1 && return sphericalhankelsecond1(x)

    v = zeros(complex(T), order + 1)
    collectsphericalhankelsecond!(v, order, x)

    return v[end]
end

# collect all spherical hankel functions of the second kind up to order n
function collectsphericalhankelsecond(order::Int, x::T) where {T}
    v = Vector{complex(T)}(undef, order + 1)
    collectsphericalhankelsecond!(v, order, x)
    return v
end

# collect all spherical hankel functions of the second kind up to order n in place in v
function collectsphericalhankelsecond!(v::AbstractArray, order::Int, x::T) where {T}
    v[1] = sphericalhankelsecond0(x)
    order < 1 && return nothing

    v[2] = sphericalhankelsecond1(x)
    order < 2 && return nothing

    for i in eachindex(v)
        i < 3 && continue

        v[i] = (2 * i - 3) * v[i - 1] / x - v[i - 2]
    end

    return nothing
end

# calculate the legendre polynomial of order zero
function legendre0(x::T) where {T}
    return T(1)
end

# calculate the legendre polynomial of order one
function legendre1(x::T) where {T}
    return x
end

# collect all legendre polynomials up to order n in place in v
function collectlegendre!(v::AbstractArray{T}, n::Int, x::T) where {T}
    v[1] = legendre0(x)
    n < 1 && return nothing
    v[2] = legendre1(x)
    n < 2 && return nothing

    for i in eachindex(v)
        i < 3 && continue

        order = i - 1
        v[i] = (2 * order - 1) / order * x * v[i - 1] - (order - 1) / (order) * v[i - 2]
    end
end

# collect all legendre polynomials up to order n
function collectlegendre(n::Int, x::T) where {T}
    v = Vector{T}(undef, n + 1)
    collectlegendre!(v, n, x)
    return v
end
