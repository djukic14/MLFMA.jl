# Vector spherical harmonics provide analytic tangential reference fields.
#
# A `TangentialSphericalFarField` stores
# `(Fθ, Fφ)`, and the interpolator continues both components across the poles with
# `VirtualExtensionFlipθSign`: `f(θ, φ) = -f(-θ, φ + π)`. Filling the two components with unrelated
# scalar harmonics produces arrays that do NOT obey that rule, so the measured "interpolation error"
# would be dominated by the reference disagreeing with the interpolator's continuation rather than by
# interpolation quality. Vector spherical harmonics obey the required continuation.
#
# The map (θ, φ) -> (-θ, φ + π) sends a point to itself in R^3 -- it is the chart continuation across
# the pole, not a different direction. Under it both basis vectors flip:
#
#     θ̂ = ( cosθcosφ,  cosθsinφ, -sinθ)   ->  (-cosθcosφ, -cosθsinφ,  sinθ) = -θ̂
#     φ̂ = (     -sinφ,      cosφ,      0) ->  (     sinφ,     -cosφ,      0) = -φ̂
#
# so a field with the same physical value at that point must negate both components. Any true
# tangential vector field therefore satisfies `VirtualExtensionFlipθSign` by construction, and the
# scalar case's unflipped `VirtualExtension` is likewise right because a scalar is chart-independent.
# What can still be wrong is the component EVALUATION, so it is validated four independent ways in
# `test/LevelTransfers/vectorharmonics.jl`: against finite differences, against the l = 1 closed forms,
# against the reflection identity above, and against orthonormality integrated with the sampling's own
# Gauss-Legendre weights.
#
# With ``Y_l^m = q_l^m(θ) e^{imφ}`` and ``s = sqrt(l(l+1))``:
#
#     B_l^m = ∇_Ω Y_l^m / s          (poloidal / spheroidal; "electric" type)
#     C_l^m = r̂ × ∇_Ω Y_l^m / s      (toroidal; "magnetic" type)
#
# and ``∇_Ω = θ̂ ∂_θ + φ̂ (1/sinθ) ∂_φ``, so
#
#     B: (F_θ, F_φ) = (∂_θ q, i m q / sinθ) e^{imφ} / s
#     C: (F_θ, F_φ) = (-i m q / sinθ, ∂_θ q) e^{imφ} / s        since r̂ × θ̂ = φ̂ and r̂ × φ̂ = -θ̂
#
# Together they span tangential band-limited fields.
#
# ON `1/sinθ`, because it shapes how the results must be read: `F_φ` carries `m q / sinθ`. Since
# ``q_l^m ~ sin^m θ`` toward the poles that stays BOUNDED for every m, and Gauss-Legendre θ nodes are
# strictly interior so nothing divides by zero. But the component is not as smooth as the field is, and
# a componentwise interpolation error near the poles may be reporting the coordinate representation
# rather than the interpolator. That is why `componenterrormetrics` reports `vector_rel_l2`/
# `vector_rel_linf` alongside the per-component numbers. Use `θbandprofile` to locate error near
# the poles; a profile helps explain error but does not set an accuracy threshold.

"""
    dnormalizedlegendre(l, m, θ)

``∂_θ`` of [`normalizedlegendre`](@ref), by recurrence in the same fully normalized quantity.

    ∂_θ q_l^m = (1/2) [ sqrt((l-m)(l+m+1)) q_l^{m+1} - sqrt((l+m)(l-m+1)) q_l^{m-1} ]

At `m = 0`, the `m - 1` neighbour is ``q_l^{-1}``, equal to ``-q_l^{1}``, not zero.
The terms add and give ``sqrt(l(l+1)) q_l^{1}``. Dropping the lower term would halve the derivative.

`m` must satisfy `0 <= m <= l`; negative orders come from the reflection in
[`vectorsphericalharmonic`](@ref).
"""
function dnormalizedlegendre(l::Integer, m::Integer, θ::T) where {T<:AbstractFloat}
    l == 0 && return zero(T)
    m == 0 && return sqrt(T(l) * T(l + 1)) * normalizedlegendre(l, 1, θ)

    upper = if m + 1 <= l
        sqrt(T(l - m) * T(l + m + 1)) * normalizedlegendre(l, m + 1, θ)
    else
        zero(T)
    end
    lower = sqrt(T(l + m) * T(l - m + 1)) * normalizedlegendre(l, m - 1, θ)
    return (upper - lower) / 2
end

"""
    SUPPORTEDVECTORKINDS

The vector spherical harmonic families this module evaluates: `:poloidal` and `:toroidal`.

Named for what they are rather than `:B`/`:C`, since every source uses different letters for these two
and the words are unambiguous.
"""
const SUPPORTEDVECTORKINDS = (:poloidal, :toroidal)

"""
    vectorsphericalharmonic(kind, l, m, θ, φ)

The tangential components `(Fθ, Fφ)` of ``B_l^m`` (`kind = :poloidal`) or ``C_l^m``
(`kind = :toroidal`), as a `Tuple` of two `Complex`.

`l >= 1` is required: at `l = 0` the surface gradient of a constant vanishes and the normalization
``sqrt(l(l+1))`` is zero, so there is no degree-zero tangential harmonic to return and this raises
rather than producing `NaN`.

Negative orders come from ``B_l^{-m} = (-1)^m conj(B_l^m)`` componentwise -- the surface gradient and
``r̂ ×`` are both real-linear, so the scalar reflection carries through unchanged, and there is one code
path to get right.
"""
function vectorsphericalharmonic(
    kind::Symbol, l::Integer, m::Integer, θ::T, φ::T
) where {T<:AbstractFloat}
    kind in SUPPORTEDVECTORKINDS || throw(
        ArgumentError(
            "vectorsphericalharmonic: kind must be one of " *
            "$(join(SUPPORTEDVECTORKINDS, ", ")), got :$kind",
        ),
    )
    l >= 1 || throw(
        ArgumentError(
            "vectorsphericalharmonic: l must be >= 1, got l = $l. There is no degree-zero " *
            "tangential harmonic -- the surface gradient of a constant vanishes.",
        ),
    )
    abs(m) <= l || throw(
        ArgumentError("vectorsphericalharmonic: need abs(m) <= l, got l = $l, m = $m")
    )

    order = abs(m)
    scale = sqrt(T(l) * T(l + 1))
    q = normalizedlegendre(l, order, θ)
    dq = dnormalizedlegendre(l, order, θ)
    phase = cis(order * φ)

    # `m q / sinθ` is bounded at every m because q ~ sin^m θ toward the poles, and Gauss-Legendre nodes
    # never sit exactly on one -- see the note at the top of this file on what that costs in smoothness.
    gradθ = complex(dq) * phase / scale
    gradφ = im * T(order) * complex(q) / sin(θ) * phase / scale

    θcomponent, φcomponent = if kind === :poloidal
        (gradθ, gradφ)
    else
        (-gradφ, gradθ)
    end

    m >= 0 && return (θcomponent, φcomponent)
    reflect = isodd(order) ? -1 : 1
    return (reflect * conj(θcomponent), reflect * conj(φcomponent))
end

"""
    exactvector(case::TangentialHarmonicCase, θ, φ)

The exact `(Fθ, Fφ)` of `case` at one direction.

The tangential counterpart of [`exactvalue`](@ref), kept a separate name rather than a method of it
because it returns two components: a caller that reached the wrong one gets a `MethodError` instead of
a scalar silently standing in for a vector.
"""
function exactvector(case::TangentialHarmonicCase, θ::T, φ::T) where {T}
    return vectorsphericalharmonic(case.kind, case.l, case.m, θ, φ)
end

"""
    tangentialbandcoefficients(case::RandomTangentialBandlimitedCase, ::Type{T}=Float64)

The coefficients of `case`, as `(poloidal, toroidal)` vectors drawn once from its seed.

TWO SEPARATE VECTORS RATHER THAN ONE COMBINED INDEX. The scalar `_coefficientindex` packs `m ∈ -l:l` for
`l = 0:L` into `1:(L+1)^2` with no gaps; here there are two families AND `l` starts at 1, so a combined
index would need an offset and a skipped degree-zero slot -- two chances to be off by one, in code whose
errors look like plausible fields. Each vector is instead indexed by the SCALAR `_coefficientindex` with
its `l = 0` entry left unused, which costs one complex number per family and cannot be misread.

The draw order is poloidal in full, then toroidal, each ascending in `l` then `m`, and it is part of what
the seed means: reordering it silently changes every case with that seed.
"""
function tangentialbandcoefficients(
    case::RandomTangentialBandlimitedCase, ::Type{T}=Float64
) where {T}
    rng = Random.Xoshiro(case.seed)
    decay = Val(case.decay)
    slots = (case.L + 1)^2

    function draw()
        coefficients = zeros(Complex{T}, slots)
        for l in 1:(case.L)
            weight = _decayweight(decay, l, T)
            for m in (-l):l
                re = 2 * rand(rng, T) - one(T)
                im = 2 * rand(rng, T) - one(T)
                coefficients[_coefficientindex(l, m)] = weight * complex(re, im)
            end
        end
        return coefficients
    end

    poloidal = draw()
    toroidal = draw()
    return (poloidal=poloidal, toroidal=toroidal)
end

"""
    exactvector(case::RandomTangentialBandlimitedCase, θ, φ, coefficients)

The exact `(Fθ, Fφ)` of a random tangential band at one direction.

`coefficients` is the precomputed [`tangentialbandcoefficients`](@ref), threaded in so a grid fill draws
once rather than per point -- the coefficients define the FUNCTION, and a per-point draw would evaluate a
different field at every direction.

Sum both harmonic families, restarting the recurrence for each `(l, m)`. Unlike the scalar band
evaluator, this does not reuse one recurrence sweep across the band.
"""
function exactvector(
    case::RandomTangentialBandlimitedCase, θ::T, φ::T, coefficients
) where {T}
    θtotal = zero(Complex{T})
    φtotal = zero(Complex{T})

    for l in 1:(case.L), m in (-l):l
        index = _coefficientindex(l, m)
        poloidalθ, poloidalφ = vectorsphericalharmonic(:poloidal, l, m, θ, φ)
        toroidalθ, toroidalφ = vectorsphericalharmonic(:toroidal, l, m, θ, φ)
        θtotal +=
            coefficients.poloidal[index] * poloidalθ +
            coefficients.toroidal[index] * toroidalθ
        φtotal +=
            coefficients.poloidal[index] * poloidalφ +
            coefficients.toroidal[index] * toroidalφ
    end
    return (θtotal, φtotal)
end

function exactvector(case::RandomTangentialBandlimitedCase, θ::T, φ::T) where {T}
    return exactvector(case, θ, φ, tangentialbandcoefficients(case, T))
end

"""
    fillexact!(farfield::MLFMA.TangentialSphericalFarField, case::TangentialHarmonicCase, sampling)

Fill both component arrays of a tangential far field with `case`'s exact values on `sampling`.

Indexed `[φindex, θindex]`, as the scalar fill is and for the same reason: `Base.size(::SphericalSampling)`
is `(Nφ, Nθ)`, the transpose of how a grid is described, and getting it backwards transposes the field
rather than failing.
"""
function fillexact!(
    farfield::FarFields.TangentialSphericalFarField,
    case::AnyTangentialCase,
    sampling::Sampling.AbstractSphericalSampling{T},
) where {T}
    θvalues = Sampling.θs(sampling)
    φvalues = Sampling.φs(sampling)
    θcomponent, φcomponent = farfield[1], farfield[2]

    # Drawn once for the whole grid, for the same reason the scalar band fill does it: the coefficients
    # ARE the function.
    coefficients = _tangentialcoefficients(case, T)

    for (θindex, θ) in enumerate(θvalues), (φindex, φ) in enumerate(φvalues)
        Fθ, Fφ = _exactvector(case, θ, φ, coefficients)
        θcomponent[φindex, θindex] = Fθ
        φcomponent[φindex, θindex] = Fφ
    end
    return farfield
end

_tangentialcoefficients(::TangentialHarmonicCase, ::Type{T}) where {T} = nothing
_tangentialcoefficients(case::RandomTangentialBandlimitedCase, ::Type{T}) where {T} =
    tangentialbandcoefficients(case, T)

_exactvector(case::TangentialHarmonicCase, θ, φ, ::Nothing) = exactvector(case, θ, φ)
_exactvector(case::RandomTangentialBandlimitedCase, θ, φ, coefficients) =
    exactvector(case, θ, φ, coefficients)
