# A one-angle grid for 2-D Helmholtz fields. It has `N = O(L)` planar `SVector{2}` directions,
# rather than `O(L²)` spherical directions, and sampled functions are vectors.
# It implements `AbstractSampling` directly; spherical product-grid methods do not apply.

"""
    CircularSampling{T}(Nφ)

Equidistant sampling of the unit circle: `Nφ` planar directions with no θ coordinate.

    φⱼ = 2π(j - 1)/Nφ    j = 1 … Nφ        (`Nφ` even, so φ → φ+π is a half-array shift)

`Nφ` must be even for the same reason `EquidistantSampling` requires it: the antipodal map `φ → φ + π`
is then exactly the index shift `j → j + Nφ÷2`, so [`mirrorindices`](@ref) is a permutation with no
interpolation. An odd `Nφ` would put the half-turn between samples and make the mirror lossy.

# Quadrature

Uniform weights `2π/Nφ` -- the trapezoidal rule on a periodic interval, which is spectrally accurate
and exact for every Fourier mode `e^{imφ}` with `|m| < Nφ`. The circle's line element is `dφ`,
so it needs neither the spherical `sin θ` Jacobian nor pole reflections.

# Bandwidth

`cutoffparameter(s) == Nφ÷2 - 1`, the exact inverse of [`NφCircular`](@ref). See that function for why
the product of two degree-`L` patterns is what sets the rate.

# See also

[`AbstractSampling`](@ref), whose interface this implements in full; [`SphericalSampling`](@ref), the
3-D counterpart.
"""
@auto_hash_equals struct CircularSampling{T<:Number} <: Sampling.AbstractSampling{T}
    Nφ::Int
    function CircularSampling{T}(Nφ::Int) where {T}
        Nφ > 0 || throw(ArgumentError("CircularSampling needs Nφ > 0, got $Nφ"))
        # `mirrorindices` requires `φ → φ+π` to be an exact index shift, hence even `Nφ`.
        iseven(Nφ) || throw(
            ArgumentError(
                "CircularSampling requires an EVEN Nφ so that φ → φ+π is the exact index shift " *
                "j → j + Nφ÷2; got Nφ = $Nφ.",
            ),
        )
        return new{T}(Nφ)
    end
end

CircularSampling(Nφ::Int) = CircularSampling{Float64}(Nφ)

"""
    CircularSamplingBuilder{T}()

Builds a [`CircularSampling{T}`](@ref) from a rate, mirroring `MLFMA.SphericalSamplingBuilder` so the
two are interchangeable at a construction site.

Takes one rate, whereas the spherical builder takes two.
"""
struct CircularSamplingBuilder{T<:Number} end

(::CircularSamplingBuilder{T})(Nφ::Int) where {T} = CircularSampling{T}(Nφ)

"""
    samplingrate(::CircularSamplingBuilder, cutoffparameter)

One rate, not two -- see [`NφCircular`](@ref) for the exactness rule and `MLFMA.samplingrate` for
why it lives on the builder.

The rate is rounded up to a multiple of four, beyond the exactness requirement.
`φ → φ + π/2` lands on a grid point only when `Nφ ≡ 0 (mod 4)`, allowing full square symmetry.
Over the 40 distinct far-field offsets at one level, this permits a 5.71x reduction rather than
2.00x. `NφCircular` gives `2(L + 1)`, a multiple of four only for odd `L`. The tree uses one
common group for all levels, so a single even-`L` level would otherwise limit the map to 2.00x.

Rounding adds two directions on even-`L` levels and none on odd ones, i.e. `1/(L + 1)` relatively.
Measured: +33 % at `L = 2`, +20 % at `L = 4`, +14 % at `L = 6`, +11 % at `L = 8`, falling to +5 % by
`L = 20`. The worst case is the smallest box, and it is paid by every 2-D map rather than only by
those using the reduction.

Rounding is independent of the translation collection. Both reduced and unreduced collections use
the same grid, so their numerical results can be compared directly.

[`NφCircular`](@ref) remains the exactness threshold. `MLFMA.CircularHarmonics` uses that threshold
with its own oversampling factor, independent of this rounding.

As a consequence, `cutoffparameter(grid) == Nφ ÷ 2 - 1` returns `L + 1` on a rounded
level, so the translation operator is built one degree higher than the rate rule asked for. That is
safe -- exactness needs `2(L + 1) < 2L + 4` -- and slightly more accurate, but it means
`cutoffparameter ∘ samplingrate` is no longer the identity, only `≥`.
"""
samplingrate(::CircularSamplingBuilder, cutoffparameter::Integer) =
    (_roundtoquarterturn(Sampling.NφCircular(cutoffparameter)),)

# The next multiple of four at or above `Nφ`. `Nφ` is even by construction here, so this adds either
# nothing or two.
_roundtoquarterturn(Nφ::I) where {I<:Integer} = I(4) * cld(Nφ, I(4))

"""
    NφCircular(cutoffparameter) -> Int

The number of circular directions needed to carry a pattern of degree `L`: `2(L + 1)`.

The 2-D analogue of [`NθNφ`](@ref). The trapezoidal rule on `[0, 2π)` with
`Nφ` points is exact for every mode `|m| < Nφ`. The translation integrand is a product of the
degree-`L` translation operator and a degree-`L` pattern, so it carries content out to `2L`, and
`2L < Nφ` needs `Nφ ≥ 2L + 1`. Rounding up to the next even value gives `2L + 2 = 2(L + 1)`, which is
both the minimum that works and an exact inverse of [`cutoffparameter`](@ref) on this grid.

This differs from `NθNφ`'s φ rate, `4(L + 1)`, which includes extra samples for spherical
θ interpolation stencils. Circular interpolation does not need them.
"""
function NφCircular(cutoffparameter::I) where {I<:Integer}
    return I(2) * (cutoffparameter + I(1))
end

"""
    φs(s::CircularSampling) -> Vector

The `Nφ` sample angles, equidistant on `[0, 2π)` with `2π` itself excluded.
"""
function φs(s::CircularSampling{T}) where {T}
    return T[2 * T(π) * (j - 1) / s.Nφ for j in 1:(s.Nφ)]
end

"""
    φ(s::CircularSampling, j) -> T

The `j`-th angle, wrapping around periodically for any integer index.

Wraparound is plain periodicity here -- `φ(s, j + Nφ) == φ(s, j)` as an angle -- with none of the
reflection the spherical grids need, because there is no pole to reflect through.
"""
function φ(s::CircularSampling{T}, j::Integer) where {T}
    return T(2 * T(π) * (mod1(j, s.Nφ) - 1) / s.Nφ)
end

Base.length(s::CircularSampling) = s.Nφ

"""
    size(s::CircularSampling)

`(Nφ,)`, a one-tuple, so sampled functions use a `Vector`.

The far-field constructors derive their array rank from the length of this tuple, so nothing else has
to be told that a 2-D problem has one angular index.
"""
Base.size(s::CircularSampling) = (s.Nφ,)

"""
    Base.enumerate(s::CircularSampling)

Enumerate the sample angles in storage order, matching the linear order of
[`cartesiansamplepoints`](@ref).
"""
Base.enumerate(s::CircularSampling) = Base.Iterators.enumerate(Sampling.φs(s))

arraytype(::Type{CircularSampling{T}}) where {T} = Vector
arraytype(::Type{CircularSampling}) = Vector
arraytype(s::CircularSampling) = Sampling.arraytype(typeof(s))

"""
    cartesiansamplepoints(s::CircularSampling) -> (k̂, weights)

The unit direction and integration weight at every sample, both `Nφ`-vectors.

`k̂` is `SVector{2,T}` -- the plane the problem lives in, not a sphere with a zero component. The
weights are the uniform `2π/Nφ` of the trapezoidal rule and sum to `2π`, the circle's full measure,
just as the spherical weights sum to `4π`.
"""
function cartesiansamplepoints(s::CircularSampling{T}) where {T}
    k̂ = Vector{SVector{2,T}}(undef, s.Nφ)
    @inbounds for (j, φ) in enumerate(Sampling.φs(s))
        sφ, cφ = sincos(φ)
        k̂[j] = SVector{2,T}(cφ, sφ)
    end
    weights = fill(2 * T(π) / s.Nφ, s.Nφ)
    return k̂, weights
end

"""
    mirrorindices(s::CircularSampling) -> Vector{Int}

Linear indices realizing the antipodal map `k̂ → -k̂`, i.e. `φ → φ + π`, so that `F[mirrorindices(s)]`
samples `f(-k̂)` wherever `F` samples `f(k̂)`.

No interpolation is needed: `φⱼ + π = φ_{j + Nφ/2}` whenever `Nφ` is even, as the constructor
requires. The spherical case also needs a θ reflection.
"""
function mirrorindices(s::CircularSampling)
    half = s.Nφ ÷ 2
    return [mod1(j + half, s.Nφ) for j in 1:(s.Nφ)]
end

"""
    cutoffparameter(s::CircularSampling) -> Int

The degree up to which this grid integrates exactly: `Nφ÷2 - 1`, the exact inverse of
[`NφCircular`](@ref).
"""
cutoffparameter(s::CircularSampling) = s.Nφ ÷ 2 - 1
