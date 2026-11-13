# The equidistant-equidistant angular grid, and the spherical continuation that makes it a torus.
#
# The midpoint grid `θᵢ = (i + 1/2)·π/Nθ`, i = 0 … Nθ-1, excludes θ = 0 and θ = π.
# No sample sits on a coordinate singularity, so the tangential components `Fθ`/`Fφ` stay well behaved,
# the FFT representation needs no special pole sample, and a Lagrange stencil crossing a pole reduces
# to an index reflection with no special case at the pole itself.
#
# WHY `Nφ` MUST BE EVEN. The continuation below maps `φ → φ + π`, which on this grid is exactly
# `j → j + Nφ/2 (mod Nφ)`. An odd `Nφ` would put that half-turn between samples and force interpolation
# inside the boundary rule, which would make the rule lossy and the two backends disagree about it.
# `EquidistantSampling` rejects an odd `Nφ` at construction rather than rounding it.

"""
    EquidistantSampling{T}(Nθ, Nφ)

Equidistant sampling in BOTH angular coordinates: a midpoint θ grid on `(0, π)` and the usual
equidistant φ grid on `[0, 2π)`.

    θᵢ = (i - 1/2)·π/Nθ    i = 1 … Nθ        (midpoint: no sample at either pole)
    φⱼ = 2π(j - 1)/Nφ      j = 1 … Nφ        (`Nφ` even, so φ → φ+π is a half-array shift)

This is the grid; the interpolation ALGORITHM is a separate choice -- see `FFTBackend` and
`LagrangeBackend`, both of which consume this same object and this same continuation rule.
"""
struct EquidistantSampling{T<:Number} <: Sampling.AbstractSphericalSampling{T}
    Nθ::Int
    Nφ::Int
    θs::Vector{T}
    φs::Vector{T}

    function EquidistantSampling{T}(Nθ::Int, Nφ::Int) where {T}
        Nθ >= 1 || error("EquidistantSampling needs Nθ >= 1, got $Nθ")
        # Not a style preference: `φ → φ + π` is `j → j + Nφ/2` ONLY for even `Nφ`, and every
        # continuation in this file depends on that being an exact index shift.
        iseven(Nφ) || error(
            "EquidistantSampling requires an EVEN Nφ so that φ → φ+π is the exact index shift " *
            "j → j + Nφ÷2; got Nφ = $Nφ. Round up to $(Nφ + 1) is wrong -- use $(Nφ + 1 + 1).",
        )
        θs = T[(i - one(T) / 2) * π / Nθ for i in 1:Nθ]
        φs = T[2 * π * (j - 1) / Nφ for j in 1:Nφ]
        return new{T}(Nθ, Nφ, θs, φs)
    end
end

EquidistantSampling(Nθ::Int, Nφ::Int) = EquidistantSampling{Float64}(Nθ, Nφ)

# `Nθ`/`Nφ` need no method: `MLFMA`'s generic `AbstractSampling` versions already read these fields.
Sampling.θs(s::EquidistantSampling) = s.θs
Sampling.φs(s::EquidistantSampling) = s.φs
Sampling.θ(s::EquidistantSampling, i::Integer) = s.θs[i]
Sampling.φ(s::EquidistantSampling, j::Integer) = s.φs[j]
# MLFMA stores direction arrays [φ, θ] -- φ contiguous. See the STORAGE ORDER note below.
Base.size(s::EquidistantSampling) = (s.Nφ, s.Nθ)

"""
    quadratureweights(sampling)

Product midpoint weights for `∫ F dΩ = ∫∫ F sinθ dφ dθ`:

    w[j, i] = Δθ · Δφ · sin(θᵢ),    Δθ = π/Nθ,  Δφ = 2π/Nφ.

Separate from anything interpolation does. Quadrature error and interpolation error are different
error sources and are tested apart; this is also the `sinθ` that stops spectral restriction and the
weighted adjoint of interpolation from coinciding (see `adjointinterpolate!`).
"""
function quadratureweights(s::EquidistantSampling{T}) where {T}
    Δθ = T(π) / s.Nθ
    Δφ = 2 * T(π) / s.Nφ
    return [Δθ * Δφ * sin(θ) for _ in 1:(s.Nφ), θ in s.θs]
end

# --- the spherical / anti-spherical continuation ---------------------------------------------------
#
# ONE rule, shared by both backends and by the tests. A scalar spherical function obeys
#
#     F(2π - θ, φ + π) = F(θ, φ)
#
# while BOTH tangential components MLFMA carries are ANTI-spherical:
#
#     Fθ(2π - θ, φ + π) = -Fθ(θ, φ),      Fφ(2π - θ, φ + π) = -Fφ(θ, φ).
#
# The identification maps a torus point to the same physical direction, but flips both local
# basis vectors --
# θ̂(2π-θ, φ+π) = -θ̂(θ, φ) and φ̂(2π-θ, φ+π) = -φ̂(θ, φ). Since the Cartesian field must be
# single-valued, `Fθ θ̂ + Fφ φ̂` is preserved only if both components flip too.
# `test/Equidistant/tangential.jl` checks the basis flip and both component parities.
#
# So SPHERICAL is for genuine SCALARS -- Helmholtz patterns, the translation operator, quadrature test
# functions -- and ANTISPHERICAL is what a tangential component uses, whichever of the two it is.
#
# `SPHERICAL` / `ANTISPHERICAL` name the two, so a caller states which component it is holding instead
# of passing a bare `±1` that is easy to get backwards.
const SPHERICAL = 1
const ANTISPHERICAL = -1

"""
    reflectindex(i, j, Nθ, Nφ) -> (i′, j′, sign)

Map a θ index OUTSIDE `1:Nθ` back onto the physical grid, returning the partner index pair and
`(-1)^reflections` -- the factor an ANTI-spherical component picks up. A spherical one picks up 1
whatever the count, since each reflection leaves it unchanged.

Both poles reduce to the same two operations -- reflect θ, half-turn φ -- because the midpoint grid
has no sample sitting on either pole to special-case:

    i < 1   (θ < 0)  ->  i′ = 1 - i,          φ → φ + π
    i > Nθ  (θ > π)  ->  i′ = 2Nθ + 1 - i,    φ → φ + π

A stencil reaching more than a full grid past a pole reflects repeatedly, and the sign accumulates;
the loop is written out rather than assumed away, because `p` is a caller's choice and `Nθ` shrinks
with the tree level.
"""
function reflectindex(i::Int, j::Int, nθ::Int, nφ::Int)
    sign = 1
    half = nφ ÷ 2
    while i < 1 || i > nθ
        i = i < 1 ? 1 - i : 2 * nθ + 1 - i
        j = j + half
        sign = -sign
    end
    return i, mod1(j, nφ), sign
end

"""
    sphericalget(F, i, j, sampling, parity) -> value

Read `F[j, i]` -- θ index `i`, φ index `j` -- for ANY integer index pair, applying the continuation
of `reflectindex` and the sign
for a component of the given `parity` (`SPHERICAL` or `ANTISPHERICAL`).

This is the accessor the Lagrange stencils index through, so the stencil code itself never mentions a
pole. It is also the reference the FFT extension is tested against -- the same rule reached two ways.
"""
@inline function sphericalget(F, i::Int, j::Int, s::EquidistantSampling, parity::Int)
    i′, j′, sgn = reflectindex(i, j, s.Nθ, s.Nφ)
    v = @inbounds F[j′, i′]
    # `sgn` is (-1)^(number of reflections). A SPHERICAL component is unchanged by each reflection, an
    # ANTI-spherical one flips, so the multiplier is `parity^reflections` -- which is `1` and `sgn`
    # respectively. Written as a branch on the parity rather than `parity^k` so it stays integer-exact
    # and free after specialization.
    return parity == SPHERICAL ? v : sgn * v
end

"""
    extend(F, sampling, parity) -> Nφ × 2Nθ array

The torus extension: continue `F` from `(0, π)` to `(0, 2π)` in θ so both coordinates are periodic and
a plain 2-D FFT applies.

Columns `1:Nθ` are `F` itself. Column `Nθ + q` is physical column `Nθ + 1 - q` with φ shifted by half
the grid and the parity sign applied: reverse θ, roll φ by `Nφ÷2`, optionally negate.
This full extension is the reference for the reduced-grid implementation.
"""
function extend(F::AbstractMatrix, s::EquidistantSampling, parity::Int)
    nθ, nφ = s.Nθ, s.Nφ
    size(F) == (nφ, nθ) || error("extend: expected $((nφ, nθ)) array, got $(size(F))")
    E = similar(F, nφ, 2 * nθ)
    @inbounds for i in 1:nθ, j in 1:nφ
        E[j, i] = F[j, i]
    end
    half = nφ ÷ 2
    @inbounds for q in 1:nθ, j in 1:nφ
        src = nθ + 1 - q
        E[j, nθ + q] = parity * F[mod1(j + half, nφ), src]
    end
    return E
end

"""
    extendedθs(sampling)

The θ values of the extended `2Nθ` grid: the same midpoint rule carried across `[0, 2π)`, so that
extended θ index `2Nθ + 1 - i` sits at `2π - θᵢ` in exact arithmetic.
"""
function extendedθs(s::EquidistantSampling{T}) where {T}
    return T[(q - one(T) / 2) * π / s.Nθ for q in 1:(2 * s.Nθ)]
end

# --- STORAGE ORDER -------------------------------------------------------------------------------
#
# Arrays sampled on an `EquidistantSampling` are `Nφ × Nθ`: φ is the CONTIGUOUS dimension, and
# `Base.size` reports `(Nφ, Nθ)`. That is MLFMA's convention, not a preference of this submodule --
# `Base.size(::SphericalSampling)` returns `(Nφ, Nθ)`, `cartesiansamplepoints` fills `k̂[j, i]`, and
# `mirrorindices` is a permutation over LINEAR indices, so every one of its consumers depends on the
# ordering. Storing the transpose here and flipping at the boundary would have cost a full transpose
# on every level transfer, in the hot path.
#
# It is also the better layout for this grid: the θ-reflection in `extend` copies contiguous φ-blocks,
# and the global φ FFT runs along the fast dimension.

"""
    mirrorindices(sampling) -> Matrix{Int}

Linear indices realizing the ANTIPODAL map `k̂ → -k̂`, i.e. `(θ, φ) → (π - θ, φ + π)`, so that
`F[mirrorindices(s)]` samples `f(-k̂)` wherever `F` samples `f(k̂)`.

EXACT on this grid, with no interpolation and no special case. Note this is a DIFFERENT map from the
torus continuation `(2π - θ, φ + π)` used by `reflectindex`: the antipode reflects about the EQUATOR
and never goes near a pole, so the absence of pole samples is irrelevant to it. The midpoint grid is
symmetric under both halves of it:

    π - θᵢ = π - (i - 1/2)π/Nθ = ((Nθ - i + 1) - 1/2)·π/Nθ = θ_{Nθ+1-i}
    φⱼ + π = φ_{j + Nφ/2}                          (`Nφ` even is enforced at construction)
"""
function Sampling.mirrorindices(s::EquidistantSampling)
    nθ, nφ = s.Nθ, s.Nφ
    half = nφ ÷ 2
    linear = reshape(1:(nθ * nφ), nφ, nθ)
    return [linear[mod1(j + half, nφ), nθ + 1 - i] for j in 1:nφ, i in 1:nθ]
end
