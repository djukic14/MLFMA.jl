# Tangential (two-component) far-field patterns on the equidistant grid.
#
# MLFMA's Maxwell far fields are k̂-transverse vector fields, carried as the pair (Fθ, Fφ) in the local
# spherical basis. Everything else in this submodule is scalar; this file is what makes the
# representation usable for the vector case.
#
# Both components are antispherical. See the continuation section of `sampling.jl` for the derivation
# and `test/Equidistant/tangential.jl` for the measurement. The short version: the torus identification
# sends a point to the SAME physical direction, and the local basis evaluated by its own formula at the
# torus coordinates flips BOTH unit vectors, so both components must flip for `Fθ θ̂ + Fφ φ̂` to stay
# single-valued. Using SPHERICAL for either component violates this continuation rule.
#
# `TangentialField` exists mainly so that parity cannot be chosen at the call site at all. There is no
# component-parity argument anywhere in this file: `resample` on a `TangentialField` hardcodes
# Use antispherical continuation for both components.

"""
    TangentialField(Fθ, Fφ)

A tangential far-field pattern on an [`EquidistantSampling`](@ref), as its two components in the local
spherical basis. Both are `Nφ × Nθ` matrices.
"""
struct TangentialField{T,M<:AbstractMatrix{T}}
    Fθ::M
    Fφ::M
    function TangentialField(Fθ::M, Fφ::M) where {T,M<:AbstractMatrix{T}}
        size(Fθ) == size(Fφ) ||
            error("TangentialField: components differ in size, $(size(Fθ)) vs $(size(Fφ)).")
        return new{T,M}(Fθ, Fφ)
    end
end

Base.size(F::TangentialField) = size(F.Fθ)
Base.eltype(::TangentialField{T}) where {T} = T

"""
    thetahat(θ, φ), phihat(θ, φ), khat(θ, φ)

The local spherical basis and the radial direction, as 3-tuples, from the standard formulas.

Evaluated at whatever `(θ, φ)` they are handed -- including `θ > π`, which is exactly the case that
makes both tangential components anti-spherical.
"""
thetahat(θ, φ) = (cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ))
phihat(θ, φ) = (-sin(φ), cos(φ), zero(θ))
khat(θ, φ) = (sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))

"""
    resample(F::TangentialField, src, dst, backend) -> TangentialField

Resample both components, each with ANTISPHERICAL parity.

There is deliberately no parity argument: for a tangential field there is only one correct choice, so
offering it would only create a way to get it wrong.
"""
function resample(
    F::TangentialField, src::EquidistantSampling, dst::EquidistantSampling, backend
)
    return TangentialField(
        resample(F.Fθ, src, dst, ANTISPHERICAL, backend),
        resample(F.Fφ, src, dst, ANTISPHERICAL, backend),
    )
end

"""
    cartesian(F::TangentialField, sampling) -> Matrix of 3-tuples

`Fθ θ̂ + Fφ φ̂` at every grid point.

This is the representation-independent view of the field, and therefore the one worth comparing
against: two different `(Fθ, Fφ)` conventions can both be "right" in their own basis, but the
Cartesian vector at a physical direction is unique.
"""
function cartesian(F::TangentialField, s::EquidistantSampling)
    return [
        F.Fθ[j, i] .* thetahat(θ, φ) .+ F.Fφ[j, i] .* phihat(θ, φ) for
        (j, φ) in enumerate(φs(s)), (i, θ) in enumerate(θs(s))
    ]
end

"""
    istransverse(F::TangentialField, sampling; atol) -> Bool

Whether `F` is everywhere orthogonal to `k̂`, which a far-field pattern must be.

Any pair of components in the `(θ̂, φ̂)` basis is transverse by construction. This checks the
Cartesian conversion to the requested tolerance; it does not detect components mixed by resampling.
"""
function istransverse(F::TangentialField, s::EquidistantSampling; atol=1e-12)
    C = cartesian(F, s)
    for (i, θ) in enumerate(θs(s)), (j, φ) in enumerate(φs(s))
        k = khat(θ, φ)
        abs(sum(C[j, i] .* k)) > atol && return false
    end
    return true
end
