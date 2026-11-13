abstract type InterpolationExpansionTrait end

"""
    f(θ,φ) = f(θ,φ+2π),     for φ < 0
    f(θ,φ) = f(θ,φ-2π),     for φ > 2π

    f(θ,φ) = f(-θ,φ+pi),    for θ< 0, 0 < φ < π
    f(θ,φ) = f(-θ,φ-pi),    for θ < 0, π < φ < 2π
    f(θ,φ) = f(2π-θ,φ+pi),  for θ > π, 0 < φ < π
    f(θ,φ) = f(2π-θ,φ-pi)   for θ > π, π < φ < 2π

Continue a scalar across either pole without changing its sign. Wrap φ periodically before
applying the pole rule; at φ = 0 or φ = π, use the same half-turn modulo 2π.
These reflections preserve the physical direction, unlike the antipodal map `(π-θ, φ+π)`.
"""
struct VirtualExtension <: InterpolationExpansionTrait end

"""
    f(θ,φ) = f(θ,φ+2π),     for φ < 0
    f(θ,φ) = f(θ,φ-2π),     for φ > 2π

    f(θ,φ) = -f(-θ,φ+pi),    for θ < 0, 0 < φ < π
    f(θ,φ) = -f(-θ,φ-pi),    for θ < 0, π < φ < 2π
    f(θ,φ) = -f(2π-θ,φ+pi),  for θ > π, 0 < φ < π
    f(θ,φ) = -f(2π-θ,φ-pi)   for θ > π, π < φ < 2π

Use the same periodic wrapping and pole reflections as `VirtualExtension`, but negate the component
at each pole crossing. Wrapping φ alone does not change the sign.
"""
struct VirtualExtensionFlipθSign <: InterpolationExpansionTrait end

"""
Equivalent to [VirtualExtension(), VirtualExtensionFlipθSign(), VirtualExtensionFlipθSign()].
For the radial component of the field the sign is not flipped while for the θ and φ
components the sign is flipped.
"""
struct SphericalInterpolation <: InterpolationExpansionTrait end

"""
Equivalent to [VirtualExtensionFlipθSign(), VirtualExtensionFlipθSign()].
For the θ and φ components the sign is flipped.
"""
struct SphericalθφInterpolation <: InterpolationExpansionTrait end

"""
Equivalent to [VirtualExtension(), VirtualExtensionFlipθSign(), VirtualExtensionFlipθSign(),
VirtualExtension(), VirtualExtensionFlipθSign(), VirtualExtensionFlipθSign()].
Which is SphericalInterpolation repeated twice.
"""
struct DoubleSphericalInterpolation <: InterpolationExpansionTrait end

"""
    f(φ) = f(φ + 2π)

Periodic continuation for `N` components on a circle. There is no θ coordinate or pole reflection,
so every component uses the same rule without a sign change.

Unlike `VirtualExtension`, this trait does not describe a spherical pole rule. There are no
conversions to spherical traits or entries in `Equidistant.componentparities`; those families reject
it rather than apply a rule for the wrong geometry. The circular interpolator checks only its arity.
"""
struct PeriodicφInterpolation{N} <: InterpolationExpansionTrait end

# `N` is the component count. Encoding it in the type gives each arity a singleton value that can
# be compared with `===` at construction: 1 for scalar operators, 3 for `[n̂₁f, n̂₂f, f']`.
PeriodicφInterpolation() = PeriodicφInterpolation{1}()

"""
Resolve the trait from the far-field representation through `defaultinterpolationtrait`.

Scalar fields use `VirtualExtension`, tangential spherical fields use `SphericalθφInterpolation`,
spherical vector fields use `SphericalInterpolation`, and paired spherical vector fields use
`DoubleSphericalInterpolation`. Planar vector-scalar fields use `PeriodicφInterpolation{3}`.
The component count alone does not determine the continuation rule.
"""
struct DefaultInterpolation <: InterpolationExpansionTrait end

"""
    traitarity(trait) -> Int

How many far-field components `trait` describes.

`VirtualExtension` describes one component, `SphericalθφInterpolation` two, and
`DoubleSphericalInterpolation` six. A trait's arity must match the far field's component count.

Not defined for [`DefaultInterpolation`](@ref), which has no arity of its own. Resolve the field's
default trait or handle that case separately before checking arity.
"""
traitarity(::VirtualExtension) = 1
traitarity(::VirtualExtensionFlipθSign) = 1
traitarity(::SphericalθφInterpolation) = 2
traitarity(::SphericalInterpolation) = 3
traitarity(::DoubleSphericalInterpolation) = 6
traitarity(::PeriodicφInterpolation{N}) where {N} = N

# Defaults depend on the representation, not just its component count. Across a pole, θ maps to
# -θ or 2π-θ and φ shifts by π; this preserves the direction but flips both tangential basis vectors.
# The antipodal map θ→π-θ, φ→φ+π instead reverses the direction and is not this continuation rule.
#
# Spherical and Cartesian vector fields have the same arity but different basis behavior.
# No CartesianVectorFarField default is defined here; an unsupported representation raises MethodError.
# `<:` matches concrete far-field types regardless of their array and buffer parameters.
function defaultinterpolationtrait(::Type{<:ScalarFarField})
    return VirtualExtension()
end

function defaultinterpolationtrait(::Type{<:TangentialSphericalFarField})
    return SphericalθφInterpolation()
end

function defaultinterpolationtrait(::Type{<:SphericalVectorFarField})
    return SphericalInterpolation()
end

function defaultinterpolationtrait(::Type{<:PairedSphericalVectorFarField})
    return DoubleSphericalInterpolation()
end

# The only representation whose default trait is a CIRCULAR one, because it is the only one that
# exists solely for a 2-D operator. Three components, no sign -- see `PeriodicφInterpolation`.
function defaultinterpolationtrait(::Type{<:PlanarVectorScalarFarField})
    return PeriodicφInterpolation{3}()
end

function converttrait(::T, dictvalue, ::T) where {T}
    return dictvalue
end

function converttrait(::VirtualExtension, dictvalue, ::DefaultInterpolation)
    return (dictvalue.virtual,)
end

function converttrait(::VirtualExtensionFlipθSign, dictvalue, ::DefaultInterpolation)
    return (dictvalue.virtualflipθsign,)
end

function converttrait(::SphericalθφInterpolation, dictvalue, ::DefaultInterpolation)
    return (dictvalue.virtualflipθsign, dictvalue.virtualflipθsign)
end

function converttrait(::SphericalInterpolation, dictvalue, ::DefaultInterpolation)
    return (dictvalue.virtual, dictvalue.virtualflipθsign, dictvalue.virtualflipθsign)
end

function converttrait(::DoubleSphericalInterpolation, dictvalue, ::DefaultInterpolation)
    return (
        dictvalue.virtual,
        dictvalue.virtualflipθsign,
        dictvalue.virtualflipθsign,
        dictvalue.virtual,
        dictvalue.virtualflipθsign,
        dictvalue.virtualflipθsign,
    )
end

function converttrait(::SphericalθφInterpolation, dictvalue, ::VirtualExtensionFlipθSign)
    return SVector(dictvalue[1], dictvalue[1])
end

function converttrait(::VirtualExtensionFlipθSign, dictvalue, ::SphericalθφInterpolation)
    return SVector(dictvalue[1])
end

function converttrait(::DefaultInterpolation, dictvalue, ::SphericalInterpolation)
    return (virtual=dictvalue[1], virtualflipθsign=dictvalue[2])
end

function converttrait(::VirtualExtension, dictvalue, ::SphericalInterpolation)
    return SVector(dictvalue[1])
end

function converttrait(::VirtualExtensionFlipθSign, dictvalue, ::SphericalInterpolation)
    return SVector(dictvalue[2])
end

function converttrait(::SphericalθφInterpolation, dictvalue, ::SphericalInterpolation)
    return SVector(dictvalue[2], dictvalue[3])
end

function converttrait(::DoubleSphericalInterpolation, dictvalue, ::SphericalInterpolation)
    return SVector(
        dictvalue[1], dictvalue[2], dictvalue[3], dictvalue[1], dictvalue[2], dictvalue[3]
    )
end

function converttrait(::DefaultInterpolation, dictvalue, ::DoubleSphericalInterpolation)
    return (virtual=dictvalue[1], virtualflipθsign=dictvalue[2])
end

function converttrait(::VirtualExtension, dictvalue, ::DoubleSphericalInterpolation)
    return SVector(dictvalue[1])
end

function converttrait(
    ::VirtualExtensionFlipθSign, dictvalue, ::DoubleSphericalInterpolation
)
    return SVector(dictvalue[2])
end

function converttrait(::SphericalθφInterpolation, dictvalue, ::DoubleSphericalInterpolation)
    return SVector(dictvalue[2], dictvalue[3])
end

function converttrait(::SphericalInterpolation, dictvalue, ::DoubleSphericalInterpolation)
    return SVector(dictvalue[1], dictvalue[2], dictvalue[3])
end
