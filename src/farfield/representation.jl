# Representation-dependent behaviour: named component accessors, the mirror/receive rule tables,
# and the transform kernel they drive.
#
# The concrete far-field type identifies the representation (`farfield.jl`).
# Dispatch uses that type, not the component count.

# Each representation groups its moment type with its two collection types, so an accessor can be
# written once and apply to all three. The collections genuinely need the same accessors as the
# moments -- callers reach for `Fθ(collection)` as readily as `Fθ(farfield)`.
const ScalarFieldLike = Union{
    ScalarFarField,ScalarRadiationFieldCollection,ScalarReceiveFieldCollection
}

const TangentialSphericalFieldLike = Union{
    TangentialSphericalFarField,
    TangentialSphericalRadiationFieldCollection,
    TangentialSphericalReceiveFieldCollection,
}

const SphericalVectorFieldLike = Union{
    SphericalVectorFarField,
    SphericalVectorRadiationFieldCollection,
    SphericalVectorReceiveFieldCollection,
}

const CartesianVectorFieldLike = Union{
    CartesianVectorFarField,
    CartesianVectorRadiationFieldCollection,
    CartesianVectorReceiveFieldCollection,
}

const PairedSphericalVectorFieldLike = Union{
    PairedSphericalVectorFarField,
    PairedSphericalRadiationFieldCollection,
    PairedSphericalReceiveFieldCollection,
}

const PlanarVectorScalarFieldLike = Union{
    PlanarVectorScalarFarField,
    PlanarVectorScalarRadiationFieldCollection,
    PlanarVectorScalarReceiveFieldCollection,
}

const AnyFieldCollection = Union{AbstractFarField,AbstractFieldCollection}

# Escape hatch for representations with no named accessor (e.g. the HH hypersingular's paired
# r1,θ1,φ1,r2,θ2,φ2 -- six components with no single conventional name each).
function component(ff::AnyFieldCollection, i)
    return ff[i]
end

# Named accessors. Each is defined ONLY for the representations that actually have that
# component, so asking for a component a representation does not have is a `MethodError` naming
# the accessor, not a silent read of whatever happens to sit at that index. `Fθ` on a scalar far
# field does not resolve; neither does `Fx` on a spherical one, nor `Fr` on a tangential one
# (which stores only θ,φ).
function F(ff::ScalarFieldLike)
    return ff[1]
end

function Fθ(ff::TangentialSphericalFieldLike)
    return ff[1]
end

function Fφ(ff::TangentialSphericalFieldLike)
    return ff[2]
end

function Fr(ff::SphericalVectorFieldLike)
    return ff[1]
end

function Fθ(ff::SphericalVectorFieldLike)
    return ff[2]
end

function Fφ(ff::SphericalVectorFieldLike)
    return ff[3]
end

# Cartesian components have their own accessors, independent of the spherical representation.
function Fx(ff::CartesianVectorFieldLike)
    return ff[1]
end

function Fy(ff::CartesianVectorFieldLike)
    return ff[2]
end

function Fz(ff::CartesianVectorFieldLike)
    return ff[3]
end

# The 2-D hypersingular representation: two IN-PLANE Cartesian components and one scalar. `Fx`/`Fy`
# are the same names the Cartesian representation uses, and deliberately so -- they mean the same
# thing, a Cartesian component of a vector. The third component is NOT a `Fz`: it is not a component
# of that vector at all but a separate scalar (`f'`, the arc-length derivative), so it gets its own
# name rather than continuing the sequence. Calling `Fz` on one of these is a `MethodError`, which is
# the correct outcome.
function Fx(ff::PlanarVectorScalarFieldLike)
    return ff[1]
end

function Fy(ff::PlanarVectorScalarFieldLike)
    return ff[2]
end

function Fs(ff::PlanarVectorScalarFieldLike)
    return ff[3]
end

# A single `dest_component <- coeff * src_component[mirrorindices]` mapping, used by both
# `mirrorrule` and `receivecouplingrule` (see `operatorutils.jl` for the latter). `Dest`/`Src` are
# compile-time component indices -- the permutation a representation's transform performs is
# fixed once you know the representation. `coeff` is a runtime field rather than a third type
# parameter because a receive rule's coefficient can depend on the specific operator instance
# (its γ), not just its type.
struct ComponentMap{Dest,Src,Coeff}
    coeff::Coeff
end

function ComponentMap{Dest,Src}(coeff) where {Dest,Src}
    return ComponentMap{Dest,Src,typeof(coeff)}(coeff)
end

function destcomponent(::ComponentMap{Dest}) where {Dest}
    return Dest
end

function srccomponent(::ComponentMap{Dest,Src}) where {Dest,Src}
    return Src
end

function coefficient(m::ComponentMap)
    return m.coeff
end

# `dest[d][i] = coeff * src[s][mirrorindices[i]]` for every mapping in `rule`, written directly
# into pre-allocated component arrays -- no `view`, no broadcast allocation, no scratch far field.
# `dest`/`src` need only support `getindex(_, componentindex)::AbstractArray` (a far field, a
# field collection, or a bare `Vector`/`SVector` of matrices all qualify). `eachindex(mirrorindices)`
# matches `eachindex` on every component array by construction -- `mirrorindices(sampling)` and
# every component here are sized to the same sampling grid.
#
# Representation-agnostic on purpose: it applies whatever rule it is handed and has no opinion
# about what the components mean.
#
# Recurses on `rule::Tuple` via `first`/`Base.tail` rather than `for mapping in rule`: `rule` is
# heterogeneous (each `ComponentMap{Dest,Src,_}` with different `Dest`/`Src` is its own concrete
# type), and the generic tuple `iterate` protocol infers `mapping` as a small `Union` across
# iterations -- correct, but the resulting union-splitting was measured to still allocate ~48
# bytes per call (see the allocation test). Peeling one element per recursive call keeps every
# `mapping` fully concrete at every step, so this compiles to a plain unrolled sequence.
function transform!(dest, src, mirrorindices, rule::Tuple)
    mapping = first(rule)
    destarray = dest[destcomponent(mapping)]
    srcarray = src[srccomponent(mapping)]
    c = coefficient(mapping)
    @inbounds for i in eachindex(mirrorindices)
        destarray[i] = c * srcarray[mirrorindices[i]]
    end
    transform!(dest, src, mirrorindices, Base.tail(rule))
    return dest
end

function transform!(dest, src, mirrorindices, ::Tuple{})
    return dest
end

# Purely geometric (component permutation + sign under the θ→π-θ, φ→φ+π reflection
# `mirrorindices` encodes) -- never a physical constant, so every coefficient here is a bare ±1.
# Operator-dependent coupling lives in `receivecouplingrule` (`operatorutils.jl`) instead; keeping
# the two separate is what stops a physical constant leaking into a geometric transform.
function mirrorrule(::Type{<:ScalarFarField})
    return (ComponentMap{1,1}(1),)
end

function mirrorrule(::Type{<:TangentialSphericalFarField})
    return (ComponentMap{1,1}(1), ComponentMap{2,2}(-1))
end

function mirrorrule(::Type{<:SphericalVectorFarField})
    return (ComponentMap{1,1}(1), ComponentMap{2,2}(1), ComponentMap{3,3}(-1))
end

# Apply the `SphericalVectorFarField` sign pattern to each r,θ,φ triple independently.
function mirrorrule(::Type{<:PairedSphericalVectorFarField})
    return (
        ComponentMap{1,1}(1),
        ComponentMap{2,2}(1),
        ComponentMap{3,3}(-1),
        ComponentMap{4,4}(1),
        ComponentMap{5,5}(1),
        ComponentMap{6,6}(-1),
    )
end

# The planar mirror leaves every component unchanged.
#
# The signs in the spherical rules above come from the BASIS VECTORS moving: θ̂ and φ̂ are defined
# relative to the direction, so the θ→π-θ, φ→φ+π reflection flips some of them. This representation
# has no such basis. Its first two components are fixed Cartesian directions (x̂, ŷ do not depend on
# k̂ at all) and its third is a scalar, so reindexing directions moves nothing.
#
# In a direct quadrature check, with `receive(k̂)` assembled and
# `radiation(-k̂)` evaluated at the antipode, the two agree componentwise to 6.6e-15 / 9.4e-15 /
# 3.2e-14. See `receivecouplingrule` in `operatorutils.jl` for the operator's coupling.
#
# This rule applies to a circle and its antipodal shift φ→φ+π.
# It does not supply the missing rule for `CartesianVectorFarField` on a sphere.
function mirrorrule(::Type{<:PlanarVectorScalarFarField})
    return (ComponentMap{1,1}(1), ComponentMap{2,2}(1), ComponentMap{3,3}(1))
end

# Deliberately absent: `mirrorrule(::Type{<:CartesianVectorFarField})`. Under the θ→π-θ, φ→φ+π
# reflection a Cartesian vector does NOT pick up the same per-component signs a spherical one
# does. This representation has no mirror rule, so mirroring it raises a `MethodError`.

# In-place: writes into a caller-supplied `dest` rather than allocating one, so this is what the
# hot path (repeated mirroring, e.g. across many leaves/directions) should call. Constrained to a
# single `F` on both sides -- mirroring maps a representation to itself, and this makes a
# mismatched pair a `MethodError` rather than a silently half-applied rule.
function mirror!(dest::F, src::F, sampling) where {F<:AbstractFarField}
    transform!(dest, src, mirrorindices(sampling), mirrorrule(F))
    return dest
end

# Convenience/reference wrapper -- allocates a fresh far field of the same concrete type as `src`
# and delegates to `mirror!`.
#
# `T` here is the stored per-component ARRAY type (e.g. `Matrix{ComplexF64}`), while the
# constructor wants the scalar element type, so `eltype(T)` recovers it. Passing the stored `T`
# straight through would build a far field of matrices-of-matrices.
#
# `basefarfieldtype` (`farfield.jl`) recovers the bare constructor from `src`'s own concrete type,
# so the result has `src`'s representation by construction rather than by re-deriving one.
function mirror(src::F, sampling) where {D,T,F<:AbstractFarField{D,T}}
    dest = basefarfieldtype(F)(eltype(T), sampling)
    return mirror!(dest, src, sampling)
end
