# Far-field moments and collections use one concrete type per representation.
# The type identifies the representation; `D` specifies only the storage length.
# In particular, these three-component representations are distinct types:
#
#     SphericalVectorFarField  != CartesianVectorFarField      (both fielddim 3)
#
# Two further properties follow from this and are relied on elsewhere:
#
#   * Accessors dispatch on the type, so `Fx` on a spherical far field is a `MethodError` rather
#     than a silent read of component 1 (see `representation.jl`).
#   * Each moment type is linked ONLY to the collection types of its own representation, so
#     `mul!(moment, collection, y)` across mismatched representations is a `MethodError` too.
#     H2Factory's `@linkmomenttocollection` generates methods constrained to one moment/collection
#     pair, so this type safety comes for free rather than needing a check.
#
# `D` remains a type parameter (H2Factory's macros own the `{D,T,P,V}` / `{D,T}` arity) and is NOT
# pinned to the representation's component count by the type system -- `fielddim` below is the
# single source of truth for what each representation's count actually is, and the constructors
# use it so nothing has to state a count by hand.
abstract type AbstractFarField{D,T} <: H2Factory.AbstractMoment{D,T} end

abstract type AbstractFieldCollection{D,T} <: H2Factory.AbstractMomentCollection{D,T} end

# Radiation and receive collections have opposite phase conventions.
# `_phase_k` in `phase/phasefactor.jl` dispatches on their abstract types.
abstract type AbstractRadiationFieldCollection{D,T} <: AbstractFieldCollection{D,T} end

abstract type AbstractReceiveFieldCollection{D,T} <: AbstractFieldCollection{D,T} end

@createmoment ScalarFarField AbstractFarField                 # F
@createmoment TangentialSphericalFarField AbstractFarField    # θ, φ
@createmoment SphericalVectorFarField AbstractFarField        # r, θ, φ
@createmoment CartesianVectorFarField AbstractFarField        # x, y, z
@createmoment PairedSphericalVectorFarField AbstractFarField  # r1,θ1,φ1, r2,θ2,φ2
# A vector IN THE PLANE plus a scalar -- the 2-D hypersingular operator's representation.
#
# This has the same component count as `SphericalVectorFarField`, but different semantics.
# It has no `Fr`/`Fθ`/`Fφ` components and must not use the θ-seam signs of `SphericalInterpolation()`.
#
# BEAST's 2-D hypersingular integrand is `(α f g (n̂ₓ·n̂ᵧ) + β f' g') G`, and on a
# curve `f.derivative` is a SCALAR. (The 3-D operator's `f.curl` is a 3-vector, which is why that one
# is six components and this is three -- it is a derivation, not a smaller copy.) Carrying `n̂ f` as
# two components is what lets `n̂ₓ·n̂ᵧ` factor across the two boxes; the third component carries `f'`.
@createmoment PlanarVectorScalarFarField AbstractFarField     # x, y, scalar

@createmomentcollection ScalarRadiationFieldCollection AbstractRadiationFieldCollection
@createmomentcollection TangentialSphericalRadiationFieldCollection AbstractRadiationFieldCollection
@createmomentcollection SphericalVectorRadiationFieldCollection AbstractRadiationFieldCollection
@createmomentcollection CartesianVectorRadiationFieldCollection AbstractRadiationFieldCollection
@createmomentcollection PairedSphericalRadiationFieldCollection AbstractRadiationFieldCollection
@createmomentcollection PlanarVectorScalarRadiationFieldCollection AbstractRadiationFieldCollection

@createmomentcollection ScalarReceiveFieldCollection AbstractReceiveFieldCollection
@createmomentcollection TangentialSphericalReceiveFieldCollection AbstractReceiveFieldCollection
@createmomentcollection SphericalVectorReceiveFieldCollection AbstractReceiveFieldCollection
@createmomentcollection CartesianVectorReceiveFieldCollection AbstractReceiveFieldCollection
@createmomentcollection PairedSphericalReceiveFieldCollection AbstractReceiveFieldCollection
@createmomentcollection PlanarVectorScalarReceiveFieldCollection AbstractReceiveFieldCollection

@linkmomenttocollection ScalarFarField ScalarRadiationFieldCollection
@linkmomenttocollection ScalarFarField ScalarReceiveFieldCollection

@linkmomenttocollection TangentialSphericalFarField TangentialSphericalRadiationFieldCollection
@linkmomenttocollection TangentialSphericalFarField TangentialSphericalReceiveFieldCollection

@linkmomenttocollection SphericalVectorFarField SphericalVectorRadiationFieldCollection
@linkmomenttocollection SphericalVectorFarField SphericalVectorReceiveFieldCollection

@linkmomenttocollection CartesianVectorFarField CartesianVectorRadiationFieldCollection
@linkmomenttocollection CartesianVectorFarField CartesianVectorReceiveFieldCollection

@linkmomenttocollection PairedSphericalVectorFarField PairedSphericalRadiationFieldCollection
@linkmomenttocollection PairedSphericalVectorFarField PairedSphericalReceiveFieldCollection

@linkmomenttocollection PlanarVectorScalarFarField PlanarVectorScalarRadiationFieldCollection
@linkmomenttocollection PlanarVectorScalarFarField PlanarVectorScalarReceiveFieldCollection

# The component count of each representation -- the single source of truth. Keyed on the
# unparameterized type so `fielddim(TangentialSphericalFarField)` works before any `D` is chosen
# (that is exactly what the constructors below need), and on `<:` so a fully instantiated
# `TangentialSphericalFarField{2,Matrix{ComplexF64},...}` resolves to the same answer.
#
# The THREE `== 3` entries are the point of this whole file: same count, three different meanings,
# three different types. `SphericalVectorFarField` is (r,θ,φ), `CartesianVectorFarField` is (x,y,z),
# and `PlanarVectorScalarFarField` is (x, y, scalar) -- and no two of them transform alike.
fielddim(::Type{<:ScalarFarField}) = 1
fielddim(::Type{<:TangentialSphericalFarField}) = 2
fielddim(::Type{<:SphericalVectorFarField}) = 3
fielddim(::Type{<:CartesianVectorFarField}) = 3
fielddim(::Type{<:PairedSphericalVectorFarField}) = 6
fielddim(::Type{<:PlanarVectorScalarFarField}) = 3

fielddim(::Type{<:ScalarRadiationFieldCollection}) = 1
fielddim(::Type{<:TangentialSphericalRadiationFieldCollection}) = 2
fielddim(::Type{<:SphericalVectorRadiationFieldCollection}) = 3
fielddim(::Type{<:CartesianVectorRadiationFieldCollection}) = 3
fielddim(::Type{<:PairedSphericalRadiationFieldCollection}) = 6
fielddim(::Type{<:PlanarVectorScalarRadiationFieldCollection}) = 3

fielddim(::Type{<:ScalarReceiveFieldCollection}) = 1
fielddim(::Type{<:TangentialSphericalReceiveFieldCollection}) = 2
fielddim(::Type{<:SphericalVectorReceiveFieldCollection}) = 3
fielddim(::Type{<:CartesianVectorReceiveFieldCollection}) = 3
fielddim(::Type{<:PairedSphericalReceiveFieldCollection}) = 6
fielddim(::Type{<:PlanarVectorScalarReceiveFieldCollection}) = 3

fielddim(x::Union{AbstractFarField,AbstractFieldCollection}) = fielddim(typeof(x))

# Constructors, generated once per concrete far-field type. All six get the identical set, and
# all of them take the component count from `fielddim` rather than restating it.
#
# The scalar element type is supplied either as a VALUE whose type is used (`F(γ, sampling)`,
# matching how the operator's γ is threaded through the call sites) or as a TYPE argument
# (`F(ComplexF64, sampling)`, replacing the old `FarField{dim,T}(sampling)` form). The type is
# passed positionally as `::Type{T}` rather than as a leading type parameter (`F{T}(sampling)`)
# because the macro-generated struct's FIRST parameter is `D`: `F{ComplexF64}` would bind
# `ComplexF64` into the `D` slot, which is legal Julia but reads as a component count and would be
# a trap for the next reader.
#
# The default leaves storage undefined for every numeric type, including BigFloat and
# Complex{BigFloat}. Request zero-filled storage with `Val(:zero)`. There is no
# other initializer form: no `Symbol`, no `Val(:undef)`, no bare `undef`.
function _farfieldconstructors(farfieldtypename)
    @eval function $farfieldtypename(
        ::Type{T}, sampling::Sampling.AbstractSampling
    ) where {T}
        return $farfieldtypename{fielddim($farfieldtypename),T}(size(sampling))
    end

    # Same, from an explicit array size instead of a sampling. Saves callers that already hold a
    # size tuple from having to restate the component count (which is the whole point of keying
    # allocation on the type).
    @eval function $farfieldtypename(
        ::Type{T}, arraysize::NTuple{N,I}
    ) where {T,N,I<:Integer}
        return $farfieldtypename{fielddim($farfieldtypename),T}(arraysize)
    end

    @eval function $farfieldtypename(
        γ::T, sampling::Sampling.AbstractSampling
    ) where {T<:Number}
        return $farfieldtypename(T, sampling)
    end

    @eval function $farfieldtypename(
        ::Type{T}, sampling::Sampling.AbstractSampling, ::Val{:zero}
    ) where {T}
        # Use `size(sampling)` for both undefined and zero-filled storage. A 1-tuple gives a
        # `Vector`; a rectangular grid gives `(Nφ, Nθ)`. Neither path requires a scalar Nφ.
        D = fielddim($farfieldtypename)
        sz = size(sampling)
        A = Array{T,length(sz)}

        fvector = Vector{A}(undef, D)
        for i in eachindex(fvector)
            fvector[i] = zeros(T, sz)
        end
        return $farfieldtypename(SVector{D,A}(fvector))
    end

    @eval function $farfieldtypename(
        γ::T, sampling::Sampling.AbstractSampling, ::Val{:zero}
    ) where {T<:Number}
        return $farfieldtypename(T, sampling, Val(:zero))
    end

    return nothing
end

for farfieldtypename in (
    :ScalarFarField,
    :TangentialSphericalFarField,
    :SphericalVectorFarField,
    :CartesianVectorFarField,
    :PairedSphericalVectorFarField,
    :PlanarVectorScalarFarField,
)
    _farfieldconstructors(farfieldtypename)
end

# The bare, unparameterized type behind a concrete far field -- `TangentialSphericalFarField` from
# `TangentialSphericalFarField{2,Matrix{ComplexF64},Ptr{ComplexF64},...}`. Lets an operation that
# must return the SAME representation as its input (mirroring, interpolation, receive coupling)
# allocate its result without naming any representation, and without re-deriving one from a
# component count. `Base.typename(F).wrapper` is the documented way to strip parameters back to
# the UnionAll; going through this helper keeps that incantation in one place.
basefarfieldtype(::Type{F}) where {F<:AbstractFarField} = Base.typename(F).wrapper
basefarfieldtype(ff::AbstractFarField) = basefarfieldtype(typeof(ff))

function Nθ(ff::AbstractFarField)
    return size(ff[1], 2)
end

function Nφ(ff::AbstractFarField)
    return size(ff[1], 1)
end

# Representation-aware accessors (Fθ, Fφ, Fr, F, Fx, Fy, Fz, component), the mirror/receive rule
# tables, and mirror/mirror! live in `representation.jl`, included right after this file -- they
# dispatch on the concrete types defined above, so those must exist first.
