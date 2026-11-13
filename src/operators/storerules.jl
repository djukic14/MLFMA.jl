# The layout and the projection into it are independent choices. `farfieldtype(op)` selects the
# layout; the store rule defined here projects BEAST's raw values into it. For example,
# `MWHyperSingular3D` stores a `ScalarFarField` by taking a Cartesian vector's radial component,
# whereas scalar Helmholtz operators store BEAST's values directly in the same layout.
# `FarFieldStorer` holds the rule and the assembly state.
#
#   farfieldtype(op)          # stored moment representation
#   farfieldstorerule(op)     # assembly projection into that representation
#   interpolationtrait(op)    # level-transfer behaviour
#
# MLFMA specializes `H2Factory.buildstorer` for its own assemblers in `assembly/store/store.jl`
# and builds a `FarFieldStorer` from the rule, without defining `storetrait` on BEAST's types.

# --- rules ----------------------------------------------------------------------------------------

"""
    AbstractFarFieldStoreRule

How a raw BEAST far-field value is projected into the moment layout `farfieldtype(op)` names.

A rule is a value, not a storer type: the same rule applies whether the moments are one flat block
(the non-tree assembler path) or one block per leaf, and whether or not a trigonometric frame cache
is needed. [`FarFieldStorer`](@ref MLFMA.Assembly.FarFieldStorer) holds a rule plus that state.
"""
abstract type AbstractFarFieldStoreRule end

"""
    StoreAsFarFieldValue()

Write BEAST's value directly, component by component, without a projection or frame cache.
Scalar Helmholtz operators use this rule because their far-field values already match the stored
moments. It serves the same purpose as `H2Factory.StoreAsIs` within MLFMA's rule-based storer.
"""
struct StoreAsFarFieldValue <: AbstractFarFieldStoreRule end

"""
    StoreCartesianAsSphericalVector()

Project a Cartesian 3-vector onto the full spherical frame, keeping all three components
`(r, θ, φ)`. Used by `MWSingleLayer3D`, which stores a `SphericalVectorFarField`.
"""
struct StoreCartesianAsSphericalVector <: AbstractFarFieldStoreRule end

"""
    StoreCartesianAsTangentialSpherical()

Project a Cartesian 3-vector onto the spherical frame and keep only the tangential components
`(θ, φ)`, discarding the radial one. Used by `MWEFIE3D`, `MWDoubleLayer3D` and
`DoubleLayerRotatedMW3D`, which store a `TangentialSphericalFarField`.
"""
struct StoreCartesianAsTangentialSpherical <: AbstractFarFieldStoreRule end

"""
    StoreCartesianAsRadialScalar()

Project a Cartesian 3-vector onto the spherical frame and keep only the radial component, giving a
single scalar. Used by `MWHyperSingular3D`.

This stores a
`ScalarFarField`, exactly like the Helmholtz scalar operators, but reaches it by projection rather
than by [`StoreAsFarFieldValue`](@ref). The layout does not determine the rule.
"""
struct StoreCartesianAsRadialScalar <: AbstractFarFieldStoreRule end

"""
    StoreDoubleCartesianAsPairedSpherical()

Project a pair of Cartesian 3-vectors, packed back to back in one 6-element value, onto the
spherical frame -- giving six components. Used by `BEAST.HH3DHyperSingularFDBIO`, which stores a
`PairedSphericalVectorFarField`.
"""
struct StoreDoubleCartesianAsPairedSpherical <: AbstractFarFieldStoreRule end

"""
    farfieldstorerule(op)

The [`AbstractFarFieldStoreRule`](@ref) for `op`'s leaf far-field assembly.

This projection table is independent of `farfieldtype(op)`: a layout does not determine how raw
values are projected into it. See the note at the top of this file.
"""
farfieldstorerule(op) = farfieldstorerule(typeof(op))

# Define each entry on `Type{...}` so both type calls and the instance forwarder resolve to it.
# Mixing type and instance entries can leave a type call in the generic forwarder, which recurses
# until a `StackOverflowError`. Separate methods make each entry's call form explicit.
farfieldstorerule(::Type{MWHyperSingular3D{U,K}}) where {U,K} =
    StoreCartesianAsRadialScalar()

farfieldstorerule(::Type{BEAST.MWSingleLayer3D{U,K}}) where {U,K} =
    StoreCartesianAsSphericalVector()

farfieldstorerule(::Type{<:BEAST.Helmholtz3DOp{U,K}}) where {U,K} = StoreAsFarFieldValue()

# The 2-D scalar operators store their value as-is, exactly as the 3-D scalar ones do -- there is no
# frame to project onto, since `ScalarFarField` is one component per direction.
farfieldstorerule(::Type{BEAST.HH2DSingleLayerFDBIO{U,K}}) where {U,K} =
    StoreAsFarFieldValue()

farfieldstorerule(::Type{BEAST.HH2DDoubleLayerFDBIO{U,K}}) where {U,K} =
    StoreAsFarFieldValue()

farfieldstorerule(::Type{BEAST.HH2DDoubleLayerTransposedFDBIO{U,K}}) where {U,K} =
    StoreAsFarFieldValue()

farfieldstorerule(::Type{BEAST.HH3DHyperSingularFDBIO{U,K}}) where {U,K} =
    StoreDoubleCartesianAsPairedSpherical()

# Store these components as-is. Unlike the 3-D rule, the circular field uses no spherical frame:
# `[n̂₁f, n̂₂f, f']` contains two fixed in-plane Cartesian components and a scalar.
#
# Verified: the factorization reproduces a dense BEAST far block at 1.59e-13 using exactly these raw
# components, with no projection anywhere.
farfieldstorerule(::Type{BEAST.HH2DHyperSingularFDBIO{U,K}}) where {U,K} =
    StoreAsFarFieldValue()

farfieldstorerule(::Type{MWEFIE3D{U,K}}) where {U,K} = StoreCartesianAsTangentialSpherical()

farfieldstorerule(::Type{MWDoubleLayer3D{U,K}}) where {U,K} =
    StoreCartesianAsTangentialSpherical()

farfieldstorerule(::Type{DoubleLayerRotatedMW3D{U,K}}) where {U,K} =
    StoreCartesianAsTangentialSpherical()
