"""
    MLFMA.Operators

Define each BEAST operator's far-field representation, radiation and receive operators, coupling,
and store rule. Also provide wrappers for operators BEAST does not supply.

This layer owns the operator property tables and store-rule types. It depends on `FarFields` for
representation types, `ComponentMap` and `fielddim`; `FarFields` does not depend on this layer.

`Seams` declares `farfieldtype`, `radiationfieldcollectiontype` and `receivecouplingrule` so
far-field kernels can call them during multiplication. This module imports those functions and
defines their methods. It also extends `interpolationtrait`, which `Interpolation` owns.

`farfieldstorerule` needs no declaration in `Seams`: its other consumers are the collection
builders in `Assembly`, which can depend on this layer directly.
"""
module Operators

# These dependencies include names used by macros and by methods called at runtime.
using BEAST
using CompScienceMeshes
using H2Factory
using H2Trees
using LinearAlgebra
using StaticArrays

using ..FarFields
using ..Harmonics
using ..Interpolation
using ..Sampling
using ..Seams

# Import functions owned by other layers so the definitions below extend them.
# `test/Package/layers.jl` checks that these methods belong to the intended functions.
import ..Interpolation: interpolationtrait
import ..Seams: farfieldtype, radiationfieldcollectiontype, receivecouplingrule

include("maxwell/mwwrappers.jl")
include("maxwell/mwfarfields.jl")
include("helmholtz/hhfarfields.jl")
include("operatorutils.jl")
# Store rules select the projection into a far-field layout, independently of `farfieldtype`.
include("storerules.jl")

# Expose these names through `MLFMA`. `@efie` and `@mw3Dhypersingular` generate `MWEFIE3D` and
# `MWHyperSingular3D`; source parsing alone cannot find them. `test/Package/callsites.jl` checks
# that the export list covers the call sites.
export AbstractFarFieldStoreRule,
    HHDoubleLayerFarField,
    HHFarField,
    HHHyperSingularFarField2D,
    HHHyperSingularFarField3D,
    HHSingleLayerFarField,
    MWEFIE3D,
    MWHyperSingular3D,
    MWSingleLayerFarField3D,
    StoreAsFarFieldValue,
    StoreCartesianAsRadialScalar,
    StoreCartesianAsSphericalVector,
    StoreCartesianAsTangentialSpherical,
    StoreDoubleCartesianAsPairedSpherical,
    _sortbasisindices!,
    checklazyreceive,
    efie,
    farfielddims,
    farfieldstorerule,
    haslazyreceive,
    hypersingular,
    radiationcollectiontype,
    radiationoperator,
    receivecollectiontype,
    receivefieldcollectiontype,
    receiveoperator,
    singlelayer,
    sortbasis!,
    farfieldvaluetype,
    α,
    γ

end
