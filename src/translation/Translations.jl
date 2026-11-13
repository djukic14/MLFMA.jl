"""
    MLFMA.Translations

The translation operators that move a far field from one box to another at the same level, and the
collections that store them.

Defines `MLFMATranslator`, plain and symmetry-reduced I2O collections, and the dummy collection for
an empty level range. Sampling supplies the symmetry policies and angular permutations.
"""
module Translations

using BEAST
using H2Factory
using H2Trees
using LinearAlgebra
using SpecialFunctions
using StaticArrays

using ..FarFields
using ..Harmonics
using ..Interpolation
using ..Operators
using ..Sampling
using ..Seams
using ..Phase

# Import explicitly so unqualified method definitions extend Sampling's `supportedsymmetry`.
import ..Sampling: supportedsymmetry

include("translation.jl")
include("translationcollection.jl")
# Load after `translationcollection.jl`: this builder reuses its translator and level indexing.
include("symmetrytranslationcollection.jl")

export DummyTranslationCollection,
    MLFMATranslator,
    SymmetryTranslationCollection,
    TRANSLATIONCANCELLATIONBUDGET,
    TranslationCollection,
    collectcylindricalhankelsecond,
    collectcylindricalhankelsecond!,
    collectlegendre,
    collectlegendre!,
    collectsphericalhankelsecond,
    collectsphericalhankelsecond!,
    legendre0,
    legendre1,
    sphericalhankelsecond,
    sphericalhankelsecond0,
    sphericalhankelsecond1,
    translationlevelindex,
    translationoperator!

end
