"""
    MLFMA.TestSupport

Provides a deterministic far-field assembler for tests without performing quadrature.
It uses `Assembly`'s `BoxFarFieldSampling` and `leafids`, so `Assembly` must be loaded first.
The unit-phase test shifter belongs to `Phase`.
"""
module TestSupport

using BEAST
using H2Factory
using H2Trees
using LinearAlgebra
using StaticArrays

using ..FarFields
using ..Harmonics
using ..Interpolation
using ..Operators
using ..Sampling
using ..Seams
using ..Assembly
using ..Phase
using ..Translations

# Generics this layer adds methods to but does not own.
import ..Assembly: quadstrategy, space
# Private assembler helper, imported independently of the compatibility exports.
import ..Assembly: _dimensionvalue
import ..FarFields: operator
import ..Seams: farfieldtype

include("dummyfarfieldassembler.jl")

export DummyFarFieldAssembler, DummyFarFieldAssemblerBuilder, dummyfarfieldvalue

end
