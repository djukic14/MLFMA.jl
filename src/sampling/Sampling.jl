"""
    MLFMA.Sampling

The angular grids a far field is represented on, and how a lattice symmetry acts on them.

Defines spherical product grids and circular grids, their builders, sample points, quadrature
weights, mirror indices and symmetry permutations. It has no dependencies on other MLFMA modules.
Its exports are also available as `MLFMA.name`.
"""
module Sampling

using AutoHashEquals
using FastGaussQuadrature
using H2Trees
using LinearAlgebra
using StaticArrays

include("abstractsampling.jl")
include("sphericalsampling.jl")
include("circularsampling.jl")
include("symmetryaction.jl")

export AbstractSampling,
    AbstractSphericalSampling,
    CircularSampling,
    CircularSamplingBuilder,
    Nθ,
    NθNφ,
    Nφ,
    NφCircular,
    SphericalSampling,
    SphericalSamplingBuilder,
    _findθclosest,
    _findθfirst,
    angularpermutation,
    angularpermutations,
    arraytype,
    cartesiansamplepoints,
    cutoffparameter,
    findθφ,
    globalindex,
    mirrorindices,
    samplingrate,
    sphericalsamplepoints,
    supportedsymmetry,
    virtualextension,
    wθs,
    θ,
    θindex,
    θs,
    θsφs,
    θφindicesfromglobalindex,
    φ,
    φindex,
    φs

end
