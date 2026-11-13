"""
    MLFMA.SphericalHarmonics

Orthonormal complex spherical harmonics and the transform pair over a set of sampling directions.

MLFMA samples each leaf basis function's far-field pattern at the finest level's angular rate.
The translation operator sets that rate for the whole box. An individual basis function has smaller
support and needs less angular bandwidth. In a measurement at 105k unknowns, the leaf radiation
collection used **66 % of a map's memory** (5.0 GB of 7.6 GB). Storing harmonic coefficients instead
compressed the patterns **10.4×** at ε = 1e-4, for a **2.5× smaller map**. This is Eibert's
SE-MLFMA (2005).

The MLFMA grids do not change. Aggregation between levels, translation, and interpolation/anterpolation
still use the ordinary tensor samplings. Harmonics appear only between the MoM basis and the leaf
group. The far-field collections wrap the storage and transforms defined here.

This submodule can be tested without a tree or assembler. Orthonormality, Parseval, and closed-form
low-order harmonics provide analytic checks. The submodule is not exported.

```julia
using MLFMA.SphericalHarmonics

t = SphericalHarmonicTransform(ComplexF64, sampling, 9)
a = analyze(t, F)             # F: samples (rows) x components (columns)
F′ = synthesize(t, a)         # back to samples
truncationorder(a, 1e-4)      # the order that tail actually needs
```
"""
module SphericalHarmonics

using FastGaussQuadrature
using LinearAlgebra
using StaticArrays

# Import the grid provider (`sphericalsamplepoints`) and its dispatch types from their owner.
import ...Sampling

include("ylm.jl")
# Transforms use the harmonics defined above.
include("transform.jl")
# Storage and aggregation use the transforms.
include("storage.jl")
# Frame projection converts synthesised Cartesian patterns to tangential far fields.
# It uses `order`/`numcoefficients` from `transform.jl`, but does not depend on storage.
include("frame.jl")
# The tangential vector basis is an alternative to three scalar Cartesian expansions.
# It uses `ylm!` and shares `gausslegendregrid`.
include("vectorharmonics.jl")

export lmindex, numcoefficients, degreeof, ylm, ylm!
export SphericalHarmonicTransform, gausslegendregrid
export synthesize, synthesize!, analyze
export degreeenergy, truncationorder
export SHETAILFLOOR, leveltruncationorder, truncatecoefficients
export SphericalHarmonicStorage, coefficientmatrix, aggregate!, aggregatelevel!
export analyzelevel!, testcoefficients!, testlevel!
export sphericalframe, parity, mirrorsynthesis, project!, projectthreaded!
export vshindex, numvshcoefficients, vshdegreeof, vshauxiliaries!
export VectorSphericalHarmonicTransform, vshsynthesize, vshsynthesize!, vshanalyze
export vshparity, vshmirrorblocks, vshdegreeenergy, vshtruncationorder
export vshleveltruncationorder

end
