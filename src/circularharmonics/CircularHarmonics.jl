"""
    MLFMA.CircularHarmonics

Orthonormal circular harmonics `ψ_m(φ) = exp(imφ)/√(2π)` and the transform pair over a set of sampling
angles (the 2-D counterpart of [`MLFMA.Harmonics.SphericalHarmonics`](@ref)).

MLFMA samples each leaf basis function's far-field pattern at the finest level's angular rate.
The translation operator sets that rate for the whole box. An individual basis function has smaller
support and needs less bandwidth. Storing harmonic coefficients removes that redundancy.

The measured benefits differ between 2-D and 3-D:

  - 3-D compresses **10.4x** at ε = 1e-4 and is **FASTER** (0.57x serial, 0.22x threaded), because the
    sampled leaf multiply there is memory-bound -- 4.5 MB per group at ~0.5 flop/byte.
  - 2-D compresses **1.47x** at ε = 1e-5 and is **1.67x SLOWER** on ordinary fixtures, because a 2-D
    leaf block is ~3.5 KB and never leaves cache. It reaches break-even only above ~64 basis functions
    per leaf and never gets past 0.89x; BLAS threading makes the low-occupancy case worse rather than
    better.

Both halves of that are the same O(L) versus O(L²) asymmetry, which shows up in three places at once:
`2L+1` coefficients against `(L+1)²`, `2(L+1)` directions against `2(L+1)·4(L+1)`, and `O(boxsize/h)`
basis functions in a box on a curve against `O((boxsize/h)²)` in a box on a surface. The first of
those is why 2-D can never win as widely however the constants fall: the sphere's product grid spends
`8(L+1)²` samples on an `(L+1)²`-dimensional space, so SHE is 5.33x ahead before any bandwidth margin,
while `2(L+1)` samples against `2M+1` coefficients is a dead heat at `M = L`.

This backend provides the same storage option as the 3-D backend. Select it to save memory at
roughly break-even time, not for speed. `MLFMA.LEAFSTORAGERECOM` remains `SampledLeafStorage`.
See [`MLFMA.Assembly.CircularHarmonicLeafStorage`](@ref) for the crossover table.

The MLFMA grids do not change. Aggregation between levels, translation, and interpolation/anterpolation
still use the ordinary `CircularSampling` grids. Harmonics appear only between the MoM basis and the
leaf group. The leaf grid must resolve the translation bandwidth, not just the pattern bandwidth.
Storing fewer coefficients and expanding on demand preserves that grid.

This submodule can be tested without a tree or assembler. Orthonormality, Parseval, and closed-form
modes provide analytic checks. The submodule is not exported.

```julia
using MLFMA.CircularHarmonics

t = CircularHarmonicTransform(ComplexF64, sampling, 7)
a = analyze(t, F)             # F: samples (rows) x components (columns)
F′ = synthesize(t, a)         # back to samples
truncationorder(a, 1e-5)      # the order that tail actually needs
```
"""
module CircularHarmonics

using LinearAlgebra
using StaticArrays

# Import the circle's grid and dispatch type from their owner.
import ...Sampling

# Storage uses the transforms and truncation rule.
include("transform.jl")
# Group storage and aggregation.
include("storage.jl")

export mindex, orderof, numcoefficients
export CircularHarmonicTransform, circulargrid
export synthesize, synthesize!, analyze
export modeparity, mirrorsynthesis
export modeenergy, truncationorder
export CHETAILFLOOR, CHETAILRATIO, leveltruncationorder, truncatecoefficients
export CircularHarmonicStorage, coefficientmatrix, aggregate!, aggregatelevel!
export analyzelevel!, testcoefficients!, testlevel!

end
