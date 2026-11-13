"""
    MLFMA.Equidistant

A far-field representation with equidistant θ and φ grids and two interchangeable interpolation
backends. Its grid, quadrature, backends, and adjoints can be checked against analytic functions
without a tree or assembler.

To use it in an MLFMA map:

    IT = typeof(MLFMA.Equidistant.EquidistantInterpolator{ComplexF64}(FFTBackend()))
    MLFMA.MLFMAStrategy(operator, X;
        samplingstrategy=MLFMA.SamplingStrategy(; samplingbuilder=EquidistantSamplingBuilder{Float64}()),
        leveltransferstrategy=MLFMA.LevelTransferStrategy(IT, phaseshifter))

The sampling builder and the interpolator family must agree -- the interpolator errors at
construction if handed a `SphericalSampling`, rather than failing at the first apply.

Both backends are allocation-free with matching scratch from `MLFMA.interimscratch` and fall back
to their allocating reference paths without it.

Both backends use `EquidistantSampling` and the same spherical continuation:

  - [`FFTBackend`](@ref) -- global spectral resampling. Exact (machine precision) for any field the
    grid resolves; unresolved content aliases.
  - [`LagrangeBackend`](@ref) -- separable local tensor-product Lagrange. Algebraically convergent in
    stencil order and in oversampling; cheaper and local, but needs oversampling to be accurate.

The FFT backend needs a source grid that resolves the field's bandwidth. The Lagrange backend
also needs oversampling for accurate local interpolation; increasing the stencil order reduces
its error.

Quadrature uses `exactquadratureweights`, not the midpoint rule. The sphere's Jacobian becomes
`|sin θ|` on the torus, whose kink prevents spectral convergence of the midpoint rule.
`exactquadratureweights` integrates this factor analytically for resolved modes; see `quadrature.jl`.
"""
module Equidistant

# Extend the owning layers' functions so the pipeline sees these methods. Importing the layers
# directly also gives access to their private helpers without adding top-level exports.
import ..Sampling
import ..Interpolation
import ..FarFields
using ..Sampling:
    Nθ,
    Nφ,
    θs,
    φs,
    θsφs,
    cartesiansamplepoints,
    cutoffparameter,
    mirrorindices,
    arraytype,
    NθNφ

using FFTW
# The real-to-real transforms and their kind constants are not among FFTW's exports, so they need
# naming explicitly. `fftscratch.jl` runs the θ direction as a DCT/DST pair rather than a complex FFT
# over a doubled torus.
using FFTW: plan_r2r!, REDFT10, REDFT01, RODFT10, RODFT01
using LinearAlgebra
using StaticArrays: SVector

# For `H2Factory.moments`, the accessor every far-field moment is read through. The interpolator
# protocol is stated in terms of moments rather than bare arrays, so the adapter in `interpolation.jl`
# needs it; nothing else in this submodule touches H2Factory.
using H2Factory

# For `H2Trees.LatticeSymmetry` in `samplinginterface.jl`.
using H2Trees

include("sampling.jl")
include("quadrature.jl")

# The MLFMA-facing accessor set, after the grid and the weights it reports.
include("samplinginterface.jl")

# Both backends dispatch `resample` on their own type; the sampling and the continuation come first.
include("fftbackend.jl")
include("lagrangebackend.jl")

# The adjoint of resampling, after both backends whose pipelines it transposes.
include("transposedresample.jl")

# Buffer-reusing versions of both spectral directions, after the reference implementations they are
# checked against. It reuses their coefficient movers but builds no torus of its own.
include("fftscratch.jl")

# The same for the local backend, after `sphericalscatter!` (which its adjoint scatters through) and
# after `_stencil`/`_stagegrid`, whose results it precomputes rather than rebuilding per apply.
include("lagrangescratch.jl")

# The diagonal translation operator, after the quadrature it is validated with.
include("translation.jl")

# The weighted adjoint of resampling, after both the backends and the weights it is defined by.
include("anterpolation.jl")

# The two-component (vector) case, after the scalar resampling it is built from.
include("tangential.jl")

# MLFMA's interpolator/anterpolator protocol over both backends. Last, because it is an adapter and
# uses nearly everything above it.
include("interpolation.jl")

export EquidistantSampling, EquidistantSamplingBuilder, SPHERICAL, ANTISPHERICAL
export Nθ, Nφ, θs, φs, extend, extendedθs, sphericalget, reflectindex
export quadratureweights, exactquadratureweights, integrate
export FFTBackend, LagrangeBackend, resample
export resampleadjoint, resamplecoefficientsadjoint, sphericalscatter!
export resamplereflectedcoefficients, resamplereflectedcoefficientsadjoint
export FFTScratch, LagrangeScratch, resample!, resampleadjoint!
export translationoperator, sphericalhankel1, legendrep, directions, excessbandwidth
export anterpolate, anterpolatedense, resamplematrix, innerproduct
export TangentialField, cartesian, istransverse, thetahat, phihat, khat
export EquidistantInterpolator, EquidistantAnterpolator, componentparities

end # module Equidistant
