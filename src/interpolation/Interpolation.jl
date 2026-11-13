"""
    MLFMA.Interpolation

Moving a far field between the angular sampling rates of two levels, in both directions.

Owns the interpolation traits, the interpolator and anterpolator families and their per-level-pair
information caches, the transposed and conjugated wrappers, the interim scratch, the builders that
select a family, and the diagnostics submodule.

It depends on Sampling for grids and angular rates, and on FarFields for field representations.
It does not depend on assembly or tree traversal.

`interpolationtrait` and `interpolatortype` are owned here and extended higher up, by Operators and
by the assembly strategies. Both are `import`ed at the top level for that reason.
"""
module Interpolation

# These imports also support names used inside macros and at call time.
using FFTW
using H2Factory
using LinearAlgebra
using SparseArrays
using StaticArrays

using ..FarFields
using ..Sampling

include("interpolationtraits.jl")
include("abstractinterpolation.jl")

include("interpolator/interpolationinfo/gausslegendreinterpolation.jl")
include("interpolator/interpolationinfo/trapezoidalinterpolation.jl")
include("interpolator/interpolationinfo/fftinterpolation.jl")
include("interpolator/interpolationinfo/sphericalinterpolation.jl")
include("interpolator/interpolationinfo/localglobalinterpolation.jl")
include("interpolator/sphericalinterpolation.jl")
include("interpolator/localglobalinterpolation.jl")
# The 2-D family. No interpolation-information type of its own: it caches nothing per level pair,
# and its FFT plans live in the per-(level, chunk) scratch instead.
include("interpolator/circularinterpolation.jl")

include("anterpolator/anterpolationinfo/gausslegendreanterpolation.jl")
include("anterpolator/anterpolationinfo/trapezoidalanterpolation.jl")
include("anterpolator/anterpolationinfo/fftanterpolation.jl")
include("anterpolator/anterpolationinfo/sphericalanterpolation.jl")
include("anterpolator/anterpolationinfo/localglobalanterpolation.jl")
include("anterpolator/sphericalanterpolation.jl")
include("anterpolator/localglobalanterpolation.jl")
include("anterpolator/circularanterpolation.jl")

# After both anterpolator families: it wraps whichever one it is given, and its
# `isrealinterpolation` trait is stated per interpolator family.
include("anterpolator/transposedinterpolation.jl")

# After `transposedinterpolation.jl`, where `isrealinterpolation` is stated: this is the fourth
# direction (conjugated, not transposed) and asks that same question to decide whether it is needed.
include("interpolator/conjugatedinterpolation.jl")

# After both anterpolator families: it dispatches on each of them.
include("interimscratch.jl")

# After every operator type it resolves to, and after `transposedinterpolator`, which it hands out
# alongside the anterpolator.
include("builders.jl")

# The diagnostics submodule needs the samplings, the
# far-field representations and the interpolators, and nothing else: it assembles nothing and
# touches no tree, which is what makes it a diagnostic of interpolation alone.
include("diagnostics/InterpolationDiagnostics.jl")

# The top level re-exports this API; `test/Package/publicapi.jl` checks its scope.
# Equidistant accesses the private `_stageprescaled!` helper through a qualified reference.

export AbstractAnterpolator,
    AbstractInterpolationBuilder,
    AbstractInterpolator,
    CircularAnterpolator,
    CircularInterimScratch,
    CircularInterpolationBuilder,
    CircularInterpolator,
    ConjugatedInterpolator,
    DefaultInterpolation,
    DoubleSphericalInterpolation,
    FFTAnterpolationInformation,
    FFTInterpolationInformation,
    GaussLegendreAnterpolationInformation,
    GaussLegendreInterpolationInformation,
    InterpolationExpansionTrait,
    LevelTransferScratch,
    LocalθGlobalφAnterpolationInformation,
    LocalθGlobalφAnterpolator,
    LocalθGlobalφInterpolationBuilder,
    LocalθGlobalφInterpolationInformation,
    LocalθGlobalφInterpolator,
    PeriodicφInterpolation,
    SphericalAnterpolationInformation,
    SphericalInterpolation,
    SphericalInterpolationInformation,
    SphericalLagrangeAnterpolator,
    SphericalLagrangeInterpolationBuilder,
    SphericalLagrangeInterpolator,
    SphericalθφInterpolation,
    TransposedInterpolator,
    TrapezoidalAnterpolationInformation,
    TrapezoidalInterpolationInformation,
    VirtualExtension,
    VirtualExtensionFlipθSign,
    _anterpolateθ!,
    _anterpolateθtransposed!,
    _anterpolateφbfft!,
    _getanterpolationmatrix,
    _getinterpolationmatrix,
    _interimshape,
    _interpolateθ!,
    _interpolateθtransposed!,
    _interpolationweights,
    _movecircularcoefficients!,
    _movecircularcoefficientsadjoint!,
    anterpolationinformation,
    anterpolator,
    buildanterpolator,
    buildinterpolator,
    buildreverseoperators,
    conjugatedinterpolator,
    converttrait,
    defaultinterpolationtrait,
    elementtype,
    fftplantype,
    getlocalglobalanterpolationinfodict,
    getlocalglobalinterpolationinfodict,
    getsphericalanterpolationinfodict,
    getsphericalinterpolationinfodict,
    ifftplantype,
    interimscratch,
    interpolationinformation,
    interpolationorder,
    interpolationtrait,
    interpolator,
    interpolatortype,
    isrealinterpolation,
    lagrangeweights,
    lagrangeweightsequidistant,
    mulrotatedrows!,
    spanningranges,
    stencil,
    stencilsize,
    takeinterim,
    traitarity,
    transposedinterpolator

export InterpolationDiagnostics

end
