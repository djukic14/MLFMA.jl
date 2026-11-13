"""
    MLFMA.InterpolationDiagnostics

Estimate the error MLFMA's angular interpolation introduces, in isolation from everything else.

This module measures angular interpolation error, excluding BEAST quadrature, addition-theorem
truncation, translation, near-field interactions, and tree construction. It evaluates the exact
angular function independently on both grids. No mesh, operator assembly, or interpolated reference
enters the comparison.

Reach it as `MLFMA.InterpolationDiagnostics`, or bind it:

```julia
using MLFMA
const ID = MLFMA.InterpolationDiagnostics

source = MLFMA.SphericalSampling{Float64}(28, 56)
target = MLFMA.SphericalSampling{Float64}(56, 112)
interpolator = MLFMA.SphericalLagrangeInterpolator{8}(ComplexF64, Float64)

cases = [
    ID.HarmonicCase(; l=10, m=0),
    ID.HarmonicCase(; l=10, m=5),
    ID.HarmonicCase(; l=10, m=10),
    ID.RandomBandlimitedCase(; L=10, seed=0x1234),
]

study = ID.interpolationerror(interpolator, source, target, cases)

ID.summary(study)
ID.worstcase(study)
```

# Sampling pairs, not trees

The API takes two explicit `MLFMA.SphericalSampling`s because that is the whole of what interpolation
depends on: a source grid, a target grid, and an interpolator. A caller who wants the grids a real
tree would use can get them from `MLFMA.sampling(γ, ε, tree; level)` and pass them in; this module
owns no tree convenience wrapper, so the diagnostic cannot quietly start depending on tree
construction -- one of the error sources it exists to exclude.

# Field Representations

`interpolationerror` supports `MLFMA.ScalarFarField` and `MLFMA.TangentialSphericalFarField`.
Use scalar cases for scalar fields and tangential harmonic cases for tangential fields. Other
representations are rejected: their exact cases must respect their own continuation rules and
supply every component, including the radial component of a spherical vector field.

# Plotting

Nothing here loads `PlotlyJS`. Pointwise fields and [`rows`](@ref) are plain arrays and
`NamedTuple`s, so any plotting package can consume them; `examples/plot_interpolation_diagnostics.jl`
shows one way, and the `MLFMAPlotlyJS` extension adds `ploterror`/`plotcurves` when `PlotlyJS` is
loaded.
"""
module InterpolationDiagnostics

# The three layers this submodule actually reads, named individually rather than borrowed from the
# top level. `...Interpolation` and not `..Interpolation` is deliberate: two dots resolve INSIDE the
# parent module, and the parent here IS Interpolation, so naming it needs the grandparent.
import ...Sampling
import ...FarFields
import ...Interpolation
using LinearAlgebra: LinearAlgebra
using Random: Random

# `summary` is Base's, extended for the study type rather than shadowed, so `ID.summary(study)` works
# and still obeys Base's contract of returning a `String`.
import Base: summary

include("cases.jl")
include("errors.jl")
include("results.jl")
include("evaluate.jl")
# After `evaluate.jl`: the vector harmonics differentiate and reflect its `normalizedlegendre`.
include("vectorharmonics.jl")
include("summary.jl")
# After `evaluate.jl` and `summary.jl`: it reuses `fillexact!`, `_ascases`, `_checkfarfieldtype` and `rho`.
include("adjointness.jl")

# Plotting entry points; the `MLFMAPlotlyJS` extension supplies their methods.
#
# Keep these fallbacks untyped and the extension's methods more specific. This avoids method
# overwriting during precompilation and selects the extension when its argument types match.
"""
    ploterror(study, i, sampling; kind=:relative)

Plot result `i`'s pointwise error on the target sphere.

Requires `PlotlyJS` -- the real method lives in the `MLFMAPlotlyJS` extension, so the core diagnostics
stay independent of any plotting package. `sampling` must be the TARGET sampling the study
interpolated onto; the extension checks that its direction count matches the error field.
"""
function ploterror(study, i, sampling; kwargs...)
    return error(
        "InterpolationDiagnostics.ploterror requires PlotlyJS: run `using PlotlyJS` to load the " *
        "MLFMAPlotlyJS extension, which supplies this method. (If PlotlyJS IS loaded, this " *
        "fallback means the arguments did not match the extension's method: it wants a study, an " *
        "integer index and an `MLFMA.SphericalSampling`.)",
    )
end

"""
    plotcurves(study; x=:rho, y=:rel_l2, group=:casekind)

Plot one error curve per group across the study's cases, with a logarithmic error axis.

Requires `PlotlyJS`, as with [`ploterror`](@ref). `x`, `y` and `group` name fields of [`rows`](@ref).
"""
function plotcurves(study; kwargs...)
    return error(
        "InterpolationDiagnostics.plotcurves requires PlotlyJS: run `using PlotlyJS` to load the " *
        "MLFMAPlotlyJS extension, which supplies this method. (If PlotlyJS IS loaded, this " *
        "fallback means the argument was not an `InterpolationErrorStudy`.)",
    )
end

end
