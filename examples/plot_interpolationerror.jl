# Relative error of interpolating a leaf far field onto a finer direction grid.
#
# Assembles the same far field twice -- once on `sampling`, once on the finer
# `samplinginterpolation` -- interpolates the coarse one up, and plots how far it lands from the
# directly-assembled fine one. The finer assembly is the reference, so what is plotted is
# interpolation error alone and not discretization error.
#
# RUN IT WITH THE TEST ENVIRONMENT: `julia --project=test examples/plot_interpolationerror.jl`, or
# activate `test` in the REPL/VSCode first. There is no `examples/Project.toml`, and the package
# environment does not have `PlotlyJS`: it is a WEAK dependency of MLFMA, backing the `MLFMAPlotlyJS`
# extension that supplies `MLFMA.sphericaltrace`, so it is not installed alongside the package. Any
# environment with PlotlyJS in it works -- `test` is simply the one that already has it.

using PlotlyJS
using LinearAlgebra
using MLFMA
using CompScienceMeshes
using BEAST

fn = joinpath(dirname(@__FILE__), "meshes/twospheres.msh")
m = CompScienceMeshes.readmesh(fn)

# CONFIGURATION: pick ONE line. Space and operator are chosen together on purpose -- they are
# coupled, and pairing them by hand is the way this script breaks. Maxwell's operators are defined on
# a div-conforming space (`raviartthomas`), Helmholtz's on a scalar one (`lagrangec0d1`); crossing
# them fails deep inside BEAST's assembly with an error that says nothing about the real mistake.
# Each line below is internally consistent, so switching configuration cannot produce that pairing.
const WAVENUMBER = 1.0

X, operator = raviartthomas(m), Maxwell3D.singlelayer(; wavenumber=WAVENUMBER)
# X, operator = raviartthomas(m), MLFMA.MWEFIE3D(; wavenumber=WAVENUMBER)
# X, operator = raviartthomas(m), Maxwell3D.doublelayer(; wavenumber=WAVENUMBER)
# X, operator = raviartthomas(m), MLFMA.MWHyperSingular3D(; wavenumber=WAVENUMBER)
# X, operator = lagrangec0d1(m), Helmholtz3D.singlelayer(; wavenumber=WAVENUMBER)
# X, operator = lagrangec0d1(m), Helmholtz3D.doublelayer(; wavenumber=WAVENUMBER)
# X, operator = lagrangec0d1(m), Helmholtz3D.hypersingular(; wavenumber=WAVENUMBER)

# stencil size for the local interpolation
stencilsize = 5

# Both grids from one `L` via `NθNφ`, so the reference is the same far field at exactly twice the
# bandwidth rather than at an unrelated hand-picked rate.
L = 50
Nθ, Nφ = MLFMA.NθNφ(L)
sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)

Nθinterpolation, Nφinterpolation = MLFMA.NθNφ(2 * L)
samplinginterpolation = MLFMA.SphericalSampling{Float64}(Nθinterpolation, Nφinterpolation)

# One construction policy, reused for both samplings -- which is the point of the builder: the two
# collections below cannot end up assembled with different quadratures or different assemblers, and
# the reference really is the same far field at a finer rate.
#
# Order 7 is what the removed `RadiationFieldCollection(op, sampling, X; quadstrat=...)` keyword
# defaulted to, spelled out here so this script measures the same interpolation error as before the
# builder migration. It is deliberately NOT `MLFMA.FARFIELDQUADSTRATEGYRECOM` (5), which is the
# recommendation for a whole map rather than the default this call used to get.
radiationbuilder = MLFMA.BEASTFieldCollectionBuilder(
    MLFMA.RadiationSide(), MLFMA.BEASTFarFieldAssemblerBuilder(), BEAST.SingleNumQStrat(7)
)

# calculate far fields
farfieldcollection = MLFMA.buildfieldcollection(radiationbuilder, operator, sampling, X)
# calculate far fields sampled at same points as interpolation for reference
extendedfarfielcollection = MLFMA.buildfieldcollection(
    radiationbuilder, operator, samplinginterpolation, X
)

# Set the values to be interpolated. The first argument fixes only the ELEMENT TYPE -- the far-field
# constructors take γ and use `typeof(γ)`, discarding the value -- so taking it from the operator
# keeps the scalar type tied to the operator rather than to a literal written here.
farfield = MLFMA.farfieldtype(operator)(BEAST.gamma(operator), sampling)
LinearAlgebra.mul!(farfield, farfieldcollection, ones(numfunctions(X)))

farfieldextended = MLFMA.farfieldtype(operator)(
    BEAST.gamma(operator), samplinginterpolation
)
LinearAlgebra.mul!(farfieldextended, extendedfarfielcollection, ones(numfunctions(X)))

# define interpolator with right stencil size
interpolator = MLFMA.SphericalLagrangeInterpolator{stencilsize}(ComplexF64, Float64)
# interpolator = MLFMA.LocalθGlobalφInterpolator{stencilsize}(ComplexF64, Float64)

# use interpolator to interpolate far field
interpolant = interpolator(farfield, sampling, samplinginterpolation)

interpolatedpoints, _ = MLFMA.sphericalsamplepoints(samplinginterpolation)

# Which field component to plot: 1..fielddim(farfieldtype(operator)). Two for a tangential spherical
# far field (Fθ, Fφ), three for a spherical vector one, six for a paired one, one for a scalar.
i = 1
relativeerror = norm.(farfieldextended[i] - interpolant[i]) ./ norm.(farfieldextended[i])

# Printed BEFORE plotting, so the numbers still arrive in a session with no display -- `display`
# raises rather than degrading when it cannot reach a browser or Electron, and it used to take the
# summary down with it after all the assembly work was already done.
println(
    "Component $i of $(nameof(MLFMA.farfieldtype(operator))), " *
    "$(Nθ)x$(Nφ) -> $(Nθinterpolation)x$(Nφinterpolation), stencil $stencilsize",
)
println(
    "Maximum error: $(maximum(relativeerror)) at $(interpolatedpoints[argmax(relativeerror)])",
)

p = PlotlyJS.plot(
    MLFMA.sphericaltrace(
        relativeerror,
        interpolatedpoints;
        mode="markers",
        marker=PlotlyJS.attr(;
            size=3, colorscale="YlOrRd", color=relativeerror[:], showscale=true
        ),
        hovertext=relativeerror[:],
    ),
)

PlotlyJS.display(p)
