# Interpolation error of MLFMA's angular interpolators, against exact angular functions.
#
# WHAT MAKES THIS DIFFERENT FROM `plot_interpolationerror.jl`, which sits next to it: that script
# assembles a real far field with BEAST on a mesh and compares interpolation against a finer
# assembly. This one assembles nothing. The reference is a closed-form spherical harmonic evaluated
# independently on both grids, so the number it reports is interpolation error and nothing else -- no
# quadrature error, no mesh, no operator, no tree. Use that one to ask "how does my problem behave",
# and this one to ask "how good is the interpolator".
#
# RUN IT WITH THE TEST ENVIRONMENT: `julia --project=test examples/plot_interpolation_diagnostics.jl`,
# or activate `test` first. `PlotlyJS` is a WEAK dependency of MLFMA -- it backs the `MLFMAPlotlyJS`
# extension -- so it is not installed alongside the package, and any environment that does have it
# works. The diagnostics themselves need no plotting package; only this script does.

using PlotlyJS
using MLFMA

const ID = MLFMA.InterpolationDiagnostics

# THE GRIDS. Built from a degree via `MLFMA.NθNφ` rather than by hand, so `supporteddegree` of the
# source is exactly SOURCEDEGREE and ρ reads directly as "fraction of the source grid's band".
# Interpolation goes coarse -> fine, which is the aggregation direction in a real MLFMA.
const SOURCEDEGREE = 13
const source = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(SOURCEDEGREE)...)
const target = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(2 * SOURCEDEGREE)...)

STENCILSIZE = 20
interpolator = MLFMA.SphericalLagrangeInterpolator{STENCILSIZE}(ComplexF64, Float64)
# The other production interpolator. Swap it in to compare -- same cases, same grids, so the numbers
# are directly comparable.
# interpolator = MLFMA.LocalθGlobalφInterpolator{STENCILSIZE}(ComplexF64, Float64)

println("source supports degree ", ID.supporteddegree(source))
println("target supports degree ", ID.supporteddegree(target))

# --- 1. A fixed degree, three orders -------------------------------------------------------------
#
# Same bandwidth, different `m`: this asks whether the interpolator cares about the DIRECTION the
# function oscillates in, which a θ/φ-separable scheme generally does.
fixeddegree = ID.interpolationerror(
    interpolator,
    source,
    target,
    [
        ID.HarmonicCase(; l=10, m=0),
        ID.HarmonicCase(; l=10, m=5),
        ID.HarmonicCase(; l=10, m=10),
        ID.RandomBandlimitedCase(; L=10, seed=0x1234),
    ],
)

println()
print(ID.summary(fixeddegree))

# --- 2. A degree sweep across and past the supported band ----------------------------------------
#
# Deliberately runs past ρ = 1. Past the source grid's supported degree the grid does not carry the
# case at all, so the error saturating near 1 there is the expected result and not a defect -- it is
# the reference point that makes the in-band numbers meaningful.
sweep = ID.interpolationerror(
    interpolator,
    source,
    target,
    vcat(
        ID.harmoniccases(2:2:(2 * SOURCEDEGREE)),
        [ID.RandomBandlimitedCase(; L=L, seed=0x1234) for L in 2:2:(2 * SOURCEDEGREE)],
    ),
)

println()
println("Worst case by rel_l2: ", ID.worstcase(sweep).case)
println("Worst case by rel_linf: ", ID.worstcase(sweep; by=:rel_linf).case)

# The table-friendly form. `rows` holds scalars only, so this is also what you would hand to a
# DataFrame outside MLFMA -- which is not a dependency of this package.
println()
println("rho      rel_l2       kind")
for row in ID.rows(sweep)
    println(
        rpad(round(row.rho; sigdigits=3), 9),
        rpad(round(row.rel_l2; sigdigits=3), 13),
        row.casekind,
    )
end

# --- 3. Plots ------------------------------------------------------------------------------------
#
# `ploterror` and `plotcurves` come from the `MLFMAPlotlyJS` extension, which `using PlotlyJS` above
# activated. Printed output comes first so a session without a display still gets the numbers.
worstindex = findfirst(==(ID.worstcase(sweep)), ID.results(sweep))

errorplot = ID.ploterror(sweep, worstindex, target)
PlotlyJS.display(errorplot)

curveplot = ID.plotcurves(sweep; x=:rho, y=:rel_l2, group=:casekind)
PlotlyJS.display(curveplot)

linfplot = ID.plotcurves(sweep; x=:rho, y=:rel_linf, group=:casekind)
PlotlyJS.display(linfplot)
