module MLFMAPlotlyJS
using PlotlyJS
using MLFMA

function MLFMA.sphericaltrace(
    fval::AbstractArray{T}, points::AbstractArray{T1}; kwargs...
) where {T,T1}
    x = []
    y = []
    z = []

    for point in points
        push!(x, point[1] * sin(point[2]) * cos(point[3]))
        push!(y, point[1] * sin(point[2]) * sin(point[3]))
        push!(z, point[1] * cos(point[2]))
    end

    return PlotlyJS.scatter3d(; x=x, y=y, z=z, kwargs...)
end

# Keep PlotlyJS optional so interpolation diagnostics work in headless runs and tests.
# These methods become available when the extension loads.
const ID = MLFMA.InterpolationDiagnostics

function MLFMA.InterpolationDiagnostics.ploterror(
    study::ID.InterpolationErrorStudy,
    i::Integer,
    sampling::MLFMA.SphericalSampling;
    kind=:relative,
    kwargs...,
)
    points, _ = MLFMA.sphericalsamplepoints(sampling)
    values = vec(ID.pointwiseerror(study, i; kind=kind))

    length(values) == length(points) || throw(
        DimensionMismatch(
            "ploterror: result $i has $(length(values)) error values but the sampling has " *
            "$(length(points)) directions. The sampling passed in is not the TARGET sampling " *
            "this study interpolated onto.",
        ),
    )

    return PlotlyJS.plot(
        MLFMA.sphericaltrace(
            values,
            points;
            mode="markers",
            marker=PlotlyJS.attr(;
                size=3, colorscale="YlOrRd", color=values, showscale=true
            ),
            hovertext=string.(values),
            kwargs...,
        ),
        PlotlyJS.Layout(; title="$(kind) interpolation error, case $i"),
    )
end

function MLFMA.InterpolationDiagnostics.plotcurves(
    study::ID.InterpolationErrorStudy; x=:rho, y=:rel_l2, group=:casekind
)
    table = ID.rows(study)
    isempty(table) && throw(ArgumentError("plotcurves: the study has no results"))

    available = join(keys(first(table)), ", ")
    for field in (x, y, group)
        field in keys(first(table)) || throw(
            ArgumentError("plotcurves: :$field is not a row field. Available: $available"),
        )
    end

    # Group and sort the NamedTuple rows without a DataFrames dependency.
    traces = PlotlyJS.AbstractTrace[]
    for label in unique(row[group] for row in table)
        rowsingroup = filter(row -> row[group] == label, table)
        order = sortperm([Float64(row[x]) for row in rowsingroup])
        push!(
            traces,
            PlotlyJS.scatter(;
                x=[Float64(rowsingroup[i][x]) for i in order],
                y=[Float64(rowsingroup[i][y]) for i in order],
                mode="lines+markers",
                name=string(label),
            ),
        )
    end

    return PlotlyJS.plot(
        traces,
        PlotlyJS.Layout(;
            title="$y versus $x",
            xaxis=PlotlyJS.attr(; title=string(x)),
            # Log scale because interpolation error spans orders of magnitude across a degree sweep;
            # on a linear axis everything inside the band collapses onto zero.
            yaxis=PlotlyJS.attr(; title=string(y), type="log"),
        ),
    )
end

end
