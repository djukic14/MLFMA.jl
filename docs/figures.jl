using MLFMA, PlotlyJS

include("translation-accuracy.jl")

function writeifchanged(path, data)
    isfile(path) && read(path) == data && return nothing
    return write(path, data)
end

# Generate only small kernel examples. No meshes or matrices are assembled here.
let output = joinpath(@__DIR__, "src", "assets", "generated")
    mkpath(output)
    writeifchanged(joinpath(output, "plotly.min.js"), read(PlotlyJS._js_path))
    phi = collect(range(0.0, 2π; length=241))
    V = MLFMA.Sampling.SVector
    directions = [V(cos(p), sin(p), 0.0) for p in phi]
    translations = Dict{String,Any}()
    for r in (0.5, 1.0, 2.0), L in (4, 8, 12)
        values = zeros(ComplexF64, length(phi))
        MLFMA.translationoperator!(values, 2π * im, directions, V(r, 0.0, 0.0), L)
        # Construct the figure through PlotlyJS; browser controls reuse these traces.
        figure = Plot(scatter(; x=phi, y=abs.(values)))
        @assert length(figure.data) == 1
        translations["$(r == 1 ? "1" : r == 2 ? "2" : "0.5")/$L"] = (;
            magnitude=figure.data[1][:y], phase=angle.(values)
        )
    end
    io = IOBuffer()
    print(io, "window.MLFMAFigures = ")
    PlotlyJS.JSON.print(io, (; phi, translations))
    println(io, ";")
    # Avoid triggering another LiveServer rebuild when the data has not changed.
    writeifchanged(joinpath(output, "figures.js"), take!(io))
    ensuretranslationaccuracy(output)
end
