module TestVisualization

using Test

@testset verbose = true "Visualization" begin
    include("visualization.jl")
end

end
