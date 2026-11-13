module TestSampling

using Test

@testset verbose = true "Sampling" begin
    include("sampling.jl")
    include("abstraction.jl")
    include("circularsampling.jl")
    include("symmetryaction.jl")
end

end
