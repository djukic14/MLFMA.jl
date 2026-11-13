module TestConstruction

using Test

@testset verbose = true "Construction" begin
    include("strategies.jl")
    include("petrovsampling.jl")
end

end
