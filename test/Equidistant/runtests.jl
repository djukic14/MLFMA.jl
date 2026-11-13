@testset verbose = true "Equidistant" begin
    include("sampling.jl")
    include("samplinginterface.jl")
    include("scratchcompatibility.jl")
    include("quadrature.jl")
    include("backends.jl")
    include("anterpolation.jl")
    include("transposed.jl")
    include("fftscratch.jl")
    include("tangential.jl")
    include("interpolation.jl")
    include("translation.jl")
    include("twolevel.jl")
    # Last: the only testset here that builds a real tree and a real map, so it is also the slowest.
    include("pipeline.jl")
    include("harmonicstorage.jl")
end
