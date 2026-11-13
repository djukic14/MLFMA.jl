module TestPackage

using Test

# Package hygiene, not physics: what the public API is, that every file still parses, and which
# H2Factory seams MLFMA extends. These hold the shape of the package steady while the layer modules
# move code around underneath them.
@testset verbose = true "Package" begin
    include("publicapi.jl")
    include("layers.jl")
    include("callsites.jl")
    include("selfreference.jl")
    include("parsecheck.jl")
    include("allocationmeasurement.jl")
    include("h2factoryinterface.jl")
    include("beastinterface.jl")
end

end
