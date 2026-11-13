module TestLevelTransfers

using Test

@testset verbose = true "LevelTransfers" begin
    include("interpolationtraits.jl")
    include("interpolation.jl")
    include("interpolationdiagnostics.jl")
    include("vectorharmonics.jl")
    include("tangentialdiagnostics.jl")
    include("farfieldinterpolation.jl")
    include("interpolatemultiple.jl")
    include("anterpolation.jl")
    include("sphericalanterpolation.jl")
    include("localglobalanterpolation.jl")
    include("circularinterpolation.jl")
    include("anterpolationcontract.jl")
    include("interpolationbuilders.jl")
    include("fftplanownership.jl")
    include("phaseshift.jl")
    include("phaseconvention.jl")
    include("transfer.jl")
    include("collections.jl")
    include("protocol.jl")
end

end
