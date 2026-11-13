# Its own `module`, following `TestLevelTransfers`: `calibration.jl` defines fixture helpers with
# generic names (`circlefixture`, `groupmomenterror`) and nothing here should be able to satisfy a
# reference in a sibling runner by accident.
module TestCircularHarmonics

using Test

@testset verbose = true "CircularHarmonics" begin
    # The transform pair, the layout and the truncation rule, which everything else is expressed in.
    include("transform.jl")
    # Group storage and aggregation, after the transform they are expressed in.
    include("storage.jl")
    # Last, and the only file here that builds a tree and runs a BEAST assembly: the empirical
    # claims -- what real 2-D patterns compress to and where the speed crossover sits.
    include("collection.jl")
    include("calibration.jl")
end

end
