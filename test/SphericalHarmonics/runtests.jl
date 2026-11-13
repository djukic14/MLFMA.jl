@testset verbose = true "SphericalHarmonics" begin
    include("ylm.jl")
    # The transform pair, after the harmonics it tabulates.
    include("transform.jl")
    # Group storage and aggregation, after the transform they are expressed in.
    include("storage.jl")
    # The frame projection and the antipodal mirror, which the Galerkin receive side rests on.
    include("frame.jl")
    # The tangential VECTOR basis. Last, since its mirror test is asserted against the same
    # `mirrorindices`/`mirrorrule` machinery `frame.jl` covers for the scalar case.
    include("vectorharmonics.jl")
end
