module TestPerformance

using Test

# Each directory's runner is its own module, so the shared fixtures have to be included HERE rather than
# relied upon from a sibling directory. `Maps/` gets away without this only because its first file
# includes the support and the rest of the directory shares one module -- accidental coupling, not a
# pattern to copy.
include("../support/util_test.jl")

@testset verbose = true "Performance" begin
    include("allocations.jl")
end

end
