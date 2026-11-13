module AllocationMeasurementTest
using Test
include("../support/util_test.jl")

increment!(counter; step=1) = (counter[] += step; nothing)
retainallocation!(sink, n) = (sink[]=zeros(n); nothing)

@testset "Allocation measurement" begin
    counter = Ref(0)
    @test warmedallocations(increment!, counter) == 0
    @test counter[] == 3
    @test warmedallocations(increment!, counter; step=2) == 0
    @test counter[] == 9

    # A real allocation must remain visible after warming and specialization.
    sink = Ref(Float64[])
    @test warmedallocations(retainallocation!, sink, 1024) >= 1024 * sizeof(Float64)
    @test length(sink[]) == 1024
end
end
