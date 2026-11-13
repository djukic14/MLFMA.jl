using PlotlyJS
using MLFMA
using Test

@testset "Visualization" begin
    Nθ, Nφ = 10, 10
    sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
    interpolatedpoints, _ = MLFMA.sphericalsamplepoints(sampling)

    fvals = zeros(length(interpolatedpoints))

    @test_nowarn MLFMA.sphericaltrace(fvals, interpolatedpoints)
end
