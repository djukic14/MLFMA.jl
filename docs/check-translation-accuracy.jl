using MLFMA, Test
include("translation-accuracy.jl")

@testset "Translation-accuracy experiment" begin
    V = MLFMA.Sampling.SVector
    k = 2π
    for (dimension, family) in ((2, "circular"), (3, "spherical"), (3, "equidistant")),
        L in (4, 10),
        refinement in (1, 2, 4)

        kd, offset = 20.0, 0.4
        result = accuracysample(dimension, family, L, refinement, kd, offset)
        grid = accuracygrid(dimension, family, L, refinement)
        directions, weights = MLFMA.cartesiansamplepoints(grid)
        D = kd / k
        displacement = dimension == 3 ? V(0.0, 0.0, D) : V(D, 0.0)
        x = dimension == 3 ? V(0.0, 0.0, offset) : V(offset, 0.0)
        values = zeros(ComplexF64, size(directions))
        effective = MLFMA.translationoperator!(values, k * im, directions, displacement, L)
        phase = MLFMA.phasefactor(
            k * im, directions, x, MLFMA.AbstractRadiationFieldCollection
        )
        fullsum = sum(weights .* values .* phase)
        @test result.approximation ≈ fullsum rtol=2e-12
        @test result.effective == effective
        @test result.directions == length(directions)
    end

    @testset "Sampled integral and finite-series reference" begin
        for dimension in (2, 3), L in (4, 8, 12)
            kd, offset = 20.0, 0.4
            result = accuracysample(
                dimension, dimension == 2 ? "circular" : "spherical", L, 4, kd, offset
            )
            series = if dimension == 3
                accuracyreference3d(kd/k, offset, L)
            else
                accuracyreference2d(kd/k, offset, L)
            end
            @test result.error ≈ series[result.effective + 1] atol=2e-13 rtol=1e-10
        end
        reference = accuracyreference3d(20/k, 0.4, 17)
        refined = accuracyreference3d(20/k, 0.4, 17; bits=384)
        @test reference ≈ refined rtol=1e-13
        # With zero offset only the zeroth spherical Bessel term remains.
        @test maximum(accuracyreference3d(0.2, 0.0, 17)) < 1e-70
    end

    @testset "The 2-D guard is visible, not bypassed" begin
        guarded = accuracysample(2, "circular", 24, 1, 3.0, 0.4)
        refined = accuracysample(2, "circular", 24, 4, 3.0, 0.4)
        @test guarded.effective < 24
        @test guarded.effective == refined.effective
        @test refined.directions == 4 * guarded.directions
        @test accuracysample(3, "spherical", 24, 1, 3.0, 0.4).effective == 24
    end
    @test_throws ArgumentError accuracygrid(2, "spherical", 4, 1)
    @test_throws ArgumentError accuracygrid(3, "circular", 4, 1)
    @test_throws ArgumentError accuracygrid(3, "spherical", 0, 1)
    @test_throws ArgumentError accuracysample(3, "spherical", 4, 1, 1.0, 0.4)
end
