using MLFMA
using Test
using FastGaussQuadrature
using StaticArrays

include("../support/util_test.jl")

@testset "Finding angles" begin
    Nθs = 4:5
    Nφs = 4:2:6

    testingθs = range(0, pi; length=100)
    testingφs = range(0, 2pi; length=100)

    for stencilsize in 3:6
        for Nθ in Nθs, Nφ in Nφs
            sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
            θs, φs = MLFMA.θsφs(sampling)

            sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)

            for θ in testingθs
                for φ in testingφs
                    θindex, φindex = MLFMA.findθφ(sampling, θ, φ, stencilsize)

                    φindex < Nφ && φindex > 1 && @test φs[φindex] <= φ <= φs[φindex + 1]
                    φindex == Nφ && @test φs[φindex] <= φ
                    φindex == 1 && @test φ <= φs[φindex + 1]

                    if isodd(stencilsize)
                        θindex < Nθ &&
                            θindex > 1 &&
                            @test abs(θs[θindex] - θ) <= abs(θs[θindex + 1] - θ)

                        θindex == Nθ && @test abs(θs[θindex] - θ) <= abs(θs[θindex - 1] - θ)

                        θindex == 1 && @test abs(θs[θindex] - θ) <= abs(θs[θindex + 1] - θ)
                    else
                        θindex < Nθ - 1 &&
                            θindex > 1 &&
                            @test θs[θindex] <= θ <= θs[θindex + 1]

                        θindex == Nθ && @test θs[θindex] <= θ

                        θindex == 1 && @test θ <= θs[θindex + 1]
                    end
                end
            end
        end
    end
end

@testset "Global index" begin
    Nθs = 4:7
    Nφs = 4:2:10

    for Nθ in Nθs, Nφ in Nφs
        sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)

        θs, φs = MLFMA.θsφs(sampling)

        k, _ = MLFMA.sphericalsamplepoints(sampling)

        for i in eachindex(θs)
            for j in eachindex(φs)
                index = MLFMA.globalindex(i, j, sampling)
                @test k[index][2] == θs[i]
                @test k[index][3] == φs[j]
            end
        end

        for i in eachindex(φs)
            for j in eachindex(θs)
                index = MLFMA.globalindex(j, i, sampling)
                @test k[index][2] == θs[j]
                @test k[index][3] == φs[i]
            end
        end
    end
end

@testset "Basic Sampling" begin
    Nθs = 3:7
    Nφs = 2:2:8
    for Nθ in Nθs, Nφ in Nφs
        sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)

        @test MLFMA.Nθ(sampling) == Nθ
        @test MLFMA.Nφ(sampling) == Nφ

        @test MLFMA.θs(sampling) == (FastGaussQuadrature.gausslegendre(Nθ)[1] .+ 1) * pi / 2
        @test MLFMA.φs(sampling) == range(0.0, 2 * pi; length=(Nφ + 1))[1:(end - 1)]

        f1(θ, φ) = θ
        f2(θ, φ) = φ

        fs = [f1, f2]

        testθs = (FastGaussQuadrature.gausslegendre(Nθ)[1] .+ 1) * pi / 2
        testφs = range(0.0, 2 * pi; length=(Nφ + 1))[1:(end - 1)]

        for i in eachindex(testθs), j in eachindex(testφs)
            @test sampling[i, j] == (testθs[i], testφs[j])
        end

        testθs = [
            reverse(-(FastGaussQuadrature.gausslegendre(Nθ)[1] .+ 1) * pi / 2)
            (FastGaussQuadrature.gausslegendre(Nθ)[1] .+ 1) * pi / 2
            (FastGaussQuadrature.gausslegendre(Nθ)[1] .+ 1) * pi / 2 .+ pi
        ]

        testφs = [
            Array(range(0.0, 2 * pi; length=(Nφ + 1)) .- 2 * pi)[1:(end - 1)]
            Array(range(0.0, 2 * pi; length=(Nφ + 1))[1:(end - 1)])
            Array(range(0.0, 2 * pi; length=(Nφ + 1))[1:(end - 1)] .+ 2 * pi)
        ]

        ii = 1
        jj = 1
        for i in (-Nθ + 1):(2 * Nθ)
            jj = 1
            for j in (-Nφ + 1):(2 * Nφ)
                @test all(sampling[i, j] .≈ (testθs[ii], testφs[jj]))
                @test MLFMA.θ(sampling, i) ≈ testθs[ii]
                @test MLFMA.φ(sampling, j) ≈ testφs[jj]

                jj += 1
            end
            ii += 1
        end

        k̂, _ = MLFMA.sphericalsamplepoints(sampling)
        for (iter, samplingpoint) in enumerate(sampling)
            @test samplingpoint[1] == k̂[iter][2]
            @test samplingpoint[2] == k̂[iter][3]
        end

        @test length(sampling) == Nθ * Nφ
    end
end
