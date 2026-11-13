using MLFMA
using FastGaussQuadrature
using Test
using StaticArrays
using CompScienceMeshes
using BEAST
using LinearAlgebra
using Test

include("../support/util_test.jl")

@testset "Barycentric Lagrange Anterpolator" begin
    anterpolator = MLFMA.SphericalLagrangeAnterpolator{10}(ComplexF64, Float64)
    @test MLFMA.stencilsize(anterpolator) == 10

    Nθs = 5:8
    Nφs = 4:2:6

    stencilsizes = 3:5
    for Nθ in Nθs, Nφ in Nφs, stencilsize in stencilsizes, anterNθ in Nθs, anterNφ in Nφs
        sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)

        anterpolationsampling = MLFMA.SphericalSampling{Float64}(anterNθ, anterNφ)
        anterpolator = MLFMA.SphericalLagrangeAnterpolator{stencilsize}(ComplexF64, Float64)
        interpolator = MLFMA.SphericalLagrangeInterpolator{stencilsize}(ComplexF64, Float64)

        antermatrix = MLFMA._getanterpolationmatrix(
            anterpolator, anterpolationsampling, sampling, ComplexF64
        )
        intermatrix = MLFMA._getinterpolationmatrix(
            interpolator, sampling, anterpolationsampling, ComplexF64
        )
        for (iter, sample) in enumerate(anterpolationsampling)
            u = zeros(ComplexF64, size(anterpolationsampling))
            u[iter] = 1.0

            farfield = MLFMA.TangentialSphericalFarField(SVector(u, 2 * u))
            anterpolatedfarfield = MLFMA.TangentialSphericalFarField(
                SVector(
                    zeros(ComplexF64, size(sampling)), zeros(ComplexF64, size(sampling))
                ),
            )
            anterpolator(anterpolatedfarfield, farfield, anterpolationsampling, sampling)
            @test anterpolatedfarfield[1][:] == antermatrix[:, iter]
            @test anterpolatedfarfield[2][:] == 2 * antermatrix[:, iter]
        end

        for i in 1:length(sampling)
            u = zeros(ComplexF64, size(sampling))
            u[i] = im * 1.0 + 1.0

            farfield = MLFMA.TangentialSphericalFarField(SVector(u, 2 * u))
            interpolatedfarfield = MLFMA.TangentialSphericalFarField(
                SVector(
                    zeros(ComplexF64, size(anterpolationsampling)),
                    zeros(ComplexF64, size(anterpolationsampling)),
                ),
            )

            interpolator(interpolatedfarfield, farfield, sampling, anterpolationsampling)
            @test interpolatedfarfield[1][:] ≈ intermatrix * u[:]
        end

        @test antermatrix ≈ transpose(intermatrix)
    end
end
