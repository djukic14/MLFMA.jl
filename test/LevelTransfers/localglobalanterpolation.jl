
using MLFMA
using FastGaussQuadrature
using LinearAlgebra
using Test

@testset "Local Global Anterpolation" begin
    Nθs = 3:7
    Nφs = 4:2:8

    stencilsizes = 3:5
    for Nθ in Nθs, Nφ in Nφs, stencilsize in stencilsizes, anterNθ in Nθs, anterNφ in Nφs
        anterNφ < Nφ && continue

        sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
        anterpolationsampling = MLFMA.SphericalSampling{Float64}(anterNθ, anterNφ)

        interpolator = MLFMA.LocalθGlobalφInterpolator{stencilsize}(ComplexF64, Float64)
        anterpolator = MLFMA.LocalθGlobalφAnterpolator{stencilsize}(ComplexF64, Float64)

        intermatrix = zeros(ComplexF64, length(anterpolationsampling), length(sampling))
        for (iter, sample) in enumerate(sampling)
            ff = MLFMA.TangentialSphericalFarField(1.0 + 0.0 * im, sampling, Val(:zero))
            ff[1][iter] = 1.0
            ff[2][iter] = 1.0
            ffinterpolated = interpolator(ff, sampling, anterpolationsampling)
            intermatrix[:, iter] = ffinterpolated[2][:]
        end

        antermatrix = zeros(ComplexF64, length(sampling), length(anterpolationsampling))
        for (iter, sample) in enumerate(anterpolationsampling)
            ff = MLFMA.TangentialSphericalFarField(
                1.0 * im, anterpolationsampling, Val(:zero)
            )
            ff[1][iter] = 1.0
            ff[2][iter] = 1.0
            ffanterpolated = anterpolator(ff, anterpolationsampling, sampling)
            antermatrix[:, iter] = ffanterpolated[2][:]
            @assert ffanterpolated[2] ≈ ffanterpolated[1]
        end

        @test antermatrix' ≈ intermatrix
    end
end
