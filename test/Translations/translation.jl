using MLFMA
using StaticArrays
using LinearAlgebra
using Test
using SphericalScattering
using SpecialFunctions
using LegendrePolynomials

include("../support/util_test.jl")
@testset "Translate dipole" begin
    function radiationfunction(κ, k, y, b, dipolediry)
        phasefactor = MLFMA.phasefactor(
            -im * κ, k, y - b, MLFMA.AbstractRadiationFieldCollection
        )

        dipolespherical = [dipolediry for k̂ in k]
        dipolespherical = MLFMA._removeradialcomponent(
            SphericalScattering.convertCartesian2Spherical.(
                dipolespherical, SphericalScattering.cart2sph.(k)
            ),
        )
        rθ = dipolespherical[1] .* phasefactor
        rφ = dipolespherical[2] .* phasefactor
        return rθ, rφ
    end

    function receivefunction(κ, k, x, a, dipoledirx)
        return radiationfunction(κ, k, a, x, dipoledirx)
    end

    function sphericaldot(fθ, fφ, gθ, gφ, k̂)
        result = zeros(ComplexF64, length(k̂))
        for i in eachindex(k̂)
            r, θ, φ = k̂[i]
            result[i] =
                (fθ[i] * cos(θ) * cos(φ) - fφ[i] * sin(φ)) *
                (gθ[i] * cos(θ) * cos(φ) - gφ[i] * sin(φ)) +
                (fθ[i] * cos(θ) * sin(φ) + fφ[i] * cos(φ)) *
                (gθ[i] * cos(θ) * sin(φ) + gφ[i] * cos(φ)) +
                fθ[i] * sin(θ) * gθ[i] * sin(θ)
        end
        return result
    end

    κ = 2 * π * 1e9 / 3e8

    a = SVector(0.0, 0.0, 0.0)
    x = [0.5; 0.0; 0.0]
    dipoledirx = [0.0; 1.0; 1.0] / sqrt(2)

    distance = 5
    b = SVector(0.0, distance, 0.0)
    y = [0.5; distance; -0.5]
    dipolediry = [1.0; 1.0; 1.0] / sqrt(3)

    result_analytical = MLFMA.dotu(_Greensdivdiv(κ, x, y, dipolediry), dipoledirx)

    d = x - a - y + b
    ϵ = 1e-10

    L = MLFMA.cutoffparameter(im * κ, ϵ, norm(d)) * 2

    Nθ, Nφ = MLFMA.NθNφ(L)
    sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
    k, weights = MLFMA.cartesiansamplepoints(sampling)
    mirrorindex = MLFMA.mirrorindices(sampling)

    translationfunction = Matrix{ComplexF64}(undef, size(sampling))
    MLFMA.translationoperator!(translationfunction, im * κ, k, a, b, L)
    receivef = receivefunction(κ, k, x, a, dipoledirx)
    radiationf = receivefunction(κ, k, y, b, dipolediry)
    radiationf = (radiationf[1][mirrorindex], -1.0 * radiationf[2][mirrorindex])

    result_numerical = sum(
        sphericaldot(receivef[1], receivef[2], radiationf[1], radiationf[2], k) .*
        (weights .* translationfunction)[:],
    )
    println(norm(result_analytical - result_numerical) / norm(result_analytical))
    @test result_analytical ≈ result_numerical rtol = 1e-10

    # @test MLFMA.cutoffparameter(translationfunction) * 2 == L
end

@testset "SphericalHankelFunction" begin
    h(l, x) =
        SpecialFunctions.sphericalbesselj(l, x) -
        im * SpecialFunctions.sphericalbessely(l, x)

    for order in 0:10
        for x in 0.1:0.1:(order + 1)
            @test h(order, x) ≈ MLFMA.sphericalhankelsecond(order, x)
        end
    end
end

@testset "Legendre Polynomials" begin
    ordermax = 1000
    for x in -1:0.1:1
        p = LegendrePolynomials.collectPl(x; lmax=ordermax)
        p2 = MLFMA.collectlegendre(ordermax, x)
        @test p.parent ≈ p2
    end
end
