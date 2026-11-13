using MLFMA
using BEAST, CompScienceMeshes
using Test
using SphericalScattering
using StaticArrays
using LinearAlgebra
using H2Factory

include("../support/util_test.jl")

function reversedsphericalunitvector(θ::T, φ::T) where {T}
    θ_minus = T(π) - θ
    φ_minus = T(0) <= φ < T(pi) ? φ + T(pi) : φ - T(pi)

    return [θ_minus, φ_minus]
end

function reversedsphericalunitvector(k̂::SVector{2,T}) where {T}
    return reversedsphericalunitvector(k̂[1], k̂[2])
end
function reversedsphericalunitvector(k̂::SVector{3,T}) where {T}
    return reversedsphericalunitvector(k̂[2], k̂[3])
end

@testset "Integrating on unit sphere" begin
    Nθ = 40
    Nφ = 82

    sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
    k̂, weights = MLFMA.sphericalsamplepoints(sampling)

    f1(k) = 1
    result1 = sum(f1.(k̂) .* weights)
    @test result1 ≈ 4π

    # f(θ,φ) = θ
    f2(k) = k[2]
    result2 = sum(f2.(k̂) .* weights)
    @test result2 ≈ 2 * π^2

    # f(θ,φ) = θ*(sin(θ) + cos(φ))
    f3(k) = k[2] * (sin(k[3]) + cos(k[3]))
    result3 = sum(f3.(k̂) .* weights)
    @test result3 ≈ 0.0 atol = eps(Float64) * 10

    # f(θ,φ) = θ^2
    f4(k) = k[2]^2
    result4 = sum(f4.(k̂) .* weights)
    @test result4 ≈ 2 * pi * (pi^2 - 4)
end

@testset "cartesian and spherical sampling points" begin
    Ns = 3:10
    for Nθ in Ns, Nφ in Ns
        isodd(Nφ) && continue
        sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
        k, weights = MLFMA.sphericalsamplepoints(sampling)
        k_cart, weights_cart = MLFMA.cartesiansamplepoints(sampling)

        # Spherical Scattering uses different convention for φ
        ksphericalcomparison = SphericalScattering.cart2sph.(k_cart)
        for i in eachindex(ksphericalcomparison)
            if ksphericalcomparison[i][3] < 0
                @test ksphericalcomparison[i] ≈ k[i] - [0.0, 0.0, 2 * pi]
            else
                @test ksphericalcomparison[i] ≈ k[i]
            end
        end
        @test all(weights .≈ weights_cart)
    end
end

@testset "Mirroring" begin
    Ns = 3:7
    fn = joinpath(pkgdir(MLFMA), "test", "assets", "simple_rectangle.msh")
    m = CompScienceMeshes.readmesh(fn)

    efie = MLFMA.efie(; wavenumber=1.0)
    doublelayer = Maxwell3D.doublelayer(; wavenumber=1.0)

    operators = [efie, doublelayer]
    negativeoperators = [
        MLFMA.efie(; wavenumber=-1.0), Maxwell3D.doublelayer(; wavenumber=-1.0)
    ]

    for X in [raviartthomas(m), buffachristiansen(m)]
        for (idoperator, operator) in enumerate(operators)
            negativeoperator = negativeoperators[idoperator]
            for Nθ in Ns, Nφ in Ns
                isodd(Nφ) && continue

                sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)

                k, weights = MLFMA.sphericalsamplepoints(sampling)
                k_cart, _ = MLFMA.cartesiansamplepoints(sampling)

                mirroredkdifferentconvention = SphericalScattering.cart2sph.(-1.0 * k_cart)

                mirroredk = SVector{3,Float64}[]

                for i in eachindex(mirroredkdifferentconvention)
                    if mirroredkdifferentconvention[i][3] < 0
                        φcomponent =
                            (mirroredkdifferentconvention[i][3] + 2 * pi) % (2 * pi)
                        push!(
                            mirroredk,
                            SVector(
                                mirroredkdifferentconvention[i][1],
                                mirroredkdifferentconvention[i][2],
                                φcomponent,
                            ),
                        )

                    else
                        push!(mirroredk, mirroredkdifferentconvention[i])
                    end
                end

                mirrorindex = MLFMA.mirrorindices(sampling)

                @test all(mirroredk[:] .≈ k[mirrorindex][:])
                @test all(mirroredk[mirrorindex][:] .≈ k[:])

                ff = MLFMA.TangentialSphericalFarField(1.0, sampling)

                RadiationFieldCollection = MLFMA.buildfieldcollection(
                    radiationbuilder(), operator, sampling, X
                )
                negativeRadiationFieldCollection = MLFMA.buildfieldcollection(
                    radiationbuilder(), negativeoperator, sampling, X
                )

                for i in 1:numfunctions(X)
                    u = zeros(numfunctions(X))
                    u[i] = 1.0

                    ff = MLFMA.TangentialSphericalFarField(im * 1.0, sampling)
                    mirroredff = MLFMA.TangentialSphericalFarField(im * 1.0, sampling)

                    LinearAlgebra.mul!(ff, RadiationFieldCollection, u)
                    LinearAlgebra.mul!(mirroredff, negativeRadiationFieldCollection, u)

                    @test all(
                        isapprox(
                            MLFMA.Fθ(MLFMA.mirror(ff, sampling)),
                            MLFMA.Fθ(mirroredff);
                            atol=1e-14,
                        ),
                    )
                    @test all(
                        isapprox(
                            MLFMA.Fφ(MLFMA.mirror(ff, sampling)),
                            MLFMA.Fφ(mirroredff);
                            atol=1e-14,
                        ),
                    )
                end
            end
        end
    end
end

@testset "Reversing spherical unit vector" begin
    θs = [0, 0.5 * pi, pi]
    φs = [0, 0.5 * pi, pi, 1.5 * pi, 2 * pi]

    results = [
        [1.0 * pi, 1.0 * pi],
        [1.0 * pi, 3 / 2 * pi],
        [1.0 * pi, 0.0],
        [1.0 * pi, pi / 2],
        [1.0 * pi, 1.0 * pi],
        [pi / 2, 1.0 * pi],
        [pi / 2, 3 / 2 * pi],
        [pi / 2, 0.0],
        [pi / 2, pi / 2],
        [pi / 2, 1.0 * pi],
        [0.0, 1.0 * pi],
        [0.0, 3 / 2 * pi],
        [0.0, 0.0],
        [0.0, pi / 2],
        [0.0, 1.0 * pi],
    ]

    i = 1
    for θ in θs
        for φ in φs
            @test all(reversedsphericalunitvector(SVector(1.0, θ, φ)) .== results[i])
            @test all(reversedsphericalunitvector(SVector(θ, φ)) .== results[i])
            @test all(reversedsphericalunitvector(θ, φ) .== results[i])
            i += 1
        end
    end
end

@testset "Complex conjugation" begin
    Nθ = 3
    Nφ = 4

    Fθ = ones(Float64, Nφ, Nθ) * (3.0 + 2.0im)
    Fφ = zeros(Float64, Nφ, Nθ) * (2.0 + 1.0im)

    ff = MLFMA.TangentialSphericalFarField(SVector(Fθ, Fφ))

    @test MLFMA.Nθ(ff) == Nθ
    @test MLFMA.Nφ(ff) == Nφ
    @test MLFMA.Fθ(ff) == Fθ
    @test MLFMA.Fφ(ff) == Fφ

    ff2 = conj(ff)

    @test MLFMA.Nθ(ff2) == Nθ
    @test MLFMA.Nφ(ff2) == Nφ
    @test MLFMA.Fθ(ff2) == conj.(Fθ)
    @test MLFMA.Fφ(ff2) == conj.(Fφ)
end
