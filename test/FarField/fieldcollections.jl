using MLFMA
using BEAST, CompScienceMeshes
using Test
using SphericalScattering
using StaticArrays
using LinearAlgebra
using H2Factory

@testset "NθNφ" begin
    L = 1:10
    result = [
        (4, 8)
        (6, 12)
        (8, 16)
        (10, 20)
        (12, 24)
        (14, 28)
        (16, 32)
        (18, 36)
        (20, 40)
        (22, 44)
    ]

    @test all(MLFMA.NθNφ.(L) .== result)
end

@testset "Manually created RadiationFieldCollection" begin
    Fθ = [1.0 4.0; 2.0 3.0; 3.0 2.0; 4.0 1.0]
    Fφ = [1.0 2.0; 1.0 2.0; 1.0 2.0; 1.0 2.0]

    sampling = MLFMA.SphericalSampling{Float64}(2, 2)

    RadiationFieldCollection = MLFMA.TangentialSphericalRadiationFieldCollection(
        SVector(Fθ, Fφ)
    )

    @test MLFMA.Fθ(RadiationFieldCollection) == Fθ
    @test MLFMA.Fφ(RadiationFieldCollection) == Fφ

    farfield = MLFMA.TangentialSphericalFarField(1.0, sampling)
    LinearAlgebra.mul!(farfield, RadiationFieldCollection, [2, 2])
    @test MLFMA.Fθ(farfield) == [10.0 10.0; 10.0 10.0]
    @test MLFMA.Fφ(farfield) == [6.0 6.0; 6.0 6.0]
    @test MLFMA.Nθ(farfield) == 2
    @test MLFMA.Nφ(farfield) == 2

    LinearAlgebra.mul!(farfield, RadiationFieldCollection, [-1.0, 1.0])
    @test MLFMA.Fθ(farfield) == [3.0 -1.0; 1.0 -3.0]
    @test MLFMA.Fφ(farfield) == [1.0 1.0; 1.0 1.0]
    @test MLFMA.Nθ(farfield) == 2
    @test MLFMA.Nφ(farfield) == 2

    farfield2 = MLFMA.TangentialSphericalFarField(
        SVector([1.0, 1.0, 1.0, 1.0], [-1.0, -2.0, -3.0, -4.0])
    )

    farfield3 = H2Factory.subset(RadiationFieldCollection, [1])
    farfield4 = H2Factory.subset(RadiationFieldCollection, [2])
end

@testset "Test removing radial component" begin
    k = [SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0), SVector(7.0, 8.0, 9.0)]

    θ, φ = MLFMA._removeradialcomponent(k)

    @test all(θ .== [2.0, 5.0, 8.0])
    @test all(φ .== [3.0, 6.0, 9.0])
end

@testset "Testing mirroring with simple rectangle" begin
    Ns = 3:6
    fns = [
        joinpath(pkgdir(MLFMA), "test", "assets", "simple_rectangle.msh"),
        joinpath(pkgdir(MLFMA), "test", "assets", "sphere.msh"),
    ]

    efie = MLFMA.efie(; wavenumber=1.0)
    doublelayer = Maxwell3D.doublelayer(; wavenumber=1.0)
    operators = [efie, doublelayer]
    negativeoperators = [
        MLFMA.efie(; wavenumber=-1.0), Maxwell3D.doublelayer(; wavenumber=-1.0)
    ]
    for (ioperator, operator) in enumerate(operators)
        negativeoperator = negativeoperators[ioperator]
        for fn in fns
            m = CompScienceMeshes.readmesh(fn)
            for X in [raviartthomas(m), buffachristiansen(m)]
                for Nθ in Ns, Nφ in Ns
                    isodd(Nφ) && continue

                    sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)

                    RadiationFieldCollection = MLFMA.buildfieldcollection(
                        radiationbuilder(), operator, sampling, X
                    )
                    negativeRadiationFieldCollection = MLFMA.buildfieldcollection(
                        radiationbuilder(), negativeoperator, sampling, X
                    )

                    u = zeros(numfunctions(X))
                    u[2] = 1.0
                    # farfield = RadiationFieldCollection * u
                    farfield = MLFMA.TangentialSphericalFarField(im * 1.0, sampling)
                    LinearAlgebra.mul!(farfield, RadiationFieldCollection, u)
                    farfieldmirrored = MLFMA.mirror(farfield, sampling)

                    u1 = zeros(ComplexF64, numfunctions(X))
                    u2 = zeros(ComplexF64, numfunctions(X))

                    LinearAlgebra.mul!(u1, RadiationFieldCollection, farfieldmirrored)
                    LinearAlgebra.mul!(u2, negativeRadiationFieldCollection, farfield)

                    @test all(u1 .≈ u2)

                    u1 =
                        transpose(MLFMA.Fθ(RadiationFieldCollection)) *
                        MLFMA.Fθ(farfield)[:] +
                        transpose(MLFMA.Fφ(RadiationFieldCollection)) *
                        MLFMA.Fφ(farfield)[:]

                    u2 = LinearAlgebra.mul!(u2, RadiationFieldCollection, farfield)
                    @test all(u1 .≈ u2)
                end
            end
        end
    end
end

@testset "Compare to BEAST.jl" begin
    κ = 2 * pi / 2
    # fn = joinpath(pkgdir(MLFMA), "test", "assets", "sphere.msh")
    fn = joinpath(pkgdir(MLFMA), "test", "assets", "simple_rectangle.msh")
    m = CompScienceMeshes.readmesh(fn)
    for X in [raviartthomas(m), buffachristiansen(m)]
        Nθ = 5
        Nφ = 4
        sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
        efie = MLFMA.efie(; wavenumber=κ)
        doublelayer = Maxwell3D.doublelayer(; wavenumber=κ)
        rotateddoublayer = BEAST.n × Maxwell3D.doublelayer(; wavenumber=κ)
        for operator in [efie, doublelayer, rotateddoublayer]
            RadiationFieldCollection = MLFMA.buildfieldcollection(
                radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(1)),
                operator,
                sampling,
                X,
            )

            sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
            farfieldsamplepoints, _ = MLFMA.cartesiansamplepoints(sampling)

            for i in 1:numfunctions(X)
                u = zeros(numfunctions(X))
                u[i] = 1.0
                farfieldscartesian = BEAST.potential(
                    BEAST.MWFarField3D(; wavenumber=κ),
                    farfieldsamplepoints,
                    u,
                    X;
                    quadstrat=BEAST.SingleNumQStrat(1),
                )

                Fθ = Vector{ComplexF64}(undef, length(farfieldsamplepoints))
                Fφ = Vector{ComplexF64}(undef, length(farfieldsamplepoints))

                farfieldspherical = SphericalScattering.convertCartesian2Spherical.(
                    farfieldscartesian, SphericalScattering.cart2sph.(farfieldsamplepoints)
                )
                farfieldθφ = MLFMA._removeradialcomponent(farfieldspherical)

                Fθ = farfieldθφ[1]
                Fφ = farfieldθφ[2]

                @test Fθ[:] ≈ MLFMA.Fθ(RadiationFieldCollection)[:, i]
                @test Fφ[:] ≈ MLFMA.Fφ(RadiationFieldCollection)[:, i]
            end

            u = ones(numfunctions(X))
            farfieldscartesian = BEAST.potential(
                BEAST.MWFarField3D(; wavenumber=κ),
                farfieldsamplepoints,
                u,
                X;
                quadstrat=BEAST.SingleNumQStrat(1),
            )

            Fθ = Vector{ComplexF64}(undef, length(farfieldsamplepoints))
            Fφ = Vector{ComplexF64}(undef, length(farfieldsamplepoints))

            farfieldspherical = SphericalScattering.convertCartesian2Spherical.(
                farfieldscartesian, SphericalScattering.cart2sph.(farfieldsamplepoints)
            )
            farfieldθφ = MLFMA._removeradialcomponent(farfieldspherical)

            Fθ = farfieldθφ[1]
            Fφ = farfieldθφ[2]

            @test Fθ[:] ≈ sum(MLFMA.Fθ(RadiationFieldCollection); dims=2)
            @test Fφ[:] ≈ sum(MLFMA.Fφ(RadiationFieldCollection); dims=2)
        end
    end
end

@testset "Mirrored RadiationFieldCollection EFIE" begin
    Ns = 3:6
    fns = [joinpath(pkgdir(MLFMA), "test", "assets", "sphere.msh")]
    κ = 1.0
    efie = MLFMA.efie(; wavenumber=κ)
    negativeefie = MLFMA.efie(; wavenumber=(-κ))
    for fn in fns
        m = CompScienceMeshes.readmesh(fn)
        for X in [raviartthomas(m), buffachristiansen(m)]
            for Nθ in Ns, Nφ in Ns
                isodd(Nφ) && continue

                sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)

                farfieldcollection = MLFMA.buildfieldcollection(
                    radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(1)),
                    efie,
                    sampling,
                    X,
                )
                mirroredfarfieldcollection = MLFMA.LazyReceiveFieldCollection(
                    efie, farfieldcollection, MLFMA.mirrorindices(sampling)
                )
                # The leaf scratch a workspace supplies: `LazyReceiveFieldCollection` implements only
                # `H2Factory.applyleafmul!`, so every read below goes through that seam.
                leafscratch = MLFMA.farfieldtype(efie)(ComplexF64, sampling, Val(:zero))

                negativefarfieldcollection = MLFMA.buildfieldcollection(
                    radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(1)),
                    negativeefie,
                    sampling,
                    X,
                )
                receivefarfieldcollection = MLFMA.buildfieldcollection(
                    receivebuilder(; quadstrategy=BEAST.SingleNumQStrat(1)),
                    efie,
                    sampling,
                    X,
                )

                @test MLFMA.Fθ(receivefarfieldcollection) ≈
                    (MLFMA.Fθ(negativefarfieldcollection))

                @test MLFMA.Fφ(receivefarfieldcollection) ≈
                    (MLFMA.Fφ(negativefarfieldcollection))

                k, _ = MLFMA.cartesiansamplepoints(sampling)

                for i in eachindex(k)
                    for j in eachindex(k)
                        farfield = MLFMA.TangentialSphericalFarField(
                            im * κ, sampling, Val(:zero)
                        )
                        farfield[1][i] = 1.0
                        farfield[2][j] = 1.0

                        result = zeros(ComplexF64, numfunctions(X))
                        resultfromirrored = zeros(ComplexF64, numfunctions(X))
                        result2 = zeros(ComplexF64, numfunctions(X))

                        LinearAlgebra.mul!(result, negativefarfieldcollection, farfield)
                        H2Factory.applyleafmul!(
                            leafscratch,
                            resultfromirrored,
                            mirroredfarfieldcollection,
                            farfield,
                            1.0,
                            0.0,
                        )
                        LinearAlgebra.mul!(result2, receivefarfieldcollection, farfield)

                        @test result ≈ resultfromirrored
                        @test result ≈ result2
                    end
                end
            end
        end
    end
end

@testset "Mirrored RadiationFieldCollection MFIE" begin
    Ns = 3:6
    fn = joinpath(pkgdir(MLFMA), "test", "assets", "sphere.msh")
    κ = 1.0
    doublelayer = Maxwell3D.doublelayer(; wavenumber=κ)
    negativedoublelayer = Maxwell3D.doublelayer(; wavenumber=(-κ))
    m = CompScienceMeshes.readmesh(fn)
    for X in [raviartthomas(m), buffachristiansen(m)]
        for Nθ in Ns, Nφ in Ns
            isodd(Nφ) && continue

            sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)

            farfieldcollection = MLFMA.buildfieldcollection(
                radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(1)),
                doublelayer,
                sampling,
                X,
            )
            mirroredfarfieldcollection = MLFMA.LazyReceiveFieldCollection(
                doublelayer, farfieldcollection, MLFMA.mirrorindices(sampling)
            )
            leafscratch = MLFMA.farfieldtype(doublelayer)(ComplexF64, sampling, Val(:zero))

            negativefarfieldcollection = MLFMA.buildfieldcollection(
                radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(1)),
                negativedoublelayer,
                sampling,
                X,
            )

            negativefarfieldcollection = MLFMA.TangentialSphericalRadiationFieldCollection(
                SVector(
                    -MLFMA.Fφ(negativefarfieldcollection) .* im * κ,
                    MLFMA.Fθ(negativefarfieldcollection) .* im * κ,
                ),
            )

            receivefarfieldcollection = MLFMA.buildfieldcollection(
                receivebuilder(; quadstrategy=BEAST.SingleNumQStrat(1)),
                doublelayer,
                sampling,
                X,
            )

            @test MLFMA.Fθ(receivefarfieldcollection) ≈
                (MLFMA.Fθ(negativefarfieldcollection))
            @test MLFMA.Fφ(receivefarfieldcollection) ≈
                (MLFMA.Fφ(negativefarfieldcollection))

            k, _ = MLFMA.cartesiansamplepoints(sampling)

            for i in eachindex(k)
                for j in eachindex(k)
                    farfield = MLFMA.TangentialSphericalFarField(
                        im * κ, sampling, Val(:zero)
                    )
                    farfield[1][i] = 1.0
                    farfield[2][j] = 1.0

                    result = zeros(ComplexF64, numfunctions(X))
                    resultfromirrored = zeros(ComplexF64, numfunctions(X))
                    result2 = zeros(ComplexF64, numfunctions(X))

                    LinearAlgebra.mul!(result, negativefarfieldcollection, farfield)
                    H2Factory.applyleafmul!(
                        leafscratch,
                        resultfromirrored,
                        mirroredfarfieldcollection,
                        farfield,
                        true,
                        false,
                    )
                    LinearAlgebra.mul!(result2, receivefarfieldcollection, farfield)

                    @test result ≈ resultfromirrored
                    @test result ≈ result2
                end
            end
        end
    end
end
