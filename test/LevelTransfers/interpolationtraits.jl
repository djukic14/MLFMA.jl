using Test
using MLFMA
using StaticArrays

@testset "Interpolationtraits" begin
    TSampling = Float64
    TInterpolation = ComplexF64
    sampling = MLFMA.SphericalSampling{TSampling}(4, 8)
    intersampling = MLFMA.SphericalSampling{TSampling}(5, 10)

    stencilsize = 5
    for interpolator in
        [MLFMA.SphericalLagrangeInterpolator, MLFMA.LocalθGlobalφInterpolator]
        virtualinterpolator = interpolator{stencilsize}(
            TInterpolation,
            sampling,
            intersampling;
            interpolationtrait=MLFMA.VirtualExtension(),
        )
        virtualanterpolator = MLFMA.anterpolator(virtualinterpolator)
        @test MLFMA.interpolationtrait(virtualinterpolator) == MLFMA.VirtualExtension()
        @test MLFMA.interpolationtrait(virtualanterpolator) == MLFMA.VirtualExtension()
        @test first(values(virtualinterpolator.interpolationinfodict)) isa SVector{1}
        @test first(values(virtualanterpolator.anterpolationinfodict)) isa SVector{1}

        virtualflipinterpolator = interpolator{stencilsize}(
            TInterpolation,
            sampling,
            intersampling;
            interpolationtrait=MLFMA.VirtualExtensionFlipθSign(),
        )
        virtualflipanterpolator = MLFMA.anterpolator(virtualflipinterpolator)
        @test MLFMA.interpolationtrait(virtualflipinterpolator) ==
            MLFMA.VirtualExtensionFlipθSign()
        @test MLFMA.interpolationtrait(virtualflipanterpolator) ==
            MLFMA.VirtualExtensionFlipθSign()
        @test first(values(virtualflipinterpolator.interpolationinfodict)) isa SVector{1}
        @test first(values(virtualflipanterpolator.anterpolationinfodict)) isa SVector{1}

        sphericalθφinterpolator = interpolator{stencilsize}(
            TInterpolation,
            sampling,
            intersampling;
            interpolationtrait=MLFMA.SphericalθφInterpolation(),
        )
        sphericalθφanterpolator = MLFMA.anterpolator(sphericalθφinterpolator)
        @test MLFMA.interpolationtrait(sphericalθφinterpolator) ==
            MLFMA.SphericalθφInterpolation()
        @test MLFMA.interpolationtrait(sphericalθφanterpolator) ==
            MLFMA.SphericalθφInterpolation()
        @test first(values(sphericalθφinterpolator.interpolationinfodict)) isa SVector{2}
        @test first(values(sphericalθφanterpolator.anterpolationinfodict)) isa SVector{2}

        sphericalinterpolator = interpolator{stencilsize}(
            TInterpolation,
            sampling,
            intersampling;
            interpolationtrait=MLFMA.SphericalInterpolation(),
        )
        sphericalanterpolator = MLFMA.anterpolator(sphericalinterpolator)
        @test MLFMA.interpolationtrait(sphericalinterpolator) ==
            MLFMA.SphericalInterpolation()
        @test MLFMA.interpolationtrait(sphericalanterpolator) ==
            MLFMA.SphericalInterpolation()
        @test first(values(sphericalinterpolator.interpolationinfodict)) isa SVector{3}
        @test first(values(sphericalanterpolator.anterpolationinfodict)) isa SVector{3}

        doublesphericalinterpolator = interpolator{stencilsize}(
            TInterpolation,
            sampling,
            intersampling;
            interpolationtrait=MLFMA.DoubleSphericalInterpolation(),
        )
        doublesphericalanterpolator = MLFMA.anterpolator(doublesphericalinterpolator)
        @test MLFMA.interpolationtrait(doublesphericalinterpolator) ==
            MLFMA.DoubleSphericalInterpolation()
        @test MLFMA.interpolationtrait(doublesphericalanterpolator) ==
            MLFMA.DoubleSphericalInterpolation()
        @test first(values(doublesphericalinterpolator.interpolationinfodict)) isa
            SVector{6}
        @test first(values(doublesphericalanterpolator.anterpolationinfodict)) isa
            SVector{6}

        defaultinterpolator = interpolator{stencilsize}(
            TInterpolation,
            sampling,
            intersampling;
            interpolationtrait=MLFMA.DefaultInterpolation(),
        )
        defaultanterpolator = MLFMA.anterpolator(defaultinterpolator)
        @test MLFMA.interpolationtrait(defaultinterpolator) == MLFMA.DefaultInterpolation()
        @test MLFMA.interpolationtrait(defaultanterpolator) == MLFMA.DefaultInterpolation()
        @test first(values(defaultinterpolator.interpolationinfodict)) isa NamedTuple
        @test first(values(defaultanterpolator.anterpolationinfodict)) isa NamedTuple

        γ = one(TInterpolation)
        farfield = MLFMA.ScalarFarField(γ, sampling)
        for i in eachindex(farfield)
            for j in eachindex(farfield[i])
                if TInterpolation <: Complex
                    farfield[i][j] = i * j + im * i
                else
                    farfield[i][j] = i * j
                end
            end
        end

        ffvirtual = virtualinterpolator(farfield, sampling, intersampling)
        for interpolator in
            [sphericalinterpolator, doublesphericalinterpolator, defaultinterpolator]
            ffequivalent = interpolator(farfield, sampling, intersampling)
            for i in eachindex(ffequivalent)
                @test ffequivalent[i] ≈ ffvirtual[i]
            end
        end

        ffvirtualflip = virtualflipinterpolator(
            farfield,
            sampling,
            intersampling;
            interpolationtrait=MLFMA.VirtualExtensionFlipθSign(),
        )
        for interpolator in [
            sphericalθφinterpolator,
            sphericalinterpolator,
            doublesphericalinterpolator,
            defaultinterpolator,
        ]
            ffequivalent = interpolator(
                farfield,
                sampling,
                intersampling;
                interpolationtrait=MLFMA.VirtualExtensionFlipθSign(),
            )
            for i in eachindex(ffequivalent)
                @test ffequivalent[i] ≈ ffvirtualflip[i]
            end
        end

        farfield = MLFMA.TangentialSphericalFarField(γ, sampling)
        for i in eachindex(farfield)
            for j in eachindex(farfield[i])
                if TInterpolation <: Complex
                    farfield[i][j] = i * j + im * i
                else
                    farfield[i][j] = i * j
                end
            end
        end

        ffsphericalθφ = sphericalθφinterpolator(
            farfield,
            sampling,
            intersampling;
            interpolationtrait=MLFMA.SphericalθφInterpolation(),
        )
        for interpolator in [
            virtualflipinterpolator,
            sphericalinterpolator,
            doublesphericalinterpolator,
            defaultinterpolator,
        ]
            ffequivalent = interpolator(
                farfield,
                sampling,
                intersampling;
                interpolationtrait=MLFMA.SphericalθφInterpolation(),
            )
            for i in eachindex(ffequivalent)
                @test ffequivalent[i] ≈ ffsphericalθφ[i]
            end
        end

        farfield = MLFMA.SphericalVectorFarField(γ, sampling)
        for i in eachindex(farfield)
            for j in eachindex(farfield[i])
                if TInterpolation <: Complex
                    farfield[i][j] = i * j + im * i
                else
                    farfield[i][j] = i * j
                end
            end
        end

        ffspherical = sphericalinterpolator(
            farfield,
            sampling,
            intersampling;
            interpolationtrait=MLFMA.SphericalInterpolation(),
        )

        for interpolator in [doublesphericalinterpolator, defaultinterpolator]
            ffequivalent = interpolator(
                farfield,
                sampling,
                intersampling;
                interpolationtrait=MLFMA.SphericalInterpolation(),
            )
            for i in eachindex(ffequivalent)
                @test ffequivalent[i] ≈ ffspherical[i]
            end
        end

        farfield = MLFMA.PairedSphericalVectorFarField(γ, sampling)
        for i in eachindex(farfield)
            for j in eachindex(farfield[i])
                if TInterpolation <: Complex
                    farfield[i][j] = i * j + im * i
                else
                    farfield[i][j] = i * j
                end
            end
        end

        ffdoublespherical = doublesphericalinterpolator(
            farfield,
            sampling,
            intersampling;
            interpolationtrait=MLFMA.DoubleSphericalInterpolation(),
        )

        for interpolator in [sphericalinterpolator, defaultinterpolator]
            ffequivalent = interpolator(
                farfield,
                sampling,
                intersampling;
                interpolationtrait=MLFMA.DoubleSphericalInterpolation(),
            )
            for i in eachindex(ffequivalent)
                @test ffequivalent[i] ≈ ffdoublespherical[i]
            end
        end

        interpolatedfarfield = MLFMA.ScalarFarField(γ, intersampling)
        for i in eachindex(interpolatedfarfield)
            for j in eachindex(interpolatedfarfield[i])
                if TInterpolation <: Complex
                    interpolatedfarfield[i][j] = i * j + im * i
                else
                    interpolatedfarfield[i][j] = i * j
                end
            end
        end

        ffvirtual = virtualanterpolator(
            interpolatedfarfield,
            intersampling,
            sampling;
            interpolationtrait=MLFMA.VirtualExtension(),
        )
        for anterpolator in
            [sphericalanterpolator, doublesphericalanterpolator, defaultanterpolator]
            ffequivalent = anterpolator(
                interpolatedfarfield,
                intersampling,
                sampling;
                interpolationtrait=MLFMA.VirtualExtension(),
            )
            for i in eachindex(ffequivalent)
                @test ffequivalent[i] ≈ ffvirtual[i]
            end
        end

        ffvirtualflip = virtualflipanterpolator(
            interpolatedfarfield,
            intersampling,
            sampling;
            interpolationtrait=MLFMA.VirtualExtensionFlipθSign(),
        )

        for anterpolator in
            [sphericalanterpolator, doublesphericalanterpolator, defaultanterpolator]
            ffequivalent = anterpolator(
                interpolatedfarfield,
                intersampling,
                sampling;
                interpolationtrait=MLFMA.VirtualExtensionFlipθSign(),
            )
            for i in eachindex(ffequivalent)
                @test ffequivalent[i] ≈ ffvirtualflip[i]
            end
        end

        interpolatedfarfield = MLFMA.TangentialSphericalFarField(γ, intersampling)
        for i in eachindex(interpolatedfarfield)
            for j in eachindex(interpolatedfarfield[i])
                if TInterpolation <: Complex
                    interpolatedfarfield[i][j] = i * j + im * i
                else
                    interpolatedfarfield[i][j] = i * j
                end
            end
        end

        ffsphericalθφ = sphericalθφanterpolator(
            interpolatedfarfield,
            intersampling,
            sampling;
            interpolationtrait=MLFMA.SphericalθφInterpolation(),
        )

        for anterpolator in
            [virtualflipanterpolator, sphericalanterpolator, doublesphericalanterpolator]
            ffequivalent = anterpolator(
                interpolatedfarfield,
                intersampling,
                sampling;
                interpolationtrait=MLFMA.SphericalθφInterpolation(),
            )
            for i in eachindex(ffequivalent)
                @test ffequivalent[i] ≈ ffsphericalθφ[i]
            end
        end

        interpolatedfarfield = MLFMA.SphericalVectorFarField(γ, intersampling)
        for i in eachindex(interpolatedfarfield)
            for j in eachindex(interpolatedfarfield[i])
                if TInterpolation <: Complex
                    interpolatedfarfield[i][j] = i * j + im * i
                else
                    interpolatedfarfield[i][j] = i * j
                end
            end
        end

        ffspherical = sphericalanterpolator(
            interpolatedfarfield,
            intersampling,
            sampling;
            interpolationtrait=MLFMA.SphericalInterpolation(),
        )

        for anterpolator in [doublesphericalanterpolator, defaultanterpolator]
            ffequivalent = anterpolator(
                interpolatedfarfield,
                intersampling,
                sampling;
                interpolationtrait=MLFMA.SphericalInterpolation(),
            )
            for i in eachindex(ffequivalent)
                @test ffequivalent[i] ≈ ffspherical[i]
            end
        end

        interpolatedfarfield = MLFMA.PairedSphericalVectorFarField(γ, intersampling)
        for i in eachindex(interpolatedfarfield)
            for j in eachindex(interpolatedfarfield[i])
                if TInterpolation <: Complex
                    interpolatedfarfield[i][j] = i * j + im * i
                else
                    interpolatedfarfield[i][j] = i * j
                end
            end
        end

        ffdoublespherical = doublesphericalanterpolator(
            interpolatedfarfield,
            intersampling,
            sampling;
            interpolationtrait=MLFMA.DoubleSphericalInterpolation(),
        )

        for anterpolator in [sphericalanterpolator, defaultanterpolator]
            ffequivalent = anterpolator(
                interpolatedfarfield,
                intersampling,
                sampling;
                interpolationtrait=MLFMA.DoubleSphericalInterpolation(),
            )
            for i in eachindex(ffequivalent)
                @test ffequivalent[i] ≈ ffdoublespherical[i]
            end
        end
    end
end
