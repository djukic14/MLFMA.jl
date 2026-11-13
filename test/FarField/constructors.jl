@testset "FarField constructor policy" begin
    sampling = MLFMA.SphericalSampling{Float64}(4, 6)

    farfieldtypes = (
        MLFMA.ScalarFarField,
        MLFMA.TangentialSphericalFarField,
        MLFMA.SphericalVectorFarField,
        MLFMA.CartesianVectorFarField,
        MLFMA.PairedSphericalVectorFarField,
    )

    @testset "default is undef, same generic method for every T (incl. BigFloat)" begin
        # `which` resolving to the identical Method object for every T -- rather than a
        # BigFloat-specific override -- is what "same policy" means here; it does not depend on
        # what bytes an undef allocation happens to contain.
        for FarFieldType in farfieldtypes
            m_f64 = which(FarFieldType, Tuple{Float64,typeof(sampling)})
            m_f32 = which(FarFieldType, Tuple{Float32,typeof(sampling)})
            m_c64 = which(FarFieldType, Tuple{ComplexF64,typeof(sampling)})
            m_big = which(FarFieldType, Tuple{BigFloat,typeof(sampling)})
            m_cbig = which(FarFieldType, Tuple{Complex{BigFloat},typeof(sampling)})

            @test m_f64 == m_f32 == m_c64 == m_big == m_cbig
        end
    end

    @testset "Val(:zero) zero-fills, for every representation and BigFloat too" begin
        for T in (Float64, Float32, ComplexF64, BigFloat, Complex{BigFloat}),
            FarFieldType in farfieldtypes

            ff = FarFieldType(T(1.0), sampling, Val(:zero))
            @test length(eachindex(ff)) == MLFMA.fielddim(FarFieldType)
            for i in eachindex(ff)
                @test all(iszero, ff[i])
            end
        end
    end

    @testset "removed initializer forms throw MethodError" begin
        @test_throws MethodError MLFMA.TangentialSphericalFarField(1.0, sampling, :zero)
        @test_throws MethodError MLFMA.TangentialSphericalFarField(1.0, sampling, undef)
        @test_throws MethodError MLFMA.TangentialSphericalFarField(
            1.0, sampling, Val(:undef)
        )
        @test_throws MethodError MLFMA.TangentialSphericalFarField(
            BigFloat(1.0), sampling, :zero
        )
    end

    @testset "explicit element type form, from a sampling or a size" begin
        for FarFieldType in farfieldtypes
            fromsampling = FarFieldType(Float32, sampling)
            @test eltype(fromsampling[1]) == Float32
            @test length(eachindex(fromsampling)) == MLFMA.fielddim(FarFieldType)

            fromsize = FarFieldType(ComplexF64, size(sampling))
            @test eltype(fromsize[1]) == ComplexF64
            @test size(fromsize[1]) == size(sampling)
            @test typeof(fromsize) <: FarFieldType
        end
    end

    # The component count is derived from the type by every constructor, so no call site has to
    # state it -- and the two three-component representations stay distinguishable regardless.
    @testset "constructed type carries the representation, not just the count" begin
        spherical = MLFMA.SphericalVectorFarField(ComplexF64, sampling)
        cartesian = MLFMA.CartesianVectorFarField(ComplexF64, sampling)

        @test length(eachindex(spherical)) == length(eachindex(cartesian)) == 3
        @test typeof(spherical) !== typeof(cartesian)
        @test !(typeof(spherical) <: MLFMA.CartesianVectorFarField)
        @test !(typeof(cartesian) <: MLFMA.SphericalVectorFarField)
    end
end
