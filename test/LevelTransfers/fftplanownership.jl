using MLFMA
using Test

@testset "FFT planning is independent of chunk ownership" begin
    child = MLFMA.SphericalSampling{Float64}(5, 8)
    parent = MLFMA.SphericalSampling{Float64}(9, 16)
    builder = MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=3)
    interpolator = MLFMA.buildinterpolator(builder, [child], [parent])
    forward = MLFMA._getinterpolationmatrix(interpolator, child, parent, ComplexF64)

    @testset "chunk compatibility does not change the operator" begin
        # Zero also verifies that construction never indexes a chunk-sized planning array.
        for nchunks in (0, 1, 7, 128)
            reverse = MLFMA.buildreverseoperators(builder, interpolator; nchunks=nchunks)
            @test length(reverse.anterpolator.anterpolationinfodict) == 1
            @test MLFMA._getanterpolationmatrix(
                reverse.anterpolator, parent, child, ComplexF64
            ) ≈ adjoint(forward)
            @test MLFMA._getanterpolationmatrix(
                reverse.transposedinterpolator, parent, child, ComplexF64
            ) ≈ transpose(forward)
        end
    end

    @testset "warmed construction allocations do not scale with chunks" begin
        allocationchild = MLFMA.SphericalSampling{Float64}(12, 24)
        allocationparent = MLFMA.SphericalSampling{Float64}(24, 48)
        allocationinterpolator = MLFMA.buildinterpolator(
            builder, [allocationchild], [allocationparent]
        )
        construct(nchunks) =
            MLFMA.buildreverseoperators(builder, allocationinterpolator; nchunks=nchunks)
        for nchunks in (1, 128), _ in 1:3
            construct(nchunks)
        end
        # Interleave measurements and take minima to exclude compilation/one-off planner work.
        small = Int[]
        large = Int[]
        for _ in 1:5
            push!(small, @allocated construct(1))
            push!(large, @allocated construct(128))
        end
        # The old extra matrices cost over 1 MiB at these shapes. Allow modest planner noise,
        # but substantially less than even one of the two old chunk-sized collections.
        tolerance = max(16_384, minimum(small) ÷ 10)
        @test abs(minimum(large) - minimum(small)) <= tolerance
    end

    @testset "shared plans work with independently scheduled scratch" begin
        anterpolator = MLFMA.buildanterpolator(builder, interpolator; nchunks=1)
        source = MLFMA.ScalarFarField(ComplexF64, parent, Val(:zero))
        for i in eachindex(source[1])
            source[1][i] = complex(sin(i), cos(2i))
        end
        original = copy(source[1])
        expected = adjoint(forward) * vec(original)
        @test maximum(abs, expected) > 1e-6

        # Model workspace chunk assignment without depending on traversal fixtures: all chunks
        # share the cached operator, but own their mutable interim arrays.
        for nchunks in (1, 4)
            scratches = [
                MLFMA.interimscratch(anterpolator, child, parent) for _ in 1:nchunks
            ]
            for i in eachindex(scratches), j in 1:(i - 1)
                for field in keys(scratches[i])
                    @test getproperty(scratches[i], field) !==
                        getproperty(scratches[j], field)
                end
            end
            output = MLFMA.ScalarFarField(ComplexF64, child, Val(:zero))
            for chunk in reverse(eachindex(scratches))
                scratch = scratches[chunk]
                for buffer in values(scratch)
                    fill!(buffer, ComplexF64(NaN))
                end
                # The matrix probe uses the tangential field's first-component continuation.
                anterpolator(
                    output,
                    source,
                    parent,
                    child;
                    interim=scratch,
                    interpolationtrait=MLFMA.VirtualExtensionFlipθSign(),
                )
                @test vec(output[1]) ≈ expected
                @test source[1] ≈ original
            end
        end
    end
end
