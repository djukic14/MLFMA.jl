module EquidistantScratchCompatibilityTests

using Test
using MLFMA
include("../support/util_test.jl")
using MLFMA.Equidistant:
    EquidistantSampling,
    EquidistantInterpolator,
    LagrangeBackend,
    LagrangeScratch,
    SPHERICAL,
    ANTISPHERICAL,
    resample,
    resampleadjoint,
    resample!,
    resampleadjoint!,
    _backendscratch

const T = ComplexF64
const CHILD = EquidistantSampling(6, 12)
const PARENT = EquidistantSampling(10, 20)

samples(s) = reshape(T[cis(0.7k) * (1 + 0.3k) for k in 1:prod(size(s))], size(s))

function checkmatching(backend)
    scratch = LagrangeScratch(CHILD, PARENT, backend, T)
    interim = (lagrangescratch=scratch,)
    @test _backendscratch(interim, backend, CHILD, PARENT) === scratch
    input, fineinput = samples(CHILD), samples(PARENT)
    up, down = similar(input, size(PARENT)), similar(input)
    interpolator = EquidistantInterpolator{T}(backend)
    anterpolator = MLFMA.anterpolator(interpolator; nchunks=1)
    field = MLFMA.ScalarFarField(T, CHILD, Val(:zero))
    finefield = MLFMA.ScalarFarField(T, PARENT, Val(:zero))
    fieldup = MLFMA.ScalarFarField(T, PARENT, Val(:zero))
    fielddown = MLFMA.ScalarFarField(T, CHILD, Val(:zero))
    field[1] .= input
    finefield[1] .= fineinput
    for parity in (SPHERICAL, ANTISPHERICAL)
        @test resample!(up, input, CHILD, PARENT, parity, backend, scratch) === up
        @test up == resample(input, CHILD, PARENT, parity, backend)
        @test resampleadjoint!(down, fineinput, CHILD, PARENT, parity, backend, scratch) ===
            down
        @test down ≈ resampleadjoint(fineinput, CHILD, PARENT, parity, backend)
        resample!(up, input, CHILD, PARENT, parity, backend, scratch)
        resampleadjoint!(down, fineinput, CHILD, PARENT, parity, backend, scratch)
        @test (@allocated resample!(up, input, CHILD, PARENT, parity, backend, scratch)) ==
            0
        @test (@allocated resampleadjoint!(
            down, fineinput, CHILD, PARENT, parity, backend, scratch
        )) == 0
        # Coverage changes allocation of the standalone immutable lookup result.
        # Measure the complete calls that consume it, including both parity conventions.
        trait = if parity == SPHERICAL
            MLFMA.VirtualExtension()
        else
            MLFMA.VirtualExtensionFlipθSign()
        end
        @test warmedallocations(
            interpolator, fieldup, field, CHILD, PARENT; interim, interpolationtrait=trait
        ) == 0
        @test fieldup[1] == up
        @test warmedallocations(
            anterpolator,
            fielddown,
            finefield,
            PARENT,
            CHILD;
            interim,
            interpolationtrait=trait,
        ) == 0
        @test fielddown[1] ≈ down
    end
    return nothing
end

function checkfallback(backend, scratch)
    interim = (lagrangescratch=scratch,)
    @test _backendscratch(interim, backend, CHILD, PARENT) === nothing
    interpolator = EquidistantInterpolator{T}(backend)
    anterpolator = MLFMA.anterpolator(interpolator; nchunks=1)
    input = MLFMA.ScalarFarField(T, CHILD, Val(:zero))
    fineinput = MLFMA.ScalarFarField(T, PARENT, Val(:zero))
    input[1] .= samples(CHILD)
    fineinput[1] .= samples(PARENT)
    up = MLFMA.ScalarFarField(T, PARENT, Val(:zero))
    upreference = MLFMA.ScalarFarField(T, PARENT, Val(:zero))
    down = MLFMA.ScalarFarField(T, CHILD, Val(:zero))
    downreference = MLFMA.ScalarFarField(T, CHILD, Val(:zero))
    interpolator(upreference, input, CHILD, PARENT)
    interpolator(up, input, CHILD, PARENT; interim=interim)
    @test up[1] == upreference[1]
    anterpolator(downreference, fineinput, PARENT, CHILD)
    anterpolator(down, fineinput, PARENT, CHILD; interim=interim)
    @test down[1] == downreference[1]
    return nothing
end

function checkrejected(backend, scratch, exception)
    input, fineinput = samples(CHILD), samples(PARENT)
    up, down = fill(T(19), size(PARENT)), fill(T(19), size(CHILD))
    fill!(scratch.stage, T(23))
    for parity in (SPHERICAL, ANTISPHERICAL)
        @test_throws exception resample!(up, input, CHILD, PARENT, parity, backend, scratch)
        @test_throws exception resampleadjoint!(
            down, fineinput, CHILD, PARENT, parity, backend, scratch
        )
        @test all(==(T(19)), up)
        @test all(==(T(19)), down)
        @test all(==(T(23)), scratch.stage)
    end
    checkfallback(backend, scratch)
    return nothing
end

@testset "Lagrange scratch compatibility" begin
    @testset "matching isotropic and anisotropic orders" begin
        for backend in (
            LagrangeBackend(2),
            LagrangeBackend(4),
            LagrangeBackend(2, 4),
            LagrangeBackend(4, 2),
        )
            checkmatching(backend)
        end
    end

    @testset "wrong orders" begin
        # Original reproducer: the smaller backend used only part of the cached order-4 weights.
        checkrejected(
            LagrangeBackend(2),
            LagrangeScratch(CHILD, PARENT, LagrangeBackend(4), T),
            ArgumentError,
        )
        # Check each axis independently, including a backend larger than the cached tables.
        for (cached, requested) in (
            (LagrangeBackend(2, 4), LagrangeBackend(4, 4)),
            (LagrangeBackend(4, 2), LagrangeBackend(4, 4)),
            (LagrangeBackend(4, 4), LagrangeBackend(2, 4)),
            (LagrangeBackend(4, 4), LagrangeBackend(4, 2)),
            (LagrangeBackend(2, 4), LagrangeBackend(4, 2)),
        )
            checkrejected(
                requested, LagrangeScratch(CHILD, PARENT, cached, T), ArgumentError
            )
        end
    end

    @testset "wrong grids" begin
        backend = LagrangeBackend(2, 4)
        # Changing only child phi or parent theta leaves the staging shape unchanged.
        for (child, parent) in (
            (EquidistantSampling(7, 12), PARENT),
            (EquidistantSampling(6, 14), PARENT),
            (CHILD, EquidistantSampling(11, 20)),
            (CHILD, EquidistantSampling(10, 22)),
            (PARENT, CHILD),
        )
            checkrejected(
                backend, LagrangeScratch(child, parent, backend, T), DimensionMismatch
            )
        end
    end

    @testset "direct input and output shapes" begin
        backend = LagrangeBackend(2, 4)
        scratch = LagrangeScratch(CHILD, PARENT, backend, T)
        input, fineinput = samples(CHILD), samples(PARENT)
        up, down = similar(fineinput), similar(input)
        for parity in (SPHERICAL, ANTISPHERICAL)
            @test_throws DimensionMismatch resample!(
                up, input[1:(end - 1), :], CHILD, PARENT, parity, backend, scratch
            )
            @test_throws DimensionMismatch resample!(
                up[1:(end - 1), :], input, CHILD, PARENT, parity, backend, scratch
            )
            @test_throws DimensionMismatch resampleadjoint!(
                down, fineinput[1:(end - 1), :], CHILD, PARENT, parity, backend, scratch
            )
            @test_throws DimensionMismatch resampleadjoint!(
                down[1:(end - 1), :], fineinput, CHILD, PARENT, parity, backend, scratch
            )
        end
    end
end

end # module
