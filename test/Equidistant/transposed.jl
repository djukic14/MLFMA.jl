module EquidistantTransposedTests

using Test
using MLFMA
using LinearAlgebra
using Random
using MLFMA.Equidistant:
    EquidistantSampling,
    SPHERICAL,
    ANTISPHERICAL,
    Nθ,
    Nφ,
    resample,
    resampleadjoint,
    resamplematrix,
    anterpolate,
    anterpolatedense,
    innerproduct,
    sphericalget,
    sphericalscatter!,
    FFTBackend,
    LagrangeBackend

const COARSE = EquidistantSampling{Float64}(6, 12)
const FINE = EquidistantSampling{Float64}(12, 24)
const BACKENDS = (FFTBackend(), LagrangeBackend(4), LagrangeBackend(6))

@testset "resampleadjoint equals the dense adjoint" begin
    # The transposed pipeline is the whole justification for `transposedresample.jl`, and none of its
    # stage transpositions is self-evident by reading -- so each is pinned against the materialized
    # operator, which follows the definition literally.
    rng = MersenneTwister(3)
    for backend in BACKENDS, parity in (SPHERICAL, ANTISPHERICAL)
        Y = randn(rng, ComplexF64, Nφ(FINE), Nθ(FINE))
        R = resamplematrix(COARSE, FINE, parity, backend)
        want = reshape(R' * vec(Y), Nφ(COARSE), Nθ(COARSE))
        @test resampleadjoint(Y, COARSE, FINE, parity, backend) ≈ want rtol = 1e-12
    end
end

@testset "the fast anterpolate equals the dense one" begin
    rng = MersenneTwister(4)
    for backend in (FFTBackend(), LagrangeBackend(6)), parity in (SPHERICAL, ANTISPHERICAL)
        G = randn(rng, ComplexF64, Nφ(FINE), Nθ(FINE))
        @test anterpolate(G, FINE, COARSE, parity, backend) ≈
            anterpolatedense(G, FINE, COARSE, parity, backend) rtol = 1e-12
    end
end

@testset "the weighted adjoint identity survives the fast path" begin
    # Re-asserted here rather than assumed from anterpolation.jl: that testset now exercises the fast
    # kernel, so this states independently that the property is a property of the OPERATOR and not an
    # artefact of how it was built.
    rng = MersenneTwister(5)
    for backend in (FFTBackend(), LagrangeBackend(6)), parity in (SPHERICAL, ANTISPHERICAL)
        f = randn(rng, ComplexF64, Nφ(COARSE), Nθ(COARSE))
        g = randn(rng, ComplexF64, Nφ(FINE), Nθ(FINE))
        lhs = innerproduct(resample(f, COARSE, FINE, parity, backend), g, FINE)
        rhs = innerproduct(f, anterpolate(g, FINE, COARSE, parity, backend), COARSE)
        @test lhs ≈ rhs rtol = 1e-11
    end
end

@testset "sphericalscatter! is the exact adjoint of sphericalget" begin
    # ⟨sphericalget(F, i, j), v⟩ == ⟨F, sphericalscatter!(·, v, i, j)⟩ for every index, including
    # several periods outside the array -- the pole reflection is where an adjoint most easily drifts.
    rng = MersenneTwister(6)
    s = EquidistantSampling{Float64}(5, 10)
    for parity in (SPHERICAL, ANTISPHERICAL)
        F = randn(rng, ComplexF64, Nφ(s), Nθ(s))
        for i in (-4):(Nθ(s) + 4), j in 1:Nφ(s)
            v = randn(rng, ComplexF64)
            scattered = zeros(ComplexF64, Nφ(s), Nθ(s))
            sphericalscatter!(scattered, v, i, j, s, parity)
            @test sphericalget(F, i, j, s, parity) * v ≈ sum(F .* scattered)
        end
    end
end

@testset "the fast path is not quietly the dense path" begin
    # Non-vacuity of the speed claim: the dense operator costs one full resample per source sample, so
    # on a grid this size it is orders slower. If these ever converge, `anterpolate` has regressed onto
    # the reference implementation.
    big_c = EquidistantSampling{Float64}(24, 48)
    big_f = EquidistantSampling{Float64}(48, 96)
    G = randn(MersenneTwister(8), ComplexF64, Nφ(big_f), Nθ(big_f))
    anterpolate(G, big_f, big_c, SPHERICAL, FFTBackend())          # warm up
    fast = @elapsed anterpolate(G, big_f, big_c, SPHERICAL, FFTBackend())
    dense = @elapsed anterpolatedense(G, big_f, big_c, SPHERICAL, FFTBackend())
    @test dense > 20 * fast
end

end # module
