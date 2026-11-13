module EquidistantAnterpolationTests

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
    θs,
    φs,
    resample,
    anterpolate,
    resamplematrix,
    innerproduct,
    exactquadratureweights,
    quadratureweights,
    FFTBackend,
    LagrangeBackend

include("ylmoracle.jl")

const COARSE = EquidistantSampling{Float64}(6, 12)
const FINE = EquidistantSampling{Float64}(12, 24)

@testset "the weighted adjoint relation holds exactly" begin
    # ⟨R f, g⟩_fine == ⟨f, A g⟩_coarse for ARBITRARY f and g -- not just for fields either grid
    # resolves, because this is an algebraic property of the operator, not an accuracy claim.
    rng = MersenneTwister(11)
    for backend in (FFTBackend(), LagrangeBackend(4), LagrangeBackend(6))
        for parity in (SPHERICAL, ANTISPHERICAL)
            f = randn(rng, ComplexF64, Nφ(COARSE), Nθ(COARSE))
            g = randn(rng, ComplexF64, Nφ(FINE), Nθ(FINE))
            lhs = innerproduct(resample(f, COARSE, FINE, parity, backend), g, FINE)
            rhs = innerproduct(f, anterpolate(g, FINE, COARSE, parity, backend), COARSE)
            @test lhs ≈ rhs rtol = 1e-11
        end
    end
end

@testset "the plain unweighted adjoint does NOT satisfy it" begin
    # Guards against the relation passing for a trivial reason -- if the two weight sets happened to
    # cancel, the testset above would be vacuous.
    rng = MersenneTwister(12)
    f = randn(rng, ComplexF64, Nφ(COARSE), Nθ(COARSE))
    g = randn(rng, ComplexF64, Nφ(FINE), Nθ(FINE))
    R = resamplematrix(COARSE, FINE, SPHERICAL, FFTBackend())
    bare = reshape(R' * vec(g), Nφ(COARSE), Nθ(COARSE))     # no weights anywhere
    lhs = innerproduct(resample(f, COARSE, FINE, SPHERICAL, FFTBackend()), g, FINE)
    @test !isapprox(lhs, innerproduct(f, bare, COARSE); rtol=1e-6)
end

@testset "anterpolation preserves the integral of a resolved field" begin
    # A pattern's total ∮ dΩ is what the coarse level must inherit; with the weighted adjoint this is
    # exact, because the constant function is in the range of interpolation.
    f = bandlimitedfield(3)
    G = sampleon(f, FINE)
    for backend in (FFTBackend(), LagrangeBackend(6))
        A = anterpolate(G, FINE, COARSE, SPHERICAL, backend)
        @test innerproduct(A, ones(size(A)), COARSE) ≈ innerproduct(G, ones(size(G)), FINE) rtol =
            1e-11
    end
end

@testset "the weighted adjoint needs the EXACT weights" begin
    # With midpoint weights the same construction inherits their ~1e-2 error, which is the concrete
    # reason `exactquadratureweights` is not optional for the reverse direction either.
    rng = MersenneTwister(13)
    f = randn(rng, ComplexF64, Nφ(COARSE), Nθ(COARSE))
    g = randn(rng, ComplexF64, Nφ(FINE), Nθ(FINE))
    R = resamplematrix(COARSE, FINE, SPHERICAL, FFTBackend())
    Wf, Wc = vec(quadratureweights(FINE)), vec(quadratureweights(COARSE))
    naive = reshape((R' * (Wf .* vec(g))) ./ Wc, Nφ(COARSE), Nθ(COARSE))
    lhs = innerproduct(resample(f, COARSE, FINE, SPHERICAL, FFTBackend()), g, FINE)
    @test !isapprox(lhs, innerproduct(f, naive, COARSE); rtol=1e-6)
end

end # module
