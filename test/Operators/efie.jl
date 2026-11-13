using MLFMA
using BEAST
using Test
using CompScienceMeshes

@testset "EFIE-Operator" begin
    for wavenumber in [1.0, 2.0, -100, 1 - 100 * im]
        alpha = -3.0
        beta = -4.0
        gamma = im * wavenumber

        efie = MLFMA.efie(; wavenumber=wavenumber, alpha=alpha)

        @test MLFMA.α(efie) == alpha
        @test MLFMA.γ(efie) == gamma
        @test BEAST.gamma(efie) == gamma

        MLFMA.MWEFIE3D(MLFMA.singlelayer(efie)) === efie
    end

    # `MWEFIE3D` guarantees α = βγ²: the single-layer constructor rejects a mismatched operator.
    # H2Factory's `@efie` macro (H2Factory/ext/H2BEAST/wrappers/maxwell/efie.jl) enforces this with
    # `isapprox(singlelayer.α, singlelayer.gamma^2 * singlelayer.β) || throw(ArgumentError(...))`.
    mismatchedsinglelayer = Maxwell3D.singlelayer(; wavenumber=10, alpha=1, beta=1)
    @test_throws ArgumentError MLFMA.MWEFIE3D(mismatchedsinglelayer)
end

@testset "MWHyperSingular3D" begin
    λ = 1.0
    m = meshsphere(λ, λ / 5)
    X = raviartthomas(m)

    hypersingular = MLFMA.hypersingular(; wavenumber=1.0)
    @test MLFMA.α(hypersingular) == 1.0 * im
    @test MLFMA.γ(hypersingular) == 1.0 * im
    @test BEAST.gamma(hypersingular) == 1.0 * im
    @test BEAST.scalartype(hypersingular) == ComplexF64

    hypersingularBEAST = Maxwell3D.hypersingular(; wavenumber=1.0)

    A = assemble(hypersingular, X, X)
    ABEAST = assemble(hypersingularBEAST, X, X)

    @test A == ABEAST
end
