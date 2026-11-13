# Coefficient aggregation, and the receive side that is its transpose.
#
# The forward property is the one the whole scheme rests on:
#
#     synthesize(Σ_j I_j a_j)  ==  Σ_j I_j synthesize(a_j)
#
# i.e. that aggregating in coefficient space and aggregating in sample space give the same group
# pattern. It holds by linearity, so what is really under test is that the LAYOUT and the reshapes
# around it do not scramble anything -- exactly the kind of thing that is easy to get subtly wrong
# and impossible to notice downstream.
#
# The reverse property is `⟨F, G⟩_W == ⟨x, V⟩`, asserted directly rather than by comparing two
# implementations. Two adjoint bugs have shipped in this package (#35 Petrov, #61 Galerkin) and both
# looked plausible from the forward side alone.

using MLFMA
using MLFMA.CircularHarmonics
using MLFMA.CircularHarmonics:
    CircularHarmonicStorage,
    CircularHarmonicTransform,
    aggregate!,
    aggregatelevel!,
    analyze,
    analyzelevel!,
    coefficientmatrix,
    ncomponents,
    numbasisfunctions,
    numcoefficients,
    numdirections,
    order,
    synthesize,
    testcoefficients!,
    testlevel!
using LinearAlgebra
using Random
using StaticArrays

# Patterns exactly band-limited to M, so analysis is lossless and any discrepancy is the aggregation
# machinery rather than truncation.
function randompatterns(t, D, nbf; seed=0)
    rng = MersenneTwister(seed)
    coeffs = [randn(rng, ComplexF64, numcoefficients(t), D) for _ in 1:nbf]
    return [synthesize(t, c) for c in coeffs], coeffs
end

@testset "Circular harmonic storage" begin
    @testset "round trip through storage preserves the coefficients" begin
        M, D, nbf = 6, 3, 12
        t = CircularHarmonicTransform(ComplexF64, M)
        patterns, coeffs = randompatterns(t, D, nbf; seed=1)
        s = CircularHarmonicStorage(t, patterns, SVector(0.0, 0.0))
        @test order(s) == M
        @test ncomponents(s) == D
        @test numbasisfunctions(s) == nbf
        @test numcoefficients(s) == numcoefficients(M)
        @test coefficientmatrix(s) === s.coefficients
        for j in 1:nbf
            @test reshape(s.coefficients[:, j], numcoefficients(M), D) ≈ coeffs[j] atol =
                1e-10
        end
    end

    @testset "the centre is a PLANE point, not a sphere point with a zero" begin
        # The 2-D branch has removed eight quiet 3-isms, every one of which typechecked where it was
        # written. An `SVector{3}` centre here would be the ninth.
        M, D, nbf = 3, 1, 4
        t = CircularHarmonicTransform(ComplexF64, M)
        patterns, _ = randompatterns(t, D, nbf; seed=9)
        s = CircularHarmonicStorage(t, patterns, SVector(1.5, -2.5))
        @test s.center isa SVector{2,Float64}
        @test s.center == SVector(1.5, -2.5)
        @test_throws Exception CircularHarmonicStorage(t, patterns, SVector(0.0, 0.0, 0.0))
    end

    @testset "aggregation in coefficient space equals aggregation in sample space" begin
        M, D, nbf = 5, 3, 20
        t = CircularHarmonicTransform(ComplexF64, M)
        patterns, _ = randompatterns(t, D, nbf; seed=2)
        s = CircularHarmonicStorage(t, patterns, SVector(0.0, 0.0))

        x = randn(MersenneTwister(3), ComplexF64, nbf)
        direct = sum(x[j] * patterns[j] for j in 1:nbf)          # the sampled leaf multiply

        A = Vector{ComplexF64}(undef, numcoefficients(M) * D)
        aggregate!(A, s, x)
        viaCHE = synthesize(t, reshape(A, numcoefficients(M), D))

        @test viaCHE ≈ direct atol = 1e-9
    end

    @testset "one component behaves like the multi-component case" begin
        M, nbf = 4, 7
        t = CircularHarmonicTransform(ComplexF64, M)
        patterns, _ = randompatterns(t, 1, nbf; seed=4)
        s = CircularHarmonicStorage(t, patterns, SVector(0.0, 0.0))
        @test ncomponents(s) == 1
        x = randn(MersenneTwister(5), ComplexF64, nbf)
        A = Vector{ComplexF64}(undef, numcoefficients(M))
        aggregate!(A, s, x)
        @test synthesize(t, A) ≈ sum(x[j] * patterns[j][:, 1] for j in 1:nbf) atol = 1e-10
    end

    @testset "aggregatelevel! equals a per-group aggregate-then-synthesise" begin
        M, D, ngroups, nbf = 5, 3, 4, 9
        t = CircularHarmonicTransform(ComplexF64, M)
        nc = numcoefficients(M)
        storages = map(1:ngroups) do g
            patterns, _ = randompatterns(t, D, nbf; seed=10 + g)
            return CircularHarmonicStorage(t, patterns, SVector(1.0 * g, 0.0))
        end
        rng = MersenneTwister(6)
        xs = [randn(rng, ComplexF64, nbf) for _ in 1:ngroups]

        F = Matrix{ComplexF64}(undef, numdirections(t), D * ngroups)
        A = Matrix{ComplexF64}(undef, nc, D * ngroups)
        aggregatelevel!(F, A, t, storages, xs)

        for g in 1:ngroups
            Ag = Vector{ComplexF64}(undef, nc * D)
            aggregate!(Ag, storages[g], xs[g])
            @test F[:, ((g - 1) * D + 1):(g * D)] ≈ synthesize(t, reshape(Ag, nc, D)) atol =
                1e-10
        end

        @test_throws DimensionMismatch aggregatelevel!(F, A, t, storages, xs[1:2])
        @test_throws DimensionMismatch aggregatelevel!(
            F, Matrix{ComplexF64}(undef, nc, D), t, storages, xs
        )
        wrongorder = CircularHarmonicTransform(ComplexF64, M + 1)
        @test_throws ArgumentError aggregatelevel!(
            Matrix{ComplexF64}(undef, numdirections(wrongorder), D * ngroups),
            Matrix{ComplexF64}(undef, numcoefficients(wrongorder), D * ngroups),
            wrongorder,
            storages,
            xs,
        )
    end

    @testset "the leaf pair is EXACTLY adjoint" begin
        # ⟨S C x, G⟩_W == ⟨x, Cᴴ Sᴴ W G⟩ is a reordering of a finite sum, so it holds on ANY grid --
        # no quadrature accuracy is involved at all. That is not a technicality: the product
        # `G · conj(ψ_m)` reaches bandwidth `L + M`, which no leaf grid resolves, and if the identity
        # needed that integral to be exact the scheme would not work.
        #
        # So the grid is varied and the order is not. `resolved` integrates the pairing exactly;
        # `starved` has FEWER directions than `2M+2` and integrates nothing exactly. Both must give
        # the same adjointness to round-off, and the second is the load-bearing one.
        M, D, nbf = 5, 2, 6
        setup = CircularHarmonicTransform(ComplexF64, M)
        patterns, _ = randompatterns(setup, D, nbf; seed=7)
        s = CircularHarmonicStorage(setup, patterns, SVector(0.0, 0.0))

        starvedφ = [2π * (j - 1) / (M + 1) for j in 1:(M + 1)]
        grids = (
            "resolved" => setup,
            "starved" => CircularHarmonicTransform(
                ComplexF64, starvedφ, fill(2π / (M + 1), M + 1), M
            ),
        )
        for (label, tg) in grids
            rng = MersenneTwister(8)
            x = randn(rng, ComplexF64, nbf)
            G = randn(rng, ComplexF64, numdirections(tg), D)

            A = Vector{ComplexF64}(undef, numcoefficients(M) * D)
            aggregate!(A, s, x)
            F = synthesize(tg, reshape(A, numcoefficients(M), D))

            B = Matrix{ComplexF64}(undef, numcoefficients(M), D)
            analyzelevel!(B, tg, G)
            V = Vector{ComplexF64}(undef, nbf)
            testcoefficients!(V, s, vec(B))

            @test sum(tg.weights .* conj(F) .* G) ≈ dot(x, V)
            @test label == "resolved" || numdirections(tg) < 2M + 2
        end
    end

    @testset "testlevel! equals a per-group analyse-then-test" begin
        M, D, ngroups, nbf = 4, 2, 3, 5
        t = CircularHarmonicTransform(ComplexF64, M)
        nc = numcoefficients(M)
        storages = map(1:ngroups) do g
            patterns, _ = randompatterns(t, D, nbf; seed=20 + g)
            return CircularHarmonicStorage(t, patterns, SVector(0.0, 1.0 * g))
        end
        G = randn(MersenneTwister(21), ComplexF64, numdirections(t), D * ngroups)

        Vs = [Vector{ComplexF64}(undef, nbf) for _ in 1:ngroups]
        B = Matrix{ComplexF64}(undef, nc, D * ngroups)
        testlevel!(Vs, B, t, storages, G)

        Bref = Matrix{ComplexF64}(undef, nc, D * ngroups)
        analyzelevel!(Bref, t, G)
        @test B ≈ Bref
        for g in 1:ngroups
            Vg = Vector{ComplexF64}(undef, nbf)
            testcoefficients!(Vg, storages[g], vec(Bref[:, ((g - 1) * D + 1):(g * D)]))
            @test Vs[g] ≈ Vg
        end

        @test_throws DimensionMismatch testlevel!(Vs[1:2], B, t, storages, G)
        @test_throws DimensionMismatch analyzelevel!(Matrix{ComplexF64}(undef, nc, 1), t, G)
        @test_throws DimensionMismatch analyzelevel!(
            B, t, randn(ComplexF64, numdirections(t) + 2, D * ngroups)
        )
    end

    @testset "empty levels are a no-op, not an error" begin
        M, D = 3, 2
        t = CircularHarmonicTransform(ComplexF64, M)
        F = zeros(ComplexF64, numdirections(t), 0)
        A = zeros(ComplexF64, numcoefficients(M), 0)
        S = CircularHarmonicStorage{ComplexF64,Float64}[]
        @test aggregatelevel!(F, A, t, S, Vector{ComplexF64}[]) === F
        @test testlevel!(Vector{ComplexF64}[], A, t, S, F) == Vector{ComplexF64}[]
    end

    @testset "construction validation" begin
        M = 3
        t = CircularHarmonicTransform(ComplexF64, M)
        @test_throws ArgumentError CircularHarmonicStorage(
            t, Matrix{ComplexF64}[], SVector(0.0, 0.0)
        )
        bad = [
            randn(ComplexF64, numdirections(t), 2), randn(ComplexF64, numdirections(t), 3)
        ]
        @test_throws DimensionMismatch CircularHarmonicStorage(t, bad, SVector(0.0, 0.0))
        @test_throws DimensionMismatch CircularHarmonicStorage(
            M, SVector(0.0, 0.0), randn(ComplexF64, numcoefficients(M) + 1, 2)
        )
    end
end
