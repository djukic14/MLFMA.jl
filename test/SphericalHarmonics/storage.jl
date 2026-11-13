# Coefficient aggregation. The property under test is the one the whole scheme rests on:
#
#     synthesize(Σ_j I_j a_j)  ==  Σ_j I_j synthesize(a_j)
#
# i.e. that aggregating in coefficient space and aggregating in sample space give the same group
# pattern. It holds by linearity, so the test is really checking that the LAYOUT and the reshapes
# around it do not scramble anything -- which is exactly the kind of thing that is easy to get subtly
# wrong and impossible to notice downstream.

using MLFMA
using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics:
    SphericalHarmonicStorage,
    SphericalHarmonicTransform,
    aggregate!,
    aggregatelevel!,
    analyze,
    analyzelevel!,
    gausslegendregrid,
    ncomponents,
    numbasisfunctions,
    numcoefficients,
    order,
    synthesize,
    testcoefficients!,
    testlevel!
using LinearAlgebra
using Random
using StaticArrays

# Patterns that are exactly band-limited to P, so analysis is lossless and any discrepancy is the
# aggregation machinery rather than truncation.
function randompatterns(t, D, nbf; seed=0)
    rng = MersenneTwister(seed)
    nd = MLFMA.SphericalHarmonics.numdirections(t)
    coeffs = [randn(rng, ComplexF64, numcoefficients(t), D) for _ in 1:nbf]
    return [synthesize(t, c) for c in coeffs], coeffs
end

@testset "Spherical harmonic storage" begin
    @testset "round trip through storage preserves the coefficients" begin
        P, D, nbf = 6, 3, 12
        t = SphericalHarmonicTransform(ComplexF64, P)
        patterns, coeffs = randompatterns(t, D, nbf; seed=1)
        s = SphericalHarmonicStorage(t, patterns, SVector(0.0, 0.0, 0.0))
        @test order(s) == P
        @test ncomponents(s) == D
        @test numbasisfunctions(s) == nbf
        @test numcoefficients(s) == numcoefficients(P)
        for j in 1:nbf
            @test reshape(s.coefficients[:, j], numcoefficients(P), D) ≈ coeffs[j] atol =
                1e-10
        end
    end

    @testset "aggregation in coefficient space equals aggregation in sample space" begin
        P, D, nbf = 5, 3, 20
        t = SphericalHarmonicTransform(ComplexF64, P)
        patterns, _ = randompatterns(t, D, nbf; seed=2)
        s = SphericalHarmonicStorage(t, patterns, SVector(0.0, 0.0, 0.0))

        x = randn(MersenneTwister(3), ComplexF64, nbf)
        direct = sum(x[j] * patterns[j] for j in 1:nbf)          # the sampled leaf multiply

        A = Vector{ComplexF64}(undef, numcoefficients(P) * D)
        aggregate!(A, s, x)
        viaSHE = synthesize(t, reshape(A, numcoefficients(P), D))

        @test viaSHE ≈ direct atol = 1e-9
    end

    @testset "one component behaves like the multi-component case" begin
        P, nbf = 4, 7
        t = SphericalHarmonicTransform(ComplexF64, P)
        patterns, _ = randompatterns(t, 1, nbf; seed=4)
        s = SphericalHarmonicStorage(t, patterns, SVector(0.0, 0.0, 0.0))
        @test ncomponents(s) == 1
        x = randn(MersenneTwister(5), ComplexF64, nbf)
        A = Vector{ComplexF64}(undef, numcoefficients(P))
        aggregate!(A, s, x)
        @test synthesize(t, A) ≈ sum(x[j] * patterns[j] for j in 1:nbf) atol = 1e-9
    end

    @testset "aggregatelevel! matches group-by-group aggregation" begin
        # The batched path is the one production must use (see storage.jl's header), so it is checked
        # against the obvious per-group loop rather than trusted.
        P, D, ngroups = 5, 3, 6
        t = SphericalHarmonicTransform(ComplexF64, P)
        nc = numcoefficients(P)
        nd = MLFMA.SphericalHarmonics.numdirections(t)

        storages = SphericalHarmonicStorage[]
        xs = Vector{ComplexF64}[]
        for g in 1:ngroups
            nbf = 3 + g                       # deliberately ragged group sizes
            patterns, _ = randompatterns(t, D, nbf; seed=10 + g)
            push!(storages, SphericalHarmonicStorage(t, patterns, SVector(0.0, 0.0, 0.0)))
            push!(xs, randn(MersenneTwister(100 + g), ComplexF64, nbf))
        end

        A = Matrix{ComplexF64}(undef, nc, D * ngroups)
        F = Matrix{ComplexF64}(undef, nd, D * ngroups)
        aggregatelevel!(F, A, t, storages, xs)

        for g in 1:ngroups
            a = Vector{ComplexF64}(undef, nc * D)
            aggregate!(a, storages[g], xs[g])
            expected = synthesize(t, reshape(a, nc, D))
            @test F[:, ((g - 1) * D + 1):(g * D)] ≈ expected atol = 1e-10
        end
    end

    @testset "shape and consistency errors are caught" begin
        P, D = 3, 3
        t = SphericalHarmonicTransform(ComplexF64, P)
        nc = numcoefficients(P)
        patterns, _ = randompatterns(t, D, 4; seed=6)
        s = SphericalHarmonicStorage(t, patterns, SVector(0.0, 0.0, 0.0))

        # rows must be a whole number of (l,m) blocks
        @test_throws DimensionMismatch SphericalHarmonicStorage(
            P, SVector(0.0, 0.0, 0.0), zeros(ComplexF64, nc + 1, 2)
        )
        @test_throws ArgumentError SphericalHarmonicStorage(
            t, Matrix{ComplexF64}[], SVector(0.0, 0.0, 0.0)
        )
        # scratch of the wrong shape
        badA = Matrix{ComplexF64}(undef, nc, D)
        badF = Matrix{ComplexF64}(undef, MLFMA.SphericalHarmonics.numdirections(t), D)
        @test_throws DimensionMismatch aggregatelevel!(
            badF, badA, t, [s, s], [ones(ComplexF64, 4), ones(ComplexF64, 4)]
        )
        # mismatched storages/vectors
        @test_throws DimensionMismatch aggregatelevel!(
            badF, badA, t, [s], Vector{ComplexF64}[]
        )
        # a storage whose order disagrees with the transform
        t2 = SphericalHarmonicTransform(ComplexF64, P + 1)
        A2 = Matrix{ComplexF64}(undef, numcoefficients(P + 1), D)
        F2 = Matrix{ComplexF64}(undef, MLFMA.SphericalHarmonics.numdirections(t2), D)
        @test_throws ArgumentError aggregatelevel!(F2, A2, t2, [s], [ones(ComplexF64, 4)])
    end
end

@testset "Spherical harmonic receive side" begin
    @testset "the leaf pair is an exact transpose pair" begin
        # ⟨F, G⟩_W == ⟨x, V⟩ with F the forward chain of x and V the reverse chain of G. This is the
        # property the whole SE-MLFMA leaf replacement rests on (Eibert Eq. 13), and it holds in exact
        # arithmetic on ANY grid -- no quadrature accuracy needed, only the SAME weights on both
        # sides. So it is asserted on a deliberately COARSE, inexact grid: if it only held where the
        # quadrature happened to be good, it would not be the identity claimed.
        Random.seed!(31)
        P, D, nbf = 5, 3, 9
        setup = SphericalHarmonicTransform(ComplexF64, P)
        patterns, _ = randompatterns(setup, D, nbf; seed=32)
        s = SphericalHarmonicStorage(setup, patterns, SVector(0.0, 0.0, 0.0))

        # A leaf grid that does NOT integrate degree-P products accurately.
        leaf = SphericalHarmonicTransform(
            ComplexF64, MLFMA.SphericalSampling{Float64}(8, 16), P
        )
        nd = MLFMA.SphericalHarmonics.numdirections(leaf)
        nc = numcoefficients(P)

        for _ in 1:5
            x = randn(ComplexF64, nbf)
            G = randn(ComplexF64, nd, D)

            A = Vector{ComplexF64}(undef, nc * D)
            aggregate!(A, s, x)
            F = synthesize(leaf, reshape(A, nc, D))

            B = Matrix{ComplexF64}(undef, nc, D)
            analyzelevel!(B, leaf, G)
            V = Vector{ComplexF64}(undef, nbf)
            testcoefficients!(V, s, vec(B))

            lhs = sum(leaf.weights .* conj.(F) .* G)     # ⟨F, G⟩_W
            @test lhs ≈ dot(x, V) rtol = 1e-11
        end
    end

    @testset "testing via coefficients equals direct quadrature against the patterns" begin
        # V_j = Σ_q w_q conj(F_j[q])·G[q] computed directly from the SAMPLED patterns, against the
        # coefficient-space form. Here the two agree only when the patterns really are the ones the
        # coefficients represent, so this checks the storage content and not just the algebra.
        Random.seed!(33)
        P, D, nbf = 6, 3, 11
        t = SphericalHarmonicTransform(ComplexF64, P)
        patterns, _ = randompatterns(t, D, nbf; seed=34)
        s = SphericalHarmonicStorage(t, patterns, SVector(0.0, 0.0, 0.0))
        nc = numcoefficients(P)

        G = randn(ComplexF64, MLFMA.SphericalHarmonics.numdirections(t), D)
        direct = [sum(t.weights .* conj.(patterns[j]) .* G) for j in 1:nbf]

        B = Matrix{ComplexF64}(undef, nc, D)
        analyzelevel!(B, t, G)
        V = Vector{ComplexF64}(undef, nbf)
        testcoefficients!(V, s, vec(B))

        @test V ≈ direct rtol = 1e-10
    end

    @testset "testlevel! matches group-by-group testing" begin
        P, D, ngroups = 4, 3, 5
        t = SphericalHarmonicTransform(ComplexF64, P)
        nc = numcoefficients(P)
        nd = MLFMA.SphericalHarmonics.numdirections(t)

        storages = SphericalHarmonicStorage[]
        for g in 1:ngroups
            patterns, _ = randompatterns(t, D, 2 + g; seed=40 + g)
            push!(storages, SphericalHarmonicStorage(t, patterns, SVector(0.0, 0.0, 0.0)))
        end
        G = randn(MersenneTwister(44), ComplexF64, nd, D * ngroups)

        B = Matrix{ComplexF64}(undef, nc, D * ngroups)
        Vs = [Vector{ComplexF64}(undef, numbasisfunctions(s)) for s in storages]
        testlevel!(Vs, B, t, storages, G)

        for g in 1:ngroups
            Bg = Matrix{ComplexF64}(undef, nc, D)
            analyzelevel!(Bg, t, G[:, ((g - 1) * D + 1):(g * D)])
            expected = Vector{ComplexF64}(undef, numbasisfunctions(storages[g]))
            testcoefficients!(expected, storages[g], vec(Bg))
            @test Vs[g] ≈ expected atol = 1e-11
        end
    end

    @testset "receive-side shape errors are caught" begin
        P, D = 3, 3
        t = SphericalHarmonicTransform(ComplexF64, P)
        nc = numcoefficients(P)
        nd = MLFMA.SphericalHarmonics.numdirections(t)
        patterns, _ = randompatterns(t, D, 4; seed=50)
        s = SphericalHarmonicStorage(t, patterns, SVector(0.0, 0.0, 0.0))

        @test_throws DimensionMismatch analyzelevel!(
            Matrix{ComplexF64}(undef, nc, D), t, randn(ComplexF64, nd + 1, D)
        )
        @test_throws DimensionMismatch analyzelevel!(
            Matrix{ComplexF64}(undef, nc + 1, D), t, randn(ComplexF64, nd, D)
        )
        @test_throws DimensionMismatch testlevel!(
            [Vector{ComplexF64}(undef, 4)],
            Matrix{ComplexF64}(undef, nc, D),
            t,
            [s, s],
            randn(ComplexF64, nd, 2D),
        )
    end
end
