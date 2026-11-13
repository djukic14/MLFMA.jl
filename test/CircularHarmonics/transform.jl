# The transform pair, the layout, and the truncation rule.
#
# The properties asserted here are analytic, which is the whole reason `CircularHarmonics` is a
# standalone submodule: orthonormality, Parseval and exactness of the trapezoidal rule are checkable
# without a tree, an assembler or a collection anywhere in sight.
#
# ONE PROPERTY IS PINNED HARDER THAN THE REST, and it is the one this family's docstrings lean on
# everywhere: `N = 2M+2` points integrate the analysis integrand EXACTLY, where the spherical grid
# only converges to it. That is asserted BOTH ways below -- exact at `2M+2`, and demonstrably not at
# `2M` -- because "exact" is a claim about a bound being tight, and a test that only checks the
# passing side would still pass if the bound were wrong.

using MLFMA
using MLFMA.CircularHarmonics
using MLFMA.CircularHarmonics:
    CHETAILFLOOR,
    CHETAILRATIO,
    CircularHarmonicTransform,
    analyze,
    circulargrid,
    leveltruncationorder,
    mindex,
    modeenergy,
    numcoefficients,
    numdirections,
    order,
    orderof,
    synthesize,
    synthesize!,
    truncatecoefficients,
    truncationorder
using LinearAlgebra
using Random

@testset "Circular harmonic transform" begin
    @testset "layout: mindex and orderof invert each other" begin
        for M in 0:6
            @test numcoefficients(M) == 2M + 1
            for m in (-M):M
                @test orderof(mindex(m, M), M) == m
            end
            @test mindex(-M, M) == 1
            @test mindex(M, M) == numcoefficients(M)
            # Centered, NOT leading -- mode 0 sits in the middle. This is the layout decision
            # `transform.jl`'s header records, and truncation depends on it.
            @test mindex(0, M) == M + 1
        end
    end

    @testset "circulargrid is NφCircular, and its weights are the circle's measure" begin
        for M in 0:5, over in 1:3
            φs, w = circulargrid(M; oversampling=over)
            @test length(φs) == length(w) == over * MLFMA.NφCircular(M)
            @test sum(w) ≈ 2π
            @test all(≈(2π / length(w)), w)
            @test φs[1] == 0
            @test maximum(φs) < 2π
        end
        @test_throws ArgumentError circulargrid(-1)
        @test_throws ArgumentError circulargrid(3; oversampling=0)
    end

    @testset "the rule is EXACT at N = 2M+2, and not at 2M" begin
        # `SᴴWS == I` says every pair `ψ_m conj(ψ_m′)`, bandwidth up to 2M, is integrated with no
        # error. On the circle that holds at the minimum rate; it is what the spherical grid, which
        # carries `sin θ` in its weight, does NOT get (see `transform.jl`'s header).
        for M in (1, 3, 7)
            t = CircularHarmonicTransform(ComplexF64, M)
            @test numdirections(t) == 2M + 2
            @test maximum(abs, t.synthesis' * (t.weights .* t.synthesis) - I) < 1e-13
        end

        # Two points short of the bound the top corner aliases onto its own conjugate and the
        # identity breaks by O(1). Asserted so that "exact at 2M+2" is a tight statement rather than
        # a safe one.
        M = 5
        φs = [2π * (j - 1) / (2M) for j in 1:(2M)]
        undersampled = CircularHarmonicTransform(ComplexF64, φs, fill(2π / (2M), 2M), M)
        @test maximum(
            abs,
            undersampled.synthesis' * (undersampled.weights .* undersampled.synthesis) - I,
        ) > 0.5
    end

    @testset "synthesize and analyze round trip" begin
        M = 6
        t = CircularHarmonicTransform(ComplexF64, M)
        rng = MersenneTwister(1)
        a = randn(rng, ComplexF64, numcoefficients(M))
        @test maximum(abs, analyze(t, synthesize(t, a)) - a) < 1e-13

        # Columns are independent components and stay independent.
        A = randn(rng, ComplexF64, numcoefficients(M), 3)
        @test maximum(abs, analyze(t, synthesize(t, A)) - A) < 1e-13

        F = Matrix{ComplexF64}(undef, numdirections(t), 3)
        synthesize!(F, t, A)
        @test F == synthesize(t, A)
    end

    @testset "Parseval: the harmonics are orthoNORMAL, not merely orthogonal" begin
        M = 5
        t = CircularHarmonicTransform(ComplexF64, M)
        a = randn(MersenneTwister(2), ComplexF64, numcoefficients(M))
        F = synthesize(t, a)
        @test sum(abs2, a) ≈ sum(t.weights .* abs2.(F))
    end

    @testset "analyze is the adjoint of synthesize, by construction" begin
        # The property `storage.jl`'s receive side rests on, asserted on the transform itself so a
        # failure points at `S` rather than at the leaf pairing built on it.
        M = 4
        t = CircularHarmonicTransform(ComplexF64, M)
        rng = MersenneTwister(3)
        a = randn(rng, ComplexF64, numcoefficients(M))
        G = randn(rng, ComplexF64, numdirections(t))
        @test dot(t.weights .* synthesize(t, a), G) ≈ dot(a, analyze(t, G))
    end

    @testset "on a CircularSampling: same grid, same order as far fields are stored in" begin
        M = 7
        s = MLFMA.CircularSampling(MLFMA.NφCircular(M))
        t = CircularHarmonicTransform(ComplexF64, s, M)
        @test numdirections(t) == length(s)
        @test maximum(abs, t.synthesis' * (t.weights .* t.synthesis) - I) < 1e-13
        # Row `q` must be the harmonics at the angle a far field stores at position `q`, or every
        # coefficient in the package would be a permutation of the truth.
        φ = MLFMA.φs(s)
        for q in eachindex(φ), m in (-M):M
            @test t.synthesis[q, mindex(m, M)] ≈ cis(m * φ[q]) / sqrt(2π)
        end
    end

    @testset "constructor validation" begin
        @test_throws DimensionMismatch CircularHarmonicTransform(
            ComplexF64, [0.0, 1.0, 2.0], [1.0, 1.0], 1
        )
        S = ones(ComplexF64, 4, 5)
        @test_throws DimensionMismatch CircularHarmonicTransform(2, S, ones(3))
        @test_throws DimensionMismatch CircularHarmonicTransform(1, S, ones(4))
        @test_throws ArgumentError CircularHarmonicTransform(
            -1, ones(ComplexF64, 4, 1), ones(4)
        )
    end

    @testset "modeenergy pools ±m, because truncation keeps them as a pair" begin
        M = 4
        a = zeros(ComplexF64, numcoefficients(M))
        a[mindex(3, M)] = 2.0
        a[mindex(-3, M)] = 1.0im
        e = modeenergy(a)
        @test length(e) == M + 1
        @test e[abs(3) + 1] ≈ 5.0
        @test sum(e) ≈ sum(abs2, a)
        @test all(iszero, e[[1, 2, 3, 5]])

        # Columns are summed too, so a multi-component expansion has one energy profile.
        A = hcat(a, a)
        @test modeenergy(A) ≈ 2 .* e

        @test_throws ArgumentError modeenergy(zeros(ComplexF64, 4))
    end

    @testset "truncationorder finds the order the tail actually needs" begin
        M = 6
        a = zeros(ComplexF64, numcoefficients(M))
        a[mindex(0, M)] = 1.0
        a[mindex(4, M)] = 1e-4
        @test truncationorder(a, 1e-2) == 0        # the 1e-4 mode holds 1e-8 of the energy
        @test truncationorder(a, 1e-6) == 4
        @test truncationorder(zeros(ComplexF64, numcoefficients(M)), 1e-8) == 0

        # Monotone in ε, and never above the order actually present.
        b = randn(MersenneTwister(4), ComplexF64, numcoefficients(M))
        orders = [truncationorder(b, ε) for ε in (1e-1, 1e-2, 1e-4, 1e-8)]
        @test issorted(orders)
        @test all(<=(M), orders)
    end

    @testset "truncatecoefficients is a CENTERED contiguous view" begin
        M = 5
        a = ComplexF64.(1:numcoefficients(M))
        v = truncatecoefficients(a, 2)
        @test v isa SubArray
        @test length(v) == numcoefficients(2)
        # Rows (M-2+1):(M+2+1) = 4:8, i.e. modes -2 … 2 -- NOT the leading 5 rows, which would be
        # modes -5 … -1. Getting this wrong is a silent low-pass into a phase ramp.
        @test collect(v) == ComplexF64.(4:8)
        for m in (-2):2
            @test v[mindex(m, 2)] == a[mindex(m, M)]
        end

        A = repeat(a, 1, 3)
        @test size(truncatecoefficients(A, 1)) == (3, 3)
        @test truncatecoefficients(a, M) == a

        @test_throws ArgumentError truncatecoefficients(a, M + 1)
        @test_throws ArgumentError truncatecoefficients(a, -1)
        @test_throws ArgumentError truncatecoefficients(zeros(ComplexF64, 4), 1)
    end

    @testset "leveltruncationorder takes the MAX over a level, never the mean" begin
        M = 8
        quiet = zeros(ComplexF64, numcoefficients(M))
        quiet[mindex(0, M)] = 1.0
        loud = copy(quiet)
        loud[mindex(6, M)] = 1e-3

        # One box needing 6 forces the whole level to 6: the batched synthesis has one `S`.
        @test leveltruncationorder([quiet], 1e-4; ratio=1) == 0
        @test leveltruncationorder([quiet, loud], 1e-4; ratio=1) == 6
        @test leveltruncationorder([loud, quiet], 1e-4; ratio=1) == 6

        # Tightening the ratio can only raise the order, never lower it.
        @test leveltruncationorder([quiet, loud], 1e-4) >=
            leveltruncationorder([quiet, loud], 1e-4; ratio=1)

        @test_throws ArgumentError leveltruncationorder([quiet], 0.0)
        @test_throws ArgumentError leveltruncationorder([quiet], 1e-4; ratio=0)
        @test_throws ArgumentError leveltruncationorder(Matrix{ComplexF64}[], 1e-4)
    end

    @testset "the tail floor stops the criterion asking for round-off" begin
        # `τ² < eps()` is unsatisfiable, so without the floor `truncationorder` runs to the full order
        # present and returns a ceiling rather than an answer -- the failure `SHETAILFLOOR` was added
        # for in 3-D, reproduced here to show the same floor is load-bearing.
        @test CHETAILFLOOR == 1e-7
        @test CHETAILRATIO == 10

        # Decaying fast enough that a 1e-14 relative tail IS reachable below the full order and a
        # 1e-20 one is not -- which is exactly the window the floor exists to keep the criterion in.
        M = 10
        a = ComplexF64[exp(-2 * abs(orderof(k, M))) for k in 1:numcoefficients(M)]
        floored = leveltruncationorder([a], 1e-9)
        @test floored == truncationorder(a, CHETAILFLOOR)
        @test floored < M
        # Without the floor the same ε would demand a relative energy of 1e-20, which no order below
        # the ceiling satisfies -- so the rule stops answering and starts returning "everything".
        @test leveltruncationorder([a], 1e-9; tailfloor=1e-12) == M
    end
end
