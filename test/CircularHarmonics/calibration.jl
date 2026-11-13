# TRUNCATION ERROR ON REAL 2-D PATTERNS, and the crossover, as CALIBRATIONS rather than footnotes.
#
# Everything in `transform.jl` and `storage.jl` is analytic: orthonormality, Parseval, adjointness,
# layout. None of it says whether a real basis function's far field compresses, or by how much, or
# whether the truncation rule picks an order that works. Those are empirical and they are what
# actually decides whether this representation is usable, so they are asserted here against a real
# BEAST assembly rather than left to a benchmark nobody runs.
#
# ------------------------------------------------------------------------------------------------
# SECTION 1 is the user's test (a): basis-pattern Fourier truncation error.
#
# Measured on the GROUP MOMENT -- `Σ_j x_j F_j` against `S_M (Σ_j x_j a_j)` -- because that is what
# the compressed leaf multiply produces. A per-pattern error would be a different quantity that
# happens to be easier to compute, and #120 and #135 both had numbers retracted for exactly that
# substitution.
#
# THE LOAD-BEARING ASSERTION IS THAT THE RULE NEVER UNDER-PREDICTS. `CHETAILRATIO = 10` is calibrated
# in `benchmark/interpolation/checalibration.jl` over 20 rows; this pins the two ends of that
# calibration so a future change to the constant, to `modeenergy`, or to the layout cannot silently
# start spending the map's error budget. Under-prediction is the dangerous direction: the map still
# runs, still converges, and is simply less accurate than it was asked to be.
#
# ------------------------------------------------------------------------------------------------
# SECTION 2 is the crossover, and it is asserted CAREFULLY because it is a timing measurement on a
# shared machine.
#
# What is NOT asserted: absolute milliseconds, or that the compressed path is faster anywhere. The
# repeat runs behind `checrossover.jl` moved the low-occupancy ratio between 1.43x and 1.69x at one
# BLAS thread, and BLAS threading moves it to 3.25x. An absolute threshold would be a flake generator.
#
# What IS asserted is the SHAPE, which was stable across every run: compressed storage is clearly
# slower at the occupancy ordinary 2-D fixtures have, and the ratio improves monotonically as
# occupancy rises. Both are WITHIN-RUN comparisons, so machine load cancels between the two halves.
# If a future change moves the crossover -- a different synthesis, a different layout, a batching
# regression -- the ordering is what breaks first.

using MLFMA
using MLFMA.CircularHarmonics
using MLFMA.CircularHarmonics:
    CHETAILRATIO,
    CircularHarmonicTransform,
    analyze,
    leveltruncationorder,
    numcoefficients,
    synthesize,
    truncatecoefficients
using BEAST, CompScienceMeshes, H2Trees
using LinearAlgebra
using Random

# NO `using Statistics` -- it is not in `test/Project.toml`, and the only thing it was here for was
# one `median`. `benchmark/interpolation/checalibration.jl` may use it (the benchmark environment has
# it); the suite must not, or the whole testset errors at LOAD and its assertions silently stop
# running while the gate still reports every other file as green.

const HT = H2Trees

# κ = 2π, so λ = 1 and `minhalfsize` reads directly as a fraction of a wavelength. 0.25 is
# `MINHALFSIZERECOM`, the λ/4 low-frequency-breakdown floor. A forced-uniform tree, so "the leaf
# grid" is one thing and a level order is meaningful.
function circlefixture(; radius=2.0, h=1 / 20, minhalfsize=0.25, ϵ=1e-5)
    κ = 2π
    operator = Helmholtz2D.singlelayer(; wavenumber=κ)
    X = lagrangec0d1(CompScienceMeshes.meshcircle(radius, h))
    tree = HT.TwoNTree(
        X;
        builder=HT.TwoNTreeBuilder(;
            minhalfsize=minhalfsize, minvalues=0, protrusion=HT.NoProtrusionCheck()
        ),
    )
    leaves = collect(HT.leaves(tree))
    leaflevel = maximum(HT.level(tree, leaf) for leaf in leaves)
    sampling, _ = MLFMA.sampling(
        im * κ,
        ϵ,
        tree;
        level=leaflevel,
        samplingbuilder=MLFMA.CircularSamplingBuilder{Float64}(),
    )
    assembler = MLFMA.BEASTFarFieldAssembler(
        MLFMA.radiationoperator(operator),
        X,
        BEAST.SingleNumQStrat(3),
        MLFMA.farfieldstorerule(operator),
        MLFMA.farfieldtype(operator),
    )
    # Each leaf's own centre: the coefficients are only summable because every column of a storage
    # shares one expansion centre, and `BoxFarFieldSampling` is what makes the assembly produce that.
    boxes = [
        MLFMA.BoxFarFieldSampling(
            sampling,
            first(MLFMA.cartesiansamplepoints(sampling)),
            HT.center(tree, leaf),
            leaf,
        ) for leaf in leaves
    ]
    patterns = assembler(tree, boxes; verbose=false)
    return (; X, tree, leaves, sampling, patterns)
end

# Worst relative group-moment error over the leaves at kept order `M`. Aggregates in coefficient
# space FIRST and synthesises once, which is the operation order the scheme uses -- synthesising per
# basis function and summing afterwards would be a different and pointlessly expensive computation.
function groupmomenterror(f, coefficients, M; ntrials=3, seed=0)
    t = CircularHarmonicTransform(ComplexF64, f.sampling, M)
    rng = MersenneTwister(seed)
    worst = 0.0
    for i in eachindex(f.patterns)
        P = f.patterns[i][1]
        a = truncatecoefficients(coefficients[i], M)
        for _ in 1:ntrials
            x = randn(rng, ComplexF64, size(P, 2))
            exact = P * x
            worst = max(worst, norm(synthesize(t, a * x) - exact) / norm(exact))
        end
    end
    return worst
end

@testset "Circular harmonic calibration on real 2-D patterns" begin
    f = circlefixture()
    L = MLFMA.cutoffparameter(f.sampling)
    # Analysed on the leaf's OWN grid: `checalibration.jl` sweeps oversampling 1 through 16 and the
    # error is identical to six digits, so there is nothing to buy here. See
    # `CircularHarmonicStorage`'s docstring.
    analysis = CircularHarmonicTransform(ComplexF64, f.sampling, L)
    coefficients = [analyze(analysis, F[1]) for F in f.patterns]

    @testset "the fixture is the regime the docstrings describe" begin
        @test length(f.sampling) == MLFMA.NφCircular(L)
        @test !isempty(f.leaves)
        # ~10 basis functions per leaf: a box on a CURVE holds O(boxsize/h), which is the whole
        # reason 2-D sits below the speed crossover. If a mesh change moves this the crossover
        # assertions below are describing a different fixture than the docstrings do.
        #
        # The MEAN, which is `numfunctions / nleaves` -- exactly what the "basis fns/leaf" column of
        # the crossover table means, since that sweep holds the level's function count fixed and
        # varies only the grouping.
        occupancy = sum(size(F[1], 2) for F in f.patterns) / length(f.patterns)
        @test 4 <= occupancy <= 24
    end

    @testset "(a) truncation error falls monotonically with the order kept" begin
        errors = [groupmomenterror(f, coefficients, M) for M in 0:L]
        @test issorted(errors; rev=true)
        # A pattern that is nearly a constant at M = 0 and resolved by the time M reaches L: the
        # spread is what makes truncation worth doing at all.
        @test errors[1] > 0.5
        @test errors[end] < 1e-9
    end

    @testset "the calibrated rule never UNDER-predicts the order needed" begin
        for ε in (1e-3, 1e-4, 1e-5)
            M = leveltruncationorder(coefficients, ε)
            @test 0 < M <= L
            @test groupmomenterror(f, coefficients, M) <= ε
        end
    end

    @testset "τ = ε alone is not enough -- which is why CHETAILRATIO exists" begin
        # The constant is not decoration. On the calibration fixtures `τ = ε` under-predicts in 8 of
        # 20 rows; here it is enough to assert that tightening is never a downgrade and that the
        # default does tighten, so a change to 1 has to justify itself against `checalibration.jl`.
        @test CHETAILRATIO > 1
        for ε in (1e-3, 1e-4, 1e-5)
            @test leveltruncationorder(coefficients, ε) >=
                leveltruncationorder(coefficients, ε; ratio=1)
        end
    end

    @testset "compression is (2L+2)/(2M+1), and it is modest -- 1.4x-2.2x, not 10x" begin
        for ε in (1e-3, 1e-4, 1e-5)
            M = leveltruncationorder(coefficients, ε)
            ratio = MLFMA.NφCircular(L) / numcoefficients(M)
            @test ratio ≈ length(f.sampling) / (2M + 1)
            # The band the docstrings quote. A circle has no `8(L+1)²`-samples-for-`(L+1)²`-modes
            # slack to exploit, so every bit of this comes from M < L -- see the module docstring.
            @test 1.2 < ratio < 3.0
        end
    end

    @testset "the speed crossover points the way the docstrings say" begin
        # Synthetic matrices: this times two multiply SHAPES and the numbers in them never enter the
        # timing. The shapes come from the fixture above -- Nφ from the leaf grid, ncoeff from the
        # calibrated order at the ε the fixture was built with.
        Nφ = length(f.sampling)
        ncoeff = numcoefficients(leveltruncationorder(coefficients, 1e-5))
        total = 1920

        function ratioat(perleaf)
            nleaves = total ÷ perleaf
            rng = MersenneTwister(11)
            P = [randn(rng, ComplexF64, Nφ, perleaf) for _ in 1:nleaves]
            C = [randn(rng, ComplexF64, ncoeff, perleaf) for _ in 1:nleaves]
            S = randn(rng, ComplexF64, Nφ, ncoeff)
            xs = [randn(rng, ComplexF64, perleaf) for _ in 1:nleaves]
            F = Matrix{ComplexF64}(undef, Nφ, nleaves)
            A = Matrix{ComplexF64}(undef, ncoeff, nleaves)

            sampled() = @inbounds for g in 1:nleaves
                @views mul!(F[:, g], P[g], xs[g])
            end
            # ONE GEMM for the level. Unbatched synthesis is a regression on both axes (#122
            # measured 1.63x in 3-D) and 2-D has less margin, so the batched form is the only one
            # timed here.
            function compressed()
                @inbounds for g in 1:nleaves
                    @views mul!(A[:, g], C[g], xs[g])
                end
                return mul!(F, S, A)
            end

            best(fn) = (fn(); minimum(@elapsed(fn()) for _ in 1:7))
            return best(compressed) / best(sampled)
        end

        # One BLAS thread, restored afterwards: the recorded table is at one thread, and threading
        # makes the low-occupancy case worse rather than better (3.25x at 8 threads), so leaving it
        # to whatever the runner set would compare against a different measurement.
        restore = BLAS.get_num_threads()
        low, high = try
            BLAS.set_num_threads(1)
            ratioat(8), ratioat(256)
        finally
            BLAS.set_num_threads(restore)
        end

        # Compressed storage is SLOWER at the occupancy an ordinary 2-D fixture has. Measured
        # 1.43x-1.69x across runs; the bound is loose because this is a shared machine.
        @test low > 1.15
        # And it improves with occupancy. A within-run comparison, so load cancels -- this is the
        # assertion that would break if the crossover moved.
        @test high < low
    end
end
