# The compressed 2-D leaf collections: all six orientations, and the receive coupling.
#
# ------------------------------------------------------------------------------------------------
# THE LOAD-BEARING TESTSET IS "AGREES WITH THE SAMPLED REPRESENTATION", and it is written the way it
# is because of #129.
#
# There, SHE leaf storage shipped with a receive kernel that was SELF-CONSISTENT -- its aggregation
# and its scatter were exact transposes of each other, every shape conformed, and every test that
# compared the compressed path against ITSELF passed. It was still wrong by 44 % against a dense
# reference on MFIE, because the sampled receive side applies `receivecouplingrule(op, F)` and the
# compressed one had assumed the geometric `mirrorrule`.
#
# So no testset here compares the compressed path only against its own algebra. Every claim about the
# receive side is checked against the SAMPLED collections -- `ScalarRadiationFieldCollection` wrapped
# in `LazyReceiveFieldCollection`, the objects a production map actually uses -- on patterns assembled
# from a real BEAST operator. That is the contract: agreeing with the sampled path, not being
# internally coherent.
#
# ------------------------------------------------------------------------------------------------
# BOTH 2-D GALERKIN OPERATORS HAPPEN TO GIVE `K = I`, and that is exactly why the coupling is tested
# on a HAND-BUILT NON-IDENTITY rule as well. A coupling machine that is only ever exercised at the
# identity is a machine nothing has checked: #129's bug was invisible for the same reason, EFIE's
# coupling being the identity that made an EFIE-only check pass.

using MLFMA
using MLFMA.CircularHarmonics
using MLFMA.CircularHarmonics:
    CircularHarmonicStorage, CircularHarmonicTransform, analyze, numcoefficients
using BEAST, CompScienceMeshes, H2Factory, H2Trees
using LinearAlgebra
using Random
using StaticArrays

const HT = H2Trees

# One leaf's worth of real 2-D patterns, referenced to that leaf's own centre -- which is what makes
# the coefficients summable at all.
function leaffixture(; operator, radius=1.0, h=1 / 16, minhalfsize=0.25, ϵ=1e-5)
    κ = 2π
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
    F = MLFMA.farfieldtype(operator)
    assembler = MLFMA.BEASTFarFieldAssembler(
        MLFMA.radiationoperator(operator),
        X,
        BEAST.SingleNumQStrat(3),
        MLFMA.farfieldstorerule(operator),
        F,
    )
    boxes = [
        MLFMA.BoxFarFieldSampling(
            sampling,
            first(MLFMA.cartesiansamplepoints(sampling)),
            HT.center(tree, leaf),
            leaf,
        ) for leaf in leaves
    ]
    patterns = assembler(tree, boxes; verbose=false)
    return (; X, tree, leaves, sampling, patterns, operator, F, κ)
end

# Compress one leaf. Analysed on the leaf's own grid -- `checalibration.jl` measured oversampling to
# be worth nothing here.
function compress(f, i, M=MLFMA.cutoffparameter(f.sampling))
    P = f.patterns[i]
    D = length(P)
    t = CircularHarmonicTransform(ComplexF64, f.sampling, M)
    G = Matrix{ComplexF64}(undef, length(f.sampling), D)
    for d in 1:D
        @views G[:, d] .= P[d][:, 1]
    end
    nbf = size(P[1], 2)
    coeff = Matrix{ComplexF64}(undef, numcoefficients(M) * D, nbf)
    for j in 1:nbf
        for d in 1:D
            @views G[:, d] .= P[d][:, j]
        end
        @views coeff[:, j] .= vec(analyze(t, G))
    end
    storage = CircularHarmonicStorage(M, SVector{2}(HT.center(f.tree, f.leaves[i])), coeff)
    return t, MLFMA.CircularHarmonicRadiationCollection(storage)
end

# The sampled collection for the same leaf, built from the same assembled patterns.
sampledcollection(f, i) =
    MLFMA.radiationcollectiontype(f.F)(SVector(Tuple(copy(m) for m in f.patterns[i])))

newmoment(f) =
    if f.F === MLFMA.ScalarFarField
        MLFMA.ScalarFarField(ComplexF64, f.sampling)
    else
        MLFMA.PlanarVectorScalarFarField(ComplexF64, f.sampling)
    end

# `fielddim`, not `length`: a far field has no `length` method, deliberately -- it is a bundle of
# component arrays, not a container of them.
momentmatrix(m) =
    hcat((collect(H2Factory.wrapvector(m, d)) for d in 1:MLFMA.fielddim(typeof(m)))...)

@testset "Circular harmonic collections" begin
    singlelayer = Helmholtz2D.singlelayer(; wavenumber=2π)
    hypersingular = Helmholtz2D.hypersingular(; wavenumber=2π)

    @testset "D comes from the storage, not from a hardcoded 2" begin
        for (op, D) in ((singlelayer, 1), (hypersingular, 3))
            f = leaffixture(; operator=op)
            _, c = compress(f, 1)
            @test MLFMA.fielddim(typeof(c)) == D
            @test MLFMA.fielddim(c) == D
            @test length(f.patterns[1]) == D
            @test size(c) ==
                (numcoefficients(MLFMA.order(c)) * D, MLFMA.numbasisfunctions(c))
        end
    end

    @testset "forward aggregation reproduces the sampled leaf multiply" begin
        for op in (singlelayer, hypersingular)
            f = leaffixture(; operator=op)
            t, che = compress(f, 1)
            D = MLFMA.fielddim(typeof(che))
            scratch = MLFMA.CircularHarmonicLeafScratch(t, D)

            x = randn(MersenneTwister(1), ComplexF64, MLFMA.numbasisfunctions(che))
            compressed = newmoment(f)
            H2Factory.applyleafmul!(scratch, compressed, che, x)

            sampled = newmoment(f)
            LinearAlgebra.mul!(sampled, sampledcollection(f, 1), x)

            # The order kept is the grid's full order, so this is representation change WITHOUT
            # truncation -- any discrepancy is the machinery, not the compression.
            @test momentmatrix(compressed) ≈ momentmatrix(sampled) atol = 1e-10
        end
    end

    @testset "no scratch is an explanatory error, not a silent allocation" begin
        f = leaffixture(; operator=singlelayer)
        _, che = compress(f, 1)
        x = randn(ComplexF64, MLFMA.numbasisfunctions(che))
        moment = newmoment(f)
        err = try
            H2Factory.applyleafmul!(nothing, moment, che, x)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("CircularHarmonicLeafScratch", err.msg)
        @test_throws ErrorException LinearAlgebra.mul!(moment, che, x)
    end

    # ============================== THE #129 CHECK ==============================
    @testset "the receive side agrees with the SAMPLED receive side" begin
        for op in (singlelayer, hypersingular)
            f = leaffixture(; operator=op)
            t, che = compress(f, 1)
            D = MLFMA.fielddim(typeof(che))
            scratch = MLFMA.CircularHarmonicLeafScratch(t, D)
            cherec = MLFMA.CircularHarmonicReceiveCollection(che, op, f.F)

            x = randn(MersenneTwister(2), ComplexF64, MLFMA.numbasisfunctions(che))

            # SAMPLED: the objects a production Galerkin map uses.
            lazy = MLFMA.LazyReceiveFieldCollection(
                op, sampledcollection(f, 1), MLFMA.mirrorindices(f.sampling)
            )
            sampled = newmoment(f)
            H2Factory.applyleafmul!(newmoment(f), sampled, lazy, x)

            # COMPRESSED: mirrored synthesis plus one component map.
            compressed = newmoment(f)
            H2Factory.applyleafmul!(scratch, compressed, cherec, x)

            @test momentmatrix(compressed) ≈ momentmatrix(sampled) atol = 1e-10

            # And the scatter, which is the transpose of the above -- checked against the sampled
            # scatter rather than against the compressed aggregation, for the reason in this file's
            # header.
            moment = newmoment(f)
            for d in 1:D
                H2Factory.wrapvector(moment, d) .= randn(
                    MersenneTwister(3 + d), ComplexF64, length(f.sampling)
                )
            end
            ysampled = zeros(ComplexF64, MLFMA.numbasisfunctions(che))
            H2Factory.applyleafmul!(newmoment(f), ysampled, lazy, moment, 1.0, false)
            ycompressed = zeros(ComplexF64, MLFMA.numbasisfunctions(che))
            H2Factory.applyleafmul!(scratch, ycompressed, cherec, moment, 1.0, false)
            @test ycompressed ≈ ysampled atol = 1e-10
        end
    end

    @testset "an operator with no Galerkin receive rule refuses to build a coupling" begin
        # The 2-D double layers are Petrov-only (#130): their patterns carry `k̂·n̂`, which flips under
        # the antipodal map, so there is no rule and guessing one is precisely the #129 failure.
        for op in (
            Helmholtz2D.doublelayer(; wavenumber=2π),
            Helmholtz2D.doublelayer_transposed(; wavenumber=2π),
        )
            @test_throws Exception MLFMA.circularreceivecoupling(op, MLFMA.ScalarFarField)
        end
    end

    @testset "the coupling machine works at a NON-identity K" begin
        # Both real 2-D Galerkin operators give K = I, so without this the whole coupling path is
        # untested. A hand-built swap-with-different-coefficients rule, unrelated to any operator,
        # exercises `_coupledgathermoment!` and `_applycircularcoupling!` where the identity path
        # cannot.
        f = leaffixture(; operator=hypersingular)
        t, che = compress(f, 1)
        scratch = MLFMA.CircularHarmonicLeafScratch(t, 3)
        K = ComplexF64[0 2 0; 0 0 3im; 5 0 0]
        cherec = MLFMA.CircularHarmonicReceiveCollection(che, K)
        @test MLFMA.coupling(cherec) == K

        x = randn(MersenneTwister(4), ComplexF64, MLFMA.numbasisfunctions(che))
        plain = MLFMA.CircularHarmonicReceiveCollection(che, Matrix{ComplexF64}(I, 3, 3))

        withK = newmoment(f)
        H2Factory.applyleafmul!(scratch, withK, cherec, x)
        withoutK = newmoment(f)
        H2Factory.applyleafmul!(scratch, withoutK, plain, x)

        # `moment_K = K * moment_I`, componentwise across the three components.
        @test momentmatrix(withK) ≈ momentmatrix(withoutK) * transpose(K) atol = 1e-10

        # The scatter must be the exact transpose of that aggregation, since the two are what the
        # forward and reverse traversals apply to the SAME collection.
        moment = newmoment(f)
        for d in 1:3
            H2Factory.wrapvector(moment, d) .= randn(
                MersenneTwister(10 + d), ComplexF64, length(f.sampling)
            )
        end
        y = zeros(ComplexF64, MLFMA.numbasisfunctions(che))
        H2Factory.applyleafmul!(scratch, y, cherec, moment, 1.0, false)
        # ⟨Rec x, m⟩ == ⟨x, Recᵀ m⟩ without conjugation anywhere -- a transpose, not an adjoint.
        lhs = sum(momentmatrix(withK) .* momentmatrix(moment))
        @test lhs ≈ transpose(x) * y atol = 1e-9

        @test_throws DimensionMismatch MLFMA.CircularHarmonicReceiveCollection(
            che, ComplexF64[1 0; 0 1]
        )
    end

    @testset "all four transpose/adjoint orientations match their matrices" begin
        f = leaffixture(; operator=singlelayer)
        t, che = compress(f, 1)
        scratch = MLFMA.CircularHarmonicLeafScratch(t, 1)
        cherec = MLFMA.CircularHarmonicReceiveCollection(che, singlelayer, f.F)
        nbf = MLFMA.numbasisfunctions(che)

        # The dense matrices the collections stand for: R = S C and Rec = K M C, with K = I here.
        C = MLFMA.CircularHarmonics.coefficientmatrix(MLFMA.storage(che))
        R = scratch.synthesis * C
        Rec = scratch.mirror * C

        rng = MersenneTwister(21)
        x = randn(rng, ComplexF64, nbf)
        moment = newmoment(f)
        H2Factory.wrapvector(moment, 1) .= randn(rng, ComplexF64, length(f.sampling))
        m = collect(H2Factory.wrapvector(moment, 1))

        # forward aggregate: R x
        fwd = newmoment(f)
        H2Factory.applyleafmul!(scratch, fwd, che, x)
        @test collect(H2Factory.wrapvector(fwd, 1)) ≈ R * x atol = 1e-10

        # transpose aggregate: Rec x
        tra = newmoment(f)
        H2Factory.applyleafmul!(scratch, tra, cherec, x)
        @test collect(H2Factory.wrapvector(tra, 1)) ≈ Rec * x atol = 1e-10

        # transpose scatter on the trial side: α Rᵀ m + β y
        y = ones(ComplexF64, nbf)
        H2Factory.applyleafmul!(scratch, y, che, moment, 2.0, 3.0)
        @test y ≈ 2.0 * (transpose(R) * m) .+ 3.0 atol = 1e-9

        # forward scatter on the test side: α Recᵀ m
        y2 = zeros(ComplexF64, nbf)
        H2Factory.applyleafmul!(scratch, y2, cherec, moment, 1.0, false)
        @test y2 ≈ transpose(Rec) * m atol = 1e-10

        # adjoint aggregate: conj(Rec) x
        adj = newmoment(f)
        MLFMA.applyconjugatedleafmul!(scratch, adj, cherec, x)
        @test collect(H2Factory.wrapvector(adj, 1)) ≈ conj(Rec) * x atol = 1e-10

        # adjoint scatter: α Rᴴ m
        y3 = zeros(ComplexF64, nbf)
        MLFMA.applyconjugatedleafmul!(scratch, y3, che, moment, 1.0, false)
        @test y3 ≈ R' * m atol = 1e-10
    end

    @testset "the adjoint aggregation handles a real x without conjugating it" begin
        f = leaffixture(; operator=singlelayer)
        t, che = compress(f, 1)
        scratch = MLFMA.CircularHarmonicLeafScratch(t, 1)
        cherec = MLFMA.CircularHarmonicReceiveCollection(che, singlelayer, f.F)
        C = MLFMA.CircularHarmonics.coefficientmatrix(MLFMA.storage(che))
        Rec = scratch.mirror * C
        x = randn(MersenneTwister(31), Float64, MLFMA.numbasisfunctions(che))
        adj = newmoment(f)
        MLFMA.applyconjugatedleafmul!(scratch, adj, cherec, x)
        @test collect(H2Factory.wrapvector(adj, 1)) ≈ conj(Rec) * x atol = 1e-10
    end

    @testset "the batched chunk multiply equals the per-leaf one" begin
        # Batching is MANDATORY here (0.28x-0.72x, see the collection file's header), so it must be
        # bit-comparable to the path it replaces or it is buying speed with silence.
        f = leaffixture(; operator=singlelayer)
        M = MLFMA.cutoffparameter(f.sampling)
        ngroups = min(6, length(f.leaves))
        t, _ = compress(f, 1)
        collections = [compress(f, i, M)[2] for i in 1:ngroups]
        scratch = MLFMA.CircularHarmonicLeafScratch(t, 1; ngroups=ngroups)
        @test MLFMA.maxgroups(scratch) == ngroups
        @test MLFMA.ncomponents(scratch) == 1
        @test MLFMA.numdirections(scratch) == length(f.sampling)

        rng = MersenneTwister(41)
        xs = [randn(rng, ComplexF64, MLFMA.numbasisfunctions(c)) for c in collections]

        perleaf = [newmoment(f) for _ in 1:ngroups]
        for g in 1:ngroups
            H2Factory.applyleafmul!(scratch, perleaf[g], collections[g], xs[g])
        end

        # Drive `_flushcircularbatch!` directly: the full `aggregateleafchunk!` needs a live traversal
        # buffer, and what is under test is the batch arithmetic, not H2Factory's dispatch.
        batched = [newmoment(f) for _ in 1:ngroups]
        for g in 1:ngroups
            @views MLFMA.CircularHarmonics.aggregate!(
                vec(scratch.coefficients[:, g:g]), MLFMA.storage(collections[g]), xs[g]
            )
            scratch.leaves[g] = g
        end
        MLFMA._flushcircularbatch!(g -> batched[g], scratch, ngroups)

        for g in 1:ngroups
            @test momentmatrix(batched[g]) ≈ momentmatrix(perleaf[g]) atol = 1e-12
        end
    end

    @testset "a ragged batch is refused, because one S serves the whole level" begin
        f = leaffixture(; operator=singlelayer)
        M = MLFMA.cutoffparameter(f.sampling)
        t, _ = compress(f, 1)
        scratch = MLFMA.CircularHarmonicLeafScratch(t, 1; ngroups=2)
        coarser = compress(f, 2, M - 2)[2]
        @test MLFMA.order(coarser) == M - 2
        @test_throws DimensionMismatch MLFMA._gathercircularcoefficients!(
            nothing, f.tree, f.leaves[2], node -> coarser, ComplexF64[], scratch, 1
        )
    end

    @testset "mirrorsynthesis is the antipodal map, checked against the sampled mirror" begin
        M = 6
        s = MLFMA.CircularSampling(MLFMA.NφCircular(M))
        t = CircularHarmonicTransform(ComplexF64, s, M)
        mirror = MLFMA.CircularHarmonics.mirrorsynthesis(t)
        idx = MLFMA.mirrorindices(s)
        # Synthesising with the mirrored matrix must equal synthesising normally and then permuting
        # the samples by MLFMA's OWN mirror indices -- not by an independently reasoned permutation.
        a = randn(MersenneTwister(51), ComplexF64, numcoefficients(M))
        @test mirror * a ≈ (t.synthesis * a)[idx] atol = 1e-12
        @test MLFMA.CircularHarmonics.modeparity(M) == [iseven(m) ? 1 : -1 for m in (-M):M]
    end
end
