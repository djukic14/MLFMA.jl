# The tangential vector-harmonic leaf collections and their six kernels.
#
# TWO LAYERS OF CHECK, and the second is the one that matters. The kernels are first compared against
# the block operator `B = [[𝒫, s𝒬], [𝒬, -s𝒫]]` written out explicitly from the transform -- that
# catches a wrong sign or a transposed block. But an algebraic check cannot tell you whether this
# backend computes the SAME OPERATOR the sampled path does, and that is exactly what went wrong when
# the SHE receive kernel shipped with a weighted sesquilinear convention that satisfied its own
# identity exactly and was 6.9e-2 wrong end to end. So the last testset compares VSH against
# `LazyReceiveFieldCollection` and `RadiationFieldCollection` on real RWG patterns.

using MLFMA
using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics:
    VectorSphericalHarmonicTransform, numvshcoefficients, vshmirrorblocks
using BEAST, CompScienceMeshes, H2Trees, H2Factory
using LinearAlgebra
using Random
using StaticArrays
using Test

# `[[𝒫, s𝒬], [𝒬, -s𝒫]]` as one dense matrix, written from the transform rather than from production
# code, so a kernel and its reference cannot share a mistake.
function vshblockoperator(t; mirrored::Bool=false)
    𝒫, 𝒬 = mirrored ? vshmirrorblocks(t) : (t.A, t.D)
    s = mirrored ? -1 : 1
    return [𝒫 (s*𝒬); 𝒬 (-s*𝒫)]
end

@testset "Vector spherical harmonic collections" begin
    P, nbf = 5, 7
    sampling = MLFMA.SphericalSampling{Float64}(2 * (P + 1), 4 * (P + 1))
    t = VectorSphericalHarmonicTransform(ComplexF64, sampling, P)
    n = numvshcoefficients(P)
    nd = size(t.A, 1)

    rng = MersenneTwister(17)
    C = randn(rng, ComplexF64, 2n, nbf)
    radiation = MLFMA.VectorSphericalHarmonicRadiationCollection(
        MLFMA.VectorSphericalHarmonicStorage(P, SVector(0.0, 0.0, 0.0), C)
    )
    # `I` is the EFIE coupling; the operator-dependent case is exercised against the sampled path
    # further down and end to end in `test/Maps/galerkin.jl`.
    receive = MLFMA.VectorSphericalHarmonicReceiveCollection(radiation, [1 0; 0 1])
    scratch = MLFMA.vshscratch(sampling, P)

    R = vshblockoperator(t) * C                    # (2nd × nbf)
    Rec = vshblockoperator(t; mirrored=true) * C

    @testset "accessors and the storage contract" begin
        @test MLFMA.fielddim(radiation) == 2
        @test MLFMA.fielddim(receive) == 2
        @test MLFMA.order(radiation) == P
        @test MLFMA.order(receive) == P
        @test MLFMA.numbasisfunctions(radiation) == nbf
        @test size(radiation) == (2n, nbf)
        @test MLFMA.radiationcollection(receive) === radiation
        # carries no data of its own -- the whole point of the mirror being diagonal
        @test Base.summarysize(receive) < Base.summarysize(radiation) + 200
        @test_throws DimensionMismatch MLFMA.VectorSphericalHarmonicStorage(
            P, SVector(0.0, 0.0, 0.0), randn(rng, ComplexF64, 2n + 1, nbf)
        )
        @test_throws ArgumentError MLFMA.VectorSphericalHarmonicStorage(
            0, SVector(0.0, 0.0, 0.0), zeros(ComplexF64, 0, nbf)
        )
    end

    moment() = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
    function randommoment()
        G = moment()
        G[1] .= reshape(randn(rng, ComplexF64, nd), size(G[1]))
        G[2] .= reshape(randn(rng, ComplexF64, nd), size(G[2]))
        return G
    end
    stacked(G) = vcat(vec(G[1]), vec(G[2]))

    @testset "the four plain kernels are the two block operators" begin
        for _ in 1:3
            x = randn(rng, ComplexF64, nbf)
            G = randommoment()

            m = moment()
            H2Factory.applyleafmul!(scratch, m, radiation, x)
            @test stacked(m) ≈ R * x rtol = 1e-11

            mr = moment()
            H2Factory.applyleafmul!(scratch, mr, receive, x)
            @test stacked(mr) ≈ Rec * x rtol = 1e-11

            # TRANSPOSE, not adjoint, and no quadrature weights -- see this file's header.
            y = zeros(ComplexF64, nbf)
            H2Factory.applyleafmul!(scratch, y, receive, G, true, false)
            @test y ≈ transpose(Rec) * stacked(G) rtol = 1e-11
            @test !isapprox(y, adjoint(Rec) * stacked(G); rtol=1e-6)   # non-vacuity

            yr = zeros(ComplexF64, nbf)
            H2Factory.applyleafmul!(scratch, yr, radiation, G, true, false)
            @test yr ≈ transpose(R) * stacked(G) rtol = 1e-11
        end
    end

    @testset "the two conjugated kernels are the adjoints" begin
        for _ in 1:3
            x = randn(rng, ComplexF64, nbf)
            G = randommoment()

            m = moment()
            MLFMA.applyconjugatedleafmul!(scratch, m, receive, x)
            @test stacked(m) ≈ conj.(Rec) * x rtol = 1e-11

            y = zeros(ComplexF64, nbf)
            MLFMA.applyconjugatedleafmul!(scratch, y, radiation, G, true, false)
            @test y ≈ adjoint(R) * stacked(G) rtol = 1e-11
        end
    end

    @testset "α and β follow mul!'s convention" begin
        G = randommoment()
        base = zeros(ComplexF64, nbf)
        H2Factory.applyleafmul!(scratch, base, receive, G, true, false)

        scaled = zeros(ComplexF64, nbf)
        H2Factory.applyleafmul!(scratch, scaled, receive, G, 2.5, false)
        @test scaled ≈ 2.5 .* base rtol = 1e-12

        preloaded = fill(7.0 + 0im, nbf)
        H2Factory.applyleafmul!(scratch, preloaded, receive, G, true, false)
        @test preloaded ≈ base rtol = 1e-12      # β = false OVERWRITES

        accumulated = copy(base)
        H2Factory.applyleafmul!(scratch, accumulated, receive, G, true, true)
        @test accumulated ≈ 2 .* base rtol = 1e-12
    end

    @testset "applying without a scratch is an error, in every arity" begin
        G = moment()
        for collection in (radiation, receive)
            @test_throws ErrorException H2Factory.applyleafmul!(
                H2Factory.EmptyChannels(), moment(), collection, randn(ComplexF64, nbf)
            )
            @test_throws ErrorException H2Factory.applyleafmul!(
                H2Factory.EmptyChannels(),
                zeros(ComplexF64, nbf),
                collection,
                G,
                true,
                false,
            )
            @test_throws ErrorException LinearAlgebra.mul!(
                moment(), collection, randn(ComplexF64, nbf)
            )
        end
    end
end

@testset "The batched tangential chunk multiply" begin
    # Checked against the PER-NODE path above, which is itself checked against the block operator and
    # (below) against the sampled collection. So the chain is asserted link by link rather than the
    # fast path being compared to a fresh restatement of the arithmetic.
    #
    # Reuses `FakeChunkTree`/`FakeChunkMap` from `sphericalharmoniccollection.jl` -- the traversal
    # context faked down to the three things `aggregateleafchunk!` actually touches.
    P = 4
    sampling = MLFMA.SphericalSampling{Float64}(2 * (P + 1), 4 * (P + 1))
    n = MLFMA.SphericalHarmonics.numvshcoefficients(P)
    rng = MersenneTwister(23)

    # RAGGED basis-function counts, interleaved with internal nodes, so the leaf filtering and the
    # sub-batching are exercised rather than assumed. `nleaves` straddles `SHEBATCHTHRESHOLD`.
    for nleaves in (4, MLFMA.SHEBATCHTHRESHOLD + 3)
        leafnodes = collect(2:2:(2nleaves))
        chunknodes = collect(1:(2nleaves))

        values = Dict{Int,Vector{Int}}()
        buffers = Dict{Int,Vector{ComplexF64}}()
        collections = Dict{Int,Any}()
        offset = 0
        for (i, node) in enumerate(leafnodes)
            nbf = 3 + i
            values[node] = collect((offset + 1):(offset + nbf))
            offset += nbf
            buffers[node] = Vector{ComplexF64}(undef, nbf)
            collections[node] = MLFMA.VectorSphericalHarmonicRadiationCollection(
                MLFMA.VectorSphericalHarmonicStorage(
                    P, SVector(0.0, 0.0, 0.0), randn(rng, ComplexF64, 2n, nbf)
                ),
            )
        end
        tree = FakeChunkTree(Set(leafnodes), values)
        A = FakeChunkMap(buffers)
        x = randn(rng, ComplexF64, offset)

        freshmoments() = Dict(
            node => MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling)) for
            node in leafnodes
        )

        perleaf = freshmoments()
        single = MLFMA.vshscratch(sampling, P)
        for node in leafnodes
            H2Factory.applyleafmul!(
                single, perleaf[node], collections[node], x[values[node]]
            )
        end

        batched = freshmoments()
        H2Factory.aggregateleafchunk!(
            A,
            tree,
            chunknodes,
            node -> batched[node],
            node -> collections[node],
            H2Factory.LinAlgebraMulFunctor(),
            x,
            1,
            MLFMA.vshscratch(sampling, P; ngroups=nleaves),
        )

        for node in leafnodes
            @test batched[node][1] ≈ perleaf[node][1] rtol = 1e-12
            @test batched[node][2] ≈ perleaf[node][2] rtol = 1e-12
        end

        # a scratch too small to hold the chunk must SUB-BATCH, not drop leaves
        subbatched = freshmoments()
        H2Factory.aggregateleafchunk!(
            A,
            tree,
            chunknodes,
            node -> subbatched[node],
            node -> collections[node],
            H2Factory.LinAlgebraMulFunctor(),
            x,
            1,
            MLFMA.vshscratch(sampling, P; ngroups=max(1, nleaves ÷ 2)),
        )
        for node in leafnodes
            @test subbatched[node][1] ≈ perleaf[node][1] rtol = 1e-12
        end
    end
end

@testset "Tangential collections agree with the sampled path" begin
    # THE TEST THAT MATTERS. Real RWG patterns, order chosen by the rule, compared against the
    # collections MLFMA builds today -- on BOTH sides, and in both arities of the receive side, since
    # a transposed matvec aggregates through it. See this file's header for why the algebraic checks
    # above are not sufficient on their own.
    f = shebuilderfixture(; protrusion=HTB.NoProtrusionCheck())
    leaves = MLFMA.leafids(f.leafsamplings)

    collections, orders = MLFMA.vshcollections(
        f.cartesianassembler, f.tree, f.leafsamplings, f.ϵ
    )
    sampled = MLFMA.RadiationFieldCollections(
        f.sampledassembler, f.tree, f.leafsamplings; verbose=false
    )

    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    @test length(collections) == length(leaves)
    @test Set(keys(collections)) == Set(leaves)
    @test length(orders) == length(f.samplings)

    @testset "the order is a choice, not a clamp, and it is BELOW the scalar rule's" begin
        for (sampling, P) in orders
            @test P >= 1
            @test P < MLFMA.vshanalysisorder(sampling)
        end
        # The measured claim this backend exists for: at the same tolerance and on the same
        # patterns, the tangential basis needs no more degrees than the Cartesian one.
        _, sheorders = MLFMA.shecollections(
            f.cartesianassembler, f.tree, f.leafsamplings, f.ϵ
        )
        for (sampling, P) in orders
            @test P <= sheorders[sampling]
        end
    end

    Random.seed!(63)
    worstradiation = 0.0
    worstreceive = 0.0
    worstaggregate = 0.0
    for (i, leaf) in enumerate(leaves)
        sampling = f.leafsamplings[i].sampling
        scratch = MLFMA.vshscratch(sampling, orders[sampling])
        nbf = length(HTB.values(f.tree, leaf))

        vshreceive = MLFMA.VectorSphericalHarmonicReceiveCollection(
            collections[leaf], MLFMA.receivecoupling(efie)
        )
        lazyreceive = MLFMA.LazyReceiveFieldCollection(
            efie, sampled[i], MLFMA.mirrorindices(sampling)
        )
        lazyscratch = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))

        # RADIATION
        x = randn(ComplexF64, nbf)
        viavsh = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        viasamples = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        H2Factory.applyleafmul!(scratch, viavsh, collections[leaf], x)
        LinearAlgebra.mul!(viasamples, sampled[i], x)
        reference = sqrt(sum(abs2, viasamples[1]) + sum(abs2, viasamples[2]))
        worstradiation = max(
            worstradiation,
            sqrt(
                sum(abs2, viavsh[1] .- viasamples[1]) +
                sum(abs2, viavsh[2] .- viasamples[2]),
            ) / reference,
        )

        # RECEIVE, both arities
        m = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        m[1] .= reshape(randn(ComplexF64, length(m[1])), size(m[1]))
        m[2] .= reshape(randn(ComplexF64, length(m[2])), size(m[2]))

        yvsh = zeros(ComplexF64, nbf)
        ylazy = zeros(ComplexF64, nbf)
        H2Factory.applyleafmul!(scratch, yvsh, vshreceive, m, true, false)
        H2Factory.applyleafmul!(lazyscratch, ylazy, lazyreceive, m, true, false)
        worstreceive = max(worstreceive, norm(yvsh - ylazy) / norm(ylazy))

        mvsh = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        mlazy = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        H2Factory.applyleafmul!(scratch, mvsh, vshreceive, x)
        H2Factory.applyleafmul!(lazyscratch, mlazy, lazyreceive, x)
        worstaggregate = max(
            worstaggregate,
            sqrt(sum(abs2, mvsh[1] .- mlazy[1]) + sum(abs2, mvsh[2] .- mlazy[2])) /
            sqrt(sum(abs2, mlazy[1]) + sum(abs2, mlazy[2])),
        )
    end

    @test worstradiation < f.ϵ
    @test worstreceive < f.ϵ
    @test worstaggregate < f.ϵ

    @testset "and they are smaller than what they replace" begin
        # A REPRESENTATION assertion: if this fails the order rule has stopped truncating, which no
        # accuracy test would catch.
        sampledbytes = sum(Base.summarysize(c) for c in sampled)
        vshbytes = sum(Base.summarysize(collections[leaf]) for leaf in leaves)
        shecollections, _ = MLFMA.shecollections(
            f.cartesianassembler, f.tree, f.leafsamplings, f.ϵ
        )
        shebytes = sum(Base.summarysize(shecollections[leaf]) for leaf in leaves)
        @test vshbytes < sampledbytes
        @test vshbytes < shebytes          # the point of a second compressed backend
    end
end

@testset "The tangential leaf-scratch channel" begin
    f = shebuilderfixture(; protrusion=HTB.NoProtrusionCheck())
    _, orders = MLFMA.vshcollections(f.cartesianassembler, f.tree, f.leafsamplings, f.ϵ)

    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    A = MLFMA.GalerkinMLFMA(
        efie,
        f.X;
        tree=f.tree,
        strategy=MLFMA.MLFMAStrategy(
            efie,
            f.X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=f.ϵ),
            nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
            ),
        ),
        verbose=false,
    )
    core = A.core

    H2Factory.withschedulechunks(A.schedules) do
        sampledpool = MLFMA.leafscratchchannel(
            complex(core.γ),
            MLFMA.farfieldtype(core.operator),
            MLFMA._leaflevels(f.tree),
            core.plans,
            core.samplings,
            core.minrelevantlevel,
        )
        pool = MLFMA.vshscratchchannel(
            orders,
            MLFMA._leaflevels(f.tree),
            core.plans,
            core.samplings,
            core.minrelevantlevel;
            groupsperchunk=level ->
                MLFMA.leavesperchunk(f.tree, level, H2Factory.nchunks(level)),
        )

        # a drop-in for the pool it replaces: same levels, same chunk counts
        @test keys(pool) == keys(sampledpool)
        @test !isempty(pool)

        for level in keys(pool)
            scratch = H2Factory.borrowbuffer(pool, level, 1, H2Factory.isForward())
            sampling = core.samplings[MLFMA._samplinglevelindex(
                level, core.minrelevantlevel, core.samplings
            )]
            P = orders[sampling]
            n = MLFMA.SphericalHarmonics.numvshcoefficients(P)
            nd = length(MLFMA.θs(sampling)) * length(MLFMA.φs(sampling))

            @test MLFMA.order(scratch) == P
            @test MLFMA.numdirections(scratch) == nd
            @test size(scratch.A) == (nd, n)
            @test size(scratch.D) == size(scratch.A)
            @test size(scratch.Aᵖ) == size(scratch.A)
            @test size(scratch.coefficients, 1) == 2n
            @test MLFMA.maxgroups(scratch) >= 1
            @test length(scratch.leaves) == MLFMA.maxgroups(scratch)
        end
    end
end

@testset "The receive coupling is the OPERATOR's, not just the mirror" begin
    # THE REGRESSION THIS ENCODES. Both compressed backends originally baked in
    # `mirrorrule(TangentialSphericalFarField)`, which EQUALS `receivecouplingrule(MWEFIE3D, ...)`.
    # For `MWDoubleLayer3D` the rule is a component SWAP with coefficient γ, and the mistake measured
    # 4.4e-01 against a dense reference where the sampled path gives 2.8e-05 -- silently, because
    # every array shape conformed. An EFIE-only check cannot see it.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    mfie = BEAST.MWDoubleLayer3D(im * κ)

    @testset "EFIE reduces to the identity" begin
        # Which is WHY the original code was right for EFIE, and why the identity branch in the
        # kernels keeps the measured path byte-for-byte what it was.
        @test MLFMA.receivecoupling(efie) ≈ [1 0; 0 1]
    end

    @testset "MFIE is the swap, with γ" begin
        g = MLFMA.γ(mfie)
        @test MLFMA.receivecoupling(mfie) ≈ [0 (-g); g 0]
        # non-vacuity: it is genuinely NOT the identity, so the assertion above discriminates
        @test !isapprox(MLFMA.receivecoupling(mfie), [1 0; 0 1])
    end

    @testset "an unrepresentable rule raises instead of guessing" begin
        # A three-component representation cannot be a 2x2 map on the tangential components. Raising
        # is the whole point: the failure mode being screened for is a plausible wrong answer.
        @test_throws ErrorException MLFMA._rulematrix((MLFMA.ComponentMap{3,3}(1),), efie)
    end
end
