# The spherical-harmonic leaf collection, against the sampled collection it replaces.
#
# The whole point is that these two are INTERCHANGEABLE from the traversal's side: given the same
# patterns, `applyleafmul!` through coefficients must produce the same moment as `mul!` through
# samples. So the tests build one set of patterns and both collections from it, rather than checking
# the coefficient path against an independent restatement of what it should do.
#
# Patterns are band-limited to `P` BY CONSTRUCTION (synthesised from random coefficients) so that the
# comparison isolates the collection. Truncation error is a separate question, measured on real RWG
# patterns in `benchmark/interpolation/shepatterns.jl`, and folding it in here would only make this
# test approximate for no extra signal.

using MLFMA
using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics:
    SphericalHarmonicStorage, SphericalHarmonicTransform, numcoefficients, sphericalframe
using H2Factory
using H2Trees
using LinearAlgebra
using OhMyThreads
using Random
using StaticArrays
using Test

# Random Cartesian patterns of exact degree `P`, returned both as coefficients (what the SHE
# collection stores) and as the tangential component matrices the sampled collection stores.
function patternfixture(sampling, P, nbf; seed=0)
    rng = MersenneTwister(seed)
    t = SphericalHarmonicTransform(ComplexF64, sampling, P)
    nc = numcoefficients(P)
    nd = size(t.synthesis, 1)
    θ̂, φ̂ = sphericalframe(sampling)

    coefficients = Matrix{ComplexF64}(undef, 3nc, nbf)
    Fθ = Matrix{ComplexF64}(undef, nd, nbf)
    Fφ = Matrix{ComplexF64}(undef, nd, nbf)
    for j in 1:nbf
        a = randn(rng, ComplexF64, nc, 3)
        coefficients[:, j] .= vec(a)
        C = t.synthesis * a
        for q in 1:nd
            Fθ[q, j] = θ̂[q, 1] * C[q, 1] + θ̂[q, 2] * C[q, 2] + θ̂[q, 3] * C[q, 3]
            Fφ[q, j] = φ̂[q, 1] * C[q, 1] + φ̂[q, 2] * C[q, 2] + φ̂[q, 3] * C[q, 3]
        end
    end
    return t, coefficients, Fθ, Fφ
end

@testset "Spherical harmonic radiation collection" begin
    P, nbf = 6, 11
    sampling = MLFMA.SphericalSampling{Float64}(2 * (P + 1), 4 * (P + 1))
    t, coefficients, Fθ, Fφ = patternfixture(sampling, P, nbf; seed=1)

    she = MLFMA.SphericalHarmonicRadiationCollection(
        SphericalHarmonicStorage(P, SVector(0.0, 0.0, 0.0), coefficients)
    )
    sampled = MLFMA.TangentialSphericalRadiationFieldCollection(SVector(Fθ, Fφ))
    scratch = MLFMA.SphericalHarmonicLeafScratch(t, sampling)

    @testset "accessors" begin
        @test MLFMA.fielddim(she) == 2          # it PRODUCES a tangential moment
        @test MLFMA.order(she) == P
        @test MLFMA.numbasisfunctions(she) == nbf
        @test size(she) == (3 * numcoefficients(P), nbf)
        @test MLFMA.maxgroups(scratch) == 1
        @test MLFMA.numdirections(scratch) ==
            length(MLFMA.θs(sampling)) * length(MLFMA.φs(sampling))
    end

    @testset "the coefficient path reproduces the sampled leaf multiply" begin
        # THE TEST THIS FILE EXISTS FOR.
        Random.seed!(31)
        for _ in 1:5
            x = randn(ComplexF64, nbf)
            viacoefficients = MLFMA.TangentialSphericalFarField{2,ComplexF64}(
                size(sampling)
            )
            viasamples = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))

            H2Factory.applyleafmul!(scratch, viacoefficients, she, x)
            LinearAlgebra.mul!(viasamples, sampled, x)

            @test vec(viacoefficients[1]) ≈ vec(viasamples[1]) rtol = 1e-11
            @test vec(viacoefficients[2]) ≈ vec(viasamples[2]) rtol = 1e-11
        end
    end

    @testset "it is linear, and repeat application does not accumulate" begin
        Random.seed!(32)
        x1 = randn(ComplexF64, nbf)
        x2 = randn(ComplexF64, nbf)
        m1 = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        m2 = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        msum = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))

        H2Factory.applyleafmul!(scratch, m1, she, x1)
        first = copy(m1[1])
        H2Factory.applyleafmul!(scratch, m1, she, x1)
        @test m1[1] == first                     # OVERWRITES, not accumulates

        H2Factory.applyleafmul!(scratch, m2, she, x2)
        H2Factory.applyleafmul!(scratch, msum, she, x1 .+ x2)
        @test vec(msum[1]) ≈ vec(m1[1]) .+ vec(m2[1]) rtol = 1e-11
        @test vec(msum[2]) ≈ vec(m1[2]) .+ vec(m2[2]) rtol = 1e-11
    end

    @testset "applying without a scratch is an error, not an allocation" begin
        # The failure mode this guards is silent: falling through to a generic `mul!` that allocates
        # the (NΩ × 3) intermediate per matvec would work, and would give back a fifth of the memory
        # the collection exists to save.
        moment = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        x = ones(ComplexF64, nbf)
        @test_throws ErrorException H2Factory.applyleafmul!(
            H2Factory.EmptyChannels(), moment, she, x
        )
        @test_throws ErrorException LinearAlgebra.mul!(moment, she, x)
    end

    @testset "construction rejects non-Cartesian storage" begin
        # Two components would be a TANGENTIAL expansion, whose harmonic content is the frame's
        # rather than the pattern's, the one thing the representation must not be.
        tangentialstorage = SphericalHarmonicStorage(
            P, SVector(0.0, 0.0, 0.0), randn(ComplexF64, 2 * numcoefficients(P), 3)
        )
        @test_throws ArgumentError MLFMA.SphericalHarmonicRadiationCollection(
            tangentialstorage
        )
    end

    @testset "scratch sizing errors are caught" begin
        @test_throws ArgumentError MLFMA.SphericalHarmonicLeafScratch(
            t, sampling; ngroups=0
        )
        # a transform built on a DIFFERENT grid than the sampling handed alongside it
        other = MLFMA.SphericalSampling{Float64}(4, 8)
        @test_throws DimensionMismatch MLFMA.SphericalHarmonicLeafScratch(t, other)
    end
end

# --- the batched chunk multiply -------------------------------------------------------------------
#
# Checked against the PER-NODE path above, which is itself checked against the sampled collection. So
# the chain is: samples -> coefficients (per node) -> coefficients (batched), each link asserted
# rather than the fast path being compared to a fresh restatement of the arithmetic.
#
# The traversal context is faked down to what `aggregateleafchunk!` actually touches, `isleaf`,
# `values`, and the per-node vector buffer. A real tree and map would exercise the same three through
# far more machinery, and the map-level check belongs in the end-to-end benchmarks, not here.

struct FakeChunkTree
    leaves::Set{Int}
    values::Dict{Int,Vector{Int}}
end
H2Trees.isleaf(t::FakeChunkTree, node::Int) = node in t.leaves
H2Trees.values(t::FakeChunkTree, node::Int) = t.values[node]

# CARRIES A SCHEDULER, because `aggregateleafchunk!` now resolves the batched projection's
# parallelism from the map through `H2Factory.LeafMulOp` rather than spawning unconditionally. A real
# map answers that from its `ExecutionScheduling`; this fake answers from a field, which is what lets
# the testsets below run the SAME batch under a serial and a parallel scheduler and require identical
# output, the projection writes one moment per group, so any aliasing between groups shows up
# there and nowhere else.
struct FakeChunkMap{S}
    buffers::Dict{Int,Vector{ComplexF64}}
    scheduler::S
end
FakeChunkMap(buffers) = FakeChunkMap(buffers, OhMyThreads.SerialScheduler())
H2Factory.aggregationvectorbuffer(A::FakeChunkMap, node::Int) = A.buffers[node]
# Answers for EVERY operation, not just `LeafMulOp`: the fake models a map with one uniform inner
# scheduler, and pinning it to a single tag would silently stop covering the kernel if the traversal
# ever asked under a different one.
H2Factory.innerscheduler(A::FakeChunkMap, ::H2Factory.AbstractH2Operation, ::Int) =
    A.scheduler

@testset "Spherical harmonic batched chunk multiply" begin
    P = 5
    sampling = MLFMA.SphericalSampling{Float64}(2 * (P + 1), 4 * (P + 1))
    t = SphericalHarmonicTransform(ComplexF64, sampling, P)
    nc = numcoefficients(P)

    # A chunk of leaves with RAGGED basis-function counts, interleaved with internal nodes so the
    # method's own leaf filtering is exercised rather than assumed.
    nleaves = 20
    leafnodes = collect(2:2:(2nleaves))            # even ids are leaves
    chunknodes = collect(1:(2nleaves))             # odd ids are internal
    rng = MersenneTwister(91)

    values = Dict{Int,Vector{Int}}()
    buffers = Dict{Int,Vector{ComplexF64}}()
    collections = Dict{Int,Any}()
    offset = 0
    for (i, node) in enumerate(leafnodes)
        nbf = 3 + i                                 # ragged on purpose
        values[node] = collect((offset + 1):(offset + nbf))
        offset += nbf
        buffers[node] = Vector{ComplexF64}(undef, nbf)
        collections[node] = MLFMA.SphericalHarmonicRadiationCollection(
            SphericalHarmonicStorage(
                P, SVector(0.0, 0.0, 0.0), randn(rng, ComplexF64, 3nc, nbf)
            ),
        )
    end
    tree = FakeChunkTree(Set(leafnodes), values)
    A = FakeChunkMap(buffers)
    x = randn(rng, ComplexF64, offset)

    momentof(dict) = node -> dict[node]
    freshmoments() = Dict(
        node => MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling)) for
        node in leafnodes
    )

    # reference: the per-node path, one leaf at a time
    reference = freshmoments()
    pernodescratch = MLFMA.SphericalHarmonicLeafScratch(t, sampling; ngroups=1)
    for node in leafnodes
        buffers[node] .= view(x, values[node])
        H2Factory.applyleafmul!(
            pernodescratch, reference[node], collections[node], buffers[node]
        )
    end

    @testset "one batch covering the whole chunk" begin
        moments = freshmoments()
        scratch = MLFMA.SphericalHarmonicLeafScratch(t, sampling; ngroups=nleaves)
        H2Factory.aggregateleafchunk!(
            A,
            tree,
            chunknodes,
            momentof(moments),
            momentof(collections),
            H2Factory.LinAlgebraMulFunctor(),
            x,
            3,
            scratch,
        )
        for node in leafnodes
            @test vec(moments[node][1]) ≈ vec(reference[node][1]) rtol = 1e-11
            @test vec(moments[node][2]) ≈ vec(reference[node][2]) rtol = 1e-11
        end
    end

    @testset "the projection's scheduler changes nothing but the timing" begin
        # THE ASSERTION THE SCHEDULER SEAM NEEDS. The projection used to spawn `nthreads()` tasks
        # unconditionally; it now runs under whatever `H2Factory.LeafMulOp` resolves to, so serial and
        # parallel are two code paths through one kernel and they must agree EXACTLY, not to a
        # tolerance. Each group writes its own leaf's moment, so there is nothing to accumulate and no
        # summation order to differ; any disagreement at all would mean two groups touching one
        # destination, which is a race and not a rounding difference.
        #
        # `nthreads() == 1` makes `DynamicScheduler` degenerate, so this only bites on a threaded
        # gate. It still pins the contract, and it is the only place either scheduler is exercised
        # against the other at all.
        parallelmap = FakeChunkMap(buffers, OhMyThreads.DynamicScheduler())
        for ngroups in (nleaves, 7)
            serialmoments = freshmoments()
            parallelmoments = freshmoments()
            for (map, moments) in ((A, serialmoments), (parallelmap, parallelmoments))
                H2Factory.aggregateleafchunk!(
                    map,
                    tree,
                    chunknodes,
                    momentof(moments),
                    momentof(collections),
                    H2Factory.LinAlgebraMulFunctor(),
                    x,
                    3,
                    MLFMA.SphericalHarmonicLeafScratch(t, sampling; ngroups=ngroups),
                )
            end
            for node in leafnodes
                @test vec(parallelmoments[node][1]) == vec(serialmoments[node][1])
                @test vec(parallelmoments[node][2]) == vec(serialmoments[node][2])
            end
        end
    end

    @testset "sub-batching when the chunk outruns the scratch" begin
        # 20 leaves through capacity 7 is three flushes, the last partial, the case where a
        # mis-sized final GEMM or a stale `leaves` entry would show up.
        moments = freshmoments()
        scratch = MLFMA.SphericalHarmonicLeafScratch(t, sampling; ngroups=7)
        H2Factory.aggregateleafchunk!(
            A,
            tree,
            chunknodes,
            momentof(moments),
            momentof(collections),
            H2Factory.LinAlgebraMulFunctor(),
            x,
            3,
            scratch,
        )
        for node in leafnodes
            @test vec(moments[node][1]) ≈ vec(reference[node][1]) rtol = 1e-11
            @test vec(moments[node][2]) ≈ vec(reference[node][2]) rtol = 1e-11
        end
    end

    @testset "below the threshold it falls back, and still agrees" begin
        # Thin levels of a non-uniform tree land here (#123 measured levels holding 1 and 15
        # leaves), so this path is exercised in production, not just in this test.
        few = leafnodes[1:4]
        @test length(few) < MLFMA.SHEBATCHTHRESHOLD
        moments = freshmoments()
        scratch = MLFMA.SphericalHarmonicLeafScratch(t, sampling; ngroups=1)
        H2Factory.aggregateleafchunk!(
            A,
            tree,
            few,
            momentof(moments),
            momentof(collections),
            H2Factory.LinAlgebraMulFunctor(),
            x,
            3,
            scratch,
        )
        for node in few
            @test vec(moments[node][1]) ≈ vec(reference[node][1]) rtol = 1e-11
            @test vec(moments[node][2]) ≈ vec(reference[node][2]) rtol = 1e-11
        end
    end

    @testset "a ragged ORDER across the batch is rejected" begin
        # One P per level is required, a batch with mixed orders has no single synthesis matrix,
        # and the failure would otherwise be a wrong-sized write rather than an error.
        moments = freshmoments()
        mixed = copy(collections)
        mixed[leafnodes[2]] = MLFMA.SphericalHarmonicRadiationCollection(
            SphericalHarmonicStorage(
                P + 1,
                SVector(0.0, 0.0, 0.0),
                randn(
                    rng,
                    ComplexF64,
                    3 * numcoefficients(P + 1),
                    length(values[leafnodes[2]]),
                ),
            ),
        )
        scratch = MLFMA.SphericalHarmonicLeafScratch(t, sampling; ngroups=nleaves)
        @test_throws DimensionMismatch H2Factory.aggregateleafchunk!(
            A,
            tree,
            chunknodes,
            momentof(moments),
            momentof(mixed),
            H2Factory.LinAlgebraMulFunctor(),
            x,
            3,
            scratch,
        )
    end
end

# --- the receive side -----------------------------------------------------------------------------
#
# The receive collection IS `Rec = P S⁻ C`, the radiation collection with the parity-folded synthesis
# in place of `S`, and the scatter is its BILINEAR TRANSPOSE, `y = α Recᵀ G`. Both halves of that
# sentence are load-bearing and both were once wrong here:
#
#   * TRANSPOSE, not adjoint. MLFMA's sampled leaf scatter is `mul!(y, radiationcollection, mirrored)`,
#     which transposes; the conjugation an adjoint matvec needs is applied separately, by
#     `ConjugatedCollectionMul`. Asserting the sesquilinear pairing here passes for a kernel that
#     conjugates `C` one time too many.
#   * NO QUADRATURE WEIGHTS. This is a moment collection, not a testing functional, the sphere's
#     weights live in the translation operator. An earlier version of this test asserted the WEIGHTED
#     pairing ⟨F,G⟩_W = ⟨x,y⟩, which the weighted kernel satisfied exactly while being 6.9e-2 wrong
#     against a dense reference end to end.
#
# So the property asserted is the transpose identity with no weights anywhere, which holds on ANY grid
# and therefore needs no quadrature accuracy of its own.
#
# The forward half of the pairing is written out here from S⁻ and the frame DIRECTLY, rather than
# obtained by calling production code with a doctored scratch. A transpose test whose two sides share
# an implementation cannot fail for the reason it exists to catch.

@testset "Spherical harmonic receive collection" begin
    P, nbf = 5, 8
    sampling = MLFMA.SphericalSampling{Float64}(2 * (P + 1), 4 * (P + 1))
    t = SphericalHarmonicTransform(ComplexF64, sampling, P)
    nc = numcoefficients(P)
    nd = size(t.synthesis, 1)
    θ̂, φ̂ = sphericalframe(sampling)

    rng = MersenneTwister(51)
    C = randn(rng, ComplexF64, 3nc, nbf)
    radiation = MLFMA.SphericalHarmonicRadiationCollection(
        SphericalHarmonicStorage(P, SVector(0.0, 0.0, 0.0), C)
    )
    # `I` is the EFIE coupling. See `MLFMA.receivecoupling`. The operator-dependent case is
    # covered against the sampled path in `shecollectionbuilder.jl` and end to end in `Maps/`.
    receive = MLFMA.SphericalHarmonicReceiveCollection(radiation, [1 0; 0 1])
    scratch = MLFMA.SphericalHarmonicLeafScratch(t, sampling)
    S⁻ = MLFMA.SphericalHarmonics.mirrorsynthesis(t)

    @testset "accessors" begin
        @test MLFMA.fielddim(receive) == 2
        @test MLFMA.order(receive) == P
        @test MLFMA.numbasisfunctions(receive) == nbf
        @test MLFMA.radiationcollection(receive) === radiation
        @test Base.summarysize(receive) < Base.summarysize(radiation) + 200  # carries no data
    end

    # `F = P S⁻ C x`, written from the pieces rather than by calling the production aggregation.
    function mirroredforward(x)
        cart = S⁻ * reshape(C * x, nc, 3)
        Fθ = [
            θ̂[q, 1] * cart[q, 1] + θ̂[q, 2] * cart[q, 2] + θ̂[q, 3] * cart[q, 3] for q in 1:nd
        ]
        Fφ = [
            φ̂[q, 1] * cart[q, 1] + φ̂[q, 2] * cart[q, 2] + φ̂[q, 3] * cart[q, 3] for q in 1:nd
        ]
        return Fθ, Fφ
    end

    function randommoment()
        G = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        G[1] .= reshape(randn(rng, ComplexF64, nd), size(G[1]))
        G[2] .= reshape(randn(rng, ComplexF64, nd), size(G[2]))
        return G
    end

    @testset "the scatter is the BILINEAR transpose of aggregate-then-mirror" begin
        for _ in 1:5
            x = randn(rng, ComplexF64, nbf)
            G = randommoment()
            Fθ, Fφ = mirroredforward(x)

            y = zeros(ComplexF64, nbf)
            H2Factory.applyleafmul!(scratch, y, receive, G, true, false)

            # NO conjugation and NO weights on either side. See this section's header.
            lhs = sum(Fθ .* vec(G[1])) + sum(Fφ .* vec(G[2]))
            @test lhs ≈ transpose(x) * y rtol = 1e-11

            # and the SESQUILINEAR pairing must NOT hold, so a kernel that conjugates `C` cannot pass
            # the test above by accident on real-valued data.
            @test !isapprox(
                sum(conj.(Fθ) .* vec(G[1])) + sum(conj.(Fφ) .* vec(G[2])),
                dot(x, y);
                rtol=1e-6,
            )
        end
    end

    @testset "the receive collection aggregates too, for a transposed matvec" begin
        # H2Factory realizes the transpose by swapping the sides, so this collection is the one that
        # AGGREGATES then. `Rec = P S⁻ C` applied forward, the same matrix the scatter transposes.
        for _ in 1:3
            x = randn(rng, ComplexF64, nbf)
            Fθ, Fφ = mirroredforward(x)

            moment = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
            H2Factory.applyleafmul!(scratch, moment, receive, x)
            @test vec(moment[1]) ≈ Fθ rtol = 1e-12
            @test vec(moment[2]) ≈ Fφ rtol = 1e-12
        end
    end

    @testset "the radiation collection scatters too, for a transposed matvec" begin
        # The mirror image of the testset above: in a transposed matvec the TRIAL side is scattered
        # onto, through `R = P S C`, no mirror, since no antipodal map is involved on that side.
        Ssynth = t.synthesis
        for _ in 1:3
            x = randn(rng, ComplexF64, nbf)
            G = randommoment()

            cart = Ssynth * reshape(C * x, nc, 3)
            Fθ = [
                θ̂[q, 1] * cart[q, 1] + θ̂[q, 2] * cart[q, 2] + θ̂[q, 3] * cart[q, 3] for
                q in 1:nd
            ]
            Fφ = [
                φ̂[q, 1] * cart[q, 1] + φ̂[q, 2] * cart[q, 2] + φ̂[q, 3] * cart[q, 3] for
                q in 1:nd
            ]

            y = zeros(ComplexF64, nbf)
            H2Factory.applyleafmul!(scratch, y, radiation, G, true, false)
            @test sum(Fθ .* vec(G[1])) + sum(Fφ .* vec(G[2])) ≈ transpose(x) * y rtol =
                1e-11
        end
    end

    @testset "the conjugated kernels are the adjoint of the transposed ones" begin
        # `ConjugatedCollectionMul` conjugates the COLLECTION and nothing else, so each conjugated
        # kernel must equal its plain twin applied to a conjugated collection, which for a linear
        # map means `conj(A) x == conj(A conj(x))` on the way up and `Aᴴ = transpose(conj(A))` on the
        # way down. Asserted against the plain kernels rather than against a formula, so the two can
        # only agree by being the same matrix.
        for _ in 1:3
            x = randn(rng, ComplexF64, nbf)
            G = randommoment()

            plain = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
            H2Factory.applyleafmul!(scratch, plain, receive, conj.(x))
            conjugated = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
            MLFMA.applyconjugatedleafmul!(scratch, conjugated, receive, x)
            @test vec(conjugated[1]) ≈ conj.(vec(plain[1])) rtol = 1e-12
            @test vec(conjugated[2]) ≈ conj.(vec(plain[2])) rtol = 1e-12

            transposed = zeros(ComplexF64, nbf)
            H2Factory.applyleafmul!(scratch, transposed, radiation, G, true, false)
            adjointed = zeros(ComplexF64, nbf)
            MLFMA.applyconjugatedleafmul!(scratch, adjointed, radiation, G, true, false)
            # `Aᴴ G == conj(Aᵀ conj(G))`
            conjG = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
            conjG[1] .= conj.(G[1])
            conjG[2] .= conj.(G[2])
            viatranspose = zeros(ComplexF64, nbf)
            H2Factory.applyleafmul!(scratch, viatranspose, radiation, conjG, true, false)
            @test adjointed ≈ conj.(viatranspose) rtol = 1e-12
            @test !isapprox(adjointed, transposed; rtol=1e-6)   # non-vacuity
        end
    end

    @testset "α and β follow mul!'s convention" begin
        G = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        G[1] .= reshape(randn(rng, ComplexF64, nd), size(G[1]))
        G[2] .= reshape(randn(rng, ComplexF64, nd), size(G[2]))

        base = zeros(ComplexF64, nbf)
        H2Factory.applyleafmul!(scratch, base, receive, G, true, false)

        scaled = zeros(ComplexF64, nbf)
        H2Factory.applyleafmul!(scratch, scaled, receive, G, 2.5, false)
        @test scaled ≈ 2.5 .* base rtol = 1e-12

        # β = false OVERWRITES, which is what the traversal's leaf scatter passes
        preloaded = fill(7.0 + 0im, nbf)
        H2Factory.applyleafmul!(scratch, preloaded, receive, G, true, false)
        @test preloaded ≈ base rtol = 1e-12

        # and a nonzero β accumulates
        accumulated = copy(base)
        H2Factory.applyleafmul!(scratch, accumulated, receive, G, true, true)
        @test accumulated ≈ 2 .* base rtol = 1e-12
    end

    @testset "applying without a scratch is an error" begin
        G = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        @test_throws ErrorException H2Factory.applyleafmul!(
            H2Factory.EmptyChannels(), zeros(ComplexF64, nbf), receive, G, true, false
        )
    end
end
