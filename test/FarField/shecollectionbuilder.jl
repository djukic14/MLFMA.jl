# Building spherical-harmonic collections from a real assembler and a real tree.
#
# Everything up to here checked the SHE machinery against synthetic band-limited patterns, where it
# is exact. This is the first test where the patterns are real RWG far fields and the order is chosen
# by the rule rather than given -- so what it asserts is the thing production depends on: that the
# compressed leaf multiply reproduces the sampled one TO THE MAP'S TOLERANCE, with nobody picking an
# order by hand.

using MLFMA
using MLFMA.SphericalHarmonics: numcoefficients
using BEAST, CompScienceMeshes, H2Trees, H2Factory
using LinearAlgebra
using Random
using StaticArrays
using Test

include("../support/util_test.jl")

const HTB = H2Trees

function shebuilderfixture(;
    radius=0.8, meshfactor=8, leaffactor=4, ϵ=1e-4, protrusion=nothing, mesh=nothing
)
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ

    mesh === nothing && (mesh = meshsphere(radius * λ, λ / meshfactor))
    X = raviartthomas(mesh)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    builder = if protrusion === nothing
        HTB.TwoNTreeBuilder(; minhalfsize=λ / leaffactor, minvalues=0)
    else
        HTB.TwoNTreeBuilder(; minhalfsize=λ / leaffactor, minvalues=0, protrusion=protrusion)
    end
    tree = HTB.TwoNTree(X; builder=builder)

    leaflevels = sort(unique(HTB.level(tree, l) for l in HTB.leaves(tree)))
    minrelevantlevel = leaflevels[begin]
    samplings = [
        first(
            MLFMA.sampling(
                γ,
                ϵ,
                MLFMA.diameter(
                    HTB.halfsize(tree, HTB.LevelIterator(tree, level)[begin]), 3
                ),
                MLFMA.SphericalSamplingBuilder{Float64}(),
            ),
        ) for level in leaflevels
    ]
    leafsamplings = MLFMA.leaffarfieldsamplings(tree, samplings, minrelevantlevel)

    farop = BEAST.MWFarField3D(; gamma=γ)
    # The EFIE assembler, which stores TANGENTIAL -- the reference path.
    sampledassembler = MLFMA.BEASTFarFieldAssembler(
        farop,
        X,
        MLFMA.FARFIELDQUADSTRATEGYRECOM,
        MLFMA.farfieldstorerule(efie),
        MLFMA.farfieldtype(efie),
    )
    # The CARTESIAN assembler `shecollections` requires. Same quadrature, deliberately: a mismatch
    # here puts a floor on every comparison below and looks like SHE error.
    cartesianassembler = MLFMA.BEASTFarFieldAssembler(
        farop,
        X,
        MLFMA.FARFIELDQUADSTRATEGYRECOM,
        MLFMA.StoreAsFarFieldValue(),
        MLFMA.CartesianVectorFarField,
    )
    return (; X, tree, leafsamplings, samplings, sampledassembler, cartesianassembler, ϵ, λ)
end

@testset "Building SHE collections from an assembler" begin
    f = shebuilderfixture(; protrusion=HTB.NoProtrusionCheck())
    leaves = MLFMA.leafids(f.leafsamplings)

    collections, orders = MLFMA.shecollections(
        f.cartesianassembler, f.tree, f.leafsamplings, f.ϵ
    )
    sampled = MLFMA.RadiationFieldCollections(
        f.sampledassembler, f.tree, f.leafsamplings; verbose=false
    )

    @testset "one collection per leaf, one order per level" begin
        @test length(collections) == length(leaves)
        @test Set(keys(collections)) == Set(leaves)
        @test length(orders) == length(f.samplings)      # uniform tree: one level, one order
    end

    @testset "the chosen order is a choice, not a clamp" begin
        # `P ≤ L` is TEMPTING AND FALSE -- see the source header. It holds when boxes are at or above
        # the mesh scale (as here), and breaks when they are below it, because then a basis
        # function's support is larger than its own box and the support sets `P`. So what is asserted
        # is the property that actually matters: the analysis grid was not the binding constraint.
        for (sampling, P) in orders
            L = MLFMA.cutoffparameter(sampling)
            @test P >= 1
            @test P < MLFMA.sheanalysisorder(sampling)      # a choice, not a clamp
            @test MLFMA.sheanalysisorder(sampling) == L + MLFMA.SHEANALYSISMARGIN
            @test P <= L                                     # true HERE: mesh λ/8, boxes λ/4
        end
    end

    @testset "the compressed leaf multiply reproduces the sampled one" begin
        # THE TEST THIS FILE EXISTS FOR: real patterns, order chosen by the rule, compared against
        # the collection MLFMA builds today.
        Random.seed!(41)
        worst = 0.0
        for (i, leaf) in enumerate(leaves)
            sampling = f.leafsamplings[i].sampling
            scratch = MLFMA.shescratch(sampling, orders[sampling])
            nbf = length(HTB.values(f.tree, leaf))
            x = randn(ComplexF64, nbf)

            viashe = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
            viasamples = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
            H2Factory.applyleafmul!(scratch, viashe, collections[leaf], x)
            LinearAlgebra.mul!(viasamples, sampled[i], x)

            reference = sqrt(sum(abs2, viasamples[1]) + sum(abs2, viasamples[2]))
            difference = sqrt(
                sum(abs2, viashe[1] .- viasamples[1]) +
                sum(abs2, viashe[2] .- viasamples[2]),
            )
            worst = max(worst, difference / reference)
        end
        # The rule is calibrated to hold the FAR MATVEC at the sampled path's accuracy; per leaf it
        # is deliberately conservative, so this sits comfortably inside the map's own tolerance.
        @test worst < f.ϵ
    end

    @testset "the collections are actually smaller than what they replace" begin
        # Not a performance assertion -- a REPRESENTATION one. If this ever fails, the order rule has
        # stopped compressing and the whole scheme is pointless, which no accuracy test would catch.
        sampledbytes = sum(Base.summarysize(c) for c in sampled)
        shebytes = sum(Base.summarysize(collections[leaf]) for leaf in leaves)
        @test shebytes < sampledbytes
        @test sampledbytes / shebytes > 3        # (16/3)((L+1)/(P+1))² floors at 5.33 in theory
    end
end

@testset "Building SHE collections on a non-uniform tree" begin
    # Mixed depth means several levels, each with its own grid, its own `L`, and therefore its own
    # order. That per-level independence is the thing to check -- a single global order would be
    # wrong on both sides: too low for the big boxes, wasteful for the small ones.
    f = shebuilderfixture(; mesh=fixturemesh("she-protruding"), leaffactor=12)
    leaflevels = sort(unique(HTB.level(f.tree, l) for l in HTB.leaves(f.tree)))
    @test length(leaflevels) > 1                 # non-vacuity: the fixture really is mixed

    collections, orders = MLFMA.shecollections(
        f.cartesianassembler, f.tree, f.leafsamplings, f.ϵ
    )
    @test Set(keys(collections)) == Set(MLFMA.leafids(f.leafsamplings))

    # Bigger boxes carry more bandwidth, so a coarser level must not get a smaller order.
    bylevel = sort([(MLFMA.cutoffparameter(s), P) for (s, P) in orders])
    @test issorted([P for (_, P) in bylevel])
    # Some fine boxes are smaller than their basis supports. Keep this counterexample fixed:
    # a newly generated mesh need not exercise P > L, even with the same mesher arguments.
    @test any(P > L for (L, P) in bylevel)   # non-vacuity: this fixture really does break P <= L
end

@testset "The SHE leaf-scratch channel" begin
    # Sized and keyed exactly like `MLFMA.leafscratchchannel`, whose pool it replaces, so the test
    # checks it AGAINST that one rather than against a restatement of the keying rule.
    f = shebuilderfixture(; protrusion=HTB.NoProtrusionCheck())
    _, orders = MLFMA.shecollections(f.cartesianassembler, f.tree, f.leafsamplings, f.ϵ)

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
    leaflevels = MLFMA._leaflevels(f.tree)

    H2Factory.withschedulechunks(A.schedules) do
        sampled = MLFMA.leafscratchchannel(
            complex(core.γ),
            MLFMA.farfieldtype(core.operator),
            leaflevels,
            core.plans,
            core.samplings,
            core.minrelevantlevel,
        )
        she = MLFMA.shescratchchannel(
            orders,
            leaflevels,
            core.plans,
            core.samplings,
            core.minrelevantlevel;
            groupsperchunk=level ->
                MLFMA.leavesperchunk(f.tree, level, H2Factory.nchunks(level)),
        )

        # SAME LEVELS, SAME CHUNK COUNTS -- it is a drop-in for the pool it replaces
        @test keys(she) == keys(sampled)
        @test !isempty(she)

        for level in keys(she)
            scratch = H2Factory.borrowbuffer(she, level, 1, H2Factory.isForward())
            sampling = core.samplings[MLFMA._samplinglevelindex(
                level, core.minrelevantlevel, core.samplings
            )]
            P = orders[sampling]
            nd = length(MLFMA.θs(sampling)) * length(MLFMA.φs(sampling))

            @test MLFMA.numdirections(scratch) == nd
            @test size(scratch.synthesis) == (nd, numcoefficients(P))
            @test size(scratch.mirror) == size(scratch.synthesis)
            @test size(scratch.θ̂) == (nd, 3)
            @test MLFMA.maxgroups(scratch) >= 1
            @test length(scratch.leaves) == MLFMA.maxgroups(scratch)
        end
    end
end

@testset "leavesperchunk bounds the biggest chunk" begin
    f = shebuilderfixture(; protrusion=HTB.NoProtrusionCheck())
    level = HTB.level(f.tree, first(HTB.leaves(f.tree)))
    nnodes = length(collect(HTB.LevelIterator(f.tree, level)))
    @test MLFMA.leavesperchunk(f.tree, level, 1) == nnodes        # one chunk holds them all
    @test MLFMA.leavesperchunk(f.tree, level, nnodes) == 1
    @test MLFMA.leavesperchunk(f.tree, level, 2) == cld(nnodes, 2)
    @test MLFMA.leavesperchunk(f.tree, level, 0) >= 1             # degenerate count is clamped
end

@testset "The SHE receive kernel agrees with the sampled one" begin
    # THE TEST THAT WOULD HAVE CAUGHT THE CONVENTION BUG. Everything else about the receive side is
    # checked against an ALGEBRAIC property -- the transpose identity in
    # `test/FarField/sphericalharmoniccollection.jl` -- and an algebraic property cannot tell you
    # which convention MLFMA actually uses. The first version of that kernel applied `Cᴴ Mᴴ Pᵀ W`,
    # was self-consistent under the weighted sesquilinear pairing, and was 6.9e-2 wrong end to end.
    #
    # So this compares the two IMPLEMENTATIONS on the same leaf and the same moment: the SHE receive
    # collection against `LazyReceiveFieldCollection`, which is what a Galerkin map uses today.
    f = shebuilderfixture(; protrusion=HTB.NoProtrusionCheck())
    leaves = MLFMA.leafids(f.leafsamplings)

    collections, orders = MLFMA.shecollections(
        f.cartesianassembler, f.tree, f.leafsamplings, f.ϵ
    )
    sampled = MLFMA.RadiationFieldCollections(
        f.sampledassembler, f.tree, f.leafsamplings; verbose=false
    )

    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    Random.seed!(77)
    worstreceive = 0.0
    worstaggregate = 0.0
    for (i, leaf) in enumerate(leaves)
        sampling = f.leafsamplings[i].sampling
        shescratch = MLFMA.shescratch(sampling, orders[sampling])
        nbf = length(HTB.values(f.tree, leaf))

        shereceive = MLFMA.SphericalHarmonicReceiveCollection(
            collections[leaf], MLFMA.receivecoupling(efie)
        )
        lazyreceive = MLFMA.LazyReceiveFieldCollection(
            efie, sampled[i], MLFMA.mirrorindices(sampling)
        )
        lazyscratch = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))

        moment = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        moment[1] .= reshape(randn(ComplexF64, length(moment[1])), size(moment[1]))
        moment[2] .= reshape(randn(ComplexF64, length(moment[2])), size(moment[2]))

        # SCATTER: the arity a forward matvec's disaggregation calls.
        yshe = zeros(ComplexF64, nbf)
        ylazy = zeros(ComplexF64, nbf)
        H2Factory.applyleafmul!(shescratch, yshe, shereceive, moment, true, false)
        H2Factory.applyleafmul!(lazyscratch, ylazy, lazyreceive, moment, true, false)
        worstreceive = max(worstreceive, norm(yshe - ylazy) / norm(ylazy))

        # AGGREGATE: the arity a transposed matvec's upward pass calls, on the same collections.
        x = randn(ComplexF64, nbf)
        mshe = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        mlazy = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling))
        H2Factory.applyleafmul!(shescratch, mshe, shereceive, x)
        H2Factory.applyleafmul!(lazyscratch, mlazy, lazyreceive, x)
        reference = sqrt(sum(abs2, mlazy[1]) + sum(abs2, mlazy[2]))
        difference = sqrt(sum(abs2, mshe[1] .- mlazy[1]) + sum(abs2, mshe[2] .- mlazy[2]))
        worstaggregate = max(worstaggregate, difference / reference)
    end

    # The SHE patterns are the sampled ones truncated at order `P`, so these carry the same
    # per-leaf truncation error the radiation comparison above does -- and nothing more.
    @test worstreceive < f.ϵ
    @test worstaggregate < f.ϵ
end
