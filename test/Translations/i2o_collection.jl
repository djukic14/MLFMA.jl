using MLFMA
using CompScienceMeshes, BEAST
using StaticArrays
using ClusterTrees
using H2Factory
using Test
using H2FactoryTestUtils
using OhMyThreads
using H2Trees

include("../support/util_test.jl")

# The raw coupling kernel is an implementation hook behind `applyi2o!`, so these testsets go through the
# public protocol entry point instead.
#
# The context carries a real scheduler rather than a placeholder: the coupling primitive dispatches on
# `threading::Scheduler`, so `nothing` there fails to match. The couplings used here need no scratch,
# hence `EmptyChannels`.
function applyi2o_forward!(out, in, coupling, receivingnode, translatingnode)
    return H2Factory.applyi2o!(
        out,
        in,
        coupling,
        receivingnode,
        translatingnode,
        H2Factory.ForwardTranslation(),
        H2Factory.Overwrite(),
        H2Factory.TranslationContext(
            OhMyThreads.SerialScheduler(), H2Factory.EmptyChannels()
        ),
    )
end

# True when a test/trial pair actually produces far interactions. An empty aggregate plan is a
# hard error in H2Trees, so pairs that produce one have to be recognized before assembly rather than
# after.
function haspetrovfarinteractions(X, Y, λ)
    tree = H2Trees.BlockTree(
        X,
        Y;
        builder=H2Trees.BlockTreeBuilder(;
            test=TwoNTreeBuilder(; minhalfsize=λ / 10, minvalues=0),
            trial=TwoNTreeBuilder(; minhalfsize=λ / 10, minvalues=0),
        ),
    )
    try
        H2Trees.buildplans(
            tree;
            builder=H2Trees.PlanBuilder(; isnear=H2Trees.isnear(; additionalbufferboxes=0)),
        )
        return true
    catch e
        isa(e, ArgumentError) || rethrow()
        return false
    end
end

@testset "Translation Collections" begin
    λ = 0.1
    m = meshsphere(1 * λ, λ / 10)
    X = raviartthomas(m)

    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=λ / 16, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
        ),
    )

    γ = ComplexF64(1.0)
    ϵ = 1.0

    SamplingStrategy = MLFMA.SphericalSampling{Float64}
    samplingfunction = MLFMA.sampling

    mintranslationlevel = H2Trees.mintranslationlevel(
        tree; TranslatingNodesIterator=H2Trees.TranslatingNodesIterator
    )

    samplings = []

    for i in mintranslationlevel:H2Trees.levels(tree)[end]
        push!(samplings, samplingfunction(γ, ϵ, tree; level=i)[1])
    end

    for mode in [H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode()]
        TFIterator = H2Trees.TranslatingNodesIterator(; isnear=H2Trees.isnear())
        galerkinplans = H2Trees.buildplans(
            tree;
            builder=H2Trees.PlanBuilder(;
                aggregationmode=mode,
                translatingnodesiterator=TFIterator,
                aggregatenode=H2Trees.istranslatingnode(;
                    TranslatingNodesIterator=TFIterator
                ),
            ),
        )
        aggregationplan = galerkinplans.trialaggregationplan
        disaggregationplan = galerkinplans.testdisaggregationplan

        # Both constructors return `(collection, channel)`; only the collection is the coupling.
        tfs, _ = MLFMA.TranslationCollection(
            γ, tree, aggregationplan, disaggregationplan, samplings
        )

        tfdummy, _ = MLFMA.DummyTranslationCollection(
            γ, tree, aggregationplan, disaggregationplan, samplings
        )

        for receivingnodes in H2Trees.DepthFirstIterator(tree)
            for translatingnode in H2Trees.TranslatingNodesIterator(tree, receivingnodes)
                s, _ = samplingfunction(
                    γ, ϵ, tree; level=H2Trees.level(tree, receivingnodes)
                )
                ff1 = MLFMA.TangentialSphericalFarField(γ, s)
                ff1[1] .= 1.0
                ff1[2] .= 1.0

                storagefarfield1 = MLFMA.TangentialSphericalFarField(γ, s)

                applyi2o_forward!(
                    storagefarfield1, ff1, tfs, receivingnodes, translatingnode
                )

                # non-vacuity: the stored collection must actually translate something here.
                # The translated *values* are checked against an independent reference
                # (`MLFMA.translationoperator!`) in the "Translation collection in MLFMAMatrix2"
                # testset below; this testset covers the protocol and the dummy short-circuit.
                @test !all(iszero.(storagefarfield1[1]))
                @test !all(iszero.(storagefarfield1[2]))

                storagefarfield1[1] .= 0.0
                storagefarfield1[2] .= 0.0
                applyi2o_forward!(
                    storagefarfield1, ff1, tfdummy, receivingnodes, translatingnode
                )
                @test all(iszero.(storagefarfield1[1]))
                @test all(iszero.(storagefarfield1[2]))
            end
        end
    end
end

@testset "Petrov Galerkin Translation collection" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ
    ϵ = 1e-5

    mx = meshspherewithcenter([0.0, 0, 0], λ, λ / 5)
    my = meshspherewithcenter([2 * λ, 0, 0], 2 * λ, λ / 5)
    mz = meshspherewithcenter([10 * λ, 0, 0], λ / 2, λ / 5)

    ms = [mx, my, mz]

    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    for mx in ms
        X = raviartthomas(mx)
        for my in ms
            Y = raviartthomas(my)
            tree = H2Trees.BlockTree(
                X,
                Y;
                builder=H2Trees.BlockTreeBuilder(;
                    test=TwoNTreeBuilder(;
                        minhalfsize=λ / 10,
                        minvalues=0,
                        protrusion=H2Trees.NoProtrusionCheck(),
                    ),
                    trial=TwoNTreeBuilder(;
                        minhalfsize=λ / 10,
                        minvalues=0,
                        protrusion=H2Trees.NoProtrusionCheck(),
                    ),
                ),
            )

            TFIterator = H2Trees.TranslatingNodesIterator(; isnear=H2Trees.isnear())
            petrovplans = H2Trees.buildplans(
                tree;
                builder=H2Trees.PlanBuilder(;
                    aggregationmode=H2Trees.AggregateMode(),
                    translatingnodesiterator=TFIterator,
                    aggregatenode=H2Trees.istranslatingnode(;
                        TranslatingNodesIterator=TFIterator
                    ),
                ),
            )
            testaggregationplan = petrovplans.testaggregationplan
            trialaggregationplan = petrovplans.trialaggregationplan
            testdisaggregationplan = petrovplans.testdisaggregationplan
            trialdisaggregationplan = petrovplans.trialdisaggregationplan
            relevantlevels = petrovplans.relevantlevels
            mintranslationlevel = petrovplans.mintranslationlevel
            minrelevantlevel = relevantlevels[begin]

            samplings, parentsamplings = MLFMA.getsamplingsparentsamplings(
                γ, ϵ, H2Trees.treewithmorelevels(tree), relevantlevels, MLFMA.sampling
            )
            translationsamplings = samplings[(mintranslationlevel - minrelevantlevel + 1):end]

            tfcollection, _ = MLFMA.TranslationCollection(
                γ, tree, trialaggregationplan, testdisaggregationplan, translationsamplings
            )

            petrovplans2 = H2Trees.buildplans(
                tree;
                builder=H2Trees.PlanBuilder(;
                    aggregationmode=H2Trees.AggregateTranslateMode(),
                    translatingnodesiterator=TFIterator,
                    aggregatenode=H2Trees.istranslatingnode(;
                        TranslatingNodesIterator=TFIterator
                    ),
                ),
            )

            testaggregationplan2 = petrovplans2.testaggregationplan
            trialaggregationplan2 = petrovplans2.trialaggregationplan
            testdisaggregationplan2 = petrovplans2.testdisaggregationplan
            trialdisaggregationplan2 = petrovplans2.trialdisaggregationplan
            tfcollection2, _ = MLFMA.TranslationCollection(
                γ,
                tree,
                trialaggregationplan2,
                testdisaggregationplan2,
                translationsamplings,
            )

            for k in keys(tfcollection.dict)
                @test tfcollection[k] ≈ tfcollection2[k]
            end
        end
    end
end

@testset "Translation collection in MLFMAMatrix2" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    mx = meshspherewithcenter([0.0, 0, 0], λ, λ / 5)
    my = meshspherewithcenter([2 * λ, 0, 0], 2 * λ, λ / 5)
    mz = meshspherewithcenter([10 * λ, 0, 0], λ / 2, λ / 5)

    ms = [mx, my, mz]

    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    # One of the nine pairs -- the smallest sphere against itself -- has no well-separated box pair
    # at all, so its aggregate plan is empty and H2Trees rejects it outright. That is the
    # modernization plan's own hazard ("do not use a geometry whose far plan is empty"), not a defect
    # here, so the pair is skipped explicitly and counted, rather than the geometry being retuned
    # until it happens to work. `skipped` is asserted below so the skip cannot quietly grow.
    exercised = 0
    skipped = 0

    for mode in [H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode()]
        for mx in ms
            X = raviartthomas(mx)
            for my in ms
                Y = raviartthomas(my)

                if !haspetrovfarinteractions(X, Y, λ)
                    skipped += 1
                    continue
                end
                exercised += 1
                A_MLFMA = MLFMA.assemble(
                    efie,
                    X,
                    Y;
                    strategy=MLFMA.MLFMAStrategy(
                        efie,
                        X,
                        Y;
                        treestrategy=MLFMA.TreeStrategy(efie; minhalfsize=λ / 10),
                        nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                            MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
                        ),
                        nearfieldstrategy=H2Factory.InteractionStrategy(;
                            isnear=H2Trees.isnear(; additionalbufferboxes=0),
                            aggregationmode=mode,
                        ),
                        farfieldstrategy=MLFMA.FarFieldStrategy(;
                            assemblerbuilder=MLFMA.DummyFarFieldAssemblerBuilder()
                        ),
                        leveltransferstrategy=MLFMA.LevelTransferStrategy(
                            efie; PhaseShifter=MLFMA.DummyPhaseShifter
                        ),
                    ),
                    verbose=true,
                )

                testtree = H2Trees.testtree(A_MLFMA)
                trialtree = H2Trees.trialtree(A_MLFMA)
                for testnode in H2Trees.DepthFirstIterator(testtree)
                    level = H2Trees.level(testtree, testnode)
                    levelindex = level - H2Factory.minrelevantlevel(A_MLFMA) + 1
                    receivecenter = H2Trees.center(testtree, testnode)
                    for trialnode in
                        H2Trees.TranslatingNodesIterator(trialtree, testtree, testnode)
                        translatingcenter = H2Trees.center(trialtree, trialnode)
                        sampling = MLFMA.samplings(A_MLFMA)[levelindex]
                        cartesiansamplepoints, _ = MLFMA.cartesiansamplepoints(sampling)
                        cutoffparameter = MLFMA.cutoffparameter(sampling)
                        # Scratch allocated here rather than borrowed from the map. This used to
                        # reach into `A_MLFMA.storagefields`/`collectedfarfields`, two of three legacy
                        # map-owned scratch pools (per-chunk, per-level `BufferStore`s -- NOT per-node
                        # moment storage); those are gone, and this was their only reader. It only ever
                        # wanted two correctly-shaped `FarField`s -- both are overwritten on the next
                        # lines -- so constructing them from the level's sampling says that outright
                        # instead of borrowing whatever pool happened to have the right shape.
                        storagefield = MLFMA.farfieldtype(efie)(ComplexF64, sampling)
                        tempfield = MLFMA.farfieldtype(efie)(ComplexF64, sampling)
                        tempfield[1] .= 1.0
                        tempfield[2] .= 1.0

                        storagefield[1] .= 0.0
                        storagefield[2] .= 0.0

                        tf = zeros(ComplexF64, size(storagefield[1]))
                        @test_nowarn applyi2o_forward!(
                            storagefield,
                            tempfield,
                            MLFMA.translationcollection(A_MLFMA),
                            testnode,
                            trialnode,
                        )

                        MLFMA.translationoperator!(
                            tf,
                            MLFMA.γ(A_MLFMA),
                            cartesiansamplepoints,
                            receivecenter,
                            translatingcenter,
                            cutoffparameter,
                        )
                        @test storagefield[1] ≈ tf .* MLFMA.weights(A_MLFMA, level)
                        @test storagefield[2] ≈ tf .* MLFMA.weights(A_MLFMA, level)

                        @test haskey(H2Factory.trialmoments(A_MLFMA), trialnode)
                    end
                end
            end
        end
    end

    # Non-vacuity for the skip above: exactly one of the nine pairs per mode has an empty far plan,
    # so eight must have run. If a future geometry or plan-builder change silently made more pairs
    # degenerate, this testset would otherwise shrink without any failure.
    @test exercised == 16
    @test skipped == 2
end
