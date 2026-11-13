using MLFMA
using BEAST, CompScienceMeshes
using StaticArrays
using Test
using LinearAlgebra
using OhMyThreads
using H2Factory
using H2Trees

include("../support/util_test.jl")

# Phase 1: the construction strategy bundle.
#
# The load-bearing claim of this slice is that bundling changed NO default. Every assertion in the
# first testset therefore compares an accessor against the `*RECOM` constant (or literal) that the
# corresponding keyword argument defaulted to before the bundle existed, rather than against a value
# copied out of a run -- a snapshot would pass just as happily if both sides drifted together.
@testset "MLFMAStrategy defaults reproduce the pre-bundle keyword defaults" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    strategy = MLFMA.MLFMAStrategy(efie, X)

    # tree
    @test MLFMA.minhalfsize(strategy) == MLFMA.λ(efie) * MLFMA.MINHALFSIZERECOM
    @test MLFMA.minvalues(strategy) == MLFMA.MINVALUESRECOM

    # sampling -- a BUILDER object now, not a functor wrapping the sampling type
    @test MLFMA.samplingbuilder(strategy) === MLFMA.SAMPLINGBUILDERRECOM
    @test MLFMA.ϵ(strategy) == 1e-5
    # ...and the `(γ, ϵ, tree)` wrapper the map constructors call is DERIVED from that builder, so
    # it cannot end up adapting a different sampling family than the strategy holds
    @test MLFMA.samplingfunction(strategy) ==
        MLFMA.SamplingFunctor(MLFMA.SAMPLINGBUILDERRECOM)
    # non-vacuity for the whole pair: the builder really does produce the sampling type it names
    @test MLFMA.SAMPLINGBUILDERRECOM(4, 6) === MLFMA.SphericalSampling{Float64}(4, 6)

    # near interactions -- now H2Factory's strategy protocol rather than a stored assembler type, so
    # the claim is that it SELECTS the assembler MLFMA always defaulted to. Checked by building the
    # assembler through H2Factory's dispatch, which is what the map constructor will do.
    nearstrategy = MLFMA.nearinteractionsstrategy(strategy)
    @test nearstrategy isa H2Factory.AbstractNearInteractionsStrategy
    @test nearstrategy.quadstrategy == MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
    builtassembler = H2Factory.buildnearinteractionsassembler(
        efie, X, X, MLFMA.γ(efie), nearstrategy
    )
    @test builtassembler isa MLFMA.NEARINTERACTIONSASSEMBLERRECOM
    # non-vacuity: that constant really is the Sparse assembler H2Factory had no strategy for
    @test MLFMA.NEARINTERACTIONSASSEMBLERRECOM ===
        H2Factory.SparseBEASTNearInteractionsAssembler

    # far field -- a BUILDER object now, not a stored assembler type
    @test MLFMA.farfieldassemblerbuilder(strategy) === MLFMA.FARFIELDASSEMBLERBUILDERRECOM
    @test MLFMA.farfieldquadstrategy(strategy) == MLFMA.FARFIELDQUADSTRATEGYRECOM
    # non-vacuity: that builder really does build the assembler MLFMA always defaulted to, which is
    # the claim the rename has to preserve
    @test MLFMA.FARFIELDASSEMBLERBUILDERRECOM(
        MLFMA.radiationoperator(efie),
        X,
        MLFMA.FARFIELDQUADSTRATEGYRECOM,
        MLFMA.farfieldstorerule(efie),
        MLFMA.farfieldtype(efie),
    ) isa MLFMA.BEASTFarFieldAssembler

    # level transfer -- parameterized by the OPERATOR's γ type, so this pins the parameterization and
    # not just the base type
    @test MLFMA.interpolatortype(strategy) ===
        MLFMA.INTERPOLATORECOM{15,typeof(MLFMA.γ(efie))}
    @test MLFMA.phaseshiftertype(strategy) === MLFMA.PHASESHIFTERRECOM

    # I2O
    @test MLFMA.translationcollectiontype(strategy) === MLFMA.TRANSLATIONRECOM

    # near field / plan -- H2Factory's strategy, read with H2Factory's accessors. `isnear` is the
    # one default H2Factory and MLFMA disagree on (H2Factory: no buffer boxes), so MLFMA passes it
    # explicitly; the other three derive from it and already matched.
    nearfield = MLFMA.nearfieldstrategy(strategy)
    @test nearfield isa H2Factory.AbstractInteractionStrategy
    @test H2Factory.nearcriterion(nearfield) ==
        H2Trees.isnear(; additionalbufferboxes=MLFMA.BUFFERBOXRECOM)
    @test H2Factory.aggregationmode(nearfield) == H2Trees.AggregateMode()
    @test H2Factory.translatingnodesiterator(nearfield) ==
        MLFMA.TRANSLATINGNODESRECOM(; isnear=H2Factory.nearcriterion(nearfield))
    @test H2Factory.aggregatenode(nearfield) == MLFMA.AGGREGATENODES(;
        TranslatingNodesIterator=H2Factory.translatingnodesiterator(nearfield)
    )
    # non-vacuity: MLFMA's buffered default is NOT H2Factory's unbuffered one, so passing it
    # explicitly is load-bearing rather than decorative
    @test H2Factory.nearcriterion(nearfield) != H2Trees.isnear()
    @test H2Factory.nearcriterion(H2Factory.InteractionStrategy()) !=
        H2Factory.nearcriterion(nearfield)

    # scheduler -- the pre-bundle default was a bare `SerialScheduler()` wrapped into an
    # `ExecutionScheduling` at construction time
    @test MLFMA.scheduler(strategy) == SerialScheduler()

    # non-vacuity: a real complex operator, so the γ-parameterized interpolator claim above is not
    # trivially satisfied by a real-valued default
    @test MLFMA.γ(efie) isa Complex
    @test !iszero(MLFMA.γ(efie))
end

@testset "MLFMAStrategy constructs in one-space and two-space forms" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    Y = raviartthomas(meshspherewithcenter([4 * λ, 0.0, 0.0], 1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    galerkin = MLFMA.MLFMAStrategy(efie, X)
    petrov = MLFMA.MLFMAStrategy(efie, X, Y)

    # `trialspace` defaults to `testspace`, so the one-space form is the two-space form with the same
    # space twice -- and nothing today differs between them, which is the property to pin: the bundle
    # must not have quietly made the Galerkin and Petrov defaults diverge.
    @test MLFMA.MLFMAStrategy(efie, X, X) == galerkin
    for accessor in (
        MLFMA.minhalfsize,
        MLFMA.minvalues,
        MLFMA.ϵ,
        MLFMA.farfieldquadstrategy,
        MLFMA.samplingfunction,
    )
        @test accessor(galerkin) == accessor(petrov)
    end

    # non-vacuity: the two spaces really are different, so the equalities above are not comparing a
    # strategy with itself
    @test numfunctions(X) == numfunctions(Y)
    @test BEAST.positions(X) != BEAST.positions(Y)
end

@testset "Overriding one sub-strategy leaves the others at their defaults" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    defaults = MLFMA.MLFMAStrategy(efie, X)
    overridden = MLFMA.MLFMAStrategy(
        efie,
        X;
        farfieldstrategy=MLFMA.FarFieldStrategy(; quadstrategy=BEAST.SingleNumQStrat(1)),
    )

    # the override took
    @test MLFMA.farfieldquadstrategy(overridden) == BEAST.SingleNumQStrat(1)
    # non-vacuity: it is not the default value already
    @test MLFMA.farfieldquadstrategy(overridden) != MLFMA.farfieldquadstrategy(defaults)

    # ...and nothing else moved, including the OTHER field of the same sub-strategy
    @test MLFMA.farfieldassemblerbuilder(overridden) ==
        MLFMA.farfieldassemblerbuilder(defaults)
    @test MLFMA.treestrategy(overridden) == MLFMA.treestrategy(defaults)
    @test MLFMA.samplingstrategy(overridden) == MLFMA.samplingstrategy(defaults)
    @test MLFMA.nearinteractionsstrategy(overridden) ==
        MLFMA.nearinteractionsstrategy(defaults)
    @test MLFMA.nearfieldstrategy(overridden) == MLFMA.nearfieldstrategy(defaults)
    @test MLFMA.leveltransferstrategy(overridden) == MLFMA.leveltransferstrategy(defaults)
    @test MLFMA.i2ostrategy(overridden) == MLFMA.i2ostrategy(defaults)
    @test MLFMA.scheduler(overridden) == MLFMA.scheduler(defaults)
end

# The desynchronization hazard the plan calls out: `isnear` defines the near/far split, and the near
# block and the traversal plan must agree on it exactly or the two together will double-count or miss
# pairs. `translatingnodesiterator` and `aggregatenode` therefore derive from `isnear`.
#
# That derivation is H2Factory's (`InteractionStrategy`), not MLFMA's, so what is pinned here is that
# MLFMA gets the benefit of it: overriding `isnear` alone propagates. Asserted by EQUALITY against
# explicit construction with the custom `isnear`, not merely by inequality against the default --
# inequality alone would also pass if the derivation used some third, unrelated value.
@testset "Overriding isnear propagates through H2Factory's near-field strategy" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    custom = H2Trees.isnear(; additionalbufferboxes=3)
    overridden = MLFMA.nearfieldstrategy(
        MLFMA.MLFMAStrategy(
            efie, X; nearfieldstrategy=H2Factory.InteractionStrategy(; isnear=custom)
        ),
    )
    defaults = MLFMA.nearfieldstrategy(MLFMA.MLFMAStrategy(efie, X))

    expectediterator = MLFMA.TRANSLATINGNODESRECOM(; isnear=custom)
    @test H2Factory.translatingnodesiterator(overridden) == expectediterator
    @test H2Factory.aggregatenode(overridden) ==
        MLFMA.AGGREGATENODES(; TranslatingNodesIterator=expectediterator)

    # non-vacuity: the custom `isnear` really does change all three, so the equalities above are not
    # comparing defaults with defaults
    @test H2Factory.nearcriterion(overridden) != H2Factory.nearcriterion(defaults)
    @test H2Factory.translatingnodesiterator(overridden) !=
        H2Factory.translatingnodesiterator(defaults)
    @test H2Factory.aggregatenode(overridden) != H2Factory.aggregatenode(defaults)
end

# The payoff of using H2Factory's near-field strategy rather than a local one: `assembleplans`
# dispatches on the tree's `treetrait`, so ONE call covers MLFMA's Galerkin `TwoNTree` and Petrov
# `BlockTree`. That is the plan-building half of the Galerkin/Petrov duplication the phase exists to
# remove, so it is pinned on both tree types rather than only the Galerkin one.
@testset "assembleplans accepts MLFMA's near-field strategy for both tree types" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    strategy = MLFMA.MLFMAStrategy(efie, X)
    builder = TwoNTreeBuilder(;
        minhalfsize=MLFMA.minhalfsize(strategy), minvalues=MLFMA.minvalues(strategy)
    )

    # `H2Trees.buildtree`, not `MLFMA.buildtree`. The latter resolved only because H2Factory
    # re-exported H2Trees' binding and `using H2Factory` pulled it into MLFMA's namespace; it was
    # never MLFMA's function. That re-export is gone, so the call site names the real owner.
    galerkintree = H2Trees.buildtree(X; builder=builder)
    petrovtree = H2Trees.buildtree(
        X, X; builder=BlockTreeBuilder(; test=builder, trial=builder)
    )

    # non-vacuity: these really are the two different tree types the dispatch has to cover
    @test galerkintree isa TwoNTree
    @test petrovtree isa H2Trees.BlockTree

    for tree in (galerkintree, petrovtree)
        plans = H2Factory.assembleplans(tree, MLFMA.nearfieldstrategy(strategy))
        @test plans isa H2Trees.PlanSet
    end
end

# The scheduler is stored bare because the per-level containers cannot be keyed until the tree's level
# indices are known. This pins that materializing honours the whole requested range -- the `KeyError`
# hazard from sizing by level COUNT instead of top level INDEX, which `GalerkinMLFMA` was actually
# bitten by (see `test/Maps/galerkin.jl`).
@testset "executionschedulingfor keys every level in the requested range" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    strategy = MLFMA.MLFMAStrategy(efie, X)

    # Positional `getfield` over the per-operation containers: `OuterScheduling`/`InnerScheduling` hold
    # one `Dict{Int,Scheduler}` per H2 operation, and this pins "every one of them is keyed over the
    # whole range" without hard-coding which operations exist today.
    function levelcounts(scheduling)
        outer = scheduling.outer
        inner = scheduling.inner
        return vcat(
            [length(getfield(outer, i)) for i in 1:nfields(outer)],
            [length(getfield(inner, i)) for i in 1:nfields(inner)],
        )
    end

    for range in (1:3, 1:6)
        scheduling = MLFMA.executionschedulingfor(strategy, range)
        @test scheduling isa H2Factory.ExecutionScheduling
        @test H2Factory.nchunks(scheduling) >= 1
        # non-vacuity: there is at least one per-operation container to check
        @test !isempty(levelcounts(scheduling))
        @test all(==(length(range)), levelcounts(scheduling))
    end

    # A non-default scheduler survives materialization, and reaches `nchunks`.
    nchunks = Threads.nthreads() == 4 ? 5 : 4
    threaded = MLFMA.MLFMAStrategy(
        efie, X; scheduler=OhMyThreads.DynamicScheduler(; nchunks=nchunks)
    )
    @test H2Factory.nchunks(MLFMA.executionschedulingfor(threaded, 1:3)) == nchunks
    # non-vacuity: distinguishable from the serial default, whatever `-t` the suite runs with
    @test nchunks != H2Factory.nchunks(MLFMA.executionschedulingfor(strategy, 1:3))
end

@testset "Strategy accessors are type-stable" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    strategy = MLFMA.MLFMAStrategy(efie, X)

    # Written out rather than looped: `@inferred` has to see a concrete call at the macro site. Looping
    # over a tuple of accessors makes the loop variable a `Union` of function types, so the closure
    # call is genuinely not inferrable and every iteration errors -- the macro would be measuring the
    # loop, not the accessors.
    #
    # Sub-strategy accessors first, then the leaf accessors that reach through two of them -- the leaf
    # ones are where an inference break would actually land, since construction sites call those.
    @test (@inferred MLFMA.treestrategy(strategy)) === MLFMA.treestrategy(strategy)
    @test (@inferred MLFMA.samplingstrategy(strategy)) === MLFMA.samplingstrategy(strategy)
    @test (@inferred MLFMA.nearinteractionsstrategy(strategy)) ===
        MLFMA.nearinteractionsstrategy(strategy)
    @test (@inferred MLFMA.nearfieldstrategy(strategy)) ===
        MLFMA.nearfieldstrategy(strategy)
    @test (@inferred MLFMA.farfieldstrategy(strategy)) === MLFMA.farfieldstrategy(strategy)
    @test (@inferred MLFMA.leveltransferstrategy(strategy)) ===
        MLFMA.leveltransferstrategy(strategy)
    @test (@inferred MLFMA.i2ostrategy(strategy)) === MLFMA.i2ostrategy(strategy)
    @test (@inferred MLFMA.scheduler(strategy)) === MLFMA.scheduler(strategy)

    @test (@inferred MLFMA.minhalfsize(strategy)) === MLFMA.minhalfsize(strategy)
    @test (@inferred MLFMA.minvalues(strategy)) === MLFMA.minvalues(strategy)
    @test (@inferred MLFMA.ϵ(strategy)) === MLFMA.ϵ(strategy)
    @test (@inferred MLFMA.samplingfunction(strategy)) === MLFMA.samplingfunction(strategy)
    @test (@inferred MLFMA.farfieldassemblerbuilder(strategy)) ===
        MLFMA.farfieldassemblerbuilder(strategy)
    @test (@inferred MLFMA.farfieldquadstrategy(strategy)) ===
        MLFMA.farfieldquadstrategy(strategy)
    @test (@inferred MLFMA.interpolatortype(strategy)) === MLFMA.interpolatortype(strategy)
    @test (@inferred MLFMA.phaseshiftertype(strategy)) === MLFMA.phaseshiftertype(strategy)
    @test (@inferred MLFMA.translationcollectiontype(strategy)) ===
        MLFMA.translationcollectiontype(strategy)

    # H2Factory's near-field accessors are type-stable on MLFMA's strategy too, since construction
    # reads them per level.
    nearfield = MLFMA.nearfieldstrategy(strategy)
    @test (@inferred H2Factory.nearcriterion(nearfield)) ===
        H2Factory.nearcriterion(nearfield)
    @test (@inferred H2Factory.aggregationmode(nearfield)) ===
        H2Factory.aggregationmode(nearfield)
end

# Phase 1 removed the pre-Phase-1 keyword surface outright rather than shimming it, so there is no
# longer a second construction path to compare against. What remains worth pinning is that a map built
# from an explicit strategy is CORRECT against a dense reference -- i.e. the accessors are wired to the
# right places, not merely present.
#
# The first testset already pins that the strategy's defaults equal the old keyword defaults value by
# value; this pins that reading them through the constructor produces a working map.
@testset "A map built from a strategy matches a dense reference" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    # Same fixture and cheap matched quadrature as the Phase 5 testset in `test/Maps/galerkin.jl`.
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))

    A = MLFMA.GalerkinMLFMA(
        efie,
        X;
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
            nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
        ),
        tree=tree,
        verbose=false,
    )

    x = ComplexF64[cis(0.7 * i) * (1 + 0.5 * cospi(i / 17)) for i in 1:numfunctions(X)]
    dense = BEAST.assemble(efie, X, X; quadstrat=quadstrat)
    reference = dense * x
    @test norm(A * x - reference) / norm(reference) < 1e-4

    # the sub-strategies actually reached the map
    @test MLFMA.ϵ(A) == 1e-4
    @test H2Trees.tree(A) === tree

    # non-vacuity: a genuinely complex input, and a reference with something in it
    @test maximum(abs, imag.(x)) > 0.1
    @test norm(reference) > 0
end

# `nearinteractionsstrategyfor` maps an assembler TYPE onto H2Factory's strategy protocol -- kept as a
# migration convenience now that the keyword surface is gone. Pinned for both assemblers MLFMA callers
# actually pass, plus the refusal to guess.
@testset "nearinteractionsstrategyfor covers the assemblers callers pass" begin
    quadstrategy = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)

    @test MLFMA.nearinteractionsstrategyfor(
        H2Factory.SparseBEASTNearInteractionsAssembler, quadstrategy
    ) isa MLFMA.SparseBEASTNearInteractionsStrategy
    @test MLFMA.nearinteractionsstrategyfor(
        H2Factory.DummyNearInteractionsAssembler, quadstrategy
    ) isa H2Factory.DummyNearInteractionsStrategy
    @test MLFMA.nearinteractionsstrategyfor(
        H2Factory.BlockBEASTNearInteractionsAssembler, quadstrategy
    ) isa H2Factory.BlockBEASTNearInteractionsStrategy

    # An instantiation resolves too, not only the bare parametric name -- the `Type{<:...}` vs
    # `Type{...}` trap that once disabled every `FarField` interpolation trait.
    @test MLFMA.nearinteractionsstrategyfor(
        H2Factory.SparseBEASTNearInteractionsAssembler{ComplexF64}, quadstrategy
    ) isa MLFMA.SparseBEASTNearInteractionsStrategy

    # ...and an unknown assembler errors rather than silently picking one, which would give the map a
    # near block that does not match what was asked for.
    @test_throws ErrorException MLFMA.nearinteractionsstrategyfor(
        H2Factory.RandomNearInteractionsAssembler, quadstrategy
    )
end

# A materialized `ExecutionScheduling` is REJECTED as `scheduler`, at construction. This has to be an
# explicit check: `H2Factory.executionscheduling(e::ExecutionScheduling, levels)` returns `e` unchanged,
# so merely not handling the case would silently pass it through and reintroduce the hazard -- a
# scheduling built for one level range reused with a deeper tree, failing mid-traversal with a `KeyError`
# on the deepest level instead of at construction.
@testset "A materialized ExecutionScheduling is rejected as scheduler" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    materialized = H2Factory.executionscheduling(SerialScheduler(), 1:9)
    @test_throws ErrorException MLFMA.MLFMAStrategy(efie, X; scheduler=materialized)

    # non-vacuity: the bare scheduler it was built from IS accepted, so the throw is about the wrapper
    # and not about the argument being a scheduler at all
    @test MLFMA.MLFMAStrategy(efie, X; scheduler=SerialScheduler()) isa MLFMA.MLFMAStrategy

    # ...and H2Factory really would have passed it through, which is why the check is needed
    @test H2Factory.executionscheduling(materialized, 1:3) === materialized

    # materializing always uses the range it is given, so a deeper tree cannot inherit a shallower range
    strategy = MLFMA.MLFMAStrategy(efie, X; scheduler=SerialScheduler())
    @test length(getfield(MLFMA.executionschedulingfor(strategy, 1:3).outer, 1)) == 3
    @test length(getfield(MLFMA.executionschedulingfor(strategy, 1:9).outer, 1)) == 9
end

# The removed keyword surface must be REJECTED, not silently ignored. This is the flip side of deleting
# the shim: a caller still passing `NearInteractionsAssembler=...` or `ϵ=...` has to find out, rather
# than get a map built with defaults that quietly ignores what they asked for.
#
# `MethodError` is the right failure -- these are simply not keywords of the constructor any more.
@testset "The removed pre-Phase-1 keywords are rejected, not ignored" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    for removed in (
        (; ϵ=1e-3),
        (; minhalfsize=1.0),
        (; minvalues=10),
        (; NearInteractionsAssembler=H2Factory.DummyNearInteractionsAssembler),
        (; FarFieldAssembler=MLFMA.DummyFarFieldAssembler),
        (; TranslationCollection=MLFMA.DummyTranslationCollection),
        (; Interpolator=MLFMA.SphericalLagrangeInterpolator{15,ComplexF64}),
        (; PhaseShifter=MLFMA.DummyPhaseShifter),
        (; samplingfunction=MLFMA.SamplingFunctor(MLFMA.SAMPLINGBUILDERRECOM)),
        (; isnear=H2Trees.isnear(; additionalbufferboxes=1)),
        (; aggregationmode=H2Trees.AggregateMode()),
        (; farfieldquadstrategy=BEAST.SingleNumQStrat(1)),
        (; nearinteractionsquadstrategy=BEAST.SingleNumQStrat(1)),
        # The remaining three the docstring lists as removed. Easy to leave out of this list precisely
        # because nothing else mentions them, which is why the list is kept in step with the docstring.
        (; translatingnodesiterator=MLFMA.TRANSLATINGNODESRECOM(; isnear=H2Trees.isnear())),
        (; aggregatenode=H2Trees.aggregateallnodes()),
        (; scheduling=H2Factory.executionscheduling(SerialScheduler(), 1:9)),
    )
        @test_throws MethodError MLFMA.assemble(efie, X; removed...)
        @test_throws MethodError MLFMA.assemble(efie, X, X; removed...)
    end

    # non-vacuity: the same calls WITHOUT those keywords are accepted, so the throws above are about the
    # keywords and not about the fixture. Stubbed so this stays cheap.
    #
    # Checked on the WORKSPACE type and its core, not on `MLFMA.GalerkinMLFMA`: since Phase 5b that name
    # is a function rather than a type, so `isa MLFMA.GalerkinMLFMA` raises a `TypeError` instead of
    # asserting anything. The core check is what keeps this specific to MLFMA -- a bare
    # `isa GalerkinH2Workspace` would pass for an iFMM map too.
    accepted = MLFMA.assemble(efie, X; strategy=stubbedstrategy(efie, X))
    @test accepted isa H2Factory.GalerkinH2Workspace
    @test H2Factory.core(accepted) isa MLFMA.GalerkinMLFMACore
end

# The leaf-storage policy is a four-member family, PER SIDE, and every member is reachable from both
# map constructors on both sides. Asserted rather than assumed because it briefly was not: the
# compressed builders used to be radiation-only, which looks like it rules out a Petrov map whose test
# side is assembled, and the resemblance is close enough that a guard forbidding the combination was
# written before it was measured. It works -- see `test/Maps/petrov.jl`.
@testset "Leaf storage: four policies, both sides, both constructors" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    storages = (
        MLFMA.SampledLeafStorage(),
        MLFMA.SphericalHarmonicLeafStorage(),
        MLFMA.VectorSphericalHarmonicLeafStorage(),
        MLFMA.CircularHarmonicLeafStorage(),
    )

    strategywith(; kwargs...) =
        MLFMA.MLFMAStrategy(efie, X; farfieldstrategy=MLFMA.FarFieldStrategy(; kwargs...))

    for storage in storages
        @test storage isa MLFMA.AbstractLeafStorage
        # Every one of them is a legal FIELD -- the strategy carries a policy, it does not validate
        # reachability, and pushing that check earlier would mean `FarFieldStrategy` knowing which
        # collections exist.
        strategy = strategywith(; leafstorage=storage)
        # `leafstorage=` IS A SHORTHAND FOR BOTH, which is what keeps every existing call site
        # meaning what it did before the field was split.
        @test MLFMA.trialleafstorage(strategy) === storage
        @test MLFMA.testleafstorage(strategy) === storage
    end

    # THE ACCESSOR THAT NO LONGER EXISTS. `leafstorage(strategy)` would have to pick a side to report
    # the moment the two differ, and a reading that is right in the common case and silently wrong in
    # the case the split exists for is worse than no reading at all.
    @test !isdefined(MLFMA, :leafstorage)

    # The default is unchanged, and that is a decision rather than an omission: compressed leaf
    # storage measures 1.67x SLOWER at ordinary 2-D leaf occupancy, so it must not become the
    # recommendation. See `CircularHarmonicLeafStorage`'s docstring for the crossover table.
    @test MLFMA.LEAFSTORAGERECOM === MLFMA.SampledLeafStorage()
    @test MLFMA.trialleafstorage(strategywith()) === MLFMA.LEAFSTORAGERECOM
    @test MLFMA.testleafstorage(strategywith()) === MLFMA.LEAFSTORAGERECOM

    @testset "the two sides are genuinely independent" begin
        # THE POINT OF THE SPLIT. Every ordered pair is settable, including the 12 mixed ones, and
        # each side reports only its own -- which is the property a single field could not have.
        for trial in storages, test in storages
            strategy = strategywith(; trialleafstorage=trial, testleafstorage=test)
            @test MLFMA.trialleafstorage(strategy) === trial
            @test MLFMA.testleafstorage(strategy) === test
        end

        # Setting one alone leaves the other at the recommendation rather than following it. Worth
        # pinning: the obvious `leafstorage=trialleafstorage` fallback would make a one-sided setting
        # silently two-sided, which is the opposite of what the keyword is for.
        onesided = strategywith(; trialleafstorage=MLFMA.CircularHarmonicLeafStorage())
        @test MLFMA.trialleafstorage(onesided) === MLFMA.CircularHarmonicLeafStorage()
        @test MLFMA.testleafstorage(onesided) === MLFMA.LEAFSTORAGERECOM
    end

    # ALL FOUR yield a radiation builder, from the TRIAL side -- slice 3 of #138 replaced the guard
    # that used to stand where `CircularHarmonicLeafStorage`'s builder is.
    for storage in storages
        strategy = strategywith(; trialleafstorage=storage)
        @test MLFMA.radiationfieldcollectionbuilder(strategy) !== nothing
        @test MLFMA.side(MLFMA.radiationfieldcollectionbuilder(strategy)) ==
            MLFMA.RadiationSide()
    end

    # AND ALL FOUR give an assembled (Petrov) receive builder, from the TEST side. There is no
    # per-storage restriction here, and two earlier versions of this file asserted one: first that a
    # compressed trial side forbade Petrov entirely, then that a Petrov test side was always SAMPLED.
    # Both were wrong. `H2Factory.SidedChannels{Test,Trial}` is independently parameterized, and an
    # assembled compressed receive side needs no mirror and no coupling -- see
    # `PetrovGalerkinMLFMACore`'s `buildleafchannels` and `test/Maps/petrov.jl`.
    for storage in storages
        strategy = strategywith(; testleafstorage=storage)
        builder = MLFMA.receivefieldcollectionbuilder(strategy)
        @test MLFMA.side(builder) == MLFMA.ReceiveSide()
    end

    # THE TWO BUILDERS READ DIFFERENT SIDES, which is the wiring the split exists to produce. A
    # strategy that compresses only the trial side must give a compressed radiation builder and a
    # SAMPLED receive one -- and the reverse for the mirror-image setting.
    mixed = strategywith(;
        trialleafstorage=MLFMA.CircularHarmonicLeafStorage(),
        testleafstorage=MLFMA.SampledLeafStorage(),
    )
    @test MLFMA.radiationfieldcollectionbuilder(mixed) isa
        MLFMA.CircularHarmonicFieldCollectionBuilder
    @test MLFMA.receivefieldcollectionbuilder(mixed) isa MLFMA.BEASTFieldCollectionBuilder

    flipped = strategywith(;
        trialleafstorage=MLFMA.SampledLeafStorage(),
        testleafstorage=MLFMA.CircularHarmonicLeafStorage(),
    )
    @test MLFMA.radiationfieldcollectionbuilder(flipped) isa
        MLFMA.BEASTFieldCollectionBuilder
    @test MLFMA.receivefieldcollectionbuilder(flipped) isa
        MLFMA.CircularHarmonicFieldCollectionBuilder

    @testset "Galerkin refuses a per-side setting it cannot honour" begin
        # Its test side is MIRRORED, so a differing `testleafstorage` has nothing to act on. The
        # check has to raise rather than ignore: a knob that reads back the value it was given and
        # changes nothing is the failure family this branch keeps removing.
        matched = strategywith(; leafstorage=MLFMA.SphericalHarmonicLeafStorage())
        @test MLFMA.assertsingleleafstorage(matched, "GalerkinMLFMA") === nothing

        for (trial, test) in (
            (MLFMA.SphericalHarmonicLeafStorage(), MLFMA.SampledLeafStorage()),
            (MLFMA.SampledLeafStorage(), MLFMA.SphericalHarmonicLeafStorage()),
            (
                MLFMA.SphericalHarmonicLeafStorage(),
                MLFMA.VectorSphericalHarmonicLeafStorage(),
            ),
        )
            strategy = strategywith(; trialleafstorage=trial, testleafstorage=test)
            @test_throws ErrorException MLFMA.assertsingleleafstorage(
                strategy, "GalerkinMLFMA"
            )
        end

        # And it raises from the CONSTRUCTOR, not only from the helper -- which is the arrangement
        # that matters, since nothing else would call the helper. Matched on the MESSAGE, not just on
        # `ErrorException`: `_galerkinmlfmacore` runs `checkboxsizes` and `checklazyreceive` first and
        # both raise the same type, so a type-only assertion would pass on the wrong error.
        @test_throws "testleafstorage" MLFMA.GalerkinMLFMA(
            efie,
            X;
            strategy=MLFMA.MLFMAStrategy(
                efie,
                X;
                farfieldstrategy=MLFMA.FarFieldStrategy(;
                    trialleafstorage=MLFMA.SphericalHarmonicLeafStorage(),
                    testleafstorage=MLFMA.SampledLeafStorage(),
                ),
            ),
            verbose=false,
        )
    end
end
