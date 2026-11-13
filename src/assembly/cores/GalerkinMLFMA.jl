# MLFMA's Galerkin map has a read-only core built here and a mutable workspace that
# `H2Factory.H2Workspace` allocates over it.
#
# The core holds construction data that multiplication never changes. The workspace owns everything
# an in-flight `mul!` writes: moments and their pooled backing, leaf vector buffers, level-transfer
# and I2O scratch channels, and materialized schedules.
#
# Several workspaces can share one core, sequentially or concurrently. `test/Maps/workspace.jl`
# checks both uses against a dense reference and checks that concurrent products agree bitwise
# with isolated products.
#
# The interpolator and anterpolator live on the core: H2Factory forwards the per-side I2I accessors
# there, and each `InterpolationPhaseTransfer` stores its interpolator. Writable transfer arrays
# arrive through `context.buffer`, borrowed per (level, chunkid) from the workspace. This keeps
# interim scratch separate between workspaces and sizes it to their schedules. See
# `src/interpolation/interimscratch.jl`. A single workspace is exclusive to one in-flight `mul!`.

"""
    GalerkinMLFMACore{T} <: H2Factory.GalerkinH2Core{T}

The read-only half of a Galerkin MLFMA map: everything built once by
[`GalerkinMLFMA`](@ref) and never written to again.

Not executable on its own. It owns no moments and no scratch. `H2Factory.H2Workspace(core, scheduler)` produces the `LinearMap` that runs a matvec, which is what [`GalerkinMLFMA`](@ref)
returns.

The field collections dominate the size of this object, which is why they are the part worth
keeping read-only and shared.
"""
struct GalerkinMLFMACore{
    T,
    TreeType,
    NearInteractionsType,
    Space,
    RadiationFieldCollections,
    ReceiveFieldCollections,
    InterType,
    AnterType,
    LevelTranslatorsType,
    SamplingType,
    PhaseShifter,
    I2OTranslatorType,
    OperatorType,
    PlansType,
    ISNEAR,
    αType,
    FarMulType,
    SchedulerType,
} <: H2Factory.GalerkinH2Core{T}
    γ::T #TODO: Remove (also from PretrovGalerkinMLFMA)
    operator::OperatorType
    ϵ::AbstractFloat
    tree::TreeType
    nearinteractions::NearInteractionsType
    space::Space
    trialfieldcollections::RadiationFieldCollections
    testfieldcollections::ReceiveFieldCollections
    i2otranslator::I2OTranslatorType
    phaseshifter::PhaseShifter
    samplings::Vector{SamplingType}
    interpolator::InterType
    anterpolator::AnterType
    # `(trial=..., test=...)` I2I level-transfer collections. Two, not one: see
    # `interpolationphasecollections`. The `phaseshifter` above is retained as the source they are
    # built from (and for direct phase queries in tests), not as a second traversal path.
    leveltranslators::LevelTranslatorsType
    minrelevantlevel::Int
    verbose::Bool
    # The whole plan set, not four separate fields as the pre-split map stored them:
    # `H2Factory.plans(core)` defaults to `core.plans` and `H2Workspace` reads it to materialize each
    # workspace's own `Schedules`. It also carries `relevantlevels`, which the channel hooks below
    # need and which the pre-split constructor could keep in a local.
    plans::PlansType
    isnear::ISNEAR
    α::αType
    # The adjoint-mode `α`: `conj.(α)`. `adjoint(α·M) = conj(α)·adjoint(M)`, so the adjoint needs the
    # conjugated coefficient while transpose does not. Precomputed and stored, same type as `α`, so
    # the hot leaf multiply allocates nothing even when `α` is a `Vector` (hypersingular, MW single
    # layer). Mirrors iFMM's `αadjoint`.
    αadjoint::αType
    farmulmode::FarMulType
    # The bare scheduler the strategy asked for, NOT scheduling state, which is per workspace. It is
    # the default a workspace inherits when one is built without an explicit scheduler, so
    # `GalerkinMLFMA(...)` still honours `MLFMAStrategy`'s choice without the workspace reaching back
    # into a strategy object. Mirrors iFMM's `defaultscheduler`.
    defaultscheduler::SchedulerType
end

# The leaf-multiply default, on the path `GalerkinMLFMA(operator, space)` actually takes: that entry
# point builds this core and hands it to `H2Factory.H2Workspace(core, core.defaultscheduler)`, which
# materializes the scheduling itself without consulting the strategy. So MLFMA's parallel default has
# to be reachable from the CORE, not only from `executionschedulingfor`. See
# `MLFMA.defaultleafmulscheduler` for why the default is parallel rather than serial.
H2Factory.leafmulscheduler(::GalerkinMLFMACore) = defaultleafmulscheduler()

"""
    assemble(operator, space; strategy=MLFMAStrategy(operator, space), tree=..., verbose=false)

Convenience alias for [`GalerkinMLFMA`](@ref).
"""
function assemble(operator, space; kwargs...)
    return GalerkinMLFMA(operator, space; kwargs...)
end

"""
    GalerkinMLFMA(operator, space; strategy=MLFMAStrategy(operator, space), tree=..., verbose=false)

Assemble a Galerkin MLFMA map: builds a [`GalerkinMLFMACore`](@ref) and returns one
`H2Factory.H2Workspace` over it, which is the executable `LinearMap`.

A single workspace is exclusive to one in-flight `mul!`. For concurrent products, build further
workspaces over the same core with `H2Factory.H2Workspace(core, scheduler)`. They share no mutable
state, so they may run at the same time. See the note at the top of this file.

Every construction choice comes from `strategy`, read through the accessors in
`strategies/strategies.jl`.

Only `strategy`, `tree` and `verbose` are accepted as keywords.
"""
function GalerkinMLFMA(operator, space; kwargs...)
    # Keyword defaults live on the CORE constructor only, so the two entry points cannot drift.
    core = GalerkinMLFMACore(operator, space; kwargs...)
    # `H2Workspace` materializes the core's default scheduler over `schedulinglevels(core)`.
    return H2Factory.H2Workspace(core, core.defaultscheduler)
end

"""
    GalerkinMLFMACore(operator, space; strategy=MLFMAStrategy(operator, space), tree=..., verbose=false)

Build the read-only core alone, without a workspace.

Takes exactly the keywords [`GalerkinMLFMA`](@ref) documents. It is where their defaults are
defined, and `GalerkinMLFMA` forwards to it. Use this when you want the construction and the
executable object separated, e.g. to inspect the core or to time construction apart from allocation.
`H2Factory.H2Workspace(core, scheduler)` then produces something that can multiply, and may be called
more than once for one core. Those workspaces are independent and may be used concurrently, one
in-flight `mul!` each. See the note at the top of this file.
"""
function GalerkinMLFMACore(
    operator,
    space;
    strategy::AbstractMLFMAStrategy=MLFMAStrategy(operator, space),
    tree=buildtree(
        space;
        builder=TwoNTreeBuilder(;
            # Qualified: `minhalfsize`/`minvalues` are no longer keywords here, but staying qualified
            # keeps this readable next to the Petrov constructor, which does the same.
            minhalfsize=Assembly.minhalfsize(strategy),
            minvalues=Assembly.minvalues(strategy),
        ),
    ),
    verbose=VERBOSERECOM,
)
    return _galerkinmlfmacore(tree, space, operator, strategy, verbose)
end

function _galerkinmlfmacore(tree, space, operator, strategy::AbstractMLFMAStrategy, verbose)
    # A tree handed in directly bypasses `TreeStrategy`'s `minhalfsize` default, which is where
    # sub-λ/4 boxes get built by accident. See `MLFMA.checkboxsizes`.
    Assembly.checkboxsizes(tree, operator)
    # A Galerkin map MIRRORS its test side, so it needs a lazy receive rule. Checked HERE rather
    # than left to the first matvec: without this the missing method surfaces as a bare
    # `MethodError` from inside a traversal, long after construction reported success.
    Operators.checklazyreceive(operator)
    # And it has ONE leaf storage, for the same reason: the test side is DERIVED from the trial
    # patterns, so a `testleafstorage` that differs would be a setting with nothing to act on. Checked
    # here rather than dropped, because a knob that silently does nothing is the harder failure to
    # find, the map builds, runs, and reports an accuracy unrelated to what was asked for.
    Assembly.assertsingleleafstorage(strategy, "GalerkinMLFMA")
    @assert H2Trees.numberoflevels(tree) > 2
    γ = Operators.γ(operator)
    ϵ = Assembly.ϵ(strategy)
    nearfieldstrategy = Assembly.nearfieldstrategy(strategy)

    verbose && @info "Assembling near matrix"
    # H2Factory's own near-assembly step: builds the assembler from the strategy via
    # `buildnearinteractionsassembler` and applies it with the near-field strategy's `isnear`, so the
    # near block and the traversal plan below cannot disagree on the split.
    nearinteractions = H2Factory.assemblenearinteractions(
        operator,
        space,
        space,
        tree,
        γ,
        Assembly.nearinteractionsstrategy(strategy),
        nearfieldstrategy;
        verbose=verbose,
    )

    verbose && @info "Calculating far fields on leaf-level"

    # H2Factory selects the plan construction from `treetrait`, for both this tree and Petrov's BlockTree.
    galerkinplans = H2Factory.assembleplans(tree, nearfieldstrategy)
    trialaggregationplan = galerkinplans.trialaggregationplan
    testdisaggregationplan = galerkinplans.testdisaggregationplan
    testaggregationplan = galerkinplans.testaggregationplan

    translatingplan = H2Trees.translatingplan(trialaggregationplan, testdisaggregationplan)

    relevantlevels = galerkinplans.relevantlevels
    minrelevantlevel = relevantlevels[begin]

    samplings, parentsamplings = getsamplingsparentsamplings(
        γ, ϵ, tree, relevantlevels, Assembly.samplingfunction(strategy, tree, space)
    )

    # Both sides through one protocol, with the strategy naming the policy. The trial side assembles
    # (`RadiationSide`); the test side MIRRORS what the trial side just built rather than assembling
    # a second time, which is what makes this map Galerkin, hence the radiation dictionary in the
    # `source` position. Both builders come from the same `FarFieldStrategy`, so they cannot end up
    # carrying different quadratures or different assemblers.
    radiationfieldcollectionsdict = getfieldcollectionsdict(
        Assembly.radiationfieldcollectionbuilder(strategy),
        operator,
        space,
        tree,
        samplings,
        minrelevantlevel;
        verbose=verbose,
    )

    receivefieldcollectionsdict = getfieldcollectionsdict(
        Assembly.lazyreceivefieldcollectionbuilder(strategy),
        operator,
        radiationfieldcollectionsdict,
        tree,
        samplings,
        minrelevantlevel;
        verbose=verbose,
    )

    # Workspace hooks allocate per-chunk scratch inside `H2Workspace`'s `withschedulechunks` block.
    # FFT planning is independent of chunk counts; mutable interim scratch belongs to the workspace.
    interpolator = Interpolation.interpolatortype(strategy)(
        samplings[2:end],
        parentsamplings[2:end];
        interpolationtrait=Interpolation.interpolationtrait(operator),
    )
    anterpolator = Interpolation.anterpolator(interpolator)
    phaseshifter = Assembly.phaseshiftertype(strategy)(
        γ, tree, relevantlevels, parentsamplings[2:end]
    )

    # Each transfer carries the interpolation trait, while the interpolator was built with the trait
    # above. Checked rather than assumed: a divergence would flip the θ-sign convention on the φ
    # component silently, with no failure anywhere. `error` rather than `@assert` so it cannot be
    # compiled out. This runs once per construction, so the check is free.
    interpolationtraitofoperator = Interpolation.interpolationtrait(operator)
    if interpolationtraitofoperator != Interpolation.interpolationtrait(interpolator)
        error(
            "interpolation trait mismatch: the operator asks for " *
            "$(interpolationtraitofoperator) but the interpolator was built with " *
            "$(Interpolation.interpolationtrait(interpolator)). The level transfers would apply a " *
            "different component-sign convention than the interpolator.",
        )
    end

    # Built from the INTERPOLATOR's trait, not the operator's: a transfer has to apply the same
    # component-sign convention as the interpolator it wraps. The check above is what guarantees the
    # two agree here.
    trialleveltranslator, testleveltranslator = Phase.interpolationphasecollections(
        phaseshifter,
        interpolator,
        anterpolator,
        samplings,
        relevantlevels[begin],
        relevantlevels,
        Interpolation.interpolationtrait(interpolator),
    )
    leveltranslators = (trial=trialleveltranslator, test=testleveltranslator)

    translationsamplings = samplings[(H2Trees.mintranslationlevel(translatingplan) - relevantlevels[begin] + 1):end]

    verbose && @info "Computing translations"
    # The second return value is the I2O scratch channel, which is per workspace and therefore not
    # kept: `buildi2ochannel` below rebuilds it from the stored collection through H2Factory's own
    # `i2ochannels` seam. Discarding it here is what makes the core hold read-only data only.
    translationcollection, _ = Assembly.translationcollectiontype(strategy)(
        γ,
        tree,
        translatingplan,
        translationsamplings;
        levelscales=getweights(translationsamplings),
        verbose=verbose,
    )

    return GalerkinMLFMACore(
        γ,
        operator,
        ϵ,
        tree,
        nearinteractions,
        space,
        radiationfieldcollectionsdict,
        receivefieldcollectionsdict,
        translationcollection,
        phaseshifter,
        samplings,
        interpolator,
        anterpolator,
        leveltranslators,
        minrelevantlevel,
        verbose,
        galerkinplans,
        H2Factory.nearcriterion(nearfieldstrategy)(tree),
        Operators.α(operator),
        conj.(Operators.α(operator)),
        H2Factory.aggregationmode(nearfieldstrategy),
        Assembly.scheduler(strategy),
    )
end

# --- Read-only accessors --------------------------------------------------------------------------
#
# Workspaces forward read-only accessors as `(core, maptrait)`. Most generic methods read the
# matching core fields directly; methods restricted to a map supertype need overrides here.

# MLFMA's level lookups also call this accessor directly on a core.
H2Factory.minrelevantlevel(A::GalerkinMLFMACore) = A.minrelevantlevel

# Adjoint multiplication conjugates α; transpose leaves it unchanged.
H2Factory.α(A::GalerkinMLFMACore, ::H2Factory.isForward) = A.α
H2Factory.α(A::GalerkinMLFMACore, ::H2Factory.isTranspose) = A.α
H2Factory.α(A::GalerkinMLFMACore, ::H2Factory.isAdjoint) = A.αadjoint

# Without this override, H2Trees' generic method would treat the core as a tree.
H2Trees.isnear(A::GalerkinMLFMACore, ::H2Factory.isForward) = A.isnear

# Unpack the stored plan set; H2Factory's defaults expect separate fields.
H2Factory.testaggregationplan(A::GalerkinMLFMACore, ::H2Factory.isForward) =
    A.plans.testaggregationplan
H2Factory.trialaggregationplan(A::GalerkinMLFMACore, ::H2Factory.isForward) =
    A.plans.trialaggregationplan
H2Factory.testdisaggregationplan(A::GalerkinMLFMACore, ::H2Factory.isForward) =
    A.plans.testdisaggregationplan
H2Factory.trialdisaggregationplan(A::GalerkinMLFMACore, ::H2Factory.isForward) =
    A.plans.trialdisaggregationplan

# Override the forward branch, leaving H2Factory to select the side for each orientation.
H2Factory.testmomentcollections(A::GalerkinMLFMACore, ::H2Factory.isForward) =
    A.testfieldcollections
H2Factory.trialmomentcollections(A::GalerkinMLFMACore, ::H2Factory.isForward) =
    A.trialfieldcollections

# All orientations share one I2O collection; the functor carries the direction.
# Concrete traits avoid ambiguity with H2Factory's trait-specific generic methods.
H2Factory.i2otranslator(A::GalerkinMLFMACore, ::H2Factory.isForward) = A.i2otranslator
H2Factory.i2otranslator(A::GalerkinMLFMACore, ::H2Factory.isTranspose) = A.i2otranslator
H2Factory.i2otranslator(A::GalerkinMLFMACore, ::H2Factory.isAdjoint) = A.i2otranslator

# Keep separate trial and test phases. Forward disaggregation uses the transpose of the test
# transfer, which does not conjugate its diagonal. O2O needs no separate translator override.
H2Factory.triali2itranslator(A::GalerkinMLFMACore) = A.leveltranslators.trial
H2Factory.testi2itranslator(A::GalerkinMLFMACore) = A.leveltranslators.test

# --- Workspace construction hooks ----------------------------------------------------------------
#
# H2Workspace calls these hooks inside `withschedulechunks`, which supplies per-level chunk counts.

# Scheduling containers are indexed by level, so span the deepest index, not just the level count.
H2Factory.schedulinglevels(A::GalerkinMLFMACore) = 1:H2Trees.levels(A.tree)[end]

H2Factory.buildmoments(A::GalerkinMLFMACore) = _corefarfieldmoments(A)

# One per-node dict serving both sides: a Galerkin map has one tree, so a second copy would only alias
# the first.
H2Factory.buildvectorbuffers(A::GalerkinMLFMACore) =
    H2Trees.computevectorbuffers(A.tree, complex(typeof(A.γ)))

H2Factory.buildi2ichannels(A::GalerkinMLFMACore) = _corei2ichannel(A)

H2Factory.buildo2ochannels(A::GalerkinMLFMACore) = _coreo2ochannel(A)

H2Factory.buildi2ochannel(A::GalerkinMLFMACore) = _buildi2ochannel(A)

# One leaf channel serves both sides of the shared tree. Select scratch by collection type:
# sampled and compressed collections need different buffers per (level, chunk).
H2Factory.buildleafchannels(A::GalerkinMLFMACore) =
    _buildleafchannels(A, A.trialfieldcollections)

_buildleafchannels(A::GalerkinMLFMACore, ::Any) = Assembly.leafscratchchannel(
    complex(A.γ),
    farfieldtype(A.operator),
    Assembly._leaflevels(A.tree),
    A.plans,
    A.samplings,
    A.minrelevantlevel,
)

# `groupsperchunk` is the batch size, and undersizing it silently drops the leaf multiply into its
# per-group fallback, which measures 1.63x SLOWER than the sampled path, i.e. worse than not having
# the feature at all. `leavesperchunk` is the upper bound `PlanSchedule`'s contiguous split allows.
_buildleafchannels(
    A::GalerkinMLFMACore,
    collections::AbstractDictionary{<:Any,<:SphericalHarmonicRadiationCollection},
) = Assembly.shescratchchannel(
    Assembly.sheorders(A.tree, collections, A.samplings, A.minrelevantlevel),
    Assembly._leaflevels(A.tree),
    A.plans,
    A.samplings,
    A.minrelevantlevel;
    groupsperchunk=level ->
        Assembly.leavesperchunk(A.tree, level, H2Factory.nchunks(level)),
)

# The 2-D backend. `ncomponents` is read off a COLLECTION rather than from `fielddim(farfieldtype(op))`,
# the two agree today, and reading the data means they cannot disagree tomorrow: the scratch's
# component count has to match what the collections actually store, and a mismatch would be a GEMM
# whose shapes still conform whenever the scratch is the narrower of the two.
#
# `groupsperchunk` sizes the batch. Undersizing costs less here than in 3-D: the 2-D batch has no
# threshold below which it becomes a regression (it wins from two leaves upward and ties at one), so a
# small batch simply flushes more often instead of falling off a cliff.
_buildleafchannels(
    A::GalerkinMLFMACore,
    collections::AbstractDictionary{<:Any,<:CircularHarmonicRadiationCollection},
) = Assembly.chescratchchannel(
    Assembly.cheorders(A.tree, collections, A.samplings, A.minrelevantlevel),
    Assembly._leaflevels(A.tree),
    A.plans,
    A.samplings,
    A.minrelevantlevel;
    ncomponents=FarFields.fielddim(typeof(first(collections))),
    groupsperchunk=level ->
        Assembly.leavesperchunk(A.tree, level, H2Factory.nchunks(level)),
)

_buildleafchannels(
    A::GalerkinMLFMACore,
    collections::AbstractDictionary{<:Any,<:VectorSphericalHarmonicRadiationCollection},
) = Assembly.vshscratchchannel(
    Assembly.vshorders(A.tree, collections, A.samplings, A.minrelevantlevel),
    Assembly._leaflevels(A.tree),
    A.plans,
    A.samplings,
    A.minrelevantlevel;
    groupsperchunk=level ->
        Assembly.leavesperchunk(A.tree, level, H2Factory.nchunks(level)),
)

"""
    _corefarfieldmoments(core)

The moment buffers and `H2Factory.Moments` for a core, shared by both map families.

Both derive them identically: one `Assembly.fields` call over the core's own tree, plans and
samplings. Written once here and called from each core's `buildmoments` hook rather than defined on
a `Union` of the two core types, so the METHOD TABLE still shows one hook per core.
`test/Package/h2factoryinterface.jl` pins those counts to catch a lost specialization, and a `Union`
method would have collapsed each pair to one, weakening exactly that check.

Same reason `_buildi2ochannel` above is a plain function rather than a shared method.
"""
function _corefarfieldmoments(core)
    buffers, testfarfields, trialfarfields = Assembly.fields(
        complex(core.γ),
        core.operator,
        core.tree,
        core.minrelevantlevel,
        core.plans.testaggregationplan,
        core.plans.trialaggregationplan,
        core.samplings,
    )
    return buffers, H2Factory.Moments(testfarfields, trialfarfields)
end

"""
    _corei2ichannel(core)
    _coreo2ochannel(core)

One interpolation or anterpolation scratch channel for a core.

The ARGUMENTS are identical for both map families; only what the caller does with the result
differs. A Galerkin core hands the channel back as it is, and a Petrov core wraps it as
`SidedChannels(channel, channel)` because its two sides share one buffer shape. That wrapping stays
at each hook, where its own comment explains why one channel serves both sides and why splitting
them per side is a separate slice.
"""
function _corei2ichannel(core)
    return Assembly.scratchi2ichannel(
        complex(core.γ),
        farfieldtype(core.operator),
        core.plans.relevantlevels,
        core.samplings,
        core.anterpolator,
    )
end

function _coreo2ochannel(core)
    return Assembly.scratcho2ochannel(
        complex(core.γ),
        farfieldtype(core.operator),
        core.plans.relevantlevels,
        core.samplings,
        core.anterpolator,
    )
end

"""
    _buildi2ochannel(core)

The `H2Factory.buildi2ochannel` hook for both MLFMA cores.

Goes through H2Factory's own `i2ochannels(translation, translatingplan)` seam rather than re-running the
translation constructor for the second return value the core constructors discard. Both of MLFMA's
current I2O collections need no scratch at all, `H2Factory.I2OTranslationCollection` and
`DummyTranslationCollection` each return `EmptyChannels()` alongside themselves, because a diagonal
translation has nothing to buffer, and `i2ochannels(::Any, translatingplan) = EmptyChannels()` is
exactly that answer. Going through the seam rather than hardcoding `EmptyChannels()` means a future
accelerated I2O (the H2Toeplitz route defines its own `i2ochannels` method, and its FFT plans MUST be
per workspace) is picked up without this hook changing.

Shared by the two cores because the translating plan is derived the same way for both: the trial
aggregation and test disaggregation plans, which is the pair `H2Factory.I2OTranslationCollection`
itself reduces to a translating plan.
"""
function _buildi2ochannel(core)
    return H2Factory.i2ochannels(
        core.i2otranslator,
        H2Trees.translatingplan(
            core.plans.trialaggregationplan, core.plans.testdisaggregationplan
        ),
    )
end
