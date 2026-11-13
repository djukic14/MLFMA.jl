# MLFMA's two-space map has a read-only core and a mutable `H2Factory.H2Workspace` over it.
# As in `GalerkinMLFMA.jl`, separate workspaces may share a core concurrently; each workspace is
# exclusive to one in-flight `mul!`.
#
# The tree, the spaces, the field collections, the
# moments, the leaf vector buffers and the level transfers are per side, because a `BlockTree`'s two
# sides are different trees with different node sets. Level-transfer scratch is shared: both sides'
# channels refer to one object. The buffers have the same shape (one
# `samplings` vector, sized for both trees and spaces, serves both), and aggregation and
# disaggregation are separate phases of one matvec, so sharing is safe.

"""
    PetrovGalerkinMLFMACore{T} <: H2Factory.PetrovGalerkinH2Core{T}

The read-only half of a two-space MLFMA map. See [`GalerkinMLFMACore`](@ref) for the core/workspace
contract; this differs only in being sided, two spaces, a `BlockTree`, and per-side field
collections.
"""
struct PetrovGalerkinMLFMACore{
    T,
    TError<:AbstractFloat,
    OperatorType,
    RadiationFieldCollectionDicts,
    ReceiveFieldCollectionDicts,
    TreeType,
    TestSpace,
    TrialSpace,
    NearInteractionsType,
    PhaseShifter,
    I2OTranslatorType,
    Interpolator,
    Anterpolator,
    LevelTranslatorsType,
    SamplingType,
    PlansType,
    ISNEAR,
    αType,
    FarMulType,
    SchedulerType,
} <: H2Factory.PetrovGalerkinH2Core{T}
    γ::T
    operator::OperatorType
    ϵ::TError
    tree::TreeType
    nearinteractions::NearInteractionsType
    testspace::TestSpace
    trialspace::TrialSpace
    trialfieldcollections::RadiationFieldCollectionDicts
    testfieldcollections::ReceiveFieldCollectionDicts
    # H2Factory's generic accessor reads this field by name, as on `GalerkinMLFMACore`.
    i2otranslator::I2OTranslatorType
    phaseshifter::PhaseShifter
    samplings::Vector{SamplingType}
    interpolator::Interpolator
    anterpolator::Anterpolator
    # `(trial=..., test=...)` I2I level-transfer collections, two, not one; see
    # `interpolationphasecollections`. For a BlockTree the two sides are genuinely different
    # trees as well as different phase conventions.
    leveltranslators::LevelTranslatorsType
    verbose::Bool
    minrelevantlevel::Int
    # The whole plan set, as on the Galerkin core: `H2Factory.plans(core)` reads it to materialize each
    # workspace's `Schedules`, and it also carries `relevantlevels` and `mintranslationlevel`, which the
    # hooks below need and the pre-split constructor could keep in locals.
    plans::PlansType
    isnear::ISNEAR
    α::αType
    # See `GalerkinMLFMACore`: the adjoint-mode `α`, precomputed as `conj.(α)`.
    αadjoint::αType
    farmulmode::FarMulType
    defaultscheduler::SchedulerType
end

# The Petrov twin of `GalerkinMLFMACore`'s override, same reason, same default. Declared on the
# concrete core rather than on `H2Factory.PetrovGalerkinH2Core`, which would be piracy and would
# capture iFMM's Petrov core too.
H2Factory.leafmulscheduler(::PetrovGalerkinMLFMACore) = defaultleafmulscheduler()
#TODO: look at CollisionDetection.jl for case that trial functions overlap

"""
    assemble(operator, testspace, trialspace; strategy=..., tree=..., verbose=false)

Convenience alias for [`PetrovGalerkinMLFMA`](@ref).
"""
function assemble(operator, testspace, trialspace; kwargs...)
    return PetrovGalerkinMLFMA(operator, testspace, trialspace; kwargs...)
end

"""
    PetrovGalerkinMLFMA(operator, testspace, trialspace;
                        strategy=MLFMAStrategy(operator, testspace, trialspace), tree=..., verbose=false)

The two-space map: builds a [`PetrovGalerkinMLFMACore`](@ref) and returns one
`H2Factory.H2Workspace` over it, which is the executable `LinearMap`.

Every construction choice comes from `strategy`, read through the accessors in
`strategies/strategies.jl`. See [`GalerkinMLFMA`](@ref) for the keyword-to-sub-strategy migration table.

`tree` defaults to a `BlockTree` with the same subdivision parameters on both sides, that is the
builder-type asymmetry [`TreeStrategy`](@ref) exists to keep out of the strategy, since a Petrov map is
legitimately built with the same space twice and so the arity cannot be inferred from the spaces.
"""
function PetrovGalerkinMLFMA(operator, testspace, trialspace; kwargs...)
    # Keyword defaults live on the CORE constructor only, so the two entry points cannot drift.
    core = PetrovGalerkinMLFMACore(operator, testspace, trialspace; kwargs...)
    return H2Factory.H2Workspace(core, core.defaultscheduler)
end

"""
    PetrovGalerkinMLFMACore(operator, testspace, trialspace; strategy=..., tree=..., verbose=false)

Build the read-only two-space core alone, without a workspace. The Petrov counterpart of
[`GalerkinMLFMACore`](@ref), and where the keywords [`PetrovGalerkinMLFMA`](@ref) documents get their
defaults.
"""
function PetrovGalerkinMLFMACore(
    operator,
    testspace,
    trialspace;
    strategy::AbstractMLFMAStrategy=MLFMAStrategy(operator, testspace, trialspace),
    tree=buildtree(
        testspace,
        trialspace;
        builder=BlockTreeBuilder(;
            test=TwoNTreeBuilder(;
                minhalfsize=Assembly.minhalfsize(strategy),
                minvalues=Assembly.minvalues(strategy),
            ),
            trial=TwoNTreeBuilder(;
                minhalfsize=Assembly.minhalfsize(strategy),
                minvalues=Assembly.minvalues(strategy),
            ),
        ),
    ),
    verbose=VERBOSERECOM,
)
    return _petrovgalerkinmlfmacore(
        tree, testspace, trialspace, operator, strategy, verbose
    )
end

function _petrovgalerkinmlfmacore(
    tree, testspace, trialspace, operator, strategy::AbstractMLFMAStrategy, verbose
)
    ϵ = Assembly.ϵ(strategy)
    nearfieldstrategy = Assembly.nearfieldstrategy(strategy)
    γ = Operators.γ(operator)

    verbose && @info "Assembling near matrix"
    nearinteractions = H2Factory.assemblenearinteractions(
        operator,
        testspace,
        trialspace,
        tree,
        γ,
        Assembly.nearinteractionsstrategy(strategy),
        nearfieldstrategy;
        verbose=verbose,
    )

    verbose && @info "Calculating far fields of trial tree"
    trialtree = H2Trees.trialtree(tree)
    testtree = H2Trees.testtree(tree)

    # `assembleplans` dispatches on `treetrait`, handling this BlockTree and Galerkin's TwoNTree.
    petrovplans = H2Factory.assembleplans(tree, nearfieldstrategy)
    trialaggregationplan = petrovplans.trialaggregationplan
    testdisaggregationplan = petrovplans.testdisaggregationplan
    relevantlevels = petrovplans.relevantlevels
    mintranslationlevel = petrovplans.mintranslationlevel
    minrelevantlevel = relevantlevels[begin]

    samplings, parentsamplings = getsamplingsparentsamplings(
        γ,
        ϵ,
        H2Trees.treewithmorelevels(tree),
        relevantlevels,
        # Both sides share this level vector, sized from their correctly paired tree and space.
        Assembly.samplingfunction(strategy, tree, testspace, trialspace),
    )

    # Both sides through one protocol, with the strategy naming the policy, the same two calls as
    # `GalerkinMLFMA` makes, differing only in which builder they ask for. Here the test side is
    # genuinely ASSEMBLED (`ReceiveSide`, over the test tree's own space) rather than mirrored: the
    # two trees hold different functions, so there is no trial pattern for the test side to mirror.
    trialfieldcollections = Assembly.getfieldcollectionsdict(
        Assembly.radiationfieldcollectionbuilder(strategy),
        operator,
        trialspace,
        trialtree,
        samplings,
        minrelevantlevel;
        verbose=verbose,
    )

    verbose && @info "Calculating far fields of test tree"
    testfieldcollections = Assembly.getfieldcollectionsdict(
        Assembly.receivefieldcollectionbuilder(strategy),
        operator,
        testspace,
        testtree,
        samplings,
        minrelevantlevel;
        verbose=verbose,
    )

    # Use the operator's component-sign convention. Planning is independent of workspace schedules.
    # Interim scratch belongs to the workspace per (level, chunk), not to the interpolator.
    interpolator = Interpolation.interpolatortype(strategy)(
        samplings[2:end],
        parentsamplings[2:end];
        interpolationtrait=Interpolation.interpolationtrait(operator),
    )
    anterpolator = Interpolation.anterpolator(interpolator)
    phaseshifter = Assembly.phaseshiftertype(strategy)(
        γ, tree, relevantlevels, parentsamplings[2:end]
    )

    # Built from the INTERPOLATOR's trait, so the transfers apply the same component-sign convention
    # as the interpolator they wrap, same check as `GalerkinMLFMA` makes.
    interpolationtraitofoperator = Interpolation.interpolationtrait(operator)
    if interpolationtraitofoperator != Interpolation.interpolationtrait(interpolator)
        error(
            "interpolation trait mismatch: the operator asks for " *
            "$(interpolationtraitofoperator) but the interpolator was built with " *
            "$(Interpolation.interpolationtrait(interpolator)). The level transfers would apply a " *
            "different component-sign convention than the interpolator.",
        )
    end

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

    translationsamplings = samplings[(mintranslationlevel - minrelevantlevel + 1):end]

    verbose && @info "Computing translations"
    # The discarded second return value is the I2O scratch channel, which is per workspace;
    # `buildi2ochannel` rebuilds it from the stored collection. See `_buildi2ochannel`.
    i2otranslator, _ = Assembly.translationcollectiontype(strategy)(
        γ,
        tree,
        trialaggregationplan,
        testdisaggregationplan,
        translationsamplings;
        levelscales=getweights(translationsamplings),
        verbose=verbose,
    )

    return PetrovGalerkinMLFMACore(
        γ,
        operator,
        ϵ,
        tree,
        nearinteractions,
        testspace,
        trialspace,
        trialfieldcollections,
        testfieldcollections,
        i2otranslator,
        phaseshifter,
        samplings,
        interpolator,
        anterpolator,
        leveltranslators,
        verbose,
        minrelevantlevel,
        petrovplans,
        H2Factory.nearcriterion(nearfieldstrategy)(tree),
        Operators.α(operator),
        conj.(Operators.α(operator)),
        H2Factory.aggregationmode(nearfieldstrategy),
        Assembly.scheduler(strategy),
    )
end

# `H2Trees.numberoflevels` doesn't support `BlockTree` directly (one side can have `minlevel > 1`), so
# the range is taken from both sides' raw top level, matching iFMM's `PetrovGalerkiniFMMCore`. Level
# INDICES, not a count. See `GalerkinMLFMACore`'s `schedulinglevels`.
#
# Keep level indices rather than a count; the two roots can start at different levels.
function _petrovschedulinglevels(tree)
    return 1:max(
        H2Trees.levels(H2Trees.testtree(tree))[end],
        H2Trees.levels(H2Trees.trialtree(tree))[end],
    )
end

# --- Read-only accessors --------------------------------------------------------------------------
#
# As on the Galerkin core: only the accessors whose forward branch is typed on a map SUPERTYPE need an
# override here, since H2Factory's other generic bodies are typed on the bare object and a core with the
# same field names satisfies them as-is. `i2otranslator` is absent from this list on purpose, the
# field rename above means the generic body finds it by name.

# MUST be defined here: H2Factory's forward body is `isnear(A::AbstractH2Map, ::isForward)`, which a
# core does not match, while `H2Trees.isnear(tree, ::Any; kwargs...)` DOES and silently returns a
# predicate built over the core as though it were a tree, a wrong answer, not an error.
H2Trees.isnear(A::PetrovGalerkinMLFMACore, ::H2Factory.isForward) = A.isnear

# The one-argument convenience arity, so MLFMA's level-indexed lookups work on a core. See the
# equivalent on `GalerkinMLFMACore`.
H2Factory.minrelevantlevel(A::PetrovGalerkinMLFMACore) = A.minrelevantlevel

# `α` per orientation. See `GalerkinMLFMACore` for why the adjoint branch conjugates and the transpose
# branch does not.
H2Factory.α(A::PetrovGalerkinMLFMACore, ::H2Factory.isForward) = A.α
H2Factory.α(A::PetrovGalerkinMLFMACore, ::H2Factory.isTranspose) = A.α
H2Factory.α(A::PetrovGalerkinMLFMACore, ::H2Factory.isAdjoint) = A.αadjoint

H2Factory.testaggregationplan(A::PetrovGalerkinMLFMACore, ::H2Factory.isForward) =
    A.plans.testaggregationplan
H2Factory.trialaggregationplan(A::PetrovGalerkinMLFMACore, ::H2Factory.isForward) =
    A.plans.trialaggregationplan
H2Factory.testdisaggregationplan(A::PetrovGalerkinMLFMACore, ::H2Factory.isForward) =
    A.plans.testdisaggregationplan
H2Factory.trialdisaggregationplan(A::PetrovGalerkinMLFMACore, ::H2Factory.isForward) =
    A.plans.trialdisaggregationplan

# Per side and genuinely different, unlike the Galerkin core where both sides are one tree: H2Factory's
# forward branch reads `A.testmomentcollections`/`A.trialmomentcollections` off a
# `PetrovGalerkinH2Map`, and a core is not one.
H2Factory.testmomentcollections(A::PetrovGalerkinMLFMACore, ::H2Factory.isForward) =
    A.testfieldcollections
H2Factory.trialmomentcollections(A::PetrovGalerkinMLFMACore, ::H2Factory.isForward) =
    A.trialfieldcollections

# The two sides store different phases, trial the radiation convention, test its conjugate, which is
# why they cannot be one collection. See `interpolationphasecollections`.
H2Factory.triali2itranslator(A::PetrovGalerkinMLFMACore) = A.leveltranslators.trial
H2Factory.testi2itranslator(A::PetrovGalerkinMLFMACore) = A.leveltranslators.test

# --- Workspace construction hooks ----------------------------------------------------------------

H2Factory.schedulinglevels(A::PetrovGalerkinMLFMACore) = _petrovschedulinglevels(A.tree)

H2Factory.buildmoments(A::PetrovGalerkinMLFMACore) = _corefarfieldmoments(A)

# Per side, because a `BlockTree`'s two sides are different trees with different node sets,
# `computevectorbuffers` already returns the pair.
function H2Factory.buildvectorbuffers(A::PetrovGalerkinMLFMACore)
    testvectorbuffer, trialvectorbuffer = H2Trees.computevectorbuffers(
        A.tree, complex(typeof(A.γ))
    )
    return H2Factory.VectorBuffers(testvectorbuffer, trialvectorbuffer)
end

# Share one channel between sides. One common `samplings` vector gives both
# sides the same buffer shape. Aggregation and disaggregation run in separate phases of a matvec,
# so the two sides do not borrow a buffer at the same time.
function H2Factory.buildi2ichannels(A::PetrovGalerkinMLFMACore)
    channel = _corei2ichannel(A)
    return H2Factory.SidedChannels(channel, channel)
end

function H2Factory.buildo2ochannels(A::PetrovGalerkinMLFMACore)
    channel = _coreo2ochannel(A)
    return H2Factory.SidedChannels(channel, channel)
end

H2Factory.buildi2ochannel(A::PetrovGalerkinMLFMACore) = _buildi2ochannel(A)

# Cover the union of both trees' leaf levels: either side may extend deeper than the other.
# A traversed leaf level without the required scratch pool fails during multiplication.
#
# Each assembled side may use a different storage policy. Select its pool from its collection type
# so the scratch matches the representation. `SidedChannels{Test,Trial}` can hold different types.
# With the trial side compressed, measured errors on a `‖Z-Zᵀ‖/‖Z‖ = 1.4` block at ε = 1e-5 were
# 9.07e-07 forward and 2.78e-07 adjoint.
#
# Two sampled sides share one pool by reference; `test/Maps/workspace.jl` checks this identity.
# Both use one `samplings` vector, so buffers have the same shape at a level, and their leaf
# multiplies run in barrier-separated phases. Separate pools would increase retained memory.
#
# Size compressed batches from each side's own tree: `groupsperchunk` counts leaves, and the two
# trees can have different node counts at the same level.
function H2Factory.buildleafchannels(A::PetrovGalerkinMLFMACore)
    leaflevels = sort!(
        unique!(
            vcat(
                Assembly._leaflevels(H2Trees.testtree(A.tree)),
                Assembly._leaflevels(H2Trees.trialtree(A.tree)),
            ),
        ),
    )
    return _petrovleafchannels(
        A,
        leaflevels,
        _petrovsidechannel(A, leaflevels, A.testfieldcollections, H2Trees.testtree(A.tree)),
        _petrovsidechannel(
            A, leaflevels, A.trialfieldcollections, H2Trees.trialtree(A.tree)
        ),
    )
end

# `nothing` means "this side needs no pool of its own", i.e. it is sampled, and takes the shared one.
# The four methods below are how that is turned back into a `SidedChannels` without ever building the
# sampled pool for a map that has no sampled side: it is the LARGEST of the three, so a fully compressed
# map allocating one would defeat the feature.
#
# The `(::Nothing, ::Nothing)` method is what keeps the all-sampled case sharing. It is strictly more
# specific than the two one-sided methods, so the three do not conflict even though those two are
# mutually ambiguous on their own.
_petrovleafchannels(A::PetrovGalerkinMLFMACore, leaflevels, ::Nothing, ::Nothing) =
    let channel = _petrovsampledchannel(A, leaflevels)
        H2Factory.SidedChannels(channel, channel)
    end

_petrovleafchannels(A::PetrovGalerkinMLFMACore, leaflevels, test, ::Nothing) =
    H2Factory.SidedChannels(test, _petrovsampledchannel(A, leaflevels))

_petrovleafchannels(A::PetrovGalerkinMLFMACore, leaflevels, ::Nothing, trial) =
    H2Factory.SidedChannels(_petrovsampledchannel(A, leaflevels), trial)

_petrovleafchannels(A::PetrovGalerkinMLFMACore, leaflevels, test, trial) =
    H2Factory.SidedChannels(test, trial)

_petrovsampledchannel(A::PetrovGalerkinMLFMACore, leaflevels) = Assembly.leafscratchchannel(
    complex(A.γ),
    farfieldtype(A.operator),
    leaflevels,
    A.plans,
    A.samplings,
    A.minrelevantlevel,
)

# Sampled collections need no pool of their own. See `_petrovleafchannels`.
_petrovsidechannel(A::PetrovGalerkinMLFMACore, leaflevels, ::Any, tree) = nothing

_petrovsidechannel(
    A::PetrovGalerkinMLFMACore,
    leaflevels,
    collections::AbstractDictionary{<:Any,<:SphericalHarmonicAssembledCollection},
    tree,
) = Assembly.shescratchchannel(
    Assembly.sheorders(tree, collections, A.samplings, A.minrelevantlevel),
    leaflevels,
    A.plans,
    A.samplings,
    A.minrelevantlevel;
    groupsperchunk=level ->
        Assembly.leavesperchunk(tree, level, H2Factory.nchunks(level)),
)

_petrovsidechannel(
    A::PetrovGalerkinMLFMACore,
    leaflevels,
    collections::AbstractDictionary{<:Any,<:VectorSphericalHarmonicAssembledCollection},
    tree,
) = Assembly.vshscratchchannel(
    Assembly.vshorders(tree, collections, A.samplings, A.minrelevantlevel),
    leaflevels,
    A.plans,
    A.samplings,
    A.minrelevantlevel;
    groupsperchunk=level ->
        Assembly.leavesperchunk(tree, level, H2Factory.nchunks(level)),
)

_petrovsidechannel(
    A::PetrovGalerkinMLFMACore,
    leaflevels,
    collections::AbstractDictionary{<:Any,<:CircularHarmonicAssembledCollection},
    tree,
) = Assembly.chescratchchannel(
    Assembly.cheorders(tree, collections, A.samplings, A.minrelevantlevel),
    leaflevels,
    A.plans,
    A.samplings,
    A.minrelevantlevel;
    ncomponents=FarFields.fielddim(typeof(first(collections))),
    groupsperchunk=level ->
        Assembly.leavesperchunk(tree, level, H2Factory.nchunks(level)),
)
