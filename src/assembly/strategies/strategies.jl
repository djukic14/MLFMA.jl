# Construction strategies group related choices and keep their defaults in one place.
# `MLFMAStrategy` combines MLFMA-specific policies with two H2Factory strategy protocols:
#
#   - `H2Factory.InteractionStrategy` (`<: AbstractInteractionStrategy`) holds
#     `isnear`/`translatingnodesiterator`/`aggregatenode`/`aggregationmode`, already derives the middle
#     two from `isnear` for the same consistency reason, and is consumed by
#     `H2Factory.assembleplans(tree, nearfieldstrategy)`, which dispatches on the tree's `treetrait`
#     and so handles MLFMA's `TwoNTree` (Galerkin) and `BlockTree` (Petrov) with one call.
#   - `H2Factory.AbstractNearInteractionsStrategy` selects the near assembler through
#     `buildnearinteractionsassembler`, consumed by `H2Factory.assemblenearinteractions`. Note this is
#     a strategy type selected by dispatch, not a stored assembler type called as a constructor.
#
# MLFMA's `isnear` default differs from H2Factory's: `InteractionStrategy` uses `H2Trees.isnear()` with no
# buffer boxes, while MLFMA's default is `BUFFERBOXRECOM`, so MLFMA passes it explicitly. The `*RECOM`
# constants remain the single source of truth for those values; they live in `AbstractMLFMAMatrix.jl`,
# which is included after this file, and that is fine because keyword defaults are evaluated at call
# time rather than at definition time.

"""
    AbstractMLFMAStrategy

Supertype for MLFMA construction strategies. Accessors are defined on this type so a future
alternative bundle only has to implement the accessor set.
"""
abstract type AbstractMLFMAStrategy end

# --- near interactions: filling in H2Factory's missing strategy -----------------------------------
#TODO: is this true that H2Factory does not ship a strategy for its sparse near-interactions assembler?
# if that is the case then it prob is good to add it to H2Factory
# + should we using blocksparsematrices instead?
"""
    SparseBEASTNearInteractionsStrategy(quadstrategy)

Select `H2Factory.SparseBEASTNearInteractionsAssembler`, MLFMA's default near-interactions
assembler (`NEARINTERACTIONSASSEMBLERRECOM`).

H2Factory ships strategies for its `BlockBEAST`, `Dummy` and `PassThrough` assemblers but not for its
`Sparse` one, so this fills that gap through H2Factory's documented extension point: a new
`struct <: AbstractNearInteractionsStrategy` plus a `buildnearinteractionsassembler` method. It is
defined here to select the sparse assembler through that protocol.
"""
struct SparseBEASTNearInteractionsStrategy{Q} <: H2Factory.AbstractNearInteractionsStrategy
    quadstrategy::Q
end

#TODO: we should always use defaultquadstrat here?
# remove NEARINTERACTIONSQUADSTRATEGYRECOM?
"""
    nearinteractionsquadstrategy(operator, testspace, trialspace)

The singular-quadrature strategy the near block should be assembled with.

Use [`NEARINTERACTIONSQUADSTRATEGYRECOM`](@ref MLFMA.Assembly.NEARINTERACTIONSQUADSTRATEGYRECOM)
by default. Its `BEAST.DoubleNumWiltonSauterQStrat` assumes triangular 3-D surface elements;
the Sauter-Schwab path would index past a 2-D segment's two vertices.

For `BEAST.HelmholtzOperator2D`, ask `BEAST.defaultquadstrat` for segment quadrature
(`DoubleNumSauterQstrat`), as BEAST's dense assembly does.
"""
nearinteractionsquadstrategy(operator, testspace, trialspace) =
    NEARINTERACTIONSQUADSTRATEGYRECOM

nearinteractionsquadstrategy(operator::BEAST.HelmholtzOperator2D, testspace, trialspace) =
    BEAST.defaultquadstrat(operator, testspace, trialspace)

function H2Factory.buildnearinteractionsassembler(
    operator, testspace, trialspace, γ, strategy::SparseBEASTNearInteractionsStrategy
)
    return H2Factory.SparseBEASTNearInteractionsAssembler{typeof(γ)}(
        operator, testspace, trialspace, strategy.quadstrategy
    )
end

"""
    nearinteractionsstrategyfor(Assembler::Type, quadstrategy)

Map a near-interactions assembler type onto the `H2Factory.AbstractNearInteractionsStrategy` that
selects it.

Prefer constructing the strategy directly when no conversion from an assembler type is needed.

Dispatches on `Type{<:Assembler}` rather than `Type{Assembler}` so both the bare parametric name
(`SparseBEASTNearInteractionsAssembler`, which is what callers actually pass, and whose `typeof` is
`UnionAll`) and an instantiation (`...{ComplexF64}`) resolve. An exact `Type{...}` signature would
match only one of the two, the same dispatch trap that once disabled every `FarField` interpolation
trait.
"""
function nearinteractionsstrategyfor(
    ::Type{<:H2Factory.SparseBEASTNearInteractionsAssembler}, quadstrategy
)
    return SparseBEASTNearInteractionsStrategy(quadstrategy)
end

function nearinteractionsstrategyfor(
    ::Type{<:H2Factory.DummyNearInteractionsAssembler}, quadstrategy
)
    return H2Factory.DummyNearInteractionsStrategy(quadstrategy)
end

function nearinteractionsstrategyfor(
    ::Type{<:H2Factory.BlockBEASTNearInteractionsAssembler}, quadstrategy
)
    return H2Factory.BlockBEASTNearInteractionsStrategy(quadstrategy)
end

# `error` rather than a silent generic fallback: guessing here would hand the map an assembler that
# does not match the requested one, and the near block would be wrong without anything failing.
function nearinteractionsstrategyfor(Assembler::Type, quadstrategy)
    return error(
        "no `AbstractNearInteractionsStrategy` is known for the near-interactions assembler " *
        "$(Assembler). Pass `strategy=MLFMAStrategy(...; nearinteractionsstrategy=...)` with the " *
        "strategy that selects it, or add a `MLFMA.nearinteractionsstrategyfor` method for this " *
        "assembler.",
    )
end

"""
    nearfieldstrategyfor(; isnear=H2Trees.isnear(; additionalbufferboxes=BUFFERBOXRECOM), kwargs...)

`H2Factory.InteractionStrategy` carrying MLFMA's `isnear` default rather than H2Factory's.

The two differ: H2Factory defaults to `H2Trees.isnear()` with NO buffer boxes, MLFMA to
`BUFFERBOXRECOM`. So building an `H2Factory.InteractionStrategy` directly in order to override just one
field, `aggregatenode`, say, silently drops MLFMA's buffer boxes and changes the near/far split.
Going through this keeps that default, and `MLFMAStrategy` uses it too so there is one source of truth.
"""
function nearfieldstrategyfor(;
    isnear=H2Trees.isnear(; additionalbufferboxes=BUFFERBOXRECOM), kwargs...
)
    return H2Factory.InteractionStrategy(; isnear=isnear, kwargs...)
end

# --- MLFMA-specific sub-strategies ----------------------------------------------------------------

"""
    TreeStrategy(operator; minhalfsize=λ(operator) * MINHALFSIZERECOM, minvalues=MINVALUESRECOM)

Tree subdivision parameters: the leaf half-size floor and the basis-function count below which
subdivision stops.

Holds parameters rather than a built `H2Trees` builder. The builder *type* is a
property of the map, not of the strategy, `GalerkinMLFMA` needs a `TwoNTreeBuilder` and
`PetrovGalerkinMLFMA` a `BlockTreeBuilder`, and which one applies cannot be inferred from the
spaces, since a Petrov map is legitimately built with the same space on both sides (see
`test/Maps/comparison.jl`). Storing a builder here would therefore have to be validated against the
map being constructed, reintroducing exactly the desynchronization this bundle removes. A fully
custom builder is still reachable: both constructors accept an explicit `tree`.
"""
struct TreeStrategy{H,V}
    minhalfsize::H
    minvalues::V
end

function TreeStrategy(
    operator; minhalfsize=λ(operator) * MINHALFSIZERECOM, minvalues=MINVALUESRECOM
)
    return TreeStrategy(minhalfsize, minvalues)
end

"""
    SamplingStrategy(; samplingbuilder=SAMPLINGBUILDERRECOM, ϵ=1e-5)

How each level's far-field direction grid is chosen. `ϵ` is the target accuracy the sampling rate is
derived from, so it belongs with the sampling builder rather than floating free: the two are only
meaningful together.

`samplingbuilder` is a builder object, [`SphericalSamplingBuilder`](@ref), called as
`samplingbuilder(Nθ, Nφ)` once the rate is known. [`samplingfunction`](@ref) wraps the builder
to accept `(γ, ϵ, tree)`.
"""
struct SamplingStrategy{B,T}
    samplingbuilder::B
    ϵ::T
    protrusionaware::Bool
end

"""
    SamplingStrategy(; samplingbuilder, ϵ, protrusionaware=false)

`protrusionaware` sizes each level's direction grid from the box that contains its basis
functions rather than the nominal box. See [`MLFMA.Assembly.protrudedhalfsize`](@ref). Basis functions are
filed under the box their centre falls in and their supports hang over the faces, so the nominal box
understates the angular bandwidth whenever elements are not small compared to boxes. Measured
protrusion is 5-25 % of the edge length on ordinary sphere fixtures and raises `L` by 1-3 at every
level, giving roughly 1.5x as many directions at the leaf.

Defaults to `false`. Enabling it grows the grids and changes accuracy, memory use and runtime;
changing that default requires measurements across the supported configurations.
"""
function SamplingStrategy(;
    samplingbuilder=SAMPLINGBUILDERRECOM, ϵ=1e-5, protrusionaware::Bool=false
)
    return SamplingStrategy(samplingbuilder, ϵ, protrusionaware)
end

"""
    SamplingStrategy(operator; samplingbuilder=samplingbuilderrecom(operator), ϵ, protrusionaware)

The same strategy, with the sampling family derived from the operator rather than defaulted to the
spherical one.

Which grid an operator radiates onto is a property of its geometry, not a preference: a 2-D Helmholtz
operator needs directions on a circle, and handing it the spherical default produces grids the 2-D
translation operator and interpolator cannot consume. Like [`nearinteractionsquadstrategy`](@ref),
the default must match the operator's dimension.
"""
function SamplingStrategy(
    operator;
    samplingbuilder=samplingbuilderrecom(operator),
    ϵ=1e-5,
    protrusionaware::Bool=false,
)
    return SamplingStrategy(samplingbuilder, ϵ, protrusionaware)
end

"""
    samplingbuilderrecom(operator)

The direction-grid family `operator`'s geometry calls for.

`SAMPLINGBUILDERRECOM` (Gauss-Legendre θ × equidistant φ on the sphere) for everything 3-D, and a
[`CircularSamplingBuilder`](@ref) for `BEAST.HelmholtzOperator2D`. Stated per operator family rather
than inferred from a mesh, because the strategy is built before any tree exists.
"""
samplingbuilderrecom(operator) = SAMPLINGBUILDERRECOM
samplingbuilderrecom(::BEAST.HelmholtzOperator2D) =
    Sampling.CircularSamplingBuilder{Float64}()

"""
    interpolatorrecom(operator)

The interpolator family `operator`'s direction grids call for, already parameterized by its `γ` type.

`INTERPOLATORECOM` at order 15 for everything 3-D; the circular family for
`BEAST.HelmholtzOperator2D`, whose grids are one-dimensional and whose resample is global, hence order
`0`. The pairing is not optional: an interpolator family only ever accepts the grids it was written
for, so this and [`samplingbuilderrecom`](@ref) have to be decided by the same fact about the
operator.
"""
interpolatorrecom(operator) = INTERPOLATORECOM{15,typeof(Operators.γ(operator))}
interpolatorrecom(operator::BEAST.HelmholtzOperator2D) =
    Interpolation.CircularInterpolator{0,typeof(Operators.γ(operator))}

"""
    protrusionaware(strategy) -> Bool

Whether this strategy sizes direction grids from the protruded box. See [`SamplingStrategy`](@ref).
"""
protrusionaware(strategy::AbstractMLFMAStrategy) =
    samplingstrategy(strategy).protrusionaware

"""
    FarFieldStrategy(; assemblerbuilder=FARFIELDASSEMBLERBUILDERRECOM,
                       quadstrategy=FARFIELDQUADSTRATEGYRECOM,
                       leafstorage=LEAFSTORAGERECOM,
                       trialleafstorage=leafstorage, testleafstorage=leafstorage)

Leaf far-field assembly: which assembler builder, its quadrature, and how the assembled patterns are
stored. Far-field quadrature is a single
numerical rule (`BEAST.SingleNumQStrat`) rather than the near block's Wilton/Sauter machinery, because
leaf far fields have no singular pairs, which is why it is a separate choice from the
near-interactions strategy rather than one shared quadrature setting.

Unlike the near side, MLFMA's far-field assembly is its own (`BEASTFarFieldAssembler`,
`DummyFarFieldAssembler`) with no H2Factory strategy protocol to dispatch through. The selection is
therefore a builder object, [`BEASTFarFieldAssemblerBuilder`](@ref MLFMA.Assembly.BEASTFarFieldAssemblerBuilder) or
[`DummyFarFieldAssemblerBuilder`](@ref MLFMA.TestSupport.DummyFarFieldAssemblerBuilder). This keeps
the field's type concrete and lets alternative builders carry construction state.

The leaf storage is an [`AbstractLeafStorage`](@ref MLFMA.Assembly.AbstractLeafStorage),
[`SampledLeafStorage`](@ref),
[`SphericalHarmonicLeafStorage`](@ref MLFMA.Assembly.SphericalHarmonicLeafStorage),
[`VectorSphericalHarmonicLeafStorage`](@ref MLFMA.Assembly.VectorSphericalHarmonicLeafStorage) or
[`CircularHarmonicLeafStorage`](@ref MLFMA.Assembly.CircularHarmonicLeafStorage), and selects what
the assembled patterns are kept as. It sits
here rather than on its own strategy because it is not independent of the two fields above: the
compressed paths substitute a store rule into the same assembler builder and analyse at the same
quadrature, so the three choices are one policy. It changes storage only, the direction grids, the
translation and the interpolation are identical either way.

# One Storage Per Side

`trialleafstorage` and `testleafstorage` are separate fields; `leafstorage=` is a shorthand
setting both, which is the common case and what every call site that does not care should keep
writing. The two are independent because a Petrov map's test side is ASSEMBLED in its own right, over
a different space, so nothing forces it to be stored the way the trial side is:

    FarFieldStrategy(; trialleafstorage=CircularHarmonicLeafStorage())          # compress trial only
    FarFieldStrategy(; leafstorage=CircularHarmonicLeafStorage())               # compress both

A Galerkin map must have equal settings. `MLFMAStrategy` does not check this because the
strategy does not know which map will consume it. `GalerkinMLFMA` raises; see
[`assertsingleleafstorage`](@ref). Its test side is mirrored rather than assembled, so a distinct
`testleafstorage` would have nothing to act on, and silently accepting a knob that does nothing is
worse than refusing it.

Use [`trialleafstorage`](@ref) or [`testleafstorage`](@ref) to read a side's setting; there is no
side-independent `leafstorage(strategy)` accessor.

The three collection builders derived from this, [`radiationfieldcollectionbuilder`](@ref) (trial),
[`receivefieldcollectionbuilder`](@ref) (test), [`lazyreceivefieldcollectionbuilder`](@ref), are
what the map constructors actually consume.
"""
struct FarFieldStrategy{A,Q,S<:AbstractLeafStorage,R<:AbstractLeafStorage}
    assemblerbuilder::A
    quadstrategy::Q
    trialleafstorage::S
    testleafstorage::R
end

function FarFieldStrategy(;
    assemblerbuilder=FARFIELDASSEMBLERBUILDERRECOM,
    quadstrategy=FARFIELDQUADSTRATEGYRECOM,
    leafstorage=LEAFSTORAGERECOM,
    trialleafstorage=leafstorage,
    testleafstorage=leafstorage,
)
    return FarFieldStrategy(
        assemblerbuilder, quadstrategy, trialleafstorage, testleafstorage
    )
end

"""
    LevelTransferStrategy(operator; Interpolator=..., PhaseShifter=PHASESHIFTERRECOM)

The child-to-parent level transfer: interpolation between direction grids, and the level phase shift.

`Interpolator` defaults to [`interpolatorrecom`](@ref)`(operator)`, `INTERPOLATORECOM{15,…}` for a 3-D
operator and the circular family for a 2-D one, in both cases parameterized by the operator's own `γ`
type, so a real-`γ` operator cannot silently get a complex interpolator or the reverse. The
interpolation *trait* is not a field: it is read from the operator at construction time and both map
constructors `error()` if the interpolator disagrees, since a mismatch would apply a different
component-sign convention than the interpolator it wraps.
"""
struct LevelTransferStrategy{I,P}
    Interpolator::Type{I}
    PhaseShifter::Type{P}
end

function LevelTransferStrategy(
    operator; Interpolator=interpolatorrecom(operator), PhaseShifter=PHASESHIFTERRECOM
)
    return LevelTransferStrategy(Interpolator, PhaseShifter)
end

"""
    LevelTransferStrategy(operator, builder::MLFMA.AbstractInterpolationBuilder; PhaseShifter=...)

The same strategy, with the interpolator named by a builder instead of by a partially applied type.

    LevelTransferStrategy(efie, MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=15))

The `Interpolator` keyword is also available. The builder holds the family's construction policy
and reverse-operator pair. See `src/interpolation/builders.jl`.

The builder carries its own element type. Passing one whose element type disagrees with the operator's
`γ` raises here rather than at the first apply, since that mismatch is what the `Interpolator` default's
`typeof(γ(operator))` parameterization exists to prevent.
"""
function LevelTransferStrategy(
    operator,
    builder::Interpolation.AbstractInterpolationBuilder;
    PhaseShifter=PHASESHIFTERRECOM,
)
    expected = typeof(Operators.γ(operator))
    Interpolation.elementtype(builder) === expected || error(
        "LevelTransferStrategy: the builder carries element type " *
        "$(Interpolation.elementtype(builder)) while the operator's γ is $expected. A real-γ operator " *
        "must not get a complex interpolator or the reverse; pass " *
        "`T = $expected` to the builder.",
    )
    return LevelTransferStrategy(Interpolation.interpolatortype(builder), PhaseShifter)
end

"""
    I2OTranslationStrategy(; TranslationCollection=TRANSLATIONRECOM)

The interaction-to-observation (far-block coupling) collection.
"""
struct I2OTranslationStrategy{C}
    TranslationCollection::C
end

function I2OTranslationStrategy(; TranslationCollection=TRANSLATIONRECOM)
    return I2OTranslationStrategy(TranslationCollection)
end

# --- the bundle ----------------------------------------------------------------------------------

"""
    MLFMAStrategy(operator, testspace, trialspace=testspace; ...)

Bundle of every construction choice an MLFMA map makes. `trialspace` defaults to `testspace` for the
Galerkin case; a Petrov-Galerkin map passes both explicitly.

Each sub-strategy is overridable on its own, and overriding one leaves the rest at their defaults:

```julia
strategy = MLFMAStrategy(
    operator,
    space;
    farfieldstrategy=FarFieldStrategy(; quadstrategy=BEAST.SingleNumQStrat(1)),
)
```

MLFMA-specific sub-strategies: [`TreeStrategy`](@ref), [`SamplingStrategy`](@ref),
[`FarFieldStrategy`](@ref MLFMA.Assembly.FarFieldStrategy), [`LevelTransferStrategy`](@ref), [`I2OTranslationStrategy`](@ref). The
near side is H2Factory's: `nearinteractionsstrategy` is an
`H2Factory.AbstractNearInteractionsStrategy` (see [`SparseBEASTNearInteractionsStrategy`](@ref)) and
`nearfieldstrategy` an `H2Factory.AbstractInteractionStrategy`, so `isnear`, `aggregationmode`,
`translatingnodesiterator` and `aggregatenode` are read with H2Factory's accessors applied to
[`nearfieldstrategy`](@ref) rather than with MLFMA ones.

`scheduler` is normally a bare scheduler rather than a materialized `ExecutionScheduling`: MLFMA
schedules per operation and per level, so the containers have to be keyed by the tree's actual level
*indices*, which are not known until the tree exists, and a strategy is built before the tree. The
map turns it into an `ExecutionScheduling` via [`executionschedulingfor`](@ref). Defaulting to a
ready-made `ExecutionScheduling` is what would let a scheduling sized for one tree be handed to
another, whose deepest level then has no scheduler entry and throws a `KeyError` mid-traversal.

A ready-made `ExecutionScheduling` is rejected at construction because it may cover another tree's
level range. H2Factory returns an existing scheduling unchanged, so an explicit check is needed
to prevent a later `KeyError` on an uncovered level.

`operator`, `testspace` and `trialspace` determine the default near-interaction quadrature.
The other defaults use the operator or fixed recommendations.
"""
struct MLFMAStrategy{T,S,N,F,FF,L,I,C} <: AbstractMLFMAStrategy
    treestrategy::T
    samplingstrategy::S
    nearinteractionsstrategy::N
    nearfieldstrategy::F
    farfieldstrategy::FF
    leveltransferstrategy::L
    i2ostrategy::I
    scheduler::C
end

function MLFMAStrategy(
    operator,
    testspace,
    trialspace=testspace;
    treestrategy=TreeStrategy(operator),
    # Through `SamplingStrategy(operator)`, so the direction-grid family follows the operator's
    # geometry. It has to match the interpolator family below, both are decided by the same fact,
    # and a spherical grid handed to a 2-D operator fails inside the translation operator rather than
    # at construction.
    samplingstrategy=SamplingStrategy(operator),
    nearinteractionsstrategy=SparseBEASTNearInteractionsStrategy(
        nearinteractionsquadstrategy(operator, testspace, trialspace)
    ),
    # Through `nearfieldstrategyfor`, which supplies MLFMA's buffered `isnear` default instead of
    # H2Factory's unbuffered one. The other three fields derive from `isnear` inside
    # `InteractionStrategy`, and its defaults for those already match what MLFMA used
    # (`TRANSLATINGNODESRECOM`/`AGGREGATENODES`/`AggregateMode`).
    nearfieldstrategy=nearfieldstrategyfor(),
    farfieldstrategy=FarFieldStrategy(),
    leveltransferstrategy=LevelTransferStrategy(operator),
    i2ostrategy=I2OTranslationStrategy(),
    scheduler=SerialScheduler(),
)
    # Checked here rather than in `executionschedulingfor` so it fails at the point the mistake is made,
    # with the tree not yet in the picture. `error` rather than a type restriction on the field: the
    # message can then say what to pass instead, and this runs once per construction.
    scheduler isa H2Factory.ExecutionScheduling && error(
        "`scheduler` must be a bare scheduler (e.g. `SerialScheduler()`, " *
        "`DynamicScheduler()`), not a materialized `ExecutionScheduling`. The per-level " *
        "containers are keyed from the tree's level indices at construction time, so a " *
        "pre-built scheduling would silently carry another tree's level range and fail " *
        "mid-traversal on the deepest level.",
    )  #TODO: I find this confusing
    return MLFMAStrategy(
        treestrategy,
        samplingstrategy,
        nearinteractionsstrategy,
        nearfieldstrategy,
        farfieldstrategy,
        leveltransferstrategy,
        i2ostrategy,
        scheduler,
    )
end

"""
    executionschedulingfor(strategy::AbstractMLFMAStrategy, levelrange)

Materialize the per-level, per-operation `ExecutionScheduling` for `levelrange` from `strategy`'s
scheduler.

`levelrange` must be sized from the tree's top level *index*, not its level *count*: with
`minlevel > 1` the two differ (`levels(tree) == 2:6` has count 5), and the traversal indexes these
containers by level index, so a count-sized range leaves the deepest level without a scheduler.

MLFMA schedules the leaf multiply in parallel when Julia has multiple threads, unlike the serial
default for other inner kernels. See [`defaultleafmulscheduler`](@ref) for the measurements.
"""
function executionschedulingfor(strategy::AbstractMLFMAStrategy, levelrange)
    return executionscheduling(
        scheduler(strategy), levelrange; leafmul=defaultleafmulscheduler()
    )
end

"""
    defaultleafmulscheduler()

The scheduler MLFMA gives `H2Factory.LeafMulOp` when the caller does not choose one.

Use parallel scheduling when Julia has multiple threads. With eight threads, compressed leaf
storage's batched projection measured speedups of 4.0x (scalar) and 1.35x (vector).

Use serial scheduling at one thread to avoid scheduling allocations. `DynamicScheduler()` still
allocates per flush at one thread.
The serial scheduler provides the allocation-free path.

A caller wanting something else passes a full `ExecutionScheduling`; this only sets the default.
"""
function defaultleafmulscheduler()
    Threads.nthreads() > 1 && return OhMyThreads.DynamicScheduler()
    return OhMyThreads.SerialScheduler()
end

# `H2Factory.leafmulscheduler` is overridden for MLFMA's two CONCRETE cores, next to where each is
# defined. See `GalerkinMLFMA.jl` and `PetrovGalerkinMLFMA.jl`. Not on H2Factory's abstract
# `GalerkinH2Core`/`PetrovGalerkinH2Core`: those are H2Factory's types, so a method on them from here
# is piracy, and it would silently capture other cores as well, which share the supertypes and have
# no batched leaf multiply to schedule.

# --- accessors -----------------------------------------------------------------------------------
#
# Functions rather than bare field access, so `MLFMAStrategy`'s layout can change without touching any
# construction-stage call site, the same reason `H2Factory`'s map accessors
# (`H2Factory.i2itranslator`, etc.) and other strategy accessors exist.
#
# Deliberately absent: `isnear`, `aggregationmode`, `translatingnodesiterator`, `aggregatenode`. Those
# are H2Factory's accessors on the interaction strategy, use e.g.
# `H2Factory.nearcriterion(MLFMA.nearfieldstrategy(strategy))`. Defining MLFMA versions would shadow
# H2Factory's exported accessors inside MLFMA.

"""
    treestrategy(strategy::AbstractMLFMAStrategy)

Returns `strategy`'s [`TreeStrategy`](@ref).
"""
treestrategy(strategy::AbstractMLFMAStrategy) = strategy.treestrategy

"""
    samplingstrategy(strategy::AbstractMLFMAStrategy)

Returns `strategy`'s [`SamplingStrategy`](@ref).
"""
samplingstrategy(strategy::AbstractMLFMAStrategy) = strategy.samplingstrategy

"""
    nearinteractionsstrategy(strategy::AbstractMLFMAStrategy)

Returns `strategy`'s `H2Factory.AbstractNearInteractionsStrategy`, for
`H2Factory.assemblenearinteractions`.
"""
nearinteractionsstrategy(strategy::AbstractMLFMAStrategy) =
    strategy.nearinteractionsstrategy

"""
    nearfieldstrategy(strategy::AbstractMLFMAStrategy)

Return `strategy`'s `H2Factory.AbstractInteractionStrategy`, for `H2Factory.assembleplans` and its
strategy accessors.
"""
nearfieldstrategy(strategy::AbstractMLFMAStrategy) = strategy.nearfieldstrategy

"""
    farfieldstrategy(strategy::AbstractMLFMAStrategy)

Returns `strategy`'s [`FarFieldStrategy`](@ref MLFMA.Assembly.FarFieldStrategy).
"""
farfieldstrategy(strategy::AbstractMLFMAStrategy) = strategy.farfieldstrategy

"""
    leveltransferstrategy(strategy::AbstractMLFMAStrategy)

Returns `strategy`'s [`LevelTransferStrategy`](@ref).
"""
leveltransferstrategy(strategy::AbstractMLFMAStrategy) = strategy.leveltransferstrategy

"""
    i2ostrategy(strategy::AbstractMLFMAStrategy)

Returns `strategy`'s [`I2OTranslationStrategy`](@ref).
"""
i2ostrategy(strategy::AbstractMLFMAStrategy) = strategy.i2ostrategy

"""
    scheduler(strategy::AbstractMLFMAStrategy)

Returns `strategy`'s bare scheduler. See [`executionschedulingfor`](@ref) for why it is stored bare.
"""
scheduler(strategy::AbstractMLFMAStrategy) = strategy.scheduler

# Leaf accessors, so construction sites never reach through two levels of field access. Each is named
# after the thing it returns rather than after its owner, matching how the map constructors already
# refer to these values.

minhalfsize(strategy::AbstractMLFMAStrategy) = treestrategy(strategy).minhalfsize
minvalues(strategy::AbstractMLFMAStrategy) = treestrategy(strategy).minvalues
"""
    samplingbuilder(strategy::AbstractMLFMAStrategy)

The strategy's sampling builder, i.e. what constructs a level's direction grid.

One of the leaf accessors: it reaches through `samplingstrategy(strategy)` so construction sites
never write two levels of field access themselves. Named after the value it returns rather than its
owner, matching how the map constructors refer to it.
"""
samplingbuilder(strategy::AbstractMLFMAStrategy) =
    samplingstrategy(strategy).samplingbuilder
ϵ(strategy::AbstractMLFMAStrategy) = samplingstrategy(strategy).ϵ
farfieldassemblerbuilder(strategy::AbstractMLFMAStrategy) =
    farfieldstrategy(strategy).assemblerbuilder
farfieldquadstrategy(strategy::AbstractMLFMAStrategy) =
    farfieldstrategy(strategy).quadstrategy
interpolatortype(strategy::AbstractMLFMAStrategy) =
    leveltransferstrategy(strategy).Interpolator
phaseshiftertype(strategy::AbstractMLFMAStrategy) =
    leveltransferstrategy(strategy).PhaseShifter
translationcollectiontype(strategy::AbstractMLFMAStrategy) =
    i2ostrategy(strategy).TranslationCollection

# --- derived accessors ----------------------------------------------------------------------------
#
# Below this line the accessors BUILD something from a strategy rather than reading a stored field.
# All four are derived rather than stored for the same reason: what they return is a stored choice
# plus a fact about the CALL SITE (which side of the map, or which call shape the caller needs), and
# storing the combination would let two positions in one map carry choices that disagree.

"""
    samplingfunction(strategy::AbstractMLFMAStrategy)

The callable the map constructors resolve each level's direction grid with:
`samplingfunction(strategy)(γ, ϵ, tree; level=level)`.

Derived from [`samplingbuilder`](@ref MLFMA.Assembly.samplingbuilder) rather than stored, so the strategy holds the one thing that
is a genuine CHOICE (which sampling family) and not the wrapper that adapts it to the
`(γ, ϵ, tree)` call shape. The immutable `SamplingFunctor` also carries optional protrusion data;
this form sets it to `nothing`.
"""
samplingfunction(strategy::AbstractMLFMAStrategy) =
    SamplingFunctor(samplingbuilder(strategy))

"""
    samplingfunction(strategy, tree, space)

As above, but protrusion-aware if the strategy asks for it, in which case the per-level protrusion
is measured ONCE here, from `tree` and `space` together, and carried by the functor. `space` is
required because the generic protrusion functor evaluates to zero; only the space knows where a
basis function's support actually reaches.
"""
function samplingfunction(strategy::AbstractMLFMAStrategy, tree, space)
    protrusionaware(strategy) || return samplingfunction(strategy)
    return SamplingFunctor(
        samplingbuilder(strategy), Assembly.levelprotrusions(tree, space)
    )
end

"""
    samplingfunction(strategy, tree::H2Trees.BlockTree, testspace, trialspace)

Choose one direction grid per level for both Petrov trees. Measure support protrusion from
each tree's own space when enabled, then use the larger required box size at each level.
Levels present on only one side use that side's requirement.
"""
function samplingfunction(
    strategy::AbstractMLFMAStrategy, tree::H2Trees.BlockTree, testspace, trialspace
)
    trees = (H2Trees.testtree(tree), H2Trees.trialtree(tree))
    protrusions = if protrusionaware(strategy)
        (
            Assembly.levelprotrusions(trees[1], testspace),
            Assembly.levelprotrusions(trees[2], trialspace),
        )
    else
        (nothing, nothing)
    end
    return PetrovSamplingFunctor(samplingbuilder(strategy), trees, protrusions)
end

"""
    radiationfieldcollectionbuilder(strategy::AbstractMLFMAStrategy)

`strategy`'s collection builder for the trial ([`RadiationSide`](@ref)) side: a
[`BEASTFieldCollectionBuilder`](@ref) under [`SampledLeafStorage`](@ref), or a
[`SphericalHarmonicFieldCollectionBuilder`](@ref) under [`SphericalHarmonicLeafStorage`](@ref MLFMA.Assembly.SphericalHarmonicLeafStorage).

Reads [`trialleafstorage`](@ref) and nothing else. Both map constructors ask for a builder and get
one; neither has, or needs, a branch on the storage, and the Galerkin receive side does not read
the strategy for it at all, since it dispatches on the radiation collections it is handed.
"""
radiationfieldcollectionbuilder(strategy::AbstractMLFMAStrategy) =
    _radiationfieldcollectionbuilder(trialleafstorage(strategy), strategy)

_radiationfieldcollectionbuilder(::SampledLeafStorage, strategy) =
    BEASTFieldCollectionBuilder(
        RadiationSide(), farfieldassemblerbuilder(strategy), farfieldquadstrategy(strategy)
    )

# ε comes from the SAMPLING strategy, not from a second tolerance of its own: the order rule holds the
# compressed leaf multiply at the accuracy the map was asked for, so a separate SHE tolerance would be
# a way for the two to disagree. See `leveltruncationorder`, which takes it as `max(ε/100, floor)`.
_radiationfieldcollectionbuilder(::SphericalHarmonicLeafStorage, strategy) =
    SphericalHarmonicFieldCollectionBuilder(
        RadiationSide(),
        farfieldassemblerbuilder(strategy),
        farfieldquadstrategy(strategy),
        ϵ(strategy),
    )

_radiationfieldcollectionbuilder(::VectorSphericalHarmonicLeafStorage, strategy) =
    VSHFieldCollectionBuilder(
        RadiationSide(),
        farfieldassemblerbuilder(strategy),
        farfieldquadstrategy(strategy),
        ϵ(strategy),
    )

# The 2-D backend. `ε` comes from the SAMPLING strategy for the same reason the spherical ones do: the
# order rule holds the compressed leaf multiply at the accuracy the map was asked for, so a separate
# tolerance would be a way for the two to disagree. See `CHETAILRATIO`, which takes it as
# `max(ε/10, floor)`, the TENTH, not the spherical hundredth.
_radiationfieldcollectionbuilder(::CircularHarmonicLeafStorage, strategy) =
    CircularHarmonicFieldCollectionBuilder(
        RadiationSide(),
        farfieldassemblerbuilder(strategy),
        farfieldquadstrategy(strategy),
        ϵ(strategy),
    )

"""
    trialleafstorage(strategy::AbstractMLFMAStrategy)
    testleafstorage(strategy::AbstractMLFMAStrategy)

How `strategy` stores the leaf far-field patterns of each side, see
[`AbstractLeafStorage`](@ref MLFMA.Assembly.AbstractLeafStorage) and [`FarFieldStrategy`](@ref MLFMA.Assembly.FarFieldStrategy).

There is no side-independent `leafstorage(strategy)` accessor because the two settings can differ.
"""
trialleafstorage(strategy::AbstractMLFMAStrategy) =
    farfieldstrategy(strategy).trialleafstorage

@doc (@doc trialleafstorage) testleafstorage(strategy::AbstractMLFMAStrategy) =
    farfieldstrategy(strategy).testleafstorage

"""
    assertsingleleafstorage(strategy, mapname)

Raise unless `strategy`'s two sides carry the same leaf storage.

For `GalerkinMLFMA`, whose test side is mirrored from the trial patterns rather than assembled: there
is no second assembly for a `testleafstorage` to govern, so a differing one would be a setting with
nothing to act on. Reject it rather than accept an ineffective setting.

`PetrovGalerkinMLFMA` does not call this. Its test side is genuinely assembled over its own space, so
the two settings are independent there.
"""
function assertsingleleafstorage(strategy::AbstractMLFMAStrategy, mapname::AbstractString)
    trial = trialleafstorage(strategy)
    test = testleafstorage(strategy)
    trial === test && return nothing
    return error(
        "$mapname mirrors its test side from the trial patterns rather than assembling it, so it " *
        "has ONE leaf storage, but this strategy carries trialleafstorage=$(trial) and " *
        "testleafstorage=$(test). Pass `leafstorage=` to set both, or use `PetrovGalerkinMLFMA`, " *
        "whose test side is assembled in its own right and can genuinely differ.",
    )
end

"""
    receivefieldcollectionbuilder(strategy::AbstractMLFMAStrategy)

`strategy`'s collection builder for an ASSEMBLED test ([`ReceiveSide`](@ref)) side, what
`PetrovGalerkinMLFMA` uses, since its test tree holds different functions than its trial tree and so
cannot mirror them.

Reads [`testleafstorage`](@ref), the mirror image of what
[`radiationfieldcollectionbuilder`](@ref) does with the trial one. Both come from the SAME
`FarFieldStrategy`, which is what still makes it impossible for a map's two sides to end up with
different quadratures or different assemblers, only the STORAGE is per side.
"""
receivefieldcollectionbuilder(strategy::AbstractMLFMAStrategy) =
    _receivefieldcollectionbuilder(testleafstorage(strategy), strategy)

# Compressed assembled receive collections use the same builders with `ReceiveSide()`.
# `receiveoperator(op)` supplies the receive sign of γ, so the stored coefficients already describe
# receive patterns. Their kernels need neither the mirror nor the component coupling used to derive
# Galerkin receive patterns from radiation collections.
#
# The radiation-calibrated order rule was also measured on `meshcircle(2.0, 1/20)` with λ/4 leaves.
# From ε = 1e-3 to 1e-6, receive and radiation orders matched for the 2-D single layer and hypersingular;
# the double layer's receive order was one lower. No receive order was higher in those measurements.
_receivefieldcollectionbuilder(::SampledLeafStorage, strategy) =
    BEASTFieldCollectionBuilder(
        ReceiveSide(), farfieldassemblerbuilder(strategy), farfieldquadstrategy(strategy)
    )

_receivefieldcollectionbuilder(::SphericalHarmonicLeafStorage, strategy) =
    SphericalHarmonicFieldCollectionBuilder(
        ReceiveSide(),
        farfieldassemblerbuilder(strategy),
        farfieldquadstrategy(strategy),
        ϵ(strategy),
    )

_receivefieldcollectionbuilder(::VectorSphericalHarmonicLeafStorage, strategy) =
    VSHFieldCollectionBuilder(
        ReceiveSide(),
        farfieldassemblerbuilder(strategy),
        farfieldquadstrategy(strategy),
        ϵ(strategy),
    )

_receivefieldcollectionbuilder(::CircularHarmonicLeafStorage, strategy) =
    CircularHarmonicFieldCollectionBuilder(
        ReceiveSide(),
        farfieldassemblerbuilder(strategy),
        farfieldquadstrategy(strategy),
        ϵ(strategy),
    )

"""
    lazyreceivefieldcollectionbuilder(strategy::AbstractMLFMAStrategy)

`strategy`'s [`LazyReceiveFieldCollectionBuilder`](@ref), the MIRRORED test side, which
`GalerkinMLFMA` uses because its test and trial patterns are the same leaf's, related by the mirror
map rather than by a second assembly.

Takes `strategy` and ignores it (the builder is fieldless) so that the map constructors read every
construction choice off the strategy uniformly, and so that a future strategy could make the
Galerkin test side configurable without touching either constructor.
"""
lazyreceivefieldcollectionbuilder(::AbstractMLFMAStrategy) =
    LazyReceiveFieldCollectionBuilder()
