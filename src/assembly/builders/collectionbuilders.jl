# A field-collection builder keeps the side, assembler builder and quadrature together.
# `fieldoperator` and `collectiontype` dispatch on the side type, so the assembled operator and
# collection wrapper agree. `MLFMAStrategy` supplies these builders through
# `radiationfieldcollectionbuilder` and related accessors in `strategies/strategies.jl`.

# --- sides ----------------------------------------------------------------------------------------

"""
    AbstractFieldCollectionSide

Which side of an operator a leaf field collection represents: [`RadiationSide`](@ref),
[`ReceiveSide`](@ref) or [`LazyReceiveSide`](@ref).

The side type selects the far-field operator ([`fieldoperator`](@ref)), collection wrapper
([`collectiontype`](@ref)) and builder method by dispatch. Callers do not need to keep separate
operator and wrapper choices in agreement.
"""
abstract type AbstractFieldCollectionSide end

"""
    RadiationSide()

The trial side: the pattern a leaf's basis functions radiate. Assembled with
`radiationoperator(op)` into a `*RadiationFieldCollection`.
"""
struct RadiationSide <: AbstractFieldCollectionSide end

"""
    ReceiveSide()

The test side, assembled in its own right: the pattern a leaf's testing functions receive.
Assembled with `receiveoperator(op)` (which differs from the radiation operator by more than a
sign for several operators (see `operatorutils.jl`)) into a `*ReceiveFieldCollection`.
"""
struct ReceiveSide <: AbstractFieldCollectionSide end

"""
    LazyReceiveSide()

The test side, not assembled: a [`LazyReceiveFieldCollection`](@ref MLFMA.FarFields.LazyReceiveFieldCollection) wrapping the radiation
collection of the same leaf plus that leaf's mirror indices, which reconstructs the receive pattern
at multiply time.

There is no [`fieldoperator`](@ref) or [`collectiontype`](@ref) method because this side assembles
nothing. [`side`](@ref) still identifies what its builder produces, so callers can distinguish a
Galerkin map's mirrored test side from a Petrov map's assembled one.
"""
struct LazyReceiveSide <: AbstractFieldCollectionSide end

# --- leaf storage ---------------------------------------------------------------------------------

"""
    AbstractLeafStorage

How a leaf stores its basis-function far-field patterns: [`SampledLeafStorage`](@ref),
[`SphericalHarmonicLeafStorage`](@ref MLFMA.Assembly.SphericalHarmonicLeafStorage),
[`VectorSphericalHarmonicLeafStorage`](@ref MLFMA.Assembly.VectorSphericalHarmonicLeafStorage) or
[`CircularHarmonicLeafStorage`](@ref MLFMA.Assembly.CircularHarmonicLeafStorage).

Independent of [`AbstractFieldCollectionSide`](@ref), which identifies the pattern, and chosen
per side: [`FarFieldStrategy`](@ref MLFMA.Assembly.FarFieldStrategy) carries a `trialleafstorage` and a `testleafstorage`, because a
Petrov map's test side is assembled over its own space and nothing ties its storage to the trial
side's. A `LazyReceiveSide` is the exception and has no storage of its own: it is derived from the
radiation collections, so it follows whatever they are, which is why `GalerkinMLFMA` requires the two
settings to agree ([`assertsingleleafstorage`](@ref MLFMA.Assembly.assertsingleleafstorage)).

Storage policies select the assembler, collection type and leaf scratch by dispatch. Some harmonic
policies assemble Cartesian components instead of using the operator's own store rule.
"""
abstract type AbstractLeafStorage end

# Read orders from stored coefficients so workspace scratch matches the core.
function _harmonicorders(tree, collections, samplings, minrelevantlevel, name)
    orders = IdDict{Any,Int}()
    for leaf in keys(collections)
        sampling = samplings[_leaflevelindex(tree, leaf, minrelevantlevel, samplings)]
        P = order(collections[leaf])
        previous = get(orders, sampling, P)
        previous == P || error(
            "leaves at one level carry different $name orders ($previous and $P). " *
            "A batched leaf multiply requires one order per level.",
        )
        orders[sampling] = P
    end
    return orders
end

# Each call builds fresh workspace buffers; only traversed leaf levels need pools.
function _harmonicscratchchannel(
    make, orders, leaflevels, plans, samplings, minrelevantlevel
)
    traversed = Set(_traversedlevels(plans))
    return Dict(
        level => begin
            sampling = samplings[_samplinglevelindex(level, minrelevantlevel, samplings)]
            H2Factory.createchannel([
                make(sampling, orders[sampling], level) for _ in 1:H2Factory.nchunks(level)
            ])
        end for level in leaflevels if level in traversed
    )
end

"""
    SampledLeafStorage()

Store each leaf's patterns as samples on that level's direction grid. This is the default.
"""
struct SampledLeafStorage <: AbstractLeafStorage end

"""
    SphericalHarmonicLeafStorage()

Store each leaf's patterns as Cartesian spherical-harmonic coefficients, as in Eibert's SE-MLFMA. The
level's grid is unchanged and so is everything above the leaf; only the leaf interface differs.

The order is chosen per level by the calibrated rule in
`SphericalHarmonics.leveltruncationorder`, never by hand (see
[`shecollections`](@ref MLFMA.Assembly.shecollections)).
"""
struct SphericalHarmonicLeafStorage <: AbstractLeafStorage end

"""
    VectorSphericalHarmonicLeafStorage()

Store each leaf's patterns as tangential vector spherical-harmonic coefficients: the `X`/`Psi`
families rather than three Cartesian scalar expansions. Like
[`SphericalHarmonicLeafStorage`](@ref MLFMA.Assembly.SphericalHarmonicLeafStorage) it changes storage only; the grids, the translation and the
interpolation are untouched.

Compresses further than the Cartesian expansion does, and by more than the coefficient count
`2P(P+2)` against `3(P+1)^2` predicts (measured 0.500x at e = 1e-3, 0.519x at 1e-4 and 0.660x at
1e-5, because the tangential basis also needs a lower degree). See `sphericalharmonics/vectorharmonics.jl`
for the reason and for the caveat that the advantage narrows as the tolerance tightens.

It also removes the frame projection from the leaf multiply outright: `X` and `Psi` are already
tangential, so synthesis writes the moment's own two components with no Cartesian stage.
"""
struct VectorSphericalHarmonicLeafStorage <: AbstractLeafStorage end

"""
    CircularHarmonicLeafStorage()

Store each leaf's patterns as circular-harmonic coefficients `a_m`, `|m| ≤ M`: the 2-D counterpart
of [`SphericalHarmonicLeafStorage`](@ref MLFMA.Assembly.SphericalHarmonicLeafStorage), for maps built on a `CircularSampling`. Like the spherical
ones it changes storage only; the grids, the translation and the interpolation are untouched.

The measured collections are smaller but not always faster.
The order is chosen per level by the calibrated rule in `CircularHarmonics.leveltruncationorder`,
never by hand.
"""
struct CircularHarmonicLeafStorage <: AbstractLeafStorage end

"""
    fieldoperator(side, op)

The BEAST far-field operator to assemble for `side` of `op`.

`RadiationSide` and `ReceiveSide` only. `LazyReceiveSide` assembles nothing (see its docstring).
"""
fieldoperator(::RadiationSide, op) = radiationoperator(op)
fieldoperator(::ReceiveSide, op) = receiveoperator(op)

"""
    collectiontype(side, F)

The concrete collection type wrapping an assembled far field of type `F` on `side`.

Takes the far-field TYPE rather than the operator so it composes with an assembler, which carries
`F` but not necessarily the original operator (`farfieldtype(assembler)`) (the same reason the
assembler-based methods in `fieldcollections.jl` can name a collection type with no operator in
sight).
"""
collectiontype(::RadiationSide, ::Type{F}) where {F<:AbstractFarField} =
    radiationcollectiontype(F)
collectiontype(::ReceiveSide, ::Type{F}) where {F<:AbstractFarField} =
    receivecollectiontype(F)

# --- builders -------------------------------------------------------------------------------------

"""
    AbstractFieldCollectionBuilder

Construction policy for one side's field collections: everything [`buildfieldcollections`](@ref)
needs beyond the tree, the operator and the space.

The subtypes share a construction protocol, not a set of fields.
[`AbstractAssembledFieldCollectionBuilder`](@ref) owns an assembler and carries `side`,
`assemblerbuilder` and `quadstrategy`; [`LazyReceiveFieldCollectionBuilder`](@ref) is a concrete
builder that has no fields and assembles nothing. Fields and their accessors therefore belong
to the assembled subtype, not this supertype.

A new leaf storage supplies its construction policy; iteration and wiring are shared.
"""
abstract type AbstractFieldCollectionBuilder end

"""
    AbstractAssembledFieldCollectionBuilder <: AbstractFieldCollectionBuilder

A builder that owns an assembler: it carries `side`, `assemblerbuilder` and `quadstrategy`, and gets
the three accessors and [`buildfarfieldassembler`](@ref) from those fields. A compressed builder adds
an `ε`, and overrides `buildfarfieldassembler` only if it assembles something other than what the
operator itself radiates, as the two harmonic ones do.

Separate from [`AbstractFieldCollectionBuilder`](@ref) so that
[`LazyReceiveFieldCollectionBuilder`](@ref), which is fieldless and has no assembler at all, does not
inherit accessors it cannot answer. Under one supertype it would, and asking it for its quadrature
would fail with "type has no field quadstrategy" rather than saying that the operation is not part
of the lazy protocol. `test/FarField/collectionbuilders.jl` pins the `MethodError`s.
"""
abstract type AbstractAssembledFieldCollectionBuilder <: AbstractFieldCollectionBuilder end

"""
    AbstractCompressedFieldCollectionBuilder{Side} <: AbstractAssembledFieldCollectionBuilder

A builder whose leaves store harmonic coefficients rather than samples: it assembles, analyses the
patterns to a per-level order, and returns one collection per leaf.

Parameterized by `Side` because both sides need it. The radiation and receive bodies are the same
three steps and differ only in the per-leaf wrapper, so with the side in the type they become two
shared methods plus two small per-builder hooks:

  - [`analysecollections`](@ref), the analysis this storage performs;
  - [`assembledreceivecollectiontype`](@ref), the wrapper its assembled receive side puts on.

Both hooks use dispatch rather than function-valued arguments.
"""
abstract type AbstractCompressedFieldCollectionBuilder{Side} <:
              AbstractAssembledFieldCollectionBuilder end

"""
    analysecollections(builder, assembler, tree, leafsamplings)

`builder`'s compressed collections, keyed by leaf, and the per-level orders it chose.

The analysis depends on the storage policy. Orders are the
second return value; callers recover them later with `sheorders` and its
siblings, which is cheaper than threading them through the core.
"""
function analysecollections end

"""
    assembledreceivecollectiontype(builder)

The wrapper for each leaf's collection on `builder`'s assembled (Petrov) receive side.

Assembled, not lazy: this side stores its own receive patterns, so it carries no mirror indices and
no antipodal parity. See the note on the lazy builder for why the two are not one thing.
"""
function assembledreceivecollectiontype end

# The two shared bodies. Everything except the per-leaf wrapper is common, which is why the side is a
# type parameter rather than a field.
function _analysedbyleaf(builder, op, tree, leafsamplings, space)
    assembler = buildfarfieldassembler(builder, op, space)
    collections, _ = analysecollections(builder, assembler, tree, leafsamplings)
    # Positional, in `leafsamplings` order, because that is what every other `buildfieldcollections`
    # method returns and what `getfieldcollectionsdict` re-keys.
    return (collections[leaf] for leaf in leafids(leafsamplings))
end

function buildfieldcollections(
    builder::AbstractCompressedFieldCollectionBuilder{RadiationSide},
    op,
    tree,
    leafsamplings,
    space;
    verbose::Bool=false,
)
    return collect(_analysedbyleaf(builder, op, tree, leafsamplings, space))
end

function buildfieldcollections(
    builder::AbstractCompressedFieldCollectionBuilder{ReceiveSide},
    op,
    tree,
    leafsamplings,
    space;
    verbose::Bool=false,
)
    wrapper = assembledreceivecollectiontype(builder)
    return [wrapper(c) for c in _analysedbyleaf(builder, op, tree, leafsamplings, space)]
end

"""
    BEASTFieldCollectionBuilder(side, assemblerbuilder, quadstrategy)

Construction policy for one materialized leaf field collection: which [side](@ref
AbstractFieldCollectionSide) to assemble, which assembler to build it with
([`BEASTFarFieldAssemblerBuilder`](@ref MLFMA.Assembly.BEASTFarFieldAssemblerBuilder) or
[`DummyFarFieldAssemblerBuilder`](@ref MLFMA.TestSupport.DummyFarFieldAssemblerBuilder)), and that
assembler's quadrature.

Consumed by [`buildfieldcollection`](@ref) (one sampling) and [`buildfieldcollections`](@ref) (one
per leaf of a tree). Normally obtained from a strategy rather than built by hand -- see
[`radiationfieldcollectionbuilder`](@ref) and [`receivefieldcollectionbuilder`](@ref).

The `BEAST` in the name is about the collections' provenance, not a restriction on
`assemblerbuilder`: any builder with the five-argument
`(operator, space, quadstrategy, storerule, farfieldtype)` call signature works here, which is what
lets `DummyFarFieldAssemblerBuilder` drop straight in.
"""
struct BEASTFieldCollectionBuilder{Side<:AbstractFieldCollectionSide,A,Q} <:
       AbstractAssembledFieldCollectionBuilder
    side::Side
    assemblerbuilder::A
    quadstrategy::Q
end

"""
    LazyReceiveFieldCollectionBuilder()

Construction policy for a leaf's [`LazyReceiveFieldCollection`](@ref MLFMA.FarFields.LazyReceiveFieldCollection): wrap the leaf's already-built
RADIATION collection together with its own level's mirror indices.

Fieldless, because it has no policy to carry -- there is no assembler and no quadrature on this
side. It exists so the lazy path goes through the same [`buildfieldcollections`](@ref) protocol as
the materialized one, which is what lets `GalerkinMLFMA` and `PetrovGalerkinMLFMA` differ only in
which builder they ask the strategy for instead of calling two differently-shaped construction
functions.
"""
struct LazyReceiveFieldCollectionBuilder <: AbstractFieldCollectionBuilder end

"""
    side(builder)

The [`AbstractFieldCollectionSide`](@ref) `builder` produces collections for.
"""
side(builder::AbstractAssembledFieldCollectionBuilder) = builder.side
# The lazy builder is fieldless: there is no assembler and no quadrature on that side, so it states
# its side rather than storing one.
side(::LazyReceiveFieldCollectionBuilder) = LazyReceiveSide()

"""
    assemblerbuilder(builder)

`builder`'s far-field assembler builder.
"""
assemblerbuilder(builder::AbstractAssembledFieldCollectionBuilder) =
    builder.assemblerbuilder

"""
    quadstrategy(builder::BEASTFieldCollectionBuilder)

`builder`'s far-field quadrature strategy. Shares its name with the accessor on the assemblers
themselves, since it is the same value before and after the assembler is built.
"""
quadstrategy(builder::AbstractAssembledFieldCollectionBuilder) = builder.quadstrategy

"""
    buildfarfieldassembler(builder::BEASTFieldCollectionBuilder, op, space)

Build the assembler for `space`'s patterns using `builder`'s policy.

The side selects both `fieldoperator(side(builder), op)` and the collection wrapper. Keeping these
choices together prevents a radiation operator from being paired with a receive wrapper.
"""
function buildfarfieldassembler(builder::AbstractAssembledFieldCollectionBuilder, op, space)
    return assemblerbuilder(builder)(
        fieldoperator(side(builder), op),
        space,
        quadstrategy(builder),
        farfieldstorerule(op),
        farfieldtype(op),
    )
end

"""
    buildfieldcollection(builder, op, sampling, space; verbose=false)

One field collection for `space` sampled on `sampling`, per `builder`'s policy.

The single-collection entry point, for callers with no tree: the whole space against one direction
grid, referenced to the global origin.
"""
function buildfieldcollection(
    builder::BEASTFieldCollectionBuilder{RadiationSide},
    op,
    sampling,
    space;
    verbose::Bool=false,
)
    assembler = buildfarfieldassembler(builder, op, space)
    return RadiationFieldCollection(assembler, sampling; verbose=verbose)
end

function buildfieldcollection(
    builder::BEASTFieldCollectionBuilder{ReceiveSide},
    op,
    sampling,
    space;
    verbose::Bool=false,
)
    assembler = buildfarfieldassembler(builder, op, space)
    return ReceiveFieldCollection(assembler, sampling; verbose=verbose)
end

# Dispatched per side rather than resolved through a function-valued helper
# (`_collectionfor(side) = RadiationFieldCollection`). A helper returning a FUNCTION would put back
# in miniature exactly what the `FieldCollectionType=` keyword was: a callable chosen in one place
# and invoked in another, with the type checker unable to say the two match. Two three-line methods
# cost less than that indirection.

"""
    buildfieldcollections(builder, op, tree, leafsamplings, space; verbose=false)

One field collection per entry of `leafsamplings`, in that vector's order, per `builder`'s policy.

`leafsamplings` is the `Vector{BoxFarFieldSampling}` from
[`leaffarfieldsamplings`](@ref MLFMA.Assembly.leaffarfieldsamplings): each leaf's own direction grid and its
own box center, so leaves at different levels of a non-uniform tree are assembled at different rates.

The order is `leafids(leafsamplings)`, NOT a fresh `H2Trees.leaves(tree)` call -- see
[`leafids`](@ref MLFMA.Assembly.leafids). `tree` is consulted for geometry, never for leaf order, so a caller
that assembles a subset of the leaves or a permuted order gets collections back in exactly the order
it asked for.

For [`LazyReceiveFieldCollectionBuilder`](@ref) the fifth argument is the leaf-keyed dictionary of
already-built RADIATION collections rather than a space -- nothing is assembled on that side, so
there is no space to assemble over. Same position, same protocol, different input; the builder type
says which.
"""
function buildfieldcollections(
    builder::BEASTFieldCollectionBuilder{RadiationSide},
    op,
    tree,
    leafsamplings,
    space;
    verbose::Bool=false,
)
    assembler = buildfarfieldassembler(builder, op, space)
    return RadiationFieldCollections(assembler, tree, leafsamplings; verbose=verbose)
end

function buildfieldcollections(
    builder::BEASTFieldCollectionBuilder{ReceiveSide},
    op,
    tree,
    leafsamplings,
    space;
    verbose::Bool=false,
)
    assembler = buildfarfieldassembler(builder, op, space)
    return ReceiveFieldCollections(assembler, tree, leafsamplings; verbose=verbose)
end

# `radiationcollections` is keyed by leaf (a `Dictionary` from `getfieldcollectionsdict`), while the
# result is positional like every other `buildfieldcollections` method -- its caller wraps it.
#
# Radiation collections are per leaf, but mirror indices depend only on the sampling grid.
# Share one index array per tree level, using that level's sampling even on non-uniform trees.
# Levels below `minrelevantlevel` use the coarsest sampling; they can hold duplicate arrays,
# but their leaf patterns are not traversed.
# Build the cache before the threaded loop and only read it there: concurrent `get!` calls
# on a shared `Dict` are unsafe. Keep the cache local to this tree and its samplings.
#
# `verbose` is accepted and ignored: this side runs no quadrature, so there is no progress to report.
# It stays in the signature so callers do not have to know which builder they hold before deciding
# whether the keyword is legal.
function buildfieldcollections(
    ::LazyReceiveFieldCollectionBuilder,
    op,
    tree,
    leafsamplings,
    radiationcollections;
    verbose::Bool=false,
)
    # Leaf order from `leafsamplings`, not from a fresh `H2Trees.leaves(tree)` call: position `i`
    # here must be the leaf whose radiation collection sits at position `i` of the dictionary the
    # materialized builder just produced.
    leaves = Assembly.leafids(leafsamplings)
    mirrorindicesbylevel = _mirrorindicesbylevel(tree, leaves, leafsamplings)

    firstcollection = LazyReceiveFieldCollection(
        op,
        radiationcollections[leaves[begin]],
        mirrorindicesbylevel[H2Trees.level(tree, leaves[begin])],
    )

    collections = Vector{typeof(firstcollection)}(undef, length(leaves))
    collections[begin] = firstcollection

    Threads.@threads for i in eachindex(leaves)
        i == firstindex(leaves) && continue

        collections[i] = LazyReceiveFieldCollection(
            op,
            radiationcollections[leaves[i]],
            mirrorindicesbylevel[H2Trees.level(tree, leaves[i])],
        )
    end

    return collections
end

# One mirror-index array per distinct LEAF LEVEL. Built from `leafsamplings[i].sampling` -- the level
# selects which array to reuse, it never determines the grid, so a non-uniform tree still gets each
# level's own sampling.
#
# Concretely typed (`Dict{Int,I}` with `I` taken from the first array) so the threaded loop's lookup
# returns a concrete type and the `LazyReceiveFieldCollection`s it builds share one concrete `I`
# parameter -- which is what keeps `collections` a `Vector` of one concrete element type.
function _mirrorindicesbylevel(tree, leaves, leafsamplings)
    firstindices = Sampling.mirrorindices(leafsamplings[begin].sampling)
    bylevel = Dict{Int,typeof(firstindices)}(
        H2Trees.level(tree, leaves[begin]) => firstindices
    )

    for (i, leaf) in enumerate(leaves)
        get!(bylevel, H2Trees.level(tree, leaf)) do
            return Sampling.mirrorindices(leafsamplings[i].sampling)
        end
    end
    return bylevel
end

"""
    tangentialassembler(builder, op, space, storagename)

Build the Cartesian assembler for a tangential harmonic builder, rejecting incompatible operators.

Both harmonic leaf storages expand a two-family tangential field, so both assemble the operator's
patterns as `CartesianVectorFarField` stored as-is rather than through the operator's own store
rule, and both must refuse an operator that radiates something else. `storagename` is the only
thing that differs, and it appears in the message.

Reject other representations rather than constructing a receive path with an invalid coupling.
See `receivecoupling`.
"""
function tangentialassembler(builder, op, space, storagename::AbstractString)
    farfieldtype(op) <: TangentialSphericalFarField || error(
        "$storagename expands a TANGENTIAL two-component far field, but $(typeof(op)) radiates a " *
        "$(farfieldtype(op)). There is no two-family expansion of that representation, so use " *
        "`SampledLeafStorage()` for this operator, or add a compressed backend for its " *
        "representation. Refusing rather than guessing.",
    )
    return assemblerbuilder(builder)(
        fieldoperator(side(builder), op),
        space,
        quadstrategy(builder),
        StoreAsFarFieldValue(),
        CartesianVectorFarField,
    )
end
