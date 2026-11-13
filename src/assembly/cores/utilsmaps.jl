# Adapt a sampling builder to the `(γ, ϵ, tree; level)` interface used by map constructors.
struct SamplingFunctor{B,P}
    samplingbuilder::B
    protrusions::P
end

SamplingFunctor(samplingbuilder) = SamplingFunctor(samplingbuilder, nothing)

# A Petrov level uses one grid large enough for both sides, even if their depths differ.
struct PetrovSamplingFunctor{B,T,P}
    samplingbuilder::B
    trees::T
    protrusions::P
end

function (f::PetrovSamplingFunctor)(γ, ϵ, tree; level::Integer)
    halfsize = nothing
    dimension = nothing
    for (i, side) in enumerate(f.trees)
        level in H2Trees.levels(side) || continue
        node = first(H2Trees.LevelIterator(side, level))
        sidedimension = length(H2Trees.center(side, node))
        if dimension !== nothing && dimension != sidedimension
            throw(DimensionMismatch("Petrov sampling requires equal spatial dimensions"))
        end
        dimension = sidedimension
        protrusion = _protrusionat(f.protrusions[i], side, nothing, level)
        required = protrudedhalfsize(H2Trees.halfsize(side, node), protrusion)
        halfsize = halfsize === nothing ? required : max(halfsize, required)
    end
    halfsize === nothing &&
        throw(ArgumentError("neither Petrov tree contains level $level"))
    return sampling(γ, ϵ, diameter(halfsize, dimension), f.samplingbuilder)
end

_samplingminimumlevel(f, tree) = H2Trees.minimumlevel(tree)
_samplingminimumlevel(f::PetrovSamplingFunctor, tree) =
    minimum(H2Trees.minimumlevel(side) for side in f.trees)

function (f::SamplingFunctor)(γ, ϵ, tree; node=nothing, level=nothing)
    return sampling(
        γ,
        ϵ,
        tree;
        samplingbuilder=f.samplingbuilder,
        node=node,
        level=level,
        protrusion=_protrusionat(f.protrusions, tree, node, level),
    )
end

# `nothing` disables protrusion adjustment through dispatch, without a run-time enlargement step.
_protrusionat(::Nothing, tree, node, level) = false
function _protrusionat(protrusions::AbstractDict, tree, node, level)
    return get(protrusions, isnothing(level) ? H2Trees.level(tree, node) : level, false)
end

function sampling(samplingbuilder, protrusions=nothing)
    return SamplingFunctor(samplingbuilder, protrusions)
end

"""
    protrudedhalfsize(halfsize, protrusion)

Return the halfsize needed to contain a group's basis functions, using H2Trees' protrusion measure.

`H2Trees.levelprotrusions` reports `(|v - center| - halfsize) / (2 * halfsize)` maximised over
support vertices and coordinates. This is the excess beyond a box face, normalized by edge length.
So the containing box is `halfsize * (1 + 2 * protrusion)`.

A group's angular bandwidth depends on source distance from the expansion centre, not nominal box
size. Basis functions are assigned by their centres, but their supports may extend beyond the faces.
The adjustment can change the required order substantially. Measured examples:

    mesh λ/8, leaf λ/4     protrusion 0.208 at the leaf level, L = 13 -> 16
    mesh λ/10, leaf λ/4    protrusion 0.197 at the leaf level, L = 13 -> 16
    mesh λ/10, leaf λ/12   protrusion 0.250 at level 4,        L = 10 -> 13

Protrusion grows as boxes shrink: one tree measured 0.052 / 0.104 / 0.208 across three levels.
A fixed-size element occupies more of a smaller box, so ignoring its support can undersample
deeper levels.
"""
protrudedhalfsize(halfsize, protrusion) = halfsize * (1 + 2 * protrusion)

"""
    checkboxsizes(tree, operator; minimum=MINHALFSIZERECOM)

Warn if any leaf box is smaller than `minimum` wavelengths in halfsize.

Below roughly `λ/4` halfsize, discarded evanescent content becomes significant and far-field error
may stop following `ϵ`: the low-frequency breakdown. `TreeStrategy` defaults `minhalfsize` to
`MINHALFSIZERECOM`, but a supplied tree bypasses that default.

Warn rather than reject the tree so diagnostics can use smaller boxes. `maxlog = 1` limits repeated
warnings, which are common for small test fixtures; passing such tests does not establish accuracy
throughout this regime.
"""
function checkboxsizes(tree, operator; minimum=MINHALFSIZERECOM)
    wavelength = λ(operator)
    smallest = Inf
    for leaf in H2Trees.leaves(tree)
        smallest = min(smallest, H2Trees.halfsize(tree, leaf))
    end
    isfinite(smallest) || return nothing
    # Allow relative slack for last-bit differences between `λ / 4` and `0.25 * λ`.
    if smallest < minimum * wavelength * (1 - 1e-8)
        @warn(
            "leaf boxes below the low-frequency breakdown floor: smallest halfsize is " *
                "$(round(smallest / wavelength; digits=4))λ against a recommended $(minimum)λ " *
                "(`MINHALFSIZERECOM`). The addition theorem's propagating-mode expansion degrades " *
                "there, so the far-field error will stop tracking ϵ regardless of the sampling rate.",
            maxlog = 1
        )
    end
    return nothing
end

"""
    levelprotrusions(tree, space) -> Dict{Int,Float64}

Return capped per-level maximum protrusions, keyed by tree level rather than H2Trees' level index.

Use the space's protrusion functor with `H2Trees.levelprotrusions`; the generic fallback returns zero.
"""
function levelprotrusions(tree, space; cap=PROTRUSIONCAP)
    values = H2Trees.levelprotrusions(tree; compute=H2Trees.defaultprotrusionfunctor(space))
    return Dict(
        level => min(values[i], cap) for
        (i, level) in enumerate(collect(H2Trees.levels(tree)))
    )
end

"""
    PROTRUSIONCAP

The largest protrusion [`levelprotrusions`](@ref) passes on for grid sizing, `0.25`.

Protrusion is normalized by box edge length, so `0.25` means an overhang of a quarter edge and
limits linear enlargement to 1.5x. H2Trees warns above `0.5`; boxes that small relative to their
elements usually need rebuilding rather than higher sampling rates. A λ/12-box tree on a λ/6 mesh
measured `1.007`, which would otherwise request roughly 3x enlargement.

Ordinary sphere fixtures with the default protrusion check measured maxima around `0.24`.
The cap bound the fixtures built with `NoProtrusionCheck()`. It complements the λ/4 halfsize warning
in `MLFMA.checkboxsizes`, but cannot repair unsuitable tree geometry.
"""
const PROTRUSIONCAP::Float64 = 0.25

function sampling(
    γ::T,
    ϵ::AbstractFloat,
    tree;
    samplingbuilder=Sampling.SphericalSamplingBuilder{Float64}(),
    node=nothing,
    level=nothing,
    protrusion=false,
) where {T<:Number}
    !xor(isnothing(node), isnothing(level)) && error("Either node or level must be nothing")

    samplednode = isnothing(level) ? node : H2Trees.LevelIterator(tree, level)[begin]
    halfsize = H2Trees.halfsize(tree, samplednode)

    # Read the spatial dimension from the tree: 2-D centres imply square boxes, not cubes.
    dimension = length(H2Trees.center(tree, samplednode))

    # `protrusion = false` uses nominal box size, which can understate the support's bandwidth.
    diameter = Assembly.diameter(protrudedhalfsize(halfsize, protrusion), dimension)

    return Assembly.sampling(γ, ϵ, diameter, samplingbuilder)
end

function γκ(γ, κ)
    !xor(isnothing(γ), isnothing(κ)) && error("Set exactly one of γ or κ")

    if γ === nothing && (κ !== nothing)
        if iszero(real(κ))
            γ = -imag(κ)
        else
            γ = im * κ
        end
    end

    return γ, κ
end

function λ(operator::BEAST.MaxwellOperator3D)
    return 2 * pi / norm(Operators.γ(operator))
end

# BEAST's 3-D Helmholtz operators subtype `MaxwellOperator3D`; 2-D operators use a separate family.
# Both use the same wavelength formula.
function λ(operator::BEAST.HelmholtzOperator2D)
    return 2 * pi / norm(Operators.γ(operator))
end

# Store the concrete far-field type to preserve the operator's representation.
# A component count alone is ambiguous: two representations have three components.
struct StorageFieldsFunctor{T,OFF,F,S}
    γ::T
    offset::OFF
    farfieldtype::Type{F}
    samplings::S
end

function (f::StorageFieldsFunctor)(tree, level)
    return f.farfieldtype(f.γ, f.samplings[level - f.offset])
end

# Build level-transfer scratch inside `withschedulechunks(schedules) do ... end`.
# `H2Factory.maxnchunks` reads the schedules from that dynamic scope.
#
# Both dictionaries use child-level keys spanning
# `(relevantlevels[begin] + 1):relevantlevels[end]`, including the deepest transfer.
# Aggregation (`_aggregatelevel!`/`aggregatechunk!`/`upwardpass!`) loops over the parent level and
# borrows I2I scratch at `level + 1`. Size that store with `maxnchunks(level - 1)`.
# Fused disaggregation under `AggregateMode()` loops over the child level; plain disaggregation
# under `AggregateTranslateMode()` loops over the parent. O2O therefore needs
# `maxnchunks(level - 1, level)`. Sizing by the child's count alone can overrun the store when
# parent and child schedules differ, a case single-threaded tests cannot expose.
#
# Both modes are covered in `test/Maps/comparison.jl`, but those fixtures do not prove the key range
# covers every tree shape. A plain-path borrow at `relevantlevels[begin]` would be outside the
# range and return `EmptyChannels()` rather than usable scratch.
#
# Each (level, chunkid) owns a `LevelTransferScratch`, including interpolation interim arrays;
# mutable scratch is not shared across in-flight tasks or workspaces. I2I has a parent-grid moment
# for forward accumulation, since interpolation overwrites its destination. O2O also has a
# child-grid moment for reverse accumulation. The diagonal phase acts on the parent grid;
# reverse application stages that input inside the operator's interim arrays.
function scratchi2ichannel(γ, FarFieldType, relevantlevels, samplings, anterpolator)
    shapeat = StorageFieldsFunctor(γ, relevantlevels[begin] - 1, FarFieldType, samplings)
    interimat = InterimScratchFunctor(anterpolator, samplings, relevantlevels[begin] - 1)
    return Dict(
        level => H2Factory.createchannel([
            LevelTransferScratch(shapeat(nothing, level - 1), nothing, interimat(level)) for _ in 1:H2Factory.maxnchunks(level - 1)
        ]) for level in (relevantlevels[begin] + 1):relevantlevels[end]
    )
end

function scratcho2ochannel(γ, FarFieldType, relevantlevels, samplings, anterpolator)
    shapeat = StorageFieldsFunctor(γ, relevantlevels[begin] - 1, FarFieldType, samplings)
    interimat = InterimScratchFunctor(anterpolator, samplings, relevantlevels[begin] - 1)
    return Dict(
        level => H2Factory.createchannel([
            LevelTransferScratch(
                shapeat(nothing, level - 1), shapeat(nothing, level), interimat(level)
            ) for _ in 1:H2Factory.maxnchunks(level - 1, level)
        ]) for level in (relevantlevels[begin] + 1):relevantlevels[end]
    )
end

# --- leaf-multiply scratch --------------------------------------------------------------------
#
# Leaf scratch uses one far field per (level, chunk), on the leaf's own grid with no `±1` offset.
# `LazyReceiveFieldCollection` requires this scratch for mirroring and a contiguous BLAS matvec;
# it has no scratch-free `mul!`. Cover every traversed level that contains leaves.
# Leaf multiplies run on the current traversal level, so `nchunks(level)` suffices here.
# Untraversed levels have no chunk schedule and must not receive a pool.
function _traversedlevels(plans)
    levels = Int[]
    for plan in (plans.trialaggregationplan, plans.testaggregationplan)
        append!(levels, H2Trees.aggregationlevels(plan))
    end
    for plan in (plans.testdisaggregationplan, plans.trialdisaggregationplan)
        append!(levels, H2Trees.disaggregationlevels(plan))
    end
    return sort!(unique!(levels))
end

_leaflevels(tree) =
    sort!(unique!([H2Trees.level(tree, leaf) for leaf in H2Trees.leaves(tree)]))

"""
    leafscratchchannel(γ, FarFieldType, leaflevels, plans, samplings, minrelevantlevel)

The per-level scratch far fields the leaf multiply borrows, as a `Dict` keyed by level.

Each channel holds `H2Factory.nchunks(level)` far fields, allowing a chunk to borrow without
allocating. [`LazyReceiveFieldCollection`](@ref MLFMA.FarFields.LazyReceiveFieldCollection) uses
them for mirroring followed by a BLAS multiply.

Only traversed leaf levels get a pool. Leaves shallower than `minrelevantlevel` are not aggregated
or disaggregated and have no chunk schedule; calling `nchunks` there would raise a `KeyError`.
"""
function leafscratchchannel(γ, FarFieldType, leaflevels, plans, samplings, minrelevantlevel)
    traversed = Set(_traversedlevels(plans))
    return Dict(
        level => H2Factory.createchannel([
            FarFieldType(
                γ,
                samplings[_samplinglevelindex(level, minrelevantlevel, samplings)],
                Val(:zero),
            ) for _ in 1:H2Factory.nchunks(level)
        ]) for level in leaflevels if level in traversed
    )
end

# Companion to `StorageFieldsFunctor`, keyed the same way: `level` is the child level, so the parent's
# sampling sits one index lower. `MLFMA.interimscratch` dispatches on the anterpolator's family and
# returns `nothing` for a family with no interim arrays.
struct InterimScratchFunctor{A,S}
    anterpolator::A
    samplings::S
    offset::Int
end

function (f::InterimScratchFunctor)(level)
    return Interpolation.interimscratch(
        f.anterpolator, f.samplings[level - f.offset], f.samplings[level - 1 - f.offset]
    )
end

struct SamplingSizeFunctor{S}
    samplings::S
    offset::Int
end

function (f::SamplingSizeFunctor)(tree, level)
    return size(f.samplings[level - f.offset])
end

# function fields(
#     γ::T, operator, tree, minrelevantlevel, aggregationplan, samplings
# ) where {T}
#     return H2Factory.storagemomentsplan(
#         aggregationplan,
#         tree,
#         MLFMA.FarField{farfielddims(operator),T},
#         SamplingSizeFunctor(samplings, minrelevantlevel - 1),
#     )
# end

function fields(
    ::T,
    operator,
    tree,
    minrelevantlevel,
    testaggregationplan,
    trialaggregationplan,
    samplings,
) where {T}
    return H2Factory.storagemomentsplan(
        testaggregationplan,
        trialaggregationplan,
        tree,
        (level) -> farfieldtype(operator){farfielddims(operator),T},
        SamplingSizeFunctor(samplings, minrelevantlevel - 1),
    )
end

"""
    getfieldcollectionsdict(builder, operator, source, tree, samplings, minrelevantlevel; verbose=false)

Every leaf's field collection, as a `Dictionary` keyed by leaf, built according to `builder`.

`builder` carries the side, assembler and quadrature policy; see
`assembly/builders/collectionbuilders.jl`.

`source` depends on the builder: the basis
space for a [`BEASTFieldCollectionBuilder`](@ref), the already-built radiation collections for a
[`LazyReceiveFieldCollectionBuilder`](@ref) (which assembles nothing and so has no space to assemble
over).

[`leaffarfieldsamplings`](@ref MLFMA.Assembly.leaffarfieldsamplings) selects each leaf's grid from
`samplings` and `minrelevantlevel`, including leaves at different levels in non-uniform trees.
Materialized builders assemble on this grid; lazy builders take mirror indices from it, keeping
each leaf's collection and indices on the same sampling.

`farfieldlocal!` uses the far-field operator's `gamma` for box centering.
`radiationoperator`/`receiveoperator` in `operatorutils.jl` supply the sign for each side.
"""
function getfieldcollectionsdict(
    builder, operator, source, tree, samplings, minrelevantlevel::Int; verbose::Bool=false
)
    # Fix leaf order here; `H2Trees.leaves` may return traversal order rather than sorted IDs.
    leafids = sort(H2Trees.leaves(tree))
    leafsamplings = Assembly.leaffarfieldsamplings(
        tree, samplings, minrelevantlevel; leaves=leafids
    )

    farfieldcollections = buildfieldcollections(
        builder, operator, tree, leafsamplings, source; verbose=verbose
    )

    # Use the assembly's leaf IDs so dictionary keys and collection values have the same order.
    return Dictionary(Assembly.leafids(leafsamplings), farfieldcollections)
end

function getsamplingsparentsamplings(γ, ϵ, tree, levels, samplingfunction)
    isempty(levels) && return Nothing[], Nothing[]

    firstsampling, _ = samplingfunction(γ, ϵ, tree; level=levels[end])
    samplings = Vector{typeof(firstsampling)}(undef, length(levels))
    parentsamplings = Vector{typeof(firstsampling)}(undef, length(levels))
    samplings[end] = firstsampling

    for (i, level) in enumerate(levels)
        i == length(levels) && continue
        samplings[i], _ = samplingfunction(γ, ϵ, tree; level=level)
    end

    indices = Dict(level => i for (i, level) in enumerate(levels))
    for (i, level) in enumerate(levels)
        level == _samplingminimumlevel(samplingfunction, tree) && continue
        if haskey(indices, level - 1)
            parentsamplings[i] = samplings[indices[level - 1]]
        else
            parentsamplings[i], _ = samplingfunction(γ, ϵ, tree; level=level - 1)
        end
    end

    return samplings, parentsamplings
end

function getweights(samplings::Vector{S}) where {S<:Sampling.AbstractSampling}
    firstsampling = samplings[1]
    _, firstweights = Sampling.cartesiansamplepoints(firstsampling)
    weightsvector = Vector{typeof(firstweights)}(undef, length(samplings))
    weightsvector[1] = firstweights
    Threads.@threads for i in eachindex(samplings)
        i == 1 && continue
        _, weightsvector[i] = Sampling.cartesiansamplepoints(samplings[i])
    end
    return weightsvector
end

function getweights(samplings::Vector{Nothing})
    return Nothing[]
end

function cutoffparameter(γ, ϵ, diameter)
    κ = abs(γ)
    κd = κ * diameter
    digitsofaccuracy = log10(1 / ϵ)

    cutoffparameter = floor(Int, κd + 1.8 * digitsofaccuracy^(2 / 3) * (κd)^(1 / 3))
    cutoffparameter = max(1, cutoffparameter)
    return cutoffparameter
end

function sampling(
    γ::T,
    ϵ::AbstractFloat,
    diameter::T2,
    samplingbuilder=Sampling.SphericalSamplingBuilder{Float64}(),
) where {T<:Number,T2<:Number}
    cutoffparameter = Sampling.cutoffparameter(γ, ϵ, diameter)
    return samplingbuilder(Sampling.samplingrate(samplingbuilder, cutoffparameter)...),
    cutoffparameter
end

"""
    diameter(halfsize, dimension) -> T

The diagonal of a `dimension`-dimensional box of the given halfsize: `2·halfsize·√dimension`.

Callers supply the spatial dimension explicitly. A square's diagonal is `2·halfsize·√2`, with
factor `√D` in general. Using the cube factor `sqrt(3)` in 2-D would overstate bandwidth and
oversample in [`cutoffparameter`](@ref).
"""
function diameter(halfsize::T, dimension::Integer) where {T<:Number}
    return T(halfsize * sqrt(dimension) * 2)
end
