# Store the component count `D` in a type parameter and a `Val{D}` field so the functor bodies can
# infer it. `storerule` is an `AbstractFarFieldStoreRule` value; it selects the projection, not
# the output layout. Callers must supply both the rule and the far-field type.
"""
    BEASTFarFieldAssembler(operator, space, quadstrategy, storerule, ::Type{F})

Assembles a leaf's basis-function far-field patterns by calling BEAST.

Carries the target far-field type `F`: the component count is derived from it via
`fielddim`, so the two can never disagree, and a caller can ask an assembler which collection type to
wrap its output in. A bare count could not answer that, since two representations can share one.

`storerule` and `farfieldtype` are independent axes and both must be given. The first is how BEAST's
raw value is projected into the moment; the second is what layout the moment has. Neither derives
from the other, since `ScalarFarField` is reached both by radial projection and by storing as is.

Neither has a constructor default. Obtain them from `farfieldstorerule(op)` and
[`farfieldtype`](@ref MLFMA.Seams.farfieldtype)`(op)` for the operator being assembled.
"""
struct BEASTFarFieldAssembler{OperatorType,Space,QuadStrategyType,StoreRule,F,D}
    operator::OperatorType
    space::Space
    quadstrategy::QuadStrategyType
    storerule::StoreRule
    farfieldtype::Type{F}
    dimensionality::Val{D}
    function BEASTFarFieldAssembler(
        operator::OperatorType,
        space::Space,
        quadstratrategy::QuadStrategyType,
        storerule::StoreRule,
        ::Type{F},
    ) where {
        OperatorType,
        Space,
        QuadStrategyType,
        StoreRule<:AbstractFarFieldStoreRule,
        F<:AbstractFarField,
    }
        D = fielddim(F)
        return new{OperatorType,Space,QuadStrategyType,StoreRule,F,D}(
            operator, space, quadstratrategy, storerule, F, Val(D)
        )
    end
end

function farfieldtype(assembler::BEASTFarFieldAssembler)
    return assembler.farfieldtype
end

"""
    BEASTFarFieldAssemblerBuilder()

Builds a [`BEASTFarFieldAssembler`](@ref MLFMA.Assembly.BEASTFarFieldAssembler) from the five things a field-collection builder holds or
can derive: `(operator, space, quadstrategy, storerule, farfieldtype)`.

This singleton functor keeps the strategy field concrete. A stored assembler type would need a
`Type{A}` field annotation to avoid inference as `UnionAll` or `DataType`. Alternative builder
objects can carry construction state, such as a fixed store rule or quadrature, without requiring
their assembler's constructor to match `BEASTFarFieldAssembler`'s positional arguments.

See [`DummyFarFieldAssemblerBuilder`](@ref MLFMA.TestSupport.DummyFarFieldAssemblerBuilder) for the test-support counterpart.
"""
struct BEASTFarFieldAssemblerBuilder end

function (::BEASTFarFieldAssemblerBuilder)(
    operator, space, quadstrategy, storerule, farfieldtype
)
    return BEASTFarFieldAssembler(operator, space, quadstrategy, storerule, farfieldtype)
end

struct AssembleMomentsFunctor end

struct BufferFunctor end

# H2Factory's `assemblemoments!`/`assemblecluster!` indexes `clusterdata[cluster]` -- one item per
# cluster (e.g. iFMM's `BEASTMomentAssembler` passes one polynomial per leaf there). The plain,
# non-tree assembler call below splits work into artificial thread-clusters purely for
# parallelism, not because they're separate boxes -- every one of them is really the SAME single
# global `BoxFarFieldSampling` (see below), so this wraps that one value and returns it regardless
# of the index used to look it up. Subtypes `AbstractArray` (not just structurally `getindex`-able)
# because `assemblemoments!`'s entry point dispatches on `clusterdata::AbstractArray`.
struct ConstantClusterData{T} <: AbstractArray{T,1}
    value::T
end

Base.size(::ConstantClusterData) = (typemax(Int),)
Base.getindex(c::ConstantClusterData, ::Int) = c.value

"""
    _indexchunks(nitems, nchunks)

Split `1:nitems` into at most `nchunks` contiguous ranges, as a `Vector{Vector{Int}}`.

Used by the non-tree assembler, whose clusters partition basis-function ids rather than boxes.
Do not emit empty chunks, including when `nitems < nchunks`. Nonpositive `nitems` returns an empty
vector; nonpositive `nchunks` is treated as one.
"""
function _indexchunks(nitems::Int, nchunks::Int)
    nitems <= 0 && return Vector{Int}[]
    chunksize = cld(nitems, max(nchunks, 1))
    return [collect(chunk) for chunk in Iterators.partition(1:nitems, chunksize)]
end

# Each leaf's assembly recipe carries its direction grid, box center and leaf id.
# Non-uniform trees can use a different grid at each level. `farfieldlocal!` subtracts the center
# before evaluating the exponential:
# `exp(γ·y·x)·exp(-γ·y·center) = exp(γ·y·(x-center))`.
# No separate phase-correction array is needed.
#
# Every per-leaf structure -- `Fs`, assembly clusters, local index maps and dictionary keys -- must
# describe the same leaf at each position. Consumers read this order from `leafsamplings` through
# `leafids`, rather than walking the tree independently. The flat path has no leaves and uses `nothing`.
struct BoxFarFieldSampling{S,P,C,L}
    sampling::S
    samplingpoints::P
    center::C
    leaf::L
end

"""
    leafids(leafsamplings)

The leaf ids `leafsamplings` was built for, in assembly order.

The single source of leaf order for everything downstream of
[`leaffarfieldsamplings`](@ref MLFMA.Assembly.leaffarfieldsamplings): moment blocks, assembly clusters, index
maps and the leaf-keyed collection dictionary all read it from here rather than calling
`H2Trees.leaves(tree)` again.
"""
leafids(leafsamplings::AbstractVector{<:BoxFarFieldSampling}) =
    [leafsampling.leaf for leafsampling in leafsamplings]

# `samplings` is keyed by level via `minrelevantlevel` the same way `StorageFieldsFunctor` and
# friends already are, so the indexing convention is shared rather than restated here.
"""
    leaffarfieldsamplings(tree, samplings, minrelevantlevel; leaves=H2Trees.leaves(tree))

One `BoxFarFieldSampling` per leaf, in the order `leaves` gives.

Each entry pairs a leaf with the direction grid for ITS OWN level, looked up as
`samplings[level - minrelevantlevel + 1]`. That is what makes non-uniform trees work: a leaf sitting
at a coarser level than its siblings gets that level's smaller grid, instead of every leaf being
forced to share the finest level's.

Downstream per-leaf arrays read the leaf ids back off these entries via
[`leafids`](@ref MLFMA.Assembly.leafids) rather than recomputing them, so the ordering here is the
single definition of per-leaf order.

The direction grid depends only on the LEVEL, so it is computed once per level and shared by
reference among all leaves at that level; only the box centre genuinely varies per leaf.
"""
function leaffarfieldsamplings(
    tree, samplings, minrelevantlevel::Int; leaves=H2Trees.leaves(tree)
)
    firstsamplingpoints, _ = Sampling.cartesiansamplepoints(samplings[begin])
    samplingpointsperlevel = Vector{typeof(firstsamplingpoints)}(undef, length(samplings))
    samplingpointsperlevel[begin] = firstsamplingpoints
    for i in eachindex(samplings)
        i == firstindex(samplings) && continue
        samplingpointsperlevel[i], _ = Sampling.cartesiansamplepoints(samplings[i])
    end

    firstlevelindex = _leaflevelindex(tree, leaves[begin], minrelevantlevel, samplings)
    firstleafsampling = BoxFarFieldSampling(
        samplings[firstlevelindex],
        samplingpointsperlevel[firstlevelindex],
        H2Trees.center(tree, leaves[begin]),
        leaves[begin],
    )

    leafsamplings = Vector{typeof(firstleafsampling)}(undef, length(leaves))
    leafsamplings[begin] = firstleafsampling

    for (i, leaf) in enumerate(leaves)
        i == firstindex(leaves) && continue

        levelindex = _leaflevelindex(tree, leaf, minrelevantlevel, samplings)
        leafsamplings[i] = BoxFarFieldSampling(
            samplings[levelindex],
            samplingpointsperlevel[levelindex],
            H2Trees.center(tree, leaf),
            leaf,
        )
    end

    return leafsamplings
end

"""
    _leaflevelindex(tree, leaf, minrelevantlevel, samplings)

Index into `samplings` for `leaf`'s own level, floored at the coarsest available sampling.
"""
function _leaflevelindex(tree, leaf, minrelevantlevel, samplings)
    return _samplinglevelindex(H2Trees.level(tree, leaf), minrelevantlevel, samplings)
end

"""
    _samplinglevelindex(level, minrelevantlevel, samplings)

The level-keyed core of [`_leaflevelindex`](@ref), for callers that already hold a LEVEL rather than
a leaf (notably `MLFMA.leafscratchchannel`, which builds one scratch far field per level and so has
no representative leaf to ask).
"""
function _samplinglevelindex(level::Int, minrelevantlevel::Int, samplings)
    levelindex = level - minrelevantlevel + 1
    if levelindex > lastindex(samplings)
        error(
            "no sampling available for a leaf at level $level: index " *
            "$levelindex exceeds the $(length(samplings)) sampling(s) spanning relevantlevels " *
            "from $minrelevantlevel. `relevantlevels` must cover every level down to the " *
            "deepest leaf of this tree.",
        )
    end
    return max(levelindex, firstindex(samplings))
end

# Read the component count from `Val{D}` without listing unrelated assembler type parameters.
# This also avoids a Julia 1.12 parser issue with wrapped callable-struct annotations: a split
# `BEASTFarFieldAssembler{OperatorType,...}` can leave its `where` parameters out of scope
# (`UndefVarError: OperatorType not defined`). Keep the callable annotations unparameterized.
_dimensionvalue(::Val{D}) where {D} = D

function (assembler::BEASTFarFieldAssembler)(sampling; verbose::Bool=false)
    D = _dimensionvalue(assembler.dimensionality)
    T = scalartype(operator(assembler))

    Fsarray = Vector{Matrix{T}}(undef, D)
    for i in eachindex(Fsarray)
        Fsarray[i] = zeros(T, length(sampling), numfunctions(space(assembler)))
    end
    Fs = SVector{D,Matrix{T}}(Fsarray)
    storer = H2Factory.buildstorer(Fs, assembler, sampling)

    cartesiansamplingpoints, _ = Sampling.cartesiansamplepoints(sampling)

    # A single box at the origin uses the same functor and storer as tree assembly.
    # Subtracting its zero center leaves the global-origin kernel unchanged.
    globalbox = BoxFarFieldSampling(
        sampling, cartesiansamplingpoints, zero(eltype(cartesiansamplingpoints)), nothing
    )

    # NOT LEAF CLUSTERS. This path has no tree; these are artificial chunks of basis-function ids,
    # split only so the assembly parallelises. `assemblemoments!` calls them clusters because that
    # is what they are on the tree path, and `ConstantClusterData` above hands every one of them the
    # same single global box precisely because they are not separate boxes.
    clusters = _indexchunks(numfunctions(space(assembler)), Threads.nthreads())

    H2Factory.assemblemoments!(
        storer,
        operator(assembler),
        space(assembler),
        clusters,
        ConstantClusterData(globalbox),
        Val(Operators.farfieldvaluetype(operator(assembler))),
        quadstrategy(assembler),
        AssembleMomentsFunctor(),
        BufferFunctor();
        verbose=verbose,
    )

    return Fs
end

function (assembler::BEASTFarFieldAssembler)(
    tree, leafsamplings::AbstractVector{<:BoxFarFieldSampling}; verbose::Bool=false
)
    D = _dimensionvalue(assembler.dimensionality)
    T = scalartype(operator(assembler))

    # `Fs[i]`, `clusters[i]`, storer indices and dictionary keys must all describe
    # `leafsamplings[i]`'s leaf. Do not use `leafclusters(tree)`, which chooses its own leaf order.
    leaves = leafids(leafsamplings)

    Fs = Vector{SVector{D,Matrix{T}}}(undef, length(leaves))

    Threads.@threads for i in eachindex(leaves)
        Fvector = Vector{Matrix{T}}(undef, D)
        for j in eachindex(Fvector)
            Fvector[j] = zeros(
                T,
                length(leafsamplings[i].samplingpoints),
                length(H2Trees.values(tree, leaves[i])),
            )
        end
        Fs[i] = SVector{D,Matrix{T}}(Fvector)
    end

    storer = H2Factory.buildstorer(Fs, assembler, leafsamplings, tree)

    clusters = [H2Trees.values(tree, leaf) for leaf in leaves]

    H2Factory.assemblemoments!(
        storer,
        operator(assembler),
        space(assembler),
        clusters,
        leafsamplings,
        Val(Operators.farfieldvaluetype(operator(assembler))),
        quadstrategy(assembler),
        AssembleMomentsFunctor(),
        BufferFunctor();
        verbose=verbose,
    )

    return Fs
end

function operator(assembler::BEASTFarFieldAssembler)
    return assembler.operator
end

function space(assembler::BEASTFarFieldAssembler)
    return assembler.space
end

function quadstrategy(assembler::BEASTFarFieldAssembler)
    return assembler.quadstrategy
end

# Deliberately NOT an `H2Factory.storetrait` method. That protocol's contract is "return a storer
# TYPE that `buildstorer` calls as a constructor", which cannot carry a rule VALUE; this assembler
# overrides `H2Factory.buildstorer` instead (see `assembly/store/store.jl`). Defining `storetrait` here as
# well would leave two ways in, one of which silently ignores the rule.
function farfieldstorerule(assembler::BEASTFarFieldAssembler)
    return assembler.storerule
end

function (f::AssembleMomentsFunctor)(
    store,
    operator,
    trial,
    functionids,
    refspace,
    computationbuffer,
    elements,
    elementids,
    ad,
    qdata,
    qstrat,
    leafsampling::BoxFarFieldSampling,
    cluster,
)
    for (e, elementid) in enumerate(elementids)
        @inbounds element = elements[elementid]
        qr = quadrule(operator, refspace, e, element, qdata, qstrat)
        for (p, point) in enumerate(leafsampling.samplingpoints)
            Assembly.farfieldlocal!(
                computationbuffer, operator, point, qr, leafsampling.center
            )
            for (r, z) in enumerate(computationbuffer)
                for (functionid, b) in (ad[elementid, r])
                    functionid ∉ functionids && continue
                    store(z * b, cluster, p, functionid)
                end
            end
        end
    end
end

# Allocate one scratch slot per shape function on the basis's reference domain.
# Read the domain from the geometry: triangles use `ReferenceSimplex{2,Float64,3}`, while 2-D
# problems use segments (`ReferenceSimplex{1,Float64,2}`). Assuming a triangle would overrun the
# segment's assembly data when the scratch is consumed.
function (::BufferFunctor)(ops, basis, refspace, clusterdata, type::Val{T}) where {T}
    geometry = BEAST.geometry(basis)
    referencedomain = CompScienceMeshes.domain(
        CompScienceMeshes.chart(geometry, first(geometry))
    )
    return Array{T}(undef, numfunctions(refspace, referencedomain))
end

@inline function farfieldkernelvals(op, y, x, center)
    return exp(op.gamma * Phase.dotu(y, cartesian(x) - center))
end

# `integrand` never reads `x`'s absolute position except through `krn` (only `normal(x)`/basis
# values, both translation-invariant), so replacing the kernel computation with
# `farfieldkernelvals` (box-centered) is a safe drop-in.
@inline function farfieldlocal!(zlocal, op, y, qr, center)
    firstindex = first(eachindex(qr))
    for (i, q) in enumerate(qr)
        x = q.point
        F = q.value
        dx = q.weight

        krn = farfieldkernelvals(op, y, x, center)
        if i == firstindex
            for r in eachindex(zlocal)
                @inbounds zlocal[r] = integrand(op, krn, y, F[r], x) * dx
            end
        else
            for r in eachindex(zlocal)
                @inbounds zlocal[r] = muladd(integrand(op, krn, y, F[r], x), dx, zlocal[r])
            end
        end
    end
end

function _removeradialcomponent(array::AbstractArray{SVector{3,T}}) where {T}
    Fθ = zeros(T, size(array))
    Fφ = zeros(T, size(array))
    for a in eachindex(array)
        Fθ[a] = array[a][2]
        Fφ[a] = array[a][3]
    end
    return Fθ, Fφ
end
