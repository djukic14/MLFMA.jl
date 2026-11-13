# The storer and its projection kernels. `operators/storerules.jl` defines the rules before
# `BEASTFarFieldAssembler` is loaded; this file also needs that assembler and `BoxFarFieldSampling`.
# Store rules select projections independently of the layout selected by `farfieldtype`.

# --- frame cache ----------------------------------------------------------------------------------

"""
    SphericalFrameCache(sampling)

Precomputed `sin`/`cos` of a sampling's θ and φ values, plus the sampling itself, for the rules that
project a Cartesian vector onto the spherical frame.

Keep the tables with their sampling so they cannot fall out of step. On the tree path, the cache
vector is indexed by cluster; leaves at the same level share a cache, while leaves at different
levels may use different direction grids.
"""
struct SphericalFrameCache{T,S}
    sinθs::T
    cosθs::T
    sinφs::T
    cosφs::T
    sampling::S
end

function SphericalFrameCache(sampling)
    return SphericalFrameCache(
        sin.(Sampling.θs(sampling)),
        cos.(Sampling.θs(sampling)),
        sin.(Sampling.φs(sampling)),
        cos.(Sampling.φs(sampling)),
        sampling,
    )
end

"""
    needsframecache(rule)

Whether `rule` projects onto the spherical frame and therefore needs trig tables.

Keep this predicate separate from [`framecachefor`](@ref), which dispatches on the sampling shape.
A no-cache specialization there would compete with the vector-sampling method: the pair
`(StoreAsFarFieldValue, Vector{BoxFarFieldSampling})` would be more specific in different arguments,
causing an ambiguous-method error on the tree path.

The rules are singletons, so this folds to a compile-time constant and the branch in `framecachefor`
disappears.
"""
needsframecache(::StoreAsFarFieldValue) = false
needsframecache(::AbstractFarFieldStoreRule) = true

"""
    framecachefor(rule, samplinginfo)

The frame cache `rule` needs, or `nothing` for a rule that does no projection.

Return `nothing` rather than an empty cache so [`StoreAsFarFieldValue`](@ref) allocates no trig
tables. Projection rules build four `sin`/`cos` arrays per sampling.
"""
function framecachefor(rule::AbstractFarFieldStoreRule, sampling)
    needsframecache(rule) || return nothing
    return SphericalFrameCache(sampling)
end

# Share one cache by reference among leaves using the same sampling object. `leaffarfieldsamplings`
# guarantees that identity within a level. The vector still has one entry per cluster.
# Use `IdDict`, not equality, so distinct sampling objects from different levels remain separate.
# Derive the concrete element type from the first cache to avoid dynamic dispatch when storing values.
function framecachefor(
    rule::AbstractFarFieldStoreRule,
    leafsamplings::AbstractVector{<:Assembly.BoxFarFieldSampling},
)
    needsframecache(rule) || return nothing
    isempty(leafsamplings) && return SphericalFrameCache[]

    firstcache = SphericalFrameCache(leafsamplings[begin].sampling)
    C = typeof(firstcache)
    bysampling = IdDict{Any,C}(leafsamplings[begin].sampling => firstcache)

    caches = Vector{C}(undef, length(leafsamplings))
    for (i, leafsampling) in enumerate(leafsamplings)
        caches[i] = get!(bysampling, leafsampling.sampling) do
            return SphericalFrameCache(leafsampling.sampling)
        end
    end
    return caches
end

# One entry per cluster on the tree path, one shared cache on the flat path, nothing when the rule
# needs none. Written as dispatch rather than a branch so the flat and no-cache cases compile to
# nothing at all.
@inline _cacheat(cache::AbstractVector, cluster::Int) = @inbounds cache[cluster]
@inline _cacheat(::Nothing, ::Int) = nothing

# --- the storer -----------------------------------------------------------------------------------

"""
    FarFieldStorer(rule, Fs, samplinginfo)
    FarFieldStorer(rule, Fs, leafsamplings, tree)

The single `H2Factory.AbstractStorer` for MLFMA's far-field assembly: a projection [rule](@ref
AbstractFarFieldStoreRule) plus the moments it writes into, the leaf-local index map, and the
rule's frame cache.

The `(v, ::Nothing, m, n)` and `(v, cluster, m, n)` methods resolve the moment block and its indices.
Each rule's `_store!` method performs the projection.

The three-argument form is the flat, non-tree path: one global moment block, no leaf-local
renumbering (`indexinleafbox === nothing`). The four-argument form is the tree path: one block per
leaf, indexed by cluster.
"""
struct FarFieldStorer{R<:AbstractFarFieldStoreRule,T,I,C} <: H2Factory.AbstractStorer
    rule::R
    Fs::T
    indexinleafbox::I
    framecache::C
end

function FarFieldStorer(rule::AbstractFarFieldStoreRule, Fs, samplinginfo)
    return FarFieldStorer(rule, Fs, nothing, framecachefor(rule, samplinginfo))
end

function FarFieldStorer(
    rule::AbstractFarFieldStoreRule,
    Fs,
    leafsamplings::AbstractVector{<:Assembly.BoxFarFieldSampling},
    tree,
)
    return FarFieldStorer(
        rule,
        Fs,
        _indexinleafbox(tree, Assembly.leafids(leafsamplings)),
        framecachefor(rule, leafsamplings),
    )
end

# Build leaf-local indices in the supplied order, unlike `H2Factory.indexinleafbox(tree)`, which
# walks the leaves itself. `Fs[cluster]`, `framecache[cluster]` and `indexinleafbox[cluster]` must
# describe the same leaf.
function _indexinleafbox(tree, leaves)
    return [
        Dict{Int,Int}(
            zip(H2Trees.values(tree, leaf), 1:length(H2Trees.values(tree, leaf)))
        ) for leaf in leaves
    ]
end

# MLFMA's far-field assemblers build their storer from their OWN rule rather than through
# `H2Factory.storetrait`, whose contract is "return a type you can call as a constructor" and so
# cannot carry a rule value. Overriding `buildstorer` for the assembler type is more specific than
# H2Factory's generic `buildstorer(F, op, storeinfo) = storetrait(op)(F, storeinfo)`, so this wins
# dispatch; H2Factory's own protocol is left completely intact for anything else that
# still uses it.
function H2Factory.buildstorer(Fs, assembler::BEASTFarFieldAssembler, samplinginfo)
    return FarFieldStorer(farfieldstorerule(assembler), Fs, samplinginfo)
end

function H2Factory.buildstorer(Fs, assembler::BEASTFarFieldAssembler, leafsamplings, tree)
    return FarFieldStorer(farfieldstorerule(assembler), Fs, leafsamplings, tree)
end

# `assemblemoments!` gives the tree path a range of candidate clusters; find the one holding basis
# function `n`. `I === Nothing` means flat moments with no per-cluster structure, so the
# range is irrelevant.
function (storer::FarFieldStorer{R,T,I})(
    v, clusters::UnitRange, m::Int, n::Int
) where {R<:AbstractFarFieldStoreRule,T,I}
    I === Nothing && return storer(v, nothing, m, n)
    for cluster in clusters
        if haskey(storer.indexinleafbox[cluster], n)
            return storer(v, cluster, m, n)
        end
    end
    return nothing
end

# The plain, non-tree assembler call still passes a real `cluster::Int` -- its artificial
# thread-clusters, which are a work partition and not real per-box structure (there is one global
# `Fs`, not one per cluster). `I === Nothing` is exactly the signal that the moments were built
# flat, so any `cluster::Int` in that case is treated identically to `::Nothing`.
#
# Keep `R<:AbstractFarFieldStoreRule`: specificity is computed from the
# signature as written, not from the struct's own field bounds: with a bare `where {R,T,C}` this
# method is more specific than the generic one below in its third parameter (`Nothing` vs free) but
# less specific in its first (`Any` vs `<:AbstractFarFieldStoreRule`), so neither wins and every
# flat-path call raises `MethodError: ... is ambiguous`. Restating the bound makes this strictly
# more specific and the ambiguity disappears.
#
function (storer::FarFieldStorer{R,T,Nothing,C})(
    v, cluster::Int, m::Int, n::Int
) where {R<:AbstractFarFieldStoreRule,T,C}
    return storer(v, nothing, m, n)
end

function (storer::FarFieldStorer)(v, ::Nothing, m::Int, n::Int)
    return _store!(storer.rule, storer.Fs, storer.framecache, v, m, n)
end

function (storer::FarFieldStorer)(v, cluster::Int, m::Int, n::Int)
    return _store!(
        storer.rule,
        @inbounds(storer.Fs[cluster]),
        _cacheat(storer.framecache, cluster),
        v,
        m,
        @inbounds(storer.indexinleafbox[cluster][n]),
    )
end

# --- projection kernels ---------------------------------------------------------------------------
#
# Each `_store!` takes the moment components for ONE cluster (or the flat set) already resolved, so
# the flat and per-cluster paths share one implementation per rule instead of two near-identical
# ones. `+=` throughout: `assemblemoments!` accumulates element contributions into the same entry.

@inline function _store!(::StoreAsFarFieldValue, Fs, ::Nothing, v, m::Int, n::Int)
    for i in eachindex(Fs)
        @inbounds Fs[i][m, n] += v[i]
    end
    return nothing
end

@inline function _store!(
    ::StoreCartesianAsSphericalVector, Fs, cache::SphericalFrameCache, v, m::Int, n::Int
)
    sinθ, cosθ, sinφ, cosφ = _frameat(cache, m)
    @inbounds Fs[1][m, n] += _radial(v, 0, sinθ, cosθ, sinφ, cosφ)
    @inbounds Fs[2][m, n] += _polar(v, 0, sinθ, cosθ, sinφ, cosφ)
    @inbounds Fs[3][m, n] += _azimuthal(v, 0, sinφ, cosφ)
    return nothing
end

@inline function _store!(
    ::StoreCartesianAsTangentialSpherical, Fs, cache::SphericalFrameCache, v, m::Int, n::Int
)
    sinθ, cosθ, sinφ, cosφ = _frameat(cache, m)
    @inbounds Fs[1][m, n] += _polar(v, 0, sinθ, cosθ, sinφ, cosφ)
    @inbounds Fs[2][m, n] += _azimuthal(v, 0, sinφ, cosφ)
    return nothing
end

@inline function _store!(
    ::StoreCartesianAsRadialScalar, Fs, cache::SphericalFrameCache, v, m::Int, n::Int
)
    sinθ, cosθ, sinφ, cosφ = _frameat(cache, m)
    @inbounds Fs[1][m, n] += _radial(v, 0, sinθ, cosθ, sinφ, cosφ)
    return nothing
end

# The six-element input holds two Cartesian vectors back to back. Offsets let the projection
# helpers index either vector without constructing views for each sample, basis function and element.
@inline function _store!(
    ::StoreDoubleCartesianAsPairedSpherical,
    Fs,
    cache::SphericalFrameCache,
    v,
    m::Int,
    n::Int,
)
    sinθ, cosθ, sinφ, cosφ = _frameat(cache, m)
    @inbounds Fs[1][m, n] += _radial(v, 0, sinθ, cosθ, sinφ, cosφ)
    @inbounds Fs[2][m, n] += _polar(v, 0, sinθ, cosθ, sinφ, cosφ)
    @inbounds Fs[3][m, n] += _azimuthal(v, 0, sinφ, cosφ)
    @inbounds Fs[4][m, n] += _radial(v, 3, sinθ, cosθ, sinφ, cosφ)
    @inbounds Fs[5][m, n] += _polar(v, 3, sinθ, cosθ, sinφ, cosφ)
    @inbounds Fs[6][m, n] += _azimuthal(v, 3, sinφ, cosφ)
    return nothing
end

# `m` is the global sample-point index. Unpack it into θ and φ indices to look up the frame.
@inline function _frameat(cache::SphericalFrameCache, m::Int)
    θindex, φindex = Sampling.θφindicesfromglobalindex(m, cache.sampling)
    return (
        @inbounds(cache.sinθs[θindex]),
        @inbounds(cache.cosθs[θindex]),
        @inbounds(cache.sinφs[φindex]),
        @inbounds(cache.cosφs[φindex]),
    )
end

# The three spherical components of a Cartesian vector stored at `v[offset+1:offset+3]`. Scalar
# indexing with an explicit offset rather than a slice: see the note on
# `StoreDoubleCartesianAsPairedSpherical`.
@inline function _radial(v, offset::Int, sinθ, cosθ, sinφ, cosφ)
    return @inbounds v[offset + 1] * sinθ * cosφ +
        v[offset + 2] * sinθ * sinφ +
        v[offset + 3] * cosθ
end

@inline function _polar(v, offset::Int, sinθ, cosθ, sinφ, cosφ)
    return @inbounds v[offset + 1] * cosθ * cosφ + v[offset + 2] * cosθ * sinφ -
        v[offset + 3] * sinθ
end

@inline function _azimuthal(v, offset::Int, sinφ, cosφ)
    return @inbounds -v[offset + 1] * sinφ + v[offset + 2] * cosφ
end
