# Assemble Cartesian patterns on a Gauss-Legendre grid, analyse them, and store truncated coefficients.
#
# The production grid resolves the translation bandwidth `L` and its excess-bandwidth term:
# `2(L+1) x 4(L+1) = 8(L+1)²` directions. The analysis grid only resolves the pattern's bandwidth.
# `gausslegendregrid` is exact in cos θ, while `MLFMA.sphericalsamplepoints` merely converges.
# It needs `(Pa+1) x (2Pa+2) = 2(Pa+1)²` directions, a quarter as many at `Pa ≈ L`.
# Setup, including analysis GEMMs, took 0.28-0.34x the sampled leaf assembly time in
# `benchmark/interpolation/shepatterns.jl`.
#
# The translation bandwidth `L` gives a scale for the analysis order, not a bound `P ≤ L`.
# `L` comes from the box diagonal; `P` depends on the basis function's support about the box centre.
# The calibration sweep measured `P*/L` from 0.65 to 1.00, but used boxes at or above the mesh scale.
#
# Support can extend beyond its box when the mesh is coarser than `minhalfsize`.
# For λ/10 elements in λ/12 boxes (halfsize 0.083λ), L = 7 but the rule returns P = 8.
#
# `L + SHEANALYSISMARGIN` is therefore a heuristic. If the rule returns the analysis order itself,
# the grid may be limiting the result. `shecollections` raises to prevent silent under-resolution.
#
# `leveltruncationorder` takes the maximum order over a level's boxes.
# All leaves at a level need one order because a batched multiply uses one synthesis matrix.

"""
    SHEANALYSISMARGIN

Degrees added to a level's `L` when choosing its analysis order, `2`.

`L` is not an upper bound on `P`: a basis function's support can extend beyond its box.
The margin allows for this and for the round-off plateau in the coefficient tail.
It is a heuristic. [`shecollections`](@ref) raises if the selected order reaches the analysis order.
"""
const SHEANALYSISMARGIN = 2

"""
    sheanalysisorder(sampling; margin=SHEANALYSISMARGIN) -> Int

Return the translation bandwidth `L` of `sampling` plus `margin` for Gauss-Legendre analysis.
"""
sheanalysisorder(sampling::AbstractSampling; margin::Integer=SHEANALYSISMARGIN) =
    cutoffparameter(sampling) + margin

# Group leaves by sampling identity. `leaffarfieldsamplings` shares one sampling object per level.
# An `IdDict` keeps structurally equal samplings from different levels separate.
function _grouptheleavesbylevel(leafsamplings)
    groups = IdDict{Any,Vector{Int}}()
    for (i, leafsampling) in enumerate(leafsamplings)
        push!(get!(() -> Int[], groups, leafsampling.sampling), i)
    end
    return groups
end

# `analyze` returns `(nc_a × 3nbf)` with component-major columns (`(c-1)nbf + j`).
# `SphericalHarmonicStorage` wants `(3nc × nbf)` with column `j` holding basis function `j`'s
# `(nc × 3)` block flattened lm-fastest. Repack and truncate to `nc` in one pass.
function _repackcoefficients(analysed, nbf, nc)
    repacked = Matrix{eltype(analysed)}(undef, 3nc, nbf)
    @inbounds for j in 1:nbf, component in 1:3
        source = (component - 1) * nbf + j
        destination = (component - 1) * nc
        for q in 1:nc
            repacked[destination + q, j] = analysed[q, source]
        end
    end
    return repacked
end

"""
    shecollections(assembler, tree, leafsamplings, ε; margin=SHEANALYSISMARGIN)
        -> (collections, orders)

Build one [`SphericalHarmonicRadiationCollection`](@ref) per leaf, and report the order chosen for
each level.

`collections` is a concretely typed `Dictionary` keyed by leaf id, as in the sampled path.
`H2Factory.buildleafchannels` identifies compressed storage by its element type.
`orders` maps each level's sampling object to the selected `P`.

`assembler` must use `StoreAsFarFieldValue` with `CartesianVectorFarField`.
Tangential components θ̂/φ̂ carry the spherical frame's pole singularity and do not compress well.
SHE expands Cartesian components and projects after synthesis. An ordinary EFIE assembler
would produce two-component storage, which
[`SphericalHarmonicRadiationCollection`](@ref) rejects.

`ε` is the map's tolerance. `leveltruncationorder` uses `max(ε/100, SHETAILFLOOR)`;
see that function for the calibration of this factor.
"""
function shecollections(
    assembler, tree, leafsamplings, ε::Real; margin::Integer=SHEANALYSISMARGIN
)
    leaves = leafids(leafsamplings)
    isempty(leaves) &&
        throw(ArgumentError("no leaves to build spherical-harmonic storage for"))
    # Infer the vector's element type from the first collection.
    # All levels share that type because `P` is a storage field, not a type parameter.
    collections = nothing
    orders = IdDict{Any,Int}()

    for (sampling, indices) in _grouptheleavesbylevel(leafsamplings)
        Pa = sheanalysisorder(sampling; margin=margin)
        θs, φs, w = SphericalHarmonics.gausslegendregrid(Pa)
        analysis = SphericalHarmonics.SphericalHarmonicTransform(ComplexF64, θs, φs, w, Pa)
        points = [
            SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)) for (θ, φ) in zip(θs, φs)
        ]

        # Use the tree assembler with explicit Gauss-Legendre directions.
        # `StoreAsFarFieldValue` needs no frame cache and does not read the accompanying `sampling`.
        boxes = [
            Assembly.BoxFarFieldSampling(
                sampling, points, H2Trees.center(tree, leaves[i]), leaves[i]
            ) for i in indices
        ]
        Fs = assembler(tree, boxes; verbose=false)

        analysed = map(Fs) do F
            nbf = size(F[1], 2)
            stacked = Matrix{ComplexF64}(undef, size(F[1], 1), 3nbf)
            for component in 1:3
                @views stacked[:, ((component - 1) * nbf + 1):(component * nbf)] .= F[component]
            end
            return SphericalHarmonics.analyze(analysis, stacked)
        end

        P = SphericalHarmonics.leveltruncationorder(analysed, ε)
        P < Pa || error(
            "the spherical-harmonic analysis grid was the binding constraint at this level: the " *
            "order rule returned $P against an analysis order of $Pa, so it is a CLAMP rather " *
            "than a choice and the patterns are under-resolved. This level's translation " *
            "bandwidth is L = $(cutoffparameter(sampling)); `P ≤ L` is NOT guaranteed (see this " *
            "file's header -- a mesh coarser than the box breaks it), so raise the `margin` " *
            "keyword until the rule returns an order strictly below it.",
        )
        nc = SphericalHarmonics.numcoefficients(P)
        orders[sampling] = P

        for (k, i) in enumerate(indices)
            leaf = leaves[i]
            nbf = size(Fs[k][1], 2)
            collection = SphericalHarmonicRadiationCollection(
                SphericalHarmonics.SphericalHarmonicStorage(
                    P,
                    SVector{3}(H2Trees.center(tree, leaf)),
                    _repackcoefficients(analysed[k], nbf, nc),
                ),
            )
            if collections === nothing
                collections = Vector{typeof(collection)}(undef, length(leaves))
            end
            collections[i] = collection
        end
    end

    return Dictionary(leaves, collections), orders
end

"""
    sheorders(tree, collections, samplings, minrelevantlevel) -> IdDict

Read each level's expansion order from its collections, keyed by the level's sampling object.
This recovers the second return value of [`shecollections`](@ref).

Collections belong to the core; `H2Factory.buildleafchannels` sizes synthesis matrices per workspace.
Reading the orders from the collections keeps scratch consistent with stored coefficients.
A smaller scratch order can silently give wrong results even when the GEMM shapes conform.

Raise if leaves at one level have different orders. A chunk uses one synthesis matrix.
"""
function sheorders(tree, collections, samplings, minrelevantlevel)
    return _harmonicorders(
        tree, collections, samplings, minrelevantlevel, "spherical-harmonic"
    )
end

# Construction policy for either assembled side. `MLFMAStrategy` selects it from
# `trialleafstorage` or `testleafstorage` through the standard `buildfieldcollections` protocol.
# A Petrov test space is assembled with `ReceiveSide()`. A Galerkin receive pattern is derived
# from radiation coefficients by the lazy builder, with parity folded into the level's scratch.

"""
    SphericalHarmonicFieldCollectionBuilder(side, assemblerbuilder, quadstrategy, ε)

Construction policy for spherical-harmonic leaf storage: assemble Cartesian patterns on a
Gauss-Legendre grid, analyse them, and keep the coefficients. Eibert's SE-MLFMA.

`ε` is the map's tolerance, used by [`shecollections`](@ref) to choose the order at construction.

Normally obtained from a strategy carrying
[`SphericalHarmonicLeafStorage`](@ref MLFMA.Assembly.SphericalHarmonicLeafStorage) rather than built by hand -- see
[`radiationfieldcollectionbuilder`](@ref) and [`receivefieldcollectionbuilder`](@ref).

`side` is [`RadiationSide`](@ref) or [`ReceiveSide`](@ref). It selects the far-field operator
through `fieldoperator`. Both sides use the same Cartesian substitution, grid, order rule, and storage.
"""
struct SphericalHarmonicFieldCollectionBuilder{
    Side<:AbstractFieldCollectionSide,A,Q,E<:Real
} <: AbstractCompressedFieldCollectionBuilder{Side}
    side::Side
    assemblerbuilder::A
    quadstrategy::Q
    ε::E
end

"""
    buildfarfieldassembler(builder::SphericalHarmonicFieldCollectionBuilder, op, space)

Build the Cartesian far-field assembler for harmonic analysis.

Use `StoreAsFarFieldValue()` with `CartesianVectorFarField` instead of the operator's store rule
and far-field type. Tangential components θ̂/φ̂ carry the frame's pole singularity and do not
compress well. Cartesian components are smooth on the sphere. Projection back to tangential
components happens after synthesis on the level's grid. See `sphericalharmoniccollection.jl`.

The side's far-field operator and sign of γ are unchanged, so the stored patterns match the sampled path.
"""
function buildfarfieldassembler(builder::SphericalHarmonicFieldCollectionBuilder, op, space)
    return tangentialassembler(builder, op, space, "Spherical-harmonic leaf storage")
end

# `verbose` is ignored because assembly runs per level; progress bars would show partial builds.
#TODO: what is this docstring doing here?
"""
    buildfieldcollections(builder::SphericalHarmonicFieldCollectionBuilder{ReceiveSide}, op, ...)

The ASSEMBLED (Petrov) test side, compressed: the same analysis, then one wrapper per leaf.

`buildfarfieldassembler` builds this one from `fieldoperator(ReceiveSide(), op)` --
`receiveoperator(op)`, which already carries the receive sign of γ -- so the coefficients that come
back ARE the receive patterns and nothing is left to correct.

WHAT AN ASSEMBLED RECEIVE SIDE DOES NOT NEED is everything that makes the Galerkin one non-trivial:
the reference to the trial leaf, the antipodal parity, and the operator's receive coupling all exist
because that side DERIVES a pattern it does not store. This side stores its own, so the #129 seam --
where assuming the geometric mirror and the operator's component map coincide was silently wrong --
is not on this path at all.
"""

# These hooks select analysis and receive wrappers for the shared compressed builder protocol.
function analysecollections(
    builder::SphericalHarmonicFieldCollectionBuilder, assembler, tree, leafsamplings
)
    return shecollections(assembler, tree, leafsamplings, builder.ε)
end

assembledreceivecollectiontype(::SphericalHarmonicFieldCollectionBuilder) =
    SphericalHarmonicAssembledReceiveCollection

# The radiation collection type selects the lazy Galerkin receive storage.
# `Y_lm(-k̂) = (-1)^l Y_lm(k̂)` is folded into `scratch.mirror`; θ̂/φ̂ signs come from the frame.
# Each receive collection references radiation storage and keeps the operator's coupling.
# It owns no coefficient data or direction permutation.
function buildfieldcollections(
    ::LazyReceiveFieldCollectionBuilder,
    op,
    tree,
    leafsamplings,
    radiationcollections::AbstractDictionary{<:Any,<:SphericalHarmonicRadiationCollection};
    verbose::Bool=false,
)
    leaves = leafids(leafsamplings)
    K = receivecoupling(op)
    return [
        SphericalHarmonicReceiveCollection(radiationcollections[leaf], K) for leaf in leaves
    ]
end

"""
    shescratch(sampling, P; ngroups=1) -> SphericalHarmonicLeafScratch

Scratch for a level: the order-`P` synthesis onto `sampling`'s directions, its parity-folded
twin for the receive side, the spherical frame, and room for `ngroups` leaves.

Collections are built once and read for the map's lifetime. Scratch belongs to a `(level, chunk)`
workspace slot and is rebuilt when the chunk count changes.
"""
function shescratch(sampling::AbstractSphericalSampling, P::Integer; ngroups::Integer=1)
    t = SphericalHarmonics.SphericalHarmonicTransform(ComplexF64, sampling, P)
    return SphericalHarmonicLeafScratch(t, sampling; ngroups=ngroups)
end

"""
    shescratchchannel(orders, leaflevels, plans, samplings, minrelevantlevel; groupsperchunk)

The `(level, chunk)` pool of [`SphericalHarmonicLeafScratch`](@ref), one entry per chunk of every
leaf level a plan actually traverses.

Like `MLFMA.leafscratchchannel`, key entries by the leaf's own level, with no offset.
Skip levels no plan traverses. Leaves shallower than `minrelevantlevel` are never aggregated,
and `H2Factory.nchunks` has no entry for those levels.

`orders` maps each level's sampling to its expansion order, as returned by [`shecollections`](@ref).

`groupsperchunk` should cover the largest leaf count in a chunk at that level.
Below `SHEBATCHTHRESHOLD`, multiplication falls back to one group at a time.
Undersizing preserves correctness but can reduce speed.
"""
function shescratchchannel(
    orders, leaflevels, plans, samplings, minrelevantlevel; groupsperchunk
)
    return _harmonicscratchchannel(
        orders, leaflevels, plans, samplings, minrelevantlevel
    ) do sampling, P, level
        return shescratch(sampling, P; ngroups=groupsperchunk(level))
    end
end

"""
    leavesperchunk(tree, level, nchunks) -> Int

The largest number of leaves one chunk at `level` can hold, which is what
[`shescratchchannel`](@ref) sizes a batch by.

`H2Factory`'s `PlanSchedule` uses contiguous `index_chunks`. The largest chunk has
`cld(nnodes, nchunks)` nodes, so this bounds its leaf count without inspecting a plan.
Oversizing allocates extra scratch; undersizing can trigger the per-group fallback.
"""
function leavesperchunk(tree, level, nchunks)
    nnodes = length(collect(H2Trees.LevelIterator(tree, level)))
    return max(1, cld(nnodes, max(nchunks, 1)))
end
