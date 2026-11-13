# A leaf radiation collection that stores spherical-harmonic coefficients instead of samples.
#
# `TangentialSphericalRadiationFieldCollection` holds two `(NΩ × N_j)` sampled component matrices;
# this holds one `(3(P+1)² × N_j)` coefficient matrix and reconstructs on demand. Measured on real
# RWG patterns: 5.3x-12x less memory depending on box size and tolerance, at a far matvec error
# indistinguishable from the sampled path once `P` is chosen by the calibrated rule
# (`benchmark/interpolation/{shepatterns,sheaccuracy,shecalibration}.jl`).
#
# The coefficients expand three Cartesian components. The tangential frame θ̂/φ̂ is singular at
# the poles, so expanding each tangential component in scalar harmonics can converge poorly.
# Synthesis therefore produces a Cartesian field, then `sphericalharmonics/frame.jl` projects it
# onto the two tangential components that MLFMA stores.
#
# Reconstruction requires scratch, as with `LazyReceiveFieldCollection`. The intermediate must
# not be allocated per matvec or stored per leaf: one `(NΩ × 3)` buffer per leaf would cost 88 MB
# at 105k unknowns, using a fifth of the memory saved by compression. H2Factory supplies scratch
# per `(level, chunk)` to `applyleafmul!`. A bare `LinearAlgebra.mul!` reports an error instead
# of allocating a buffer.
#
# `@linkmomenttocollection` assumes `moment[i] = collection[i] * y` independently for each component.
# Here all three Cartesian components share one synthesis and are mixed by projection, so there
# is no independent matrix for each output component.

"""
    SphericalHarmonicLeafScratch(synthesis, mirror, θ̂, φ̂, coefficients, cartesian, leaves)

The per-`(level, chunk)` working set for a spherical-harmonic leaf multiply.

The fields have two roles:

  - `synthesis`, `mirror`, `θ̂`, `φ̂` depend only on the level's direction grid and order. Leaves
    share them by reference. `synthesis` is `S`; `mirror` is `S` with the `(-1)^l`
    parity folded in (see [`mirrorsynthesis`](@ref MLFMA.Harmonics.SphericalHarmonics.mirrorsynthesis)), which
    is what the Galerkin receive side uses instead of a mirror-index permutation.

  - `coefficients` and `cartesian` are genuine scratch, sized for the largest batch this chunk will
    multiply, one group for the per-node path, the whole chunk for the batched one.

No quadrature weights are stored here. MLFMA's sampled leaf scatter transposes its collection
without weights at the leaf; the sphere's quadrature is applied in translation. A kernel written
as `Sᴴ W` would not match that contract. See this file's receive-side header.
"""
struct SphericalHarmonicLeafScratch{C<:Complex,T<:Real}
    synthesis::Matrix{C}
    mirror::Matrix{C}
    θ̂::Matrix{T}
    φ̂::Matrix{T}
    coefficients::Matrix{C}
    cartesian::Matrix{C}
    # Leaf ids for the current batch let projection tasks index disjoint outputs.
    # Allocate these `maxgroups` `Int`s once, not once per chunk per matvec.
    leaves::Vector{Int}
end

"""
    SphericalHarmonicLeafScratch(t, sampling; ngroups=1)

Build the scratch for a level whose transform is `t` and whose directions are `sampling`, sized to
multiply `ngroups` leaves at once.
"""
function SphericalHarmonicLeafScratch(
    t::SphericalHarmonics.SphericalHarmonicTransform{T,C},
    sampling::AbstractSphericalSampling;
    ngroups::Integer=1,
) where {T,C}
    ngroups >= 1 || throw(ArgumentError("ngroups must be at least 1, got $ngroups"))
    θ̂, φ̂ = SphericalHarmonics.sphericalframe(sampling)
    nd = size(t.synthesis, 1)
    size(θ̂, 1) == nd || throw(
        DimensionMismatch(
            "the transform has $nd directions but the sampling has $(size(θ̂, 1))"
        ),
    )
    return SphericalHarmonicLeafScratch(
        t.synthesis,
        SphericalHarmonics.mirrorsynthesis(t),
        θ̂,
        φ̂,
        Matrix{C}(undef, SphericalHarmonics.numcoefficients(t), 3ngroups),
        Matrix{C}(undef, nd, 3ngroups),
        Vector{Int}(undef, ngroups),
    )
end

maxgroups(s::SphericalHarmonicLeafScratch) = size(s.cartesian, 2) ÷ 3
numdirections(s::SphericalHarmonicLeafScratch) = size(s.cartesian, 1)

"""
    SphericalHarmonicRadiationCollection(storage)

A leaf's radiation patterns as spherical-harmonic coefficients.

`storage` is a [`SphericalHarmonicStorage`](@ref MLFMA.Harmonics.SphericalHarmonics.SphericalHarmonicStorage)
with three (Cartesian) components, whose columns are this leaf's basis functions in the order
`H2Trees.values(tree, leaf)` gives, the same order the sampled collection's columns are in, so the
two are interchangeable from the traversal's point of view.

Declares `D = 2`: the moment it produces is a `TangentialSphericalFarField`. The three Cartesian
components are an internal representation, not what this collection hands back.
"""
struct SphericalHarmonicRadiationCollection{T<:Complex,S} <:
       AbstractRadiationFieldCollection{2,T}
    storage::S

    function SphericalHarmonicRadiationCollection(
        storage::SphericalHarmonics.SphericalHarmonicStorage{C}
    ) where {C<:Complex}
        SphericalHarmonics.ncomponents(storage) == 3 || throw(
            ArgumentError(
                "spherical-harmonic leaf storage must be CARTESIAN (3 components), got " *
                "$(SphericalHarmonics.ncomponents(storage)); see this file's header for why",
            ),
        )
        return new{C,typeof(storage)}(storage)
    end
end

# `fielddim` counts the two output components, not the three internal Cartesian components.
fielddim(::Type{<:SphericalHarmonicRadiationCollection}) = 2

storage(c::SphericalHarmonicRadiationCollection) = c.storage
order(c::SphericalHarmonicRadiationCollection) = SphericalHarmonics.order(c.storage)
numbasisfunctions(c::SphericalHarmonicRadiationCollection) =
    SphericalHarmonics.numbasisfunctions(c.storage)
Base.size(c::SphericalHarmonicRadiationCollection) =
    (SphericalHarmonics.numcoefficients(c.storage) * 3, numbasisfunctions(c))

"""
    SphericalHarmonicAssembledReceiveCollection(radiation)

A leaf's assembled receive patterns as spherical-harmonic coefficients. This is the Petrov test
side under [`SphericalHarmonicLeafStorage`](@ref MLFMA.Assembly.SphericalHarmonicLeafStorage).

Wraps a [`SphericalHarmonicRadiationCollection`](@ref) whose Cartesian coefficients were analysed
from `receiveoperator(op)`'s patterns. The two types store patterns from different sides but use
the same kernels, so this wrapper forwards to the radiation collection.

The separate `<: AbstractReceiveFieldCollection` type identifies the side for map constructors,
scratch pools, and tests.

Unlike [`SphericalHarmonicReceiveCollection`](@ref), the Galerkin wrapper, this collection owns its
receive coefficients. It applies neither antipodal parity nor the operator's `receivecoupling`.
"""
struct SphericalHarmonicAssembledReceiveCollection{T<:Complex,R} <:
       AbstractReceiveFieldCollection{2,T}
    radiation::R
end

SphericalHarmonicAssembledReceiveCollection(
    radiation::SphericalHarmonicRadiationCollection{C}
) where {C<:Complex} =
    SphericalHarmonicAssembledReceiveCollection{C,typeof(radiation)}(radiation)

fielddim(::Type{<:SphericalHarmonicAssembledReceiveCollection}) = 2
radiationcollection(c::SphericalHarmonicAssembledReceiveCollection) = c.radiation
storage(c::SphericalHarmonicAssembledReceiveCollection) = storage(c.radiation)
order(c::SphericalHarmonicAssembledReceiveCollection) = order(c.radiation)
numbasisfunctions(c::SphericalHarmonicAssembledReceiveCollection) =
    numbasisfunctions(c.radiation)
Base.size(c::SphericalHarmonicAssembledReceiveCollection) = size(c.radiation)

"""
    SphericalHarmonicAssembledCollection

Either SHE collection whose coefficients were assembled and analysed,
[`SphericalHarmonicRadiationCollection`](@ref) or
[`SphericalHarmonicAssembledReceiveCollection`](@ref).

Both types use the same scratch, order accessors, and kernels after analysis.
The union excludes [`SphericalHarmonicReceiveCollection`](@ref), which reuses radiation coefficients
and needs different kernels.
"""
const SphericalHarmonicAssembledCollection = Union{
    SphericalHarmonicRadiationCollection,SphericalHarmonicAssembledReceiveCollection
}

# --- the leaf multiply ----------------------------------------------------------------------------

"""
    H2Factory.applyleafmul!(scratch, moment, collection, x)

Aggregate a leaf: `moment .= project(S · (C x))`.

Aggregation has three steps:

 1. `A = C x`, one GEMV against the coefficient matrix, `3(P+1)²` rows rather than `2NΩ`.
    This reads the compressed data and reduces traffic for the memory-bound leaf multiply.
 2. `Cart = S · reshape(A, ncoeff, 3)`, synthesis onto the level's directions. `S` is shared by
    every leaf at the level, so batching this step across a chunk reduces repeated matrix loads.
 3. the frame projection, 3 Cartesian components down to (θ̂, φ̂).
"""
function H2Factory.applyleafmul!(
    scratch::SphericalHarmonicLeafScratch,
    moment::TangentialSphericalFarField,
    collection::SphericalHarmonicAssembledCollection,
    x::AbstractVector,
)
    @views A = scratch.coefficients[:, 1:3]
    SphericalHarmonics.aggregate!(vec(A), storage(collection), x)
    @views Cart = scratch.cartesian[:, 1:3]
    LinearAlgebra.mul!(Cart, scratch.synthesis, A)
    return _projectinto!(moment, Cart, scratch.θ̂, scratch.φ̂)
end

# Write the projection straight into the moment's two component arrays, reading group `g`'s three
# Cartesian columns. `SphericalHarmonics.project!` does the same arithmetic into a `(nd, 2ngroups)`
# matrix, the layout the moment arena stores, but a moment reached through the accessor
# is two separate arrays, so this writes them directly rather than staging and copying.
function _projectinto!(moment, cartesian, θ̂, φ̂, g=1)
    θcomponent, φcomponent = moment[1], moment[2]
    @inbounds for q in axes(cartesian, 1)
        x = cartesian[q, 3g - 2]
        y = cartesian[q, 3g - 1]
        z = cartesian[q, 3g]
        θcomponent[q] = θ̂[q, 1] * x + θ̂[q, 2] * y + θ̂[q, 3] * z
        φcomponent[q] = φ̂[q, 1] * x + φ̂[q, 2] * y + φ̂[q, 3] * z
    end
    return moment
end

# Reject multiplication without scratch, as with `LazyReceiveFieldCollection`.
# Reconstruction needs an intermediate, but allocating it per matvec would lose the memory savings.
function H2Factory.applyleafmul!(
    ::Any,
    ::AbstractFarField,
    collection::SphericalHarmonicAssembledCollection,
    ::AbstractVector,
)
    return error(
        "a $(typeof(collection)) needs a `SphericalHarmonicLeafScratch` to multiply through: " *
        "its reconstruction goes via an (NΩ × 3) intermediate that must be neither allocated " *
        "per matvec nor stored per leaf. The core must implement `H2Factory.buildleafchannels` " *
        "returning that scratch (see `MLFMA.leafscratchchannel` for the sampled analogue).",
    )
end

function LinearAlgebra.mul!(
    ::AbstractFarField, collection::SphericalHarmonicAssembledCollection, ::AbstractVector
)
    return error(
        "a $(typeof(collection)) has no plain `mul!`: it must be applied through " *
        "`H2Factory.applyleafmul!` with a `SphericalHarmonicLeafScratch`. See this file's header.",
    )
end

# --- the batched chunk multiply -------------------------------------------------------------------
#
# Per-node `applyleafmul!` measured 1.63x slower than the sampled multiply because it reloads
# `S`, 2.4 MB at P = 9, for each leaf. One GEMM across a chunk reuses that matrix and measured
# 12x faster than the sampled path at a full uniform level
# (`benchmark/interpolation/shewrite.jl`, strategy B').
#
# The first pass fills the coefficient batch with one GEMV per group. After the GEMM, a second
# pass projects into each leaf's moment. Both passes reuse scratch. `scratch.leaves` retains the
# leaf ids between passes so projection can run in parallel without allocating a fresh list.
#
# Below the batch threshold, aggregation uses the per-node path. Small batches do not amortise
# synthesis well: SHE measured 3.7x slower than sampled at 8 groups and about 10x at 1 group.
# Non-uniform trees can have leaf levels this small; ordinary sphere cases included 1 and 15 leaves.

"""
    SHEBATCHTHRESHOLD

Leaves per batch below which the spherical-harmonic leaf multiply falls back to one group at a time.

The crossing against the sampled leaf multiply was measured between 16 and 32 groups.
At 16 groups, runtime was 1.17x and 1.54x of baseline across two runs; at 32 it was 0.38x and 0.41x.
The threshold enables batching at 16. See `benchmark/interpolation/shewrite.jl`.

This controls whether to batch. [`SHEPROJECTIONTHREADTHRESHOLD`](@ref) separately controls whether
projection in a batch flush may run in parallel. A measured 2.8x jump across 15 -> 16 leaves comes
from enabling batching, not from the projection threshold.
"""
const SHEBATCHTHRESHOLD = 16

"""
    SHEPROJECTIONTHREADTHRESHOLD

Groups in a single flush below which the projection runs serially regardless of the scheduler
`H2Factory.LeafMulOp` resolves to.

This differs from `SHEBATCHTHRESHOLD`: a chunk larger than scratch capacity flushes at `maxgroups`
and again for any remainder. That final flush can contain a single group, too little work to
justify spawning projection tasks.

The two thresholds currently match because they use the same benchmark sweep, but they can be
tuned independently.
"""
const SHEPROJECTIONTHREADTHRESHOLD = SHEBATCHTHRESHOLD

function H2Factory.aggregateleafchunk!(
    A,
    tree,
    chunknodes,
    moment,
    momentcollection,
    leafmul!,
    x,
    level,
    scratch::SphericalHarmonicLeafScratch,
)
    return _aggregateharmonicchunk!(
        A,
        tree,
        chunknodes,
        moment,
        momentcollection,
        leafmul!,
        x,
        level,
        scratch,
        _gathercoefficients!,
        _flushbatch!,
    )
end

# SHE and VSH share the traversal; gathering and synthesis keep their own kernels.
function _aggregateharmonicchunk!(
    A,
    tree,
    chunknodes,
    moment,
    momentcollection,
    leafmul!,
    x,
    level,
    scratch,
    gather!::G,
    flush!::F,
) where {G,F}
    nleaves = count(node -> H2Trees.isleaf(tree, node), chunknodes)
    if nleaves < SHEBATCHTHRESHOLD ||
        !_batchable(tree, chunknodes, momentcollection, leafmul!)
        for node in chunknodes
            H2Trees.isleaf(tree, node) || continue
            H2Factory.aggregateleaf!(
                A, tree, node, moment, momentcollection, leafmul!, x, scratch
            )
        end
        return nothing
    end

    # Resolve the scheduler once; it is unchanged across this chunk's flushes.
    scheduler = H2Factory.innerscheduler(A, H2Factory.LeafMulOp(), level)

    capacity = maxgroups(scratch)
    filled = 0
    for node in chunknodes
        H2Trees.isleaf(tree, node) || continue
        filled += 1
        scratch.leaves[filled] = node
        gather!(A, tree, node, momentcollection, x, scratch, filled)
        # Sub-batch when the chunk outruns the scratch. `S` is streamed once per sub-batch, so this
        # costs a re-stream per `capacity` groups rather than per group, still the batched regime
        # as long as `capacity` is comfortably above the threshold.
        if filled == capacity
            flush!(moment, scratch, filled, scheduler)
            filled = 0
        end
    end
    filled > 0 && flush!(moment, scratch, filled, scheduler)
    return nothing
end

"""
    projectionscheduler(scheduler, filled)

`scheduler`, or `SerialScheduler()` when the flush is too small to be worth spawning for.

The scheduler selects the map's policy for nested parallelism. The threshold determines whether
this flush is large enough to benefit. A tail of one group runs serially regardless of that policy.
See [`SHEPROJECTIONTHREADTHRESHOLD`](@ref).
"""
function projectionscheduler(scheduler, filled::Int)
    filled >= SHEPROJECTIONTHREADTHRESHOLD && return scheduler
    return OhMyThreads.SerialScheduler()
end

# The batch implements unconjugated `P S C x` against an assembled collection. H2Factory swaps sides for
# transpose and adjoint traversals. A Galerkin receive collection needs `M` in place of `S` and
# its operator's coupling; the adjoint also uses a `ConjugatedCollectionMul`.
# Those paths must use their own kernels rather than this batch.
#
# The batch does not otherwise call `leafmul!`, so this guard is its only check of the multiply mode.
#
# Reverse orientations use the per-node fallback. Assembled Petrov receive collections can use
# the plain synthesis batch when the multiply is unconjugated.
_batchable(tree, chunknodes, momentcollection, leafmul!) = false

function _batchable(tree, chunknodes, momentcollection, ::H2Factory.LinAlgebraMulFunctor)
    for node in chunknodes
        H2Trees.isleaf(tree, node) || continue
        return _batchableradiation(momentcollection(node))
    end
    return false
end

# A predicate rather than an `isa` against a fixed type, so a second compressed backend opts in from
# its own file instead of this one growing a `Union` it has to be kept in step with.
_batchableradiation(::Any) = false
_batchableradiation(::SphericalHarmonicAssembledCollection) = true

function _gathercoefficients!(A, tree, node, momentcollection, x, scratch, g)
    collection = momentcollection(node)
    # Through the accessor, not the field: an assembled receive collection batches through this same
    # path and holds its storage one level down.
    store = storage(collection)
    SphericalHarmonics.numcoefficients(store) == size(scratch.coefficients, 1) || throw(
        DimensionMismatch(
            "leaf $node expands to order $(order(collection)), but this level's scratch is " *
            "sized for $(size(scratch.coefficients, 1)) coefficients. One order per LEVEL is " *
            "required, a ragged batch has no single synthesis matrix.",
        ),
    )
    buffer = H2Factory.aggregationvectorbuffer(A, node)
    buffer .= view(x, H2Trees.values(tree, node))
    @views SphericalHarmonics.aggregate!(
        vec(scratch.coefficients[:, (3g - 2):(3g)]), store, buffer
    )
    return nothing
end

# One GEMM synthesises the sub-batch, then projection writes the moments. The GEMM uses BLAS
# threads. Parallel projection measured a 4.0x speedup with eight threads at 35 groups,
# but added overhead at one group.
#
# Projection uses the scheduler resolved for `H2Factory.LeafMulOp`, so it respects the map's
# execution policy. Measurements against `Threads.@threads` show the tradeoff:
#
#   * A `SerialScheduler` compiles to the plain loop, 177.1 us against 176.8 us hand-written, and
#     allocation-free, where `Threads.@threads` cost 1632 bytes per flush even at one thread.
#   * At eight threads OhMyThreads takes 51.3 us against `Threads.@threads`' 44.7 us,
#     about 15% overhead for the configurable scheduler.
#
# MLFMA defaults this kernel to parallel execution; see `executionschedulingfor`.
# H2Factory's serial inner default would lose the measured 4.0x speedup.
function _flushbatch!(moment, scratch, filled, scheduler)
    @views LinearAlgebra.mul!(
        scratch.cartesian[:, 1:(3filled)],
        scratch.synthesis,
        scratch.coefficients[:, 1:(3filled)],
    )
    OhMyThreads.tforeach(1:filled; scheduler=projectionscheduler(scheduler, filled)) do g
        return _projectinto!(
            moment(scratch.leaves[g]), scratch.cartesian, scratch.θ̂, scratch.φ̂, g
        )
    end
    return nothing
end

# --- the receive side -----------------------------------------------------------------------------
#
# The Galerkin receive collection reuses radiation coefficients. The sampled geometric mirror uses
# `MLFMA.mirrorindices` to permute NΩ samples under k̂ -> -k̂, then applies the ±1 component table
# `mirrorrule`. It needs a scratch far field for BLAS. In coefficient space, the mirror is diagonal:
#
#     Y_lm(-k̂) = (-1)^l Y_lm(k̂)
#
# The sign for each degree is precomputed into `scratch.mirror` once per level. Projection supplies
# the component signs: θ̂ is even under the antipodal map and φ̂ is odd, as in
# `mirrorrule(TangentialSphericalFarField)`. `test/SphericalHarmonics/frame.jl` checks both identities.
# The receive wrapper stores a radiation reference and a component coupling, not separate patterns
# or sample indices. Mutable work buffers belong to the `(level, chunk)` scratch.
#
# Write the radiation collection as `R = P S C`: coefficients, synthesis, then frame projection.
# Let `Π` denote the sampled geometric mirror, including its component signs:
#
#     Π P S = P M,        M = S·diag((-1)^l) = `scratch.mirror`
#
# Since `θ̂` is even and `φ̂` is odd, `Y_lm(-k̂) = (-1)^l Y_lm(k̂)` gives the tangential mirror rule.
# When the operator's coupling is the identity, the Galerkin receive collection is
#
#     Rec = Π R = P M C
#
# In that case, substituting `M` for `S` is sufficient. Other operators also need the component map
# derived below. Neither case needs a separate set of receive coefficients.
#
# `Rec` is a moment collection, not a weighted testing functional. MLFMA's sampled leaf scatter is
# `y = α·transpose(L)·Π m` (see
# `LazyReceiveFieldCollection`'s read side, which is `receivefields!` followed by a plain collection
# `mul!`), with the sphere's quadrature weight living in the translation operator, not at the leaf.
# Applying `Sᴴ W` would satisfy the weighted pairing ⟨F,G⟩_W but would not match the sampled path.
# End-to-end measurements of that weighted form gave 6.9e-2 error against a dense reference.
#
# Scatter uses a transpose, matching the sampled collection. `ConjugatedCollectionMul` supplies
# the conjugation needed in the adjoint orientation. See `assembly/cores/adjointleafmul.jl`.

# --- the operator's receive coupling ---------------------------------------------------------------
#
# `Pi_mirror . (P S) = P M` describes the geometric mirror: the antipodal
# sample permutation composed with `mirrorrule`, which is `theta-hat` even and `phi-hat` odd. But the
# sampled receive side does not apply `mirrorrule`; it applies `receivecouplingrule(op, F)`, which is
# operator-dependent. For `MWEFIE3D` the two coincide:
#
#     mirrorrule(TangentialSphericalFarField)          == (1->1: +1, 2->2: -1)
#     receivecouplingrule(MWEFIE3D, Tangential...)     == (1->1: +1, 2->2: -1)  , the same
#     receivecouplingrule(MWDoubleLayer3D, Tangential) == (1->2:  g, 2->1:  g)  , a SWAP
#
# Treating these rules as interchangeable measured 4.4e-01 relative error on an MFIE map against
# a dense reference, compared with 2.8e-05 for the sampled path. Array shapes do not detect this error.
#
# Write `Perm` for the pure permutation and `R_op` for the operator's component
# map, `Pi_mirror = Diag(1,-1) . Perm` gives `Perm . (P S) = Diag(1,-1) . (P M)` and hence
#
#     Pi_op . (P S) = R_op . Diag(1,-1) . (P M) = K . (P M),      K := R_op . mirrorrule
#
# The receive collection applies mirrored synthesis followed by a 2x2 component map, shared across
# directions and leaves. `K = I` for EFIE, so it uses the plain mirrored kernel.
# For MFIE the map is `[[0,-g],[g,0]]`.

"""
    receivecoupling(op) -> SMatrix{2,2}

The 2x2 component map a compressed receive collection applies after its mirrored synthesis:
`receivecouplingrule(op, TangentialSphericalFarField)` composed with `mirrorrule`, both as matrices.

The map is derived from both rule tables. A rule that addresses components outside the two-component
tangential layout raises an error during construction.
"""
function receivecoupling(op)
    return _rulematrix(receivecouplingrule(op, TangentialSphericalFarField), op) *
           _rulematrix(mirrorrule(TangentialSphericalFarField), op)
end

function _rulematrix(rule, op)
    K = zeros(MMatrix{2,2,ComplexF64})
    for mapping in rule
        d, s = destcomponent(mapping), srccomponent(mapping)
        (d in 1:2 && s in 1:2) || error(
            "compressed leaf storage supports two-component tangential far fields only, but " *
            "$(typeof(op))'s receive rule maps component $s to $d. Use `SampledLeafStorage()` " *
            "for this operator, or extend the compressed receive kernels to its representation.",
        )
        K[d, s] += coefficient(mapping)
    end
    return SMatrix(K)
end

# `K == I` is the EFIE case. Skip component mixing on this common path.
_isplaincoupling(K) = K == LinearAlgebra.I

"""
    SphericalHarmonicReceiveCollection(radiation, coupling)

The Galerkin receive collection for a leaf whose radiation patterns are stored as spherical-harmonic
coefficients.

Holds a reference to the radiation collection and the operator's coupling map. The antipodal mirror is a
per-degree sign folded into the level's `scratch.mirror`, and the tangential component signs fall out
of the frame projection, so unlike the sampled
[`LazyReceiveFieldCollection`](@ref MLFMA.FarFields.LazyReceiveFieldCollection), there is no index array and no mirror pass. See this file's
receive-side header.
"""
struct SphericalHarmonicReceiveCollection{T<:Complex,R} <:
       AbstractReceiveFieldCollection{2,T}
    radiation::R
    # The operator's receive coupling, reduced to one 2x2 component map, see
    # `receivecoupling` for the derivation.
    coupling::SMatrix{2,2,T,4}

    function SphericalHarmonicReceiveCollection(
        radiation::SphericalHarmonicRadiationCollection{C}, coupling::AbstractMatrix
    ) where {C<:Complex}
        size(coupling) == (2, 2) || throw(
            DimensionMismatch(
                "the receive coupling is a 2x2 map on the tangential components, got " *
                "$(size(coupling))",
            ),
        )
        return new{C,typeof(radiation)}(radiation, SMatrix{2,2,C}(coupling))
    end
end

fielddim(::Type{<:SphericalHarmonicReceiveCollection}) = 2
radiationcollection(c::SphericalHarmonicReceiveCollection) = c.radiation
coupling(c::SphericalHarmonicReceiveCollection) = c.coupling
storage(c::SphericalHarmonicReceiveCollection) = storage(c.radiation)
order(c::SphericalHarmonicReceiveCollection) = order(c.radiation)
numbasisfunctions(c::SphericalHarmonicReceiveCollection) = numbasisfunctions(c.radiation)

"""
    H2Factory.applyleafmul!(scratch, y, collection, moment, α, β)

Scatter an incoming moment onto a leaf's degrees of freedom: `y .= α * Cᵀ Mᵀ Pᵀ moment + β * y`.

The transpose of [`SphericalHarmonicRadiationCollection`](@ref)'s aggregation with `M` for `S`, which
is what `Rec = Π R` means when the operator's coupling is the identity. The displayed formula
assumes that case; the implementation also applies the transpose of a nonidentity coupling.
See this file's receive-side header.

`α`/`β` follow `LinearAlgebra.mul!`'s convention, and `β` arrives as `false` from the traversal's
overwriting leaf scatter, which is why the accumulation below short-circuits on it rather than
multiplying by zero.
"""
function H2Factory.applyleafmul!(
    scratch::SphericalHarmonicLeafScratch,
    y::AbstractVector,
    collection::SphericalHarmonicReceiveCollection,
    moment::TangentialSphericalFarField,
    α::Number,
    β::Number,
)
    return _shescatter!(
        y,
        scratch,
        storage(collection),
        moment,
        scratch.mirror,
        α,
        β,
        _Transposed(),
        coupling(collection),
    )
end

function H2Factory.applyleafmul!(
    ::Any,
    ::AbstractVector,
    collection::SphericalHarmonicReceiveCollection,
    ::AbstractFarField,
    ::Number,
    ::Number,
)
    return error(
        "a $(typeof(collection)) needs a `SphericalHarmonicLeafScratch` to multiply through, " *
        "for the same reason the radiation collection does. See this file's header.",
    )
end

# --- the reverse orientations ---------------------------------------------------------------------
#
# H2Factory swaps sides for transpose and adjoint traversals. Each collection therefore needs both
# aggregation and scatter methods. These methods are explicit because the collections are not matrices.
#
# The six cases share two kernels. With `R = P S C` and `Rec = P M C` (identity coupling), and
# `Y` standing for whichever of `S`/`M` the side owns:
#
#                    aggregate                       scatter
#     forward        R x        = P S C x            α Recᵀ m = α Cᵀ Mᵀ Pᵀ m
#     transpose      Rec x      = P M C x            α Rᵀ m   = α Cᵀ Sᵀ Pᵀ m
#     adjoint        conj(Rec) x                     α Rᴴ m   = α Cᴴ Sᴴ Pᵀ m
#
# so `_sheaggregate!(moment, scratch, storage, x, Y)` and
# `_shescatter!(y, scratch, storage, moment, Y, α, β, mode)` cover these cases. The orientation picks `Y`,
# and the scatter's `mode` picks transpose or adjoint. `P` is real, so it never carries a conjugation
# and `Pᵀ` appears unchanged in all three scatters.
#
# The adjoint's `α` arrives already conjugated from `H2Factory.α(core, ::isAdjoint)`.
# The conjugated kernels live in `assembly/cores/adjointleafmul.jl`.

# Transpose aggregation, test side: `moment = Rec x = P M C x` before component coupling.
function H2Factory.applyleafmul!(
    scratch::SphericalHarmonicLeafScratch,
    moment::TangentialSphericalFarField,
    collection::SphericalHarmonicReceiveCollection,
    x::AbstractVector,
)
    _sheaggregate!(moment, scratch, storage(collection), x, scratch.mirror)
    return _applycoupling!(moment, coupling(collection))
end

"""
    _applycoupling!(moment, K) -> moment

Mix a moment's two tangential components in place by `K`, the operator's receive coupling.

The aggregation-side counterpart of the fold in `_coupledbackprojectinto!`. In place and with no
temporary: the two components are read into scalars before either is written, so a swap
(`MWDoubleLayer3D`) is safe.
"""
function _applycoupling!(moment, K)
    _isplaincoupling(K) && return moment
    θcomponent = H2Factory.wrapvector(moment, 1)
    φcomponent = H2Factory.wrapvector(moment, 2)
    @inbounds for q in eachindex(θcomponent)
        a, b = θcomponent[q], φcomponent[q]
        θcomponent[q] = K[1, 1] * a + K[1, 2] * b
        φcomponent[q] = K[2, 1] * a + K[2, 2] * b
    end
    return moment
end

# Transpose scatter, trial side: `y = α Rᵀ moment + β y`.
function H2Factory.applyleafmul!(
    scratch::SphericalHarmonicLeafScratch,
    y::AbstractVector,
    collection::SphericalHarmonicAssembledCollection,
    moment::TangentialSphericalFarField,
    α::Number,
    β::Number,
)
    return _shescatter!(
        y, scratch, storage(collection), moment, scratch.synthesis, α, β, _Transposed()
    )
end

function H2Factory.applyleafmul!(
    ::Any,
    ::AbstractVector,
    collection::SphericalHarmonicAssembledCollection,
    ::AbstractFarField,
    ::Number,
    ::Number,
)
    return error(
        "a $(typeof(collection)) needs a `SphericalHarmonicLeafScratch` to scatter through, " *
        "see this file's header. This arity is the leaf of a TRANSPOSED matvec, where the trial " *
        "side is the one being scattered onto.",
    )
end

# --- the two shared kernels -----------------------------------------------------------------------

# Select `transpose` or `adjoint` by type so the wrapper resolves at compile time.
# A ternary over `transpose(Y)`/`Y'` would return a small `Union` at GEMM dispatch.
struct _Transposed end
struct _Adjointed end

_reverse(::_Transposed, A) = transpose(A)
_reverse(::_Adjointed, A) = adjoint(A)

"""
    _sheaggregate!(moment, scratch, storage, x, Y) -> moment

`moment = P Y C x`, the leaf aggregation for whichever side owns synthesis matrix `Y`.

`Y` is `scratch.synthesis` on the radiation side and `scratch.mirror` on the Galerkin receive side.
The receive caller applies the operator's component coupling after this kernel.
"""
function _sheaggregate!(moment, scratch, storage, x, Y)
    @views A = scratch.coefficients[:, 1:3]
    SphericalHarmonics.aggregate!(vec(A), storage, x)
    @views Cart = scratch.cartesian[:, 1:3]
    LinearAlgebra.mul!(Cart, Y, A)
    return _projectinto!(moment, Cart, scratch.θ̂, scratch.φ̂)
end

"""
    _shescatter!(y, scratch, storage, moment, Y, α, β, mode) -> y

`y = α (P Y C)ᵀ moment + β y`, or the adjoint of the same when `mode` is `_Adjointed()`.

Three stages mirroring [`_sheaggregate!`](@ref) in reverse: the frame back-projection `Pᵀ`, the
reversed synthesis, and the reversed coefficient multiply.
"""
function _shescatter!(y, scratch, storage, moment, Y, α, β, mode, K=LinearAlgebra.I)
    @views Cart = scratch.cartesian[:, 1:3]
    _backprojectinto!(Cart, moment, scratch.θ̂, scratch.φ̂, K)
    @views A = scratch.coefficients[:, 1:3]
    LinearAlgebra.mul!(A, _reverse(mode, Y), Cart)
    coefficients = SphericalHarmonics.coefficientmatrix(storage)
    LinearAlgebra.mul!(y, _reverse(mode, coefficients), vec(A), α, iszero(β) ? false : β)
    return y
end

# `Pᵀ`, the 2 -> 3 back-projection `θ̂ G_θ + φ̂ G_φ`, using the same frame arrays the forward
# projection reads.
# `Pᵀ` with the operator's coupling transposed into it: `Cart = Pᵀ Kᵀ moment`. Folded here rather
# than staged into a scratch pair because `cartesian` has no free `(NΩ × 2)` block to stage into, and
# the fold is four extra multiplies per direction against a GEMM that follows.
_backprojectinto!(cartesian, moment, θ̂, φ̂, K) =
    if _isplaincoupling(K)
        _backprojectinto!(cartesian, moment, θ̂, φ̂)
    else
        _coupledbackprojectinto!(cartesian, moment, θ̂, φ̂, K)
    end

function _coupledbackprojectinto!(cartesian, moment, θ̂, φ̂, K)
    θcomponent, φcomponent = moment[1], moment[2]
    @inbounds for q in axes(cartesian, 1)
        a, b = θcomponent[q], φcomponent[q]
        mθ = K[1, 1] * a + K[2, 1] * b
        mφ = K[1, 2] * a + K[2, 2] * b
        cartesian[q, 1] = θ̂[q, 1] * mθ + φ̂[q, 1] * mφ
        cartesian[q, 2] = θ̂[q, 2] * mθ + φ̂[q, 2] * mφ
        cartesian[q, 3] = θ̂[q, 3] * mθ + φ̂[q, 3] * mφ
    end
    return cartesian
end

function _backprojectinto!(cartesian, moment, θ̂, φ̂)
    θcomponent, φcomponent = moment[1], moment[2]
    @inbounds for q in axes(cartesian, 1)
        mθ = θcomponent[q]
        mφ = φcomponent[q]
        cartesian[q, 1] = θ̂[q, 1] * mθ + φ̂[q, 1] * mφ
        cartesian[q, 2] = θ̂[q, 2] * mθ + φ̂[q, 2] * mφ
        cartesian[q, 3] = θ̂[q, 3] * mθ + φ̂[q, 3] * mφ
    end
    return cartesian
end

# `A = C conj(x)`, through the real/imaginary split rather than a conjugated temporary, the leaf
# multiply runs once per leaf per matvec and must not allocate. `reinterpret(reshape, ...)` views the
# same memory as a `2 x n` real matrix, so the two `mul!`s read `x` in place. Identical in shape to
# H2Factory's `conjmul!`, which solves the same problem for the sampled collections, and used for the
# same reason: there is no conjugate-without-transpose BLAS call.
#
# Only adjoint aggregation needs it, via `conj(C) x == conj(C conj(x))`.
function _aggregateconjugated!(A, coefficients::AbstractMatrix, x::DenseVector{<:Complex})
    parts = reinterpret(reshape, real(eltype(x)), x)
    LinearAlgebra.mul!(A, coefficients, view(parts, 1, :))
    LinearAlgebra.mul!(
        A, coefficients, view(parts, 2, :), -im * one(eltype(A)), one(eltype(A))
    )
    return A
end

# A real `x` needs no conjugation at all, and a non-dense one cannot be reinterpreted, the traversal
# always passes H2Factory's dense per-node aggregation buffer, so the last method serves direct
# callers (tests) only and its one temporary is not on any hot path.
_aggregateconjugated!(A, coefficients::AbstractMatrix, x::AbstractVector{<:Real}) =
    LinearAlgebra.mul!(A, coefficients, x)
_aggregateconjugated!(A, coefficients::AbstractMatrix, x::AbstractVector) =
    LinearAlgebra.mul!(A, coefficients, conj.(x))
