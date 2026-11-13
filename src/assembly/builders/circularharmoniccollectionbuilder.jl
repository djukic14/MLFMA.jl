# Assemble on the level's circular grid, analyse, and store truncated harmonic coefficients.
#
# Unlike the spherical builder, this builder uses the leaf's own grid for analysis.
# `shecollections` uses a separate Gauss-Legendre grid because `MLFMA.sphericalsamplepoints`
# includes `sin θ` in its weights and needs about 4x as many θ nodes for accurate analysis.
# Substituting `x = cos θ` avoids that cost.
#
# The circle has no Jacobian factor. Its trapezoidal rule is exact for resolved modes.
# In `benchmark/interpolation/checalibration.jl`, oversampling factors 1, 2, 4, 8 and 16
# gave group-moment errors identical to six digits, so no separate analysis grid is used.
#
# Real basis patterns are not band-limited: content above `Nφ/2` aliases into retained modes.
# In the calibration fixtures, the spectrum fell to ~1e-14 by |m| = 16 with a Nyquist bin at 14.
# This is a measured property of those fixtures, not a general bound. Callers can use
# `circulargrid`'s `oversampling` option to check their geometry.
#
# `leveltruncationorder` takes the maximum order over a level's boxes.
# All leaves at a level need one order because a batched multiply uses one synthesis matrix.
#
# A 2-D far field has no spherical frame or pole singularity, so no Cartesian substitution is needed.
# Use the operator's store rule and far-field type for scalar and three-component hypersingular fields.
# `_grouptheleavesbylevel` groups by sampling identity; each level shares one sampling object.

"""
    checollections(assembler, tree, leafsamplings, ε) -> (collections, orders)

Build one [`CircularHarmonicRadiationCollection`](@ref) per leaf, and report the order chosen for each
level.

`collections` is a concretely typed `Dictionary` keyed by leaf id, as in the sampled path.
`H2Factory.buildleafchannels` identifies compressed storage by its element type.
`orders` is keyed by each level's sampling object.

`assembler` uses the operator's store rule and far-field type, with no Cartesian substitution.

`ε` is the map's tolerance. `leveltruncationorder` uses `max(ε/10, CHETAILFLOOR)`, rather than
the spherical rule's hundredth. See `CHETAILRATIO` for the 20-row calibration and the reason they differ.
"""
function checollections(assembler, tree, leafsamplings, ε::Real)
    leaves = leafids(leafsamplings)
    isempty(leaves) &&
        throw(ArgumentError("no leaves to build circular-harmonic storage for"))
    collections = nothing
    orders = IdDict{Any,Int}()

    for (sampling, indices) in _grouptheleavesbylevel(leafsamplings)
        sampling isa CircularSampling || error(
            "circular-harmonic leaf storage expands a pattern on a `CircularSampling`, but this " *
            "level's grid is a $(typeof(sampling)). Pass " *
            "`samplingbuilder=MLFMA.CircularSamplingBuilder{Float64}()` in the SamplingStrategy, " *
            "or use `SphericalHarmonicLeafStorage()` for a 3-D map.",
        )
        # Analyse at the level's full grid order. Resolved modes are exact; unresolved content can alias.
        M₀ = cutoffparameter(sampling)
        analysis = CircularHarmonics.CircularHarmonicTransform(ComplexF64, sampling, M₀)

        points, _ = Sampling.cartesiansamplepoints(sampling)
        boxes = [
            Assembly.BoxFarFieldSampling(
                sampling, points, H2Trees.center(tree, leaves[i]), leaves[i]
            ) for i in indices
        ]
        Fs = assembler(tree, boxes; verbose=false)

        # Stack component-major columns into `(nc × D·nbf)` for analysis.
        analysed = map(Fs) do F
            D = length(F)
            nbf = size(F[1], 2)
            stacked = Matrix{ComplexF64}(undef, size(F[1], 1), D * nbf)
            for component in 1:D
                @views stacked[:, ((component - 1) * nbf + 1):(component * nbf)] .= F[component]
            end
            return CircularHarmonics.analyze(analysis, stacked)
        end

        M = CircularHarmonics.leveltruncationorder(analysed, ε)
        # `M == M₀` retains every mode the level's grid represents, so it needs no clamp check.
        # This loses no resolved modes but does not rule out aliasing in the sampled pattern.
        nc = CircularHarmonics.numcoefficients(M)
        orders[sampling] = M

        for (k, i) in enumerate(indices)
            leaf = leaves[i]
            D = length(Fs[k])
            nbf = size(Fs[k][1], 2)
            collection = CircularHarmonicRadiationCollection(
                CircularHarmonics.CircularHarmonicStorage(
                    M,
                    SVector{2}(H2Trees.center(tree, leaf)),
                    _repackcircularcoefficients(analysed[k], nbf, nc, M₀, M, D),
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

# `analyze` returns `(2M₀+1 × D·nbf)` with component-major columns (`(c-1)nbf + j`).
# `CircularHarmonicStorage` wants `((2M+1)·D × nbf)` with column `j` holding basis function `j`'s
# `(nc × D)` block flattened m-fastest. Repack and truncate in one pass.
#
# Truncation uses a centred slice: `m` runs `-M₀:M₀`, so order `≤ M` occupies
# rows `(M₀-M+1):(M₀+M+1)`. The first `2M+1` rows would keep the wrong band, `-M₀ … -M₀+2M`.
# In contrast, spherical indexing `l² + l + m + 1` puts low degrees in the leading block.
function _repackcircularcoefficients(analysed, nbf, nc, M₀, M, D)
    offset = M₀ - M
    repacked = Matrix{eltype(analysed)}(undef, nc * D, nbf)
    @inbounds for j in 1:nbf, component in 1:D
        source = (component - 1) * nbf + j
        destination = (component - 1) * nc
        for q in 1:nc
            repacked[destination + q, j] = analysed[offset + q, source]
        end
    end
    return repacked
end

"""
    cheorders(tree, collections, samplings, minrelevantlevel) -> IdDict

Read each level's expansion order from its collections, keyed by the level's sampling object.
This recovers the second return value of [`checollections`](@ref).

Collections belong to the core; `H2Factory.buildleafchannels` sizes synthesis matrices per workspace.
Reading orders from the collections keeps scratch consistent with stored coefficients.
A smaller scratch order can silently give wrong results even when the GEMM shapes conform.

Raise if leaves at one level have different orders. A chunk uses one synthesis matrix.
"""
function cheorders(tree, collections, samplings, minrelevantlevel)
    return _harmonicorders(
        tree, collections, samplings, minrelevantlevel, "circular-harmonic"
    )
end

# Construction policy.

"""
    CircularHarmonicFieldCollectionBuilder(side, assemblerbuilder, quadstrategy, ε)

Construction policy for circular-harmonic leaf storage: assemble on the level's own grid, analyse, and
keep the coefficients.

`ε` is the map's tolerance, used by [`checollections`](@ref) to choose the order.

`side` is [`RadiationSide`](@ref) or [`ReceiveSide`](@ref). It selects the assembled operator
through `fieldoperator`. Both sides use the same analysis, order rule, and storage.

The lazy receive side derives patterns from radiation collections through
[`LazyReceiveFieldCollectionBuilder`](@ref), as in the sampled path.

Normally obtained from a strategy carrying [`CircularHarmonicLeafStorage`](@ref MLFMA.Assembly.CircularHarmonicLeafStorage) rather than built by
hand -- see [`radiationfieldcollectionbuilder`](@ref) and [`receivefieldcollectionbuilder`](@ref).
"""
struct CircularHarmonicFieldCollectionBuilder{
    Side<:AbstractFieldCollectionSide,A,Q,E<:Real
} <: AbstractCompressedFieldCollectionBuilder{Side}
    side::Side
    assemblerbuilder::A
    quadstrategy::Q
    ε::E
end

# Use the default `buildfarfieldassembler` from `AbstractAssembledFieldCollectionBuilder`.
# Circular harmonics need no Cartesian substitution for the operator's field representation.

#TODO: what is this docstring doing here?
"""
    buildfieldcollections(builder::CircularHarmonicFieldCollectionBuilder{ReceiveSide}, op, ...)

The ASSEMBLED (Petrov) test side, compressed: `checollections` again, then one wrapper per leaf.

It really is that short, and the reason is worth stating rather than leaving to look like an
oversight. `checollections` compresses whatever the assembler it is handed produces, and
`buildfarfieldassembler` builds this one from `fieldoperator(ReceiveSide(), op)` -- i.e.
`receiveoperator(op)`, which already carries the receive sign of γ. So the coefficients that come back
ARE the receive patterns; there is nothing left to correct.

WHAT AN ASSEMBLED RECEIVE SIDE DOES NOT NEED, and it is everything that makes the Galerkin one
non-trivial. [`CircularHarmonicReceiveCollection`](@ref) carries a reference to the RADIATION
collection, the antipodal parity (as `scratch.mirror`) and the operator's `D x D` receive coupling,
because it reconstructs a pattern it does not store. This side stores its own, so the mirror is not
applied and the coupling is not applied -- and consequently the #129 seam, which is where the
Galerkin path can be silently wrong, does not exist here at all. That in turn is why the 2-D DOUBLE
LAYER can use this: it is Petrov-only (#130) precisely because it has no mirrored receive rule, and
this side never asks for one.

Which is also why the wrapper is a distinct TYPE rather than reusing the radiation collection whose
kernels it borrows: the collections must answer `AbstractReceiveFieldCollection` to say which side
they are, and a type that lies about that would make every downstream side check vacuous.
"""

# These hooks select analysis and receive wrappers for the shared compressed builder protocol.
function analysecollections(
    builder::CircularHarmonicFieldCollectionBuilder, assembler, tree, leafsamplings
)
    return checollections(assembler, tree, leafsamplings, builder.ε)
end

assembledreceivecollectiontype(::CircularHarmonicFieldCollectionBuilder) =
    CircularHarmonicAssembledReceiveCollection

# The radiation collection type selects the lazy Galerkin receive storage.
# `ψ_m(-k̂) = (-1)^m ψ_m(k̂)` is folded into `scratch.mirror`.
# Receive collections reference radiation storage and keep the operator's coupling.
# They own no coefficient data or mirror indices.
function buildfieldcollections(
    ::LazyReceiveFieldCollectionBuilder,
    op,
    tree,
    leafsamplings,
    radiationcollections::AbstractDictionary{<:Any,<:CircularHarmonicRadiationCollection};
    verbose::Bool=false,
)
    leaves = leafids(leafsamplings)
    K = circularreceivecoupling(op, farfieldtype(op))
    return [
        CircularHarmonicReceiveCollection(radiationcollections[leaf], K) for leaf in leaves
    ]
end

"""
    chescratch(sampling, M; ncomponents, ngroups=1) -> CircularHarmonicLeafScratch

The leaf scratch for a level: the order-`M` synthesis onto `sampling`'s directions, its parity-folded
twin for the receive side, and room for `ngroups` leaves of `ncomponents` components each.

Collections are built once and read for the map's lifetime. Scratch belongs to a `(level, chunk)`
workspace slot and is rebuilt when the chunk count changes.
"""
function chescratch(
    sampling::CircularSampling, M::Integer; ncomponents::Integer, ngroups::Integer=1
)
    t = CircularHarmonics.CircularHarmonicTransform(ComplexF64, sampling, M)
    return CircularHarmonicLeafScratch(t, ncomponents; ngroups=ngroups)
end

"""
    chescratchchannel(orders, leaflevels, plans, samplings, minrelevantlevel;
                      ncomponents, groupsperchunk)

The `(level, chunk)` pool of [`CircularHarmonicLeafScratch`](@ref), one entry per chunk of every leaf
level a plan actually traverses.

Like [`shescratchchannel`](@ref), key entries by the leaf's own level, with no offset.
Skip levels no plan traverses.

`groupsperchunk` sizes the batch. Undersizing preserves correctness but causes more frequent flushes.
The 2-D batch has no minimum threshold: it ties at one leaf and is faster from two leaves upward.
"""
function chescratchchannel(
    orders, leaflevels, plans, samplings, minrelevantlevel; ncomponents, groupsperchunk
)
    return _harmonicscratchchannel(
        orders, leaflevels, plans, samplings, minrelevantlevel
    ) do sampling, P, level
        return chescratch(
            sampling, P; ncomponents=ncomponents, ngroups=groupsperchunk(level)
        )
    end
end
