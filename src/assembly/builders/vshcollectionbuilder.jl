# Build tangential vector-harmonic leaf collections from an assembler.
#
# The grid and analysis-order checks match `shecollectionbuilder.jl`.
# `L` gives a scale for the order, not an upper bound; reaching the analysis order raises an error.
#
# The tangential store rule needs a `SphericalFrameCache` built from a sampling object.
# The Gauss-Legendre grid is an explicit direction list, so setup uses `StoreAsFarFieldValue` to
# assemble Cartesian components. It projects them onto the grid's frame once per leaf before analysis.
# The matvec needs no Cartesian-to-tangential projection.
#
# `vshtruncationorder` uses Parseval over both coefficient families, with no extra normalization.
# As in the scalar case, τ must be stricter than `ε` because leaf errors accumulate across a level.
# A tolerance floor prevents the round-off plateau from forcing an unnecessarily high order.

"""
    vshanalysisorder(sampling; margin=SHEANALYSISMARGIN) -> Int

The Gauss-Legendre analysis order for a level whose production grid is `sampling`.

Share [`SHEANALYSISMARGIN`](@ref) with the scalar builder. The margin allows for basis support
that extends beyond its box, regardless of the angular basis. Measured VSH orders are lower than
scalar orders, but the same analysis-order check applies.
"""
vshanalysisorder(sampling::AbstractSampling; margin::Integer=SHEANALYSISMARGIN) =
    cutoffparameter(sampling) + margin

# Truncate `(a, b)` to degree `P` and stack them into storage's `(2n × nbf)` layout.
# Each family occupies a separate block; `vshindex` makes each retained block contiguous.
function _packvshcoefficients(a, b, n)
    nbf = size(a, 2)
    packed = Matrix{eltype(a)}(undef, 2n, nbf)
    @inbounds for j in 1:nbf
        for q in 1:n
            packed[q, j] = a[q, j]
            packed[n + q, j] = b[q, j]
        end
    end
    return packed
end

"""
    vshcollections(assembler, tree, leafsamplings, ε; margin=SHEANALYSISMARGIN)
        -> (collections, orders)

Build one [`VectorSphericalHarmonicRadiationCollection`](@ref) per leaf, and report the order chosen
for each level.

Like [`shecollections`](@ref), return a concretely typed `Dictionary` keyed by leaf id and
`orders` keyed by each level's sampling object. `assembler` must assemble Cartesian components.
Setup projects these onto the analysis grid's frame; multiplication needs no such projection.
"""
function vshcollections(
    assembler, tree, leafsamplings, ε::Real; margin::Integer=SHEANALYSISMARGIN
)
    leaves = leafids(leafsamplings)
    isempty(leaves) && throw(ArgumentError("no leaves to build tangential storage for"))
    collections = nothing
    orders = IdDict{Any,Int}()

    for (sampling, indices) in _grouptheleavesbylevel(leafsamplings)
        Pa = vshanalysisorder(sampling; margin=margin)
        θs, φs, w = SphericalHarmonics.gausslegendregrid(Pa)
        analysis = SphericalHarmonics.VectorSphericalHarmonicTransform(
            ComplexF64, θs, φs, w, Pa
        )
        θ̂, φ̂ = SphericalHarmonics.sphericalframe(θs, φs)
        points = [
            SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)) for (θ, φ) in zip(θs, φs)
        ]

        boxes = [
            Assembly.BoxFarFieldSampling(
                sampling, points, H2Trees.center(tree, leaves[i]), leaves[i]
            ) for i in indices
        ]
        Fs = assembler(tree, boxes; verbose=false)

        analysed = map(Fs) do F
            Fθ, Fφ = _projectcartesian(F, θ̂, φ̂)
            return SphericalHarmonics.vshanalyze(analysis, Fθ, Fφ)
        end

        P = SphericalHarmonics.vshleveltruncationorder(analysed, ε)
        P < Pa || error(
            "the tangential analysis grid was the binding constraint at this level: the order " *
            "rule returned $P against an analysis order of $Pa, so it is a CLAMP rather than a " *
            "choice and the patterns are under-resolved. This level's translation bandwidth is " *
            "L = $(cutoffparameter(sampling)); `P ≤ L` is NOT guaranteed (see " *
            "`shecollectionbuilder.jl`'s header -- a mesh coarser than the box breaks it), so " *
            "raise the `margin` keyword until the rule returns an order strictly below it.",
        )
        n = SphericalHarmonics.numvshcoefficients(P)
        orders[sampling] = P

        for (k, i) in enumerate(indices)
            leaf = leaves[i]
            a, b = analysed[k]
            collection = VectorSphericalHarmonicRadiationCollection(
                VectorSphericalHarmonicStorage(
                    P, SVector{3}(H2Trees.center(tree, leaf)), _packvshcoefficients(a, b, n)
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

# Project Cartesian components onto the analysis frame at setup, producing two `(nd × nbf)` matrices.
function _projectcartesian(F, θ̂, φ̂)
    nd, nbf = size(F[1])
    Fθ = Matrix{ComplexF64}(undef, nd, nbf)
    Fφ = Matrix{ComplexF64}(undef, nd, nbf)
    @inbounds for j in 1:nbf, q in 1:nd
        x, y, z = F[1][q, j], F[2][q, j], F[3][q, j]
        Fθ[q, j] = θ̂[q, 1] * x + θ̂[q, 2] * y + θ̂[q, 3] * z
        Fφ[q, j] = φ̂[q, 1] * x + φ̂[q, 2] * y + φ̂[q, 3] * z
    end
    return Fθ, Fφ
end

"""
    vshorders(tree, collections, samplings, minrelevantlevel) -> IdDict

Recover the per-level tangential order from already-built collections, keyed by that level's sampling.

As in [`sheorders`](@ref), read orders from stored coefficients to keep scratch consistent.
A smaller scratch order can silently give wrong results even when the GEMM shapes conform.
"""
function vshorders(tree, collections, samplings, minrelevantlevel)
    return _harmonicorders(tree, collections, samplings, minrelevantlevel, "tangential")
end

"""
    vshscratch(sampling, P; ngroups=1) -> VectorSphericalHarmonicLeafScratch

The leaf scratch for a level: the order-`P` tangential synthesis onto `sampling`'s directions, its
parity-folded twin, and room for `ngroups` leaves.
"""
function vshscratch(sampling::AbstractSphericalSampling, P::Integer; ngroups::Integer=1)
    t = SphericalHarmonics.VectorSphericalHarmonicTransform(ComplexF64, sampling, P)
    return VectorSphericalHarmonicLeafScratch(t; ngroups=ngroups)
end

"""
    vshscratchchannel(orders, leaflevels, plans, samplings, minrelevantlevel; groupsperchunk)

Build a workspace-owned `(level, chunk)` pool of [`VectorSphericalHarmonicLeafScratch`](@ref).
Use the same keys as [`shescratchchannel`](@ref) and
[`leafscratchchannel`](@ref MLFMA.Assembly.leafscratchchannel).
"""
function vshscratchchannel(
    orders, leaflevels, plans, samplings, minrelevantlevel; groupsperchunk
)
    return _harmonicscratchchannel(
        orders, leaflevels, plans, samplings, minrelevantlevel
    ) do sampling, P, level
        return vshscratch(sampling, P; ngroups=groupsperchunk(level))
    end
end

# Construction policy.

"""
    VSHFieldCollectionBuilder(side, assemblerbuilder, quadstrategy, ε)

Construction policy for tangential vector-harmonic leaf storage. Use the same
`buildfieldcollections(builder, op, tree, leafsamplings, source)` protocol as
[`SphericalHarmonicFieldCollectionBuilder`](@ref).

`side` is [`RadiationSide`](@ref) or [`ReceiveSide`](@ref) and selects the assembled far-field operator.
The Galerkin receive side uses the lazy builder. It references radiation coefficients and uses
the level's parity-folded blocks rather than storing its own patterns.
"""
struct VSHFieldCollectionBuilder{Side<:AbstractFieldCollectionSide,A,Q,E<:Real} <:
       AbstractCompressedFieldCollectionBuilder{Side}
    side::Side
    assemblerbuilder::A
    quadstrategy::Q
    ε::E
end

# Assemble Cartesian components for projection onto the analysis frame at setup.
function buildfarfieldassembler(builder::VSHFieldCollectionBuilder, op, space)
    return tangentialassembler(
        builder, op, space, "Tangential vector-harmonic leaf storage"
    )
end

#TODO: what is this docstring doing here?
"""
    buildfieldcollections(builder::VSHFieldCollectionBuilder{ReceiveSide}, op, ...)

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
    builder::VSHFieldCollectionBuilder, assembler, tree, leafsamplings
)
    return vshcollections(assembler, tree, leafsamplings, builder.ε)
end

assembledreceivecollectiontype(::VSHFieldCollectionBuilder) =
    VectorSphericalHarmonicAssembledReceiveCollection

# The radiation collection type selects the lazy Galerkin receive storage.
# The antipodal map is `(-1)^l` on `X` and `(-1)^(l+1)` on `Ψ`, folded into the level's blocks.
# Receive collections reference radiation storage and keep the operator's coupling.
# They own no coefficient data or mirror indices.
function buildfieldcollections(
    ::LazyReceiveFieldCollectionBuilder,
    op,
    tree,
    leafsamplings,
    radiationcollections::AbstractDictionary{
        <:Any,<:VectorSphericalHarmonicRadiationCollection
    };
    verbose::Bool=false,
)
    K = receivecoupling(op)
    return [
        VectorSphericalHarmonicReceiveCollection(radiationcollections[leaf], K) for
        leaf in leafids(leafsamplings)
    ]
end
