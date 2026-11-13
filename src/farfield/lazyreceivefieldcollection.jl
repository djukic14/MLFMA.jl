# Tests use the ordinary `radiationcollection * mirrored-far-field` path as an independent reference.

"""
    LazyReceiveFieldCollection(operator, farfieldcollection, mirrorindices)

The test side of a leaf, derived from the trial side instead of assembled.

This H2Factory leaf moment collection requires a leaf scratch buffer. It is not matrix-like.
It implements the two `applyleafmul!` arities used by the traversal, but no `mul!`/`conjmul!` of its own.

It wraps the RADIATION collection and derives the receive side on demand from `mirrorindices` and
[`receivecouplingrule`](@ref MLFMA.Seams.receivecouplingrule), so the test side is never assembled or
stored.

The workspace must provide a leaf channel. It owns one far field per `(level, chunk)`
(see [`leafscratchchannel`](@ref MLFMA.Assembly.leafscratchchannel)), so mirroring into a real far
field and calling BLAS is allocation-free, and measured 27-45x faster than a fused loop
(see `benchmark/README.md`).

There is no fallback without a leaf channel. `H2Factory.buildleafchannels` defaults to
`EmptyChannels()`, so the core must implement it. `_requiresleafscratch` reports a specific error
when scratch is missing.

The operator and the wrapped collection must agree on representation. A collection of a different
representation would be indexed with component indices meant for another layout and would silently
produce wrong numbers wherever those indices happen to be in range, which they always are once the
component counts coincide. That is checked once at construction rather than in the per-sample
kernels.
"""
struct LazyReceiveFieldCollection{OperatorType,FFCol,I}
    operator::OperatorType
    farfieldcollection::FFCol
    mirrorindices::I

    # `applyleafmul!` takes its component rule from `operator` and the far field's type, but reads
    # matrices out of `farfieldcollection`. Those two sources of truth have to agree: a collection of
    # a different representation would be indexed with component indices meant for another layout,
    # and would silently produce wrong numbers wherever the indices happen to be in range (which they
    # always are once the component counts coincide -- e.g. a
    # `SphericalVectorRadiationFieldCollection` driven by a tangential θ↔φ rule).
    #
    # Checked once here rather than in the kernels, which run per sample point. An operator with no
    # `farfieldtype` method fails here with a `MethodError` naming it, which is also the right
    # outcome -- there is no defensible rule to apply for an operator whose representation is
    # unknown.
    function LazyReceiveFieldCollection(
        operator::OperatorType, farfieldcollection::FFCol, mirrorindices::I
    ) where {OperatorType,FFCol,I}
        # `I` is unconstrained, so reject a sampling here before `transform!` tries to use it
        # as an index array. The caller must supply mirror indices.
        I <: AbstractArray{<:Integer} || error(
            "LazyReceiveFieldCollection stores mirror INDICES, not a sampling: expected an " *
            "AbstractArray of integers, got $I. Pass `MLFMA.mirrorindices(sampling)`.",
        )
        ExpectedCollection = radiationfieldcollectiontype(operator)
        farfieldcollection isa ExpectedCollection || error(
            "LazyReceiveFieldCollection needs a radiation collection matching its operator's " *
            "far-field representation: $(nameof(OperatorType)) radiates " *
            "$(nameof(farfieldtype(operator))), so it requires a " *
            "$(nameof(ExpectedCollection)), but got a $(nameof(FFCol))",
        )
        return new{OperatorType,FFCol,I}(operator, farfieldcollection, mirrorindices)
    end
end

# The constructor takes mirror indices, not a sampling. Callers write
# `LazyReceiveFieldCollection(op, collection, MLFMA.mirrorindices(sampling))`, which says what is
# actually being stored.

function operator(mirroredffcol::LazyReceiveFieldCollection)
    return mirroredffcol.operator
end

function farfieldcollection(mirroredffcol::LazyReceiveFieldCollection)
    return mirroredffcol.farfieldcollection
end

function mirrorindices(mirroredffcol::LazyReceiveFieldCollection)
    return mirroredffcol.mirrorindices
end

# The mirror-and-scale kernel both leaf multiplies are built on: materializes (mirror + scale) `src`'s
# data into `dest` via the `receivecouplingrule` table for the far field's own concrete type and
# `op`. Only the six operator/far-field-type pairs `receivecouplingrule` covers work here (see
# `operatorutils.jl`); `DoubleLayerRotatedMW3D`, `HH3DDoubleLayerFDBIO` and
# `HH3DDoubleLayerTransposedFDBIO` have no lazy-receive support and fail as a missing method.
#
# Constrained to a single `F` on both sides, exactly like `mirror!`: a receive coupling rule permutes
# components WITHIN one representation, it never converts between representations. With two
# independent `AbstractFarField`s a `SphericalVectorFarField` source could be written into a
# `CartesianVectorFarField` destination purely because both have three components, silently
# reinterpreting r,θ,φ as x,y,z. This makes that a `MethodError`.
#
# Safe to require the FULL concrete type (not just the same representation): a pool-wrapped moment
# from `H2Factory.wrapbuffer` and a workspace's freshly allocated leaf scratch resolve to the
# identical `{D,T,P,V}` -- both end up `Matrix`/`Ptr`/`Memory`-backed -- which is what lets the
# scratch path below hand one straight to the other. Pinned in `test/FarField/rules.jl`.
function receivefields!(dest::F, src::F, mirrorindices, op) where {F<:AbstractFarField}
    transform!(dest, src, mirrorindices, receivecouplingrule(op, F))
    return dest
end

# --- THE TWO LEAF MULTIPLIES --------------------------------------------------------------------
#
# During disaggregation, `receivefields!` writes `scratch[dest][s] = coeff *
# farfield[src][idx[s]]` for every mapping in the rule, which is exactly the materialized receive
# collection's component `dest` -- so handing `scratch` to the RADIATION collection's ordinary `mul!`
# computes
#
#     result[basis] = α * Σ_dest Σ_s C[dest][s,basis] * scratch[dest][s] + β * result[basis]
#                   = α * Σ_mappings coeff * Σ_s C[dest][s,basis] * farfield[src][idx[s]] + β * (...)
#
# one contiguous `gemv` per component.
#
# `α::AbstractVector` needs no special case: H2Factory's own vector-`α` collection `mul!` weights
# component `i` by `α[i]`, and `scratch`'s component `i` IS the receive collection's component `i`, so
# the weight lands on the term it should. That path is not optional -- leaf disaggregation calls
# `leafmul!(..., α(A), false)` (H2Factory `farinteractions/disaggregation.jl`) and `MLFMA.α` is
# genuinely a vector for `MWSingleLayer3D` (3 entries) and `HH3DHyperSingularFDBIO` (6).
#
# `β::Number` deliberately, so a vector `β` is a clear `MethodError` listing what IS accepted rather
# than an arithmetic failure inside the collection. H2Factory defines result-side vector-`β` overloads
# ONLY for single-component collections (`MType{1}`/`MColType{1}`, where they index `β[1]`), so there
# is no general-`D` meaning to mirror and inventing one would be guessing. Nothing reaches it either:
# both leaf disaggregation sites pass a literal `false`.
function H2Factory.applyleafmul!(
    scratch::AbstractFarField,
    result::AbstractArray,
    mirroredfcol::LazyReceiveFieldCollection,
    farfield::AbstractFarField,
    α,
    β::Number,
)
    receivefields!(scratch, farfield, mirrorindices(mirroredfcol), operator(mirroredfcol))
    return LinearAlgebra.mul!(result, farfieldcollection(mirroredfcol), scratch, α, β)
end

# WRITE-THROUGH SIDE (aggregation, reached when a transposed/adjoint map aggregates the test side).
# The transpose of the read side, and NOT `receivefields!` run backwards: transposing a mapping swaps
# which end is read from and which is written into, because `(ABC)^T = C^T B^T A^T` swaps operand
# order. So `receivefields!`'s roles are exchanged here -- component `srccomponent` of the moment is
# written, component `destcomponent` of the scratch is read -- and the mirror moves to the READ index:
#
#     mul!(scratch, C, y)          =>  scratch[i][s] = Σ_basis C[i][s,basis] * y[basis]
#     moment[src][s] = coeff * scratch[dest][idx[s]]
#
# `idx` is a genuine permutation of the sample index space AND an involution (`idx[idx[i]] == i`):
# `mirrorindices` encodes the θ→π-θ, φ→φ+π reflection, whose own inverse is itself, so `P^T = P^{-1} =
# P` and the transposed map gathers with the same index array.
#
# Getting the swap wrong is invisible on every rule shipped today -- each is either diagonal
# (`dest == src`, nothing to swap) or a two-way swap with the SAME coefficient both ways
# (`MWDoubleLayer3D`: `{(1,2,γ),(2,1,γ)}`), which is symmetric as a set of mappings. So
# `test/FarField/rules.jl` pins it with a hand-built ASYMMETRIC rule unrelated to any real operator,
# checking the defining bilinear transpose property `<Ax,y> == <x,A^T y>` directly rather than
# trusting today's rules to be discriminating.
#
# Only the 4-argument (α = true, β = false) arity exists because that is the only one the aggregation
# leaf multiply ever calls: `upwardpass!` passes no `α`/`β` at all (H2Factory
# `farinteractions/traversalhelpers.jl`).
function H2Factory.applyleafmul!(
    scratch::AbstractFarField,
    moment::AbstractFarField,
    mirroredfcol::LazyReceiveFieldCollection,
    y::AbstractArray,
)
    LinearAlgebra.mul!(scratch, farfieldcollection(mirroredfcol), y)
    _lazyreceivescatter!(
        moment,
        scratch,
        mirrorindices(mirroredfcol),
        receivecouplingrule(operator(mirroredfcol), typeof(moment)),
    )
    return moment
end

# The gather half of the write-through. Peels `rule::Tuple` one `ComponentMap` at a time via
# `first`/`Base.tail` rather than iterating it, same reason as `transform!` in representation.jl: the
# generic tuple iteration protocol infers a small `Union` across a heterogeneous tuple, which
# measurably allocates even though every element is isbits.
function _lazyreceivescatter!(moment, scratch, idx, rule::Tuple)
    mapping = first(rule)
    destarray = moment[srccomponent(mapping)]
    S = scratch[destcomponent(mapping)]
    c = coefficient(mapping)

    @inbounds for sampleindex in eachindex(idx)
        destarray[sampleindex] = c * S[idx[sampleindex]]
    end
    _lazyreceivescatter!(moment, scratch, idx, Base.tail(rule))
    return moment
end

function _lazyreceivescatter!(moment, scratch, idx, ::Tuple{})
    return moment
end

# `borrowbuffer` returns `EmptyChannels()` for a level with no scratch pool, and
# `H2Factory.buildleafchannels` defaults to `EmptyChannels()` for a core that does not implement it --
# so this is reachable by a construction mistake (a core that forgot the hook, or a leaf at a level the
# pool does not cover), not by a caller doing anything exotic.
#
# Caught here rather than left as a `MethodError` on `applyleafmul!` because the message is the whole
# value: the generic `applyleafmul!` would otherwise fall through to `LinearAlgebra.mul!`, and the
# resulting "no method matching mul!(::Vector, ::LazyReceiveFieldCollection, ::TangentialSphericalFarField)"
# names none of the three things that actually went wrong.
function _requiresleafscratch(mirroredfcol)
    return error(
        "LazyReceiveFieldCollection requires a leaf scratch buffer, but the traversal borrowed " *
        "`EmptyChannels()` -- meaning this workspace's leaf channel has no pool for this leaf's " *
        "level. It is not a general matrix-like collection and implements no `mul!` of its own: " *
        "the core must implement `H2Factory.buildleafchannels` (see `MLFMA.leafscratchchannel`), " *
        "covering every traversed level that holds leaves. Operator: " *
        "$(nameof(typeof(operator(mirroredfcol)))).",
    )
end

function H2Factory.applyleafmul!(
    ::H2Factory.EmptyChannels,
    ::AbstractArray,
    mirroredfcol::LazyReceiveFieldCollection,
    ::AbstractFarField,
    ::Any,
    ::Number,
)
    return _requiresleafscratch(mirroredfcol)
end

function H2Factory.applyleafmul!(
    ::H2Factory.EmptyChannels,
    ::AbstractFarField,
    mirroredfcol::LazyReceiveFieldCollection,
    ::AbstractArray,
)
    return _requiresleafscratch(mirroredfcol)
end
