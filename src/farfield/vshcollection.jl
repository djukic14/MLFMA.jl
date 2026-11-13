# Leaf field collections over the TANGENTIAL VECTOR harmonics, the third leaf storage, after the
# sampled one and Eibert's Cartesian scalar SHE.
#
# These collections share the scalar backend's scratch and batching contracts; see
# `sphericalharmoniccollection.jl`. Galerkin receive collections reuse radiation coefficients.
# Receive scatter transposes the collection and applies no quadrature weights at the leaf.
#
# With `C` for the coefficient matrix and `B` for a side's synthesis:
#
#     sampled     leaf pattern stored as 2NΩ samples per basis function
#     SHE         R = P S C          three CARTESIAN scalar expansions, then a frame projection
#     VSH         R = B C            two TANGENTIAL families, and no projection at all
#
# The frame projection is gone because `X` and `Ψ` are already tangential: synthesis writes `F_θ` and
# `F_φ` directly, which is what the moment stores. SHE has to synthesise three Cartesian components
# and then project, reading 3NΩ to write 2NΩ.
#
# With `A` and `D` the two synthesis blocks (`vectorharmonics.jl` explains
# why four components collapse to two) a side's operator is
#
#     B = [[𝒫, s𝒬], [𝒬, -s𝒫]]
#
# with `(𝒫, 𝒬, s) = (A, D, +1)` on the radiation side and `(Aᵖ, Dᵖ, -1)` on the receive side, `ᵖ`
# being the per-degree parity fold. That single parameterization is why the six orientation/side
# kernels below are two functions and a sign, rather than six bodies, and why the receive collection
# stores nothing: it is the radiation collection read against the level's other pair of blocks.
#
# In measurements on RWG patterns, both bases analysed from the same
# assembled Cartesian samples on the SAME grid: 0.500x Eibert's coefficient count at ε = 1e-3, 0.519x
# at 1e-4, 0.660x at 1e-5. Twice what the count ratio `2P(P+2) / 3(P+1)²` predicts, because VSH also
# needs a lower DEGREE. See `vectorharmonics.jl`'s header for the reason and for the caveat that the
# advantage narrows as ε tightens.

"""
    VectorSphericalHarmonicStorage(order, center, coefficients)

One group's basis-function far-field patterns as tangential vector-harmonic coefficients.

`coefficients` is `2·numvshcoefficients(order) × nbasisfunctions`, column `j` holding basis function
`j`'s `X`-family coefficients followed by its `Ψ`-family ones. Contiguous per basis function and
contiguous per family, which is what lets the aggregation be one GEMV and the synthesis read each
family as a submatrix view rather than a gather.

`center` is the expansion center the coefficients are referenced to; patterns about a different center
differ by `exp(-iκ k̂·Δc)`, which is not a coefficient-space operation, so only collections sharing a
center may be summed. Same contract as
[`SphericalHarmonicStorage`](@ref MLFMA.Harmonics.SphericalHarmonics.SphericalHarmonicStorage).
"""
struct VectorSphericalHarmonicStorage{C<:Complex,T<:Real}
    order::Int
    center::SVector{3,T}
    coefficients::Matrix{C}

    function VectorSphericalHarmonicStorage(
        order::Integer, center::SVector{3,T}, coefficients::Matrix{C}
    ) where {T<:Real,C<:Complex}
        order >= 1 ||
            throw(ArgumentError("a tangential expansion needs degree >= 1, got $order"))
        expected = 2 * SphericalHarmonics.numvshcoefficients(order)
        size(coefficients, 1) == expected || throw(
            DimensionMismatch(
                "coefficients has $(size(coefficients, 1)) rows, expected $expected = " *
                "2·P(P+2) for degree $order",
            ),
        )
        return new{C,T}(Int(order), center, coefficients)
    end
end

order(s::VectorSphericalHarmonicStorage) = s.order
numbasisfunctions(s::VectorSphericalHarmonicStorage) = size(s.coefficients, 2)
coefficientmatrix(s::VectorSphericalHarmonicStorage) = s.coefficients

# --- the scratch ----------------------------------------------------------------------------------

"""
    VectorSphericalHarmonicLeafScratch(t; ngroups=1)

The per-`(level, chunk)` working set for a tangential-harmonic leaf multiply.

Splits the same way [`SphericalHarmonicLeafScratch`](@ref) does. `A`/`D` and their parity-folded twins
`Aᵖ`/`Dᵖ` depend only on the LEVEL's directions and order, so every leaf at a level shares them by
reference; `coefficients`, `θfield` and `φfield` are genuine scratch sized for the largest batch.

FOUR SHARED MATRICES rather than SHE's two, because the receive side needs the parity-folded pair as
well. They are not free but they are per level, not per leaf, and the order VSH needs is lower, so
in practice this is smaller than the SHE pair it replaces.
"""
struct VectorSphericalHarmonicLeafScratch{C<:Complex}
    order::Int
    A::Matrix{C}
    D::Matrix{C}
    Aᵖ::Matrix{C}
    Dᵖ::Matrix{C}
    # `(2n × ngroups)`: column `g` is `[a_g; b_g]`, so the aggregation GEMV writes it contiguously and
    # the synthesis reads each family as `view(coefficients, 1:n, 1:k)`, a real submatrix, which is
    # what keeps the batched synthesis a single GEMM per family.
    coefficients::Matrix{C}
    θfield::Matrix{C}
    φfield::Matrix{C}
    leaves::Vector{Int}
end

# Unlike `SphericalHarmonicLeafScratch`, this type has no real frame arrays `θ̂`/`φ̂`.
# It needs only the complex element type; a second parameter could not be inferred from the fields.

function VectorSphericalHarmonicLeafScratch(
    t::SphericalHarmonics.VectorSphericalHarmonicTransform{T,C}; ngroups::Integer=1
) where {T,C}
    ngroups >= 1 || throw(ArgumentError("ngroups must be at least 1, got $ngroups"))
    Aᵖ, Dᵖ = SphericalHarmonics.vshmirrorblocks(t)
    n = SphericalHarmonics.numvshcoefficients(t)
    nd = size(t.A, 1)
    return VectorSphericalHarmonicLeafScratch(
        SphericalHarmonics.order(t),
        t.A,
        t.D,
        Aᵖ,
        Dᵖ,
        Matrix{C}(undef, 2n, ngroups),
        Matrix{C}(undef, nd, ngroups),
        Matrix{C}(undef, nd, ngroups),
        Vector{Int}(undef, ngroups),
    )
end

order(s::VectorSphericalHarmonicLeafScratch) = s.order
maxgroups(s::VectorSphericalHarmonicLeafScratch) = size(s.coefficients, 2)
numdirections(s::VectorSphericalHarmonicLeafScratch) = size(s.θfield, 1)

# The two block pairs, named by the side that owns them. `s` is the sign in `[[𝒫, s𝒬], [𝒬, -s𝒫]]`;
# see this file's header.
_radiationblocks(s::VectorSphericalHarmonicLeafScratch) = (s.A, s.D, 1)
_receiveblocks(s::VectorSphericalHarmonicLeafScratch) = (s.Aᵖ, s.Dᵖ, -1)

# --- the collections ------------------------------------------------------------------------------

"""
    VectorSphericalHarmonicRadiationCollection(storage)

A leaf's radiation collection over the tangential vector harmonics.

Interchangeable with [`SphericalHarmonicRadiationCollection`](@ref) and the sampled
`RadiationFieldCollection` from the traversal's point of view: it declares `D = 2` and produces a
`TangentialSphericalFarField`. Unlike SHE it produces it DIRECTLY. There are no Cartesian components
here, internal or otherwise.
"""
struct VectorSphericalHarmonicRadiationCollection{T<:Complex,S} <:
       AbstractRadiationFieldCollection{2,T}
    storage::S

    function VectorSphericalHarmonicRadiationCollection(
        storage::VectorSphericalHarmonicStorage{C}
    ) where {C<:Complex}
        return new{C,typeof(storage)}(storage)
    end
end

fielddim(::Type{<:VectorSphericalHarmonicRadiationCollection}) = 2
storage(c::VectorSphericalHarmonicRadiationCollection) = c.storage
order(c::VectorSphericalHarmonicRadiationCollection) = order(c.storage)
numbasisfunctions(c::VectorSphericalHarmonicRadiationCollection) =
    numbasisfunctions(c.storage)
Base.size(c::VectorSphericalHarmonicRadiationCollection) =
    (size(coefficientmatrix(c.storage), 1), numbasisfunctions(c))

"""
    VectorSphericalHarmonicReceiveCollection(radiation, coupling)

The Galerkin receive collection for a leaf stored as tangential vector harmonics.

Holds a reference to the radiation collection and the operator's coupling map.
The antipodal mirror is `(-1)^l` on the `X` family and `(-1)^(l+1)` on `Ψ`,
derived in `vectorharmonics.jl` and checked there against `MLFMA.mirrorindices`/`mirrorrule`, and
both signs are already folded into the level's `Aᵖ`/`Dᵖ` blocks and the receive side's sign. So there
is no index array, no mirror pass, and no separate set of receive coefficients.
"""
struct VectorSphericalHarmonicReceiveCollection{T<:Complex,R} <:
       AbstractReceiveFieldCollection{2,T}
    radiation::R
    # The operator's receive coupling. See `receivecoupling` in
    # `sphericalharmoniccollection.jl`, whose derivation applies here unchanged: the tangential
    # mirror is a property of the SPHERE, the coupling a property of the OPERATOR, and only for
    # `MWEFIE3D` do the two coincide.
    coupling::SMatrix{2,2,T,4}

    function VectorSphericalHarmonicReceiveCollection(
        radiation::VectorSphericalHarmonicRadiationCollection{C}, coupling::AbstractMatrix
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

fielddim(::Type{<:VectorSphericalHarmonicReceiveCollection}) = 2
radiationcollection(c::VectorSphericalHarmonicReceiveCollection) = c.radiation
coupling(c::VectorSphericalHarmonicReceiveCollection) = c.coupling
storage(c::VectorSphericalHarmonicReceiveCollection) = storage(c.radiation)
order(c::VectorSphericalHarmonicReceiveCollection) = order(c.radiation)
numbasisfunctions(c::VectorSphericalHarmonicReceiveCollection) =
    numbasisfunctions(c.radiation)

"""
    VectorSphericalHarmonicAssembledReceiveCollection(radiation)

A leaf's assembled receive patterns over the tangential vector harmonics. This is the Petrov test
side under [`VectorSphericalHarmonicLeafStorage`](@ref MLFMA.Assembly.VectorSphericalHarmonicLeafStorage).

Wraps a [`VectorSphericalHarmonicRadiationCollection`](@ref) whose `X`/`Ψ` coefficients were analysed
from `receiveoperator(op)`'s patterns, so it holds this side's own data and every kernel forwards to
the radiation one. The separate type exists so `<: AbstractReceiveFieldCollection` still says which
side the collection is on. See the SHE twin's docstring, whose argument applies here verbatim.

Not [`VectorSphericalHarmonicReceiveCollection`](@ref), which is the GALERKIN side: that one derives
its patterns from the trial leaf's through the `(-1)^l` / `(-1)^(l+1)` family parities and the
operator's receive coupling. This one applies neither.
"""
struct VectorSphericalHarmonicAssembledReceiveCollection{T<:Complex,R} <:
       AbstractReceiveFieldCollection{2,T}
    radiation::R
end

VectorSphericalHarmonicAssembledReceiveCollection(
    radiation::VectorSphericalHarmonicRadiationCollection{C}
) where {C<:Complex} =
    VectorSphericalHarmonicAssembledReceiveCollection{C,typeof(radiation)}(radiation)

fielddim(::Type{<:VectorSphericalHarmonicAssembledReceiveCollection}) = 2
radiationcollection(c::VectorSphericalHarmonicAssembledReceiveCollection) = c.radiation
storage(c::VectorSphericalHarmonicAssembledReceiveCollection) = storage(c.radiation)
order(c::VectorSphericalHarmonicAssembledReceiveCollection) = order(c.radiation)
numbasisfunctions(c::VectorSphericalHarmonicAssembledReceiveCollection) =
    numbasisfunctions(c.radiation)
Base.size(c::VectorSphericalHarmonicAssembledReceiveCollection) = size(c.radiation)

"""
    VectorSphericalHarmonicAssembledCollection

Either VSH collection whose coefficients were ASSEMBLED and analysed. Excludes
[`VectorSphericalHarmonicReceiveCollection`](@ref), which stores nothing of its own. See the SHE
twin.
"""
const VectorSphericalHarmonicAssembledCollection = Union{
    VectorSphericalHarmonicRadiationCollection,
    VectorSphericalHarmonicAssembledReceiveCollection,
}

# --- the two shared kernels -----------------------------------------------------------------------

# `Kᵀ moment` in the scratch's own field columns, or the moment's own arrays untouched when `K` is the
# identity, which is the EFIE case, and means the measured kernel is still exactly what runs there.
function _stagecoupled!(scratch, moment, K)
    Fθ = H2Factory.wrapvector(moment, 1)
    Fφ = H2Factory.wrapvector(moment, 2)
    _isplaincoupling(K) && return Fθ, Fφ
    @views stagedθ = scratch.θfield[:, 1]
    @views stagedφ = scratch.φfield[:, 1]
    @inbounds for q in eachindex(Fθ)
        a, b = Fθ[q], Fφ[q]
        stagedθ[q] = K[1, 1] * a + K[2, 1] * b
        stagedφ[q] = K[1, 2] * a + K[2, 2] * b
    end
    return stagedθ, stagedφ
end

"""
    _vshaggregate!(moment, scratch, storage, x, blocks) -> moment

`moment = B C x`, with `B = [[𝒫, s𝒬], [𝒬, -s𝒫]]` the block operator of whichever side owns `blocks`.

Two stages and no third: the coefficient GEMV, then the synthesis. There is no frame projection,
that is the structural saving over SHE, whose synthesis lands in Cartesian components and needs a
third pass to reach the tangential moment.
"""
function _vshaggregate!(moment, scratch, storage, x, blocks)
    𝒫, 𝒬, s = blocks
    n = size(𝒫, 2)
    @views c = scratch.coefficients[:, 1]
    LinearAlgebra.mul!(c, coefficientmatrix(storage), x)
    @views a = scratch.coefficients[1:n, 1]
    @views b = scratch.coefficients[(n + 1):(2n), 1]

    Fθ = H2Factory.wrapvector(moment, 1)
    Fφ = H2Factory.wrapvector(moment, 2)
    LinearAlgebra.mul!(Fθ, 𝒫, a)
    LinearAlgebra.mul!(Fθ, 𝒬, b, s, true)
    LinearAlgebra.mul!(Fφ, 𝒬, a)
    LinearAlgebra.mul!(Fφ, 𝒫, b, -s, true)
    return moment
end

"""
    _vshscatter!(y, scratch, storage, moment, blocks, α, β, mode) -> y

`y = α (B C)ᵀ moment + β y`, or the adjoint of the same when `mode` is `_Adjointed()`.

`Bᵀ = [[𝒫ᵀ, 𝒬ᵀ], [s𝒬ᵀ, -s𝒫ᵀ]]`, and `Bᴴ` is the same with `s` unchanged because the sign is real.
No quadrature weights are applied at the leaf. This is a moment collection, not a testing
functional; see the receive-side header in `sphericalharmoniccollection.jl`.
"""
function _vshscatter!(y, scratch, storage, moment, blocks, α, β, mode, K=LinearAlgebra.I)
    𝒫, 𝒬, s = blocks
    n = size(𝒫, 2)
    # `Kᵀ moment`, STAGED rather than folded: unlike the scalar backend this one has a free
    # `(NΩ × 2)` block in its own scratch, so the four `mul!`s below stay unchanged and the coupling
    # costs one pass instead of four extra scalar products inside each of them.
    Fθ, Fφ = _stagecoupled!(scratch, moment, K)

    @views a = scratch.coefficients[1:n, 1]
    @views b = scratch.coefficients[(n + 1):(2n), 1]
    LinearAlgebra.mul!(a, _reverse(mode, 𝒫), Fθ)
    LinearAlgebra.mul!(a, _reverse(mode, 𝒬), Fφ, true, true)
    LinearAlgebra.mul!(b, _reverse(mode, 𝒬), Fθ, s, false)
    LinearAlgebra.mul!(b, _reverse(mode, 𝒫), Fφ, -s, true)

    @views c = scratch.coefficients[:, 1]
    LinearAlgebra.mul!(
        y, _reverse(mode, coefficientmatrix(storage)), c, α, iszero(β) ? false : β
    )
    return y
end

# --- the six orientation/side kernels --------------------------------------------------------------
#
# Same six as SHE, and for the same reason: H2Factory realizes the transpose and the adjoint by
# swapping the sides, so each collection aggregates in one orientation and scatters in another, and a
# forward matvec exercises only half of them.

# FORWARD aggregation, trial side.
function H2Factory.applyleafmul!(
    scratch::VectorSphericalHarmonicLeafScratch,
    moment::TangentialSphericalFarField,
    collection::VectorSphericalHarmonicAssembledCollection,
    x::AbstractVector,
)
    return _vshaggregate!(
        moment, scratch, storage(collection), x, _radiationblocks(scratch)
    )
end

# FORWARD scatter, test side.
function H2Factory.applyleafmul!(
    scratch::VectorSphericalHarmonicLeafScratch,
    y::AbstractVector,
    collection::VectorSphericalHarmonicReceiveCollection,
    moment::TangentialSphericalFarField,
    α::Number,
    β::Number,
)
    return _vshscatter!(
        y,
        scratch,
        storage(collection),
        moment,
        _receiveblocks(scratch),
        α,
        β,
        _Transposed(),
        coupling(collection),
    )
end

# Transpose aggregation, test side.
function H2Factory.applyleafmul!(
    scratch::VectorSphericalHarmonicLeafScratch,
    moment::TangentialSphericalFarField,
    collection::VectorSphericalHarmonicReceiveCollection,
    x::AbstractVector,
)
    _vshaggregate!(moment, scratch, storage(collection), x, _receiveblocks(scratch))
    return _applycoupling!(moment, coupling(collection))
end

# Transpose scatter, trial side.
function H2Factory.applyleafmul!(
    scratch::VectorSphericalHarmonicLeafScratch,
    y::AbstractVector,
    collection::VectorSphericalHarmonicAssembledCollection,
    moment::TangentialSphericalFarField,
    α::Number,
    β::Number,
)
    return _vshscatter!(
        y,
        scratch,
        storage(collection),
        moment,
        _radiationblocks(scratch),
        α,
        β,
        _Transposed(),
    )
end

# ADJOINT aggregation, test side: `moment = conj(Rec) x == conj(Rec conj(x))`.
# The final conjugation covers the coefficients, synthesis blocks, and coupling; `s` is real.
# It is safe because this leaf overwrites the moment. See `adjointleafmul.jl`.
function applyconjugatedleafmul!(
    scratch::VectorSphericalHarmonicLeafScratch,
    moment::TangentialSphericalFarField,
    collection::VectorSphericalHarmonicReceiveCollection,
    x::AbstractVector,
)
    𝒫, 𝒬, s = _receiveblocks(scratch)
    n = size(𝒫, 2)
    @views c = scratch.coefficients[:, 1]
    _aggregateconjugated!(c, coefficientmatrix(storage(collection)), x)
    @views a = scratch.coefficients[1:n, 1]
    @views b = scratch.coefficients[(n + 1):(2n), 1]

    Fθ = H2Factory.wrapvector(moment, 1)
    Fφ = H2Factory.wrapvector(moment, 2)
    LinearAlgebra.mul!(Fθ, 𝒫, a)
    LinearAlgebra.mul!(Fθ, 𝒬, b, s, true)
    LinearAlgebra.mul!(Fφ, 𝒬, a)
    LinearAlgebra.mul!(Fφ, 𝒫, b, -s, true)
    # BEFORE the conjugation. See the scalar twin in `adjointleafmul.jl` for why.
    _applycoupling!(moment, coupling(collection))
    conj!(Fθ)
    conj!(Fφ)
    return moment
end

# ADJOINT aggregation on an ASSEMBLED test side: `moment = conj(Rec) x` with `Rec` built from the
# Radiation blocks: a Petrov map's compressed test collections store their own patterns. The
# kernel uses radiation blocks rather than receive blocks and applies no coupling before conjugation.
function applyconjugatedleafmul!(
    scratch::VectorSphericalHarmonicLeafScratch,
    moment::TangentialSphericalFarField,
    collection::VectorSphericalHarmonicAssembledReceiveCollection,
    x::AbstractVector,
)
    𝒫, 𝒬, s = _radiationblocks(scratch)
    n = size(𝒫, 2)
    @views c = scratch.coefficients[:, 1]
    _aggregateconjugated!(c, coefficientmatrix(storage(collection)), x)
    @views a = scratch.coefficients[1:n, 1]
    @views b = scratch.coefficients[(n + 1):(2n), 1]

    Fθ = H2Factory.wrapvector(moment, 1)
    Fφ = H2Factory.wrapvector(moment, 2)
    LinearAlgebra.mul!(Fθ, 𝒫, a)
    LinearAlgebra.mul!(Fθ, 𝒬, b, s, true)
    LinearAlgebra.mul!(Fφ, 𝒬, a)
    LinearAlgebra.mul!(Fφ, 𝒫, b, -s, true)
    conj!(Fθ)
    conj!(Fφ)
    return moment
end

# ADJOINT scatter, trial side. `α` arrives already conjugated from `H2Factory.α(core, ::isAdjoint)`.
function applyconjugatedleafmul!(
    scratch::VectorSphericalHarmonicLeafScratch,
    result::AbstractVector,
    collection::VectorSphericalHarmonicAssembledCollection,
    moment::TangentialSphericalFarField,
    α::Number,
    β::Number,
)
    return _vshscatter!(
        result,
        scratch,
        storage(collection),
        moment,
        _radiationblocks(scratch),
        α,
        β,
        _Adjointed(),
    )
end

# --- the guards -----------------------------------------------------------------------------------

# Over the ASSEMBLED UNION rather than its two members separately, and the difference is not
# cosmetic: the kernels above dispatch on the union, so a guard naming only the radiation member would
# be MORE specific in its collection argument while the kernel is more specific in its scratch and
# moment arguments, an ambiguity, which Julia reports at the call rather than at definition.
for Collection in
    (:VectorSphericalHarmonicAssembledCollection, :VectorSphericalHarmonicReceiveCollection)
    @eval function H2Factory.applyleafmul!(
        ::Any, ::AbstractFarField, collection::$Collection, ::AbstractVector
    )
        return error(
            "a $(typeof(collection)) needs a `VectorSphericalHarmonicLeafScratch` to multiply " *
            "through: it reconstructs via the level's synthesis blocks, which must be neither " *
            "allocated per matvec nor stored per leaf. The core must implement " *
            "`H2Factory.buildleafchannels` returning that scratch. See `MLFMA.vshscratchchannel`.",
        )
    end

    @eval function H2Factory.applyleafmul!(
        ::Any,
        ::AbstractVector,
        collection::$Collection,
        ::AbstractFarField,
        ::Number,
        ::Number,
    )
        return error(
            "a $(typeof(collection)) needs a `VectorSphericalHarmonicLeafScratch` to scatter " *
            "through, for the same reason it needs one to aggregate.",
        )
    end

    @eval function LinearAlgebra.mul!(
        ::AbstractFarField, collection::$Collection, ::AbstractVector
    )
        return error(
            "a $(typeof(collection)) has no plain `mul!`: it must be applied through " *
            "`H2Factory.applyleafmul!` with a `VectorSphericalHarmonicLeafScratch`.",
        )
    end
end

# --- the batched chunk multiply --------------------------------------------------------------------
#
# Same trade as SHE's, and mandatory for the same reason: the blocks are shared by every leaf at a
# level, so one GEMM per family across a chunk streams them once instead of once per leaf. The
# per-node path above is correct but re-streams them.
#
# The staging pass here is a COPY rather than SHE's frame projection, the synthesis already produced
# the moment's own components, so it moves 2NΩ words per leaf where SHE's projection read 3NΩ to
# write 2NΩ.

_batchableradiation(::VectorSphericalHarmonicAssembledCollection) = true

function H2Factory.aggregateleafchunk!(
    A,
    tree,
    chunknodes,
    moment,
    momentcollection,
    leafmul!,
    x,
    level,
    scratch::VectorSphericalHarmonicLeafScratch,
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
        _gathervshcoefficients!,
        _flushvshbatch!,
    )
end

function _gathervshcoefficients!(A, tree, node, momentcollection, x, scratch, g)
    collection = momentcollection(node)
    coefficients = coefficientmatrix(storage(collection))
    size(coefficients, 1) == size(scratch.coefficients, 1) || throw(
        DimensionMismatch(
            "leaf $node expands to order $(order(collection)), but this level's scratch is " *
            "sized for order $(order(scratch)). One order per LEVEL is required, a ragged " *
            "batch has no single pair of synthesis blocks.",
        ),
    )
    buffer = H2Factory.aggregationvectorbuffer(A, node)
    buffer .= view(x, H2Trees.values(tree, node))
    @views LinearAlgebra.mul!(scratch.coefficients[:, g], coefficients, buffer)
    return nothing
end

# Scheduled through `LeafMulOp`, exactly as the scalar `_flushbatch!` is and for the same reasons,
# see that function. Measured 1.35x from eight threads here against the scalar route's 4.0x: this
# projection is two column copies per group where the scalar one recombines a frame, so there is less
# to parallelise, but the sign is the same and one policy serves both.
function _flushvshbatch!(moment, scratch, filled, scheduler)
    A, D, s = _radiationblocks(scratch)
    n = size(A, 2)
    @views a = scratch.coefficients[1:n, 1:filled]
    @views b = scratch.coefficients[(n + 1):(2n), 1:filled]
    @views Fθ = scratch.θfield[:, 1:filled]
    @views Fφ = scratch.φfield[:, 1:filled]
    LinearAlgebra.mul!(Fθ, A, a)
    LinearAlgebra.mul!(Fθ, D, b, s, true)
    LinearAlgebra.mul!(Fφ, D, a)
    LinearAlgebra.mul!(Fφ, A, b, -s, true)

    OhMyThreads.tforeach(1:filled; scheduler=projectionscheduler(scheduler, filled)) do g
        return _copyintomoment!(moment(scratch.leaves[g]), scratch, g)
    end
    return nothing
end

function _copyintomoment!(moment, scratch, g)
    @views H2Factory.wrapvector(moment, 1) .= scratch.θfield[:, g]
    @views H2Factory.wrapvector(moment, 2) .= scratch.φfield[:, g]
    return moment
end
