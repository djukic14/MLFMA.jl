# A leaf radiation collection that stores circular-harmonic coefficients instead of samples.
# This is the 2-D counterpart of `sphericalharmoniccollection.jl`.
#
# Where a sampled 2-D collection holds `D` matrices of `(Nφ × N_j)`, this holds one
# `((2M+1)·D × N_j)` coefficient matrix and reconstructs on demand. Measured: 1.47x less memory at
# ε = 1e-5 on a λ/4 leaf. Ordinary fixtures did not run faster; see
# `CircularHarmonicLeafStorage`'s docstring for the table and `MLFMA.CircularHarmonics` for why the
# 3-D speedup does not transfer.
#
# ------------------------------------------------------------------------------------------------
# Three differences from the spherical backend:
#
# 1. No frame projection. The spherical scalar backend stores three Cartesian components and
#    produces two tangential ones because θ̂/φ̂ are singular at the poles. Here `x̂`/`ŷ` do not
#    depend on `k̂`, and scalars need no frame. Stored and output components coincide:
#    `fielddim` is `ncomponents(storage)`. The `_projectinto!`/`_backprojectinto!` steps become
#    copies. The unbatched path skips even that copy by synthesising into the moment's arrays.
#
# 2. No sampling argument for scratch. `SphericalHarmonicLeafScratch` needs the sampling to build
#    `sphericalframe`. Here the transform alone determines everything the level shares.
#
# 3. No batch threshold. The spherical backend uses `SHEBATCHTHRESHOLD = 16` to avoid small batches.
#    In 2-D, batching beat per-leaf synthesis from two leaves upward: 0.61x runtime at 2, 0.36x at 8,
#    and 0.28x at 64. At one leaf it was similar (0.98x). These measurements support batching at
#    every size. The per-node path remains for orientations that cannot use the batch kernel.
#
# ------------------------------------------------------------------------------------------------
# Batching reduces overhead even though the 2-D synthesis matrix fits in cache.
#
# In 3-D `S` is `(1568 × 300)` = 2.4 MB, so a per-leaf synthesis re-streams it once per leaf and the
# leaf multiply is memory-bound. A 2-D `S` is `(28 × 19)` = 8.3 KB and fits in L1, so repeatedly
# loading a large matrix is not the main cost here.
#
# Batching measured 0.28x-0.72x runtime here, compared with 3-D's 0.29x-0.38x. It replaces 480 tiny
# `mul!` calls into BLAS with one GEMM. These measurements attribute the benefit to call overhead,
# not memory traffic; they do not establish how the benefit scales with larger grids.
#
# ------------------------------------------------------------------------------------------------
# `@linkmomenttocollection` assumes `moment[i] = collection[i] * y` with an independent matrix
# for each component. Here each component uses the shared synthesis matrix and its own slice of
# one coefficient block, so that macro does not apply.

"""
    CircularHarmonicLeafScratch(synthesis, mirror, coefficients, samples, leaves, ncomponents)

The per-`(level, chunk)` working set for a circular-harmonic leaf multiply.

As with `SphericalHarmonicLeafScratch`, some fields are shared and others are mutable scratch:

  - `synthesis` and `mirror` depend only on the level's grid and order. Leaves share them by
    reference. `mirror` is `S` with the `(-1)^m` antipodal parity folded in (see
    [`mirrorsynthesis`](@ref MLFMA.Harmonics.CircularHarmonics.mirrorsynthesis)), which is what the Galerkin
    receive side uses instead of a mirror-index permutation.
  - `coefficients` and `samples` are genuine scratch, sized for the largest batch this chunk will
    multiply.

No quadrature weights are stored here. MLFMA's sampled leaf scatter transposes its collection
without weights at the leaf; angular quadrature is applied in translation. A receive kernel
written as `Sᴴ W` would satisfy a weighted pairing but would not match the sampled path.
"""
struct CircularHarmonicLeafScratch{C<:Complex}
    synthesis::Matrix{C}
    mirror::Matrix{C}
    coefficients::Matrix{C}
    samples::Matrix{C}
    # Which leaves the batch currently in `coefficients`/`samples` belongs to, so the write-out can be
    # indexed. Filling a fresh vector per chunk per matvec would allocate on the hot path.
    leaves::Vector{Int}
    ncomponents::Int
end

"""
    CircularHarmonicLeafScratch(t, ncomponents; ngroups=1)

Build the scratch for a level whose transform is `t`, for moments of `ncomponents` components, sized
to multiply `ngroups` leaves at once.

Takes no sampling: unlike the spherical case there is no frame to tabulate, so the transform alone
determines everything the level shares.
"""
function CircularHarmonicLeafScratch(
    t::CircularHarmonics.CircularHarmonicTransform{T,C},
    ncomponents::Integer;
    ngroups::Integer=1,
) where {T,C}
    ngroups >= 1 || throw(ArgumentError("ngroups must be at least 1, got $ngroups"))
    ncomponents >= 1 ||
        throw(ArgumentError("ncomponents must be at least 1, got $ncomponents"))
    D = Int(ncomponents)
    nd = size(t.synthesis, 1)
    return CircularHarmonicLeafScratch(
        t.synthesis,
        CircularHarmonics.mirrorsynthesis(t),
        Matrix{C}(undef, CircularHarmonics.numcoefficients(t), D * ngroups),
        Matrix{C}(undef, nd, D * ngroups),
        Vector{Int}(undef, ngroups),
        D,
    )
end

ncomponents(s::CircularHarmonicLeafScratch) = s.ncomponents
maxgroups(s::CircularHarmonicLeafScratch) = size(s.samples, 2) ÷ s.ncomponents
numdirections(s::CircularHarmonicLeafScratch) = size(s.samples, 1)

"""
    CircularHarmonicRadiationCollection(storage)

A leaf's radiation patterns as circular-harmonic coefficients.

`storage` is a [`CircularHarmonicStorage`](@ref MLFMA.Harmonics.CircularHarmonics.CircularHarmonicStorage)
whose columns are this leaf's basis functions in the order `H2Trees.values(tree, leaf)` gives, the
same order a sampled collection's columns are in, so the two are interchangeable from the traversal's
point of view.

`D` is the storage's own component count, NOT a fixed number. The spherical collection declares 2
while storing 3, because its Cartesian representation is internal; here the stored components are the
produced ones, so a `ScalarFarField` map gets `D = 1` and the 2-D hypersingular's
`PlanarVectorScalarFarField` gets `D = 3` from the same type.
"""
struct CircularHarmonicRadiationCollection{D,T<:Complex,S} <:
       AbstractRadiationFieldCollection{D,T}
    storage::S
end

function CircularHarmonicRadiationCollection(
    storage::CircularHarmonics.CircularHarmonicStorage{C}
) where {C<:Complex}
    D = CircularHarmonics.ncomponents(storage)
    return CircularHarmonicRadiationCollection{D,C,typeof(storage)}(storage)
end

fielddim(::Type{<:CircularHarmonicRadiationCollection{D}}) where {D} = D

storage(c::CircularHarmonicRadiationCollection) = c.storage
order(c::CircularHarmonicRadiationCollection) = CircularHarmonics.order(c.storage)
numbasisfunctions(c::CircularHarmonicRadiationCollection) =
    CircularHarmonics.numbasisfunctions(c.storage)
Base.size(c::CircularHarmonicRadiationCollection{D}) where {D} =
    (CircularHarmonics.numcoefficients(c.storage) * D, numbasisfunctions(c))

# --- the operator's receive coupling ---------------------------------------------------------------
#
# The mirror identity `Π_mirror · S = M` is geometric: the antipodal map composed with
# `mirrorrule`. The sampled receive side does not apply `mirrorrule`; it applies
# `receivecouplingrule(op, F)`, which depends on the operator. Treating the two as interchangeable
# gave 44 % relative error on a 3-D MFIE map despite matching array shapes.
#
# Writing `Perm` for the pure permutation and `R_op` for the operator's component map:
#
#     Π_op · S = R_op · mirrorrule · M = K · M,      K := R_op · mirrorrule
#
# A receive collection therefore applies mirrored synthesis followed by a `D × D` component map, constant
# over directions and over leaves.
#
# Both supported 2-D Galerkin operators give `K = I`. The single layer's rule is `(1,1)→+1`,
# matching `ScalarFarField`'s `mirrorrule`, also `(1,1)→+1`. The hypersingular rule and mirror
# rule are identity on all three components. Direct quadrature checks gave
# 6.6e-15/9.4e-15/3.2e-14 componentwise agreement; see `representation.jl`.
# The 2-D double layers have no receive rule and support only Petrov assembly. Building a mirrored
# collection for them raises through `receivecouplingrule`'s fallback.
#
# The implementation remains generic over `D`. `K = I` holds for these two operators, not for
# every possible operator on the representation.

"""
    circularreceivecoupling(op, F) -> SMatrix{D,D}

The `D × D` component map a compressed 2-D receive collection applies after its mirrored synthesis:
`receivecouplingrule(op, F)` composed with `mirrorrule(F)`, both as matrices.

The map is derived from both rule tables. A rule that addresses components outside `F`'s layout
raises an error during construction.
"""
function circularreceivecoupling(op, ::Type{F}) where {F<:AbstractFarField}
    D = fielddim(F)
    return _circularrulematrix(receivecouplingrule(op, F), op, D, F) *
           _circularrulematrix(mirrorrule(F), op, D, F)
end

function _circularrulematrix(rule, op, D::Int, ::Type{F}) where {F}
    K = zeros(ComplexF64, D, D)
    for mapping in rule
        d, s = destcomponent(mapping), srccomponent(mapping)
        (d in 1:D && s in 1:D) || error(
            "compressed circular leaf storage supports far fields of $D component(s), but " *
            "$(typeof(op))'s rule for $(nameof(F)) maps component $s to $d. Use " *
            "`SampledLeafStorage()` for this operator, or extend the compressed receive kernels " *
            "to its representation.",
        )
        K[d, s] += coefficient(mapping)
    end
    return K
end

# `K == I` is what both 2-D Galerkin operators give, and it is worth branching on so that the common
# path is a plain copy with no per-direction arithmetic at all.
_isplaincircularcoupling(K) = K == LinearAlgebra.I

"""
    CircularHarmonicReceiveCollection(radiation, coupling)

The Galerkin receive collection for a leaf whose radiation patterns are stored as circular-harmonic
coefficients.

Holds a reference to the radiation collection and the operator's coupling. The antipodal mirror
is a per-mode sign folded into the level's `scratch.mirror`, so unlike the sampled
[`LazyReceiveFieldCollection`](@ref MLFMA.FarFields.LazyReceiveFieldCollection) there is no index array and no mirror pass.
"""
struct CircularHarmonicReceiveCollection{D,T<:Complex,R,K} <:
       AbstractReceiveFieldCollection{D,T}
    radiation::R
    coupling::K
end

function CircularHarmonicReceiveCollection(
    radiation::CircularHarmonicRadiationCollection{D,C}, coupling::AbstractMatrix
) where {D,C<:Complex}
    size(coupling) == (D, D) || throw(
        DimensionMismatch(
            "the receive coupling is a $D x $D map on this far field's components, got " *
            "$(size(coupling))",
        ),
    )
    K = SMatrix{D,D,C}(coupling)
    return CircularHarmonicReceiveCollection{D,C,typeof(radiation),typeof(K)}(radiation, K)
end

"""
    CircularHarmonicReceiveCollection(radiation, op, F)

The same, with the coupling derived from the operator. See [`circularreceivecoupling`](@ref).
"""
CircularHarmonicReceiveCollection(radiation, op, ::Type{F}) where {F<:AbstractFarField} =
    CircularHarmonicReceiveCollection(radiation, circularreceivecoupling(op, F))

fielddim(::Type{<:CircularHarmonicReceiveCollection{D}}) where {D} = D
radiationcollection(c::CircularHarmonicReceiveCollection) = c.radiation
coupling(c::CircularHarmonicReceiveCollection) = c.coupling
storage(c::CircularHarmonicReceiveCollection) = storage(c.radiation)
order(c::CircularHarmonicReceiveCollection) = order(c.radiation)
numbasisfunctions(c::CircularHarmonicReceiveCollection) = numbasisfunctions(c.radiation)

"""
    CircularHarmonicAssembledReceiveCollection(radiation)

A leaf's assembled receive patterns as circular-harmonic coefficients. This is the Petrov test
side under [`CircularHarmonicLeafStorage`](@ref MLFMA.Assembly.CircularHarmonicLeafStorage).

Wraps a [`CircularHarmonicRadiationCollection`](@ref) whose coefficients were analysed from
`receiveoperator(op)`'s patterns. The two types store patterns from different sides but use the
same kernels, so this wrapper forwards to the radiation collection.

The separate `<: AbstractReceiveFieldCollection` type identifies the side for map constructors,
scratch pools, and tests. See `test/Maps/circularharmonicstorage.jl`.

Unlike [`CircularHarmonicReceiveCollection`](@ref), the Galerkin wrapper, this collection owns its
receive coefficients. It applies neither antipodal parity nor the operator's receive coupling.
It needs no `mirrorrule` or `receivecouplingrule`, so it supports the Petrov-only 2-D double layers.
"""
struct CircularHarmonicAssembledReceiveCollection{D,T<:Complex,R} <:
       AbstractReceiveFieldCollection{D,T}
    radiation::R
end

CircularHarmonicAssembledReceiveCollection(
    radiation::CircularHarmonicRadiationCollection{D,C}
) where {D,C<:Complex} =
    CircularHarmonicAssembledReceiveCollection{D,C,typeof(radiation)}(radiation)

fielddim(::Type{<:CircularHarmonicAssembledReceiveCollection{D}}) where {D} = D
radiationcollection(c::CircularHarmonicAssembledReceiveCollection) = c.radiation
storage(c::CircularHarmonicAssembledReceiveCollection) = storage(c.radiation)
order(c::CircularHarmonicAssembledReceiveCollection) = order(c.radiation)
numbasisfunctions(c::CircularHarmonicAssembledReceiveCollection) =
    numbasisfunctions(c.radiation)
Base.size(c::CircularHarmonicAssembledReceiveCollection) = size(c.radiation)

"""
    CircularHarmonicAssembledCollection

Either compressed 2-D collection whose coefficients were assembled and analysed,
[`CircularHarmonicRadiationCollection`](@ref) or
[`CircularHarmonicAssembledReceiveCollection`](@ref).

Both types use the same scratch, order accessors, and kernels after analysis.
The union excludes [`CircularHarmonicReceiveCollection`](@ref), which reuses radiation coefficients
and needs different kernels.
"""
const CircularHarmonicAssembledCollection = Union{
    CircularHarmonicRadiationCollection,CircularHarmonicAssembledReceiveCollection
}

# --- the six orientations -------------------------------------------------------------------------
#
# H2Factory realizes the transpose and the adjoint by swapping the sides, so
# each collection needs both arities. With `R = S C` the radiation collection and `Rec = K M C` the
# receive one, no `P`, because there is no frame projection in 2-D:
#
#                    aggregate                    scatter
#     forward        R x   = S C x                α Recᵀ m = α Cᵀ Mᵀ Kᵀ m
#     transpose      Rec x = K M C x              α Rᵀ m   = α Cᵀ Sᵀ m
#     adjoint        conj(Rec) x                  α Rᴴ m   = α Cᴴ Sᴴ m
#
# `_cheaggregate!` and `_chescatter!` cover these cases. The orientation picks `S` or `M`, and the
# scatter's `mode` picks transpose or adjoint. The adjoint's `α` arrives already conjugated from
# `H2Factory.α(core, ::isAdjoint)`; its kernels live in `assembly/cores/adjointleafmul.jl`.

"""
    H2Factory.applyleafmul!(scratch, moment, collection, x)

Aggregate a leaf: `moment .= S · (C x)`.

First, `A = C x` applies one GEMV to the coefficient matrix, with `(2M+1)·D` rows rather than
`Nφ·D`. Synthesis then writes into the moment's component arrays. It needs no frame projection
or intermediate array of samples.
"""
function H2Factory.applyleafmul!(
    scratch::CircularHarmonicLeafScratch,
    moment::AbstractFarField,
    collection::CircularHarmonicAssembledCollection,
    x::AbstractVector,
)
    return _cheaggregate!(moment, scratch, storage(collection), x, scratch.synthesis)
end

# Reject multiplication without scratch, as with `LazyReceiveFieldCollection` and the spherical
# collections. Scratch must be reused, not allocated per matvec.
function H2Factory.applyleafmul!(
    ::Any,
    ::AbstractFarField,
    collection::CircularHarmonicAssembledCollection,
    ::AbstractVector,
)
    return error(
        "a $(typeof(collection)) needs a `CircularHarmonicLeafScratch` to multiply through: its " *
        "reconstruction goes via a coefficient block that must be neither allocated per matvec " *
        "nor stored per leaf. The core must implement `H2Factory.buildleafchannels` returning " *
        "that scratch (see `MLFMA.leafscratchchannel` for the sampled analogue).",
    )
end

function LinearAlgebra.mul!(
    ::AbstractFarField, collection::CircularHarmonicAssembledCollection, ::AbstractVector
)
    return error(
        "a $(typeof(collection)) has no plain `mul!`: it must be applied through " *
        "`H2Factory.applyleafmul!` with a `CircularHarmonicLeafScratch`. See this file's header.",
    )
end

# Transpose aggregation, test side: `moment = Rec x = K M C x`.
function H2Factory.applyleafmul!(
    scratch::CircularHarmonicLeafScratch,
    moment::AbstractFarField,
    collection::CircularHarmonicReceiveCollection,
    x::AbstractVector,
)
    _cheaggregate!(moment, scratch, storage(collection), x, scratch.mirror)
    return _applycircularcoupling!(moment, coupling(collection))
end

"""
    H2Factory.applyleafmul!(scratch, y, collection, moment, α, β)

Scatter an incoming moment onto a leaf's degrees of freedom: `y .= α · Cᵀ Mᵀ Kᵀ moment + β · y`.

The transpose of the receive collection's aggregation, which is what `Rec = K M C` means concretely.
`β` arrives as `false` from the traversal's overwriting leaf scatter, which is why the accumulation
short-circuits on it rather than multiplying by zero.
"""
function H2Factory.applyleafmul!(
    scratch::CircularHarmonicLeafScratch,
    y::AbstractVector,
    collection::CircularHarmonicReceiveCollection,
    moment::AbstractFarField,
    α,
    β::Number,
)
    return _chescatter!(
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

# Transpose scatter, trial side: `y = α Rᵀ moment + β y`.
function H2Factory.applyleafmul!(
    scratch::CircularHarmonicLeafScratch,
    y::AbstractVector,
    collection::CircularHarmonicAssembledCollection,
    moment::AbstractFarField,
    α,
    β::Number,
)
    return _chescatter!(
        y, scratch, storage(collection), moment, scratch.synthesis, α, β, _Transposed()
    )
end

function H2Factory.applyleafmul!(
    ::Any,
    ::AbstractVector,
    collection::Union{
        CircularHarmonicAssembledCollection,CircularHarmonicReceiveCollection
    },
    ::AbstractFarField,
    ::Any,
    ::Number,
)
    return error(
        "a $(typeof(collection)) needs a `CircularHarmonicLeafScratch` to scatter through. See " *
        "this file's header.",
    )
end

# --- the two shared kernels -----------------------------------------------------------------------

"""
    _cheaggregate!(moment, scratch, storage, x, Y) -> moment

`moment = Y C x`, the leaf aggregation for whichever side owns synthesis matrix `Y`.

`Y` is `scratch.synthesis` on the radiation side and `scratch.mirror` on the Galerkin receive side.
The receive caller applies the operator's component coupling after this kernel.

Writes one component at a time straight into the moment, with no `(Nφ × D)` intermediate: the
spherical version needs one because a frame projection has to read all three Cartesian components
together, and there is nothing to project here.
"""
function _cheaggregate!(moment, scratch, storage, x, Y)
    D = CircularHarmonics.ncomponents(storage)
    nc = CircularHarmonics.numcoefficients(storage)
    @views A = scratch.coefficients[:, 1:D]
    CircularHarmonics.aggregate!(vec(A), storage, x)
    for d in 1:D
        @views LinearAlgebra.mul!(H2Factory.wrapvector(moment, d), Y, A[:, d])
    end
    return moment
end

"""
    _chescatter!(y, scratch, storage, moment, Y, α, β, mode, K=I) -> y

`y = α (K Y C)ᵀ moment + β y`, or the adjoint of the same when `mode` is `_Adjointed()`.

Three stages mirroring [`_cheaggregate!`](@ref) in reverse: the coupling transpose, the reversed
synthesis, and the reversed coefficient multiply. Unlike aggregation, this stages into
`scratch.samples`, because the reversed synthesis is one GEMM over all `D` components and needs them
contiguous, a moment's components are separate arrays.
"""
function _chescatter!(y, scratch, storage, moment, Y, α, β, mode, K=LinearAlgebra.I)
    D = CircularHarmonics.ncomponents(storage)
    @views G = scratch.samples[:, 1:D]
    _gathermoment!(G, moment, K)
    scalar = _foldalpha!(G, α)
    @views A = scratch.coefficients[:, 1:D]
    LinearAlgebra.mul!(A, _reverse(mode, Y), G)
    coefficients = CircularHarmonics.coefficientmatrix(storage)
    LinearAlgebra.mul!(
        y, _reverse(mode, coefficients), vec(A), scalar, iszero(β) ? false : β
    )
    return y
end

# `α` may be a scalar or a vector. H2Factory's leaf disaggregation passes `MLFMA.α(operator)`;
# `HH2DHyperSingularFDBIO` supplies three entries (`[alpha, alpha, beta]`). The collection's
# scatter methods must accept both forms to avoid falling through to the generic matrix kernel.
#
# A vector `α` scales each component. It is folded into `G` because the
# final `mul!` flattens all `D` components into one vector and BLAS has no per-block scaling. Folding
# is exact and commutes with what follows: `α[d]` multiplies column `d`, `Yᵀ` acts within a column,
# and `Cᵀ`'s block `d` reads exactly that column, so `Σ_d α[d] Cᵀ_d Yᵀ G_d` is what comes out either
# way.
#
# Apply `α` after `Kᵀ`, matching the sampled path, where
# `receivefields!` applies the mirror and the coupling and the following `mul!` then takes `α`.
#
# Do not conjugate `α` here, even in the adjoint orientation. It arrives already conjugated from
# `H2Factory.α(core, ::isAdjoint)`.
_foldalpha!(G, α::Number) = α

function _foldalpha!(G, α::AbstractVector)
    length(α) == size(G, 2) || throw(
        DimensionMismatch(
            "α has $(length(α)) entries but the moment has $(size(G, 2)) components"
        ),
    )
    @inbounds for d in axes(G, 2)
        αd = α[d]
        isone(αd) && continue
        for q in axes(G, 1)
            G[q, d] *= αd
        end
    end
    return true
end

# `Kᵀ moment`, staged into the samples block. Two methods rather than a branch inside the loop, so the
# identity case, used by both supported 2-D Galerkin operators, compiles to a plain copy.
_gathermoment!(G, moment, K) =
    if _isplaincircularcoupling(K)
        _gathermoment!(G, moment)
    else
        _coupledgathermoment!(G, moment, K)
    end

function _gathermoment!(G, moment)
    @inbounds for d in axes(G, 2)
        component = H2Factory.wrapvector(moment, d)
        for q in axes(G, 1)
            G[q, d] = component[q]
        end
    end
    return G
end

"""
    _coupledgathermoment!(G, moment, K) -> G

Gather a moment's components into `G` while mixing them by `K`: `G[:, d] = Σ_e K[e, d] * moment[e]`,
i.e. `Kᵀ` applied across components.

The coupled counterpart of `_gathermoment!`, which gathers without mixing. Each output is
accumulated in a scalar before it is written to `G`. Inputs are read from `moment`; the caller
must keep that storage separate from `G` when the coupling mixes components.
"""
function _coupledgathermoment!(G, moment, K)
    D = size(G, 2)
    @inbounds for q in axes(G, 1)
        for d in 1:D
            acc = zero(eltype(G))
            for e in 1:D
                acc += K[e, d] * H2Factory.wrapvector(moment, e)[q]
            end
            G[q, d] = acc
        end
    end
    return G
end

"""
    _applycircularcoupling!(moment, K) -> moment

Mix a moment's components in place by `K`. The aggregation-side counterpart of the fold in
[`_coupledgathermoment!`](@ref MLFMA.FarFields._coupledgathermoment!).

Each direction's components are captured in a tuple before any is written, so component
permutations are safe. This avoids a separate array of samples; tuple allocation depends on
compiler specialization.
"""
function _applycircularcoupling!(moment, K)
    _isplaincircularcoupling(K) && return moment
    D = size(K, 1)
    @inbounds for q in eachindex(H2Factory.wrapvector(moment, 1))
        source = ntuple(e -> H2Factory.wrapvector(moment, e)[q], D)
        for d in 1:D
            acc = zero(eltype(K))
            for e in 1:D
                acc += K[d, e] * source[e]
            end
            H2Factory.wrapvector(moment, d)[q] = acc
        end
    end
    return moment
end

# `A = C conj(x)`, through the real/imaginary split rather than a conjugated temporary, the leaf
# multiply runs once per leaf per matvec and must not allocate. Identical in shape to the spherical
# file's `_aggregateconjugated!` and to H2Factory's `conjmul!`, and used for the same reason: there is
# no conjugate-without-transpose BLAS call. Only adjoint aggregation needs it, via
# `conj(C) x == conj(C conj(x))`.
function _aggregatecircularconjugated!(
    A, coefficients::AbstractMatrix, x::DenseVector{<:Complex}
)
    parts = reinterpret(reshape, real(eltype(x)), x)
    LinearAlgebra.mul!(A, coefficients, view(parts, 1, :))
    LinearAlgebra.mul!(
        A, coefficients, view(parts, 2, :), -im * one(eltype(A)), one(eltype(A))
    )
    return A
end

_aggregatecircularconjugated!(A, coefficients::AbstractMatrix, x::AbstractVector{<:Real}) =
    LinearAlgebra.mul!(A, coefficients, x)
_aggregatecircularconjugated!(A, coefficients::AbstractMatrix, x::AbstractVector) =
    LinearAlgebra.mul!(A, coefficients, conj.(x))

# --- the batched chunk multiply -------------------------------------------------------------------
#
# One GEMM synthesises the chunk instead of one call per leaf. Although `S` is only 8.3 KB,
# reducing call overhead measured 0.28x-0.72x runtime. See this file's header.
#
# There is no threshold like `SHEBATCHTHRESHOLD`. Batching beat the per-leaf form from two leaves
# upward and tied at one in the measurements. Only orientations that cannot batch use the per-node path.

function H2Factory.aggregateleafchunk!(
    A,
    tree,
    chunknodes,
    moment,
    momentcollection,
    leafmul!,
    x,
    level,
    scratch::CircularHarmonicLeafScratch,
)
    if !_circularbatchable(tree, chunknodes, momentcollection, leafmul!)
        for node in chunknodes
            H2Trees.isleaf(tree, node) || continue
            H2Factory.aggregateleaf!(
                A, tree, node, moment, momentcollection, leafmul!, x, scratch
            )
        end
        return nothing
    end

    capacity = maxgroups(scratch)
    filled = 0
    for node in chunknodes
        H2Trees.isleaf(tree, node) || continue
        filled += 1
        scratch.leaves[filled] = node
        _gathercircularcoefficients!(A, tree, node, momentcollection, x, scratch, filled)
        # Sub-batch when the chunk outruns the scratch, so `S` is streamed once per `capacity` groups
        # rather than once per group.
        if filled == capacity
            _flushcircularbatch!(moment, scratch, filled)
            filled = 0
        end
    end
    filled > 0 && _flushcircularbatch!(moment, scratch, filled)
    return nothing
end

# The batch implements unconjugated `S C x` for assembled collections. H2Factory swaps sides in
# transpose and adjoint traversals. A Galerkin receive collection needs `M` for `S` and the
# operator's coupling; the adjoint also uses a `ConjugatedCollectionMul`. Those paths must use
# their own kernels rather than this batch.
#
# The batch does not otherwise call `leafmul!`, so this guard is its only check of the multiply mode.
_circularbatchable(tree, chunknodes, momentcollection, leafmul!) = false

function _circularbatchable(
    tree, chunknodes, momentcollection, ::H2Factory.LinAlgebraMulFunctor
)
    for node in chunknodes
        H2Trees.isleaf(tree, node) || continue
        return _circularbatchableradiation(momentcollection(node))
    end
    return false
end

_circularbatchableradiation(::Any) = false
_circularbatchableradiation(::CircularHarmonicAssembledCollection) = true

function _gathercircularcoefficients!(A, tree, node, momentcollection, x, scratch, g)
    collection = momentcollection(node)
    # Through the accessor, not the field: an assembled receive collection batches through this same
    # path and holds its storage one level down.
    store = storage(collection)
    D = ncomponents(scratch)
    CircularHarmonics.numcoefficients(store) == size(scratch.coefficients, 1) || throw(
        DimensionMismatch(
            "leaf $node expands to order $(order(collection)), but this level's scratch is " *
            "sized for $(size(scratch.coefficients, 1)) coefficients. One order per LEVEL is " *
            "required, a ragged batch has no single synthesis matrix.",
        ),
    )
    CircularHarmonics.ncomponents(store) == D || throw(
        DimensionMismatch(
            "leaf $node stores $(CircularHarmonics.ncomponents(store)) " *
            "components, but this level's scratch is sized for $D",
        ),
    )
    buffer = H2Factory.aggregationvectorbuffer(A, node)
    buffer .= view(x, H2Trees.values(tree, node))
    @views CircularHarmonics.aggregate!(
        vec(scratch.coefficients[:, (D * (g - 1) + 1):(D * g)]), store, buffer
    )
    return nothing
end

function _flushcircularbatch!(moment, scratch, filled)
    D = ncomponents(scratch)
    @views LinearAlgebra.mul!(
        scratch.samples[:, 1:(D * filled)],
        scratch.synthesis,
        scratch.coefficients[:, 1:(D * filled)],
    )
    for g in 1:filled
        _writeinto!(moment(scratch.leaves[g]), scratch.samples, D, g)
    end
    return nothing
end

# Write-out copies the synthesised components into the moment. No frame projection is needed.
#
# This copy runs serially rather than using the scheduler for `H2Factory.LeafMulOp`.
# Spherical projection performs 9 multiply-adds per direction over 1568 directions; this copies
# only `Nφ · D` = 28 or 84 complex numbers per leaf. Spawning tasks is unlikely to help at that size.
# Spherical measurements gave a 4.0x speedup at 35 groups, but threading one group cost 0.824 ms
# against 0.122 ms serial. The circular copy has about two orders of magnitude less work.
#
# `LeafMulOp` permits scheduling inside a batched multiply; it does not require every backend
# to consult a scheduler.
function _writeinto!(moment, samples, D, g)
    @inbounds for d in 1:D
        component = H2Factory.wrapvector(moment, d)
        column = D * (g - 1) + d
        for q in eachindex(component)
            component[q] = samples[q, column]
        end
    end
    return moment
end
