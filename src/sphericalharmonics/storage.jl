# Spherical-harmonic storage for a group's basis-function patterns, and the coefficient aggregation
# that replaces the sampled leaf multiply.
#
# Aggregate coefficients before synthesising the group's field:
#
#     A_g = Σ_j I_j a_j          coefficient aggregation, ONE GEMV per group
#     F_g = S A_g                synthesis, ONE GEMM for the WHOLE LEVEL
#
# rather than `F_g = Σ_j I_j F_j` over sampled patterns. Never reconstruct an individual basis
# pattern -- that would spend a synthesis per basis function instead of per group and throw the
# advantage away.
#
# SHE does ~1.76x more arithmetic than the sampled multiply, but the sampled multiply is
# memory-bound. It streams a `(2NΩ x N_j)` matrix -- 4.5 MB per group at the λ/4 leaf --
# and does about 0.5 flop per byte with it. The coefficient block is 10.4x smaller and `S` is shared
# by every group at the level, so it stays resident: ~8.7 flop/byte. Runtime follows BYTES here, so
# compressing the storage compresses the time. Measured 0.57x (serial BLAS) / 0.22x (threaded) against
# the sampled path -- see `benchmark/interpolation/shesynthesis.jl`.
#
# Batch synthesis to retain the measured speedup. `S` depends only on the level's sampling
# and `P`, never on the group. One `S * [A_1 … A_k]` keeps it resident; k skinny per-group GEMMs
# re-stream it and measure 1.63x SLOWER than the sampled path -- i.e. unbatched SHE is a regression on
# both axes. `aggregatelevel!` below exists so that the batched form is the easy thing to call.
#
# LAYOUT: `(lm, component)` flattened lm-FASTEST, basis functions along columns. The alternative
# (component-fastest) was benchmarked and is within noise -- 1.001-1.006x over three configurations --
# because the permute lands on a `(P+1)² x D` block that is negligible beside the GEMV and the GEMM.
# So this is chosen for CLARITY, not speed: lm-fastest makes `reshape(A_g, ncoeff, D)` free and is
# exactly the shape synthesis consumes.

"""
    SphericalHarmonicStorage(order, center, coefficients)

One group's basis-function far-field patterns, stored as spherical-harmonic coefficients instead of
samples.

`coefficients` is `((order+1)² * ncomponents) × nbasisfunctions`, with the row index running over
`(l, m)` fastest and the component slowest -- so column `j` reshapes to the `(ncoeff, ncomponents)`
matrix of basis function `j`'s expansion at no cost.

`center` is the expansion center the coefficients are referenced to. Patterns about a different center
differ by `exp(-iκ k̂·Δc)`, which is not a coefficient-space operation, so a group's coefficients can
only be summed if they share this center -- see [`aggregate!`](@ref).
"""
struct SphericalHarmonicStorage{C<:Complex,T<:Real}
    order::Int
    ncomponents::Int
    center::SVector{3,T}
    coefficients::Matrix{C}

    function SphericalHarmonicStorage(
        order::Integer, center::SVector{3,T}, coefficients::Matrix{C}
    ) where {T<:Real,C<:Complex}
        nc = numcoefficients(order)
        rows = size(coefficients, 1)
        rows % nc == 0 || throw(
            DimensionMismatch(
                "coefficients has $rows rows, which is not a multiple of " *
                "$(nc) = (order+1)² for order $order",
            ),
        )
        return new{C,T}(Int(order), rows ÷ nc, center, coefficients)
    end
end

order(s::SphericalHarmonicStorage) = s.order
ncomponents(s::SphericalHarmonicStorage) = s.ncomponents
numcoefficients(s::SphericalHarmonicStorage) = numcoefficients(s.order)
numbasisfunctions(s::SphericalHarmonicStorage) = size(s.coefficients, 2)

"""
    coefficientmatrix(storage)

The raw `(ncoeff * ncomponents) × nbasisfunctions` matrix.

Exposed so the receive side can apply `Cᴴ` directly. [`testcoefficients!`](@ref) is the same product
with a result vector supplied; this is for callers that already hold one and want `mul!`'s `α`/`β`
arithmetic, which the traversal's leaf scatter needs.
"""
coefficientmatrix(s::SphericalHarmonicStorage) = s.coefficients

"""
    SphericalHarmonicStorage(t, patterns, center) -> storage

Analyse sampled patterns into coefficients. `patterns` is a vector, one entry per basis function, each
`ndirections × ncomponents` on `t`'s grid.

`t` must be a setup transform built on [`gausslegendregrid`](@ref), not on an MLFMA leaf
sampling. The coefficients here have to be the field's actual harmonic content, and the leaf grid's
quadrature is not accurate enough for that (it needs ~4x the θ nodes; see `transform.jl`'s header).
"""
function SphericalHarmonicStorage(
    t::SphericalHarmonicTransform{T,C}, patterns::AbstractVector{<:AbstractMatrix}, center
) where {T,C}
    isempty(patterns) && throw(ArgumentError("need at least one pattern"))
    D = size(first(patterns), 2)
    nc = numcoefficients(t)
    coefficients = Matrix{C}(undef, nc * D, length(patterns))
    for (j, F) in enumerate(patterns)
        size(F, 2) == D ||
            throw(DimensionMismatch("pattern $j has $(size(F, 2)) components, expected $D"))
        @views coefficients[:, j] .= vec(analyze(t, F))
    end
    return SphericalHarmonicStorage(order(t), SVector{3}(center), coefficients)
end

"""
    aggregate!(A, storage, x) -> A

`A .= storage.coefficients * x` -- the group's aggregated expansion, `A_g = Σ_j I_j a_j`.

`A` is a vector of length `numcoefficients(storage) * ncomponents(storage)`; `reshape(A, ncoeff, D)`
is then the matrix [`synthesize`](@ref) wants.

Every column must share the storage's `center` for this sum to be meaningful.
The caller is responsible for that agreement.
"""
function aggregate!(A::AbstractVector, storage::SphericalHarmonicStorage, x::AbstractVector)
    return LinearAlgebra.mul!(A, storage.coefficients, x)
end

"""
    aggregatelevel!(F, A, t, storages, xs) -> F

Aggregate every group of a level and synthesise them all in ONE GEMM.

`storages` and `xs` are matched vectors over the level's groups; `A` is scratch of size
`(ncoeff, D * ngroups)` and `F` receives `(ndirections, D * ngroups)`, group `g` occupying columns
`((g-1)D+1):(gD)` in both.

The single trailing `synthesize!` is the whole reason this function exists rather than a loop calling
`synthesize` per group: `S` is shared across the level, and re-streaming it per group is measurably
worse than the sampled path it is meant to replace. See this file's header.
"""
function aggregatelevel!(
    F::AbstractMatrix,
    A::AbstractMatrix,
    t::SphericalHarmonicTransform,
    storages::AbstractVector{<:SphericalHarmonicStorage},
    xs::AbstractVector{<:AbstractVector},
)
    length(storages) == length(xs) || throw(
        DimensionMismatch(
            "$(length(storages)) storages but $(length(xs)) coefficient vectors"
        ),
    )
    isempty(storages) && return F
    nc = numcoefficients(t)
    D = ncomponents(first(storages))
    size(A) == (nc, D * length(storages)) ||
        throw(DimensionMismatch("A is $(size(A)), expected $((nc, D * length(storages)))"))
    for (g, (storage, x)) in enumerate(zip(storages, xs))
        order(storage) == order(t) || throw(
            ArgumentError(
                "storage $g has order $(order(storage)), transform has $(order(t))"
            ),
        )
        ncomponents(storage) == D || throw(
            DimensionMismatch(
                "storage $g has $(ncomponents(storage)) components, expected $D"
            ),
        )
        @views aggregate!(vec(A[:, ((g - 1) * D + 1):(g * D)]), storage, x)
    end
    return synthesize!(F, t, A)
end

# --- the receive side -----------------------------------------------------------------------------
#
# The two operations form an adjoint pair. With `C` the storage's coefficient matrix,
# `S` the synthesis matrix and `W` the quadrature weights:
#
#     forward   x -> A = C x        -> F = S A
#     reverse   G -> B = Sᴴ W G     -> V = Cᴴ B
#
#     ⟨F, G⟩_W = (S C x)ᴴ W G = xᴴ Cᴴ Sᴴ W G = ⟨x, V⟩
#
# so the reverse chain is the exact adjoint of the forward one, in exact arithmetic, on ANY grid --
# no quadrature accuracy is required for it, only that the SAME `W` appears in both. This is Eibert's
# Eq. (13) and it is a reordering of a finite sum, not an approximated integral. In particular the
# product `G·conj(Y_lm)` has bandwidth `L + P` which the leaf grid does not resolve; if the identity
# needed that integral to be exact the scheme would not work at all.
#
# `MLFMA.InterpolationDiagnostics.adjointpairingerror` is NOT reused for this: it pairs an
# interpolator with an anterpolator across two SAMPLINGS, which is a different operator shape from a
# coefficient-space-to-sample-space map over one sampling. The property is the same idea; the test
# writes it out directly rather than bending the signature.

"""
    analyzelevel!(B, t, G) -> B

Project a whole level's incoming fields onto the harmonics: `B = Sᴴ (w .* G)`.

`G` is `(ndirections, D * ngroups)` as [`aggregatelevel!`](@ref) produces, and `B` receives
`(ncoeff, D * ngroups)` in the same column layout. One GEMM for the level, for the same reason the
forward direction batches: `S` is shared by every group.
"""
function analyzelevel!(B::AbstractMatrix, t::SphericalHarmonicTransform, G::AbstractMatrix)
    size(G, 1) == numdirections(t) || throw(
        DimensionMismatch(
            "G has $(size(G, 1)) rows but the transform has $(numdirections(t)) directions",
        ),
    )
    size(B) == (numcoefficients(t), size(G, 2)) || throw(
        DimensionMismatch("B is $(size(B)), expected $((numcoefficients(t), size(G, 2)))"),
    )
    return LinearAlgebra.mul!(B, t.synthesis', t.weights .* G)
end

"""
    testcoefficients!(V, storage, B) -> V

`V .= storage.coefficients' * B` -- the tested values `V_j = Σ_lm conj(a_{j,lm}) B_lm`, Eibert's
Eq. (13).

The exact adjoint of [`aggregate!`](@ref) by construction: the same matrix, applied on the other side.
`B` is a vector of length `numcoefficients(storage) * ncomponents(storage)`, i.e. what
[`analyzelevel!`](@ref) writes for one group, flattened.
"""
function testcoefficients!(
    V::AbstractVector, storage::SphericalHarmonicStorage, B::AbstractVector
)
    return LinearAlgebra.mul!(V, storage.coefficients', B)
end

"""
    testlevel!(Vs, B, t, storages, G) -> Vs

The mirror of [`aggregatelevel!`](@ref): analyse the level's incoming fields in one GEMM, then test
each group's basis functions against its own coefficients.

`Vs` is a vector of per-group result vectors, `Vs[g]` of length `numbasisfunctions(storages[g])`.
"""
function testlevel!(
    Vs::AbstractVector{<:AbstractVector},
    B::AbstractMatrix,
    t::SphericalHarmonicTransform,
    storages::AbstractVector{<:SphericalHarmonicStorage},
    G::AbstractMatrix,
)
    length(Vs) == length(storages) || throw(
        DimensionMismatch("$(length(Vs)) result vectors but $(length(storages)) storages"),
    )
    isempty(storages) && return Vs
    D = ncomponents(first(storages))
    analyzelevel!(B, t, G)
    for (g, storage) in enumerate(storages)
        order(storage) == order(t) || throw(
            ArgumentError(
                "storage $g has order $(order(storage)), transform has $(order(t))"
            ),
        )
        @views testcoefficients!(Vs[g], storage, vec(B[:, ((g - 1) * D + 1):(g * D)]))
    end
    return Vs
end
