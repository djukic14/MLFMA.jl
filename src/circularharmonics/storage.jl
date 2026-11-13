# Circular-harmonic storage for a group's basis-function patterns, and the coefficient aggregation
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
# ------------------------------------------------------------------------------------------------
# Measured storage and runtime in 2-D.
#
# SMALLER: yes, always. On `meshcircle(4.0, 1/20)` at ε = 1e-5, λ/4 leaves, the leaf grid is `Nφ = 28`
# (L = 13) and the calibrated level order is `M = 9`, i.e. `2(L+1)/(2M+1) = 28/19 = 1.47x`. It ranges
# 1.33x-1.90x over the tolerances and box sizes in `benchmark/interpolation/checalibration.jl`.
#
# FASTER: essentially NEVER at that tolerance -- break-even at best, and only far above the leaf
# occupancy any ordinary 2-D fixture reaches. Measured (Nφ = 28, 3840 basis functions in the level,
# 1 BLAS thread, sampled `F_g = P_g x_g` per group against compressed `A_g = C_g x_g` per group plus
# one batched `S [A_1 … A_k]` per level), at BOTH the calibrated order and a looser one:
#
#     basis fns/leaf   leaves   leaf block    M = 9 (ε=1e-5)   M = 7 (ε≈1e-3)
#              8         480       3.5 KB          1.67x            1.44x
#             16         240       7.0 KB          1.34x            1.24x
#             32         120      14.0 KB          1.07x            0.97x
#             64          60      28.0 KB          0.98x            0.84x
#            256          15     112.0 KB          0.95x            0.80x
#           1024           3     448.0 KB          0.89x            0.76x
#
# A 503-function `meshcircle` over 60 leaves sits at ~8 basis functions per leaf, where this is 1.67x
# SLOWER. `LEAFSTORAGERECOM` therefore stays `SampledLeafStorage` for 2-D.
#
# Compression and runtime must use the maximum order for the level, not the median leaf order:
#
#   * The median per-leaf order (M = 7) with a `2M+2` count suggests 1.75x compression.
#     Batching requires one M per level. The calibrated maximum is 9 and the count is `2M+1`,
#     giving 28/19 = 1.47x.
#   * Using `ncoeff = 15` gives a crossover near ~32 basis functions per leaf. With 19 coefficients,
#     the crossover is near ~64 basis functions per leaf and the best runtime ratio is 0.89x,
#     rather than 0.81x.
#
# BLAS threading did not improve this result: in 3-D, SHE measured 0.57x
# serial and 0.22x threaded, because a batched GEMM over a large level parallelises well. Here at 8
# threads the low-occupancy case gets WORSE -- 3.25x at 8 basis functions per leaf, 2.56x at 16 --
# since the per-group products are far too small to amortise thread dispatch, and the high-occupancy
# case does not improve (0.98x at 64, 1.03x at 1024).
#
# The 3-D speedup depends on the sampled multiply
# being MEMORY-BOUND: it streams a `(2NΩ × N_j)` block -- 4.5 MB per group at the λ/4 leaf -- at about
# 0.5 flop/byte, so compressing the bytes compresses the time. A 3.5 KB 2-D leaf block is nowhere near
# that regime; it is in cache either way, and the compressed path's extra synthesis is then pure added
# arithmetic. The mechanism is the O(L) versus O(L²) asymmetry showing up in LEAF OCCUPANCY: a box on
# a CURVE holds `O(boxsize/h)` basis functions, a box on a SURFACE `O((boxsize/h)²)`, so a λ/4 box
# holds ~10 in 2-D and ~100+ in 3-D.
#
# There is a THIRD instance of the same asymmetry hiding in the compression itself, and it is why 2-D
# can never win as widely as 3-D no matter how the constants fall. On the sphere, a band-limited
# function needs `(L+1)²` coefficients while the product grid spends `8(L+1)²` samples -- so SHE wins
# 5.33x EVEN AT `P = L`, before any bandwidth margin. The circle has no such slack: `2(L+1)` samples
# against `2M+1` coefficients is `≈ 1` at `M = L`. Every bit of 2-D compression has to come from
# `M < L`, and `M` is only ~30 % below `L` at ε = 1e-5.
#
# Batch synthesis to retain the measured speedup. Unbatched SHE measured 1.63x slower than sampled
# in 3-D. `S` depends only on the level's sampling and `M`, never on the group.
# The table above is already measured with
# batching; without it the 2-D numbers would be worse still, and there is less margin here to lose.
# `aggregatelevel!` below exists so that the batched form is the easy thing to call.
#
# ------------------------------------------------------------------------------------------------
# LAYOUT: `(m, component)` flattened m-FASTEST, basis functions along columns -- structurally the same
# choice `SphericalHarmonicStorage` made and for the same reason, that `reshape(A_g, ncoeff, D)` is
# then free and is exactly the shape synthesis consumes. The component-fastest alternative was
# benchmarked on the 3-D side and came out within noise (1.001-1.006x), so this is chosen for clarity;
# nothing about the circle changes that argument.

"""
    CircularHarmonicStorage(order, center, coefficients)

One group's basis-function far-field patterns, stored as circular-harmonic coefficients instead of
samples.

`coefficients` is `((2·order+1) * ncomponents) × nbasisfunctions`, with the row index running over `m`
fastest -- centered, `-order:order` -- and the component slowest, so column `j` reshapes to the
`(ncoeff, ncomponents)` matrix of basis function `j`'s expansion at no cost.

`center` is the expansion center the coefficients are referenced to, an `SVector{2}`.
Patterns about a different center differ by `exp(-γ k̂·Δc)`, which is
not a coefficient-space operation, so a group's coefficients can only be summed if they share this
center -- see [`aggregate!`](@ref).
"""
struct CircularHarmonicStorage{C<:Complex,T<:Real}
    order::Int
    ncomponents::Int
    center::SVector{2,T}
    coefficients::Matrix{C}

    function CircularHarmonicStorage(
        order::Integer, center::SVector{2,T}, coefficients::Matrix{C}
    ) where {T<:Real,C<:Complex}
        nc = numcoefficients(order)
        rows = size(coefficients, 1)
        rows % nc == 0 || throw(
            DimensionMismatch(
                "coefficients has $rows rows, which is not a multiple of " *
                "$(nc) = 2M+1 for order $order",
            ),
        )
        return new{C,T}(Int(order), rows ÷ nc, center, coefficients)
    end
end

order(s::CircularHarmonicStorage) = s.order
ncomponents(s::CircularHarmonicStorage) = s.ncomponents
numcoefficients(s::CircularHarmonicStorage) = numcoefficients(s.order)
numbasisfunctions(s::CircularHarmonicStorage) = size(s.coefficients, 2)

"""
    coefficientmatrix(storage)

The raw `(ncoeff * ncomponents) × nbasisfunctions` matrix.

Exposed so the receive side can apply `Cᴴ` directly. [`testcoefficients!`](@ref) is the same product
with a result vector supplied; this is for callers that already hold one and want `mul!`'s `α`/`β`
arithmetic, which the traversal's leaf scatter needs.
"""
coefficientmatrix(s::CircularHarmonicStorage) = s.coefficients

"""
    CircularHarmonicStorage(t, patterns, center) -> storage

Analyse sampled patterns into coefficients. `patterns` is a vector, one entry per basis function, each
`ndirections × ncomponents` on `t`'s grid.

`t` may be built on the leaf's own MLFMA grid. Oversampling was unnecessary in the measured cases.
The circle's quadrature is exact for resolved modes (see `transform.jl`), but a real basis pattern is not
band-limited, so its content above `N/2` folds back onto the modes being kept instead of being
dropped. `benchmark/interpolation/checalibration.jl` sweeps `oversampling` over 1, 2, 4, 8 and 16 and
the group-moment error is IDENTICAL TO SIX DIGITS at every order tested. The reason is that the folded
content is already below the error being measured: on the λ/4 fixture the pattern's own spectrum has
fallen to ~1e-14 by `|m| = 16`, beyond the leaf grid's Nyquist bin at 14.

Oversampling did not improve those measurements and costs a proportionally larger assembly.
It is available on [`circulargrid`](@ref) because "band-limited enough by `N/2`" is a property of these fixtures rather
than a theorem, and a caller with an unusually large box relative to its mesh should be able to check.
"""
function CircularHarmonicStorage(
    t::CircularHarmonicTransform{T,C}, patterns::AbstractVector{<:AbstractMatrix}, center
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
    return CircularHarmonicStorage(order(t), SVector{2}(center), coefficients)
end

"""
    aggregate!(A, storage, x) -> A

`A .= storage.coefficients * x` -- the group's aggregated expansion, `A_g = Σ_j I_j a_j`.

`A` is a vector of length `numcoefficients(storage) * ncomponents(storage)`; `reshape(A, ncoeff, D)`
is then the matrix [`synthesize`](@ref) wants.

Every column must share the storage's `center` for this sum to be meaningful.
The caller is responsible for that agreement.
"""
function aggregate!(A::AbstractVector, storage::CircularHarmonicStorage, x::AbstractVector)
    return LinearAlgebra.mul!(A, storage.coefficients, x)
end

"""
    aggregatelevel!(F, A, t, storages, xs) -> F

Aggregate every group of a level and synthesise them all in ONE GEMM.

`storages` and `xs` are matched vectors over the level's groups; `A` is scratch of size
`(ncoeff, D * ngroups)` and `F` receives `(ndirections, D * ngroups)`, group `g` occupying columns
`((g-1)D+1):(gD)` in both.

The single trailing `synthesize!` is the whole reason this function exists rather than a loop calling
`synthesize` per group -- see this file's header on why batching is mandatory rather than an
optimization.
"""
function aggregatelevel!(
    F::AbstractMatrix,
    A::AbstractMatrix,
    t::CircularHarmonicTransform,
    storages::AbstractVector{<:CircularHarmonicStorage},
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
# no quadrature accuracy is required for it, only that the SAME `W` appears in both. It is a
# reordering of a finite sum, not an approximated integral. In particular the product `G·conj(ψ_m)`
# has bandwidth `L + M`, which the leaf grid does not resolve; if the identity needed that integral to
# be exact the scheme would not work at all. (On the circle the leaf grid's rule IS exact to `|m| < N`,
# so this is a weaker requirement here than on the sphere -- but the argument is the one that matters,
# because it is what makes the pair adjoint on a grid that resolves nothing in particular.)

"""
    analyzelevel!(B, t, G) -> B

Project a whole level's incoming fields onto the harmonics: `B = Sᴴ (w .* G)`.

`G` is `(ndirections, D * ngroups)` as [`aggregatelevel!`](@ref) produces, and `B` receives
`(ncoeff, D * ngroups)` in the same column layout. One GEMM for the level, for the same reason the
forward direction batches: `S` is shared by every group.
"""
function analyzelevel!(B::AbstractMatrix, t::CircularHarmonicTransform, G::AbstractMatrix)
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

`V .= storage.coefficients' * B` -- the tested values `V_j = Σ_m conj(a_{j,m}) B_m`.

The exact adjoint of [`aggregate!`](@ref) by construction: the same matrix, applied on the other side.
`B` is a vector of length `numcoefficients(storage) * ncomponents(storage)`, i.e. what
[`analyzelevel!`](@ref) writes for one group, flattened.
"""
function testcoefficients!(
    V::AbstractVector, storage::CircularHarmonicStorage, B::AbstractVector
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
    t::CircularHarmonicTransform,
    storages::AbstractVector{<:CircularHarmonicStorage},
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
