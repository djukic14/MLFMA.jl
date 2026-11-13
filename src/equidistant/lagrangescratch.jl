# Buffer-reusing local interpolation: the same operator as `resample`/`resampleadjoint` for
# `LagrangeBackend`, with the stencils and staging array hoisted into a caller-owned scratch so a
# warmed-up apply allocates NOTHING.
#
# `_stencil` depends only on the grids and order. Cache its weights and resolved indices once
# per level pair rather than rebuilding them for each apply.
#
# Store stencils column-major in one matrix, not a `Vector{Tuple{Int,Vector{Float64}}}`. The
# tuple-of-vectors form is what the allocating routines build, and it costs a pointer chase per
# destination index in the inner loop. `weights[:, i]` is contiguous and the loop over stencil taps
# walks it in order.

"""
    LagrangeStencils(p, positions)

The `p`-point Lagrange stencil at every entry of `positions`, precomputed.

`positions[i]` is target `i`'s location in SOURCE cells (0-based), and `lefts[i]` is the resulting
first stencil node -- possibly out of range at either end, which `sphericalget` and φ periodicity
resolve exactly as they do on the allocating path.

Pass positions computed with the reference's formulas, not an affine `(ratio, offset)` pair.
`ratio*i - ratio` and `(i-1)*ratio` can round differently in Float64, changing the stencil weights.
"""
struct LagrangeStencils
    lefts::Vector{Int}
    weights::Matrix{Float64}
    indices::Matrix{Int}
end

# `nsrc` is the source φ count. Cache `mod1(left + b, nsrc)` in `indices` to avoid integer
# division for every tap of every output point.
function LagrangeStencils(p::Int, positions::AbstractVector{Float64}, nsrc::Int)
    lefts = Vector{Int}(undef, length(positions))
    weights = Matrix{Float64}(undef, p, length(positions))
    indices = Matrix{Int}(undef, p, length(positions))
    for (i, t) in enumerate(positions)
        left, w = _stencil(p, t)
        lefts[i] = left
        for a in 1:p
            weights[a, i] = w[a]
            indices[a, i] = mod1(left + a, nsrc)
        end
    end
    return LagrangeStencils(lefts, weights, indices)
end

"""
    ReflectedStencils(p, positions, nθ, nφ)

The θ stencils with the SPHERICAL CONTINUATION ALREADY RESOLVED: for every tap, the physical row it
reads, the φ half-turn it needs, and the sign it carries.

Cache the reflection so the θ loop need not call [`sphericalget`](@ref) for each tap.
`reflectindex(i, j, Nθ, Nφ)` depends only on the grid and indices, not on the data.

TWO THINGS FOLD OUT, both of which keep the inner loop free of branches:

  - The φ dependence is a PURE ADDITIVE SHIFT, so probing `reflectindex` at `j = 1` gives the offset
    for every `j`. Each reflection adds `Nφ÷2`, so the offset is 0 or `Nφ÷2` and nothing else.
  - The sign is `(-1)^reflections` and multiplication by `±1` is EXACT, so folding it into a second
    weight table costs no accuracy: `w * (sgn * v)` and `(w * sgn) * v` are the same Float64. That is
    what keeps this bitwise against the allocating reference, and it is the only reason the sign may
    leave the inner loop.
"""
struct ReflectedStencils
    weights::Matrix{Float64}
    antiweights::Matrix{Float64}
    indices::Matrix{Int}
    shifts::Matrix{Int}
end

function ReflectedStencils(p::Int, positions::AbstractVector{Float64}, nθ::Int, nφ::Int)
    m = length(positions)
    weights = Matrix{Float64}(undef, p, m)
    antiweights = Matrix{Float64}(undef, p, m)
    indices = Matrix{Int}(undef, p, m)
    shifts = Matrix{Int}(undef, p, m)
    for (i, t) in enumerate(positions)
        left, w = _stencil(p, t)
        for a in 1:p
            i′, j′, sgn = reflectindex(left + a, 1, nθ, nφ)
            indices[a, i] = i′
            shifts[a, i] = j′ - 1
            weights[a, i] = w[a]
            antiweights[a, i] = sgn * w[a]
        end
    end
    return ReflectedStencils(weights, antiweights, indices, shifts)
end

# The weight table for a parity. SPHERICAL is unchanged by a reflection; ANTISPHERICAL flips.
@inline reflectedweights(r::ReflectedStencils, parity::Int) =
    parity == SPHERICAL ? r.weights : r.antiweights

"""
    LagrangeTranspose(stencils, ncoarse)

The φ stencils TRANSPOSED: for every coarse φ row, the fine rows that scatter into it and with what
weight, in CSR form.

The adjoint's φ stage could scatter with `out[index[b,j′], i] += w * v`. Reversing that map makes it
a gather: each output cell accumulates in a register and is written once.

ORDER IS PRESERVED, so this stays bitwise. The scatter visits `(j′, b)` in ascending order and each
`(j′, b)` reaches its destination once, so a destination cell accumulates in ascending `(j′, b)`. This
constructor fills each bucket in that same order, so the summation order per output cell is identical.
The θ stage's reorder does not preserve summation order: its shifts make the arrival order depend
on the column. That stage can therefore differ from the reference by round-off.
"""
struct LagrangeTranspose
    rowstart::Vector{Int}
    sources::Vector{Int}
    weights::Vector{Float64}
end

function LagrangeTranspose(stencils::LagrangeStencils, ncoarse::Int)
    p, m = size(stencils.indices)
    counts = zeros(Int, ncoarse)
    for j′ in 1:m, b in 1:p
        counts[stencils.indices[b, j′]] += 1
    end
    rowstart = Vector{Int}(undef, ncoarse + 1)
    rowstart[1] = 1
    for jc in 1:ncoarse
        rowstart[jc + 1] = rowstart[jc] + counts[jc]
    end
    total = rowstart[end] - 1
    sources = Vector{Int}(undef, total)
    weights = Vector{Float64}(undef, total)
    cursor = copy(rowstart)
    for j′ in 1:m, b in 1:p
        jc = stencils.indices[b, j′]
        k = cursor[jc]
        cursor[jc] += 1
        sources[k] = j′
        weights[k] = stencils.weights[b, j′]
    end
    return LagrangeTranspose(rowstart, sources, weights)
end

# The two position formulas, written exactly as `resample`/`resampleadjoint` write them inline.
_φpositions(src::EquidistantSampling, dst::EquidistantSampling) =
    [(j - 1) * (src.Nφ / dst.Nφ) for j in 1:(dst.Nφ)]
_θpositions(src::EquidistantSampling, dst::EquidistantSampling) =
    [(i - 0.5) * (src.Nθ / dst.Nθ) - 0.5 for i in 1:(dst.Nθ)]

"""
    LagrangeScratch(child, parent, backend, ::Type{T})

Precomputed stencils and staging storage for one level pair's local interpolation, both directions.

| field        | shape          | interpolation          | anterpolation          |
|:------------ |:-------------- |:---------------------- |:---------------------- |
| `φstencils`  | `Nφ_p` targets | the φ gather           | the φ scatter          |
| `θstencils`  | `Nθ_p` targets | the θ gather           | the θ scatter          |
| `stage`      | `(Nφ_p, Nθ_c)` | between the two stages | between the two stages |
| `parentgrid` | `(Nφ_p, Nθ_p)` | result                 | --                     |
| `childgrid`  | `(Nφ_c, Nθ_c)` | --                     | result                 |

Both directions share the stencils because the adjoint is a scatter over the SAME stencils the forward
gathers through -- the weights are real, so transpose and adjoint coincide and no second table is
needed. `stagegrid` carries the `Nθ_src × Nφ_dst` sampling the θ stage reflects through; its half-turn
must use the DESTINATION φ count, which is a silent pole-only error if taken from `src`.
"""
struct LagrangeScratch{T,S}
    φstencils::LagrangeStencils
    φtranspose::LagrangeTranspose
    θstencils::ReflectedStencils
    stagegrid::S
    stage::Matrix{T}
    parentgrid::Matrix{T}
    childgrid::Matrix{T}
end

function LagrangeScratch(
    child::EquidistantSampling,
    parent::EquidistantSampling,
    backend::LagrangeBackend,
    ::Type{T},
) where {T<:Complex}
    pθ, pφ = θorder(backend), φorder(backend)
    # The θ stencils resolve against the STAGE grid -- source θ resolution, destination φ count --
    # because that is the array the θ stage reads. Taking the half-turn from `child.Nφ` instead is a
    # silent pole-only error, which is why `_stagegrid` is a named thing.
    stage = _stagegrid(child, parent)
    φstencils = LagrangeStencils(pφ, _φpositions(child, parent), child.Nφ)
    return LagrangeScratch(
        φstencils,
        LagrangeTranspose(φstencils, child.Nφ),
        ReflectedStencils(pθ, _θpositions(child, parent), stage.Nθ, stage.Nφ),
        stage,
        Matrix{T}(undef, parent.Nφ, child.Nθ),
        Matrix{T}(undef, parent.Nφ, parent.Nθ),
        Matrix{T}(undef, child.Nφ, child.Nθ),
    )
end

# The cached table heights retain both orders, including anisotropic backends.
@inline _lagrangeordersmatch(scratch::LagrangeScratch, backend::LagrangeBackend) =
    size(scratch.θstencils.weights, 1) == θorder(backend) &&
    size(scratch.φstencils.weights, 1) == φorder(backend)

@inline _lagrangegridsmatch(scratch::LagrangeScratch, src, dst) =
    size(scratch.stage) == (dst.Nφ, src.Nθ) &&
    size(scratch.parentgrid) == (dst.Nφ, dst.Nθ) &&
    size(scratch.childgrid) == (src.Nφ, src.Nθ)

@inline function _checklagrangescratch(scratch::LagrangeScratch, backend, src, dst)
    _lagrangeordersmatch(scratch, backend) || throw(
        ArgumentError(
            "LagrangeScratch stencil orders (θ, φ) = " *
            "$((size(scratch.θstencils.weights, 1), size(scratch.φstencils.weights, 1))) " *
            "do not match backend orders $((θorder(backend), φorder(backend)))",
        ),
    )
    _lagrangegridsmatch(scratch, src, dst) || throw(
        DimensionMismatch(
            "LagrangeScratch grids do not match source $(size(src)) and destination $(size(dst))",
        ),
    )
    return nothing
end

"""
    resample!(out, F, src, dst, parity, backend::LagrangeBackend, scratch) -> out

[`resample`](@ref) with the stencils precomputed and the staging array taken from `scratch`.
Allocates nothing once warmed up.

Throws `ArgumentError` for incompatible scratch orders and `DimensionMismatch` for incompatible
scratch grids or input/output shapes, before modifying any buffers.

Identical arithmetic to the allocating form, in the identical order, so the two agree BITWISE -- there
is no transform here to introduce a different algorithm, which is why this can make a stronger claim
than the spectral path's `≈`.
"""
function resample!(
    out::AbstractMatrix,
    F::AbstractMatrix,
    src::EquidistantSampling,
    dst::EquidistantSampling,
    parity::Int,
    backend::LagrangeBackend,
    scratch::LagrangeScratch,
)
    _checklagrangescratch(scratch, backend, src, dst)
    size(F) == size(src) ||
        throw(DimensionMismatch("resample! input must have size $(size(src))"))
    size(out) == size(dst) ||
        throw(DimensionMismatch("resample! output must have size $(size(dst))"))
    pθ, pφ = θorder(backend), φorder(backend)

    φ = scratch.φstencils
    φw, φidx = φ.weights, φ.indices
    stage = scratch.stage
    @inbounds for i in 1:(src.Nθ), j′ in 1:(dst.Nφ)
        acc = zero(eltype(stage))
        for b in 1:pφ
            acc += φw[b, j′] * F[φidx[b, j′], i]     # φ is plain periodic; the wrap is precomputed
        end
        stage[j′, i] = acc
    end

    # Keep `a` innermost for this gather: accumulate in a register, then write each output once.
    # The adjoint uses a different loop order because it scatters.
    θ = scratch.θstencils
    θw = reflectedweights(θ, parity)
    nφstage = dst.Nφ
    @inbounds for i′ in 1:(dst.Nθ), j′ in 1:(dst.Nφ)
        acc = zero(eltype(out))
        for a in 1:pθ
            jj = j′ + θ.shifts[a, i′]
            jj > nφstage && (jj -= nφstage)
            acc += θw[a, i′] * stage[jj, θ.indices[a, i′]]
        end
        out[j′, i′] = acc
    end
    return out
end

"""
    resampleadjoint!(out, Y, coarse, fine, parity, backend::LagrangeBackend, scratch) -> out

[`resampleadjoint`](@ref) with the stencils precomputed and the staging array taken from `scratch`.
Allocates nothing once warmed up.

Throws `ArgumentError` for incompatible scratch orders and `DimensionMismatch` for incompatible
scratch grids or input/output shapes, before modifying any buffers.

As in the allocating form, the samplings name the FORWARD direction: `coarse` first, then `fine`,
while the data flows fine -> coarse. Every forward gather becomes a scatter over the identical
stencil, with the stages run in reverse.

Unlike [`resample!`](@ref), this can differ from its allocating reference by round-off. The θ scatter
reorders contributions when two reflected taps reach the same source row.
"""
function resampleadjoint!(
    out::AbstractMatrix,
    Y::AbstractMatrix,
    coarse::EquidistantSampling,
    fine::EquidistantSampling,
    parity::Int,
    backend::LagrangeBackend,
    scratch::LagrangeScratch,
)
    _checklagrangescratch(scratch, backend, coarse, fine)
    size(Y) == size(fine) ||
        throw(DimensionMismatch("resampleadjoint! input must have size $(size(fine))"))
    size(out) == size(coarse) ||
        throw(DimensionMismatch("resampleadjoint! output must have size $(size(coarse))"))
    pθ, pφ = θorder(backend), φorder(backend)

    θ = scratch.θstencils
    θw = reflectedweights(θ, parity)
    stage = scratch.stage
    nφstage = fine.Nφ
    fill!(stage, zero(eltype(stage)))
    # Keep `j′` innermost for this scatter so each tap adds a shifted column with fixed weights
    # and indices. Reflected taps can reach the same row in a different order from the reference,
    # so the result need not be bitwise identical.
    @inbounds for i′ in 1:(fine.Nθ)
        for a in 1:pθ
            w = θw[a, i′]
            idx = θ.indices[a, i′]
            sh = θ.shifts[a, i′]
            # The shift is 0 or Nφ/2 and nothing else, so the wrap is not a per-tap test -- it splits
            # `j′` into two CONTIGUOUS ranges that need no branch at all. Each `j′` still writes a
            # distinct cell, so splitting cannot change any cell's accumulation order.
            lim = nφstage - sh
            for j′ in 1:lim
                stage[j′ + sh, idx] += w * Y[j′, i′]
            end
            for j′ in (lim + 1):nφstage
                stage[j′ - lim, idx] += w * Y[j′, i′]
            end
        end
    end

    # GATHER over the transposed stencil rather than a scatter over the forward one: register
    # accumulator, one store per output cell, no write hazard, and no `fill!`. Bitwise-neutral --
    # `LagrangeTranspose` preserves each cell's summation order by construction.
    tr = scratch.φtranspose
    trstart, trsrc, trw = tr.rowstart, tr.sources, tr.weights
    @inbounds for i in 1:(coarse.Nθ), jc in 1:(coarse.Nφ)
        acc = zero(eltype(out))
        for c in trstart[jc]:(trstart[jc + 1] - 1)
            acc += trw[c] * stage[trsrc[c], i]
        end
        out[jc, i] = acc
    end
    return out
end
