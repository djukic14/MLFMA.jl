# local-local backend: tensor-product barycentric Lagrange on the same equidistant grids.
#
# Resample separably: φ for every source θ row, then θ for every destination
# φ column. That turns `N_target · p_θ · p_φ` work into roughly `Nθ·Nφ_f·p_φ + Nθ_f·Nφ_f·p_θ` and keeps
# the inner loops walking contiguous memory.
#
# The midpoint grid avoids a special case at the poles: a θ stencil that runs off
# either end is read through `sphericalget`, which reflects the index, half-turns φ, and applies the
# component's parity sign. The stencil arithmetic below never mentions a pole.

"""
    LagrangeBackend{Pθ,Pφ}()

Local tensor-product interpolation of order `Pθ × Pφ` on an `EquidistantSampling`. `Pθ = Pφ = 4` or `6`
are the sensible starting points; the error is algebraic in the order, against the FFT backend's
spectral accuracy.
"""
struct LagrangeBackend{Pθ,Pφ} end
LagrangeBackend(p::Int) = LagrangeBackend{p,p}()
LagrangeBackend(pθ::Int, pφ::Int) = LagrangeBackend{pθ,pφ}()

θorder(::LagrangeBackend{Pθ,Pφ}) where {Pθ,Pφ} = Pθ
φorder(::LagrangeBackend{Pθ,Pφ}) where {Pθ,Pφ} = Pφ

# Barycentric weights for `p` EQUISPACED nodes at 0, 1, …, p-1. These depend only on `p`, so they are a
# property of the order and not of any grid -- computed once here and reused for every offset, level
# and direction. Barycentric form is used because it is the numerically stable way to evaluate a
# Lagrange basis; the naive product form loses digits well before `p` gets interesting.
function _barycentricconstants(p::Int)
    λ = ones(Float64, p)
    for j in 1:p, m in 1:p
        m == j && continue
        λ[j] /= ((j - 1) - (m - 1))
    end
    return λ
end

"""
    lagrangeweights(p, x) -> Vector

The `p` Lagrange basis values at position `x`, measured in GRID CELLS from the first stencil node.
`x ∈ [0, p-1]`. Exact node hits are returned as a unit vector rather than a division by zero -- which
is not a defensive nicety here, since factor-two grid refinement puts targets exactly on source nodes
half the time.
"""
function lagrangeweights(p::Int, x::Float64)
    λ = _barycentricconstants(p)
    w = zeros(Float64, p)
    for j in 1:p
        if abs(x - (j - 1)) < 1e-13
            w[j] = 1.0
            return w
        end
    end
    s = 0.0
    for j in 1:p
        w[j] = λ[j] / (x - (j - 1))
        s += w[j]
    end
    return w ./ s
end

# The stencil for a target coordinate: first node index and the weights. `t` is the target position in
# SOURCE cell units (0-based), so node `floor(t)` is its left neighbour; the stencil is centred on it as
# far as the order allows. No clamping to the array -- `sphericalget` and φ periodicity handle the ends,
# which is exactly why the two coordinates can share this routine.
function _stencil(p::Int, t::Float64)
    left = floor(Int, t) - (p ÷ 2 - 1)
    return left, lagrangeweights(p, t - left)
end

"""
    resample(F, src, dst, parity, backend::LagrangeBackend) -> Nφ(dst) × Nθ(dst) array

Local separable interpolation of one component between two `EquidistantSampling`s.

Both coordinates are equidistant, so a target's position in source cells is affine in its index and
the stencils REPEAT: for a factor-two refinement there are only two distinct fractional offsets in
each coordinate. They are precomputed per destination index here, before the data loop, so the inner
work is multiply-add only.
"""
function resample(
    F::AbstractMatrix,
    src::EquidistantSampling,
    dst::EquidistantSampling,
    parity::Int,
    backend::LagrangeBackend,
)
    pθ, pφ = θorder(backend), φorder(backend)
    T = complex(float(eltype(F)))

    # --- φ stage: every source θ column onto the destination φ grid ---------------------------------
    # φ_j = 2π(j-1)/Nφ on both, so target j' sits at source cell (j'-1)·Nφ_src/Nφ_dst.
    φratio = src.Nφ / dst.Nφ
    φstencils = [_stencil(pφ, (j - 1) * φratio) for j in 1:(dst.Nφ)]

    stageφ = zeros(T, dst.Nφ, src.Nθ)
    @inbounds for i in 1:(src.Nθ), (j′, (left, w)) in enumerate(φstencils)
        acc = zero(T)
        for b in 1:pφ
            acc += w[b] * F[mod1(left + b, src.Nφ), i]   # φ is plain periodic: no parity, no reflection
        end
        stageφ[j′, i] = acc
    end

    # --- θ stage: onto the destination θ grid, through the spherical accessor ---------------------
    # Midpoint grids: θ_i = (i-1/2)h. Target θ_{i′} sits at source cell index
    #   t = θ_{i′}/h_src - 1/2 = (i′-1/2)·Nθ_src/Nθ_dst - 1/2   (0-based cells)
    θratio = src.Nθ / dst.Nθ
    θstencils = [_stencil(pθ, (i - 0.5) * θratio - 0.5) for i in 1:(dst.Nθ)]

    out = zeros(T, dst.Nφ, dst.Nθ)
    stage = _stagegrid(src, dst)
    @inbounds for (i′, (left, w)) in enumerate(θstencils), j′ in 1:(dst.Nφ)
        acc = zero(T)
        for a in 1:pθ
            # `stageφ` lives on (destination φ) × (source θ), so the accessor needs the DESTINATION φ
            # count for its half-turn -- hence the temporary sampling rather than `src`.
            acc += w[a] * sphericalget(stageφ, left + a, j′, stage, parity)
        end
        out[j′, i′] = acc
    end
    return out
end

# The intermediate grid the φ stage produces: source θ resolution, destination φ resolution. The θ
# reflection's half-turn must be `Nφ_dst ÷ 2` there, not `Nφ_src ÷ 2`, and getting that wrong is a
# silent half-turn error only visible near the poles -- so the grid is named rather than inlined.
_stagegrid(src::EquidistantSampling{T}, dst::EquidistantSampling) where {T} =
    EquidistantSampling{T}(src.Nθ, dst.Nφ)
