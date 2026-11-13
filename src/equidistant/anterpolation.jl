# Anterpolation as the WEIGHTED adjoint of interpolation.
#
# Aggregation interpolates child patterns up to the parent
# grid; disaggregation must push parent contributions back down. For the resulting matrix-vector
# product to be the adjoint of the forward one, the down-step has to be the adjoint of the up-step
# In the quadrature-weighted inner product, ⟨u,v⟩ = ∮ u v̄ dΩ,
# not the bare Euclidean dot product on samples. Writing R for interpolation coarse -> fine:
#
#     ⟨R f, g⟩_fine = ⟨f, A g⟩_coarse   for all f, g   ⟺   A = W_c⁻¹ Rᴴ W_f
#
# The plain adjoint `Rᴴ` omits the change in weights. The identity holds algebraically for the
# chosen weights; `exactquadratureweights` also makes the inner product accurate for resolved fields.
#
# `resamplematrix` and `anterpolatedense` are dense references for checking the fast adjoint.
# They require O(N²) storage and are suitable only for small grids.

"""
    resamplematrix(src, dst, parity, backend) -> Matrix

The dense matrix of the (linear) `resample` operator from `src` to `dst`, built by applying it to
each unit vector in turn.

Reference-grade only: this is `length(dst) × length(src)` dense. See this file's header.
"""
function resamplematrix(
    src::EquidistantSampling, dst::EquidistantSampling, parity::Int, backend
)
    ncoarse = Nθ(src) * Nφ(src)
    nfine = Nθ(dst) * Nφ(dst)
    R = Matrix{ComplexF64}(undef, nfine, ncoarse)
    e = zeros(ComplexF64, Nφ(src), Nθ(src))
    for c in 1:ncoarse
        fill!(e, 0)
        e[c] = 1
        R[:, c] = vec(resample(e, src, dst, parity, backend))
    end
    return R
end

"""
    anterpolate(G, src, dst, parity, backend) -> Matrix

Push a pattern on the FINE grid `src` down to the COARSE grid `dst`, as the quadrature-weighted
adjoint of interpolating `dst -> src`.

Satisfies `⟨resample(f), G⟩_src == ⟨f, anterpolate(G)⟩_dst` exactly, where `⟨u,v⟩ = ∮ u v̄ dΩ` is
evaluated with [`exactquadratureweights`](@ref). Note the argument order mirrors `resample`: `src`
is where `G` lives and `dst` is where the result lands, which is the OPPOSITE of the interpolation
whose adjoint this is.
"""
function anterpolate(
    G::AbstractMatrix,
    src::EquidistantSampling,
    dst::EquidistantSampling,
    parity::Int,
    backend,
)
    # `resampleadjoint` names the FORWARD direction, which is `dst -> src`: coarse to fine.
    return resampleadjoint(exactquadratureweights(src) .* G, dst, src, parity, backend) ./
           exactquadratureweights(dst)
end

"""
    anterpolatedense(G, src, dst, parity, backend) -> Matrix

The same operator as [`anterpolate`](@ref), built by materializing the interpolation matrix.

This reference follows `W_c⁻¹ Rᴴ W_f` directly; it is not a fallback.
`test/Equidistant/transposed.jl` compares it with the fast path for both backends and both parities.

Costs one full `resample` per source sample. Use it on small grids only.
"""
function anterpolatedense(
    G::AbstractMatrix,
    src::EquidistantSampling,
    dst::EquidistantSampling,
    parity::Int,
    backend,
)
    R = resamplematrix(dst, src, parity, backend)      # note: coarse -> fine, hence dst -> src
    Wf = vec(exactquadratureweights(src))
    Wc = vec(exactquadratureweights(dst))
    return reshape((R' * (Wf .* vec(G))) ./ Wc, Nφ(dst), Nθ(dst))
end

"""
    innerproduct(u, v, sampling)

`∮ u v̄ dΩ`, the inner product the weighted adjoint above is taken with respect to.
"""
function innerproduct(u::AbstractMatrix, v::AbstractMatrix, s::EquidistantSampling)
    return sum(exactquadratureweights(s) .* u .* conj(v))
end
