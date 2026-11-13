# Fast adjoints of `resample`, with the same asymptotic cost as the forward map.
# The dense `resamplematrix` is a reference for small grids, not an apply-time fallback.
#
# `resampleadjoint(Y, coarse, fine, parity, backend)` names the forward sampling pair but sends
# fine-grid data to the coarse grid. It computes the Euclidean adjoint, without quadrature weights.
#
# Each backend reverses its forward stages:
#
#   * Lagrange weights are real, so every forward gather becomes a scatter over the same stencil,
#     with no conjugation.
#   * The FFT pipeline is a chain of linear maps, so the adjoint reverses the chain and adjoints each
#     link. `fft` and `ifft` swap roles, and the normalizations cancel exactly: the forward `fft(E)/N`
#     has adjoint `ifft`, and the forward `ifft(P)*N` has adjoint `fft`. The midpoint phase is diagonal,
#     so it conjugates; the coefficient move transposes; `extend` scatters back through the same
#     reflection it gathered through.
#
# `test/Equidistant/transposed.jl` compares both backends and parities with the dense reference.

"""
    sphericalscatter!(F, v, i, j, sampling, parity)

Accumulate `v` into `F` at the physical partner of the (possibly out-of-range) index `(i, j)`, where
`i` is the θ index and `j` the φ index. `F` is stored `[φ, θ]` like every other sampled array.

The exact adjoint of [`sphericalget`](@ref): same reflection, same sign, `+=` instead of read. The
sign is real, so this serves the transpose and the adjoint alike.
"""
@inline function sphericalscatter!(
    F, v, i::Int, j::Int, s::EquidistantSampling, parity::Int
)
    i′, j′, sgn = reflectindex(i, j, s.Nθ, s.Nφ)
    @inbounds F[j′, i′] += parity == SPHERICAL ? v : sgn * v
    return F
end

"""
    resamplecoefficientsadjoint(v, Nsrc, Ndst) -> length-`Nsrc` vector

Transpose of [`resamplecoefficients`](@ref): consumes length-`Ndst` coefficients, produces length-`Nsrc`.

Walks the same frequency bookkeeping in the same order, reading where the forward wrote. Its
coefficients are 1 and 1/2 -- both real -- so transpose and adjoint coincide, and the Nyquist split
transposes into a sum over the two half-bins.
"""
function resamplecoefficientsadjoint(v::AbstractVector{T}, Nsrc::Int, Ndst::Int) where {T}
    out = zeros(T, Nsrc)
    Ndst == Nsrc && return copyto!(out, v)
    fsrc = _signedfrequencies(Nsrc)
    keep = Ndst ÷ 2
    @inbounds for (k, m) in enumerate(fsrc)
        absm = abs(m)
        absm > keep && continue
        if iseven(Nsrc) && absm == Nsrc ÷ 2 && Ndst > Nsrc
            out[k] += v[_indexof(m, Ndst)] / 2 + v[_indexof(-m, Ndst)] / 2
        elseif iseven(Ndst) && absm == keep && Ndst < Nsrc
            out[k] += v[_indexof(keep, Ndst)]
        else
            out[k] += v[_indexof(m, Ndst)]
        end
    end
    return out
end

"""
    resamplereflectedcoefficientsadjoint(v, Nsrc, Ndst) -> length-`Nsrc` vector

Transpose of [`resamplereflectedcoefficients`](@ref), the θ counterpart of
[`resamplecoefficientsadjoint`](@ref).

Its coefficients are `±1` and `±1/2` -- all real -- so transpose and adjoint coincide here too. The
reflected top bin retains its signs: the up-split transposes into a difference of the two half-bins,
and the down-fold returns the destination top bin negated to the negative source frequency.
"""
function resamplereflectedcoefficientsadjoint(
    v::AbstractVector{T}, Nsrc::Int, Ndst::Int
) where {T}
    out = zeros(T, Nsrc)
    Ndst == Nsrc && return copyto!(out, v)
    keep = Ndst ÷ 2
    @inbounds for k in 1:Nsrc
        m = _signedfrequency(k, Nsrc)
        absm = abs(m)
        absm > keep && continue
        if iseven(Nsrc) && absm == Nsrc ÷ 2 && Ndst > Nsrc
            out[k] += v[_indexof(m, Ndst)] / 2 - v[_indexof(-m, Ndst)] / 2
        elseif iseven(Ndst) && absm == keep && Ndst < Nsrc
            out[k] += m < 0 ? -v[_indexof(keep, Ndst)] : v[_indexof(keep, Ndst)]
        else
            out[k] += v[_indexof(m, Ndst)]
        end
    end
    return out
end

"""
    resampleadjoint(Y, coarse, fine, parity, ::FFTBackend) -> coarse-shaped array

Adjoint of the spectral `resample(·, coarse, fine, parity, FFTBackend())`.
"""
function resampleadjoint(
    Y::AbstractMatrix,
    coarse::EquidistantSampling,
    fine::EquidistantSampling,
    parity::Int,
    ::FFTBackend,
)
    T = complex(float(eltype(Y)))
    nθe_c, nφ_c = 2 * coarse.Nθ, coarse.Nφ
    nθe_f, nφ_f = 2 * fine.Nθ, fine.Nφ

    # adjoint of `Efine[:, 1:fine.Nθ]`: the discarded columns contribute nothing, so they stay zero
    Z = zeros(T, nφ_f, nθe_f)
    @inbounds Z[:, 1:(fine.Nθ)] .= Y

    # adjoint of `ifft(padded) * (nθe_f * nφ_f)` -- the scale and the 1/N in ifft's adjoint cancel
    P = fft(Z)

    # Reverse and conjugate the two midpoint phase passes around the coefficient moves.
    # Undo the destination phase first, on fine-grid frequencies.
    δc = π / (2 * coarse.Nθ)
    δf = π / (2 * fine.Nθ)
    @inbounds for (k, m) in enumerate(_signedfrequencies(nθe_f))
        c = conj(cis(m * δf))
        for j in axes(P, 1)
            P[j, k] *= c
        end
    end

    # adjoint of the θ then φ coefficient moves, in reverse order. θ is dim 2, φ is dim 1.
    P2 = zeros(T, nφ_f, nθe_c)
    for j in 1:nφ_f
        P2[j, :] .= resamplereflectedcoefficientsadjoint(view(P, j, :), nθe_c, nθe_f)
    end
    P3 = zeros(T, nφ_c, nθe_c)
    for k in 1:nθe_c
        P3[:, k] .= resamplecoefficientsadjoint(view(P2, :, k), nφ_c, nφ_f)
    end

    # adjoint of the first forward pass: COARSE frequencies, after the moves
    @inbounds for (k, m) in enumerate(_signedfrequencies(nθe_c))
        c = conj(cis(-m * δc))
        for j in axes(P3, 1)
            P3[j, k] *= c
        end
    end

    # adjoint of `fft(E) / (nθe_c * nφ_c)`
    Eadj = ifft(P3)

    # adjoint of `extend`: scatter every torus column back onto its physical partner
    out = zeros(T, coarse.Nφ, coarse.Nθ)
    @inbounds for i in 1:nθe_c, j in 1:nφ_c
        sphericalscatter!(out, Eadj[j, i], i, j, coarse, parity)
    end
    return out
end

"""
    resampleadjoint(Y, coarse, fine, parity, backend::LagrangeBackend) -> coarse-shaped array

Adjoint of the local `resample(·, coarse, fine, parity, backend)`.

Every forward gather becomes a scatter over the identical stencil, with the stages run in reverse.
The weights are real, so this is simultaneously the transpose and the adjoint.
"""
function resampleadjoint(
    Y::AbstractMatrix,
    coarse::EquidistantSampling,
    fine::EquidistantSampling,
    parity::Int,
    backend::LagrangeBackend,
)
    pθ, pφ = θorder(backend), φorder(backend)
    T = complex(float(eltype(Y)))

    # The forward stencils, built from the FORWARD direction coarse -> fine so they match exactly.
    φratio = coarse.Nφ / fine.Nφ
    φstencils = [_stencil(pφ, (j - 1) * φratio) for j in 1:(fine.Nφ)]
    θratio = coarse.Nθ / fine.Nθ
    θstencils = [_stencil(pθ, (i - 0.5) * θratio - 0.5) for i in 1:(fine.Nθ)]
    stage = _stagegrid(coarse, fine)

    # adjoint of the θ stage
    stageadj = zeros(T, fine.Nφ, coarse.Nθ)
    @inbounds for (i′, (left, w)) in enumerate(θstencils), j′ in 1:(fine.Nφ)
        y = Y[j′, i′]
        iszero(y) && continue
        for a in 1:pθ
            sphericalscatter!(stageadj, w[a] * y, left + a, j′, stage, parity)
        end
    end

    # adjoint of the φ stage
    out = zeros(T, coarse.Nφ, coarse.Nθ)
    @inbounds for i in 1:(coarse.Nθ), (j′, (left, w)) in enumerate(φstencils)
        v = stageadj[j′, i]
        iszero(v) && continue
        for b in 1:pφ
            out[mod1(left + b, coarse.Nφ), i] += w[b] * v
        end
    end
    return out
end
