# Buffer-reusing spectral resampling without a doubled torus. Same operator as `resample`/
# `resampleadjoint` in `fftbackend.jl`/`transposedresample.jl`, computed on `Nθ` θ points instead of
# `2Nθ`, with every temporary in caller-owned scratch so a warmed-up apply allocates nothing.
#
# The allocating `resample` is an independent reference for checking this reduced-grid path,
# including the reflected θ Nyquist split in `resamplereflectedcoefficients`.
#
# `extend` reflects the physical `Nφ × Nθ` grid into a periodic `Nφ × 2Nθ` torus.
# The reflected half is redundant: after a φ-DFT, mode `m` has θ symmetry
#
#     x[n] = ε · x[2Nθ-1-n],   ε = σ(-1)^m
#
# Here σ = +1 for SPHERICAL and -1 for ANTISPHERICAL. A DCT-II (ε = +1) or DST-II (ε = -1)
# uses this symmetry to transform `Nθ` values rather than `2Nθ`. Physical storage remains `Nφ × Nθ`.
#
# The implementation uses three conventions:
#
#   * Group rows by ε. The cosine block always uses REDFT10/REDFT01 and the sine block
#     RODFT10/RODFT01; only the permutation depends on σ.
#   * Pack real and imaginary parts into alternating rows of dense real matrices. FFTW's r2r
#     plans receive these buffers directly, without reinterpreted views.
#   * Keep raw r2r coefficients. For the sine block, `a_m = -i·S_{m-1}/(2Nθ)` and
#     `y = i·RODFT01(a)` cancel the `-i` and `+i` factors. The `Nd/Ns` rescaling is included in
#     the final global scale. The forward identity is `Ŷ[κ] = e^{iκδ}·(-2i)Σx sin(...)`;
#     its minus sign is required.
#
# With zero-based r2r indices, cosine index κ is frequency κ (0..N-1); frequency N vanishes.
# Sine index κ is frequency κ+1 (1..N). Both increase with |m|, so θ resampling copies the leading
# coefficients and zeroes the rest. The sine block's top frequency needs a half-bin correction.
#
# This path does not build a doubled torus. The allocating reference uses `extend`.

"""
    FFTScratch(child, parent, ::Type{T})

Every buffer and plan one level pair's spectral resampling needs, in both directions.

The two directions share storage because they run in separate applies and their stage shapes are
mirror images -- interpolation goes child -> parent, anterpolation parent -> child:

| field             | shape          | interpolation              | anterpolation               |
|:----------------- |:-------------- |:-------------------------- |:--------------------------- |
| `childspectrum`   | `(Nφ_c, Nθ_c)` | φ modes, in-place fft      | result of the φ adjoint     |
| `cossrc`/`sinsrc` | `(Nφ_c, Nθ_c)` | child-width θ coefficients | child-width θ coefficients  |
| `cosmid`/`sinmid` | `(Nφ_p, Nθ_c)` | after the φ resample       | after the φ adjoint's input |
| `cosdst`/`sindst` | `(Nφ_p, Nθ_p)` | destination θ coefficients | source θ coefficients       |
| `parentspectrum`  | `(Nφ_p, Nθ_p)` | φ modes, in-place ifft     | φ modes, in-place fft       |
| `parentgrid`      | `(Nφ_p, Nθ_p)` | result                     | --                          |
| `childgrid`       | `(Nφ_c, Nθ_c)` | --                         | result                      |

The φ resample runs between the two θ transforms. Packing and the forward θ transform use the
child φ count before resampling to the parent count. `cosmid` and `sinmid` are staging buffers,
never inputs to a plan.

The two real blocks together are exactly the size of one complex `(Nφ_p, ·)` array, so splitting by ε
costs no memory over storing the modes interleaved.

`interimscratch` builds this object once per (level, chunk). Each concurrent chunk owns its buffers
and plans; neither is shared through the interpolator.
"""
struct FFTScratch{T,R,P,Q}
    childspectrum::Matrix{T}
    parentspectrum::Matrix{T}
    cossrc::Matrix{R}
    sinsrc::Matrix{R}
    cosmid::Matrix{R}
    sinmid::Matrix{R}
    cosdst::Matrix{R}
    sindst::Matrix{R}
    parentgrid::Matrix{T}
    childgrid::Matrix{T}
    φcoefficients::Vector{T}
    φplans::P
    θplans::Q
end

function FFTScratch(
    child::EquidistantSampling, parent::EquidistantSampling, ::Type{T}
) where {T<:Complex}
    R = real(T)
    nφc, nθc = child.Nφ, child.Nθ
    nφp, nθp = parent.Nφ, parent.Nθ
    # The ε split pairs each φ-mode index with its neighbour, and `extend`'s own reflection pairs φ
    # with φ+π. Both need an even count, and every grid MLFMA builds has one.
    iseven(nφc) && iseven(nφp) ||
        error("FFTScratch: Nφ must be even on both grids, got $nφc and $nφp.")

    childspectrum = Matrix{T}(undef, nφc, nθc)
    parentspectrum = Matrix{T}(undef, nφp, nθp)
    # The source blocks use the child φ count: the φ resample runs between the two θ
    # transforms, so the forward transform and the pack that feeds it are done at the smaller width.
    cossrc = Matrix{R}(undef, nφc, nθc)
    sinsrc = Matrix{R}(undef, nφc, nθc)
    cosdst = Matrix{R}(undef, nφp, nθp)
    sindst = Matrix{R}(undef, nφp, nθp)

    φplans = (
        childforward=plan_fft!(childspectrum, 1),
        childinverse=plan_ifft!(childspectrum, 1),
        parentforward=plan_fft!(parentspectrum, 1),
        parentinverse=plan_ifft!(parentspectrum, 1),
    )
    # Eight, over just TWO shapes -- `(Nφ_c, Nθ_c)` and `(Nφ_p, Nθ_p)` -- because the staging pair is
    # never transformed. All eight are used: the forward direction runs `fwdsrc` then `invdst`, the
    # adjoint runs `fwddst` then `invsrc`. θ is dimension 2.
    θplans = (
        cosforwardsrc=plan_r2r!(cossrc, REDFT10, 2),
        sinforwardsrc=plan_r2r!(sinsrc, RODFT10, 2),
        cosinversesrc=plan_r2r!(cossrc, REDFT01, 2),
        sininversesrc=plan_r2r!(sinsrc, RODFT01, 2),
        cosforwarddst=plan_r2r!(cosdst, REDFT10, 2),
        sinforwarddst=plan_r2r!(sindst, RODFT10, 2),
        cosinversedst=plan_r2r!(cosdst, REDFT01, 2),
        sininversedst=plan_r2r!(sindst, RODFT01, 2),
    )

    return FFTScratch(
        childspectrum,
        parentspectrum,
        cossrc,
        sinsrc,
        Matrix{R}(undef, nφp, nθc),   # cosmid: parent φ, child θ -- staging, never transformed
        Matrix{R}(undef, nφp, nθc),   # sinmid
        cosdst,
        sindst,
        Matrix{T}(undef, nφp, nθp),
        Matrix{T}(undef, nφc, nθc),
        Vector{T}(undef, max(nφc, nφp)),
        φplans,
        θplans,
    )
end

# The ε classes. ε = σ(-1)^m, and for an even-length DFT the parity of the SIGNED frequency at 1-based
# index j is always the parity of `j-1` (the wrap subtracts an even number). So the split is a plain
# deinterleave and needs no index table: under SPHERICAL the odd rows carry ε = +1, under ANTISPHERICAL
# they carry ε = -1 and the two blocks simply trade places.
@inline _cosrow(t::Int, parity::Int) = parity == SPHERICAL ? 2t - 1 : 2t
@inline _sinrow(t::Int, parity::Int) = parity == SPHERICAL ? 2t : 2t - 1

# Complex `M` -> two real blocks with re/im on alternating rows, grouped by ε.
function _packblocks!(C::AbstractMatrix, S::AbstractMatrix, M::AbstractMatrix, parity::Int)
    nφ, n = size(M)
    @inbounds for t in 1:(nφ ÷ 2)
        jc, js = _cosrow(t, parity), _sinrow(t, parity)
        for k in 1:n
            vc, vs = M[jc, k], M[js, k]
            C[2t - 1, k] = real(vc)
            C[2t, k] = imag(vc)
            S[2t - 1, k] = real(vs)
            S[2t, k] = imag(vs)
        end
    end
    return C, S
end

# The exact inverse permutation of `_packblocks!`, and also its transpose.
function _unpackblocks!(
    M::AbstractMatrix, C::AbstractMatrix, S::AbstractMatrix, parity::Int
)
    nφ, n = size(M)
    @inbounds for t in 1:(nφ ÷ 2)
        jc, js = _cosrow(t, parity), _sinrow(t, parity)
        for k in 1:n
            M[jc, k] = complex(C[2t - 1, k], C[2t, k])
            M[js, k] = complex(S[2t - 1, k], S[2t, k])
        end
    end
    return M
end

# Scale every entry of column `k` -- the half/double bookkeeping at a boundary coefficient.
@inline function _scalecolumn!(A::AbstractMatrix, k::Int, c::Real)
    @inbounds for j in axes(A, 1)
        A[j, k] *= c
    end
    return A
end

# `dst` row-pair `u` += c * `src` row-pair `t`, over `n` columns. One complex φ-mode is two real rows.
@inline function _addrowpair!(
    dst::AbstractMatrix, u::Int, src::AbstractMatrix, t::Int, c::Real, n::Int
)
    @inbounds for k in 1:n
        dst[2u - 1, k] += c * src[2t - 1, k]
        dst[2u, k] += c * src[2t, k]
    end
    return dst
end

# Resample φ within a packed block, between the θ transforms. Both grids have even Nφ, so
# `jd - 1 ≡ m (mod 2)` on the destination as on the source. The move preserves frequency parity
# and therefore ε. Nyquist splitting and folding also stay within one block: `+Nφ/2` and `-Nφ/2`
# differ by an even number, as do `±keep`.
#
# `cld(jd, 2)` recovers the row-pair from the natural index in all four (parity, block) combinations,
# because the block already fixes the parity of `jd`.
function _resampleφblock!(
    dst::AbstractMatrix,
    src::AbstractMatrix,
    nφsrc::Int,
    nφdst::Int,
    parity::Int,
    iscos::Bool,
    n::Int,
)
    fill!(dst, zero(eltype(dst)))
    keep = nφdst ÷ 2
    @inbounds for t in 1:(nφsrc ÷ 2)
        js = iscos ? _cosrow(t, parity) : _sinrow(t, parity)
        m = _signedfrequency(js, nφsrc)
        absm = abs(m)
        absm > keep && continue
        if iseven(nφsrc) && absm == nφsrc ÷ 2 && nφdst > nφsrc
            _addrowpair!(dst, cld(_indexof(m, nφdst), 2), src, t, 0.5, n)
            _addrowpair!(dst, cld(_indexof(-m, nφdst), 2), src, t, 0.5, n)
        elseif iseven(nφdst) && absm == keep && nφdst < nφsrc
            _addrowpair!(dst, cld(_indexof(keep, nφdst), 2), src, t, 1.0, n)
        else
            _addrowpair!(dst, cld(_indexof(m, nφdst), 2), src, t, 1.0, n)
        end
    end
    return dst
end

# Transpose of [`_resampleφblock!`](@ref): same frequency walk, reading where the forward wrote.
function _resampleφblockadjoint!(
    out::AbstractMatrix,
    v::AbstractMatrix,
    nφsrc::Int,
    nφdst::Int,
    parity::Int,
    iscos::Bool,
    n::Int,
)
    fill!(out, zero(eltype(out)))
    keep = nφdst ÷ 2
    @inbounds for t in 1:(nφsrc ÷ 2)
        js = iscos ? _cosrow(t, parity) : _sinrow(t, parity)
        m = _signedfrequency(js, nφsrc)
        absm = abs(m)
        absm > keep && continue
        if iseven(nφsrc) && absm == nφsrc ÷ 2 && nφdst > nφsrc
            _addrowpair!(out, t, v, cld(_indexof(m, nφdst), 2), 0.5, n)
            _addrowpair!(out, t, v, cld(_indexof(-m, nφdst), 2), 0.5, n)
        elseif iseven(nφdst) && absm == keep && nφdst < nφsrc
            _addrowpair!(out, t, v, cld(_indexof(keep, nφdst), 2), 1.0, n)
        else
            _addrowpair!(out, t, v, cld(_indexof(m, nφdst), 2), 1.0, n)
        end
    end
    return out
end

"""
    resample!(out, F, src, dst, parity, ::FFTBackend, scratch) -> out

[`resample`](@ref) on the reduced θ grid, with every temporary taken from `scratch`. Allocates nothing
once warmed up.

`out` must be `Nφ(dst) × Nθ(dst)`. `scratch` must have been built for exactly the `(src, dst)` pair --
`interimscratch` does that, and the shapes are asserted rather than trusted, because a mismatched
buffer here would be an out-of-bounds write under `@inbounds` rather than an error.
"""
function resample!(
    out::AbstractMatrix,
    F::AbstractMatrix,
    src::EquidistantSampling,
    dst::EquidistantSampling,
    parity::Int,
    ::FFTBackend,
    scratch::FFTScratch,
)
    nφs, Ns = src.Nφ, src.Nθ
    nφd, Nd = dst.Nφ, dst.Nθ
    @assert size(scratch.childspectrum) == (nφs, Ns)
    @assert size(scratch.parentspectrum) == (nφd, Nd)
    @assert size(out) == (nφd, Nd)

    # φ modes, unnormalized. The whole normalization is one scalar at the end.
    A = scratch.childspectrum
    copyto!(A, F)
    scratch.φplans.childforward * A

    # Pack and transform θ at the CHILD φ count, before the φ resample widens the array. Both stages
    # then do `nφs` rows of work instead of `nφd`.
    C, S = scratch.cossrc, scratch.sinsrc
    _packblocks!(C, S, A, parity)
    scratch.θplans.cosforwardsrc * C
    scratch.θplans.sinforwardsrc * S

    # φ resample, on the packed blocks. Legal because the move preserves φ-mode parity and so never
    # crosses between the ε classes -- see `_resampleφblock!`.
    Cm, Sm = scratch.cosmid, scratch.sinmid
    _resampleφblock!(Cm, C, nφs, nφd, parity, true, Ns)
    _resampleφblock!(Sm, S, nφs, nφd, parity, false, Ns)

    # θ resample: head copy, zero tail. Both blocks index monotonically in |m|, so there is nothing
    # else to do -- except at the boundary coefficient of the SIN block, which is a half bin on
    # whichever grid holds it as its top frequency. The cos block needs no such case: its
    # frequency-Nθ coefficient is identically zero, which is exactly why a DCT has only N of them.
    Cd, Sd = scratch.cosdst, scratch.sindst
    fill!(Cd, zero(eltype(Cd)))
    fill!(Sd, zero(eltype(Sd)))
    ncopy = min(Ns, Nd)
    @inbounds for k in 1:ncopy, j in 1:nφd
        Cd[j, k] = Cm[j, k]
        Sd[j, k] = Sm[j, k]
    end
    if Nd > Ns
        _scalecolumn!(Sd, ncopy, 0.5)
    elseif Nd < Ns
        _scalecolumn!(Sd, ncopy, 2.0)
    end

    scratch.θplans.cosinversedst * Cd
    scratch.θplans.sininversedst * Sd

    P = scratch.parentspectrum
    _unpackblocks!(P, Cd, Sd, parity)
    scratch.φplans.parentinverse * P

    # One scalar for the whole pipeline: `ifft!` already divided by nφd, the φ forward was
    # unnormalized (÷nφs), and the r2r round trip carries a factor 2Ns.
    scale = nφd / (nφs * 2 * Ns)
    @inbounds for k in 1:Nd, j in 1:nφd
        out[j, k] = P[j, k] * scale
    end
    return out
end

"""
    resampleadjoint!(out, Y, coarse, fine, parity, ::FFTBackend, scratch) -> out

[`resampleadjoint`](@ref) on the reduced θ grid, with every temporary taken from `scratch`. Allocates
nothing once warmed up.

As in the allocating form, the samplings name the FORWARD direction: `coarse` first, then `fine`,
while the data flows fine -> coarse.

DERIVED STAGE BY STAGE FROM `resample!`, reversing the chain and transposing each link. The two links
worth stating, because neither is guessable and both are off-by-a-factor-of-two traps:

    (REDFT10)ᵀ = REDFT01 ∘ diag(2,1,…,1)      (RODFT10)ᵀ = RODFT01 ∘ diag(1,…,1,2)
    (REDFT01)ᵀ = diag(1/2,1,…,1) ∘ REDFT10    (RODFT01)ᵀ = diag(1,…,1,1/2) ∘ RODFT10

FFTW's DCT-III is the transpose of its DCT-II only up to the first coefficient, which the III form
weights by 1 where the transpose wants 2; the sine pair has the same discrepancy at the LAST
coefficient instead. Everything else transposes into itself: the pack and unpack permutations swap,
and the head copy becomes a truncation with the same boundary factor.
"""
function resampleadjoint!(
    out::AbstractMatrix,
    Y::AbstractMatrix,
    coarse::EquidistantSampling,
    fine::EquidistantSampling,
    parity::Int,
    ::FFTBackend,
    scratch::FFTScratch,
)
    nφs, Ns = coarse.Nφ, coarse.Nθ
    nφd, Nd = fine.Nφ, fine.Nθ
    @assert size(scratch.childspectrum) == (nφs, Ns)
    @assert size(scratch.parentspectrum) == (nφd, Nd)
    @assert size(out) == (nφs, Ns)

    # adjoint of the final `ifft!` and scalar: `ifft!ᴴ = (1/nφd)·fft!`, and the scalars are real, so
    # they all commute out to the single factor applied at the end.
    P = scratch.parentspectrum
    copyto!(P, Y)
    scratch.φplans.parentforward * P

    Cd, Sd = scratch.cosdst, scratch.sindst
    _packblocks!(Cd, Sd, P, parity)

    # adjoint of the forward's INVERSE r2r pair
    scratch.θplans.cosforwarddst * Cd
    scratch.θplans.sinforwarddst * Sd
    _scalecolumn!(Cd, 1, 0.5)
    _scalecolumn!(Sd, Nd, 0.5)

    # adjoint of the head copy: truncate, same boundary factor
    Cm, Sm = scratch.cosmid, scratch.sinmid
    fill!(Cm, zero(eltype(Cm)))
    fill!(Sm, zero(eltype(Sm)))
    ncopy = min(Ns, Nd)
    @inbounds for k in 1:ncopy, j in 1:nφd
        Cm[j, k] = Cd[j, k]
        Sm[j, k] = Sd[j, k]
    end
    if Nd > Ns
        _scalecolumn!(Sm, ncopy, 0.5)
    elseif Nd < Ns
        _scalecolumn!(Sm, ncopy, 2.0)
    end

    # adjoint of the φ resample, narrowing back to the child φ count
    C, S = scratch.cossrc, scratch.sinsrc
    _resampleφblockadjoint!(C, Cm, nφs, nφd, parity, true, Ns)
    _resampleφblockadjoint!(S, Sm, nφs, nφd, parity, false, Ns)

    # adjoint of the forward's FORWARD r2r pair
    _scalecolumn!(C, 1, 2.0)
    _scalecolumn!(S, Ns, 2.0)
    scratch.θplans.cosinversesrc * C
    scratch.θplans.sininversesrc * S

    A = scratch.childspectrum
    _unpackblocks!(A, C, S, parity)

    # adjoint of the φ forward transform: `fft!ᴴ = nφs·ifft!`
    scratch.φplans.childinverse * A

    # nφd/(nφs·2Ns) · 1/nφd · nφs collapses to 1/(2Ns)
    scale = 1 / (2 * Ns)
    @inbounds for k in 1:Ns, j in 1:nφs
        out[j, k] = A[j, k] * scale
    end
    return out
end

"""
    resamplecoefficientsadjoint!(out, v, Nsrc, Ndst) -> out

[`resamplecoefficientsadjoint`](@ref) writing into a caller-owned buffer, which it zeroes first.
Same reason as [`resamplecoefficients!`](@ref): it is called once per column.
"""
function resamplecoefficientsadjoint!(
    out::AbstractVector{T}, v::AbstractVector, Nsrc::Int, Ndst::Int
) where {T}
    fill!(out, zero(T))
    Ndst == Nsrc && return copyto!(out, v)
    keep = Ndst ÷ 2
    @inbounds for k in 1:Nsrc
        m = _signedfrequency(k, Nsrc)
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
