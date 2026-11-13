# FFT-FFT backend: global spectral resampling in both angular coordinates.
#
# Once the torus extension exists this is conceptually just "evaluate the same Fourier series at more
# points", which zero-padding does exactly. Three things make it fiddly enough to keep in one tested
# routine rather than spread around, and every one of them is a silent-wrong-answer if got wrong:
#
#   1. THE MIDPOINT OFFSET. Samples sit at `(q + 1/2)h`, not `qh`, so the DFT of the samples is not the
#      Fourier coefficient: `Ĝ_m = F̂_m e^{imδ}` with `δ = h/2`. Blindly doing FFT → pad → IFFT rotates
#      the whole pattern by half a cell. φ needs no such term: `φ_j = 2πj/Nφ` has offset zero.
#
#      THE TWO HALVES OF THAT CORRECTION ARE APPLIED SEPARATELY, on either side of the coefficient
#      move, rather than fused into one `e^{im(δ_f - δ_c)}` pass before it. Fusing is valid only while
#      the move preserves frequency, and the Nyquist split is precisely where it does not: the half
#      that lands on `-N/2` would carry the phase of `+N/2`. Splitting the pass costs one extra
#      diagonal multiply and removes a whole class of "correct except at the top bin" bug.
#   2. SIGNED-FREQUENCY PADDING. Coefficients must be placed by the frequency they REPRESENT, not by
#      copying leading and trailing blocks. `_signedfrequencies` makes that explicit.
#   3. NORMALIZATION. Coefficients are kept normalized (`fft(F)/N`) so they do not depend on the number
#      of samples, and the inverse multiplies back up. Otherwise the error is a level-dependent
#      amplitude drift, which looks like a physics bug.

"""
    FFTBackend()

Global spectral interpolation and restriction on an `EquidistantSampling`. Exact for functions whose
content is resolved by the source grid; the residual is floating-point, not truncation.
"""
struct FFTBackend end

# Signed frequency of each 1-based DFT index for a length-`N` transform, in the standard ordering
# `0, 1, …, ⌊N/2⌋, -⌊(N-1)/2⌋, …, -1`.
_signedfrequencies(N::Int) = [k <= N ÷ 2 ? k : k - N for k in 0:(N - 1)]

"""
    _signedfrequency(index, N)

The signed DFT frequency at 1-based `index`, without materializing the whole table.

Returns `_signedfrequencies(N)[index]` without allocating a frequency table in the buffered loops.
"""
@inline _signedfrequency(index::Int, N::Int) =
    (index - 1) <= N ÷ 2 ? (index - 1) : (index - 1 - N)

"""
    resamplecoefficients(Ĝ, Nsrc, Ndst) -> Ĝ′

Move normalized DFT coefficients from a length-`Nsrc` grid to a length-`Ndst` one, matching SIGNED
frequencies. Padding (`Ndst > Nsrc`) is interpolation; truncation (`Ndst < Nsrc`) is an ideal spectral
low-pass, which is the anti-aliasing filter a restriction needs -- there is no separate filter step.

THE NYQUIST BIN IS SPLIT IN HALF between `+N/2` and `-N/2` when upsampling an even-length transform,
and the two halves are recombined when downsampling. That single bin is genuinely ambiguous in sign,
and splitting it is the choice that makes the result the band-limited trigonometric interpolant rather
than one arbitrarily-rotated cousin of it. By construction MLFMA keeps content well below Nyquist
(`Nθ ≥ L+1`, `Nφ ≥ 2(L+1)`), so this bin should be negligible -- but "should be negligible" is a
reason to handle it correctly and cheaply, not a reason to guess.
"""
function resamplecoefficients(Ĝ::AbstractVector{T}, Nsrc::Int, Ndst::Int) where {T}
    return resamplecoefficients!(zeros(T, Ndst), Ĝ, Nsrc, Ndst)
end

"""
    resamplecoefficients!(Ĝ′, Ĝ, Nsrc, Ndst) -> Ĝ′

[`resamplecoefficients`](@ref) writing into a caller-owned `Ĝ′`, which it zeroes first.
Reuse this buffer across rows and columns. Clearing it here prevents the Nyquist accumulation from
including values left by an earlier call.
"""
function resamplecoefficients!(
    Ĝ′::AbstractVector{T}, Ĝ::AbstractVector, Nsrc::Int, Ndst::Int
) where {T}
    fill!(Ĝ′, zero(T))
    Ndst == Nsrc && return copyto!(Ĝ′, Ĝ)
    keep = Ndst ÷ 2                     # highest |m| representable on the destination
    for k in 1:Nsrc
        m = _signedfrequency(k, Nsrc)
        absm = abs(m)
        absm > keep && continue          # truncated: the low-pass
        if iseven(Nsrc) && absm == Nsrc ÷ 2 && Ndst > Nsrc
            # split the source Nyquist across ±m on the finer grid
            Ĝ′[_indexof(m, Ndst)] += Ĝ[k] / 2
            Ĝ′[_indexof(-m, Ndst)] += Ĝ[k] / 2
        elseif iseven(Ndst) && absm == keep && Ndst < Nsrc
            # both source ±keep fold onto the destination's single Nyquist bin
            Ĝ′[_indexof(keep, Ndst)] += Ĝ[k]
        else
            Ĝ′[_indexof(m, Ndst)] += Ĝ[k]
        end
    end
    return Ĝ′
end

_indexof(m::Int, N::Int) = mod(m, N) + 1

"""
    resamplereflectedcoefficients(Ĝ, Nsrc, Ndst) -> Ĝ′

[`resamplecoefficients`](@ref) for a sequence living on a HALF-SAMPLE-REFLECTED torus -- which θ
always is, and which φ never is.

The top bin is a sine mode, not a symmetric pair. `resamplecoefficients` splits it evenly across
`±N/2`, which reconstructs a COS mode; on a reflected torus that mode is identically zero at every
sample (`cos(π(n+1/2)) == 0`), so the content actually sitting in the bin is the SIN mode and an even
split would reconstruct a different function. Negate the half landing on `-N/2` when upsampling,
and subtract the two source halves when downsampling. This preserves the torus reflection even
when the top bin contains data.

CALLERS MUST PASS TRUE FOURIER COEFFICIENTS, not midpoint-convention ones. This routine moves content
BETWEEN frequencies, so a phase attached per SOURCE frequency would follow that content to the wrong
destination. [`resample`](@ref) divides the source convention out before the move and applies the
destination's after it, for this reason.
"""
function resamplereflectedcoefficients(Ĝ::AbstractVector{T}, Nsrc::Int, Ndst::Int) where {T}
    return resamplereflectedcoefficients!(zeros(T, Ndst), Ĝ, Nsrc, Ndst)
end

"""
    resamplereflectedcoefficients!(Ĝ′, Ĝ, Nsrc, Ndst) -> Ĝ′

[`resamplereflectedcoefficients`](@ref) writing into a caller-owned `Ĝ′`, which it ZEROES first.
Same reason as [`resamplecoefficients!`](@ref): it is called once per destination row.
"""
function resamplereflectedcoefficients!(
    Ĝ′::AbstractVector{T}, Ĝ::AbstractVector, Nsrc::Int, Ndst::Int
) where {T}
    fill!(Ĝ′, zero(T))
    Ndst == Nsrc && return copyto!(Ĝ′, Ĝ)
    keep = Ndst ÷ 2
    for k in 1:Nsrc
        m = _signedfrequency(k, Nsrc)
        absm = abs(m)
        absm > keep && continue
        if iseven(Nsrc) && absm == Nsrc ÷ 2 && Ndst > Nsrc
            # half to +m and MINUS half to -m: the antisymmetric pair, not the symmetric one
            Ĝ′[_indexof(m, Ndst)] += Ĝ[k] / 2
            Ĝ′[_indexof(-m, Ndst)] -= Ĝ[k] / 2
        elseif iseven(Ndst) && absm == keep && Ndst < Nsrc
            # both source halves fold onto the single destination top bin, antisymmetrically
            Ĝ′[_indexof(keep, Ndst)] += m < 0 ? -Ĝ[k] : Ĝ[k]
        else
            Ĝ′[_indexof(m, Ndst)] += Ĝ[k]
        end
    end
    return Ĝ′
end

"""
    resample(F, src, dst, parity, ::FFTBackend) -> Nφ(dst) × Nθ(dst) array

Spectral resampling of one component between two `EquidistantSampling`s, in either direction. Extends
to the torus, transforms, moves coefficients by signed frequency with the midpoint phase correction in
θ, inverts, and keeps the physical columns.

Deliberately the EXPLICIT doubled-grid form: it builds the whole `Nφ × 2Nθ` extension rather than
exploiting the symmetry. That is the reference implementation any optimized version gets checked
against, so it is written to be obviously right.
"""
function resample(
    F::AbstractMatrix,
    src::EquidistantSampling,
    dst::EquidistantSampling,
    parity::Int,
    ::FFTBackend,
)
    E = extend(complex(float.(F)), src, parity)
    nφ_src, nθe_src = size(E)          # arrays are [φ, θ]: φ contiguous
    nθe_dst, nφ_dst = 2 * dst.Nθ, dst.Nφ

    # φ first (no offset, so no phase term), then θ (midpoint offset, so a phase term).
    Ĝ = fft(E) ./ (nθe_src * nφ_src)

    # θ midpoint correction, FIRST HALF: divide out the source's own `e^{imδ}` so what gets moved is
    # the true Fourier coefficient. δ = h/2 = π/(2Nθ). θ is the SECOND dimension.
    δsrc = π / (2 * src.Nθ)
    δdst = π / (2 * dst.Nθ)
    @inbounds for (k, m) in enumerate(_signedfrequencies(nθe_src))
        c = cis(-m * δsrc)
        for j in axes(Ĝ, 1)
            Ĝ[j, k] *= c
        end
    end

    padded = zeros(eltype(Ĝ), nφ_dst, nθe_dst)
    # separable: resample φ (dim 1) for each source θ mode, then θ (dim 2) for each destination φ.
    # φ is a plain torus; θ is a reflected one, and the two move their top bin differently.
    tmp = zeros(eltype(Ĝ), nφ_dst, nθe_src)
    for k in 1:nθe_src
        tmp[:, k] .= resamplecoefficients(view(Ĝ, :, k), nφ_src, nφ_dst)
    end
    for j in 1:nφ_dst
        padded[j, :] .= resamplereflectedcoefficients(view(tmp, j, :), nθe_src, nθe_dst)
    end

    # SECOND HALF: put the coefficients into the DESTINATION's midpoint convention, by the frequency
    # each one now represents rather than the one it arrived as.
    @inbounds for (k, m) in enumerate(_signedfrequencies(nθe_dst))
        c = cis(m * δdst)
        for j in axes(padded, 1)
            padded[j, k] *= c
        end
    end

    Efine = ifft(padded) .* (nθe_dst * nφ_dst)
    return Efine[:, 1:(dst.Nθ)]
end
