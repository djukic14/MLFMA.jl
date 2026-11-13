# Spectral resampling between two CircularSampling grids.
#
# A circular far field is periodic in φ, sampled at `φ_j = 2πj/N`. Zero-padding its DFT evaluates
# the same Fourier interpolant on a finer grid. There is no local stencil order to tune:
# `stencilsize == 0` denotes global interpolation, as in the equidistant FFT backend.
#
# ------------------------------------------------------------------------------------------------
# Continuation Traits
#
# A local θ stencil on a spherical grid can reach
# past the pole, and continuing past the pole lands at `φ + π` on the other side -- so every
# component needs a rule for the sign it picks up under that reflection. `VirtualExtension` says "no
# sign change", `VirtualExtensionFlipθSign` says "negate", and the composite traits are per-component
# lists of those two.
#
# A circle has no pole, θ coordinate, or reflection. Its only continuation is `f(φ) = f(φ + 2π)`,
# with no sign change. `VirtualExtension()` instead describes continuation across a spherical pole.
#
# The trait separates geometry checks from per-component arithmetic:
#
#   * `MLFMA.PeriodicφInterpolation()` describes one periodic component for `BEAST.HelmholtzOperator2D`.
#     It has no spherical trait conversions or `Equidistant.componentparities` entry, so a
#     spherical interpolator given a 2-D operator's trait raises `MethodError` rather than applying
#     a pole rule.
#   * The circular apply checks only the trait's arity: how many
#     components the caller says the field has -- and applies the identical spectral resample to each
#     of them. Every component on the circle uses the same periodic continuation.
#
# This family accepts any trait of matching arity, including spherical traits, without applying
# their pole signs. Matrix diagnostics use this with TangentialSphericalFarField probes.
#
# ------------------------------------------------------------------------------------------------
# Forward And Adjoint Operators
#
# With `F_N` the length-`N` DFT matrix and `P` the coefficient move (below), interpolation from a
# child grid of `N` directions to a parent grid of `M` is
#
#     R = (1/N) F_Mᴴ P F_N              i.e.   g = ifft(P · fft(f)) · M/N
#
# `P` is real (its only non-unit entries are the halves of a split Nyquist bin), so
#
#     Rᴴ = (1/N) F_Nᴴ Pᵀ F_M = IFFT_N ∘ Pᵀ ∘ FFT_M
#
# No extra scaling remains. `_movecircularcoefficientsadjoint!` applies `Pᵀ`; together with the
# transforms, it gives the exact adjoint in exact arithmetic. The contract tests check its matrix.
#
# The coefficient move preserves conjugate frequency pairs, making R real in exact arithmetic.
# Thus `isrealinterpolation` is
# `true` here and both `transposedinterpolator` and `conjugatedinterpolator` correctly decline to wrap
# it. `test/LevelTransfers/circularinterpolation.jl` checks this declaration against the matrix.
#
# ------------------------------------------------------------------------------------------------
# Nyquist Bin
#
# `NφCircular(L) = 2(L+1)` puts a degree-`L` pattern's highest mode at `|m| = L = N/2 - 1`, exactly
# one below the Nyquist bin `N/2`. A degree-L pattern leaves that bin empty, but arbitrary samples
# need not. Upsampling splits the source bin evenly across `±N/2` on the destination grid;
# downsampling folds the two frequencies into the destination Nyquist bin. This preserves a real
# interpolation operator, allowing the wrappers above to skip conjugation.
#
# This kernel duplicates the plain-torus coefficient moves in Equidistant because Interpolation
# loads first and does not depend on that submodule. `test/LevelTransfers/circularinterpolation.jl`
# checks bitwise agreement across source and destination sizes.

"""
    CircularInterpolator{T}(; interpolationtrait=DefaultInterpolation())
    CircularInterpolator{0,T}(samplings, parentsamplings; interpolationtrait=...)

Spectral interpolation between two [`CircularSampling`](@ref) grids, in MLFMA's interpolator protocol.

Exact for any pattern the source grid resolves: the residual is floating point, not truncation. The
`stencilsize` parameter is `0`, meaning global -- every source sample contributes to every
destination sample, so there is no order to choose.

The second form is the shape the MAP CONSTRUCTORS use (`interpolatortype(strategy)` yields the type
and calls it with the level-pair vectors). The samplings are validated and then discarded: this family
caches nothing per level pair, so the pairs serve only to catch a grid mismatch at construction rather
than at the first apply. Its FFT plans live in the per-(level, chunk) scratch instead -- see
[`MLFMA.Interpolation.interimscratch`](@ref) -- so concurrent chunks never share one.

The stored `interpolationtrait` is used for construction checks, not apply-time arithmetic.
Map constructors compare it with the operator's trait and reject a mismatch. Each call checks
the per-call trait's arity, then applies the same resampling to every component.
"""
struct CircularInterpolator{stencilsize,T} <:
       Interpolation.AbstractInterpolator{stencilsize,T}
    # An abstract field keeps `CircularInterpolator{0,ComplexF64}` concrete and callable by strategies.
    # The stored trait is read during construction, not during an apply.
    interpolationtrait::Interpolation.InterpolationExpansionTrait
end

function CircularInterpolator{T}(; interpolationtrait=DefaultInterpolation()) where {T}
    return CircularInterpolator{0,T}(interpolationtrait)
end

CircularInterpolator(T::DataType; interpolationtrait=DefaultInterpolation()) =
    CircularInterpolator{T}(; interpolationtrait=interpolationtrait)

function CircularInterpolator{stencilsize,T}(
    samplings::AbstractVector,
    parentsamplings::AbstractVector;
    interpolationtrait=DefaultInterpolation(),
) where {stencilsize,T}
    length(samplings) == length(parentsamplings) || error(
        "CircularInterpolator: got $(length(samplings)) child samplings but " *
        "$(length(parentsamplings)) parent samplings; they are level PAIRS.",
    )
    for s in Iterators.flatten((samplings, parentsamplings))
        s isa Sampling.CircularSampling || error(
            "CircularInterpolator: this family interpolates between CircularSampling grids, got " *
            "$(typeof(s)). Pass `samplingbuilder=MLFMA.CircularSamplingBuilder{...}()` in the " *
            "SamplingStrategy so the level grids match the interpolator.",
        )
    end
    return CircularInterpolator{stencilsize,T}(interpolationtrait)
end

# Per-family arity check, in the same spirit as `Equidistant.checkarity`: a trait of the wrong arity is
# a caller error, and without this it surfaces from inside the component loop as a `BoundsError` naming
# neither the trait nor the far field.
#
# This family only needs the trait's component count, so this check covers the
# trait handling here rather than a preliminary to a parity table. See this file's header.
#
# `DefaultInterpolation` carries no arity of its own -- it means "whatever this far field's default
# is" -- so it passes by definition rather than by measurement.
@inline _checkcirculararity(::Interpolation.DefaultInterpolation, ::Type) = nothing

@inline function _checkcirculararity(trait, ::Type{F}) where {F}
    arity = Interpolation.traitarity(trait)
    arity == FarFields.fielddim(F) && return nothing
    return error(
        "interpolation trait arity mismatch: $(nameof(typeof(trait))) describes $arity " *
        "component(s), but $(nameof(F)) has $(FarFields.fielddim(F)). Pass a trait of matching " *
        "arity, or omit `interpolationtrait` to use " *
        "$(nameof(typeof(Interpolation.defaultinterpolationtrait(F)))).",
    )
end

"""
    CircularInterimScratch(childsampling, parentsampling, T)

One level pair's worth of spectral scratch: the two working spectra and four IN-PLACE FFT plans.

There is deliberately no staging vector for `prescale`. Both directions copy their input into a
spectrum buffer before transforming it, so the anterpolator's diagonal is applied during a copy that
had to happen anyway -- unlike the `LocalθGlobalφ` family, which transforms the caller's array in
place and therefore needs a private copy to stage into.

Built per (level, chunk) by [`MLFMA.Interpolation.interimscratch`](@ref) rather than stored on the interpolator, so
each concurrent task owns its own plans and buffers. That is the same arrangement
`Equidistant.FFTScratch` has, for the same reason: the interpolator lives on the read-only core, and a
plan applied in place is not something two chunks may share.

Both directions of the pair are served by one object -- interpolation runs child → parent and
anterpolation parent → child, so between them they need exactly these two spectra.
"""
struct CircularInterimScratch{T,FC,IC,FP,IP}
    childspectrum::Vector{T}
    parentspectrum::Vector{T}
    fftchild::FC
    ifftchild::IC
    fftparent::FP
    ifftparent::IP
end

function CircularInterimScratch(
    childsampling::Sampling.CircularSampling,
    parentsampling::Sampling.CircularSampling,
    ::Type{T},
) where {T}
    childspectrum = Vector{T}(undef, length(childsampling))
    parentspectrum = Vector{T}(undef, length(parentsampling))
    return CircularInterimScratch(
        childspectrum,
        parentspectrum,
        plan_fft!(childspectrum),
        plan_ifft!(childspectrum),
        plan_fft!(parentspectrum),
        plan_ifft!(parentspectrum),
    )
end

# Missing scratch, the wrong scratch type, or mismatched spectrum lengths select the allocating path.
_circularscratch(interim, Nchild::Int, Nparent::Int) = nothing

function _circularscratch(interim::NamedTuple, Nchild::Int, Nparent::Int)
    hasproperty(interim, :circularscratch) || return nothing
    scratch = interim.circularscratch
    scratch isa CircularInterimScratch || return nothing
    length(scratch.childspectrum) == Nchild || return nothing
    length(scratch.parentspectrum) == Nparent || return nothing
    return scratch
end

# Buffer and transform accessors, as method pairs rather than branches: a scratch-backed apply carries
# no runtime test, and a standalone one resolves to plain `fft!`/`ifft!` on freshly allocated vectors.
_childvector(::Nothing, N::Int, ::Type{T}) where {T} = Vector{T}(undef, N)
_childvector(scratch::CircularInterimScratch, ::Int, ::Type) = scratch.childspectrum
_parentvector(::Nothing, N::Int, ::Type{T}) where {T} = Vector{T}(undef, N)
_parentvector(scratch::CircularInterimScratch, ::Int, ::Type) = scratch.parentspectrum

_fftchild!(v, ::Nothing) = fft!(v)
_fftchild!(v, scratch::CircularInterimScratch) = (scratch.fftchild * v; v)
_ifftchild!(v, ::Nothing) = ifft!(v)
_ifftchild!(v, scratch::CircularInterimScratch) = (scratch.ifftchild * v; v)
_fftparent!(v, ::Nothing) = fft!(v)
_fftparent!(v, scratch::CircularInterimScratch) = (scratch.fftparent * v; v)
_ifftparent!(v, ::Nothing) = ifft!(v)
_ifftparent!(v, scratch::CircularInterimScratch) = (scratch.ifftparent * v; v)

"""
    _signedcircularfrequency(index, N)

The signed DFT frequency at 1-based `index` for a length-`N` transform, in the standard ordering
`0, 1, …, ⌊N/2⌋, -⌊(N-1)/2⌋, …, -1`.
"""
@inline _signedcircularfrequency(index::Int, N::Int) =
    (index - 1) <= N ÷ 2 ? (index - 1) : (index - 1 - N)

@inline _circularindexof(m::Int, N::Int) = mod(m, N) + 1

"""
    _movecircularcoefficients!(ĝ′, ĝ, Nsrc, Ndst) -> ĝ′

Move DFT coefficients from a length-`Nsrc` grid to a length-`Ndst` one, matching SIGNED frequencies,
writing into a caller-owned `ĝ′` which it ZEROES first.

Padding (`Ndst > Nsrc`) is interpolation; truncation (`Ndst < Nsrc`) is an ideal spectral low-pass,
which is the anti-aliasing a restriction needs -- there is no separate filter step. The source Nyquist
bin is split evenly across `±N/2` going up and folded back coming down; see this file's header for why
that bin is handled at all when MLFMA's own rate rule keeps it empty.

The zeroing is not the caller's job: the `+=` accumulation is what makes the split work, so a caller
that forgot would pick up the previous component's contents.
"""
function _movecircularcoefficients!(
    ĝ′::AbstractVector{T}, ĝ::AbstractVector, Nsrc::Int, Ndst::Int
) where {T}
    fill!(ĝ′, zero(T))
    Ndst == Nsrc && return copyto!(ĝ′, ĝ)
    keep = Ndst ÷ 2
    @inbounds for k in 1:Nsrc
        m = _signedcircularfrequency(k, Nsrc)
        absm = abs(m)
        absm > keep && continue
        if iseven(Nsrc) && absm == Nsrc ÷ 2 && Ndst > Nsrc
            ĝ′[_circularindexof(m, Ndst)] += ĝ[k] / 2
            ĝ′[_circularindexof(-m, Ndst)] += ĝ[k] / 2
        elseif iseven(Ndst) && absm == keep && Ndst < Nsrc
            ĝ′[_circularindexof(keep, Ndst)] += ĝ[k]
        else
            ĝ′[_circularindexof(m, Ndst)] += ĝ[k]
        end
    end
    return ĝ′
end

"""
    _movecircularcoefficientsadjoint!(x̂, ŷ, Nsrc, Ndst) -> x̂

The exact TRANSPOSE of [`_movecircularcoefficients!`](@ref): consumes length-`Ndst` coefficients and
produces length-`Nsrc` ones, with `Nsrc`/`Ndst` still naming the FORWARD direction.

Written as an assignment per source bin rather than an accumulation, because the transpose of a map
that sends each source bin to one or two destinations is a gather. Dropped bands assign zero
explicitly, which is what makes this a total transpose rather than one that leaves stale content in
the truncated modes.

The move's weights are real, so no conjugation appears here and this is `Pᵀ = Pᴴ` -- which is what
makes the anterpolator built on it the exact adjoint of the interpolator.
"""
function _movecircularcoefficientsadjoint!(
    x̂::AbstractVector{T}, ŷ::AbstractVector, Nsrc::Int, Ndst::Int
) where {T}
    Ndst == Nsrc && return copyto!(x̂, ŷ)
    keep = Ndst ÷ 2
    @inbounds for k in 1:Nsrc
        m = _signedcircularfrequency(k, Nsrc)
        absm = abs(m)
        if absm > keep
            x̂[k] = zero(T)
        elseif iseven(Nsrc) && absm == Nsrc ÷ 2 && Ndst > Nsrc
            x̂[k] = (ŷ[_circularindexof(m, Ndst)] + ŷ[_circularindexof(-m, Ndst)]) / 2
        elseif iseven(Ndst) && absm == keep && Ndst < Nsrc
            x̂[k] = ŷ[_circularindexof(keep, Ndst)]
        else
            x̂[k] = ŷ[_circularindexof(m, Ndst)]
        end
    end
    return x̂
end

"""
    (interpolator::CircularInterpolator)(interpolatedfarfield, farfield, farfieldsampling,
                                        interpolationsampling; interpolationtrait, interim, postscale)

Resample every component of `farfield` from `farfieldsampling` onto `interpolationsampling`, writing
into `interpolatedfarfield`.

`postscale`, when given, is a diagonal applied to the result in the same pass that writes it -- the
same fused contract the other families offer, so a level transfer's phase costs no extra sweep.
"""
function (interpolator::CircularInterpolator{stencilsize,T})(
    interpolatedfarfield::F,
    farfield::F,
    farfieldsampling::Sampling.CircularSampling,
    interpolationsampling::Sampling.CircularSampling;
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
    interim=nothing,
    postscale=nothing,
) where {stencilsize,T,F<:FarFields.AbstractFarField}
    matrices = H2Factory.moments(farfield)
    resultmatrices = H2Factory.moments(interpolatedfarfield)

    @assert length(farfieldsampling) == length(first(matrices))
    @assert length(interpolationsampling) == length(first(resultmatrices))
    _checkcirculararity(interpolationtrait, F)

    N = length(farfieldsampling)
    M = length(interpolationsampling)
    S = eltype(first(resultmatrices))

    scratch = _circularscratch(interim, N, M)
    childspectrum = _childvector(scratch, N, S)
    parentspectrum = _parentvector(scratch, M, S)

    for component in eachindex(matrices)
        copyto!(childspectrum, matrices[component])
        _fftchild!(childspectrum, scratch)
        _movecircularcoefficients!(parentspectrum, childspectrum, N, M)
        _ifftparent!(parentspectrum, scratch)
        # `M / N` is the normalization the pair `fft` (unnormalized) / `ifft` (1/M) leaves over; see
        # the operator written out in this file's header.
        _writecircularscaled!(resultmatrices[component], parentspectrum, M / N, postscale)
    end

    return nothing
end

# The result write, with the optional diagonal folded into the same pass. Two methods rather than a
# branch, so an unscaled apply compiles to a plain scaled copy with nothing to predict -- the same
# shape as `_scaleresult!` in the local/global family, and for the same reason. Named apart from that
# one because it takes a separate source: the spectral pipeline lands in the scratch and copies out,
# where the local/global φ stage rescales the result in place.
#
# `(v * scale) * coefficient`, NOT `v * (scale * coefficient)`: those differ in the last bit, and the
# first is what the separate two-pass form computes, which is what lets the fused path be compared with
# `==` rather than `≈`.
function _writecircularscaled!(result, source, scale, ::Nothing)
    @inbounds for i in eachindex(result, source)
        result[i] = source[i] * scale
    end
    return result
end

function _writecircularscaled!(result, source, scale, postscale)
    @inbounds for i in eachindex(result, source, postscale)
        result[i] = (source[i] * scale) * postscale[i]
    end
    return result
end

# The vector-of-far-fields entry point the abstract interpolator declares. This family has no batched
# form, and saying so names the reason rather than failing with a `MethodError` several frames away.
function (interpolator::CircularInterpolator)(
    farfields::AbstractArray{F,1}, interpolatedfarfields::AbstractArray{F,1}, args...
) where {F<:FarFields.AbstractFarField}
    return error(
        "Not implemented: this interpolator uses the special structure of CircularSampling"
    )
end
