# MLFMA's interpolator/anterpolator PROTOCOL, implemented over the two equidistant backends.
#
# Everything else in this submodule speaks in bare matrices: `resample(F, src, dst, parity, backend)`
# takes an `Nφ × Nθ` array and a parity and returns another array. This file is the adapter to what the
# rest of MLFMA actually calls -- a mutating functor on far-field MOMENTS, which carry several
# components at once and decide their own sign behaviour through an `InterpolationExpansionTrait`.
#
# `EquidistantAnterpolator` is the bare Euclidean
# adjoint (`resampleadjoint`), NOT the quadrature-weighted adjoint (`anterpolate`) that lives in
# `anterpolation.jl`. Those are different operators and both are correct -- for different callers:
#
#   * `anterpolate` = `W_c⁻¹ Rᴴ W_f` is the adjoint in the inner product `∮ u v̄ dΩ`. It is what a
#     standalone two-level kernel wants, because there the coarse-grid result is integrated against a
#     receive pattern with the COARSE weights, so the weight change has to happen inside the transfer.
#   * MLFMA's protocol wants `Rᴴ`. Its contract is asserted literally as `anter ≈ adjoint(inter)` in
#     `test/LevelTransfers/anterpolationcontract.jl`, and it is bare because the quadrature weights are
#     folded into the LEAF receive patterns once (via `cartesiansamplepoints`) rather than being
#     re-applied at every level. Handing the weighted operator to the traversal would apply the weight
#     ratio at every level transfer and change the operator.
#
# `anterpolate` therefore stays exactly where it is and keeps its own tests; it is not deprecated by
# this file and is not reachable from it.
#
# TRAITS. MLFMA already has a vocabulary for the sign a component picks up across the θ seam, and it
# is the same statement as this submodule's `parity` -- just named per far-field representation
# instead of per array. The mapping is one-to-one and mechanical:
#
#     VirtualExtension          <-> SPHERICAL       (no sign flip)
#     VirtualExtensionFlipθSign <-> ANTISPHERICAL   (sign flips)
#
# Composite traits list parities in component order. `SphericalθφInterpolation` is `[flip, flip]`:
# both tangential basis vectors reverse under continuation, as described in `tangential.jl`.
#
# Matching scratch supplies the FFT buffers and plans, or the Lagrange stencils and buffers.
# Missing scratch or incompatible grids/orders selects the allocating reference path. The resampling
# kernels allocate nothing once warmed up; anterpolation with `prescale` still allocates a staging copy.

"""
    componentparities(trait, ::Type{F}) -> NTuple{N,Int}

The per-component continuation parity a far field of type `F` interpolated under `trait` needs, as
`SPHERICAL` / `ANTISPHERICAL` entries in component order.

This is the whole of the translation between MLFMA's `InterpolationExpansionTrait` vocabulary and this
submodule's `parity` argument. `DefaultInterpolation` resolves through
`MLFMA.defaultinterpolationtrait(F)`, which is why the far-field type is a parameter at all; that
function never returns `DefaultInterpolation`, so the recursion terminates in one step.
"""
componentparities(::Interpolation.VirtualExtension, ::Type) = (SPHERICAL,)
componentparities(::Interpolation.VirtualExtensionFlipθSign, ::Type) = (ANTISPHERICAL,)
componentparities(::Interpolation.SphericalθφInterpolation, ::Type) =
    (ANTISPHERICAL, ANTISPHERICAL)
componentparities(::Interpolation.SphericalInterpolation, ::Type) =
    (SPHERICAL, ANTISPHERICAL, ANTISPHERICAL)
componentparities(::Interpolation.DoubleSphericalInterpolation, ::Type) =
    (SPHERICAL, ANTISPHERICAL, ANTISPHERICAL, SPHERICAL, ANTISPHERICAL, ANTISPHERICAL)

function componentparities(::Interpolation.DefaultInterpolation, ::Type{F}) where {F}
    return componentparities(Interpolation.defaultinterpolationtrait(F), F)
end

"""
    checkarity(parities, ::Type{F}, trait)

Reject a trait whose component count differs from `F` before entering the component loop.

Traits carry an arity: `VirtualExtension` is one component, `SphericalθφInterpolation` two,
`SphericalInterpolation` three. The error reports the trait and far-field types and their counts.
"""
@inline function checkarity(parities, ::Type{F}, trait) where {F}
    length(parities) == FarFields.fielddim(F) && return nothing
    return error(
        "interpolation trait arity mismatch: $(nameof(typeof(trait))) describes " *
        "$(length(parities)) component(s), but $(nameof(F)) has $(FarFields.fielddim(F)). " *
        "Pass a trait of matching arity, or omit `interpolationtrait` to use " *
        "$(nameof(typeof(Interpolation.defaultinterpolationtrait(F)))).",
    )
end

"""
    EquidistantInterpolator{T}(backend; interpolationtrait=MLFMA.DefaultInterpolation())

Interpolation between two [`EquidistantSampling`](@ref) grids, in MLFMA's interpolator protocol,
delegating to `backend`.

`T` is the moment element type. One type serves both backends because they differ only in the object
that `resample` dispatches on -- which is also what `isrealinterpolation` keys off, so the two families
stay distinguishable where it matters without a second struct.

The `stencilsize` type parameter (MLFMA's, reported by `MLFMA.stencilsize`) is the Lagrange stencil
order for [`LagrangeBackend`](@ref) and **0** for [`FFTBackend`](@ref), 0 meaning "global -- every
source sample contributes". For an anisotropic `LagrangeBackend{Pθ,Pφ}` it is `max(Pθ, Pφ)`: the
protocol's slot holds one number, and the backend itself keeps both orders, so nothing is lost except
in that one reported figure.

Each call selects parity through its `interpolationtrait` keyword, which defaults to the far field's
`defaultinterpolationtrait`. The stored trait does not control an apply. Map constructors compare it
with the operator's trait and reject a mismatch in component-sign conventions.
"""
struct EquidistantInterpolator{stencilsize,T,B} <:
       Interpolation.AbstractInterpolator{stencilsize,T}
    backend::B
    # An abstract field keeps `EquidistantInterpolator{S,T,B}` concrete and callable by strategies.
    # Construction reads the stored trait; applies use the per-call keyword instead.
    interpolationtrait::Interpolation.InterpolationExpansionTrait
end

function EquidistantInterpolator{T}(
    backend::FFTBackend; interpolationtrait=Interpolation.DefaultInterpolation()
) where {T}
    return EquidistantInterpolator{0,T,typeof(backend)}(backend, interpolationtrait)
end

function EquidistantInterpolator{T}(
    backend::LagrangeBackend{Pθ,Pφ}; interpolationtrait=Interpolation.DefaultInterpolation()
) where {T,Pθ,Pφ}
    return EquidistantInterpolator{max(Pθ, Pφ),T,typeof(backend)}(
        backend, interpolationtrait
    )
end

EquidistantInterpolator(
    T::DataType, backend; interpolationtrait=Interpolation.DefaultInterpolation()
) = EquidistantInterpolator{T}(backend; interpolationtrait=interpolationtrait)

"""
    EquidistantInterpolator{stencilsize,T,B}(samplings, parentsamplings; interpolationtrait=...)

The construction shape the MAP CONSTRUCTORS use: `interpolatortype(strategy)` yields this type and
calls it with the level-pair vectors.

The samplings are validated and then discarded. They catch grid mismatches at construction;
per-level buffers, plans, and stencils belong to scratch, not to this interpolator.
Both backends are singletons, so the backend is recovered from `B` rather than passed.

The trait passed here IS stored: the map constructors compare `MLFMA.interpolationtrait(interpolator)`
against the operator's and `error()` on a mismatch, so a constructor that dropped it would fail that
check with `DefaultInterpolation` against, say, `SphericalθφInterpolation`.
"""
function EquidistantInterpolator{stencilsize,T,B}(
    samplings::AbstractVector,
    parentsamplings::AbstractVector;
    interpolationtrait=Interpolation.DefaultInterpolation(),
) where {stencilsize,T,B}
    length(samplings) == length(parentsamplings) || error(
        "EquidistantInterpolator: got $(length(samplings)) child samplings but " *
        "$(length(parentsamplings)) parent samplings; they are level PAIRS.",
    )
    for s in Iterators.flatten((samplings, parentsamplings))
        s isa EquidistantSampling || error(
            "EquidistantInterpolator: this family interpolates between EquidistantSampling " *
            "grids, got $(typeof(s)). Pass `samplingbuilder=EquidistantSamplingBuilder{...}()` " *
            "in the SamplingStrategy so the level grids match the interpolator.",
        )
    end
    return EquidistantInterpolator{stencilsize,T,B}(B(), interpolationtrait)
end

backend(interpolator::EquidistantInterpolator) = interpolator.backend

"""
    EquidistantAnterpolator(interpolator)

The ADJOINT of an [`EquidistantInterpolator`](@ref) -- `Rᴴ`, bare, with no quadrature weights. See
this file's header for why that is the right operator here and `anterpolate` is not.
"""
struct EquidistantAnterpolator{stencilsize,T,I} <:
       Interpolation.AbstractAnterpolator{stencilsize,T}
    interpolator::I
end

function EquidistantAnterpolator(
    interpolator::EquidistantInterpolator{stencilsize,T}
) where {stencilsize,T}
    return EquidistantAnterpolator{stencilsize,T,typeof(interpolator)}(interpolator)
end

# Map constructors pass `nchunks` to every family. This family accepts but ignores it:
# plans and buffers are built per (level, chunk) by `interimscratch`.
function Interpolation.anterpolator(
    interpolator::EquidistantInterpolator; nchunks::Int=Threads.nthreads()
)
    return EquidistantAnterpolator(interpolator)
end

backend(anterpolator::EquidistantAnterpolator) = backend(anterpolator.interpolator)

# MEASURED, not reasoned -- see the note in `MLFMA.isrealinterpolation`'s docstring about probing being
# too expensive to do at construction, and `test/Equidistant/interpolation.jl` for the assertions that
# hold these to the actual operator.
#
# Lagrange: `max|imag(R)| == 0` exactly, at p = 4 and p = 6 and under both parities. Its weights are
# real barycentric values and its adjoint is a plain transpose, so transpose and adjoint are one
# operator and `transposedinterpolator` correctly declines to wrap it.
#
# FFT: ALSO REAL -- `max|imag(R)| / max|real(R)| ≤ 1.8e-16`, i.e. FFTW round-off. Measured over ten
# source/destination shape pairs under both parities: up, down, equal, and odd as well as even `Nθ`,
# chosen so the Nyquist bin -- the only place this could break -- is exercised rather than avoided.
# What declaring `true` actually asserts is `transpose(R) == adjoint(R)`, measured at 1.1e-16 … 3.3e-16
# relative, which is what lets `transposedinterpolator` decline to wrap the reverse direction.
#
# The trait describes real values in exact arithmetic, not bitwise equality. Both consumers
# (`conjugatedinterpolation.jl`, `transposedinterpolation.jl`) use it for exactly one decision -- may
# conjugation be skipped -- and skipping it perturbs the result by the size of the imaginary part.
# `test/Equidistant/interpolation.jl` therefore carries `declaresreal` and `exactlyreal` as separate
# columns: this family is `true`/`false`, the Lagrange one `true`/`true`.
#
# The reflected θ Nyquist split must preserve this real operator; see `fftbackend.jl`.
Interpolation.isrealinterpolation(
    ::EquidistantInterpolator{S,T,<:LagrangeBackend}
) where {S,T} = true
Interpolation.isrealinterpolation(::EquidistantInterpolator{S,T,<:FFTBackend}) where {S,T} =
    true

# The FFT family's scratch: one `FFTScratch` per level pair, holding every stage buffer and its
# in-place plans. Built here rather than on the interpolator because `interimscratch` runs once per
# (level, chunk), so each concurrent chunk gets private plans and buffers -- see `fftscratch.jl`.
#
# Dispatches on the ANTERPOLATOR, as the protocol requires, and reads the BACKEND off the interpolator
# it wraps, so one method covers both directions of the pair.
function Interpolation.interimscratch(
    anterpolator::EquidistantAnterpolator{S,T,<:EquidistantInterpolator{S,T,<:FFTBackend}},
    childsampling::EquidistantSampling,
    parentsampling::EquidistantSampling,
) where {S,T}
    return (fftscratch=FFTScratch(childsampling, parentsampling, complex(float(T))),)
end

# Lagrange scratch caches stencils as well as buffers. Read the orders from the wrapped interpolator.
function Interpolation.interimscratch(
    anterpolator::EquidistantAnterpolator{
        S,T,<:EquidistantInterpolator{S,T,<:LagrangeBackend}
    },
    childsampling::EquidistantSampling,
    parentsampling::EquidistantSampling,
) where {S,T}
    backend = anterpolator.interpolator.backend
    return (
        lagrangescratch=LagrangeScratch(
            childsampling, parentsampling, backend, complex(float(T))
        ),
    )
end

Interpolation.interimscratch(::EquidistantAnterpolator, childsampling, parentsampling) =
    nothing

# Reads the scratch tolerantly, exactly as `MLFMA._interimmatrix` and friends do: a shape mismatch, a
# hand-built NamedTuple, or no scratch at all falls back to the allocating reference path rather than
# writing out of bounds. That fallback is what keeps every standalone caller working unchanged.
# Dispatching the LOOKUP on the backend, not just the kernel, so a `LagrangeScratch` can never be
# handed to a spectral apply (or the reverse) -- that would be a MethodError deep inside the kernel
# rather than the graceful fallback this layer promises.
_backendscratch(interim, ::FFTBackend, src, dst) = _fftscratch(interim, src, dst)
_backendscratch(interim, backend::LagrangeBackend, src, dst) =
    _lagrangescratch(interim, backend, src, dst)

_lagrangescratch(
    interim, ::LagrangeBackend, src::EquidistantSampling, dst::EquidistantSampling
) = nothing

function _lagrangescratch(
    interim::NamedTuple,
    backend::LagrangeBackend,
    src::EquidistantSampling,
    dst::EquidistantSampling,
)
    hasproperty(interim, :lagrangescratch) || return nothing
    scratch = interim.lagrangescratch
    scratch isa LagrangeScratch || return nothing
    _lagrangeordersmatch(scratch, backend) || return nothing
    _lagrangegridsmatch(scratch, src, dst) || return nothing
    return scratch
end

_fftscratch(interim, src::EquidistantSampling, dst::EquidistantSampling) = nothing

function _fftscratch(
    interim::NamedTuple, src::EquidistantSampling, dst::EquidistantSampling
)
    hasproperty(interim, :fftscratch) || return nothing
    scratch = interim.fftscratch
    scratch isa FFTScratch || return nothing
    # Buffers use the physical grids, not doubled tori.
    size(scratch.childspectrum) == (src.Nφ, src.Nθ) || return nothing
    size(scratch.parentspectrum) == (dst.Nφ, dst.Nθ) || return nothing
    return scratch
end

"""
    (interpolator::EquidistantInterpolator)(interpolatedfarfield, farfield, farfieldsampling,
                                            interpolationsampling; interpolationtrait, interim, postscale)

Resample every component of `farfield` from `farfieldsampling` onto `interpolationsampling`, writing
into `interpolatedfarfield`.

`postscale`, when given, is a diagonal applied to the result in the same pass that writes it -- the
same fused contract the other families offer, so a level transfer's phase costs no extra sweep.
`interim` supplies backend scratch when its grids and stencil orders match. Otherwise the
call uses the allocating reference path.
"""
function (interpolator::EquidistantInterpolator{stencilsize,T})(
    interpolatedfarfield::F,
    farfield::F,
    farfieldsampling::EquidistantSampling,
    interpolationsampling::EquidistantSampling;
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
    interim=nothing,
    postscale=nothing,
) where {stencilsize,T,F<:FarFields.AbstractFarField}
    @assert Sampling.Nθ(farfieldsampling) == Sampling.Nθ(farfield)
    @assert Sampling.Nφ(farfieldsampling) == Sampling.Nφ(farfield)

    matrices = H2Factory.moments(farfield)
    resultmatrices = H2Factory.moments(interpolatedfarfield)
    parities = componentparities(interpolationtrait, F)
    checkarity(parities, F, interpolationtrait)

    scratch = _backendscratch(
        interim, interpolator.backend, farfieldsampling, interpolationsampling
    )
    for component in eachindex(matrices)
        resampled = _resampled(
            matrices[component],
            farfieldsampling,
            interpolationsampling,
            parities[component],
            interpolator.backend,
            scratch,
        )
        _writescaled!(resultmatrices[component], resampled, postscale)
    end

    return nothing
end

"""
    (anterpolator::EquidistantAnterpolator)(anterpolatedfarfield, farfield, farfieldsampling,
                                            anterpolationsampling; interpolationtrait, interim,
                                            prescale, stagetransform)

Apply the adjoint of interpolation, taking `farfield` on the PARENT grid `farfieldsampling` down to
the CHILD grid `anterpolationsampling`.

`prescale` and `stagetransform` carry the same meaning as in the other families and go through the
same `Interpolation._stageprescaled!`: the diagonal is applied to a private copy of the input.
The caller's moment is not mutated, so concurrent children can read the same parent moment.
`stagetransform=conj` lets `TransposedInterpolator` conjugate the staged product without changing
shared input. A non-identity `stagetransform` requires `prescale`.
"""
function (anterpolator::EquidistantAnterpolator{stencilsize,T})(
    anterpolatedfarfield::F,
    farfield::F,
    farfieldsampling::EquidistantSampling,
    anterpolationsampling::EquidistantSampling;
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
    interim=nothing,
    prescale=nothing,
    stagetransform=identity,
) where {stencilsize,T,F<:FarFields.AbstractFarField}
    @assert Sampling.Nθ(farfieldsampling) == Sampling.Nθ(farfield)
    @assert Sampling.Nφ(farfieldsampling) == Sampling.Nφ(farfield)

    matrices = H2Factory.moments(farfield)
    resultmatrices = H2Factory.moments(anterpolatedfarfield)
    parities = componentparities(interpolationtrait, F)
    checkarity(parities, F, interpolationtrait)

    scratch = _backendscratch(
        interim, anterpolator.interpolator.backend, anterpolationsampling, farfieldsampling
    )
    staging = prescale === nothing ? nothing : similar(matrices[1])

    for component in eachindex(matrices)
        source = Interpolation._stageprescaled!(
            staging, matrices[component], prescale, stagetransform
        )
        # `resampleadjoint` names its samplings in the FORWARD direction -- coarse first, then fine --
        # so the child comes before the parent here even though the data flows the other way.
        anterpolated = _resampledadjoint(
            source,
            anterpolationsampling,
            farfieldsampling,
            parities[component],
            anterpolator.interpolator.backend,
            scratch,
        )
        copyto!(resultmatrices[component], anterpolated)
    end

    return nothing
end

# Buffered when a matching `FFTScratch` arrived, allocating reference otherwise. Two methods rather
# than a branch, so a scratch-backed apply carries no runtime test and a standalone call resolves to
# exactly the code it always ran.
#
# The buffered forms return the scratch's OWN output array, which the caller immediately copies into
# the moment. That is safe because the copy happens before the next component reuses the buffer, and
# it is why `parentgrid`/`childgrid` can be single buffers rather than one per component.
_resampled(F, src, dst, parity, backend, ::Nothing) = resample(F, src, dst, parity, backend)

function _resampled(F, src, dst, parity, backend::FFTBackend, scratch::FFTScratch)
    return resample!(scratch.parentgrid, F, src, dst, parity, backend, scratch)
end

function _resampled(F, src, dst, parity, backend::LagrangeBackend, scratch::LagrangeScratch)
    return resample!(scratch.parentgrid, F, src, dst, parity, backend, scratch)
end

_resampledadjoint(Y, coarse, fine, parity, backend, ::Nothing) =
    resampleadjoint(Y, coarse, fine, parity, backend)

function _resampledadjoint(
    Y, coarse, fine, parity, backend::FFTBackend, scratch::FFTScratch
)
    return resampleadjoint!(scratch.childgrid, Y, coarse, fine, parity, backend, scratch)
end

function _resampledadjoint(
    Y, coarse, fine, parity, backend::LagrangeBackend, scratch::LagrangeScratch
)
    return resampleadjoint!(scratch.childgrid, Y, coarse, fine, parity, backend, scratch)
end

# The result write, with the optional diagonal folded in. Two methods rather than a branch, so an
# unscaled apply compiles to a plain `copyto!` with nothing to predict -- the same shape as
# `_scaleresult!` in the local/global family, and for the same reason.
_writescaled!(result, resampled, ::Nothing) = copyto!(result, resampled)

function _writescaled!(result, resampled, postscale)
    @inbounds for i in eachindex(result, resampled, postscale)
        result[i] = resampled[i] * postscale[i]
    end
    return result
end
