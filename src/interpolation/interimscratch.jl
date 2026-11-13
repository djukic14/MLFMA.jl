# Per-transfer interim scratch for the interpolators and anterpolators.
#
# The read-only core owns the interpolators and anterpolators; each workspace owns its scratch.
# `H2Factory.borrowbuffer(_, level, chunkid, _)` supplies one `LevelTransferScratch` per
# (level, chunk), so concurrent chunks do not share mutable buffers or plans.
# Missing buffers or buffers with the wrong shape cause a fresh allocation. Standalone calls
# can therefore omit scratch without risking writes outside a mismatched buffer.

"""
    LevelTransferScratch(parentmoment, childmoment, interim)

One chunk's worth of level-transfer scratch, borrowed per (level, chunk) by H2Factory's traversal and
reached through `context.buffer` in `H2Factory.applydirectedtransfer!`.

  - `parentmoment` -- a moment on the PARENT direction grid. The forward transfer stages through it when
    accumulating; the reverse directions always need it, since the diagonal phase acts on the parent grid.
  - `childmoment` -- a moment on the CHILD grid, or `nothing`. Only the accumulating reverse direction
    needs it, so the I2I side leaves it empty.
  - `interim` -- the interpolator/anterpolator's own interim arrays for this level pair, built by
    [`interimscratch`](@ref), or `nothing` to let them allocate per call.
"""
struct LevelTransferScratch{P,C,I}
    parentmoment::P
    childmoment::C
    interim::I
end

"""
    interimscratch(anterpolator, childsampling, parentsampling)

Build the interim arrays one level transfer between `childsampling` and `parentsampling` needs, as a
`NamedTuple` whose fields the interpolator and anterpolator read through [`takeinterim`](@ref).

Dispatches on the anterpolator, which stores its interpolator and needs at least as much scratch.
One method per family covers both directions. The fallback returns `nothing`, so families without
a scratch method allocate their interim arrays per call.

The shapes follow from which sampling plays which role in each direction. Interpolation runs
child -> parent (`farfieldsampling = childsampling`, `interpolationsampling = parentsampling`) and
anterpolation runs parent -> child, so both stage through an interim grid mixing the parent's `Nθ` with
the child's `Nφ`, and one set of arrays serves both directions.
"""
interimscratch(::Any, childsampling, parentsampling) = nothing

function interimscratch(
    ::Interpolation.SphericalLagrangeAnterpolator{stencilsize,T},
    childsampling,
    parentsampling,
) where {stencilsize,T}
    return (
        interimmatrix=Matrix{T}(
            undef, (Sampling.Nθ(parentsampling), Sampling.Nφ(childsampling))
        ),
        interimθvector=Vector{T}(undef, Sampling.Nφ(childsampling)),
        interimφvector=Vector{T}(undef, Sampling.Nθ(parentsampling)),
        prescalematrix=Matrix{T}(undef, size(parentsampling)),
    )
end

function interimscratch(
    ::Interpolation.LocalθGlobalφAnterpolator{stencilsize,T}, childsampling, parentsampling
) where {stencilsize,T}
    # `ftpadmatrix` serves the interpolator's zero-padded FT buffer and the anterpolator's
    # `shiftedmatrix`: both are shaped `size(parentsampling)`, and interpolation and anterpolation are
    # separate applies, so one array covers both.
    # `prescalematrix` holds `in .* prescale` on the parent grid without changing the caller's moment.
    # The kernel can discard this copy without restoring it by inverse FFT. It must not alias
    # `ftpadmatrix`, which receives the first `fftshift!`. Components reuse the staging array in turn.
    return (
        ftpadmatrix=Matrix{T}(undef, size(parentsampling)),
        interimmatrix=Matrix{T}(
            undef, (Sampling.Nφ(childsampling), Sampling.Nθ(parentsampling))
        ),
        interimθvector=Vector{T}(undef, Sampling.Nφ(childsampling)),
        prescalematrix=Matrix{T}(undef, size(parentsampling)),
    )
end

# The circular family's scratch: two spectra and four in-place FFT plans, in ONE object rather than as
# loose fields. The plans are the reason -- they have to be built per (level, chunk) so no two
# concurrent tasks apply the same in-place plan, which is the same arrangement `Equidistant.FFTScratch`
# has and the reason neither family stores its plans on the interpolator.
#
# `complex(float(T))` rather than `T`: the transforms need a complex array, and a family built with a
# real element type would otherwise hand `plan_fft!` something it cannot plan for.
function interimscratch(
    ::Interpolation.CircularAnterpolator{stencilsize,T},
    childsampling::Sampling.CircularSampling,
    parentsampling::Sampling.CircularSampling,
) where {stencilsize,T}
    return (
        circularscratch=Interpolation.CircularInterimScratch(
            childsampling, parentsampling, complex(float(T))
        ),
    )
end

# Field readers that tolerate a missing scratch, so the apply sites never branch on `nothing` themselves.
# Each family reads only the fields its own `interimscratch` method produces, and an anterpolator is
# always built from an interpolator of the same family, so the pairing cannot come apart.
_interimmatrix(scratch) = scratch.interimmatrix
_interimmatrix(::Nothing) = nothing
_interimθvector(scratch) = scratch.interimθvector
_interimθvector(::Nothing) = nothing
_interimφvector(scratch) = scratch.interimφvector
_interimφvector(::Nothing) = nothing
_ftpadmatrix(scratch) = scratch.ftpadmatrix
_ftpadmatrix(::Nothing) = nothing
# Hand-built scratch NamedTuples may omit `prescalematrix`. Treat that as a missing buffer
# and allocate on demand, rather than raising a field-access error.
_prescalematrix(scratch) =
    hasproperty(scratch, :prescalematrix) ? scratch.prescalematrix : nothing
_prescalematrix(::Nothing) = nothing

"""
    takeinterim(buffer, T, shape)

Return `buffer` if it already has `shape`, otherwise a fresh `Array{T}` of that shape.

The workspace sizes buffers from its samplings; each transfer checks against its own samplings.
If the shapes differ, allocate instead of writing outside the buffer. A standalone call may pass
`nothing` to request a fresh array.
"""
takeinterim(::Nothing, ::Type{T}, shape) where {T} = _freshinterim(T, shape)

function takeinterim(buffer, ::Type{T}, shape) where {T}
    _interimshape(buffer) == shape && return buffer
    return _freshinterim(T, shape)
end

_freshinterim(::Type{T}, shape::NTuple{2,Int}) where {T} = Matrix{T}(undef, shape)
_freshinterim(::Type{T}, shape::Int) where {T} = Vector{T}(undef, shape)

_interimshape(buffer::AbstractMatrix) = size(buffer)
_interimshape(buffer::AbstractVector) = length(buffer)
