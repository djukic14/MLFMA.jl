# MLFMA's level (I2I/O2O) transfer, expressed for H2Factory's level-transfer protocol.
#
# H2Factory owns traversal, direction handling and channels. This file supplies the transfer data
# and extends the public application protocol:
#
#   * `H2Factory.AbstractI2ITranslationTrait`  -- supertype of `InterpolationPhaseI2I`.
#   * `H2Factory.presenttransfer`              -- maps the stored entry to the object the
#                                                 direction/application steps see.
#   * `H2Factory.applydirectedtransfer!`       -- applies a transfer with its direction resolved.
#
# Direction handling needs no extension: `H2Factory._directedtransfer` turns a
# transfer into `T` / `transpose(T)` / `adjoint(T)` / `conj(T)`, so defining `Base.transpose`,
# `Base.adjoint` and `Base.conj` on the transfer type is enough to join the shared direction path.
#
# The transfer itself is `Phase ∘ Interp`: interpolate a child moment onto the parent's direction
# grid, then apply that child sector's diagonal re-referencing phase on the parent grid. A diagonal
# transposes to itself and conjugates under adjoint, so:
#
#     transpose(Phase ∘ Interp) = Interp^T ∘ Phase
#     adjoint(Phase ∘ Interp)   = Interp^H ∘ conj(Phase)
#     conj(Phase ∘ Interp)      = conj(Phase) ∘ conj(Interp)
#
# Conjugation preserves the child-to-parent order and conjugates both factors. An adjoint map's
# aggregation still runs child-to-parent; it uses H2Factory's `ConjugateTranslation`.
#
# The reverse directions use different interpolation operators. Anterpolation is adjoint
# interpolation, `Interp^H`. That equals `Interp^T` only when the operator is real, as in the
# barycentric-Lagrange family. FFT-based local/global interpolation need not be real. The transpose
# direction applies `MLFMA.transposedinterpolator(anterpolator)` and the adjoint direction applies the
# anterpolator itself. See `src/interpolation/anterpolator/transposedinterpolation.jl` for the measured
# difference and why H2Factory's contract requires honouring it, and
# `test/LevelTransfers/anterpolationcontract.jl` for the pins.
#
# Because transpose leaves the phase unconjugated, the radiation/receive conjugation cannot come
# from the direction step -- it belongs to the side. The trial and test collections therefore store
# different phases. `test/LevelTransfers/phaseconvention.jl` checks the side conjugation and
# distinguishes transpose from adjoint on the phase diagonal.

"""
    InterpolationPhaseI2I()

`H2Factory.AbstractI2ITranslationTrait` for MLFMA's interpolation-plus-phase level transfer.

Selects [`InterpolationPhaseTransfer`](@ref) as the applied transfer, alongside H2Factory's own
`DenseI2I` (one dense matrix per transfer) and `SeparableI2I` (per-axis 1D matrices).
"""
struct InterpolationPhaseI2I <: H2Factory.AbstractI2ITranslationTrait end

"""
    InterpolationPhaseTransfer(interpolator, anterpolator, phase, childsampling, parentsampling,
                               interpolationtrait)

One MLFMA level transfer: interpolation from `childsampling` up to `parentsampling`, followed by the
diagonal `phase` on the parent grid.

`phase` is the child sector's phase for this side, conjugated for the receive side. The constructor
stores `conjphase` and the non-forward operators, so application does not rebuild them. For complex
interpolation, two operators wrap a native apply with in-place conjugation of the moments. These
passes allocate nothing but still cost time.

  - `anterpolator` -- adjoint interpolation, applied in the adjoint direction alongside `conjphase`.
  - `transposedinterpolator` -- plain transposed interpolation, applied in the transpose direction
    alongside `phase`. Identical to `anterpolator` for a real interpolation operator, and a distinct
    operator otherwise; `MLFMA.transposedinterpolator` decides which.
  - `conjugatedinterpolator` -- `conj(Interp)`, applied in the conjugate direction alongside
    `conjphase`. That direction is not a reversal (see the header), so this one runs child-to-parent
    like `interpolator` itself, and is `interpolator` for a real operator;
    `MLFMA.conjugatedinterpolator` decides.

Applied through H2Factory's level-transfer protocol, so the direction is chosen by the traversal,
never here.
"""
struct InterpolationPhaseTransfer{I,A,TA,CI,P,SC,SP,T}
    interpolator::I
    anterpolator::A
    transposedinterpolator::TA
    conjugatedinterpolator::CI
    phase::P
    conjphase::P
    childsampling::SC
    parentsampling::SP
    interpolationtrait::T
end

# Derive the transpose and conjugate operators here so callers cannot supply mismatched variants.
function InterpolationPhaseTransfer(
    interpolator, anterpolator, phase, childsampling, parentsampling, interpolationtrait
)
    return InterpolationPhaseTransfer(
        interpolator,
        anterpolator,
        Interpolation.transposedinterpolator(anterpolator),
        Interpolation.conjugatedinterpolator(interpolator),
        phase,
        conj.(phase),
        childsampling,
        parentsampling,
        interpolationtrait,
    )
end

# The stored entry is the applied transfer, so presenting it is the identity (contrast
# `SeparableI2I`, which wraps a raw tuple into a `SeparableTransfer`).
H2Factory.presenttransfer(::InterpolationPhaseI2I, raw) = raw

# --- Direction ------------------------------------------------------------------------------------
# Each immutable wrapper selects an `applydirectedtransfer!` method through its type.
# The wrappers hold the transfer by reference and allocate nothing once inlined.

struct TransposedInterpolationPhaseTransfer{T}
    transfer::T
end

struct AdjointInterpolationPhaseTransfer{T}
    transfer::T
end

# An adjoint map's aggregation uses conjugation, not transpose: it still runs child-to-parent.
# See H2Factory's `ConjugateTranslation`.
struct ConjugatedInterpolationPhaseTransfer{T}
    transfer::T
end

Base.transpose(t::InterpolationPhaseTransfer) = TransposedInterpolationPhaseTransfer(t)
Base.adjoint(t::InterpolationPhaseTransfer) = AdjointInterpolationPhaseTransfer(t)
Base.conj(t::InterpolationPhaseTransfer) = ConjugatedInterpolationPhaseTransfer(t)

# Involutions, so a doubly-directed transfer collapses instead of nesting wrappers.
Base.transpose(t::TransposedInterpolationPhaseTransfer) = t.transfer
Base.adjoint(t::AdjointInterpolationPhaseTransfer) = t.transfer
Base.conj(t::ConjugatedInterpolationPhaseTransfer) = t.transfer

# --- Application ----------------------------------------------------------------------------------
# Accumulation needs a staging moment because interpolation overwrites its destination: a parent
# moment for forward/conjugate application and a child moment for transpose/adjoint application.
# Forward overwrite writes directly to `out`. Reverse application stages the phased parent input
# inside the operator's `interim`, leaving the traversal's input unchanged.
#
# Both channels supply `LevelTransferScratch`; I2I leaves `childmoment` empty because only reverse
# accumulation needs it. A bare parent-grid buffer is also accepted for forward application.
# Its missing `interim` is passed as `nothing`, allowing interpolation to allocate per call.
# `test/LevelTransfers/protocol.jl` checks allocation overhead relative to a bare interpolation call,
# not the interpolator's own allocation count.
_parentscratch(buffer::Interpolation.LevelTransferScratch) = buffer.parentmoment
_parentscratch(buffer) = buffer

_childscratch(buffer::Interpolation.LevelTransferScratch) = buffer.childmoment

_interim(buffer::Interpolation.LevelTransferScratch) = buffer.interim
_interim(buffer) = nothing

function H2Factory.applydirectedtransfer!(
    out,
    in,
    T::InterpolationPhaseTransfer,
    accumulate::Bool,
    context::H2Factory.TranslationContext,
)
    return _applyforwardtransfer!(out, in, T, T.interpolator, T.phase, accumulate, context)
end

# Conjugate application follows the forward order with the stored conjugated factors.
# For real interpolation, such as barycentric Lagrange, the interpolator needs no wrapper or passes.
function H2Factory.applydirectedtransfer!(
    out,
    in,
    T::ConjugatedInterpolationPhaseTransfer,
    accumulate::Bool,
    context::H2Factory.TranslationContext,
)
    return _applyforwardtransfer!(
        out,
        in,
        T.transfer,
        T.transfer.conjugatedinterpolator,
        T.transfer.conjphase,
        accumulate,
        context,
    )
end

# The two child-to-parent directions differ only in the operator and diagonal supplied here.
# `postscale` applies the diagonal on the parent grid during the final interpolation write.
# `LocalθGlobalφ` combines it with `interN / N`; Lagrange interpolation scales each result row
# after `mul!`. This saves a separate grid pass. The kernels keep the scale-then-diagonal order,
# giving bitwise agreement with the two-pass form, checked with `==` in `test/LevelTransfers/`.
function _applyforwardtransfer!(
    out, in, transfer, operator, phase, accumulate::Bool, context
)
    if accumulate
        # The operator overwrites its destination, so accumulating needs a parent-grid staging step.
        scratch = _parentscratch(context.buffer)
        operator(
            scratch,
            in,
            transfer.childsampling,
            transfer.parentsampling;
            interpolationtrait=transfer.interpolationtrait,
            interim=_interim(context.buffer),
            postscale=phase,
        )
        H2Factory.add!(out, scratch)
    else
        operator(
            out,
            in,
            transfer.childsampling,
            transfer.parentsampling;
            interpolationtrait=transfer.interpolationtrait,
            interim=_interim(context.buffer),
            postscale=phase,
        )
    end
    return out
end

function H2Factory.applydirectedtransfer!(
    out,
    in,
    T::TransposedInterpolationPhaseTransfer,
    accumulate::Bool,
    context::H2Factory.TranslationContext,
)
    return _applyreversetransfer!(
        out,
        in,
        T.transfer,
        T.transfer.transposedinterpolator,
        T.transfer.phase,
        accumulate,
        context,
    )
end

function H2Factory.applydirectedtransfer!(
    out,
    in,
    T::AdjointInterpolationPhaseTransfer,
    accumulate::Bool,
    context::H2Factory.TranslationContext,
)
    return _applyreversetransfer!(
        out,
        in,
        T.transfer,
        T.transfer.anterpolator,
        T.transfer.conjphase,
        accumulate,
        context,
    )
end

# Transpose uses the unchanged phase and transposed interpolation. Adjoint uses the conjugated phase
# and anterpolation (= adjoint interpolation). The operators coincide only for real interpolation.
function _applyreversetransfer!(
    out, in, transfer, reverseoperator, phase, accumulate::Bool, context
)
    # `prescale` stages `in .* phase` in the operator's parent-grid `interim`, without mutating `in`.
    # The phase needs a separate pass before the global FFT; fusing it into overlapping θ stencils
    # would repeat multiplies. Owning the staged copy lets `_anterpolateφbfft!` destroy it and skip
    # the inverse FFT used to restore borrowed input. On a 108x54 grid, that FFT cost 31.05 µs per
    # component versus 11.59 µs for the phase pass. This uses no additional parent-sized buffer.
    if accumulate
        # Reverse accumulation stages on the child grid because the operator overwrites its output.
        childscratch = _childscratch(context.buffer)
        reverseoperator(
            childscratch,
            in,
            transfer.parentsampling,
            transfer.childsampling;
            interpolationtrait=transfer.interpolationtrait,
            interim=_interim(context.buffer),
            prescale=phase,
        )
        H2Factory.add!(out, childscratch)
    else
        reverseoperator(
            out,
            in,
            transfer.parentsampling,
            transfer.childsampling;
            interpolationtrait=transfer.interpolationtrait,
            interim=_interim(context.buffer),
            prescale=phase,
        )
    end
    return out
end
