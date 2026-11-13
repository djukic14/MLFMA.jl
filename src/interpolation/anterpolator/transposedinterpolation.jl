# Plain transposed interpolation, distinct from adjoint interpolation (anterpolation).
#
# SphericalLagrangeInterpolator has real barycentric weights, so its transpose equals its adjoint.
# LocalθGlobalφInterpolator can have complex entries because its φ step assigns the unsplit
# Nyquist bin `|m| = Nφ/2` to one signed frequency. Its transpose and adjoint must remain distinct.
# `test/LevelTransfers/anterpolationcontract.jl` checks that distinction against the matrices.
#
# A bilinear Galerkin entry ``∫∫ t G b`` has no conjugation, so its forward test side needs the plain
# transpose. Reverse-direction dispatch must preserve Julia's distinction between transpose and adjoint.
#
# `MLFMA.NθNφ(L) = (2(L+1), 4(L+1))` places the φ Nyquist order at `2(L+1)`, above degree L.
# A field confined to that degree does not excite the ambiguous bin, but arbitrary input can.

"""
    isrealinterpolation(interpolator)

Whether `interpolator`'s operator has real entries, so that its transpose and its adjoint coincide.

This determines whether [`transposedinterpolator`](@ref) reuses the anterpolator or builds a wrapper.
Each family declares it rather than materializing a matrix to test it during construction.

`true` for the real-weight barycentric-Lagrange family; `false` for the local/global family with its
unsplit φ Nyquist bin. `test/LevelTransfers/anterpolationcontract.jl` checks the actual matrices.
"""
isrealinterpolation(::Interpolation.SphericalLagrangeInterpolator) = true
isrealinterpolation(::Interpolation.LocalθGlobalφInterpolator) = false
# The circular family splits an even-length Nyquist bin across `±N/2`, preserving conjugate pairs
# and a real operator in exact arithmetic. The circular tests cover odd and even grid sizes.
isrealinterpolation(::Interpolation.CircularInterpolator) = true
isrealinterpolation(anterpolator::Interpolation.AbstractAnterpolator) =
    isrealinterpolation(Interpolation.interpolator(anterpolator))

"""
    TransposedInterpolator(anterpolator)

The plain TRANSPOSE of an interpolation operator, built from its adjoint (an anterpolator).

Applies ``I^T = conj ∘ I^H ∘ conj``, which is exact rather than approximate: with
``I^H = conj(I^T)``, conjugating on both sides of the adjoint recovers the transpose identically. Only
needed for families where the two differ -- see [`transposedinterpolator`](@ref), which returns the
anterpolator itself when the operator is real.

With `prescale`, the wrapped anterpolator stages and conjugates the scaled input in private scratch;
the caller's moment is untouched. Without it, this wrapper conjugates the input in place and undoes
that conjugation in `finally`, even if the wrapped call throws. This unscaled path requires exclusive
access to the input during the call.

The wrapper adds no allocations. Double conjugation is exact, but a wrapped FFT round trip may
perturb the restored input by round-off. The wrapper cannot undo those changes or other mutations
made by a failing wrapped operator.
"""
struct TransposedInterpolator{stencilsize,T,A} <:
       Interpolation.AbstractAnterpolator{stencilsize,T}
    anterpolator::A
end

function TransposedInterpolator(
    anterpolator::Interpolation.AbstractAnterpolator{stencilsize,T}
) where {stencilsize,T}
    return TransposedInterpolator{stencilsize,T,typeof(anterpolator)}(anterpolator)
end

# The wrapper holds an anterpolator rather than an interpolator, so the inherited accessors -- which
# read a `.interpolator` field -- need forwarding.
Interpolation.interpolator(t::TransposedInterpolator) =
    Interpolation.interpolator(t.anterpolator)

anterpolator(t::TransposedInterpolator) = t.anterpolator

# Interim scratch is shaped by the wrapped anterpolator, since that is what actually runs. Forwarded
# so a `TransposedInterpolator` handed to `interimscratch` cannot silently fall through to the
# `::Any` method and return `nothing`, which would turn a scratch-backed transfer into an allocating
# one without failing anything.
interimscratch(t::TransposedInterpolator, childsampling, parentsampling) =
    interimscratch(t.anterpolator, childsampling, parentsampling)

"""
    transposedinterpolator(anterpolator)

The operator to apply in a TRANSPOSE direction, given the adjoint operator `anterpolator`.

Returns `anterpolator` unchanged when the interpolation operator is real -- transpose and adjoint are
then the same matrix, and wrapping would buy three conjugation passes for nothing -- and a
[`TransposedInterpolator`](@ref) otherwise. Symmetric with `anterpolator(interpolator)`: one accessor
per reverse direction, so a caller states which algebra it means instead of conjugating at the call
site.
"""
function transposedinterpolator(anterpolator::Interpolation.AbstractAnterpolator)
    isrealinterpolation(anterpolator) && return anterpolator
    return TransposedInterpolator(anterpolator)
end

transposedinterpolator(::Nothing) = nothing

function (t::TransposedInterpolator)(
    transposedfarfield::F,
    farfield::F,
    # The wrapper reads no grid data. Accept any sampling supported by the wrapped anterpolator.
    farfieldsampling::Sampling.AbstractSampling{T1},
    transposesampling::Sampling.AbstractSampling{T1};
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
    interim=nothing,
    prescale=nothing,
) where {F<:FarFields.AbstractFarField,T1}
    # conj -> adjoint -> conj, then put the input back. `Base.conj!` on a moment conjugates every
    # component in place and allocates nothing.
    #
    # With `prescale`, use private staging so concurrent children can read the same parent input.
    #
    # WITH `prescale` the input is NEVER TOUCHED. The wrapped anterpolator stages `conj(x .* D)` into
    # its own buffer -- `stagetransform=conj` conjugates the PRODUCT, which is the whole conjugation
    # this wrapper needs:
    #
    #     Iᵀ(D·x) = conj( Iᴴ( conj(D·x) ) )
    #
    # so staging `conj(x*D)` and conjugating the result is the entire transpose. Forwarding `D`
    # unconjugated would compute `Iᵀ` against the unconjugated phase -- plausible-looking and wrong.
    #
    # In-place conjugation of a shared parent would race with reads by other children.
    if prescale !== nothing
        t.anterpolator(
            transposedfarfield,
            farfield,
            farfieldsampling,
            transposesampling;
            interpolationtrait=interpolationtrait,
            interim=interim,
            prescale=prescale,
            stagetransform=conj,
        )
        conj!(transposedfarfield)
        return transposedfarfield
    end

    # Without `prescale`, conjugate in place and undo that conjugation in `finally`, even on failure.
    #
    # This path is for direct/standalone callers. The traversal always supplies a diagonal and so
    # always takes the branch above, which is the one that has to be concurrency-safe.
    conj!(farfield)
    try
        t.anterpolator(
            transposedfarfield,
            farfield,
            farfieldsampling,
            transposesampling;
            interpolationtrait=interpolationtrait,
            interim=interim,
        )
    finally
        conj!(farfield)
    end

    conj!(transposedfarfield)
    return transposedfarfield
end

function (t::TransposedInterpolator)(
    farfields::AbstractArray{F,1}, transposedfarfields::AbstractArray{F,1}, args...
) where {F<:FarFields.AbstractFarField}
    return error(
        "Not implemented: this operator uses the special structure of SphericalSampling, as does the " *
        "anterpolator it wraps",
    )
end
