# The CONJUGATE of an interpolation operator -- conjugated, NOT transposed.
#
# The fourth direction, and the only one that is not a reversal. An adjoint map's AGGREGATION needs it:
# writing a far operator as `A = α · (I2I_test · L_test)^T · T_i2o · (I2I_trial · L_trial)`,
#
#     A^H = conj(α) · (conj(I2I_trial) · conj(L_trial))^T · T_i2o^H · conj(I2I_test) · conj(L_test)
#
# and aggregation still runs child-to-parent, so the test side wants `conj(I2I_test)` with no transpose
# anywhere. See H2Factory's `ConjugateTranslation`, which is the direction that reaches this.
#
# Built the same way `TransposedInterpolator` reaches the transpose, one level over: that one applies
# `I^T = conj ∘ I^H ∘ conj` around the ADJOINT operator, this one applies `conj(I) = conj ∘ I ∘ conj`
# around the FORWARD one. Both are exact rather than approximate -- conjugating an operator's input and
# its output conjugates the operator -- and both are needed only for a family whose operator is complex.
# `isrealinterpolation` (stated per family in `transposedinterpolation.jl`) decides; for the
# barycentric-Lagrange family `conj(I) == I` and `conjugatedinterpolator` hands the interpolator
# straight back, so that family pays nothing at all here and only its PHASE gets conjugated.
#
# Same in-place discipline as `TransposedInterpolator`, and for the same reason: the input belongs to
# the traversal, so it is conjugated in place and restored from a `finally`. That is also what makes
# this need NO extra scratch -- the obvious alternative, staging `conj(in)` into a child-grid buffer,
# would need one the I2I channel does not allocate (its `LevelTransferScratch` carries a parent moment
# and `nothing` for the child; only the O2O channel has both).

"""
    ConjugatedInterpolator(interpolator)

The CONJUGATE of an interpolation operator, `conj(I)`, built from the interpolator itself.

Applies ``conj(I) = conj ∘ I ∘ conj``. Conjugated but NOT transposed -- distinct from both
[`TransposedInterpolator`](@ref) and an anterpolator, and the direction an ADJOINT map's aggregation
applies. See [`conjugatedinterpolator`](@ref), which returns the interpolator unchanged when its
operator is real.

The conjugations are done IN PLACE and the input is restored before returning, from a `finally`, so it
happens even when the wrapped interpolator throws. An apply allocates nothing beyond what the wrapped
interpolator allocates. As with `TransposedInterpolator`, the restore is only as faithful as the wrapped
operator: a family that perturbs its input on an FFT round trip returns it to round-off, not bitwise.
"""
struct ConjugatedInterpolator{stencilsize,T,I} <:
       Interpolation.AbstractInterpolator{stencilsize,T}
    interpolator::I
end

function ConjugatedInterpolator(
    interpolator::Interpolation.AbstractInterpolator{stencilsize,T}
) where {stencilsize,T}
    return ConjugatedInterpolator{stencilsize,T,typeof(interpolator)}(interpolator)
end

interpolator(c::ConjugatedInterpolator) = c.interpolator

# Shaped by the wrapped interpolator, since that is what actually runs. Forwarded for the same reason
# `TransposedInterpolator` forwards it: without a method here the `::Any` fallback returns `nothing`,
# which turns a scratch-backed transfer into an allocating one without failing anything.
interimscratch(c::ConjugatedInterpolator, childsampling, parentsampling) =
    interimscratch(c.interpolator, childsampling, parentsampling)

interpolationtrait(c::ConjugatedInterpolator) = interpolationtrait(c.interpolator)

"""
    conjugatedinterpolator(interpolator)

The operator to apply in a CONJUGATE direction, given the forward `interpolator`.

Returns `interpolator` unchanged when its operator is real -- `conj(I) == I` then, and wrapping would
buy two conjugation passes for nothing -- and a [`ConjugatedInterpolator`](@ref) otherwise. Symmetric
with `anterpolator(interpolator)` and `transposedinterpolator(anterpolator)`: one accessor per
direction, so a caller states which algebra it means instead of conjugating at the call site.
"""
function conjugatedinterpolator(interpolator::Interpolation.AbstractInterpolator)
    isrealinterpolation(interpolator) && return interpolator
    return ConjugatedInterpolator(interpolator)
end

conjugatedinterpolator(::Nothing) = nothing

function (c::ConjugatedInterpolator)(
    conjugatedfarfield::F,
    farfield::F,
    # The wrapper reads no grid data. Accept any sampling supported by the wrapped interpolator.
    farfieldsampling::Sampling.AbstractSampling{T1},
    interpolationsampling::Sampling.AbstractSampling{T1};
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
    interim=nothing,
    postscale=nothing,
) where {F<:FarFields.AbstractFarField,T1}
    # conj -> forward -> conj, then put the input back. `Base.conj!` on a moment conjugates every
    # component in place and allocates nothing.
    #
    # Restore the caller's input in `finally`, including when the wrapped call throws.
    # Do not forward `postscale`: this wrapper computes `conj ∘ I ∘ conj`, so a diagonal applied inside would
    # come back conjugated: the caller asks for `D·conj(I)·x` and would get `conj(D)·conj(I)·x`.
    # Apply the diagonal during the trailing conjugation to avoid allocating `conj.(D)`.
    conj!(farfield)
    try
        c.interpolator(
            conjugatedfarfield,
            farfield,
            farfieldsampling,
            interpolationsampling;
            interpolationtrait=interpolationtrait,
            interim=interim,
        )
    finally
        conj!(farfield)
    end

    # Only conjugate the output after a successful apply.
    _conjugatescaled!(conjugatedfarfield, postscale)
    return conjugatedfarfield
end

# The trailing conjugation, with the optional output diagonal folded into the same pass.
#
# `conj(v) * coefficient` is exactly what `conj!` followed by `elementwisemul!` computed, in that order,
# so the fused form is BITWISE identical to the unfused one rather than merely close.
_conjugatescaled!(farfield, ::Nothing) = conj!(farfield)

function _conjugatescaled!(farfield, postscale)
    for i in eachindex(farfield)
        component = farfield[i]
        @inbounds for j in eachindex(component, postscale)
            component[j] = conj(component[j]) * postscale[j]
        end
    end
    return farfield
end

function (c::ConjugatedInterpolator)(
    farfields::AbstractArray{F,1}, conjugatedfarfields::AbstractArray{F,1}, args...
) where {F<:FarFields.AbstractFarField}
    return error(
        "Not implemented: this operator uses the special structure of SphericalSampling, as does the " *
        "interpolator it wraps",
    )
end
