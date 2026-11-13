
struct LocalθGlobalφInterpolator{stencilsize,T,TSampling,FFT,IFFT,DICT,Trait} <:
       Interpolation.AbstractInterpolator{stencilsize,T}
    interpolationinfodict::DICT
    interpolationtrait::Trait

    function LocalθGlobalφInterpolator{stencilsize,T}(
        TSampling; interpolationtrait=DefaultInterpolation()
    ) where {stencilsize,T}
        # No plans exist yet; use `T` for the unused FFT and IFFT type parameters.

        dict = Dict{
            Tuple{
                Sampling.SphericalSampling{TSampling},Sampling.SphericalSampling{TSampling}
            },
            LocalθGlobalφInterpolationInformation{stencilsize,TSampling,T,T},
        }()

        return new{stencilsize,T,TSampling,T,T,typeof(dict),typeof(interpolationtrait)}(
            dict, interpolationtrait
        )
    end

    function LocalθGlobalφInterpolator{stencilsize}(
        T, TSampling; interpolationtrait=DefaultInterpolation()
    ) where {stencilsize}
        return LocalθGlobalφInterpolator{stencilsize,T}(
            TSampling; interpolationtrait=interpolationtrait
        )
    end

    function LocalθGlobalφInterpolator{stencilsize,T}(
        samplings::Vector{Sampling.SphericalSampling{TSampling}},
        intersamplings::Vector{Sampling.SphericalSampling{TSampling}};
        interpolationtrait=DefaultInterpolation(),
    ) where {stencilsize,T,TSampling}
        if isempty(samplings)
            return LocalθGlobalφInterpolator{stencilsize,T}(
                TSampling; interpolationtrait=interpolationtrait
            )
        end

        dict = getlocalglobalinterpolationinfodict(
            T, stencilsize, samplings, intersamplings, interpolationtrait
        )

        return new{
            stencilsize,
            T,
            TSampling,
            fftplantype(first(first(values(dict)))),
            ifftplantype(first(first(values(dict)))),
            typeof(dict),
            typeof(interpolationtrait),
        }(
            dict, interpolationtrait
        )
    end

    function LocalθGlobalφInterpolator{stencilsize}(
        T::DataType,
        samplings::Vector{Sampling.SphericalSampling{TSampling}},
        intersamplings::Vector{Sampling.SphericalSampling{TSampling}};
        interpolationtrait=DefaultInterpolation(),
    ) where {stencilsize,TSampling}
        return LocalθGlobalφInterpolator{stencilsize,T}(
            samplings, intersamplings; interpolationtrait=interpolationtrait
        )
    end

    function LocalθGlobalφInterpolator{stencilsize,T}(
        sampling::Sampling.SphericalSampling{TSampling},
        intersampling::Sampling.SphericalSampling{TSampling};
        interpolationtrait=DefaultInterpolation(),
    ) where {stencilsize,T,TSampling}
        return LocalθGlobalφInterpolator{stencilsize}(
            T, [sampling], [intersampling]; interpolationtrait=interpolationtrait
        )
    end

    function LocalθGlobalφInterpolator{stencilsize}(
        T::DataType,
        sampling::Sampling.SphericalSampling{TSampling},
        intersampling::Sampling.SphericalSampling{TSampling};
        interpolationtrait=DefaultInterpolation(),
    ) where {stencilsize,TSampling}
        return LocalθGlobalφInterpolator{stencilsize,T}(
            sampling, intersampling; interpolationtrait=interpolationtrait
        )
    end
end

function (interpolator::Interpolation.LocalθGlobalφInterpolator{stencilsize,T})(
    farfields::AbstractArray{F,1},
    interpolatedfarfields::AbstractArray{F,1},
    farfieldsampling,
    interpolationsampling,
) where {stencilsize,F<:FarFields.AbstractFarField,T}
    return error(
        "Not implemented: This interpolator uses the special structure of SphericalSampling"
    )
end

function (interpolator::Interpolation.LocalθGlobalφInterpolator{stencilsize,T})(
    interpolatedfarfield::F,
    farfield::F,
    farfieldsampling::Sampling.SphericalSampling{T1},
    interpolationsampling::Sampling.SphericalSampling{T1};
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
    interim=nothing,
    postscale=nothing,
) where {stencilsize,T,T1,F<:FarFields.AbstractFarField}
    @assert Sampling.Nθ(farfieldsampling) == Sampling.Nθ(farfield)
    @assert Sampling.Nφ(farfieldsampling) == Sampling.Nφ(farfield)

    matrices = H2Factory.moments(farfield)
    resultmatrices = H2Factory.moments(interpolatedfarfield)

    # The θ stage writes into the TOP `Nφ(child)` rows of the parent-sized result, which the φ FFT then
    # zero-pads out to the full parent grid -- hence a view per component rather than another buffer.
    #
    # `Val(fielddim(F))` supplies a compile-time component count so the tuple of views needs no
    # heap allocation.
    interpolatedθfvals = ntuple(
        i -> view(resultmatrices[i], 1:Sampling.Nφ(farfieldsampling), :),
        Val(FarFields.fielddim(F)),
    )

    ftpadmatrix = Interpolation.takeinterim(
        Interpolation._ftpadmatrix(interim), T, size(interpolationsampling)
    )

    information = interpolationinformation(
        interpolator,
        farfieldsampling,
        interpolationsampling,
        resultmatrices[1],
        interpolationtrait,
    )

    for matrixjd in eachindex(matrices)
        Interpolation._interpolateθ!(
            interpolatedθfvals[matrixjd],
            matrices[matrixjd],
            information[matrixjd].θinformation,
        )
    end

    _interpolateφFFT!(
        resultmatrices,
        interpolatedθfvals,
        ftpadmatrix,
        information,
        (Sampling.Nφ(interpolationsampling), Sampling.Nθ(interpolationsampling)),
        postscale,
    )

    return nothing
end

function _interpolateθ!(
    interimmatrix::AbstractArray{T,2},
    fvals::AbstractArray{T,2},
    θinformation::GaussLegendreInterpolationInformation{stencilsize,T1},
) where {T,T1,stencilsize}
    for i in eachindex(θinformation.conformindices)
        weights = θinformation.weights[i]
        indices = θinformation.conformindices[i]

        iteration = θinformation.conformiterations[i]

        LinearAlgebra.mul!(
            view(interimmatrix, :, iteration), view(fvals, :, indices), weights
        )
    end

    for i in eachindex(θinformation.posextendingindices)
        iter = θinformation.nonconformiterations[i]

        weights = θinformation.posextendingweights[i]
        indices = θinformation.posextendingindices[i]
        negweights = θinformation.negextendingweights[i]
        negindices = θinformation.negextendingindices[i]
        LinearAlgebra.mul!(view(interimmatrix, :, iter), view(fvals, :, indices), weights)
        # Add the mirrored contribution as two contiguous row blocks. `mulrotatedrows!` derives
        # the rotation from the row count and may round differently from a permuted view.
        Interpolation.mulrotatedrows!(
            view(interimmatrix, :, iter), fvals, negindices, negweights, true, true
        )
    end
end

function _interpolateφFFT!(results, matrices, ftpad, information, newsize, postscale)
    interN = newsize[1]
    N = size(matrices[1], 1)
    Npad = floor(Int, interN / 2 - N / 2)

    ftpad[1:Npad, :] .= zero(eltype(ftpad))
    ftpad[(end - Npad + 1):end, :] .= zero(eltype(ftpad))
    ft = @view ftpad[(Npad + 1):(Npad + N), :]

    #TODO: consider using ffts! instead of fftshift!(fft(...)) -> look at FourierTools

    for j in eachindex(matrices)
        fftplan = information[j].φinformation.fftplan
        ifftplan = information[j].φinformation.ifftplan

        fftshift!(ft, fftplan * matrices[j], 1)

        fftshift!(results[j], ftpad, 1)

        ifftplan * results[j]

        _scaleresult!(results[j], interN / N, postscale)
    end

    return results
end

# The φ stage's final write, with the OPTIONAL `postscale` diagonal folded into it.
#
# Combine `postscale` with the existing `interN / N` normalization pass over the parent grid.
#
# `(v * scale) * coefficient`, NOT `v * (scale * coefficient)`. Those differ in the last bit, and the
# first preserves the rounding of interpolate-then-scale. Do not precombine the factors.
#
# `::Nothing` is a distinct method rather than a branch: `postscale` is a compile-time type, so an
# un-scaled apply resolves to the same single `mul!` it always did, with nothing to predict.
_scaleresult!(result, scale, ::Nothing) = LinearAlgebra.mul!(result, result, scale)

function _scaleresult!(result, scale, postscale)
    @inbounds for i in eachindex(result, postscale)
        result[i] = (result[i] * scale) * postscale[i]
    end
    return result
end

function interpolationinformation(
    interpolator::LocalθGlobalφInterpolator{stencilsize},
    sampling::Sampling.SphericalSampling,
    interpolationsampling::Sampling.SphericalSampling,
    resultmatrix,
    interpolationtrait,
) where {stencilsize}
    if haskey(interpolator.interpolationinfodict, (sampling, interpolationsampling))
        return converttrait(
            interpolationtrait,
            interpolator.interpolationinfodict[(sampling, interpolationsampling)],
            Interpolation.interpolationtrait(interpolator),
        )
    end

    return _localglobalinterpolationinformation(
        eltype(resultmatrix),
        Val(stencilsize),
        sampling,
        interpolationsampling,
        interpolationtrait,
    )
end
