struct LocalθGlobalφInterpolationInformation{stencilsize,T1,FFT,IFFT}
    θinformation::Interpolation.GaussLegendreInterpolationInformation{stencilsize,T1}
    φinformation::FFTInterpolationInformation{FFT,IFFT}

    function LocalθGlobalφInterpolationInformation{stencilsize}(
        sampling::Sampling.SphericalSampling{T1},
        interpolationsampling::Sampling.SphericalSampling{T1},
        fftmatrix::AbstractArray{T,2},
        ifftmatrix::AbstractArray{T,2};
        interpolationtrait=VirtualExtensionFlipθSign(),
    ) where {stencilsize,T,T1}
        θinformation = Interpolation.GaussLegendreInterpolationInformation{stencilsize}(
            sampling, interpolationsampling; interpolationtrait=interpolationtrait
        )
        φinformation = FFTInterpolationInformation(fftmatrix, ifftmatrix)

        return new{
            stencilsize,T1,typeof(φinformation.fftplan),typeof(φinformation.ifftplan)
        }(
            θinformation, φinformation
        )
    end
end

function fftplantype(
    ::LocalθGlobalφInterpolationInformation{stencilsize,T1,FFT,IFFT}
) where {stencilsize,T1,FFT,IFFT}
    return FFT
end

function ifftplantype(
    ::LocalθGlobalφInterpolationInformation{stencilsize,T1,FFT,IFFT}
) where {stencilsize,T1,FFT,IFFT}
    return IFFT
end

function getlocalglobalinterpolationinfodict(
    T::DataType,
    stencilsize,
    samplings::Vector{Sampling.SphericalSampling{TSampling}},
    interpolationsamplings::Vector{Sampling.SphericalSampling{TSampling}},
    interpolationtrait,
) where {TSampling}
    infos = Vector{
        typeof(
            _localglobalinterpolationinformation(
                T,
                stencilsize,
                samplings[end],
                interpolationsamplings[end],
                interpolationtrait,
            ),
        ),
    }(
        undef, length(samplings)
    )

    dictkeys = Vector{Tuple{typeof(samplings[end]),typeof(samplings[end])}}(
        undef, length(samplings)
    )

    for i in eachindex(samplings)
        infos[i] = _localglobalinterpolationinformation(
            T, stencilsize, samplings[i], interpolationsamplings[i], interpolationtrait
        )
        dictkeys[i] = (samplings[i], interpolationsamplings[i])
    end

    return Dict(zip(dictkeys, infos))
end

function _resultinterpolatedθfvals(T, sampling, interpolationsampling)
    resultmatrix = Matrix{T}(undef, size(interpolationsampling))
    interpolatedθfvals = view(resultmatrix, 1:Sampling.Nφ(sampling), :)

    return resultmatrix, interpolatedθfvals
end

function _localglobalinterpolationinformation(
    T, stencilsize::Int, sampling, interpolationsampling, interpolationtrait
)
    return _localglobalinterpolationinformation(
        T, Val(stencilsize), sampling, interpolationsampling, interpolationtrait
    )
end

function _localglobalinterpolationinformation(
    T, ::Val{stencilsize}, sampling, interpolationsampling, ::DefaultInterpolation
) where {stencilsize}
    resultmatrix, interpolatedθfvals = _resultinterpolatedθfvals(
        T, sampling, interpolationsampling
    )

    info = LocalθGlobalφInterpolationInformation{stencilsize}(
        sampling,
        interpolationsampling,
        interpolatedθfvals,
        resultmatrix;
        interpolationtrait=VirtualExtension(),
    )

    infoflipped = LocalθGlobalφInterpolationInformation{stencilsize}(
        sampling,
        interpolationsampling,
        interpolatedθfvals,
        resultmatrix;
        interpolationtrait=VirtualExtensionFlipθSign(),
    )

    return (virtual=info, virtualflipθsign=infoflipped)
end

function _localglobalinterpolationinformation(
    T, ::Val{stencilsize}, sampling, interpolationsampling, interpolationtrait
) where {stencilsize}
    resultmatrix, interpolatedθfvals = _resultinterpolatedθfvals(
        T, sampling, interpolationsampling
    )

    info = LocalθGlobalφInterpolationInformation{stencilsize}(
        sampling,
        interpolationsampling,
        interpolatedθfvals,
        resultmatrix;
        interpolationtrait=interpolationtrait,
    )

    return SVector(info)
end

function _localglobalinterpolationinformation(
    T, ::Val{stencilsize}, sampling, interpolationsampling, ::SphericalθφInterpolation
) where {stencilsize}
    resultmatrix, interpolatedθfvals = _resultinterpolatedθfvals(
        T, sampling, interpolationsampling
    )

    info = LocalθGlobalφInterpolationInformation{stencilsize}(
        sampling,
        interpolationsampling,
        interpolatedθfvals,
        resultmatrix;
        interpolationtrait=VirtualExtensionFlipθSign(),
    )

    return SVector(info, info)
end

function _localglobalinterpolationinformation(
    T, ::Val{stencilsize}, sampling, interpolationsampling, ::SphericalInterpolation
) where {stencilsize}
    resultmatrix, interpolatedθfvals = _resultinterpolatedθfvals(
        T, sampling, interpolationsampling
    )

    infor = LocalθGlobalφInterpolationInformation{stencilsize}(
        sampling,
        interpolationsampling,
        interpolatedθfvals,
        resultmatrix;
        interpolationtrait=VirtualExtension(),
    )

    infoθφ = LocalθGlobalφInterpolationInformation{stencilsize}(
        sampling,
        interpolationsampling,
        interpolatedθfvals,
        resultmatrix;
        interpolationtrait=VirtualExtensionFlipθSign(),
    )

    return SVector(infor, infoθφ, infoθφ)
end

function _localglobalinterpolationinformation(
    T, ::Val{stencilsize}, sampling, interpolationsampling, ::DoubleSphericalInterpolation
) where {stencilsize}
    resultmatrix, interpolatedθfvals = _resultinterpolatedθfvals(
        T, sampling, interpolationsampling
    )

    infor = LocalθGlobalφInterpolationInformation{stencilsize}(
        sampling,
        interpolationsampling,
        interpolatedθfvals,
        resultmatrix;
        interpolationtrait=VirtualExtension(),
    )

    infoθφ = LocalθGlobalφInterpolationInformation{stencilsize}(
        sampling,
        interpolationsampling,
        interpolatedθfvals,
        resultmatrix;
        interpolationtrait=VirtualExtensionFlipθSign(),
    )

    return SVector(infor, infoθφ, infoθφ, infor, infoθφ, infoθφ)
end
