
struct LocalθGlobalφAnterpolationInformation{stencilsize,T1,FFT,BFFT,IFFT}
    θinformation::GaussLegendreAnterpolationInformation{stencilsize,T1}
    φinformation::FFTAnterpolationInformation{FFT,BFFT,IFFT}

    function LocalθGlobalφAnterpolationInformation{stencilsize}(
        sampling::Sampling.SphericalSampling{T1},
        anterpolationsampling::Sampling.SphericalSampling{T1},
        fftmatrix,
        bfftmatrix;
        interpolationtrait=VirtualExtensionFlipθSign(),
    ) where {stencilsize,T1}
        θinformation = Interpolation.GaussLegendreAnterpolationInformation{stencilsize}(
            sampling, anterpolationsampling; interpolationtrait=interpolationtrait
        )
        φinformation = FFTAnterpolationInformation(fftmatrix, bfftmatrix)
        return new{
            stencilsize,
            T1,
            typeof(φinformation.fftplan),
            typeof(φinformation.bfftplan),
            typeof(φinformation.ifftplan),
        }(
            θinformation, φinformation
        )
    end
end

function getlocalglobalanterpolationinfodict(
    T::DataType,
    interpolationinfodict::Dict{NTuple{2,SphericalSampling{TSampling}}},
    stencilsize,
    interpolationtrait,
    nchunks::Int,
) where {TSampling}
    samplings = Vector{Sampling.SphericalSampling{TSampling}}()
    intersamplings = Vector{Sampling.SphericalSampling{TSampling}}()

    for key in keys(interpolationinfodict)
        push!(samplings, key[1])
        push!(intersamplings, key[2])
    end

    matrix = Matrix{T}(undef, size(intersamplings[end]))

    # Planning needs only a representative layout; chunk buffers belong to workspaces.
    interimmatrix = Matrix{T}(
        undef, Sampling.Nφ(samplings[end]), Sampling.Nθ(intersamplings[end])
    )

    intfotype = typeof(
        _localglobalanterpolationinfo(
            stencilsize,
            samplings[end],
            intersamplings[end],
            matrix,
            interimmatrix,
            interpolationtrait,
        ),
    )

    infos = Vector{intfotype}(undef, length(samplings))
    dictkeys = Vector{eltype(keys(interpolationinfodict))}(undef, length(infos))

    for i in eachindex(samplings)
        matrix = Matrix{T}(undef, size(intersamplings[i]))
        interimmatrix = Matrix{T}(
            undef, Sampling.Nφ(samplings[i]), Sampling.Nθ(intersamplings[i])
        )

        infos[i] = _localglobalanterpolationinfo(
            stencilsize,
            samplings[i],
            intersamplings[i],
            matrix,
            interimmatrix,
            interpolationtrait,
        )
    end

    for (i, key) in enumerate(keys(interpolationinfodict))
        dictkeys[i] = (key[2], key[1])
    end

    return Dict(zip(dictkeys, infos))
end

function _localglobalanterpolationinfo(
    stencilsize::Int, sampling, intersampling, matrix, interimmatrix, interpolationtrait
)
    return _localglobalanterpolationinfo(
        Val(stencilsize), sampling, intersampling, matrix, interimmatrix, interpolationtrait
    )
end

function _localglobalanterpolationinfo(
    ::Val{stencilsize},
    sampling,
    intersampling,
    matrix,
    interimmatrix,
    ::DefaultInterpolation,
) where {stencilsize}
    return (
        virtual=LocalθGlobalφAnterpolationInformation{stencilsize}(
            intersampling,
            sampling,
            matrix,
            interimmatrix;
            interpolationtrait=VirtualExtension(),
        ),
        virtualflipθsign=LocalθGlobalφAnterpolationInformation{stencilsize}(
            intersampling,
            sampling,
            matrix,
            interimmatrix;
            interpolationtrait=VirtualExtensionFlipθSign(),
        ),
    )
end

function _localglobalanterpolationinfo(
    ::Val{stencilsize}, sampling, intersampling, matrix, interimmatrix, interpolationtrait
) where {stencilsize}
    return SVector(
        LocalθGlobalφAnterpolationInformation{stencilsize}(
            intersampling,
            sampling,
            matrix,
            interimmatrix;
            interpolationtrait=interpolationtrait,
        ),
    )
end

function _localglobalanterpolationinfo(
    ::Val{stencilsize},
    sampling,
    intersampling,
    matrix,
    interimmatrix,
    ::SphericalθφInterpolation,
) where {stencilsize}
    info = LocalθGlobalφAnterpolationInformation{stencilsize}(
        intersampling,
        sampling,
        matrix,
        interimmatrix;
        interpolationtrait=VirtualExtensionFlipθSign(),
    )

    return SVector(info, info)
end

function _localglobalanterpolationinfo(
    ::Val{stencilsize},
    sampling,
    intersampling,
    matrix,
    interimmatrix,
    ::SphericalInterpolation,
) where {stencilsize}
    infor = LocalθGlobalφAnterpolationInformation{stencilsize}(
        intersampling,
        sampling,
        matrix,
        interimmatrix;
        interpolationtrait=VirtualExtension(),
    )

    infoθφ = LocalθGlobalφAnterpolationInformation{stencilsize}(
        intersampling,
        sampling,
        matrix,
        interimmatrix;
        interpolationtrait=VirtualExtensionFlipθSign(),
    )

    return SVector(infor, infoθφ, infoθφ)
end

function _localglobalanterpolationinfo(
    ::Val{stencilsize},
    sampling,
    intersampling,
    matrix,
    interimmatrix,
    ::DoubleSphericalInterpolation,
) where {stencilsize}
    infor = LocalθGlobalφAnterpolationInformation{stencilsize}(
        intersampling,
        sampling,
        matrix,
        interimmatrix;
        interpolationtrait=VirtualExtension(),
    )

    infoθφ = LocalθGlobalφAnterpolationInformation{stencilsize}(
        intersampling,
        sampling,
        matrix,
        interimmatrix;
        interpolationtrait=VirtualExtensionFlipθSign(),
    )

    return SVector(infor, infoθφ, infoθφ, infor, infoθφ, infoθφ)
end
