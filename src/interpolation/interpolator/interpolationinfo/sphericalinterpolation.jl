#TODO: different stencil size in θ and φ (less strict requirements for φ problably)
struct SphericalInterpolationInformation{stencilsize,T1}
    θinformation::GaussLegendreInterpolationInformation{stencilsize,T1}
    φinformation::TrapezoidalInterpolationInformation{stencilsize,T1}

    function SphericalInterpolationInformation{stencilsize}(
        sampling::Sampling.SphericalSampling{T1},
        interpolationsampling::Sampling.SphericalSampling{T1};
        interpolationtrait=VirtualExtensionFlipθSign(),
    ) where {T1,stencilsize}
        θinformation = GaussLegendreInterpolationInformation{stencilsize}(
            sampling, interpolationsampling; interpolationtrait=interpolationtrait
        )
        φinformation = TrapezoidalInterpolationInformation{stencilsize}(
            sampling, interpolationsampling
        )

        return new{stencilsize,T1}(θinformation, φinformation)
    end
end

function getsphericalinterpolationinfodict(
    stencilsize,
    samplings::Vector{Sampling.SphericalSampling{TSampling}},
    interpolationsamplings::Vector{Sampling.SphericalSampling{TSampling}},
    interpolationtrait,
) where {TSampling}
    infos = Vector{
        typeof(
            _sphericalinterpolationinfo(
                stencilsize, samplings[end], interpolationsamplings[end], interpolationtrait
            ),
        ),
    }(
        undef, length(samplings)
    )
    dictkeys = Vector{Tuple{typeof(samplings[1]),typeof(samplings[1])}}(
        undef, length(samplings)
    )

    for i in eachindex(samplings)
        infos[i] = _sphericalinterpolationinfo(
            stencilsize, samplings[i], interpolationsamplings[i], interpolationtrait
        )
        dictkeys[i] = (samplings[i], interpolationsamplings[i])
    end

    return Dict(zip(dictkeys, infos))
end

function _sphericalinterpolationinfo(
    stencilsize::Int, sampling, interpolationsampling, interpolationtrait
)
    return _sphericalinterpolationinfo(
        Val(stencilsize), sampling, interpolationsampling, interpolationtrait
    )
end

function _sphericalinterpolationinfo(
    ::Val{stencilsize}, sampling, interpolationsampling, ::DefaultInterpolation
) where {stencilsize}
    info = SphericalInterpolationInformation{stencilsize}(
        sampling, interpolationsampling; interpolationtrait=VirtualExtension()
    )

    infoflipping = SphericalInterpolationInformation{stencilsize}(
        sampling, interpolationsampling; interpolationtrait=VirtualExtensionFlipθSign()
    )

    return (virtual=info, virtualflipθsign=infoflipping)
end

function _sphericalinterpolationinfo(
    ::Val{stencilsize}, sampling, interpolationsampling, interpolationtrait
) where {stencilsize}
    info = SphericalInterpolationInformation{stencilsize}(
        sampling, interpolationsampling; interpolationtrait=interpolationtrait
    )

    return SVector(info)
end

function _sphericalinterpolationinfo(
    ::Val{stencilsize}, sampling, interpolationsampling, ::SphericalθφInterpolation
) where {stencilsize}
    info = SphericalInterpolationInformation{stencilsize}(
        sampling, interpolationsampling; interpolationtrait=VirtualExtensionFlipθSign()
    )

    return SVector(info, info)
end

function _sphericalinterpolationinfo(
    ::Val{stencilsize}, sampling, interpolationsampling, ::SphericalInterpolation
) where {stencilsize}
    infor = SphericalInterpolationInformation{stencilsize}(
        sampling, interpolationsampling; interpolationtrait=VirtualExtension()
    )

    infoθφ = SphericalInterpolationInformation{stencilsize}(
        sampling, interpolationsampling; interpolationtrait=VirtualExtensionFlipθSign()
    )

    return SVector(infor, infoθφ, infoθφ)
end

function _sphericalinterpolationinfo(
    ::Val{stencilsize}, sampling, interpolationsampling, ::DoubleSphericalInterpolation
) where {stencilsize}
    infor = SphericalInterpolationInformation{stencilsize}(
        sampling, interpolationsampling; interpolationtrait=VirtualExtension()
    )

    infoθφ = SphericalInterpolationInformation{stencilsize}(
        sampling, interpolationsampling; interpolationtrait=VirtualExtensionFlipθSign()
    )

    return SVector(infor, infoθφ, infoθφ, infor, infoθφ, infoθφ)
end
