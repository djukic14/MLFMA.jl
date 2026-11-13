struct SphericalAnterpolationInformation{stencilsize,T1}
    θinformation::GaussLegendreAnterpolationInformation{stencilsize,T1}
    φinformation::TrapezoidalAnterpolationInformation{stencilsize,T1}

    function SphericalAnterpolationInformation{stencilsize}(
        sampling::Sampling.SphericalSampling{T},
        anterpolationsampling::Sampling.SphericalSampling{T};
        interpolationtrait=VirtualExtensionFlipθSign(),
    ) where {stencilsize,T}
        θinformation = GaussLegendreAnterpolationInformation{stencilsize}(
            sampling, anterpolationsampling; interpolationtrait=interpolationtrait
        )
        φinformation = TrapezoidalAnterpolationInformation{stencilsize}(
            sampling, anterpolationsampling
        )

        return new{stencilsize,T}(θinformation, φinformation)
    end
end

function getsphericalanterpolationinfodict(
    interpolator::Interpolation.SphericalLagrangeInterpolator{stencilsize,T,TSampling},
    interpolationtrait,
) where {stencilsize,T,TSampling}
    interpolationinfodict = interpolator.interpolationinfodict

    samplings = Vector{Sampling.SphericalSampling{TSampling}}()
    intersamplings = Vector{Sampling.SphericalSampling{TSampling}}()

    for key in keys(interpolationinfodict)
        push!(samplings, key[1])
        push!(intersamplings, key[2])
    end

    if isempty(samplings)
        return Dict{
            eltype(keys(interpolationinfodict)),
            SVector{SphericalAnterpolationInformation{stencilsize,T}},
        }()
    end

    infos = Vector{
        typeof(
            _sphericalanterpolationinfo(
                stencilsize, samplings[end], intersamplings[end], interpolationtrait
            ),
        ),
    }(
        undef, length(keys(interpolationinfodict))
    )

    dictkeys = Vector{eltype(keys(interpolationinfodict))}(undef, length(infos))

    for (i, key) in enumerate(keys(interpolationinfodict))
        infos[i] = _sphericalanterpolationinfo(
            stencilsize, samplings[i], intersamplings[i], interpolationtrait
        )
        dictkeys[i] = (key[2], key[1])
    end

    return Dict(zip(dictkeys, infos))
end

function _sphericalanterpolationinfo(
    stencilsize::Int, sampling, intersampling, interpolationtrait
)
    return _sphericalanterpolationinfo(
        Val(stencilsize), sampling, intersampling, interpolationtrait
    )
end

function _sphericalanterpolationinfo(
    ::Val{stencilsize}, sampling, intersampling, ::DefaultInterpolation
) where {stencilsize}
    return (
        virtual=SphericalAnterpolationInformation{stencilsize}(
            intersampling, sampling; interpolationtrait=VirtualExtension()
        ),
        virtualflipθsign=SphericalAnterpolationInformation{stencilsize}(
            intersampling, sampling; interpolationtrait=VirtualExtensionFlipθSign()
        ),
    )
end

function _sphericalanterpolationinfo(
    ::Val{stencilsize}, sampling, intersampling, interpolationtrait
) where {stencilsize}
    return SVector(
        SphericalAnterpolationInformation{stencilsize}(
            intersampling, sampling; interpolationtrait=interpolationtrait
        ),
    )
end

function _sphericalanterpolationinfo(
    ::Val{stencilsize}, sampling, intersampling, ::SphericalθφInterpolation
) where {stencilsize}
    info = SphericalAnterpolationInformation{stencilsize}(
        intersampling, sampling; interpolationtrait=VirtualExtensionFlipθSign()
    )

    return SVector(info, info)
end

function _sphericalanterpolationinfo(
    ::Val{stencilsize}, sampling, intersampling, ::SphericalInterpolation
) where {stencilsize}
    infor = SphericalAnterpolationInformation{stencilsize}(
        intersampling, sampling; interpolationtrait=VirtualExtension()
    )
    infoθφ = SphericalAnterpolationInformation{stencilsize}(
        intersampling, sampling; interpolationtrait=VirtualExtensionFlipθSign()
    )

    return SVector(infor, infoθφ, infoθφ)
end

function _sphericalanterpolationinfo(
    ::Val{stencilsize}, sampling, intersampling, ::DoubleSphericalInterpolation
) where {stencilsize}
    infor = SphericalAnterpolationInformation{stencilsize}(
        intersampling, sampling; interpolationtrait=VirtualExtension()
    )
    infoθφ = SphericalAnterpolationInformation{stencilsize}(
        intersampling, sampling; interpolationtrait=VirtualExtensionFlipθSign()
    )

    return SVector(infor, infoθφ, infoθφ, infor, infoθφ, infoθφ)
end
