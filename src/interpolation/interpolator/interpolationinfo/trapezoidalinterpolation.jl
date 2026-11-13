struct TrapezoidalInterpolationInformation{stencilsize,T1}
    conformindices::Vector{UnitRange{Int}}
    weights::Vector{SVector{stencilsize,T1}}
    nonconformindices::Vector{SVector{stencilsize,Int}}
    conformiterations::UnitRange{Int}
    nonconformiterations::Vector{Int}

    function TrapezoidalInterpolationInformation{stencilsize}(
        sampling::Sampling.SphericalSampling{T},
        interpolationsampling::Sampling.SphericalSampling{T},
    ) where {stencilsize,T}
        indices = Vector{Union{UnitRange{Int}}}(undef, Sampling.Nφ(interpolationsampling))

        nonconformindices = SVector{stencilsize,Int}[]

        weights = Vector{SVector{stencilsize,T}}(undef, Sampling.Nφ(interpolationsampling))

        equidistantweights = Interpolation.lagrangeweightsequidistant(stencilsize - 1, T)

        firstconformindex = 0
        lastconformindex = 0
        unconformiterations = Int[]

        for (iter, φ) in enumerate(Sampling.φs(interpolationsampling))
            φindex = Sampling._findφ(sampling, φ)
            _φindices = Interpolation.stencil(φindex, stencilsize)
            _localφs = Sampling.φ.(Ref(sampling), _φindices)

            _weights = _interpolationweights(equidistantweights, _localφs, φ)

            weights[iter] = SVector{stencilsize,T}(_weights)

            if all(>(0), _φindices) && all(<=(Sampling.Nφ(sampling)), _φindices)
                indices[iter] = _φindices
                if iszero(firstconformindex)
                    firstconformindex = iter
                else
                    lastconformindex = iter
                end
            else
                _nonconformindices = Int[]

                for (i, _φindex) in enumerate(_φindices)
                    if _φindex <= 0 || _φindex > Sampling.Nφ(sampling)
                        push!(
                            _nonconformindices,
                            Sampling.virtualextension(1, _φindex, sampling)[2],
                        )

                    else
                        push!(_nonconformindices, _φindex)
                    end
                end
                push!(nonconformindices, _nonconformindices)
                push!(unconformiterations, iter)
            end
        end

        lastconformindex = iszero(lastconformindex) ? firstconformindex : lastconformindex

        if iszero(firstconformindex)
            return new{stencilsize,T}(
                Union{UnitRange{Int}}[],
                weights,
                nonconformindices,
                0:-1,
                unconformiterations,
            )
        else
            conformiterations = firstconformindex:lastconformindex
            return new{stencilsize,T}(
                indices[conformiterations],
                weights,
                nonconformindices,
                conformiterations,
                unconformiterations,
            )
        end
    end
end
