
struct TrapezoidalAnterpolationInformation{stencilsize,T1}
    conformindices::Vector{Vector{Int}}
    weights::Vector{Vector{T1}}

    function TrapezoidalAnterpolationInformation{stencilsize}(
        sampling::Sampling.SphericalSampling{T},
        anterpolationsampling::Sampling.SphericalSampling{T},
    ) where {stencilsize,T}
        indices = Vector{Vector{Int}}(undef, Sampling.Nφ(anterpolationsampling))
        weights = Vector{Vector{T}}(undef, Sampling.Nφ(anterpolationsampling))

        equidistantweights = Interpolation.lagrangeweightsequidistant(stencilsize - 1, T)
        for (iter, φ) in enumerate(Sampling.φs(sampling))
            φindex = Sampling._findφ(anterpolationsampling, φ)
            _φindices = Interpolation.stencil(φindex, stencilsize)

            _localφs = Sampling.φ.(Ref(anterpolationsampling), _φindices)

            _weights = _interpolationweights(equidistantweights, _localφs, φ)

            for (i, _φindex) in enumerate(_φindices)
                conformφindex =
                    if _φindex > 0 && _φindex <= Sampling.Nφ(anterpolationsampling)
                        _φindex
                    else
                        Sampling.virtualextension(1, _φindex, anterpolationsampling)[2]
                    end

                if isassigned(indices, conformφindex)
                    push!(indices[conformφindex], iter)
                    push!(weights[conformφindex], _weights[i])
                else
                    indices[conformφindex] = [iter]
                    weights[conformφindex] = [_weights[i]]
                end
            end
        end
        return new{stencilsize,T}(indices, weights)
    end
end
