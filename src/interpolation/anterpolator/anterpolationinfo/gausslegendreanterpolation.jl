# `UnitRange` column indices keep the views strided so `mul!` can use its fast kernel.
# Skipping zero weights can leave gaps, as in `[1, 2, 4, 5]`. `spanningranges` spans each set
# and fills the gaps with zero weights, preserving the weighted sum.
#
# Empty ranges mark entries with no contribution. `isassigned` cannot mark missing entries
# in a vector of the bits type `UnitRange{Int}`.
#
# Both `_anterpolateθtransposed!` (Lagrange) and `_anterpolateθ!` (global-φ) consume this type.
# `test/LevelTransfers/interpolation.jl` checks them with one shared information object.
# Their mirrored-row products may sum in different orders, so compare them approximately.
struct GaussLegendreAnterpolationInformation{stencilsize,T1}
    conformindices::Vector{UnitRange{Int}}
    weights::Vector{Vector{T1}}
    nonconformindices::Vector{UnitRange{Int}}
    nonconformweights::Vector{Vector{T1}}
    flippingφindices::Vector{Int}

    function GaussLegendreAnterpolationInformation{stencilsize}(
        sampling::Sampling.SphericalSampling{T},
        anterpolationsampling::Sampling.SphericalSampling{T};
        interpolationtrait=VirtualExtensionFlipθSign(),
    ) where {stencilsize,T}
        I = Int
        indices = Vector{Vector{Int}}(undef, Sampling.Nθ(anterpolationsampling))
        weights = Vector{Vector{T}}(undef, Sampling.Nθ(anterpolationsampling))

        nonconformindices = Vector{Vector{Int}}(undef, Sampling.Nθ(anterpolationsampling))
        nonconformweights = Vector{Vector{T}}(undef, Sampling.Nθ(anterpolationsampling))
        lastθindex = 1

        _localθs = Vector{T}(undef, stencilsize)

        for (iter, θ) in enumerate(Sampling.θs(sampling))
            θindex = if isodd(stencilsize)
                Sampling._findθclosest(anterpolationsampling, θ, lastθindex)
            else
                Sampling._findθfirst(anterpolationsampling, θ, lastθindex)
            end

            lastθindex = θindex

            _θindices = _θinformation!(_localθs, θindex, stencilsize, anterpolationsampling)
            _weights = _interpolationweights(
                Interpolation.lagrangeweights(_localθs), _localθs, θ
            )

            for (i, _θindex) in enumerate(_θindices)
                iszero(_weights[i]) && continue

                if _θindex > 0 && _θindex <= Sampling.Nθ(anterpolationsampling)
                    if isassigned(indices, _θindex)
                        push!(indices[_θindex], iter)
                        push!(weights[_θindex], _weights[i])
                    else
                        indices[_θindex] = [iter]
                        weights[_θindex] = [_weights[i]]
                    end
                else
                    conformθindex = Sampling.virtualextension(
                        _θindex, 1, anterpolationsampling
                    )[1]
                    if isassigned(nonconformindices, conformθindex)
                        push!(nonconformindices[conformθindex], iter)
                        push!(
                            nonconformweights[conformθindex],
                            _getnegativeweights(_weights[i], interpolationtrait),
                        )
                    else
                        nonconformindices[conformθindex] = [iter]
                        nonconformweights[conformθindex] = [
                            _getnegativeweights(_weights[i], interpolationtrait)
                        ]
                    end
                end
            end
        end

        flippingφindices = vcat(
            ((Sampling.Nφ(anterpolationsampling) ÷ 2) + 1):Sampling.Nφ(
                anterpolationsampling
            ),
            1:(Sampling.Nφ(anterpolationsampling) ÷ 2),
        )

        # Built by pushing, stored as ranges with gap-padded weights -- see `MLFMA.spanningranges` for why the
        # storage type matters and why the sets are not always contiguous to begin with.
        conformranges, conformweights = Interpolation.spanningranges(indices, weights, T)
        nonconformranges, paddednonconformweights = Interpolation.spanningranges(
            nonconformindices, nonconformweights, T
        )

        return new{stencilsize,T}(
            conformranges,
            conformweights,
            nonconformranges,
            paddednonconformweights,
            flippingφindices,
        )
    end
end
