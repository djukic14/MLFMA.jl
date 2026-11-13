# The optional `postscale` keyword applies the level transfer's diagonal after interpolation.
# The interpolator does not store the phase; each family chooses how to apply the scaling.
struct SphericalLagrangeInterpolator{stencilsize,T,TSampling,Info,Trait} <:
       Interpolation.AbstractInterpolator{stencilsize,T}
    interpolationinfodict::Info
    interpolationtrait::Trait

    function SphericalLagrangeInterpolator{stencilsize,T}(
        TSampling::DataType; interpolationtrait=DefaultInterpolation()
    ) where {stencilsize,T}
        dict = Dict{
            Tuple{
                Sampling.SphericalSampling{TSampling},Sampling.SphericalSampling{TSampling}
            },
            SphericalInterpolationInformation{stencilsize,TSampling},
        }()

        return new{stencilsize,T,TSampling,typeof(dict),typeof(interpolationtrait)}(
            dict, interpolationtrait
        )
    end

    function SphericalLagrangeInterpolator{stencilsize}(
        T::DataType, TSampling::DataType; interpolationtrait=DefaultInterpolation()
    ) where {stencilsize}
        return SphericalLagrangeInterpolator{stencilsize,T}(
            TSampling; interpolationtrait=interpolationtrait
        )
    end

    function SphericalLagrangeInterpolator{stencilsize,T}(
        samplings::Vector{Sampling.SphericalSampling{TSampling}},
        intersamplings::Vector{Sampling.SphericalSampling{TSampling}};
        interpolationtrait=DefaultInterpolation(),
    ) where {stencilsize,T,TSampling}
        if isempty(samplings)
            return SphericalLagrangeInterpolator{stencilsize,T}(
                TSampling; interpolationtrait=interpolationtrait
            )
        end

        dict = getsphericalinterpolationinfodict(
            stencilsize, samplings, intersamplings, interpolationtrait
        )

        return new{stencilsize,T,TSampling,typeof(dict),typeof(interpolationtrait)}(
            dict, interpolationtrait
        )
    end

    function SphericalLagrangeInterpolator{stencilsize}(
        T::DataType,
        samplings::Vector{Sampling.SphericalSampling{TSampling}},
        intersamplings::Vector{Sampling.SphericalSampling{TSampling}};
        interpolationtrait=DefaultInterpolation(),
    ) where {stencilsize,TSampling}
        return SphericalLagrangeInterpolator{stencilsize,T}(
            samplings, intersamplings; interpolationtrait=interpolationtrait
        )
    end

    function SphericalLagrangeInterpolator{stencilsize,T}(
        sampling::Sampling.SphericalSampling{TSampling},
        intersampling::Sampling.SphericalSampling{TSampling};
        interpolationtrait=DefaultInterpolation(),
    ) where {stencilsize,T,TSampling}
        return SphericalLagrangeInterpolator{stencilsize,T}(
            [sampling], [intersampling]; interpolationtrait=interpolationtrait
        )
    end

    function SphericalLagrangeInterpolator{stencilsize}(
        T::DataType,
        sampling::Sampling.SphericalSampling{TSampling},
        intersampling::Sampling.SphericalSampling{TSampling};
        interpolationtrait=DefaultInterpolation(),
    ) where {stencilsize,TSampling}
        return SphericalLagrangeInterpolator{stencilsize,T}(
            sampling, intersampling; interpolationtrait=interpolationtrait
        )
    end
end

function (interpolator::Interpolation.SphericalLagrangeInterpolator{stencilsize,T})(
    farfields::AbstractArray{F,1},
    interpolatedfarfields::AbstractArray{F,1},
    farfieldsampling,
    interpolationsampling,
) where {stencilsize,F<:FarFields.AbstractFarField,T}
    return error(
        "Not implemented: This interpolator uses the special structure of SphericalSampling"
    )
end

function (interpolator::Interpolation.SphericalLagrangeInterpolator{stencilsize,T})(
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

    interimmatrix = Interpolation.takeinterim(
        Interpolation._interimmatrix(interim),
        T,
        (Sampling.Nθ(interpolationsampling), Sampling.Nφ(farfieldsampling)),
    )

    # Stage θ products in a contiguous vector, then copy each result into a strided interim row.
    # This avoids the temporary that `mul!` allocates for a strided output.
    interimθvector = Interpolation.takeinterim(
        Interpolation._interimθvector(interim), T, Sampling.Nφ(farfieldsampling)
    )

    information = interpolationinformation(
        interpolator, farfieldsampling, interpolationsampling, interpolationtrait
    )

    for matrixjd in eachindex(matrices)
        _interpolateθtransposed!(
            interimmatrix,
            matrices[matrixjd],
            information[matrixjd].θinformation,
            interimθvector,
        )
        _interpolateφtransposed!(
            resultmatrices[matrixjd], interimmatrix, information[matrixjd].φinformation
        )
    end

    # Scale in one contiguous pass. Fusing into the φ loop would scale strided rows instead.
    _applypostscale!(interpolatedfarfield, postscale)
    return nothing
end

_applypostscale!(farfield, ::Nothing) = farfield
_applypostscale!(farfield, postscale) =
    H2Factory.elementwisemul!(farfield, farfield, postscale)

function _interpolateθtransposed!(
    interimmatrix::AbstractArray{T,2},
    fvals::AbstractArray{T,2},
    θinformation::GaussLegendreInterpolationInformation{stencilsize,T1},
    interimvector::AbstractVector{T},
) where {T,T1,stencilsize}
    # EVERY product goes through `interimvector` before reaching the interim matrix. See the apply above
    # for why: the destinations here are strided row views, and writing a `mul!` result straight into one
    # allocates a temporary per call.
    for i in eachindex(θinformation.conformindices)
        weights = θinformation.weights[i]
        indices = θinformation.conformindices[i]

        iteration = θinformation.conformiterations[i]
        LinearAlgebra.mul!(interimvector, view(fvals, :, indices), weights)
        view(interimmatrix, iteration, :) .= interimvector
    end

    for i in eachindex(θinformation.posextendingindices)
        iter = θinformation.nonconformiterations[i]

        weights = θinformation.posextendingweights[i]
        indices = θinformation.posextendingindices[i]
        negweights = θinformation.negextendingweights[i]
        negindices = θinformation.negextendingindices[i]

        LinearAlgebra.mul!(interimvector, view(fvals, :, indices), weights)
        # Add the mirrored contribution as two contiguous row blocks. `mulrotatedrows!` derives
        # the rotation from the row count and may round differently from a permuted view.
        Interpolation.mulrotatedrows!(
            interimvector, fvals, negindices, negweights, true, true
        )
        view(interimmatrix, iter, :) .= interimvector
    end
end

function _interpolateφtransposed!(
    resultmatrix::M,
    interimmatrix::M,
    φinformation::TrapezoidalInterpolationInformation{stencilsize,T1},
) where {T,T1,stencilsize,M<:AbstractArray{T,2}}
    for (i, iter) in enumerate(φinformation.conformiterations)
        weights = φinformation.weights[iter]
        indices = φinformation.conformindices[i]

        @inbounds LinearAlgebra.mul!(
            view(resultmatrix, iter, :), view(interimmatrix, :, indices), weights
        )
    end

    for (i, iter) in enumerate(φinformation.nonconformiterations)
        weights = φinformation.weights[iter]
        indices = φinformation.nonconformindices[i]

        @inbounds LinearAlgebra.mul!(
            view(resultmatrix, iter, :), view(interimmatrix, :, indices), weights
        )
    end
end

function interpolationinformation(
    interpolator::Interpolation.SphericalLagrangeInterpolator{stencilsize},
    sampling::Sampling.SphericalSampling,
    interpolationsampling::Sampling.SphericalSampling,
    interpolationtrait,
) where {stencilsize}
    if haskey(interpolator.interpolationinfodict, (sampling, interpolationsampling))
        return converttrait(
            interpolationtrait,
            interpolator.interpolationinfodict[(sampling, interpolationsampling)],
            Interpolation.interpolationtrait(interpolator),
        )
    end

    return _sphericalinterpolationinfo(
        Val(stencilsize), sampling, interpolationsampling, interpolationtrait
    )
end
