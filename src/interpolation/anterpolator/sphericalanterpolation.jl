
# Anterpolation is adjoint interpolation. For this real-weight family it also equals the transpose.
struct SphericalLagrangeAnterpolator{stencilsize,T,TSampling,Interpolator,DICT} <:
       Interpolation.AbstractAnterpolator{stencilsize,T}
    interpolator::Interpolator
    anterpolationinfodict::DICT

    function SphericalLagrangeAnterpolator{stencilsize,T}(
        TSampling::DataType; interpolationtrait=DefaultInterpolation()
    ) where {stencilsize,T}
        dict = Dict{
            Tuple{
                Sampling.SphericalSampling{TSampling},Sampling.SphericalSampling{TSampling}
            },
            SphericalInterpolationInformation{stencilsize,TSampling},
        }()
        interpolator = Interpolation.SphericalLagrangeInterpolator{stencilsize,T}(
            TSampling; interpolationtrait=interpolationtrait
        )

        return new{stencilsize,T,TSampling,typeof(interpolator),typeof(dict)}(
            interpolator, dict
        )
    end

    function SphericalLagrangeAnterpolator{stencilsize}(
        T::DataType, TSampling::DataType; interpolationtrait=DefaultInterpolation()
    ) where {stencilsize}
        return SphericalLagrangeAnterpolator{stencilsize,T}(
            TSampling; interpolationtrait=interpolationtrait
        )
    end

    function SphericalLagrangeAnterpolator{stencilsize,T}(
        interpolator::SphericalLagrangeInterpolator{stencilsize,T,TSampling}
    ) where {stencilsize,T,TSampling}
        dict = getsphericalanterpolationinfodict(
            interpolator, interpolationtrait(interpolator)
        )

        return new{stencilsize,T,TSampling,typeof(interpolator),typeof(dict)}(
            interpolator, dict
        )
    end

    function SphericalLagrangeAnterpolator{stencilsize}(
        interpolator::Interpolation.SphericalLagrangeInterpolator{stencilsize,T,TSampling}
    ) where {stencilsize,T,TSampling}
        return SphericalLagrangeAnterpolator{stencilsize,T}(interpolator)
    end
end

# Accept but ignore `nchunks` so map constructors can pass it to every family.
# This family gets its buffers from workspace scratch and builds no FFT plans.
function anterpolator(
    interpolator::Interpolation.SphericalLagrangeInterpolator{stencilsize};
    nchunks::Int=Threads.nthreads(),
) where {stencilsize}
    return SphericalLagrangeAnterpolator{stencilsize}(interpolator)
end

function (anterpolator::Interpolation.SphericalLagrangeAnterpolator{stencilsize,T})(
    farfields::AbstractArray{F,1},
    anterpolatedfarfields::AbstractArray{F,1},
    farfieldsampling,
    anterpolationsampling,
) where {stencilsize,F<:FarFields.AbstractFarField,T}
    return error(
        "Not implemented: This anterpolator uses the special structure of SphericalSampling"
    )
end

function (anterpolator::Interpolation.SphericalLagrangeAnterpolator{stencilsize,T})(
    anterpolatedfarfield::F,
    farfield::F,
    farfieldsampling::Sampling.SphericalSampling{T1},
    anterpolationsampling::Sampling.SphericalSampling{T1};
    interpolationtrait=defaultinterpolationtrait(F),
    interim=nothing,
    prescale=nothing,
    stagetransform=identity,
) where {stencilsize,T,F<:FarFields.AbstractFarField,T1}
    @assert Sampling.Nθ(farfieldsampling) == Sampling.Nθ(farfield)
    @assert Sampling.Nφ(farfieldsampling) == Sampling.Nφ(farfield)

    resultmatrices = H2Factory.moments(anterpolatedfarfield)
    matrices = H2Factory.moments(farfield)

    # Anterpolation runs parent -> child, so `farfieldsampling` is the PARENT and
    # `anterpolationsampling` the child -- the mirror image of the interpolator's naming, which is why
    # one `interimmatrix` shape serves both directions.
    interimmatrix = Interpolation.takeinterim(
        Interpolation._interimmatrix(interim),
        T,
        (Sampling.Nθ(farfieldsampling), Sampling.Nφ(anterpolationsampling)),
    )
    interimφvector = Interpolation.takeinterim(
        Interpolation._interimφvector(interim), T, Sampling.Nθ(farfieldsampling)
    )
    interimθvector = Interpolation.takeinterim(
        Interpolation._interimθvector(interim), T, Sampling.Nφ(anterpolationsampling)
    )

    anterpolationinfo = Interpolation.anterpolationinformation(
        anterpolator, farfieldsampling, anterpolationsampling, interpolationtrait
    )

    # Stage `prescale` in a private parent-grid buffer, using the same diagonal contract as the
    # local/global family. These kernels do not mutate their input.
    prescalematrix = Interpolation.takeinterim(
        Interpolation._prescalematrix(interim), T, size(farfieldsampling)
    )

    for matrixjd in eachindex(matrices)
        source = Interpolation._stageprescaled!(
            prescalematrix, matrices[matrixjd], prescale, stagetransform
        )
        _anterpolateφtransposed!(
            interimmatrix, source, anterpolationinfo[matrixjd].φinformation, interimφvector
        )
        _anterpolateθtransposed!(
            resultmatrices[matrixjd],
            interimmatrix,
            anterpolationinfo[matrixjd].θinformation,
            interimθvector,
        )
    end

    return nothing
end

function _anterpolateφtransposed!(
    interimmatrix::I,
    fvals::F,
    φinformation::TrapezoidalAnterpolationInformation{stencilsize,T1},
    interimvector::Vector{T},
) where {T,T1,stencilsize,F<:AbstractArray{T,2},I<:AbstractArray{T,2}}
    for i in eachindex(φinformation.conformindices)
        weights = φinformation.weights[i]
        indices = φinformation.conformindices[i]

        @inbounds localfvals = view(transpose(fvals), :, indices)

        LinearAlgebra.mul!(interimvector, localfvals, weights)

        interimmatrix[:, i] .= interimvector
    end
end

function _anterpolateθtransposed!(
    resultmatrix::AbstractArray{T,2},
    fvals::AbstractArray{T,2},
    θinformation::GaussLegendreAnterpolationInformation{stencilsize,T1},
    interimvector::Vector{T},
) where {T,T1,stencilsize}
    flippingindices = θinformation.flippingφindices

    for i in eachindex(θinformation.conformindices)
        weights = θinformation.weights[i]
        indices = θinformation.conformindices[i]

        localfvals = view(transpose(fvals), :, indices)

        LinearAlgebra.mul!(interimvector, localfvals, weights)

        view(resultmatrix, :, i) .= interimvector
    end

    for i in eachindex(θinformation.nonconformindices)
        indices = θinformation.nonconformindices[i]
        # An empty range means this child θ receives no extended contribution. Both consumers of
        # `GaussLegendreAnterpolationInformation` use this sentinel.
        isempty(indices) && continue

        weights = θinformation.nonconformweights[i]
        localfvals = view(transpose(fvals), flippingindices, indices)

        LinearAlgebra.mul!(interimvector, localfvals, weights)

        view(resultmatrix, :, i) .+= interimvector
    end
end

function anterpolationinformation(
    anterpolator::Interpolation.SphericalLagrangeAnterpolator,
    farfieldsampling::Sampling.SphericalSampling,
    anterpolationsampling::Sampling.SphericalSampling,
    interpolationtrait,
)
    return anterpolationinformation(
        anterpolator,
        Interpolation.interpolationtrait(anterpolator),
        farfieldsampling,
        anterpolationsampling,
        interpolationtrait,
    )
end

function anterpolationinformation(
    anterpolator::SphericalLagrangeAnterpolator{stencilsize},
    anterpolatortrait,
    farfieldsampling::Sampling.SphericalSampling,
    anterpolationsampling::Sampling.SphericalSampling,
    interpolationtrait,
) where {stencilsize}
    if haskey(anterpolator.anterpolationinfodict, (farfieldsampling, anterpolationsampling))
        return converttrait(
            interpolationtrait,
            anterpolator.anterpolationinfodict[(farfieldsampling, anterpolationsampling)],
            anterpolatortrait,
        )
    end

    return _sphericalanterpolationinfo(
        Val(stencilsize), anterpolationsampling, farfieldsampling, interpolationtrait
    )
end
