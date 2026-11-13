
struct LocalθGlobalφAnterpolator{stencilsize,T,TSampling,Interpolator,DICT} <:
       Interpolation.AbstractAnterpolator{stencilsize,T}
    interpolator::Interpolator
    anterpolationinfodict::DICT

    function LocalθGlobalφAnterpolator{stencilsize,T}(
        TSampling::DataType; interpolationtrait=DefaultInterpolation()
    ) where {stencilsize,T}
        # It is not clear what FFT and BFFT should be here since they are determined by the
        # FFT and bFFT plans. We thus just set them to T since they are not used in this
        # case
        return LocalθGlobalφAnterpolator{stencilsize}(
            LocalθGlobalφInterpolator{stencilsize}(
                T, TSampling; interpolationtrait=interpolationtrait
            ),
        )
    end

    function LocalθGlobalφAnterpolator{stencilsize}(
        T::DataType, TSampling::DataType; interpolationtrait=DefaultInterpolation()
    ) where {stencilsize}
        return LocalθGlobalφAnterpolator{stencilsize,T}(
            TSampling; interpolationtrait=interpolationtrait
        )
    end

    # `nchunks` is accepted for compatibility only; FFT planning uses representative matrices.
    # Per-apply buffers belong to workspace scratch; see `interimscratch.jl`.
    function LocalθGlobalφAnterpolator{stencilsize,T}(
        interpolator::LocalθGlobalφInterpolator{stencilsize,T,TSampling};
        nchunks::Int=Threads.nthreads(),
    ) where {stencilsize,T,TSampling}
        if isempty(interpolator.interpolationinfodict)
            dict = Dict{
                Tuple{
                    Sampling.SphericalSampling{TSampling},
                    Sampling.SphericalSampling{TSampling},
                },
                LocalθGlobalφAnterpolationInformation{stencilsize,TSampling,T,T},
            }()

            return new{stencilsize,T,TSampling,typeof(interpolator),typeof(dict)}(
                interpolator, dict
            )
        end

        dict = getlocalglobalanterpolationinfodict(
            T,
            interpolator.interpolationinfodict,
            stencilsize,
            interpolationtrait(interpolator),
            nchunks,
        )

        return new{stencilsize,T,TSampling,typeof(interpolator),typeof(dict)}(
            interpolator, dict
        )
    end

    function LocalθGlobalφAnterpolator{stencilsize}(
        interpolator::LocalθGlobalφInterpolator{stencilsize,T,TSampling};
        nchunks::Int=Threads.nthreads(),
    ) where {stencilsize,T,TSampling}
        return LocalθGlobalφAnterpolator{stencilsize,T}(interpolator; nchunks=nchunks)
    end
end

function anterpolator(
    interpolator::LocalθGlobalφInterpolator{stencilsize}; nchunks::Int=Threads.nthreads()
) where {stencilsize}
    return LocalθGlobalφAnterpolator{stencilsize}(interpolator; nchunks=nchunks)
end

function (anterpolator::Interpolation.LocalθGlobalφAnterpolator{stencilsize,T})(
    farfields::AbstractArray{F,1},
    anterpolatedfarfields::AbstractArray{F,1},
    farfieldsampling,
    anterpolationsampling,
) where {stencilsize,F<:FarFields.AbstractFarField,T}
    return error(
        "Not implemented: This anterpolator uses the special structure of SphericalSampling"
    )
end

function (anterpolator::Interpolation.LocalθGlobalφAnterpolator{stencilsize,T})(
    anterpolatedfarfield::F,
    farfield::F,
    farfieldsampling::Sampling.SphericalSampling{T1},
    anterpolationsampling::Sampling.SphericalSampling{T1};
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
    interim=nothing,
    prescale=nothing,
    stagetransform=identity,
) where {stencilsize,T,F<:FarFields.AbstractFarField,T1}
    @assert Sampling.Nθ(farfieldsampling) == Sampling.Nθ(farfield)
    @assert Sampling.Nφ(farfieldsampling) == Sampling.Nφ(farfield)

    resultmatrices = H2Factory.moments(anterpolatedfarfield)
    matrices = H2Factory.moments(farfield)

    interimsampling = Sampling.SphericalSampling{Float64}(
        Sampling.Nθ(farfieldsampling), Sampling.Nφ(anterpolationsampling)
    )

    # `farfieldsampling` is the PARENT here (anterpolation runs parent -> child), which is why the
    # `ftpadmatrix` slot serves both this and the interpolator's padded FT buffer.
    shiftedmatrix = Interpolation.takeinterim(
        Interpolation._ftpadmatrix(interim), T, size(farfieldsampling)
    )
    interimmatrix = Interpolation.takeinterim(
        Interpolation._interimmatrix(interim), T, size(interimsampling)
    )

    anterpolationinformation = Interpolation.anterpolationinformation(
        anterpolator,
        farfieldsampling,
        anterpolationsampling,
        matrices[1],
        interimmatrix,
        interpolationtrait,
    )
    interimθvector = Interpolation.takeinterim(
        Interpolation._interimθvector(interim), T, Sampling.Nφ(anterpolationsampling)
    )

    # Apply the parent-grid diagonal before the FFT, using private staging to preserve the caller's
    # input. The kernel may discard this staged copy and skip the inverse FFT used for restoration.
    prescalematrix = Interpolation.takeinterim(
        Interpolation._prescalematrix(interim), T, size(farfieldsampling)
    )

    for matrixjd in eachindex(matrices)
        source = _stageprescaled!(
            prescalematrix, matrices[matrixjd], prescale, stagetransform
        )
        Interpolation._anterpolateφbfft!(
            interimmatrix,
            source,
            shiftedmatrix,
            anterpolationinformation[matrixjd].φinformation,
            (Sampling.Nφ(anterpolationsampling), Sampling.Nθ(farfieldsampling)),
            prescale === nothing,
        )

        Interpolation._anterpolateθ!(
            resultmatrices[matrixjd],
            interimmatrix,
            anterpolationinformation[matrixjd].θinformation,
            interimθvector,
        )
    end

    return nothing
end

function _anterpolateφbfft!(
    result::Matrix{T},
    matrix,
    shiftedmatrix::Matrix{T},
    φinfo::FFTAnterpolationInformation{FFT,BFFT,IFFT},
    newsize::Tuple{I,I},
    restoreinput::Bool,
) where {T<:Complex,I<:Integer,FFT,BFFT,IFFT}
    fftplan = φinfo.fftplan
    bfftplan = φinfo.bfftplan
    ifftplan = φinfo.ifftplan

    anterN = newsize[1]
    N = size(matrix, 1)
    Npad = floor(Int, N / 2 - anterN / 2)

    fftnopad = @view shiftedmatrix[(Npad + 1):(end - Npad), :]

    fftplan * matrix

    fftshift!(shiftedmatrix, matrix, 1)
    fftshift!(result, fftnopad, 1)

    bfftplan * result
    LinearAlgebra.mul!(result, result, 1 / anterN)

    # Restore the caller's input after the in-place FFT, but discard a private staged copy.
    restoreinput && (ifftplan * matrix)

    return nothing
end

# Stage `in .* prescale` into the apply's own parent-grid buffer, or hand the input straight back when
# there is no diagonal. Without `prescale`, no copy or transform is applied and the kernel must
# restore any input it mutates.
_stageprescaled!(::Any, matrix, ::Nothing, ::Any) = matrix

function _stageprescaled!(staging, matrix, prescale, transform)
    @inbounds for i in eachindex(staging, matrix, prescale)
        staging[i] = transform(matrix[i] * prescale[i])
    end
    return staging
end

function _anterpolateθ!(
    resultmatrix::AbstractArray{T,2},
    fvals::AbstractArray{T,2},
    θinformation::GaussLegendreAnterpolationInformation{stencilsize,T1},
    interimvector::Vector{T},
) where {T,T1,stencilsize}
    # `mulrotatedrows!` recovers the rotation from the row count. The shared information type also
    # stores `flippingφindices` for the Lagrange kernel, which indexes through that permutation.
    for i in eachindex(θinformation.conformindices)
        weights = θinformation.weights[i]
        indices = θinformation.conformindices[i]

        localfvals = view(fvals, :, indices)
        LinearAlgebra.mul!(interimvector, localfvals, weights)

        view(resultmatrix, :, i) .= interimvector
    end

    for i in eachindex(θinformation.nonconformindices)
        indices = θinformation.nonconformindices[i]
        # An empty range means this child θ receives no extended contribution.
        isempty(indices) && continue

        weights = θinformation.nonconformweights[i]
        # Two contiguous row blocks rather than `view(fvals, flippingindices, indices)`: the flipping
        # index vector rotates by Nφ/2 and makes the view non-strided. See `mulrotatedrows!`.
        Interpolation.mulrotatedrows!(interimvector, fvals, indices, weights)
        view(resultmatrix, :, i) .+= interimvector
    end
end

function anterpolationinformation(
    anterpolator::LocalθGlobalφAnterpolator{stencilsize},
    farfieldsampling::Sampling.SphericalSampling,
    anterpolationsampling::Sampling.SphericalSampling,
    matrix,
    interimmatrix,
    interpolationtrait,
) where {stencilsize}
    return anterpolationinformation(
        anterpolator,
        Interpolation.interpolationtrait(anterpolator),
        farfieldsampling,
        anterpolationsampling,
        matrix,
        interimmatrix,
        interpolationtrait,
    )
end

function anterpolationinformation(
    anterpolator::LocalθGlobalφAnterpolator{stencilsize},
    anterpolatortrait,
    farfieldsampling::Sampling.SphericalSampling,
    anterpolationsampling::Sampling.SphericalSampling,
    matrix,
    interimmatrix,
    interpolationtrait,
) where {stencilsize}
    if haskey(anterpolator.anterpolationinfodict, (farfieldsampling, anterpolationsampling))
        return converttrait(
            interpolationtrait,
            anterpolator.anterpolationinfodict[(farfieldsampling, anterpolationsampling)],
            anterpolatortrait,
        )
    end

    return _localglobalanterpolationinfo(
        Val(stencilsize),
        anterpolationsampling,
        farfieldsampling,
        matrix,
        interimmatrix,
        interpolationtrait,
    )
end
