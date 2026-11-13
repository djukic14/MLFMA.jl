struct FFTInterpolationInformation{FFT,IFFT}
    fftplan::FFT
    ifftplan::IFFT
end

function FFTInterpolationInformation(
    fftmatrix::AbstractArray{T,2}, ifftmatrix::AbstractArray{T,2}
) where {T}
    fftplan = plan_fft!(fftmatrix, 1)
    ifftplan = plan_ifft!(ifftmatrix, 1)
    return FFTInterpolationInformation(fftplan, ifftplan)
end
