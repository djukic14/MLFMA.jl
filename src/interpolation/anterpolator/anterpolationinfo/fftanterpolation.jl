
struct FFTAnterpolationInformation{FFT,BFFT,IFFT}
    fftplan::FFT
    bfftplan::BFFT
    ifftplan::IFFT
end

# Compatibility with callers supplying buffer collections. New construction passes one matrix
# per transform layout, independently of the number of workspace chunks.
function FFTAnterpolationInformation(fftmatrix::AbstractVector, bfftmatrix::AbstractVector)
    return FFTAnterpolationInformation(fftmatrix[1], bfftmatrix[1])
end

function FFTAnterpolationInformation(fftmatrix::AbstractMatrix, bfftmatrix::AbstractVector)
    return FFTAnterpolationInformation(fftmatrix, bfftmatrix[1])
end

function FFTAnterpolationInformation(fftmatrix::AbstractMatrix, bfftmatrix::AbstractMatrix)
    fftplan = plan_fft!(fftmatrix, 1)
    bfftplan = plan_bfft!(bfftmatrix, 1)
    ifftplan = inv(fftplan)
    return FFTAnterpolationInformation(fftplan, bfftplan, ifftplan)
end
