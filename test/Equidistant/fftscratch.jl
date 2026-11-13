module EquidistantFFTScratchTests

# The buffer-reusing spectral path: same operator, no allocations.
#
# TWO CLAIMS, AND THEY NEED DIFFERENT EVIDENCE. That the buffered path computes the same thing is
# checked against the ALLOCATING REFERENCE in `fftbackend.jl`, which exists for exactly this purpose.
# That it allocates nothing is checked with `@allocated` on a warmed-up apply, asserted as `== 0`
# rather than "small" -- a threshold would drift upward one commit at a time.
#
# WHY NOT BITWISE IN THE FORWARD DIRECTION. The adjoint agrees with its reference to the bit; the
# forward direction agrees to 3.3e-16 and NOT bitwise. That is FFTW, not a mistake in the arithmetic:
# `plan_fft!` and out-of-place `fft` select different algorithms for the same size, so their last bits
# differ. Everything this file controls -- the order of the normalization and phase multiply, the
# stage order, the coefficient arithmetic -- is written to match the reference exactly, and the
# residual is the transform itself. Worth stating because `≈` here is a weaker claim than the one
# `_scaleresult!` in the local/global family can make, and the difference has a specific cause.

using Test
using MLFMA
using LinearAlgebra
using MLFMA.Equidistant:
    EquidistantSampling,
    EquidistantInterpolator,
    FFTScratch,
    FFTBackend,
    LagrangeBackend,
    SPHERICAL,
    ANTISPHERICAL,
    resample,
    resample!,
    resampleadjoint,
    resampleadjoint!,
    extend,
    resamplecoefficients,
    resamplecoefficients!,
    resamplereflectedcoefficients,
    resamplereflectedcoefficients!,
    resamplereflectedcoefficientsadjoint

const T = ComplexF64
const CHILD = EquidistantSampling{Float64}(18, 36)
const PARENT = EquidistantSampling{Float64}(26, 52)

@testset "resamplecoefficients! matches the allocating form bitwise" begin
    for (Nsrc, Ndst) in ((36, 52), (52, 36), (36, 36), (8, 16), (16, 8))
        v = randn(T, Nsrc)
        want = resamplecoefficients(v, Nsrc, Ndst)
        got = resamplecoefficients!(Vector{T}(undef, Ndst), v, Nsrc, Ndst)
        @test got == want
    end

    # It must ZERO the buffer, not accumulate into whatever was there. Reusing a dirty buffer is the
    # failure mode the in-place form introduces and the allocating one cannot have.
    dirty = fill(T(999), 52)
    v = randn(T, 36)
    @test resamplecoefficients!(dirty, v, 36, 52) == resamplecoefficients(v, 36, 52)
end

@testset "the reflected coefficient move and its adjoint" begin
    for (Nsrc, Ndst) in ((36, 52), (52, 36), (36, 36), (8, 16), (16, 8))
        v = randn(T, Nsrc)
        want = resamplereflectedcoefficients(v, Nsrc, Ndst)
        @test resamplereflectedcoefficients!(Vector{T}(undef, Ndst), v, Nsrc, Ndst) == want

        w = randn(T, Ndst)
        # Allocating only: the reduced θ path resamples by head copy and never calls a buffered
        # adjoint coefficient mover, so the in-place form was removed rather than left unreachable.
        adj = resamplereflectedcoefficientsadjoint(w, Nsrc, Ndst)

        # The defining property, checked as a bilinear form rather than entry by entry: whatever the
        # top-bin bookkeeping turns out to be, the two routines must be exact transposes. `sum(w .* R v)`
        # against `sum(Rᵀw .* v)` -- no conjugation, because this is the transpose, not the adjoint.
        @test sum(w .* want) ≈ sum(adj .* v)

        # It differs from the plain move ONLY at the top bin, and only when the grids differ. Pins that
        # this is a targeted change and not a second, divergent implementation.
        plain = resamplecoefficients(v, Nsrc, Ndst)
        differing = findall(!iszero, want - plain)
        keep = min(Nsrc, Ndst) ÷ 2
        @test all(i -> abs(MLFMA.Equidistant._signedfrequency(i, Ndst)) == keep, differing)
    end
end

@testset "the buffered spectral path computes the reference operator" begin
    scratch = FFTScratch(CHILD, PARENT, T)
    F = randn(T, 36, 18)
    Y = randn(T, 52, 26)
    for parity in (SPHERICAL, ANTISPHERICAL)
        want = resample(F, CHILD, PARENT, parity, FFTBackend())
        got = resample!(similar(want), F, CHILD, PARENT, parity, FFTBackend(), scratch)
        # `≈` not `==`, and the header says why: in-place and out-of-place FFTW plans differ in the
        # last bits. 1e-14 is far tighter than the 3.3e-16 measured, and far looser than bitwise.
        @test norm(got - want) / norm(want) < 1e-14

        wantadj = resampleadjoint(Y, CHILD, PARENT, parity, FFTBackend())
        gotadj = resampleadjoint!(
            similar(wantadj), Y, CHILD, PARENT, parity, FFTBackend(), scratch
        )
        # NOT bitwise any more, and deliberately so: the reduced path reaches this operator through
        # DCT/DST transforms on Nθ points, which is different arithmetic from the reference's complex
        # FFT on 2Nθ. Bitwise agreement was a property of mirroring the reference stage for stage, and
        # that mirroring is exactly what dropping the doubling gave up.
        @test norm(gotadj - wantadj) / norm(wantadj) < 1e-14
    end
end

@testset "reusing one scratch does not leak state between calls" begin
    # The buffers are written by every stage of every call, so a stage that reads before writing would
    # produce a right answer once and a wrong one afterwards. Running the same input twice through one
    # scratch, with a DIFFERENT parity in between, is what catches that.
    scratch = FFTScratch(CHILD, PARENT, T)
    F = randn(T, 36, 18)
    first = copy(
        resample!(zeros(T, 52, 26), F, CHILD, PARENT, SPHERICAL, FFTBackend(), scratch)
    )
    resample!(zeros(T, 52, 26), F, CHILD, PARENT, ANTISPHERICAL, FFTBackend(), scratch)
    resampleadjoint!(
        zeros(T, 36, 18), randn(T, 52, 26), CHILD, PARENT, SPHERICAL, FFTBackend(), scratch
    )
    again = resample!(zeros(T, 52, 26), F, CHILD, PARENT, SPHERICAL, FFTBackend(), scratch)
    @test again == first
end

@testset "a scratch-backed apply allocates exactly nothing" begin
    for (nc, np) in ((18, 26), (40, 64))
        child = EquidistantSampling{Float64}(nc, 2nc)
        parent = EquidistantSampling{Float64}(np, 2np)
        interpolator = EquidistantInterpolator{T}(
            FFTBackend(); interpolationtrait=MLFMA.SphericalθφInterpolation()
        )
        anterpolator = MLFMA.anterpolator(interpolator; nchunks=1)
        scratch = MLFMA.interimscratch(anterpolator, child, parent)
        @test scratch !== nothing

        input = MLFMA.TangentialSphericalFarField(T, child, Val(:zero))
        input[1] .= randn(T, size(input[1]))
        input[2] .= randn(T, size(input[2]))
        up = MLFMA.TangentialSphericalFarField(T, parent, Val(:zero))
        down = MLFMA.TangentialSphericalFarField(T, child, Val(:zero))

        interpolator(up, input, child, parent; interim=scratch)      # warm up
        anterpolator(down, up, parent, child; interim=scratch)
        interpolator(up, input, child, parent; interim=scratch)
        anterpolator(down, up, parent, child; interim=scratch)

        # `== 0`, not a threshold. Measured before this path existed: 739,664 B interpolating and
        # 520,000 B anterpolating at 18x36 -> 26x52; 4,058,608 and 2,764,000 at 40x80 -> 64x128.
        @test (@allocated interpolator(up, input, child, parent; interim=scratch)) == 0
        @test (@allocated anterpolator(down, up, parent, child; interim=scratch)) == 0
    end
end

@testset "the scratch is optional and mismatches fall back rather than corrupt" begin
    interpolator = EquidistantInterpolator{T}(
        FFTBackend(); interpolationtrait=MLFMA.SphericalθφInterpolation()
    )
    input = MLFMA.TangentialSphericalFarField(T, CHILD, Val(:zero))
    input[1] .= randn(T, size(input[1]))
    input[2] .= randn(T, size(input[2]))

    noscratch = MLFMA.TangentialSphericalFarField(T, PARENT, Val(:zero))
    interpolator(noscratch, input, CHILD, PARENT)

    # A scratch built for the WRONG level pair must be ignored, not written through -- its buffers are
    # indexed under `@inbounds`, so silently accepting it would be an out-of-bounds write rather than
    # an error. This is the same tolerance `MLFMA.takeinterim` promises for the other families.
    wrong = (
        fftscratch=FFTScratch(
            EquidistantSampling{Float64}(4, 8), EquidistantSampling{Float64}(6, 12), T
        ),
    )
    mismatched = MLFMA.TangentialSphericalFarField(T, PARENT, Val(:zero))
    interpolator(mismatched, input, CHILD, PARENT; interim=wrong)
    @test mismatched[1] == noscratch[1]

    # A NamedTuple with no `fftscratch` field at all (several hand-built ones exist in the level
    # transfer tests) must also fall back rather than raise a FieldError.
    unrelated = (interimmatrix=zeros(T, 2, 2),)
    other = MLFMA.TangentialSphericalFarField(T, PARENT, Val(:zero))
    interpolator(other, input, CHILD, PARENT; interim=unrelated)
    @test other[1] == noscratch[1]
end

@testset "the Lagrange family is scratch-backed too" begin
    # It used to decline scratch, and this testset asserted the `nothing` so it read as a decision.
    # It now provides one -- and most of what it hoists are STENCILS, which depend only on the two
    # grids and the order and were being rebuilt identically on every apply.
    lagrange = EquidistantInterpolator{T}(
        LagrangeBackend(6); interpolationtrait=MLFMA.SphericalθφInterpolation()
    )
    anterpolator = MLFMA.anterpolator(lagrange; nchunks=1)
    scratch = MLFMA.interimscratch(anterpolator, CHILD, PARENT)
    @test scratch !== nothing

    input = MLFMA.TangentialSphericalFarField(T, CHILD, Val(:zero))
    input[1] .= randn(T, size(input[1]))
    input[2] .= randn(T, size(input[2]))
    up = MLFMA.TangentialSphericalFarField(T, PARENT, Val(:zero))
    down = MLFMA.TangentialSphericalFarField(T, CHILD, Val(:zero))

    # BITWISE against the no-scratch path, which the spectral family cannot claim: there is no
    # transform here to pick a different algorithm, only the same gather in the same order. That is
    # also what pins the precomputed stencils -- rewriting the target position as `ratio*i - ratio`
    # instead of `(i-1)*ratio` is not the same Float64, and cost exactly this equality when tried.
    reference = MLFMA.TangentialSphericalFarField(T, PARENT, Val(:zero))
    lagrange(reference, input, CHILD, PARENT)
    lagrange(up, input, CHILD, PARENT; interim=scratch)
    @test up[1] == reference[1]
    @test up[2] == reference[2]

    anterpolator(down, up, PARENT, CHILD; interim=scratch)
    lagrange(up, input, CHILD, PARENT; interim=scratch)
    anterpolator(down, up, PARENT, CHILD; interim=scratch)

    # Measured before this path existed: 420 KiB and 1954 allocations per apply at 28x56 -> 54x108,
    # the COUNT tracking Nφ + Nθ because it was one weight vector per destination index.
    @test (@allocated lagrange(up, input, CHILD, PARENT; interim=scratch)) == 0
    @test (@allocated anterpolator(down, up, PARENT, CHILD; interim=scratch)) == 0
end

end # module
