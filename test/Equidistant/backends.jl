module EquidistantBackendTests

using Test
using MLFMA
using LinearAlgebra
using Random
using MLFMA.Equidistant:
    EquidistantSampling,
    SPHERICAL,
    ANTISPHERICAL,
    θs,
    φs,
    resample,
    FFTBackend,
    LagrangeBackend

include("ylmoracle.jl")

@testset "resampling to the same grid is the identity" begin
    s = EquidistantSampling{Float64}(8, 16)
    F = randn(ComplexF64, 16, 8)   # [φ, θ]
    # Every target IS a source node, so a correct Lagrange stencil collapses to a unit weight vector.
    # This is the sharpest available check on the index arithmetic, and it is EXACT.
    for p in (4, 6, 8)
        @test resample(F, s, s, SPHERICAL, LagrangeBackend(p)) == F
    end
    @test relL2(resample(F, s, s, SPHERICAL, FFTBackend()), F) < 1e-13
end

@testset "the θ Nyquist mode is a sin, not a cos" begin
    # THE REGRESSION TEST FOR THE ONE MODE THE DOUBLED-GRID REFERENCE GOT WRONG, against an analytic
    # oracle that shares no machinery with the implementation.
    #
    # Take the φ-mode m = 1. Under SPHERICAL it has ε = -1, so its θ sequence is ANTISYMMETRIC under
    # the half-sample reflection. Fill it with pure θ-Nyquist content: the mode sin(Nθ·θ) samples to
    # (-1)^(q-1) on its own grid. The correct interpolant on ANY finer grid is that same sin evaluated
    # at the finer midpoints.
    #
    # The old ±N/2 split reconstructed cos(Nθ_src·θ) instead. That is not a small error and not a
    # tolerance question: measured 1.414 == √2 relative, i.e. the two answers are ORTHOGONAL. It stayed
    # invisible because band-limited input has nothing in this bin, so only random data ever hit it.
    src = EquidistantSampling{Float64}(18, 36)
    dst = EquidistantSampling{Float64}(26, 52)
    Ns, Nd = MLFMA.Nθ(src), MLFMA.Nθ(dst)

    F = [
        cis(2π * (j - 1) / MLFMA.Nφ(src)) * (-1.0)^(q - 1) for
        j in 1:(MLFMA.Nφ(src)), q in 1:Ns
    ]
    want = [
        cis(2π * (j - 1) / MLFMA.Nφ(dst)) * sin(π * Ns * (q - 0.5) / Nd) for
        j in 1:(MLFMA.Nφ(dst)), q in 1:Nd
    ]
    wrong = [
        cis(2π * (j - 1) / MLFMA.Nφ(dst)) * cos(π * Ns * (q - 0.5) / Nd) for
        j in 1:(MLFMA.Nφ(dst)), q in 1:Nd
    ]

    got = resample(F, src, dst, SPHERICAL, FFTBackend())
    @test relL2(got, want) < 1e-13
    # Non-vacuity: pin that the wrong answer really is wrong, so a future change cannot satisfy the
    # assertion above by making both branches agree.
    @test relL2(got, wrong) > 1
end

@testset "the spectral resampling operator is real" begin
    # Resampling maps samples of a function to samples of the SAME function, so its matrix is real --
    # the Lagrange backend's is exactly real, and the FFT backend must reach the same operator.
    #
    # Stated as its own test because the imaginary part is a load-bearing diagnostic: before the θ
    # Nyquist fix it was over 1% of the real part, and that was the single most visible symptom of the
    # bug. Nothing else in the suite would have flagged it.
    coarse = EquidistantSampling{Float64}(10, 20)
    fine = EquidistantSampling{Float64}(14, 28)
    for parity in (SPHERICAL, ANTISPHERICAL)
        R = MLFMA.Equidistant.resamplematrix(coarse, fine, parity, FFTBackend())
        @test maximum(abs, imag(R)) / maximum(abs, real(R)) < 1e-14
    end
end

@testset "FFT backend is exact for any field the grid resolves" begin
    # Critical sampling Nθ = L+1 is enough: the extension is band-limited to L on the torus, so
    # zero-padding evaluates the SAME series at more points, with no approximation anywhere.
    for L in (1, 3, 5, 8, 12)
        f = bandlimitedfield(L)
        nθ = L + 1
        src = EquidistantSampling{Float64}(nθ, 2nθ)
        dst = EquidistantSampling{Float64}(2nθ, 4nθ)
        got = resample(sampleon(f, src), src, dst, SPHERICAL, FFTBackend())
        @test relL2(got, sampleon(f, dst)) < 1e-12
    end
end

@testset "FFT backend aliases honestly below the band limit" begin
    # A method that stayed accurate here would be hiding an error, not avoiding one.
    L = 6
    f = bandlimitedfield(L)
    src = EquidistantSampling{Float64}(5, 10)      # Nθ < L+1 = 7
    dst = EquidistantSampling{Float64}(10, 20)
    @test relL2(
        resample(sampleon(f, src), src, dst, SPHERICAL, FFTBackend()), sampleon(f, dst)
    ) > 1e-2
    # ...and recovers the moment the grid is adequate.
    src2 = EquidistantSampling{Float64}(7, 14)
    dst2 = EquidistantSampling{Float64}(14, 28)
    @test relL2(
        resample(sampleon(f, src2), src2, dst2, SPHERICAL, FFTBackend()), sampleon(f, dst2)
    ) < 1e-12
end

@testset "Lagrange converges algebraically in stencil order" begin
    L = 6
    f = bandlimitedfield(L)
    nθ = 3 * (L + 1)                                # oversampled: local stencils require it
    src = EquidistantSampling{Float64}(nθ, 2nθ)
    dst = EquidistantSampling{Float64}(2nθ, 4nθ)
    Fc, ex = sampleon(f, src), sampleon(f, dst)
    errs = [
        relL2(resample(Fc, src, dst, SPHERICAL, LagrangeBackend(p)), ex) for
        p in (4, 6, 8, 10)
    ]
    @test issorted(errs; rev=true)                  # strictly improving with order
    @test errs[1] < 5e-2
    @test errs[end] < 1e-4
end

@testset "Lagrange converges with oversampling" begin
    L = 6
    f = bandlimitedfield(L)
    errs = map((1, 2, 4)) do γ
        nθ = γ * (L + 1)
        src = EquidistantSampling{Float64}(nθ, 2nθ)
        dst = EquidistantSampling{Float64}(2nθ, 4nθ)
        relL2(
            resample(sampleon(f, src), src, dst, SPHERICAL, LagrangeBackend(6)),
            sampleon(f, dst),
        )
    end
    @test issorted(errs; rev=true)
    @test errs[end] < 1e-3
end

@testset "parity is load-bearing" begin
    # sin θ obeys F(2π-θ) = -F(θ): it is ANTI-spherical, so the spherical rule must visibly fail on it.
    src = EquidistantSampling{Float64}(8, 16)
    dst = EquidistantSampling{Float64}(16, 16)
    a(θ, φ) = sin(θ) + 0.4sin(3θ)
    Fa, exa = sampleon(a, src), sampleon(a, dst)
    @test relL2(resample(Fa, src, dst, ANTISPHERICAL, FFTBackend()), exa) < 1e-13
    @test relL2(resample(Fa, src, dst, SPHERICAL, FFTBackend()), exa) > 1e-3
end

@testset "the two backends agree where both are accurate" begin
    # Not a substitute for the analytic checks above -- agreement between two wrong methods proves
    # nothing -- but with both independently pinned to the oracle, this pins the shared API.
    L = 4
    f = bandlimitedfield(L)
    nθ = 4 * (L + 1)
    src = EquidistantSampling{Float64}(nθ, 2nθ)
    dst = EquidistantSampling{Float64}(2nθ, 4nθ)
    Fc = sampleon(f, src)
    a = resample(Fc, src, dst, SPHERICAL, FFTBackend())
    b = resample(Fc, src, dst, SPHERICAL, LagrangeBackend(10))
    @test relL2(b, a) < 1e-5
end

end # module
