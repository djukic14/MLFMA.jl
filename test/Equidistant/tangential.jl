module EquidistantTangentialTests

using Test
using MLFMA
using LinearAlgebra
using Random
using MLFMA.Equidistant:
    EquidistantSampling,
    SPHERICAL,
    ANTISPHERICAL,
    Nθ,
    Nφ,
    θs,
    φs,
    resample,
    sphericalget,
    TangentialField,
    cartesian,
    istransverse,
    thetahat,
    phihat,
    khat,
    FFTBackend,
    LagrangeBackend

# Two independent smooth tangential fields, both from Y₁₁ ~ sin θ e^{iφ}:
#   ∇ₛY   = θ̂ cos θ e^{iφ} + φ̂ i e^{iφ}
#   k̂ × ∇ₛY = θ̂ (-i e^{iφ}) + φ̂ cos θ e^{iφ}
gradY(θ, φ) = (cos(θ) * cis(φ), im * cis(φ))
curlY(θ, φ) = (-im * cis(φ), cos(θ) * cis(φ))

tangential(f, s) = TangentialField(
    [f(θ, φ)[1] for φ in φs(s), θ in θs(s)], [f(θ, φ)[2] for φ in φs(s), θ in θs(s)]
)

@testset "the local basis flips under the torus identification" begin
    # This is the geometric fact the whole parity question reduces to. k̂ is preserved (same physical
    # direction), but BOTH tangent vectors reverse.
    for (θ, φ) in ((0.7, 1.1), (2.3, 4.0), (1.5, 0.2), (0.3, 5.5))
        θ2, φ2 = 2π - θ, φ + π
        @test all(isapprox.(khat(θ2, φ2), khat(θ, φ); atol=1e-12))
        @test all(isapprox.(thetahat(θ2, φ2), -1 .* thetahat(θ, φ); atol=1e-12))
        @test all(isapprox.(phihat(θ2, φ2), -1 .* phihat(θ, φ); atol=1e-12))
    end
end

@testset "both tangential components are ANTI-spherical" begin
    # Measured, not assumed -- and on two independent fields, so it cannot be an accident of one.
    for f in (gradY, curlY)
        for (θ, φ) in ((0.7, 1.1), (2.3, 4.0), (1.5, 0.2))
            a = f(θ, φ)
            b = f(2π - θ, φ + π)
            @test b[1] ≈ -a[1]                     # Fθ flips
            @test b[2] ≈ -a[2]                     # Fφ flips TOO -- not spherical
        end
    end
end

@testset "the Cartesian field is single-valued on the torus" begin
    # The physical constraint the parity rule exists to satisfy: one direction, one vector.
    for f in (gradY, curlY)
        for (θ, φ) in ((0.7, 1.1), (2.3, 4.0), (1.5, 0.2))
            a = f(θ, φ)[1] .* thetahat(θ, φ) .+ f(θ, φ)[2] .* phihat(θ, φ)
            θ2, φ2 = 2π - θ, φ + π
            b = f(θ2, φ2)[1] .* thetahat(θ2, φ2) .+ f(θ2, φ2)[2] .* phihat(θ2, φ2)
            @test all(isapprox.(a, b; atol=1e-12))
        end
    end
end

@testset "the stored continuation reproduces the analytic components" begin
    # `sphericalget` with ANTISPHERICAL must agree with evaluating the field at torus angles, for
    # indices well outside the stored array -- separately for each component.
    s = EquidistantSampling{Float64}(12, 24)
    F = tangential(gradY, s)
    errθ = errφ = 0.0
    for i in (-5):(Nθ(s) + 5), j in 1:Nφ(s)
        θ, φ = (i - 0.5) * π / Nθ(s), φs(s)[j]
        errθ = max(errθ, abs(sphericalget(F.Fθ, i, j, s, ANTISPHERICAL) - gradY(θ, φ)[1]))
        errφ = max(errφ, abs(sphericalget(F.Fφ, i, j, s, ANTISPHERICAL) - gradY(θ, φ)[2]))
    end
    @test errθ < 1e-13
    @test errφ < 1e-13
end

@testset "using SPHERICAL for Fφ is silently wrong" begin
    # Non-vacuity: the wrong parity must produce a visibly different continuation, or the tests above
    # would not be testing anything.
    s = EquidistantSampling{Float64}(12, 24)
    F = tangential(gradY, s)
    worst = 0.0
    for i in (-5):0, j in 1:Nφ(s)
        θ, φ = (i - 0.5) * π / Nθ(s), φs(s)[j]
        worst = max(worst, abs(sphericalget(F.Fφ, i, j, s, SPHERICAL) - gradY(θ, φ)[2]))
    end
    @test worst > 1e-2
end

@testset "resampling a tangential field is exact for the FFT backend" begin
    # End-to-end: resample both components, rebuild the Cartesian vector on the FINE grid, and compare
    # against the analytic field there. This is the check that a scalar test cannot make.
    for f in (gradY, curlY)
        src = EquidistantSampling{Float64}(6, 12)
        dst = EquidistantSampling{Float64}(12, 24)
        got = cartesian(resample(tangential(f, src), src, dst, FFTBackend()), dst)
        want = cartesian(tangential(f, dst), dst)
        @test maximum(norm(collect(g) .- collect(w)) for (g, w) in zip(got, want)) < 1e-12
    end
end

@testset "resampling a tangential field converges for Lagrange" begin
    f = gradY
    errs = map((4, 8)) do γ
        src = EquidistantSampling{Float64}(2γ, 4γ)
        dst = EquidistantSampling{Float64}(4γ, 8γ)
        got = cartesian(resample(tangential(f, src), src, dst, LagrangeBackend(6)), dst)
        want = cartesian(tangential(f, dst), dst)
        maximum(norm(collect(g) .- collect(w)) for (g, w) in zip(got, want))
    end
    @test errs[2] < errs[1]
    @test errs[2] < 1e-4
end

@testset "resampling preserves transversality" begin
    src = EquidistantSampling{Float64}(6, 12)
    dst = EquidistantSampling{Float64}(12, 24)
    F = tangential(gradY, src)
    @test istransverse(F, src)
    @test istransverse(resample(F, src, dst, FFTBackend()), dst)
end

@testset "TangentialField rejects mismatched components" begin
    @test_throws ErrorException TangentialField(
        zeros(ComplexF64, 8, 4), zeros(ComplexF64, 6, 4)
    )
end

end # module
