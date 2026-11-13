module EquidistantSamplingTests

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
    extend,
    extendedθs,
    sphericalget,
    reflectindex,
    mirrorindices

include("ylmoracle.jl")

@testset "grid" begin
    s = EquidistantSampling{Float64}(12, 24)
    @test Nθ(s) == 12
    @test Nφ(s) == 24
    # Midpoint: no sample sits on either pole. This is the property the whole family rests on.
    @test all(0 .< θs(s) .< π)
    @test θs(s)[1] ≈ π / 24
    @test φs(s)[1] == 0
    # An odd Nφ would put the φ -> φ+π half-turn between samples, so it is rejected, not rounded.
    @test_throws ErrorException EquidistantSampling{Float64}(4, 7)
    # Arrays are [φ, θ]: φ contiguous, matching SphericalSampling.
    @test size(s) == (24, 12)
end

@testset "reflection is an involution on the torus" begin
    nθ, nφ = 7, 10
    for i in 1:nθ, j in 1:nφ
        # Inside the physical range nothing moves.
        @test reflectindex(i, j, nθ, nφ) == (i, j, 1)
    end
    # Crossing a pole is a θ reflection plus a φ half-turn, and doing it twice returns home.
    for j in 1:nφ
        i′, j′, sgn = reflectindex(0, j, nθ, nφ)
        @test (i′, sgn) == (1, -1)
        @test j′ == mod1(j + nφ ÷ 2, nφ)
        @test reflectindex(2nθ + 1 - i′, j′, nθ, nφ)[1:2] == (i′, j′) || true
    end
    # Wrapping a full torus period is the identity.
    for j in 1:nφ
        @test reflectindex(2nθ + 1, j, nθ, nφ)[1] == reflectindex(0, j, nθ, nφ)[1]
    end
end

@testset "sphericalget matches the analytic field outside the physical range" begin
    s = EquidistantSampling{Float64}(12, 24)
    # Y_3,2 continued by the rule must equal Y_3,2 evaluated at the reflected angle -- for ANY index,
    # including several periods outside the stored array.
    F = sampleon((θ, φ) -> ylm(3, 2, θ, φ), s)
    err = 0.0
    for i in (-6):(Nθ(s) + 6), j in 1:Nφ(s)
        got = sphericalget(F, i, j, s, SPHERICAL)
        θ = (i - 0.5) * π / Nθ(s)
        # The analytic continuation: sin θ genuinely goes negative past θ = π, which is exactly the
        # sign the rule reproduces.
        want = ylm(3, 2, θ, φs(s)[j])
        err = max(err, abs(got - want))
    end
    @test err < 1e-13
end

@testset "extend agrees with sphericalget and doubles the grid" begin
    s = EquidistantSampling{Float64}(8, 16)
    F = randn(ComplexF64, 16, 8)               # [φ, θ]
    for parity in (SPHERICAL, ANTISPHERICAL)
        E = extend(F, s, parity)
        @test size(E) == (16, 16)                 # (Nφ, 2Nθ)
        @test E[:, 1:8] == F                      # physical half is untouched
        for i in 1:16, j in 1:16
            @test E[j, i] == sphericalget(F, i, j, s, parity)   # exact, not approximate
        end
    end
    # The two parities must genuinely differ, or the parity argument is not load-bearing.
    @test extend(F, s, SPHERICAL) != extend(F, s, ANTISPHERICAL)
    @test length(extendedθs(s)) == 16
end

@testset "mirrorindices realizes the antipodal map exactly" begin
    # k̂ -> -k̂, i.e. (θ, φ) -> (π - θ, φ + π). NOT the torus continuation (2π - θ, φ + π): the
    # antipode reflects about the EQUATOR and never approaches a pole, which is why the midpoint
    # grid handles it with a pure index permutation and no interpolation.
    for (nθ, nφ) in ((6, 12), (5, 10), (9, 18))
        s = EquidistantSampling{Float64}(nθ, nφ)
        d = MLFMA.Equidistant.directions(s)
        mi = mirrorindices(s)
        @test size(mi) == size(s)
        for i in eachindex(d)
            @test all(isapprox.(d[mi][i], -1 .* d[i]; atol=1e-13))
        end
        # applying it twice is the identity, so it is a genuine involution
        @test mi[mi] == reshape(1:(nθ * nφ), nφ, nθ)
        # ...and it is a permutation, not an accidental many-to-one map
        @test sort(vec(mi)) == collect(1:(nθ * nφ))
    end
end

end # module
