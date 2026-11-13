# An INDEPENDENT oracle: closed-form low-order harmonics, written out by hand from a table, plus the
# NAIVE factorial construction at low degree. Neither shares any machinery with the normalized
# recurrence under test -- which is the point. The naive form is only valid at low degree (it overflows
# for large l, which is why the production code does not use it), so it is exercised exactly where it
# is still trustworthy.

using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics: lmindex, numcoefficients, degreeof, ylm, ylm!

# Y_lm from the textbook definition, forming P_l^m and its normalization SEPARATELY.
function naiveylm(l::Int, m::Int, θ::Float64, φ::Float64)
    am = abs(m)
    x = cos(θ)
    pmm = 1.0
    if am > 0
        s = sqrt(max(0.0, 1 - x^2))
        for i in 1:am
            pmm *= -(2i - 1) * s
        end
    end
    p = if l == am
        pmm
    else
        pmm1 = x * (2am + 1) * pmm
        if l == am + 1
            pmm1
        else
            pll = 0.0
            for ll in (am + 2):l
                pll = ((2ll - 1) * x * pmm1 - (ll + am - 1) * pmm) / (ll - am)
                pmm, pmm1 = pmm1, pll
            end
            pll
        end
    end
    norm = sqrt((2l + 1) / (4π) * factorial(l - am) / factorial(l + am))
    v = norm * p * cis(am * φ)
    return m < 0 ? (-1)^am * conj(v) : v
end

@testset "Spherical harmonics" begin
    @testset "index map" begin
        # Contiguity is the property the layout relies on: every mode of degree <= P must land in
        # 1:(P+1)^2 exactly once, with no gaps.
        for P in 0:6
            seen = [lmindex(l, m) for l in 0:P for m in (-l):l]
            @test sort(seen) == collect(1:numcoefficients(P))
        end
        # degreeof inverts lmindex in its first argument
        for l in 0:8, m in (-l):l
            @test degreeof(lmindex(l, m)) == l
        end
    end

    @testset "closed forms" begin
        # Straight from a table, independent of any recurrence.
        for (θ, φ) in ((0.7, 1.1), (2.3, 5.0), (π / 2, 0.0))
            x, s = cos(θ), sin(θ)
            @test ylm(0, 0, θ, φ) ≈ sqrt(1 / (4π))
            @test ylm(1, 0, θ, φ) ≈ sqrt(3 / (4π)) * x
            @test ylm(1, 1, θ, φ) ≈ -sqrt(3 / (8π)) * s * cis(φ)
            @test ylm(1, -1, θ, φ) ≈ sqrt(3 / (8π)) * s * cis(-φ)
            @test ylm(2, 0, θ, φ) ≈ sqrt(5 / (16π)) * (3x^2 - 1)
            @test ylm(2, 2, θ, φ) ≈ sqrt(15 / (32π)) * s^2 * cis(2φ)
        end
    end

    @testset "agrees with the naive construction where the naive one is valid" begin
        for (θ, φ) in ((0.3, 0.2), (1.9, 4.4), (3.0, 1.0))
            for l in 0:8, m in (-l):l
                @test ylm(l, m, θ, φ) ≈ naiveylm(l, m, θ, φ) atol = 1e-12
            end
        end
    end

    @testset "Condon-Shortley: Y_{l,-m} = (-1)^m conj(Y_{l,m})" begin
        buf = Vector{ComplexF64}(undef, numcoefficients(12))
        ylm!(buf, 12, 1.234, 2.345)
        for l in 0:12, m in 1:l
            @test buf[lmindex(l, -m)] ≈ (-1)^m * conj(buf[lmindex(l, m)])
        end
    end

    @testset "no overflow at high degree" begin
        # The whole reason the production code runs the recurrence on normalized functions. The naive
        # construction returns NaN here (factorial(171) is Inf); this must stay finite and bounded --
        # |Y_lm| <= sqrt((2l+1)/4π) pointwise.
        P = 120
        buf = Vector{ComplexF64}(undef, numcoefficients(P))
        ylm!(buf, P, 0.9, 0.4)
        @test all(isfinite, buf)
        for l in 0:P, m in (-l):l
            @test abs(buf[lmindex(l, m)]) <= sqrt((2l + 1) / (4π)) + 1e-9
        end
    end

    @testset "poles are finite and only m = 0 survives" begin
        for θ in (0.0, Float64(π))
            buf = Vector{ComplexF64}(undef, numcoefficients(10))
            ylm!(buf, 10, θ, 0.77)
            @test all(isfinite, buf)
            for l in 1:10, m in 1:l
                @test abs(buf[lmindex(l, m)]) < 1e-14
                @test abs(buf[lmindex(l, -m)]) < 1e-14
            end
            # Y_l0(0) = sqrt((2l+1)/4π), and at θ=π it alternates by (-1)^l.
            for l in 0:10
                expected = sqrt((2l + 1) / (4π)) * (θ == 0 ? 1 : (-1)^l)
                @test real(buf[lmindex(l, 0)]) ≈ expected
            end
        end
    end

    @testset "ylm! bounds are checked" begin
        @test_throws ArgumentError ylm!(Vector{ComplexF64}(undef, 8), 3, 0.5, 0.5)
        @test_throws ArgumentError ylm(2, 3, 0.5, 0.5)
    end
end
