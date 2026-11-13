# The 2-D translation operator, against the identity it exists to satisfy.
#
# THE ORACLE IS EXACT AND INDEPENDENT. A translation operator's whole job is the diagonalized addition
# theorem: aggregating a point source at offset `d`, translating across `R`, and integrating over the
# directions must reproduce the Green's function at the true distance `|R + d|`. That is checkable to
# machine precision against a closed form, with no reference implementation and no tolerance to
# negotiate -- so this tests the physics, not a previous run of the same code.
#
# THE CONSTANT IS BEAST'S, NOT THIS FILE'S. `G₂ᴰ = -(i/4) H₀⁽²⁾(κr)` is read off
# `HH2DSingleLayerFDBIO`'s own kernel (`green = -im/4 * hankelh2(0, k*R)`). Restating it here from a
# textbook would let a convention mismatch with BEAST pass this test and fail end to end, which is
# exactly the failure this file is meant to make impossible.
#
# THE SAME CONTRACT HOLDS IN 3-D and is asserted alongside, because the 2-D operator was built to match
# it: `Σ_q w_q T exp(-γ k̂·d)` is the Green's function ITSELF in both dimensions. Checking both here
# keeps that a stated property of the translation layer rather than a coincidence of one branch.

using LinearAlgebra
using MLFMA
using SpecialFunctions
using StaticArrays
using Test

# The package's own radiation-convention phase, so the test cannot silently disagree with the
# aggregation about a sign.
_aggregationphase(γ, k̂, d) =
    MLFMA.phasefactor(γ, k̂, d, MLFMA.AbstractRadiationFieldCollection)

_green2d(κ, r) = -im / 4 * SpecialFunctions.hankelh2(0, κ * r)
_green3d(κ, r) = cis(-κ * r) / (4π * r)

@testset "2D translation operator" begin
    @testset "the diagonalized addition theorem, to machine precision" begin
        κ = 2.0
        γ = im * κ

        # Several geometries, including one where R is nearly parallel to d and one where it is
        # nearly perpendicular -- the φ dependence is the whole content of the operator, so a bug in
        # the `cos(mΔ)` folding would show up as an orientation-dependent error.
        cases = (
            (SVector{2,Float64}(5.0, 3.0), SVector{2,Float64}(0.3, -0.25)),
            (SVector{2,Float64}(-6.0, 1.0), SVector{2,Float64}(-0.4, -0.05)),
            (SVector{2,Float64}(0.0, 7.5), SVector{2,Float64}(0.0, 0.35)),
            (SVector{2,Float64}(4.0, -4.0), SVector{2,Float64}(0.28, 0.28)),
        )

        for (R, d) in cases
            target = _green2d(κ, norm(R + d))
            # L = 14 sits in the measured accuracy floor for these geometries (κ|R| ≈ 8-15). See the
            # divergence testset below for why "larger L" is NOT automatically better here.
            L = 14
            s = MLFMA.CircularSampling{Float64}(MLFMA.NφCircular(L))
            k̂, w = MLFMA.cartesiansamplepoints(s)

            T = Vector{ComplexF64}(undef, length(s))
            MLFMA.translationoperator!(T, γ, k̂, R, L)
            approx = sum(T .* _aggregationphase(γ, k̂, d) .* w)

            @test approx ≈ target rtol = 1e-11
        end
    end

    @testset "it converges, and the rate is spectral" begin
        κ = 2.0
        γ = im * κ
        R = SVector{2,Float64}(5.0, 3.0)
        d = SVector{2,Float64}(0.3, -0.25)
        target = _green2d(κ, norm(R + d))

        errors = map((4, 6, 8, 10)) do L
            s = MLFMA.CircularSampling{Float64}(MLFMA.NφCircular(L))
            k̂, w = MLFMA.cartesiansamplepoints(s)
            T = Vector{ComplexF64}(undef, length(s))
            MLFMA.translationoperator!(T, γ, k̂, R, L)
            abs(sum(T .* _aggregationphase(γ, k̂, d) .* w) / target - 1)
        end

        # Strictly decreasing, and by orders of magnitude per step rather than polynomially -- each
        # added pair of modes should buy a factor of hundreds in this regime.
        @test issorted(errors; rev=true)
        @test errors[1] < 1e-3
        @test errors[end] < 1e-11
        @test errors[1] / errors[end] > 1e6
    end

    @testset "the cancellation guard stops the asymptotic series diverging" begin
        # `Hₘ⁽²⁾(x)` grows like `(m-1)!(2/x)^m`, so this series DIVERGES past its optimal order --
        # unlike the 3-D one, where the analogous guard is only an optimization. The guard spends a
        # bounded number of digits on cancellation and stops; asking for an absurd cutoff must
        # therefore still give a usable operator rather than noise.
        κ = 2.0
        γ = im * κ
        R = SVector{2,Float64}(5.0, 3.0)
        d = SVector{2,Float64}(0.3, -0.25)
        target = _green2d(κ, norm(R + d))

        for L in (60, 400)
            s = MLFMA.CircularSampling{Float64}(MLFMA.NφCircular(L))
            k̂, w = MLFMA.cartesiansamplepoints(s)
            T = Vector{ComplexF64}(undef, length(s))
            used = MLFMA.translationoperator!(T, γ, k̂, R, L)

            @test used < L                       # it clamped
            @test all(isfinite, T)
            # The budget is 1e6 against |H₀|, i.e. ~6 of the ~16 digits, so ~1e-9 is what survives.
            # Asserting the level the budget IMPLIES, rather than a number tuned until green.
            @test sum(T .* _aggregationphase(γ, k̂, d) .* w) ≈ target rtol = 1e-9
        end
    end

    @testset "the guard does not bind at the rate a real box asks for" begin
        # THE PROPERTY THAT ACTUALLY MATTERS, and the reason the guard is written as a cancellation
        # budget rather than as a cap near `κ|R|`. At a λ/4 leaf box the excess-bandwidth rule asks
        # for L = 11 while `κ|R|` to the nearest interaction-list neighbour is only 6.28 -- so a
        # `κ|R|` cap would silently gut the accuracy of every leaf-level translation. Measured: the
        # margin term dominates at small boxes, which is exactly the regime MLFMA runs in.
        halfsize = 0.25
        κ = 2π
        γ = im * κ

        diameter = 2 * halfsize * sqrt(2)              # a SQUARE's diagonal, not a cube's
        L = floor(Int, κ * diameter + 1.8 * 4^(2 / 3) * cbrt(κ * diameter))
        @test L > κ * 4 * halfsize                     # the cap-at-κ|R| rule really would have bound

        R = SVector{2,Float64}(4 * halfsize, 0.0)
        d = SVector{2,Float64}(0.3 * halfsize, 0.2 * halfsize)

        s = MLFMA.CircularSampling{Float64}(MLFMA.NφCircular(L))
        k̂, w = MLFMA.cartesiansamplepoints(s)
        T = Vector{ComplexF64}(undef, length(s))
        used = MLFMA.translationoperator!(T, γ, k̂, R, L)

        @test used == L                                # not clamped
        @test sum(T .* _aggregationphase(γ, k̂, d) .* w) ≈ _green2d(κ, norm(R + d)) rtol =
            1e-10
    end

    @testset "the 3D operator satisfies the same contract" begin
        # Stated here because the 2-D operator was NORMALIZED to match this. If the 3-D convention
        # ever moves, the two must stop agreeing loudly rather than drift apart.
        κ = 2.0
        γ = im * κ
        R = SVector{3,Float64}(6.0, 2.0, -3.0)
        d = SVector{3,Float64}(0.35, -0.2, 0.15)

        L = 16
        s = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(L)...)
        k̂, w = MLFMA.cartesiansamplepoints(s)
        T = Array{ComplexF64}(undef, size(k̂))
        MLFMA.translationoperator!(T, γ, k̂, R, L)

        approx = sum(T .* _aggregationphase(γ, k̂, d) .* w)
        @test approx ≈ _green3d(κ, norm(R + d)) rtol = 1e-10
    end

    @testset "the two dimensions do not dispatch into each other" begin
        # The 3-D body is spherical Hankels times Legendre polynomials and the 2-D body is cylindrical
        # Hankels; before this work the 3-D signature was generic in `D`, so a 2-D argument reached it
        # and produced confident nonsense. Each must now be reachable only by its own dimension.
        γ = 2.0im
        circular = MLFMA.CircularSampling{Float64}(16)
        k̂2, _ = MLFMA.cartesiansamplepoints(circular)
        spherical = MLFMA.SphericalSampling{Float64}(8, 16)
        k̂3, _ = MLFMA.cartesiansamplepoints(spherical)

        R2 = SVector{2,Float64}(3.0, 1.0)
        R3 = SVector{3,Float64}(3.0, 1.0, 2.0)

        @test_throws MethodError MLFMA.translationoperator!(
            Vector{ComplexF64}(undef, length(circular)), γ, k̂2, R3, 6
        )
        @test_throws MethodError MLFMA.translationoperator!(
            Array{ComplexF64}(undef, size(k̂3)), γ, k̂3, R2, 6
        )
    end
end
