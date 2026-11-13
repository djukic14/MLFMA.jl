using MLFMA
using LinearAlgebra
using Test

const IDV = MLFMA.InterpolationDiagnostics

# GROUND TRUTH FOR THE TANGENTIAL REPRESENTATION, pinned four independent ways before anything is built
# on it. These functions are the reference the tangential interpolation diagnostics will be measured
# against, so an error here would make that diagnostic authoritative and wrong at the same time -- the
# same reason the scalar harmonics in `interpolationdiagnostics.jl` are pinned three ways rather than
# spot-checked.
#
# The four are chosen to fail independently: finite differences catch a wrong recurrence, the l = 1 closed
# forms catch a wrong normalization, the reflection identity catches a wrong component assignment (which
# is what would silently break the interpolator's pole continuation), and orthonormality catches a wrong
# scale factor that the others might share.
@testset "Vector spherical harmonic ground truth" begin
    @testset "dnormalizedlegendre against central differences" begin
        # h = 1e-6 gives ~1e-10 truncation on these functions, so agreement at 1e-7 is the formula being
        # right rather than the comparison being loose.
        function centraldifference(l, m, θ; h=1e-6)
            return (
                IDV.normalizedlegendre(l, m, θ + h) - IDV.normalizedlegendre(l, m, θ - h)
            ) / (2h)
        end

        worst = 0.0
        for l in 0:14, m in 0:l, θ in (0.05, 0.3, 1.1, 2.0, 2.9, 3.1)
            analytic = IDV.dnormalizedlegendre(l, m, θ)
            numeric = centraldifference(l, m, θ)
            worst = max(worst, abs(analytic - numeric) / max(abs(numeric), 1e-6))
        end
        @test worst < 1e-7

        # THE m = 0 TRAP, pinned explicitly rather than left to the sweep above: reading the general
        # formula with a zeroed `m - 1` neighbour halves the derivative at every m = 0, and m = 0 is the
        # axisymmetric case that dominates the θ error budget. `q_l^{-1} = -q_l^{1}`, so the terms add.
        for (l, θ) in ((1, 1.1), (4, 0.7), (9, 2.3))
            @test IDV.dnormalizedlegendre(l, 0, θ) ≈
                sqrt(l * (l + 1)) * IDV.normalizedlegendre(l, 1, θ)
            # And it is NOT half that, which is what the naive reading gives.
            @test !isapprox(
                IDV.dnormalizedlegendre(l, 0, θ),
                sqrt(l * (l + 1)) * IDV.normalizedlegendre(l, 1, θ) / 2,
            )
        end

        @test IDV.dnormalizedlegendre(0, 0, 1.1) == 0   # a constant has no θ dependence
    end

    @testset "l = 1 closed forms" begin
        # Y_1^0 = sqrt(3/4π) cosθ, so ∂θ Y = -sqrt(3/4π) sinθ and the m = 0 field has no φ component.
        # s = sqrt(l(l+1)) = sqrt(2).
        θ, φ = 1.1, 0.7
        expected = -sqrt(3 / (4π)) * sin(θ) / sqrt(2)

        poloidal = IDV.vectorsphericalharmonic(:poloidal, 1, 0, θ, φ)
        @test poloidal[1] ≈ expected
        @test poloidal[2] ≈ 0 atol = 1e-14

        # The toroidal field is the poloidal one rotated by r̂ ×: (Fθ, Fφ) -> (-Fφ, Fθ).
        toroidal = IDV.vectorsphericalharmonic(:toroidal, 1, 0, θ, φ)
        @test toroidal[1] ≈ 0 atol = 1e-14
        @test toroidal[2] ≈ expected

        # Non-vacuity: the field is not identically zero, so the two `atol` checks above mean something.
        @test abs(expected) > 0.1
    end

    @testset "the reflection identity the interpolator's continuation assumes" begin
        # `SphericalθφInterpolation` continues both components as f(θ,φ) = -f(-θ, φ+π). That map sends a
        # point to ITSELF in R^3, and both basis vectors flip under it, so a genuine tangential field
        # satisfies this automatically. Checking it here is what verifies the COMPONENT ASSIGNMENT: swap
        # Fθ and Fφ, or drop a sign in the toroidal rotation, and this fails while the closed forms and
        # the derivative sweep above still pass.
        for kind in IDV.SUPPORTEDVECTORKINDS,
            (l, m) in ((1, 0), (2, 1), (3, -2), (5, 5), (7, 3)),
            (θ, φ) in ((0.4, 0.9), (1.3, 2.2), (2.6, 5.1))

            here = IDV.vectorsphericalharmonic(kind, l, m, θ, φ)
            reflected = IDV.vectorsphericalharmonic(kind, l, m, -θ, φ + π)
            @test reflected[1] ≈ -here[1] atol = 1e-12
            @test reflected[2] ≈ -here[2] atol = 1e-12
            # Non-vacuity: at least one component is well away from zero, so this is not 0 ≈ -0.
            @test max(abs(here[1]), abs(here[2])) > 1e-3
        end
    end

    @testset "orthonormality under the sampling's own quadrature" begin
        # Integrated with Gauss-Legendre weights in θ and the trapezoidal rule in φ, exactly as the
        # scalar orthonormality check in `interpolationdiagnostics.jl` does. A wrong sqrt(l(l+1)) scale
        # shows up here and nowhere else above.
        sampling = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(8)...)
        θvalues = MLFMA.θs(sampling)
        wθ = MLFMA.wθs(sampling)
        φvalues = MLFMA.φs(sampling)
        dφ = 2π / length(φvalues)

        function innerproduct(kind1, l1, m1, kind2, l2, m2)
            total = zero(ComplexF64)
            for (i, θ) in enumerate(θvalues)
                weight = wθ[i] * (π / 2) * sin(θ) * dφ
                for φ in φvalues
                    a = IDV.vectorsphericalharmonic(kind1, l1, m1, θ, φ)
                    b = IDV.vectorsphericalharmonic(kind2, l2, m2, θ, φ)
                    total += weight * (a[1] * conj(b[1]) + a[2] * conj(b[2]))
                end
            end
            return total
        end

        for kind in IDV.SUPPORTEDVECTORKINDS, (l, m) in ((1, 0), (2, 1), (3, -1), (4, 4))
            @test innerproduct(kind, l, m, kind, l, m) ≈ 1 rtol = 1e-8
        end

        # Distinct (l, m) within a family are orthogonal ...
        @test abs(innerproduct(:poloidal, 2, 1, :poloidal, 3, 1)) < 1e-8
        @test abs(innerproduct(:poloidal, 2, 1, :poloidal, 2, 0)) < 1e-8
        # ... and the two families are orthogonal to each other, which is the property that makes them a
        # basis rather than two overlapping sets.
        @test abs(innerproduct(:poloidal, 2, 1, :toroidal, 2, 1)) < 1e-8
        @test abs(innerproduct(:poloidal, 3, 2, :toroidal, 3, 2)) < 1e-8
    end

    @testset "argument validation" begin
        @test_throws ArgumentError IDV.vectorsphericalharmonic(:radial, 1, 0, 1.1, 0.7)
        # l = 0 has no tangential harmonic: the surface gradient of a constant vanishes and the
        # normalization is zero, so this must raise rather than return NaN.
        @test_throws ArgumentError IDV.vectorsphericalharmonic(:poloidal, 0, 0, 1.1, 0.7)
        @test_throws ArgumentError IDV.vectorsphericalharmonic(:poloidal, 2, 3, 1.1, 0.7)
        @test_throws ArgumentError IDV.vectorsphericalharmonic(:toroidal, 2, -3, 1.1, 0.7)
    end
end
