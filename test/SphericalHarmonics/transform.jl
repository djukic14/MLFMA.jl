# The transform pair on a real MLFMA sampling. Every assertion here has analytic ground truth --
# orthonormality, Parseval, exact round-trip of a band-limited field -- so nothing is checked against
# another implementation of the same thing.
#
# WHICH GRID, AND WHY IT IS NOT THE MLFMA ONE. Exactness claims here run on `gausslegendregrid`:
# Gauss-Legendre in `cos θ`, where a product of two harmonics is a polynomial and the rule is exact.
# `MLFMA.sphericalsamplepoints` is NOT that rule -- it puts Gauss-Legendre nodes in `θ` and carries
# `sin θ` as an explicit weight factor, so its integrand is not polynomial and it is never exact for
# harmonics at any resolution. That is measured in the last testset rather than asserted away, because
# synthesis onto the MLFMA leaf grid is a genuine use of it and needs no exactness at all -- see the
# algebraic identity in the header of `src/sphericalharmonics/transform.jl`.

using MLFMA
using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics:
    SphericalHarmonicTransform,
    analyze,
    degreeenergy,
    lmindex,
    numcoefficients,
    synthesize,
    synthesize!,
    truncationorder,
    ylm
using LinearAlgebra
using Random

# The exact grid, built by the module's own setup-analysis constructor.
exact(P) = SphericalHarmonicTransform(ComplexF64, P)

@testset "Spherical harmonic transform" begin
    @testset "orthonormality on the grid" begin
        P = 8
        t = exact(P)
        # analyze(synthesize(I)) is the Gram matrix of the harmonics under the grid quadrature. If the
        # grid resolves them, it is the identity -- this single check validates S, w and both
        # directions at once.
        G = analyze(
            t,
            Matrix{ComplexF64}(I, numcoefficients(P), numcoefficients(P)) |>
            a -> synthesize(t, a),
        )
        @test G ≈ Matrix{ComplexF64}(I, size(G)...) atol = 1e-10
    end

    @testset "analysis is exactly the adjoint of synthesis" begin
        # ⟨S a, b⟩_w == ⟨a, Sᴴ W b⟩ to round-off, for random a and b. This is the property the
        # SE-MLFMA leaf pairing rests on, and it must hold WITHOUT the quadrature being exact for the
        # product -- so the grid here is deliberately too coarse for degree P.
        Random.seed!(11)
        P = 10
        t = SphericalHarmonicTransform(
            ComplexF64, MLFMA.SphericalSampling{Float64}(12, 24), P
        )
        for _ in 1:5
            a = randn(ComplexF64, numcoefficients(P))
            b = randn(ComplexF64, MLFMA.SphericalHarmonics.numdirections(t))
            lhs = sum(t.weights .* conj.(synthesize(t, a)) .* b)
            rhs = dot(a, analyze(t, b))
            @test lhs ≈ rhs rtol = 1e-12
        end
    end

    @testset "round trip of a band-limited field is exact" begin
        Random.seed!(3)
        for P in (2, 5, 9)
            t = exact(P)
            a = randn(ComplexF64, numcoefficients(P))
            F = synthesize(t, a)
            @test analyze(t, F) ≈ a atol = 1e-10
        end
    end

    @testset "Parseval" begin
        Random.seed!(5)
        P = 7
        t = exact(P)
        a = randn(ComplexF64, numcoefficients(P))
        F = synthesize(t, a)
        # ∮|F|² dΩ == Σ|a|², because the harmonics are orthonormal.
        @test sum(t.weights .* abs2.(F)) ≈ sum(abs2, a) rtol = 1e-10
        @test sum(degreeenergy(a)) ≈ sum(abs2, a) rtol = 1e-12
    end

    @testset "multi-component (matrix) form matches column by column" begin
        Random.seed!(7)
        P = 6
        t = exact(P)
        A = randn(ComplexF64, numcoefficients(P), 3)
        F = synthesize(t, A)
        @test size(F) == (MLFMA.SphericalHarmonics.numdirections(t), 3)
        for c in 1:3
            @test F[:, c] ≈ synthesize(t, A[:, c])
        end
        @test analyze(t, F) ≈ A atol = 1e-10
    end

    @testset "synthesize! matches synthesize" begin
        P = 4
        t = exact(P)
        a = randn(MersenneTwister(1), ComplexF64, numcoefficients(P))
        out = Vector{ComplexF64}(undef, MLFMA.SphericalHarmonics.numdirections(t))
        @test synthesize!(out, t, a) ≈ synthesize(t, a)
    end

    @testset "a single harmonic analyses to a single coefficient" begin
        P = 6
        t = exact(P)
        θs, φs, _ = gausslegendregrid(P)
        for (l, m) in ((0, 0), (3, -2), (5, 5), (6, 1))
            F = [ylm(l, m, θs[q], φs[q]) for q in eachindex(θs)]
            a = analyze(t, F)
            @test a[lmindex(l, m)] ≈ 1 atol = 1e-10
            others = [q for q in eachindex(a) if q != lmindex(l, m)]
            @test maximum(abs, a[others]) < 1e-10
        end
    end

    @testset "degreeenergy bins by degree" begin
        P = 5
        a = zeros(ComplexF64, numcoefficients(P))
        a[lmindex(3, -1)] = 2         # |.|² = 4, degree 3
        a[lmindex(3, 2)] = 1          # |.|² = 1, degree 3
        a[lmindex(1, 0)] = 3          # |.|² = 9, degree 1
        e = degreeenergy(a)
        @test e[3 + 1] ≈ 5
        @test e[1 + 1] ≈ 9
        @test sum(e) ≈ 14
        @test_throws ArgumentError degreeenergy(zeros(ComplexF64, 7))
    end

    @testset "truncationorder" begin
        P = 6
        a = zeros(ComplexF64, numcoefficients(P))
        a[lmindex(0, 0)] = 1
        a[lmindex(4, 1)] = 1e-3       # holds 1e-6 of the energy, i.e. 1e-3 in field terms
        # ε² must exceed the tail fraction: at ε=1e-2 the tail (1e-6) is dropped, at 1e-4 it is kept.
        @test truncationorder(a, 1e-2) == 0
        @test truncationorder(a, 1e-4) == 4
        @test truncationorder(zeros(ComplexF64, numcoefficients(3)), 1e-6) == 0
    end

    @testset "constructor rejects mismatched shapes" begin
        P = 3
        S = zeros(ComplexF64, 10, numcoefficients(P))
        @test_throws DimensionMismatch SphericalHarmonicTransform(P, S, zeros(9))
        @test_throws DimensionMismatch SphericalHarmonicTransform(P + 1, S, zeros(10))
        @test_throws DimensionMismatch analyze(exact(P), zeros(ComplexF64, 3))
        @test_throws ArgumentError gausslegendregrid(-1)
        @test_throws ArgumentError gausslegendregrid(3; oversampling=0)
    end

    @testset "the MLFMA sampling: adjoint-consistent always, accurate only if oversampled 4x" begin
        # ADJOINTNESS holds on ANY grid, because analysis is DEFINED as S's adjoint -- that is the
        # property the SE-MLFMA leaf pairing rests on, and it needs no quadrature accuracy at all.
        # ACCURACY is a separate matter: Gauss-Legendre in θ against a sin θ factor is not integrating
        # a polynomial, so there is no exactness theorem -- only spectral convergence, and it needs
        # about 4x the θ nodes to arrive. Both halves are pinned because both are load bearing and
        # they point opposite ways.
        Random.seed!(19)
        P = 6
        a = randn(ComplexF64, numcoefficients(P))
        errs = Float64[]
        for Nθ in (P + 1, 2P + 2, 4P + 4)
            t = SphericalHarmonicTransform(
                ComplexF64, MLFMA.SphericalSampling{Float64}(Nθ, 2Nθ), P
            )
            b = randn(ComplexF64, MLFMA.SphericalHarmonics.numdirections(t))
            @test sum(t.weights .* conj.(synthesize(t, a)) .* b) ≈ dot(a, analyze(t, b)) rtol =
                1e-12
            push!(errs, maximum(abs, analyze(t, synthesize(t, a)) .- a))
        end
        # Measured at degree 6: 1.2e0 -> 1.6e-3 -> 1.3e-15. Thresholds are loose enough to be about
        # the SHAPE (unusable at the natural rate, converged only after 4x) rather than the digits.
        @test errs[1] > 1e-2
        @test errs[2] > 1e-6
        @test errs[3] < 1e-12
        # The exact grid needs none of that oversampling: round-off at the minimum rate.
        @test maximum(abs, analyze(exact(P), synthesize(exact(P), a)) .- a) < 1e-12
    end
end

@testset "The calibrated per-level order rule" begin
    # `leveltruncationorder` is what decides P in production, so what is tested is the RULE's
    # documented behaviour -- max over boxes, the ε/100 calibration, and the floor -- not that it
    # happens to return some number on some input.
    using MLFMA.SphericalHarmonics:
        SHETAILFLOOR, leveltruncationorder, truncatecoefficients, truncationorder

    # A box whose spectrum is exactly band-limited to `P`: everything above it is zero, so any
    # sensible criterion must return exactly `P`.
    function bandlimited(P, maxP)
        a = zeros(ComplexF64, numcoefficients(maxP), 2)
        a[1:numcoefficients(P), :] .= randn(
            MersenneTwister(P), ComplexF64, numcoefficients(P), 2
        )
        return a
    end

    @testset "it takes the MAX over the level's boxes, never the mean" begin
        # The batch needs ONE synthesis matrix, so the loudest box sets the order for everyone.
        boxes = [bandlimited(P, 12) for P in (3, 9, 4, 5)]
        @test leveltruncationorder(boxes, 1e-4) == 9
        @test leveltruncationorder(reverse(boxes), 1e-4) == 9   # order-independent
        @test leveltruncationorder([boxes[1]], 1e-4) == 3
    end

    @testset "τ is ε/100, and that is tighter than τ = ε" begin
        # The calibration only matters where the tail is graded rather than exactly zero, so use a
        # geometrically decaying spectrum -- the shape a real pattern has.
        maxP = 20
        a = zeros(ComplexF64, numcoefficients(maxP), 1)
        for q in axes(a, 1)
            a[q, 1] = 10.0^(-degreeof(q) / 2)
        end
        strict = leveltruncationorder([a], 1e-4)
        loose = truncationorder(a, 1e-4)                     # the τ = ε rule, for comparison
        @test strict >= loose                                 # never LESS conservative
        @test strict == truncationorder(a, 1e-6)              # and it IS the ε/100 criterion
    end

    @testset "the floor engages before the tail criterion becomes unsatisfiable" begin
        # At ε = 1e-6, ε/100 asks for a relative tail below 1e-16 -- round-off. The fixture must have
        # a ROUND-OFF PLATEAU for that to bite: a spectrum whose tail is exactly zero terminates for
        # any positive τ, so it cannot show the effect. Real coefficients computed in double
        # precision always carry such a plateau, and chasing it is what produced the ceiling values
        # in shecalibration.jl's last column.
        maxP = 14
        a = zeros(ComplexF64, numcoefficients(maxP), 1)
        for q in axes(a, 1)
            l = degreeof(q)
            a[q, 1] = l <= 6 ? 10.0^(-l) : 3e-9      # signal down to degree 6, then noise
        end

        @test leveltruncationorder([a], 1e-6) == 6                     # floored: stops at the signal
        # Unfloored, τ² sits BELOW the plateau's energy, no order ever satisfies the criterion, and
        # it runs to the full degree present -- the ceiling, exactly as in shecalibration.jl.
        @test leveltruncationorder([a], 1e-6; tailfloor=1e-30) == maxP
        # The separation is real but not wide: the plateau has to sit between τ_floored² = 1e-14 and
        # τ_unfloored² = 1e-16. A NOISIER analysis genuinely does raise the order the rule returns
        # (a 1e-8 plateau gives 11 rather than 6 even floored), which is the rule behaving correctly
        # -- it cannot distinguish signal from noise it was handed.
        @test max(1e-6 / 100, SHETAILFLOOR) == SHETAILFLOOR            # the floor is what binds
        @test max(1e-4 / 100, SHETAILFLOOR) == 1e-6                    # at 1e-4 it does not
    end

    @testset "truncatecoefficients is a row slice, and rejects growing" begin
        a = randn(MersenneTwister(5), ComplexF64, numcoefficients(8), 3)
        @test truncatecoefficients(a, 5) == a[1:numcoefficients(5), :]
        @test size(truncatecoefficients(a, 8)) == size(a)
        v = randn(MersenneTwister(6), ComplexF64, numcoefficients(8))
        @test truncatecoefficients(v, 4) == v[1:numcoefficients(4)]
        @test_throws ArgumentError truncatecoefficients(a, 9)
    end

    @testset "invalid tolerances are rejected" begin
        a = bandlimited(3, 6)
        @test_throws ArgumentError leveltruncationorder([a], 0.0)
        @test_throws ArgumentError leveltruncationorder([a], -1e-4)
        @test_throws ArgumentError leveltruncationorder(Matrix{ComplexF64}[], 1e-4)
    end
end
