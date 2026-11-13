using MLFMA
using LinearAlgebra
using Test

const IDT = MLFMA.InterpolationDiagnostics

@testset "Tangential interpolation diagnostics" begin
    T = ComplexF64
    source = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(13)...)
    target = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(26)...)
    interpolator = MLFMA.SphericalLagrangeInterpolator{8}(T, Float64)

    tangential(case) = only(
        IDT.results(
            IDT.interpolationerror(
                interpolator,
                source,
                target,
                case;
                farfieldtype=MLFMA.TangentialSphericalFarField,
            ),
        ),
    )

    @testset "case construction and validation" begin
        case = IDT.TangentialHarmonicCase(; kind=:poloidal, l=4, m=2)
        @test IDT.casebandwidth(case) == 4
        @test IDT.casekind(case) === :poloidal
        @test IDT.casekind(IDT.TangentialHarmonicCase(; kind=:toroidal, l=4, m=2)) ===
            :toroidal
        # The FAMILY is part of the kind, so a mixed study can group its curves.
        @test IDT.casekind(case) !==
            IDT.casekind(IDT.TangentialHarmonicCase(; kind=:toroidal, l=4, m=2))

        @test_throws ArgumentError IDT.TangentialHarmonicCase(; kind=:radial, l=4, m=2)
        # No degree-zero tangential harmonic exists: the surface gradient of a constant vanishes.
        @test_throws ArgumentError IDT.TangentialHarmonicCase(; kind=:poloidal, l=0, m=0)
        @test_throws ArgumentError IDT.TangentialHarmonicCase(; kind=:poloidal, l=2, m=3)
    end

    @testset "fillexact! fills both components, in the array's index order" begin
        case = IDT.TangentialHarmonicCase(; kind=:poloidal, l=4, m=2)
        field = MLFMA.TangentialSphericalFarField(T, source, Val(:zero))
        IDT.fillexact!(field, case, source)

        θvalues, φvalues = MLFMA.θs(source), MLFMA.φs(source)
        # `[φindex, θindex]`, the transpose of how a grid is described -- a transposed fill would still
        # look like a valid field, so this is checked against the point it claims to be.
        expected = IDT.vectorsphericalharmonic(:poloidal, 4, 2, θvalues[5], φvalues[3])
        @test field[1][3, 5] ≈ expected[1]
        @test field[2][3, 5] ≈ expected[2]
        @test maximum(abs, field[1]) > 0.01   # non-vacuity
        @test maximum(abs, field[2]) > 0.01
    end

    # THE SCALAR PATH MUST BE UNTOUCHED. Component-aware metrics were added so tangential studies could
    # exist, not to change what a scalar number means -- `interpolationdiagnostics.jl` and seven of the
    # nine benchmark groups read `rel_l2` and would silently change meaning if this drifted.
    @testset "scalar results are unchanged by component awareness" begin
        result = only(
            IDT.results(
                IDT.interpolationerror(
                    interpolator, source, target, IDT.HarmonicCase(; l=10, m=0)
                ),
            ),
        )
        @test length(result.component_rel_l2) == 1
        @test result.component_rel_l2[1] ≈ result.rel_l2
        @test result.vector_rel_l2 ≈ result.rel_l2
        @test result.vector_rel_linf ≈ result.rel_linf
        # The value itself, so a refactor that changed the combination cannot pass unnoticed.
        @test result.rel_l2 ≈ 0.00598216 rtol = 1e-4
    end

    @testset "component metrics expose the θ/φ asymmetry" begin
        result = tangential(IDT.TangentialHarmonicCase(; kind=:poloidal, l=10, m=5))
        @test length(result.component_rel_l2) == 2
        @test length(result.component_rel_linf) == 2
        @test all(isfinite, result.component_rel_l2)
        # The two components genuinely differ -- if they did not, a combined metric would suffice and
        # this whole split would be decoration.
        @test !(result.component_rel_l2[1] ≈ result.component_rel_l2[2])

        # `vector_rel_l2 == rel_l2` identically: the same sum of squares grouped two ways. Pinned so
        # nobody reads a difference into two numbers that cannot differ.
        @test result.vector_rel_l2 ≈ result.rel_l2
        # The linf pair is where the vector view actually says something different.
        @test result.vector_rel_linf != result.rel_linf
    end

    @testset "a zero component reports zero, not Inf or an error" begin
        # A poloidal m = 0 field has Fφ ≡ 0 EXACTLY -- and (l=1, m=0) is the first tangential case
        # anyone reaches for, so this edge case is on the happy path, not in a corner.
        result = tangential(IDT.TangentialHarmonicCase(; kind=:poloidal, l=10, m=0))
        @test result.component_rel_l2[2] == 0
        @test result.component_rel_linf[2] == 0
        @test result.component_rel_l2[1] > 0
        # And the combined number is unaffected by the zero component.
        @test isfinite(result.rel_l2)
        @test result.rel_l2 ≈ result.component_rel_l2[1] rtol = 1e-12

        # The toroidal m = 0 field is the mirror: Fθ ≡ 0 instead.
        mirror = tangential(IDT.TangentialHarmonicCase(; kind=:toroidal, l=10, m=0))
        @test mirror.component_rel_l2[1] == 0
        @test mirror.component_rel_l2[2] > 0
    end

    # C = r̂ × B swaps the components, so the per-component errors must swap while the combined error
    # stays put. An independent check on the whole chain -- case, fill, interpolation, metrics -- that
    # no single-family test could make.
    @testset "poloidal and toroidal are exact mirrors" begin
        for (l, m) in ((10, 0), (10, 5), (10, 10), (6, 3))
            poloidal = tangential(IDT.TangentialHarmonicCase(; kind=:poloidal, l=l, m=m))
            toroidal = tangential(IDT.TangentialHarmonicCase(; kind=:toroidal, l=l, m=m))

            @test poloidal.rel_l2 ≈ toroidal.rel_l2 rtol = 1e-10
            @test poloidal.component_rel_l2[1] ≈ toroidal.component_rel_l2[2] rtol = 1e-10
            @test poloidal.component_rel_l2[2] ≈ toroidal.component_rel_l2[1] rtol = 1e-10
        end
    end

    @testset "θbandprofile attributes the error in θ" begin
        result = IDT.interpolationerror(
            interpolator,
            source,
            target,
            IDT.TangentialHarmonicCase(; kind=:poloidal, l=10, m=5);
            farfieldtype=MLFMA.TangentialSphericalFarField,
        )
        profile = IDT.θbandprofile(result, 1)

        # The bands partition the grid, so the two hemispheres together account for all of it.
        @test sum(profile.north) + sum(profile.south) ≈ 1 rtol = 1e-10
        # Symmetric hemispheres: an asymmetry here is what a sign error in the pole continuation would
        # produce, which is why the helper keeps them apart rather than summing.
        @test all(isapprox.(profile.north, profile.south; rtol=1e-6))
        # THE ATTRIBUTION RESULT: the error does NOT pile up at the poles despite Fφ's 1/sinθ factor.
        # If this ever inverts, a componentwise tangential number is reporting coordinates rather than
        # the interpolator, and must not be used to reject an interpolator family.
        @test profile.north[1] < 1e-6
        @test profile.north[end] > 0.1

        @test_throws ArgumentError IDT.θbandprofile(result, 1; bands=0)
        @test_throws ArgumentError IDT.θbandprofile(result, 1; bands=1000)
    end

    @testset "random tangential bands" begin
        @testset "construction and coefficients" begin
            case = IDT.RandomTangentialBandlimitedCase(; L=3, seed=0x1234)
            @test IDT.casebandwidth(case) == 3
            @test IDT.casekind(case) === :randomtangential
            # L >= 1: there is no degree-zero tangential harmonic to superpose.
            @test_throws ArgumentError IDT.RandomTangentialBandlimitedCase(; L=0)
            @test_throws ArgumentError IDT.RandomTangentialBandlimitedCase(;
                L=3, decay=:cubic
            )

            coefficients = IDT.tangentialbandcoefficients(case)
            # Two separate vectors, each indexed by the SCALAR packing with the l = 0 slot unused --
            # chosen over a combined index precisely so there is no offset to get wrong.
            @test length(coefficients.poloidal) == (3 + 1)^2
            @test coefficients.poloidal[1] == 0
            @test coefficients.toroidal[1] == 0
            # The two families must be independently drawn, or "both families in one field" is a lie
            # and the case is just a poloidal band counted twice.
            @test coefficients.poloidal[2:end] != coefficients.toroidal[2:end]
            # Reproducible from the seed: a study is a baseline, not a sample.
            @test IDT.tangentialbandcoefficients(case).poloidal == coefficients.poloidal
            @test IDT.tangentialbandcoefficients(
                IDT.RandomTangentialBandlimitedCase(; L=3, seed=0x4321)
            ).poloidal != coefficients.poloidal
        end

        @testset "it fills both components and interpolates sanely" begin
            case = IDT.RandomTangentialBandlimitedCase(; L=6, seed=0x1234)
            field = MLFMA.TangentialSphericalFarField(T, source, Val(:zero))
            IDT.fillexact!(field, case, source)
            # A mixed band excites BOTH components everywhere -- unlike a poloidal m = 0 harmonic.
            @test maximum(abs, field[1]) > 0.01
            @test maximum(abs, field[2]) > 0.01

            result = tangential(case)
            @test isfinite(result.rel_l2)
            @test result.rel_l2 > 0
            @test all(>(0), result.component_rel_l2)
            @test result.vector_rel_l2 ≈ result.rel_l2
        end

        @testset "error grows with bandwidth" begin
            errors = map((4, 10, 20)) do L
                tangential(IDT.RandomTangentialBandlimitedCase(; L=L, seed=0x1234)).rel_l2
            end
            @test issorted(errors)
            @test errors[end] > 100 * errors[1]
        end

        # THE FINDING THIS CASE EXISTS TO PIN. On a random band -- both families mixed, which is what a
        # real far-field pattern is -- the two interpolator families are essentially INDISTINGUISHABLE
        # in accuracy (measured ratio 0.993 to 1.000). Global φ's advantage is real but confined to pure
        # sectoral content, and a full band carries little of its energy there; the total stays
        # θ-limited, which is exactly what the θ/φ isolation study predicts. So the case for global φ
        # rests on speed and on φ-exactness, NOT on accuracy for realistic content -- and this test
        # exists so nobody re-derives an accuracy advantage that the mixed-content evidence does not show.
        @testset "the two families are indistinguishable on mixed content" begin
            case = IDT.RandomTangentialBandlimitedCase(; L=10, seed=0x1234)
            ratios = map((8, 20)) do p
                errors = map((
                    MLFMA.SphericalLagrangeInterpolator, MLFMA.LocalθGlobalφInterpolator
                )) do F
                    only(
                        IDT.results(
                            IDT.interpolationerror(
                                F{p}(T, Float64),
                                source,
                                target,
                                case;
                                farfieldtype=MLFMA.TangentialSphericalFarField,
                            ),
                        ),
                    ).rel_l2
                end
                return errors[2] / errors[1]
            end
            @test all(r -> 0.95 < r <= 1.02, ratios)
        end
    end

    @testset "still-unsupported representations raise" begin
        case = IDT.TangentialHarmonicCase(; kind=:poloidal, l=4, m=2)
        for F in (MLFMA.SphericalVectorFarField, MLFMA.PairedSphericalVectorFarField)
            @test_throws ErrorException IDT.interpolationerror(
                interpolator, source, target, case; farfieldtype=F
            )
        end
    end
end
