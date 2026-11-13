using MLFMA
using LinearAlgebra
using LegendrePolynomials
using Test

const ID = MLFMA.InterpolationDiagnostics

# The exact angular functions are the GROUND TRUTH of this whole module, and they are hand-implemented
# (MLFMA depends on neither LegendrePolynomials nor SpecialFunctions, and a diagnostic is a poor
# reason to add one). So they are pinned three independent ways rather than spot-checked: against
# closed forms, against an outside implementation available in the test environment, and against the
# orthonormality integral computed with the sampling's own quadrature. A wrong normalization, a
# dropped Condon-Shortley sign or a transposed recurrence fails at least two of the three.
@testset "Spherical harmonic ground truth" begin
    θ, φ = 0.7, 1.3

    @testset "closed forms" begin
        @test ID.sphericalharmonic(0, 0, θ, φ) ≈ 1 / sqrt(4π)
        @test ID.sphericalharmonic(1, 0, θ, φ) ≈ sqrt(3 / (4π)) * cos(θ)
        # Negative because P_l^m carries the Condon-Shortley (-1)^m: the sign is part of the
        # convention this module documents, not an accident to be normalized away.
        @test ID.sphericalharmonic(1, 1, θ, φ) ≈ -sqrt(3 / (8π)) * sin(θ) * cis(φ)
        @test ID.sphericalharmonic(2, 0, θ, φ) ≈ 0.25 * sqrt(5 / π) * (3cos(θ)^2 - 1)
        @test ID.sphericalharmonic(2, 2, θ, φ) ≈ 0.25 * sqrt(15 / (2π)) * sin(θ)^2 * cis(2φ)
    end

    @testset "Y_l^{-m} = (-1)^m conj(Y_l^m)" begin
        for l in 0:6, m in 1:l
            y = ID.sphericalharmonic(l, m, θ, φ)
            @test ID.sphericalharmonic(l, -m, θ, φ) ≈ (isodd(m) ? -conj(y) : conj(y))
            @test abs(ID.sphericalharmonic(l, -m, θ, φ)) ≈ abs(y)
        end
    end

    # An outside implementation of P_l^m, scaled by our documented normalization. `big` factorials
    # because (l+m)! overflows Float64's exact range well before l = 12 -- which is itself the reason
    # `normalizedlegendre` recurses in the normalized quantity instead of scaling afterwards.
    @testset "against LegendrePolynomials" begin
        for testθ in (0.01, 0.7, 1.5707963, 3.1)
            for l in 0:12, m in 0:l
                scale = sqrt(
                    BigFloat(2l + 1) / (4 * BigFloat(π)) * factorial(big(l - m)) /
                    factorial(big(l + m)),
                )
                reference = Float64(scale * Plm(cos(testθ), l, m))
                @test ID.normalizedlegendre(l, m, testθ) ≈ reference atol = 1e-13
            end
        end
    end

    # ∫ Y_l^m conj(Y_l'^m') dΩ = δ δ, integrated on the Gauss-Legendre θ grid MLFMA itself uses. This
    # is the test that would catch a normalization error the closed forms above happen to agree with,
    # and it also confirms the θ/φ measure this module assumes when it fills a grid.
    @testset "orthonormality on the sampling quadrature" begin
        sampling = MLFMA.SphericalSampling{Float64}(24, 48)
        θvalues = MLFMA.θs(sampling)
        φvalues = MLFMA.φs(sampling)
        wθvalues = MLFMA.wθs(sampling)

        function inner(l1, m1, l2, m2)
            total = zero(ComplexF64)
            for (i, θi) in enumerate(θvalues)
                weight = wθvalues[i] * (π / 2) * sin(θi) * (2π / length(φvalues))
                for φj in φvalues
                    total +=
                        weight *
                        ID.sphericalharmonic(l1, m1, θi, φj) *
                        conj(ID.sphericalharmonic(l2, m2, θi, φj))
                end
            end
            return total
        end

        for (l, m) in ((0, 0), (1, 0), (3, 2), (5, -3), (5, 5), (8, -8))
            @test inner(l, m, l, m) ≈ 1 atol = 1e-12
        end
        @test abs(inner(5, 3, 5, 2)) < 1e-12
        @test abs(inner(5, 3, 4, 3)) < 1e-12
        @test abs(inner(5, 3, 5, -3)) < 1e-12
    end
end

@testset "Case construction and validation" begin
    @testset "HarmonicCase" begin
        case = ID.HarmonicCase(; l=4, m=-2)
        @test case.l == 4
        @test case.m == -2
        @test ID.casebandwidth(case) == 4
        @test ID.casekind(case) === :harmonic

        @test_throws ArgumentError ID.HarmonicCase(; l=-1, m=0)
        @test_throws ArgumentError ID.HarmonicCase(; l=2, m=3)
        @test_throws ArgumentError ID.HarmonicCase(; l=2, m=-3)
        # l = 0, m = 0 is the one degenerate case that must be ALLOWED: a constant is a legitimate
        # exact function, and rejecting it would be an off-by-one in the validation.
        @test ID.HarmonicCase(; l=0, m=0).l == 0
    end

    @testset "RandomBandlimitedCase" begin
        case = ID.RandomBandlimitedCase(; L=6, seed=0x99)
        @test case.L == 6
        @test case.seed == 0x99
        @test case.decay === :none
        @test ID.casebandwidth(case) == 6
        @test ID.casekind(case) === :randombandlimited

        @test_throws ArgumentError ID.RandomBandlimitedCase(; L=-1)
        @test_throws ArgumentError ID.RandomBandlimitedCase(; L=4, decay=:exponential)
        for decay in (:none, :inverse_l, :inverse_l2)
            @test ID.RandomBandlimitedCase(; L=3, decay=decay).decay === decay
        end
    end

    @testset "seed determines the function" begin
        a = ID.RandomBandlimitedCase(; L=4, seed=0x2024)
        b = ID.RandomBandlimitedCase(; L=4, seed=0x2024)
        c = ID.RandomBandlimitedCase(; L=4, seed=0x2025)
        # Same seed must give bit-identical coefficients, or a study is a sample rather than a
        # baseline and two runs cannot be compared.
        @test ID.bandcoefficients(a) == ID.bandcoefficients(b)
        @test ID.bandcoefficients(a) != ID.bandcoefficients(c)
        @test length(ID.bandcoefficients(a)) == 25
    end

    @testset "decay weights by degree" begin
        # The l = 0 coefficient is unweighted by all three (1/(1+0) == 1), so the discrimination has
        # to be looked for at higher l -- comparing only the first entry would pass for :none too.
        plain = ID.bandcoefficients(ID.RandomBandlimitedCase(; L=3, seed=0x7, decay=:none))
        decayed = ID.bandcoefficients(
            ID.RandomBandlimitedCase(; L=3, seed=0x7, decay=:inverse_l)
        )
        squared = ID.bandcoefficients(
            ID.RandomBandlimitedCase(; L=3, seed=0x7, decay=:inverse_l2)
        )
        @test plain[1] == decayed[1] == squared[1]
        highest = length(plain)
        @test decayed[highest] ≈ plain[highest] / 4
        @test squared[highest] ≈ plain[highest] / 16
    end

    @testset "harmoniccases families" begin
        cases = ID.harmoniccases(4:4:8)
        @test length(cases) == 6
        @test all(case isa ID.HarmonicCase for case in cases)
        @test [(case.l, case.m) for case in cases] == [(4, 0), (4, 2), (4, 4), (8, 0), (8, 4), (8, 8)]

        @test [c.m for c in ID.harmoniccases(6:6; families=(:axisymmetric,))] == [0]
        @test [c.m for c in ID.harmoniccases(6:6; families=(:sectoral,))] == [6]
        @test_throws ArgumentError ID.harmoniccases(2:2; families=(:radial,))
    end
end

@testset "Bandwidth helpers" begin
    # supporteddegree is the exact inverse of MLFMA.NθNφ -- the property that makes rho == 1 mean
    # "at the grid's design degree" rather than an arbitrary fraction of it.
    for L in (2, 5, 10, 50)
        Nθ, Nφ = MLFMA.NθNφ(L)
        @test ID.supporteddegree(MLFMA.SphericalSampling{Float64}(Nθ, Nφ)) == L
    end

    sampling = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(10)...)
    @test ID.rho(ID.HarmonicCase(; l=10, m=0), sampling) ≈ 1.0
    @test ID.rho(ID.HarmonicCase(; l=5, m=0), sampling) ≈ 0.5
    @test ID.rho(ID.RandomBandlimitedCase(; L=20), sampling) ≈ 2.0
    @test ID.rho(ID.HarmonicCase(; l=3, m=0), sampling) isa Float64

    # A grid too coarse to support any varying harmonic must say so rather than divide by zero.
    @test_throws ArgumentError ID.rho(
        ID.HarmonicCase(; l=1, m=0), MLFMA.SphericalSampling{Float64}(2, 4)
    )
end

@testset "interpolationerror" begin
    source = MLFMA.SphericalSampling{Float64}(28, 56)
    target = MLFMA.SphericalSampling{Float64}(56, 112)
    interpolator = MLFMA.SphericalLagrangeInterpolator{8}(ComplexF64, Float64)

    cases = [
        ID.HarmonicCase(; l=6, m=0),
        ID.HarmonicCase(; l=6, m=3),
        ID.HarmonicCase(; l=6, m=6),
        ID.RandomBandlimitedCase(; L=6, seed=0x1234),
    ]
    study = ID.interpolationerror(interpolator, source, target, cases)

    @testset "shapes and metadata" begin
        @test length(study) == length(cases)
        @test length(ID.results(study)) == length(cases)

        meta = ID.metadata(study)
        @test meta.sourcesize == (28, 56)
        @test meta.targetsize == (56, 112)
        # One stage, described the same way a chain's stages are -- there is no separate singular
        # `stencilsize` field, because a chain has no single stencil size to report.
        @test length(meta.stages) == 1
        @test meta.stencilsizes == [8]
        @test only(meta.stages).stencilsize == 8
        @test only(meta.stages).interpolator === interpolator
        @test meta.farfieldtype === MLFMA.ScalarFarField
        # The trait the interpolation actually ran under, which is the representation's default and
        # NOT the interpolator's own field -- the three-argument call ignores that field.
        @test meta.interpolationtrait ===
            MLFMA.defaultinterpolationtrait(MLFMA.ScalarFarField)

        for result in ID.results(study)
            @test result.source_size == (28, 56)
            @test result.target_size == (56, 112)
            # Pointwise fields keep the ARRAY shape, (Nφ, Nθ), which is the transpose of the grid
            # description above. Both appear in this package; the result type commits to one of each
            # and this pins which is which.
            @test size(result.abs_error) == size(target)
            @test size(result.rel_error) == size(target)
            @test size(result.abs_error) == (112, 56)
        end
    end

    @testset "target is evaluated independently of source" begin
        # Not merely "the shapes differ": interpolate onto a target of the SAME size as the source
        # and check the reference is still the exact function there, which a copied-from-source
        # reference would also satisfy -- so also check a target on a DIFFERENT grid whose exact
        # values cannot be any subset of the source's.
        # Nθ odd and Nφ not a multiple of the source's, so no target direction coincides with a
        # source one and nothing could be lifted across by accident. Nφ must stay even --
        # `SphericalSampling` asserts it, since the φ grid is mirrored.
        case = ID.HarmonicCase(; l=4, m=2)
        odd = MLFMA.SphericalSampling{Float64}(31, 58)
        oddstudy = ID.interpolationerror(interpolator, source, odd, case)
        result = only(ID.results(oddstudy))
        @test result.target_size == (31, 58)
        @test size(result.abs_error) == size(odd)

        exact = MLFMA.ScalarFarField(ComplexF64, odd, Val(:zero))
        ID.fillexact!(exact, case, odd)
        θvalues = MLFMA.θs(odd)
        φvalues = MLFMA.φs(odd)
        # Spot-check the fill against the harmonic evaluated directly, INCLUDING the index order:
        # a transposed fill would still be a valid-looking field.
        @test exact[1][3, 5] ≈ ID.sphericalharmonic(4, 2, θvalues[5], φvalues[3])
        @test exact[1][1, 1] ≈ ID.sphericalharmonic(4, 2, θvalues[1], φvalues[1])
        # Non-vacuity: the exact function is not identically zero on either grid.
        @test maximum(abs, exact[1]) > 0.01
    end

    @testset "metrics are finite and nonvacuous" begin
        for result in ID.results(study)
            @test isfinite(result.rel_l2)
            @test isfinite(result.rel_linf)
            @test isfinite(result.max_abs)
            @test result.rel_l2 > 0
            @test all(isfinite, result.abs_error)
            @test all(isfinite, result.rel_error)
            # The floored pointwise denominator must not leak Inf/NaN even where the harmonic has
            # genuine zeros on the grid.
            @test !any(isnan, result.rel_error)
        end
    end

    @testset "rows and worstcase" begin
        table = ID.rows(study)
        @test length(table) == length(cases)
        @test all(row -> keys(row) == keys(table[1]), table)
        @test table[1].casekind === :harmonic
        @test table[end].casekind === :randombandlimited
        @test table[end].l === nothing
        @test table[end].L == 6
        @test table[1].L === nothing
        # Table-friendly means scalars only: no pointwise arrays leak into a row.
        @test all(row -> all(v -> !(v isa AbstractArray), values(row)), table)

        worst = ID.worstcase(study)
        @test worst in ID.results(study)
        @test worst.rel_l2 == maximum(r.rel_l2 for r in ID.results(study))
        byinf = ID.worstcase(study; by=:rel_linf)
        @test byinf.rel_linf == maximum(r.rel_linf for r in ID.results(study))
        @test_throws ArgumentError ID.worstcase(study; by=:rel_error)
    end

    @testset "pointwiseerror" begin
        @test ID.pointwiseerror(study, 1) === ID.results(study)[1].rel_error
        @test ID.pointwiseerror(study, 1; kind=:absolute) === ID.results(study)[1].abs_error
        @test_throws ArgumentError ID.pointwiseerror(study, 1; kind=:squared)
    end

    @testset "summary" begin
        text = ID.summary(study)
        @test text isa String
        @test occursin("InterpolationErrorStudy", text)
        @test occursin("harmonic l=6 m=3", text)
        @test occursin("random L=6", text)
        # One header block plus one line per case, so a reader can line the table up with the cases.
        @test count(==('\n'), text) >= length(cases) + 3
    end

    @testset "single case returns a one-element study" begin
        one = ID.interpolationerror(interpolator, source, target, cases[1])
        @test one isa ID.InterpolationErrorStudy
        @test length(one) == 1
        # Uniform return is the point: every helper works on whatever interpolationerror gave back.
        @test ID.summary(one) isa String
        @test ID.worstcase(one) === only(ID.results(one))
    end

    @testset "unsupported far-field types raise" begin
        # `TangentialSphericalFarField` is no longer in this list: it is supported, with its own exact
        # cases (see `tangentialdiagnostics.jl`). The two below still are not -- both carry a RADIAL
        # component that a tangential harmonic does not supply, so they need their own case type rather
        # than a widened guard.
        for F in (MLFMA.SphericalVectorFarField, MLFMA.PairedSphericalVectorFarField)
            @test_throws ErrorException ID.interpolationerror(
                interpolator, source, target, cases[1]; farfieldtype=F
            )
        end

        # A SCALAR case against the now-supported tangential representation must still fail, and fail
        # at the fill rather than half-filling component 1 and leaving Fφ zero -- which is exactly the
        # "fake a vector field with scalars" this module refuses to do.
        @test_throws MethodError ID.interpolationerror(
            interpolator,
            source,
            target,
            cases[1];
            farfieldtype=MLFMA.TangentialSphericalFarField,
        )
    end
end

@testset "Error responds to degree and to interpolation order" begin
    source = MLFMA.SphericalSampling{Float64}(28, 56)
    target = MLFMA.SphericalSampling{Float64}(56, 112)

    @testset "higher degree is harder" begin
        interpolator = MLFMA.SphericalLagrangeInterpolator{6}(ComplexF64, Float64)
        study = ID.interpolationerror(
            interpolator,
            source,
            target,
            [ID.HarmonicCase(; l=l, m=0) for l in (2, 6, 10, 14)],
        )
        errors = [r.rel_l2 for r in ID.results(study)]
        # Strictly increasing, not merely "worst is last": the diagnostic is supposed to resolve the
        # trend, and a metric that only orders the endpoints would be a much weaker instrument.
        @test issorted(errors)
        @test all(errors[i] < errors[i + 1] for i in 1:(length(errors) - 1))
        # And the growth has to be substantial, or the ordering could be numerical noise.
        @test errors[end] > 100 * errors[1]
    end

    @testset "higher order does not catastrophically worsen a smooth case" begin
        case = ID.HarmonicCase(; l=4, m=2)
        errors = map((4, 6, 8, 10)) do stencil
            interpolator = MLFMA.SphericalLagrangeInterpolator{stencil}(ComplexF64, Float64)
            study = ID.interpolationerror(interpolator, source, target, case)
            return only(ID.results(study)).rel_l2
        end
        @test all(isfinite, errors)
        @test errors[end] <= errors[1]
        @test all(e -> e < 1e-2, errors)
    end

    @testset "rho above one is where it falls apart" begin
        # Pins the interpretation supporteddegree exists to support: inside the band the error is
        # small, and past it the grid cannot represent the case at all. If this ever inverts, either
        # supporteddegree or the interpolator changed meaning.
        interpolator = MLFMA.SphericalLagrangeInterpolator{8}(ComplexF64, Float64)
        inband = only(
            ID.results(
                ID.interpolationerror(
                    interpolator, source, target, ID.HarmonicCase(; l=4, m=0)
                ),
            ),
        )
        outofband = only(
            ID.results(
                ID.interpolationerror(
                    interpolator, source, target, ID.HarmonicCase(; l=26, m=0)
                ),
            ),
        )
        @test inband.rho < 1
        @test outofband.rho > 1
        @test inband.rel_l2 < 1e-4
        @test outofband.rel_l2 > 1e-2
    end
end

# Both production interpolators resample θ and φ separably, θ first. These pin that a chain through
# `intermediatesampling` really does split the two steps -- which is what the interpolation refactor's
# diagnostics phase is built on, so a change that broke the split silently would invalidate every
# θ-versus-φ conclusion drawn from it.
@testset "Chained studies isolate θ from φ" begin
    source = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(13)...)
    target = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(26)...)
    middle = ID.intermediatesampling(source, target)
    interpolator = MLFMA.SphericalLagrangeInterpolator{8}(ComplexF64, Float64)

    cases = [
        ID.HarmonicCase(; l=10, m=0),
        ID.HarmonicCase(; l=10, m=5),
        ID.HarmonicCase(; l=10, m=10),
        ID.RandomBandlimitedCase(; L=10, seed=0x1234),
    ]

    @testset "intermediatesampling takes θ from the target and φ from the source" begin
        @test MLFMA.Nθ(middle) == MLFMA.Nθ(target)
        @test MLFMA.Nφ(middle) == MLFMA.Nφ(source)
        # The NODES, not just the counts. Equal counts would not be enough: the split relies on the
        # stage landing on nodes it already has, which is a statement about node positions.
        @test MLFMA.θs(middle) == MLFMA.θs(target)
        @test MLFMA.φs(middle) == MLFMA.φs(source)
    end

    @testset "a same-grid stage is the exact identity" begin
        # Exactly zero, not merely small. The interpolation weights detect the coincident node and
        # assign it weight one, rather than dividing by a difference that happens to be tiny.
        for case in (ID.HarmonicCase(; l=10, m=0), ID.HarmonicCase(; l=10, m=10))
            result = only(
                ID.results(ID.interpolationerror(interpolator, source, source, case))
            )
            @test result.rel_l2 == 0
            @test result.max_abs == 0
        end
    end

    @testset "a two-stage chain at equal orders reproduces the single stage" begin
        single = ID.interpolationerror(interpolator, source, target, cases)
        chained = ID.interpolationerror(
            [interpolator, interpolator], [source, middle, target], cases
        )
        @test length(chained) == length(single)
        for (a, b) in zip(ID.results(single), ID.results(chained))
            # The decomposition is the SAME arithmetic regrouped, so this is a round-off-level
            # agreement rather than an approximation -- which is what licenses reading a θ-only or
            # φ-only chain as a component of the single-stage error.
            @test b.rel_l2 ≈ a.rel_l2 rtol = 1e-8
            @test b.rel_linf ≈ a.rel_linf rtol = 1e-8
        end
    end

    @testset "θ and φ contributions separate on an axisymmetric case" begin
        axisymmetric = ID.HarmonicCase(; l=10, m=0)
        total = only(
            ID.results(ID.interpolationerror(interpolator, source, target, axisymmetric))
        )
        θonly = only(
            ID.results(ID.interpolationerror(interpolator, source, middle, axisymmetric))
        )
        φonly = only(
            ID.results(ID.interpolationerror(interpolator, middle, target, axisymmetric))
        )
        # m = 0 has no φ dependence, so the φ stage resamples a function that is constant along φ
        # and must contribute nothing, leaving the θ stage owning the whole error. The two grids have
        # different Nφ, but every φ column is identical here, so the count cancels out of a RELATIVE
        # norm and the two are comparable.
        @test φonly.rel_l2 < 1e-13
        @test θonly.rel_l2 ≈ total.rel_l2 rtol = 1e-8
    end

    @testset "separate θ and φ orders" begin
        # The order-split prototype: two interpolators of different stencil size, which measures
        # `pθ != pφ` before any such production API exists.
        case = ID.HarmonicCase(; l=10, m=5)
        function splitorder(pθ, pφ)
            return only(
                ID.results(
                    ID.interpolationerror(
                        [
                            MLFMA.SphericalLagrangeInterpolator{pθ}(ComplexF64, Float64),
                            MLFMA.SphericalLagrangeInterpolator{pφ}(ComplexF64, Float64),
                        ],
                        [source, middle, target],
                        case,
                    ),
                ),
            ).rel_l2
        end

        low, θhigh, φhigh, both = splitorder(8, 8),
        splitorder(20, 8), splitorder(8, 20),
        splitorder(20, 20)

        # Raising the θ order alone recovers nearly everything raising both does; raising the φ order
        # alone recovers almost nothing. Asserted as orderings with wide factors rather than as
        # measured numbers, so what is pinned is the finding and not one machine's digits.
        @test θhigh < low / 10
        @test φhigh > low / 2
        @test θhigh < 2 * both
    end

    @testset "metadata describes every stage" begin
        chained = ID.interpolationerror(
            [
                MLFMA.SphericalLagrangeInterpolator{20}(ComplexF64, Float64),
                MLFMA.SphericalLagrangeInterpolator{8}(ComplexF64, Float64),
            ],
            [source, middle, target],
            cases[1],
        )
        meta = ID.metadata(chained)
        @test length(meta.stages) == 2
        @test meta.stencilsizes == [20, 8]
        @test meta.sourcesize == (MLFMA.Nθ(source), MLFMA.Nφ(source))
        @test meta.targetsize == (MLFMA.Nθ(target), MLFMA.Nφ(target))
        @test meta.stages[1].targetsize == (MLFMA.Nθ(middle), MLFMA.Nφ(middle))
        @test meta.stages[2].sourcesize == (MLFMA.Nθ(middle), MLFMA.Nφ(middle))
        # ρ and the supported degree describe the grid the case STARTS on, not an intermediate.
        @test meta.supporteddegree == ID.supporteddegree(source)
        @test only(ID.results(chained)).rho == ID.rho(cases[1], source)
        text = ID.summary(chained)
        @test occursin("2 stages", text)
        @test occursin("stage 2", text)
    end

    @testset "chain length is checked" begin
        @test_throws DimensionMismatch ID.interpolationerror(
            [interpolator], [source, middle, target], cases[1]
        )
        @test_throws DimensionMismatch ID.interpolationerror(
            [interpolator, interpolator], [source, target], cases[1]
        )
        @test_throws ArgumentError ID.interpolationerror([], [source, target], cases[1])
    end
end

# The API claims to take any production interpolator, and only `SphericalLagrangeInterpolator` was
# exercised above. This pins the other one -- and does it with the case that can actually tell them
# apart, which is the point worth getting right: `LocalθGlobalφInterpolator` differs from the Lagrange
# one only in φ, so at m = 0 (no φ dependence at all) the two agree to the last bit and a smoke test
# there would pass even if the second path were never taken. The sectoral case m = l, with the fastest
# φ oscillation its degree allows, is where the difference lives.
@testset "LocalθGlobalφInterpolator" begin
    source = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(13)...)
    target = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(26)...)
    lagrange = MLFMA.SphericalLagrangeInterpolator{8}(ComplexF64, Float64)
    localglobal = MLFMA.LocalθGlobalφInterpolator{8}(ComplexF64, Float64)

    @testset "it runs and produces sane results" begin
        study = ID.interpolationerror(
            localglobal,
            source,
            target,
            [
                ID.HarmonicCase(; l=4, m=2),
                ID.HarmonicCase(; l=10, m=10),
                ID.RandomBandlimitedCase(; L=10, seed=0x1234),
            ],
        )
        @test length(study) == 3
        for result in ID.results(study)
            @test isfinite(result.rel_l2)
            @test result.rel_l2 > 0
            @test result.rel_l2 < 1
            @test size(result.abs_error) == size(target)
            @test !any(isnan, result.rel_error)
        end
        @test ID.metadata(study).stencilsizes == [8]
        @test ID.summary(study) isa String
    end

    @testset "identical to the Lagrange interpolator where φ does not vary" begin
        # m = 0 is φ-independent, so a global-in-φ scheme and a local one have nothing to disagree
        # about. Equality here is what proves the sectoral difference below is about φ and not about
        # two unrelated implementations.
        axisymmetric = ID.HarmonicCase(; l=10, m=0)
        a = only(ID.results(ID.interpolationerror(lagrange, source, target, axisymmetric)))
        b = only(
            ID.results(ID.interpolationerror(localglobal, source, target, axisymmetric))
        )
        @test a.rel_l2 ≈ b.rel_l2
        @test a.rel_linf ≈ b.rel_linf
    end

    @testset "better than the Lagrange interpolator on a sectoral case" begin
        # Measured: 1.48e-3 for the Lagrange scheme against 2.06e-4 for the global-φ one, a factor of
        # about 7. Asserted as a factor of 2 so the property is pinned without the test becoming a
        # transcription of one machine's numbers.
        sectoral = ID.HarmonicCase(; l=10, m=10)
        a = only(ID.results(ID.interpolationerror(lagrange, source, target, sectoral)))
        b = only(ID.results(ID.interpolationerror(localglobal, source, target, sectoral)))
        @test b.rel_l2 < a.rel_l2 / 2
    end
end

@testset "errormetrics" begin
    exact = ComplexF64[1 2; 3 4]
    @test ID.errormetrics(exact, exact).rel_l2 == 0
    @test ID.errormetrics(exact, exact).max_abs == 0

    perturbed = copy(exact)
    perturbed[2, 2] += 0.5
    metrics = ID.errormetrics(perturbed, exact)
    @test metrics.max_abs ≈ 0.5
    @test metrics.rel_l2 ≈ 0.5 / norm(exact)
    @test metrics.rel_linf ≈ 0.5 / 4
    @test size(metrics.abs_error) == size(exact)

    @test_throws DimensionMismatch ID.errormetrics(ComplexF64[1 2], exact)
    # An identically zero reference has no relative error; saying so beats returning NaN.
    @test_throws ErrorException ID.errormetrics(exact, zeros(ComplexF64, 2, 2))
end

@testset "Plotting helpers are absent without PlotlyJS" begin
    # The core diagnostics must not need PlotlyJS. These stubs exist so that calling one without the
    # extension says what to load instead of raising UndefVarError; if the extension IS loaded in this
    # session the real method takes over, so only check the functions exist and are callable names.
    @test ID.ploterror isa Function
    @test ID.plotcurves isa Function
end
