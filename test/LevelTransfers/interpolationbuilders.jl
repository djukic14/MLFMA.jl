using MLFMA
using BEAST
using LinearAlgebra
using Test

const IDB = MLFMA.InterpolationDiagnostics

# The builders are a CONSTRUCTION LAYER, so what has to be pinned is that they resolve to exactly the
# types hand-written construction produces -- otherwise "thin" is an intention rather than a property.
@testset "Interpolation builders" begin
    T = ComplexF64
    samplings = [MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(L)...) for L in (3, 5)]
    parentsamplings = [MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(L)...) for L in (6, 10)]

    @testset "resolve to today's parameterized types" begin
        lagrange = MLFMA.SphericalLagrangeInterpolationBuilder(; order=8)
        localglobal = MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=8)

        @test MLFMA.interpolationorder(lagrange) == 8
        @test MLFMA.interpolationorder(localglobal) == 8
        @test MLFMA.elementtype(lagrange) === T

        # The order stays a TYPE parameter -- this is the assertion that would fail if anyone moved it
        # to a runtime field, which is the thing that would ripple through the information types.
        @test MLFMA.interpolatortype(lagrange) === MLFMA.SphericalLagrangeInterpolator{8,T}
        @test MLFMA.interpolatortype(localglobal) === MLFMA.LocalθGlobalφInterpolator{8,T}

        @test MLFMA.elementtype(
            MLFMA.SphericalLagrangeInterpolationBuilder(; order=8, T=ComplexF32)
        ) === ComplexF32
    end

    @testset "built operators match hand-written construction" begin
        for (builder, hand) in (
            (
                MLFMA.SphericalLagrangeInterpolationBuilder(; order=6),
                MLFMA.SphericalLagrangeInterpolator{6}(T, samplings, parentsamplings),
            ),
            (
                MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=6),
                MLFMA.LocalθGlobalφInterpolator{6}(T, samplings, parentsamplings),
            ),
        )
            built = MLFMA.buildinterpolator(builder, samplings, parentsamplings)
            @test typeof(built) === typeof(hand)
            @test MLFMA.stencilsize(built) == 6
            # The information cache is populated at construction, one entry per level pair, rather than
            # on the first apply -- which is the whole reason the pairs are passed in.
            @test length(built.interpolationinfodict) == length(samplings)
            @test length(built.interpolationinfodict) == length(hand.interpolationinfodict)
        end
    end

    # A FAMILY OWNS A PAIR. `buildreverseoperators` returning both is what stops a caller from building
    # an anterpolator and reusing it for the transpose direction -- the mistake that
    # `transposedinterpolation.jl` exists to prevent.
    @testset "buildreverseoperators returns both reverse operators" begin
        @testset "real operator: the two are the same object" begin
            builder = MLFMA.SphericalLagrangeInterpolationBuilder(; order=6)
            interpolator = MLFMA.buildinterpolator(builder, samplings, parentsamplings)
            ops = MLFMA.buildreverseoperators(builder, interpolator; nchunks=2)

            @test ops.anterpolator isa MLFMA.SphericalLagrangeAnterpolator
            @test ops.transposedinterpolator === ops.anterpolator
            @test MLFMA.isrealinterpolation(interpolator)
        end

        @testset "complex operator: they are different operators" begin
            builder = MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=6)
            interpolator = MLFMA.buildinterpolator(builder, samplings, parentsamplings)
            ops = MLFMA.buildreverseoperators(builder, interpolator; nchunks=2)

            @test ops.anterpolator isa MLFMA.LocalθGlobalφAnterpolator
            @test ops.transposedinterpolator isa MLFMA.TransposedInterpolator
            @test ops.transposedinterpolator !== ops.anterpolator
            @test !MLFMA.isrealinterpolation(interpolator)
            # And they really are the adjoint and the transpose of the same operator.
            source, target = samplings[1], parentsamplings[1]
            inter = MLFMA._getinterpolationmatrix(interpolator, source, target, T)
            @test MLFMA._getanterpolationmatrix(ops.anterpolator, target, source, T) ≈
                adjoint(inter)
            @test MLFMA._getanterpolationmatrix(
                ops.transposedinterpolator, target, source, T
            ) ≈ transpose(inter)
        end
    end

    @testset "LevelTransferStrategy accepts a builder" begin
        efie = MLFMA.efie(; wavenumber=2π)
        builder = MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=15)

        fromkeyword = MLFMA.LevelTransferStrategy(
            efie; Interpolator=MLFMA.LocalθGlobalφInterpolator{15,ComplexF64}
        )
        frombuilder = MLFMA.LevelTransferStrategy(efie, builder)
        # A spelling change, not a behaviour change: same stored type either way.
        @test frombuilder.Interpolator === fromkeyword.Interpolator
        @test frombuilder.PhaseShifter === fromkeyword.PhaseShifter

        # The element-type invariant the `typeof(γ(operator))` default protects has to survive the new
        # spelling, or the builder becomes a way around it.
        mismatched = MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=15, T=Float64)
        @test_throws ErrorException MLFMA.LevelTransferStrategy(efie, mismatched)
        err = try
            MLFMA.LevelTransferStrategy(efie, mismatched)
        catch e
            sprint(showerror, e)
        end
        @test occursin("element type", err)
        @test occursin("Float64", err)
    end
end

# WHAT AN ANTERPOLATION DIAGNOSTIC MEASURES, and what it cannot. See the long note at the top of
# `src/interpolation/diagnostics/adjointness.jl`: the adjointness residual is a correctness invariant at
# round-off, and anterpolation has no approximation error independent of the interpolator's, so the
# quantity worth reporting is the error the reverse direction puts into a Galerkin PAIRING.
@testset "adjointpairingerror" begin
    T = ComplexF64
    source = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(13)...)
    target = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(26)...)
    supported = IDB.supporteddegree(source)

    function operators(θorder)
        builder = MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=θorder)
        interpolator = MLFMA.buildinterpolator(builder, [source], [target])
        ops = MLFMA.buildreverseoperators(builder, interpolator; nchunks=1)
        return interpolator, ops.anterpolator
    end

    @testset "shape and fields" begin
        interpolator, anterpolator = operators(8)
        cases = [
            IDB.HarmonicCase(; l=6, m=3), IDB.RandomBandlimitedCase(; L=6, seed=0x1234)
        ]
        rows = IDB.adjointpairingerror(interpolator, anterpolator, source, target, cases)

        @test length(rows) == length(cases)
        @test all(row -> keys(row) == keys(rows[1]), rows)
        @test rows[1].casekind === :harmonic
        @test rows[end].casekind === :randombandlimited
        @test rows[1].bandwidth == 6
        @test rows[1].rho == IDB.rho(cases[1], source)
        @test all(row -> isfinite(row.pairing) && isfinite(row.adjointness), rows)
        # A single case comes back as a one-element vector, as with `interpolationerror`.
        @test length(
            IDB.adjointpairingerror(interpolator, anterpolator, source, target, cases[1])
        ) == 1
    end

    @testset "adjointness is a round-off invariant, not a varying quantity" begin
        # The point of reporting it next to `pairing`: it must stay at round-off across order and
        # bandwidth, so that a `pairing` number is never read off an operator that stopped being the
        # adjoint. Pinned across a range wide enough to include the ill-conditioned end.
        for θorder in (8, 20)
            interpolator, anterpolator = operators(θorder)
            rows = IDB.adjointpairingerror(
                interpolator,
                anterpolator,
                source,
                target,
                [IDB.RandomBandlimitedCase(; L=L, seed=0x11) for L in (4, 10, 20)],
            )
            @test all(row -> row.adjointness < 1e-10, rows)
        end
    end

    @testset "pairing error grows with rho and shrinks with order" begin
        cases = [IDB.RandomBandlimitedCase(; L=L, seed=0x11) for L in (4, 10, 20)]

        low = IDB.adjointpairingerror(operators(8)..., source, target, cases)
        high = IDB.adjointpairingerror(operators(20)..., source, target, cases)

        # Monotone in bandwidth at fixed order, and substantially so -- a metric that only ordered the
        # endpoints would be a much weaker instrument.
        errors = [row.pairing for row in low]
        @test issorted(errors)
        @test errors[end] > 100 * errors[1]

        # Higher order helps where the grid can still represent the case (ρ < 1) and cannot rescue a
        # case the grid does not carry (ρ = 1.54), which is the same interpretation ρ has everywhere
        # else in these diagnostics.
        @test high[1].pairing < low[1].pairing / 100
        @test high[2].pairing < low[2].pairing / 10
        @test high[end].pairing > low[end].pairing / 10

        # Non-vacuity: the in-band numbers must be genuinely small, or "grows with ρ" says nothing.
        @test low[1].pairing < 1e-4
    end

    @testset "it tracks interpolation error without being equal to it" begin
        # The pairing contracts the error field against another field, so it is normally SMALLER than
        # the field norm `interpolationerror` reports. Pinning the relationship rather than either
        # number keeps the two diagnostics honest about measuring different things.
        interpolator, anterpolator = operators(8)
        case = IDB.RandomBandlimitedCase(; L=10, seed=0x11)

        pairing =
            only(IDB.adjointpairingerror(interpolator, anterpolator, source, target, case)).pairing
        rel =
            only(IDB.results(IDB.interpolationerror(interpolator, source, target, case))).rel_l2

        @test pairing < rel
        @test pairing > rel / 1000
    end

    @testset "unsupported far-field types raise" begin
        interpolator, anterpolator = operators(8)
        # `SphericalVectorFarField` is still unsupported outright -- it carries a radial component no
        # tangential harmonic supplies.
        @test_throws ErrorException IDB.adjointpairingerror(
            interpolator,
            anterpolator,
            source,
            target,
            IDB.HarmonicCase(; l=4, m=0);
            farfieldtype=MLFMA.SphericalVectorFarField,
        )
        # The tangential representation IS supported now, but not with a scalar case: that combination
        # must fail at the fill rather than quietly filling one component.
        @test_throws MethodError IDB.adjointpairingerror(
            interpolator,
            anterpolator,
            source,
            target,
            IDB.HarmonicCase(; l=4, m=0);
            farfieldtype=MLFMA.TangentialSphericalFarField,
        )
    end
end
