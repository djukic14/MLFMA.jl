using Test
using MLFMA
using BEAST, CompScienceMeshes
using H2Factory
using H2Trees
@testset "Sorting RWG functions" begin
    λ = 1.0

    m1 = CompScienceMeshes.meshsphere(0.5 * λ, λ / 10)
    m2 = meshcuboid(0.5 * λ, 0.5 * λ, 0.5 * λ, λ / 10)
    m3 = meshrectangle(0.5 * λ, 0.5 * λ, λ / 10)
    for m in [m1, m2, m3]
        X = raviartthomas(m)

        tree = TwoNTree(
            X;
            builder=TwoNTreeBuilder(;
                minhalfsize=λ / 10, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
            ),
        )
        tree1 = deepcopy(tree)

        X1 = deepcopy(X)
        X2 = deepcopy(X)

        X = MLFMA.sortbasis!(tree, X)
        indices = MLFMA._sortbasisindices!(tree1, X1)

        @test X.fns == X2.fns[indices]
        @test X.pos == X2.pos[indices]

        sort!(indices)
        @test indices == 1:numfunctions(X2)
    end
end

@testset "Far-field representation mapping" begin
    efie = MLFMA.efie(; wavenumber=1.0)
    doublelayer = Maxwell3D.doublelayer(; wavenumber=1.0)
    rotateddoublelayer = BEAST.n × doublelayer
    singlelayer = MLFMA.singlelayer(efie)
    hypersingular = MLFMA.hypersingular(; wavenumber=1.0)
    hh_single = Helmholtz3D.singlelayer(; wavenumber=1.0)
    hh_double = Helmholtz3D.doublelayer(; wavenumber=1.0)
    hh_doubletransposed = Helmholtz3D.doublelayer_transposed(; wavenumber=1.0)
    hh_hyper = Helmholtz3D.hypersingular(; wavenumber=1.0)

    cases = (
        (efie, MLFMA.TangentialSphericalFarField),
        (doublelayer, MLFMA.TangentialSphericalFarField),
        (rotateddoublelayer, MLFMA.TangentialSphericalFarField),
        (singlelayer, MLFMA.SphericalVectorFarField),
        (hypersingular, MLFMA.ScalarFarField),
        (hh_single, MLFMA.ScalarFarField),
        (hh_double, MLFMA.ScalarFarField),
        (hh_doubletransposed, MLFMA.ScalarFarField),
        (hh_hyper, MLFMA.PairedSphericalVectorFarField),
    )

    for (op, ExpectedFarFieldType) in cases
        @test MLFMA.farfieldtype(op) === ExpectedFarFieldType
        # `farfielddims(op)` is derived from `farfieldtype(op)`; pin that the two agree rather
        # than duplicating the per-operator component-count table the derivation replaced.
        @test MLFMA.farfielddims(op) == MLFMA.fielddim(ExpectedFarFieldType)

        # Each operator's moment type must pair with collection types of its OWN representation.
        # `@linkmomenttocollection` only generates `mul!` for matching pairs, so a mismatch here
        # would surface later as a MethodError deep in a matvec rather than as a wiring error.
        @test MLFMA.radiationfieldcollectiontype(op) ===
            MLFMA.radiationcollectiontype(ExpectedFarFieldType)
        @test MLFMA.receivefieldcollectiontype(op) ===
            MLFMA.receivecollectiontype(ExpectedFarFieldType)
        @test MLFMA.fielddim(MLFMA.radiationfieldcollectiontype(op)) ==
            MLFMA.farfielddims(op)
        @test MLFMA.fielddim(MLFMA.receivefieldcollectiontype(op)) == MLFMA.farfielddims(op)
    end

    # Component count round-trips through every representation, including both three-component
    # ones -- `fielddim` is the single source of truth and nothing infers a representation back
    # out of it.
    sampling = MLFMA.SphericalSampling{Float64}(4, 6)
    for (D, FarFieldType) in (
        (1, MLFMA.ScalarFarField),
        (2, MLFMA.TangentialSphericalFarField),
        (3, MLFMA.SphericalVectorFarField),
        (3, MLFMA.CartesianVectorFarField),
        (6, MLFMA.PairedSphericalVectorFarField),
    )
        @test MLFMA.fielddim(FarFieldType) == D
        @test MLFMA.fielddim(MLFMA.radiationcollectiontype(FarFieldType)) == D
        @test MLFMA.fielddim(MLFMA.receivecollectiontype(FarFieldType)) == D

        ff = FarFieldType(1.0 + 0.0im, sampling, Val(:zero))
        @test typeof(ff) <: FarFieldType
        @test MLFMA.fielddim(ff) == D
        @test length(eachindex(ff)) == D
    end

    # The distinction the previous, count-keyed scheme could not express: no operator maps to
    # Cartesian today, but the type exists, is constructible, and is not the spherical one.
    @test MLFMA.fielddim(MLFMA.SphericalVectorFarField) ==
        MLFMA.fielddim(MLFMA.CartesianVectorFarField)
    @test MLFMA.SphericalVectorFarField !== MLFMA.CartesianVectorFarField
    @test MLFMA.radiationcollectiontype(MLFMA.SphericalVectorFarField) !==
        MLFMA.radiationcollectiontype(MLFMA.CartesianVectorFarField)
end

# Farfield modernization plan §5 (revised): γ(operator) is far-field OPERATOR COUPLING, owned by
# `receivecouplingrule` -- not the H2 map's global/leaf weak-form scaling, owned by
# `H2Factory.α(core)` (applied once per leaf during disaggregation, see
# H2Factory/src/farinteractions/disaggregation.jl). Pin that these stay independent quantities for
# MWDoubleLayer3D, the one operator whose receivecouplingrule actually carries γ: changing
# wavenumber (γ) must not move α, and changing alpha must not move γ. If a future edit ever made
# α depend on γ (or vice versa), that would be exactly the double-counting §5's revision is
# guarding against.
@testset "α(core) and γ(operator) stay independent for MWDoubleLayer3D" begin
    samealpha = Maxwell3D.doublelayer(; wavenumber=1.0, alpha=2.5)
    higherwavenumber = Maxwell3D.doublelayer(; wavenumber=3.7, alpha=2.5)
    higheralpha = Maxwell3D.doublelayer(; wavenumber=1.0, alpha=9.1)

    @test MLFMA.γ(samealpha) != MLFMA.γ(higherwavenumber)
    @test MLFMA.α(samealpha) == MLFMA.α(higherwavenumber)

    @test MLFMA.α(samealpha) != MLFMA.α(higheralpha)
    @test MLFMA.γ(samealpha) == MLFMA.γ(higheralpha)

    # The receive coupling itself is a function of γ only -- α never enters it, so an operator.alpha
    # change cannot silently perturb the far-field component algebra alongside the intended
    # H2Factory.α(core) leaf scaling.
    rep = MLFMA.farfieldtype(samealpha)
    rule_lowalpha = MLFMA.receivecouplingrule(samealpha, rep)
    rule_highalpha = MLFMA.receivecouplingrule(higheralpha, rep)
    @test all(
        MLFMA.coefficient(a) == MLFMA.coefficient(b) for
        (a, b) in zip(rule_lowalpha, rule_highalpha)
    )
end
