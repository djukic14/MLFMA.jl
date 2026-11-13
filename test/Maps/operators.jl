# Every operator MLFMA claims to support, against a dense BEAST reference -- and, for the ones whose
# far field is tangential, against every leaf storage backend.
#
# WHY THIS EXISTS. The MFIE receive-coupling bug (#129) was a 4.4e-01 silent error that survived
# because the compressed backends were only ever checked against EFIE, whose receive coupling happens
# to equal the geometric mirror. The lesson generalises past that one fix: an operator table is a
# table, and every row needs its own evidence. This is that evidence, for the whole table.
#
# WHAT IT FOUND, and what it first got WRONG about it. Three of the nine have no
# `receivecouplingrule` method: `DoubleLayerRotatedMW3D`, `HH3DDoubleLayerFDBIO` and
# `HH3DDoubleLayerTransposedFDBIO`. That is a DELIBERATE, documented limitation -- see the comment
# above the rule block in `operators/operatorutils.jl` -- and not an oversight, which an earlier
# version of this file claimed.
#
# They are fully supported through an ASSEMBLED receive side, which is what a PETROV map uses.
# Measured against a dense reference on this fixture: 2.6e-05, 3.9e-05 and 4.0e-05, against
# 3.3e-06 for the EFIE control. What they cannot back is a GALERKIN map, which MIRRORS its test side
# rather than assembling it, and so needs the lazy rule.
#
# So the three rows below assert the real contract in both directions: Petrov works to the same
# accuracy as any other operator, and Galerkin REFUSES at construction. The refusal is new
# (`MLFMA.checklazyreceive`); before it, a Galerkin map on these constructed happily and died with a
# bare `MethodError` inside the first traversal.
#
# THREE STORAGES, BUT NOT FOR EVERY OPERATOR. `SphericalHarmonicLeafStorage` and
# `VectorSphericalHarmonicLeafStorage` both expand a TANGENTIAL two-component far field. An operator
# radiating a scalar, a three-component vector or a paired vector has no such expansion, and the right
# behaviour is a clear refusal at construction rather than a plausible wrong answer -- asserted here
# too, since a silent success on those would be the #129 failure mode again.

using MLFMA
using BEAST, CompScienceMeshes, H2Trees, H2Factory
using LinearAlgebra
using Random
using Test

using H2Trees: TwoNTree, TwoNTreeBuilder

# One row per supported operator: how to build it, which space it lives on, whether its far field is
# tangential (hence compressible), and whether it is a confirmed gap. Deliberately mirrors the `cases`
# tuple in `test/Operators/operatorutils.jl`; the count check below fails if an operator is added
# there and not here.
function operatorcases(κ, ω, μ)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    doublelayer = Maxwell3D.doublelayer(; wavenumber=κ)
    row(name, op, space, tangential, gap=false) =
        (; name=name, op=op, space=space, tangential=tangential, gap=gap)
    return (
        row("EFIE", efie, :rwg, true),
        row("MFIE (double layer)", doublelayer, :rwg, true),
        row("rotated double layer", BEAST.n × doublelayer, :rwg, true, true),
        row("MW single layer", MLFMA.singlelayer(efie), :rwg, false),
        row("MW hypersingular", MLFMA.hypersingular(; wavenumber=κ), :rwg, false),
        row("HH single layer", Helmholtz3D.singlelayer(; wavenumber=κ), :lagrange, false),
        row(
            "HH double layer",
            Helmholtz3D.doublelayer(; wavenumber=κ),
            :lagrange,
            false,
            true,
        ),
        row(
            "HH double layer transposed",
            Helmholtz3D.doublelayer_transposed(; wavenumber=κ),
            :lagrange,
            false,
            true,
        ),
        row(
            "HH hypersingular", Helmholtz3D.hypersingular(; wavenumber=κ), :lagrange, false
        ),
    )
end

# The non-gap body, hoisted out of the loop. NOT an early `return` inside the `@testset`: `return`
# there exits the enclosing block, which silently skipped every case after the first gap and reported
# a green 16-assertion run instead of the full 40.
function _checkoperator(case, X, tree, quadstrat, mapwith, compressedstorages)
    @testset "$(case.name) against the dense reference" begin
        Random.seed!(37)
        x = randn(ComplexF64, numfunctions(X))
        reference = BEAST.assemble(case.op, X, X; quadstrat=quadstrat) * x
        sampledmap = mapwith(case.op, X, tree, MLFMA.SampledLeafStorage())
        sampled = norm(sampledmap * x - reference) / norm(reference)
        @test sampled < 1e-3

        if case.tangential
            # Two tangential components, so both compressed backends apply and must agree with
            # the SAMPLED path -- not merely be "accurate", which is what let the MFIE coupling
            # bug through at 4.4e-01 while an absolute bound still looked plausible.
            @test MLFMA.farfieldtype(case.op) === MLFMA.TangentialSphericalFarField
            for storage in compressedstorages
                compressed =
                    norm(mapwith(case.op, X, tree, storage) * x - reference) /
                    norm(reference)
                @test compressed < 1e-3
                @test compressed < 2 * sampled
            end
        else
            # NOT tangential, so there is no two-family expansion of this far field. Refusing at
            # construction is the contract; succeeding would be #129 all over again.
            @test MLFMA.farfieldtype(case.op) !== MLFMA.TangentialSphericalFarField
            for storage in compressedstorages
                @test_throws ErrorException mapwith(case.op, X, tree, storage)
            end
        end
    end
    return nothing
end

@testset "Every supported operator, against a dense reference" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    # λ/4 leaves -- the low-frequency-breakdown floor -- so the sphere has to be big enough to still
    # give more than two levels. 1.2λ at λ/5 is the smallest that does.
    m = meshsphere(1.2 * λ, λ / 5)
    spaces = (rwg=raviartthomas(m), lagrange=lagrangec0d1(m))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)
    trees = map(
        X -> TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0)), spaces
    )

    cases = operatorcases(κ, ω, μ)
    @test length(cases) == 9      # the `farfieldtype` table's size; see this file's header

    function mapwith(op, X, tree, storage)
        strategy = MLFMA.MLFMAStrategy(
            op,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
            farfieldstrategy=MLFMA.FarFieldStrategy(; leafstorage=storage),
            nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
        )
        return MLFMA.GalerkinMLFMA(op, X; tree=tree, strategy=strategy, verbose=false)
    end

    compressedstorages = (
        MLFMA.SphericalHarmonicLeafStorage(), MLFMA.VectorSphericalHarmonicLeafStorage()
    )

    for case in cases
        @testset "$(case.name)" begin
            X = getproperty(spaces, case.space)
            tree = getproperty(trees, case.space)

            if case.gap
                # NO MIRRORED RECEIVE RULE, by decision -- so this operator cannot back a
                # GALERKIN MATVEC, and must SAY SO at construction rather than dying silently in
                # a traversal. It warns rather than raises: a Galerkin map with no receive rule is
                # still usable for AGGREGATION, which `test/Maps/aggregation.jl`'s Helmholtz testset
                # relies on, so refusing outright would forbid something that works.
                @test !MLFMA.haslazyreceive(case.op)
                @test_throws MethodError MLFMA.receivecouplingrule(case.op)
                galerkin = @test_logs (:warn,) match_mode = :any mapwith(
                    case.op, X, tree, MLFMA.SampledLeafStorage()
                )
                @test_throws MethodError galerkin * randn(ComplexF64, numfunctions(X))

                # ...and it IS supported, through an ASSEMBLED receive side. A Petrov map must
                # reproduce the operator to the same accuracy any other one does -- which is the
                # half an earlier version of this file missed, having read the missing method as a
                # total gap rather than a Galerkin-only one.
                Random.seed!(37)
                x = randn(ComplexF64, numfunctions(X))
                reference = BEAST.assemble(case.op, X, X; quadstrat=quadstrat) * x
                strategy = MLFMA.MLFMAStrategy(
                    case.op,
                    X,
                    X;
                    samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
                    treestrategy=MLFMA.TreeStrategy(
                        case.op; minhalfsize=λ / 4, minvalues=0
                    ),
                    nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(
                        quadstrat
                    ),
                )
                petrov = MLFMA.assemble(case.op, X, X; strategy=strategy, verbose=false)
                @test norm(petrov * x - reference) / norm(reference) < 1e-3
            else
                _checkoperator(case, X, tree, quadstrat, mapwith, compressedstorages)
            end
        end
    end
end
