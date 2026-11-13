using MLFMA
using BEAST, CompScienceMeshes
using StaticArrays
using Test
using LinearAlgebra
using H2FactoryTestUtils
using H2Factory
using H2Trees

include("../support/util_test.jl")

# Galerkin vs Petrov-Galerkin, and `AggregateMode` vs `AggregateTranslateMode`, against a dense far
# reference. This is the only place the Petrov map is checked numerically, and the only place the two
# aggregation modes are compared, so those four combinations are the point of the file.
#
# FIXTURE COST. This used to be `meshspherewithcenter(2λ, λ/10)` (18849 functions, a 5.3 GB dense
# matrix) assembled at `DoubleNumWiltonSauterQStrat(7, ...)`, and took ~58 of the suite's ~71 minutes
# -- roughly 80% of total test time in one file. Both halves of that were unnecessary:
#
#   - The near assembler is `DummyNearInteractionsAssembler` and the comparison runs against
#     `farinteractions`, so every near/singular block the dense assembly computes is thrown away
#     unused. Those are exactly the expensive ones (Sauter-Schwab on touching elements).
#   - Order 7 vs order 3 on the *far* rule changes the measured discrepancy from 4.2e-5 to 2.8e-5
#     (on a random draw; 4.58e-5 on the deterministic input used below),
#     both far below the multipole error actually being bounded. Measured: 87.3 s vs 14.4 s for the
#     dense assembly alone. What is under test is MLFMA's far-field machinery, not BEAST's quadrature
#     accuracy, so the reference only has to be assembled the same way on both sides -- the same
#     reasoning as the Phase 5 testset in `galerkin.jl`.
#
# At 1λ / λ8 the tree still has 3 levels, 50 leaves and 1878 translating pairs, so nothing about the
# far path goes untested. Do not shrink further without re-checking those: `minhalfsize = λ/4` on a
# smaller sphere runs into the empty-far-plan hazard.
#
# The `@show`n error diagnostics this file used to print are now real assertions. They were computed
# but never tested, which is how the wrong Petrov adjoint below stayed invisible.
@testset "Galerkin and Petrov agree against a dense far reference" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    m = meshspherewithcenter([0.0, 0.0, 0.0], 1.0 * λ, λ / 8)
    Y = raviartthomas(m)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    # Matched, deliberately cheap quadrature on both sides -- see the fixture note above.
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)

    # Only the NEAR block is stubbed -- `stubbedstrategy` is not used here because this file needs the
    # real I2O translation and the real far-field quadrature: it compares against a dense reference, and
    # the far path is the thing under test. The dense side is `farinteractions`, so a dummy near block is
    # what makes the two comparable.
    function strategyfor(aggregationmode, spaces...)
        return MLFMA.MLFMAStrategy(
            efie,
            spaces...;
            treestrategy=MLFMA.TreeStrategy(efie; minhalfsize=λ / 4),
            nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
            ),
            nearfieldstrategy=H2Factory.InteractionStrategy(;
                isnear=H2Trees.isnear(; additionalbufferboxes=1),
                aggregationmode=aggregationmode,
            ),
        )
    end

    petrovaggregate = MLFMA.assemble(
        efie, Y, Y; strategy=strategyfor(H2Trees.AggregateMode(), Y, Y), verbose=false
    )
    petrovaggregatetranslate = MLFMA.assemble(
        efie,
        Y,
        Y;
        strategy=strategyfor(H2Trees.AggregateTranslateMode(), Y, Y),
        verbose=false,
    )
    galerkinaggregatetranslate = MLFMA.assemble(
        efie, Y; strategy=strategyfor(H2Trees.AggregateTranslateMode(), Y), verbose=false
    )
    galerkinaggregate = MLFMA.assemble(
        efie, Y; strategy=strategyfor(H2Trees.AggregateMode(), Y), verbose=false
    )

    @test length(H2Factory.trialmoments(galerkinaggregatetranslate)) ==
        length(H2Factory.trialmoments(petrovaggregate))

    # --- non-vacuity: the fixture must actually produce far interactions to compare ---------------
    # The Galerkin tree is a plain `TwoNTree`; the Petrov map holds a `BlockTree`, which has no
    # `levels`/`DepthFirstIterator` of its own, so introspection goes through this one.
    tree = H2Trees.tree(galerkinaggregatetranslate)
    @test H2Trees.numberoflevels(tree) > 2
    farpairs = sum(
        length(collect(H2Trees.TranslatingNodesIterator(tree, node))) for
        node in H2Trees.DepthFirstIterator(tree)
    )
    @test farpairs > 0

    # Deterministic input, built without any RNG. `randn` draws from the GLOBAL RNG, and this file
    # runs after other random-using tests, so an unrelated future test change upstream would shift
    # the draw. That matters here specifically because `adjointerror` below is asserted against a
    # measured magnitude rather than a tolerance: a moving input would make that pin flaky. Varied
    # magnitude and irrational-ish phase step, so this is not accidentally a special vector.
    x = ComplexF64[cis(0.7 * i) * (1 + 0.5 * cospi(i / 17)) for i in 1:numfunctions(Y)]
    # non-vacuity: a genuinely complex input with varying magnitude, not a constant or a real vector
    @test maximum(abs, imag.(x)) > 0.1
    @test maximum(abs, x) / minimum(abs, x) > 1.5

    # --- the two map types aggregate to identical moments -----------------------------------------
    H2Factory.aggregate!(petrovaggregate, x)
    H2Factory.aggregate!(galerkinaggregatetranslate, x)

    momentscale = 0.0
    for box in keys(H2Factory.trialmomentcollections(galerkinaggregatetranslate))
        galerkinmoments = H2Factory.trialmomentcollections(galerkinaggregatetranslate)[box]
        petrovmoments = H2Factory.trialmomentcollections(petrovaggregate)[box]
        @test galerkinmoments[1] ≈ petrovmoments[1]
        @test galerkinmoments[2] ≈ petrovmoments[2]
        momentscale = max(momentscale, maximum(abs, galerkinmoments[1]))
    end
    # non-vacuity: the moments compared above must not be all zero
    @test momentscale > 0

    # --- dense far reference ----------------------------------------------------------------------
    A = BEAST.assemble(efie, Y, Y; quadstrat=quadstrat)
    farmatrix = H2Factory.farinteractions(A, petrovaggregate)

    reference = farmatrix * x
    relerror(y) = maximum(abs.(y .- reference)) / maximum(abs.(reference))

    # Bounds are the measured errors (all 4.58e-5 on this input) with headroom. The two aggregation modes are
    # different traversals of the same mathematics, so each is checked against the dense reference
    # rather than only against the other.
    @test relerror(petrovaggregate * x) < 1e-4
    @test relerror(petrovaggregatetranslate * x) < 1e-4
    @test relerror(galerkinaggregate * x) < 1e-4
    @test relerror(galerkinaggregatetranslate * x) < 1e-4

    # Galerkin and Petrov agree to round-off, far tighter than either agrees with the reference:
    # they share the far-field machinery, so a discrepancy here is a bookkeeping difference between
    # the two constructors rather than an approximation error. Measured 2.3e-15.
    @test maximum(abs.(galerkinaggregatetranslate * x .- petrovaggregate * x)) /
          maximum(abs.(reference)) < 1e-12

    # --- reversed directions, Petrov --------------------------------------------------------------
    # The EFIE matrix is symmetric, so `transpose(farmatrix) == farmatrix` and this check cannot
    # distinguish a correctly-transposed matvec from one that silently ran forward. What it does pin
    # is that the transposed Petrov path runs and stays accurate -- which is more than the adjoint
    # below manages. A non-symmetric fixture (distinct test/trial spaces) would give it teeth and
    # belongs with Phase 6's Petrov rework.
    @test maximum(abs.(transpose(farmatrix) .- farmatrix)) / maximum(abs.(farmatrix)) <
        1e-12
    transposereference = transpose(farmatrix) * x
    @test maximum(abs.(transpose(petrovaggregate) * x .- transposereference)) /
          maximum(abs.(transposereference)) < 1e-4

    # The Petrov adjoint, which this fixture pinned as a KNOWN BUG at relative error 1.22 (O(1), not
    # a tolerance issue, and returned silently) until the conjugated leaf multiply landed --
    # `src/assembly/cores/adjointleafmul.jl`, which supplies the `α` and both-leaf-collection conjugations
    # H2Factory's orientation-invariant defaults do not. Measured 3.9e-5 now, the same order as the
    # transpose bound right above, so it takes the same 1e-4.
    adjointreference = adjoint(farmatrix) * x
    adjointerror =
        maximum(abs.(adjoint(petrovaggregate) * x .- adjointreference)) /
        maximum(abs.(adjointreference))
    @test adjointerror < 1e-4
    # non-vacuity for the whole adjoint discussion: the reference must genuinely be complex, or
    # conjugation would be a no-op and none of the above would mean anything
    @test maximum(abs.(conj(farmatrix) * x .- reference)) / maximum(abs.(reference)) > 0.1

    # --- the full matvec list, forward ------------------------------------------------------------
    # `nearmul=false`: the near assembler is a dummy, so there is no near block to compare -- the
    # dense side is `farinteractions` and the map is far-only, which is what makes them comparable.
    #
    # `ϵ = 1e-3` rather than the 1e-4 used for the explicit bounds above: `checkinteractions` draws
    # its OWN random vectors from the global RNG, so a 2.4x margin over the measured 4.2e-5 would be
    # thin across draws. What it uniquely covers is plumbing -- sizes, that the near/far partition is
    # complete, and the 5-argument α/β forms -- none of which needs a tight tolerance. The tight,
    # draw-independent signal lives in the explicit assertions on the deterministic `x`. Same
    # reasoning as the Phase 5 testset in `galerkin.jl`.
    @test checkinteractions(
        farmatrix, galerkinaggregatetranslate; n=3, ϵ=1e-3, nearmul=false, verbose=false
    )

    # Both reverse directions of the GALERKIN map, whose test side is a mirrored
    # `LazyReceiveFieldCollection` rather than an assembled one -- so this is the fixture that
    # exercises the lazy collection's conjugated aggregation arity, which the Petrov map above never
    # reaches. It carries the history of two separate fixes: the write-through `mul!` (previously
    # both directions THREW), and then the conjugated leaf multiply (the adjoint then returned a
    # silently wrong answer, O(1), while the transpose was already correct -- transposing needs no
    # conjugation, so it was never going to expose a conjugation bug).
    @test checkinteractions(
        transpose(farmatrix),
        transpose(galerkinaggregatetranslate);
        n=3,
        ϵ=1e-3,
        nearmul=false,
        verbose=false,
    )

    @test checkinteractions(
        adjoint(farmatrix),
        adjoint(galerkinaggregatetranslate);
        n=3,
        ϵ=1e-3,
        nearmul=false,
        verbose=false,
    )

    # The deterministic-input version of the same claim, kept alongside `checkinteractions`'s own
    # random sweep because it is the figure this file's history is written in: 1.22 while broken,
    # 3.9e-5 now. `checkinteractions` draws from the global RNG, so it cannot pin a number.
    adjointreference2 = farmatrix' * x
    adjointerror2 =
        maximum(abs.(galerkinaggregatetranslate' * x .- adjointreference2)) /
        maximum(abs.(adjointreference2))
    @test adjointerror2 < 1e-3
end
