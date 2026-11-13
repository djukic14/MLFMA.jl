# TEST (d): A COMPRESSED-LEAF 2-D MATVEC, END TO END, AGAINST DENSE BEAST.
#
# `test/CircularHarmonics/` verifies the representation in isolation -- the transform pair, the
# storage, the six orientations, the receive coupling against the sampled collections. All of that is
# about ONE LEAF. This file is where the compressed leaf error has to compose with interpolation,
# translation, the level phase shifts and the addition theorem and still land inside the tolerance the
# map was asked for.
#
# ------------------------------------------------------------------------------------------------
# WHAT THIS IS ALLOWED TO CONCLUDE ABOUT THE ADJOINT, AND WHAT IT IS NOT.
#
# `test/Maps/helmholtz2d.jl` measured that every 2-D block assembled with the same test and trial
# space is SYMMETRIC to round-off. So on a Galerkin fixture `dense'` is just `conj(dense)`, and an
# adjoint that merely conjugated the forward map would score exactly as well as a correct one. That is
# the #35/#61 failure shape verbatim.
#
# The sharp adjoint check for this representation is therefore NOT here. It is in
# `test/CircularHarmonics/collection.jl`, which pins `conj(Rec) x` and `Rᴴ m` against explicitly
# constructed dense matrices, where a missing or misplaced conjugation cannot hide. What this file
# adds is that those kernels COMPOSE correctly through a whole traversal -- which is a real question
# and a different one.
#
# THE SHARP END-TO-END ADJOINT IS THE PETROV TESTSET BELOW, on two separated circles where
# `‖Z-Zᵀ‖/‖Z‖ = 1.4` and a conjugation-only adjoint cannot pass. BOTH of a Petrov map's sides are
# assembled -- the test tree holds different functions, so there is nothing to mirror -- and each
# chooses its storage independently, which is why that testset runs all FOUR combinations.
#
# It briefly looked like an incompatibility and was briefly guarded against as one. The guard was
# wrong twice over: the Petrov core was handing BOTH sides the same channel, and the compressed
# builders were radiation-only, so `testleafstorage` had nothing to reach. Both are omissions rather
# than design limits, and removing them is what made this file's adjoint claim sharp instead of
# vacuous.
#
# ------------------------------------------------------------------------------------------------
# THE ASSERTIONS ARE AGAINST ε AND AGAINST THE SAMPLED MAP, not against recorded numbers.
#
# The claim worth defending is "compressing the leaves does not spend the map's error budget", so each
# case checks the compressed error against the tolerance AND against what the sampled map achieved on
# the identical fixture. Pinning the measured values instead would fail on any change to the order
# rule that was in fact an improvement.

using MLFMA
using BEAST
using CompScienceMeshes
using H2Factory
using H2Trees
using LinearAlgebra
using StaticArrays
using Test

@testset "Circular-harmonic leaf storage, end to end" begin
    κ = 2π                                    # λ = 1, so the mesh radius is in wavelengths
    X = lagrangec0d1(CompScienceMeshes.meshcircle(1.5, 1 / 6))
    tree = H2Trees.TwoNTree(
        X; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.25, minvalues=0)
    )

    function mapwith(operator, storage, ϵ)
        strategy = MLFMA.MLFMAStrategy(
            operator,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(operator; ϵ=ϵ),
            farfieldstrategy=MLFMA.FarFieldStrategy(; leafstorage=storage),
        )
        return MLFMA.GalerkinMLFMA(operator, X; tree=tree, strategy=strategy, verbose=false)
    end

    # Worst relative error over every unit vector, in both directions. Returns a pair so a caller
    # cannot check one direction and forget the other -- which is how #35 and #61 survived. A single
    # random draw is not enough: it has read as ~6e-4 in this package family on a direction later
    # proven genuinely broken, because averaging hides a bad column.
    function worstcolumns(A, dense)
        forward = 0.0
        adjoint_ = 0.0
        adjointreference = dense'
        for j in axes(dense, 2)
            e = zeros(ComplexF64, size(dense, 2))
            e[j] = 1
            forward = max(forward, norm(A * e - dense * e) / norm(dense * e))
        end
        for j in axes(dense, 1)
            e = zeros(ComplexF64, size(dense, 1))
            e[j] = 1
            adjoint_ = max(
                adjoint_,
                norm(adjoint(A) * e - adjointreference * e) / norm(adjointreference * e),
            )
        end
        return (forward=forward, adjoint=adjoint_)
    end

    # BOTH GALERKIN-CAPABLE 2-D OPERATORS, and the pair is the point rather than redundancy: the
    # single layer is one component and a scalar `α`, the hypersingular is THREE components and a
    # VECTOR `α`. The vector case is what caught a real gap -- the scatter arities were typed
    # `α::Number`, which did not error but silently failed to match, dropping the call through to
    # H2Factory's generic `applyleafmul!` and a `MethodError` on `mul!` many frames away.
    operators = (
        ("single layer", Helmholtz2D.singlelayer(; wavenumber=κ), 1),
        ("hypersingular", Helmholtz2D.hypersingular(; wavenumber=κ), 3),
    )

    @testset "$name: compressed matches dense, forward and adjoint" for (
            name, operator, D
        ) in operators

        dense = BEAST.assemble(operator, X, X)

        for ϵ in (1e-3, 1e-5)
            sampled = worstcolumns(mapwith(operator, MLFMA.SampledLeafStorage(), ϵ), dense)
            compressed = worstcolumns(
                mapwith(operator, MLFMA.CircularHarmonicLeafStorage(), ϵ), dense
            )

            # THE CLAIM: compressing the leaves does not spend the map's error budget.
            @test compressed.forward < ϵ
            @test compressed.adjoint < ϵ

            # NO COMPARISON AGAINST `sampled` AT ALL, and `sampled` is computed above only so this
            # note can be checked against numbers. Non-vacuity -- that the compressed map really is
            # compressed rather than a silent fallback to sampled storage -- is covered by the
            # "really is compressed" testset below, which pins the collection types and that the
            # coefficient count is strictly under the direction count.
            #
            # THIS USED TO BOUND `compressed < 30 * sampled`, AND THAT COMPARISON IS NOT STABLE.
            # The two errors have different sources: the compressed one is set by the harmonic tail
            # truncation, which `CHETAILRATIO` calibrates against ε, while the sampled one is set by
            # the direction count. Rounding `Nφ` up to a multiple of four (so the quarter turn is a
            # grid point) improved the sampled path without touching the compressed one, and the
            # ratio moved for a reason that says nothing about compression quality. Measured at
            # ε = 1e-3, grid 22 -> 24: sampled 1.2e-5 -> 4.22e-6, compressed 3.63e-5 -> 3.48e-5.
            # A ratio between an ε-calibrated error and a grid-calibrated one is not a property
            # worth pinning; the absolute bound above is the claim that matters.
            #
            # Nor is the ORDER of the two stable, which is why this is not merely a looser ratio:
            # measured on the hypersingular at ε = 1e-5, the compressed path is the more accurate of
            # the two (2.110e-6 against 2.195e-6). Truncating the harmonic tail is not obliged to
            # cost accuracy on every operator.
        end
    end

    @testset "the compressed map really is compressed, and really is 2-D" begin
        # Non-vacuity: without this, every assertion above would pass just as well if the strategy
        # had quietly fallen back to sampled storage.
        operator = Helmholtz2D.singlelayer(; wavenumber=κ)
        A = mapwith(operator, MLFMA.CircularHarmonicLeafStorage(), 1e-5)
        collections = H2Factory.core(A).trialfieldcollections
        @test eltype(collections) <: MLFMA.CircularHarmonicRadiationCollection
        @test eltype(H2Factory.core(A).testfieldcollections) <:
            MLFMA.CircularHarmonicReceiveCollection

        # One order per LEVEL, which the batched leaf multiply requires -- a ragged level has no
        # single synthesis matrix. `cheorders` raises on a disagreement, so reaching a value at all
        # is the assertion.
        core = H2Factory.core(A)
        orders = MLFMA.cheorders(
            core.tree, collections, core.samplings, core.minrelevantlevel
        )
        @test !isempty(orders)

        # PER LEVEL, not one order for the map. This fixture has leaves at more than one level, so
        # `orders` genuinely carries several entries -- which is the rule working, not a defect, and
        # is why the check below iterates rather than taking `only`.
        for (sampling, M) in orders
            # Compressed, i.e. strictly fewer coefficients than that grid has directions. If the rule
            # ever returned a grid's full order this would be a representation change with no
            # compression -- legal, but not what the feature is for.
            @test MLFMA.CircularHarmonics.numcoefficients(M) < length(sampling)
            @test M < MLFMA.cutoffparameter(sampling)
        end
    end

    @testset "the hypersingular's VECTOR α reaches the leaf scatter" begin
        # The gap the operator pair above exists to catch, pinned directly rather than only through
        # its end-to-end consequence: `MLFMA.α` is a three-entry vector here, and a scatter arity
        # typed `α::Number` would not match it.
        operator = Helmholtz2D.hypersingular(; wavenumber=κ)
        @test MLFMA.α(operator) isa AbstractVector
        @test length(MLFMA.α(operator)) == 3
        @test MLFMA.fielddim(MLFMA.farfieldtype(operator)) == 3
    end

    @testset "Petrov: every storage combination, on a NON-SYMMETRIC block" begin
        # THE SHARP ADJOINT CHECK. Everything above is Galerkin, where the block is symmetric to
        # round-off and `dense'` is merely `conj(dense)` -- so an adjoint that only conjugated the
        # forward map would score exactly as well as a correct one. Here the two spaces differ, the
        # block is genuinely non-symmetric, and transpose and adjoint are different operators.
        #
        # AND THE PLACE THE FOUR COMBINATIONS ARE EXERCISED. A Petrov map assembles both sides, so
        # `trialleafstorage` and `testleafstorage` are independent; the two MIXED settings are the
        # ones that make the map hold two different leaf scratch types at once.
        trialspace = lagrangec0d1(CompScienceMeshes.meshcircle(1.0, 1 / 8))
        testspace = lagrangec0d1(
            CompScienceMeshes.translate(
                CompScienceMeshes.meshcircle(1.0, 1 / 8), SVector{2,Float64}(5.0, 0.0)
            ),
        )

        function petrovwith(operator, trial, test, ϵ)
            strategy = MLFMA.MLFMAStrategy(
                operator,
                testspace,
                trialspace;
                samplingstrategy=MLFMA.SamplingStrategy(operator; ϵ=ϵ),
                farfieldstrategy=MLFMA.FarFieldStrategy(;
                    trialleafstorage=trial, testleafstorage=test
                ),
            )
            return MLFMA.PetrovGalerkinMLFMA(
                operator, testspace, trialspace; strategy=strategy, verbose=false
            )
        end

        sampled = MLFMA.SampledLeafStorage()
        compressed = MLFMA.CircularHarmonicLeafStorage()

        # THE DOUBLE LAYER IS HERE AND NOT IN THE GALERKIN SETS ABOVE, and it is the operator this
        # feature is most for: it is Petrov-only (#130) precisely because it has no mirrored receive
        # rule, so the GALERKIN compressed receive side cannot exist for it -- while an ASSEMBLED one
        # needs no such rule and works. It is also the operator whose receive side wants a LOWER order
        # than its radiation side (#139's measurement), so a radiation-calibrated `CHETAILRATIO` can
        # only over-predict here.
        petrovoperators = (
            ("single layer", Helmholtz2D.singlelayer(; wavenumber=κ)),
            ("hypersingular", Helmholtz2D.hypersingular(; wavenumber=κ)),
            ("double layer", Helmholtz2D.doublelayer(; wavenumber=κ)),
        )

        @testset "$name" for (name, operator) in petrovoperators
            dense = BEAST.assemble(operator, testspace, trialspace)

            # Non-vacuity for the adjoint claim: if this block were symmetric the testset would prove
            # nothing, exactly as the Galerkin ones cannot.
            @test norm(dense - transpose(dense)) / norm(dense) > 1.0

            for ϵ in (1e-3, 1e-5)
                errors = Dict{Tuple{Bool,Bool},Float64}()
                for (trial, test) in (
                    (sampled, sampled),
                    (compressed, sampled),
                    (sampled, compressed),
                    (compressed, compressed),
                )
                    A = petrovwith(operator, trial, test, ϵ)
                    core = H2Factory.core(A)

                    # Each side carries what it was asked for and only that side. Without this every
                    # accuracy assertion below would pass just as well on a silent fallback -- and a
                    # `testleafstorage` that was quietly ignored would pass them too.
                    if trial === compressed
                        @test eltype(core.trialfieldcollections) <:
                            MLFMA.CircularHarmonicRadiationCollection
                    else
                        @test eltype(core.trialfieldcollections) <:
                            MLFMA.AbstractRadiationFieldCollection
                        @test !(
                            eltype(core.trialfieldcollections) <:
                            MLFMA.CircularHarmonicRadiationCollection
                        )
                    end
                    if test === compressed
                        # ASSEMBLED, not the mirrored Galerkin receive collection: this side stores
                        # its own coefficients and applies neither the antipodal parity nor the
                        # operator's coupling, which is what lets the double layer use it at all.
                        @test eltype(core.testfieldcollections) <:
                            MLFMA.CircularHarmonicAssembledReceiveCollection
                        @test !(
                            eltype(core.testfieldcollections) <:
                            MLFMA.CircularHarmonicReceiveCollection
                        )
                    else
                        @test eltype(core.testfieldcollections) <:
                            MLFMA.AbstractReceiveFieldCollection
                        @test !(
                            eltype(core.testfieldcollections) <:
                            MLFMA.CircularHarmonicAssembledReceiveCollection
                        )
                    end

                    e = worstcolumns(A, dense)
                    @test e.forward < ϵ
                    @test e.adjoint < ϵ
                    errors[(trial === compressed, test === compressed)] = e.forward
                end

                # THE TWO ERRORS DO NOT ADD. The prediction on record was the conservative "two sides
                # each under ε compose to under 2ε"; measured, compressing both lands at or below the
                # QUADRATURE of compressing each alone, because the two truncations act on different
                # patterns and their errors are uncorrelated. Held as an assertion rather than a note,
                # since this task's parent had five predictions refuted by measurement.
                #
                # Against the SAMPLED baseline subtracted out, so what is being composed is the two
                # compression errors rather than the map's own.
                base = errors[(false, false)]
                a = max(errors[(true, false)] - base, 0.0)
                b = max(errors[(false, true)] - base, 0.0)
                @test errors[(true, true)] - base <= 1.3 * sqrt(a^2 + b^2) + 1e-12
            end
        end
    end
end
