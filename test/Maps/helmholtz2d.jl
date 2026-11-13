# THE 2-D MATVEC, END TO END, AGAINST A DENSE BEAST REFERENCE.
#
# Everything below this file was verified in pieces -- the circular grid, the 2-D translation
# operator, the leaf far fields, the operator mappings, the φ interpolator and its adjoint -- and each
# piece was checked against its own oracle. This is where the pieces have to agree with BEAST about a
# whole operator, which is the only claim a user cares about.
#
# WHAT MAKES A 2-D MATVEC A DIFFERENT TEST FROM THE 3-D ONES, rather than the same test on a smaller
# fixture. Five hardcoded 3-isms had to be removed to get here, none of which any 3-D fixture could
# see: `diameter`'s `√3`, `BufferFunctor`'s triangle, `farfieldkernelvals`' unrolled 3-component dot,
# `NEARINTERACTIONSQUADSTRATEGYRECOM`, and `translationcollection.jl`'s `SVector{3}`. A sixth
# (`α`) was simply missing. They were found one at a time by RUNNING this, never by reading, and each
# failed somewhere that named neither the dimension nor MLFMA -- one of them several frames inside
# BEAST. So this file is the regression test for a class of fault, not for one bug.
#
# ---------------------------------------------------------------------------------------------
# THE ADJOINT NEEDS A NON-SYMMETRIC BLOCK, AND THE OBVIOUS FIXTURE DOES NOT HAVE ONE.
#
# Measured: every 2-D block assembled with the SAME test and trial space is symmetric to round-off --
# `‖Z - Zᵀ‖/‖Z‖` is 5.0e-16 for the single layer and 1.7e-15 for both double layers. So `dense'` is
# just `conj(dense)` there, and an "adjoint" map that merely conjugated the forward one would score
# EXACTLY as well as a correct one. That is the #35/#61 failure shape verbatim, and it is why the
# Galerkin testset below is not allowed to be the whole adjoint story.
#
# The sharp test is the Petrov testset: two separated circles, so test and trial spaces differ and
# `‖Z - Zᵀ‖/‖Z‖ ≈ 1.4`. There the transpose and the adjoint are genuinely different operators and a
# conjugation-only adjoint cannot pass.
#
# ---------------------------------------------------------------------------------------------
# WHY AN ε SWEEP RATHER THAN ONE TOLERANCE. A single relative error below a threshold says the map is
# close on one fixture; an error that TRACKS ε says the multipole machinery is what is producing the
# accuracy. Measured on the single layer, worst column over every unit vector:
#
#     ε      levels 3:3    levels 3:4
#     1e-3   1.20e-5       4.77e-6
#     1e-5   2.57e-7       1.43e-7
#     1e-7   6.13e-9       1.48e-8
#
# Two decades of ε buys about two decades of accuracy. The assertions bound each row generously
# rather than pinning these numbers, so the SHAPE is what is under test.
#
# WORST COLUMN, NOT A RANDOM DRAW. A single random vector has read as ~6e-4 in this package family on
# a direction later proven genuinely broken, because averaging hides a bad column. Every unit vector
# is cheap at this size.
#
# FIXTURE COST: 2m44s, about 7 % of the suite, for ~1000 matvecs across nine maps. Most of it is the ε
# sweep, which is the file's main claim and not trimmable without giving that up. If it ever needs to
# shrink, the honest cut is the ε = 1e-7 row (the 1e-3 → 1e-5 pair already shows the trend), NOT the
# Petrov testset -- that one is small and is the only sharp adjoint check in 2-D.

using MLFMA
using BEAST
using CompScienceMeshes
using H2Factory
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test

@testset "Helmholtz2D end to end" begin
    κ = 2π                                    # λ = 1, so the mesh radius is in wavelengths
    mesh = CompScienceMeshes.meshcircle(1.5, 1 / 6)
    X = lagrangec0d1(mesh)

    # Worst relative error over every unit vector, in both directions. Returns a pair so a caller
    # cannot check one direction and forget the other -- which is exactly how #35 and #61 survived.
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

    @testset "the same-space blocks really are symmetric" begin
        # Stated as a measurement because the Galerkin adjoint assertions below are only as strong as
        # this is weak. If a future 2-D operator's block stops being symmetric, this fails and the
        # note at the top of the file stops being true -- which is the point of pinning it.
        for operator in (
            Helmholtz2D.singlelayer(; wavenumber=κ),
            Helmholtz2D.doublelayer(; wavenumber=κ),
            Helmholtz2D.doublelayer_transposed(; wavenumber=κ),
        )
            Z = BEAST.assemble(operator, X, X)
            @test norm(Z - transpose(Z)) / norm(Z) < 1e-12
            # And NOT Hermitian, so `adjoint` is not `transpose` either -- otherwise the Petrov
            # testset's distinction would be vacuous too.
            @test norm(Z - Z') / norm(Z) > 1.0
        end
    end

    @testset "Galerkin single layer, and the error tracks ε" begin
        operator = Helmholtz2D.singlelayer(; wavenumber=κ)
        dense = BEAST.assemble(operator, X, X)

        # TWO FIXTURES, and the difference between them is the whole reason the second exists:
        #   0.375 -> levels [1,2,3], relevantlevels 3:3, so NO level transfer happens.
        #   0.25  -> levels [1,2,3,4], relevantlevels 3:4, so exactly ONE level transfer does, which
        #            is the only place the circular interpolator and anterpolator run inside a matvec.
        # The `relevantlevels` assertions are not decoration: if a tree or plan change silently
        # collapsed the second fixture to one relevant level, both would still pass the error bounds
        # while the interpolator went untested.
        fixtures = (
            (
                minhalfsize=0.375,
                relevantlevels=3:3,
                bounds=(1e-3 => 1e-4, 1e-5 => 1e-6, 1e-7 => 1e-7),
            ),
            (
                minhalfsize=0.25,
                relevantlevels=3:4,
                bounds=(1e-3 => 1e-4, 1e-5 => 1e-6, 1e-7 => 1e-7),
            ),
        )

        for fixture in fixtures
            tree = H2Trees.TwoNTree(
                X;
                builder=H2Trees.TwoNTreeBuilder(;
                    minhalfsize=fixture.minhalfsize, minvalues=0
                ),
            )
            plans = H2Factory.assembleplans(
                tree,
                H2Factory.InteractionStrategy(;
                    isnear=H2Trees.isnear(; additionalbufferboxes=1)
                ),
            )
            @test plans.relevantlevels == fixture.relevantlevels

            for (ϵ, bound) in fixture.bounds
                strategy = MLFMA.MLFMAStrategy(
                    operator, X; samplingstrategy=MLFMA.SamplingStrategy(operator; ϵ=ϵ)
                )
                A = MLFMA.GalerkinMLFMA(
                    operator, X; tree=tree, strategy=strategy, verbose=false
                )
                worst = worstcolumns(A, dense)
                @test worst.forward < bound
                @test worst.adjoint < bound
                # Non-vacuity: a map that returned zeros would have relative error 1, and a map that
                # was exact to round-off would mean the reference had become the map. Neither is what
                # a multipole approximation does.
                @test worst.forward > 1e-16
            end
        end
    end

    @testset "the level transfer is what the second fixture adds" begin
        # The circular interpolator runs only when there is more than one relevant level, so this
        # pins that the interpolator/anterpolator built for a 2-D map are the circular ones and that
        # a transfer is actually scheduled. Without it, "the 2-D matvec works" could be true of a
        # single-level FMM and say nothing about #133's product.
        operator = Helmholtz2D.singlelayer(; wavenumber=κ)
        tree = H2Trees.TwoNTree(
            X; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.25, minvalues=0)
        )
        A = MLFMA.GalerkinMLFMA(operator, X; tree=tree, verbose=false)

        @test MLFMA.interpolator(A) isa MLFMA.CircularInterpolator
        @test MLFMA.anterpolator(A) isa MLFMA.CircularAnterpolator
        # The trait the level transfers carry, which the map constructor checks the interpolator
        # against. A mismatch here is an `error()` at construction, so reaching this line already
        # proves agreement -- asserted anyway so the intended value is written down.
        @test MLFMA.interpolationtrait(MLFMA.interpolator(A)) ===
            MLFMA.PeriodicφInterpolation()
    end

    @testset "α reaches the block" begin
        # `α(op) = op.alpha` was inferred from BEAST giving the 2-D operators the same `(alpha, gamma)`
        # field pair as the 3-D ones. With the default `alpha = 1` that claim is UNFALSIFIABLE -- the
        # factorization agrees whether or not α is applied -- so it is checked at values where the two
        # possibilities differ, including a complex one.
        tree = H2Trees.TwoNTree(
            X; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.25, minvalues=0)
        )
        Random.seed!(5)
        x = randn(ComplexF64, numfunctions(X))

        for alpha in (1.0, 2.5, -0.75 + 0.5im)
            operator = BEAST.HH2DSingleLayerFDBIO(alpha, im * κ)
            @test MLFMA.α(operator) == alpha
            dense = BEAST.assemble(operator, X, X)
            A = MLFMA.GalerkinMLFMA(operator, X; tree=tree, verbose=false)
            @test norm(A * x - dense * x) / norm(dense * x) < 1e-4
        end
    end

    @testset "the double layers refuse a Galerkin map, by design" begin
        # #130 in 2-D: a mirrored receive rule is a physics claim per operator. The double layers
        # carry `k̂·n̂`, which flips under the antipodal map, so they have no `receivecouplingrule` and
        # a Galerkin map on them must FAIL rather than run on a silently wrong receive side. They are
        # supported through the assembled receive side -- see the Petrov testset below.
        tree = H2Trees.TwoNTree(
            X; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.25, minvalues=0)
        )
        for operator in (
            Helmholtz2D.doublelayer(; wavenumber=κ),
            Helmholtz2D.doublelayer_transposed(; wavenumber=κ),
        )
            @test !MLFMA.haslazyreceive(operator)
            A = MLFMA.GalerkinMLFMA(operator, X; tree=tree, verbose=false)
            # Construction succeeds and the MATVEC is what fails -- worth pinning, because it means
            # the warning MLFMA emits at construction is the user's only early signal.
            @test_throws MethodError A * ones(ComplexF64, numfunctions(X))
        end
    end

    @testset "Petrov, all three operators, on a genuinely non-symmetric block" begin
        # THE SHARP ADJOINT TEST. Two separated circles, so test and trial spaces differ and the block
        # is far from symmetric -- measured `‖Z - Zᵀ‖/‖Z‖ ≈ 1.4` -- which is what makes `adjoint`
        # distinguishable from `conj ∘ forward`. Also the only 2-D exercise of the Petrov constructor
        # and of the assembled (non-mirrored) receive side.
        trialspace = lagrangec0d1(CompScienceMeshes.meshcircle(1.0, 1 / 8))
        testspace = lagrangec0d1(
            CompScienceMeshes.translate(
                CompScienceMeshes.meshcircle(1.0, 1 / 8), SVector{2,Float64}(5.0, 0.0)
            ),
        )

        for operator in (
            Helmholtz2D.singlelayer(; wavenumber=κ),
            Helmholtz2D.doublelayer(; wavenumber=κ),
            Helmholtz2D.doublelayer_transposed(; wavenumber=κ),
        )
            dense = BEAST.assemble(operator, testspace, trialspace)
            # Non-vacuity for the adjoint claim: if this block were symmetric, everything below would
            # pass for a map that only conjugates.
            @test norm(dense - transpose(dense)) / norm(dense) > 1.0

            A = MLFMA.PetrovGalerkinMLFMA(operator, testspace, trialspace; verbose=false)
            worst = worstcolumns(A, dense)
            @test worst.forward < 1e-7
            @test worst.adjoint < 1e-7
            @test worst.forward > 1e-16
        end
    end
end
