using MLFMA
using BEAST, CompScienceMeshes
using StaticArrays
using ClusterTrees
using Test
using LinearAlgebra
using PlotlyJS
using H2Factory
using SparseArrays
using H2FactoryTestUtils
using OhMyThreads
using Random
using H2Trees

include("../support/util_test.jl")
@testset "MLFMAMatrix utils" begin
    κ = 2 * pi
    ϵ = 1e-5

    m = CompScienceMeshes.read_gmsh_mesh(
        joinpath(pkgdir(MLFMA), "test", "assets", "tworects.msh")
    )
    X = raviartthomas(m)
    efie = MLFMA.efie(; wavenumber=κ)

    points = BEAST.positions(X)
    root_center, root_size = H2Trees.boundingbox(points)

    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=root_size / 5, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
        ),
    )

    A_MLFMA = MLFMA.GalerkinMLFMA(
        efie,
        X;
        tree=tree,
        strategy=stubbedstrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
            farfieldstrategy=MLFMA.FarFieldStrategy(;
                assemblerbuilder=MLFMA.DummyFarFieldAssemblerBuilder()
            ),
            nearfieldstrategy=H2Factory.InteractionStrategy(;
                isnear=H2Trees.isnear(; additionalbufferboxes=1)
            ),
        ),
    )

    @test size(A_MLFMA) == (numfunctions(X), numfunctions(X))
    @test size(A_MLFMA) == size(MLFMA.nearinteractions(A_MLFMA))
    @test size(A_MLFMA, 1) == numfunctions(X)
    @test size(A_MLFMA, 2) == numfunctions(X)
    @test MLFMA.γ(A_MLFMA) == im * κ
    @test MLFMA.ϵ(A_MLFMA) == ϵ
    @test H2Trees.tree(A_MLFMA) == tree
    @test H2Factory.α(A_MLFMA) == MLFMA.α(efie)
end

# Regression pin: a tree built with `minlevel > 1` has level *indices* that run past its level
# *count* (e.g. `levels(tree) == 2:6` while `numberoflevels(tree) == 5`). Sizing the per-level
# scheduling containers from the count leaves the deepest level with no scheduler entry, which
# stays invisible until the traversal actually reaches that level and `outerscheduler` throws a
# `KeyError`. `PetrovGalerkinMLFMA` already sizes from the raw top level index for this reason.
@testset "GalerkinMLFMA scheduling covers trees with minlevel > 1" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    ϵ = 1e-3

    m = meshsphere(1 * λ, λ / 10)
    X = raviartthomas(m)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=λ / 10,
            minlevel=2,
            minvalues=0,
            protrusion=H2Trees.NoProtrusionCheck(),
        ),
    )

    # non-vacuity: this fixture only exercises the bug if the indices really do exceed the count
    @test H2Trees.levels(tree)[end] > H2Trees.numberoflevels(tree)

    # near interactions and translation are stubbed out for speed -- this pins the scheduling container
    # sizing, which the aggregation traversal exercises regardless. The far-field ASSEMBLER stays real
    # (only its quadrature is cheapened), since the dummy one does not produce collections sized for the
    # real per-level sampling.
    A = MLFMA.GalerkinMLFMA(
        efie,
        X;
        tree=tree,
        strategy=stubbedstrategy(efie, X; samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ)),
    )

    x = ones(numfunctions(X))
    @test (H2Factory.aggregate!(A, x); true) # must not throw on the deepest level
end

# This fixed mesh gives the default builder leaves above the aggregation range.
# Their sampling and collection lookup must still work, without indexing level zero.
@testset "Default convenience constructor handles its own non-uniform tree" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    m = fixturemesh("default-nonuniform")
    X = raviartthomas(m)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    # no `tree`, so this goes through the default builder including `MINVALUESRECOM`
    A = MLFMA.GalerkinMLFMA(efie, X; strategy=stubbedstrategy(efie, X))

    leaflevels = sort(
        unique(
            H2Trees.level.(Ref(H2Trees.tree(A)), collect(H2Trees.leaves(H2Trees.tree(A))))
        ),
    )

    # non-vacuity: this only guards the gap while the default really does produce one
    @test length(leaflevels) > 1
    @test leaflevels[begin] < H2Factory.minrelevantlevel(A)

    # both indexing sites are total over the leaves
    @test length(H2Factory.trialmomentcollections(A)) ==
        length(H2Trees.leaves(H2Trees.tree(A)))
    @test length(H2Factory.testmomentcollections(A)) ==
        length(H2Trees.leaves(H2Trees.tree(A)))

    x = ones(numfunctions(X))
    @test (H2Factory.aggregate!(A, x); true)
end

# Phase 5: the Galerkin map against a dense BEAST reference.
#
# Fixture chosen by measurement, not by taste. `meshsphere(0.8λ, λ/6)` at `minhalfsize = λ/12` is the
# smallest sphere tried that still yields a non-empty far plan -- 0.5λ at λ/5 and λ/6 both fail with
# `ArgumentError: Empty AggregatePlan not supported`, which is exactly the hazard the modernization
# plan warns about ("do not use a geometry whose far plan is empty").
#
# Both sides use the SAME, deliberately cheap quadrature. What is under test is that MLFMA's near/far
# decomposition reproduces a dense matrix assembled the same way -- not BEAST's quadrature accuracy.
# Using `NEARINTERACTIONSQUADSTRATEGYRECOM` here instead would make the dense assembly dominate the
# runtime for no extra signal, which is how `comparison.jl` came to take ~58 minutes.
@testset "GalerkinMLFMA against a dense reference" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    ϵ = 1e-4

    m = meshsphere(0.8 * λ, λ / 6)
    X = raviartthomas(m)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)

    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))

    strategy = MLFMA.MLFMAStrategy(
        efie,
        X;
        samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
        nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
    )
    A_MLFMA = MLFMA.GalerkinMLFMA(efie, X; tree=tree, strategy=strategy, verbose=false)

    # --- non-vacuity: the fixture must actually exercise both halves of the method ----------------
    @test H2Trees.numberoflevels(tree) > 2

    near = H2Factory.nearinteractions(A_MLFMA)
    @test nnz(SparseArrays.sparse(near)) > 0

    farpairs = sum(
        length(collect(H2Trees.TranslatingNodesIterator(tree, node))) for
        node in H2Trees.DepthFirstIterator(tree)
    )
    @test farpairs > 0

    # the I2O collection actually stores couplings
    @test !isempty(MLFMA.translationcollection(A_MLFMA).dict)

    # at least one aggregation and one disaggregation level
    @test H2Trees.minaggregationlevel(H2Factory.trialaggregationplan(A_MLFMA)) >= 1
    @test H2Trees.mindisaggregationlevel(H2Factory.testdisaggregationplan(A_MLFMA)) >= 1

    # --- dense reference -------------------------------------------------------------------------
    A = BEAST.assemble(efie, X, X; quadstrat=quadstrat)

    # `checkinteractions` covers the plan's whole matvec list in one call: size, near-only
    # (`nearmul!`), far-only (`farmul!`), the plain `mul!` and the `*` form, plus that
    # `farinteractions ≈ A - nearinteractions` (i.e. the partition is complete). `ϵ` here is the
    # comparison rtol -- measured worst case is the far-only path at ~9e-5, so 1e-3 leaves ~10x.
    @test checkinteractions(A, A_MLFMA; n=3, ϵ=1e-3, verbose=false)

    # Explicit tolerances alongside, so a regression says WHICH path drifted rather than just that
    # `checkinteractions` failed. Bounds are the measured errors with roughly an order of magnitude
    # of headroom: full 2.5e-6, near exactly 0, far 8.9e-5.
    x = randn(ComplexF64, numfunctions(X))
    reference = A * x
    @test norm(A_MLFMA * x - reference) / norm(reference) < 1e-4

    ynear = similar(x)
    H2Factory.nearmul!(ynear, A_MLFMA, x)
    nearreference = near * x
    # the near block is extracted from the same dense matrix, so this is exact, not approximate
    @test norm(ynear - nearreference) / norm(nearreference) < 1e-12

    yfar = similar(x)
    H2Factory.farmul!(yfar, A_MLFMA, x)
    farreference = H2Factory.farinteractions(A, A_MLFMA) * x
    @test norm(yfar - farreference) / norm(farreference) < 1e-3
    # non-vacuity: the far path must contribute something, or the bound above is meaningless
    @test norm(farreference) / norm(reference) > 1e-3

    # --- workspace reuse -------------------------------------------------------------------------
    # Repeating a matvec through the same map must give the same answer: the traversal's scratch
    # buffers have to be reset per call, not accumulated into.
    first = A_MLFMA * x
    second = A_MLFMA * x
    @test first ≈ second
    # and a different input must still be handled correctly afterwards
    x2 = randn(ComplexF64, numfunctions(X))
    @test norm(A_MLFMA * x2 - A * x2) / norm(A * x2) < 1e-4

    # --- serial and threaded scheduling agree ----------------------------------------------------
    # The default is serial; a threaded outer traversal must produce the same result, which is what
    # rules out a chunk-shared buffer being written by two tasks at once.
    #
    # `nchunks` is EXPLICIT. A bare `DynamicScheduler()` takes its chunk count from
    # `Threads.nthreads()`, which is ONE unless the suite is run with `-t2` or more, so this testset
    # spent the whole refactor asserting that serial agrees with serial. A real out-of-bounds in the
    # level-transfer scratch sizing survived underneath it and only appeared at `-t4`; see the
    # `maxnchunks` note in src/assembly/cores/utilsmaps.jl. Chunking is by `nchunks`, not by thread count, so 4
    # chunks exercise chunk ids 1..4 even on one thread.
    A_threaded = MLFMA.GalerkinMLFMA(
        efie,
        X;
        tree=tree,
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
            nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
            scheduler=OhMyThreads.DynamicScheduler(; nchunks=4),
        ),
        verbose=false,
    )
    # Non-vacuity with teeth, replacing `>= 1`: without more than one chunk nothing here is threaded.
    @test H2Factory.nchunks(H2Factory.scheduling(A_threaded)) == 4
    @test norm(A_threaded * x - reference) / norm(reference) < 1e-4
    @test A_threaded * x ≈ A_MLFMA * x

    # --- reversed directions --------------------------------------------------------------------
    # Both reverse directions with `checkinteractions` (a full `n`-vector sweep), not a single
    # random vector: `tworects.jl` found a single vector can measure misleadingly small (~6e-4,
    # looked passing) on a direction whose full sweep then proved real per-column errors. `n=3`
    # matches this file's own forward check above.
    #
    @test checkinteractions(
        transpose(A), transpose(A_MLFMA); n=3, ϵ=1e-3, nearmul=false, verbose=false
    )

    # ADJOINT, and THIS FIXTURE IS THE ONE THAT EXERCISES THE LAST PIECE OF IT. Its adjoint
    # aggregation CROSSES LEVELS -- 4 levels, leaves at 3 and 4, `minrelevantlevel` 3, so the test
    # side genuinely transfers upward and the adjoint's I2I functor is invoked (3 times, instrumented)
    # rather than the 0 times it sees on `comparison.jl`/`petrovdirections.jl`.
    #
    # It was a `@test_skip` through two separate fixes. The conjugated leaf multiply
    # (`src/assembly/cores/adjointleafmul.jl`) handled `α` and both leaf collections but left this fixture at
    # 0.3428 far-only, because H2Factory applied the AGGREGATION level transfer unconjugated in every
    # orientation. `H2Factory.ConjugateTranslation` plus `MLFMA.ConjugatedInterpolator` closed that.
    # Far-only against `farinteractions`, before and after that second fix:
    #     transpose                       7.156e-5    7.156e-5
    #     adjoint, native traversal       0.3428      7.990e-5
    #     adjoint, via conj(A^T conj(x))  7.990e-5    7.990e-5
    # and native-vs-identity went 0.3428 -> 6.6e-16, i.e. the two routes are now the same computation
    # to machine precision rather than merely both inside a tolerance.
    #
    # WORTH KNOWING FOR THE NEXT BUG HERE: with `nearmul` on, the near block dilutes a far-field error
    # by ~35x (this one read as 9.98e-3 on a full matvec against an O(1) far-only error), so a
    # reverse-direction failure should be measured with `farmul!` against `farinteractions` before
    # anything is concluded from a full-matvec figure.
    @test checkinteractions(
        adjoint(A), adjoint(A_MLFMA); n=3, ϵ=1e-3, nearmul=false, verbose=false
    )
end

# Check mixed-depth aggregation against a dense reference, not only successful construction.
# The fixed mesh keeps populated leaves at two depths inside the aggregation range.
# Regenerating it in CI can change that coverage because Gmsh output varies by platform.
@testset "GalerkinMLFMA on a non-uniform tree" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    m = fixturemesh("mixed-aggregation")
    X = raviartthomas(m)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)

    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))

    # --- non-vacuity: the fixture must actually BE non-uniform ------------------------------------
    # Without this the testset degrades into a second uniform-tree check the moment a mesher or
    # builder default shifts, and still passes -- which is the failure mode that matters, because it
    # is invisible.
    leaflevels = [H2Trees.level(tree, leaf) for leaf in H2Trees.leaves(tree)]
    levels = sort(unique(leaflevels))
    counts = [count(==(l), leaflevels) for l in levels]
    @test length(levels) > 1
    # and both depths genuinely populated, so this is a mixed tree rather than one stray box
    @test minimum(counts) >= 5

    strategy = MLFMA.MLFMAStrategy(
        efie,
        X;
        samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
        nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
    )
    A_MLFMA = MLFMA.GalerkinMLFMA(efie, X; tree=tree, strategy=strategy, verbose=false)

    # Every leaf level here is INSIDE the aggregation range, so all of them are really traversed.
    # That makes this the mixed-depth AGGREGATION case specifically -- distinct from the
    # shallow-leaf-is-pure-near case that `_leaflevelindex` floors for, which the "Default
    # convenience constructor" testset above already pins through `minrelevantlevel`.
    @test levels[begin] >=
        H2Trees.minaggregationlevel(H2Factory.trialaggregationplan(A_MLFMA))
    @test levels[end] > levels[begin]  # ...and the deeper leaves transfer through a level the
    # shallower ones are already resident at, which is the whole point

    A = BEAST.assemble(efie, X, X; quadstrat=quadstrat)

    x = randn(ComplexF64, numfunctions(X))
    reference = A * x
    @test norm(A_MLFMA * x - reference) / norm(reference) < 1e-5

    # Far-only, against `farinteractions`, is where a level mismatch would actually show: a leaf
    # aggregated at the wrong rate leaves the near block untouched, and the near block dilutes a
    # far-field error by ~35x on a full matvec (see the adjoint note above).
    yfar = similar(x)
    H2Factory.farmul!(yfar, A_MLFMA, x)
    farreference = H2Factory.farinteractions(A, A_MLFMA) * x
    @test norm(yfar - farreference) / norm(farreference) < 1e-3
    @test norm(farreference) / norm(reference) > 1e-3

    near = H2Factory.nearinteractions(A_MLFMA)
    ynear = similar(x)
    H2Factory.nearmul!(ynear, A_MLFMA, x)
    @test norm(ynear - near * x) / norm(near * x) < 1e-12

    # The reverse direction separately: the disaggregation side keys its scratch by level too, so a
    # mixed-depth mistake there would not show up in the forward map at all.
    @test checkinteractions(
        adjoint(A), adjoint(A_MLFMA); n=3, ϵ=1e-3, nearmul=false, verbose=false
    )
end

@testset "GalerkinMLFMA with compressed leaf storage" begin
    # THE END-TO-END CONTRACT FOR SE-MLFMA, and the reason it is stated as "agrees with the SAMPLED
    # map" rather than "is accurate": everything below the map -- the collection, the order rule, the
    # batched multiply, the receive kernel -- is already checked against algebraic properties in
    # `test/FarField/`. What no algebraic property can check is whether the compressed path computes
    # the SAME OPERATOR as the sampled one, and that is exactly what went wrong once: a receive kernel
    # that satisfied its own adjoint identity exactly was 6.9e-2 off against a dense reference.
    #
    # ALL THREE ORIENTATIONS, because H2Factory realizes the transpose and the adjoint by swapping the
    # sides -- so each of the two collections is aggregated in one orientation and scattered in
    # another, four kernels in total, and the forward matvec exercises only two of them.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    # SIZED BY THE TREE, not by taste. λ/4 leaves are the low-frequency-breakdown floor
    # (`MLFMA.MINHALFSIZERECOM`), so the sphere has to be big enough that λ/4 boxes still give more
    # than two levels -- 0.8λ gives exactly two and the constructor rejects it. This is also a
    # NON-UNIFORM tree (leaf levels [3, 4]), which is the case that exercises the per-level order.
    X = raviartthomas(meshsphere(1.2 * λ, λ / 5))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0))
    @test length(MLFMA._leaflevels(tree)) > 1

    function mapwith(storage)
        strategy = MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
            farfieldstrategy=MLFMA.FarFieldStrategy(; leafstorage=storage),
            nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
        )
        return MLFMA.GalerkinMLFMA(efie, X; tree=tree, strategy=strategy, verbose=false)
    end

    sampled = mapwith(MLFMA.SampledLeafStorage())
    she = mapwith(MLFMA.SphericalHarmonicLeafStorage())
    vsh = mapwith(MLFMA.VectorSphericalHarmonicLeafStorage())

    @testset "the strategy actually selected it" begin
        # Otherwise every numerical assertion below would pass by comparing the sampled path to
        # itself, which is the failure mode a policy field invites.
        samplecore, shecore = H2Factory.core(sampled), H2Factory.core(she)
        @test valtype(shecore.trialfieldcollections) <:
            MLFMA.SphericalHarmonicRadiationCollection
        @test valtype(shecore.testfieldcollections) <:
            MLFMA.SphericalHarmonicReceiveCollection
        # ...and the control really is the other thing, so this is a comparison and not a tautology
        @test !(
            valtype(samplecore.trialfieldcollections) <:
            MLFMA.SphericalHarmonicRadiationCollection
        )
        @test valtype(samplecore.testfieldcollections) <: MLFMA.LazyReceiveFieldCollection

        vshcore = H2Factory.core(vsh)
        @test valtype(vshcore.trialfieldcollections) <:
            MLFMA.VectorSphericalHarmonicRadiationCollection
        @test valtype(vshcore.testfieldcollections) <:
            MLFMA.VectorSphericalHarmonicReceiveCollection
        # ...and it compresses, which no accuracy assertion would catch if the order rule stopped
        # truncating.
        @test Base.summarysize(shecore.trialfieldcollections) <
            Base.summarysize(samplecore.trialfieldcollections) / 3
        # and the TANGENTIAL basis compresses further than the Cartesian one -- the measured claim
        # the second compressed backend exists for.
        @test Base.summarysize(vshcore.trialfieldcollections) <
            Base.summarysize(shecore.trialfieldcollections)
    end

    A = BEAST.assemble(efie, X, X; quadstrat=quadstrat)
    x = randn(ComplexF64, numfunctions(X))
    b = randn(ComplexF64, numfunctions(X))

    @testset "it computes the same operator as the sampled map" begin
        for (backend, M) in (("SHE", she), ("VSH", vsh)),
            (name, apply, reference) in (
                ("forward", m -> m * x, A * x),
                ("transpose", m -> transpose(m) * b, transpose(A) * b),
                ("adjoint", m -> adjoint(m) * b, adjoint(A) * b),
            )

            sampledresult = apply(sampled)
            compressed = apply(M)
            samplederror = norm(sampledresult - reference) / norm(reference)
            error = norm(compressed - reference) / norm(reference)

            @test error < 1e-4
            # The real assertion: the compressed path is not merely accurate, it is accurate to the
            # SAME degree the sampled path is. A factor here is a convention mismatch, not a
            # truncation choice -- measured 1.000 for both backends on this fixture, so 2x is loose
            # by three orders of magnitude against the failure it screens for.
            @test error < 2 * samplederror
            @test norm(compressed - sampledresult) / norm(sampledresult) < 1e-4
        end
    end

    @testset "the far path alone, where a leaf-kernel error is not diluted" begin
        # The near block is IDENTICAL across the three maps and is ~97 % of the norm here, so a full
        # matvec hides a leaf error by about that factor. This is the sensitive comparison.
        yfarsampled = similar(x)
        H2Factory.farmul!(yfarsampled, sampled, x)
        @test norm(yfarsampled) / norm(A * x) > 1e-3   # non-vacuity: the far part is not noise
        for M in (she, vsh)
            yfar = similar(x)
            H2Factory.farmul!(yfar, M, x)
            @test norm(yfar - yfarsampled) / norm(yfarsampled) < 1e-3
        end
    end

    @testset "the leaf scratch pool follows the collections" begin
        # `buildleafchannels` dispatches on the collection type rather than on a flag, so this pins
        # that the workspace really got the SHE pool -- a sampled pool would be a `MethodError` deep
        # inside a traversal, which the matvecs above would have caught, but not legibly.
        for (M, Scratch) in (
            (she, MLFMA.SphericalHarmonicLeafScratch),
            (vsh, MLFMA.VectorSphericalHarmonicLeafScratch),
        )
            H2Factory.withschedulechunks(M.schedules) do
                channels = H2Factory.buildleafchannels(H2Factory.core(M))
                @test !isempty(channels)
                for level in keys(channels)
                    scratch = H2Factory.borrowbuffer(
                        channels, level, 1, H2Factory.isForward()
                    )
                    @test scratch isa Scratch
                    @test MLFMA.maxgroups(scratch) >= 1
                end
            end
        end
    end
end

@testset "Compressed leaf storage across operators" begin
    # THE TEST THAT WOULD HAVE CAUGHT THE MFIE BUG, and the reason it is a separate testset rather
    # than another assertion inside the one above: what it varies is the OPERATOR, and the defect it
    # screens for is precisely the assumption that one operator's receive coupling is evidence about
    # another's.
    #
    # `receivecouplingrule(MWEFIE3D)` happens to equal `mirrorrule(TangentialSphericalFarField)`;
    # `receivecouplingrule(MWDoubleLayer3D)` is a component swap with γ. Baking in the mirror
    # measured 4.4e-01 on MFIE against a dense reference, where the sampled path gives 2.8e-05.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1.2 * λ, λ / 5))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0))
    Random.seed!(19)
    x = randn(ComplexF64, numfunctions(X))

    for (opname, op) in (
        ("EFIE", MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))),
        ("MFIE", BEAST.MWDoubleLayer3D(im * κ)),
    )
        reference = BEAST.assemble(op, X, X; quadstrat=quadstrat) * x

        errors = map((
            MLFMA.SampledLeafStorage(),
            MLFMA.SphericalHarmonicLeafStorage(),
            MLFMA.VectorSphericalHarmonicLeafStorage(),
        )) do storage
            strategy = MLFMA.MLFMAStrategy(
                op,
                X;
                samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
                farfieldstrategy=MLFMA.FarFieldStrategy(; leafstorage=storage),
                nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(
                    quadstrat
                ),
            )
            A = MLFMA.GalerkinMLFMA(op, X; tree=tree, strategy=strategy, verbose=false)
            return norm(A * x - reference) / norm(reference)
        end

        # Every compressed backend agrees with the SAMPLED one for this operator, which is the
        # statement that fails at 4.4e-01 when the coupling is assumed rather than derived. Stated as
        # a ratio against the sampled error rather than an absolute bound, so the assertion does not
        # quietly weaken if the fixture's accuracy changes.
        for (i, name) in enumerate(("sampled", "SHE", "VSH"))
            @test errors[i] < 1e-3
            i == 1 || @test errors[i] < 2 * errors[1]
        end
    end
end
