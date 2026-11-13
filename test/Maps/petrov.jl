using MLFMA
using BEAST, CompScienceMeshes
using StaticArrays
using Test
using OhMyThreads
using H2Factory
using Random
using LinearAlgebra
using H2Trees

include("../support/util_test.jl")

# Regression pin, MOVED not retired: both map constructors must give every scheduled chunk its OWN
# interpolation scratch, sized from the map's own `scheduling` rather than from `Threads.nthreads()`.
#
# WHERE IT USED TO LIVE. The interpolator and anterpolator carried borrow/return POOLS
# (`interimmatrices`/`interimθvectors`/`interimφvectors`/`ftpadmatrices`) on the read-only core, and this
# file pinned their pool sizes. `PetrovGalerkinMLFMA` once passed `nchunks` to neither
# `Interpolator(...)` nor `anterpolator(...)` while `GalerkinMLFMA` passed both, so a serially scheduled
# Petrov map still allocated one pool per thread and a map scheduled into more chunks than there are
# threads allocated too FEW -- for containers the traversal borrows by chunk id.
#
# WHERE IT LIVES NOW. Those pools are gone (src/interpolation/interimscratch.jl): the interim arrays ride
# in the `MLFMA.LevelTransferScratch` that the level-transfer channels hand out per (level, chunkid), so
# they are per WORKSPACE and per chunk by construction. The under-allocation half of the old defect is
# therefore structurally impossible now, and this file pins what replaced it: one scratch set per chunk,
# all distinct, shaped so the interpolator actually USES them.
#
# The scheduler is built with an explicit `nchunks` rather than the default: the two would coincide on
# any single-threaded runner (`Threads.nthreads() == 1` gives `nchunks == 1` too), and the pin has to
# distinguish them to mean anything.
#
# FIXTURE: `meshsphere(2λ, λ/6)` at `minhalfsize = λ/4` is the smallest tried that reaches TWO relevant
# levels, which is what this needs: with one relevant level there is no level transfer at all and the
# channels are empty. Near interactions and translation are stubbed and the far-field quadrature is the
# cheapest available, since only construction is under test.
@testset "Both map constructors give every chunk its own interpolation scratch" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    X = raviartthomas(meshsphere(2 * λ, λ / 6))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    # Bounded, and never equal to the thread count whatever `-t` the suite is run with. A literal 4
    # would make the non-vacuity guard below FAIL (not go vacuous) on a 4-threaded runner, while
    # `Threads.nthreads() + 3` would scale the scratch allocation with the runner.
    nchunks = Threads.nthreads() == 4 ? 5 : 4
    # A BARE scheduler: the strategy rejects a materialized `ExecutionScheduling`, and the map keys the
    # per-level containers from its own tree's level range.
    scheduler = OhMyThreads.DynamicScheduler(; nchunks=nchunks)
    @test H2Factory.nchunks(H2Factory.executionscheduling(scheduler, 1:12)) == nchunks

    # `stubbedstrategy` covers all three stubs (near, translation, cheap far quadrature): only
    # construction is under test here, and matvec VALUES are undefined under it.
    strategyfor(spaces...) = stubbedstrategy(
        efie,
        spaces...;
        treestrategy=MLFMA.TreeStrategy(efie; minhalfsize=λ / 4),
        nearfieldstrategy=H2Factory.InteractionStrategy(;
            isnear=H2Trees.isnear(; additionalbufferboxes=1)
        ),
        scheduler=scheduler,
    )

    galerkin = MLFMA.assemble(efie, X; strategy=strategyfor(X), verbose=false)
    petrov = MLFMA.assemble(efie, X, X; strategy=strategyfor(X, X), verbose=false)

    # --- non-vacuity ------------------------------------------------------------------------------
    # The channels only carry transfers when the tree has more than one relevant level (see the fixture
    # note), and the chosen `nchunks` has to differ from the thread count or the pin proves nothing.
    @test length(MLFMA.samplings(galerkin)) > 1
    @test nchunks != Threads.nthreads()

    # The pools this file used to inspect must be GONE from the core, not merely bypassed: a leftover
    # field would mean shared mutable state stayed on the read-only half.
    for A in (galerkin, petrov)
        interpolatorfields = fieldnames(typeof(MLFMA.interpolator(A)))
        anterpolatorfields = fieldnames(typeof(MLFMA.anterpolator(A)))
        @test :interimmatrices ∉ interpolatorfields
        @test :ftpadmatrices ∉ interpolatorfields
        @test :interimmatrices ∉ anterpolatorfields
        @test :interimθvectors ∉ anterpolatorfields
        @test :interimφvectors ∉ anterpolatorfields
    end

    # Both channel kinds, because they are built by two different functions
    # (`scratchi2ichannel`/`scratcho2ochannel`) that size their tuples independently -- checking one
    # would let the other regress, which is the same hole the old two-constructor version guarded.
    channelsof(A) = (H2Factory.triali2ichannel(A), H2Factory.trialo2ochannel(A))

    function scratchsets(A)
        return [store.buffers for channel in channelsof(A) for store in values(channel)]
    end

    # Asserted per map rather than in a loop, so a failure names which constructor regressed.
    for (name, A) in (("Galerkin", galerkin), ("Petrov", petrov))
        sets = scratchsets(A)
        @test !isempty(sets)
        @test all(set -> length(set) == nchunks, sets)

        # The claim the old pool version could not make: each chunk's arrays are DISTINCT objects, so
        # two chunks running concurrently cannot write the same interim buffer. `===`-identity, because
        # equality would be satisfied by two `undef` arrays that happen to match.
        for set in sets
            interims = [scratch.interim.interimmatrix for scratch in set]
            @test length(unique(objectid.(interims))) == nchunks
        end
    end

    # And the scratch has to be the shape the interpolator asks for, or `MLFMA.takeinterim` silently
    # falls back to allocating a fresh array per apply -- correct, but exactly the cost this change was
    # made to remove. `takeinterim` returning the SAME object is what proves the buffer is used.
    #
    # CHECKED AGAINST WHAT THE FAMILY ITSELF DECLARES, not against one family's field names. This block
    # used to spell out the barycentric-Lagrange layout (`interimmatrix` at `(Nθ(parent), Nφ(child))`,
    # plus `interimθvector` and `interimφvector`) and so broke the moment `INTERPOLATORECOM` became the
    # local/global family, whose scratch is `ftpadmatrix`/`interimmatrix`/`interimθvector` with the
    # matrix TRANSPOSED. Nothing was wrong in production -- the channels build their scratch from
    # `interimscratch(anterpolator, ...)`, so it always matched whichever family was in use -- but the
    # test encoded the old default's spelling rather than the property it meant to assert.
    #
    # Rebuilding the reference from `interimscratch` is stronger as well as family-agnostic: it checks
    # EVERY field the family declares, and cannot go stale when a family's scratch changes.
    #
    # `level` keys the CHILD of the transfer, so the parent's sampling sits one index lower.
    for A in (galerkin, petrov)
        samplings = MLFMA.samplings(A)
        for channel in channelsof(A), (level, store) in pairs(channel)
            childsampling = samplings[MLFMA.levelindex(A, level)]
            parentsampling = samplings[MLFMA.levelindex(A, level - 1)]
            interim = first(store.buffers).interim

            reference = MLFMA.interimscratch(
                MLFMA.anterpolator(A), childsampling, parentsampling
            )
            # Non-vacuity: a family with no interim scratch would make every assertion below empty.
            @test reference !== nothing
            @test keys(interim) == keys(reference)

            for name in keys(reference)
                stored = getfield(interim, name)
                shape = MLFMA._interimshape(getfield(reference, name))
                @test MLFMA.takeinterim(stored, ComplexF64, shape) === stored
            end
        end
    end
end

# THE TWO SIDES OF A PETROV MAP ARE INDEPENDENT, in what they store and in the leaf pool they need,
# and until this was measured they were neither.
#
# BOTH sides of a Petrov map are ASSEMBLED -- the test tree holds different functions, so there is
# nothing to mirror -- which is exactly why each can choose its own storage. Two things had to be true
# for that and neither was:
#
#  1. `buildleafchannels` returned `SidedChannels(channel, channel)` unconditionally, so a
#     compressed-trial Petrov map BUILT and then died at the first matvec with the collection's own
#     "needs a `*LeafScratch`" error -- which names the scratch rather than the choice that made it
#     unobtainable, and arrives after the whole construction has been paid for.
#  2. The compressed builders were RADIATION-only, so `testleafstorage` had no builder to reach.
#
# `SidedChannels{Test,Trial}` was independently parameterized all along, and an assembled compressed
# receive side needs no mirror and no coupling -- both of those exist only because the GALERKIN receive
# side derives a pattern it does not store. So this testset covers all three compressed combinations on
# both 3-D backends, and asserts the sides really do carry different representations rather than only
# that the numbers come out.
#
# AGREEMENT WITH SAMPLED, NOT ABSOLUTE ACCURACY, is the assertion. Leaf compression is a small
# perturbation of one stage, and this fixture's error is dominated by others -- so a bound on the
# absolute error would pass whether or not the compressed path ran at all. Comparing the two maps
# directly is what makes a broken compressed leaf visible.
@testset "Petrov gives each side the leaf pool and the storage its own collections need" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    trialspace = raviartthomas(meshsphere(0.7λ, λ / 7))
    testspace = raviartthomas(
        CompScienceMeshes.translate(
            meshsphere(0.7λ, λ / 7), SVector{3,Float64}(4λ, 0.0, 0.0)
        ),
    )

    function petrovwith(trial, test)
        strategy = MLFMA.MLFMAStrategy(
            efie,
            testspace,
            trialspace;
            samplingstrategy=MLFMA.SamplingStrategy(efie; ϵ=1e-4),
            farfieldstrategy=MLFMA.FarFieldStrategy(;
                trialleafstorage=trial, testleafstorage=test
            ),
        )
        return MLFMA.PetrovGalerkinMLFMA(
            efie, testspace, trialspace; strategy=strategy, verbose=false
        )
    end

    sampled = MLFMA.SampledLeafStorage()
    reference = petrovwith(sampled, sampled)
    x = randn(MersenneTwister(9), ComplexF64, numfunctions(trialspace))
    y = randn(MersenneTwister(11), ComplexF64, numfunctions(testspace))
    baseline = reference * x
    adjointbaseline = adjoint(reference) * y
    @test norm(baseline) > 0
    # The all-sampled map shares ONE pool, which is the behaviour every fixture predating #139 has
    # and which the per-side split must not disturb. Stated here as the control for the
    # `!==` assertions below, so the pair reads as one claim rather than two.
    @test H2Factory.trialleafchannel(reference) === H2Factory.testleafchannel(reference)

    # BOTH compressed backends, because they own different scratch types, different order rules and
    # different synthesis blocks -- a pool built for one cannot serve the other, so one method each is
    # what makes this work. `Radiation` and `Receive` are the two collection types that side must end
    # up with when it is the compressed one.
    for (storage, Radiation, Receive) in (
        (
            MLFMA.SphericalHarmonicLeafStorage(),
            MLFMA.SphericalHarmonicRadiationCollection,
            MLFMA.SphericalHarmonicAssembledReceiveCollection,
        ),
        (
            MLFMA.VectorSphericalHarmonicLeafStorage(),
            MLFMA.VectorSphericalHarmonicRadiationCollection,
            MLFMA.VectorSphericalHarmonicAssembledReceiveCollection,
        ),
    )
        @testset "$(nameof(typeof(storage))) on each side" begin
            # ALL THREE compressed combinations. The two mixed ones are the interesting cases -- they
            # are the only ones where the map holds two DIFFERENT leaf scratch types at once, which is
            # the thing `SidedChannels{Test,Trial}` exists for and the thing one shared channel broke.
            for (trial, test) in
                ((storage, sampled), (sampled, storage), (storage, storage))
                A = petrovwith(trial, test)
                core = H2Factory.core(A)

                # Each side carries the representation it was ASKED for, and only that side. Without
                # this, a silent fallback to sampled storage would pass every accuracy check below --
                # and a `testleafstorage` that was quietly ignored would pass them too.
                if trial === storage
                    @test eltype(core.trialfieldcollections) <: Radiation
                else
                    @test eltype(core.trialfieldcollections) <:
                        MLFMA.TangentialSphericalRadiationFieldCollection
                end
                if test === storage
                    @test eltype(core.testfieldcollections) <: Receive
                    # An ASSEMBLED receive collection, not the mirrored Galerkin one -- the
                    # distinction the two types exist to make, and the reason this side needs neither
                    # `mirrorrule` nor `receivecouplingrule`.
                    @test eltype(core.testfieldcollections) <:
                        MLFMA.AbstractReceiveFieldCollection
                else
                    @test eltype(core.testfieldcollections) <:
                        MLFMA.TangentialSphericalReceiveFieldCollection
                end

                # AND THE TWO SIDES GET DIFFERENT LEAF POOLS. This is the `SidedChannels{Test,Trial}`
                # property the whole fix turns on, asserted directly rather than only through the
                # matvec: one shared channel is what made a compressed Petrov map die at its first
                # leaf multiply. The all-sampled case still SHARES one pool -- pinned in
                # `test/Maps/workspace.jl`, and deliberately not changed here, since a sampled pool is
                # sized for the full direction grid and doubling it is a memory change of its own.
                @test H2Factory.trialleafchannel(A) !== H2Factory.testleafchannel(A)

                # The matvec runs at all -- which it did not before -- and agrees with the sampled
                # map. Measured: 1.4e-7 to 4.3e-7 forward across the six cases here.
                @test norm(A * x - baseline) / norm(baseline) < 1e-3
                # And the adjoint, which reaches the OTHER arity of both compressed kernels: on a
                # compressed TEST side that is the conjugated AGGREGATION, which is a kernel no
                # forward matvec touches.
                @test norm(adjoint(A) * y - adjointbaseline) / norm(adjointbaseline) < 1e-3
            end

            # THE TWO ERRORS DO NOT ADD. Compressing both sides is measured at or below the QUADRATURE
            # of compressing each alone, never their sum -- the two truncations act on different
            # patterns and their errors are uncorrelated. Held as an assertion because the prediction
            # on record was the conservative "under 2ε", and this branch has already had five
            # predictions refuted by measurement.
            onlytrial = petrovwith(storage, sampled)
            onlytest = petrovwith(sampled, storage)
            both = petrovwith(storage, storage)
            errorof(A) = norm(A * x - baseline) / norm(baseline)
            a, b = errorof(onlytrial), errorof(onlytest)
            @test errorof(both) <= 1.2 * sqrt(a^2 + b^2)
        end
    end
end
