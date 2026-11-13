# Allocation and type-stability contracts for the hot paths.
#
# ALLOCATION AND STABILITY, NOT TIMING. Wall-clock thresholds are noisy enough on shared machines to
# fail for reasons that have nothing to do with MLFMA, so nothing here measures time.
#
# Two kinds of claim live below, and they are not equally robust:
#
#   SCALE-FREE claims -- a ratio, or "does not grow" -- hold regardless of Julia version, thread count
#   or mesh. These are the primary pins.
#
#   ABSOLUTE byte budgets depend on the fixture and on Julia's allocation behaviour. They are pinned
#   with the measured value quoted and roughly 3x headroom, so a real regression trips them while a
#   version bump does not. If one of them fails after an upgrade, re-measure before assuming a bug.
#
# `@allocated` measures the FIRST call as compilation, so every path below is warmed before it is
# measured, and the warmup is not incidental -- removing it makes every number here meaningless.

using Test
using LinearAlgebra
using CompScienceMeshes
using BEAST
using H2Factory
using H2Trees
using MLFMA
using OhMyThreads

@testset "Workspace reuse avoids reconstruction" begin
    # THE CONTRACT THE CORE/WORKSPACE SPLIT EXISTS FOR: a second executable object over one core must
    # cost a small fraction of redoing construction. Before the split there was no way to obtain one
    # without rebuilding everything, so this claim could not even be stated.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))
    strategy = stubbedstrategy(efie, X)

    # Warm both paths. Also the objects the rest of the testset uses.
    core = MLFMA.GalerkinMLFMACore(efie, X; tree=tree, strategy=strategy, verbose=false)
    H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler())

    coreallocs = @allocated MLFMA.GalerkinMLFMACore(
        efie, X; tree=tree, strategy=strategy, verbose=false
    )
    workspaceallocs = @allocated H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler())

    # NON-VACUITY: the ratio below only means something if constructing a core is genuinely expensive.
    # Measured 111.8 MiB on this fixture; if it ever came out near zero the ratio would pass trivially.
    @test coreallocs > 10 * 2^20

    # THE PIN. Measured 4.49 MiB against 111.8 MiB, i.e. 4.0%. Bounded at 15% for headroom.
    @test workspaceallocs / coreallocs < 0.15
    # And in absolute terms, measured 4.49 MiB.
    @test workspaceallocs < 15 * 2^20
end

@testset "A matvec neither grows nor drifts across calls" begin
    # A LEAK PIN, and scale-free: whatever one matvec costs, the tenth must cost the same. Growth here
    # means the traversal is accumulating into something it should be reusing -- the failure mode the
    # workspace's per-chunk pools exist to prevent.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))
    A = MLFMA.GalerkinMLFMA(
        efie, X; tree=tree, strategy=stubbedstrategy(efie, X), verbose=false
    )

    x = ComplexF64[cis(0.7 * i) * (1 + 0.5 * cospi(i / 17)) for i in 1:numfunctions(X)]
    y = similar(x)

    LinearAlgebra.mul!(y, A, x)   # warm: the first call is compilation

    allocs = [(@allocated LinearAlgebra.mul!(y, A, x)) for _ in 1:5]

    # non-vacuity: a matvec on this fixture does allocate, so "no growth" is a real constraint rather
    # than a statement about zero. Measured 1.94 MiB per matvec.
    @test minimum(allocs) > 0
    # THE PIN: flat across calls. Measured bit-identical five times over.
    @test maximum(allocs) / minimum(allocs) < 1.05
    # Absolute budget, measured 1.94 MiB. Generous, because it scales with the mesh and the level count.
    @test maximum(allocs) < 8 * 2^20
end

@testset "A threaded workspace over one core behaves the same way" begin
    # The plan asks for serial AND threaded variants. Only the SCALE-FREE claims are made here:
    # absolute allocations under a `DynamicScheduler` scale with the runner's thread count, so pinning
    # bytes would make this test's verdict depend on the machine it runs on.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))
    core = MLFMA.GalerkinMLFMACore(
        efie, X; tree=tree, strategy=stubbedstrategy(efie, X), verbose=false
    )

    serial = H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler())
    # EXPLICIT `nchunks`, not the default. `DynamicScheduler()` derives its chunk count from
    # `Threads.nthreads()`, which is ONE unless the suite is run with `-t2` or more -- so the default makes
    # every "threaded" claim below vacuous on a default runner, and it is what let a real chunk-sizing bug
    # in `scratchi2ichannel`/`scratcho2ochannel` sit green for the whole refactor (it only surfaced at
    # `-t4`; see the `maxnchunks` note in src/assembly/cores/utilsmaps.jl). Chunking is by `nchunks`, not by thread
    # count, so an explicit 4 exercises chunk ids 1..4 even single-threaded.
    threaded = H2Factory.H2Workspace(
        core, OhMyThreads.DynamicScheduler(; nchunks=4); nchunks=4
    )

    # Genuinely different scheduling over ONE core, or the comparison is vacuous.
    @test H2Factory.core(threaded) === core
    @test H2Factory.nchunks(H2Factory.scheduling(serial)) == 1
    # Non-vacuity with teeth, replacing `>= 1`: more than one chunk is the whole point.
    @test H2Factory.nchunks(H2Factory.scheduling(threaded)) == 4

    x = ComplexF64[cis(0.7 * i) * (1 + 0.5 * cospi(i / 17)) for i in 1:numfunctions(X)]
    y = similar(x)

    LinearAlgebra.mul!(y, threaded, x)   # warm
    allocs = [(@allocated LinearAlgebra.mul!(y, threaded, x)) for _ in 1:5]
    @test maximum(allocs) / minimum(allocs) < 1.10   # looser than serial: task spawning varies

    # NO VALUE COMPARISON HERE, and the reason is worth recording rather than leaving as an omission.
    # `stubbedstrategy` supplies `H2Factory.DummyI2OTranslationCollection`, whose coupling
    # returns its output buffer UNCHANGED -- it does not even zero it -- so disaggregation reads
    # uninitialised moment memory and the matvec result is undefined. Asserting
    # `serial * x ≈ threaded * x` here produced `NaN` on one side and a large finite value on the other,
    # from two workspaces' separate `undef` buffers. That is not a scheduling bug; it is what a no-op
    # translation means.
    #
    # Serial-versus-threaded agreement IS pinned, with a real I2O and against a dense reference, by the
    # "serial and threaded scheduling agree" testset in test/Maps/galerkin.jl. This file's job is the
    # allocation behaviour, and it deliberately does not restate a correctness claim on a fixture that
    # cannot support one.
end

@testset "Hot accessors are type-stable" begin
    # `@inferred` needs a concrete call at the macro site, so these are written out rather than looped:
    # iterating over a tuple of functions makes the variable a `Union` and `@inferred` errors on it.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))
    A = MLFMA.GalerkinMLFMA(
        efie, X; tree=tree, strategy=stubbedstrategy(efie, X), verbose=false
    )

    # Read-only accessors, which resolve workspace -> core through H2Factory's forwarding. An instability
    # here lands in the traversal's inner loops.
    @inferred H2Trees.tree(A)
    @inferred H2Factory.trialmomentcollections(A)
    @inferred H2Factory.testmomentcollections(A)
    @inferred H2Factory.trialmoments(A)
    @inferred H2Factory.testmoments(A)
    @inferred H2Factory.i2otranslator(A)
    @inferred H2Factory.triali2itranslator(A)
    @inferred H2Factory.testi2itranslator(A)
    @inferred H2Factory.minrelevantlevel(A)
    @inferred H2Factory.farmulmode(A)
    @inferred MLFMA.samplings(A)
    @inferred MLFMA.interpolator(A)
    @inferred MLFMA.anterpolator(A)
    @inferred MLFMA.γ(A)
    @test true   # reaching here means every `@inferred` above held
end

@testset "Traversal phases: aggregation is allocation-free, and the far path is bounded" begin
    # The phase's acceptance criterion is "hot direction application paths allocate zero or have an
    # explicitly justified small bound". These are the PHASE entry points -- `aggregate!` covers the leaf
    # multiply and every I2I level transfer, `farmul!` adds I2O and disaggregation -- rather than the
    # inner kernels below them, which take per-node scratch arguments the traversal owns and cannot be
    # called meaningfully in isolation.
    #
    # A REAL I2O collection here, unlike the testsets above: with the dummy translation the I2O phase does
    # nothing at all, so pinning `farmul!` under it would measure an empty step. Only the near block is
    # stubbed.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))
    A = MLFMA.GalerkinMLFMA(
        efie,
        X;
        tree=tree,
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
            nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
            ),
        ),
        verbose=false,
    )

    x = ComplexF64[cis(0.7 * i) * (1 + 0.5 * cospi(i / 17)) for i in 1:numfunctions(X)]
    y = similar(x)

    H2Factory.aggregate!(A, x)      # warm
    H2Factory.farmul!(y, A, x)      # warm

    aggregateallocs = [(@allocated H2Factory.aggregate!(A, x)) for _ in 1:3]
    farallocs = [(@allocated H2Factory.farmul!(y, A, x)) for _ in 1:3]

    # non-vacuity: a tree with level transfers to perform, or "aggregation allocates nothing" would be a
    # statement about an empty traversal.
    @test H2Trees.numberoflevels(tree) > 2

    # AGGREGATION IS EFFECTIVELY ALLOCATION-FREE. Measured 224 bytes on this fixture, bit-identical
    # across repeats -- the leaf multiply writes into the workspace's moments and the I2I transfers borrow
    # from its channels, so neither should allocate. Bounded at 4 KiB, which is generous for "a handful of
    # bytes" while still failing loudly if a per-node temporary appears.
    #
    # The figure was 105 bytes when this was written. It is prose, not an assertion, and the bound is
    # loose enough that the drift went unnoticed until someone measured again; the number is quoted here
    # so a LARGE change is visible on reading, not so that a small one fails.
    #
    # SERIAL SCHEDULING, which is this strategy's default. A `DynamicScheduler` allocates for task
    # spawning, so this bound is not meaningful for one and is deliberately not asserted there.
    @test maximum(aggregateallocs) < 4 * 1024
    @test maximum(aggregateallocs) == minimum(aggregateallocs)

    # The far path costs more, and the contrast localizes where: measured 65104 bytes against
    # aggregation's 224, so I2O plus disaggregation accounts for essentially all of a matvec's
    # allocation. Pinned as a bounded, stable budget rather than as zero.
    #
    # This read 1.7334 MiB when written, so the far path has become roughly 28x cheaper and nothing
    # here noticed. The cause was not identified; it is not the symmetry reduction, which costs MORE
    # (see `TRANSLATIONRECOM` in `src/MLFMA/AbstractMLFMAMatrix.jl`), and the far traversal itself lives
    # in H2Factory. Left as a bounded budget rather than tightened: a tight bound on a number whose
    # movement is unexplained would fail on the next legitimate change without saying why.
    @test maximum(farallocs) == minimum(farallocs)
    @test maximum(farallocs) < 8 * 2^20
    @test maximum(farallocs) > maximum(aggregateallocs)
end

@testset "Far-field leaf assembly allocates little beyond what it retains" begin
    # The other loop the phase names. Expressed as allocated/retained, which is the ratio H2Trees uses for
    # its own contracts and the one the exploratory `test/performance/test_farfieldallocations.jl` says it
    # lacks a baseline for. This is that baseline: measured 1.068, i.e. the assembly allocates under 7%
    # more than the collections it produces.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))
    strategy = stubbedstrategy(efie, X)   # the near block is irrelevant here and stays cheap

    plans = H2Factory.assembleplans(tree, MLFMA.nearfieldstrategy(strategy))
    minrelevantlevel = plans.relevantlevels[begin]
    samplings, _ = MLFMA.getsamplingsparentsamplings(
        MLFMA.γ(efie),
        MLFMA.ϵ(strategy),
        tree,
        plans.relevantlevels,
        MLFMA.samplingfunction(strategy),
    )

    build() = MLFMA.getfieldcollectionsdict(
        MLFMA.radiationfieldcollectionbuilder(strategy),
        efie,
        X,
        tree,
        samplings,
        minrelevantlevel,
    )

    collections = build()   # warm
    allocated = @allocated build()
    retained = Base.summarysize(collections)

    # non-vacuity: the collections must actually be large, or the ratio says nothing. Measured 85 MiB.
    @test retained > 10 * 2^20
    # THE PIN. Measured 1.068; bounded at 1.5, which still catches a doubling of working memory.
    @test allocated / retained < 1.5
end
