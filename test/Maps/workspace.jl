# The core/workspace split: Galerkin (Phase 5b) in the first five testsets, Petrov (Phase 6 slice 1) in
# the last two.
#
# What is pinned here is the OWNERSHIP boundary, not the numerics -- galerkin.jl already checks the
# matvec against a dense reference. These tests fail if read-only data drifts onto the workspace or, the
# direction that actually matters, if matvec scratch drifts back onto the core.
#
# Two workspaces over one core are pinned BOTH ways: sequentially, and concurrently from two tasks. The
# concurrent testset could not be written while MLFMA's interpolator and anterpolator carried scratch pools
# on the core, sized for one workspace's `nchunks` -- N workspaces queued on them and serialized. Every
# array a level transfer writes now comes from the workspace, borrowed per (level, chunkid); see
# `src/interpolation/interimscratch.jl` and the header of `src/assembly/cores/GalerkinMLFMA.jl`.

using Test
using LinearAlgebra
using CompScienceMeshes
using BEAST
using H2Factory
using H2Trees
using MLFMA
using OhMyThreads

include("../support/util_test.jl")

function hasconjugatelevelphases(trial, test)
    trial.transfers === nothing && return test.transfers === nothing
    test.transfers === nothing && return false
    keys(trial.transfers) == keys(test.transfers) || return false
    for level in keys(trial.transfers)
        triallevel, testlevel = trial.transfers[level], test.transfers[level]
        keys(triallevel) == keys(testlevel) || return false
        for sector in keys(triallevel)
            t, s = triallevel[sector], testlevel[sector]
            t.phase == conj.(s.phase) || return false
            t.childsampling == s.childsampling || return false
            t.parentsampling == s.parentsampling || return false
        end
    end
    return true
end

@testset "Sided transfers follow the phase convention, not object identity" begin
    parent = MLFMA.CircularSampling{Float64}(28)
    child = MLFMA.CircularSampling{Float64}(12)
    interpolator = MLFMA.CircularInterpolator{ComplexF64}()
    anterpolator = MLFMA.anterpolator(interpolator)
    phase = cis.(range(0.2, 1.1; length=length(parent)))
    shifter = MLFMA.PhaseShifter([Dict(0 => phase)], 1)
    trial, test = MLFMA.interpolationphasecollections(
        shifter,
        interpolator,
        anterpolator,
        [parent, child],
        1,
        1:2,
        MLFMA.PeriodicφInterpolation{2}();
        dims=2,
    )
    @test maximum(abs, imag.(phase)) > 0.1
    @test hasconjugatelevelphases(trial, test)
    @test !hasconjugatelevelphases(trial, trial)

    emptytrial, emptytest = MLFMA.interpolationphasecollections(
        shifter,
        interpolator,
        anterpolator,
        [parent],
        1,
        1:1,
        MLFMA.PeriodicφInterpolation{2}();
        dims=2,
    )
    @test emptytrial === emptytest
    @test hasconjugatelevelphases(emptytrial, emptytest)
    @test !hasconjugatelevelphases(emptytrial, test)
end

# Same fixture recipe as galerkin.jl's dense-reference testset, and for the same measured reason: the
# smallest sphere tried whose far plan is non-empty, with a deliberately cheap matched quadrature on
# both sides.
@testset "The Galerkin constructor returns a workspace over a core" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    m = meshsphere(0.8 * λ, λ / 6)
    X = raviartthomas(m)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))
    strategy = stubbedstrategy(efie, X)

    A = MLFMA.GalerkinMLFMA(efie, X; tree=tree, strategy=strategy, verbose=false)

    # non-vacuity: a two-level tree would exercise neither the level transfers nor the split
    @test H2Trees.numberoflevels(tree) > 2

    @test A isa H2Factory.GalerkinH2Workspace
    @test A isa H2Factory.GalerkinH2Map          # still the same abstract map the traversal wants

    core = H2Factory.core(A)
    @test core isa MLFMA.GalerkinMLFMACore
    @test core isa H2Factory.GalerkinH2Core
    # The core is deliberately NOT executable: `mul!(y, core, x)` must not compile into a traversal
    # that then fails deep inside for want of moments.
    @test !(core isa H2Factory.AbstractH2Map)

    # `eltype` must survive the split -- the workspace takes `T` from the core rather than storing it.
    @test eltype(A) == eltype(core)
    @test eltype(A) == typeof(MLFMA.γ(efie))

    # Orientation is a property of the wrapper, never of the shared data.
    @test H2Factory.core(transpose(A)) === core
    @test H2Factory.core(adjoint(A)) === core

    # The core constructor is the documented way to get one without a workspace, and it must produce
    # the same kind of object the convenience constructor wraps.
    barecore = MLFMA.GalerkinMLFMACore(efie, X; tree=tree, strategy=strategy, verbose=false)
    @test barecore isa MLFMA.GalerkinMLFMACore
    @test typeof(barecore) == typeof(core)
end

@testset "Matvec scratch lives on the workspace, not the core" begin
    # A field-name pin. It is the direction that matters: putting read-only data on the workspace only
    # wastes memory, whereas putting mutable scratch back on the core silently makes two workspaces
    # over one core alias each other's buffers, which no numerical test on ONE workspace would catch.
    corefields = fieldnames(MLFMA.GalerkinMLFMACore)

    for scratch in (
        :farfields,            # trial moments
        :testfarfields,        # test moments
        :matvecbuffers,        # the pooled memory the moments wrap
        :vectorbuffer,         # per-node leaf mul! scratch
        :i2ichannel,
        :o2ochannel,
        :i2ochannel,
        :leafchannel,          # the scratch a lazy receive collection's leaf mul! stages through
        :scheduling,           # sizes the pools, so it is per workspace by definition
        :schedules,
    )
        @test !(scratch in corefields)
    end

    # ...and the read-only half is present, so the pin above is not passing merely because the struct
    # is empty.
    for readonly in (
        :tree,
        :nearinteractions,
        :trialfieldcollections,
        :testfieldcollections,
        :i2otranslator,
        :leveltranslators,
        :interpolator,
        :anterpolator,
        :phaseshifter,
        :samplings,
        :plans,
        :defaultscheduler,
    )
        @test readonly in corefields
    end

    # The workspace side of the same boundary: H2Factory owns these names, so this asserts the hooks
    # actually delivered something rather than that MLFMA stored it.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))
    A = MLFMA.GalerkinMLFMA(
        efie, X; tree=tree, strategy=stubbedstrategy(efie, X), verbose=false
    )

    buffers = H2Factory.workspacebuffers(A)
    @test !isempty(H2Factory.trialmoments(A))
    @test !isempty(H2Factory.testmoments(A))
    @test H2Factory.momentbuffers(A) === buffers.momentbuffers
    # Scratch is the same object whichever way the map is read -- it belongs to the workspace, not to
    # the orientation it is being read in.
    @test H2Factory.workspacebuffers(transpose(A)) === buffers
    @test H2Factory.workspacebuffers(adjoint(A)) === buffers

    # The leaf-multiply scratch: one far field per (level, chunk), shaped for the LEAF's own level,
    # that `LazyReceiveFieldCollection`'s leaf `mul!` mirrors into. A Galerkin core has one tree, so
    # both sides answer with the same bare channel -- pinned as identity so that splitting it per
    # side, a memory/behaviour change, cannot happen silently.
    leafchannels = H2Factory.trialleafchannel(A)
    @test leafchannels === H2Factory.testleafchannel(A)
    @test leafchannels === buffers.leafchannels
    @test !isempty(leafchannels)

    # Keyed by the leaf's own level, with a buffer whose type and sampling match the moment the leaf
    # multiply will hand it. `receivefields!` requires the two to be the SAME concrete type, so a
    # mismatch here is a `MethodError` in the middle of a matvec rather than anything subtler.
    for leaf in H2Trees.leaves(H2Trees.tree(A))
        level = H2Trees.level(H2Trees.tree(A), leaf)
        # A leaf with no moment is pure near-interaction: no plan aggregates or disaggregates it, so
        # it never borrows leaf scratch either and there is nothing here to compare against. The
        # pool is per LEVEL, so its siblings at the same level still cover that key.
        (haskey(leafchannels, level) && haskey(H2Factory.trialmoments(A), leaf)) || continue
        buffer = H2Factory.borrowbuffer(leafchannels, level, 1, H2Factory.maptrait(A))
        @test typeof(buffer) === typeof(H2Factory.trialmoments(A)[leaf])
        @test size(buffer[1]) == size(MLFMA.sampling(A, level))
    end
end

@testset "Read-only accessors resolve through the workspace" begin
    # A forwarding list that covers only the forward orientation is the documented failure mode here,
    # so every accessor is exercised in all three. `H2Trees.isnear` is the one that would return a
    # WRONG ANSWER rather than erroring if the core forgot its override: H2Factory's forward body is
    # typed on `AbstractH2Map` (which a core is not), while `H2Trees.isnear(tree, ::Any)` matches and
    # silently builds a predicate over the core as though it were a tree.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))
    A = MLFMA.GalerkinMLFMA(
        efie, X; tree=tree, strategy=stubbedstrategy(efie, X), verbose=false
    )
    core = H2Factory.core(A)

    @test H2Trees.tree(A) === tree
    @test H2Trees.tree(transpose(A)) === tree
    @test H2Trees.tree(adjoint(A)) === tree

    # Not just "does not throw": the predicate must answer the four-argument far-extraction query,
    # which is the call a predicate built over the wrong object has no method for.
    @test H2Trees.isnear(A) === core.isnear
    @test H2Trees.isnear(transpose(A)) === core.isnear
    @test H2Trees.isnear(adjoint(A)) === core.isnear

    for accessor in (
        H2Factory.nearinteractions,
        H2Factory.trialmomentcollections,
        H2Factory.testmomentcollections,
        H2Factory.trialaggregationplan,
        H2Factory.testaggregationplan,
        H2Factory.trialdisaggregationplan,
        H2Factory.testdisaggregationplan,
        H2Factory.i2otranslator,
        H2Factory.minrelevantlevel,
        H2Factory.farmulmode,
        H2Factory.verbose,
    )
        # Every orientation must resolve. Whether the ANSWER flips with orientation is H2Factory's
        # business (`nearinteractions` transposes, `verbose` does not), so this checks resolution, and
        # the forward answer is compared to the core's directly.
        @test accessor(A) == accessor(core, H2Factory.isForward())
        @test (accessor(transpose(A)); true)
        @test (accessor(adjoint(A)); true)
    end

    # `operator` is FORWARD-ONLY, and not because of anything this split changed: H2Factory's
    # one-argument entry point for it is typed `AbstractH2Map` (singular) where its siblings use
    # `AbstractH2Maps` (the union that also covers `TransposeMap`/`AdjointMap`), so
    # `H2Factory.operator(transpose(A))` has thrown a `MethodError` for every H2 map, pre-split
    # included. Pinned at the arity that does resolve, so this testset documents the asymmetry instead
    # of tripping over it.
    @test H2Factory.operator(A) == H2Factory.operator(core, H2Factory.isForward())
    @test H2Factory.operator(A) === core.operator
    @test_throws MethodError H2Factory.operator(transpose(A))

    # The test side stores conjugate phases. Empty immutable collections may be identical.
    @test H2Factory.triali2itranslator(A) === core.leveltranslators.trial
    @test H2Factory.testi2itranslator(A) === core.leveltranslators.test
    @test hasconjugatelevelphases(
        H2Factory.triali2itranslator(A), H2Factory.testi2itranslator(A)
    )

    # MLFMA's own accessors, which needed workspace->core forwarding added for this phase. Callers hold
    # the workspace, so reading them off the map has to keep working.
    @test MLFMA.samplings(A) === core.samplings
    @test MLFMA.interpolator(A) === core.interpolator
    @test MLFMA.anterpolator(A) === core.anterpolator
    @test MLFMA.phaseshifter(A) === core.phaseshifter
    @test MLFMA.γ(A) == core.γ
    @test MLFMA.ϵ(A) == core.ϵ
    # Answered for the CORE as well as the workspace, which it was not: `translationcollection` used to
    # delegate to H2Factory's one-argument `i2otranslator`, and that resolves through `maptrait(A)`, which
    # no core defines. So the workspace form passed while the core form was a `MethodError` -- invisible
    # because every caller in the suite held a workspace. Both arities are pinned now.
    @test MLFMA.translationcollection(A) === core.i2otranslator
    @test MLFMA.translationcollection(core) === core.i2otranslator
    @test MLFMA.translationcollection(core) === MLFMA.translationcollection(A)
    # And the other MLFMA accessors answer on a core too, which is the invariant the one above broke.
    for accessor in (
        MLFMA.samplings,
        MLFMA.interpolator,
        MLFMA.anterpolator,
        MLFMA.phaseshifter,
        MLFMA.translationcollection,
    )
        @test accessor(core) === accessor(A)
    end

    # The level-indexed lookups are written against the accessors rather than the fields, so one
    # definition serves core, workspace and pre-split Petrov map alike.
    level = H2Factory.minrelevantlevel(A)
    @test MLFMA.levelindex(A, level) == 1
    @test MLFMA.sampling(A, level) === MLFMA.samplings(A)[1]
    @test MLFMA.sampling(A, level) === MLFMA.sampling(core, level)
end

@testset "Two workspaces over one core, used sequentially, both multiply correctly" begin
    # THE point of the split, and the strongest available check that no matvec state stayed on the
    # core: if any did, the second workspace would either alias the first's buffers or find them
    # already written, and its answer would drift from the dense reference.
    #
    # SEQUENTIAL on purpose, and kept even though the concurrent testset below now covers more: bitwise
    # agreement between two workspaces run one after the other isolates "state left behind on the core"
    # from "state raced on concurrently", and only this one can fail for the first reason alone.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    ϵ = 1e-4

    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))

    core = MLFMA.GalerkinMLFMACore(
        efie,
        X;
        tree=tree,
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
            nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
        ),
        verbose=false,
    )

    first = H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler())
    second = H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler())

    # Genuinely two workspaces over ONE core -- otherwise the agreement below would be trivial.
    @test first !== second
    @test H2Factory.core(first) === core
    @test H2Factory.core(second) === core
    @test H2Factory.workspacebuffers(first) !== H2Factory.workspacebuffers(second)
    @test H2Factory.trialmoments(first) !== H2Factory.trialmoments(second)
    # ...including the leaf-multiply scratch, down to the buffers rather than only the enclosing
    # dicts: two dicts holding the same far fields would still race, and the bitwise agreement below
    # would not catch it, since these two run sequentially.
    @test H2Factory.trialleafchannel(first) !== H2Factory.trialleafchannel(second)
    for level in keys(H2Factory.trialleafchannel(first))
        @test H2Factory.borrowbuffer(
            H2Factory.trialleafchannel(first), level, 1, H2Factory.maptrait(first)
        ) !== H2Factory.borrowbuffer(
            H2Factory.trialleafchannel(second), level, 1, H2Factory.maptrait(second)
        )
    end

    Adense = BEAST.assemble(efie, X, X; quadstrat=quadstrat)
    x = ComplexF64[cis(0.7 * i) * (1 + 0.5 * cospi(i / 17)) for i in 1:numfunctions(X)]
    reference = Adense * x
    @test norm(reference) > 0     # non-vacuity: the bounds below mean nothing against a zero vector

    y1 = first * x
    y2 = second * x
    @test norm(y1 - reference) / norm(reference) < 1e-4
    @test norm(y2 - reference) / norm(reference) < 1e-4
    # Bitwise-equal is the real claim: two workspaces over one core do the same arithmetic in the same
    # order, so anything shared-and-mutated would show up here before it showed up in the tolerance.
    @test y1 == y2

    # And the first workspace still works after the second has run, which is what fails if the second
    # construction reached into the core and overwrote something the first was holding.
    @test first * x == y1

    # A workspace may be sized differently from its core's default without rebuilding the core -- the
    # `nchunks` dial is per workspace, which is why `scheduling` and `schedules` are workspace fields.
    chunky = H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler(); nchunks=3)
    @test H2Factory.nchunks(H2Factory.scheduling(chunky)) == 3
    @test H2Factory.nchunks(H2Factory.scheduling(first)) == 1
    @test norm(chunky * x - reference) / norm(reference) < 1e-4
end

@testset "Two workspaces over one core, used concurrently, both multiply correctly" begin
    # What the split was for, and what could NOT be claimed until the interpolator's interim scratch
    # moved off the core (task #45): with the scratch pooled on the core, N workspaces queued on pools
    # sized for one and serialized. Now every array a level transfer writes is reached through
    # `context.buffer`, which the traversal borrows per (level, chunkid) from the WORKSPACE, so two
    # workspaces share nothing mutable.
    #
    # The hazard this would catch is a write race, which shows up as a wrong answer rather than an error,
    # so both tasks are checked against the dense reference AND against the same product computed with
    # nothing else running.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    ϵ = 1e-4

    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))

    core = MLFMA.GalerkinMLFMACore(
        efie,
        X;
        tree=tree,
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
            nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
        ),
        verbose=false,
    )

    # Serially scheduled workspaces, run from two concurrent TASKS. The scheduling is deliberately not
    # the thing under test: a `DynamicScheduler` inside each workspace would test H2Factory's own
    # intra-matvec threading, which `allocations.jl` already covers, and would make it much harder to say
    # which of the two overlapping matvecs corrupted what.
    A = H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler())
    B = H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler())
    @test H2Factory.core(A) === H2Factory.core(B) === core
    @test H2Factory.workspacebuffers(A) !== H2Factory.workspacebuffers(B)

    Adense = BEAST.assemble(efie, X, X; quadstrat=quadstrat)
    x = ComplexF64[cis(0.7 * i) * (1 + 0.5 * cospi(i / 17)) for i in 1:numfunctions(X)]
    u = ComplexF64[cis(1.3 * i) * (1 - 0.25 * sinpi(i / 11)) for i in 1:numfunctions(X)]
    reference, referenceu = Adense * x, Adense * u
    @test norm(reference) > 0
    @test norm(referenceu) > 0

    # Uncontended results first, so the comparison below is against this map's own arithmetic rather
    # than only against the dense reference's tolerance.
    quiet, quietu = A * x, B * u

    # Repeated inside each task to widen the overlap window: one matvec each could finish before the
    # other starts, and the point is to have them in flight together.
    repeats = 4
    ys = Vector{Vector{ComplexF64}}(undef, repeats)
    us = Vector{Vector{ComplexF64}}(undef, repeats)
    @sync begin
        Threads.@spawn for i in 1:repeats
            ys[i] = A * x
        end
        Threads.@spawn for i in 1:repeats
            us[i] = B * u
        end
    end

    # Non-vacuity limit, stated rather than asserted (asserting it would fail the suite on a
    # single-threaded runner, which is a property of the runner and not of this code): with
    # `Threads.nthreads() == 1` the two tasks interleave only at yield points, so this degrades to a
    # sequential check. It still fails on shared mutable state either way, but the write-race half of the
    # claim needs `-t2` or better to bite -- and Julia defaults to ONE thread, so `Pkg.test()` without
    # `JULIA_NUM_THREADS`/`-t` takes the degraded path. The `@info` makes that visible in the log rather
    # than letting a green run imply coverage it did not have.
    Threads.nthreads() > 1 ||
        @info "concurrent workspace testset ran on one thread: write-race coverage is degraded"

    for i in 1:repeats
        @test norm(ys[i] - reference) / norm(reference) < 1e-4
        @test norm(us[i] - referenceu) / norm(referenceu) < 1e-4
        # Bitwise, as in the sequential testset: identical arithmetic in identical order, so a single
        # word written by the other task shows up here well before it breaches the tolerance above.
        @test ys[i] == quiet
        @test us[i] == quietu
    end

    # Neither workspace is left damaged by having run alongside the other.
    @test A * x == quiet
    @test B * u == quietu
end

@testset "The Petrov constructor returns a sided workspace over a sided core" begin
    # Phase 6 slice 1: the same ownership boundary on the two-space side. What differs from the Galerkin
    # testsets above is SIDEDNESS -- two spaces, a `BlockTree`, per-side field collections, moments and
    # leaf vector buffers -- so that is what this checks, rather than repeating the Galerkin assertions.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    tree = H2Trees.BlockTree(
        X,
        X;
        builder=H2Trees.BlockTreeBuilder(;
            test=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0),
            trial=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0),
        ),
    )
    A = MLFMA.PetrovGalerkinMLFMA(
        efie, X, X; tree=tree, strategy=stubbedstrategy(efie, X, X), verbose=false
    )

    @test A isa H2Factory.PetrovGalerkinH2Workspace
    @test A isa H2Factory.PetrovGalerkinH2Map

    core = H2Factory.core(A)
    @test core isa MLFMA.PetrovGalerkinMLFMACore
    @test core isa H2Factory.PetrovGalerkinH2Core
    @test !(core isa H2Factory.AbstractH2Map)
    @test eltype(A) == eltype(core)

    # Same scratch-absent / read-only-present pin as the Galerkin core gets, since the failure it
    # guards against -- scratch drifting back onto a shared core -- is identical.
    corefields = fieldnames(MLFMA.PetrovGalerkinMLFMACore)
    for scratch in (
        :trialfarfields,
        :testfarfields,
        :matvecbuffers,
        :testvectorbuffer,
        :trialvectorbuffer,
        :i2ichannel,
        :o2ochannel,
        :i2ochannel,
        :leafchannel,
        :scheduling,
        :schedules,
    )
        @test !(scratch in corefields)
    end
    for readonly in (
        :tree,
        :testspace,
        :trialspace,
        :i2otranslator,
        :leveltranslators,
        :plans,
        :samplings,
    )
        @test readonly in corefields
    end

    # The field rename this slice made: `translationcollection` -> `i2otranslator`, which is the name
    # H2Factory's generic accessor reads, so the two `i2otranslator` overrides the pre-split map needed
    # are gone. Pinned because the rename is what makes their absence correct.
    @test !(:translationcollection in corefields)
    @test H2Factory.i2otranslator(A) === core.i2otranslator
    @test H2Factory.i2otranslator(transpose(A)) === core.i2otranslator
    @test H2Factory.i2otranslator(adjoint(A)) === core.i2otranslator
    @test MLFMA.translationcollection(A) === core.i2otranslator
    # On the Petrov CORE too, not only the workspace -- see the equivalent pin in the Galerkin accessor
    # testset for why the core arity needed its own method.
    @test MLFMA.translationcollection(core) === core.i2otranslator

    # --- what is genuinely per side ---------------------------------------------------------------
    testtree = H2Trees.testtree(tree)
    trialtree = H2Trees.trialtree(tree)
    @test H2Trees.testtree(A) === testtree
    @test H2Trees.trialtree(A) === trialtree

    # Equal spaces may share read-only values, but each side needs its own mutable moments.
    @test H2Factory.testmomentcollections(A) !== H2Factory.trialmomentcollections(A)
    @test H2Factory.testmoments(A) !== H2Factory.trialmoments(A)
    @test hasconjugatelevelphases(
        H2Factory.triali2itranslator(A), H2Factory.testi2itranslator(A)
    )
    @test H2Factory.triali2itranslator(A) === core.leveltranslators.trial
    @test H2Factory.testi2itranslator(A) === core.leveltranslators.test

    # Sided leaf vector buffers, from `buildvectorbuffers` returning a `VectorBuffers` pair.
    @test H2Factory.testvectorbuffer(A, first(H2Trees.leaves(testtree))) !== nothing
    @test H2Factory.trialvectorbuffer(A, first(H2Trees.leaves(trialtree))) !== nothing
    @test H2Factory.workspacebuffers(A).vectorbuffers isa H2Factory.VectorBuffers

    # --- level-transfer scratch is SHARED between the sides, deliberately -------------------------
    # `buildi2ichannels`/`buildo2ochannels` wrap ONE channel in `SidedChannels`, reproducing the
    # pre-split map's single `i2ichannel`/`o2ochannel` field, which H2Factory's generic accessors handed
    # to both sides. Pinned as identity so that giving each side its own -- a memory/behaviour change --
    # cannot happen silently.
    @test H2Factory.workspacebuffers(A).i2ichannels isa H2Factory.SidedChannels
    @test H2Factory.triali2ichannel(A) === H2Factory.testi2ichannel(A)
    @test H2Factory.trialo2ochannel(A) === H2Factory.testo2ochannel(A)

    # The leaf channel is shared between the sides for the same reason and with the same caveat: one
    # `samplings` vector serves both trees, so a level's buffer has one shape whichever side borrows
    # it, and the two leaf multiplies are barrier-separated phases of one matvec.
    #
    # CONDITIONAL ON BOTH SIDES BEING SAMPLED, which they are here and in every fixture that predates
    # #139. A compressed side needs a pool of a different TYPE, so it gets its own -- pinned from the
    # other direction in `test/Maps/petrov.jl`. This assertion is what stopped that split from
    # spreading to the sampled case as a side effect, which is exactly what it was written for: a
    # sampled pool is sized for the full direction grid per (level, chunk), so doubling it is a
    # memory change and belongs to its own slice.
    @test H2Factory.workspacebuffers(A).leafchannels isa H2Factory.SidedChannels
    @test H2Factory.trialleafchannel(A) === H2Factory.testleafchannel(A)

    # MLFMA's own accessors forward from the Petrov workspace too, not just the Galerkin one.
    @test MLFMA.samplings(A) === core.samplings
    @test MLFMA.interpolator(A) === core.interpolator
    @test MLFMA.γ(A) == core.γ
    @test MLFMA.sampling(A, H2Factory.minrelevantlevel(A)) === MLFMA.samplings(A)[1]

    # Two workspaces over one Petrov core, structurally. Their matvec VALUES cannot be checked here --
    # `stubbedstrategy`'s dummy I2O returns its output buffer unzeroed, so products under it are
    # undefined -- which is why the concurrent testset below builds its own real-I2O fixture instead of
    # reusing this one.
    second = H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler())
    @test H2Factory.core(second) === core
    @test H2Factory.workspacebuffers(second) !== H2Factory.workspacebuffers(A)
    @test H2Factory.trialmoments(second) !== H2Factory.trialmoments(A)
end

@testset "Two Petrov workspaces over one core, used concurrently" begin
    # The Galerkin concurrent testset above does not cover this: a Petrov core is sided, and its
    # `buildi2ichannels`/`buildo2ochannels` deliberately wrap ONE channel in `SidedChannels` so both sides
    # share it. That sharing is within a workspace, and this is what pins that it stays within one -- if
    # two workspaces ever came to share a channel, the products below would disagree.
    #
    # Its own fixture, with a REAL I2O and a DUMMY near block, because the sided testset above runs under
    # `stubbedstrategy` where matvec values are undefined. With a dummy near block the map's full matvec IS
    # its far matvec, so `H2Factory.farinteractions` is the right reference -- the same recipe
    # petrovdirections.jl uses, and comparing a full matvec against a far-only reference is the mistake
    # that recipe exists to avoid. One space twice: rectangularity is petrovdirections.jl's concern, not
    # this one.
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)

    X = raviartthomas(meshsphere(0.8 * λ, λ / 6))
    tree = H2Trees.BlockTree(
        X,
        X;
        builder=H2Trees.BlockTreeBuilder(;
            test=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0),
            trial=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0),
        ),
    )
    core = MLFMA.PetrovGalerkinMLFMACore(
        efie,
        X,
        X;
        tree=tree,
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
            nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
            ),
        ),
        verbose=false,
    )

    A = H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler())
    B = H2Factory.H2Workspace(core, OhMyThreads.SerialScheduler())
    @test H2Factory.core(A) === H2Factory.core(B) === core
    @test H2Factory.workspacebuffers(A) !== H2Factory.workspacebuffers(B)
    # The sided channels are shared WITHIN each workspace and must not be shared BETWEEN them. Both
    # halves are asserted, because only the pair distinguishes "sided sharing" from "leaked to the core".
    @test H2Factory.triali2ichannel(A) === H2Factory.testi2ichannel(A)
    @test H2Factory.triali2ichannel(A) !== H2Factory.triali2ichannel(B)
    @test H2Factory.trialo2ochannel(A) !== H2Factory.trialo2ochannel(B)
    @test H2Factory.trialleafchannel(A) === H2Factory.testleafchannel(A)
    @test H2Factory.trialleafchannel(A) !== H2Factory.trialleafchannel(B)

    far = H2Factory.farinteractions(BEAST.assemble(efie, X, X; quadstrat=quadstrat), A)
    x = ComplexF64[cis(0.7 * i) * (1 + 0.5 * cospi(i / 17)) for i in 1:numfunctions(X)]
    u = ComplexF64[cis(-0.41 * i) * (1 + 0.3 * sinpi(i / 13)) for i in 1:numfunctions(X)]
    referencex, referenceu = far * x, far * u
    @test norm(referencex) > 0
    @test norm(referenceu) > 0

    quiet, quietu = A * x, B * u

    repeats = 4
    ys = Vector{Vector{ComplexF64}}(undef, repeats)
    us = Vector{Vector{ComplexF64}}(undef, repeats)
    @sync begin
        Threads.@spawn for i in 1:repeats
            ys[i] = A * x
        end
        Threads.@spawn for i in 1:repeats
            us[i] = B * u
        end
    end

    # Same one-thread caveat as the Galerkin concurrent testset: on a default runner the two tasks
    # interleave only at yield points.
    Threads.nthreads() > 1 || @info "concurrent Petrov workspace testset ran on one thread"

    # The tolerance is petrovdirections.jl's measured forward bound on this recipe, not a fresh guess.
    for i in 1:repeats
        @test maximum(abs.(ys[i] .- referencex)) / maximum(abs.(referencex)) < 5e-3
        @test maximum(abs.(us[i] .- referenceu)) / maximum(abs.(referenceu)) < 5e-3
        @test ys[i] == quiet
        @test us[i] == quietu
    end

    @test A * x == quiet
    @test B * u == quietu
end
