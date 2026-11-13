using Test, LinearAlgebra
using CompScienceMeshes, BEAST
using H2Factory
using MLFMA

include("../support/util_test.jl")

@testset "LazyReceiveFieldCollection Maxwell" begin
    λ = 1.0
    k = 2π / λ

    m = meshsphere(λ, λ / 10)
    X = raviartthomas(m)

    sampling = MLFMA.SphericalSampling{Float64}(4, 8)
    efie = MLFMA.MWEFIE3D(; wavenumber=k)
    doublelayer = Maxwell3D.doublelayer(; wavenumber=k)
    singlelayer = Maxwell3D.singlelayer(; wavenumber=k)
    hypersingular = MLFMA.MWHyperSingular3D(; wavenumber=k)

    for operator in [efie, doublelayer, singlelayer, hypersingular]
        radiationcol = MLFMA.buildfieldcollection(
            radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(3)), operator, sampling, X
        )

        # Mirror INDICES, not a sampling: the collection stores indices and never needs the sampling
        # for anything else, so there is no constructor that takes one.
        mirroredradiationcol = MLFMA.LazyReceiveFieldCollection(
            operator, radiationcol, MLFMA.mirrorindices(sampling)
        )

        # The leaf scratch a workspace's leaf channel supplies. `LazyReceiveFieldCollection` is a
        # scratch-only leaf collection -- it implements the two `applyleafmul!` arities the traversal
        # calls and no `mul!`/`conjmul!` of its own -- so every call below goes through that seam,
        # exactly as the traversal does.
        leafscratch = MLFMA.farfieldtype(operator)(ComplexF64, sampling, Val(:zero))

        receivecol = MLFMA.buildfieldcollection(
            receivebuilder(; quadstrategy=BEAST.SingleNumQStrat(3)), operator, sampling, X
        )

        farfield = MLFMA.farfieldtype(operator)(BEAST.gamma(operator), sampling)
        farfield2 = MLFMA.farfieldtype(operator)(BEAST.gamma(operator), sampling)

        u = Float64.(1:numfunctions(X))
        u = u + im * reverse(u)

        # WRITE-THROUGH direction: `applyleafmul!(scratch, moment, lazy, y)` -- one `gemv` per
        # component into the borrowed scratch, then a gather through `receivecouplingrule` --
        # compared against the real, independently BEAST-assembled `ReceiveFieldCollection`. This is
        # the strongest check on the direction: the reference shares no code with it and was
        # assembled from the mesh, not derived from `radiationcol`.
        H2Factory.applyleafmul!(leafscratch, farfield, mirroredradiationcol, u)
        LinearAlgebra.mul!(farfield2, receivecol, u)
        for i in eachindex(farfield)
            @test farfield[i] ≈ farfield2[i]
        end

        @test warmedallocations(
            H2Factory.applyleafmul!, leafscratch, farfield, mirroredradiationcol, u
        ) == 0

        # READ direction -- populate `farfield` via the real, non-lazy `receivecol` so this
        # exercises the scratch path against real BEAST-assembled data for every operator here,
        # `doublelayer` (γ != 1 via k = 2π/λ) included.
        LinearAlgebra.mul!(farfield, receivecol, u)

        result = Vector{ComplexF64}(undef, numfunctions(X))
        result2 = Vector{ComplexF64}(undef, numfunctions(X))

        LinearAlgebra.mul!(result, receivecol, farfield)
        H2Factory.applyleafmul!(
            leafscratch, result2, mirroredradiationcol, farfield, true, false
        )

        @test result ≈ result2

        LinearAlgebra.mul!(result, receivecol, farfield, 1.0, 2.0)
        H2Factory.applyleafmul!(
            leafscratch, result2, mirroredradiationcol, farfield, 1.0, 2.0
        )

        @test result ≈ result2

        # Per-component α, where the operator has one: leaf disaggregation passes `MLFMA.α(operator)`
        # straight in, and the scratch path forwards it to H2Factory's own weighted collection `mul!`.
        weights = MLFMA.α(operator)
        if weights isa AbstractVector
            LinearAlgebra.mul!(result, receivecol, farfield, weights, 0.0)
            H2Factory.applyleafmul!(
                leafscratch, result2, mirroredradiationcol, farfield, weights, 0.0
            )
            @test result ≈ result2
        end

        @test warmedallocations(
            H2Factory.applyleafmul!,
            leafscratch,
            result2,
            mirroredradiationcol,
            farfield,
            true,
            false,
        ) == 0
    end
end

@testset "LazyReceiveFieldCollection Helmholtz" begin
    λ = 1.0
    k = 2π / λ

    m = meshsphere(λ, λ / 10)
    X = lagrangec0d1(m)

    sampling = MLFMA.SphericalSampling{Float64}(4, 8)
    singlelayer = Helmholtz3D.singlelayer(; wavenumber=k)
    hypersingular = Helmholtz3D.hypersingular(; wavenumber=k)

    for operator in [singlelayer, hypersingular]
        radiationcol = MLFMA.buildfieldcollection(
            radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(3)), operator, sampling, X
        )

        # Mirror INDICES, not a sampling: the collection stores indices and never needs the sampling
        # for anything else, so there is no constructor that takes one.
        mirroredradiationcol = MLFMA.LazyReceiveFieldCollection(
            operator, radiationcol, MLFMA.mirrorindices(sampling)
        )

        # The leaf scratch a workspace's leaf channel supplies. `LazyReceiveFieldCollection` is a
        # scratch-only leaf collection -- it implements the two `applyleafmul!` arities the traversal
        # calls and no `mul!`/`conjmul!` of its own -- so every call below goes through that seam,
        # exactly as the traversal does.
        leafscratch = MLFMA.farfieldtype(operator)(ComplexF64, sampling, Val(:zero))

        receivecol = MLFMA.buildfieldcollection(
            receivebuilder(; quadstrategy=BEAST.SingleNumQStrat(3)), operator, sampling, X
        )

        farfield = MLFMA.farfieldtype(operator)(BEAST.gamma(operator), sampling)
        farfield2 = MLFMA.farfieldtype(operator)(BEAST.gamma(operator), sampling)

        u = Float64.(1:numfunctions(X))
        u = u + im * reverse(u)

        # WRITE-THROUGH direction: `applyleafmul!(scratch, moment, lazy, y)` -- one `gemv` per
        # component into the borrowed scratch, then a gather through `receivecouplingrule` --
        # compared against the real, independently BEAST-assembled `ReceiveFieldCollection`. This is
        # the strongest check on the direction: the reference shares no code with it and was
        # assembled from the mesh, not derived from `radiationcol`.
        H2Factory.applyleafmul!(leafscratch, farfield, mirroredradiationcol, u)
        LinearAlgebra.mul!(farfield2, receivecol, u)
        for i in eachindex(farfield)
            @test farfield[i] ≈ farfield2[i]
        end

        @test warmedallocations(
            H2Factory.applyleafmul!, leafscratch, farfield, mirroredradiationcol, u
        ) == 0

        # READ direction -- populate `farfield` via the real, non-lazy `receivecol` so this
        # exercises the scratch path against real BEAST-assembled data.
        LinearAlgebra.mul!(farfield, receivecol, u)

        result = Vector{ComplexF64}(undef, numfunctions(X))
        result2 = Vector{ComplexF64}(undef, numfunctions(X))

        LinearAlgebra.mul!(result, receivecol, farfield)
        H2Factory.applyleafmul!(
            leafscratch, result2, mirroredradiationcol, farfield, true, false
        )

        @test result ≈ result2

        LinearAlgebra.mul!(result, receivecol, farfield, 1.0, 2.0)
        H2Factory.applyleafmul!(
            leafscratch, result2, mirroredradiationcol, farfield, 1.0, 2.0
        )

        @test result ≈ result2

        # Per-component α, where the operator has one: leaf disaggregation passes `MLFMA.α(operator)`
        # straight in, and the scratch path forwards it to H2Factory's own weighted collection `mul!`.
        weights = MLFMA.α(operator)
        if weights isa AbstractVector
            LinearAlgebra.mul!(result, receivecol, farfield, weights, 0.0)
            H2Factory.applyleafmul!(
                leafscratch, result2, mirroredradiationcol, farfield, weights, 0.0
            )
            @test result ≈ result2
        end

        @test warmedallocations(
            H2Factory.applyleafmul!,
            leafscratch,
            result2,
            mirroredradiationcol,
            farfield,
            true,
            false,
        ) == 0
    end
end

@testset "LazyReceiveFieldCollection is shareable across tasks, its scratch is not" begin
    # THE THREAD-SAFETY CONTRACT, and it is sharper than "concurrent calls agree with sequential ones"
    # because this collection is not self-contained: it is a SCRATCH-ONLY leaf collection, so every
    # `applyleafmul!` writes into a borrowed far field. That splits the question in two, and only the
    # first half is about the collection at all:
    #
    #   * the COLLECTION is read-only state (operator, radiation collection, mirror indices), so one
    #     instance may be shared by every task;
    #   * the SCRATCH is written on every call, so it must be per-task -- which is exactly why
    #     H2Factory's leaf channel hands out one far field per (level, chunk) rather than one per
    #     collection, and why `MLFMA.leafscratchchannel` sizes it by `nchunks(level)`.
    #
    # Asserting the first without the second would pass on an implementation that is not thread-safe at
    # all, so both are checked: N tasks share ONE collection with N scratches and N results, and every
    # result must equal the sequential one EXACTLY. Exactly, not approximately -- each task performs
    # identical arithmetic in identical order, so any difference is interference, not round-off, and a
    # tolerance would hide precisely the bug this is for.
    λ = 1.0
    k = 2π / λ
    m = meshsphere(λ, λ / 10)
    X = raviartthomas(m)
    sampling = MLFMA.SphericalSampling{Float64}(4, 8)
    operator = MLFMA.MWEFIE3D(; wavenumber=k)

    radiationcol = MLFMA.buildfieldcollection(
        radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(3)), operator, sampling, X
    )
    lazy = MLFMA.LazyReceiveFieldCollection(
        operator, radiationcol, MLFMA.mirrorindices(sampling)
    )

    u = Float64.(1:numfunctions(X))
    u = u + im * reverse(u)
    incoming = MLFMA.farfieldtype(operator)(BEAST.gamma(operator), sampling)
    LinearAlgebra.mul!(
        incoming,
        MLFMA.buildfieldcollection(
            receivebuilder(; quadstrategy=BEAST.SingleNumQStrat(3)), operator, sampling, X
        ),
        u,
    )

    ntasks = 4

    # EACH TASK GETS A DIFFERENT INPUT, and that is what makes the check discriminating rather than
    # decorative. Run every task on the SAME input and a stolen scratch computes the same answer as the
    # rightful owner, so interference is invisible and the testset passes on an implementation with no
    # thread safety at all. Distinct inputs make a stolen scratch a WRONG NUMBER. Verified by
    # deliberately sharing one scratch across the tasks below: 3 of 4 results were corrupted on the
    # first trial, so the assertions have teeth.
    scratches = [
        MLFMA.farfieldtype(operator)(ComplexF64, sampling, Val(:zero)) for _ in 1:ntasks
    ]
    incomings = [
        MLFMA.farfieldtype(operator)(BEAST.gamma(operator), sampling) for _ in 1:ntasks
    ]
    for t in 1:ntasks, i in eachindex(incomings[t])
        incomings[t][i] .= incoming[i] .* t
    end
    us = [u .* t for t in 1:ntasks]

    # READ direction: y = α·transpose(C)·incoming. Sequential reference per task, computed through ONE
    # scratch so the references themselves cannot be contaminated.
    reference = MLFMA.farfieldtype(operator)(ComplexF64, sampling, Val(:zero))
    sequentialreads = [Vector{ComplexF64}(undef, numfunctions(X)) for _ in 1:ntasks]
    for t in 1:ntasks
        H2Factory.applyleafmul!(
            reference, sequentialreads[t], lazy, incomings[t], true, false
        )
    end

    concurrentreads = [Vector{ComplexF64}(undef, numfunctions(X)) for _ in 1:ntasks]
    Threads.@sync for t in 1:ntasks
        Threads.@spawn H2Factory.applyleafmul!(
            scratches[t], concurrentreads[t], lazy, incomings[t], true, false
        )
    end
    for t in 1:ntasks
        @test concurrentreads[t] == sequentialreads[t]
    end

    # WRITE-THROUGH direction: moment = C·u, the arity an adjoint/transposed map aggregates through.
    sequentialmoments = [
        MLFMA.farfieldtype(operator)(BEAST.gamma(operator), sampling) for _ in 1:ntasks
    ]
    for t in 1:ntasks
        H2Factory.applyleafmul!(reference, sequentialmoments[t], lazy, us[t])
    end

    concurrentmoments = [
        MLFMA.farfieldtype(operator)(BEAST.gamma(operator), sampling) for _ in 1:ntasks
    ]
    Threads.@sync for t in 1:ntasks
        Threads.@spawn H2Factory.applyleafmul!(
            scratches[t], concurrentmoments[t], lazy, us[t]
        )
    end
    for t in 1:ntasks, i in eachindex(sequentialmoments[t])
        @test concurrentmoments[t][i] == sequentialmoments[t][i]
    end

    # NON-VACUITY. The results must be non-trivial, and -- the part that matters here -- the per-task
    # inputs must actually produce DIFFERENT answers, or "each task got its own result back" is
    # satisfied by any implementation whatsoever.
    @test maximum(abs, sequentialreads[1]) > 1e-6
    @test maximum(abs, sequentialmoments[1][1]) > 1e-6
    @test sequentialreads[1] != sequentialreads[2]
    @test sequentialmoments[1][1] != sequentialmoments[2][1]
    # ...and the suite must actually be running more than one thread for the concurrent half to mean
    # anything. This is reported rather than asserted: `Threads.@spawn` on a single-threaded runner
    # still exercises the task boundary and the sharing, just not true parallelism, and CI covers the
    # parallel case in its 4-thread row (see .github/workflows/Runtests.yml). An @test here would fail
    # the default local run for no defect.
    Threads.nthreads() == 1 &&
        @info "lazy-receive concurrency ran on ONE thread; the parallel case is CI's 4-thread row"
    @test Threads.nthreads() >= 1
end
