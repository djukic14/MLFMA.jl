module SharedCoreDirectionsTest
using Test, LinearAlgebra, MLFMA, H2Factory, H2Trees, OhMyThreads, BEAST, CompScienceMeshes

@testset "Independent workspaces share grids, not mutable direction state" begin
    op = BEAST.Helmholtz2D.singlelayer(wavenumber=2π)
    X = BEAST.lagrangec0d1(CompScienceMeshes.meshcircle(1.5, 1 / 3))
    Y = BEAST.lagrangec0d1(CompScienceMeshes.meshcircle(1.4, 0.4))
    builder(n) = H2Trees.TwoNTreeBuilder(
        minhalfsize=0.25, minvalues=n, protrusion=H2Trees.NoProtrusionCheck()
    )
    strategy(spaces...) = MLFMA.MLFMAStrategy(
        op,
        spaces...;
        samplingstrategy=MLFMA.SamplingStrategy(op; ϵ=1e-5, protrusionaware=true),
        nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
            BEAST.defaultquadstrat(op, first(spaces), last(spaces))
        ),
    )
    galerkin = MLFMA.GalerkinMLFMACore(
        op, X; tree=H2Trees.TwoNTree(X; builder=builder(0)), strategy=strategy(X)
    )
    petrov = MLFMA.PetrovGalerkinMLFMACore(
        op,
        X,
        Y;
        tree=H2Trees.BlockTree(
            X, Y; builder=H2Trees.BlockTreeBuilder(test=builder(0), trial=builder(4))
        ),
        strategy=strategy(X, Y),
    )
    directions = (identity, transpose, adjoint)
    for (core, testspace, trialspace) in ((galerkin, X, X), (petrov, X, Y))
        @test length(core.samplings) >= 2
        @testset "$(nameof(typeof(core)))" begin
            levels = H2Factory.schedulinglevels(core)
            schedules = (
                H2Factory.serialexecutionscheduling(levels; nchunks=1),
                H2Factory.defaultexecutionscheduling(
                    levels; outer=DynamicScheduler(), nchunks=3
                ),
                H2Factory.defaultexecutionscheduling(
                    levels; outer=DynamicScheduler(), nchunks=2
                ),
            )
            maps = map(s -> H2Factory.H2Workspace(core, s), schedules)
            @test map(A -> H2Factory.nchunks(H2Factory.scheduling(A)), maps) == (1, 3, 2)
            for i in eachindex(maps), j in 1:(i - 1)
                @test H2Factory.core(maps[i]) === H2Factory.core(maps[j])
                @test H2Factory.workspacebuffers(maps[i]) !==
                    H2Factory.workspacebuffers(maps[j])
                @test H2Factory.triali2ichannel(maps[i]) !==
                    H2Factory.triali2ichannel(maps[j])
            end

            for collection in values(core.leveltranslators),
                (level, transfers) in collection.transfers

                index = level - core.minrelevantlevel + 1
                for transfer in values(transfers)
                    @test transfer.childsampling === core.samplings[index]
                    @test transfer.parentsampling === core.samplings[index - 1]
                end
            end

            dense = BEAST.assemble(op, testspace, trialspace)
            far = H2Factory.farinteractions(dense, first(maps))
            @test norm(far) > 0
            core === petrov && @test size(far, 1) != size(far, 2)
            inputs = map(directions) do direction
                n = size(direction(far), 2)
                ComplexF64[cis(0.31 * i) * (1 + 0.2cos(i)) for i in 1:n]
            end
            quiet = [
                direction(A) * input for
                A in maps, (direction, input) in zip(directions, inputs)
            ]
            for i in eachindex(maps), d in eachindex(directions)
                reference = directions[d](far) * inputs[d]
                @test norm(quiet[i, d] - reference) / norm(reference) < 1e-3
            end

            results = Array{Vector{ComplexF64}}(undef, length(maps), 3, 3)
            @sync for i in eachindex(maps)
                Threads.@spawn for repetition in 1:3, offset in 0:2
                    d = mod1(i + offset, 3)
                    results[i, d, repetition] = directions[d](maps[i]) * inputs[d]
                end
            end
            for i in eachindex(maps), d in 1:3, repetition in 1:3
                @test results[i, d, repetition] == quiet[i, d]
            end
        end
    end
    Threads.nthreads() > 1 ||
        @info "Shared-core direction test ran on one thread; repeat with -t4 for race coverage"
end
end
