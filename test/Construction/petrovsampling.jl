module PetrovSamplingTest
using Test, MLFMA, H2Factory, H2Trees, BEAST, CompScienceMeshes

function blocktree(testspace, trialspace, testminvalues, trialminvalues)
    builder(n) = H2Trees.TwoNTreeBuilder(;
        minhalfsize=0.2, minvalues=n, protrusion=H2Trees.NoProtrusionCheck()
    )
    return H2Trees.BlockTree(
        testspace,
        trialspace;
        builder=H2Trees.BlockTreeBuilder(;
            test=builder(testminvalues), trial=builder(trialminvalues)
        ),
    )
end

struct CountingBuilder{B}
    builder::B
    calls::Base.RefValue{Int}
end
function (b::CountingBuilder)(rates...)
    b.calls[] += 1
    return b.builder(rates...)
end
MLFMA.Sampling.samplingrate(b::CountingBuilder, degree::Integer) =
    MLFMA.Sampling.samplingrate(b.builder, degree)

@testset "Petrov grids are shared and cover both support requirements" begin
    fine = BEAST.lagrangec0d1(CompScienceMeshes.meshcircle(1.0, 0.1))
    coarse = BEAST.lagrangec0d1(CompScienceMeshes.meshcircle(1.0, 0.5))
    op = BEAST.Helmholtz2D.singlelayer(wavenumber=2π)
    γ = MLFMA.γ(op)

    for (testspace, trialspace) in ((fine, coarse), (coarse, fine)),
        (testmin, trialmin) in ((0, 100), (100, 0), (0, 0)),
        protrusionaware in (false, true)

        tree = blocktree(testspace, trialspace, testmin, trialmin)
        sides = (H2Trees.testtree(tree), H2Trees.trialtree(tree))
        strategy = MLFMA.MLFMAStrategy(
            op,
            testspace,
            trialspace;
            samplingstrategy=MLFMA.SamplingStrategy(
                op; ϵ=1e-4, protrusionaware=protrusionaware
            ),
        )
        f = MLFMA.samplingfunction(strategy, tree, testspace, trialspace)
        requirements = if protrusionaware
            (
                MLFMA.levelprotrusions(sides[1], testspace),
                MLFMA.levelprotrusions(sides[2], trialspace),
            )
        else
            (nothing, nothing)
        end
        levels = sort!(
            union(collect(H2Trees.levels(sides[1])), collect(H2Trees.levels(sides[2])))
        )
        grids, parents = MLFMA.getsamplingsparentsamplings(
            γ, MLFMA.ϵ(strategy), H2Trees.treewithmorelevels(tree), levels, f
        )

        for (i, level) in enumerate(levels)
            expected = [
                first(
                    MLFMA.Assembly.sampling(
                        γ,
                        MLFMA.ϵ(strategy),
                        side;
                        level=level,
                        samplingbuilder=MLFMA.samplingbuilder(strategy),
                        protrusion=if requirements[j] === nothing
                            false
                        else
                            requirements[j][level]
                        end,
                    ),
                ) for (j, side) in enumerate(sides) if level in H2Trees.levels(side)
            ]
            @test length(grids[i]) == maximum(length, expected)
            i > 1 && @test parents[i] === grids[i - 1]
        end
        for side in sides
            for leaf in MLFMA.leaffarfieldsamplings(side, grids, first(levels))
                level = H2Trees.level(side, leaf.leaf)
                @test leaf.sampling === grids[findfirst(==(level), levels)]
            end
        end
    end

    @testset "the larger test-side requirement wins" begin
        tree = blocktree(fine, fine, 0, 0)
        sides = (H2Trees.testtree(tree), H2Trees.trialtree(tree))
        level = last(H2Trees.levels(sides[1]))
        testlarger = (Dict(level => 0.25), Dict(level => 0.0))
        triallarger = reverse(testlarger)
        testfirst = MLFMA.Assembly.PetrovSamplingFunctor(
            MLFMA.CircularSamplingBuilder{Float64}(), sides, testlarger
        )
        trialfirst = MLFMA.Assembly.PetrovSamplingFunctor(
            MLFMA.CircularSamplingBuilder{Float64}(), sides, triallarger
        )
        nominal = first(
            MLFMA.Assembly.sampling(
                γ,
                1e-4,
                sides[1];
                level=level,
                samplingbuilder=MLFMA.CircularSamplingBuilder{Float64}(),
            ),
        )
        @test length(first(testfirst(γ, 1e-4, tree; level=level))) > length(nominal)
        @test first(testfirst(γ, 1e-4, tree; level=level)) ===
            first(trialfirst(γ, 1e-4, tree; level=level))
    end

    @testset "each grid is constructed once, including parent lookups" begin
        tree = blocktree(fine, fine, 0, 0)
        builder = CountingBuilder(MLFMA.CircularSamplingBuilder{Float64}(), Ref(0))
        sides = (H2Trees.testtree(tree), H2Trees.trialtree(tree))
        f = MLFMA.Assembly.PetrovSamplingFunctor(builder, sides, (nothing, nothing))
        levels = collect(H2Trees.levels(sides[1]))
        grids, parents = MLFMA.getsamplingsparentsamplings(γ, 1e-4, sides[1], levels, f)
        @test builder.calls[] == length(levels)
        for i in 2:length(levels)
            @test parents[i] === grids[i - 1]
        end
    end
end
end
