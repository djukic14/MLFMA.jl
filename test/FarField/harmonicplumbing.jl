using Dictionaries

struct HarmonicOrderTree
    levels::Dict{Int,Int}
end
H2Trees.level(tree::HarmonicOrderTree, node::Int) = tree.levels[node]

struct HarmonicPoolPlan
    aggregation::Vector{Int}
    disaggregation::Vector{Int}
end
H2Trees.aggregationlevels(plan::HarmonicPoolPlan) = plan.aggregation
H2Trees.disaggregationlevels(plan::HarmonicPoolPlan) = plan.disaggregation

@testset "Harmonic order lookup" begin
    tree = HarmonicOrderTree(Dict(10 => 1, 20 => 2, 21 => 2, 30 => 3))
    samplings = [
        MLFMA.SphericalSampling{Float64}(8, 16), MLFMA.SphericalSampling{Float64}(10, 20)
    ]
    center = SVector(0.0, 0.0, 0.0)
    makers = (
        P -> MLFMA.SphericalHarmonicRadiationCollection(
            SphericalHarmonicStorage(P, center, zeros(ComplexF64, 3(P + 1)^2, 1))
        ),
        P -> MLFMA.VectorSphericalHarmonicRadiationCollection(
            MLFMA.VectorSphericalHarmonicStorage(
                P, center, zeros(ComplexF64, 2 * P * (P + 2), 1)
            ),
        ),
        P -> MLFMA.CircularHarmonicRadiationCollection(
            MLFMA.CircularHarmonics.CircularHarmonicStorage(
                P, SVector(0.0, 0.0), zeros(ComplexF64, 2P + 1, 1)
            ),
        ),
    )
    for (lookup, make) in zip((MLFMA.sheorders, MLFMA.vshorders, MLFMA.cheorders), makers)
        collections = Dict(10 => make(2), 20 => make(2), 21 => make(2), 30 => make(3))
        orders = lookup(tree, collections, samplings, 2)
        @test orders isa IdDict
        @test length(orders) == 2
        @test orders[samplings[1]] == 2
        @test orders[samplings[2]] == 3
        @test isempty(lookup(tree, empty(collections), samplings, 2))
        collections[21] = make(3)
        @test_throws ErrorException lookup(tree, collections, samplings, 2)
        @test_throws ErrorException lookup(tree, Dict(30 => make(3)), samplings[1:1], 2)
    end
end

@testset "Harmonic scratch pools" begin
    plan = HarmonicPoolPlan([2], [3])
    plans = (
        trialaggregationplan=plan,
        testaggregationplan=plan,
        testdisaggregationplan=plan,
        trialdisaggregationplan=plan,
    )
    spherical = [
        MLFMA.SphericalSampling{Float64}(8, 16), MLFMA.SphericalSampling{Float64}(10, 20)
    ]
    circular = [MLFMA.CircularSampling{Float64}(16), MLFMA.CircularSampling{Float64}(20)]
    for (builder, samplings) in (
        (MLFMA.shescratchchannel, spherical),
        (MLFMA.vshscratchchannel, spherical),
        (
            (args...; kwargs...) ->
                MLFMA.chescratchchannel(args...; ncomponents=2, kwargs...),
            circular,
        ),
    )
        orders = IdDict(samplings[1] => 2, samplings[2] => 3)
        # Level 1 has neither a sampling order nor a chunk count: it must be skipped.
        H2Factory.withnchunks(Dict(2 => 2, 3 => 1)) do
            build() =
                builder(orders, [1, 2, 3], plans, samplings, 2; groupsperchunk=l -> l + 1)
            pools, other = build(), build()
            @test Set(keys(pools)) == Set([2, 3])
            for level in (2, 3)
                firstscratch = H2Factory.borrowbuffer(
                    pools, level, 1, H2Factory.isForward()
                )
                @test MLFMA.maxgroups(firstscratch) == level + 1
                @test firstscratch !==
                    H2Factory.borrowbuffer(other, level, 1, H2Factory.isForward())
                if level == 2
                    second = H2Factory.borrowbuffer(pools, level, 2, H2Factory.isForward())
                    @test firstscratch.coefficients !== second.coefficients
                    @test !Base.mightalias(firstscratch.coefficients, second.coefficients)
                end
            end
        end
    end
end
