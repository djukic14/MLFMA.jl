using MLFMA
using BEAST, CompScienceMeshes
using H2Factory
using Test
using H2Trees

include("../support/util_test.jl")

# Exploratory allocation-vs-struct-size measurement for the box-centered leaf far-field path
# (`MLFMA.BoxFarFieldSampling`/`MLFMA.leaffarfieldsamplings`, `beastfarfieldassembler.jl`) --
# mirrors H2Trees' own performance-contract pattern (`H2Trees/test/performance/allocations.jl`):
# warm up (compile + run once, discarded) before measuring so `@timed` never counts JIT
# compilation, then report `allocated_bytes / Base.summarysize(result)`.
#
# No hard budget is asserted here yet -- unlike H2Trees' `budgets.jl`, there is no prior measured
# baseline for this operation to set one from. This is deliberately a reporting/diagnostic test
# (via `@info`) to gather that baseline first; a real regression budget can be added once we know
# what a normal ratio looks like here.
function perf_measure(f, args...)
    f(args...)
    f(args...)
    stats = @timed f(args...)
    return stats.bytes, stats.value
end

function farfieldperf_leafsamplings(tree, samplings, minrelevantlevel)
    return MLFMA.leaffarfieldsamplings(tree, samplings, minrelevantlevel)
end

function farfieldperf_collectionsdict(
    operator, space, tree, samplings, minrelevantlevel, quadstrat
)
    return MLFMA.getfieldcollectionsdict(
        operator,
        space,
        tree,
        samplings,
        minrelevantlevel,
        quadstrat,
        MLFMA.BEASTFarFieldAssembler,
        false,
    )
end

@testset "Leaf far-field allocations vs BoxFarFieldSampling size" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ

    # Sized to give a genuinely large number of leaves (so per-leaf struct overhead actually
    # shows up) without paying for near-interaction assembly at all -- this benchmark only calls
    # the leaf far-field path directly, never `GalerkinMLFMA`/`PetrovGalerkinMLFMA`.
    m = meshsphere(2 * λ, λ / 10)
    X = raviartthomas(m)
    @show numfunctions(X)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    tree = buildtree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=λ / 4, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
        ),
    )
    leaves = collect(H2Trees.leaves(tree))
    minrelevantlevel = H2Trees.level(tree, leaves[begin])
    @test length(unique(H2Trees.level.(Ref(tree), leaves))) == 1 # uniform, per the builder above

    ϵ = 1e-4
    halfsize = H2Trees.halfsize(tree, leaves[begin])
    diameter = MLFMA.diameter(halfsize, 3)
    sampling, _ = MLFMA.sampling(γ, ϵ, diameter, MLFMA.SphericalSamplingBuilder{Float64}())
    samplings = [sampling]

    quadstrat = BEAST.SingleNumQStrat(4)

    leafsamplingsbytes, leafsamplings = perf_measure(
        farfieldperf_leafsamplings, tree, samplings, minrelevantlevel
    )
    structsize = Base.summarysize(leafsamplings)

    @profview collectionsbytes, _ = perf_measure(
        farfieldperf_collectionsdict, efie, X, tree, samplings, minrelevantlevel, quadstrat
    )

    @info "Leaf far-field allocation report" nleaves = length(leaves) numfunctions = numfunctions(
        X
    ) samplingsize = size(sampling) leafsamplings_summarysize = structsize leafsamplings_allocated_bytes =
        leafsamplingsbytes leafsamplings_allocation_ratio = leafsamplingsbytes / structsize full_collections_allocated_bytes =
        collectionsbytes full_collections_allocation_ratio = collectionsbytes / structsize

    @test structsize > 0
    @test collectionsbytes > 0
end
