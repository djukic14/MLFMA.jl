using MLFMA
using BEAST, CompScienceMeshes
using H2Factory
using H2Trees
using Random
using StaticArrays
using Test
using LinearAlgebra

include("../support/util_test.jl")

# ONE LEAF ORDER, FOLLOWED BY EVERYTHING.
#
# The tree assembly path builds five things per leaf -- the moment blocks `Fs`, the per-leaf
# sampling recipe, the assembly clusters, the storer's leaf-local index maps, and the leaf-keyed
# collection dictionary. Each used to derive its own order from its own `H2Trees.leaves(tree)` call.
# They agreed only because that call is deterministic; a disagreement would have assembled one
# leaf's far field into another leaf's slot, silently.
#
# ON NON-VACUITY, because it shapes every test below. The obvious check, find a tree where
# `sort(leaves(tree)) != leaves(tree)` so that pinning the order visibly reorders something, used to
# be impossible to write: `H2Trees.leaves(tree)` returned ascending ids on every fixture measured
# (balanced, adaptive, 4-level uneven-density, and both `BlockTree` subtrees), so `sort` was a no-op.
# That was a property of how `DepthFirstIterator` happened to number those trees, never a guarantee,
# and it no longer holds now that `TreeIndex` numbers nodes along a Hilbert walk. The last assertion
# in the first testset is that check, and it now has teeth.
#
# So the ordering tests here drive the builders with a DELIBERATELY PERMUTED leaf order instead.
# That is strictly stronger than a fixture that merely happens to be unsorted: it proves every
# per-leaf array follows the order it is given, for an order nothing else could have produced by
# accident.

@testset "_indexchunks partitions basis-function ids" begin
    # The non-tree path's "clusters" are a work partition, not boxes. What matters is that the
    # partition is exact -- the hand-rolled `round`-the-endpoints version it replaced had three
    # separate chances at an off-by-one.
    for (nitems, nchunks) in
        ((10, 4), (3, 4), (1, 1), (997, 7), (64, 8), (5, 1), (2, 16), (1000, 3))
        chunks = MLFMA._indexchunks(nitems, nchunks)

        # exact cover of 1:nitems, no duplicates, no gaps
        @test sort(vcat(chunks...)) == collect(1:nitems)
        # contiguous, ascending
        @test all(chunk == collect(chunk[begin]:chunk[end]) for chunk in chunks)
        # never more chunks than asked for, and never an empty one
        @test length(chunks) <= max(nchunks, 1)
        @test all(!isempty, chunks)
    end

    # degenerate: no basis functions is no work, not one empty chunk
    @test MLFMA._indexchunks(0, 4) == Vector{Int}[]

    # fewer items than chunks gives one item each, not trailing empties
    @test MLFMA._indexchunks(3, 8) == [[1], [2], [3]]
end

@testset "leaffarfieldsamplings takes the leaf order and carries it" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    strategy = stubbedstrategy(efie, X)
    tree = TwoNTree(X)

    plans = H2Factory.assembleplans(tree, MLFMA.nearfieldstrategy(strategy))
    minrel = plans.relevantlevels[begin]
    samplings, _ = MLFMA.getsamplingsparentsamplings(
        MLFMA.γ(efie),
        MLFMA.ϵ(strategy),
        tree,
        plans.relevantlevels,
        MLFMA.samplingfunction(strategy),
    )

    # default is the tree's own order, so callers that do not care are unaffected
    default = MLFMA.leaffarfieldsamplings(tree, samplings, minrel)
    @test MLFMA.leafids(default) == H2Trees.leaves(tree)

    # ...and an explicit order is honoured verbatim, including one nothing else would produce
    permuted = shuffle(MersenneTwister(20240815), sort(H2Trees.leaves(tree)))
    @test permuted != H2Trees.leaves(tree)   # non-vacuity of the permutation itself
    explicit = MLFMA.leaffarfieldsamplings(tree, samplings, minrel; leaves=permuted)
    @test MLFMA.leafids(explicit) == permuted

    # each entry still describes ITS leaf -- centre and grid follow the id, not the position
    for (i, leaf) in enumerate(permuted)
        @test explicit[i].leaf == leaf
        @test explicit[i].center == H2Trees.center(tree, leaf)
        levelindex = MLFMA._leaflevelindex(tree, leaf, minrel, samplings)
        @test explicit[i].sampling === samplings[levelindex]
    end

    # THE CONTRACT, stated against MLFMA rather than against H2Trees. The production path
    # (`getfieldcollectionsdict`) installs SORTED leaf order, and every per-leaf array reads its
    # order back off `leafids` instead of calling `H2Trees.leaves` again. That is MLFMA's decision,
    # so it holds whatever order the tree walk happens to return.
    #
    # An earlier version of this asserted that the walk order and the sorted order DIFFER, which was
    # a mistake: it pinned a fact about H2Trees' traversal, so restoring ascending ids upstream would
    # have failed this file with nothing here actually wrong. Non-vacuity comes from the explicit
    # permutation above, which is stronger anyway: it drives the builder with an order nothing else
    # could have produced.
    sorted = MLFMA.leaffarfieldsamplings(
        tree, samplings, minrel; leaves=sort(H2Trees.leaves(tree))
    )
    @test issorted(MLFMA.leafids(sorted))
    @test Set(MLFMA.leafids(sorted)) == Set(H2Trees.leaves(tree))
    @test length(MLFMA.leafids(sorted)) == length(H2Trees.leaves(tree))
end

# THE ORDERING TEST THAT IS NOT VACUOUS. Assemble the same tree twice, once in the natural order and
# once in a shuffled one, and require every leaf's far field to come out identical. Any per-leaf
# array that failed to follow the given order puts one leaf's pattern in another leaf's slot, and
# these blocks differ per leaf (different basis functions, and `DummyFarFieldAssembler` is
# deliberately distinct in all three indices), so a mismatch cannot cancel.
@testset "Assembly is invariant under the leaf order it is given" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    strategy = stubbedstrategy(efie, X)
    tree = TwoNTree(X)

    plans = H2Factory.assembleplans(tree, MLFMA.nearfieldstrategy(strategy))
    minrel = plans.relevantlevels[begin]
    samplings, _ = MLFMA.getsamplingsparentsamplings(
        MLFMA.γ(efie),
        MLFMA.ϵ(strategy),
        tree,
        plans.relevantlevels,
        MLFMA.samplingfunction(strategy),
    )
    builder = MLFMA.radiationfieldcollectionbuilder(strategy)

    leafids = sort(H2Trees.leaves(tree))
    permuted = shuffle(MersenneTwister(987654), leafids)
    @test permuted != leafids

    # production entry point, natural order
    natural = MLFMA.getfieldcollectionsdict(builder, efie, X, tree, samplings, minrel)
    @test collect(keys(natural)) == leafids

    # same thing through the same threading, permuted
    ls = MLFMA.leaffarfieldsamplings(tree, samplings, minrel; leaves=permuted)
    shuffled = MLFMA.Dictionary(
        MLFMA.leafids(ls), MLFMA.buildfieldcollections(builder, efie, tree, ls, X)
    )
    @test collect(keys(shuffled)) == permuted

    for leaf in leafids
        # EXACT equality: the assembly is deterministic, so anything short of it would be tolerating
        # a difference rather than proving there is none.
        @test MLFMA.Fθ(natural[leaf]) == MLFMA.Fθ(shuffled[leaf])
        @test MLFMA.Fφ(natural[leaf]) == MLFMA.Fφ(shuffled[leaf])

        # the block's width is the leaf's own basis-function count -- an ordering slip between `Fs`
        # and the clusters shows up here even when two leaves happen to hold equal patterns
        @test size(MLFMA.Fθ(shuffled[leaf]), 2) == length(H2Trees.values(tree, leaf))
    end

    # non-vacuity for "a mismatch could not cancel": neighbouring leaves really do hold different
    # patterns, so comparing leaf-by-leaf is discriminating.
    distinctwidths = unique(length(H2Trees.values(tree, leaf)) for leaf in leafids)
    @test length(distinctwidths) > 1 ||
        MLFMA.Fθ(natural[leafids[1]]) != MLFMA.Fθ(natural[leafids[2]])

    # THE LAZY SIDE follows the same order: each collection wraps the radiation collection of ITS
    # leaf, by identity, under a permutation that would expose an index slip.
    lazy = MLFMA.Dictionary(
        MLFMA.leafids(ls),
        MLFMA.buildfieldcollections(
            MLFMA.LazyReceiveFieldCollectionBuilder(), efie, tree, ls, shuffled
        ),
    )
    for leaf in leafids
        @test MLFMA.farfieldcollection(lazy[leaf]) === shuffled[leaf]
    end

    # ...and mirror indices are still shared once per level under the permuted order, i.e. the
    # sharing is keyed by the leaf's level and not by its position in the assembly order
    bylevel = Dict{Int,Vector{Int}}()
    for leaf in permuted
        push!(get!(bylevel, H2Trees.level(tree, leaf), Int[]), leaf)
    end
    for (_, levelleaves) in bylevel
        reference = MLFMA.mirrorindices(lazy[levelleaves[begin]])
        for leaf in levelleaves
            @test MLFMA.mirrorindices(lazy[leaf]) === reference
        end
    end
    @test length(unique(objectid, [MLFMA.mirrorindices(c) for c in lazy])) == length(bylevel)
end

@testset "The storer's index maps follow the given leaf order" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    strategy = stubbedstrategy(efie, X)
    tree = TwoNTree(X)

    plans = H2Factory.assembleplans(tree, MLFMA.nearfieldstrategy(strategy))
    minrel = plans.relevantlevels[begin]
    samplings, _ = MLFMA.getsamplingsparentsamplings(
        MLFMA.γ(efie),
        MLFMA.ϵ(strategy),
        tree,
        plans.relevantlevels,
        MLFMA.samplingfunction(strategy),
    )

    permuted = shuffle(MersenneTwister(555), sort(H2Trees.leaves(tree)))
    ls = MLFMA.leaffarfieldsamplings(tree, samplings, minrel; leaves=permuted)
    maps = MLFMA._indexinleafbox(tree, MLFMA.leafids(ls))

    @test length(maps) == length(permuted)

    # `maps[cluster]` must renumber the basis functions of the leaf that `leafsamplings[cluster]`
    # describes -- this is the array the storer indexes with the same `cluster` it uses for
    # `Fs[cluster]`, so a mismatch here writes into the wrong block.
    for (i, leaf) in enumerate(permuted)
        functions = H2Trees.values(tree, leaf)
        @test length(maps[i]) == length(functions)
        @test Set(keys(maps[i])) == Set(functions)
        @test Set(values(maps[i])) == Set(1:length(functions))
        for (position, functionid) in enumerate(functions)
            @test maps[i][functionid] == position
        end
    end

    # non-vacuity: under the natural order these maps would be in a DIFFERENT arrangement, so the
    # assertions above are about the order actually being followed and not about it being ignorable.
    naturalmaps = MLFMA._indexinleafbox(tree, sort(H2Trees.leaves(tree)))
    @test [sort(collect(keys(m))) for m in maps] != [sort(collect(keys(m))) for m in naturalmaps]
end
