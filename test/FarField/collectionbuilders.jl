using MLFMA
using BEAST, CompScienceMeshes
using H2Factory
using H2Trees
using StaticArrays
using Test
using LinearAlgebra

include("../support/util_test.jl")

# The builder protocol: one construction-policy object per field collection, replacing the four
# independent arguments (`op`'s side, `FarFieldAssembler=`, `quadstrat=`, `FieldCollectionType=`)
# that used to have to agree with each other.
#
# The load-bearing claim is that routing through a builder changed NOTHING about the collections
# produced -- so every assertion below compares against the low-level assembler-based constructor,
# which is the one path the builders themselves call and the one the removed keyword-taking
# constructors called too.

@testset "Field collection sides" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    @test MLFMA.RadiationSide() isa MLFMA.AbstractFieldCollectionSide
    @test MLFMA.ReceiveSide() isa MLFMA.AbstractFieldCollectionSide
    @test MLFMA.LazyReceiveSide() isa MLFMA.AbstractFieldCollectionSide

    # The side resolves BOTH halves of what used to be two separate arguments, and resolves them to
    # the same functions the generated constructors were handed by name.
    @test MLFMA.fieldoperator(MLFMA.RadiationSide(), efie) == MLFMA.radiationoperator(efie)
    @test MLFMA.fieldoperator(MLFMA.ReceiveSide(), efie) == MLFMA.receiveoperator(efie)

    F = MLFMA.farfieldtype(efie)
    @test MLFMA.collectiontype(MLFMA.RadiationSide(), F) ===
        MLFMA.radiationcollectiontype(F)
    @test MLFMA.collectiontype(MLFMA.ReceiveSide(), F) === MLFMA.receivecollectiontype(F)

    # non-vacuity: the two sides genuinely differ, so pairing one side's operator with the other
    # side's collection type -- exactly what two independent arguments allowed -- was a real hazard
    # and not a distinction without a difference.
    @test MLFMA.radiationoperator(efie) != MLFMA.receiveoperator(efie)
    @test MLFMA.radiationcollectiontype(F) !== MLFMA.receivecollectiontype(F)

    # `LazyReceiveSide` assembles nothing, so it deliberately has neither method. A `MethodError`
    # rather than a fallback that quietly hands back the radiation operator: there is no assembly on
    # that side to hand an operator to.
    @test_throws MethodError MLFMA.fieldoperator(MLFMA.LazyReceiveSide(), efie)
    @test_throws MethodError MLFMA.collectiontype(MLFMA.LazyReceiveSide(), F)
end

@testset "Builders carry the side, the assembler and the quadrature" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(1 * λ, λ / 6))

    quad = BEAST.SingleNumQStrat(3)
    builder = MLFMA.BEASTFieldCollectionBuilder(
        MLFMA.RadiationSide(), MLFMA.BEASTFarFieldAssemblerBuilder(), quad
    )

    @test MLFMA.side(builder) === MLFMA.RadiationSide()
    @test MLFMA.assemblerbuilder(builder) === MLFMA.BEASTFarFieldAssemblerBuilder()
    @test MLFMA.quadstrategy(builder) === quad
    @test MLFMA.side(MLFMA.LazyReceiveFieldCollectionBuilder()) === MLFMA.LazyReceiveSide()

    # THE LAZY BUILDER IS NOT AN ASSEMBLED ONE, and the type hierarchy has to say so. It is
    # fieldless by design: no assembler, no quadrature. When the shared accessors briefly sat on the
    # common supertype it inherited all three, so asking for its quadrature failed with "type has no
    # field quadstrategy" instead of saying the operation is not part of its protocol. A `MethodError`
    # names the operation; a `FieldError` names an implementation detail of a struct the caller never
    # asked about.
    #
    # `side` is the one it DOES answer, above, because it states its side rather than storing one.
    lazy = MLFMA.LazyReceiveFieldCollectionBuilder()
    @test_throws MethodError MLFMA.assemblerbuilder(lazy)
    @test_throws MethodError MLFMA.quadstrategy(lazy)
    @test_throws MethodError MLFMA.buildfarfieldassembler(lazy, efie, X)

    # And the hierarchy that produces those: only the assembler-owning builders carry the defaults.
    @test MLFMA.LazyReceiveFieldCollectionBuilder <: MLFMA.AbstractFieldCollectionBuilder
    @test !(
        MLFMA.LazyReceiveFieldCollectionBuilder <:
        MLFMA.AbstractAssembledFieldCollectionBuilder
    )
    @test MLFMA.BEASTFieldCollectionBuilder <: MLFMA.AbstractAssembledFieldCollectionBuilder

    # The assembler the builder builds is the one the three parts describe -- this is the single
    # point where the side, the assembler builder and the quadrature meet, so it is where a
    # mismatch between them would show.
    assembler = MLFMA.buildfarfieldassembler(builder, efie, X)
    @test assembler isa MLFMA.BEASTFarFieldAssembler
    @test MLFMA.operator(assembler) == MLFMA.radiationoperator(efie)
    @test MLFMA.quadstrategy(assembler) === quad
    @test MLFMA.space(assembler) === X
    @test MLFMA.farfieldtype(assembler) === MLFMA.farfieldtype(efie)

    # The RECEIVE side of the same builder shape assembles the receive operator, which is the
    # distinction the `FieldCollectionType=` keyword used to have to be kept in step with by hand.
    receiveassembler = MLFMA.buildfarfieldassembler(
        MLFMA.BEASTFieldCollectionBuilder(
            MLFMA.ReceiveSide(), MLFMA.BEASTFarFieldAssemblerBuilder(), quad
        ),
        efie,
        X,
    )
    @test MLFMA.operator(receiveassembler) == MLFMA.receiveoperator(efie)

    # The assembler builder is a genuine choice, not a fixed detail: swapping it swaps the assembler
    # and nothing else.
    dummyassembler = MLFMA.buildfarfieldassembler(
        MLFMA.BEASTFieldCollectionBuilder(
            MLFMA.RadiationSide(), MLFMA.DummyFarFieldAssemblerBuilder(), quad
        ),
        efie,
        X,
    )
    @test dummyassembler isa MLFMA.DummyFarFieldAssembler
    @test MLFMA.operator(dummyassembler) == MLFMA.radiationoperator(efie)
    @test MLFMA.quadstrategy(dummyassembler) === quad
end

@testset "buildfieldcollection matches the assembler-based constructor" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(1 * λ, λ / 6))
    sampling = MLFMA.SphericalSampling{Float64}(6, 8)
    quad = BEAST.SingleNumQStrat(3)

    for (sideobj, operatorof, collectiontypeof, lowlevel) in (
        (
            MLFMA.RadiationSide(),
            MLFMA.radiationoperator,
            MLFMA.radiationcollectiontype,
            MLFMA.RadiationFieldCollection,
        ),
        (
            MLFMA.ReceiveSide(),
            MLFMA.receiveoperator,
            MLFMA.receivecollectiontype,
            MLFMA.ReceiveFieldCollection,
        ),
    )
        builder = MLFMA.BEASTFieldCollectionBuilder(
            sideobj, MLFMA.BEASTFarFieldAssemblerBuilder(), quad
        )
        built = MLFMA.buildfieldcollection(builder, efie, sampling, X)

        # The type the side resolves to...
        @test built isa collectiontypeof(MLFMA.farfieldtype(efie))

        # ...and the DATA, against the low-level path built by hand from the same four parts. Equal
        # to the last bit: the assembly is deterministic, so anything short of exact equality would
        # be hiding a difference rather than tolerating one.
        reference = lowlevel(
            MLFMA.BEASTFarFieldAssembler(
                operatorof(efie),
                X,
                quad,
                MLFMA.farfieldstorerule(efie),
                MLFMA.farfieldtype(efie),
            ),
            sampling,
        )
        @test typeof(built) === typeof(reference)
        @test MLFMA.Fθ(built) == MLFMA.Fθ(reference)
        @test MLFMA.Fφ(built) == MLFMA.Fφ(reference)
    end

    # non-vacuity: the two sides produce DIFFERENT numbers, so the equality checks above are
    # discriminating rather than comparing a value against itself.
    radiation = MLFMA.buildfieldcollection(
        MLFMA.BEASTFieldCollectionBuilder(
            MLFMA.RadiationSide(), MLFMA.BEASTFarFieldAssemblerBuilder(), quad
        ),
        efie,
        sampling,
        X,
    )
    receive = MLFMA.buildfieldcollection(
        MLFMA.BEASTFieldCollectionBuilder(
            MLFMA.ReceiveSide(), MLFMA.BEASTFarFieldAssemblerBuilder(), quad
        ),
        efie,
        sampling,
        X,
    )
    @test MLFMA.Fθ(radiation) != MLFMA.Fθ(receive)

    # The quadrature is really the builder's, not a default reached past it.
    coarse = MLFMA.buildfieldcollection(
        MLFMA.BEASTFieldCollectionBuilder(
            MLFMA.RadiationSide(),
            MLFMA.BEASTFarFieldAssemblerBuilder(),
            BEAST.SingleNumQStrat(1),
        ),
        efie,
        sampling,
        X,
    )
    @test MLFMA.Fθ(coarse) != MLFMA.Fθ(radiation)
end

@testset "getfieldcollectionsdict routes both sides through one builder protocol" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(1 * λ, λ / 6))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 6, minvalues=0))
    strategy = stubbedstrategy(efie, X)

    plans = H2Factory.assembleplans(tree, MLFMA.nearfieldstrategy(strategy))
    minrelevantlevel = plans.relevantlevels[begin]
    samplings, _ = MLFMA.getsamplingsparentsamplings(
        MLFMA.γ(efie),
        MLFMA.ϵ(strategy),
        tree,
        plans.relevantlevels,
        MLFMA.samplingfunction(strategy),
    )
    # THE ORDER THE PRODUCTION PATH INSTALLS. `getfieldcollectionsdict` pins it with a `sort`
    # precisely so that a change to how `TreeIndex` numbers nodes reorders nothing downstream; see
    # the comment at its `leafids = sort(...)`. This compared against `H2Trees.leaves(tree)` while
    # the two happened to agree, and started failing when node numbering moved to a Hilbert walk.
    # The production path was already correct.
    #
    # Deliberately NOT asserted here: that the tree's own walk order differs from the sorted one.
    # That is a fact about H2Trees, not about MLFMA, and pinning it would make this file fail if
    # upstream ever restored ascending ids, with nothing here actually wrong. The contract below is
    # stated against `sort` and against `issorted`, both of which hold whatever order the tree walk
    # returns.
    leaves = sort(H2Trees.leaves(tree))

    radiationdict = MLFMA.getfieldcollectionsdict(
        MLFMA.radiationfieldcollectionbuilder(strategy),
        efie,
        X,
        tree,
        samplings,
        minrelevantlevel,
    )

    @test collect(keys(radiationdict)) == leaves
    # The contract itself, independent of what the tree walk returns: the keys come out ascending,
    # and they are every leaf, exactly once.
    @test issorted(collect(keys(radiationdict)))
    @test Set(collect(keys(radiationdict))) == Set(H2Trees.leaves(tree))
    @test length(collect(keys(radiationdict))) == length(H2Trees.leaves(tree))
    @test all(
        c isa MLFMA.radiationcollectiontype(MLFMA.farfieldtype(efie)) for c in radiationdict
    )

    # The LAZY side takes the radiation dictionary in the `source` position -- same protocol, same
    # argument order, different input, with the builder type saying which.
    lazydict = MLFMA.getfieldcollectionsdict(
        MLFMA.lazyreceivefieldcollectionbuilder(strategy),
        efie,
        radiationdict,
        tree,
        samplings,
        minrelevantlevel,
    )

    @test collect(keys(lazydict)) == leaves
    @test all(c isa MLFMA.LazyReceiveFieldCollection for c in lazydict)

    # Each lazy collection wraps ITS OWN leaf's radiation collection -- by identity, so a shifted or
    # reversed pairing fails rather than passing on equal-looking numbers.
    for leaf in leaves
        @test MLFMA.farfieldcollection(lazydict[leaf]) === radiationdict[leaf]
        @test MLFMA.operator(lazydict[leaf]) === efie
    end

    # CORRECTNESS BY VALUE: its own level's mirror indices, equal to the ones that level's sampling
    # produces. What this pins is that the indices come from the SAME per-leaf recipe the radiation
    # collections were assembled against, rather than from a separately recomputed level lookup.
    for (i, leaf) in enumerate(leaves)
        levelindex = MLFMA._leaflevelindex(tree, leaf, minrelevantlevel, samplings)
        @test MLFMA.mirrorindices(lazydict[leaf]) ==
            MLFMA.mirrorindices(samplings[levelindex])
    end

    # SHARING BY IDENTITY within a level. `mirrorindices` depends only on the direction grid, so
    # building it per leaf produced one identical array per leaf -- 10k arrays on a 10k-leaf level,
    # each retained by its own collection. Leaves on one level must now hold the SAME object.
    #
    # `==` above and `===` here are both load-bearing and neither implies the other: equal-by-value
    # would pass on duplicated arrays, and shared-by-identity would pass on a shared WRONG array.
    bylevel = Dict{Int,Vector{eltype(leaves)}}()
    for leaf in leaves
        push!(get!(bylevel, H2Trees.level(tree, leaf), eltype(leaves)[]), leaf)
    end

    for (level, levelleaves) in bylevel
        reference = MLFMA.mirrorindices(lazydict[levelleaves[begin]])
        for leaf in levelleaves
            @test MLFMA.mirrorindices(lazydict[leaf]) === reference
        end
    end

    # non-vacuity: at least one level really does hold several leaves, so the identity assertions
    # above compare distinct collections rather than a leaf with itself.
    @test maximum(length, values(bylevel)) > 1

    # ...and the total number of DISTINCT index arrays is EXACTLY the number of distinct leaf levels,
    # not the leaf count -- which is the property the sharing exists for, stated as the cache's own
    # key so it cannot drift from the implementation.
    #
    # Deliberately `==` and not `<=`: an upper bound alone is satisfied by sharing MORE than the key
    # allows (one array for the whole tree), which on a non-uniform tree would mean a level mirroring
    # against another level's grid.
    distinctarrays = length(unique(objectid, [MLFMA.mirrorindices(c) for c in lazydict]))
    @test distinctarrays == length(bylevel)
    @test distinctarrays < length(leaves)

    # The RECEIVE side of the same protocol assembles instead of mirroring -- what Petrov uses.
    receivedict = MLFMA.getfieldcollectionsdict(
        MLFMA.receivefieldcollectionbuilder(strategy),
        efie,
        X,
        tree,
        samplings,
        minrelevantlevel,
    )
    @test all(
        c isa MLFMA.receivecollectiontype(MLFMA.farfieldtype(efie)) for c in receivedict
    )

    # `FieldCollectionType=` is gone: the builder says which collection is produced, and the keyword
    # that used to decide it is REJECTED rather than ignored.
    #
    # Pinned here, and with EVERY positional argument correct, because that is the only way the throw
    # can be about the keyword. The same assertion with a short argument list -- which is how it was
    # first written -- throws on arity and would keep passing even if the new builder-shaped method
    # started swallowing `FieldCollectionType`, i.e. it would pin nothing.
    @test_throws MethodError MLFMA.getfieldcollectionsdict(
        MLFMA.radiationfieldcollectionbuilder(strategy),
        efie,
        X,
        tree,
        samplings,
        minrelevantlevel;
        FieldCollectionType=MLFMA.RadiationFieldCollections,
    )
    # non-vacuity: those exact positional arguments, minus the keyword, are ACCEPTED -- so the throw
    # above is the keyword and not the fixture. (Same call as `radiationdict` above; repeated rather
    # than reused so the pair reads as one claim.)
    accepted = MLFMA.getfieldcollectionsdict(
        MLFMA.radiationfieldcollectionbuilder(strategy),
        efie,
        X,
        tree,
        samplings,
        minrelevantlevel,
    )
    @test collect(keys(accepted)) == leaves
end

@testset "A strategy hands out builders that cannot disagree" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(1 * λ, λ / 8))

    quad = BEAST.SingleNumQStrat(2)
    strategy = MLFMA.MLFMAStrategy(
        efie,
        X;
        farfieldstrategy=MLFMA.FarFieldStrategy(;
            assemblerbuilder=MLFMA.DummyFarFieldAssemblerBuilder(), quadstrategy=quad
        ),
    )

    radiation = MLFMA.radiationfieldcollectionbuilder(strategy)
    receive = MLFMA.receivefieldcollectionbuilder(strategy)

    @test MLFMA.side(radiation) === MLFMA.RadiationSide()
    @test MLFMA.side(receive) === MLFMA.ReceiveSide()

    # THE POINT OF DERIVING THE BUILDERS FROM ONE `FarFieldStrategy` rather than storing three:
    # the trial and test sides cannot end up with different assemblers or different quadratures,
    # which independent keywords at two call sites allowed and which no test could have caught
    # because both maps would still assemble successfully.
    @test MLFMA.assemblerbuilder(radiation) === MLFMA.assemblerbuilder(receive)
    @test MLFMA.quadstrategy(radiation) === MLFMA.quadstrategy(receive)
    @test MLFMA.assemblerbuilder(radiation) === MLFMA.DummyFarFieldAssemblerBuilder()
    @test MLFMA.quadstrategy(radiation) === quad

    # non-vacuity: those are the values this strategy was BUILT with, not the defaults, so the
    # accessors really do read through to `FarFieldStrategy`.
    @test MLFMA.assemblerbuilder(radiation) !== MLFMA.FARFIELDASSEMBLERBUILDERRECOM
    @test MLFMA.quadstrategy(radiation) != MLFMA.FARFIELDQUADSTRATEGYRECOM
end

# The end of the chain: what the two map constructors actually END UP HOLDING. The testsets above
# check the builders in isolation; these check that each constructor asks for the right pair, which
# is the thing the `FieldCollectionType=` keyword used to decide by hand at four call sites.
@testset "Galerkin and Petrov maps hold the collections their builders produce" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    # 1λ / λ8, the same fixture size as test/Maps/comparison.jl: coarser meshes leave the default
    # builder with a 2-level tree, which the map constructor rejects outright.
    X = raviartthomas(meshsphere(1 * λ, λ / 8))
    strategy = stubbedstrategy(efie, X)

    RadiationCollection = MLFMA.radiationcollectiontype(MLFMA.farfieldtype(efie))
    ReceiveCollection = MLFMA.receivecollectiontype(MLFMA.farfieldtype(efie))

    # GALERKIN: radiation trial side, LAZY receive test side -- the test patterns are the trial
    # patterns mirrored, never assembled a second time.
    galerkin = MLFMA.assemble(efie, X; strategy=strategy)
    @test all(c isa RadiationCollection for c in H2Factory.trialmomentcollections(galerkin))
    @test all(
        c isa MLFMA.LazyReceiveFieldCollection for
        c in H2Factory.testmomentcollections(galerkin)
    )
    # and the mirroring is by IDENTITY: every lazy test collection wraps the very trial collection
    # the same leaf radiates with, which is what makes the second assembly unnecessary rather than
    # merely redundant
    trials = H2Factory.trialmomentcollections(galerkin)
    for leaf in keys(trials)
        @test MLFMA.farfieldcollection(H2Factory.testmomentcollections(galerkin)[leaf]) ===
            trials[leaf]
    end

    # PETROV: radiation trial side, ASSEMBLED receive test side. Same space on both sides here (a
    # legitimate Petrov construction, see test/Maps/comparison.jl), which isolates the builder
    # choice -- the sides differ because the BUILDERS differ, not because the spaces do.
    petrov = MLFMA.assemble(efie, X, X; strategy=strategy)
    @test all(c isa RadiationCollection for c in H2Factory.trialmomentcollections(petrov))
    @test all(c isa ReceiveCollection for c in H2Factory.testmomentcollections(petrov))

    # non-vacuity: the two maps really do differ on the test side, so the assertions above
    # distinguish the constructors rather than restating one fact twice.
    @test !(ReceiveCollection <: MLFMA.LazyReceiveFieldCollection)
end

# The flip side of item 6: the keyword-taking constructors are GONE, not deprecated. A caller still
# passing `quadstrat=`/`FarFieldAssembler=`, or still calling the `(op, sampling, space)` form, has
# to find out -- otherwise they get a collection built with defaults that quietly ignores what they
# asked for, which is precisely the desynchronization the builder exists to remove.
@testset "The removed field-collection construction surface is rejected" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    X = raviartthomas(meshsphere(1 * λ, λ / 6))
    sampling = MLFMA.SphericalSampling{Float64}(6, 8)

    for collection in (MLFMA.RadiationFieldCollection, MLFMA.ReceiveFieldCollection)
        # the whole `(op, sampling, space)` form, with and without the keywords
        @test_throws MethodError collection(efie, sampling, X)
        @test_throws MethodError collection(
            efie, sampling, X; quadstrat=BEAST.SingleNumQStrat(3)
        )
        @test_throws MethodError collection(
            efie, sampling, X; FarFieldAssembler=MLFMA.BEASTFarFieldAssembler
        )
    end

    # non-vacuity: the LOW-LEVEL form these names kept is accepted, so the throws above are about
    # the removed surface and not about the fixture.
    assembler = MLFMA.BEASTFarFieldAssembler(
        MLFMA.radiationoperator(efie),
        X,
        BEAST.SingleNumQStrat(3),
        MLFMA.farfieldstorerule(efie),
        MLFMA.farfieldtype(efie),
    )
    @test MLFMA.RadiationFieldCollection(assembler, sampling) isa
        MLFMA.radiationcollectiontype(MLFMA.farfieldtype(efie))
end
# `getfieldcollectionsdict`'s removed `FieldCollectionType=` keyword is pinned in the
# "routes both sides through one builder protocol" testset above rather than here, because rejecting
# it can only be demonstrated with a call whose positional arguments are otherwise VALID -- and that
# needs the tree/samplings fixture only that testset builds.
