# SYMMETRY-REDUCED TRANSLATIONS, THROUGH A WHOLE MATVEC.
#
# The pieces below this file were each checked against their own oracle: the lattice group and its
# canonicalization in `H2Trees`, the symmetry side table in `H2Factory`, and the angular action
# (`k̂[π[j]] == S k̂[j]`, plus `T(k̂, Sd) == T(S⁻¹k̂, d)`) in `test/Sampling/symmetryaction.jl`. What
# none of them can see is whether the three agree about WHICH symmetry belongs to which interaction,
# and that is the only question here.
#
# THE FAILURE THIS FILE EXISTS FOR IS SILENT. A symmetry-reduced collection has the right shapes and
# produces a plausible field whatever permutation it applies, so every mistake available, a reversed
# inverse convention, a level off by one, a `couplingkey` swap picking up a neighbouring
# interaction's tag, shows up as a slightly wrong answer and nothing else.
#
# WHY THE BOUND IS 1e-12 AND NOT EXACT EQUALITY. `T(k̂, Sd) == T(S⁻¹k̂, d)` is exact in exact
# arithmetic, but the reduced path evaluates the right-hand side and the unreduced one the left, from
# different floating-point inputs. Measured on the Galerkin fixture: 89 of 106 couplings differ in
# their last bits, worst relative difference 8.3e-15. So exact equality is the wrong assertion,
# it was tried, and it fails on correct code. The bound is still decisive, because a WRONG
# permutation reads a different direction's translation entirely and misses by order one, not by
# 1e-15. Nothing between those two scales is a thing this code can do.
#
# NON-VACUITY, which matters more here than usual. The antipodal map is its own inverse, so on a grid
# that supports only `OppositeSymmetry` a REVERSED inverse convention passes everything. The
# reduction-ratio assertion is what rules that out: a ratio above 2 is only reachable with chiral
# elements, whose permutation and inverse permutation differ.

using MLFMA
using BEAST
using CompScienceMeshes
using H2Factory
using H2Trees
using LinearAlgebra
using Random
using StaticArrays
using Test

include("../support/util_test.jl")

# A grid nobody has written an angular action for. It used to be possible to point the refusal test
# at a real sampling (`EquidistantSampling` was the standing example), but every sampling this
# package ships now has one, which is itself required, since `TRANSLATIONRECOM` is the reduced
# collection and a sampling without an action could not build a default map at all. So the case is
# kept alive with a type that exists only to have no method, rather than deleted along with the
# grid that used to illustrate it.
struct NoActionSampling <: MLFMA.AbstractSampling{Float64} end

@testset "Symmetry-reduced translations" begin
    # THE BASELINE HAS TO BE REQUESTED EXPLICITLY NOW. `TRANSLATIONRECOM` is the symmetry-reduced
    # collection, so a map built without a strategy is no longer the unreduced one, and every
    # comparison in this file would quietly become "reduced against reduced", passing perfectly
    # while testing nothing.
    function unreducedstrategy(operator, spaces...; kwargs...)
        return MLFMA.MLFMAStrategy(
            operator,
            spaces...;
            i2ostrategy=MLFMA.I2OTranslationStrategy(;
                TranslationCollection=MLFMA.TranslationCollection
            ),
            kwargs...,
        )
    end

    function symmetrystrategy(operator, spaces...; kwargs...)
        return MLFMA.MLFMAStrategy(
            operator,
            spaces...;
            i2ostrategy=MLFMA.I2OTranslationStrategy(;
                TranslationCollection=MLFMA.SymmetryTranslationCollection
            ),
            kwargs...,
        )
    end

    # Every stored coupling of `reduced`, compared to `full`'s, plus the reduction the wrapper claims.
    # Returns the offending keys rather than a bool so a failure names an interaction, and the worst
    # relative difference so a regression can be read as a number instead of a pass/fail.
    function comparecollections(reduced, full; atol=1e-12)
        fullkeys = collect(keys(full))
        differences = map(fullkeys) do key
            canonical = collect(full[key])
            return norm(collect(reduced[key]) - canonical) / norm(canonical)
        end
        return (
            npairs=length(fullkeys),
            samekeys=Set(fullkeys) == Set(collect(keys(reduced))),
            mismatches=fullkeys[differences .> atol],
            worstdifference=isempty(differences) ? 0.0 : maximum(differences),
            ratio=H2Factory.translationstats(full).nunique /
                  H2Factory.translationstats(reduced).nunique,
        )
    end

    # How many DISTINCT permutation objects the collection reads through, BY IDENTITY.
    #
    # `_permutationsupplier` returns a closure over tables built once per (level, symmetry), so this
    # is bounded by the group order times the level count, NOT by the interaction count. Nothing
    # pinned that, and no value-based assertion can: a supplier rewritten to call
    # `angularpermutation` inline per pair returns EQUAL permutations every time, so every check in
    # this file still passes while storage grows with the interaction count.
    #
    # Not hypothetical. H2Factory's Toeplitz reduction shipped with exactly that shape, and for a
    # scalar kernel the per-interaction permutations cost MORE memory than the reduction saved
    # (measured 1.4x bigger than not reducing at all) until they were shared.
    function distinctpermutations(tfs)
        return length(Set(objectid(tfs[key].permutation) for key in collect(keys(tfs))))
    end

    # How many stored couplings actually READ THROUGH a non-identity permutation. A reduction ratio
    # says orbits merged; this says the permutation machinery ran, which is the part that can be
    # silently wrong.
    function nontrivialpermutations(tfs)
        return count(collect(keys(tfs))) do key
            permutation = tfs[key].permutation
            return permutation != Int32.(eachindex(permutation))
        end
    end

    @testset "the common policy is the smallest group over the levels" begin
        # `CircularSampling`'s quarter turns need `Nφ % 4 == 0` and the rate rule does not guarantee
        # it at every level, so a mixed stack has to fall back for ALL levels, canonicalization
        # happens once, with one policy. Pinned on hand-built stacks because a real level stack that
        # happens to be uniformly divisible by four (like the fixtures below) never exercises the
        # fallback.
        alldivisible = [MLFMA.CircularSampling(8), MLFMA.CircularSampling(16)]
        @test MLFMA.supportedsymmetry(alldivisible, Val(2)) isa H2Trees.FullLatticeSymmetry

        onelevelnot = [MLFMA.CircularSampling(8), MLFMA.CircularSampling(10)]
        @test MLFMA.supportedsymmetry(onelevelnot, Val(2)) isa H2Trees.OppositeSymmetry
    end

    @testset "Galerkin: identical couplings, and a matvec that agrees" begin
        κ = 2π
        mesh = CompScienceMeshes.meshcircle(1.5, 1 / 6)
        X = lagrangec0d1(mesh)
        operator = Helmholtz2D.singlelayer(; wavenumber=κ)

        # `minhalfsize = 0.25` gives relevantlevels 3:4, i.e. two translating levels rather than one,
        # so the per-level permutation tables are actually indexed by more than one level. With a
        # single level a level-index error is unobservable.
        tree = H2Trees.TwoNTree(
            X; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.25, minvalues=0)
        )

        full = MLFMA.GalerkinMLFMA(
            operator, X; tree=tree, strategy=unreducedstrategy(operator, X), verbose=false
        )
        reduced = MLFMA.GalerkinMLFMA(
            operator, X; tree=tree, strategy=symmetrystrategy(operator, X), verbose=false
        )

        fulltfs = MLFMA.translationcollection(full)
        reducedtfs = MLFMA.translationcollection(reduced)

        @test reducedtfs isa H2Factory.SymmetryReducedI2OTranslationCollection
        # The baseline really is unreduced, otherwise "identical to the unreduced collection" would
        # be comparing two reduced ones.
        @test H2Factory.translationsymmetrygroup(fulltfs) === nothing

        comparison = comparecollections(reducedtfs, fulltfs)
        @test comparison.npairs > 0
        @test comparison.samekeys
        @test isempty(comparison.mismatches)
        # Measured 8.3e-15 on this fixture. Asserted loosely so the SHAPE is pinned, round-off, not
        # approximation, without turning a last-bit change into a failure.
        @test comparison.worstdifference < 1e-12
        # Above 2 only with chiral elements: the antipodal map alone gives exactly 2.00x, because
        # `q == -q` holds only for `q == 0`, which is never a translation. Measured 56 unique
        # translations down to 12, i.e. 4.67x, so the quarter turns really are being used and a
        # reversed inverse convention could not hide here.
        @test comparison.ratio > 2.5
        # Stronger than the ratio: orbits merging is not the same as a permutation being APPLIED,
        # and only the latter can be silently wrong. Measured 89 of 106 couplings here.
        @test nontrivialpermutations(reducedtfs) > 0

        # And the whole matvec, in both directions. Compared against the unreduced map rather than a
        # tolerance on BEAST: the two maps differ ONLY in how the translation is stored, so anything
        # above round-off is a bug and not an approximation.
        Random.seed!(11)
        x = randn(ComplexF64, numfunctions(X))
        @test norm(reduced * x - full * x) / norm(full * x) < 1e-14
        @test norm(adjoint(reduced) * x - adjoint(full) * x) / norm(adjoint(full) * x) <
            1e-14

        # Non-vacuity for the comparison above: the unreduced map is itself right. (Its own accuracy
        # is `test/Maps/helmholtz2d.jl`'s subject; this only rules out both maps being zero.)
        dense = BEAST.assemble(operator, X, X)
        @test norm(reduced * x - dense * x) / norm(dense * x) < 1e-4
    end

    # TWO TREES: THE ROOT OFFSET DECIDES WHETHER A SYMMETRY APPLIES AT ALL.
    #
    # `H2Trees` folds a block tree's root-center offset out before taking the integer key, which is
    # sound for the plain trait, every interaction sharing a key has the same physical
    # displacement. A SYMMETRY MERGES DIFFERENT KEYS, and then the stored direction
    # `halfsize*canonical + offset` transforms to `halfsize*(S*canonical) + S*offset` where the
    # interaction needs `+ offset`. So the reduction is valid only where the displacements lie on a
    # lattice, which `H2Trees` checks per interaction and refuses otherwise.
    #
    # THE PREVIOUS VERSION OF THIS TESTSET WAS VACUOUS, and that is why both cases are pinned now.
    # It used two circles five units apart, which produce exactly ONE far interaction carrying the
    # identity symmetry, so it asserted reconstruction on a case where the symmetry machinery
    # never ran, and would have passed whatever the two-tree path did.
    @testset "Petrov: a non-lattice offset falls back, loudly" begin
        κ = 2π
        base = CompScienceMeshes.meshcircle(1.0, 1 / 8)
        trialspace = lagrangec0d1(base)
        testspace = lagrangec0d1(
            CompScienceMeshes.translate(base, SVector{2,Float64}(5.37, 0.0))
        )
        operator = Helmholtz2D.singlelayer(; wavenumber=κ)

        # A WARNING AND NOT AN ERROR, because this is a legitimate map the reduction cannot serve
        # rather than a mistake: refusing to build would break a configuration that works today for
        # a user who did nothing wrong. It has to be AUDIBLE, though, silently handing back an
        # unreduced collection would let someone conclude the feature does not pay on block trees
        # when it was never able to run.
        reduced = @test_logs (:warn, r"falling back to the unreduced") match_mode = :any MLFMA.PetrovGalerkinMLFMA(
            operator,
            testspace,
            trialspace;
            strategy=symmetrystrategy(operator, testspace, trialspace),
            verbose=false,
        )

        # What came back really is the unreduced collection, not a symmetry-reduced one that
        # happened not to reduce.
        reducedtfs = MLFMA.translationcollection(reduced)
        @test !(reducedtfs isa H2Factory.SymmetryReducedI2OTranslationCollection)
        @test H2Factory.translationsymmetrygroup(reducedtfs) === nothing

        # And it is a working map, agreeing with the one built without the strategy at all.
        full = MLFMA.PetrovGalerkinMLFMA(
            operator,
            testspace,
            trialspace;
            strategy=unreducedstrategy(operator, testspace, trialspace),
            verbose=false,
        )
        Random.seed!(23)
        x = randn(ComplexF64, numfunctions(trialspace))
        y = randn(ComplexF64, numfunctions(testspace))
        @test norm(reduced * x - full * x) / norm(full * x) < 1e-14
        @test norm(adjoint(reduced) * y - adjoint(full) * y) / norm(adjoint(full) * y) <
            1e-14
    end

    @testset "Petrov: two trees the lattice admits, and the swapped coupling key" begin
        # THE REASON THIS TESTSET EXISTS. `couplingkey(r, t, ForwardTranslation()) == (r, t)`, but
        # transpose and adjoint both return the SWAPPED pair. A wrapper that rebuilt the symmetry
        # lookup from its arguments instead of from the key it already holds would pick up another
        # interaction's symmetry in exactly those two directions, and on a Galerkin block, which
        # is symmetric to round-off in 2-D, that is nearly invisible.
        #
        # WHY SHIFT 4.0 AND NOT A DERIVED MULTIPLE OF SOME HALFSIZE. Whether a shift lands on the
        # lattice depends on the boxes the MAP's tree strategy chooses, not on the mesh extents,
        # the root halfsize is rounded to a power-of-two multiple of `minhalfsize`, so it is not
        # readable off the mesh. Rather than derive it and be wrong, this asserts the two properties
        # the testset actually depends on: that construction is admitted, and that non-identity
        # permutations are genuinely applied. If a tree-building change breaks either, the failure
        # names which one.
        # WHY `minvalues` IS PINNED HERE. This mesh carries 51 basis functions, and the default
        # `minvalues` is 100, so with the default the tree stops at its root: ONE box per side. Two
        # single boxes 4.0 apart, with a root halfsize of 1.0, are two box widths apart and
        # therefore each other's near neighbours, so the map has no far interactions at all and the
        # aggregate plan comes out empty, which `H2Trees` refuses. Pinning a `minvalues` that
        # actually subdivides restores the 16 couplings this testset was written against, and stops
        # a default change from silently collapsing the fixture again.
        κ = 2π
        base = CompScienceMeshes.meshcircle(1.0, 1 / 8)
        trialspace = lagrangec0d1(base)
        testspace = lagrangec0d1(
            CompScienceMeshes.translate(base, SVector{2,Float64}(4.0, 0.0))
        )
        operator = Helmholtz2D.singlelayer(; wavenumber=κ)
        subdividing = MLFMA.TreeStrategy(operator; minhalfsize=0.25, minvalues=30)

        full = MLFMA.PetrovGalerkinMLFMA(
            operator,
            testspace,
            trialspace;
            strategy=unreducedstrategy(
                operator, testspace, trialspace; treestrategy=subdividing
            ),
            verbose=false,
        )
        reduced = MLFMA.PetrovGalerkinMLFMA(
            operator,
            testspace,
            trialspace;
            strategy=symmetrystrategy(
                operator, testspace, trialspace; treestrategy=subdividing
            ),
            verbose=false,
        )

        # The fixture really did subdivide. Without this, a future default change puts the trees back
        # to one box each and the assertions below would fail somewhere deep in a plan builder
        # instead of saying that the fixture collapsed.
        blocktree = H2Factory.core(reduced).tree
        @test length(H2Trees.leaves(H2Trees.testtree(blocktree))) > 1
        @test length(H2Trees.leaves(H2Trees.trialtree(blocktree))) > 1

        reducedtfs = MLFMA.translationcollection(reduced)
        comparison = comparecollections(reducedtfs, MLFMA.translationcollection(full))
        @test comparison.npairs > 1
        @test comparison.samekeys
        @test isempty(comparison.mismatches)
        @test comparison.worstdifference < 1e-12
        # NON-VACUITY, and stronger than a reduction ratio: it asks whether a non-identity
        # permutation is actually APPLIED, which is the thing a mis-keyed symmetry would corrupt.
        # Measured 4 of 16 couplings on this fixture.
        @test nontrivialpermutations(reducedtfs) > 0
        # And the permutations are SHARED per (level, symmetry) rather than built per interaction,
        # see `distinctpermutations`. Bounded by group order x level count, which is far below the
        # pair count here; the strict inequality is what fails if the sharing is ever lost.
        @test distinctpermutations(reducedtfs) < comparison.npairs

        Random.seed!(13)
        x = randn(ComplexF64, numfunctions(trialspace))
        y = randn(ComplexF64, numfunctions(testspace))
        # The block really is non-symmetric, so transpose and adjoint are different operators from
        # the forward one and a mis-keyed symmetry in those directions cannot hide. Asserted on the
        # maps rather than a dense reference, which is `test/Maps/helmholtz2d.jl`'s subject.
        @test norm(full * x - transpose(full) * x) / norm(full * x) > 0.1

        @test norm(reduced * x - full * x) / norm(full * x) < 1e-14
        @test norm(adjoint(reduced) * y - adjoint(full) * y) / norm(adjoint(full) * y) <
            1e-14
        @test norm(transpose(reduced) * y - transpose(full) * y) /
              norm(transpose(full) * y) < 1e-14
    end

    @testset "three dimensions: the order-16 subgroup in a real matvec" begin
        # WHERE THE REDUCTION ACTUALLY PAYS. Two dimensions is the correctness proof; the measured
        # storage case, 9.69x at 16 wavelengths, growing with problem size, is entirely three
        # dimensional. This is also the first exercise of a NON-FULL policy end to end: the grid
        # takes the order-16 subgroup fixing the polar axis, not all 48, because the excluded
        # elements carry the polar axis into the equatorial plane where a Gauss-Legendre θ rule has
        # no samples.
        ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
        # A TWO-WAVELENGTH SPHERE, not one. Measured: the 1λ sphere puts every far interaction on a
        # SINGLE level (`pairsbylevel == [1302, 0]`), and with one level an off-by-one in the
        # permutation tables' level index is unobservable. This fixture gives `[1736, 136]`, and the
        # assertion below is what holds that property in place. `minhalfsize = λ/4` is
        # `MINHALFSIZERECOM`, so the fixture also stays out of the low-frequency-breakdown warning.
        Y = raviartthomas(meshsphere(2 * λ, λ / 6))
        efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

        tree = H2Trees.TwoNTree(
            Y; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0)
        )

        # The near block is stubbed on BOTH sides. It is identical between them by construction and
        # is most of the construction cost, while the far path is the entire subject here.
        function strategyfor(collection)
            return MLFMA.MLFMAStrategy(
                efie,
                Y;
                nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                    MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
                ),
                i2ostrategy=MLFMA.I2OTranslationStrategy(;
                    TranslationCollection=collection
                ),
            )
        end

        full = MLFMA.GalerkinMLFMA(
            efie,
            Y;
            tree=tree,
            strategy=strategyfor(MLFMA.TranslationCollection),
            verbose=false,
        )
        reduced = MLFMA.GalerkinMLFMA(
            efie,
            Y;
            tree=tree,
            strategy=strategyfor(MLFMA.SymmetryTranslationCollection),
            verbose=false,
        )

        fulltfs = MLFMA.translationcollection(full)
        reducedtfs = MLFMA.translationcollection(reduced)
        @test reducedtfs isa H2Factory.SymmetryReducedI2OTranslationCollection
        @test H2Factory.translationsymmetrygroup(fulltfs) === nothing

        # Levels that actually CARRY translations, not entries in the per-level vector. The vector
        # has one entry per translating level in the plan and a level can legitimately hold zero
        # interactions, counting entries would have passed on the 1λ fixture, whose second level
        # is empty, which is exactly the case that cannot see a level-index error.
        @test count(>(0), H2Factory.translationstats(fulltfs).pairsbylevel) >= 2

        comparison = comparecollections(reducedtfs, fulltfs)
        @test comparison.npairs > 0
        @test comparison.samekeys
        @test isempty(comparison.mismatches)
        @test comparison.worstdifference < 1e-12
        # Measured 354 unique translations down to 49 over 1872 pairs, i.e. 7.22x, close to the
        # 9.1x this policy was projected to give on an 8λ sphere, and the reason the reduction is
        # worth having at all. Asserted only above 2.5, which is the non-vacuity threshold: above 2
        # needs chiral elements, and chiral elements are what make a reversed inverse convention
        # detectable at all, since the antipodal map is its own inverse and passes either way.
        @test comparison.ratio > 2.5
        @test nontrivialpermutations(reducedtfs) > 0

        # A LOOSER BOUND THAN THE TWO-DIMENSIONAL TESTSETS, chosen for robustness across fixtures
        # rather than because this one needs it: measured 7.3e-15 forward and 7.2e-15 adjoint here,
        # but 3.0e-14 on a coarser sphere tried first. The spread is ordinary summation round-off,
        # a three-dimensional grid has Nθ*Nφ directions instead of Nφ and each output accumulates
        # over more interactions, and it is fixture-dependent in a way a tight bound would make
        # brittle for no gain. 1e-12 is still eleven orders clear of what this checks for: a wrong
        # permutation reads a different direction's translation and misses by order one.
        Random.seed!(17)
        x = randn(ComplexF64, numfunctions(Y))
        @test norm(reduced * x - full * x) / norm(full * x) < 1e-12
        @test norm(adjoint(reduced) * x - adjoint(full) * x) / norm(adjoint(full) * x) <
            1e-12
        # Not comparing two zeros.
        @test norm(full * x) > 0
    end

    @testset "a sampling with no angular action refuses rather than reducing nothing" begin
        # `NoSymmetry()` would read as "reduction available, just empty" and let a caller conclude the
        # feature does not pay on a grid it was never able to act on. The refusal names the two
        # functions to define instead.
        @test_throws ErrorException MLFMA.supportedsymmetry(NoActionSampling())
        @test_throws "supportedsymmetry is not defined" MLFMA.supportedsymmetry(
            NoActionSampling()
        )
    end

    @testset "EquidistantSampling has an action, and it is its own fact" begin
        # This grid used to be the refusal example. It no longer is: its θ nodes are the MIDPOINT
        # rule `(i - ½)π/Nθ`, and `θᵢ + θ_{Nθ+1-i} = π` makes `θ → π - θ` the same index reversal
        # the Gauss-Legendre nodes' midpoint symmetry provides, a different fact reaching the same
        # conclusion. Without it, flipping `TRANSLATIONRECOM` to the reduced collection would have
        # made every equidistant map fail to CONSTRUCT, which is how this was found.
        for (Nθ, Nφ) in ((4, 8), (6, 12), (8, 16))
            sampling = MLFMA.Equidistant.EquidistantSampling{Float64}(Nθ, Nφ)
            @test MLFMA.supportedsymmetry(sampling) == H2Trees.AxisPreservingSymmetry(3)

            # The action is checked against the grid itself rather than against the spherical
            # implementation it shares code with: `k̂[π[j]]` must BE `S k̂[j]`, on every element.
            k̂, _ = MLFMA.cartesiansamplepoints(sampling)
            group = H2Trees.symmetrygroup(MLFMA.supportedsymmetry(sampling), Val(3))
            for symmetry in group
                permutation = MLFMA.angularpermutation(sampling, symmetry)
                @test sort(Int.(permutation)) == collect(eachindex(k̂))
                @test maximum(eachindex(k̂)) do j
                    return norm(
                        SVector{3,Float64}(H2Trees.applysymmetry(symmetry, Tuple(k̂[j]))) -
                        k̂[permutation[j]],
                    )
                end < 1e-14
            end
        end
    end
end
