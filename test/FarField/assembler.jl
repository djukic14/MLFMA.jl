using MLFMA
using BEAST, CompScienceMeshes
using LinearAlgebra
using Test
using H2Factory
using SphericalScattering
using H2Trees

include("../support/util_test.jl")

@testset "Splitting FarFieldAssembler" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ

    m = meshsphere(2 * λ, λ / 5)

    for X in [raviartthomas(m), buffachristiansen(m)]
        L = 6
        Nθ, Nφ = MLFMA.NθNφ(L)
        sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
        efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

        farfieldoperator = BEAST.MWFarField3D(; gamma=γ)
        quadstrat = BEAST.SingleNumQStrat(1)

        assembler = MLFMA.BEASTFarFieldAssembler(
            farfieldoperator,
            X,
            quadstrat,
            MLFMA.farfieldstorerule(efie),
            MLFMA.farfieldtype(efie),
        )
        tree = TwoNTree(
            X;
            builder=TwoNTreeBuilder(;
                minhalfsize=λ / 10, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
            ),
        )

        leaves = collect(H2Trees.leaves(tree))
        levels = unique(H2Trees.level.(Ref(tree), leaves))
        @test length(levels) == 1 # this tree is forced uniform -- see the non-uniform testset below
        minrelevantlevel = levels[begin]

        leafsamplings = MLFMA.leaffarfieldsamplings(tree, [sampling], minrelevantlevel)

        Fmultiples = assembler(tree, leafsamplings; verbose=true)

        samplingpoints, _ = MLFMA.cartesiansamplepoints(sampling)

        # Spot-check a handful of leaves, not every leaf -- each check re-runs a full
        # (multi-threaded) `BEAST.potential` quadrature pass over the whole mesh as its ground
        # truth, which does not scale to checking every leaf once the tree has many of them.
        for (i, leaf) in enumerate(leaves)
            i > 3 && break

            ids = H2Trees.values(tree, leaf)
            center = H2Trees.center(tree, leaf)
            expectedphase = MLFMA.phasefactor(
                γ, samplingpoints, center, MLFMA.AbstractRadiationFieldCollection
            )

            # Spot-check one function per leaf (not every function in every leaf) -- enough to
            # confirm the per-leaf centering is correct everywhere in the tree without repeating
            # a full `BEAST.potential` quadrature pass (one per basis function) for every column.
            functionid = first(ids)
            u = zeros(numfunctions(X))
            u[functionid] = 1.0

            # ground truth, independent of the production assembler path: absolute
            # (origin-referenced) BEAST far field, explicitly phase-shifted to this leaf's own
            # center.
            #
            # This used to double as a WORKAROUND: the plain (non-tree) assembler path raised an
            # ambiguity `MethodError` for the tangential rule, so it could not serve as a reference
            # at all. That bug is fixed -- collapsing four generated storers into one
            # `FarFieldStorer` made the specificity mistake unconditional instead of hiding it in
            # the one storer that annotated `cluster::Int`, see `store.jl` -- and
            # `test/FarField/storerules.jl` now pins the flat path for every rule. The independent
            # reference stays regardless: its value was never the workaround, but that ground truth
            # comes from BEAST rather than from a second call into the code under test.
            farfieldscartesian = BEAST.potential(
                farfieldoperator, samplingpoints, u, X; quadstrat=quadstrat
            )
            farfieldspherical = SphericalScattering.convertCartesian2Spherical.(
                farfieldscartesian, SphericalScattering.cart2sph.(samplingpoints)
            )
            Fθ, Fφ = MLFMA._removeradialcomponent(farfieldspherical)

            column = findfirst(==(functionid), ids)
            @test vec(Fθ .* expectedphase) ≈ Fmultiples[i][1][:, column]
            @test vec(Fφ .* expectedphase) ≈ Fmultiples[i][2][:, column]
        end
    end
end

@testset "Splitting FarFieldAssembler on a non-uniform tree" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ

    m = meshsphere(2 * λ, λ / 5)
    X = raviartthomas(m)

    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    farfieldoperator = BEAST.MWFarField3D(; gamma=γ)
    quadstrat = BEAST.SingleNumQStrat(1)

    assembler = MLFMA.BEASTFarFieldAssembler(
        farfieldoperator,
        X,
        quadstrat,
        MLFMA.farfieldstorerule(efie),
        MLFMA.farfieldtype(efie),
    )

    # `minvalues=0` with the default (BEAST-aware) protrusion check -- unlike the uniform testset
    # above, which additionally forces `protrusion=NoProtrusionCheck()` to get uniform depth --
    # leaves end up at different depths depending on local point density, which is exactly what
    # this testset needs to exercise.
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 20, minvalues=0))

    leaves = collect(H2Trees.leaves(tree))
    leaflevels = H2Trees.level.(Ref(tree), leaves)
    levels = sort(unique(leaflevels))
    @test length(levels) > 1 # confirms the fixture actually has mixed-depth leaves

    minrelevantlevel = levels[begin]

    ϵ = 1e-5
    samplings = Vector{MLFMA.SphericalSampling{Float64}}(undef, length(levels))
    for (i, level) in enumerate(levels)
        halfsize = H2Trees.halfsize(tree, H2Trees.LevelIterator(tree, level)[begin])
        diameter = MLFMA.diameter(halfsize, 3)
        samplings[i], _ = MLFMA.sampling(
            γ, ϵ, diameter, MLFMA.SphericalSamplingBuilder{Float64}()
        )
    end

    leafsamplings = MLFMA.leaffarfieldsamplings(tree, samplings, minrelevantlevel)

    Fmultiples = assembler(tree, leafsamplings; verbose=true)

    # The size check below is cheap and runs for every leaf. The `BEAST.potential`-based ground
    # truth check is not (a full multi-threaded quadrature pass over the whole mesh per call), so
    # it only runs once per distinct level -- same reasoning as the uniform testset above, but
    # bounded by level here instead of by leaf count, so every distinct grid size actually gets
    # exercised regardless of how many leaves share it.
    checkedlevels = Set{Int}()
    for (i, leaf) in enumerate(leaves)
        level = leaflevels[i]
        expectedsampling = samplings[level - minrelevantlevel + 1]
        expectedNθ, expectedNφ = MLFMA.Nθ(expectedsampling), MLFMA.Nφ(expectedsampling)

        # this leaf's own far field must be sized for its own level's grid, not some shared one
        @test size(Fmultiples[i][1], 1) == expectedNθ * expectedNφ

        level in checkedlevels && continue
        push!(checkedlevels, level)

        ids = H2Trees.values(tree, leaf)
        functionid = first(ids)
        u = zeros(numfunctions(X))
        u[functionid] = 1.0

        samplingpoints, _ = MLFMA.cartesiansamplepoints(expectedsampling)
        farfieldscartesian = BEAST.potential(
            farfieldoperator, samplingpoints, u, X; quadstrat=quadstrat
        )

        # ground truth, independent of the production path: absolute (origin-referenced) BEAST
        # far field -- same pattern as `test_farfieldcollection.jl`'s "Compare to BEAST.jl"
        # testset -- explicitly phase-shifted to this leaf's own center.
        farfieldspherical = SphericalScattering.convertCartesian2Spherical.(
            farfieldscartesian, SphericalScattering.cart2sph.(samplingpoints)
        )
        Fθ, Fφ = MLFMA._removeradialcomponent(farfieldspherical)

        center = H2Trees.center(tree, leaf)
        expectedphase = MLFMA.phasefactor(
            γ, samplingpoints, center, MLFMA.AbstractRadiationFieldCollection
        )

        column = findfirst(==(functionid), ids)
        @test vec(Fθ .* expectedphase) ≈ Fmultiples[i][1][:, column]
        @test vec(Fφ .* expectedphase) ≈ Fmultiples[i][2][:, column]
    end
end

@testset "Leaf far-field samplings when a leaf is shallower than minrelevantlevel" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ
    ϵ = 1e-4

    # A plan's `relevantlevels` starts at the shallowest level where a transfer OPERATION happens,
    # which can be strictly deeper than the shallowest level where a LEAF exists: coarse leaves with
    # no well-separated partners are pure near-interaction and never aggregate. `minvalues=0` with
    # the default (BEAST-aware) protrusion check produces exactly that on this mesh, while keeping
    # `minhalfsize` at λ/12. Indexing `samplings` by `level - minrelevantlevel + 1` then yields 0 for
    # those leaves, which used to be a `BoundsError`.
    m = meshsphere(0.8 * λ, λ / 5)
    X = raviartthomas(m)
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0))

    leaves = collect(H2Trees.leaves(tree))
    leaflevels = sort(unique(H2Trees.level.(Ref(tree), leaves)))

    plans = H2Trees.buildplans(tree)
    relevantlevels = plans.relevantlevels
    minrelevantlevel = relevantlevels[begin]

    # non-vacuity: this fixture is only a regression test while the gap actually exists
    @test length(leaflevels) > 1
    @test leaflevels[begin] < minrelevantlevel

    samplings, _ = MLFMA.getsamplingsparentsamplings(
        γ, ϵ, tree, relevantlevels, MLFMA.sampling
    )

    leafsamplings = MLFMA.leaffarfieldsamplings(tree, samplings, minrelevantlevel)

    # the per-leaf collection must stay total -- one well-formed entry for every leaf, including
    # the ones below `minrelevantlevel`
    @test length(leafsamplings) == length(leaves)
    for (i, leaf) in enumerate(leafsamplings)
        @test !isempty(leaf.samplingpoints)
        @test leaf.center == H2Trees.center(tree, leaves[i])
    end

    # leaves at or below `minrelevantlevel` land on the coarsest available grid, deeper leaves on
    # their own level's grid
    for (i, leaf) in enumerate(leaves)
        expected = samplings[max(
            H2Trees.level(tree, leaf) - minrelevantlevel + 1, firstindex(samplings)
        )]
        @test leafsamplings[i].sampling == expected
    end

    # The floor is one-sided on purpose. A leaf DEEPER than the last sampling means
    # `relevantlevels` and the tree have gone out of sync, and that pattern really would be read at
    # the wrong rate, so it must fail loudly rather than silently reuse the finest grid. Simulate
    # that by claiming a `minrelevantlevel` shallower than the samplings were built for.
    @test_throws ErrorException MLFMA.leaffarfieldsamplings(
        tree, samplings, minrelevantlevel - 1
    )
end

# BOTH POLICY AXES ARE ARGUMENTS, NOT DEFAULTS.
#
# Both assemblers used to offer a three-argument convenience constructor that defaulted
# `storerule` and `FarFieldType` independently. Nothing called it -- every construction site,
# production and test, reads both from the operator via `farfieldstorerule`/`farfieldtype` -- and
# the default PAIR was not self-consistent: `BEASTFarFieldAssembler`'s defaulted to a radial
# projection (one scalar out) alongside `TangentialSphericalFarField` (`fielddim` 2), so anyone
# taking the defaults got a storer writing into a layout it did not fit. There is no correct fixed
# pair to choose, because the two axes are independent properties of the operator, so the
# convenience form is gone rather than repaired.
@testset "Assembler constructors require both policy axes" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    X = raviartthomas(meshsphere(1 * λ, λ / 5))
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    farfieldoperator = BEAST.MWFarField3D(; gamma=im * κ)
    quadstrat = BEAST.SingleNumQStrat(1)

    # the five-argument form is the whole interface, and it round-trips both axes
    assembler = MLFMA.BEASTFarFieldAssembler(
        farfieldoperator,
        X,
        quadstrat,
        MLFMA.farfieldstorerule(efie),
        MLFMA.farfieldtype(efie),
    )
    @test MLFMA.farfieldstorerule(assembler) === MLFMA.farfieldstorerule(efie)
    @test MLFMA.farfieldtype(assembler) === MLFMA.farfieldtype(efie)

    # no partial construction: neither axis may be left to the constructor to guess
    @test_throws MethodError MLFMA.BEASTFarFieldAssembler(farfieldoperator, X, quadstrat)
    @test_throws MethodError MLFMA.DummyFarFieldAssembler(farfieldoperator, X, quadstrat)
    @test_throws MethodError MLFMA.BEASTFarFieldAssembler(
        farfieldoperator, X, quadstrat, MLFMA.farfieldstorerule(efie)
    )

    # ...and the keyword spelling those defaults used is gone with them
    @test_throws MethodError MLFMA.BEASTFarFieldAssembler(
        farfieldoperator,
        X,
        quadstrat;
        storerule=MLFMA.farfieldstorerule(efie),
        FarFieldType=MLFMA.farfieldtype(efie),
    )
    @test_throws MethodError MLFMA.DummyFarFieldAssembler(
        farfieldoperator,
        X,
        quadstrat;
        storerule=MLFMA.StoreCartesianAsTangentialSpherical(),
        FarFieldType=MLFMA.TangentialSphericalFarField,
    )

    # the pair that used to be the default is exactly the incoherent one: what the rule produces
    # must have as many components as the layout holds
    @test MLFMA.fielddim(MLFMA.ScalarFarField) == 1
    @test MLFMA.fielddim(MLFMA.TangentialSphericalFarField) == 2
    hypersingular = MLFMA.MWHyperSingular3D(; wavenumber=κ)
    @test MLFMA.farfieldtype(hypersingular) === MLFMA.ScalarFarField
    @test MLFMA.farfieldstorerule(hypersingular) === MLFMA.StoreCartesianAsRadialScalar()
end
