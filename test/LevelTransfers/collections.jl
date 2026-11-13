using MLFMA
using Test
using CompScienceMeshes, BEAST
using StaticArrays
using LinearAlgebra
using H2Factory
using H2Trees

include("../support/util_test.jl")

# Resolves level transfers through H2Factory's OWN `leveltransfer` on a real tree, so the sector-key
# convention and the per-level indexing are checked against the protocol rather than against a
# restatement of MLFMA's own arithmetic.
@testset "InterpolationPhaseTransfer collections" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ
    ϵ = 1e-3
    T = ComplexF64

    m = meshsphere(2 * λ, λ / 8)
    X = raviartthomas(m)
    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=λ / 8, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
        ),
    )

    plans = H2Trees.buildplans(tree)
    relevantlevels = plans.relevantlevels
    minrelevantlevel = relevantlevels[begin]

    samplings, parentsamplings = MLFMA.getsamplingsparentsamplings(
        γ, ϵ, tree, relevantlevels, MLFMA.sampling
    )

    interpolator = MLFMA.SphericalLagrangeInterpolator{6,T}(
        samplings[2:end], parentsamplings[2:end]
    )
    anterpolator = MLFMA.anterpolator(interpolator)
    phaseshifter = MLFMA.PhaseShifter(γ, tree, relevantlevels, parentsamplings[2:end])
    trait = MLFMA.interpolationtrait(MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ)))

    trialcollection, testcollection = MLFMA.interpolationphasecollections(
        phaseshifter,
        interpolator,
        anterpolator,
        samplings,
        minrelevantlevel,
        relevantlevels,
        trait,
    )

    @test H2Factory.i2itrait(trialcollection) isa MLFMA.InterpolationPhaseI2I
    @test H2Factory.selector(trialcollection) isa H2Factory.SectorSelector{3}

    # The two collections must be distinct objects holding conjugate phases -- the whole reason
    # MLFMA cannot share one the way iFMM does.
    @test trialcollection !== testcollection

    checked = 0
    for level in relevantlevels
        level <= minrelevantlevel && continue
        for child in H2Trees.LevelIterator(tree, level)
            parent = H2Trees.parent(tree, child)
            parent == 0 && continue

            trialtransfer = H2Factory.leveltransfer(trialcollection, tree, parent, child)
            testtransfer = H2Factory.leveltransfer(testcollection, tree, parent, child)

            @test trialtransfer isa MLFMA.InterpolationPhaseTransfer
            @test testtransfer isa MLFMA.InterpolationPhaseTransfer

            # the phase resolved through the protocol must be the one `PhaseShifter` holds for this
            # edge's sector -- i.e. the 1-based sector key lines up with H2Trees' 0-based sector
            @test trialtransfer.phase ≈ MLFMA.findphaseshift(phaseshifter, tree, child)

            # sides are conjugates, non-vacuously
            @test trialtransfer.phase ≈ conj.(testtransfer.phase)
            @test !(trialtransfer.phase ≈ testtransfer.phase)

            # today's disaggregation phase (the opposite sector) is what the TEST side stores, which
            # is what makes `transpose(test I2I)` reproduce it without conjugating in the direction
            @test testtransfer.phase ≈
                MLFMA.findoppositephaseshift(phaseshifter, tree, child)

            # each transfer is sized for its own edge: child grid in, parent grid out
            @test size(trialtransfer.childsampling) ==
                size(samplings[level - minrelevantlevel + 1])
            @test size(trialtransfer.parentsampling) ==
                size(samplings[level - 1 - minrelevantlevel + 1])
            @test size(trialtransfer.phase) == size(trialtransfer.parentsampling)

            checked += 1
            checked >= 12 && break
        end
        checked >= 12 && break
    end
    @test checked >= 12 # the loop actually resolved transfers
end
