using MLFMA
using Test
using CompScienceMeshes, BEAST
using StaticArrays
using LinearAlgebra
using H2Factory
using H2Trees

include("../support/util_test.jl")

# Pins the phase convention that MLFMA's level-transfer design rests on, BEFORE the transfer types
# that consume it exist. The claim under test:
#
#   MLFMA needs two distinct I2I level-transfer families -- one per side -- even for a Galerkin map,
#   because the receive side's level phase is the conjugate of the radiation side's.
#
# That is unlike iFMM, whose polynomial level transfers are real and side-independent, so it returns
# one shared collection from all four `H2Factory.{trial,test}{i2i,o2o}translator` accessors. If these
# tests ever fail, the two-collection design is wrong and must be revisited before the transfer
# types are changed to match it.
@testset "Level-transfer phase convention (trial vs test side)" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ

    sampling = MLFMA.SphericalSampling{Float64}(10, 20)
    k̂s, _ = MLFMA.cartesiansamplepoints(sampling)

    # A deliberately generic offset: no zero component and no symmetry between components, so a
    # phase that accidentally came out real, or a trial/test pair that accidentally agreed, cannot
    # slip through. Scaled by λ so κ·Δ is order 1 -- an offset small against the wavelength gives a
    # phase within O(0.01) of 1, where `conj` is nearly a no-op and the test would prove nothing.
    Δ = λ * SVector(0.17, -0.31, 0.23)

    radiationphase = MLFMA.phasefactor(γ, k̂s, Δ, MLFMA.AbstractRadiationFieldCollection)
    receivephase = MLFMA.phasefactor(γ, k̂s, Δ, MLFMA.AbstractReceiveFieldCollection)

    # Non-vacuity: the fixture is only discriminating if the phase is genuinely complex, and not
    # so close to real that conj() is a no-op within tolerance.
    @test eltype(radiationphase) <: Complex
    @test maximum(abs, imag.(radiationphase)) > 0.1
    @test !(radiationphase ≈ receivephase)

    # The claim: the two sides' level phases are conjugates of one another.
    @test radiationphase ≈ conj.(receivephase)

    # ...and that conjugation traces back to the operators themselves, not to a coincidence of this
    # particular offset: `receiveoperator` carries -γ where `radiationoperator` carries +γ.
    # (`BEAST.gamma` has no method for the far-field operator types, so read the field directly.)
    #
    # Checked across every operator family that defines both accessors, Maxwell and Helmholtz alike,
    # since the whole two-collection design rests on this convention holding generally rather than
    # for one family. Some receive operators additionally carry an `amplitude`; only the γ relation
    # is asserted here, which is what the level phase is built from.
    operators = (
        "MWEFIE3D" => MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ)),
        "MWSingleLayer3D" => Maxwell3D.singlelayer(; wavenumber=κ),
        "MWHyperSingular3D" => MLFMA.MWHyperSingular3D(; wavenumber=κ),
        "MWDoubleLayer3D" => Maxwell3D.doublelayer(; wavenumber=κ),
        "DoubleLayerRotatedMW3D" => BEAST.n × Maxwell3D.doublelayer(; wavenumber=κ),
        "HH3DSingleLayerFDBIO" => Helmholtz3D.singlelayer(; wavenumber=κ),
        "HH3DDoubleLayerFDBIO" => Helmholtz3D.doublelayer(; wavenumber=κ),
        "HH3DDoubleLayerTransposedFDBIO" =>
            Helmholtz3D.doublelayer_transposed(; wavenumber=κ),
        "HH3DHyperSingularFDBIO" => Helmholtz3D.hypersingular(; wavenumber=κ),
    )

    # guards against the list being silently trimmed to one family again
    @test count(p -> startswith(first(p), "MW"), operators) >= 4
    @test count(p -> startswith(first(p), "HH"), operators) >= 4

    for (name, op) in operators
        radiationγ = MLFMA.radiationoperator(op).gamma
        receiveγ = MLFMA.receiveoperator(op).gamma
        @test radiationγ == -receiveγ
        # non-vacuity: `0 == -0` would pass the above trivially
        @test !iszero(radiationγ)
    end

    # Transpose and adjoint must stay distinguishable for this transfer. The level phase enters the
    # transfer as a diagonal operator, so transpose leaves it alone while adjoint conjugates it --
    # which is exactly why the direction step cannot be allowed to smuggle in the conjugation that
    # actually belongs to the side. Spelled out on a small diagonal to keep the claim about the
    # algebra, not about any MLFMA type.
    d = Diagonal(vec(radiationphase)[1:4])
    @test transpose(d) ≈ d
    @test !(adjoint(d) ≈ d)
    @test adjoint(d) ≈ conj(d)
end

@testset "Level-transfer phase: PhaseShifter sector vs opposite sector" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ
    ϵ = 1e-3

    m = meshsphere(5 * λ, λ / 10)
    X = raviartthomas(m)
    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=λ / 10, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
        ),
    )

    levels = H2Trees.levels(tree)
    samplings = Vector{MLFMA.SphericalSampling{Float64}}(undef, length(levels) - 1)
    for (i, level) in enumerate(levels[2:end])
        samplings[i] = MLFMA.sampling(γ, ϵ, tree; level=level)[1]
    end

    p = MLFMA.PhaseShifter(γ, tree, levels[2:end], samplings)

    # The aggregation phase for a node's own sector and the disaggregation phase (its opposite
    # sector) are conjugates. Today MLFMA reaches the second via `findoppositephaseshift`; under the
    # level-transfer protocol the same value has to come from the TEST side's stored phase, because
    # the protocol's transpose direction will not conjugate a diagonal for us.
    checked = 0
    for level in levels
        level <= 2 && continue
        for node in H2Trees.LevelIterator(tree, level)
            H2Trees.parent(tree, node) == 0 && continue

            own = MLFMA.findphaseshift(p, tree, node)
            opposite = MLFMA.findoppositephaseshift(p, tree, node)

            @test own ≈ conj(opposite)
            # non-vacuity: a real phase would make the above trivially true
            @test !(own ≈ opposite)

            checked += 1
            checked >= 8 && break
        end
        checked >= 8 && break
    end
    @test checked >= 8 # the loop actually exercised the claim
end
