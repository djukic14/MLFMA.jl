# The 2-D operator mappings, checked by ASSEMBLING A FAR BLOCK AND COMPARING TO BEAST.
#
# WHY THIS SHAPE OF TEST. The mappings are five one-line table entries, and every one of them is a
# CONVENTION -- which far-field operator, which sign of γ, which amplitude, which store rule. A test
# that merely checks each entry returns the expected type would restate the table rather than check
# it, and would have passed happily through #129, where a receive coupling was hardcoded to the wrong
# rule and the map ran on, wrong by 44 %.
#
# So this composes the whole factorization and compares against a dense BEAST block:
#
#     Z[m, n] = Σ_q w_q · L_m(k̂_q) · T(k̂_q) · R_n(k̂_q)
#
# with `R` the radiation pattern about the trial centre (`radiationoperator`, +γ), `L` the receive
# pattern about the test centre (`receiveoperator`, −γ), and `T` the translation operator for the
# vector between centres. If any of the three conventions disagrees, the number is wrong. Nothing is
# restated from `operatorutils.jl` -- the test asks for the operators through the same accessors the
# map constructor uses.
#
# THE DERIVATION, since the sign placement is the whole content. Writing R = c_test − c_trial and
# d = (x − c_test) − (y − c_trial), so that x − y = R + d:
#
#     G(|x − y|) = Σ_q w_q T_R(k̂) exp(−γ k̂·d)
#                = Σ_q w_q T_R(k̂) · exp(−γ k̂·(x − c_test)) · exp(+γ k̂·(y − c_trial))
#
# which is exactly `receive × translation × radiation`. That identity was verified against the closed
# form in `test/Translations/translation2d.jl`; here it is verified against BEAST's own assembly of
# the operator, which is the thing that ultimately has to agree.
#
# TWO SEPARATE MESHES, far apart, so every pair is genuinely a FAR interaction -- the factorization
# is only valid there, and splitting one mesh geometrically would leave near pairs in the block.

using BEAST
using CompScienceMeshes
using LinearAlgebra
using MLFMA
using StaticArrays
using Test

@testset "Helmholtz2D operator mappings" begin
    κ = 2.0
    γ = im * κ

    trialmesh = CompScienceMeshes.meshcircle(0.3, 0.12)
    testmesh = CompScienceMeshes.translate(trialmesh, SVector{2,Float64}(6.0, 1.5))
    trialspace = lagrangec0d1(trialmesh)
    testspace = lagrangec0d1(testmesh)

    trialcentre = SVector{2,Float64}(0.0, 0.0)
    testcentre = SVector{2,Float64}(6.0, 1.5)

    # Well inside the accuracy floor measured in `translation2d.jl` for this κ|R|.
    L = 20
    sampling = MLFMA.CircularSampling{Float64}(MLFMA.NφCircular(L))
    k̂, w = MLFMA.cartesiansamplepoints(sampling)

    # One helper for both sides: the only differences are which operator the mapping hands back and
    # which convention re-references the pattern to its own box centre.
    #
    # The ORIGIN-referenced assembler plus an explicit phase, rather than the tree-taking path -- this
    # test has two loose meshes and no tree. The two phase conventions are the package's own
    # primitives, not restated here:
    #
    #   radiation: `phasefactor(γ, k̂, c, ::Radiation) = exp(-γ k̂·c)`  turns ∫g e^{+γk̂·y} into
    #              ∫g e^{+γk̂·(y-c)}
    #   receive:   `phasefactor(γ, k̂, c, ::Receive)   = exp(+γ k̂·c)`  turns ∫f e^{-γk̂·x} into
    #              ∫f e^{-γk̂·(x-c)}
    #
    # so the sign that distinguishes the two sides is taken from `_phase_k`, exactly as the
    # aggregation and disaggregation take it.
    function pattern(operator, space, centre, collectionfamily)
        assembler = MLFMA.BEASTFarFieldAssembler(
            operator,
            space,
            BEAST.SingleNumQStrat(8),
            MLFMA.StoreAsFarFieldValue(),
            MLFMA.ScalarFarField,
        )
        atorigin = assembler(sampling; verbose=false)
        phase = MLFMA.phasefactor(γ, k̂, centre, collectionfamily)
        return phase .* atorigin[1]
    end

    @testset "the table resolves to the right shapes" begin
        for operator in (
            Helmholtz2D.singlelayer(; wavenumber=κ),
            Helmholtz2D.doublelayer(; wavenumber=κ),
            Helmholtz2D.doublelayer_transposed(; wavenumber=κ),
        )
            @test MLFMA.farfieldtype(operator) === MLFMA.ScalarFarField
            @test MLFMA.farfieldstorerule(operator) === MLFMA.StoreAsFarFieldValue()
            @test MLFMA.farfielddims(operator) == 1
            # Both sides must exist, and the receive side must carry the OPPOSITE sign of γ --
            # that is what makes it a receive pattern rather than a second radiation pattern.
            @test MLFMA.radiationoperator(operator).gamma == operator.gamma
            @test MLFMA.receiveoperator(operator).gamma == -operator.gamma
        end
    end

    @testset "only the single layer claims a Galerkin receive rule" begin
        # #130: a mirrored receive rule is a physics claim per operator. The single layer's kernel is
        # symmetric so the coupling is the identity; the double layers carry `k̂·n̂`, which flips under
        # the antipodal map, so they are Petrov-only and must NOT silently claim otherwise.
        @test MLFMA.haslazyreceive(Helmholtz2D.singlelayer(; wavenumber=κ))
        @test !MLFMA.haslazyreceive(Helmholtz2D.doublelayer(; wavenumber=κ))
        @test !MLFMA.haslazyreceive(Helmholtz2D.doublelayer_transposed(; wavenumber=κ))
    end

    @testset "the factorization reproduces a dense BEAST far block" begin
        translation = Vector{ComplexF64}(undef, length(sampling))
        MLFMA.translationoperator!(translation, γ, k̂, testcentre - trialcentre, L)

        for operator in (
            Helmholtz2D.singlelayer(; wavenumber=κ),
            Helmholtz2D.doublelayer(; wavenumber=κ),
            Helmholtz2D.doublelayer_transposed(; wavenumber=κ),
        )
            R = pattern(
                MLFMA.radiationoperator(operator),
                trialspace,
                trialcentre,
                MLFMA.AbstractRadiationFieldCollection,
            )
            Lp = pattern(
                MLFMA.receiveoperator(operator),
                testspace,
                testcentre,
                MLFMA.AbstractReceiveFieldCollection,
            )

            # Z = Lᵀ · diag(w ⊙ T) · R, the diagonalized far interaction.
            Z = transpose(Lp) * ((w .* translation) .* R)

            reference = BEAST.assemble(operator, testspace, trialspace)

            @test size(Z) == size(reference)
            @test norm(Z - reference) / norm(reference) < 1e-8
        end
    end
end

# The construction-path accessors, which are separate from the far-field mappings above: these are
# what `TreeStrategy` and the near-field strategy consult before any far field exists.
#
# All three were missing or 3-D-only, and all three were found by PROBING A 2-D MATVEC rather than by
# reading -- which is the point. Each failed in a different place, none of them named the dimension,
# and the quadrature one failed several frames inside BEAST.
@testset "Helmholtz2D construction-path accessors" begin
    κ = 2π
    operator2d = Helmholtz2D.singlelayer(; wavenumber=κ)
    operator3d = BEAST.Helmholtz3D.singlelayer(; wavenumber=κ)

    space2d = lagrangec0d1(CompScienceMeshes.meshcircle(1.5, 1 / 6))
    space3d = raviartthomas(meshsphere(1.0, 0.3))

    @testset "γ and λ resolve for a 2D operator" begin
        # `MLFMA.γ` forwards to `BEAST.gamma`, which BEAST defines for 3-D operators only; `λ` is
        # defined for `BEAST.MaxwellOperator3D`, which BEAST's 3-D HELMHOLTZ operators subtype (a
        # BEAST naming quirk) while the 2-D ones do not. So neither reached the 2-D family.
        @test MLFMA.γ(operator2d) == operator2d.gamma
        @test MLFMA.λ(operator2d) ≈ 1.0                    # κ = 2π, so λ = 1
        # Both dimensions must agree at the same wavenumber -- the formula is the same, only the
        # dispatch differed.
        @test MLFMA.λ(operator2d) ≈ MLFMA.λ(operator3d)
        @test MLFMA.γ(operator2d) == MLFMA.γ(operator3d)
    end

    @testset "the near-field quadrature is the operator's, not a stored 3D constant" begin
        # 3-D IS UNCHANGED, and identically so: `===` against the constant itself, not `==` against a
        # rebuilt equal one. That is what makes this a safe default to derive rather than store.
        @test MLFMA.nearinteractionsquadstrategy(operator3d, space3d, space3d) ===
            MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM

        # 2-D gets BEAST's own default, which is a DIFFERENT TYPE. The stored constant is a
        # `DoubleNumWiltonSauterQStrat`, whose Sauter-Schwab path assumes triangles and dies inside
        # `SauterSchwabQuadrature.reorder!` on a two-vertex segment -- deep in BEAST, far from
        # anything MLFMA wrote.
        quadstrategy = MLFMA.nearinteractionsquadstrategy(operator2d, space2d, space2d)
        @test quadstrategy == BEAST.defaultquadstrat(operator2d, space2d, space2d)
        @test !(quadstrategy isa BEAST.DoubleNumWiltonSauterQStrat)

        # And the whole point: a strategy built with the defaults now works for a 2-D operator.
        @test MLFMA.MLFMAStrategy(operator2d, space2d) isa MLFMA.MLFMAStrategy
    end

    @testset "the direction grid and the interpolator follow the operator too" begin
        # SAME SHAPE AS THE QUADRATURE ENTRY ABOVE, and for the same reason: a stored constant that
        # was correct where it was written is silently wrong one dimension over. These two are a PAIR
        # -- an interpolator family only accepts the grids it was written for -- so they are decided
        # by the same fact about the operator rather than left to a caller to keep in step.
        #
        # 3-D IS UNCHANGED, and identically so: `===` against the stored constants themselves, not
        # `==` against rebuilt equal ones. That is what makes these safe to derive rather than store.
        @test MLFMA.samplingbuilderrecom(operator3d) === MLFMA.SAMPLINGBUILDERRECOM
        @test MLFMA.interpolatorrecom(operator3d) ===
            MLFMA.INTERPOLATORECOM{15,typeof(MLFMA.γ(operator3d))}
        efie = BEAST.MWSingleLayer3D(im * κ)
        @test MLFMA.samplingbuilderrecom(efie) === MLFMA.SAMPLINGBUILDERRECOM
        @test MLFMA.interpolatorrecom(efie) ===
            MLFMA.INTERPOLATORECOM{15,typeof(MLFMA.γ(efie))}

        # 2-D gets the circular pair, parameterized by the operator's own γ so a real-γ operator
        # cannot silently pick up a complex interpolator.
        @test MLFMA.samplingbuilderrecom(operator2d) isa MLFMA.CircularSamplingBuilder
        @test MLFMA.interpolatorrecom(operator2d) ===
            MLFMA.CircularInterpolator{0,typeof(MLFMA.γ(operator2d))}

        # Reached through the strategy, which is where it actually matters: a caller who writes
        # `MLFMAStrategy(operator2d, space2d)` and nothing else must get grids and an interpolator
        # that agree with each other AND with the 2-D translation operator.
        strategy2d = MLFMA.MLFMAStrategy(operator2d, space2d)
        @test MLFMA.interpolatortype(strategy2d) ===
            MLFMA.CircularInterpolator{0,ComplexF64}
        @test MLFMA.samplingstrategy(strategy2d).samplingbuilder isa
            MLFMA.CircularSamplingBuilder

        strategy3d = MLFMA.MLFMAStrategy(operator3d, space3d)
        @test MLFMA.interpolatortype(strategy3d) === MLFMA.INTERPOLATORECOM{15,ComplexF64}
        @test MLFMA.samplingstrategy(strategy3d).samplingbuilder ===
            MLFMA.SAMPLINGBUILDERRECOM

        # The trait the map constructors check the interpolator against. It has to be the 2-D one, or
        # construction stops at the agreement check with a message about a component-sign convention.
        @test MLFMA.interpolationtrait(operator2d) === MLFMA.PeriodicφInterpolation()

        # An explicit override still wins -- the derivation is a default, not a policy.
        overridden = MLFMA.MLFMAStrategy(
            operator2d,
            space2d;
            samplingstrategy=MLFMA.SamplingStrategy(;
                samplingbuilder=MLFMA.CircularSamplingBuilder{Float64}(), ϵ=1e-3
            ),
        )
        @test MLFMA.samplingstrategy(overridden).ϵ == 1e-3
    end
end
