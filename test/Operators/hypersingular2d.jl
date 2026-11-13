# THE 2-D HYPERSINGULAR OPERATOR, which is a DERIVATION and not a smaller copy of the 3-D one.
#
# The difference is visible in the refspace: a 2-D Lagrange basis carries `(value, derivative)` with
# `derivative` a SCALAR arc-length derivative, where a 3-D one carries `(value, curl)` with `curl` an
# `SVector{3}`. There is no `curl` field on a 2-D refspace at all, so the 3-D far-field integrand
# would not merely be wrong here -- it would not dispatch.
#
# BEAST's 2-D integrand (`BEAST/src/helmholtz2d/hh2dops.jl`) is
#
#     (α · f(x) · g(y) · (n̂ₓ·n̂ᵧ)  +  β · f'(x) · g'(y)) · G(x-y)
#
# and substituting the diagonalized Green's function factors it into a THREE-component pattern per
# box, weighted `[α, α, β]`:
#
#     pattern = [n̂₁f, n̂₂f, f']
#     Z[m,n]  = Σ_q w_q T_q ( α(L₁R₁ + L₂R₂) + β L₃R₃ )
#
# Carrying `n̂` as components is the point: it is what lets `n̂ₓ·n̂ᵧ` factor across the two boxes.
#
# WHY A NEW REPRESENTATION RATHER THAN `SphericalVectorFarField`, which also has three components.
# Because that one MEANS (r,θ,φ). Reusing it would give `Fr`/`Fθ`/`Fφ` nonsense to return and would
# hand these components `SphericalInterpolation()` -- a θ-seam sign rule -- as their default trait, on
# a geometry with no θ and no seam. That collision is exactly what #69-#78 restructured the far-field
# types to make unrepresentable, so this file asserts the two types stay distinct.

using BEAST
using CompScienceMeshes
using H2Trees
using LinearAlgebra
using MLFMA
using Random
using StaticArrays
using Test

@testset "Helmholtz2D hypersingular" begin
    κ = 2.0
    γ = im * κ
    operator = Helmholtz2D.hypersingular(; wavenumber=κ)

    @testset "the representation is its own type, not a three-component lookalike" begin
        @test MLFMA.fielddim(MLFMA.PlanarVectorScalarFarField) == 3
        @test MLFMA.fielddim(MLFMA.SphericalVectorFarField) == 3
        # Same count, different type -- the property the far-field type split exists to provide.
        @test MLFMA.PlanarVectorScalarFarField !== MLFMA.SphericalVectorFarField
        @test MLFMA.PlanarVectorScalarFarField !== MLFMA.CartesianVectorFarField

        sampling = MLFMA.CircularSampling{Float64}(8)
        ff = MLFMA.PlanarVectorScalarFarField(ComplexF64, sampling)
        @test length(MLFMA.component(ff, 1)) == 8

        # `Fx`/`Fy` mean what they mean everywhere else. The third component is NOT `Fz` -- it is not
        # a component of that vector at all but a separate scalar -- so asking for one is an error.
        ff[1] .= 1
        ff[2] .= 2
        ff[3] .= 3
        @test all(MLFMA.Fx(ff) .== 1)
        @test all(MLFMA.Fy(ff) .== 2)
        @test all(MLFMA.Fs(ff) .== 3)
        @test_throws MethodError MLFMA.Fz(ff)
        @test_throws MethodError MLFMA.Fθ(ff)

        # Identity mirror rule, because the in-plane Cartesian basis does not move under the
        # antipodal map and the third component is a scalar. Contrast the spherical rules, whose
        # signs come from θ̂/φ̂ moving.
        rule = MLFMA.mirrorrule(MLFMA.PlanarVectorScalarFarField)
        @test length(rule) == 3
        @test all(m -> m.coeff == 1, rule)
    end

    @testset "the operator table resolves, and to the 2-D answers" begin
        @test MLFMA.farfieldtype(operator) === MLFMA.PlanarVectorScalarFarField
        @test MLFMA.farfielddims(operator) == 3
        @test MLFMA.farfieldstorerule(operator) === MLFMA.StoreAsFarFieldValue()
        # `[α, α, β]` -- two then one, following the integrand's two terms. The 3-D operator is
        # three then three.
        @test MLFMA.α(operator) == [operator.alpha, operator.alpha, operator.beta]
        @test length(MLFMA.α(operator)) == 3
        @test length(MLFMA.α(BEAST.Helmholtz3D.hypersingular(; wavenumber=κ))) == 6
        # Receive carries the opposite sign of γ, which is what makes it a receive pattern.
        @test MLFMA.radiationoperator(operator) isa MLFMA.HHHyperSingularFarField2D
        @test MLFMA.receiveoperator(operator) isa MLFMA.HHHyperSingularFarField2D
        @test MLFMA.radiationoperator(operator).gamma == operator.gamma
        @test MLFMA.receiveoperator(operator).gamma == -operator.gamma
        # Sizes the assembler's accumulation buffer. Without it the generic `HHFarField` method
        # returns a bare scalar type and the assembler dies converting an `SVector{3}` to a
        # `ComplexF64`, naming neither the operator nor the missing method.
        @test MLFMA.farfieldvaluetype(MLFMA.radiationoperator(operator)) ===
            SVector{3,ComplexF64}
    end

    @testset "the interpolation trait carries the arity and no sign" begin
        @test MLFMA.interpolationtrait(operator) === MLFMA.PeriodicφInterpolation{3}()
        @test MLFMA.traitarity(MLFMA.interpolationtrait(operator)) == 3
        # The scalar 2-D operators keep arity 1, and the bare constructor still means 1.
        @test MLFMA.PeriodicφInterpolation() === MLFMA.PeriodicφInterpolation{1}()
        @test MLFMA.interpolationtrait(Helmholtz2D.singlelayer(; wavenumber=κ)) ===
            MLFMA.PeriodicφInterpolation{1}()
        @test MLFMA.defaultinterpolationtrait(MLFMA.PlanarVectorScalarFarField) ===
            MLFMA.PeriodicφInterpolation{3}()
        # Still a circular trait, so a SPHERICAL family handed it must fail rather than apply a pole
        # rule -- the same protection the arity-1 case has.
        @test_throws MethodError MLFMA.converttrait(
            MLFMA.PeriodicφInterpolation{3}(), (1, 2, 3), MLFMA.DefaultInterpolation()
        )
    end

    @testset "the factorization reproduces a dense BEAST far block" begin
        # THE PHYSICS CLAIM, through MLFMA's own assembler, store rule and α table. Two separated
        # meshes so every pair is genuinely far -- the factorization is only valid there.
        trialmesh = CompScienceMeshes.meshcircle(0.3, 0.12)
        testmesh = CompScienceMeshes.translate(trialmesh, SVector{2,Float64}(6.0, 1.5))
        trialspace = lagrangec0d1(trialmesh)
        testspace = lagrangec0d1(testmesh)
        trialcentre = SVector{2,Float64}(0.0, 0.0)
        testcentre = SVector{2,Float64}(6.0, 1.5)

        L = 20
        sampling = MLFMA.CircularSampling{Float64}(MLFMA.NφCircular(L))
        k̂, w = MLFMA.cartesiansamplepoints(sampling)

        function pattern(op, space, centre, family)
            assembler = MLFMA.BEASTFarFieldAssembler(
                op,
                space,
                BEAST.SingleNumQStrat(8),
                MLFMA.farfieldstorerule(operator),
                MLFMA.farfieldtype(operator),
            )
            atorigin = assembler(sampling; verbose=false)
            phase = MLFMA.phasefactor(γ, k̂, centre, family)
            return [phase .* atorigin[c] for c in 1:3]
        end

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

        translation = Vector{ComplexF64}(undef, length(sampling))
        MLFMA.translationoperator!(translation, γ, k̂, testcentre - trialcentre, L)

        αs = MLFMA.α(operator)
        Z = zeros(ComplexF64, numfunctions(testspace), numfunctions(trialspace))
        for c in 1:3
            Z .+= αs[c] .* (transpose(Lp[c]) * ((w .* translation) .* R[c]))
        end

        reference = BEAST.assemble(operator, testspace, trialspace)
        @test size(Z) == size(reference)
        @test norm(Z - reference) / norm(reference) < 1e-10      # measured 1.84e-13

        # NON-VACUITY, and a discriminator: both terms must be substantial, or a test that dropped
        # one of them would still pass. Measured ‖·‖ = 3.4e-2 and 2.4e-2.
        vec = zeros(ComplexF64, size(Z))
        scal = zeros(ComplexF64, size(Z))
        for c in 1:2
            vec .+= αs[c] .* (transpose(Lp[c]) * ((w .* translation) .* R[c]))
        end
        scal .+= αs[3] .* (transpose(Lp[3]) * ((w .* translation) .* R[3]))
        @test norm(vec) / norm(reference) > 0.1
        @test norm(scal) / norm(reference) > 0.1
    end

    @testset "it is Galerkin-able, and that was measured rather than assumed" begin
        # #130's question. The double layers have NO mirrored receive rule -- their `k̂·n̂` flips under
        # the antipodal map -- and are Petrov-only. This one's `n̂` is contracted with the OTHER
        # pattern's `n̂` and never with `k̂`, so the coupling is the identity. Verified componentwise
        # against direct quadrature (6.6e-15 / 9.4e-15 / 3.2e-14) before it was written down; #129
        # shipped a plausible coupling that was wrong by 44 %.
        @test MLFMA.haslazyreceive(operator)
        rule = MLFMA.receivecouplingrule(operator)
        @test length(rule) == 3
        @test all(m -> m.coeff == 1, rule)
    end

    @testset "end to end, Galerkin and Petrov" begin
        endκ = 2π
        endoperator = Helmholtz2D.hypersingular(; wavenumber=endκ)
        X = lagrangec0d1(CompScienceMeshes.meshcircle(1.5, 1 / 6))
        dense = BEAST.assemble(endoperator, X, X)
        Random.seed!(5)
        x = randn(ComplexF64, numfunctions(X))

        # `0.25` gives relevantlevels 3:4, so the circular interpolator runs inside the matvec on a
        # THREE-component far field -- the only place the arity-3 trait is exercised end to end.
        for minhalfsize in (0.375, 0.25)
            tree = H2Trees.TwoNTree(
                X; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=minhalfsize, minvalues=0)
            )
            A = MLFMA.GalerkinMLFMA(endoperator, X; tree=tree, verbose=false)
            @test norm(A * x - dense * x) / norm(dense * x) < 1e-5      # measured 1.5e-7, 3.9e-7
            @test norm(adjoint(A) * x - dense' * x) / norm(dense' * x) < 1e-5
            @test norm(A * x) > 1e-12                                    # non-vacuity
        end

        # THE SHARP ADJOINT TEST. A same-space 2-D block is symmetric to round-off, so there `dense'`
        # is just `conj(dense)` and an adjoint that only conjugated would score identically -- the
        # #35/#61 shape. Two separated circles make the block genuinely non-symmetric.
        trialspace = lagrangec0d1(CompScienceMeshes.meshcircle(1.0, 1 / 8))
        testspace = lagrangec0d1(
            CompScienceMeshes.translate(
                CompScienceMeshes.meshcircle(1.0, 1 / 8), SVector{2,Float64}(5.0, 0.0)
            ),
        )
        petrovdense = BEAST.assemble(endoperator, testspace, trialspace)
        @test norm(petrovdense - transpose(petrovdense)) / norm(petrovdense) > 1.0

        B = MLFMA.PetrovGalerkinMLFMA(endoperator, testspace, trialspace; verbose=false)
        Random.seed!(7)
        u = randn(ComplexF64, numfunctions(trialspace))
        v = randn(ComplexF64, numfunctions(testspace))
        @test norm(B * u - petrovdense * u) / norm(petrovdense * u) < 1e-7   # measured 1.2e-10
        @test norm(adjoint(B) * v - petrovdense' * v) / norm(petrovdense' * v) < 1e-7
    end
end
