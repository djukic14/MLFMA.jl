# Tangential vector spherical harmonics, checked against analytic ground truth and against MLFMA's
# own mirror machinery.
#
# WHAT IS AT RISK HERE, and therefore what these tests are shaped around:
#
#   * The two Legendre auxiliaries. `∂_θY` comes from a recurrence, not from differencing, so it is
#     checked against differencing -- and `A = -imY/(√(l(l+1))sinθ)` against the closed-form l = 1
#     harmonics, where the whole basis can be written down.
#   * ORTHONORMALITY, which everything else rests on: the analysis is the synthesis's adjoint, and
#     that is only an inverse because the basis is orthonormal.
#   * The MIRROR, which was DERIVED. `a' = (-1)^l a` on X and `b' = (-1)^(l+1) b` on Ψ is an algebraic
#     claim about MLFMA's `mirrorindices`/`mirrorrule`, so it is asserted against those rather than
#     against a restatement -- exactly as `frame.jl` does for the scalar case.
#   * The 1/sinθ. Poles are excluded by construction, but the interesting question is whether the
#     near-pole rows cost accuracy on the grids MLFMA uses. They do not, and the round-trip test
#     below is what says so.

using MLFMA
using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics:
    VectorSphericalHarmonicTransform,
    numcoefficients,
    lmindex,
    numvshcoefficients,
    vshanalyze,
    vshauxiliaries!,
    vshdegreeenergy,
    vshdegreeof,
    vshindex,
    vshmirrorblocks,
    vshparity,
    vshsynthesize,
    vshsynthesize!,
    vshtruncationorder,
    gausslegendregrid,
    numdirections
using LinearAlgebra
using Random
using Test

@testset "Tangential mode indexing" begin
    for P in 1:8
        @test numvshcoefficients(P) == P * (P + 2)
        # contiguous and gap-free over 1:P(P+2), which is what makes truncation a row slice
        seen = [vshindex(l, m) for l in 1:P for m in (-l):l]
        @test sort(seen) == collect(1:numvshcoefficients(P))
        for l in 1:P, m in (-l):l
            @test vshdegreeof(vshindex(l, m)) == l
            # the scalar index shifted by exactly one -- the property the transform's row slice uses
            @test vshindex(l, m) == lmindex(l, m) - 1
        end
    end
end

@testset "The Legendre auxiliaries" begin
    P = 6
    Y = Vector{ComplexF64}(undef, numcoefficients(P))
    D = similar(Y)
    A = similar(Y)

    @testset "∂_θY against central differences" begin
        # The recurrence is the thing under test, so the reference must NOT be a recurrence.
        worst = 0.0
        for θ in (0.4, 1.0, π / 2, 2.3, 2.9), φ in (0.3, 2.7)
            vshauxiliaries!(Y, D, A, P, θ, φ)
            h = 1e-6
            Yplus = Vector{ComplexF64}(undef, numcoefficients(P))
            Yminus = similar(Yplus)
            ylm!(Yplus, P, θ + h, φ)
            ylm!(Yminus, P, θ - h, φ)
            difference = (Yplus .- Yminus) ./ (2h)
            for l in 1:P, m in (-l):l
                q = lmindex(l, m)
                r = sqrt(l * (l + 1))
                scale = max(abs(difference[q]), 1.0)
                worst = max(worst, abs(D[q] * r - difference[q]) / scale)
            end
        end
        # central differences at h = 1e-6 are good to about 1e-10; anything much above that is the
        # recurrence being wrong, not the reference being coarse
        @test worst < 1e-8
    end

    @testset "against the closed-form degree-1 harmonics" begin
        # Y_1,0 = √(3/4π) cosθ, Y_1,±1 = ∓√(3/8π) sinθ e^{±iφ} -- Condon-Shortley, matching `ylm.jl`.
        for θ in (0.6, 1.2, 2.4), φ in (0.0, 1.7)
            vshauxiliaries!(Y, D, A, P, θ, φ)
            r = sqrt(2.0)                         # √(l(l+1)) at l = 1
            @test D[lmindex(1, 0)] * r ≈ -sqrt(3 / (4π)) * sin(θ) rtol = 1e-12
            @test D[lmindex(1, 1)] * r ≈ -sqrt(3 / (8π)) * cos(θ) * cis(φ) rtol = 1e-12
            @test A[lmindex(1, 0)] ≈ 0 atol = 1e-14
            @test A[lmindex(1, 1)] * r ≈ -im * (-sqrt(3 / (8π)) * sin(θ) * cis(φ)) / sin(θ) rtol =
                1e-12
        end
    end

    @testset "the l = 0 slot is zeroed, and poles are refused" begin
        vshauxiliaries!(Y, D, A, P, 1.0, 0.5)
        @test iszero(D[lmindex(0, 0)])
        @test iszero(A[lmindex(0, 0)])
        # 1/sinθ is in the definition of the basis, so this is a precondition and not a limitation
        @test_throws DomainError vshauxiliaries!(Y, D, A, P, 0.0, 0.5)
        @test_throws DomainError vshauxiliaries!(Y, D, A, P, Float64(π), 0.5)
    end
end

@testset "The tangential basis is orthonormal" begin
    # EVERYTHING RESTS ON THIS. `vshanalyze` is the synthesis's adjoint, which is only its inverse
    # because ⟨X,X⟩ = ⟨Ψ,Ψ⟩ = I and ⟨X,Ψ⟩ = 0.
    P = 5
    t = VectorSphericalHarmonicTransform(ComplexF64, gausslegendregrid(P + 4)..., P)
    n = numvshcoefficients(P)
    w = t.weights

    # ⟨X_i, X_j⟩ = Σ_q w_q (conj(X_θi) X_θj + conj(X_φi) X_φj), with X_θ = A and X_φ = D
    XX = t.A' * (w .* t.A) .+ t.D' * (w .* t.D)
    # Ψ_θ = D, Ψ_φ = -A
    ΨΨ = t.D' * (w .* t.D) .+ t.A' * (w .* t.A)
    XΨ = t.A' * (w .* t.D) .- t.D' * (w .* t.A)

    @test norm(XX - I, Inf) < 1e-12
    @test norm(ΨΨ - I, Inf) < 1e-12
    @test norm(XΨ, Inf) < 1e-12
    @test size(t.A) == (numdirections(t), n)
end

@testset "Synthesis and analysis invert each other" begin
    Random.seed!(31)
    for P in (3, 6, 9)
        t = VectorSphericalHarmonicTransform(ComplexF64, gausslegendregrid(P + 2)..., P)
        n = numvshcoefficients(P)
        for _ in 1:3
            a = randn(ComplexF64, n)
            b = randn(ComplexF64, n)
            Fθ, Fφ = vshsynthesize(t, a, b)
            arecovered, brecovered = vshanalyze(t, Fθ, Fφ)
            @test arecovered ≈ a rtol = 1e-11
            @test brecovered ≈ b rtol = 1e-11
        end

        # batched: one expansion per column, which is the shape the leaf multiply uses
        A = randn(ComplexF64, n, 4)
        B = randn(ComplexF64, n, 4)
        Fθ, Fφ = vshsynthesize(t, A, B)
        Ar, Br = vshanalyze(t, Fθ, Fφ)
        @test Ar ≈ A rtol = 1e-11
        @test Br ≈ B rtol = 1e-11
    end
end

@testset "The 1/sinθ costs no accuracy on the grids MLFMA uses" begin
    # The one thing that could have sunk this basis. `Y_lm ~ (sinθ)^|m|` cancels the `1/sinθ`
    # exactly, so the near-pole rows are not amplified -- asserted here rather than argued, at the
    # analysis orders the builder actually uses and at the tightest θ the grid produces.
    Random.seed!(88)
    P = 10
    for Pa in (P + 2, P + 4, P + 8)
        t = VectorSphericalHarmonicTransform(ComplexF64, gausslegendregrid(Pa)..., P)
        n = numvshcoefficients(P)
        a, b = randn(ComplexF64, n), randn(ComplexF64, n)
        Fθ, Fφ = vshsynthesize(t, a, b)
        ar, br = vshanalyze(t, Fθ, Fφ)
        @test norm(vcat(ar - a, br - b)) / norm(vcat(a, b)) < 1e-12

        # and the operator itself is well conditioned -- measured 2.4-2.6, so 10 is loose by a
        # factor of four and would still catch a genuinely singular basis
        block = [t.A t.D; t.D -t.A]
        @test cond(block) < 10
    end
end

@testset "The antipodal mirror is one sign per degree, per family" begin
    # DERIVED, SO CHECKED AGAINST MLFMA'S OWN MACHINERY. `mirrorindices` is the sample permutation
    # and `mirrorrule` the ±1 component table that the sampled receive side uses; the claim is that
    # together they are `(-1)^l` on X and `(-1)^(l+1)` on Ψ. Restating the algebra here would test
    # nothing.
    Random.seed!(90)
    P = 6
    s = MLFMA.SphericalSampling{Float64}(2 * (P + 1), 4 * (P + 1))
    t = VectorSphericalHarmonicTransform(ComplexF64, s, P)
    indices = vec(MLFMA.mirrorindices(s))
    n = numvshcoefficients(P)

    rule = MLFMA.mirrorrule(MLFMA.TangentialSphericalFarField)
    signs = Dict(MLFMA.destcomponent(m) => MLFMA.coefficient(m) for m in rule)
    @test signs[1] == 1 && signs[2] == -1     # the fixture is the rule this test encodes

    @testset "the parity table" begin
        p = vshparity(P)
        @test length(p) == n
        for l in 1:P, m in (-l):l
            @test p[vshindex(l, m)] == (iseven(l) ? 1 : -1)
        end
    end

    @testset "mirrored synthesis IS the permuted forward one" begin
        for _ in 1:5
            a = randn(ComplexF64, n)
            b = randn(ComplexF64, n)
            Fθ, Fφ = vshsynthesize(t, a, b)
            Mθ, Mφ = vshsynthesize(t, a, b; mirrored=true)
            # `mirrorrule`: θ gathered as is, φ negated
            @test Mθ ≈ signs[1] .* Fθ[indices] rtol = 1e-11
            @test Mφ ≈ signs[2] .* Fφ[indices] rtol = 1e-11
        end
    end

    @testset "and equals scaling the coefficients per family" begin
        # The form the receive collection uses: the mirror is a diagonal on the COEFFICIENTS, so it
        # can be folded into the level's blocks once instead of permuting NΩ samples per leaf.
        p = vshparity(P)
        for _ in 1:3
            a = randn(ComplexF64, n)
            b = randn(ComplexF64, n)
            Mθ, Mφ = vshsynthesize(t, a, b; mirrored=true)
            viacoefficients = vshsynthesize(t, p .* a, .-(p .* b))
            @test Mθ ≈ viacoefficients[1] rtol = 1e-11
            @test Mφ ≈ viacoefficients[2] rtol = 1e-11
        end
    end

    @testset "the folded blocks match the keyword" begin
        Aᵖ, Dᵖ = vshmirrorblocks(t)
        @test Aᵖ ≈ t.A .* transpose(vshparity(P))
        @test Dᵖ ≈ t.D .* transpose(vshparity(P))
    end
end

@testset "Degree energy and the truncation order" begin
    P = 8
    t = VectorSphericalHarmonicTransform(ComplexF64, gausslegendregrid(P + 2)..., P)
    n = numvshcoefficients(P)

    @testset "Parseval: coefficient energy is field energy" begin
        Random.seed!(101)
        a, b = randn(ComplexF64, n), randn(ComplexF64, n)
        Fθ, Fφ = vshsynthesize(t, a, b)
        fieldenergy = sum(t.weights .* (abs2.(Fθ) .+ abs2.(Fφ)))
        @test sum(vshdegreeenergy(a, b)) ≈ fieldenergy rtol = 1e-10
    end

    @testset "a band-limited expansion truncates at its own band" begin
        for band in 2:5
            a = zeros(ComplexF64, n)
            b = zeros(ComplexF64, n)
            Random.seed!(4band)
            for l in 1:band, m in (-l):l
                a[vshindex(l, m)] = randn(ComplexF64)
                b[vshindex(l, m)] = randn(ComplexF64)
            end
            @test vshtruncationorder(a, b, 1e-12) == band
            # the energy above the band really is zero, so this is exact rather than tolerant
            @test sum(@view vshdegreeenergy(a, b)[(band + 1):end]) == 0
        end
    end

    @testset "a tighter tolerance never asks for a lower order" begin
        Random.seed!(102)
        a, b = randn(ComplexF64, n), randn(ComplexF64, n)
        orders = [vshtruncationorder(a, b, ε) for ε in (1e-1, 1e-2, 1e-4, 1e-8)]
        @test issorted(orders)
    end
end
