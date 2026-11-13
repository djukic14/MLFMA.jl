# The frame projection and the antipodal mirror.
#
# The mirror testset is the point of this file. The claim it checks -- that the sample permutation
# `MLFMA.mirrorindices` and the ±1 table `MLFMA.mirrorrule` together become ONE SIGN PER DEGREE in
# coefficient space -- is what lets the Galerkin receive side under SHE carry no data at all. It was
# derived, not measured, so it is asserted here against MLFMA's OWN mirror machinery rather than
# against a restatement of the same algebra.

using MLFMA
using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics:
    SphericalHarmonicTransform,
    degreeof,
    lmindex,
    mirrorsynthesis,
    numcoefficients,
    parity,
    project!,
    projectthreaded!,
    sphericalframe,
    synthesize,
    ylm!
using LinearAlgebra
using Random
using StaticArrays

@testset "Spherical frame" begin
    @testset "θ̂ and φ̂ are an orthonormal tangential pair" begin
        s = MLFMA.SphericalSampling{Float64}(10, 20)
        θ̂, φ̂ = sphericalframe(s)
        k̂, _ = MLFMA.cartesiansamplepoints(s)
        for q in eachindex(vec(k̂))
            t = SVector(θ̂[q, 1], θ̂[q, 2], θ̂[q, 3])
            p = SVector(φ̂[q, 1], φ̂[q, 2], φ̂[q, 3])
            r = vec(k̂)[q]
            @test norm(t) ≈ 1 atol = 1e-14
            @test norm(p) ≈ 1 atol = 1e-14
            @test abs(dot(t, p)) < 1e-14
            @test abs(dot(t, r)) < 1e-14   # tangential: both ⊥ k̂
            @test abs(dot(p, r)) < 1e-14
            @test cross(t, p) ≈ r atol = 1e-14   # right-handed (θ̂ × φ̂ = r̂)
        end
    end

    @testset "projection matches the store rule's own tangential components" begin
        # The projection has to agree with what `StoreCartesianAsTangentialSpherical` produces at
        # assembly, since SHE replaces that step. Checked against a direct dot product rather than
        # against the storer, whose kernel takes one value at a time.
        Random.seed!(71)
        s = MLFMA.SphericalSampling{Float64}(8, 16)
        θ̂, φ̂ = sphericalframe(s)
        nd = length(θ̂[:, 1])
        ngroups = 4

        C = randn(ComplexF64, nd, 3ngroups)
        F = Matrix{ComplexF64}(undef, nd, 2ngroups)
        project!(F, C, θ̂, φ̂)

        for g in 1:ngroups, q in 1:nd
            v = SVector(C[q, 3g - 2], C[q, 3g - 1], C[q, 3g])
            @test F[q, 2g - 1] ≈ θ̂[q, 1] * v[1] + θ̂[q, 2] * v[2] + θ̂[q, 3] * v[3] atol =
                1e-14
            @test F[q, 2g] ≈ φ̂[q, 1] * v[1] + φ̂[q, 2] * v[2] + φ̂[q, 3] * v[3] atol = 1e-14
        end
    end

    @testset "threaded projection agrees with the serial one" begin
        Random.seed!(72)
        s = MLFMA.SphericalSampling{Float64}(12, 24)
        θ̂, φ̂ = sphericalframe(s)
        nd = size(θ̂, 1)
        ngroups = 9
        C = randn(ComplexF64, nd, 3ngroups)
        serial = Matrix{ComplexF64}(undef, nd, 2ngroups)
        threaded = similar(serial)
        project!(serial, C, θ̂, φ̂)
        projectthreaded!(threaded, C, θ̂, φ̂)
        @test serial == threaded          # same arithmetic in the same order, so EXACT
    end

    @testset "projection shape errors are caught" begin
        s = MLFMA.SphericalSampling{Float64}(6, 12)
        θ̂, φ̂ = sphericalframe(s)
        nd = size(θ̂, 1)
        @test_throws DimensionMismatch project!(
            Matrix{ComplexF64}(undef, nd, 2), randn(ComplexF64, nd + 1, 3), θ̂, φ̂
        )
        # 4 Cartesian columns is not a whole number of groups
        @test_throws DimensionMismatch project!(
            Matrix{ComplexF64}(undef, nd, 2), randn(ComplexF64, nd, 4), θ̂, φ̂
        )
        # 2 groups of Cartesian into room for 1 tangential
        @test_throws DimensionMismatch project!(
            Matrix{ComplexF64}(undef, nd, 2), randn(ComplexF64, nd, 6), θ̂, φ̂
        )
    end
end

@testset "Antipodal mirror in coefficient space" begin
    @testset "parity is (-1)^l per coefficient" begin
        for P in 0:6
            p = parity(P)
            @test length(p) == numcoefficients(P)
            for l in 0:P, m in (-l):l
                @test p[lmindex(l, m)] == (isodd(l) ? -1 : 1)
            end
        end
    end

    @testset "Y_lm(-k̂) = (-1)^l Y_lm(k̂)" begin
        # The parity relation itself, straight off `ylm!` -- everything below rests on it.
        P = 7
        forward = Vector{ComplexF64}(undef, numcoefficients(P))
        mirrored = similar(forward)
        Random.seed!(73)
        for _ in 1:20
            θ = π * rand()
            φ = 2π * rand()
            ylm!(forward, P, θ, φ)
            ylm!(mirrored, P, π - θ, mod2pi(φ + π))
            for q in eachindex(forward)
                @test mirrored[q] ≈ (isodd(degreeof(q)) ? -forward[q] : forward[q]) atol =
                    1e-12
            end
        end
    end

    @testset "the sign REPLACES MLFMA.mirrorindices, on MLFMA's own permutation" begin
        # THE LOAD-BEARING TEST. `synthesize` of the parity-scaled coefficients must equal the
        # forward synthesis read through `MLFMA.mirrorindices` -- i.e. the diagonal really is the
        # same operator as the sample permutation the sampled receive side uses.
        Random.seed!(74)
        P = 6
        s = MLFMA.SphericalSampling{Float64}(2 * (P + 1), 4 * (P + 1))
        t = SphericalHarmonicTransform(ComplexF64, s, P)
        indices = vec(MLFMA.mirrorindices(s))

        for _ in 1:5
            a = randn(ComplexF64, numcoefficients(P), 3)
            forward = synthesize(t, a)
            viaparity = synthesize(t, parity(P) .* a)
            @test viaparity ≈ forward[indices, :] rtol = 1e-12
        end
    end

    @testset "mirrorsynthesis is that scaling, folded into the matrix" begin
        Random.seed!(75)
        P = 5
        s = MLFMA.SphericalSampling{Float64}(2 * (P + 1), 4 * (P + 1))
        t = SphericalHarmonicTransform(ComplexF64, s, P)
        S⁻ = mirrorsynthesis(t)
        @test size(S⁻) == size(t.synthesis)
        a = randn(ComplexF64, numcoefficients(P), 3)
        @test S⁻ * a ≈ synthesize(t, parity(P) .* a) rtol = 1e-13
        # applying it twice is the identity, since (-1)^l squares away
        @test mirrorsynthesis(SphericalHarmonicTransform(P, S⁻, t.weights)) ≈ t.synthesis
    end

    @testset "the tangential COMPONENT signs come out of the frame" begin
        # The other half of the claim: projecting the mirrored Cartesian field must reproduce
        # `mirrorrule(TangentialSphericalFarField) == (+1 on θ̂, -1 on φ̂)` without that table being
        # consulted. Checked against MLFMA's rule rather than against the literal signs, so a change
        # to the rule surfaces here.
        Random.seed!(76)
        P = 6
        s = MLFMA.SphericalSampling{Float64}(2 * (P + 1), 4 * (P + 1))
        t = SphericalHarmonicTransform(ComplexF64, s, P)
        θ̂, φ̂ = sphericalframe(s)
        nd = size(θ̂, 1)
        indices = vec(MLFMA.mirrorindices(s))

        rule = MLFMA.mirrorrule(MLFMA.TangentialSphericalFarField)
        signs = Dict(MLFMA.destcomponent(m) => MLFMA.coefficient(m) for m in rule)
        @test signs[1] == 1 && signs[2] == -1   # the fixture is the rule this test encodes

        a = randn(ComplexF64, numcoefficients(P), 3)
        C = synthesize(t, a)
        Cmirrored = synthesize(t, parity(P) .* a)

        F = Matrix{ComplexF64}(undef, nd, 2)
        Fmirrored = similar(F)
        project!(F, C, θ̂, φ̂)
        project!(Fmirrored, Cmirrored, θ̂, φ̂)

        # tangential component `d` of the mirrored pattern == rule sign × the forward pattern read
        # through the mirror permutation
        @test Fmirrored[:, 1] ≈ signs[1] * F[indices, 1] rtol = 1e-12
        @test Fmirrored[:, 2] ≈ signs[2] * F[indices, 2] rtol = 1e-12
    end
end
