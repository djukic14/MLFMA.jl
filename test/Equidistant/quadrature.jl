module EquidistantQuadratureTests

using Test
using MLFMA
using LinearAlgebra
using Random
using MLFMA.Equidistant:
    EquidistantSampling, θs, φs, quadratureweights, exactquadratureweights, integrate

include("ylmoracle.jl")

@testset "∮dΩ = 4π" begin
    for nθ in (4, 8, 12, 24)
        s = EquidistantSampling{Float64}(nθ, 2nθ)
        F = ones(2nθ, nθ)     # [φ, θ]
        @test integrate(F, s) ≈ 4π rtol = 1e-14
        # The plain midpoint rule is only O(h²) here -- the torus Jacobian |sin θ| has a kink -- and
        # this asserts that gap rather than leaving it as a footnote.
        @test abs(sum(quadratureweights(s) .* F) - 4π) / 4π > 1e-4
    end
end

@testset "harmonic orthogonality is exact for resolved modes" begin
    s = EquidistantSampling{Float64}(12, 24)
    pairs = ((0, 0), (1, 0), (2, 1), (3, -2), (4, 3), (5, 0))
    for (l1, m1) in pairs, (l2, m2) in pairs
        A = [ylm(l1, m1, θ, φ) * conj(ylm(l2, m2, θ, φ)) for φ in φs(s), θ in θs(s)]
        want = if (l1, m1) == (l2, m2)
            4π / (2l1 + 1) * factorial(l1 + abs(m1)) / factorial(l1 - abs(m1))
        else
            0.0
        end
        @test abs(integrate(A, s) - want) < 1e-11 * max(1.0, abs(want))
    end
end

end # module
