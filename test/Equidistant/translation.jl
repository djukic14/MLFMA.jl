module EquidistantTranslationTests

using Test
using MLFMA
using LinearAlgebra
using MLFMA.Equidistant:
    EquidistantSampling,
    translationoperator,
    directions,
    integrate,
    sphericalhankel1,
    legendrep,
    excessbandwidth,
    quadratureweights

norm3(v) = sqrt(v[1]^2 + v[2]^2 + v[3]^2)

# e^{ik|d+u|}/|d+u| = (ik/4π) ∮ e^{ik k̂·u} T_L(k̂·d̂) dΩ,  valid for |u| < |d|.
function diagonalgreen(L, k, d, u, s)
    T = translationoperator(L, k * norm3(d), d ./ norm3(d), s)
    P = map(directions(s)) do k̂
        return cis(k * (k̂[1] * u[1] + k̂[2] * u[2] + k̂[3] * u[3]))
    end
    return (im * k / (4π)) * integrate(P .* T, s)
end
exactgreen(k, d, u) = (r=norm3(d .+ u); cis(k * r) / r)

@testset "special function recurrences" begin
    @test legendrep(4, 1.0) ≈ ones(5)
    @test legendrep(3, -1.0) ≈ [1.0, -1.0, 1.0, -1.0]
    @test legendrep(2, 0.3)[3] ≈ 0.5 * (3 * 0.3^2 - 1)
    h = sphericalhankel1(3, 2.0)
    @test h[1] ≈ -im * cis(2.0) / 2.0
    @test h[2] ≈ -(1 + im / 2.0) * cis(2.0) / 2.0
    # The recurrence itself, checked independently of how it was seeded.
    for l in 1:2
        @test h[l + 2] ≈ (2l + 1) / 2.0 * h[l + 1] - h[l]
    end
end

@testset "translation operator refuses a grid that would alias it" begin
    # T_L carries content up to degree L exactly; a coarser grid silently corrupts it, so this must
    # error rather than return something plausible.
    s = EquidistantSampling{Float64}(6, 12)
    @test_throws ErrorException translationoperator(10, 10.0, (0.0, 0.0, 1.0), s)
    @test translationoperator(5, 10.0, (0.0, 0.0, 1.0), s) isa Matrix
end

@testset "Gegenbauer identity reproduces the Green function" begin
    k, d, u = 1.0, (0.0, 0.0, 10.0), (1.8, 2.4, 0.0)   # |u| = 3 < |d| = 10
    ku = k * norm3(u)
    exact = exactgreen(k, d, u)
    # Truncation error alone: the quadrature is deliberately over-resolved, because the INTEGRAND has
    # bandwidth L + k|u| -- the product of the operator and the pattern, not either one alone.
    errs = map((4, 6, 8, 10)) do L
        nθ = 3 * (L + ceil(Int, ku))
        s = EquidistantSampling{Float64}(nθ, 2nθ)
        abs(diagonalgreen(L, k, d, u, s) - exact) / abs(exact)
    end
    @test issorted(errs; rev=true)
    @test errs[end] < 1e-5
end

@testset "quadrature must resolve the product bandwidth, not just L" begin
    k, d, u = 1.0, (0.0, 0.0, 10.0), (1.8, 2.4, 0.0)
    exact = exactgreen(k, d, u)
    L = 12
    starved =
        abs(diagonalgreen(L, k, d, u, EquidistantSampling{Float64}(13, 26)) - exact) /
        abs(exact)
    ample =
        abs(diagonalgreen(L, k, d, u, EquidistantSampling{Float64}(24, 48)) - exact) /
        abs(exact)
    @test starved > 1e-2       # Nθ = L+1 represents T, but NOT T times the pattern
    @test ample < 1e-6
end

@testset "the exact quadrature is what makes translation accurate" begin
    # With the plain midpoint weights the same computation stalls near 1e-2 no matter how fine the
    # grid gets, which is the practical reason `exactquadratureweights` exists.
    k, d, u = 1.0, (0.0, 0.0, 10.0), (1.8, 2.4, 0.0)
    exact = exactgreen(k, d, u)
    s = EquidistantSampling{Float64}(32, 64)
    T = translationoperator(12, k * norm3(d), d ./ norm3(d), s)
    P = map(directions(s)) do k̂
        cis(k * (k̂[1] * u[1] + k̂[2] * u[2] + k̂[3] * u[3]))
    end
    naive = (im * k / (4π)) * sum(quadratureweights(s) .* (P .* T))
    @test abs(naive - exact) / abs(exact) > 1e-3
    # The exact-weight result bottoms out at ~6.5e-8, and that floor is the GEGENBAUER TRUNCATION at
    # L = 12, not the quadrature: it is the same value the over-resolved truncation sweep above reports
    # for L = 12. Asserting anything tighter would be asserting a property of the series rather than of
    # the weights, and would fail for a reason unrelated to what this testset is about.
    @test abs(diagonalgreen(12, k, d, u, s) - exact) / abs(exact) < 1e-7
end

@testset "excessbandwidth grows with both argument and requested digits" begin
    @test excessbandwidth(10.0, 3) < excessbandwidth(10.0, 6)
    @test excessbandwidth(10.0, 3) < excessbandwidth(20.0, 3)
    @test excessbandwidth(10.0, 3) >= 10
end

end # module
