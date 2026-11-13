module EquidistantSamplingInterfaceTests

using Test
using MLFMA
using LinearAlgebra
using StaticArrays
using MLFMA.Equidistant:
    EquidistantSampling,
    EquidistantSamplingBuilder,
    exactquadratureweights,
    quadratureweights,
    directions

const S = EquidistantSampling{Float64}(12, 24)

@testset "the accessors EXTEND MLFMA's functions rather than shadowing them" begin
    # This is the whole point of the file. The pipeline calls `MLFMA.Nθ(s)`; a same-named function
    # living only in the submodule would never be reached, and the failure would be a MethodError deep
    # inside an assembler rather than anything pointing here. So each is called through `MLFMA.`.
    @test S isa MLFMA.AbstractSampling
    @test S isa MLFMA.AbstractSampling{Float64}
    @test MLFMA.Nθ(S) == 12
    @test MLFMA.Nφ(S) == 24
    @test eltype(S) == Float64
    @test length(S) == 12 * 24
    @test length.(MLFMA.θsφs(S)) == (12, 24)
    @test MLFMA.arraytype(S) == Matrix
    @test MLFMA.arraytype(EquidistantSampling{Float64}) == Matrix
    @test size(MLFMA.mirrorindices(S)) == size(S)
end

@testset "size reports (Nφ, Nθ), matching SphericalSampling" begin
    # A sampled array is [φ, θ]. Asserted against the spherical grid directly, so the two cannot drift.
    @test size(S) == (24, 12)
    @test size(S) == size(MLFMA.SphericalSampling{Float64}(12, 24))
end

@testset "cutoffparameter inverts NθNφ" begin
    # `NθNφ(L)` is the shared sizing rule; `cutoffparameter` must be its inverse on this grid too, or a
    # level built from a cutoff would report a different one back.
    for L in 0:8
        nθ, nφ = MLFMA.NθNφ(L)
        @test MLFMA.cutoffparameter(EquidistantSampling{Float64}(nθ, nφ)) == L
    end
    # ...and it agrees with the spherical grid at the same rate, which is what makes the two
    # interchangeable at a construction site.
    for (nθ, nφ) in ((6, 12), (12, 24), (20, 40))
        @test MLFMA.cutoffparameter(EquidistantSampling{Float64}(nθ, nφ)) ==
            MLFMA.cutoffparameter(MLFMA.SphericalSampling{Float64}(nθ, nφ))
    end
end

@testset "cartesiansamplepoints returns directions and EXACT weights" begin
    k̂, w = MLFMA.cartesiansamplepoints(S)
    @test size(k̂) == size(S)
    @test size(w) == size(S)
    @test all(v -> isapprox(norm(v), 1; atol=1e-14), k̂)
    # same directions the submodule's own `directions` produces, so there is one source of truth
    d = directions(S)
    @test maximum(norm(collect(k̂[i]) .- collect(d[i])) for i in eachindex(d)) < 1e-14
    # the weights are the spectrally exact ones, NOT the midpoint rule -- handing back the naive
    # weights here would silently cap every downstream integral near 1e-2
    @test w == exactquadratureweights(S)
    @test sum(w) ≈ 4π rtol = 1e-13
    @test !isapprox(sum(quadratureweights(S)), 4π; rtol=1e-13)
end

@testset "spherical points preserve Cartesian types, ordering and exact weights" begin
    for T in (Float32, Float64), (nθ, nφ) in ((1, 2), (7, 10), (12, 24))
        s = EquidistantSampling{T}(nθ, nφ)
        points, weights = MLFMA.sphericalsamplepoints(s)
        cartesian, cartesianweights = MLFMA.cartesiansamplepoints(s)
        @test points isa Matrix{SVector{3,T}}
        @test weights isa Matrix{T}
        @test size(points) == size(weights) == size(s)
        @test weights == cartesianweights == exactquadratureweights(s)
        @test sum(weights) ≈ T(4π) rtol = 20eps(T)
        @test weights != quadratureweights(s)
        for (q, (θ, φ)) in enumerate(s)
            @test points[q] == SVector{3,T}(1, θ, φ)
            r, θq, φq = points[q]
            @test cartesian[q] ≈
                SVector{3,T}(r * sin(θq) * cos(φq), r * sin(θq) * sin(φq), r * cos(θq)) rtol =
                10eps(T)
        end
        @test points[2] == SVector{3,T}(1, MLFMA.θs(s)[1], MLFMA.φs(s)[2])
        if nθ > 1
            @test points[nφ + 1] == SVector{3,T}(1, MLFMA.θs(s)[2], MLFMA.φs(s)[1])
        end
    end
end

@testset "the builder mirrors SphericalSamplingBuilder" begin
    b = EquidistantSamplingBuilder{Float64}()
    @test b(6, 12) == EquidistantSampling{Float64}(6, 12) ||
        (MLFMA.Nθ(b(6, 12)) == 6 && MLFMA.Nφ(b(6, 12)) == 12)
    @test b(6, 12) isa EquidistantSampling{Float64}
    @test EquidistantSamplingBuilder{Float32}()(4, 8) isa EquidistantSampling{Float32}
end

@testset "enumerate iterates in storage order" begin
    e = collect(enumerate(S))
    @test length(e) == length(S)
    @test e[1] == (1, (MLFMA.θs(S)[1], MLFMA.φs(S)[1]))
    # second entry advances φ, not θ -- φ is the contiguous dimension
    @test e[2] == (2, (MLFMA.θs(S)[1], MLFMA.φs(S)[2]))
end

@testset "wθs is deliberately NOT defined" begin
    # Gauss-Legendre carries a per-θ weight because its nodes are built for the sinθ factor. This grid
    # has no such vector, and `exactquadratureweights` takes over the role as a full array. A method
    # returning something plausible would be worse than none, so its absence is asserted.
    @test_throws MethodError MLFMA.wθs(S)
end

end # module
