using CompScienceMeshes
using BEAST
using MLFMA
using Test
using StaticArrays

# `@test_nowarn` is the wrong assertion for these: `BEAST.potential` writes a ProgressMeter to stderr,
# which is progress output rather than a warning, so the check failed on output it should not have been
# policing. It also asserted nothing about the result. Each call is now checked for the properties that
# actually matter -- one value per evaluation point, all finite, and not uniformly zero (which a
# silently-empty assembly would produce).
function checkfarfield(values, points)
    @test length(values) == length(points)
    flat = collect(Iterators.flatten(values))
    @test all(isfinite, flat)
    @test !all(iszero, flat)
    return values
end

@testset "Maxwell3D FarFields" begin
    λ = 1.0
    m = meshsphere(0.5 * λ, λ / 10)
    X = raviartthomas(m)

    ffMLFMA = MLFMA.MWSingleLayerFarField3D(; wavenumber=1.0, amplitude=1.0)
    u = Array(1:numfunctions(X))

    T, P = range(0.0; stop=π, length=7), range(0.0; stop=2π, length=13)
    points = vec([point(cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)) for θ in T, ϕ in P])
    checkfarfield(potential(ffMLFMA, points, u, X), points)
end

@testset "Helmholtz FarFields" begin
    λ = 1.0
    m = meshsphere(0.5 * λ, λ / 10)
    X = lagrangec0d1(m)
    u = Array(1:numfunctions(X))

    T, P = range(0.0; stop=π, length=7), range(0.0; stop=2π, length=13)
    points = vec([point(cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)) for θ in T, ϕ in P])

    ffS = MLFMA.HHSingleLayerFarField(; wavenumber=1.0)
    checkfarfield(potential(ffS, points, u, X; type=ComplexF64), points)

    ffD = MLFMA.HHDoubleLayerFarField(; wavenumber=1.0)
    checkfarfield(potential(ffD, points, u, X; type=ComplexF64), points)

    ffH = MLFMA.HHHyperSingularFarField3D(; wavenumber=1.0)
    checkfarfield(potential(ffH, points, u, X; type=SVector{6,ComplexF64}), points)
end
