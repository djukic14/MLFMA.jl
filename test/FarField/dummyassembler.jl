using MLFMA
using BEAST, CompScienceMeshes
using StaticArrays
using ClusterTrees
using Test
using LinearAlgebra
using SparseArrays

include("../support/util_test.jl")
@testset "Dummy Far Field" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    m = CompScienceMeshes.read_gmsh_mesh(
        joinpath(pkgdir(MLFMA), "test", "assets", "tworects.msh")
    )
    X = raviartthomas(m)
    farfieldoperator = BEAST.MWFarField3D(; wavenumber=κ)

    farfieldassembler = MLFMA.DummyFarFieldAssembler(
        farfieldoperator,
        X,
        BEAST.DoubleNumWiltonSauterQStrat(2, 3, 2, 3, 2, 2, 3, 2),
        MLFMA.StoreCartesianAsTangentialSpherical(),
        MLFMA.TangentialSphericalFarField,
    )

    @test MLFMA.quadstrategy(farfieldassembler) ==
        BEAST.DoubleNumWiltonSauterQStrat(2, 3, 2, 3, 2, 2, 3, 2)
    @test MLFMA.space(farfieldassembler) == X
    @test MLFMA.operator(farfieldassembler) == farfieldoperator

    s, _ = MLFMA.sampling(κ, 1e-2, 1.0)
    samplepoints = MLFMA.cartesiansamplepoints(s)[1]

    # The entries are `dummyfarfieldvalue`, not ones -- see the note atop
    # src/testsupport/dummyfarfieldassembler.jl. Rebuilt here from the documented formula rather than by
    # calling the assembler twice, so this pins the CONTRACT and not merely self-consistency.
    F = farfieldassembler(samplepoints)
    np, nf = length(samplepoints), numfunctions(X)
    @test F == SVector(
        ComplexF64[MLFMA.dummyfarfieldvalue(ComplexF64, 1, p, n) for p in 1:np, n in 1:nf],
        ComplexF64[MLFMA.dummyfarfieldvalue(ComplexF64, 2, p, n) for p in 1:np, n in 1:nf],
    )
    @test eltype(F[1]) == ComplexF64
    @test size(F[1]) == (np, nf)

    # --- the properties every test using this assembler depends on --------------------------------
    # A fill of ones satisfied none of these, which is precisely how it hid transpose, conjugation and
    # test/trial side-swap bugs. Pinned so that blindness cannot come back unnoticed.

    # Genuinely complex, so a missing `conj` changes the answer.
    @test maximum(abs, imag.(F[1])) / maximum(abs, F[1]) > 0.1
    # Distinct per COMPONENT, so supplying Fθ where Fφ belongs is visible.
    @test F[1] != F[2]
    # Distinct per SAMPLE POINT and per BASIS FUNCTION, so a transposed or axis-swapped collection is
    # visible. A small slice keeps this O(1) on a mesh of any size.
    block = F[1][1:min(6, np), 1:min(6, nf)]
    @test allunique(vec(block))
    @test all(
        block[i, :] != block[j, :] for i in axes(block, 1) for j in axes(block, 1) if i < j
    )
    @test all(
        block[:, i] != block[:, j] for i in axes(block, 2) for j in axes(block, 2) if i < j
    )
    # Bounded modulus, so aggregating over a leaf stays a value check rather than a tolerance argument.
    @test all(0.5 .< abs.(F[1]) .< 2.0)
    # Deterministic and RNG-free: a second call is bit-identical.
    @test farfieldassembler(samplepoints) == F

    @test_nowarn farfieldassembler(samplepoints, verbose=true)
end
