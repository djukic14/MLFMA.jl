# Leaf far-field assembly on a 2-D problem, against direct quadrature.
#
# WHY THIS EXISTS. Two lines on the leaf-assembly path were hardcoded to three dimensions, and both
# were CORRECT when written -- which is exactly why nothing caught them:
#
#   * `farfieldkernelvals` unrolled its dot product over three components, with a comment stating
#     `y`/`cartesian(x)`/`center` are "always `SVector{3}` here".
#   * `BufferFunctor` sized its per-element scratch with a hardcoded `ReferenceSimplex{2,Float64,3}`
#     -- a TRIANGLE. A 2-D problem is meshed with SEGMENTS, so the buffer came out one slot too long
#     and the assembly data was indexed off its end.
#
# Every 3-D fixture in the suite passes either way, so only a genuinely 2-D assembly can hold these.
#
# THE ORACLE IS INDEPENDENT. A radiation pattern is `R_n(k̂) = ∫ g_n(y) exp(γ k̂·y) dy`, so it is
# checked by evaluating that integral directly over the mesh -- own quadrature, own basis evaluation,
# no reuse of the assembler's kernel. Reusing `farfieldkernelvals` here would test it against itself.

using BEAST
using CompScienceMeshes
using LinearAlgebra
using MLFMA
using StaticArrays
using Test

# `R_n(k̂) = ∫ g_n(y) exp(γ k̂·y) dy`, by direct quadrature over the basis function's own support.
function _directradiationpattern(X, mesh, γ, k̂, n; order=9)
    refspace = BEAST.refspace(X)
    cells = CompScienceMeshes.cells(mesh)
    accumulated = zero(ComplexF64)
    for shape in X.fns[n]
        element = CompScienceMeshes.chart(mesh, cells[shape.cellid])
        for (point, weight) in CompScienceMeshes.quadpoints(element, order)
            value = refspace(point)[shape.refid].value
            accumulated +=
                weight *
                shape.coeff *
                value *
                exp(γ * dot(k̂, CompScienceMeshes.cartesian(point)))
        end
    end
    return accumulated
end

@testset "2D leaf far-field assembly" begin
    mesh = CompScienceMeshes.meshcircle(1.0, 0.3)
    X = lagrangec0d1(mesh)
    γ = 2.0im

    sampling = MLFMA.CircularSampling{Float64}(16)
    k̂, _ = MLFMA.cartesiansamplepoints(sampling)

    operator = MLFMA.HHSingleLayerFarField(; gamma=γ)
    assembler = MLFMA.BEASTFarFieldAssembler(
        operator,
        X,
        BEAST.SingleNumQStrat(8),
        MLFMA.StoreAsFarFieldValue(),
        MLFMA.ScalarFarField,
    )
    F = assembler(sampling; verbose=false)

    @testset "it assembles at all, with the right shape" begin
        # `size` is `(ndirections, nbasisfunctions)` -- and a `Matrix`, because the far field of a
        # 1-tuple grid is a `Vector` PER BASIS FUNCTION, stacked.
        @test size(F[1]) == (length(sampling), numfunctions(X))
        @test all(isfinite, F[1])
        @test any(!iszero, F[1])
    end

    @testset "every pattern matches direct quadrature" begin
        worst = 0.0
        for n in 1:numfunctions(X), q in eachindex(k̂)
            reference = _directradiationpattern(X, mesh, γ, k̂[q], n)
            worst = max(worst, abs(F[1][q, n] - reference) / abs(reference))
        end
        # Both sides integrate the same smooth integrand; the only difference is the quadrature
        # order, so they agree to well past what an MLFMA ever needs.
        @test worst < 1e-10
    end

    @testset "the receive operator is the conjugate convention" begin
        # `radiationoperator`/`receiveoperator` differ only in the sign of γ, which is what makes the
        # receive pattern the conjugate for a real grid. Asserted on 2-D input because that sign is
        # carried through the same `farfieldkernelvals` this file exists to pin.
        receiveoperator = MLFMA.HHSingleLayerFarField(; gamma=(-γ))
        receiveassembler = MLFMA.BEASTFarFieldAssembler(
            receiveoperator,
            X,
            BEAST.SingleNumQStrat(8),
            MLFMA.StoreAsFarFieldValue(),
            MLFMA.ScalarFarField,
        )
        L = receiveassembler(sampling; verbose=false)
        @test size(L[1]) == size(F[1])
        # `exp(-γ k̂·y)` against `exp(+γ k̂·y)` with a purely imaginary γ and a real grid.
        @test all(L[1] .≈ conj.(F[1]))
    end
end
