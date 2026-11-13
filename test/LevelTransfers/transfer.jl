using MLFMA
using Test
using CompScienceMeshes, BEAST
using StaticArrays
using LinearAlgebra
using H2Factory

include("../support/util_test.jl")

# Verifies MLFMA's level transfer against explicit dense matrices, one direction at a time.
#
# The transfer is `Phase ∘ Interp`, so the dense reference is `Diagonal(phase) * intermatrix`. Every
# direction is checked against that ONE reference put through plain LinearAlgebra -- forward against
# `M`, transpose against `transpose(M)`, adjoint against `adjoint(M)` -- so a direction that quietly
# conjugated (or failed to conjugate) the phase cannot pass.
@testset "InterpolationPhaseTransfer against a dense reference" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ
    T = ComplexF64

    # Deliberately different θ/φ counts on the two grids, so the transfer is a NON-SQUARE,
    # non-symmetric operator: transpose is then structurally distinguishable from the forward
    # direction, not merely numerically.
    childsampling = MLFMA.SphericalSampling{Float64}(8, 16)
    parentsampling = MLFMA.SphericalSampling{Float64}(6, 12)

    stencil = 4
    interpolator = MLFMA.SphericalLagrangeInterpolator{stencil}(T, Float64)
    anterpolator = MLFMA.SphericalLagrangeAnterpolator{stencil}(T, Float64)

    # A generic offset scaled by λ, so κ·Δ is order 1 and the phase is strongly complex -- otherwise
    # transpose and adjoint agree to within tolerance and the distinction proves nothing.
    Δ = λ * SVector(0.19, -0.27, 0.11)
    k̂s, _ = MLFMA.cartesiansamplepoints(parentsampling)
    phase = MLFMA.phasefactor(γ, k̂s, Δ, MLFMA.AbstractRadiationFieldCollection)

    # non-vacuity of the fixture itself
    @test maximum(abs, imag.(phase)) > 0.1
    @test size(childsampling) != size(parentsampling)

    trait = MLFMA.SphericalθφInterpolation()
    transfer = MLFMA.InterpolationPhaseTransfer(
        interpolator, anterpolator, phase, childsampling, parentsampling, trait
    )

    # Dense reference, built independently of the transfer object.
    intermatrix = MLFMA._getinterpolationmatrix(
        interpolator, childsampling, parentsampling, T
    )
    M = Diagonal(vec(phase)) * intermatrix

    @test size(M) == (length(parentsampling), length(childsampling))
    # the reference must genuinely distinguish the directions
    @test !(transpose(M) ≈ adjoint(M))

    # `applydirectedtransfer!` stages through `context.buffer`, which must be a moment on the parent
    # grid. Only the θ component is compared: the trait flips the θ sign for φ, so one component is
    # enough to pin the algebra without restating the trait's convention here.
    parentscratch = MLFMA.TangentialSphericalFarField(T, parentsampling)
    context = H2Factory.TranslationContext(nothing, parentscratch)

    @testset "forward" begin
        for i in 1:length(childsampling)
            u = zeros(T, size(childsampling))
            u[i] = 1.0 + 0.5im
            in = MLFMA.TangentialSphericalFarField(SVector(u, copy(u)))
            out = MLFMA.TangentialSphericalFarField(T, parentsampling)

            H2Factory.applydirectedtransfer!(out, in, transfer, false, context)
            @test vec(out[1]) ≈ M * vec(u)

            i >= 3 && break # a few columns is enough; each is a full interpolation pass
        end
    end

    @testset "transpose" begin
        for i in 1:length(parentsampling)
            u = zeros(T, size(parentsampling))
            u[i] = 1.0 + 0.5im
            in = MLFMA.TangentialSphericalFarField(SVector(u, copy(u)))
            out = MLFMA.TangentialSphericalFarField(T, childsampling)

            H2Factory.applydirectedtransfer!(out, in, transpose(transfer), false, context)
            @test vec(out[1]) ≈ transpose(M) * vec(u)
            # and it is NOT the adjoint -- the phase must stay unconjugated here
            @test !(vec(out[1]) ≈ adjoint(M) * vec(u))

            i >= 3 && break
        end
    end

    @testset "adjoint" begin
        for i in 1:length(parentsampling)
            u = zeros(T, size(parentsampling))
            u[i] = 1.0 + 0.5im
            in = MLFMA.TangentialSphericalFarField(SVector(u, copy(u)))
            out = MLFMA.TangentialSphericalFarField(T, childsampling)

            H2Factory.applydirectedtransfer!(out, in, adjoint(transfer), false, context)
            @test vec(out[1]) ≈ adjoint(M) * vec(u)
            @test !(vec(out[1]) ≈ transpose(M) * vec(u))

            i >= 3 && break
        end
    end

    @testset "direction wrappers are involutions" begin
        @test transpose(transpose(transfer)) === transfer
        @test adjoint(adjoint(transfer)) === transfer
    end

    # Accumulate must add T*in onto whatever `out` already holds, rather than overwrite it. The
    # reverse directions also need the CHILD-grid staging moment, which only the o2o channel's
    # `LevelTransferScratch` carries, so this exercises both buffer shapes the apply methods accept.
    # `interim = nothing` on purpose: an interpolator built from the empty-dict constructor has no
    # sampling pair registered, so this pins the allocate-per-call fallback path.
    @testset "accumulate adds instead of overwriting" begin
        tuplecontext = H2Factory.TranslationContext(
            nothing,
            MLFMA.LevelTransferScratch(
                MLFMA.TangentialSphericalFarField(T, parentsampling),
                MLFMA.TangentialSphericalFarField(T, childsampling),
                nothing,
            ),
        )

        u = zeros(T, size(childsampling))
        u[2] = 1.0 + 0.5im
        in = MLFMA.TangentialSphericalFarField(SVector(u, copy(u)))

        seed = fill(T(0.25 - 0.75im), size(parentsampling))
        out = MLFMA.TangentialSphericalFarField(SVector(copy(seed), copy(seed)))
        H2Factory.applydirectedtransfer!(out, in, transfer, true, context)
        @test vec(out[1]) ≈ vec(seed) + M * vec(u)
        # non-vacuity: the seed must actually be visible in the result
        @test !(vec(out[1]) ≈ M * vec(u))

        up = zeros(T, size(parentsampling))
        up[2] = 1.0 + 0.5im
        inp = MLFMA.TangentialSphericalFarField(SVector(up, copy(up)))

        seedchild = fill(T(-0.4 + 0.6im), size(childsampling))
        outt = MLFMA.TangentialSphericalFarField(SVector(copy(seedchild), copy(seedchild)))
        H2Factory.applydirectedtransfer!(outt, inp, transpose(transfer), true, tuplecontext)
        @test vec(outt[1]) ≈ vec(seedchild) + transpose(M) * vec(up)

        outa = MLFMA.TangentialSphericalFarField(SVector(copy(seedchild), copy(seedchild)))
        H2Factory.applydirectedtransfer!(outa, inp, adjoint(transfer), true, tuplecontext)
        @test vec(outa[1]) ≈ vec(seedchild) + adjoint(M) * vec(up)
        @test !(vec(outa[1]) ≈ vec(outt[1])) # transpose and adjoint stay distinct here too

        # the reverse directions must not have written through to their input
        @test vec(inp[1]) ≈ vec(up)
    end
end
