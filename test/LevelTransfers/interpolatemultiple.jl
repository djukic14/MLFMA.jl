using MLFMA
using BEAST, CompScienceMeshes
using StaticArrays
using ClusterTrees
using Test
using LinearAlgebra
using PlotlyJS
using H2Factory
using H2Trees

include("../support/util_test.jl")

# `MLFMA.applyphasefactor!` was removed once leaf far-field assembly started computing
# box-centered moments directly (via `MLFMA.leaffarfieldsamplings`/`BoxFarFieldSampling`, see
# `beastfarfieldassembler.jl`) instead of correcting a globally-assembled collection afterward.
# This test builds one flat, non-tree-chunked `RadiationFieldCollection` over the whole space (to
# exercise the interpolator against a single shared sampling grid) and still needs the old
# per-leaf-column-block phase correction applied to it -- reproduced locally here since it's no
# longer part of the production API.
function applyboxphase!(farfieldcollection, γ, samplingpoints, tree)
    phase = Array{typeof(γ)}(undef, length(samplingpoints))
    for node in H2Trees.leaves(tree)
        MLFMA.phasefactor!(
            phase,
            γ,
            samplingpoints,
            H2Trees.center(tree, node),
            MLFMA.AbstractRadiationFieldCollection,
        )
        ids = H2Trees.values(tree, node)
        for i in eachindex(farfieldcollection)
            farfieldcollection[i][:, ids] .*= phase
        end
    end
    return farfieldcollection
end

@testset "Interpolate Multiple Far Fields" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    ϵ = 1e-7

    m = CompScienceMeshes.read_gmsh_mesh(
        joinpath(pkgdir(MLFMA), "test", "assets", "tworects50MHz.msh")
    )

    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    doublelayer = Maxwell3D.doublelayer(; wavenumber=κ)
    rotateddoublelayer = BEAST.n × doublelayer

    for X in [raviartthomas(m), buffachristiansen(m)]
        for operator in [efie, doublelayer, rotateddoublelayer]
            points = BEAST.positions(X)

            tree = TwoNTree(
                X;
                builder=TwoNTreeBuilder(;
                    minhalfsize=λ / 10, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
                ),
            )

            x = ones(numfunctions(X))

            levels = H2Trees.levels(tree)[3:(end)]

            level = levels[end]
            sampling, _ = MLFMA.sampling(
                κ,
                ϵ,
                tree;
                samplingbuilder=MLFMA.SphericalSamplingBuilder{Float64}(),
                level=level,
            )
            farfieldcollection = MLFMA.buildfieldcollection(
                radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(1)),
                operator,
                sampling,
                X;
                verbose=true,
            )

            applyboxphase!(
                farfieldcollection, im * κ, MLFMA.cartesiansamplepoints(sampling)[1], tree
            )

            l = numfunctions(X)
            farfields = Vector{Any}(undef, l)
            for i in 1:l
                u = zeros(numfunctions(X))
                u[i] = 1
                # farfields[i] = farfieldcollection * u
                farfield = MLFMA.TangentialSphericalFarField(im * κ, sampling)
                LinearAlgebra.mul!(farfield, farfieldcollection, u)
                farfields[i] = farfield
            end

            farfields = convert(Vector{typeof(farfields[1])}, farfields)

            localglobalinterpolator = MLFMA.LocalθGlobalφInterpolator{15}(
                ComplexF64, Float64
            )
            locallocalinterpolator = MLFMA.SphericalLagrangeInterpolator{15}(
                ComplexF64, Float64
            )

            interpolatorsampling = MLFMA.SphericalSampling{Float64}(
                MLFMA.Nθ(sampling) * 2, MLFMA.Nφ(sampling) * 2
            )
            interpolatedfarfields = similar(farfields)
            interpolatedfarfields2 = similar(farfields)

            for i in eachindex(farfields)
                interpolatedfarfields[i] = localglobalinterpolator(
                    farfields[i], sampling, interpolatorsampling
                )
                interpolatedfarfields2[i] = locallocalinterpolator(
                    farfields[i], sampling, interpolatorsampling
                )
            end
            # interpolatedfarfields = localglobalinterpolator(
            # farfields, sampling, interpolatorsampling
            # )
            # interpolatedfarfields2 = locallocalinterpolator(
            #     farfields, sampling, interpolatorsampling
            # )

            farfieldcollection2 = MLFMA.buildfieldcollection(
                radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(1)),
                operator,
                interpolatorsampling,
                X;
                verbose=true,
            )
            applyboxphase!(
                farfieldcollection2,
                im * κ,
                MLFMA.cartesiansamplepoints(interpolatorsampling)[1],
                tree,
            )

            for i in 1:l
                u = zeros(numfunctions(X))
                u[i] = 1
                farfield = MLFMA.TangentialSphericalFarField(im * κ, interpolatorsampling)
                LinearAlgebra.mul!(farfield, farfieldcollection2, u)
                @test MLFMA.Fθ(interpolatedfarfields[i]) ≈ MLFMA.Fθ(farfield) rtol = ϵ
                @test MLFMA.Fφ(interpolatedfarfields[i]) ≈ MLFMA.Fφ(farfield) rtol = ϵ

                @test MLFMA.Fθ(interpolatedfarfields2[i]) ≈ MLFMA.Fθ(farfield) rtol = ϵ
                @test MLFMA.Fφ(interpolatedfarfields2[i]) ≈ MLFMA.Fφ(farfield) rtol = ϵ
            end
        end
    end
end
