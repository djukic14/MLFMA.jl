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

@testset "Farfield Interpolation" begin
    function findnode(tree, point)
        for node in H2Trees.leaves(tree)
            if any(H2Trees.values(tree, node) .== point)
                return node
            end
        end
    end

    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ
    ϵ = 1e-5

    m = CompScienceMeshes.read_gmsh_mesh(
        joinpath(pkgdir(MLFMA), "test", "assets", "twospheres50MHz.msh")
    )
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    doublelayer = Maxwell3D.doublelayer(; wavenumber=κ)
    rotateddoublelayer = BEAST.n × doublelayer

    for X in [raviartthomas(m), buffachristiansen(m)]
        tree = TwoNTree(
            X;
            builder=TwoNTreeBuilder(;
                minhalfsize=λ / 10, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
            ),
        )
        x = ones(numfunctions(X))
        levels = H2Trees.levels(tree)[(end - 1):(end)]
        farfieldcollections = Vector{MLFMA.AbstractRadiationFieldCollection}(
            undef, length(levels)
        )

        for operator in [efie, doublelayer, rotateddoublelayer]
            i = 1
            for level in levels
                i
                sampling, _ = MLFMA.sampling(
                    γ,
                    ϵ,
                    tree;
                    samplingbuilder=MLFMA.SphericalSamplingBuilder{Float64}(),
                    level=level,
                )
                farfieldcollections[i] = MLFMA.buildfieldcollection(
                    radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(1)),
                    operator,
                    sampling,
                    X,
                )
                i += 1
            end

            interpolator = MLFMA.SphericalLagrangeInterpolator{12}(ComplexF64, Float64)
            i = 2
            for level in levels
                level == levels[begin] && continue
                sampling, _ = MLFMA.sampling(
                    γ,
                    ϵ,
                    tree;
                    samplingbuilder=MLFMA.SphericalSamplingBuilder{Float64}(),
                    level=level,
                )
                parentsampling, _ = MLFMA.sampling(
                    γ,
                    ϵ,
                    tree;
                    samplingbuilder=MLFMA.SphericalSamplingBuilder{Float64}(),
                    level=level - 1,
                )

                farfield = MLFMA.TangentialSphericalFarField(γ, sampling)
                calculatedfarfield = MLFMA.TangentialSphericalFarField(γ, parentsampling)

                for j in 1:numfunctions(X)
                    x = zeros(numfunctions(X))
                    x[j] = 1.0
                    LinearAlgebra.mul!(farfield, farfieldcollections[i], x)
                    elementwisemul!(
                        farfield,
                        farfield,
                        MLFMA.phasefactor(
                            γ,
                            MLFMA.cartesiansamplepoints(sampling)[1],
                            H2Trees.center(tree, findnode(tree, j)),
                            MLFMA.AbstractRadiationFieldCollection,
                        ),
                    )

                    interpolatedfarfield = interpolator(farfield, sampling, parentsampling)

                    LinearAlgebra.mul!(calculatedfarfield, farfieldcollections[i - 1], x)
                    elementwisemul!(
                        calculatedfarfield,
                        calculatedfarfield,
                        MLFMA.phasefactor(
                            γ,
                            MLFMA.cartesiansamplepoints(parentsampling)[1],
                            H2Trees.center(tree, findnode(tree, j)),
                            MLFMA.AbstractRadiationFieldCollection,
                        ),
                    )

                    errorθ =
                        norm.(
                            MLFMA.Fθ(interpolatedfarfield)[:] -
                            MLFMA.Fθ(calculatedfarfield)[:],
                        ) ./ norm.(MLFMA.Fθ(calculatedfarfield)[:])
                    errorφ =
                        norm.(
                            MLFMA.Fφ(interpolatedfarfield)[:] -
                            MLFMA.Fφ(calculatedfarfield)[:],
                        ) ./ norm.(MLFMA.Fφ(calculatedfarfield)[:])

                    @test sum(errorθ) / length(errorθ) < ϵ
                    @test sum(errorφ) / length(errorφ) < ϵ
                end
                i += 1
            end
        end
    end
end
