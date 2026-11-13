using MLFMA
using BEAST, CompScienceMeshes
using StaticArrays
using ClusterTrees
using Test
using LinearAlgebra
using PlotlyJS
using SphericalScattering
using H2Factory
using H2FactoryTestUtils
using OhMyThreads
using H2Trees

include("../support/util_test.jl")

# EVERY `DynamicScheduler` HERE STATES `ntasks` EXPLICITLY, and that is not decoration. A bare
# `DynamicScheduler()` takes its chunk count from `Threads.nthreads()`, which is ONE unless the suite
# is run with `-t2` or more -- so on a default runner these testsets, among the largest in the suite,
# spent the whole refactor asserting that serial agrees with serial.
#
# Chunking is by TASK COUNT, not by thread count, so `ntasks=4` exercises chunk ids 1..4 even on one
# thread. That is the half that matters most here: the real bug this guards against was a chunk-INDEX
# error -- level-transfer scratch sized by one level's chunk count and indexed with the PARENT level's
# chunkid -- which is invisible at one chunk and reproduces at four however many threads actually run.
# CI additionally runs the whole suite once at `JULIA_NUM_THREADS=4` for the genuinely concurrent
# half; see `.github/workflows/Runtests.yml`.

@testset "Aggregation Maxwell" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    ϵ = 1e-3

    m = meshsphere(1 * λ, λ / 10)

    # m = CompScienceMeshes.read_gmsh_mesh(
    #     joinpath(pkgdir(MLFMA), "test", "assets", "tworects50MHz.msh")
    # )
    X = raviartthomas(m)

    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    doublelayer = Maxwell3D.doublelayer(; wavenumber=κ)
    singlelayer = Maxwell3D.singlelayer(; wavenumber=κ)
    hypersingular = MLFMA.MWHyperSingular3D(; wavenumber=κ)

    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=λ / 10,
            minlevel=2,
            minvalues=0,
            protrusion=H2Trees.NoProtrusionCheck(),
        ),
    )

    for operator in [hypersingular, singlelayer, efie, doublelayer]
        levels = H2Trees.levels(tree)

        farfieldcollections = Vector{MLFMA.AbstractRadiationFieldCollection}(
            undef, length(levels)
        )

        i = 1
        for level in levels
            sampling, _ = MLFMA.sampling(
                κ,
                ϵ,
                tree;
                samplingbuilder=MLFMA.SphericalSamplingBuilder{Float64}(),
                level=level,
            )
            farfieldcollections[i] = MLFMA.buildfieldcollection(
                radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(2)),
                operator,
                sampling,
                X;
                verbose=true,
            )

            i += 1
        end
        for PhaseShifter in [MLFMA.PhaseShifter]
            A_MLFMA = MLFMA.GalerkinMLFMA(
                operator,
                X;
                tree=tree,
                strategy=stubbedstrategy(
                    operator,
                    X;
                    samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                    farfieldstrategy=MLFMA.FarFieldStrategy(;
                        assemblerbuilder=MLFMA.BEASTFarFieldAssemblerBuilder(),
                        quadstrategy=BEAST.SingleNumQStrat(2),
                    ),
                    leveltransferstrategy=MLFMA.LevelTransferStrategy(
                        operator;
                        Interpolator=MLFMA.SphericalLagrangeInterpolator{15,ComplexF64},
                        PhaseShifter=PhaseShifter,
                    ),
                    scheduler=OhMyThreads.DynamicScheduler(; ntasks=4),
                ),
                verbose=true,
            )

            A_MLFMA2 = MLFMA.GalerkinMLFMA(
                operator,
                X;
                tree=tree,
                strategy=stubbedstrategy(
                    operator,
                    X;
                    samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                    farfieldstrategy=MLFMA.FarFieldStrategy(;
                        quadstrategy=BEAST.SingleNumQStrat(2)
                    ),
                    leveltransferstrategy=MLFMA.LevelTransferStrategy(
                        operator;
                        Interpolator=MLFMA.LocalθGlobalφInterpolator{15,ComplexF64},
                        PhaseShifter=PhaseShifter,
                    ),
                    nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
                        aggregatenode=H2Trees.aggregateallnodes()
                    ),
                    scheduler=OhMyThreads.DynamicScheduler(; ntasks=4),
                ),
            )

            A_MLFMA3 = MLFMA.GalerkinMLFMA(
                operator,
                X;
                tree=tree,
                strategy=stubbedstrategy(
                    operator,
                    X;
                    samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                    farfieldstrategy=MLFMA.FarFieldStrategy(;
                        quadstrategy=BEAST.SingleNumQStrat(2)
                    ),
                    leveltransferstrategy=MLFMA.LevelTransferStrategy(
                        operator;
                        Interpolator=MLFMA.SphericalLagrangeInterpolator{15,ComplexF64},
                        PhaseShifter=PhaseShifter,
                    ),
                    nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
                        aggregatenode=H2Trees.aggregateallnodes()
                    ),
                    scheduler=OhMyThreads.SerialScheduler(),
                ),
                verbose=true,
            )

            A_MLFMA4 = MLFMA.GalerkinMLFMA(
                operator,
                X;
                tree=tree,
                strategy=stubbedstrategy(
                    operator,
                    X;
                    samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                    farfieldstrategy=MLFMA.FarFieldStrategy(;
                        quadstrategy=BEAST.SingleNumQStrat(2)
                    ),
                    leveltransferstrategy=MLFMA.LevelTransferStrategy(
                        operator;
                        Interpolator=MLFMA.SphericalLagrangeInterpolator{15,ComplexF64},
                        PhaseShifter=PhaseShifter,
                    ),
                    nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
                        aggregatenode=H2Trees.aggregaterootonly()
                    ),
                    scheduler=OhMyThreads.SerialScheduler(),
                ),
            )

            x = ones(numfunctions(X))

            H2Factory.aggregate!(A_MLFMA, x)
            H2Factory.aggregate!(A_MLFMA2, x)
            H2Factory.aggregate!(A_MLFMA3, x)
            H2Factory.aggregate!(A_MLFMA4, x)

            for (i, level) in enumerate(levels)
                sampling, _ = MLFMA.sampling(
                    κ,
                    ϵ,
                    tree;
                    samplingbuilder=MLFMA.SphericalSamplingBuilder{Float64}(),
                    level=level,
                )

                for node in H2Trees.LevelIterator(tree, level)
                    xlocal = zeros(ComplexF64, numfunctions(X))
                    xlocal[H2Trees.values(tree, node)] .= 1.0

                    phase = MLFMA.phasefactor(
                        im * κ,
                        MLFMA.cartesiansamplepoints(sampling)[1],
                        H2Trees.center(tree, node),
                        MLFMA.AbstractRadiationFieldCollection,
                    )
                    testfarfield = MLFMA.farfieldtype(operator)(im * κ, sampling)
                    LinearAlgebra.mul!(testfarfield, farfieldcollections[i], xlocal)
                    elementwisemul!(testfarfield, testfarfield, phase)

                    for i in eachindex(testfarfield)
                        @test H2Factory.trialmoments(A_MLFMA2)[node][i] ≈ testfarfield[i] rtol =
                            ϵ
                        @test H2Factory.trialmoments(A_MLFMA3)[node][i] ≈ testfarfield[i] rtol =
                            ϵ
                        @test H2Factory.trialmoments(A_MLFMA2)[node][i] ≈
                            H2Factory.trialmoments(A_MLFMA3)[node][i] rtol = ϵ
                    end

                    if H2Trees.storenode(H2Factory.trialaggregationplan(A_MLFMA), node)
                        for i in eachindex(testfarfield)
                            @test H2Factory.trialmoments(A_MLFMA)[node][i] ≈ testfarfield[i] rtol =
                                ϵ
                        end

                        @test haskey(H2Factory.trialmoments(A_MLFMA), node)
                    end

                    if H2Trees.storenode(H2Factory.trialaggregationplan(A_MLFMA4), node)
                        @test node == 1
                        for i in eachindex(testfarfield)
                            @test H2Factory.trialmoments(A_MLFMA4)[node][i] ≈
                                testfarfield[i] rtol = ϵ
                        end
                        @test haskey(H2Factory.trialmoments(A_MLFMA4), node)
                    end
                end
            end
        end
    end
end

@testset "Aggregation far field" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    ϵ = 1e-3

    m = meshsphere(0.5 * λ, λ / 10)

    # m = CompScienceMeshes.read_gmsh_mesh(
    #     joinpath(pkgdir(MLFMA), "test", "assets", "tworects50MHz.msh")
    # )
    X = buffachristiansen(m)

    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    operator = efie

    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=λ / 10,
            minlevel=2,
            minvalues=0,
            protrusion=H2Trees.NoProtrusionCheck(),
        ),
    )

    A_MLFMA = MLFMA.GalerkinMLFMA(
        operator,
        X;
        tree=tree,
        strategy=stubbedstrategy(
            operator,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
            farfieldstrategy=MLFMA.FarFieldStrategy(;
                quadstrategy=BEAST.SingleNumQStrat(1)
            ),
            nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
                aggregatenode=H2Trees.aggregateallnodes()
            ),
            scheduler=OhMyThreads.DynamicScheduler(; ntasks=4),
        ),
        verbose=true,
    )

    x = ones(numfunctions(X))

    sampling = MLFMA.samplings(A_MLFMA)[1]
    farfieldsamplepoints, _ = MLFMA.cartesiansamplepoints(sampling)
    farfieldscartesian = BEAST.potential(
        BEAST.MWFarField3D(; wavenumber=κ),
        farfieldsamplepoints,
        x,
        X;
        quadstrat=BEAST.SingleNumQStrat(1),
    )

    Fθ = Vector{ComplexF64}(undef, length(farfieldsamplepoints))
    Fφ = Vector{ComplexF64}(undef, length(farfieldsamplepoints))

    farfieldspherical = SphericalScattering.convertCartesian2Spherical.(
        farfieldscartesian, SphericalScattering.cart2sph.(farfieldsamplepoints)
    )

    farfieldθφ = MLFMA._removeradialcomponent(farfieldspherical)

    phasefactor = MLFMA.phasefactor(
        im * κ,
        farfieldsamplepoints,
        H2Trees.center(tree, 1),
        MLFMA.AbstractRadiationFieldCollection,
    )

    Fθ = farfieldθφ[1] .* phasefactor
    Fφ = farfieldθφ[2] .* phasefactor

    H2Factory.aggregate!(A_MLFMA, x)

    errorθ = norm.(Fθ - MLFMA.Fθ(H2Factory.trialmoments(A_MLFMA)[1])) / maximum(norm.(Fθ))
    errorφ = norm.(Fφ - MLFMA.Fφ(H2Factory.trialmoments(A_MLFMA)[1])) / maximum(norm.(Fφ))

    @test H2Factory.trialmoments(A_MLFMA)[1][1] ≈ Fθ rtol = ϵ
    @test H2Factory.trialmoments(A_MLFMA)[1][2] ≈ Fφ rtol = ϵ
end

@testset "Aggregation PetrovGalerkinMLFMA" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    # we are testing the result of the aggregation of the MLFMAMatrix2 with == with
    # MLFMAMatrix as the gold truth. The accuracy ϵ does not matter for this test

    ϵ = 1e-1

    # The geometies for trial and testing tree should coincide, overlap and be completely
    # disjoint with a lot of distance between them. Also different sizes (different number
    # of levels in the tree) should be tested.
    mx = meshspherewithcenter([0.0, 0, 0], λ, λ / 5)
    my = meshspherewithcenter([2 * λ, 0, 0], 2 * λ, λ / 5)
    mz = meshspherewithcenter([10 * λ, 0, 0], λ / 2, λ / 5)

    # these functions are all symmetric with regard to trial and testtree thus we can use
    # them for both
    functionarray = [
        (tree) ->
            (testtree, trialtree, trialnode) ->
                H2Trees.istranslatingnode(testtree, trialtree, trialnode),
        H2Trees.aggregateallnodes(),
        H2Trees.aggregaterootonly(),
    ]
    operator = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

    for m in [mx, my, mz]
        X = raviartthomas(m)
        x = ones(numfunctions(X))

        for m2 in [mx, my, mz]
            Y = raviartthomas(m2)

            treetrialtest = H2Trees.BlockTree(
                Y,
                X;
                builder=H2Trees.BlockTreeBuilder(;
                    test=TwoNTreeBuilder(;
                        minhalfsize=λ / 5,
                        minvalues=0,
                        protrusion=H2Trees.NoProtrusionCheck(),
                    ),
                    trial=TwoNTreeBuilder(;
                        minhalfsize=λ / 5,
                        minvalues=0,
                        protrusion=H2Trees.NoProtrusionCheck(),
                    ),
                ),
            )

            display(treetrialtest)
            tree = H2Trees.trialtree(treetrialtest)
            tree2 = H2Trees.testtree(treetrialtest)

            levels = H2Trees.levels(tree)

            for PhaseShifter in [MLFMA.PhaseShifter]
                A_MLFMA = MLFMA.GalerkinMLFMA(
                    operator,
                    X;
                    tree=tree,
                    strategy=stubbedstrategy(
                        operator,
                        X;
                        samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                        farfieldstrategy=MLFMA.FarFieldStrategy(;
                            assemblerbuilder=MLFMA.DummyFarFieldAssemblerBuilder()
                        ),
                        leveltransferstrategy=MLFMA.LevelTransferStrategy(
                            operator;
                            Interpolator=MLFMA.SphericalLagrangeInterpolator{3,ComplexF64},
                            PhaseShifter=PhaseShifter,
                        ),
                        nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
                            aggregatenode=H2Trees.aggregateallnodes()
                        ),
                        scheduler=OhMyThreads.DynamicScheduler(; ntasks=4),
                    ),
                    verbose=true,
                )

                H2Factory.aggregate!(A_MLFMA, x)

                for funtrialnode in functionarray
                    A_MLFMA2 = MLFMA.PetrovGalerkinMLFMA(
                        operator,
                        Y,
                        X;
                        tree=treetrialtest,
                        strategy=stubbedstrategy(
                            operator,
                            Y,
                            X;
                            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                            farfieldstrategy=MLFMA.FarFieldStrategy(;
                                assemblerbuilder=MLFMA.DummyFarFieldAssemblerBuilder()
                            ),
                            leveltransferstrategy=MLFMA.LevelTransferStrategy(
                                operator;
                                Interpolator=MLFMA.SphericalLagrangeInterpolator{
                                    3,ComplexF64
                                },
                                PhaseShifter=PhaseShifter,
                            ),
                            nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
                                aggregatenode=funtrialnode
                            ),
                        ),
                    )

                    H2Factory.aggregate!(A_MLFMA2, x)

                    for (i, level) in enumerate(levels)
                        for node in H2Trees.LevelIterator(tree, level)
                            if !H2Trees.storenode(
                                H2Factory.trialaggregationplan(A_MLFMA2), node
                            )
                                continue
                            end

                            @test MLFMA.Fθ(H2Factory.trialmoments(A_MLFMA)[node]) ==
                                MLFMA.Fθ(H2Factory.trialmoments(A_MLFMA2)[node])
                            @test MLFMA.Fφ(H2Factory.trialmoments(A_MLFMA)[node]) ==
                                MLFMA.Fφ(H2Factory.trialmoments(A_MLFMA2)[node])
                        end
                    end
                end
            end
        end
    end
end

# @testset "Aggregation Transpose MLFMAMatrix2" begin
#     ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
#     # we are testing the result of the aggregation of the MLFMAMatrix2 with == with
#     # MLFMAMatrix as the gold truth. The accuracy ϵ does not matter for this test

#     ϵ = 1e-1

#     # The geometies for trial and testing tree should coincide, overlap and be completely
#     # disjoint with a lot of distance between them. Also different sizes (different number
#     # of levels in the tree) should be tested.
#     mx = meshspherewithcenter([0.0, 0, 0], λ, λ / 5)
#     my = meshspherewithcenter([2 * λ, 0, 0], 2 * λ, λ / 5)
#     mz = meshspherewithcenter([10 * λ, 0, 0], λ / 2, λ / 5)

#     # these functions are all symmetric with regard to trial and testtree thus we can use
#     # them for both
#     # TODO: fix this to work with isnear
#     functionarray = [
#         (tree) ->
#             (testtree, trialtree, trialnode) ->
#                 H2Trees.istranslatingnode(testtree, trialtree, trialnode),
#         H2Trees.aggregateallnodes(),
#         H2Trees.aggregaterootonly(),
#     ]
#     efie = MLFMA.efie(; wavenumber=κ, alpha=-im * ω * μ)
#     doublelayer = Maxwell3D.doublelayer(; wavenumber=κ)

#     for operator in [doublelayer, efie]
#         for m2 in [mx, my, mz]
#             Y = raviartthomas(m2)
#             y = ones(numfunctions(Y))

#             for m in [mx, my, mz]
#                 X = raviartthomas(m)

#                 treetrialtest = TwoNTree(Y, X, λ / 5)

#                 display(treetrialtest)
#                 tree2 = H2Trees.testtree(treetrialtest)

#                 levels = H2Trees.levels(tree2)

#                 for PhaseShifter in [MLFMA.PhaseShifter]
#                     A_MLFMA = MLFMA.GalerkinMLFMA(
#                         ϵ,
#                         tree2,
#                         Y,
#                         operator;
#                         verbose=true,
#                         NearInteractionsAssembler=H2Factory.DummyNearInteractionsAssembler,
#                         TranslationCollection=MLFMA.DummyTranslationCollection,
#                         Interpolator=MLFMA.SphericalLagrangeInterpolator{3,ComplexF64},
#                         PhaseShifter=PhaseShifter,
#                         FarFieldassemblerbuilder=MLFMA.DummyFarFieldAssemblerBuilder(),
#                         aggregatenode=H2Trees.aggregateallnodes(),
#                     )

#                     H2Factory.aggregate!(A_MLFMA, y)

#                     for funtrialnode in functionarray
#                         for funtestnode in functionarray
#                             A_MLFMA2 = MLFMA.PetrovGalerkinMLFMA(
#                                 ϵ,
#                                 treetrialtest,
#                                 Y,
#                                 X,
#                                 operator;
#                                 NearInteractionsAssembler=H2Factory.DummyNearInteractionsAssembler,
#                                 TranslationCollection=MLFMA.DummyTranslationCollection,
#                                 Interpolator=MLFMA.SphericalLagrangeInterpolator{
#                                     3,ComplexF64
#                                 },
#                                 PhaseShifter=PhaseShifter,
#                                 FarFieldassemblerbuilder=MLFMA.DummyFarFieldAssemblerBuilder(),
#                                 aggregatetrialnode=funtrialnode,
#                                 aggregatetestnode=funtestnode,
#                             )

#                             H2Factory.aggregate!(transpose(A_MLFMA2), y)

#                             for (i, level) in enumerate(levels)
#                                 for node in H2Trees.LevelIterator(tree2, level)
#                                     if !H2Trees.storenode(
#                                         A_MLFMA2.testaggregationplan, node
#                                     )
#                                         continue
#                                     end

#                                     @test MLFMA.Fθ(A_MLFMA.farfields[node]) ==
#                                         conj.(MLFMA.Fθ(A_MLFMA2.testfarfields[node]))
#                                     @test MLFMA.Fφ(A_MLFMA.farfields[node]) ==
#                                         conj.(MLFMA.Fφ(A_MLFMA2.testfarfields[node]))
#                                 end
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end

@testset "Aggregation Helmholtz" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    ϵ = 1e-3

    m = meshsphere(1 * λ, λ / 10)

    X = lagrangec0d1(m)

    singlelayer = Helmholtz3D.singlelayer(; wavenumber=κ)
    doublelayer = Helmholtz3D.doublelayer(; wavenumber=κ)
    doublelayertransposed = Helmholtz3D.doublelayer_transposed(; wavenumber=κ)
    hypersingular = Helmholtz3D.hypersingular(; wavenumber=κ) #TODO: does not work

    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=λ / 10,
            minlevel=2,
            minvalues=0,
            protrusion=H2Trees.NoProtrusionCheck(),
        ),
    )

    for operator in [singlelayer, doublelayer, doublelayertransposed]
        levels = H2Trees.levels(tree)

        farfieldcollections = Vector{MLFMA.AbstractRadiationFieldCollection}(
            undef, length(levels)
        )

        i = 1
        for level in levels
            sampling, _ = MLFMA.sampling(
                κ,
                ϵ,
                tree;
                samplingbuilder=MLFMA.SphericalSamplingBuilder{Float64}(),
                level=level,
            )
            farfieldcollections[i] = MLFMA.buildfieldcollection(
                radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(2)),
                operator,
                sampling,
                X;
                verbose=true,
            )

            i += 1
        end
        for PhaseShifter in [MLFMA.PhaseShifter]
            A_MLFMA = MLFMA.GalerkinMLFMA(
                operator,
                X;
                tree=tree,
                strategy=stubbedstrategy(
                    operator,
                    X;
                    samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                    farfieldstrategy=MLFMA.FarFieldStrategy(;
                        quadstrategy=BEAST.SingleNumQStrat(2)
                    ),
                    leveltransferstrategy=MLFMA.LevelTransferStrategy(
                        operator;
                        Interpolator=MLFMA.SphericalLagrangeInterpolator{15,ComplexF64},
                        PhaseShifter=PhaseShifter,
                    ),
                    scheduler=OhMyThreads.DynamicScheduler(; ntasks=4),
                ),
                verbose=true,
            )

            A_MLFMA2 = MLFMA.GalerkinMLFMA(
                operator,
                X;
                tree=tree,
                strategy=stubbedstrategy(
                    operator,
                    X;
                    samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                    farfieldstrategy=MLFMA.FarFieldStrategy(;
                        quadstrategy=BEAST.SingleNumQStrat(2)
                    ),
                    leveltransferstrategy=MLFMA.LevelTransferStrategy(
                        operator;
                        Interpolator=MLFMA.LocalθGlobalφInterpolator{15,ComplexF64},
                        PhaseShifter=PhaseShifter,
                    ),
                    nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
                        aggregatenode=H2Trees.aggregateallnodes()
                    ),
                ),
            )

            A_MLFMA3 = MLFMA.GalerkinMLFMA(
                operator,
                X;
                tree=tree,
                strategy=stubbedstrategy(
                    operator,
                    X;
                    samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                    farfieldstrategy=MLFMA.FarFieldStrategy(;
                        quadstrategy=BEAST.SingleNumQStrat(2)
                    ),
                    leveltransferstrategy=MLFMA.LevelTransferStrategy(
                        operator;
                        Interpolator=MLFMA.SphericalLagrangeInterpolator{15,ComplexF64},
                        PhaseShifter=PhaseShifter,
                    ),
                    nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
                        aggregatenode=H2Trees.aggregateallnodes()
                    ),
                    scheduler=OhMyThreads.DynamicScheduler(; ntasks=4),
                ),
                verbose=true,
            )

            A_MLFMA4 = MLFMA.GalerkinMLFMA(
                operator,
                X;
                tree=tree,
                strategy=stubbedstrategy(
                    operator,
                    X;
                    samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                    farfieldstrategy=MLFMA.FarFieldStrategy(;
                        quadstrategy=BEAST.SingleNumQStrat(2)
                    ),
                    leveltransferstrategy=MLFMA.LevelTransferStrategy(
                        operator;
                        Interpolator=MLFMA.SphericalLagrangeInterpolator{15,ComplexF64},
                        PhaseShifter=PhaseShifter,
                    ),
                    nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
                        aggregatenode=H2Trees.aggregaterootonly()
                    ),
                ),
            )

            x = ones(numfunctions(X))

            H2Factory.aggregate!(A_MLFMA, x)
            H2Factory.aggregate!(A_MLFMA2, x)
            H2Factory.aggregate!(A_MLFMA3, x)
            H2Factory.aggregate!(A_MLFMA4, x)

            for (i, level) in enumerate(levels)
                sampling, _ = MLFMA.sampling(
                    κ,
                    ϵ,
                    tree;
                    samplingbuilder=MLFMA.SphericalSamplingBuilder{Float64}(),
                    level=level,
                )

                for node in H2Trees.LevelIterator(tree, level)
                    xlocal = zeros(ComplexF64, numfunctions(X))
                    xlocal[H2Trees.values(tree, node)] .= 1.0

                    phase = MLFMA.phasefactor(
                        im * κ,
                        MLFMA.cartesiansamplepoints(sampling)[1],
                        H2Trees.center(tree, node),
                        MLFMA.AbstractRadiationFieldCollection,
                    )

                    testfarfield = MLFMA.farfieldtype(operator)(im * κ, sampling)
                    LinearAlgebra.mul!(testfarfield, farfieldcollections[i], xlocal)
                    elementwisemul!(testfarfield, testfarfield, phase)

                    for i in eachindex(testfarfield)
                        @test H2Factory.trialmoments(A_MLFMA2)[node][i] ≈ testfarfield[i] rtol =
                            ϵ
                        @test H2Factory.trialmoments(A_MLFMA3)[node][i] ≈ testfarfield[i] rtol =
                            ϵ
                    end

                    if H2Trees.storenode(H2Factory.trialaggregationplan(A_MLFMA), node)
                        for i in eachindex(testfarfield)
                            @test H2Factory.trialmoments(A_MLFMA)[node][i] ≈ testfarfield[i] rtol =
                                ϵ
                        end

                        @test haskey(H2Factory.trialmoments(A_MLFMA), node)
                    end

                    if H2Trees.storenode(H2Factory.trialaggregationplan(A_MLFMA4), node)
                        @test node == 1

                        for i in eachindex(testfarfield)
                            @test H2Factory.trialmoments(A_MLFMA4)[node][i] ≈
                                testfarfield[i] rtol = ϵ
                        end

                        @test haskey(H2Factory.trialmoments(A_MLFMA4), node)
                    end
                end
            end
        end
    end
end

# @testset "Matrix-Vector Product Helmholtz" begin
#     ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
#     additionalbufferboxes = 1
#     # we are testing the result of the aggregation of the MLFMAMatrix2 with == with
#     # MLFMAMatrix as the gold truth. The accuracy ϵ does not matter for this test

#     ϵ = 1e-8

#     # The geometies for trial and testing tree should coincide, overlap and be completely
#     # disjoint with a lot of distance between them. Also different sizes (different number
#     # of levels in the tree) should be tested.
#     mx = meshspherewithcenter([0.0, 0, 0], λ, λ / 5)
#     my = meshspherewithcenter([2 * λ, 0, 0], 2 * λ, λ / 5)
#     mz = meshspherewithcenter([10 * λ, 0, 0], λ / 2, λ / 5)

#     # m = CompScienceMeshes.read_gmsh_mesh(
#     #     joinpath(pkgdir(MLFMA), "test", "assets", "tworects50MHz.msh")
#     # )

#     singlelayer = Helmholtz3D.singlelayer(; wavenumber=κ)
#     doublelayer = Helmholtz3D.doublelayer(; wavenumber=κ, alpha=2.0 * im)
#     doublelayertransposed = Helmholtz3D.doublelayer_transposed(;
#         wavenumber=κ, alpha=2.0 * im
#     )
#     hypersingular = Helmholtz3D.hypersingular(;
#         wavenumber=κ, alpha=2.0 + 0.0 * im, beta=1.5 * im
#     )

#     for operator in [singlelayer, doublelayer, doublelayertransposed, hypersingular]
#         for mx in [mx, my, mz]
#             X = lagrangec0d1(mx)
#             for my in [mx, my, mz]
#                 Y = lagrangec0d1(my)
#                 sampling = MLFMA.SphericalSampling{Float64}(4, 8)

#                 A_MLFMA = MLFMA.assemble(
#                     operator,
#                     X,
#                     Y;
#                     ϵ=ϵ,
#                     NearInteractionsAssembler=H2Factory.DummyNearInteractionsAssembler,
#                     additionalbufferboxes=additionalbufferboxes,
#                 )

#                 A_MLFMA * ones(numfunctions(Y))

#                 A = assemble(
#                     operator,
#                     X,
#                     Y;
#                     quadstrat=BEAST.DoubleNumWiltonSauterQStrat(7, 7, 7, 7, 1, 1, 1, 1),
#                 )

#                 farmatrix = getfarmatrix(A, A_MLFMA)

#                 u = ones(numfunctions(Y))

#                 error =
#                     maximum(norm.(farmatrix * u - A_MLFMA * u)) /
#                     maximum(norm.(farmatrix * u))

#                 println(maximum(error))
#             end
#         end
#     end
# end
