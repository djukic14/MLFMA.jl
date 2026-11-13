using MLFMA
using BEAST, CompScienceMeshes
using StaticArrays
using ClusterTrees
using Test
using LinearAlgebra
using PlotlyJS
using IterativeSolvers
using H2Factory
using H2FactoryTestUtils
using H2Trees

include("../support/util_test.jl")

@testset "MLFMA matrix tworects" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    ϵ = 1e-4

    m = CompScienceMeshes.read_gmsh_mesh(
        joinpath(pkgdir(MLFMA), "test", "assets", "tworects50MHz.msh")
    )
    # m = meshsphere(4 * λ, λ / 10)

    X = raviartthomas(m)

    efie = MLFMA.efie(; wavenumber=κ)
    # efie = Maxwell3D.singlelayer(; wavenumber=κ, alpha=1.0, beta=2.0)
    # efie = MLFMA.MWHyperSingular3D(; wavenumber=κ)

    # Build the tree explicitly with `minvalues=0` so leaf size is driven by `minhalfsize`, which
    # is what MLFMA's accuracy model assumes. `TwoNTreeBuilder`'s default `minvalues=70` stops
    # subdivision after two levels on a mesh this size no matter how small `minhalfsize` gets, and
    # so can never satisfy `GalerkinMLFMA`'s own `numberoflevels(tree) > 2` assertion.
    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=MLFMA.λ(efie) * MLFMA.MINHALFSIZERECOM, minvalues=0
        ),
    )

    A_MLFMA = MLFMA.assemble(
        efie,
        X;
        tree=tree,
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
            leveltransferstrategy=MLFMA.LevelTransferStrategy(
                efie; Interpolator=MLFMA.SphericalLagrangeInterpolator{8,ComplexF64}
            ),
            nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(
                MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
            ),
        ),
        verbose=true,
    )

    @test size(A_MLFMA) == (numfunctions(X), numfunctions(X))
    @test size(A_MLFMA) == size(MLFMA.nearinteractions(A_MLFMA))
    @test size(A_MLFMA, 1) == numfunctions(X)
    @test size(A_MLFMA, 2) == numfunctions(X)

    x = randn(ComplexF64, numfunctions(X))

    A_MLFMA * ones(numfunctions(X))

    A = assemble(efie, X, X; quadstrat=MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM)

    checkinteractions(A, A_MLFMA; n=numfunctions(X), ϵ=ϵ * 10, verbose=true)

    # TRANSPOSE across all `numfunctions(X)` basis vectors, not a single random vector: this file
    # found one draw measuring ~6e-4 -- indistinguishable from "passing" `ϵ*10 = 1e-3` -- on a
    # direction whose full sweep then proved real per-column errors.
    @test checkinteractions(
        transpose(A), transpose(A_MLFMA); n=numfunctions(X), ϵ=ϵ * 10, verbose=false
    )

    # ADJOINT, over the same full sweep, and this fixture is the second of the two whose adjoint
    # aggregation CROSSES LEVELS -- see `galerkin.jl`'s equivalent for the measurements. It was a
    # `@test_skip` through two separate fixes: the conjugated leaf multiply
    # (`src/assembly/cores/adjointleafmul.jl`) for `α` and the leaf collections, then
    # `H2Factory.ConjugateTranslation` + `MLFMA.ConjugatedInterpolator` for the aggregation level
    # transfer. This one runs the LAGRANGE family, whose interpolation operator is real, so
    # `conjugatedinterpolator` hands the interpolator straight back and only the PHASE is conjugated
    # here -- a different code path from `galerkin.jl`'s global-φ default, which is why both are
    # asserted rather than one standing in for the other.
    @test checkinteractions(
        adjoint(A), adjoint(A_MLFMA); n=numfunctions(X), ϵ=ϵ * 10, verbose=false
    )

    # dense-only, so unaffected by the gap below
    σsquared, x = IterativeSolvers.powm(A * A'; verbose=false)

    # `estimateerror` and `estimateerrorpowermethod` used to provide the accuracy estimate here.
    # Neither exists in H2FactoryTestUtils any more, so those calls are removed rather than guarded --
    # they were failing with `UndefVarError`, not because of the write-path gap. Re-establishing an
    # accuracy estimate for this map belongs to Phase 5, which builds one against a dense reference.
    #
    # What remains is the relative-error estimate via the power method. `(A_MLFMA - A)'` adjoints the
    # map, so this was a `@test_skip` for as long as the adjoint was wrong -- a power-method figure
    # computed through a wrong adjoint says nothing about accuracy however small it comes out (one
    # draw measured 5.5e-7 while the direction was genuinely broken). It is a real assertion now that
    # the adjoint is verified column by column above.
    @test begin
        σerrorsquared, _ = IterativeSolvers.powm(
            (A_MLFMA - A) * (A_MLFMA - A)'; verbose=false, maxiter=50, tol=ϵ / 10
        )
        norm(sqrt(σerrorsquared) / sqrt(σsquared)) < ϵ
    end

    A_MLFMA = MLFMA.assemble(
        efie,
        X;
        tree=tree, # as above: the default builder cannot reach three levels on this mesh
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
            leveltransferstrategy=MLFMA.LevelTransferStrategy(
                efie; Interpolator=MLFMA.LocalθGlobalφInterpolator{8,ComplexF64}
            ),
            nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(
                MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
            ),
            nearfieldstrategy=H2Factory.InteractionStrategy(;
                isnear=H2Trees.isnear(; additionalbufferboxes=2)
            ),
        ),
        verbose=true,
    )

    A_MLFMA * ones(numfunctions(X))

    # checkinteractions(A, A_MLFMA; n=numfunctions(X), ϵ=ϵ * 10, verbose=false)
    # checkinteractions(
    #     transpose(A), transpose(A_MLFMA); n=numfunctions(X), ϵ=ϵ * 10, verbose=false
    # )
    # checkinteractions(
    #     adjoint(A), adjoint(A_MLFMA); n=numfunctions(X), ϵ=ϵ * 10, verbose=false
    # )

    @test size(A_MLFMA) == (numfunctions(X), numfunctions(X))
    @test size(A_MLFMA) == size(MLFMA.nearinteractions(A_MLFMA))
    @test size(A_MLFMA, 1) == numfunctions(X)
    @test size(A_MLFMA, 2) == numfunctions(X)

    @time begin
        A_MLFMA = MLFMA.assemble(
            efie,
            X;
            tree=tree, # as above: the default builder cannot reach three levels on this mesh
            strategy=MLFMA.MLFMAStrategy(
                efie,
                X;
                samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
                leveltransferstrategy=MLFMA.LevelTransferStrategy(
                    efie; Interpolator=MLFMA.SphericalLagrangeInterpolator{15,ComplexF64}
                ),
                nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(
                    MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
                ),
            ),
        )
        result, ch = IterativeSolvers.gmres(
            A_MLFMA, ones(numfunctions(X)); log=true, reltol=ϵ
        )
    end

    resultBEAST, chBEAST = IterativeSolvers.gmres(
        A, ones(numfunctions(X)); log=true, reltol=ϵ
    )

    @test result ≈ resultBEAST rtol = ϵ

    for k in keys(H2Factory.trialmomentcollections(A_MLFMA))
        @test H2Factory.trialmomentcollections(A_MLFMA)[k][1] ==
            H2Factory.testmomentcollections(A_MLFMA)[k].farfieldcollection[1]
        @test H2Factory.trialmomentcollections(A_MLFMA)[k][2] ==
            H2Factory.testmomentcollections(A_MLFMA)[k].farfieldcollection[2]
    end
end
