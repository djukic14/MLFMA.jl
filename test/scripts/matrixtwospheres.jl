using MLFMA
using BEAST, CompScienceMeshes
using StaticArrays
using ClusterTrees
using Test
using LinearAlgebra
using PlotlyJS
using H2FactoryTestUtils
using H2Trees

include("../support/util_test.jl")

ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
ϵ = 1e-5

m = meshtwospheres([0.0, 0, 0], 0.2 * λ, [10 * λ, 0, 0], 0.2 * λ, λ / 10)
X = raviartthomas(m)

efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

doublelayer = Maxwell3D.doublelayer(; alpha=1.0, wavenumber=κ)
doublelayer = BEAST.MWDoubleLayer3D(2.0, κ * im)
additionalbufferboxes = 0
for operator in [doublelayer, efie]
    @show operator

    # `isnear` here used to read `H2Trees(; ...)`, which is not a call to anything -- `scripts/` is
    # excluded from the suite, so it never ran. Fixed to `H2Trees.isnear` while migrating.
    A_MLFMA = MLFMA.GalerkinMLFMA(
        operator,
        X;
        strategy=MLFMA.MLFMAStrategy(
            operator,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ),
            nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
            ),
            nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
                isnear=H2Trees.isnear(; additionalbufferboxes=additionalbufferboxes)
            ),
        ),
    )

    A = assemble(
        operator,
        X,
        X;
        quadstrat=BEAST.DoubleNumWiltonSauterQStrat(10, 11, 6, 7, 5, 5, 4, 3),
    )

    y1 = ComplexF64.(1:size(A, 1))
    y2 = deepcopy(y1)

    LinearAlgebra.mul!(y1, A, ones(numfunctions(X)), 1.0, 1.0)
    LinearAlgebra.mul!(y2, A_MLFMA, ones(numfunctions(X)), 1.0, 1.0)

    @test y1 ≈ y2 rtol = ϵ

    checkinteractions(A, A_MLFMA, ϵ)
end
