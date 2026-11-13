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
additionalbufferboxes = 1
# we are testing the result of the aggregation of the MLFMAMatrix2 with == with
# MLFMAMatrix as the gold truth. The accuracy ϵ does not matter for this test

ϵ = 1e-4

# The geometies for trial and testing tree should coincide, overlap and be completely
# disjoint with a lot of distance between them. Also different sizes (different number
# of levels in the tree) should be tested.
mx = meshspherewithcenter([0.0, 0, 0], λ, λ / 5)
my = meshspherewithcenter([2 * λ, 0, 0], 2 * λ, λ / 5)
mz = meshspherewithcenter([10 * λ, 0, 0], 2 * λ, λ / 5)

efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))

X = raviartthomas(mx)
Y = raviartthomas(mz)

A = assemble(
    efie, Y, X; quadstrat=BEAST.DoubleNumWiltonSauterQStrat(7, 7, 7, 7, 7, 7, 7, 7)
)

# `isnear` here used to read `H2Trees(; additionalbufferboxes=1)`, which is not a call to anything --
# `scripts/` is excluded from the suite, so it never ran. Fixed to `H2Trees.isnear` while migrating.
A_MLFMA = MLFMA.assemble(
    efie,
    Y,
    X;
    strategy=MLFMA.MLFMAStrategy(
        efie,
        Y,
        X;
        treestrategy=MLFMA.TreeStrategy(efie; minhalfsize=λ / 4),
        samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
        nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
            MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
        ),
        nearfieldstrategy=MLFMA.nearfieldstrategyfor(;
            isnear=H2Trees.isnear(; additionalbufferboxes=1)
        ),
    ),
    verbose=true,
)

farmatrix = getfarmatrix(A, A_MLFMA)

y = randn(numfunctions(X))
error = maximum(norm.(farmatrix * y - A_MLFMA * y)) / maximum(norm.(farmatrix * y))

x = randn(numfunctions(Y))
error = maximum(norm.(farmatrix' * x - A_MLFMA' * x)) / maximum(norm.(farmatrix' * x))
error =
    maximum(norm.(transpose(farmatrix) * x - transpose(A_MLFMA) * x)) /
    maximum(norm.(transpose(farmatrix) * x))
