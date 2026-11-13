# Forward, transpose and adjoint of a Petrov map on a GENUINELY NON-SYMMETRIC fixture.
#
# WHY A SEPARATE FIXTURE. comparison.jl checks the same three directions, but on a symmetric EFIE with
# one space used twice: there `transpose(farmatrix) == farmatrix`, so its transpose check cannot tell a
# correctly-transposed matvec from one that silently ran forward, and it says so. Here the two spaces are
# different meshes of the same sphere, so the far matrix is RECTANGULAR -- 942x1227 as measured -- and
# transposing is not a no-op under any reading. This is the first test in the suite where the transposed
# Petrov path has teeth.
#
# FIXTURE NOTES, all measured rather than chosen:
#   * Two separated small spheres were tried first and threw `ArgumentError: Empty AggregatePlan not
#     supported` -- each tree too shallow. The recipe below is galerkin.jl's measured one
#     (0.8 lambda, `minhalfsize = lambda/12`, `minvalues = 0`), which is known to give a non-empty far
#     plan, with the two meshes differing only in element size.
#   * The near block is DUMMY and the far quadrature real, as in comparison.jl, so the map's full matvec
#     IS its far matvec and is directly comparable to `farinteractions`. Comparing a full matvec against
#     a far-only reference is what a first attempt did, and it read as a forward error of 78.
using Test
using LinearAlgebra
using CompScienceMeshes
using BEAST
using H2Factory
using H2Trees
using MLFMA

@testset "Petrov directions on a non-symmetric fixture" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)

    X = raviartthomas(meshsphere(0.8 * λ, λ / 5))   # test, coarser
    Y = raviartthomas(meshsphere(0.8 * λ, λ / 6))   # trial, finer

    tree = H2Trees.BlockTree(
        X,
        Y;
        builder=H2Trees.BlockTreeBuilder(;
            test=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0),
            trial=TwoNTreeBuilder(; minhalfsize=λ / 12, minvalues=0),
        ),
    )
    A = MLFMA.assemble(
        efie,
        X,
        Y;
        tree=tree,
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X,
            Y;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
            nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
            ),
        ),
        verbose=false,
    )

    dense = BEAST.assemble(efie, X, Y; quadstrat=quadstrat)
    far = H2Factory.farinteractions(dense, A)

    # --- non-vacuity: without these the direction checks below prove nothing -----------------------
    # Rectangular, so transpose is genuinely a different operator -- the property comparison.jl lacks.
    @test size(far, 1) != size(far, 2)
    @test size(far) == (numfunctions(X), numfunctions(Y))
    # Genuinely complex, so adjoint is genuinely different from transpose. Measured 0.90.
    @test maximum(abs, imag.(far)) / maximum(abs, far) > 0.1
    @test numfunctions(X) != numfunctions(Y)

    # Deterministic, RNG-free inputs, one per side: the transposed and adjoint maps take a test-length
    # vector, the forward map a trial-length one.
    x = ComplexF64[cis(0.7 * i) * (1 + 0.5 * cospi(i / 17)) for i in 1:numfunctions(Y)]
    u = ComplexF64[cis(-0.41 * i) * (1 + 0.3 * sinpi(i / 13)) for i in 1:numfunctions(X)]
    @test maximum(abs, imag.(x)) > 0.1
    @test maximum(abs, imag.(u)) > 0.1

    rel(got, ref) = maximum(abs.(got .- ref)) / maximum(abs.(ref))

    # --- forward and transpose --------------------------------------------------------------------
    # Bounds are the measured errors with roughly an order of magnitude of headroom: forward 2.6e-4,
    # transpose 6.7e-4. Larger than comparison.jl's 4.6e-5 because the test mesh here is coarser.
    @test rel(A * x, far * x) < 5e-3
    @test rel(transpose(A) * u, transpose(far) * u) < 5e-3

    # --- adjoint ----------------------------------------------------------------------------------
    # This was the fixture the Petrov adjoint bug was pinned on: 0.994 relative error, O(1) rather
    # than a tolerance issue, returned silently. Fixed by the conjugated leaf multiply
    # (`src/assembly/cores/adjointleafmul.jl`), which supplies the three conjugations H2Factory's default
    # adjoint does not -- `α` and both leaf collections. Measured 6.848e-4 now, the same accuracy as
    # the transpose, so this bound is the same 5e-3 the other two directions use.
    adjointerror = rel(adjoint(A) * u, adjoint(far) * u)
    @test adjointerror < 5e-3

    # `A^H u == conj(A^T conj(u))` is an identity for any linear operator, so the transpose path
    # gives an INDEPENDENT route to the adjoint -- one that never touches `leafmul` or the adjoint
    # `α`. It was the diagnostic that localized the bug (it reproduced the true adjoint while the
    # native traversal did not), and it keeps its value as a check that the native path did not
    # merely move the error: the two now agree to 11 digits, not just to this test's tolerance.
    @test rel(conj.(transpose(A) * conj.(u)), adjoint(far) * u) < 5e-3
    @test rel(adjoint(A) * u, conj.(transpose(A) * conj.(u))) < 1e-10

    # Duality is the property the adjoint exists to have, and it is not implied by the bound above:
    # `<u, Ax> == <A^H u, x>` couples the forward and adjoint traversals directly, with no dense
    # reference in it at all. Measured 7e-14.
    @test abs(dot(u, A * x) - dot(adjoint(A) * u, x)) / abs(dot(u, A * x)) < 1e-10
end
