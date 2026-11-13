module EquidistantPipelineTests

# THE EQUIDISTANT GRID DRIVING THE REAL MLFMA PIPELINE, end to end, against a dense BEAST reference.
#
# Every other testset in this directory exercises the submodule in isolation. This one swaps the
# angular representation underneath an actual `GalerkinMLFMA` -- tree, leaf far-field assembly,
# aggregation, translation, disaggregation, near field -- and checks the matvec.
#
# WHAT THE CONTROL IS FOR, and it carries most of the value here. The absolute error (~7e-3 on this
# mesh) is NOT set by the angular representation: it is dominated by a source both configurations
# share, and it does not fall below that floor no matter how far ϵ is tightened. So an absolute
# threshold would pin the wrong thing. What is diagnostic is that the equidistant result tracks the
# shipped Gauss-Legendre configuration to several significant figures at every ϵ -- including the
# NON-MONOTONE rise from 1e-5 to 1e-7, which both paths show together. A wiring bug (wrong parity, a
# transposed grid, an unweighted adjoint) would break that agreement long before it moved the floor.
#
# AND THEN THERE IS THE DIFFERENTIAL TESTSET AT THE BOTTOM, which is the one that actually gets
# underneath that floor. Both strategies share the tree, the leaf far fields and the near field, so
# differencing their matvecs cancels every error source except the angular path -- exposing a residual
# of 2.7e-5 falling to 1.5e-6, two to three orders below the floor the comparisons above are stuck at.
# Anything sensitive enough to matter is measured there; the ϵ-tracking testsets are the coarse screen.
#
# BOTH MAP DIRECTIONS ARE ASSERTED. `A * x` was correct on this grid for a full session while `A' * x`
# raised a `MethodError` -- see the adjoint testset. A file that tested only the matvec would have
# reported this representation as working.
#
# Correspondingly this file does NOT verify interpolation accuracy in absolute terms -- `backends.jl`
# does that against analytic ground truth, where the FFT backend reaches 2e-16. It verifies that the
# pipeline uses the representation correctly, in both directions.
#
# Measured when written (tworects, κ=2π, tree at minhalfsize = rootsize/5):
#
#     ϵ       Gauss-Legendre   equi+FFT       equi+Lagrange p=10
#     1e-2    9.5305e-3        9.5295e-3      9.5294e-3
#     1e-3    7.1907e-3        7.1906e-3      7.1906e-3
#     1e-5    6.8653e-3        6.8661e-3      6.8661e-3
#     1e-7    7.1884e-3        7.1885e-3      7.1885e-3

using Test
using MLFMA
using MLFMA.Equidistant:
    EquidistantSampling,
    EquidistantSamplingBuilder,
    FFTBackend,
    LagrangeBackend,
    EquidistantInterpolator
using BEAST
using CompScienceMeshes
using H2Trees
using LinearAlgebra
using Random

const EQB = EquidistantSamplingBuilder{Float64}()

mesh = CompScienceMeshes.read_gmsh_mesh(
    joinpath(pkgdir(MLFMA), "test", "assets", "tworects.msh")
)
const X = raviartthomas(mesh)
const EFIE = MLFMA.efie(; wavenumber=2 * pi)
_, rootsize = H2Trees.boundingbox(BEAST.positions(X))
const TREE = TwoNTree(
    X;
    builder=TwoNTreeBuilder(;
        minhalfsize=rootsize / 5, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
    ),
)

const DENSE = BEAST.assemble(EFIE, X, X)
Random.seed!(11)
const XVEC = randn(ComplexF64, numfunctions(X))
const WANT = DENSE * XVEC

const PHASESHIFTER = MLFMA.phaseshiftertype(MLFMA.MLFMAStrategy(EFIE, X))

const WANTADJOINT = DENSE' * XVEC

"""
Relative error of one strategy's matvec against the dense reference.
"""
function relerr(strategy)
    A = MLFMA.GalerkinMLFMA(EFIE, X; tree=TREE, strategy=strategy)
    return norm(A * XVEC - WANT) / norm(WANT)
end

"""
Relative error of one strategy's ADJOINT matvec against the dense reference.
"""
function adjointrelerr(strategy)
    A = MLFMA.GalerkinMLFMA(EFIE, X; tree=TREE, strategy=strategy)
    return norm(A' * XVEC - WANTADJOINT) / norm(WANTADJOINT)
end

"""
Both matvec directions for one strategy, so the map is built once.
"""
function bothdirections(strategy)
    A = MLFMA.GalerkinMLFMA(EFIE, X; tree=TREE, strategy=strategy)
    return (forward=A * XVEC, adjoint=A' * XVEC)
end

gausslegendre(ϵ) =
    MLFMA.MLFMAStrategy(EFIE, X; samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ))

function equidistant(backend, ϵ)
    IT = typeof(EquidistantInterpolator{ComplexF64}(backend))
    return MLFMA.MLFMAStrategy(
        EFIE,
        X;
        samplingstrategy=MLFMA.SamplingStrategy(; samplingbuilder=EQB, ϵ=ϵ),
        leveltransferstrategy=MLFMA.LevelTransferStrategy(IT, PHASESHIFTER),
    )
end

@testset "the equidistant grid resolves per level through the real sampling function" begin
    sf = MLFMA.sampling(EQB)
    for level in H2Trees.levels(TREE)
        sampling, L = sf(MLFMA.γ(EFIE), 1e-5, TREE; level=level)
        @test sampling isa EquidistantSampling
        # Nφ = 2Nθ is this representation's shape, and `cutoffparameter` must invert the rate the
        # sampling function chose -- if it did not, every level pair would be built for the wrong band.
        @test MLFMA.Nφ(sampling) == 2 * MLFMA.Nθ(sampling)
        @test iseven(MLFMA.Nφ(sampling))
        @test MLFMA.cutoffparameter(sampling) == L
    end
end

@testset "the interpolator is constructible the way the map constructors build it" begin
    # `interpolatortype(strategy)(samplings, parentsamplings; interpolationtrait=...)` is the exact
    # shape `GalerkinMLFMA` uses, and it is why the trait is a FIELD rather than a type parameter --
    # `EquidistantInterpolator{S,T,B}` has to be concrete and callable.
    IT = typeof(EquidistantInterpolator{ComplexF64}(FFTBackend()))
    @test isconcretetype(IT)

    child = EquidistantSampling{Float64}(6, 12)
    parent = EquidistantSampling{Float64}(12, 24)
    interpolator = IT([child], [parent]; interpolationtrait=MLFMA.interpolationtrait(EFIE))

    # The map constructors `error()` when these disagree, so a constructor that dropped the trait
    # would fail construction rather than produce a wrong answer -- but only for operators whose
    # trait is not the default, which is why this is asserted explicitly.
    @test MLFMA.interpolationtrait(interpolator) == MLFMA.interpolationtrait(EFIE)
    @test MLFMA.interpolationtrait(interpolator) == MLFMA.SphericalθφInterpolation()

    # `nchunks` is passed unconditionally by both map constructors.
    @test MLFMA.anterpolator(interpolator; nchunks=1) isa MLFMA.AbstractAnterpolator

    # A grid mismatch is caught at construction, naming the fix, rather than at the first apply.
    spherical = MLFMA.SphericalSampling{Float64}(6, 12)
    @test_throws ErrorException IT([spherical], [parent])
    @test_throws ErrorException IT([child, child], [parent])
end

@testset "the equidistant pipeline matches the Gauss-Legendre control across ϵ" begin
    for ϵ in (1e-2, 1e-3, 1e-5, 1e-7)
        control = relerr(gausslegendre(ϵ))
        for backend in (FFTBackend(), LagrangeBackend(10))
            got = relerr(equidistant(backend, ϵ))
            # 2% relative agreement between the two ERRORS. Loose enough not to pin one machine's
            # round-off, tight enough that any of the wiring bugs this exists to catch -- a flipped
            # parity, a transposed grid, the weighted adjoint in place of the bare one -- moves it by
            # orders rather than percent.
            @test isapprox(got, control; rtol=0.02)
        end
    end
end

@testset "the equidistant matvec is genuinely close to the dense reference" begin
    # Non-vacuity for the testset above: agreeing with the control would be worthless if both were
    # simply wrong. This is the absolute statement, at the floor the mesh and near-field impose.
    @test relerr(equidistant(FFTBackend(), 1e-5)) < 1e-2
    @test relerr(gausslegendre(1e-5)) < 1e-2
end

@testset "THE ADJOINT MAP runs and is as accurate as the forward one" begin
    # THIS DIRECTION WAS BROKEN AND THE FORWARD TESTS ABOVE ALL PASSED. `ConjugatedInterpolator` --
    # the conjugate-only aggregation an adjoint map applies, added in #104 -- had its functor typed to
    # `SphericalSampling`, so on this grid it FAILED TO DISPATCH rather than failing a comparison:
    # `A * x` was correct while `A' * x` raised a `MethodError` from inside `_applyforwardtransfer!`.
    # Nothing that tests only the matvec can see that, which is the whole reason this testset exists.
    for backend in (FFTBackend(), LagrangeBackend(10))
        @test adjointrelerr(equidistant(backend, 1e-5)) < 1e-2
        # The adjoint is not systematically worse than the forward direction. A conjugation applied in
        # the wrong place -- the #35/#61/#104 family -- typically survives as a small but persistent
        # degradation rather than an outright failure, so this compares the two directions rather than
        # only bounding one.
        @test isapprox(
            adjointrelerr(equidistant(backend, 1e-5)),
            adjointrelerr(gausslegendre(1e-5));
            rtol=0.02,
        )
    end
end

@testset "differencing the two configurations isolates the angular representation" begin
    # THE SENSITIVE TEST, and the only one here that can see the representation at all.
    #
    # Both strategies share the tree, the leaf far fields and the near field, so subtracting one
    # matvec from the other CANCELS every error source except the far-field angular path. That is what
    # gets underneath the ~7e-3 floor the dense comparison is stuck at: the residual difference is
    # 2.7e-5 falling to 1.5e-6 across the ϵ sweep -- two to three orders below the floor, and
    # DECREASING, which is the signature of two valid discretizations converging to the same operator.
    #
    # A defect too small to move the floor shows up here immediately. Measured when written:
    #
    #     ϵ       forward    adjoint
    #     1e-2    2.732e-5   2.594e-5
    #     1e-3    4.118e-6   3.961e-6
    #     1e-5    1.849e-6   1.766e-6
    #     1e-7    1.512e-6   1.442e-6
    diffs = map((1e-2, 1e-3, 1e-5)) do ϵ
        control = bothdirections(gausslegendre(ϵ))
        got = bothdirections(equidistant(FFTBackend(), ϵ))
        forward = norm(got.forward - control.forward) / norm(control.forward)
        adj = norm(got.adjoint - control.adjoint) / norm(control.adjoint)
        # Bounded well under the dense-comparison floor, so this cannot pass by being insensitive.
        @test forward < 1e-4
        @test adj < 1e-4
        # The adjoint path must track the forward one. A misplaced conjugation would separate them
        # while leaving both individually small.
        @test isapprox(adj, forward; rtol=0.1)
        forward
    end

    # Convergent, not merely small: refining ϵ must bring the two representations TOGETHER. A wiring
    # bug contributes a fixed offset that refinement cannot remove, so this ordering is what
    # distinguishes "two good discretizations" from "two discretizations that happen to be close".
    @test diffs[2] < diffs[1]
    @test diffs[3] < diffs[2]
end

end # module
