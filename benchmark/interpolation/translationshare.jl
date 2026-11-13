# What fraction of the far-field matvec is I2O TRANSLATION?
#
# RUN AS:  julia --startup-file=no --project=test benchmark/interpolation/translationshare.jl
# The `benchmark` environment resolves a registry BEAST whose `blockassembler` has no `primer`, so any
# study that builds a real map must run under `test`. Budget ~25 min: the 6.0e8 mesh alone takes 7
# minutes, and each frequency builds the map twice.
#
# ANSWER: 70 %, flat across 105k-411k unknowns and 6-7 levels. Table in README.md.
#
# NOTE what this does NOT license: antipodal symmetry (#110) cannot collect any of this 70 %.
# F(-k) = conj(F(k)) holds per basis function, but a box moment is a COMPLEX-coefficient sum of
# those and the symmetry does not survive complex scaling -- it is gone after the first
# aggregation, and translation sees only aggregated moments.
#
# THIS IS THE ONE NUMBER THE RING FEATURE DEPENDS ON. Ragged rings save ~16 % of the SAMPLES. That
# saving is collected in full, with no ragged penalty, only by translation -- an elementwise product
# with T_L over the grid. The local interpolation kernel measured SLOWER on ragged grids (#118,
# B = 0.44), so if translation is a small share of the matvec the whole feature is a net loss.
#
# WHY THE PREVIOUS ATTEMPT WAS INVALID. Translation was estimated as
#     farmul! - aggregate! - disaggregate!
# which returned -2.4 % -- an impossible negative share. The assumption is that those two calls
# PARTITION the far path. They COVER it: this script's own output shows `disaggregate!` dropping
# 3021 -> 475 ms when translation is swapped out, so `disaggregate!` CONTAINS the I2O work. So
# agg + disagg is already the whole far path and the "remainder" is noise about zero -- ~1 % on
# tworects, -2.4 % here. Both retracted. (An intermediate diagnosis, that the two merely OVERLAP,
# was also wrong: disaggregate! is a superset, not a partial overlap.)
#
# HOW THIS ONE IS DIFFERENT. `DummyI2OTranslationCollection`'s `i2otranslation!` is `return out` --
# it applies no coupling. Building the SAME fixture with it, and differencing `farmul!` against the
# real collection, isolates translation BY CONSTRUCTION: every other stage (aggregation, the leaf
# multiplies, the O2O downward pass, all the traversal bookkeeping) is bit-for-bit the same work in
# both runs. Nothing overlaps, so nothing can be double-counted.
#
#     t_far(real) - t_far(dummy) = I2O translation cost.
#
# The result is NOT numerically correct with the dummy -- y comes out wrong. That is fine and
# expected: this measures time, not values.
#
# Fixture as established: sphere R = 4.79 m, edge length λ/10, leaf boxes λ/4, ϵ = 1e-4, and NO near
# field (`DummyNearInteractionsStrategy`) -- near-field assembly is what put every earlier sphere
# fixture over budget and is irrelevant to angular sampling.

using MLFMA, H2Factory, BEAST, CompScienceMeshes, H2Trees, LinearAlgebra, Random, Printf

best(f, n=5) = (f(); minimum(@elapsed(f()) for _ in 1:n))

const R = 4.79

# Build a map over an ALREADY-BUILT tree with the given I2O collection, time the far path, and let
# the map go. Returns (far, agg, disagg) in seconds. One map alive at a time keeps peak memory at
# roughly what a single run needs rather than two.
function timefar(efie, X, tree, N, Collection)
    A = MLFMA.GalerkinMLFMA(
        efie,
        X;
        tree=tree,
        strategy=MLFMA.MLFMAStrategy(
            efie,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
            nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
            ),
            i2ostrategy=MLFMA.I2OTranslationStrategy(; TranslationCollection=Collection),
        ),
        verbose=false,
    )

    Random.seed!(3)
    x = randn(ComplexF64, N)
    y = similar(x)

    H2Factory.farmul!(y, A, x)
    tf = best(() -> H2Factory.farmul!(y, A, x))
    ta = best(() -> H2Factory.aggregate!(A, x))
    H2Factory.aggregate!(A, x)
    td = best(() -> H2Factory.disaggregate!(y, A))

    A = nothing
    GC.gc()
    return (tf, ta, td)
end

function run(f)
    c = 3e8
    λ = c / f
    ω = 2π * f
    μ = 4π * 1e-7
    κ = ω / c
    h = λ / 10

    tmesh = @elapsed X = raviartthomas(meshsphere(R, h))
    N = numfunctions(X)
    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0))
    lv = H2Trees.numberoflevels(tree)
    @printf(
        "f=%.2e  λ=%.3f  R/λ=%.2f  N=%d  levels=%d  (mesh %.0f s)\n",
        f,
        λ,
        R / λ,
        N,
        lv,
        tmesh
    )
    flush(stdout)

    real = timefar(efie, X, tree, N, MLFMA.TranslationCollection)
    dummy = timefar(efie, X, tree, N, MLFMA.DummyTranslationCollection)

    ttrans = real[1] - dummy[1]
    @printf("    far  real %7.1f ms   dummy %7.1f ms\n", real[1] * 1e3, dummy[1] * 1e3)
    @printf(
        "    agg  real %7.1f ms   dummy %7.1f ms   (CONTROL: must match)\n",
        real[2] * 1e3,
        dummy[2] * 1e3
    )
    @printf(
        "    dis  real %7.1f ms   dummy %7.1f ms   (must NOT match: contains I2O)\n",
        real[3] * 1e3,
        dummy[3] * 1e3
    )
    @printf(
        "    ==> I2O TRANSLATION %7.1f ms = %5.1f%% of the far path\n\n",
        ttrans * 1e3,
        100 * ttrans / real[1]
    )
    flush(stdout)
    return nothing
end

println("Sphere R=$(R) m, edge λ/10, NO near field.")
println(
    "Translation isolated by swapping in DummyI2OTranslationCollection, NOT by subtraction.\n",
)
for f in (3.0e8, 4.2e8, 6.0e8)
    try
        run(f)
    catch e
        @printf("f=%.2e FAILED: %s\n\n", f, sprint(showerror, e))
        flush(stdout)
    end
end
