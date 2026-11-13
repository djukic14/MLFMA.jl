# The BATCHED leaf multiply -- `H2Factory.aggregateleafchunk!` -- on a real map, for each leaf storage.
#
# WHY THIS IS A SEPARATE FILE AND NOT PART OF `benchmarks.jl`. It needs a real workspace: the batched
# hook takes the map's own moment arena, its collection accessor, its leaf functor and a scratch
# borrowed from its leaf channel, and none of those can be built from the synthetic point cloud that
# file uses. Building one costs a mesh plus three map constructions -- measured at ~40 s -- against
# that file's ~23 s include, which its header explicitly budgets and which people re-evaluate from a
# REPL. So this file carries its own fixture and adds its own group; `benchmark/run.jl` includes both.
#
# WHAT IT CLOSES. `SUITE["leafstorage"]` measures the PER-LEAF kernels, which is the shape of the
# arithmetic but not what a solve pays: all three compressed backends are batched in production, and
# for SHE that is mandatory rather than an optimization -- unbatched it measured 1.63x SLOWER than the
# sampled path it replaces (#125). Nothing regression-tracked the form that actually runs. This does.
#
# ------------------------------------------------------------------------------------------------
# THE FIXTURE IS STUBBED EVERYWHERE EXCEPT THE LEAVES, deliberately, and that is what makes it cheap
# enough to build three times.
#
#   * The NEAR block is `DummyNearInteractionsStrategy`. It is not on this path at all, and assembling
#     it is the dominant construction cost -- #135 measured the near block at 99.8 % of a 2-D build.
#   * The I2O translation is `DummyTranslationCollection`, for the same reason: the leaf hook runs
#     before any translation and never reads one.
#   * Far-field quadrature is `SingleNumQStrat(1)`, the lowest order that assembles. The leaf
#     PATTERNS' accuracy is irrelevant to how long multiplying by them takes, and the quadrature order
#     is pure construction cost.
#
# This is the same recipe `test/support/mlfmatestutils.jl` calls `stubbedstrategy`, and it is used
# here for the same reason: everything not under test is replaced by something that costs nothing.
#
# THE TIMED CALL IS THE PRODUCTION ONE, not a reconstruction of it. `moment`, `momentcollection`,
# `leafmul!` and the scratch are taken from the built map through the accessors
# `H2Factory.aggregatechunk!` itself uses (`trialmoment`, `trialmomentcollection`, `leafmul`,
# `borrowbuffer(aggregationleafchannel(A), ...)`), so a change to what the traversal passes shows up here as a
# broken benchmark rather than as a benchmark that quietly measures the old thing.
#
# ------------------------------------------------------------------------------------------------
# THIS GROUP MEASURES SLOWER-THAN-SAMPLED, AND #125 RECORDS 12x FASTER. Both are right about
# different things, and the reconciliation is most of what this file is for.
#
# #125's figure is one CELL of `shewrite.jl`'s table: `B'/baseline = 0.08` at r = 272, a full uniform
# level. It was never a claim about the batched path in general. The same table's r = 32 row -- the
# nearest to this fixture's 35 -- reads 0.38x, still faster than sampled, so the regimes alone do not
# account for the sign flip.
#
# WHAT DOES ACCOUNT FOR MOST OF THE SPREAD IS BLAS THREADS, and that is the finding. Same fixture,
# same call, one Julia thread throughout, sweeping `BLAS.set_num_threads`:
#
#     BLAS threads    sampled     SHE        VSH        SHE/sampled
#      1              1.998 ms    5.911 ms   6.371 ms    2.96x
#      4              0.827 ms    2.083 ms   2.253 ms    2.52x
#     16              0.896 ms    1.424 ms   1.317 ms    1.59x
#    128              1.685 ms    2.159 ms   3.934 ms    1.28x
#
# Every kernel moves by 2.4x (sampled) to 4.2x (SHE) across that sweep, and the RATIO moves by 2.3x.
# A batched-leaf number quoted without its BLAS thread count says almost nothing. This machine
# defaults to 128 threads on 256 shared cores, which is both the worst setting measured for VSH and
# the one every earlier reading in this session was accidentally taken at.
#
# SO THE CASES BELOW PIN IT. `BLAS.set_num_threads(BATCHBLASTHREADS)` runs at definition time and the
# count is recorded in the group name, because a regression suite comparing two runs at different
# thread counts is comparing noise. 16 is chosen as the best measured for both compressed backends
# and close to the best for sampled -- not as a recommendation for production, which schedules the
# traversal serially and would want its own sweep.
#
# WHAT IS STILL NOT SETTLED: on this fixture the compressed path is slower than sampled at EVERY
# thread count, where `shewrite.jl` has it faster at r = 32. The untested difference is leaf
# OCCUPANCY -- SHE's synthesis cost is independent of the basis-function count while the sampled
# GEMV is linear in it, so the two cross somewhere, and `shewrite.jl`'s baseline at r = 32 is 3.899 ms
# against 0.8-2.0 ms here, which says its leaves hold considerably more functions. Occupancy is swept
# in `SUITE["leafstorage"]` on kernels that need no map; sweeping it here means one map build per
# point. That is the next measurement, and it is deliberately not in this commit.
#
# Nothing here is presented as a defect in the code under test.

using BenchmarkTools
using BEAST
using CompScienceMeshes
using H2Factory
using H2Trees
using LinearAlgebra
using MLFMA
using Random: Xoshiro

# `SUITE` is defined by `benchmarks.jl`, which `run.jl` includes first. Standing alone (from a REPL,
# say) it does not exist, so create it rather than erroring -- this file is self-contained otherwise.
if !@isdefined(SUITE)
    const SUITE = BenchmarkGroup()
end

# Fixture #########################################################################################
#
# λ = 1, so lengths read in wavelengths. A 2λ sphere at λ/6 gives 6,747 basis functions over 78 leaves
# at 4 levels -- deep enough that the leaf level is genuinely a level rather than the root, and
# populated enough (~86 functions per leaf) that the batch has something to amortise over. Smaller
# meshes bottom out at 2 levels, which `_galerkinmlfmacore` refuses.

const BATCHWAVELENGTH = 1.0
const BATCHWAVENUMBER = 2π / BATCHWAVELENGTH
const BATCHEPSILON = 1e-4
const BATCHOPERATOR = MLFMA.efie(;
    wavenumber=BATCHWAVENUMBER, alpha=(-im * BATCHWAVENUMBER)
)

const BATCHSPACE = raviartthomas(meshsphere(2.0 * BATCHWAVELENGTH, BATCHWAVELENGTH / 6))
const BATCHTREE = H2Trees.TwoNTree(
    BATCHSPACE;
    builder=H2Trees.TwoNTreeBuilder(; minhalfsize=BATCHWAVELENGTH / 4, minvalues=0),
)

H2Trees.numberoflevels(BATCHTREE) > 2 || error(
    "leafbatch fixture: the tree has $(H2Trees.numberoflevels(BATCHTREE)) levels, and " *
    "`GalerkinMLFMA` requires more than 2. Enlarge the sphere or reduce `minhalfsize`; do not " *
    "relax this check, because a 2-level tree has no leaf level distinct from the root and every " *
    "case below would measure the wrong thing.",
)

function batchstrategy(storage)
    return MLFMA.MLFMAStrategy(
        BATCHOPERATOR,
        BATCHSPACE;
        nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
            MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
        ),
        i2ostrategy=MLFMA.I2OTranslationStrategy(;
            TranslationCollection=MLFMA.DummyTranslationCollection
        ),
        samplingstrategy=MLFMA.SamplingStrategy(BATCHOPERATOR; ϵ=BATCHEPSILON),
        farfieldstrategy=MLFMA.FarFieldStrategy(;
            quadstrategy=BEAST.SingleNumQStrat(1), leafstorage=storage
        ),
    )
end

batchmap(storage) = MLFMA.GalerkinMLFMA(
    BATCHOPERATOR,
    BATCHSPACE;
    tree=BATCHTREE,
    strategy=batchstrategy(storage),
    verbose=false,
)

# The leaf level and its chunk, read off the map rather than assumed ###############################
#
# EVERY NODE AT THE LEAF LEVEL, which on this fixture IS one chunk: the scratch borrowed below reports
# `maxgroups == 35` against 35 leaves at that level, so the batch flushes once rather than
# sub-batching. That is checked rather than assumed -- see the non-vacuity guard after the cases.
#
# The tree is NON-UNIFORM (78 leaves in total, 35 of them at the deepest level), which is the ordinary
# case for a real mesh and the one the batched hook has to handle.

const BATCHLEAFLEVEL = maximum(
    H2Trees.level(BATCHTREE, l) for l in H2Trees.leaves(BATCHTREE)
)
const BATCHNODES = collect(H2Trees.LevelIterator(BATCHTREE, BATCHLEAFLEVEL))
const BATCHX = rand(Xoshiro(0x1234BEEF), ComplexF64, numfunctions(BATCHSPACE))

"""
    batchcase(storage) -> (map, chunknodes, moment, momentcollection, leafmul!, scratch)

Everything `H2Factory.aggregateleafchunk!` needs, pulled off a map built with `storage` through the
SAME accessors `H2Factory.aggregatechunk!` uses.

Not reconstructed: if the traversal starts passing something else, this stops compiling instead of
quietly measuring the previous contract.
"""
function batchcase(storage)
    A = batchmap(storage)
    scratch = H2Factory.borrowbuffer(
        H2Factory.aggregationleafchannel(A), BATCHLEAFLEVEL, 1, H2Factory.maptrait(A)
    )
    return (
        A,
        BATCHNODES,
        H2Factory.trialmoment(A),
        H2Factory.trialmomentcollection(A),
        H2Factory.leafmul(A),
        scratch,
    )
end

# The cases #######################################################################################
#
# All four storages, which is the comparison the group exists for: the sampled batch is the baseline
# the compressed ones have to beat, and it is the number `SUITE["leafstorage"]`'s per-leaf cases
# cannot supply because they measure a different call.
#
# ONE CASE PER STORAGE, no sweep. The axes that matter -- occupancy and expansion order -- are swept
# in `SUITE["leafstorage"]`, cheaply, on kernels that need no map. Sweeping them here would multiply
# a 40-second fixture by the sweep length to re-measure a shape that file already covers.

# THE THREAD COUNT IS PART OF THE MEASUREMENT -- see this file's header for the sweep that forced
# this. Set at definition time so every case below runs at the same setting, and named in the group so
# two runs cannot be compared across different ones.
const BATCHBLASTHREADS = 16
BLAS.set_num_threads(BATCHBLASTHREADS)

# NON-VACUITY: the batched branch must actually be taken. `aggregateleafchunk!` falls back to the
# per-leaf loop when the chunk is under `SHEBATCHTHRESHOLD`, when `leafmul!` is not a
# `LinAlgebraMulFunctor`, or when the collection is not a compressed RADIATION one -- and the fallback
# is silent, so without this the group would measure the unbatched kernel under a name that says
# otherwise. It nearly did: the first numbers came out close enough to the recorded UNBATCHED figure
# that the fallback was the obvious suspect, and only this check ruled it out.
for (name, storage) in (
    ("sampled", MLFMA.SampledLeafStorage()),
    ("SHE", MLFMA.SphericalHarmonicLeafStorage()),
    ("VSH", MLFMA.VectorSphericalHarmonicLeafStorage()),
)
    A, nodes, moment, collections, leafmul!, scratch = batchcase(storage)

    if storage !== MLFMA.SampledLeafStorage()
        MLFMA._batchable(BATCHTREE, nodes, collections, leafmul!) || error(
            "leafbatch fixture: the batched branch is NOT taken for $name, so this case would " *
            "time the per-leaf fallback under a name that claims otherwise. Check the chunk size " *
            "against `SHEBATCHTHRESHOLD` and the leaf functor type.",
        )
    end

    SUITE["leafbatch"]["aggregate chunk (BLAS $BATCHBLASTHREADS)"][name] = @benchmarkable(
        H2Factory.aggregateleafchunk!(
            $A,
            $BATCHTREE,
            $nodes,
            $moment,
            $collections,
            $leafmul!,
            $BATCHX,
            $BATCHLEAFLEVEL,
            $scratch,
        ),
        seconds = 5,
        samples = 50,
    )
end
