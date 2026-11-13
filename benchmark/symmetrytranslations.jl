# WHAT SYMMETRY-REDUCED TRANSLATIONS COST AND WHAT THEY BUY, on real maps.
#
# `MLFMA.SymmetryTranslationCollection` stores one translation per SYMMETRY ORBIT instead of one per
# direction, and reads each interaction's own translation through an index permutation. The
# reduction in UNIQUE TRANSLATION COUNT is pinned by the test suite (7.22x on a 2λ sphere). Three
# things it does not pin, and which decide whether this should ever become the default:
#
#   1. BYTES. A count ratio is not a memory ratio -- the reduced collection also stores a
#      `PermutedTranslation` per pair and one permutation vector per (level, symmetry).
#   2. CONSTRUCTION TIME. Fewer translations to build, but a canonicalization pass and a permuted
#      store to assemble.
#   3. MATVEC TIME. Every coupling is now a GATHER through a permutation rather than a contiguous
#      read. Measured in isolation the gather ran at 0.93-1.12x of an identical-shape control; that
#      was a loop microbenchmark, not a traversal.
#
# WHY A SEPARATE ENTRY POINT AND NOT A CASE IN `SUITE`. The headline quantity is the resident size of
# a stored object, which `BenchmarkTools` does not measure -- it measures allocation during a timed
# call. `benchmark/interpolation/` set the precedent for a suite that reports something other than
# time beside the time, and for keeping it out of `SUITE` so the two cannot be included into one
# another. It is also minutes per fixture, which does not belong in the default run.
#
#     julia --project=benchmark benchmark/symmetrytranslations.jl
#
# MEASURED (16 BLAS threads, 1 Julia thread). Ratios are unreduced / reduced, so above 1 means the
# reduction wins:
#
#     radius  unknowns  pairs   unique        collection MiB     build s      matvec ms
#      1.5λ     4206     1494   364→39  9.33x  15.70→2.09 7.50x  7.09→7.16 0.99x   47.6→46.9 1.02x
#      2.0λ     6747     1898   356→41  8.68x  16.47→2.23 7.38x 11.10→11.60 0.96x  66.5→66.4 1.00x
#      2.5λ    10656     9764  948→121  7.83x  50.93→7.29 6.98x 17.45→17.35 1.01x 287.6→296.5 0.97x
#
# THE ANSWER IS: ~7x SMALLER, AND FREE. Construction is 0.96-1.01x and the matvec 0.97-1.02x, both
# inside the run-to-run spread -- the canonicalization pass is paid back by building fewer
# translations, and the permutation gather costs nothing measurable in a traversal, which is what
# the 0.93-1.12x loop microbenchmark predicted.
#
# THE BYTE RATIO TRAILS THE COUNT RATIO by about 12 %, and that gap is the honest price of the
# scheme: a `PermutedTranslation` per pair plus one permutation vector per (level, symmetry). It is
# a fixed overhead, so it matters least where the reduction matters most.
#
# WHAT THIS SWEEP DOES NOT SHOW, stated because an earlier draft of this comment claimed it did.
# The projections behind this work had the reduction GROWING with problem size (9.10x at 8λ across,
# 9.69x at 16λ). Here it DECREASES, 9.33x → 7.83x. That is not a contradiction and is not evidence
# either way: these spheres are 3λ to 5λ ACROSS, an entirely different regime, and the 2.5λ row
# gains a tree level (1898 → 9764 pairs), so its ratio mixes a size change with a level-structure
# change. Reading a trend off three points that are not comparable would be the same mistake as
# reading a translation cost off one `@elapsed`. Settling the scaling question needs a sweep that
# holds the level count fixed, which is a different and much more expensive fixture.
#
# A CORRECTNESS GUARD RUNS BEFORE EVERY TIMING. A benchmark that does not check its output will
# happily report a speedup for a broken kernel, and this particular kernel fails SILENTLY -- a wrong
# permutation gives the right shapes and a plausible field. The row is refused, not flagged, if the
# two maps disagree.

using BenchmarkTools
using BEAST
using CompScienceMeshes
using H2Factory
using H2Trees
using LinearAlgebra
using MLFMA
using Printf
using Random: Xoshiro

# Pinned for the same reason `leafbatch.jl` pins it: this machine defaults to 128 BLAS threads on 256
# shared cores, the far-field leaf multiplies inside a matvec are BLAS-bound, and comparing two runs
# taken at different thread counts is comparing noise. The reduced and unreduced columns are affected
# identically, so the RATIO is robust either way -- the absolute times are not.
const SYMBLASTHREADS = 16
LinearAlgebra.BLAS.set_num_threads(SYMBLASTHREADS)

const SYMWAVELENGTH = 1.0
const SYMWAVENUMBER = 2π / SYMWAVELENGTH
const SYMEPSILON = 1e-4
const SYMOPERATOR = MLFMA.efie(; wavenumber=SYMWAVENUMBER, alpha=(-im * SYMWAVENUMBER))

# `minhalfsize = λ/4` is `MINHALFSIZERECOM`, so no fixture here sits under the low-frequency
# breakdown floor. The mesh is λ/6 throughout: element density does not affect translation storage
# (that is set by the tree geometry), so holding it fixed keeps the sweep about SIZE.
function symmap(radius, collection)
    space = raviartthomas(meshsphere(radius * SYMWAVELENGTH, SYMWAVELENGTH / 6))
    tree = H2Trees.TwoNTree(
        space; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=SYMWAVELENGTH / 4, minvalues=0)
    )
    # `GalerkinMLFMA` asserts this internally and the AssertionError names neither the radius nor
    # the fixture. A 1λ sphere at `minhalfsize = λ/4` bottoms out at two levels, which is how this
    # was found -- see the same hazard in `leafbatch.jl`.
    H2Trees.numberoflevels(tree) > 2 || error(
        "symmetrytranslations: a $(radius)λ sphere at minhalfsize λ/4 gives " *
        "$(H2Trees.numberoflevels(tree)) levels, and a map needs more than 2. Enlarge the " *
        "sphere; do not lower `minhalfsize`, which would put the fixture under the " *
        "low-frequency breakdown floor and measure a configuration nobody runs.",
    )
    strategy = MLFMA.MLFMAStrategy(
        SYMOPERATOR,
        space;
        # The near block is stubbed. It is identical between the two columns, is not on the path
        # under test, and #135 measured it at 99.8 % of a build -- leaving it in would bury the
        # construction difference this file exists to report.
        nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
            MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
        ),
        samplingstrategy=MLFMA.SamplingStrategy(SYMOPERATOR; ϵ=SYMEPSILON),
        i2ostrategy=MLFMA.I2OTranslationStrategy(; TranslationCollection=collection),
    )
    return space,
    tree,
    MLFMA.GalerkinMLFMA(SYMOPERATOR, space; tree=tree, strategy=strategy, verbose=false)
end

# Minimum of a few builds. `@belapsed` is wrong here: a map construction is seconds long and
# allocates gigabytes, so BenchmarkTools' sampling machinery would spend minutes to learn what three
# runs already say, and the garbage collector dominates the spread either way.
function buildseconds(radius, collection; repetitions=2)
    return minimum(1:repetitions) do _
        return @elapsed symmap(radius, collection)
    end
end

function report(radius)
    space, _, full = symmap(radius, MLFMA.TranslationCollection)
    _, _, reduced = symmap(radius, MLFMA.SymmetryTranslationCollection)

    fulltfs = MLFMA.translationcollection(full)
    reducedtfs = MLFMA.translationcollection(reduced)
    fullstats = H2Factory.translationstats(fulltfs)
    reducedstats = H2Factory.translationstats(reducedtfs)

    # THE GUARD. Refuse the row rather than report it: every number below would look fine on a map
    # that silently used the wrong translation.
    x = rand(Xoshiro(0x5EED1234), ComplexF64, numfunctions(space))
    reference = full * x
    disagreement = norm(reduced * x - reference) / norm(reference)
    disagreement < 1e-11 || error(
        "symmetrytranslations: the reduced map disagrees with the unreduced one by " *
        "$(disagreement) at radius $(radius)λ. That is far above the ~1e-14 round-off the two " *
        "paths differ by, so this is a correctness failure and no timing below would mean " *
        "anything.",
    )

    # `summarysize` and not `sizeof`: both collections SHARE their translation arrays between pairs
    # (that is what the unique-count dedup already does), and `summarysize` counts a shared object
    # once. It is the same accounting on both sides, which is what makes the ratio meaningful.
    fullbytes = Base.summarysize(fulltfs)
    reducedbytes = Base.summarysize(reducedtfs)

    y = similar(reference)
    fullmatvec = @belapsed mul!($y, $full, $x)
    reducedmatvec = @belapsed mul!($y, $reduced, $x)

    return (
        radius=radius,
        unknowns=numfunctions(space),
        pairs=fullstats.npairs,
        uniquefull=fullstats.nunique,
        uniquereduced=reducedstats.nunique,
        countratio=fullstats.nunique / reducedstats.nunique,
        fullbytes=fullbytes,
        reducedbytes=reducedbytes,
        byteratio=fullbytes / reducedbytes,
        fullbuild=buildseconds(radius, MLFMA.TranslationCollection),
        reducedbuild=buildseconds(radius, MLFMA.SymmetryTranslationCollection),
        fullmatvec=fullmatvec,
        reducedmatvec=reducedmatvec,
        disagreement=disagreement,
    )
end

# Starts at 1.5λ: a 1λ sphere at `minhalfsize = λ/4` has only two levels and no map can be built on
# it. Three sizes is not enough to fit an exponent, but it is enough to see which way the ratio moves.
const SYMRADII = (1.5, 2.0, 2.5)

function main()
    @printf("BLAS threads: %d, Julia threads: %d\n\n", SYMBLASTHREADS, Threads.nthreads())
    rows = [report(radius) for radius in SYMRADII]

    println("Translation storage and its cost. Ratios are unreduced / reduced;")
    println("above 1 means the reduction wins, below 1 means it costs.\n")
    @printf(
        "%-7s %8s %7s %17s %19s %17s %15s\n",
        "radius",
        "unknowns",
        "pairs",
        "unique  (ratio)",
        "collection  (ratio)",
        "build s (ratio)",
        "matvec (ratio)",
    )
    for r in rows
        @printf(
            "%5.1fλ %8d %7d %6d→%-5d %5.2fx %7.2f→%-7.2f %5.2fx %5.2f→%-5.2f %5.2fx %5.1f→%-5.1f %5.2fx\n",
            r.radius,
            r.unknowns,
            r.pairs,
            r.uniquefull,
            r.uniquereduced,
            r.countratio,
            r.fullbytes / 2^20,
            r.reducedbytes / 2^20,
            r.byteratio,
            r.fullbuild,
            r.reducedbuild,
            r.fullbuild / r.reducedbuild,
            r.fullmatvec * 1e3,
            r.reducedmatvec * 1e3,
            r.fullmatvec / r.reducedmatvec,
        )
    end
    println()
    println("Worst map disagreement: ", maximum(r -> r.disagreement, rows))
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
