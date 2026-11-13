# Benchmark suite for MLFMA's far-field layer (`src/farfield/`).
#
# DEFINES ONLY -- including this file builds the fixture and populates `SUITE`, and runs nothing.
# That is deliberate: it makes the file safe to re-evaluate from a REPL/VSCode session while
# developing. `benchmark/run.jl` is the entry point that actually runs it. See `benchmark/README.md`.
#
# Safe, but no longer sub-second: the construction group's mesh fixture at the bottom pulls in
# `CompScienceMeshes` and shells out to gmsh at include time, and its two spaces over that mesh took a
# first include from ~11 s to ~23 s and a re-include from ~0.7 s to ~2.8 s, and made both depend on a
# working mesher. Index `SUITE` directly rather than re-including if that is in your way -- everything
# outside `SUITE["construction"]` is built from the synthetic point cloud and never touches gmsh.
#
# What is measured: `FarField` construction and mirroring, the box-centering phase factors, the
# field-collection `mul!`/`conjmul!` kernels -- the operations `src/farfield/` owns -- and, in the
# separate `"construction"` group at the end, the one-off cost of BUILDING the per-leaf collections
# those kernels then run on.
#
# THE TWO GROUPS ANSWER DIFFERENT QUESTIONS and must not be read together. `"fieldcollections"` is
# per-matvec: it runs on every iteration of a solve, so nanoseconds matter. `"construction"` runs
# once per operator assembly, so its unit is milliseconds and a regression there costs setup time,
# not solve time. Mixing them into one group would put a 2-second BEAST quadrature next to a 300 ns
# `gemv` and invite comparing the two numbers, which means nothing.
#
# What is deliberately NOT measured, so its absence does not read as an oversight:
#
#   * Aggregation/disaggregation traversal and interpolation. Different layers, and the
#     workspace/scheduling machinery they need makes them a separate suite. Translation used to be
#     listed here for the same reason and no longer is: that reason applies to the I2O APPLY, which
#     lives in H2Factory, and not to `translationoperator!`, which takes an array, a γ, a direction
#     grid, a transfer vector and a cutoff. See the `"translation"` group.
#   * The flat `buildfieldcollection` path next to the tree `buildfieldcollections` one, which would
#     separate raw quadrature cost from the per-leaf ordering, clustering and box-centering around it.
#   * `farfieldlocal!`/`_store!` at the inner-loop level, which is what would attribute construction
#     cost to quadrature versus projection versus frame-cache lookup versus the matrix writes.
#   * Leaf-count scaling across two or three mesh sizes.
#
# Those three are wanted, in that order, and are held back deliberately: each one multiplies the
# construction group rather than adding to it, and gmsh variability makes the last one the noisiest of
# the three. The current group covers the LAYERS and the store rules; those cover the WHY.

using BenchmarkTools
using BEAST
using CompScienceMeshes
using H2Factory
using H2Trees
# One use: pinning the store-rule sweep to the full set of `AbstractFarFieldStoreRule` subtypes, so
# adding a sixth rule fails the fixture instead of quietly leaving the sweep incomplete.
using InteractiveUtils: subtypes
using LinearAlgebra
using MLFMA
using Random: Xoshiro
using StaticArrays

const SUITE = BenchmarkGroup()

# Fixture parameters ##############################################################################
#
# λ = 1, so every length below reads directly in wavelengths.

const WAVELENGTH = 1.0
const WAVENUMBER = 2π / WAVELENGTH
const MINHALFSIZE = WAVELENGTH / 4
const NLEVELS = 8
const EPSILON = 1e-4

# The operator supplies γ rather than a hand-built `im * κ`, so the benchmarks run against the same
# convention (and sign) production does. EFIE: two components, exercising Fθ/Fφ.
const OPERATOR = MLFMA.MWEFIE3D(; wavenumber=WAVENUMBER)
const GAMMA = BEAST.gamma(OPERATOR)
const NCOMPONENTS = MLFMA.farfielddims(OPERATOR)
# The operator's own far-field type. Benchmarks construct through this rather than through a
# component count, matching how production code allocates moments.
const FARFIELDTYPE = MLFMA.farfieldtype(OPERATOR)

# Deterministic tree ##############################################################################
#
# `NLEVELS` levels at `MINHALFSIZE` means a root halfsize of `MINHALFSIZE * 2^(NLEVELS-1)` = 32λ, so
# the points sit on a shell of that radius: the bounding box then has exactly that halfsize and
# subdivision reaches `MINHALFSIZE` at level `NLEVELS`. `minvalues=1` keeps the population criterion
# from stopping subdivision before the halfsize criterion does.
#
# A shell (not a filled ball) because that is what a real scatterer's basis-function centers look
# like, and it keeps the point count -- and so the tree build, the one non-trivial fixture cost --
# modest for the depth reached.

"""
    shellpoints(n, radius; seed=0x12345678)

`n` deterministic points on the sphere of the given radius, from a seeded `Xoshiro` so the same
points are produced on every run. Benchmark timings are only comparable across runs if the tree is
bit-identical, which means the point set must be.
"""
function shellpoints(n::Int, radius::Float64; seed=0x12345678)
    rng = Xoshiro(seed)
    points = Vector{SVector{3,Float64}}(undef, n)
    for i in eachindex(points)
        z = 2 * rand(rng) - 1
        φ = 2π * rand(rng)
        r = sqrt(1 - z^2)
        points[i] = radius * SVector(r * cos(φ), r * sin(φ), z)
    end
    return points
end

const ROOTHALFSIZE = MINHALFSIZE * 2^(NLEVELS - 1)
const POINTS = shellpoints(60_000, ROOTHALFSIZE)
const TREE = H2Trees.TwoNTree(
    POINTS; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=MINHALFSIZE, minvalues=1)
)

# Fail loudly rather than benchmark a different regime than the one this file documents. `error`, not
# `@assert`: `@assert` is removable at some optimization levels, and a silently shallower tree would
# quietly change every sampling size below.
H2Trees.numberoflevels(TREE) == NLEVELS || error(
    "benchmark fixture: expected a $NLEVELS-level tree at minhalfsize $MINHALFSIZE, got " *
    "$(H2Trees.numberoflevels(TREE)) levels ($(H2Trees.levels(TREE))). The point set or the " *
    "builder parameters no longer produce the intended depth; fix the fixture rather than the " *
    "check, or every per-level sampling size below refers to a level that does not exist.",
)

# Levels ##########################################################################################
#
# MEASURED, not assumed: `H2Trees.buildplans(TREE).relevantlevels` is `3:8` for this fixture -- the
# tree has 8 levels but levels 1 and 2 carry no aggregation, so no sampling is ever built for them.
# It is hardcoded rather than recomputed because `buildplans` costs ~50 s on this tree, which would
# dwarf the whole suite; the depth check above is what ties this recorded value to a verified tree
# shape. Rerun `buildplans` if the fixture changes.
const RELEVANTLEVELS = 3:8

# Finest (the leaf level), middle, and coarsest RELEVANT level. Not every level, and specifically not
# levels 1-2: they are outside `RELEVANTLEVELS`, and their samplings are enormous. Measured sizes for
# this fixture, one `FarField` (2 components, ComplexF64):
#
#   level 8  Nθ=28   Nφ=56     1,568 dirs    0.05 MiB   <- leaf
#   level 6  Nθ=70   Nφ=140    9,800 dirs    0.30 MiB
#   level 5  Nθ=120  Nφ=240   28,800 dirs    0.88 MiB   <- middle
#   level 4  Nθ=216  Nφ=432   93,312 dirs    2.85 MiB
#   level 3  Nθ=400  Nφ=800  320,000 dirs    9.77 MiB   <- coarsest relevant
#   level 2  Nθ=762  Nφ=1524   1.16M dirs   35.44 MiB   (outside RELEVANTLEVELS)
#   level 1  Nθ=1474 Nφ=2948   4.35M dirs  132.61 MiB   (outside RELEVANTLEVELS)
#
# Three levels spanning 1,568 to 320,000 directions is a 200x spread, which is what makes a
# size-scaling regression visible; adding levels 1-2 would only add allocator and GC noise.
const BENCHLEVELS = (8, 5, 3)

const SAMPLINGS = Dict(
    level => MLFMA.sampling(GAMMA, EPSILON, TREE; level=level)[1] for level in BENCHLEVELS
)

# Non-vacuity guard on the level sweep itself. If the samplings ever coincide, "three levels" would
# be three copies of one measurement -- passing benchmarks that compare a size against itself.
length(unique(size.(values(SAMPLINGS)))) == length(BENCHLEVELS) || error(
    "benchmark fixture: the levels $BENCHLEVELS do not have distinct sampling sizes " *
    "($(sort([size(s) for s in values(SAMPLINGS)]))), so the level sweep measures one size " *
    "repeatedly instead of a range.",
)
issubset(BENCHLEVELS, RELEVANTLEVELS) || error(
    "benchmark fixture: $BENCHLEVELS is not inside the relevant levels $RELEVANTLEVELS, so at " *
    "least one benchmarked level never carries a far field in production.",
)

# Per-level fixture ###############################################################################
#
# Everything a timed expression reads is built here, so no case measures its own setup: the sampling,
# its direction vectors, a real box center from that level, a zeroed far field, a preallocated phase
# array, and the mirror index map.

const DIRECTIONS = Dict(
    level => MLFMA.cartesiansamplepoints(SAMPLINGS[level])[1] for level in BENCHLEVELS
)

# A real node center from the level in question, not an invented point -- `dotu(k̂, center)` grows with
# |center|, and an origin-centered box would make the phase identically 1.
const CENTERS = Dict(
    level => H2Trees.center(TREE, H2Trees.LevelIterator(TREE, level)[begin]) for
    level in BENCHLEVELS
)

const FARFIELDS = Dict(
    level => FARFIELDTYPE(GAMMA, SAMPLINGS[level], Val(:zero)) for level in BENCHLEVELS
)
const PHASEBUFFERS = Dict(
    level => Array{typeof(GAMMA)}(undef, size(SAMPLINGS[level])) for level in BENCHLEVELS
)
const MIRRORINDICES = Dict(
    level => MLFMA.mirrorindices(SAMPLINGS[level]) for level in BENCHLEVELS
)

# `seconds` per case: the coarsest level is ~200x the finest, so a flat budget would spend the same
# wall clock on 20 ms calls as on 30 µs ones and starve the sample count at the coarse end.
_budget(level) = level == 3 ? 2 : 1

# FarField ########################################################################################

for level in BENCHLEVELS
    sampling = SAMPLINGS[level]
    farfield = FARFIELDS[level]
    seconds = _budget(level)

    SUITE["farfield"]["construct zero"]["level $level"] = @benchmarkable(
        $FARFIELDTYPE($GAMMA, $sampling, Val(:zero)), seconds = seconds, samples = 50,
    )
    SUITE["farfield"]["construct undef"]["level $level"] = @benchmarkable(
        $FARFIELDTYPE($GAMMA, $sampling), seconds = seconds, samples = 50,
    )
    SUITE["farfield"]["mirror"]["level $level"] = @benchmarkable(
        MLFMA.mirror($farfield, $sampling), seconds = seconds, samples = 50
    )

    # Accessors are `getindex` on an `SVector` and should be free. Benchmarked as a canary rather
    # than a hot path: if this ever stops being a couple of nanoseconds, `Fθ`/`Fφ` have lost their
    # concrete return type and gone dynamic, which is exactly the kind of regression that stays
    # invisible in a passing test suite.
    SUITE["farfield"]["accessors"]["level $level"] = @benchmarkable(
        MLFMA.Fθ($farfield)[1] + MLFMA.Fφ($farfield)[1], seconds = 1, samples = 50
    )
end

# Phase factors ###################################################################################
#
# The box-centering phase, per direction. `phasefactor` allocates its result; `phasefactor!` fills a
# caller-provided array, which is what the production paths use.

for level in BENCHLEVELS
    directions = DIRECTIONS[level]
    center = CENTERS[level]
    buffer = PHASEBUFFERS[level]
    seconds = _budget(level)

    SUITE["phasefactor"]["allocate radiation"]["level $level"] = @benchmarkable(
        MLFMA.phasefactor(
            $GAMMA, $directions, $center, MLFMA.AbstractRadiationFieldCollection
        ),
        seconds = seconds,
        samples = 50,
    )
    SUITE["phasefactor"]["allocate receive"]["level $level"] = @benchmarkable(
        MLFMA.phasefactor(
            $GAMMA, $directions, $center, MLFMA.AbstractReceiveFieldCollection
        ),
        seconds = seconds,
        samples = 50,
    )
    # In-place, radiation only. The receive convention differs by the sign of `k` alone (`_phase_k`
    # in `src/phase/phasefactor.jl`), so a second in-place case would measure the same loop.
    SUITE["phasefactor"]["in-place radiation"]["level $level"] = @benchmarkable(
        MLFMA.phasefactor!(
            $buffer, $GAMMA, $directions, $center, MLFMA.AbstractRadiationFieldCollection
        ),
        seconds = seconds,
        samples = 50,
    )
end

# Field collections ###############################################################################
#
# Collections are LEAF objects: one per leaf box, shaped (directions x basis functions in that leaf).
# So the axis that varies here is the leaf's basis-function count at the leaf sampling -- NOT the
# tree level. Sweeping level would mean claiming collections exist at level 3, which they do not,
# and a level-3-shaped collection would be 0.95 GiB per component against 2.4 MiB for a real one.
#
# `NBASIS` brackets `MLFMA.MINVALUESRECOM` (100), the subdivision threshold that sets how many basis
# functions a leaf actually holds, so the middle value is the realistic one and the outer two show
# how the kernel scales around it.

const NBASIS = (25, 100, 400)
const LEAFLEVEL = BENCHLEVELS[1]
const LEAFSAMPLING = SAMPLINGS[LEAFLEVEL]
const NDIRECTIONS = MLFMA.Nθ(LEAFSAMPLING) * MLFMA.Nφ(LEAFSAMPLING)

function collectionmatrix(nbasis::Int; seed=0x87654321)
    rng = Xoshiro(seed)
    return rand(rng, ComplexF64, NDIRECTIONS, nbasis)
end

# Collections are built per OPERATOR, not once: the component count and the meaning of those
# components are now carried by the far-field type (`MLFMA.farfieldtype`), and there is no
# constructor that guesses a representation from a bare `SVector` of matrices any more. The three
# operators below span what actually differs at this layer:
#
#   * `MWEFIE3D`                -> TangentialSphericalFarField,   2 components, SCALAR α
#   * `MWSingleLayer3D`         -> SphericalVectorFarField,        3 components, VECTOR α (3)
#   * `HH3DHyperSingularFDBIO`  -> PairedSphericalVectorFarField,  6 components, VECTOR α (6)
#
# The α distinction is the reason the last two are here rather than being redundant with the first:
# `MLFMA.α` is a vector for them, H2Factory's leaf disaggregation passes it straight into the
# result-side multiply, and that per-component-weighted path is a different kernel from the scalar
# one. It also went unmeasured AND untested until it was found broken, so it is measured now.
const COLLECTIONOPERATORS = (
    (name="efie", operator=MLFMA.MWEFIE3D(; wavenumber=WAVENUMBER)),
    (name="singlelayer", operator=Maxwell3D.singlelayer(; wavenumber=WAVENUMBER)),
    (name="hh_hypersingular", operator=Helmholtz3D.hypersingular(; wavenumber=WAVENUMBER)),
)

# Distinct seed per component so no two components share data -- identical components would let a
# component-index mix-up measure (and time) the same memory twice.
function collectioncomponents(nbasis::Int, ncomponents::Int)
    return SVector{ncomponents,Matrix{ComplexF64}}([
        collectionmatrix(nbasis; seed=0x87654321 + 0x01010101 * UInt32(component)) for
        component in 1:ncomponents
    ])
end

# Every entry independently random, and a distinct seed per component. An earlier version filled each
# component by broadcasting ONE column across the whole grid, which is cheap but leaves the field
# constant along θ -- harmless for timings, but it is exactly the kind of accidental structure a
# future kernel could exploit (or a bug could hide in) without anyone noticing the fixture was the
# reason.
# `MLFMA.receivefields` (the allocating wrapper) no longer exists: `LazyReceiveFieldCollection` is a
# scratch-only leaf collection now, and the wrapper had no production caller once the leaf channel
# supplied the destination. Kept here purely as the "before" side of the allocating/in-place pair
# below -- it is what the lazy path used to have to do on every multiply.
function allocatingreceivefields(
    src::F, mirrorindices, op
) where {D,T,F<:MLFMA.AbstractFarField{D,T}}
    dest = MLFMA.basefarfieldtype(F){D,eltype(T)}(size(src[1]))
    return MLFMA.receivefields!(dest, src, mirrorindices, op)
end

function randomfarfield(FarFieldType, sampling; seed=0x5A5A0000)
    farfield = FarFieldType(ComplexF64, sampling, Val(:zero))
    for component in eachindex(farfield)
        rng = Xoshiro(seed + 0x00010001 * UInt32(component))
        farfield[component] .= rand(rng, ComplexF64, size(farfield[component]))
    end
    return farfield
end

filledfarfield(FarFieldType) = randomfarfield(FarFieldType, LEAFSAMPLING)

# THE MATERIALIZED RECEIVE COLLECTION the lazy one stands in for -- built from the SAME
# `receivecouplingrule` the lazy path walks, so the `receive mul!` lane is apples-to-apples with the
# two lazy lanes rather than merely the same SHAPE of BLAS call on unrelated random data.
#
# Derivation. The read side computes
#     result[basis] = Σ_mappings coeff * Σ_s C[dest][s,basis] * ff[src][idx[s]]
# and a collection `mul!` computes `result[basis] = Σ_i Σ_t R[i][t,basis] * ff[i][t]`. Substituting
# `t = idx[s]` (legal: `idx` is a permutation AND an involution, so `s = idx[t]`) turns the first into
#     result[basis] = Σ_mappings coeff * Σ_t C[dest][idx[t],basis] * ff[src][t]
# and matching component `i = src` gives
#     R[src][t,basis] = coeff * C[dest][idx[t],basis].
# Note `src` indexes R while `dest` indexes C -- the same end-swap the write-through direction needs,
# and the reason this is derived here rather than eyeballed: on every rule shipped today (each either
# diagonal or a symmetric two-way swap) the un-swapped version happens to agree, so a mistake would be
# invisible. Verified against the fused `mul!(result, lazy, ff)` -- which existed at the time and has
# since been deleted -- for all five operator/representation pairs, scalar and vector α. The live
# check is now in `test/FarField/rules.jl`, where the same construction is the ground truth for both
# `applyleafmul!` arities; if the derivation were wrong, that testset fails.
function materializedreceive(operator, FarFieldType, radiation, mirrorindices)
    ncomponents = MLFMA.fielddim(FarFieldType)
    components = [Matrix{ComplexF64}(undef, size(radiation[1])) for _ in 1:ncomponents]
    for mapping in MLFMA.receivecouplingrule(operator, FarFieldType)
        dest = MLFMA.destcomponent(mapping)
        src = MLFMA.srccomponent(mapping)
        coeff = MLFMA.coefficient(mapping)
        for basis in axes(radiation[dest], 2), t in eachindex(mirrorindices)
            components[src][t, basis] = coeff * radiation[dest][mirrorindices[t], basis]
        end
    end
    return MLFMA.receivecollectiontype(FarFieldType)(
        SVector{ncomponents,Matrix{ComplexF64}}(components)
    )
end

for case in COLLECTIONOPERATORS
    operator = case.operator
    FarFieldType = MLFMA.farfieldtype(operator)
    ncomponents = MLFMA.fielddim(FarFieldType)
    α = MLFMA.α(operator)
    farfield = filledfarfield(FarFieldType)

    for nbasis in NBASIS
        components = collectioncomponents(nbasis, ncomponents)
        radiation = MLFMA.radiationcollectiontype(FarFieldType)(components)
        # The production shape: the receive side derived from the radiation data rather than
        # assembled again.
        lazy = MLFMA.LazyReceiveFieldCollection(
            operator, radiation, MLFMA.mirrorindices(LEAFSAMPLING)
        )
        # ...and the collection that derivation would produce if it were materialized once and stored.
        # `mul!(result, receive, farfield)` computes exactly what the two lazy lanes compute, so the
        # three are directly comparable: same numbers, three ways of getting them.
        receive = materializedreceive(
            operator, FarFieldType, radiation, MLFMA.mirrorindices(LEAFSAMPLING)
        )
        result = Vector{ComplexF64}(undef, nbasis)
        coefficients = collectionmatrix(nbasis; seed=0x33339999)[1:nbasis, 1]
        group = "$(case.name) nbasis $nbasis"

        SUITE["fieldcollections"]["radiation mul!"][group] = @benchmarkable(
            LinearAlgebra.mul!($result, $radiation, $farfield), seconds = 1, samples = 50
        )
        SUITE["fieldcollections"]["receive mul!"][group] = @benchmarkable(
            LinearAlgebra.mul!($result, $receive, $farfield), seconds = 1, samples = 50
        )
        SUITE["fieldcollections"]["radiation conjmul!"][group] = @benchmarkable(
            H2Factory.conjmul!($result, $radiation, $farfield), seconds = 1, samples = 50
        )

        # THE LAZY LANE, and the only implementation there is: `H2Factory.applyleafmul!` is what the
        # traversal calls, `LazyReceiveFieldCollection` implements nothing else, and it mirrors the
        # incoming moment into the borrowed scratch before handing the radiation collection one
        # contiguous `gemv` per component.
        #
        # Read against `receive mul!` above, which is the SAME numbers via a materialized
        # `ReceiveFieldCollection` (built from the operator's own rule -- see `materializedreceive`)
        # and therefore the ceiling: no mirroring at multiply time at all. The gap between the two is
        # what the lazy form costs; `receivefields!` in the group below is the part of that gap the
        # mirroring accounts for, and it does not account for all of it.
        #
        # The scratch is allocated ONCE, outside the benchmarkable, exactly as a workspace's leaf
        # channel allocates it once per workspace -- timing its construction here would measure the
        # cost the leaf channel exists to remove.
        leafscratch = FarFieldType(ComplexF64, LEAFSAMPLING, Val(:zero))
        SUITE["fieldcollections"]["lazy receive mul!"][group] = @benchmarkable(
            H2Factory.applyleafmul!($leafscratch, $result, $lazy, $farfield, true, false),
            seconds = 1,
            samples = 50,
        )

        # Per-component α: the path leaf disaggregation actually takes for a vector-α operator.
        # Measured for every operator so the scalar and weighted paths stay comparable.
        # The scratch path needs no vector-α kernel of its own -- it forwards to H2Factory's own
        # per-component collection `mul!` -- so this lane measures whether that is also true of the
        # COST, or whether the weighted collection path is slower than the scalar one.
        SUITE["fieldcollections"]["lazy receive mul! (α weights)"][group] = @benchmarkable(
            H2Factory.applyleafmul!($leafscratch, $result, $lazy, $farfield, $α, false),
            seconds = 1,
            samples = 50,
        )

        # WRITE-THROUGH: the aggregation direction, `moment = f(collection, coefficients)`, reached
        # whenever a transposed/adjoint map aggregates the test side. Its own destination moment so
        # it cannot clobber the read-side fixture above.
        writethroughmoment = FarFieldType(ComplexF64, LEAFSAMPLING, Val(:zero))
        # ...and the scratch lane for the same direction: one `gemv` per component into the scratch,
        # then a gather. Its own scratch, so the read-side lanes' fixture stays untouched.
        writethroughscratch = FarFieldType(ComplexF64, LEAFSAMPLING, Val(:zero))
        SUITE["fieldcollections"]["lazy receive write-through mul!"][group] = @benchmarkable(
            H2Factory.applyleafmul!(
                $writethroughscratch, $writethroughmoment, $lazy, $coefficients
            ),
            seconds = 1,
            samples = 50,
        )
    end
end

# The mirror-and-scale step, both ways round. It depends on the operator and the sampling, not on the
# basis count, so it is measured per level, where its cost actually varies.
#
# TWO LANES, and the distinction is the whole point of the leaf channel:
#
#   * `receivefields (allocating)` -- mirrors into a FRESHLY ALLOCATED far field. No longer in `src/`
#                         at all (see `allocatingreceivefields` above); this is what the lazy path had
#                         to do on every multiply before a workspace owned leaf scratch.
#   * `receivefields!` -- the IN-PLACE step the scratch path actually runs inside
#                         `H2Factory.applyleafmul!`, writing into a preallocated buffer.
#
# The gap between them is the allocation the leaf channel removes; `receivefields!` alone is the
# honest overhead to add to a materialized `mul!` when reading the three receive lanes above. Swept
# over every operator in `COLLECTIONOPERATORS`, not just EFIE: the rule length and component count
# differ per representation (1, 2, 3 and 6 components appear), and that is precisely what this step
# loops over.
for case in COLLECTIONOPERATORS, level in BENCHLEVELS
    operator = case.operator
    FarFieldType = MLFMA.farfieldtype(operator)
    sampling = SAMPLINGS[level]
    indices = MIRRORINDICES[level]
    source = randomfarfield(FarFieldType, sampling)
    # Allocated once, outside the benchmarkable, exactly as a workspace's leaf channel allocates it
    # once per workspace -- timing its construction would measure the cost this lane exists to show
    # has been removed.
    scratch = FarFieldType(ComplexF64, sampling, Val(:zero))
    group = "$(case.name) level $level"

    SUITE["fieldcollections"]["receivefields (allocating)"][group] = @benchmarkable(
        allocatingreceivefields($source, $indices, $operator),
        seconds = _budget(level),
        samples = 50,
    )
    SUITE["fieldcollections"]["receivefields!"][group] = @benchmarkable(
        MLFMA.receivefields!($scratch, $source, $indices, $operator),
        seconds = _budget(level),
        samples = 50,
    )
end

# Representative workflow #########################################################################
#
# What aggregating one leaf actually costs end to end: coefficients -> leaf moment -> box-centered
# moment. Catches regressions that only appear when the steps run together (an extra copy at a
# boundary, a type that goes dynamic only in composition) and that no isolated case above would show.

function leafaggregation!(
    farfield, phasebuffer, collection, coefficients, directions, center
)
    LinearAlgebra.mul!(farfield, collection, coefficients)
    MLFMA.phasefactor!(
        phasebuffer, GAMMA, directions, center, MLFMA.AbstractRadiationFieldCollection
    )
    return H2Factory.elementwisemul!(farfield, farfield, phasebuffer)
end

# EFIE only, and at the realistic basis count: this group is about composition cost, not about
# sweeping the representation axis (the `fieldcollections` group above does that).
let nbasis = 100
    operator = OPERATOR
    farfield = filledfarfield(FARFIELDTYPE)
    collection = MLFMA.radiationcollectiontype(FARFIELDTYPE)(
        collectioncomponents(nbasis, MLFMA.fielddim(FARFIELDTYPE))
    )
    coefficients = collectionmatrix(nbasis; seed=0x33339999)[1:nbasis, 1]

    SUITE["workflows"]["aggregate one leaf"]["nbasis $nbasis"] = @benchmarkable(
        leafaggregation!(
            $farfield,
            $(PHASEBUFFERS[LEAFLEVEL]),
            $collection,
            $coefficients,
            $(DIRECTIONS[LEAFLEVEL]),
            $(CENTERS[LEAFLEVEL]),
        ),
        seconds = 2,
        samples = 50,
    )
end

# Translation #####################################################################################
#
# `src/translation/` -- the addition-theorem operator itself, and the special functions under it.
#
# WHAT THIS IS AND IS NOT THE 70 %. `#120` measured I2O translation at ~70 % of the far matvec, but
# that share is H2Factory APPLYING the stored operators, which is not MLFMA code. What MLFMA owns is
# BUILDING them: one `translationoperator!` call per translating direction per level, at construction.
# So this group belongs on the construction side of the header's split, and its unit is what it is --
# a cost paid once per assembly, not once per iteration. It is grouped here rather than under
# `"construction"` because it shares that group's units but none of its fixture: no mesh, no gmsh, no
# BEAST, no plans.
#
# The header used to list translation among the things deliberately not measured, on the grounds that
# it needs workspace and scheduling machinery. That is true of the APPLY and false of the kernel:
# `translationoperator!` takes an array, a γ, a direction grid, a transfer vector and a cutoff, and
# nothing else.
#
# COST MODEL, and |R| IS AN AXIS IN 2-D AND NOT IN 3-D -- measured, not assumed. The 3-D operator is
# `ndirections x L`, a Legendre three-term recurrence per direction up to the level's cutoff, plus ONE
# Hankel evaluation of length L for the whole grid; |R| enters only that single Hankel argument. Both a
# near (2-edge) and a far (diagonal) transfer vector were run at every level and agreed to three
# digits -- 150.238 ms against 150.356 ms at level 3, 3.953 against 3.953 at level 5 -- so only ONE 3-D
# vector is kept. A second case measuring the same number is what the fixture guard at the top of this
# file exists to prevent.
#
# The 2-D operator genuinely differs, because its truncation is `min(cutoff, ceil(κ|R|) + margin)` --
# a CORRECTNESS requirement rather than an optimization, since the series is asymptotic and diverges
# past the optimal order (see `translation.jl`). So a SHORT transfer vector is CHEAPER: at order 55,
# |R| = 1λ clamps the series well below the grid's cutoff and costs 70.8 µs, against 179.3 µs at
# |R| = 4λ. That is the cutoff rule showing up as runtime, and it is why |R| is swept here.
#
# THE HANKELS ARE MEASURED SEPARATELY, and what that separation shows is NOT what it was added to
# show. `translationoperator!` calls the ALLOCATING `collectsphericalhankelsecond` once per
# translating direction, which reads like a per-pair cost worth removing. It is not: measured, the
# allocating and in-place forms take the SAME time (5.360 against 5.390 µs at order 13, 31.890 against
# 32.121 at order 55) because the special-function evaluation dominates the allocation completely. The
# gap is MEMORY only -- 288 B to 3.2 KiB per call -- so the two cases are kept as an allocation pin
# rather than as a speed comparison, and nobody should spend an afternoon threading a buffer through.

# Transfer vectors are integer multiples of the box EDGE (`2 * halfsize`) on a `TwoNTree`. With one
# buffer box the nearest translating pair sits two edges apart, so `NEARTRANSFER` is the shortest
# vector this kernel is ever called with at a level, and `FARTRANSFER` a representative diagonal one.
# Both are built per level, because the edge length is what changes down the tree.
_halfsizeat(level) = MINHALFSIZE * 2.0^(NLEVELS - level)

# `far` is unused by the suite and kept deliberately: it is the vector the header's independence
# measurement was taken against, so the claim can be re-checked by pointing the case set at it.
const TRANSFERVECTORS = Dict(
    level => (
        near=2 * (2 * _halfsizeat(level)) * SVector(1.0, 0.0, 0.0),
        far=(2 * _halfsizeat(level)) * SVector(4.0, 3.0, 2.0),
    ) for level in BENCHLEVELS
)

const TRANSLATIONBUFFERS = Dict(
    level => Array{typeof(GAMMA)}(undef, size(SAMPLINGS[level])) for level in BENCHLEVELS
)

for level in BENCHLEVELS
    L = MLFMA.cutoffparameter(SAMPLINGS[level])
    directions = DIRECTIONS[level]
    buffer = TRANSLATIONBUFFERS[level]
    seconds = _budget(level)

    # One vector, not both: see this section's header for the measurement that made the second one
    # redundant.
    R = TRANSFERVECTORS[level].near
    SUITE["translation"]["operator 3-D"]["level $level"] = @benchmarkable(
        MLFMA.translationoperator!($buffer, $GAMMA, $directions, $R, $L),
        seconds = seconds,
        samples = 50,
    )

    # The per-pair Hankel setup, allocating and in-place. `translationoperator!` currently calls the
    # allocating form, so the second case is what the first one could cost.
    x = abs(GAMMA) * norm(TRANSFERVECTORS[level].near)
    hankelbuffer = Vector{ComplexF64}(undef, L + 1)
    SUITE["translation"]["spherical hankels (allocating)"]["level $level"] = @benchmarkable(
        MLFMA.collectsphericalhankelsecond($L, $x), seconds = 1, samples = 50,
    )
    SUITE["translation"]["spherical hankels!"]["level $level"] = @benchmarkable(
        MLFMA.collectsphericalhankelsecond!($hankelbuffer, $L, $x),
        seconds = 1,
        samples = 50,
    )
end

# The 2-D operator, on its own fixture ############################################################
#
# A `CircularSampling` needs only its `Nφ`, so this costs a constructor call and no tree -- which is
# why the 2-D kernel can sit in this file at all while the rest of it is a 3-D EFIE fixture. The three
# orders are a λ/4 leaf grid (L = 13, `Nφ` = 28, the size `#138` calibrated against) and its two
# coarser parents.
#
# `|R|` IS SWEPT HERE AND NOT IN 3-D, because 2-D truncation depends on it -- see this section's header.

const CIRCULARORDERS = (13, 27, 55)
const CIRCULARSAMPLINGS = Dict(
    L => MLFMA.CircularSampling(MLFMA.NφCircular(L)) for L in CIRCULARORDERS
)
const CIRCULARDIRECTIONS = Dict(
    L => MLFMA.cartesiansamplepoints(CIRCULARSAMPLINGS[L])[1] for L in CIRCULARORDERS
)
const CIRCULARBUFFERS = Dict(
    L => Array{typeof(GAMMA)}(undef, size(CIRCULARSAMPLINGS[L])) for L in CIRCULARORDERS
)

# Non-vacuity, same shape as the 3-D level guard: three orders that produced one grid size would be
# one measurement repeated.
length(unique(size(CIRCULARSAMPLINGS[L]) for L in CIRCULARORDERS)) ==
length(CIRCULARORDERS) || error(
    "benchmark fixture: the circular orders $CIRCULARORDERS do not have distinct grid sizes, " *
    "so the 2-D sweep measures one size repeatedly.",
)

for L in CIRCULARORDERS
    directions = CIRCULARDIRECTIONS[L]
    buffer = CIRCULARBUFFERS[L]

    # In wavelengths: the nearest translating pair at a λ/4 leaf is ~1λ apart, and coarser levels
    # translate proportionally further. Both are inside the regime where the cutoff binds.
    for (name, distance) in (("near", 1.0), ("far", 4.0))
        R = distance * WAVELENGTH * SVector(1.0, 0.0)
        SUITE["translation"]["operator 2-D ($name)"]["order $L"] = @benchmarkable(
            MLFMA.translationoperator!($buffer, $GAMMA, $directions, $R, $L),
            seconds = 1,
            samples = 50,
        )
    end

    x = abs(GAMMA) * WAVELENGTH
    cylbuffer = Vector{ComplexF64}(undef, L + 1)
    SUITE["translation"]["cylindrical hankels (allocating)"]["order $L"] = @benchmarkable(
        MLFMA.collectcylindricalhankelsecond($L, $x), seconds = 1, samples = 50,
    )
    SUITE["translation"]["cylindrical hankels!"]["order $L"] = @benchmarkable(
        MLFMA.collectcylindricalhankelsecond!($cylbuffer, $L, $x),
        seconds = 1,
        samples = 50,
    )
end

# Compressed leaf storage #########################################################################
#
# The leaf multiply under each of the four `AbstractLeafStorage` policies, on ONE leaf, at the same
# grid, over a sweep of leaf occupancy. `SUITE["fieldcollections"]` already measures the sampled
# kernels; this group is what lets the compressed ones be compared against them and against each
# other, which nothing in this suite could do before.
#
# DIRECTION FIRST, because the neighbouring group measures the OTHER one and the two are not
# comparable. `SUITE["fieldcollections"]["radiation mul!"]` is `mul!(result, collection, farfield)` --
# the SCATTER, writing `nbasis` values. Everything named "aggregate" here is
# `mul!(moment, collection, x)`, which writes the whole `1568 x 2` moment. The aggregate is ~4x the
# scatter at these occupancies for that reason alone, and reading one against the other -- which the
# first draft of this section did -- looks like a 5x regression that is not there.
#
# WHAT IS MEASURED IS THE PER-LEAF PATH, AND PRODUCTION RUNS THE BATCHED ONE. Stated first because it
# bounds every number here. `H2Factory.aggregateleafchunk!` synthesises a whole chunk against one
# shared matrix, and for all three compressed backends that is mandatory rather than an optimization:
# unbatched SHE measured 1.63x SLOWER than the sampled path it replaces (#125), and batching is worth
# 0.28x-0.72x in 2-D (#138). The batched form is not here because it needs a workspace, a tree and
# `momentcollection`/`moment` as callables -- the machinery this file's header keeps out. So read
# these as the kernel's shape, not as what a solve pays, and scale by the factors above.
#
# OCCUPANCY IS THE AXIS, because it is the one that decided #138. Compression is a fixed win in
# MEMORY and a function of basis-functions-per-leaf in TIME: the coefficient GEMV shrinks with the
# representation while the synthesis does not, so a sparsely populated leaf pays the synthesis over
# too few columns. 2-D measured 1.67x slower at 8 functions per leaf and 0.89x at 1024. An ordinary
# `meshcircle` sits at ~8. A single occupancy would hide that entirely.
#
# THE ORDERS ARE THE CALIBRATED ONES FOR THIS GRID, not round numbers. `P = 9` is what the SHE rule
# picks for this fixture's leaf level at this tolerance -- the same value the `(1568 x 300)` synthesis
# matrix in this file's `fieldcollections` note refers to -- and `M = 9` is what `#138`'s table
# records for a λ/4 leaf. Benchmarking a compressed representation at its UNCOMPRESSED order would
# measure a kernel no map ever runs.
#
# RANDOM COEFFICIENTS, NEVER ZEROS. A zero-filled operand changes which arithmetic the hardware
# performs and has produced a phantom asymmetry in this package family before.
#
# ------------------------------------------------------------------------------------------------
# WHAT THE FIRST RUNS SAID, recorded so the group has a baseline -- and one caveat that has to come
# first, because it nearly put a wrong finding in this file.
#
# THIS IS A SHARED MACHINE AND SINGLE-RUN DIFFERENCES UNDER ~20 % ARE NOISE. Other users' jobs were at
# ~58 % of a core while these were taken. One run had VSH aggregation at 161-169 µs and SHE at
# 116-156, which reads as a clear ordering and was briefly written down here as an unexplained defect;
# three consecutive reps then put VSH at 120-133 against SHE at 142-161, i.e. the other way round. The
# claims below are the ones that survived repetition. Re-run before believing any comparison here that
# is closer than a factor of ~1.3.
#
# UNBATCHED SHE IS SLOWER THAN SAMPLED, and that is the known result rather than a regression:
# 109-161 µs against 78-87 for the sampled aggregate, i.e. ~1.4-1.9x, against the 1.63x #125 recorded
# by a different route. It is exactly why the batched chunk path exists, and it is why those two cases
# sit next to each other.
#
# VSH BEATS SHE IN BOTH DIRECTIONS, and by more than the noise floor in the scatter: 41-54 µs against
# 151-168 for the scatter (~3x), 120-133 against 142-161 for the aggregate (~1.2x, repeated three
# times). Both follow from the same fact and this is the first measurement of it -- `X`/`Psi` are
# already tangential, so the VSH kernel carries no Cartesian intermediate and no frame projection,
# which `vshcollection.jl` states as the representation's third advantage and which nothing had
# checked. The gap is larger in the scatter because that is where SHE's `_backprojectinto!` writes a
# full `(NΩ x 3)` intermediate.

const LEAFORDER3D = 9
const LEAFORDER2D = 9
# Leaf occupancy: an ordinary mesh sits at the low end, and the high end is past 2-D's measured
# crossover, so the sweep brackets the point where compression stops costing time.
const LEAFOCCUPANCIES = (8, 32, 128)

_randomcoefficients(rows, nbasis, seed) = rand(Xoshiro(seed), ComplexF64, rows, nbasis)

# --- 3-D, against the same leaf sampling the sampled kernels use ---------------------------------

const SHESCRATCH = MLFMA.shescratch(LEAFSAMPLING, LEAFORDER3D)
const VSHSCRATCH = MLFMA.vshscratch(LEAFSAMPLING, LEAFORDER3D)

for nbasis in LEAFOCCUPANCIES
    moment = FARFIELDTYPE(GAMMA, LEAFSAMPLING, Val(:zero))
    x = collectionmatrix(nbasis; seed=0x5150ABCD)[1, :]
    y = zeros(ComplexF64, nbasis)

    shestorage = MLFMA.SphericalHarmonics.SphericalHarmonicStorage(
        LEAFORDER3D,
        CENTERS[LEAFLEVEL],
        _randomcoefficients(
            3 * MLFMA.SphericalHarmonics.numcoefficients(LEAFORDER3D), nbasis, 0x11AA22BB
        ),
    )
    shecollection = MLFMA.SphericalHarmonicRadiationCollection(shestorage)

    vshstorage = MLFMA.VectorSphericalHarmonicStorage(
        LEAFORDER3D,
        CENTERS[LEAFLEVEL],
        _randomcoefficients(
            2 * MLFMA.SphericalHarmonics.numvshcoefficients(LEAFORDER3D), nbasis, 0x33CC44DD
        ),
    )
    vshcollection = MLFMA.VectorSphericalHarmonicRadiationCollection(vshstorage)

    # The SAMPLED kernel at the identical grid and occupancy, so the three are comparable. The
    # `fieldcollections` group measures this too, but over its own basis counts.
    sampledcollection = MLFMA.radiationcollectiontype(FARFIELDTYPE)(
        collectioncomponents(nbasis, MLFMA.fielddim(FARFIELDTYPE))
    )

    SUITE["leafstorage"]["aggregate (moment <- x) sampled 3-D"]["nbasis $nbasis"] = @benchmarkable(
        LinearAlgebra.mul!($moment, $sampledcollection, $x), seconds = 1, samples = 50,
    )
    SUITE["leafstorage"]["aggregate (moment <- x) SHE"]["nbasis $nbasis"] = @benchmarkable(
        H2Factory.applyleafmul!($SHESCRATCH, $moment, $shecollection, $x),
        seconds = 1,
        samples = 50,
    )
    SUITE["leafstorage"]["aggregate (moment <- x) VSH"]["nbasis $nbasis"] = @benchmarkable(
        H2Factory.applyleafmul!($VSHSCRATCH, $moment, $vshcollection, $x),
        seconds = 1,
        samples = 50,
    )

    # The SCATTER arity, which a forward matvec reaches on the other side of the tree and which no
    # aggregation case exercises: it runs the coefficient matrix transposed and the synthesis reversed.
    SUITE["leafstorage"]["scatter (y <- moment) SHE"]["nbasis $nbasis"] = @benchmarkable(
        H2Factory.applyleafmul!(
            $SHESCRATCH, $y, $shecollection, $moment, $(one(ComplexF64)), false
        ),
        seconds = 1,
        samples = 50,
    )
    SUITE["leafstorage"]["scatter (y <- moment) VSH"]["nbasis $nbasis"] = @benchmarkable(
        H2Factory.applyleafmul!(
            $VSHSCRATCH, $y, $vshcollection, $moment, $(one(ComplexF64)), false
        ),
        seconds = 1,
        samples = 50,
    )
end

# --- 2-D, on the circular fixture the translation group already builds ---------------------------
#
# The 2-D single layer, whose far field is one scalar component -- so `D = 1` here against the EFIE's
# 2 above, and the two dimensions are NOT comparable case for case. They are in one group because they
# are one policy axis, not because their numbers belong side by side.

const CIRCULARLEAFORDER = CIRCULARORDERS[1]
const CIRCULARLEAFSAMPLING = CIRCULARSAMPLINGS[CIRCULARLEAFORDER]
const OPERATOR2D = Helmholtz2D.singlelayer(; wavenumber=WAVENUMBER)
const FARFIELDTYPE2D = MLFMA.farfieldtype(OPERATOR2D)
const NCOMPONENTS2D = MLFMA.fielddim(FARFIELDTYPE2D)
const CHESCRATCH = MLFMA.chescratch(
    CIRCULARLEAFSAMPLING, LEAFORDER2D; ncomponents=NCOMPONENTS2D
)

for nbasis in LEAFOCCUPANCIES
    moment2d = FARFIELDTYPE2D(GAMMA, CIRCULARLEAFSAMPLING, Val(:zero))
    x = rand(Xoshiro(0x6161FEED + UInt32(nbasis)), ComplexF64, nbasis)
    y = zeros(ComplexF64, nbasis)

    chestorage = MLFMA.CircularHarmonics.CircularHarmonicStorage(
        LEAFORDER2D,
        SVector(0.3, -0.7),
        _randomcoefficients(
            NCOMPONENTS2D * MLFMA.CircularHarmonics.numcoefficients(LEAFORDER2D),
            nbasis,
            0x55EE66FF,
        ),
    )
    checollection = MLFMA.CircularHarmonicRadiationCollection(chestorage)

    SUITE["leafstorage"]["aggregate (moment <- x) CHE 2-D"]["nbasis $nbasis"] = @benchmarkable(
        H2Factory.applyleafmul!($CHESCRATCH, $moment2d, $checollection, $x),
        seconds = 1,
        samples = 50,
    )
    SUITE["leafstorage"]["scatter (y <- moment) CHE 2-D"]["nbasis $nbasis"] = @benchmarkable(
        H2Factory.applyleafmul!(
            $CHESCRATCH, $y, $checollection, $moment2d, $(one(ComplexF64)), false
        ),
        seconds = 1,
        samples = 50,
    )
end

# Sampling and operator accessors #################################################################
#
# `src/sampling/` and `src/operators/`, the two remaining layers with no coverage. Both are
# CONSTRUCTION-side, so they belong with the group below rather than with the per-matvec kernels; they
# are kept separate from it because they need no mesh, no gmsh and no BEAST assembly, only the tree the
# rest of this file already built.
#
# WHAT IS ALREADY COVERED AND IS NOT REPEATED HERE. `SUITE["construction"]["leaf samplings"]` measures
# `leaffarfieldsamplings`, which is `src/assembly/beast/`'s per-leaf recipe, and
# `SUITE["construction"]["mirror indices by level"]` measures the 3-D `mirrorindices`. What was left
# untimed in `src/sampling/` is the grid constructor and `cartesiansamplepoints`.
#
# ALL OF THE COST IS IN `cartesiansamplepoints`, AND NONE OF IT IS IN THE GRID -- measured, and it is
# why the two cases below are shaped differently. A `SphericalSampling` is a SIXTEEN-BYTE descriptor
# holding two integers and nothing else, so `MLFMA.sampling` is the rate rule plus a two-field
# struct: 140 ns and zero allocations, IDENTICAL at every level. It is therefore ONE case rather than
# a level sweep -- three copies of one number is what this file's fixture guards exist to prevent,
# and the first draft of this section had exactly that. Same for the 2-D constructor at 30 ns.
#
# The materialisation is where the work is, and in ISOLATION it is linear in the grid: 556 / 58 / 19
# microseconds at levels 3 / 5 / 8, against 320,000 / 28,800 / 1,568 directions -- 1.7 to 2.0 ns per
# direction once the small-grid overhead is out.
#
# INSIDE THE SUITE THE LEVEL-5 CASE READS 7x HIGHER, ~415 microseconds, and that is REPRODUCIBLE
# rather than noise -- it was briefly written off as the shared-machine variance the `leafstorage`
# group warns about, and it is not that. It is the level-3 case immediately before it: 9.84 MiB of
# `SVector{3,Float64}` per sample, whose collection lands on whoever runs next. Recorded rather than
# fixed, because the fix is either to reorder cases (fragile, and it would only move the cost onto a
# different neighbour) or to give the case its own `gcsample`, which changes what every other case in
# the file measures. READ THIS CASE AGAINST ITSELF ACROSS RUNS, not against its neighbours -- which
# is what a regression suite is for anyway.
#
# THE LEVEL TRANSFERS IN `src/phase/` ARE DELIBERATELY ABSENT, and this is the one place in the file
# where that is a judgement rather than a boundary. `interpolationphasecollections` is construction
# and would fit, but it takes an interpolator/anterpolator PAIR, which is `benchmark/interpolation/`'s
# fixture and not this one; and the transfer APPLY needs the workspace machinery the header keeps
# out. Building a second interpolator fixture here to measure one constructor would duplicate that
# file's setup for less than it costs.

# One level, not a sweep: the constructor is level-independent -- see this section's header.
SUITE["sampling"]["build 3-D grid"]["level $(BENCHLEVELS[1])"] = @benchmarkable(
    MLFMA.sampling($GAMMA, $EPSILON, $TREE; level=($(BENCHLEVELS[1]))),
    seconds = 1,
    samples = 50,
)

for level in BENCHLEVELS
    sampling = SAMPLINGS[level]
    seconds = _budget(level)
    SUITE["sampling"]["cartesian points 3-D"]["level $level"] = @benchmarkable(
        MLFMA.cartesiansamplepoints($sampling), seconds = seconds, samples = 50,
    )
end

# One order, not a sweep, for the same reason as the 3-D constructor above.
SUITE["sampling"]["build 2-D grid"]["order $(CIRCULARORDERS[1])"] = @benchmarkable(
    MLFMA.CircularSampling($(MLFMA.NφCircular(CIRCULARORDERS[1]))),
    seconds = 1,
    samples = 50,
)

for L in CIRCULARORDERS
    sampling = CIRCULARSAMPLINGS[L]
    SUITE["sampling"]["cartesian points 2-D"]["order $L"] = @benchmarkable(
        MLFMA.cartesiansamplepoints($sampling), seconds = 1, samples = 50,
    )
    # The 2-D counterpart of `SUITE["construction"]["mirror indices by level"]`, which covers 3-D only.
    SUITE["sampling"]["mirror indices 2-D"]["order $L"] = @benchmarkable(
        MLFMA.mirrorindices($sampling), seconds = 1, samples = 50,
    )
end

# --- operator accessors ---------------------------------------------------------------------------
#
# The same shape as `SUITE["farfield"]["accessors"]`, and here for the same reason: these are on every
# construction path, they are expected to fold away to a constant, and one of them has been a real
# defect before -- `farfieldstorerule` once resolved through a `Union` mixing `Type{X}` and instance
# arms, which answered the two call forms by different routes and died with a `StackOverflow` on the
# documented one. A case that suddenly reports nanoseconds instead of picoseconds is that class of
# regression becoming visible.
#
# A TRIPWIRE, NOT A MEASUREMENT. All ten cases sit at 20-30 ns and zero allocations, which is
# BenchmarkTools' floor for a non-inlined call rather than the cost of the accessor -- they fold to a
# constant, which is the whole expectation. What the group is for is the case that stops doing so: an
# accessor that starts allocating, or reports microseconds, has become dynamic, and on this code path
# every map construction then pays for it once per leaf.
#
# Both dimensions, because the 2-D operators reach different methods of every one of these.

for (name, op) in (("efie", OPERATOR), ("helmholtz2d singlelayer", OPERATOR2D))
    SUITE["operators"]["farfieldtype"][name] = @benchmarkable(
        MLFMA.farfieldtype($op), seconds = 1, samples = 50,
    )
    SUITE["operators"]["farfieldstorerule"][name] = @benchmarkable(
        MLFMA.farfieldstorerule($op), seconds = 1, samples = 50,
    )
    SUITE["operators"]["alpha"][name] = @benchmarkable(
        MLFMA.α($op), seconds = 1, samples = 50,
    )
    SUITE["operators"]["radiationoperator"][name] = @benchmarkable(
        MLFMA.radiationoperator($op), seconds = 1, samples = 50,
    )
    SUITE["operators"]["receiveoperator"][name] = @benchmarkable(
        MLFMA.receiveoperator($op), seconds = 1, samples = 50,
    )
end

# Construction ####################################################################################
#
# BUILDING the per-leaf collections, as opposed to multiplying by them. Everything above this line
# runs once per matvec; everything below runs once per operator assembly. See the header for why
# they are separate groups.
#
# The five cases are the five layers `getfieldcollectionsdict` goes through:
#
#   leaf samplings            per-leaf recipe: level lookup, per-level direction points, centers, ids
#   radiation collections     the real BEAST quadrature -- the expensive one, swept over store rules
#   lazy receive collections  wrapping radiation collections; should be nearly free
#   frame cache               the per-sampling trig tables the storer reads
#   mirror indices by level   the per-level mirror map the lazy side shares
#
# `radiation collections` carries one entry per far-field store rule; the other four are EFIE-only.
# See the sweep table further down for which operator stands for which rule, and for the two things
# its rows can and cannot be compared against.
#
# THREE OF THESE EXIST TO PIN A SHARING PROPERTY, not just a duration. `frame cache` and
# `mirror indices by level` both used to build one object PER LEAF; they now build one per distinct
# sampling and per distinct level respectively. On this fixture that is 4 objects instead of 37,099,
# so a regression that reverts the sharing shows up as a four-order-of-magnitude jump rather than as
# a subtle slowdown. The `@assert`-free guards below hold the fixture to the shape that makes that
# true.
#
# WHICH FIXTURE. The synthetic `TREE` has no BEAST space behind it, so the four cases that only need
# a tree and samplings use it and get the cleaner signal its 37,099 leaves provide. Real radiation
# assembly needs a space, and its cost is dominated by mesh-dependent quadrature, so it gets its own
# deliberately small mesh below rather than being run on a fixture sized for the kernels.

const CONSTRUCTIONLEAVES = sort(H2Trees.leaves(TREE))

# Every relevant level, not just `BENCHLEVELS`: `leaffarfieldsamplings` indexes this vector by
# `level - minrelevantlevel + 1`, so it has to be dense over `RELEVANTLEVELS` exactly as the vector
# `getfieldcollectionsdict` receives from `getsamplingsparentsamplings` is.
const CONSTRUCTIONSAMPLINGS = [
    MLFMA.sampling(GAMMA, EPSILON, TREE; level=level)[1] for level in RELEVANTLEVELS
]
const MINRELEVANTLEVEL = first(RELEVANTLEVELS)

const CONSTRUCTIONLEAFSAMPLINGS = MLFMA.leaffarfieldsamplings(
    TREE, CONSTRUCTIONSAMPLINGS, MINRELEVANTLEVEL; leaves=CONSTRUCTIONLEAVES
)

# Measured on this fixture: leaves sit on levels 5-8 (34,636 / 2,205 / 233 / 25 of them). Recorded as
# a guard because BOTH sharing cases below are vacuous on a single-level fixture -- "one cache per
# level" and "one cache per leaf" are the same statement when every leaf shares a level, and the
# benchmark would happily report a healthy number for a regressed implementation.
const CONSTRUCTIONLEAFLEVELS = unique(
    H2Trees.level(TREE, leaf) for leaf in CONSTRUCTIONLEAVES
)
length(CONSTRUCTIONLEAFLEVELS) > 1 || error(
    "benchmark fixture: every leaf of TREE sits on level $(only(CONSTRUCTIONLEAFLEVELS)), so the " *
    "frame-cache and mirror-index cases cannot distinguish per-level sharing from per-leaf " *
    "construction. Fix the fixture rather than dropping the cases.",
)

# The sharing itself, checked once at fixture-build time. A benchmark cannot assert, so this is the
# only place the property is pinned in this file -- the correctness tests live in
# `test/FarField/leaforder.jl` and `test/FarField/storerules.jl`.
let
    caches = MLFMA.framecachefor(
        MLFMA.farfieldstorerule(OPERATOR), CONSTRUCTIONLEAFSAMPLINGS
    )
    bylevel = MLFMA._mirrorindicesbylevel(
        TREE, CONSTRUCTIONLEAVES, CONSTRUCTIONLEAFSAMPLINGS
    )
    nuniquecaches = length(unique(objectid, caches))
    nuniquecaches == length(CONSTRUCTIONLEAFLEVELS) || error(
        "benchmark fixture: `framecachefor` produced $nuniquecaches distinct caches for " *
        "$(length(CONSTRUCTIONLEAFLEVELS)) distinct leaf levels over " *
        "$(length(CONSTRUCTIONLEAFSAMPLINGS)) leaves. The per-sampling sharing this case exists " *
        "to measure is not happening.",
    )
    length(bylevel) == length(CONSTRUCTIONLEAFLEVELS) || error(
        "benchmark fixture: `_mirrorindicesbylevel` keyed $(length(bylevel)) entries for " *
        "$(length(CONSTRUCTIONLEAFLEVELS)) distinct leaf levels.",
    )
end

# The lazy builder reads `radiationcollections[leaf]` and wraps it without touching its matrices, so
# ONE shared synthetic collection stands in for all 37,099 leaves. Building a correctly-sized real
# collection per leaf would cost ~1.25 MiB each -- tens of GiB of fixture -- to measure a wrap that
# never reads them. It does have to be the right REPRESENTATION: `LazyReceiveFieldCollection`'s
# constructor rejects a collection whose type does not match the operator's `farfieldtype`.
const LAZYRADIATION = MLFMA.Dictionary(
    CONSTRUCTIONLEAVES,
    fill(
        MLFMA.radiationcollectiontype(FARFIELDTYPE)(
            collectioncomponents(25, MLFMA.fielddim(FARFIELDTYPE))
        ),
        length(CONSTRUCTIONLEAVES),
    ),
)

# The mesh fixture, for the cases that need a real space ###
#
# A 1λ sphere at λ/8. Chosen as the smallest mesh that still produces a tree deep enough to have a
# leaf level distinct from the root, because the quadrature cost scales with the mesh and these cases
# already dominate the group at 2-4 s per sample. `meshsphere` shells out to gmsh, so the mesh is
# reproducible for a given gmsh version but not across versions -- hence the recorded counts below,
# which turn a mesher change into a loud failure instead of a silent shift in the baseline.
#
# ONE MESH, TWO SPACES. The store-rule sweep needs both: `Maxwell3D`'s operators are defined on a
# div-conforming space and `Helmholtz3D`'s on a scalar one, so no single space reaches all five store
# rules. Sharing the mesh keeps the geometry (and the single gmsh call) common to both.
const MESH = meshsphere(WAVELENGTH, WAVELENGTH / 8)
const MESHSPACE = raviartthomas(MESH)
const MESHSCALARSPACE = lagrangec0d1(MESH)

# Tree, per-level samplings and per-leaf recipe for one space. A function rather than two copies of
# the same six lines, because the two spaces have to be set up IDENTICALLY for their timings to mean
# anything -- and because `TwoNTree` derives the tree from the space's own positions, so the scalar
# space gets its own tree (vertex-centred, coarser) rather than reusing the RWG one.
#
# `GAMMA` for every space: all five sweep operators carry the same wavenumber (verified --
# `BEAST.gamma` is `2πim` for each), so one γ gives every case on a space the same direction grids.
# That is what makes the per-rule timings comparable rather than a grid-size comparison in disguise.
function constructionmeshfixture(space)
    tree = H2Trees.TwoNTree(space)
    relevantlevels = 2:H2Trees.numberoflevels(tree)
    samplings = [
        MLFMA.sampling(GAMMA, EPSILON, tree; level=level)[1] for level in relevantlevels
    ]
    leaves = sort(H2Trees.leaves(tree))
    return (
        space=space,
        tree=tree,
        leafsamplings=MLFMA.leaffarfieldsamplings(
            tree, samplings, first(relevantlevels); leaves=leaves
        ),
    )
end

# Recorded shape of both fixtures: 3,213 RWG functions over 56 leaves all on level 3, and 1,073
# Lagrange functions over 32 leaves split across levels 2 and 3. The leaf COUNTS are what the
# per-leaf construction cost scales with, so they are what makes a run comparable with an earlier
# one; they are checked rather than trusted because gmsh supplies them.
const MESHFIXTURES = (
    div=constructionmeshfixture(MESHSPACE), scalar=constructionmeshfixture(MESHSCALARSPACE)
)

let recorded = (div=(3213, 56), scalar=(1073, 32))
    for (key, (nfns, nleaves)) in pairs(recorded)
        fixture = MESHFIXTURES[key]
        actual = (numfunctions(fixture.space), length(fixture.leafsamplings))
        actual == (nfns, nleaves) || error(
            "benchmark fixture: the $key construction space now has $(actual[1]) functions over " *
            "$(actual[2]) leaves, not the recorded $nfns and $nleaves. The mesher changed, so the " *
            "`radiation collections` timings are not comparable with earlier runs; update the " *
            "recorded numbers deliberately once, rather than removing the check.",
        )
    end
end

# The production builder, with the production quadrature strategy -- the point is what an assembly
# actually costs, so a cheaper `SingleNumQStrat` would measure a configuration nobody runs.
const RADIATIONBUILDER = MLFMA.BEASTFieldCollectionBuilder(
    MLFMA.RadiationSide(),
    MLFMA.FARFIELDASSEMBLERBUILDERRECOM,
    MLFMA.FARFIELDQUADSTRATEGYRECOM,
)

# THE STORE-RULE SWEEP, applied to `radiation collections` only. One operator per
# `AbstractFarFieldStoreRule`, so the group answers what the `farfieldstorerule`/`farfieldtype` split
# costs per REPRESENTATION rather than only for the EFIE:
#
#   efie              StoreCartesianAsTangentialSpherical    2 components   div-conforming
#   singlelayer       StoreCartesianAsSphericalVector        3 components   div-conforming
#   mw_hypersingular  StoreCartesianAsRadialScalar           1 component    div-conforming
#   hh_singlelayer    StoreAsFarFieldValue                   1 component    scalar
#   hh_hypersingular  StoreDoubleCartesianAsPairedSpherical  6 components   scalar
#
# That is all five rules `src/operators/storerules.jl` defines. The three operators it maps to
# `StoreCartesianAsTangentialSpherical` are represented by the EFIE alone: `radiationoperator` sends
# all three to the same `MWFarField3D`, so a second one would re-measure an identical assembly.
#
# READ IT AS TWO BLOCKS, NOT AS FIVE COMPARABLE ROWS. The Helmholtz rules only exist on a scalar
# space, so those two rows run over 1,073 Lagrange functions and 32 leaves while the Maxwell rows run
# over 3,213 RWG functions and 56 leaves. A Helmholtz row against a Maxwell row is therefore mostly a
# comparison of problem sizes. Each row is comparable with the other rows on its own space, and with
# itself across runs, which is what a regression check actually needs.
#
# WHERE THE COMPARISON IS GENUINELY CONTROLLED: `singlelayer` against `mw_hypersingular`. Same space,
# same tree, same direction grids, and the same far-field operator -- `radiationoperator` maps both
# `MWSingleLayer3D` and `MWHyperSingular3D` to `MWSingleLayerFarField3D`. The ONLY difference between
# those two timings is the projection: three spherical components versus one radial scalar. A gap
# there is the store rule and nothing else. Every other pair in the sweep changes the far-field
# operator too, so a difference is quadrature AND projection and cannot be charged to the rule.
#
# The other four cases stay EFIE-only, deliberately. `leaf samplings`, `lazy receive collections` and
# `mirror indices by level` never touch a store rule -- they are per-leaf bookkeeping over a tree and
# a direction grid, so a sweep would run the same code five times. `frame cache` DOES depend on the
# rule, and is the obvious next candidate, but it belongs with the storer microbenchmarks rather than
# here: on `hh_singlelayer` it would measure a `return nothing`, since `StoreAsFarFieldValue`
# short-circuits on `needsframecache`, and reporting that next to four real timings invites reading a
# no-op as a speedup.
const CONSTRUCTIONOPERATORS = (
    (name="efie", operator=MLFMA.MWEFIE3D(; wavenumber=WAVENUMBER), fixture=:div),
    (
        name="singlelayer",
        operator=Maxwell3D.singlelayer(; wavenumber=WAVENUMBER),
        fixture=:div,
    ),
    (
        name="mw_hypersingular",
        operator=MLFMA.MWHyperSingular3D(; wavenumber=WAVENUMBER),
        fixture=:div,
    ),
    (
        name="hh_singlelayer",
        operator=Helmholtz3D.singlelayer(; wavenumber=WAVENUMBER),
        fixture=:scalar,
    ),
    (
        name="hh_hypersingular",
        operator=Helmholtz3D.hypersingular(; wavenumber=WAVENUMBER),
        fixture=:scalar,
    ),
)

# EVERY RULE REACHED EXACTLY ONCE, checked against the type tree rather than assumed from the table
# above. The point of the sweep is coverage of `farfieldstorerule`'s whole range, and neither failure
# mode is visible in the output: a table that drifted into measuring one rule twice while missing
# another still reports five healthy rows, and a sixth rule added to `storerules.jl` leaves the sweep
# looking complete while silently no longer being it. `subtypes` is exact here because
# `AbstractFarFieldStoreRule`'s subtypes are a flat set of five singletons -- if that ever gains a
# layer, this check is where it has to be reconsidered.
let swept = Set(MLFMA.farfieldstorerule(case.operator) for case in CONSTRUCTIONOPERATORS),
    defined = Set(rule() for rule in subtypes(MLFMA.AbstractFarFieldStoreRule))

    missingrules = sort([string(nameof(typeof(r))) for r in setdiff(defined, swept)])
    duplicates = length(CONSTRUCTIONOPERATORS) - length(swept)
    swept == defined || error(
        "benchmark fixture: the construction sweep covers $(length(swept)) of " *
        "$(length(defined)) store rules. Not measured: $(join(missingrules, ", ")). " *
        "Measured more than once: $duplicates duplicate(s).",
    )
end

SUITE["construction"]["leaf samplings"]["efie"] = @benchmarkable(
    MLFMA.leaffarfieldsamplings(
        $TREE, $CONSTRUCTIONSAMPLINGS, $MINRELEVANTLEVEL; leaves=($CONSTRUCTIONLEAVES)
    ),
    seconds = 5,
    samples = 20,
)

# 2-4 s per evaluation, so the sample count is what the budget buys, not what precision would want.
# `evals=1` is mandatory rather than stylistic: BenchmarkTools' automatic tuning would otherwise probe
# for an eval count on a case where a single call already exceeds the whole group's budget.
#
# Five operators at five samples measured 92 s, and that is essentially the whole cost of the
# construction group -- the four synthetic cases together take under a second. `seconds = 60` per case
# is a ceiling that binds only if an evaluation slows by ~3x, where stopping early is what you want.
for case in CONSTRUCTIONOPERATORS
    fixture = MESHFIXTURES[case.fixture]
    SUITE["construction"]["radiation collections"][case.name] = @benchmarkable(
        MLFMA.buildfieldcollections(
            $RADIATIONBUILDER,
            $(case.operator),
            $(fixture.tree),
            $(fixture.leafsamplings),
            $(fixture.space),
        ),
        seconds = 60,
        samples = 5,
        evals = 1,
    )
end

SUITE["construction"]["lazy receive collections"]["efie"] = @benchmarkable(
    MLFMA.buildfieldcollections(
        MLFMA.LazyReceiveFieldCollectionBuilder(),
        $OPERATOR,
        $TREE,
        $CONSTRUCTIONLEAFSAMPLINGS,
        $LAZYRADIATION,
    ),
    seconds = 5,
    samples = 20,
)

SUITE["construction"]["frame cache"]["efie"] = @benchmarkable(
    MLFMA.framecachefor($(MLFMA.farfieldstorerule(OPERATOR)), $CONSTRUCTIONLEAFSAMPLINGS),
    seconds = 5,
    samples = 20,
)

SUITE["construction"]["mirror indices by level"]["efie"] = @benchmarkable(
    MLFMA._mirrorindicesbylevel($TREE, $CONSTRUCTIONLEAVES, $CONSTRUCTIONLEAFSAMPLINGS),
    seconds = 5,
    samples = 20,
)

# REPL/VSCode helpers #############################################################################
#
# `include`ing this file runs nothing. These are the convenient ways to run part of it by hand; see
# `benchmark/README.md`.

"""
    run_farfield_benchmarks(; kwargs...)

Run the whole suite. `kwargs` are forwarded to `BenchmarkTools.run`.
"""
run_farfield_benchmarks(; kwargs...) = run(SUITE; kwargs...)

"""
    run_farfield_group(name; kwargs...)

Run one top-level group -- `"farfield"`, `"phasefactor"`, `"fieldcollections"`, `"workflows"`,
`"construction"`. `keys(SUITE)` lists them.
"""
run_farfield_group(name; kwargs...) = run(SUITE[name]; kwargs...)

"""
    profilecase(target; evals=1000)
    profilecase(path::AbstractString...; evals=1000)

Run a suite case `evals` times, shaped for a profiler rather than for measurement. `target` is a
`Benchmark` or a `BenchmarkGroup` you already indexed out of `SUITE`; the second form takes the key
path instead. These are equivalent:

```julia
@profview profilecase(
    SUITE[\"fieldcollections\"][\"lazy receive mul!\"][\"nbasis 100\"]; evals=20_000
)
@profview profilecase(\"fieldcollections\", \"lazy receive mul!\", \"nbasis 100\"; evals=20_000)
```

Passing a GROUP runs every leaf under it, which is the convenient form when a group holds a single case:

```julia
@profview profilecase(SUITE[\"workflows\"][\"aggregate one leaf\"]; evals=20_000)
```

Note what that means for a group with several leaves -- `SUITE[\"farfield\"][\"mirror\"]` has one per
level. `evals` applies to EACH leaf, and the profiler collects them into ONE merged profile, so a frame
shared between levels is attributed to all of them together. Index down to a single leaf when you need
to know which level a cost belongs to.

`samples=1` with a large `evals` puts the repetitions inside BenchmarkTools' own inner loop, and
`gctrial`/`gcsample` are off because they call `GC.gc()` around the trial and around EVERY sample --
a full collection over a heap holding this fixture, 50 times per `run`.

Profiling `for _ in 1:10; run(case); end` instead spends a large share of its samples in that
bookkeeping rather than in the operation. How large is case- and heap-dependent, so treat these as
illustrations and not as constants -- measured by leaf-frame classification:

lazy receive mul!:    run-loop 45.8% gc/alloc (12356 samples, ~12 s)  vs  0.0-2.4% here
aggregate one leaf:   run-loop 13.1% gc/alloc (214 samples)           vs  0.0% here

Passing a `Benchmark` profiles more cleanly than passing the enclosing single-leaf `BenchmarkGroup`
(0.0% vs 8.5% on `aggregate one leaf`, and the group took ~60% longer for the same `evals`) because the
group form repeats BenchmarkTools' per-benchmark bookkeeping. Index down to the leaf for the tightest
profile; the group form is for convenience.

Pick `evals` so the total is order a second: a 60 µs case wants ~20_000, a 13 ms one ~100. Too few and
the profile has too few samples to read.

Returns a `Trial`, which is not the point. `samples=1` means its one measurement is a TYPICAL draw
rather than a minimum, and that is the larger error, not the profiler: measured on
`lazy receive mul!`, `run` reports 62.85 µs (minimum over 50 samples, whose own spread is
62.85 / 85.22 / 269.44 µs for min / median / max), while this returns 81.3 µs without a profiler
attached and 88.98 µs with one -- i.e. it lands near `run`'s MEDIAN, and profiler overhead accounts for
only the last ~9%. Use `run_farfield_group`/`run` for numbers, this for finding where the time goes.
"""
function profilecase(
    target::Union{BenchmarkTools.Benchmark,BenchmarkTools.BenchmarkGroup}; evals=1000
)
    # An empty group runs nothing and reports nothing, which under a profiler looks like a flamegraph
    # with your code missing from it rather than like an error. Reachable by indexing `SUITE` with a
    # typo, because `BenchmarkGroup`'s `getindex` AUTO-CREATES missing keys -- that is what lets this
    # file build the suite with `SUITE["a"]["b"]["c"] = @benchmarkable(...)` and it means
    # `SUITE["farfeild"]` is an empty group, not a `KeyError`.
    if target isa BenchmarkTools.BenchmarkGroup && isempty(BenchmarkTools.leaves(target))
        error(
            "profilecase: that BenchmarkGroup holds no benchmarks, so there is nothing to profile. " *
            "`BenchmarkGroup` indexing auto-creates missing keys, so this is most likely a mistyped " *
            "key -- `keys(SUITE)` lists the top level. Use the string form " *
            "`profilecase(\"farfield\", \"mirror\", \"level 8\")` to get the key checked for you.",
        )
    end
    return run(target; samples=1, evals=evals, gctrial=false, gcsample=false)
end

function profilecase(path::AbstractString...; evals=1000)
    target = SUITE
    for (depth, key) in enumerate(path)
        if !(target isa BenchmarkTools.BenchmarkGroup)
            error(
                "profilecase: SUITE[$(join(map(repr, path[1:(depth - 1)]), "]["))] is a single " *
                "benchmark, so it cannot be indexed further with $(repr(key)). Drop the trailing " *
                "keys.",
            )
        end
        # Checked rather than left to `getindex`, which would auto-create the missing key and hand
        # back an empty group.
        haskey(target, key) || error(
            "profilecase: no key $(repr(key)) under " *
            "SUITE[$(join(map(repr, path[1:(depth - 1)]), "]["))]. Available here: " *
            "$(sort!(string.(collect(keys(target))))).",
        )
        target = target[key]
    end
    return profilecase(target; evals=evals)
end

nothing
