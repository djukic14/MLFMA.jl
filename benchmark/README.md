# MLFMA far-field benchmarks

Benchmarks for `src/farfield/`: `FarField` construction and mirroring, the box-centering phase
factors, and the field-collection `mul!`/`conjmul!` kernels — plus a separate `"construction"` group
for the one-off cost of *building* the per-leaf collections those kernels run on.

Read the two apart. Everything but `"construction"` is **per matvec**, so its unit is nanoseconds and
it runs on every iteration of a solve. `"construction"` is **per operator assembly**, its unit is
milliseconds, and a regression there costs setup time, not solve time.

`benchmarks.jl` **defines only** — including it builds the fixture, populates `SUITE`, and runs
nothing. That is what makes it safe to re-evaluate while developing. `run.jl` is the only file that
starts a run.

Interpolation is a **separate suite** with its own entry point and README:
[`benchmark/interpolation/`](interpolation/README.md). It reports accuracy alongside runtime, since an
interpolation change is only an improvement if it wins on both, and it deliberately defines `TIMINGS`
rather than a second `SUITE` so the two cannot be included into one another.

## First time

```julia
using Pkg
Pkg.activate("benchmark")
Pkg.instantiate()
```

The environment resolves MLFMA, H2Factory and H2Trees from their local checkouts via `[sources]` in
`benchmark/Project.toml`, so no manual `Pkg.develop` step is needed. `benchmark/Manifest.toml` is
gitignored.

## From the command line

```console
julia --project=benchmark benchmark/run.jl
```

Prints the full `BenchmarkGroup` and then a flat, sorted table of minimum time / memory / allocations
per case.

## From a REPL or VSCode

Include once, then run whatever you are currently working on:

```julia
include("benchmark/benchmarks.jl")

keys(SUITE)                                  # "farfield", "phasefactor", "fieldcollections",
                                             # "workflows", "construction"
keys(SUITE["farfield"])                      # "construct zero", "construct undef", "mirror", "accessors"

run(SUITE["phasefactor"])                    # one group
run(SUITE["farfield"]["mirror"])             # one case, all levels
run(SUITE["farfield"]["mirror"]["level 8"])  # one case, one level
```

Two helpers exist for the same thing:

```julia
run_farfield_benchmarks()                    # the whole suite
run_farfield_group("fieldcollections")       # one top-level group
```

Re-including is safe: ~23 s the first time (nearly all of it package loading and compilation), ~2.8 s
after that. The fixture is rebuilt on each include, and redefining its `const`s is fine.

**Both numbers grew with the construction group, and include now needs gmsh.** They were ~11 s and
~0.7 s before it existed, and ~19 s / ~2.5 s before the store-rule sweep added a second space. That
group's mesh fixture (`MESH = meshsphere(...)`, then a `raviartthomas` and a `lagrangec0d1` space over
it) is built at include time, so `include`ing this file now loads `CompScienceMeshes` and shells out to
the mesher: a few tenths of a second of the re-include once gmsh is warm, several seconds on the first
call in a session, and an error rather than a slow path if gmsh is unavailable. That is the trade for
measuring real BEAST assembly at all — the cases need real spaces, and a space needs a mesh.

If the re-include cost is in your way while iterating on a kernel case, index `SUITE` directly
instead of re-including. Everything outside `SUITE["construction"]` is built from the synthetic point
cloud and never touches gmsh.

**Editing `src/` does not take effect on re-include.** MLFMA is a package, so `include`ing this file
again re-runs the fixture against the *already-loaded* MLFMA. Either load `Revise` before `using
MLFMA` (i.e. before the first `include` here), or restart the REPL. Comparing a "before" and "after"
number without one of those measures the same code twice.

## Profiling one case

`profilecase` takes the same key path you would use to index `SUITE`:

```julia
using Profile   # or nothing extra, if you use VSCode's @profview
include("benchmark/benchmarks.jl")

@profview profilecase("fieldcollections", "lazy receive mul!", "efie nbasis 100"; evals=20_000)
@profview profilecase("farfield", "mirror", "level 3"; evals=200)
```

Pick `evals` so the total is order a second — a 60 µs case wants ~20,000, a 13 ms one ~100.

All three forms are equivalent:

```julia
profilecase(SUITE["workflows"]["aggregate one leaf"]; evals=20_000)   # a group: runs every leaf under it
profilecase(SUITE["workflows"]["aggregate one leaf"]["nbasis 100"])   # a single Benchmark
profilecase("workflows", "aggregate one leaf", "nbasis 100")          # the key path
```

The string form checks each key and lists the alternatives on a typo. Worth using, because
`BenchmarkGroup` indexing **auto-creates** missing keys — that is how the suite builds itself with
`SUITE["a"]["b"]["c"] = ...` — so `SUITE["farfeild"]` is an empty group rather than a `KeyError`, and
profiling it would give you a flamegraph with your code simply absent. `profilecase` refuses an empty
group for that reason.

It is not the same as profiling `for _ in 1:10; run(SUITE[...]); end`. That form spends a large share of
its samples in BenchmarkTools' `GC.gc()` around the trial and around every sample, rather than in the
operation. The share is case- and heap-dependent, so these are illustrations, not constants:

| case | `for _ in 1:10; run(case); end` | `profilecase` |
|---|---:|---:|
| `lazy receive mul!` | 45.8% gc/alloc (12,356 samples, ~12 s) | 0.0–2.4% |
| `aggregate one leaf` | 13.1% gc/alloc (214 samples) | 0.0% |

The `lazy receive mul!` figures were taken when that lane measured the fused kernel, which has since
been deleted. The *point* they illustrate is about BenchmarkTools' harness, not about the kernel, so
they still make it; do not read them as current timings for that lane.

Passing a `Benchmark` profiles slightly more cleanly than passing its enclosing single-leaf group (0.0%
vs 8.5% on `aggregate one leaf`, and the group took ~60% longer for the same `evals`), since the group
form repeats BenchmarkTools' per-benchmark bookkeeping. Index to the leaf when you want the tightest
profile; the group form is for convenience.

For a group with several leaves — `SUITE["farfield"]["mirror"]` has one per level — `evals` applies to
each, and the profiler merges them into **one** profile, so a frame shared across levels is attributed
to all of them at once. Index to a single leaf when you need to know which level a cost belongs to.

Its return value is a `Trial`, but ignore the timings — and the reason is not mainly profiler overhead.
`samples=1` gives one *typical* measurement where `run` gives the minimum over many, and on this case
the per-sample spread is wide:

| | `lazy receive mul!`, `efie nbasis 100` (fused-era numbers) |
|---|---:|
| `run` (min of 50 samples) | 62.85 µs |
| `run` sample spread (min / median / max) | 62.85 / 85.22 / 269.44 µs |
| `profilecase`, no profiler | 81.30 µs |
| `profilecase`, profiler active | 88.98 µs |

`profilecase` lands near `run`'s *median*, and the profiler itself accounts for only the last ~9%. That
is why `run` exists for numbers: taking a minimum over 50 samples is what makes a 5% regression visible
against a distribution whose max is 4× its min.

Because the fixture lives in module-level `const`s, you can also just call the operation directly on
exactly the data the benchmark used, which is the cleanest option when you want to vary something:

```julia
farfield, sampling = FARFIELDS[3], SAMPLINGS[3]
MLFMA.mirror(farfield, sampling)                     # warm up first, or you profile the compiler
@profview for _ in 1:200; MLFMA.mirror(farfield, sampling); end
```

For allocation rather than time, `Profile.Allocs` with `sample_rate=1` (the default undercounts):

```julia
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1 MLFMA.mirror(FARFIELDS[3], SAMPLINGS[3])
results = Profile.Allocs.fetch()
```

Each entry carries `.type`, `.size` and `.stacktrace`. On `mirror` at level 3 that attributes ~2,400 of
its ~3,600 allocations to small `Vector{Int64}`/`Memory{Int64}` from `mirrorindices` building the index
map, not to copying field data.

## The fixture

Deterministic and asserted, not incidental:

- λ = 1, so every length reads in wavelengths. γ comes from `MLFMA.MWEFIE3D(; wavenumber=2π/λ)`, so
  the sign convention matches production.
- 60,000 seeded points on a shell of radius 32λ, `minhalfsize = λ/4`, `minvalues = 1` → **exactly 8
  levels**. The suite `error`s at include time if the depth is not 8, rather than silently
  benchmarking a shallower tree.
- Per-level spherical samplings at `ϵ = 1e-4`.

### Which levels, and why not all of them

`H2Trees.buildplans(TREE).relevantlevels` is `3:8` for this fixture — the tree has 8 levels, but
levels 1 and 2 carry no aggregation, so no far field is ever built for them. Measured sizes of one
2-component `ComplexF64` `FarField`:

| level | Nθ × Nφ | directions | FarField |
|------:|--------:|-----------:|---------:|
| 8 (leaf) | 28 × 56 | 1,568 | 0.05 MiB |
| 6 | 70 × 140 | 9,800 | 0.30 MiB |
| 5 (middle) | 120 × 240 | 28,800 | 0.88 MiB |
| 4 | 216 × 432 | 93,312 | 2.85 MiB |
| 3 (coarsest relevant) | 400 × 800 | 320,000 | 9.77 MiB |
| 2 | 762 × 1524 | 1.16 M | 35.44 MiB |
| 1 | 1474 × 2948 | 4.35 M | 132.61 MiB |

The suite runs levels **8, 5, 3** — finest, middle, coarsest relevant. That is a 200× spread in
direction count, which is what makes a size-scaling regression visible. Levels 1–2 are excluded
because they are outside `relevantlevels` *and* because at 35–133 MiB per far field the measurement
would be dominated by the allocator and GC rather than by far-field code.

`relevantlevels` is hardcoded rather than recomputed: `buildplans` costs ~50 s on this tree, which
would dwarf the suite. The tree-depth check is what ties that recorded value to a verified shape —
rerun `buildplans` if the fixture changes.

### Collections are swept by basis count, not by level

Field collections are **leaf** objects, shaped (directions × basis functions in that leaf). "One per
level" is not a thing that exists: a level-3-shaped collection would be 0.95 GiB per component
against 2.4 MiB for a real one. So `SUITE["fieldcollections"]` varies the leaf's basis-function count
(25 / 100 / 400, bracketing `MLFMA.MINVALUESRECOM = 100`, the subdivision threshold that sets how
many functions a leaf holds) at the leaf sampling.

`SUITE["fieldcollections"]` also sweeps an **operator** axis, because representation is now carried by
the far-field type rather than inferred from a component count, and because `MLFMA.α` is a *vector*
for some operators:

| operator | far-field type | components | `MLFMA.α` |
|---|---|---:|---|
| `MWEFIE3D` | `TangentialSphericalFarField` | 2 | scalar |
| `MWSingleLayer3D` | `SphericalVectorFarField` | 3 | vector (3) |
| `HH3DHyperSingularFDBIO` | `PairedSphericalVectorFarField` | 6 | vector (6) |

The vector-α cases are not redundant with the scalar one: leaf disaggregation passes `α(core)` into
the result-side multiply, and that resolves to a different H2Factory collection method than the scalar
form -- the scratch path forwards `α` straight through rather than weighting anything itself, so this
lane measures whether the weighted collection method costs more than the plain one. Case keys are
therefore `"<operator> nbasis <n>"`.

Both directions of the lazy receive path are measured: the read side (disaggregation) and the
**write-through** side (aggregation, reached when a transposed/adjoint map aggregates the test side).
The read side is also swept with per-component α weights.

Every lazy lane calls `H2Factory.applyleafmul!`, because that is the entire public surface of
`LazyReceiveFieldCollection`: it is a scratch-only leaf collection and implements no `mul!` or
`conjmul!` of its own. There are no direct-`mul!` or `conjmul!` lanes, and no *fused* lane — the fused
kernels were deleted when the leaf channel made them unreachable. See
`src/farfield/lazyreceivefieldcollection.jl`.

### The two receive lanes

| lane | what it does |
|---|---|
| `receive mul!` | the materialized `ReceiveFieldCollection` — one `gemv` per component, no mirroring at all. The ceiling. |
| `lazy receive mul!` | what the traversal runs: mirror into the borrowed leaf-channel scratch (`receivefields!`), then one `gemv` per component. |

Both compute **the same numbers**. `receive` is not independent random data of the right shape — it
is built by `materializedreceive`, applying the operator's own `receivecouplingrule` to the radiation
collection (`R[src][t,:] = coeff * C[dest][idx[t],:]`). That identity is derived in the source and is
the ground truth `test/FarField/rules.jl` checks both `applyleafmul!` arities against, so if it were
wrong the suite fails. The two lanes therefore differ only in *how* they get there, which is what
makes the comparison meaningful.

Measured (minimum of 50 samples, this machine, `nbasis 400`, leaf sampling at level 8, zero
allocations in both lanes). Two runs several minutes apart, to show the spread — treat these as
indicative, not as a baseline to diff against:

| operator | `receive mul!` | `lazy receive mul!` | ratio |
|---|---:|---:|---:|
| `efie` | 42 / 48 µs | 95 / 125 µs | 2.3× / 2.6× |
| `singlelayer` | 89 / 84 µs | 157 / 149 µs | 1.8× / 1.8× |
| `hh_hypersingular` | 171 / 188 µs | 276 / 291 µs | 1.6× / 1.5× |

Run-to-run drift is ~30% on the smallest lane, which is more than several of the differences anyone
would want to read out of a single run. Compare lanes *within* one run, and re-run before believing a
regression.

The gap between the two lanes is the price of not storing a second collection per leaf. It is
consistently *larger* than `receivefields!` alone accounts for (≈2 µs for EFIE at this sampling), so
something beyond the mirroring costs the difference — not yet chased down. **Do not read the remaining
gap as "the cost of mirroring".**

For scale on what the leaf channel bought: before it existed this path ran a fused elementwise kernel,
measured at 2744 / 5013 / 12290 µs for the three operators above — well over an order of magnitude
slower than the lane measured today, far outside the noise above. Those kernels no longer exist, so
there is nothing left to benchmark against; the figure is recorded only so the ratio above reads as
"close to the ceiling" rather than "slow".

### `receivefields` and `receivefields!`

Two lanes, swept per operator **and** per level (the rule length and component count differ per
representation, so EFIE alone would not cover the step):

- `receivefields` — the **allocating** reference, mirroring into a freshly allocated far field. What
  the lazy path had to do before a workspace owned leaf scratch.
- `receivefields!` — the **in-place** step the scratch path actually runs inside
  `H2Factory.applyleafmul!`, writing into a preallocated buffer.

The gap between them is exactly what the leaf channel removes. At level 5, EFIE: 65 µs / 922 kB
allocated versus 37 µs / **0 B**. At the coarsest level the two converge in time (the copy dominates)
while the allocation still goes from ~30 MB to zero, which is the part that matters under threading.

The scratch buffers in every lane are allocated once, outside the `@benchmarkable`, exactly as a
workspace allocates its leaf channel once — timing their construction would measure the cost the leaf
channel exists to remove.

`receivefields` is the exception — it depends on the sampling and operator, not the basis count, so it
is swept by level, on EFIE.

### Construction

`SUITE["construction"]` covers the five layers `getfieldcollectionsdict` walks:

| case | what it measures | fixture |
| --- | --- | --- |
| `leaf samplings` | `leaffarfieldsamplings` — level lookup, per-level direction points, centers, leaf ids | synthetic tree, 37,099 leaves |
| `radiation collections` | `buildfieldcollections` through the real BEAST quadrature, one entry per far-field **store rule** | 1λ sphere at λ/8; 3,213 RWG functions over 56 leaves, or 1,073 Lagrange over 32 |
| `lazy receive collections` | `LazyReceiveFieldCollectionBuilder` wrapping radiation collections | synthetic tree |
| `frame cache` | `framecachefor` — the storer's trig tables | synthetic tree |
| `mirror indices by level` | `_mirrorindicesbylevel` — the map the lazy side shares | synthetic tree |

Three of these exist to pin a **sharing property**, not just a duration. `frame cache` and
`mirror indices by level` both used to build one object per *leaf*; they now build one per distinct
sampling and per distinct level. On this fixture that is 4 objects instead of 37,099, so losing the
sharing shows up as a four-order-of-magnitude jump rather than as a subtle slowdown. The fixture
`error`s at include time if the tree stops having leaves on more than one level, because that is
exactly the shape in which "per level" and "per leaf" become indistinguishable.

`radiation collections` is the only case that needs a real mesh and BEAST space, and at 2–4 s per
evaluation the five of them are essentially the whole group: 92 s of its runtime as measured, against
under a second for the four synthetic cases together. It therefore uses a deliberately small mesh rather than a fixture sized for the
kernels, `evals=1` (a single call already exceeds any tuning budget), and recorded function/leaf counts
that fail loudly if the mesher changes underneath it — gmsh output is reproducible for a given version,
not across versions, and a silently different mesh would move the baseline without moving the code.

#### The store-rule sweep

`radiation collections` carries one operator per `AbstractFarFieldStoreRule`, which is what answers
whether the `farfieldstorerule`/`farfieldtype` split costs different amounts per representation. The
other four cases are EFIE-only: three of them never touch a store rule at all, and `frame cache` is
held for the storer microbenchmarks (on `StoreAsFarFieldValue` it short-circuits on `needsframecache`
and would report a `return nothing` next to four real timings).

| case | store rule | components | space |
| --- | --- | ---: | --- |
| `efie` | `StoreCartesianAsTangentialSpherical` | 2 | `raviartthomas` |
| `singlelayer` | `StoreCartesianAsSphericalVector` | 3 | `raviartthomas` |
| `mw_hypersingular` | `StoreCartesianAsRadialScalar` | 1 | `raviartthomas` |
| `hh_singlelayer` | `StoreAsFarFieldValue` | 1 | `lagrangec0d1` |
| `hh_hypersingular` | `StoreDoubleCartesianAsPairedSpherical` | 6 | `lagrangec0d1` |

That is all five rules `src/operators/storerules.jl` defines, checked against
`subtypes(AbstractFarFieldStoreRule)` at include time so that adding a sixth fails the fixture instead
of leaving an incomplete sweep looking complete.

**Read it as two blocks, not as five comparable rows.** The Helmholtz rules only exist on a scalar
space, so those two rows run over 1,073 functions and 32 leaves against the Maxwell rows' 3,213 and 56.
A Helmholtz row against a Maxwell row is mostly a comparison of problem sizes. Each row is comparable
with the others on its own space, and with itself across runs, which is what a regression check needs.

**`singlelayer` against `mw_hypersingular` is the one controlled pair.** Same space, same tree, same
direction grids, and the same far-field operator — `radiationoperator` maps both `MWSingleLayer3D` and
`MWHyperSingular3D` to `MWSingleLayerFarField3D` — so the only difference between those two timings is
three spherical components versus one radial scalar. Every other pair changes the far-field operator
too, so a gap there is quadrature *and* projection and cannot be charged to the rule.

What that pair says as measured: 2.14 s / 244 MiB for three components against 2.07 s / 90 MiB for one.
The projection costs ~3% in time and 2.7× in memory, i.e. **the store rule shows up in what gets
stored, not in what gets computed** — quadrature dominates the clock either way. Worth knowing before
optimizing a projection kernel to save assembly time.

## Deliberately not measured

- **Aggregation/disaggregation traversal, interpolation, translation** — different layers, and the
  workspace/scheduling machinery they need makes them a separate suite.
- **The flat `buildfieldcollection` path** next to the tree `buildfieldcollections` one, which would
  separate raw quadrature cost from the per-leaf ordering, clustering and box-centering around it.
- **`farfieldlocal!`/`_store!` at the inner-loop level**, which is what would attribute construction
  cost to quadrature versus projection versus frame-cache lookup versus the matrix writes.
- **Leaf-count scaling** across two or three mesh sizes.
- **CI comparison.** No `ci.jl` and no `.github/workflows/Benchmarks.yml` yet, by design: get the
  local suite answering useful questions first, then mirror H2Trees' `benchmark/ci.jl`.

The middle three are wanted, in that order. They are held back deliberately: each multiplies the
construction group rather than adding to it, and gmsh variability makes leaf-count scaling the noisiest
of the three. The group as it stands covers the *layers* and the store rules; those three cover the
*why*.

`run.jl` deliberately does not call `tune!`. Every case sets its own `seconds`/`samples`; tuning
rewrites evaluations-per-sample from whatever the machine is doing at that moment, which is the
opposite of what run-to-run comparison wants. Tune by hand in a REPL if a case needs it.
