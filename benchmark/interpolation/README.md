# Interpolation accuracy and runtime studies

Measures what MLFMA's angular level-to-level interpolation costs in **accuracy** and in **runtime**,
against exact angular functions, with nothing else in the way. No mesh, no operator, no BEAST
assembly, no translation, no near field, no addition-theorem truncation, no tree — the reference is a
closed-form spherical harmonic evaluated independently on both grids. That is what makes these numbers
usable as a before/after for a change to `src/interpolation/`.

This directory exists because `test/` is for correctness pins and these are development sweeps.
Nothing here is a test; nothing here should become one.

```console
julia --project=benchmark benchmark/interpolation/run.jl              # accuracy, ~25 s after load
julia --project=benchmark benchmark/interpolation/run.jl --timings    # and runtime, ~2 min more
```

`benchmarks.jl` defines and runs nothing, so it is safe to re-include from a REPL while developing;
`run.jl` is the only thing that starts a run. Runtime is opt-in because the two axes answer different
questions and take very different times — checking whether a change moved the error budget should not
require waiting for `BenchmarkTools`.

Accuracy is measured through `MLFMA.InterpolationDiagnostics`, runtime through `BenchmarkTools`, and
the two are never printed in one table. **A faster kernel that loses a digit is a regression, and
nothing in a `Trial` would say so.**

## How to read ρ

```text
ρ = casebandwidth(case) / supporteddegree(sourcesampling)
```

`supporteddegree` is `MLFMA.cutoffparameter`, the exact inverse of `MLFMA.NθNφ` — a grid built as
`NθNφ(L)` supports degree `L`. So **ρ = 1 is precisely where the source grid stops carrying the
case**, with no fudge factor:

- **ρ < 1** — the grid represents the function and the error is the interpolator's. This is the only
  region where a number is a statement about interpolation quality.
- **ρ ≥ 1** — the grid does not represent the function at all. A large error there is arithmetic, not
  a defect, and it is kept in the tables deliberately: it is the reference point that makes the
  in-band numbers mean something.

ρ is an interpretation aid for these sweeps, **not** a truncation model. What degree a real MLFMA
level needs comes from `MLFMA.sampling(γ, ε, tree; level)`, which accounts for box size, wavenumber
and target accuracy.

The default grids are `NθNφ(13)` → `NθNφ(26)`, i.e. 28×56 → 54×108 in (Nθ, Nφ): coarse to fine, the
aggregation direction, at the usual level-to-level doubling.

## The groups

| # | Group | Question |
|:--|:--|:--|
| 1 | stencil/order sweep | how `rel_l2`/`rel_linf` depend on stencil size `p` across ρ |
| 2 | interpolator family | `SphericalLagrangeInterpolator` against `LocalθGlobalφInterpolator` |
| 3 | sampling margin | how much margin a requested tolerance needs, at fixed case degree |
| 4/5 | θ and φ isolation | which direction limits the error |
| 6 | separate θ/φ orders | whether `pθ != pφ` is worth an API |
| 7 | θ versus cos θ | whether the θ step should interpolate in `x = cos θ` |
| 8 | polar error profile | whether the error concentrates near the poles |
| 9 | forward field vs reverse pairing | the two tolerance numbers, side by side |
| 10 | tangential default-flip diligence | whether the global-φ default costs accuracy |
| 11 | equidistant vs Gauss-Legendre | what spectral-in-both-angles buys, and what it costs |
| 12 | cost at equal tolerance | the grid each family needs for a target ε, and its cost there |
| 13 | anisotropic stencil orders | whether pθ and pφ can differ, and what that is worth |

Groups 4 and 5 of the handoff plan are printed as **one** table, because the comparison that matters
is total versus θ-only versus φ-only at fixed case and order, and reading that off two tables invites
lining rows up by hand.

### How θ and φ are separated

Both production interpolators resample θ and φ **separably, θ first**. So with

```julia
middle = ID.intermediatesampling(source, target)   # target's θ nodes, source's φ nodes
```

a study `source -> middle` is the θ half alone — its φ step lands on nodes it already has, which is
the **exact** identity (measured: `rel_l2 == 0.0`, because the barycentric weights detect the
coincident node rather than dividing by something small) — and `middle -> target` is the φ half alone.
Chaining both through `ID.interpolationerror([a, b], [source, middle, target], cases)` reproduces the
single-stage study to every digit at equal stencil sizes, so this is the same arithmetic regrouped and
not an approximation of it. Group 6 exploits that to measure `pθ != pφ` before any such API exists, by
passing two interpolators of different stencil size. `test/LevelTransfers/interpolationdiagnostics.jl`
pins all of it.

## What the current numbers say

Measured on one machine at the grids above; the orderings are the durable part, not the digits. Every
figure below is `rel_l2`.

### θ is the limiting direction, not φ

At p=8, ρ=0.77, splitting the error into halves:

| case | total | θ only | φ only (local) | φ only (global/FFT) |
|:--|--:|--:|--:|--:|
| `Y_10^0` axisymmetric | 5.98e-3 | 5.98e-3 | 1.7e-16 | 1.1e-16 |
| `Y_10^5` mixed | 7.26e-3 | 7.25e-3 | 6.9e-6 | 1.4e-15 |
| `Y_10^10` sectoral | 1.48e-3 | 2.06e-4 | 1.42e-3 | 2.8e-15 |
| random L=10 | 4.11e-3 | 4.11e-3 | 1.8e-4 | 1.0e-15 |

θ owns essentially the whole error for axisymmetric, mixed and random-band content. Only the sectoral
`m = l` case is φ-limited. This is why the plan's early guess — that the worst cases are `m = 0` or
`m ≈ l/2`, so θ or sampling margin is limiting — holds up.

### Global (FFT) φ interpolation is exact for the scalar band-limited cases measured here

The `φ only (global)` column above never exceeds **5.7e-15** across every order and case in these
studies, including at ρ = 1.54 where local φ interpolation gives 0.15. That is not "better", it is
exact to round-off, and the mechanism is understood rather than incidental: zero-padding the φ DFT is
exact interpolation of a trigonometric polynomial, and along φ a band-limited scalar far field is one.

**Scope of that claim, stated because it is the strongest result in this file.** What is measured is
`ScalarFarField{ComplexF64}`, single harmonics and random band-limited superpositions, on the grids
above. The argument should carry to any representation whose φ dependence is band-limited on a uniform
φ grid — but the tangential and vector representations add continuation and sign rules across the φ
seam and at the poles that nothing here exercises, so treat φ-exactness as established for the scalar
case and *expected*, not verified, for the others. Re-run this group once the diagnostics grow
representation-aware exact cases.

Within that scope, `LocalθGlobalφInterpolator` reduces the interpolation problem to θ alone.

Group 2 shows this end to end: the two families agree to the last digit on every non-sectoral case
(their θ code is the same code), and on `Y_20^20` the global one is 0.00253 against 0.15 at p=8, and
2.24e-4 against 0.031 at p=20.

### Interpolating θ in `x = cos θ` is exact for even `m` — and destructive for odd `m`

Production interpolates in **θ itself**: `_θinformation!` builds stencil nodes as
`MLFMA.θ.(Ref(sampling), θindices)` and takes barycentric weights on those. The θ factor of `Y_l^m` is
a polynomial of degree `l` in `x = cos θ` **times** `(1-x²)^(m/2)`, so the change of variable should be
exact once the stencil spans the degree — but only when that second factor is itself a polynomial,
i.e. for even `m`. Group 7 measures both, with the same nodes, the same stencils and production's own
barycentric formula, so the only difference is the variable:

| θ factor | θ p=8 | θ p=12 | θ p=20 | cosθ p=8 | cosθ p=12 | cosθ p=20 |
|:--|--:|--:|--:|--:|--:|--:|
| l=4 m=0 | 9.13e-6 | 7.59e-8 | 3.58e-12 | **4.6e-15** | 2.24e-13 | 4.0e-10 |
| l=10 m=0 | 5.98e-3 | 1.39e-3 | 5.41e-5 | 4.57e-3 | **8.01e-13** | 1.12e-9 |
| l=10 m=5 | 7.25e-3 | 1.41e-3 | 4.67e-5 | 4.02e-3 | 2.99e-3 | **3.14** |
| l=10 m=10 | 2.06e-4 | 2.48e-5 | 5.46e-7 | 6.41e-5 | **4.02e-16** | 2.6e-11 |
| l=20 m=0 (ρ>1) | 0.203 | 0.222 | 0.255 | 0.271 | 0.436 | 0.225 |

Reading it:

- **Even `m`, stencil spanning the degree (`p ≥ l+1`): exact.** l=4 m=0 at p=8, l=10 m=0 and m=10 at
  p=12 all land at round-off — nine to ten orders of magnitude below interpolating in θ.
- **Stencil too short: almost no gain.** l=10 at p=8 (`p < l+1`) improves only from 5.98e-3 to
  4.57e-3. The win is exactness, not a uniform improvement.
- **Odd `m`: worse, and catastrophically so at high order.** l=10 m=5 gives 2.99e-3 at p=12 against
  θ's 1.41e-3, and **3.14 at p=20**. `(1-x²)^(m/2)` for odd `m` is `sin^m θ`, which has a square-root
  branch point at both poles: analytic in θ, hopeless for polynomial interpolation in x, and worse as
  the stencil reaches further toward the poles.

So a plain change of variable is **not** the free win the `m = 0` row suggests. A real far field is a
mixture over all `m`, so switching the θ variable to `x` would make the even-`m` content exact and
wreck the odd-`m` content, with no way to separate them at that point in the pipeline. Anyone
implementing "interpolate in cos θ" on the strength of the axisymmetric row would ship a regression
that the existing tests, which use mixed-`m` fields, might well average away.

Two further constraints on the idea, both measured rather than assumed:

- **The pole extension cannot survive the change of variable.** It mirrors θ → −θ, and `cos` is even,
  so a mirrored node is a *duplicate* node in `x` and the barycentric weights are singular. Group 7 is
  therefore restricted to the 40 of 54 target θ whose whole stencil is interior; a production version
  would need a different polar treatment, not a translated one.
- **High order in `x` is worse than moderate order**, even where it is exact: l=4 m=0 goes 4.6e-15 at
  p=8 → 2.24e-13 at p=12 → 4.0e-10 at p=20. Conditioning, not bandwidth.

### The error concentrates away from the poles

Share of the squared absolute error by θ band at p=8, counted in from each pole (rows 9–27 are the
interior of each hemisphere):

| case | N rows 1–4 | N rows 5–8 | N rows 9–27 | S rows 1–4 | S rows 5–8 | S rows 9–27 |
|:--|--:|--:|--:|--:|--:|--:|
| `Y_10^0` | 8.0e-12 | 2.1e-7 | 0.5 | 8.0e-12 | 2.1e-7 | 0.5 |
| `Y_10^5` | 1.7e-12 | 1.2e-7 | 0.5 | 1.7e-12 | 1.2e-7 | 0.5 |
| `Y_10^10` | 6.0e-17 | 1.5e-11 | 0.5 | 6.0e-17 | 1.5e-11 | 0.5 |

Effectively **all** of it is interior; the polar bands hold 1e-7 or less. So the virtual-extension
polar treatment is not the problem, and the two hemispheres are symmetric to three digits, which is
also a check that the extension applies no asymmetric sign. Gauss–Legendre θ nodes are near-uniform in
θ and cluster in `x` toward the poles, so the sparsest sampling relative to how fast `P_l^m(cos θ)`
oscillates is near the equator — where the error duly sits.

### `pθ != pφ` is real, but a distant second to using global φ

At ρ=0.77, chaining two Lagrange interpolators:

| case | 8/8 | 12/8 | 20/8 | 8/20 | 12/12 | 20/20 |
|:--|--:|--:|--:|--:|--:|--:|
| `Y_10^0` axisymmetric | 5.98e-3 | 1.39e-3 | 5.41e-5 | 5.98e-3 | 1.39e-3 | 5.41e-5 |
| `Y_10^5` mixed | 7.26e-3 | 1.41e-3 | 5.01e-5 | 7.25e-3 | 1.41e-3 | 4.67e-5 |
| `Y_10^10` sectoral | 1.48e-3 | 1.43e-3 | 1.42e-3 | 2.06e-4 | 1.02e-4 | 8.01e-7 |
| random L=10 | 4.11e-3 | 9.0e-4 | 1.82e-4 | 4.11e-3 | 8.81e-4 | 3.23e-5 |

For axisymmetric and mixed content `pφ` is irrelevant — `20/8` matches `20/20` — and for sectoral
content it is the only thing that matters. But a *random band* contains sectoral content, and there
`20/8` costs a factor of 5.6 against `20/20` (1.82e-4 vs 3.23e-5). So on realistic mixed content, a
cheap `pφ` is not free with local φ interpolation. With global φ it is, because that step is exact at
any order — which makes "use global φ, tune `pθ` only" strictly better than "tune `pθ` and `pφ`
separately", and is exactly the shape the plan's `LocalθGlobalφInterpolationBuilder(; θorder)`
already has.

### Sampling margin buys about as much as stencil order

At fixed case degree 10, varying only the source grid (target stays 2×source):

| source | ρ | p=8 | p=20 |
|:--|--:|--:|--:|
| `NθNφ(13)` | 0.77 | 5.98e-3 | 5.41e-5 |
| `NθNφ(16)` | 0.63 | 1.63e-3 | 2.95e-6 |
| `NθNφ(20)` | 0.50 | 3.65e-4 | 9.24e-8 |
| `NθNφ(26)` | 0.38 | 5.68e-5 | 1.12e-9 |
| `NθNφ(32)` | 0.31 | 1.24e-5 | 2.81e-11 |

Roughly an order of magnitude per margin step at p=8. Going from ρ=0.77 to ρ=0.5 is worth about as
much as raising p from 8 to 14. Both dials are available to a sampling strategy, and a margin costs
grid points in every level's storage while an order costs only apply time — which is the trade Phase 7
of the plan has to price.

### A field tolerance and a matrix tolerance are not the same number

Group 9 reports both, per case, because quoting one as the other is the easy mistake:

- **field `rel_l2`** — the forward interpolation error as a norm over the whole target grid.
- **`pairing`** — what the reverse direction contributes to the scalar a Galerkin test side actually
  computes, `|⟨f_fine, g⟩ − ⟨f_coarse, A g⟩| / |⟨f_fine, g⟩|`, referenced to exact resampling.

`pairing` is consistently **smaller**, and for a structural reason rather than by luck: it contracts the
error field against the test function, so oscillatory parts of the error cancel in an inner product
instead of accumulating in a norm. Measured `pairing / field` across the case set, at both θorder 8 and
20: **0.09 to 0.62**. So a field tolerance is the conservative one and over-states the error reaching the
matrix by roughly 2–10×, while a pairing number quoted as a field number would under-state it.

Neither substitutes for the other, and there is no third "anterpolation accuracy" number to be had:
`src/interpolation/diagnostics/adjointness.jl` sets out why anterpolate-and-compare measures nothing and
why the adjointness residual is a correctness invariant instead of an accuracy curve. `adjointness` is
printed in the same table for exactly that reason — it must stay at round-off (measured 2e-16 to 5e-15),
and if it ever does not, the `pairing` column beside it was read off an operator that is no longer the
adjoint.

### Should the default interpolator be global φ? (group 10)

The evidence for changing `INTERPOLATORECOM`, and the framing is the part that is easy to get wrong.

**Local/global is NOT broadly more accurate.** On random tangential bands — both families mixed, which is
what a far-field pattern actually is — the two families are indistinguishable: measured ratio **0.993 to
1.000** at both θorder 8 and 20, across L = 4, 10 and 20. Its factor-of-7 advantage is confined to *pure
sectoral* modes (`m = l`), which carry little of a full band's energy. This is not a surprise given
group 4/5: θ owns the error for non-sectoral content, so making the φ step exact cannot reduce a total
that θ dominates.

**Nothing regresses.** No ratio exceeds 1.0, combined or per-component — the per-component columns exist
because `Fφ` carries a `1/sinθ` factor `Fθ` does not, so the two could trade against each other under a
flat total. They do not. The `pole share` column (fraction of squared error in the outermost four θ rows
of both hemispheres) stays at 1e-8 or below, so global φ does not push error toward the coordinate
singularity.

**The speed win is real but ORDER-DEPENDENT**, on the tangential apply (2 components, the representation
a Maxwell operator carries):

| θorder | `lagrange` | `localglobal` | speedup |
|--:|--:|--:|--:|
| 8 | 195.3 µs | 202.6 µs | 0.96 |
| 12 | 377.5 µs | 287.7 µs | 1.31 |
| **15 (`INTERPOLATORECOM`)** | **517.2 µs** | **370.3 µs** | **1.40** |
| 20 | 851.2 µs | 526.9 µs | 1.62 |
| 24 | 1.229 ms | 691.7 µs | 1.78 |

The crossover sits between p=8 and p=12: below it, global φ is a wash or marginally slower, because its
FFT φ step costs a fixed amount while the local φ step is cheap at small stencils. At the production
default it is 1.40× faster, and the margin grows with order. **Quoting a speedup without its order is
therefore wrong** — the p=20 figure does not describe the default.

**Allocations differ only in the STANDALONE path.** Constant across order: standalone 230.05 KiB / 17
allocs for `lagrange` against 274.27 KiB / 19 allocs for `localglobal`. Most of that is the output field
both allocate; the difference is the FFT interim. On the PRODUCTION scratch-backed transfer path there is
**no gap at all**: both families, both directions, allocate zero. (An earlier revision of this section
reported 352 and then 224 bytes for `localglobal forward` — see the "Phase 6 baseline" note below for why
that was a property of how the benchmark built its interpolator rather than of the family.)

So the defensible claim is: **a faster default at the shipped stencil with no measured accuracy
regression for the scalar and tangential spherical representations, improving φ-limited sectoral modes,
and allocation-free on the production path.** Not "more accurate", and not "faster" without
qualification. `SphericalVectorFarField` and `PairedSphericalVectorFarField` remain unmeasured.

### The equidistant representation: exact in both angles, and 1.7–2.5× the cost (group 11)

The single largest accuracy result in this file, and the one whose framing matters most, because both
halves are true at once and either one alone is misleading.

`src/equidistant/` resamples **spectrally in both angles** — extend to the torus, FFT, move
coefficients by signed frequency, inverse — where `LocalθGlobalφ` does that in φ only and keeps a local
barycentric stencil in θ. Group 4/5 established that **θ owns essentially the whole error**. Making θ
spectral too therefore ought to collapse the error to round-off, and it does. Same `(Nθ, Nφ)`, same
cases, `rel_l2`:

| case | GL `localglobal` p=15 | **equi FFT** | equi Lagrange p=8 | equi Lagrange p=16 |
|:--|--:|--:|--:|--:|
| `Y_10^0` axisym | 4.56e-4 | **2.65e-15** | 1.08e-3 | 4.99e-6 |
| `Y_10^5` mixed | 4.26e-4 | **2.30e-15** | 5.40e-4 | 2.34e-6 |
| `Y_18^0` axisym | 0.143 | **5.94e-15** | 0.0595 | 0.0118 |
| `Y_26^13` mixed | 0.853 | **5.50e-15** | 0.187 | 0.104 |
| random L=10 | 2.80e-4 | **2.04e-15** | 4.93e-4 | 2.12e-6 |
| random L=26 | 0.340 | **5.44e-15** | 0.156 | 0.107 |

Eleven orders of magnitude at the shipped stencil. The **tangential** table (11b) says the same thing on
the representation a Maxwell operator actually carries — random tangential bands at L = 4, 10, 20, 26 come
out at 9.4e-16 to 5.2e-15 against `localglobal`'s 1.3e-9 to 0.343 — which is also the check that the
equidistant continuation and sign rules across the poles and the φ seam are right, since both components
flip under `SphericalθφInterpolation` and a wrong rule would show here first.

**ρ does not predict this column, and ρi does.** `supporteddegree` is `MLFMA.cutoffparameter`, a
**quadrature** bound: the degree at which the grid still integrates a *product* of two harmonics exactly,
hence ≈ Nθ/2. Interpolation only has to *represent* one harmonic, so its band is the doubled torus's own
Nyquist — about twice that. The Gauss-Legendre family never gets to exploit the gap because its θ step is
not spectral; the equidistant one does, which is why its column stays at round-off out to ρ = 2. It is not
unbounded, and the cliff was measured rather than assumed: at Nθ = 28, `rel_l2` is 1.3e-14 at l = 27, then
**0.688 at l = 28** and 1.41 by l = 29. Exactly `Nθ − 1`. Group 11 prints both ratios so the table cannot
be read as magic.

**What it costs.** Same grids, two-component `TangentialSphericalFarField`, preallocated output,
scratch-backed, minimum of the trial:

| child → parent | GL `localglobal` p=15 | equi FFT | ratio | equi Lagrange p=10 |
|:--|--:|--:|--:|--:|
| 18×36 → 26×52 | 43.2 µs | 71.1 µs | 1.7× | 279.6 µs |
| 26×52 → 40×80 | 80.2 µs | 141.8 µs | 1.8× | — |
| 40×80 → 64×128 | 164.0 µs | 404.3 µs | **2.5×** | 1.625 ms |

The equidistant FFT column has moved twice. It was 122.4 / 260.1 / 782.1 µs with the doubled θ torus
(#107), then 86.2 / 171.9 / 479.9 µs once the doubling was dropped, then the figures above once the φ
resample moved between the two θ transforms (#111). Cumulatively **1.72× / 1.83× / 1.93×**.

**The ratio still grows with the grid**, and that part is structural rather than a missing
optimization: the equidistant FFT is a global spectral transform in both angles, while `LocalθGlobalφ`
pays a local θ stencil plus 1-D φ FFTs. Different asymptotics, and no amount of constant-factor work
changes that. At the file's own grid pair (28×56 → 54×108) the forward apply is 141.5 µs against 338.2 µs
and the reverse 184.8 µs against 321.2 µs — note the equidistant family is **symmetric** between
directions where the local one is not (141 vs 185), which is what a spectral operator that is its own
structure in both directions should look like. All three are now allocation-free. The Lagrange backend was the last holdout at 420 KiB / 1954 allocs
per apply; #112 hoisted its stencils — which depend only on the two grids and the order, and were being
rebuilt identically every apply — into a scratch. **That bought the allocation property and almost no
time: 1.03–1.13×.** Worth recording as a negative result rather than dressed up, because it says the
per-apply stencil rebuild was never the cost. Its value is the 0-allocation guarantee under threading,
where GC pressure serializes across chunks and the single-threaded figure understates it.

**Where its time actually went, found by profiling rather than by reasoning about it: 95% of the
kernel's samples were inside `sphericalget`/`reflectindex`, and 4.2% in the multiply-add they serve.**
The θ stage resolved the spherical continuation *per tap, per output point* — `reflectindex` runs a
`while` loop and a `mod1` division to walk an out-of-range θ index back onto the grid. None of that
depends on the data: for a fixed stencil and grid it is a constant, the same kind the weights are.
Precomputing the resolved row, the φ half-turn and the sign (`ReflectedStencils`) leaves an inner loop
of two table reads and a multiply-add:

| | before | after | |
|:--|--:|--:|--:|
| 18×36 → 26×52 | 279.6 µs | 122.1 µs | 2.29× |
| 28×56 → 54×108 | 1107.9 µs | 437.3 µs | 2.53× |
| 40×80 → 64×128 | 1625.1 µs | 691.5 µs | 2.35× |

Against the original allocating path that is **2.4–2.6×**, and it stays bitwise, because the sign is
`±1` and `w * (sgn * v)` is the same Float64 as `(w * sgn) * v` — which is the only reason the sign is
allowed to leave the inner loop and fold into a second weight table.

**The two directions then wanted opposite loop nests, which is worth knowing because symmetry is the
tempting default.** The `(a, i′)`-indexed tables are loop invariants across `j′`, so hoisting `a`
outward makes them free — and that is right for the *adjoint*, whose θ stage is a scatter with no
register to accumulate into. Applied to the *forward*, the same change replaces one store to
`out[j′, i′]` with `p` read-modify-writes and measured **7–9% slower**. Register accumulation beats
loop-invariant hoisting when there is a register to accumulate into.

Keeping `(i′, j′, a)` forward and `(i′, a, j′)` adjoint:

| | after θ precompute | after loop order | |
|:--|--:|--:|--:|
| reverse, 18×36 → 26×52 | 180.5 µs | 124.7 µs | 1.45× |
| reverse, 28×56 → 54×108 | 717.4 µs | 455.3 µs | 1.58× |
| reverse, 40×80 → 64×128 | 1060.9 µs | 714.9 µs | 1.48× |

The directions are now balanced — 437.6 µs forward against 455.3 µs reverse — which is what a transpose
pair should look like, and they were not before.

**The adjoint gives up bitwise agreement to get this**, deliberately: it now matches its allocating
reference to 2–6e-17, under one ulp, because two taps of one stencil can resolve onto the same source
row (a stencil reaching past a pole reflects back onto rows it already covers) and the reorder changes
which arrives first. Floating-point addition is not associative. The *forward* direction keeps its
bitwise equality, and that is the one that pins the stencils and continuation tables as exactly the
reference's — so the property is not lost, only narrowed to where it is load-bearing.

**Then the φ stage turned out to be hiding behind θ.** With θ fixed, re-profiling put **87.9% of the
forward kernel in the φ stage** — specifically `mod1(left + b, Nφ)`, an integer division per tap per
output point, wrapping the periodic φ stencil. Same class of constant as the θ tables and the weights:
it depends on the grids and the order, never on the data. Precomputing it is the single largest step
of the three.

| forward | original | +stencils (#112) | +θ precompute | +loop order | +φ wrap |
|:--|--:|--:|--:|--:|--:|
| 18×36 → 26×52 | 317.2 µs | 279.6 | 122.1 | 122.0 | **56.4** |
| 28×56 → 54×108 | 1158.9 µs | 1107.9 | 437.3 | 437.6 | **224.9** |
| 40×80 → 64×128 | 1673.8 µs | 1625.1 | 691.5 | 691.3 | **331.2** |

| reverse | original | +θ precompute | +loop order | +φ wrap | +wrap split, φ gather |
|:--|--:|--:|--:|--:|--:|
| 18×36 → 26×52 | 263.7 µs | 180.5 | 124.7 | 69.0 | **45.1** |
| 28×56 → 54×108 | 1033.5 µs | 717.4 | 455.3 | 268.0 | **173.0** |
| 40×80 → 64×128 | 1528.7 µs | 1060.9 | 714.9 | 475.2 | **252.0** |

**5.0–5.6× forward and 5.9–6.1× reverse against the original.** Every step after the first was the same
finding wearing a different hat: *per-element work that does not depend on the data*, sitting in an
innermost loop and reading as arithmetic. Each one was invisible until the one above it was removed,
which is the argument for re-profiling after every step rather than planning the sequence up front.

The last two both landed on the adjoint. The θ wrap `jj > Nφ && (jj -= Nφ)` was a compare-and-subtract
per tap, but the shift is only ever 0 or Nφ/2 — so `j′` splits into two contiguous branch-free ranges,
and each `j′` still writes a distinct cell, so no accumulation order changes. And the φ stage became a
**gather over a transposed stencil** (`LagrangeTranspose`, CSR) instead of a scatter over the forward
one: register accumulator, one store per output cell, no write hazard, no `fill!`. That one is
bitwise-neutral, because the scatter visits `(j′, b)` in ascending order and the transpose fills each
bucket in that same order — unlike the θ reorder, where a shift makes the arrival order column
dependent. Same class of change, opposite answer on exactness, and the difference is worth keeping
straight.

**Where it stops.** The forward leaves **6 leaf samples** in the whole kernel and the reverse 70, now
flat across the remaining loops with no single line above 17%. Both are bounded by memory traffic
rather than by anything addressable in the loop, and the reverse is now the *faster* direction
(173.0 µs against 224.9 µs) having started as the slower one. I do not have another mechanism to
propose; anything past here is vectorisation or blocking, with no evidence yet that either would pay.

**None of which changes the strategic picture.** At equal tolerance the Lagrange backend was ~29× the
equidistant FFT family and is now ~5×, and roughly at parity with the GL local/global family rather
than 6× behind it. That is a real change in its standing, but not enough to make it the production
choice — the equidistant FFT family still wins on both axes at once. It remains a comparison baseline
and a locality demonstration, now an honest one: measured rather than assumed slow.

**Neither axis decides this alone.** A slower kernel that gains eleven digits is no more a regression than
a faster one that loses a digit is an improvement, which is the whole reason this file keeps the two
tables apart. What the pair actually says is that the comparison at *equal grid size* is the wrong one for
a production decision — see group 12, which asks the right one and reverses the answer.

**Two FFT shapes were measured against the current one** (raw FFTW plans at 56×56 → 108×108, 2000
samples), because "it is slower" is only actionable with a direction:

| shape | time | speedup |
|:--|--:|--:|
| 2-D forward on child + 2-D inverse on parent (**current**) | 144.5 µs | — |
| fully separable 1-D passes, resampling interleaved, θ first | 126.9 µs | 1.14× |
| same, φ first | 131.3 µs | 1.10× |
| 2-D on a **non-doubled** θ grid | 71.2 µs | **2.03×** |

The 2-D choice was right — FFTW implements a 2-D transform as passes over each dimension anyway, and
the only reason the separable form wins at all is that interleaving the resampling lets each dimension's
*inverse* run while the other is still at child length. That is worth 14%, and has not been taken.

**The doubling was the one to take, and it has been** — the θ direction now runs as a DCT-II/DST-II pair
on `Nθ` points instead of a complex FFT on `2Nθ`, because the extended torus's second half is built by
reflection and is therefore redundant. On its own that delivered **1.38–1.63×** against a probe that
promised 2.03×, because the probe timed bare FFTW plans and so measured a *ceiling*: the real kernel
also pays a pack/unpack pass for the ε split, and at that point the φ resample still ran first and did
not halve.

Moving the φ resample **between the two θ transforms** (#111) then added a further **1.19–1.21×**, and
the interesting part is why — it is mostly not the smaller θ transform. Running the resample on the
packed real blocks replaced `resamplecoefficients!` called once per θ column, each with its own `fill!`
and frequency walk, by a single frequency walk copying whole rows. The stage that looked like a
data-movement detail was a per-column function call in disguise.

Together the two reach **1.72–1.93×**, which lands about where the probe said — so the probe was a fair
estimate of the *combined* achievable win and a poor one for either step alone. That is the lesson to
carry: a shape probe on isolated transforms picks a direction, it does not predict a speedup, and it
says nothing at all about which stage will turn out to dominate.

### Cost at equal tolerance — where the equal-grid comparison reverses (group 12)

Group 11 is measured at equal `(Nθ, Nφ)`, and on that axis the equidistant FFT family looks 1.7–2.5×
*more expensive*. That comparison charges it for accuracy nobody asked for. A sampling strategy does not
choose a grid; it chooses a **tolerance**, and the grid follows — so group 12 fixes the content's
bandwidth (13, tangential), sweeps the *requested degree* that `MLFMA.NθNφ(D)` turns into a grid, and
finds the coarsest grid each family needs to reach a target ε.

| ε | best GL family | equi FFT | ratio |
|:--|--:|--:|--:|
| 1e-3 | 251.7 µs — p=15 at Nθ=34 | 55.0 µs at Nθ=14 | **4.6×** |
| 1e-6 | 549.5 µs — p=20 at Nθ=46 | 55.5 µs at Nθ=14 | **9.9×** |
| 1e-9 | 1479.4 µs — p=20 at Nθ=70 | 55.4 µs at Nθ=14 | **26.7×** |
| 1e-12 | *not reached at any swept degree* | 55.4 µs at Nθ=14 | — |

**The sign flips and the margin grows with the tolerance.** The equidistant FFT column is flat because it
reaches every one of these tolerances on the *same* grid — Nθ=14, the coarsest that represents degree-13
content at all. It is exact there, so tightening ε costs it nothing. The local family pays for each
digit with resolution.

That gap is the one group 11 already named: `cutoffparameter` is a **quadrature** bound (≈Nθ/2), but
interpolation only has to *represent* one harmonic, and the equidistant grid's representation band is
Nθ−1. A family that exploits the gap runs at roughly half the requested degree; one that does not, cannot.

**The table above is an upper bound, and the binding constraint is elsewhere — measured since.**

The equidistant column runs at Nθ=14 because that is all *interpolation* needs. A full matvec is not
allowed to. Translation is a pointwise product with `T_L`, and a product has twice the bandwidth of its
factors — which is precisely why the quadrature rate is 2(L+1). Sweeping the rate through the whole
two-level loop (aggregate → translate → disaggregate → integrate) locates the constraint exactly, and
`test/Equidistant/twolevel.jl` now pins it:

| what varies | behaviour |
|:--|:--|
| **parent** grid — where the product lands | truncation-limited (2.7e-10) at Nθ=26, 9.6e-8 at 20, **1.8e-2 at 13** |
| **child** grid — only interpolated *from* | 2.7e-10 unchanged all the way down to Nθ=13, and still 2.7e-10 at 10 |

So the level a product lands on needs the quadrature rate; a grid that is only interpolated *from* may
sit far below it. Every level translates in MLFMA, so this is a floor on every level — and the earlier
guess that interior levels could simply halve their rate is **wrong**, for exactly this reason.

Re-running the comparison with that floor imposed (each family at `max(translation floor, its own
interpolation requirement)`) gives the number that actually applies:

| ε | equi FFT at the floor | best GL, interpolation-limited | ratio |
|:--|--:|--:|--:|
| 1e-3 | 52.3 µs at Nθ=16 | 251.7 µs at Nθ=34 | **4.8×** |
| 1e-6 | 117.0 µs at Nθ=20 | 549.5 µs at Nθ=46 | **4.7×** |
| 1e-9 | 227.3 µs at Nθ=26 | 1479.4 µs at Nθ=70 | **6.5×** |

Still decisively in the equidistant family's favour, and now defensible — but 4.7–6.5×, not 4.6–26.7×.
The floor binds the spectral family and not the local one, because GL's interpolation requirement
(34/46/70) already exceeds it while the spectral family's (14) does not. The 1e-12 row is gone: at L=12
the method is truncation-limited at 2.7e-10, which no grid fixes.

*Critical sampling has no margin.* Nθ=14 for degree-13 content is exactly the representation limit, and
the failure past it is a cliff, not a slope — measured at 0.688 for the first unrepresentable degree.
The local family degrades gracefully; the spectral one does not. Production would want headroom, and
headroom is cheap here (the FFT column is still 2.4e-15 at Nθ=18), but it is not free and it is not
optional.

**A harness note, because it nearly went in as a result.** The GL rows were first timed through the bare
`localglobal(p)` constructor this file uses for accuracy work, which stores no per-level information and
so rebuilds it every apply. That reported p=20 at 10.4 ms on Nθ=30 and 8.9 ms on Nθ=70 — *faster on a
grid 2.3× larger*. A cost that falls as the grid grows is a broken harness, not a fast kernel, and the
non-monotonicity is the only thing that gave it away. Group 12 builds GL through `buildinterpolator`,
as the map constructors do.

### Can pθ and pφ differ? By one order, worth 2–9% (group 13)

`LagrangeBackend{Pθ,Pφ}` has always taken two orders and every caller has always passed one number to
both. Group 6 asks this of the Gauss-Legendre family, where θ nodes are non-uniform and θ is the
limiting direction; on the equidistant grid **both** coordinates are uniform, so that result does not
transfer and the question is open.

Sweeping both orders independently (tangential L=10, 28×56 → 54×108, `rel_l2`):

| | pφ=4 | pφ=6 | pφ=8 | pφ=10 | pφ=12 | pφ=16 |
|:--|--:|--:|--:|--:|--:|--:|
| pθ=4 | 0.00998 | 0.00876 | 0.00869 | 0.00869 | 0.00869 | 0.00869 |
| pθ=8 | 0.0045 | 0.000986 | 0.000489 | 0.000451 | 0.000448 | 0.000448 |
| pθ=12 | 0.00447 | 0.000875 | 0.000195 | 5.38e-05 | 3.07e-05 | 2.85e-05 |
| pθ=16 | 0.00447 | 0.000874 | 0.000193 | 4.56e-05 | 1.14e-05 | 2.1e-06 |

**Both directions saturate, and θ is the harder one** — its floor at pθ=4 is 0.00869 against φ's 0.00447
at pφ=4. So the GL family's answer does carry over after all, on a grid where the reason for it does
not: it is not the node distribution, it is that the field varies faster in θ.

The tension is that θ is *also* the more expensive order. The stages are separable, so the tap count is
`Nθ_dst·pθ + Nθ_src·pφ` per φ column — here 54·pθ + 28·pφ, making a θ order 1.9× the cost of a φ one.
θ sets the floor so its order cannot come down; the only saving available is trimming φ to where it
stops contributing. (Cost model checked, not assumed: measured across ten pairs it holds at 2.23–2.94
µs per 1000 taps, mostly 2.5–2.6.)

**How far apart may they be? Exactly the ratio of the two floors, and that ratio is small.** Sweeping
each order with the other maxed out:

| p | θ floor | φ floor | ratio |
|--:|--:|--:|--:|
| 4 | 8.69e-3 | 4.47e-3 | 1.94 |
| 8 | 4.48e-4 | 1.93e-4 | 2.32 |
| 12 | 2.85e-5 | 1.14e-5 | 2.50 |
| 14 | 7.46e-6 | 3.46e-6 | 2.16 |
| 16 | 2.10e-6 | 2.10e-6 | 1.00 |

One order step buys about **2.1×**. The θ/φ ratio peaks at **2.5** — barely more than a single step. So
φ has roughly **one order** of slack and no more; dropping it two costs ~4× accuracy to save ~6% of the
work, which is never the right trade. (The ratio falls back to 1.0 by p=16 because both directions have
hit a floor that is no longer the stencil's — the grid's.)

| target ε | cheapest pair | Δ | taps | cheapest pθ = pφ | saving |
|:--|:--|--:|--:|--:|--:|
| 5e-3 | (5, 5) | 0 | 410 | 410 | — |
| 1e-3 | (8, 6) | 2 | 600 | 656 | **1.09×** |
| 1e-4 | (11, 10) | 1 | 874 | 902 | **1.03×** |
| 1e-5 | (14, 13) | 1 | 1120 | 1148 | **1.02×** |

So: **yes, by one order, worth 2–9%**, shrinking as the tolerance tightens and zero at the loosest.

**An earlier version of this section claimed 6–13% and a two-order gap. That was an artifact of the
sweep.** It stepped the orders by 2, which forced every reported optimum to differ by 2 and — worse —
compared it against a symmetric pair that was itself unnecessarily coarse, inflating the saving from
both ends. At unit resolution the gap is 1 and the saving roughly halves. A parameter sweep whose grid
is coarser than the effect being measured does not merely lose precision; it can manufacture the result.

Worth knowing mostly because the intuition that a tensor-product stencil should be square is *nearly*
right, and fails by about one order — for two independent reasons pulling oppositely, since θ needs the
higher order and also costs more to raise.

### The reverse path, decomposed — the Phase 6 baseline

Disaggregation runs inside every *forward* matvec, and anterpolation costs ~1.6× interpolation at the
shipped default. The `"reverse split"` timing group splits that so an optimization is aimed by
measurement. Anterpolation runs parent → child and resamples θ and φ separably, so `MIDDLE` (the target's
`Nθ` with the source's `Nφ`) separates them, exactly as it does for interpolation.

The two sub-kernels are timed **directly**, one component each, so `2 × (θ + φ)` must reconstruct the
two-component apply. It does, to 99.6–99.7%:

| θorder | interp. apply | anterp. apply | θ kernel | φ kernel | 2×(θ+φ) |
|--:|--:|--:|--:|--:|--:|
| 8 | 208.8 µs | 385.1 µs | 114.0 µs | 77.8 µs | 383.5 µs |
| **15** | **372.0 µs** | **589.4 µs** | **216.1 µs** | **77.6 µs** | 587.4 µs |
| 20 | 527.0 µs | 735.5 µs | 289.1 µs | 77.7 µs | 733.6 µs |

- **The θ kernel is the target:** 74% of the reverse cost at the shipped order, and the part that scales
  with the stencil. The φ kernel is flat across order, which is right for an FFT.
- **The cause is measured, not guessed.** The θ kernel's `mul!` arithmetic is only ~11 µs of its 216 µs
  (56 calls at ~199 ns). The cost is in its NONCONFORM branch, which reads
  `view(fvals, flippingφindices, indices)` — rows indexed through a permutation `Vector{Int}`, hence not
  strided, so `mul!` takes a generic path at **4.23 µs** against **211 ns** for the contiguous
  `view(fvals, :, indices)`. A 20× penalty on 28 of the 56 calls; 28 contiguous + 28 fancy accounts for
  125 µs of the 216 µs.
- **The fix this predicted has since been made**, in both directions: `flippingφindices` is
  `vcat(Nφ÷2+1:Nφ, 1:Nφ÷2)`, a circular shift, so the fancy view decomposes into two *contiguous* row
  blocks — one 4.23 µs call becomes two ~211 ns ones. The same pattern was in the interpolator's
  `_interpolateθ!`. Results in the next section.
- A scratch-backed anterpolation apply allocates **nothing**; the interpolation row's 19 allocations are
  the three-argument form allocating its own output field.

**Two measurement traps, both hit while producing this table.** The first version timed whole applies over
an intermediate grid, the way the *accuracy* split does — but an anterpolation apply always runs its φ
FFT even when `Nφ` does not change, so both "stages" double-counted it. The tell was mild and I misread
it: φ + θ came to 1.15–1.28× the full apply, which reads as ordinary per-call overhead. Those numbers
(φ ~191 µs flat, θ 303–654 µs, and a claimed "~191 µs floor") were committed in `ba7982a` and are wrong.
Second, and kept because it is real: interpolation information — FFT plans included — is cached keyed by
`(child, parent)`, so applying an operator to a pair it was not built for rebuilds that information and
measures plan construction; done that way the φ stage read **12.8–29.5 ms** against a 0.39–0.74 ms apply.
Both point the same way: **a decomposition is not to be trusted until its parts are shown to sum to the
whole.** This group now asserts that by construction, which is why the `2×(θ+φ)` column is printed.

### Where both θ kernels ended up

The table above is the **pre-optimization baseline** and is kept as that. What the three θ-kernel commits
actually delivered, at the shipped stencil 15 on the same grid pair:

| | θ kernel (1 cmp) | φ kernel (1 cmp) | apply (2 cmp) |
|---|--:|--:|--:|
| anterpolation, before | 216.1 µs | 77.6 µs | 589.4 µs |
| anterpolation, after | **13.9 µs** | 77.4 µs | **185.2 µs** (3.2×) |
| interpolation, before | 131.0 µs | 54.0 µs | 370.0 µs |
| interpolation, after | **18.4 µs** | 55.0 µs | **146.8 µs** (2.5×) |

Two defects, both about which `mul!` kernel the θ product reaches, and both present in both directions:
`Vector{Int}` column indices (non-strided view, generic path, 4.19 µs against 212 ns) and rows indexed
through the `flippingφindices` permutation (same problem, 4.23 µs against 289 ns as two contiguous
blocks). The interpolator carried a third: `SVector{stencilsize}` conform weights, 878 ns against 199 ns
for a `Vector`, which is the opposite of the usual static-array intuition and is why it is written down.

**The φ columns are the control.** They are unchanged across every row, which is what says the speedup came
from θ and not from a harness difference. Both directions are now at their floor — φ is 84% of what remains
in the reverse path and ~75% in the forward one — so further θ work would chase single-digit percentages.
Retiring the FFT's Nyquist-bin asymmetry is the separate research item that would move φ.

Accuracy is unaffected beyond round-off. Verified two ways over a sweep spanning the small grids whose
index sets contain gaps: all 120 interpolation matrices bitwise identical, and 122 dense random-field
applies agreeing to a worst relative difference of **5.3e-16**. The dense check is the one that matters —
a unit-impulse matrix column has one nonzero term per inner product, so it cannot see a change in
summation order, and the rotation split does change it.

### Two housekeeping notes on the sweep

- **`rel_l2` bottoms out near 1e-14 and then rises.** `Y_2^0` runs 1.5e-14 at p=16, 1.29e-14 at p=20,
  6.5e-14 at p=24. That is the round-off/conditioning floor of high-order barycentric weights, not a
  bandwidth effect. Do not read improvements below ~1e-13 as real.
- **The knee at ρ = 1 is sharp.** At p=8: 5.3e-8 at ρ=0.15, 1.8e-4 at ρ=0.46, 6.0e-3 at ρ=0.77,
  4.2e-2 at ρ=1.08, 0.33 at ρ=2.0. Two decades of degradation across the last 30% of the band.

## Runtime

Full `--timings` output is reproducible with the command at the top; a runtime win is not an
improvement on its own, and the accuracy sections above are the other half of every judgement here.
Measured on one machine, 28×56 → 54×108, `ScalarFarField{ComplexF64}`, minimum of the trial:

| apply | p=8 | p=12 | p=20 |
|:--|--:|--:|--:|
| `lagrange full` | 104 μs | 174 μs | 429 μs |
| `lagrange θ stage` | 76 μs | 130 μs | 316 μs |
| `lagrange φ stage` | 93 μs | 145 μs | 350 μs |
| `localglobal full` | 104 μs | 146 μs | **264 μs** |

Interpolation-information construction is a separate group, because it happens once per
(level, sampling pair) in a real core while apply happens on every matvec: `lagrange` 15.5 ms /
8.5 MiB at p=8 and 36.2 ms / 20.4 MiB at p=20; `localglobal` 14.3 ms / 5.0 MiB and 33.0 ms / 11.2 MiB.

### The level transfer, per direction — what the transposed wrapper costs

The `"transfer"` group applies whole `InterpolationPhaseTransfer`s through H2Factory's own direction
path, on a two-component `TangentialSphericalFarField` with scratch-backed context. `TangentialSpherical`
and `Overwrite` are chosen so the numbers describe the traversal; the interesting quantity is a
*difference*, not any single row.

| p | family | forward | transpose | adjoint | transpose − adjoint |
|--:|:--|--:|--:|--:|--:|
| 8 | `lagrange` | 230.0 µs | 735.9 µs | 741.1 µs | −5.2 µs (noise) |
| 8 | `localglobal` | 287.3 µs | **531.4 µs** | 496.8 µs | +34.6 µs (**+7.0%**) |
| 20 | `lagrange` | 1.003 ms | 1.882 ms | 1.896 ms | −14 µs (noise) |
| 20 | `localglobal` | 668.8 µs | **858.5 µs** | 826.5 µs | +32.0 µs (**+3.9%**) |

Both reverse directions do identical work apart from which operator applies — the anterpolator (adjoint)
or `transposedinterpolator` (transpose). For `lagrange` those are the same object, so its two rows
agreeing to noise is the **control** that makes the `localglobal` difference attributable to the wrapper
alone.

- **`TransposedInterpolator` costs ~33 µs, and that is nearly constant** — three in-place conjugation
  passes over the moments, O(N) and independent of stencil order — so its relative cost *falls* as order
  rises: 7.0% at p=8, 3.9% at p=20.
- **It is affordable as a bridge.** `localglobal` transpose *including* the wrapper is 1.39× faster at
  p=8 and 2.19× faster at p=20 than the `lagrange` default it would replace. A native transposed FFT φ
  kernel would recover the 4–7%, but nothing about a default flip has to wait for it.
- **The reverse direction costs far more than the forward one**, and the asymmetry is much worse for the
  local family: 3.2× (`lagrange` p=8, 741 vs 230 µs) against 1.7× (`localglobal`, 497 vs 287 µs). Since
  disaggregation runs inside the *forward* matvec, that cost is paid on every product — which is why the
  reverse path was optimized first. **These numbers predate the θ-kernel work**; both directions have since
  been reduced (see "Where both θ kernels ended up"), and at the shipped order the remaining asymmetry is
  1.26× rather than 1.6×. This group has not been re-run at the new baseline, so read the rows above as the
  relative comparison they were built for, not as current absolute costs.
- **`localglobal forward` allocated 352 bytes / 2 allocations per apply**, where every other row in this
  group is allocation-free. Small, but it is per transfer per matvec, so it scales with the tree. The
  figure is FIXED-SIZE — identical at Nθ 8, 14, 28 and 42 and at orders 4 and 15 — so it is bookkeeping,
  not anything proportional to the data. **Now 224 bytes**, and both halves are worth recording:
    - *Fixed (128 bytes).* The apply built its per-component view container as
      `SVector{D}([view(…) for i in 1:D])` with `D = length(eachindex(farfield))` — a RUNTIME value, so the
      `SVector` type was runtime-dependent and uninferable, and the comprehension materialized a `Vector`
      first. `fielddim` reads the component count off the far field TYPE, so `ntuple(…, Val(fielddim(F)))`
      is a compile-time-sized tuple of views needing no heap. Verified that `fielddim` equals the runtime
      count for all five representations before relying on it.
    - *The other 224 bytes were never a production cost, and this is the correction.* The mechanism
      recorded here was right as far as it went: `converttrait` rebuilds a tuple of two
      `LocalθGlobalφInterpolationInformation` structs and it escapes into the φ stage. The conclusion
      drawn from it — that removing them "touches both families and every trait" — was wrong.
      `converttrait(::T, v, ::T)` is the **identity**, so an interpolator whose stored trait matches the
      one its applies request converts nothing and allocates nothing. The tuple is built only when
      converting *from* `DefaultInterpolation`, and that happened only because **this benchmark and the
      allocation test** built their interpolator through `buildinterpolator` without passing
      `interpolationtrait`, while both map constructors pass the operator's concrete trait. Measured, same
      grids, forward apply: trait defaulted 224 B, trait given **0 B**. All four scratch-backed applies
      are now pinned at exactly 0 in `test/LevelTransfers/interpolation.jl`, with a companion testset
      asserting the defaulted path still pays, so the figure stays attached to its real cause. `@inline`
      on `interpolationinformation` was tried and changed nothing, which is consistent with the escape
      never having been the point.

      **The general lesson outlives the bytes.** A benchmark that constructs its own operand can measure
      a cost the real caller never pays, and it looks exactly like a genuine finding — this one survived
      as a documented "known, unfixed" item through two attempts to remove it and a task that called for
      redesigning the trait plumbing. Build the object the way production builds it, or the measurement
      is of the harness.
- **A LATER REGRESSION AND WHAT IT EXPOSED, worth keeping here.** Putting the forward θ kernel on the fast
  `mul!` path made `lagrange forward` allocate 440 times / 26 KB per apply, and the full suite passed. Two
  reasons it got through. (1) That family writes strided ROW views of its interim matrix, and a strided
  `mul!` destination forces a temporary once the other operands are BLAS-eligible — so the same change that
  is free for the column-writing family is not free here. The fix is the one `_anterpolateθ!` always used:
  compute into a contiguous vector and copy the row, which also made it FASTER than before the whole
  exercise (625.8 → 363.4 µs, still 0 allocations). (2) The only absolute allocation pins covered the
  default family, and `protocol.jl`'s contract measures the transfer layer's overhead RELATIVE to the bare
  kernel — when the kernel itself starts allocating, both sides rise together and the delta stays in
  budget. There is now an absolute per-family, per-direction pin in
  `test/LevelTransfers/interpolation.jl`.

Reading it:

- **Global φ is free at low order and a discount at high order.** `localglobal` ties `lagrange` at
  p=8 (104 vs 104 μs) and is **1.6× faster at p=20** (264 vs 429 μs) — while being *exact* in φ rather
  than approximate. The FFT φ step does not get more expensive with `p`; only the θ step does. On both
  axes at once, at p=20, it simply wins.
- **The stage timings do not add up, and must not be read as if they did.** Each is a separate call
  with its own output field, and the φ stage runs on the larger middle grid, so 76 + 93 ≠ 104 at p=8. A
  fused `pθ != pφ` kernel would cost less than the sum. What the θ-stage row *does* give is a lower
  bound: any local-θ scheme at pθ=20 pays at least 316 μs, which already exceeds `localglobal`'s 264 μs
  for the whole job. That is the cost argument against pursuing the order split ahead of global φ.
- **A Phase 6 candidate, stated as a hypothesis.** `localglobal full` at p=20 (264 μs) beats even
  `lagrange`'s θ stage alone (316 μs), though both do the same amount of θ work. The two θ kernels
  differ in memory access: `_interpolateθtransposed!` writes *row* views of a transposed interim
  (`view(interimmatrix, iteration, :)`), while `_interpolateθ!` writes *column* views
  (`view(interimmatrix, :, iter)`). That points at the transposed interim as the cost, but this
  measurement does not isolate it — chaining `localglobal` onto the middle grid would add an FFT/IFFT
  round trip rather than measuring its θ step alone. It needs its own experiment before anyone rewrites
  a kernel on the strength of it.
- **13–15 allocations and 95–183 KiB per apply**, all of it the standalone path's output field and
  fallback interim. This is the direct-call path, not the workspace one; whether the production
  level-transfer path allocates is Phase 5's question and is not measured here.

### A third measurement trap: zeros are not a neutral input

Recorded with the other two because it cost a wrong table and the file's own `preparedfield` comment had
already warned about it. Group 11's fixture initially built its moments with `Val(:zero)` and never filled
them. The equidistant Lagrange backend then reported anterpolation at **131 µs against interpolation's
1.166 ms** — a 9× forward/reverse asymmetry in the direction no other family shows, with identical
allocation counts on both sides. Filling the inputs with a real tangential band made it 1.139 ms against
1.166 ms, i.e. symmetric, and the asymmetry disappeared entirely.

The tell was that the number was *suspiciously good* in a direction with no mechanism to be good, which is
the same shape as a silent no-op — so the first thing checked was that the reverse apply produced non-zero
output at all. It did. **An all-zero operand changes which arithmetic the hardware performs, so it is not a
neutral choice of input; it is a choice that measures a different kernel.**

## Should the equidistant FFT family become `INTERPOLATORECOM`? No — and group 12's premise is why

**Verdict: do not promote. On a real matvec the equidistant FFT family is 1.07–1.74× SLOWER than
`LocalθGlobalφ` at every ϵ, at identical accuracy.** Measured on the `pipeline.jl` fixture (tworects,
κ=2π, minhalfsize = rootsize/5) against the dense reference:

| ϵ | GL p=15 | equi FFT | ratio | GL error | equi error |
|:--|--:|--:|--:|--:|--:|
| 3e-1 | 1.090 ms | 1.171 ms | 1.07× | 2.3618e-2 | 2.3528e-2 |
| 1e-1 | 1.296 ms | 2.253 ms | 1.74× | 1.0155e-2 | 1.0184e-2 |
| 3e-2 | 1.968 ms | 2.777 ms | 1.41× | 9.5291e-3 | 9.5277e-3 |
| 1e-2 | 2.053 ms | 2.269 ms | 1.11× | 9.5305e-3 | 9.5295e-3 |
| 1e-3 | 2.433 ms | 3.784 ms | 1.56× | 7.1907e-3 | 7.1906e-3 |
| 1e-5 | 3.520 ms | 5.530 ms | 1.57× | 6.8653e-3 | 6.8661e-3 |

Build times are equal (~0.3 s both; the first row of any run is compilation, not construction).

**Why the 4.7–6.5× of group 12 does not survive.** Three facts compose:

1. **The sampling rule is family-independent.** `SamplingStrategy(; ϵ)` maps ϵ → L → `(2(L+1), 4(L+1))`,
   and both families receive *identical* grids — verified per level: at ϵ=1e-2 both get
   Nθ = [50, 30, 20, 12] for L = [24, 14, 9, 5]. At equal grid the spectral family is 1.7–2.5× more
   expensive (group 11), and that is what the matvec shows, diluted by the near-field and translation
   stages.
2. **There is no tolerance headroom to trade.** The end-to-end error is floored near 7e-3 by the near
   field and the mesh from ϵ=1e-2 downward, so tightening ϵ buys nothing and production runs coarse.
3. **And coarsening does not favour the spectral family either** — the decisive check. Sweeping ϵ up to
   3e-1, the two families' errors track each other to 3 significant figures the whole way, and at
   ϵ=1e-1 the spectral family is marginally *worse*. Its exact interpolation never becomes the
   advantage, because **interpolation error is not the limiting term at any production ϵ**: the error
   is set by the truncation of `L`, both families use the same `L`, so both truncate identically.

**That is the correction to group 12's premise.** Group 12 fixed the content bandwidth and swept the
grid to find each family's *interpolation* limit, which is a regime the production pipeline never
enters — there the grid is chosen by `L`, and `L` is chosen by the translation truncation. A family
that interpolates exactly wins only where interpolation is the bottleneck, and in this pipeline it
never is. The 4.7–6.5× is a real property of the kernels and a real answer to the question group 12
asked; it is simply not the question production asks.

**What would change the answer.** A larger or higher-frequency problem, where the far-field stages
take a larger share of the matvec and the near-field floor sits lower, changes the dilution factor and
could make ϵ matter — this is one fixture, and the conclusion is stated for this regime. The other
route is a family-aware sampling rule that lets the spectral family run below `2(L+1)`; #113 measured
its two-level loop reaching 1e-3 at Nθ=16 where `2(L+1)`=28. But that headroom is bounded by the
*translation* floor, not by interpolation, so it is available to whatever family can tolerate it —
and the check above suggests the end-to-end error stops improving before it can be exploited.

## sinθ ring sampling (#118): the storage saving does NOT convert into runtime

**Verdict: the ragged ring path is 5–7× SLOWER than the tensor path, and the largest term is
structural rather than an implementation defect.** Measured at kernel level (no tree — see the
baseline caveat below).

A single ragged-vs-tensor number confounds two opposite effects, so it is decomposed:

- **A = tensor / ring-uniform** — the ring kernel on a *uniform* ring grid against the tensor kernel.
  Same points, same arithmetic (the local pair is bitwise), so this is pure bookkeeping overhead.
- **B = ring-uniform / ring-ragged** — the same ring kernel on a ragged grid. This isolates what the
  smaller rings actually buy.

| backend | A (overhead) | B (raggedness) | NET = A·B |
|---|--:|--:|--:|
| Lagrange p=6 | 0.435 | 0.438 | **0.191** |
| Lagrange p=10 | 0.356 | 0.412 | **0.147** |
| spectral (#117 reference) | 0.153 | 1.013 | **0.154** |

Accuracy is matched throughout (ragged tracks uniform to ~4 %), so these are like-for-like.

**B < 1 is the finding: raggedness makes the local kernel SLOWER, by ~2.4×.** The width-grouped
factorization builds one `n × Nθ_src` φ-stage per *distinct destination width*, so the φ stage costs
`Σ_{distinct n} n` where the uniform grid pays `maxNφ` once — measured 2.50 / 3.56 / 2.92× at parent
grids (24,48) / (36,72) / (48,96). The θ stage does shrink, to `fill` ≈ 0.84. The product is worse.

**`Σ_{distinct n} n > maxNφ` whenever there is more than one width, so this is not a tuning problem —
it is a property of the factorization.** The obvious alternative, resampling per destination POINT
(`pθ` source rings at each target's φ), is worse still: it abandons separability, costing
`pθ·pφ·length(dst)` against the separable `(pφ·Nθ_src + pθ·Nθ_dst)·Nφ_dst`, i.e. ~2.6× at p=6 by the
same accounting. *(That one is a derivation, not a measurement — it was not implemented.)* So
**φ-first-then-θ separability and ragged rings are in direct conflict**: any factorization pays either
the sum-over-widths or the loss of separability.

**B ≈ 1.0 for the spectral path is expected, not a surprise:** #117's reference inflates every ring to
`maxNφ` and does the full rectangular work, so it saves storage and nothing else. Its A = 0.153 is the
cost of the extra inflate/compress transforms.

*(An earlier version added: "whether the mode-space version (#119) can overcome that is open — but it
must beat A ≈ 0.15." That gate was wrong. A = 0.153 measures the reference's inflate/compress stages,
which a mode-space kernel does not perform, so it does not carry over. The real gate is the FFT
batching penalty, and it is measured below — #119 fails it.)*

**Caveat on the baseline, which makes the picture worse rather than better.** All three columns are
*allocating reference* implementations, so A is a fair reference-to-reference comparison. But the
production tensor Lagrange path is scratch-backed and 5–6× faster than its own reference (#112), and
there is no scratch-backed ring path at all. Against production the ragged ring kernels are ~30×
slower.

**Why there is no permanent benchmark group for this.** Groups 1–13 exist to track things worth
keeping fast. This result argues for not building more of the ragged path, so a group would re-measure
a dead end on every run. The numbers and the structural argument are here instead.

**THE RING SAMPLING HAS BEEN DELETED.** After #120 measured the translation share and closed the
feature (below), keeping ~1100 lines of `src/equidistant/ring*.jl` plus their tests and benchmarks as
"infrastructure" was not worth the test-suite time and the reading cost, so it was removed. The code is
intact in history — `10eee37` (#114 sampling), `af40bd9` (#115 bandwidth rule), `f7f33d6` (#116 local
Lagrange), `85aa28e` (#117 spectral), `2a0775b` (#118 benchmark) — and every number in this file was
measured before the removal. Nothing here is retracted by the deletion; the sections stay because the
*reasoning* is what stops this being re-proposed, and the structural argument above (φ-first
separability versus ragged rings) applies to any future attempt regardless of implementation.

## The far-path stage split, measured properly: translation is 70 % (#120)

**Verdict: I2O translation is 70 % of the far-field matvec, flat across a 4× range in N and a level of
tree depth. The ring feature's premise was right; two earlier attempts to size it were not.**
Reproduce with `translationshare.jl`.

### The method, and why the previous one was invalid

The earlier number came from `farmul! − aggregate! − disaggregate!`, which assumes those two calls
partition the far path. **They do not partition it — they cover it.** The run below shows why: with
translation swapped out, `disaggregate!` drops from 3021 ms to 475 ms, so `disaggregate!` *contains*
the I2O work. `aggregate! + disaggregate!` is therefore already the whole far path, and the
"remainder" was noise about zero. On tworects that noise read **~1 %**; on a 6-level sphere the same
subtraction returned **−2.4 %**, which is impossible and is what exposed the error. Both figures are
retracted, and "translation is a negligible share" was never a measurement.

The valid method swaps the collection instead of subtracting phases:

```julia
i2ostrategy = MLFMA.I2OTranslationStrategy(;
    TranslationCollection = MLFMA.DummyTranslationCollection
)
```

`DummyI2OTranslationCollection`'s `i2otranslation!` is `return out` — it applies no coupling. Building
the *same* fixture with it and differencing `farmul!` isolates translation **by construction**: every
other stage (aggregation, both leaf multiplies, the O2O downward pass, all traversal bookkeeping) is
the same work in both runs, so nothing can overlap and nothing can be double-counted. The result is
numerically wrong — `y` is garbage — which is fine, because this measures time, not values.

`aggregate!` is reported for both builds as the control: it must match, and it does at every size.

### The numbers

Sphere R = 4.79 m, edge length λ/10, leaf boxes λ/4, ϵ = 1e-4, near field skipped
(`DummyNearInteractionsStrategy` — near-field assembly is what put every earlier sphere fixture over
budget, and it is irrelevant to angular sampling).

| N | levels | far (real) | far (dummy) | `aggregate!` real / dummy | **I2O translation** | share |
|--:|--:|--:|--:|--:|--:|--:|
| 104,958 | 6 | 3690 ms | 1096 ms | 632 / 618 ms | 2594 ms | **70.3 %** |
| 203,439 | 6 | 7290 ms | 2233 ms | 1303 / 1253 ms | 5057 ms | **69.4 %** |
| 411,150 | 7 | 16993 ms | 5073 ms | 2848 / 2808 ms | 11921 ms | **70.2 %** |

So the far path decomposes as **aggregation ~17 % / I2O translation ~70 % / O2O disaggregation ~13 %**.
The share is stable, not trending — it does not need to be extrapolated to a larger problem.

*(The earlier reading that "the downward path is ~83 %" was correct but uninformative: 70 of those 83
points are translation, not disaggregation. A separate claim that the aggregation share falls with
tree depth is also retracted — 18.7 % → 17.0 % across a level is flat, and the 41–45 % seen on
tworects is a fixture difference, plate versus sphere, not a depth effect.)*

### What it does to the ring arithmetic

Translation dispatches to `elementwisemul!` (`i2otranslationcollection.jl`, `DiagonalTranslation`), a
pure elementwise product over the grid — so it collects the ~16 % sample saving 1:1 with no ragged
penalty. Interpolation pays the penalty measured in #118. With `net = 0.70 × 0.84 + 0.30 × penalty`:

| interpolation penalty | source | net |
|---|---|--:|
| 2.28× | #118 local ragged backend (B = 0.438) | **1.27** — loses |
| 1.22–1.64× | mode-space, measured below | **0.95–1.07** — break-even |
| 1.00× (hypothetical, no penalty) | — | 0.89 — the ceiling |

**Even at 70 % translation the ring feature does not pay.** The absolute ceiling, assuming an
interpolation kernel that costs exactly what the tensor one costs, is ~11 %; every kernel actually
measured gives that back and more.

## The mode-space fast path (#119): the FFT batching penalty exceeds the element saving

**Verdict: do not build it.** The measurement below is preserved here rather than as a runnable
script: it depended on the ring sampling, which has since been DELETED (see the note at the end of the
ring section above). Recover `modespace.jl` from commit `62b7a8d` if it ever needs re-running.

#119 was parked on the claim that it "must beat A ≈ 0.15", which was wrong — that A belongs to #117's
*reference*, which inflates every ring to `maxNφ` and runs the full rectangular path plus two extra
transform stages. The mode-space version does none of that, so the reference's overhead does not
transfer to it. The real risk is different and is measurable with no kernel written at all: mode space
replaces **one batched rectangular FFT** with **many small independent ones** — per ring in φ, and per
mode over a contiguous band of rings in θ. FFTW's batched transforms over contiguous memory are much
more efficient per element than a pile of small ones.

Transform work only, plans pre-created on both sides so plan creation is not what is timed:

| (Nθ,Nφ) | fill | φ× | θ× | **NET** |
|---|--:|--:|--:|--:|
| (16,32) | 0.906 | 0.45 | 0.50 | **0.47** |
| (24,48) | 0.875 | 0.78–0.82 | 0.82–1.03 | **0.80–0.92** |
| (32,64) | 0.852 | 0.39–0.42 | 0.36–0.49 | **0.39–0.43** |
| (48,96) | 0.837 | 0.94–1.16 | 0.42–0.43 | **0.61–0.69** |
| (64,128) | 0.814 | 0.49–0.50 | 0.98–1.05 | **0.79–0.82** |

Ranges are across three runs. **NET < 1 at every size in every run** — the ragged form's transform work
is 1.1–2.6× *slower* than batched-uniform, despite saving 8–19 % of the elements. That is the batching
penalty, and it is larger than the prize. It is also a floor, not a final figure: the mode-space kernel
must additionally pay for the mode bookkeeping the uniform path skips.

Feeding the best case back into the model gives net ≈ 0.95, i.e. ~5 % — inside the noise of the
measurement it came from, and negative at three of the five grid sizes.

## Where the 70 % actually points

Translation cost is `(number of translating box pairs) × (grid size at that level)`, and the product is
elementwise, so only those two factors are available.

**Antipodal symmetry (#110) is NOT a candidate, despite halving a grid.** `F(−k̂) = conj(F(k̂))` holds
for a single basis function's pattern with real `f_n` and purely imaginary `γ`. A box moment is
`Σ_n x_n F_n(k̂)` with **complex** solution coefficients, and Hermitian symmetry does not survive
complex scaling — so the symmetry is gone after the first aggregation, and translation operates
entirely on aggregated moments. #110's prize is halving the stored leaf radiation collection, with **no
flop reduction anywhere**; it does not reach this 70 % and should not be justified by it.

**The grid-size factor is what #114–#119 attacked, and it is now measured out.** The ~16 % available
there is real but is spent on the interpolation penalty, whichever backend collects it.

**That leaves the pair count, which is a function of tree shape and is an existing knob.** Every number
here was taken at `minhalfsize = λ/4`. Larger leaf boxes mean a shallower tree with fewer, larger
translations, trading translation count against leaf near-field work and against grid size per level —
and the near field was switched off in this fixture, so this study cannot see that side of the trade
at all. Whether λ/4 is near the optimum is **unmeasured**, and it is cheap to measure: sweep
`minhalfsize` on the existing sphere fixture with the near field switched back on. That is the honest
next question the 70 % raises; no prediction is offered here about which way it goes.

## What is deliberately not measured

- **The ring path end to end in a real matvec.** #118 was scoped to the interpolation kernels because
  the ring sampling is not wired into the tree/map layer. It still is not — and after the two sections
  above there is no longer a reason to wire it: the stage split that would have justified the work has
  been measured, and it does not justify it. See #120 for the one piece that was built anyway
  (`_farfieldconstructors` now shapes from `size(sampling)` rather than a scalar `Nφ`, which is a
  correctness improvement independent of ring sampling).
- **Vector far fields.** `SphericalVectorFarField` and `PairedSphericalVectorFarField` have continuation
  and sign rules at the poles and the φ seam that exact cases must respect, and filling each component
  with an unrelated scalar harmonic would produce numbers that describe no field the interpolator
  carries. Scalar and *tangential* cases both exist (groups 10 and 11b); these two do not.
- **Anterpolation accuracy.** Group 9 measures the adjoint *pairing*, and `adjointness` rides along as an
  invariant, but the reverse direction's own information types and indexing are otherwise unexercised.
- **Cost at equal tolerance.** Every runtime row here compares families at equal `(Nθ, Nφ)`. Since the
  equidistant family reaches round-off where the local one sits at 3e-4, the production question — what
  grid each needs to hit a requested ε, and what *that* costs — is not answered by any table in this file.
- **`adjointpairingerror` and `intermediatesampling` on non-spherical grids.** Both are deliberately still
  typed to `SphericalSampling`, and the reason is in `evaluate.jl`: the first forms an inner product and so
  is only the Galerkin pairing it claims to be under the sampling's own quadrature, and the second presumes
  a separable θ-then-φ interpolator, which the equidistant FFT backend is not.
- **The production traversal.** No tree, no workspace, no scratch. Whether the level-transfer path
  allocates is Phase 5's question and belongs with the allocation contracts in
  `test/Performance/allocations.jl`, not here.
- **Anything at more than one machine's precision.** Every study is `Float64`.

## Doing all of a parent's I2I translations at once: DECIDED, do not build

**Verdict: no. The mechanism that would have paid does not, and two of three levels measure batching
as a regression.** `benchmark/interpolation/i2ibatching.jl`.

`i2ipass!` loops over a node's children, interpolating each onto the parent grid and accumulating.
The interpolation operator is shared by every child at a level, and the per-child phase lands on the
parent grid after interpolation (#105), so the children *could* go through one batched call. Summing
children first is not available — different centres, different phases.

Measuring the ceiling rather than building it: the θ stencil (a GEMM) and the φ resample (an FFT) run
eight times separately against once with eight columns, plus the parent accumulate. No implementation
beats that ceiling.

| p | child grid | parent grid | φ FFT | θ GEMM | accumulate | all three |
|--:|--:|--:|--:|--:|--:|--:|
| 6 | 14×28 | 26×52 | 0.96× | 0.43× | 2.49× | **0.68×** |
| 13 | 28×56 | 54×108 | 0.98× | 2.11× | 2.42× | **1.84×** |
| 26 | 54×108 | 106×212 | 0.98× | 0.75× | 2.49× | **0.87×** |

*(separate / batched; above 1 means batching wins)*

**The φ FFT gains nothing — 0.96–0.98× at every level.** That was the hypothesis, and it is wrong:
FFTW's per-call overhead is already negligible at these sizes, so `howmany` batching buys nothing and
marginally loses. **The θ GEMM is shape luck, not a trend** — 0.43×, 2.11×, 0.75× as the dimensions
land on better or worse BLAS kernels. Nothing here is a structural win.

**The accumulate's 2.5× is probably illusory, and this is the part that settles it.** It compares
eight write passes over the parent against one — but a batched path must first MATERIALIZE eight
staged interpolation outputs, which today's loop does not: it writes interpolation results straight
into the parent. The benchmark charges the staging write to neither side. Accounting for it, batching
is strictly worse on traffic at that stage, so the only consistent "win" in the table is an artifact
of what was left out.

**The prize was bounded and small to begin with.** The far path is aggregation ~17 % / I2O translation
~70 % / O2O disaggregation ~13 % (#120); all I2I work in both directions is at most ~20–25 % of the
far matvec. Even taking the best row at face value, 1.84× is worth `(1 − 1/1.84) × 0.25 ≈ 11 %` — and
two of three levels say it is a regression.

**This is the SHE crossover again, from the other side.** That study measured batching at 8 groups as
3.7× SLOWER than not batching, with the crossover at 16–32. A parent has at most 8 children, fewer on
a non-uniform tree. Different mechanism, same conclusion: eight is too few to amortise anything.

The 70 % still points where it did — at the pair count, which is tree shape.

## Batching I2O by transfer vector: DECIDED, do not build

**Verdict: no, and for a reason that generalises — the translation operator already fits in L2, so
there is no re-streaming to eliminate.** `benchmark/interpolation/i2ovectorbatching.jl`.

`translateintonode!` loops over (receiving, translating) PAIRS and streams the translation operator
`T` once per pair. `T` depends only on the relative offset, so many pairs share one — grouping the
loop by offset would stream it once per group. Since I2O translation is 70 % of the far matvec
(#120), a win here would matter more than anything else available.

Sphere 2λ, mesh λ/8, λ/4 leaves, N = 12510.

| level | pairs | distinct vectors | mean | median | max | grid `nd` |
|--:|--:|--:|--:|--:|--:|--:|
| 3 | 1736 | 262 | 6.6 | 6 | 16 | 3528 |
| 4 | 26808 | 850 | 31.5 | 26 | 76 | 1568 |

| `nd` | pairs | per-pair | grouped | ratio |
|--:|--:|--:|--:|--:|
| 3528 | 6 | 26.3 µs | 26.6 µs | **0.99×** |
| 1568 | 26 | 51.1 µs | 51.7 µs | **0.99×** |

**The premise was wrong twice.** The batch was predicted to be "hundreds of pairs"; it is a median of
**6 and 26**. And there are 850 distinct vectors at level 4, not the textbook 316 — the interaction
list here is not `7³-3³`, because `additionalbufferboxes` changes it. (The geometric binning is sound:
the offset residuals are 4e-16 and 7e-16, so the quotients really are integers.)

**But the batch size is not why it fails.** The kernel measures 0.99× even at 26 pairs, which is
inside the 16–32 crossover where SHE batching started to win. The reason is simpler: `T` is `nd`
complex numbers — **56 KB at level 3, 25 KB at level 4** — so it is resident in L2 across the whole
pair loop already. The hypothesis assumed `T` was being evicted between pairs and re-fetched from
memory. It is not, so grouping removes traffic that was never there.

**This generalises, which is the useful part.** Any "stream the shared translation operator once"
optimization on this stage is pointless for the same reason. The 70 % is not operator-bound: it is
bound by the SOURCE AND TARGET MOMENT traffic, which scales as `pairs × nd` and which no reordering
reduces, plus the arithmetic itself. Reducing it needs fewer pairs or fewer directions — the two
factors #120 already named — not a better loop order.

## The near field: bandwidth-bound already, and ordering does nothing

**Verdict: no lever here.** `benchmark/interpolation/nearfield.jl`.

Sphere 2λ, mesh λ/8, λ/4 leaves, N = 12510, nnz = 17.7 M (1414 per row), 4 threads.
Measured stream triad bandwidth 24.3 GB/s.

| | time | share of matvec |
|--|--:|--:|
| near | 0.0267 s | 5.9 % |
| far | 0.4202 s | 93.6 % |
| full | 0.4489 s | |

| kernel | time | achieved | of stream | ratio |
|--|--:|--:|--:|--:|
| bare CSC SpMV | 0.0265 s | 16.1 GB/s | 66.1 % | — |
| tree-ordered SpMV | 0.0267 s | 15.9 GB/s | 65.7 % | **0.99×** |

*(permutation verified: 3.3e-16 against the unpermuted product)*

**Reordering into leaf-traversal order buys nothing.** The hypothesis was that the near matrix
inherits the mesher's numbering, so its gather is scattered where tree order would be blocked. It
measures 0.99×. And at 66 % of a streaming triad the kernel is already close to what an SpMV with a
gather can achieve, so there is not much headroom to chase even with a better format.

**The near SHARE is strongly fixture-dependent and the 5.9 % here should not be generalised.** #121
measured near 0.0365 s against far 0.0803 s on a coarser mesh (λ/6, N = 6756) — about 30 %. This run
is at 4 threads and the far path is threaded while the SpMV is not, which inflates the far share
further. What transfers between the two fixtures is not the share but the ceiling: the SpMV is
bandwidth-limited in both, so the achievable win is bounded by roughly a third of whatever the near
share happens to be.

Combined with the two translation results above, every kernel-level and loop-order lever examined
this session is closed. What remains is what #120 named: **fewer pairs or fewer directions**.
