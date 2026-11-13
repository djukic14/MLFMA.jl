# Angular sampling

A larger box can radiate a more rapidly varying angular pattern. It therefore
needs more directions. MLFMA chooses a cutoff `L`, then constructs the level's grid.

For the standard spherical grid,

```math
\begin{aligned}
N_\theta&=2(L+1),\\
N_\phi&=4(L+1),\\
N_\Omega&=8(L+1)^2.
\end{aligned}
```

Change box width and tolerance below. The points are directions on the unit
sphere, not locations on the physical box. The default grid uses Gauss-Legendre
nodes **in θ**, with the spherical Jacobian included in its weights.

```@raw html
<div class="mlfma-explorer" data-kind="sampling"></div>
```

The figure uses the production cutoff rule for a cube of width `a`, with
`k=2π` and wavelength one. If basis support extends outside its box,
`protrusionaware=true` can increase the size used by this rule.

```jldoctest
julia> using MLFMA

julia> s, L = MLFMA.sampling(2π * im, 1e-4, sqrt(3) / 2);

julia> (MLFMA.Nθ(s), MLFMA.Nφ(s)) == MLFMA.NθNφ(L)
true

julia> length(s) == 8 * (L + 1)^2
true
```

## Other grids

`MLFMA.CircularSampling` puts directions on a circle for 2-D problems.
`MLFMA.Equidistant.EquidistantSampling` uses midpoint θ nodes and uniform φ
nodes. Its FFT and local-Lagrange backends are separate from the standard
spherical interpolators.

The harmonic analysis grid is different again: it uses Gauss-Legendre nodes
in `cos θ`. That substitution matters when computing coefficients;
see [Harmonic storage](../storage/harmonics.md).
