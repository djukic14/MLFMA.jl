# MLFMA.jl

MLFMA builds fast matrix-vector products for wave-kernel matrices. It uses
BEAST to assemble boundary-element operators, H2Trees to group basis functions,
and H2Factory to run the hierarchical multiplication.

[Construct a map](getting-started.md), or start with
[far fields and directions](fields/patterns.md) to see how the method works.

## From an integral to three small operations

For example, a scalar single-layer boundary-element matrix contains integrals

```math
A_{ij}=\int_\Gamma\int_\Gamma
v_i(x)\,\frac{e^{-\mathrm{i}k|x-y|}}{4\pi|x-y|}\,u_j(y)\,\mathrm{d}y\,\mathrm{d}x.
```

For well-separated test and trial boxes, MLFMA approximates this block by

```math
A_{ij}\approx\sum_q w_q\,V_i(\widehat k_q)\,
T_{ab}(\widehat k_q)\,U_j(\widehat k_q).
```

`U` is the trial function's radiation pattern, `T` translates between box
centers, and `V` is the test function's receive pattern. Their phase conventions
and normalization belong together. The weights `w` integrate over directions.
Vector operators use component-valued patterns and the operator's component coupling.

```@raw html
<div class="method-flow" aria-label="Radiation, translation and reception">
<span>Trial coefficients</span><b>→</b><span>Radiation<br><code>U x</code></span><b>→</b><span>Translation<br><code>T .* field</code></span><b>→</b><span>Reception<br><code>Vᵀ field</code></span><b>→</b><span>Tested values</span>
</div>
```

The tree lets many functions share these operations. I2I transfers combine
children into parents; O2O transfers pass incoming fields back to children.
Near blocks are assembled directly. The
[H2Factory guide](https://djukic14.github.io/H2Factory.jl/dev/) explains the
traversal, core/workspace split and scheduling.

## Explore the ingredients

- [Angular sampling](fields/sampling.md): how many directions does a box need?
- [Level transfers](transfers/i2i.md): what do interpolation and a phase shift do?
- [Harmonic storage](storage/harmonics.md): which leaf data can be compressed?
- [Box translations](translations/i2o.md): how do fields travel between separated boxes?

## References

- J. Song, C.-C. Lu and W. C. Chew, *Multilevel fast multipole algorithm for
  electromagnetic scattering by large complex objects*, IEEE Transactions on
  Antennas and Propagation, 45(10), 1488–1493, 1997.
  [DOI](https://doi.org/10.1109/8.633855).
- T. F. Eibert, *A diagonalized multilevel fast multipole method with spherical
  harmonics expansion of the k-space integrals*, IEEE Transactions on Antennas
  and Propagation, 53(2), 814–817, 2005.
  [Author's publication list](https://www.ee.cit.tum.de/en/hft/people/professors/prof-dr-ing-thomas-eibert/publications-until-october-2008/).
