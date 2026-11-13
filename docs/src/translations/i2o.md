# Translation between boxes: I2O

After aggregation, each box has one field on its direction grid. I2O multiplies
it by a translation function for the displacement between separated boxes:

```math
F_a^{\mathrm{in}}(\widehat k_q)\mathrel{+}=
T_L(\widehat k_q,R)\,F_b^{\mathrm{out}}(\widehat k_q).
```

This is diagonal in direction: there is no dense matrix coupling different
directions. `MLFMA.translationoperator!` constructs the arrays; the collection
reuses them for repeated displacements at a level. The full coupling includes
the level's integration weights.

## Inspect a translation

These curves come from the production Julia kernel. Choose separation in
wavelengths and cutoff, then switch between magnitude and phase. The slice
uses `R=(r,0,0)` and `k̂=(cos φ,sin φ,0)`.

```@raw html
<div class="mlfma-explorer" data-kind="translation"></div>
```

Increasing the cutoff does not repair an invalid geometric separation.
The expansion is intended for separated source and target supports; near
blocks are handled directly by the near-interaction strategy.

## Separation, cutoff and accuracy

How well does the translated field reproduce a point-source Green's function?
Fix a local offset `x = a e` and move the box center by `d = D e` along the
same axis. Here `e` is the z-axis in 3-D and the x-axis in 2-D; wavelength is one.

```math
G_L(d,x)=\sum_q w_q\,T_L(\widehat k_q,d)\,
e^{-\mathrm{i}k\widehat k_q\cdot x},\qquad
\varepsilon_L=\frac{|G_L-G(|d+x|)|}{|G(|d+x|)|}.
```

The exact kernels use the package's outgoing-wave convention:

```math
G_{3D}(r)=\frac{e^{-\mathrm{i}kr}}{4\pi r},\qquad
G_{2D}(r)=-\frac{\mathrm{i}}4 H_0^{(2)}(kr).
```

The default offset is `a=0.4`. This is a collinear stress configuration,
not a computed maximum over all orientations. The shaded region fails the
expansion's separation condition `D > |x|`; those points are not evaluated.

```@raw html
<div class="mlfma-explorer" data-kind="accuracy"></div>
```

The production translation kernel generates every solid curve. For each fixed
cutoff, angular refinement multiplies the grid counts by 1, 2 or 4 without
changing the requested cutoff. In 3-D, compare the standard Gauss-Legendre-in-θ
grid with the equidistant grid and its exact angular weights. In 2-D the circle
count uses the production builder's quarter-turn rounding.

Collinearity makes the 3-D integrand independent of φ, so generation sums its φ
weights first. This saves work without changing the quadrature mathematically;
floating-point rounding can differ from summing the full product grid.

Here `L` is passed explicitly to the translation kernel. It is not inferred
again from the refined grid. The 2-D cancellation guard can reduce that order;
hovering a curve shows the effective order and number of directions at that point.
The 3-D kernel uses the full requested order.

### Reading the curves

At large `kD`, a plateau can come from truncation: adding angular samples cannot
recover terms omitted by `L`. At smaller electrical separations, high-order
Hankel terms can amplify cancellation in the sampled integral. A larger cutoff
is therefore not automatically more accurate in finite precision.

Turn on the integrated reference to remove angular quadrature from the same
finite expansion. The 3-D reference uses 256-bit arithmetic. The 2-D reference
uses Float64 cylindrical Bessel functions and the effective cutoff, so it is
not a high-precision oracle. Comparing refinements and this reference helps
distinguish truncation, angular integration and floating-point effects; it does
not turn any one curve into a guaranteed matrix-error bound.

The offline dataset covers `L=1:24`, four offsets and `0.5 ≤ kD ≤ 1000`.
These are figure-generation limits, not restrictions of MLFMA. Edit
`docs/translation-accuracy.jl` to generate another range. Zero errors have no
position on a logarithmic axis and are not replaced by an artificial floor.

## Symmetry reduction

For a supported grid symmetry `S`,

```math
T(\widehat k,S R)=T(S^{-1}\widehat k,R).
```

The same array serves a symmetry-related displacement through an angular
permutation. The displacement uses `S`, while directions use its inverse.

```@raw html
<div class="mlfma-explorer" data-kind="symmetry"></div>
```

```julia
i2o = MLFMA.I2OTranslationStrategy(
    TranslationCollection=MLFMA.SymmetryTranslationCollection,
)
strategy = MLFMA.MLFMAStrategy(op, X; i2ostrategy=i2o)
```

The collection chooses a symmetry group supported by every level's grid.
For non-lattice offsets between block-tree roots, it warns and falls back to
unreduced storage. Symmetry reduction adds no interpolation, although rotated
coordinates can change the final rounding bits.
