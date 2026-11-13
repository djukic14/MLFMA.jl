# Choosing parameters

Start with `MLFMA.MLFMAStrategy(op, X)`. Override a sub-strategy to change one
part of construction.

| Change | Pass to `MLFMAStrategy` | What it controls |
|:--|:--|:--|
| Angular resolution | `samplingstrategy=MLFMA.SamplingStrategy(op; ϵ=1e-4)` | Grid resolution from tolerance and box size |
| Support outside a box | `samplingstrategy=MLFMA.SamplingStrategy(op; protrusionaware=true)` | Includes measured basis-support extension |
| Leaf size | `treestrategy=MLFMA.TreeStrategy(op; minhalfsize=λ/4)` | Minimum box half-side length |
| Leaf storage | `farfieldstrategy=MLFMA.FarFieldStrategy(; leafstorage=MLFMA.VectorSphericalHarmonicLeafStorage())` | Sampled or compressed collections |
| Angular transfer | `leveltransferstrategy=MLFMA.LevelTransferStrategy(op, builder)` | Interpolation family and stencil |
| Translation symmetry | `i2ostrategy=MLFMA.I2OTranslationStrategy(; TranslationCollection=MLFMA.SymmetryTranslationCollection)` | Reuses symmetry-related I2O arrays |

`ϵ` guides angular sampling and harmonic truncation. It is not a bound on the
final matrix error: quadrature, leaf size and interpolation also contribute.

## Interpolation builders

```julia
builder = MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=15)
transfer = MLFMA.LevelTransferStrategy(op, builder)
strategy = MLFMA.MLFMAStrategy(op, X; leveltransferstrategy=transfer)
```

Use `MLFMA.SphericalLagrangeInterpolationBuilder(; order=15)` for local
interpolation in both angles. Operator-aware defaults choose circular
interpolation for 2-D. A builder's element type must match `γ(op)`; set its
`T` keyword for another precision or a real `γ`.

## Assembly quadrature

Far-field quadrature goes into `MLFMA.FarFieldStrategy(; quadstrategy=...)`.
Near-field quadrature goes into
`MLFMA.SparseBEASTNearInteractionsStrategy(quadstrategy)`, passed as
`nearinteractionsstrategy`. These integrate basis functions; they are separate
from angular quadrature on the sphere.

For workspace reuse and parallel execution, see
[H2Factory](https://djukic14.github.io/H2Factory.jl/dev/).
