# Choosing leaf storage

Leaf collections can dominate a map's storage: they contain a pattern for every
basis function. Keep those patterns as samples or as harmonic coefficients.

| Policy | Stored per basis function | Applies to |
|:--|:--|:--|
| `SampledLeafStorage()` | One value per direction and component | All supported representations |
| `SphericalHarmonicLeafStorage()` | `3(P+1)²` Cartesian scalar coefficients | Tangential spherical operators |
| `VectorSphericalHarmonicLeafStorage()` | `2P(P+2)` tangential coefficients | Tangential spherical operators |
| `CircularHarmonicLeafStorage()` | `D(2M+1)` coefficients | Supported 2-D representations with `D` components |

Start with sampled storage. For a large tangential problem, compare SHE and VSH
against it. Compression saves pattern data, but synthesis adds work and scratch;
a smaller coefficient count alone does not guarantee a faster product.
Circular compression is especially worth measuring rather than assuming faster.

```julia
fields = MLFMA.FarFieldStrategy(
    leafstorage=MLFMA.VectorSphericalHarmonicLeafStorage(),
)
strategy = MLFMA.MLFMAStrategy(op, X; farfieldstrategy=fields)
```

`op` must have a tangential representation here. For example,
`MLFMA.efie(wavenumber=k)` selects it; a native Maxwell single layer does not.

Petrov maps may choose `trialleafstorage` and `testleafstorage` independently.
Galerkin maps derive receive patterns from radiation patterns and require
matching policies. [Harmonic storage](harmonics.md) explains the data layout.
