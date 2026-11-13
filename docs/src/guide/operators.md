# Supported operators

The operator chooses the far-field representation and component convention.
Leaf storage is a separate choice.

| Operator family | Moment representation |
|:--|:--|
| 3-D Helmholtz single layer | Scalar |
| 3-D Helmholtz double layers | Scalar |
| 3-D Helmholtz hypersingular | Two spherical-vector groups |
| Maxwell single layer | Spherical vector |
| Maxwell double layers and `MLFMA.efie` | Tangential spherical |
| `MLFMA.MWHyperSingular3D` | Scalar (radial projection) |
| 2-D Helmholtz single and double layers | Scalar |
| 2-D Helmholtz hypersingular | Two in-plane components and one scalar |

Inspect the actual choice rather than inferring it from a component count:

```julia
MLFMA.farfieldtype(op)
MLFMA.farfieldstorerule(op)
MLFMA.interpolationtrait(op)
```

The first selects the layout, the second projects assembled values into it,
and the third handles continuation of angular components at the poles.

Some double-layer operators lack a Galerkin mirrored-receive rule. Use an
assembled Petrov receive side for these; inspect `MLFMA.haslazyreceive(op)`
before choosing a Galerkin map for a new operator.

SHE and VSH storage require a **tangential spherical** operator. They are not
storage options for every three-dimensional operator. Keep sampled storage for
a native Maxwell single layer; choose an EFIE wrapper only
when that is the operator you intend to assemble.

Circular harmonic storage supports the corresponding 2-D representations.
See [Leaf storage](../storage/choosing.md) before choosing a compressed policy.
