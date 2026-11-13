# Far fields and directions

A leaf collection stores one radiation pattern for each basis function in the
leaf. A pattern says how that function contributes in each direction `k̂`.
It is referenced to the leaf box's center.

For a scalar trial function, the radiation phase has the form

```math
U_j(\widehat k)=\int_{\operatorname{supp}u_j}
u_j(y)\,e^{+\gamma\widehat k\cdot(y-c)}\,\mathrm{d}y,
\qquad \gamma=\mathrm{i}k.
```

Vector operators have vector-valued integrands and may project onto a spherical
frame. The receive pattern uses its own operator and phase convention; it is
not obtained by blindly conjugating the radiation pattern.

## Samples and components

`MLFMA.ScalarFarField` holds one array. A
`MLFMA.TangentialSphericalFarField` holds two, accessed with `MLFMA.Fθ(field)`
and `MLFMA.Fφ(field)`. A spherical vector field also has `MLFMA.Fr(field)`.

On a spherical product grid, each component is stored as `[φ, θ]`, with φ
contiguous in memory. A leaf's sampled collection adds one column per basis
function. Multiplication combines these patterns into one field.

```jldoctest
julia> using MLFMA

julia> s = MLFMA.SphericalSampling{Float64}(6, 12);

julia> f = MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(s));

julia> size(MLFMA.Fθ(f))
(12, 6)
```

The [sampling grid](sampling.md) sets the number of directions.
[Harmonic storage](../storage/harmonics.md) changes the stored collections,
not the sampled field used by the tree traversal.
