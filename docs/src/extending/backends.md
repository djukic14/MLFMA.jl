# Sampling, transfers and storage

These are separate extension points. A grid does not automatically supply an
interpolator, and a leaf-storage format does not change I2O.

## A direction grid

Subtype `MLFMA.Sampling.AbstractSampling{T}` and implement
`length`, `size`, `enumerate`, `MLFMA.Sampling.arraytype`,
`MLFMA.Sampling.cartesiansamplepoints`, `MLFMA.Sampling.mirrorindices`
and `MLFMA.Sampling.cutoffparameter`.

A spherical product grid subtypes
`MLFMA.Sampling.AbstractSphericalSampling{T}` and also supplies θ/φ nodes and
`MLFMA.Sampling.sphericalsamplepoints`. Keep coordinates, weights and array
ordering consistent. The [sampling API](../api/sampling.md) gives the signatures.

## A level transfer

Select an interpolator and phase shifter through `MLFMA.LevelTransferStrategy`.
The standard path builds interpolation information per grid pair, obtains
reverse operators with `MLFMA.Interpolation.anterpolator` and
`MLFMA.Interpolation.transposedinterpolator`, then wraps them in
`MLFMA.InterpolationPhaseTransfer`.

Use `MLFMA.InterpolationDiagnostics.interpolationerror` with analytic scalar
or tangential cases. Check transpose and adjoint pairings separately. Writable
apply buffers must belong to the workspace, not a shared operator.

## A leaf-storage format

Implement
`MLFMA.Assembly.buildfieldcollections(builder, op, tree, leafsamplings, source)`.
The compressed builders share assembly wiring and specialize
`MLFMA.Assembly.analysecollections` and
`MLFMA.Assembly.assembledreceivecollectiontype`.

The resulting collection needs `H2Factory.applyleafmul!` kernels and matching
scratch. Add batching after the per-leaf multiply agrees with sampled storage.
Check both assembled Petrov receive collections and the Galerkin wrapper
that reuses radiation coefficients.
