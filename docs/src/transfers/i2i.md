# Level transfers: I2I and O2O

A child and its parent have different direction grids and expansion centers.
An I2I transfer must change both. In the radiation convention,

```math
F_p(\widehat k)\approx
e^{-\gamma\widehat k\cdot(c_p-c_c)}\,
\mathcal I_{c\to p}F_c(\widehat k).
```

`I` interpolates child samples onto the parent grid. The phase factor
re-references the pattern from the child center to the parent center.
`MLFMA.InterpolationPhaseTransfer` combines these operations. The receive
side uses the opposite phase sign.

## Interpolation on one angular slice

Both standard spherical families interpolate locally in θ. They differ in φ:

- `SphericalLagrangeInterpolator` uses a local polynomial stencil.
- `LocalθGlobalφInterpolator` evaluates the periodic Fourier interpolant.

The figure isolates **φ**, using `cos(mφ)` at a fixed polar angle. Change the
source count, target count, mode and stencil. Resolved Fourier modes can be
reconstructed to round-off; modes beyond the source grid's Nyquist limit alias.

```@raw html
<div class="mlfma-explorer" data-kind="transfer"></div>
```

The error is for this slice, not a full transfer or matrix. The stencil setting
affects only local interpolation. A warning marks the even-grid Nyquist case.

## The phase shift

Change the center displacement to see the radiation phase
`exp(-ik δ cos(φ))`, where `δ` points from child center to parent center.
Its magnitude is one for real `k`, but its real and
imaginary parts oscillate faster as displacement grows.

```@raw html
<div class="mlfma-explorer" data-kind="phase"></div>
```

## Use the production interpolator

The diagnostics evaluate exact angular functions independently on both grids.
This check exercises both angles, not the browser's slice.

```jldoctest
julia> using MLFMA

julia> const ID = MLFMA.InterpolationDiagnostics;

julia> child = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(6)...);

julia> parent = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(10)...);

julia> interpolator = MLFMA.LocalθGlobalφInterpolator{8}(ComplexF64, Float64);

julia> study = ID.interpolationerror(interpolator, child, parent, [ID.HarmonicCase(l=2, m=1)]);

julia> isfinite(ID.worstcase(study).rel_l2)
true
```

Pass a builder through `MLFMA.LevelTransferStrategy(op, builder)` to use it in
a map. [Choosing parameters](../guide/parameters.md) shows the call.

## O2O and reversed products

O2O returns incoming fields from parent to child. The code keeps both plain
transpose and adjoint interpolation: they differ for a complex operator.
Do not substitute one for the other just because both reverse the grid sizes.
The transfer protocol selects the required direction and phase convention.
