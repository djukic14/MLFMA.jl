# Harmonic storage

Harmonics compress the **stored leaf patterns**. They do not replace the
directional samples used by I2I, I2O and O2O.

```math
F_j(\widehat k)\approx\sum_{\ell=0}^{P}\sum_{m=-\ell}^{\ell}
c_{\ell m,j}Y_\ell^m(\widehat k).
```

An expansion needs `(P+1)²` coefficients per scalar component. The matrix `C`
stores one basis function per column. Given leaf inputs `x`, multiplication
first forms `a=Cx`, then synthesizes the **combined** pattern. It does not
reconstruct each basis function separately.

## See truncation

This scalar example expands `exp(-iz cos θ)`. Increasing `z` makes the pattern
more oscillatory. Change retained degree `P` and compare its real part with the
reconstruction. Coefficients are computed by quadrature, not by a curve fit.

```@raw html
<div class="mlfma-explorer" data-kind="harmonics"></div>
```

This axisymmetric scalar illustration is not a prediction of the order or error
for an assembled RWG pattern. It does not compare SHE with VSH.

## SHE and VSH

SHE expands three Cartesian components separately, then projects the reconstructed
field onto θ and φ. Expanding arbitrary θ and φ components as independent scalar
functions would ignore their coordinate behavior at the poles.

VSH uses two tangential harmonic families instead. It writes θ and φ components
directly, with no Cartesian projection. It stores one coefficient block for
the `X` family followed by one for `Ψ`.

Each collection stores its order, expansion center and coefficient matrix.
All leaves at one level use a common order, chosen from assembled coefficient
tails. Analysis uses Gauss-Legendre nodes in `cos θ`. Reaching the analysis
order raises an error rather than silently accepting an under-resolved analysis.

Coefficient matrices belong to the core. Working coefficient and sample buffers
belong to workspace scratch pools. Leaves at one level can batch their synthesis
against shared matrices.

## A small transform check

```jldoctest
julia> using MLFMA

julia> const SH = MLFMA.SphericalHarmonics;

julia> θ, φ, w = SH.gausslegendregrid(3);

julia> t = SH.SphericalHarmonicTransform(ComplexF64, θ, φ, w, 3);

julia> a = zeros(ComplexF64, 16, 1); a[SH.lmindex(2, 1), 1] = 1;

julia> maximum(abs.(SH.analyze(t, SH.synthesize(t, a)) .- a)) < 1e-12
true
```

Circular storage uses the analogous Fourier expansion with modes `-M:M`.
See the [storage choices](choosing.md) for its component count and use cases.
