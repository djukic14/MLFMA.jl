# Building the guide

From the package root:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/check-explorers.jl
julia --project=docs docs/make.jl
```

`make.jl` generates the local Plotly library and I2O slices before Documenter
builds the site. Generated files are ignored. No matrix assembly runs in the build.
The figure checks need Node.js; they compare browser calculations with Julia.
They also check the translation-accuracy experiment against full angular sums,
integrated finite-series references and the production 2-D cutoff guard.

The accuracy dataset is cached by its generator and numerical source files.
It is loaded only on pages containing that explorer. To change its ranges,
edit `translation-accuracy.jl`; no mesh assembly is needed.

Browser figures illustrate small analytic cases. In particular, the transfer
figure isolates azimuthal interpolation, and the harmonic figure is scalar and
axisymmetric. Their error displays are not matrix-accuracy estimates.
