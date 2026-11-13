# Constructing a map

The two inputs are a BEAST operator and a basis space. For a Maxwell
single-layer operator with RWG functions:

```julia
using MLFMA, BEAST, CompScienceMeshes

λ = 1.0
k = 2π / λ
mesh = meshsphere(λ / 2, λ / 8)
X = raviartthomas(mesh)
op = Maxwell3D.singlelayer(wavenumber=k)

A = MLFMA.GalerkinMLFMA(op, X)
x = ones(ComplexF64, numfunctions(X))
y = A * x
```

`A` is a linear map, not a dense matrix. Use `mul!(y, A, x)` to reuse an
output vector, or pass it to an iterative solver. Construction is the expensive
step; repeated products reuse its operators and workspace.

These assembly examples are not executed during the documentation build.

## Scalar Helmholtz

The same constructor accepts scalar operators. For example, use continuous
piecewise-linear nodal functions with a scalar single layer:

```julia
X = lagrangec0d1(mesh)
op = Helmholtz3D.singlelayer(wavenumber=k)
A = MLFMA.GalerkinMLFMA(op, X)
```

For two spaces, use `MLFMA.PetrovGalerkinMLFMA(op, testspace, trialspace)`.
The first space determines rows, the second columns. See
[H2Factory](https://djukic14.github.io/H2Factory.jl/dev/) for the two-space
map and workspace conventions.

## Set options on the strategy

```julia
strategy = MLFMA.MLFMAStrategy(op, X;
    samplingstrategy=MLFMA.SamplingStrategy(op; ϵ=1e-4),
    treestrategy=MLFMA.TreeStrategy(op; minhalfsize=λ / 4),
)
A = MLFMA.GalerkinMLFMA(op, X; strategy=strategy)
```

The map constructor accepts `strategy`, `tree` and `verbose`. Numerical choices
belong to the strategy, as listed in [Choosing parameters](guide/parameters.md).
