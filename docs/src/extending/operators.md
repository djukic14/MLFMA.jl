# Adding an operator

MLFMA needs the radiation and receive patterns, their representation, angular
continuation and amplitudes, not just a kernel evaluator. Start with an existing
operator using the same representation: scalar examples are in
`src/operators/helmholtz/hhfarfields.jl`, Maxwell wrappers in `src/operators/maxwell/`.

## Operator properties

For a new operator type `MyOperator`, provide the applicable methods:

```julia
MLFMA.Operators.γ(op::MyOperator) = op.gamma
MLFMA.Operators.α(op::MyOperator) = op.alpha
MLFMA.Seams.farfieldtype(::MyOperator) = MLFMA.ScalarFarField
MLFMA.Operators.farfieldstorerule(::MyOperator) = MLFMA.StoreAsFarFieldValue()
MLFMA.Interpolation.interpolationtrait(::MyOperator) = MLFMA.VirtualExtension()
MLFMA.Assembly.λ(op::MyOperator) = 2π / abs(op.gamma)
```

These scalar choices are an example, not defaults for a vector operator.
`α` supplies amplitudes applied by the map. Keep their placement consistent
with the assembled patterns so they are not applied twice.

## Pattern assembly

Implement `MLFMA.Operators.radiationoperator(op::MyOperator)` and
`MLFMA.Operators.receiveoperator(op::MyOperator)` to return the BEAST far-field
operators used to assemble trial and test collections. Their BEAST assembly
methods must produce the values consumed by the store rule.

For Galerkin's lazy receive path, also implement
`MLFMA.Seams.receivecouplingrule(op::MyOperator, ::Type{YourFarField})`.
Use the actual representation: two components do not automatically mean the
same frame or receive rule.

First check separately assembled radiation and receive patterns. Then compare
a small map against dense assembly in forward, transpose and adjoint orientations.
H2Factory and H2FactoryTestUtils cover the generic map protocol; MLFMA's
`test/Operators` and `test/Maps` show the operator-specific checks.
