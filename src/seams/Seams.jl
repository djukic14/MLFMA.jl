"""
    MLFMA.Seams

Shared function declarations for calls from lower layers to higher ones.

Far-field kernels call these functions; `Operators` and `Assembly` supply the methods.
This module contains no implementations, types or policy tables. Builders belong in
`Assembly`, where they can use operator properties directly.
"""
module Seams

"""
    receivecouplingrule(operator, ::Type{<:AbstractFarField})

Return the component-mixing rule from radiation to receive fields.

The rule uses the component indices of the supplied far-field representation.
Lazy receive kernels and harmonic collections call it; `Operators` defines the rules.
"""
function receivecouplingrule end

"""
    radiationfieldcollectiontype(operator)

Return the radiation-field collection type for `operator`.

[`LazyReceiveFieldCollection`](@ref MLFMA.FarFields.LazyReceiveFieldCollection) uses it
to check the wrapped representation, not just its component count.
"""
function radiationfieldcollectiontype end

"""
    farfieldtype(operator)

Return the far-field representation type for `operator`, such as scalar or
tangential-spherical. Equal component counts do not imply equal representations.

`Operators` defines the table; `Assembly` and the dummy assembler add methods.
"""
function farfieldtype end

export farfieldtype, radiationfieldcollectiontype, receivecouplingrule

end
