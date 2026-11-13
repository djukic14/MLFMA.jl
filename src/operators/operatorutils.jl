
# These scalar 2-D and 3-D operators share `(alpha, gamma)`: `alpha` scales the whole block.
# The far-field operators read `gamma` but omit this amplitude; the H2 core applies it once via
# `leafmul!(..., α(A), false)`.
#
# List the 2-D types individually. `BEAST.HelmholtzOperator2D` also includes the hypersingular
# operator, which needs a vector of amplitudes for its three components, as defined below.
function α(
    operator::Union{
        BEAST.MWDoubleLayer3D,
        BEAST.DoubleLayerRotatedMW3D,
        BEAST.HH3DSingleLayerFDBIO,
        BEAST.HH3DDoubleLayerFDBIO,
        BEAST.HH3DDoubleLayerTransposedFDBIO,
        BEAST.HH2DSingleLayerFDBIO,
        BEAST.HH2DDoubleLayerFDBIO,
        BEAST.HH2DDoubleLayerTransposedFDBIO,
    },
)
    return operator.alpha
end

function α(op::MWEFIE3D)
    return singlelayer(op).α
end

function α(op::MWHyperSingular3D)
    return -singlelayer(op).β * (singlelayer(op).gamma)^2
end

function α(operator::BEAST.MWSingleLayer3D)
    return [operator.α - operator.β * (operator.gamma)^2, operator.α, operator.α]
end

function α(op::BEAST.HH3DHyperSingularFDBIO)
    return [op.alpha, op.alpha, op.alpha, op.beta, op.beta, op.beta]
end

# `[α, α, β]`: two then one, where the 3-D operator is three then three. The split follows the
# integrand `(α f g (n̂ₓ·n̂ᵧ) + β f' g') G`: `α` weights the two in-plane components carrying `n̂ f`,
# and `β` weights the single scalar component carrying `f'`.
#
# This operator needs a vector, not the scalar `operator.alpha` returned by the method above.
function α(op::BEAST.HH2DHyperSingularFDBIO)
    return [op.alpha, op.alpha, op.beta]
end

function γ(op) #TODO: this can probably be removed (unnecessary wrapper)
    return BEAST.gamma(op)
end

# BEAST defines `gamma` for its 3-D operators only. The 2-D field uses the same convention,
# `γ = ik`, so read it directly. Adding a `BEAST.gamma` method on BEAST's types would be type piracy.
function γ(op::BEAST.HelmholtzOperator2D)
    return op.gamma
end

function sortbasis!(tree, sapce)
    newindices = _sortbasisindices!(tree, sapce)
    sapce.fns .= sapce.fns[newindices]
    sapce.pos .= sapce.pos[newindices]
    return sapce
end

function _sortbasisindices!(tree, basis)
    newindices = Vector{Int}(undef, numfunctions(basis))

    lastindex = 0
    for node in H2Trees.leaves(tree)
        points = H2Trees.values(tree, node)

        newpositionids = (1:length(points)) .+ lastindex

        newindices[points] .= newpositionids

        tree(node).data.values .= newpositionids

        lastindex += length(points)
    end

    return newindices
end

function interpolationtrait(::Union{MWEFIE3D,MWDoubleLayer3D,DoubleLayerRotatedMW3D})
    return SphericalθφInterpolation()
end

function interpolationtrait(::MWSingleLayer3D)
    return SphericalInterpolation()
end

function interpolationtrait(
    ::Union{
        BEAST.HH3DSingleLayerFDBIO,
        BEAST.HH3DDoubleLayerFDBIO,
        BEAST.HH3DDoubleLayerTransposedFDBIO,
        MWHyperSingular3D,
    },
)
    return VirtualExtension()
end

function interpolationtrait(::BEAST.HH3DHyperSingularFDBIO)
    return DoubleSphericalInterpolation()
end

# Scalar 2-D fields have one component with plain 2π periodicity and no sign change. A circle has
# no poles or spherical θ seam. `PeriodicφInterpolation` has no `converttrait` methods, so pairing
# it with a spherical grid raises a `MethodError` instead of applying a spherical pole rule.
function interpolationtrait(::BEAST.HelmholtzOperator2D)
    return PeriodicφInterpolation()
end

# The hypersingular field has three components (`[n̂₁f, n̂₂f, f']`) with the same periodicity.
# `_checkcirculararity` rejects a trait whose component count differs from the field's.
function interpolationtrait(::BEAST.HH2DHyperSingularFDBIO)
    return PeriodicφInterpolation{3}()
end

function radiationoperator(op::Union{MWEFIE3D,MWDoubleLayer3D,DoubleLayerRotatedMW3D})
    return BEAST.MWFarField3D(; gamma=BEAST.gamma(op))
end

function radiationoperator(op::Union{MWSingleLayer3D,MWHyperSingular3D})
    return Operators.MWSingleLayerFarField3D(; gamma=BEAST.gamma(op))
end

function receiveoperator(op::MWEFIE3D)
    return BEAST.MWFarField3D(; gamma=(-BEAST.gamma(op)))
end

function receiveoperator(op::MWDoubleLayer3D)
    return BEAST.MWDoubleLayerFarField3D(;
        gamma=(-BEAST.gamma(op)), amplitude=BEAST.gamma(op)
    )
end

function receiveoperator(op::DoubleLayerRotatedMW3D)
    return BEAST.MWDoubleLayerRotatedFarField3D(;
        gamma=(-BEAST.gamma(op)), amplitude=BEAST.gamma(op)
    )
end

function receiveoperator(op::Union{MWSingleLayer3D,MWHyperSingular3D})
    return Operators.MWSingleLayerFarField3D(; gamma=(-BEAST.gamma(op)))
end

# The concrete far-field representation for each operator. Component counts and collection types
# are derived from this table below.
function farfieldtype(::Union{MWEFIE3D,MWDoubleLayer3D,DoubleLayerRotatedMW3D})
    return TangentialSphericalFarField
end

function farfieldtype(::BEAST.MWSingleLayer3D)
    return SphericalVectorFarField
end

function farfieldtype(::Operators.MWHyperSingular3D)
    return ScalarFarField
end

function farfieldvaluetype(op::BEAST.MWFarField)
    return SVector{3,scalartype(op)}
end

function radiationoperator(
    op::Union{BEAST.HH3DSingleLayerFDBIO,BEAST.HH3DDoubleLayerTransposedFDBIO}
)
    return Operators.HHSingleLayerFarField(; gamma=BEAST.gamma(op))
end

function radiationoperator(op::BEAST.HH3DDoubleLayerFDBIO)
    return Operators.HHDoubleLayerFarField(; gamma=BEAST.gamma(op))
end

function radiationoperator(op::BEAST.HH3DHyperSingularFDBIO)
    return Operators.HHHyperSingularFarField3D(; gamma=BEAST.gamma(op))
end

function receiveoperator(op::BEAST.HH3DSingleLayerFDBIO)
    return Operators.HHSingleLayerFarField(; gamma=(-BEAST.gamma(op)))
end
function receiveoperator(op::BEAST.HH3DDoubleLayerFDBIO)
    return Operators.HHSingleLayerFarField(;
        gamma=(-BEAST.gamma(op)), amplitude=BEAST.gamma(op)
    )
end

function receiveoperator(op::BEAST.HH3DDoubleLayerTransposedFDBIO)
    return Operators.HHDoubleLayerFarField(;
        gamma=(-BEAST.gamma(op)), amplitude=(-BEAST.gamma(op))
    )
end

function receiveoperator(op::BEAST.HH3DHyperSingularFDBIO)
    return Operators.HHHyperSingularFarField3D(; gamma=(-BEAST.gamma(op)))
end

# The receive pattern uses the opposite sign of γ. Use `γ(op)` because BEAST's accessor covers
# only its 3-D operators.
function radiationoperator(op::BEAST.HH2DHyperSingularFDBIO)
    return Operators.HHHyperSingularFarField2D(; gamma=γ(op))
end

function receiveoperator(op::BEAST.HH2DHyperSingularFDBIO)
    return Operators.HHHyperSingularFarField2D(; gamma=(-γ(op)))
end

function farfieldtype(
    ::Union{
        BEAST.HH3DSingleLayerFDBIO,
        BEAST.HH3DDoubleLayerFDBIO,
        BEAST.HH3DDoubleLayerTransposedFDBIO,
    },
)
    return ScalarFarField
end

function farfieldtype(::BEAST.HH3DHyperSingularFDBIO)
    return PairedSphericalVectorFarField
end

# BEAST's 2-D integrand is `(α f g (n̂ₓ·n̂ᵧ) + β f' g') G`, and on a curve `f.derivative` is a scalar
# where the 3-D operator's `f.curl` is a 3-vector. So the pattern is `[n̂₁f, n̂₂f, f']`: two in-plane
# components carrying the normal so that `n̂ₓ·n̂ᵧ` factors across the two boxes, plus one scalar.
# Verified against a dense BEAST far block at 1.59e-13.
function farfieldtype(::BEAST.HH2DHyperSingularFDBIO)
    return PlanarVectorScalarFarField
end

# --- the 2-D Helmholtz operators ------------------------------------------------------------------
#
# `HHSingleLayerFarField` and `HHDoubleLayerFarField` also assemble on 2-D spaces. Their worst
# relative errors against direct quadrature were 1e-10 and 5e-16 respectively. The kernel
# `exp(γ·dot(y, cartesian(p)))` works in either dimension; a curve segment's normal is an `SVector{2}`.
#
# Read `op.gamma` directly because BEAST's accessor covers only 3-D operators. The field uses
# MLFMA's convention, `γ = ik`. The 2-D hypersingular operator has separate methods above and uses
# `PlanarVectorScalarFarField`, not a spherical projection.
function radiationoperator(
    op::Union{BEAST.HH2DSingleLayerFDBIO,BEAST.HH2DDoubleLayerTransposedFDBIO}
)
    return Operators.HHSingleLayerFarField(; gamma=op.gamma)
end

function radiationoperator(op::BEAST.HH2DDoubleLayerFDBIO)
    return Operators.HHDoubleLayerFarField(; gamma=op.gamma)
end

function receiveoperator(op::BEAST.HH2DSingleLayerFDBIO)
    return Operators.HHSingleLayerFarField(; gamma=(-op.gamma))
end

function receiveoperator(op::BEAST.HH2DDoubleLayerFDBIO)
    return Operators.HHSingleLayerFarField(; gamma=(-op.gamma), amplitude=op.gamma)
end

function receiveoperator(op::BEAST.HH2DDoubleLayerTransposedFDBIO)
    return Operators.HHDoubleLayerFarField(; gamma=(-op.gamma), amplitude=(-op.gamma))
end

function farfieldtype(
    ::Union{
        BEAST.HH2DSingleLayerFDBIO,
        BEAST.HH2DDoubleLayerFDBIO,
        BEAST.HH2DDoubleLayerTransposedFDBIO,
    },
)
    return ScalarFarField
end

# `D` (the component count/storage length) is fully determined by the far-field type, so it is a
# derived quantity rather than a second per-operator table to keep in sync with the one above.
function farfielddims(op)
    return fielddim(farfieldtype(op))
end

# Each moment type pairs with exactly one radiation and one receive collection type (the pairing
# `@linkmomenttocollection` establishes in `farfield.jl`). Keyed on the far-field type rather than
# on the operator, so adding an operator means adding one `farfieldtype` method and nothing else.
radiationcollectiontype(::Type{<:ScalarFarField}) = ScalarRadiationFieldCollection
function radiationcollectiontype(::Type{<:TangentialSphericalFarField})
    return TangentialSphericalRadiationFieldCollection
end
function radiationcollectiontype(::Type{<:SphericalVectorFarField})
    return SphericalVectorRadiationFieldCollection
end
function radiationcollectiontype(::Type{<:CartesianVectorFarField})
    return CartesianVectorRadiationFieldCollection
end
function radiationcollectiontype(::Type{<:PairedSphericalVectorFarField})
    return PairedSphericalRadiationFieldCollection
end
function radiationcollectiontype(::Type{<:PlanarVectorScalarFarField})
    return PlanarVectorScalarRadiationFieldCollection
end

receivecollectiontype(::Type{<:ScalarFarField}) = ScalarReceiveFieldCollection
function receivecollectiontype(::Type{<:TangentialSphericalFarField})
    return TangentialSphericalReceiveFieldCollection
end
function receivecollectiontype(::Type{<:SphericalVectorFarField})
    return SphericalVectorReceiveFieldCollection
end
function receivecollectiontype(::Type{<:CartesianVectorFarField})
    return CartesianVectorReceiveFieldCollection
end
function receivecollectiontype(::Type{<:PairedSphericalVectorFarField})
    return PairedSphericalReceiveFieldCollection
end
function receivecollectiontype(::Type{<:PlanarVectorScalarFarField})
    return PlanarVectorScalarReceiveFieldCollection
end

function radiationfieldcollectiontype(op)
    return radiationcollectiontype(farfieldtype(op))
end

function receivefieldcollectiontype(op)
    return receivecollectiontype(farfieldtype(op))
end

function farfieldvaluetype(op::Operators.HHFarField)
    return scalartype(op)
end

function farfieldvaluetype(op::Operators.HHHyperSingularFarField3D)
    return SVector{6,scalartype(op)}
end

# This sizes the assembler's three-component accumulation
# buffer. Falling through to the generic `HHFarField` method above -- which returns a bare
# `scalartype` -- makes the buffer a `Vector{ComplexF64}` while the integrand returns `SVector{3}`,
# and the failure is `MethodError: Cannot convert SVector{3, ComplexF64} to ComplexF64` from inside
# the assembler, naming neither the operator nor the missing method.
function farfieldvaluetype(op::Operators.HHHyperSingularFarField2D)
    return SVector{3,scalartype(op)}
end

# Only the six operator/representation pairs `receivefields` in `lazyreceivefieldcollection.jl`
# currently implements get a rule here -- `DoubleLayerRotatedMW3D`, `HH3DDoubleLayerFDBIO` and
# `HH3DDoubleLayerTransposedFDBIO` have no lazy-receive support today (only a BEAST-assembled
# `ReceiveFieldCollection` via `receiveoperator`), and this preserves that rather than silently
# adding it.
#
# γ stays here permanently for `MWDoubleLayer3D` -- this is NOT a staging step toward moving it
# into `H2Factory.α(core)`. `α(core)` was traced into `H2Factory/src/farinteractions/
# disaggregation.jl`: a single global/leaf-level scalar already carrying BEAST's
# `operator.alpha`/`.beta` weak-form scaling, structurally different from γ here, which is a
# coefficient inside a component *permutation* (the θ↔φ swap), not a uniform prefactor. The
# farfield modernization plan §5 was revised to say this explicitly after tracing `α(core)`; do
# not "fix" this by relocating γ into it.
function receivecouplingrule(op::MWEFIE3D, ::Type{<:TangentialSphericalFarField})
    return (ComponentMap{1,1}(1), ComponentMap{2,2}(-1))
end

function receivecouplingrule(::Operators.MWHyperSingular3D, ::Type{<:ScalarFarField})
    return (ComponentMap{1,1}(-1),)
end

function receivecouplingrule(op::MWDoubleLayer3D, ::Type{<:TangentialSphericalFarField})
    g = γ(op)
    return (ComponentMap{1,2}(g), ComponentMap{2,1}(g))
end

function receivecouplingrule(::BEAST.MWSingleLayer3D, ::Type{<:SphericalVectorFarField})
    return (ComponentMap{1,1}(-1), ComponentMap{2,2}(1), ComponentMap{3,3}(-1))
end

function receivecouplingrule(::BEAST.HH3DSingleLayerFDBIO, ::Type{<:ScalarFarField})
    return (ComponentMap{1,1}(1),)
end

# The 2-D single layer, identical to its 3-D counterpart: a symmetric scalar kernel, so the receive
# pattern is the mirrored radiation pattern with coefficient +1.
#
# The 2-D double layer and its transpose have no mirrored receive rule, just as in 3-D. Their
# `k̂·n̂` factor changes sign under the antipodal map, so an identity coupling would give wrong
# results. Use Petrov-Galerkin to assemble their receive side. Galerkin construction warns when
# the rule is missing.
function receivecouplingrule(::BEAST.HH2DSingleLayerFDBIO, ::Type{<:ScalarFarField})
    return (ComponentMap{1,1}(1),)
end

# Identity on all three components: `receive(k̂)` against
# `radiation(-k̂)` by direct quadrature agreed to 6.6e-15 / 9.4e-15 / 3.2e-14.
#
# It differs from the double layers (which have no rule and are Petrov-only) for a reason that is
# visible in the integrand: their kernel carries `k̂·n̂`, which flips under the antipodal map, while
# here `n̂` is contracted with the other pattern's `n̂` and never with `k̂`. It differs from the 3-D
# hypersingular's signs because those come from θ̂/φ̂ moving under the spherical reflection; the
# in-plane Cartesian basis here does not move at all.
#
# `test/Operators/hypersingular2d.jl` checks the coupling against a dense reference, not just its
# factorization algebra.
function receivecouplingrule(
    ::BEAST.HH2DHyperSingularFDBIO, ::Type{<:PlanarVectorScalarFarField}
)
    return (ComponentMap{1,1}(1), ComponentMap{2,2}(1), ComponentMap{3,3}(1))
end

function receivecouplingrule(
    ::BEAST.HH3DHyperSingularFDBIO, ::Type{<:PairedSphericalVectorFarField}
)
    return (
        ComponentMap{1,1}(-1),
        ComponentMap{2,2}(1),
        ComponentMap{3,3}(-1),
        ComponentMap{4,4}(-1),
        ComponentMap{5,5}(1),
        ComponentMap{6,6}(-1),
    )
end

function receivecouplingrule(op)
    return receivecouplingrule(op, farfieldtype(op))
end

"""
    haslazyreceive(op) -> Bool

Whether `op` has a mirrored (lazy) receive path, i.e. a [`receivecouplingrule`](@ref MLFMA.Seams.receivecouplingrule) method.

`DoubleLayerRotatedMW3D`, `HH3DDoubleLayerFDBIO` and `HH3DDoubleLayerTransposedFDBIO` lack this
rule. Petrov maps support them through an assembled receive side (`receiveoperator`); errors
against a dense reference on a 1.2λ sphere were 2.6e-05, 3.9e-05 and 4.0e-05 respectively.
The 2-D double layer and its transpose also lack a mirrored receive rule.

A full Galerkin matvec needs this rule because it mirrors the test side instead of assembling it.
The lazy collection resolves the rule during multiplication, so a missing method fails there.
[`checklazyreceive`](@ref) warns at construction; aggregation alone remains usable.
"""
haslazyreceive(op) = applicable(receivecouplingrule, op, farfieldtype(op))

"""
    checklazyreceive(op)

Warn if `op` has no mirrored receive side, which a Galerkin matvec needs. See
[`haslazyreceive`](@ref).

Warn rather than raise because aggregation does not need the receive rule.
`test/Maps/aggregation.jl` uses such a Helmholtz map without disaggregating. A full matvec would
fail with a `MethodError` when the traversal first needs the rule.

`maxlog=1` for the same reason [`checkboxsizes`](@ref MLFMA.Assembly.checkboxsizes) uses it: one clear notice, not one per map in
a sweep.
"""
function checklazyreceive(op)
    haslazyreceive(op) && return nothing
    @warn(
        "$(typeof(op)) has no mirrored (lazy) receive rule, so a GALERKIN MATVEC on it will " *
            "fail: that side is built by mirroring the trial patterns, and " *
            "`MLFMA.receivecouplingrule` has " *
            "no method for this operator/representation pair " *
            "($(farfieldtype(op))). It IS supported through an assembled receive side: build a " *
            "PETROV map instead, `MLFMA.assemble(operator, testspace, trialspace)`. To make it work " *
            "Galerkin, add the `receivecouplingrule` method -- and verify it against a dense " *
            "reference rather than against its own algebra. Aggregation alone is unaffected.",
        maxlog = 1
    )
    return nothing
end
