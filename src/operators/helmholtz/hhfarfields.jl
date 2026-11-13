abstract type HHFarField{K,U} <: BEAST.FarField end

struct HHSingleLayerFarField{K,U} <: HHFarField{K,U}
    gamma::K
    amplitude::U
end

struct HHDoubleLayerFarField{K,U} <: HHFarField{K,U}
    gamma::K
    amplitude::U
end

struct HHHyperSingularFarField3D{K,U} <: HHFarField{K,U}
    gamma::K
    amplitude::U
end

# The 2-D hypersingular far field has its own type because its integrand returns three components
# using `f.derivative` (a scalar). The 3-D integrand returns six using `f.curl` (a 3-vector).
struct HHHyperSingularFarField2D{K,U} <: HHFarField{K,U}
    gamma::K
    amplitude::U
end

function HHSingleLayerFarField(; gamma=nothing, wavenumber=nothing, amplitude=nothing)
    gamma, _ = BEAST.gamma_wavenumber_handler(gamma, wavenumber)

    @assert gamma !== nothing

    amplitude === nothing && (amplitude = 1)

    return HHSingleLayerFarField(gamma, amplitude)
end

function HHDoubleLayerFarField(; gamma=nothing, wavenumber=nothing, amplitude=nothing)
    gamma, _ = BEAST.gamma_wavenumber_handler(gamma, wavenumber)

    @assert gamma !== nothing

    amplitude === nothing && (amplitude = 1)

    return HHDoubleLayerFarField(gamma, amplitude)
end

function HHHyperSingularFarField3D(; gamma=nothing, wavenumber=nothing, amplitude=nothing)
    gamma, _ = BEAST.gamma_wavenumber_handler(gamma, wavenumber)

    @assert gamma !== nothing

    amplitude === nothing && (amplitude = 1)

    return HHHyperSingularFarField3D(gamma, amplitude)
end

function HHHyperSingularFarField2D(; gamma=nothing, wavenumber=nothing, amplitude=nothing)
    gamma, _ = BEAST.gamma_wavenumber_handler(gamma, wavenumber)

    @assert gamma !== nothing

    amplitude === nothing && (amplitude = 1)

    return HHHyperSingularFarField2D(gamma, amplitude)
end

function BEAST.kernelvals(op::HHFarField, y, p)
    return exp(op.gamma * dot(y, cartesian(p)))
end

function BEAST.integrand(op::HHSingleLayerFarField, krn, y, f, p)
    return op.amplitude * krn * f.value
end

function BEAST.integrand(op::HHDoubleLayerFarField, krn, y, f, p)
    return op.amplitude * krn * dot(y, normal(p)) * f.value
end

function BEAST.integrand(op::HHHyperSingularFarField3D, krn, y, f, p)
    n = normal(p)
    return (op.amplitude * krn) * vcat(n * f.value, f.curl)
end

# Three components: the in-plane normal weighted by the basis value, then the scalar arc-length
# derivative. Use `f.derivative`; a 2-D Lagrange refspace has no `curl` field.
#
# `vcat(n * f.value, f.derivative)` gives `SVector{3}` because `n` is `SVector{2}` and `f.derivative`
# is a scalar. In the 3-D case both halves are 3-vectors and
# the result is `SVector{6}`.
#
# Carrying `n̂` as components (rather than contracting it with anything here) is what lets the
# integrand's `n̂ₓ·n̂ᵧ` factor across the two boxes: it becomes the dot product of the two patterns'
# first two components. Verified end to end at 1.59e-13 against a dense BEAST far block.
function BEAST.integrand(op::HHHyperSingularFarField2D, krn, y, f, p)
    n = normal(p)
    return (op.amplitude * krn) * vcat(n * f.value, f.derivative)
end

function BEAST.scalartype(::HHFarField{K,U}) where {K,U}
    return K
end

# These methods extend BEAST's functions on MLFMA's own `HHFarField` type, so they are not type
# piracy. See `operators/maxwell/mwfarfields.jl` for the quadrature call convention and the
# ownership caveat for the `MWFarField` methods.
function BEAST.quaddata(op::HHFarField, rs, els, qs::BEAST.SingleNumQStrat)
    return quadpoints(rs, els, (qs.quad_rule,))
end

function BEAST.quadrule(op::HHFarField, rs, q, element, qdata, qs::BEAST.SingleNumQStrat)
    return qdata[1, q]
end
