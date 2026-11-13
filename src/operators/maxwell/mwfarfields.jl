struct MWSingleLayerFarField3D{K,U} <: BEAST.MWFarField
    gamma::K
    amplitude::U
end

function MWSingleLayerFarField3D(; gamma=nothing, wavenumber=nothing, amplitude=nothing)
    gamma, _ = BEAST.gamma_wavenumber_handler(gamma, wavenumber)
    @assert gamma !== nothing

    amplitude === nothing && (amplitude = 1)

    return MWSingleLayerFarField3D(gamma, amplitude)
end

function MWSingleLayerFarField3D(op::MWSingleLayer3D{T,U}) where {T,U}
    return MWSingleLayerFarField3D(op.gamma, sqrt(op.α * op.β))
end

function BEAST.integrand(op::MWSingleLayerFarField3D, krn, y, f, p)
    return op.amplitude * (krn * f[1])
end

function BEAST.scalartype(
    ::Union{
        BEAST.MWFarField3D{K,U},
        BEAST.MWDoubleLayerFarField3D{K,U},
        MWSingleLayerFarField3D{K,U},
        BEAST.MWDoubleLayerRotatedFarField3D{K,U},
    },
) where {K,U}
    return K
end

# These methods are type piracy: BEAST owns both `quaddata`/`quadrule` and `MWFarField`.
# Loading MLFMA adds them to BEAST's method table for the whole session.
#
# BEAST's `Functional` methods have these bodies, but `FarField` is not a `Functional`.
# Its `quadrule(::FarField, ...)` takes eight arguments, not the six used here. These methods
# therefore add support without overwriting existing methods; `MWFarField3DTD` is unrelated.
#
# Qualify the definitions so they extend BEAST's functions rather than shadowing its exports with
# local functions. `test/Package/beastinterface.jl` checks these bindings.
function BEAST.quaddata(op::BEAST.MWFarField, rs, els, qs::BEAST.SingleNumQStrat)
    return quadpoints(rs, els, (qs.quad_rule,))
end

function BEAST.quadrule(
    op::BEAST.MWFarField, rs, q, element, qdata, qs::BEAST.SingleNumQStrat
)
    return qdata[1, q]
end
