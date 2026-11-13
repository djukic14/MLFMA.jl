
@efie MWEFIE3D

function efie(; gamma=nothing, wavenumber=nothing, alpha=nothing)
    return MWEFIE3D(; gamma=gamma, wavenumber=wavenumber, alpha=alpha)
end

function singlelayer(op::MWEFIE3D{T,U}) where {T,U}
    return op.operator
end

@mw3Dhypersingular MWHyperSingular3D

function hypersingular(; gamma=nothing, wavenumber=nothing, beta=nothing)
    return MWHyperSingular3D(; gamma=gamma, wavenumber=wavenumber, beta=beta)
end

function singlelayer(op::MWHyperSingular3D{T,U}) where {T,U}
    return op.operator
end
