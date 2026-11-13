"""
    AbstractInterpolator{T}

An interpolator resamples a far-field pattern on a different angular grid.

# Type parameters

  - `T`: stencilsize in interpolation
"""
abstract type AbstractInterpolator{stencilsize,T} end

function stencilsize(interpolator::AbstractInterpolator{S,T}) where {S,T}
    return S
end

function (interpolator::Interpolation.AbstractInterpolator{T})(
    farfield::F,
    farfieldsampling,
    interpolationsampling;
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
) where {D,TF,T,F<:FarFields.AbstractFarField{D,TF}}
    # Result keeps `farfield`'s own representation -- interpolation resamples the direction grid,
    # it does not reinterpret what the components mean.
    interpolatedfarfield = FarFields.basefarfieldtype(F)(eltype(TF), interpolationsampling)
    interpolator(
        interpolatedfarfield,
        farfield,
        farfieldsampling,
        interpolationsampling;
        interpolationtrait=interpolationtrait,
    )
    return interpolatedfarfield
end

function (interpolator::Interpolation.AbstractInterpolator)(
    interpolatedfarfields::F,
    farfields::F,
    farfieldsampling,
    interpolationsampling;
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
) where {F<:FarFields.AbstractFarField}
    return error("Not implemented: To be implemented for derived type")
end

function interpolationtrait(interpolator::AbstractInterpolator)
    return interpolator.interpolationtrait
end

"""
    AbstractAnterpolator{T}

An abstract type representing an anterpolator. Anterpolation is the adjoint operation to interpolation.
More information can be, for example, found in https://doi.org/10.1002/9781118844977 on page 194.

# Type parameters

  - `T`: stencilsize in anterpolation
"""
abstract type AbstractAnterpolator{stencilsize,T} end

function stencilsize(interpolator::AbstractAnterpolator{S,T}) where {S,T}
    return S
end
function (anterpolator::Interpolation.AbstractAnterpolator)(
    farfield::F,
    farfieldsampling,
    anterpolationsampling;
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
) where {D,T,F<:FarFields.AbstractFarField{D,T}}
    # See the interpolator functor above -- preserve the representation.
    anterpolatedfarfield = FarFields.basefarfieldtype(F)(eltype(T), anterpolationsampling)

    anterpolator(
        anterpolatedfarfield,
        farfield,
        farfieldsampling,
        anterpolationsampling;
        interpolationtrait=interpolationtrait,
    )
    return anterpolatedfarfield
end
function (anterpolator::Interpolation.AbstractAnterpolator)(
    anterpolatedfarfields::AbstractArray{F,1},
    farfields::AbstractArray{F,1},
    farfieldsampling,
    anterpolationsampling;
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
) where {F<:FarFields.AbstractFarField}
    error("Not implemented: To be implemented for derived type")
    return nothing
end

function interpolator(anterpolator::AbstractAnterpolator)
    return anterpolator.interpolator
end

function interpolationtrait(anterpolator::AbstractAnterpolator)
    return interpolationtrait(interpolator(anterpolator))
end

function lagrangeweightsequidistant(samples::AbstractArray{T,1}) where {T}
    n = length(samples) - 1
    return lagrangeweightsequidistant(n, T)
end

function lagrangeweightsequidistant(n::Int, T::DataType)
    result = Vector{T}(undef, n + 1)
    for i in eachindex(result)
        result[i] = T(-1)^(i - 1) * T(binomial(n, i - 1))
    end
    return result
end

function lagrangeweights(samples::AbstractArray{T,1}) where {T}
    weights = Vector{T}(undef, length(samples))
    for i in eachindex(samples)
        weight = T(1)
        for j in eachindex(samples)
            if i != j
                weight *= (samples[i] - samples[j])
            end
        end
        weights[i] = inv(weight)
    end
    return weights
end

"""
    stencil(θindex::Int, φindex::Int, stencilsize::Int)

Generate a stencil of indices around a given θ and φ index.

# Arguments

  - `θindex::Int`: The θ index around which to generate the stencil.
  - `φindex::Int`: The φ index around which to generate the stencil.
  - `stencilsize::Int`: The size of the stencil.

# Returns

Two vectors of indices, one for θ and one for φ.
"""
function stencil(θindex::Int, φindex::Int, stencilsize::Int)
    return Interpolation.stencil(θindex, stencilsize),
    Interpolation.stencil(φindex, stencilsize)
end
"""
    stencil(index::Int, stencilsize::Int)

Generate a stencil of indices around a given index.

# Arguments

  - `index::Int`: The index around which to generate the stencil.
  - `stencilsize::Int`: The size of the stencil.

# Returns

A vector of indices.
"""
function stencil(index::Int, stencilsize::Int)
    offset = iseven(stencilsize) ? 1 : 0

    indices =
        (index - floor(Int, stencilsize / 2) + offset):(index + floor(Int, stencilsize / 2))

    return indices
end

function _getmatrixarraysfromfarfields(
    interpolatedfarfields::AbstractArray{F,1}, farfields::AbstractArray{F,1}
) where {D,TF,F<:FarFields.AbstractFarField{D,TF}}
    T = eltype(TF)
    matrices = Vector{SVector{D,Matrix{T}}}(undef, length(farfields))
    resultmatrices = Vector{SVector{D,Matrix{T}}}(undef, length(farfields))
    id = 1
    #TODO: too many allocations here
    for i in eachindex(farfields)
        matrices[id] = SVector{D,Matrix{T}}([
            farfields[i][j] for j in eachindex(farfields[i])
        ])
        resultmatrices[id] = SVector{D,Matrix{T}}([
            interpolatedfarfields[i][j] for j in eachindex(interpolatedfarfields[i])
        ])
        id += 1
    end
    return matrices, resultmatrices
end

function _getinterpolationmatrix(
    interpolator::Interpolation.AbstractInterpolator,
    # Build the matrix by applying the operator to unit vectors. The grid traversal uses only
    # `length` and `enumerate`; the operator and probe fields must also support these samplings.
    sampling::Sampling.AbstractSampling,
    interpolationsampling::Sampling.AbstractSampling,
    T::DataType,
)
    intermatrix = spzeros(T, length(interpolationsampling), length(sampling))
    for (iter, sample) in enumerate(sampling)
        # Probe component 1 with the tangential field's default trait. `SphericalθφInterpolation`
        # flips the sign across the θ boundary; a scalar probe uses `VirtualExtension` without
        # that flip and would produce a different matrix.
        ff = FarFields.TangentialSphericalFarField(one(T), sampling, Val(:zero))
        ff2 = FarFields.TangentialSphericalFarField(
            one(T), interpolationsampling, Val(:zero)
        )
        ff[1][iter] = 1.0

        interpolator(ff2, ff, sampling, interpolationsampling)
        intermatrix[:, iter] = ff2[1][:]
    end
    return intermatrix
end

function _getanterpolationmatrix(
    anterpolator::Interpolation.AbstractAnterpolator,
    anterpolationsampling,
    sampling,
    T::DataType,
)
    antermatrix = spzeros(T, length(sampling), length(anterpolationsampling))

    for (iter, sample) in enumerate(anterpolationsampling)
        # See `_getinterpolationmatrix` above for why this names the tangential type explicitly.
        ff = FarFields.TangentialSphericalFarField(
            one(T), anterpolationsampling, Val(:zero)
        )
        ff2 = FarFields.TangentialSphericalFarField(one(T), sampling, Val(:zero))
        ff[1][iter] = 1.0

        anterpolator(ff2, ff, anterpolationsampling, sampling)
        antermatrix[:, iter] = ff2[1][:]
    end
    return antermatrix
end

function anterpolator(::Nothing)
    return nothing
end

"""
    mulrotatedrows!(result, fvals, indices, weights, α=true, β=false)

`result = view(fvals, flippingφindices, indices) * weights`, computed as two contiguous row blocks
instead of one permuted view.

`flippingφindices` is always `vcat(Nφ÷2+1:Nφ, 1:Nφ÷2)` -- a rotation by half the φ count, which is why it
never has to be passed in: the rotation is recovered from `size(fvals, 1)`, and `SphericalSampling`
guarantees an even `Nφ`.

The permutation makes the view non-strided. Two contiguous blocks let `mul!` use its fast path:

    y[1:half]        = fvals[half+1:2half, indices] * weights
    y[half+1:2half]  = fvals[1:half,       indices] * weights

The fast path may sum in a different order from the permuted form. Compare results with `isapprox`,
not bitwise equality.

`α` and `β` are `mul!`'s scaling factors, so an accumulating caller passes `(true, true)`: the interpolator's
extending branch adds its mirrored contribution onto a column the positive branch already wrote.
"""
function mulrotatedrows!(result, fvals, indices, weights, α=true, β=false)
    half = size(fvals, 1) ÷ 2
    LinearAlgebra.mul!(
        view(result, 1:half), view(fvals, (half + 1):(2 * half), indices), weights, α, β
    )
    LinearAlgebra.mul!(
        view(result, (half + 1):(2 * half)), view(fvals, 1:half, indices), weights, α, β
    )
    return result
end

"""
    spanningranges(sets, weightsets, ::Type{T})

Convert push-built index sets and their weights into `(ranges, paddedweights)`, where each range spans its
set and the weights are zero-padded across any gap.

Shared by the interpolation and anterpolation information constructors -- both build their θ index sets by
pushing, and both feed them to a `mul!` that wants a strided column index.

Unassigned or empty entries become `1:0`, the empty range -- how the θ kernels recognise "this child θ
receives nothing", since `isassigned` cannot do that once the storage holds a bits type.

The sets may have gaps because the constructors skip zero weights. Padding those gaps with zeros
preserves the weighted sum and gives every entry a `UnitRange`. This keeps the column view strided
and lets `mul!` use its fast kernel without a union of index types.
"""
function spanningranges(
    sets::Vector{Vector{Int}}, weightsets::Vector{Vector{T}}, ::Type{T}
) where {T}
    ranges = Vector{UnitRange{Int}}(undef, length(sets))
    padded = Vector{Vector{T}}(undef, length(sets))

    for i in eachindex(sets)
        if !isassigned(sets, i) || isempty(sets[i])
            ranges[i] = 1:0
            padded[i] = T[]
            continue
        end

        set = sets[i]
        weights = weightsets[i]
        low, high = extrema(set)
        ranges[i] = low:high

        entries = zeros(T, high - low + 1)
        for (k, index) in enumerate(set)
            entries[index - low + 1] = weights[k]
        end
        padded[i] = entries
    end
    return ranges, padded
end
