"""
    GaussLegendreInterpolationInformation{stencilsize,T1}

Precomputed θ stencils and weights for one (source, target) Gauss-Legendre θ grid pair, shared by both
interpolator families: the Lagrange kernel in `sphericalinterpolation.jl` and the global-φ kernel in
`localglobalinterpolation.jl` dispatch on this same type. A change to any field here has to be checked
against both.

Both kernels read the same fields in the same order. The row-writing family stages through a
contiguous vector; it does not need different stencil storage. `test/LevelTransfers/interpolation.jl`
checks both kernels against one shared information object.

Ranges and plain `Vector` weights let the θ products use the fast `mul!` kernel:

  - `Vector{Int}` column indices make `view(fvals, :, indices)` non-strided, forcing LinearAlgebra's
    generic path. The extending sets are therefore stored
    as spanning `UnitRange`s with their weights zero-padded across any gap (see `MLFMA.spanningranges`).
  - `SVector{stencilsize,T1}` weights also miss this fast path; use `Vector{T1}` weights instead.

An EMPTY extending range (`1:0`) means "this target θ takes no extended contribution". `isassigned`
cannot express that once the storage is a bits type, so the kernels test `isempty` instead.
"""
struct GaussLegendreInterpolationInformation{stencilsize,T1}
    conformindices::Vector{UnitRange{Int}}
    posextendingindices::Vector{UnitRange{Int}}
    negextendingindices::Vector{UnitRange{Int}}
    weights::Vector{Vector{T1}}
    posextendingweights::Vector{Vector{T1}}
    negextendingweights::Vector{Vector{T1}}
    flippingφindices::Vector{Int}
    conformiterations::UnitRange{Int}
    nonconformiterations::Vector{Int}

    function GaussLegendreInterpolationInformation{stencilsize}(
        sampling::Sampling.SphericalSampling{T},
        interpolationsampling::Sampling.SphericalSampling{T};
        interpolationtrait=VirtualExtensionFlipθSign(),
    ) where {stencilsize,T}
        conformindices = UnitRange{Int}[]
        posextendingindices = Vector{Int}[]
        negextendingindices = Vector{Int}[]

        conformweights = Vector{T}[]
        posextendingweights = Vector{T}[]
        negextendingweights = Vector{T}[]

        lastθindex = 1

        _localθs = Vector{T}(undef, stencilsize)

        firstconformindex = 0
        lastconformindex = 0
        nonconformiterations = Int[]

        for (iter, θ) in enumerate(Sampling.θs(interpolationsampling))
            θindex = if isodd(stencilsize)
                Sampling._findθclosest(sampling, θ, lastθindex)
            else
                Sampling._findθfirst(sampling, θ, lastθindex)
            end
            lastθindex = θindex

            _θindices = _θinformation!(_localθs, θindex, stencilsize, sampling)
            _weights = _interpolationweights(
                Interpolation.lagrangeweights(_localθs), _localθs, θ
            )

            if all(>(0), _θindices) && all(<=(Sampling.Nθ(sampling)), _θindices)
                push!(conformindices, _θindices)
                push!(conformweights, Vector{T}(_weights))
                if iszero(firstconformindex)
                    firstconformindex = iter
                else
                    lastconformindex = iter
                end

            else
                _posweights = T[]
                _negweights = T[]

                _posextendingindices = Int[]
                _negextendingindices = Int[]
                for (i, _θindex) in enumerate(_θindices)
                    if 0 < _θindex <= Sampling.Nθ(sampling)
                        push!(_posextendingindices, _θindex)
                        push!(_posweights, _weights[i])
                    else
                        pushfirst!(
                            _negextendingindices,
                            Sampling.virtualextension(_θindex, 1, sampling)[1],
                        )
                        pushfirst!(
                            _negweights,
                            _getnegativeweights(_weights[i], interpolationtrait),
                        )
                    end
                end

                push!(posextendingindices, _posextendingindices)
                push!(negextendingindices, _negextendingindices)
                push!(posextendingweights, _posweights)
                push!(negextendingweights, _negweights)

                push!(nonconformiterations, iter)
            end
        end

        flippingφindices = vcat(
            ((Sampling.Nφ(sampling) ÷ 2) + 1):Sampling.Nφ(sampling),
            1:(Sampling.Nφ(sampling) ÷ 2),
        )

        lastconformindex = iszero(lastconformindex) ? firstconformindex : lastconformindex
        # Spanning ranges with gap-padded weights, for the strided `mul!` path -- see the type docstring
        # and `MLFMA.spanningranges`.
        posranges, paddedposweights = Interpolation.spanningranges(
            posextendingindices, posextendingweights, T
        )
        negranges, paddednegweights = Interpolation.spanningranges(
            negextendingindices, negextendingweights, T
        )

        return new{stencilsize,T}(
            conformindices,
            posranges,
            negranges,
            conformweights,
            paddedposweights,
            paddednegweights,
            flippingφindices,
            firstconformindex:lastconformindex,
            nonconformiterations,
        )
    end
end

function _getnegativeweights(weights, ::VirtualExtensionFlipθSign)
    return -weights
end

function _getnegativeweights(weights, ::VirtualExtension)
    return weights
end

"""
    _interpolationweights(barycentricweights, samples, newsample)

The barycentric Lagrange weights evaluating an interpolant at `newsample`.

Second-form barycentric interpolation: `wᵢ / (x - xᵢ)`, normalized by its own sum. `samples` are the
nodes, `barycentricweights` their precomputed barycentric weights.

Handles the exact-hit case explicitly. If `newsample` coincides with a node the reciprocal is
infinite, so the result is the corresponding unit vector instead: the interpolant reproduces the
sample there. Without that branch the formula would produce `Inf/Inf`.
"""
function _interpolationweights(barycentricweights, samples, newsample)
    invl = 1 ./ (newsample .- samples)
    if any(isinf, invl)
        index = findfirst(isinf, invl)
        weights = zeros(length(samples))
        weights[index] = 1
        return weights
    end
    weightsinvl = barycentricweights .* invl
    denom = sum(weightsinvl)
    return weightsinvl ./ denom
end

function _θinformation!(
    localθs, θindex::Int, stencilsize::Int, sampling::Sampling.SphericalSampling
)
    θindices = Interpolation.stencil(θindex, stencilsize)
    localθs .= Sampling.θ.(Ref(sampling), θindices)

    return θindices
end
