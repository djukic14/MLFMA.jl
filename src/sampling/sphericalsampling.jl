"""
    struct SphericalSampling{T<:Number} <: MLFMA.AbstractSphericalSampling{T}

Gauss-Legendre sampling on (0, π) in θ and equidistant sampling on [0, 2π) in φ.
Quadrature assumes periodicity in φ. Interpolation assumes the following continuation:

```math
\t f(θ, φ) = f(θ, φ + 2π),          for φ < 0,
\t f(θ, φ) = f(θ, φ - 2π),          for φ ≥ 2π,

\t f(θ, φ) = ±f(-θ, φ + π),         for θ ≤ 0, 0 ≤ φ < π,
\t f(θ, φ) = ±f(-θ, φ - π),         for θ ≤ 0, π ≤ φ < 2π,
\t f(θ, φ) = ±f(2π - θ, φ + π),     for θ ≥ π, 0 ≤ φ < π,
\t f(θ, φ) = ±f(2π - θ, φ - π),     for θ ≥ π, π ≤ φ < 2π,
```

These rules apply to functions sampled on `SphericalSampling`.

# Fields

  - `Nθ::Int`: Number of θ values.
  - `Nφ::Int`: Number of φ values.

Node vectors and θ weights are computed by `θs`, `φs` and `wθs`; they are not stored fields.

# Usage

```julia
using MLFMA
using Test

sampling = MLFMA.SphericalSampling{Float64}(10, 20)
@show arraytype = MLFMA.arraytype(sampling){Float64}
k_cart, integrationweights = MLFMA.cartesiansamplepoints(sampling)
k_spherical, integrationweights = MLFMA.sphericalsamplepoints(sampling)

@test MLFMA.Nθ(sampling) == 10
@test MLFMA.Nφ(sampling) == 20

# f(θ, φ) = θ
f(x) = x[2]
fsampled = arraytype(undef, size(sampling))
fsampled .= f.(k_spherical)

result = sum(fsampled .* integrationweights)
@test result ≈ 2 * π^2

# first θ value and second φ value
@show sampling[1, 2]

# wrapping around is also included for indices <0 and >Nθ, Nφ
@show sampling[-1, 25]

# get f at the first θ value and first φ value
@show fsampled[1, 1, sampling]

# here we also support wrapping indices
@show fsampled[-1, 25, sampling]
```

# Constructor

  - `SphericalSampling{T}(Nθ::Int, Nφ::Int) where {T}`: Constructs a new `SphericalSampling`
    object with the given number of θ and φ values. `T` is the type of the samples.
"""
@auto_hash_equals struct SphericalSampling{T<:Number} <:
                         Sampling.AbstractSphericalSampling{T}
    Nθ::Int
    Nφ::Int
    function SphericalSampling{T}(Nθ::Int, Nφ::Int) where {T}
        @assert iseven(Nφ) && Nφ > 0 && Nθ > 0
        return new{T}(Nθ, Nφ)
    end
end
#TODO: is auto_hash_equals actually needed here?

"""
    SphericalSamplingBuilder{T}()

Builds a [`SphericalSampling{T}`](@ref) once the rate `(Nθ, Nφ)` has been derived from the target
accuracy and the box diameter.

The builder instance provides a concrete callable type and can carry construction state without
changing the caller's interface.

`T` is the sampling points' element type, not the operator's -- the grid is real for a complex `γ`.
"""
struct SphericalSamplingBuilder{T<:Number} end

(::SphericalSamplingBuilder{T})(Nθ::Int, Nφ::Int) where {T} = SphericalSampling{T}(Nθ, Nφ)

"""
    samplingrate(samplingbuilder, cutoffparameter) -> Tuple

The grid rates a builder needs to carry a pattern of degree `cutoffparameter`, as a tuple splatted
straight into the builder.

Each builder defines its grid's rule: spherical grids need two rates and circles need one.
Rate selection is separate from construction because `CircularSamplingBuilder` already takes one
argument, `Nφ`; a `builder(L)` method would be ambiguous with that rate.
"""
samplingrate(::SphericalSamplingBuilder, cutoffparameter::Integer) =
    Sampling.NθNφ(cutoffparameter)

"""
    cartesiansamplepoints(s::MLFMA.SphericalSampling{T}) where {T}

Generate Cartesian sampling points and weights for integrals on the surface of the
unit sphere `∂𝐵(0)`

```math
∫_{∂𝐵(0)} f(k̂) d S(k̂).
```

The integration assumes a `2π` periodic function in the spherical coordinate φ.

# Arguments

  - `s::MLFMA.SphericalSampling{T}`: The spherical sampling object.

# Returns

  - `k̂::Matrix{SVector{3,T}}`: Matrix of SVectors pointing from origin to surface of unit
    sphere.
  - `weights::Matrix{T}`: Matrix of integration weights.
"""
function cartesiansamplepoints(s::Sampling.SphericalSampling{T}) where {T}
    Nφ = Sampling.Nφ(s)

    θs = Sampling.θs(s)
    φs = Sampling.φs(s)
    wθ = Sampling.wθs(s)

    k̂ = Matrix{SVector{3,T}}(undef, size(s))
    weights = Matrix{T}(undef, size(s))

    sinθs = sin.(θs)
    cosθs = cos.(θs)
    sinφs = sin.(φs)
    cosφs = cos.(φs)

    for i in eachindex(θs)
        @inbounds integrationweight = Sampling._integrationweight(sinθs[i], wθ[i], Nφ)
        for j in eachindex(φs)
            @inbounds k̂[j, i] = SVector{3,T}(
                sinθs[i] * cosφs[j], sinθs[i] * sinφs[j], cosθs[i]
            )

            @inbounds weights[j, i] = integrationweight
        end
    end
    return k̂, weights
end

# Quadrature assumes periodicity in φ on [0, 2*pi].
"""
    sphericalsamplepoints(s::MLFMA.SphericalSampling{T}) -> Tuple{Matrix{SVector{3,T}}, Matrix{T}}

Generate sampling points in spherical coordinates and weights to perform integrals on the
surface of the unit sphere `∂𝐵(0)`

```math
∫_{∂𝐵(0)} f(k̂) d S(k̂).
```

The integration assumes a `2π` periodic function in φ.

# Arguments

  - `s::MLFMA.SphericalSampling{T}`: The `SphericalSampling` object.

# Returns

  - `k̂::Matrix{SVector{3,T}}`: Spherical coordinate tuples `(1, θ, φ)` on the unit sphere.
  - `weights::Matrix{T}`: Matrix of integration weights.
"""
function sphericalsamplepoints(s::Sampling.SphericalSampling{T}) where {T}
    Nφ = Sampling.Nφ(s)

    θs = Sampling.θs(s)
    φs = Sampling.φs(s)
    wθ = Sampling.wθs(s)

    k̂ = Matrix{SVector{3,T}}(undef, size(s))
    weights = Matrix{T}(undef, size(s))

    sinθs = sin.(θs)

    for i in eachindex(θs)
        @inbounds integrationweight = Sampling._integrationweight(sinθs[i], wθ[i], Nφ)

        @inbounds for j in eachindex(φs)
            k̂[j, i] = SVector(T(1), θs[i], φs[j])

            @inbounds weights[j, i] = integrationweight
        end
    end
    return k̂, weights
end

# Return the Gauss-Legendre weights in θ before the integration Jacobian is applied.
function wθs(s::SphericalSampling{T}) where {T}
    _, wθs = FastGaussQuadrature.gausslegendre(Nθ(s))
    return T.(wθs)
end

"""
    cutoffparameter(s::SphericalSampling) -> Int

Return the spherical-harmonic bandwidth associated with this grid's θ rate.

# Arguments

  - `s::SphericalSampling`: The `SphericalSampling` object.

# Returns

  - `cutoff::Int`: The cutoff parameter.
"""
function cutoffparameter(s::SphericalSampling)
    return ceil(Int, Sampling.Nθ(s) / 2) - 1
end

# Map an extended θ index to a stored node index.
function θindex(s::SphericalSampling, θindex::Integer)
    θindex = θindex < 1 ? (-θindex) : (θindex - 1)
    return θindex % Nθ(s) + 1
end

# Continue θ by reflection across the poles within the supported index range.
function θ(s::SphericalSampling{T}, θindex::Integer) where {T}
    (θindex < -2 * Nθ(s) || θindex > 2 * Nθ(s)) &&
        throw(error("Out of bounds: θindex: ", θindex))

    0 < θindex <= Nθ(s) && return θs(s)[θindex]

    θcoefficient = θindex < 1 ? -1 : 1
    θoffset = θindex < 1 ? pi * ((-θindex) ÷ Nθ(s)) : pi * ((θindex - 1) ÷ Nθ(s))
    θindex = Sampling.θindex(s, θindex)

    return T(θcoefficient * (θs(s)[θindex] + θoffset))
end

# Map an extended φ index to a stored node index.
function φindex(s::SphericalSampling, φindex::Integer)
    φindex = φindex < 1 ? (-φindex + 1) : (φindex - 1)
    return φindex % Sampling.Nφ(s) + 1
end

# Continue φ beyond the stored interval.
function φ(s::SphericalSampling{T}, φindex::Integer) where {T}
    0 < φindex <= Nφ(s) && return φs(s)[φindex]

    φcoefficient = φindex < 1 ? -1 : 1
    φoffset =
        φindex < 1 ? 2 * pi * ((-φindex + 1) ÷ Nφ(s)) : 2 * pi * ((φindex - 1) ÷ Nφ(s))
    φindex = Sampling.φindex(s, φindex)

    return T(φcoefficient * (φs(s)[φindex] + φoffset))
end

"""
    virtualextension(θindex::Integer, φindex::Integer, s::SphericalSampling)

Virtually extend the indices of θ and φ according to

```math
\t f(θ, φ) = f(θ, φ + 2π),          for φ < 0,
\t f(θ, φ) = f(θ, φ - 2π),          for φ ≥ 2π,

\t f(θ, φ) = -f(-θ, φ + π),         for θ ≤ 0, 0 ≤ φ < π,
\t f(θ, φ) = -f(-θ, φ - π),         for θ ≤ 0, π ≤ φ < 2π,
\t f(θ, φ) = -f(2π - θ, φ + π),     for θ ≥ π, 0 ≤ φ < π,
\t f(θ, φ) = -f(2π - θ, φ - π),     for θ ≥ π, π ≤ φ < 2π,
```

# Arguments

  - `θindex::Int`: The index of the θ value to get the virtual extension for.
  - `φindex::Int`: The index of the φ value to get the virtual extension for.
  - `s::SphericalSampling`: The spherical sampling strategy to use.

# Returns

The stored θ index, stored φ index and sign coefficient. Crossing either pole reflects θ,
shifts φ by half a turn and changes the sign; wrapping φ alone preserves the sign.
"""
function virtualextension(θindex::Integer, φindex::Integer, s::SphericalSampling)
    localθindex = θindex
    localφindex = φindex

    coefficient = 1

    if φindex < 1
        localφindex = Nφ(s) + localφindex + Nφ(s) * (-localφindex ÷ Nφ(s))
    elseif φindex > Nφ(s)
        localφindex = localφindex - (localφindex ÷ Nφ(s)) * Nφ(s)
        iszero(localφindex) ? localφindex = Nφ(s) : nothing
    end

    if θindex < 1 && localφindex <= (Nφ(s) ÷ 2)
        localφindex += Nφ(s) ÷ 2
        localθindex = localθindex * -1 + 1
        coefficient = -1
    elseif θindex < 1 && localφindex > (Nφ(s) ÷ 2)
        localθindex = localθindex * -1 + 1
        localφindex -= Nφ(s) ÷ 2
        coefficient = -1
    elseif θindex > Nθ(s) && localφindex <= (Nφ(s) ÷ 2)
        localφindex += Nφ(s) ÷ 2
        coefficient = -1
        localθindex = 2 * Nθ(s) - localθindex + 1
    elseif θindex > Nθ(s) && localφindex > (Nφ(s) ÷ 2)
        localφindex -= Nφ(s) ÷ 2
        coefficient = -1
        localθindex = 2 * Nθ(s) - localθindex + 1
    end

    localθindex < 1 ||
        localθindex > Nθ(s) && throw(error("Out of bounds: θindex: ", localθindex))

    return localθindex, localφindex, coefficient
end

"""
    globalindex(θindex, φindex, s)

Compute the global index corresponding to the given θindex and φindex in the
`SphericalSampling` s.

# Arguments

  - `θindex::Integer`: The index of the θ angle.
  - `φindex::Integer`: The index of the φ angle.
  - `s::SphericalSampling`: The SphericalSampling object.

# Returns

  - The global index corresponding to the given θindex and φindex.
"""
function globalindex(θindex::Integer, φindex::Integer, s::SphericalSampling)
    0 < θindex <= Sampling.Nθ(s) &&
        0 < φindex <= Sampling.Nφ(s) &&
        return φindex + (θindex - 1) * Sampling.Nφ(s)

    θindex = Sampling.θindex(s, θindex)
    φindex = Sampling.φindex(s, φindex)
    return φindex + (θindex - 1) * Sampling.Nφ(s)
end

"""
    θφindicesfromglobalindex(globalindex::Int, s::AbstractSphericalSampling)

Convert a linear index to θ and φ indices for an `AbstractSphericalSampling` product grid.

This uses `Nφ` and `[φ, θ]` storage order. Its inverse [`globalindex`](@ref) is specific to
`SphericalSampling` because out-of-range indices use that grid's wraparound and reflection rules.
The far-field store in `assembly/store/store.jl` uses this conversion during BEAST assembly.

# Arguments

  - `globalindex::Int`: The global index to convert.
  - `s::AbstractSphericalSampling`: A spherical product grid.

# Returns

  - `θindex::Int`: The θ index corresponding to the global index.
  - `φindex::Int`: The φ index corresponding to the global index.
"""
function θφindicesfromglobalindex(globalindex::Int, s::Sampling.AbstractSphericalSampling)
    θindex = (globalindex - 1) ÷ Sampling.Nφ(s) + 1
    φindex = globalindex - (θindex - 1) * Sampling.Nφ(s)

    return θindex, φindex
end

"""
    findθφ(s::SphericalSampling{T}, θ::T, φ::T, stencilsize::Integer) where {T}

Find the θ and φ indices used to place an interpolation stencil.
Odd stencil sizes use the closest θ node. Even sizes use the last θ node at or below the target,
clamped to leave a following node. Both use the φ interval containing the target.

# Arguments

  - `s::SphericalSampling{T}`: The SphericalSampling object.
  - `θ::T`: The θ value.
  - `φ::T`: The φ value.
  - `stencilsize::Integer`: The size of the stencil used for interpolation.

# Returns

  - The selected θ and φ indices.
"""
function findθφ(s::SphericalSampling{T}, θ::T, φ::T, stencilsize::Integer) where {T}
    # Odd and even stencils need different θ anchors to preserve interpolation symmetry.
    if isodd(stencilsize)
        return Sampling._findθclosest(s, θ), Sampling._findφ(s, φ)
    else
        return Sampling._findθfirst(s, θ), Sampling._findφ(s, φ)
    end
end

"""
    mirrorindices(s::MLFMA.SphericalSampling)

Compute the mirror indices for a given `SphericalSampling` object `s`. The mirror indices
are used to mirror a function sampled at `f(k̂)` to `f(-k̂)` by using the symmetries in
`SphericalSampling`.

# Arguments

  - `s::MLFMA.SphericalSampling`: The `SphericalSampling` object.

# Returns

  - `indices::Array{Int}`: An array of mirror indices.
"""
function mirrorindices(s::Sampling.SphericalSampling)
    Nθ = Sampling.Nθ(s)
    Nφ = Sampling.Nφ(s)
    # indices = Array(Int(1):(Nθ * Nφ))
    indices = reshape(Array(1:(Nθ * Nφ)), size(s))

    for blockindex in Int(1):ceil(Int, Nθ / 2)
        reverseblock = Nθ - blockindex + Int(1)

        blockindices = Sampling._φblockindex(s, blockindex)
        reverseblockindices = Sampling._φblockindex(s, reverseblock)

        temp = indices[blockindices[1]:blockindices[Int(Nφ / 2)]]
        indices[blockindices[1]:blockindices[Int(Nφ / 2)]] = indices[blockindices[Int(Nφ / 2) + 1]:blockindices[end]]
        indices[blockindices[Int(Nφ / 2) + Int(1)]:blockindices[end]] = temp

        if reverseblock == blockindex
            continue
        end

        temp = indices[reverseblockindices[1]:reverseblockindices[Int(Nφ / 2)]]
        indices[reverseblockindices[1]:reverseblockindices[Int(Nφ / 2)]] = indices[reverseblockindices[Int(Nφ / 2) + Int(1)]:reverseblockindices[end]]
        indices[reverseblockindices[Int(Nφ / 2) + Int(1)]:reverseblockindices[end]] = temp

        temp = indices[blockindices]
        indices[blockindices] = indices[reverseblockindices]
        indices[reverseblockindices] = temp
    end

    return indices
end

"""
    Base.enumerate(s::SphericalSampling)

Enumerate spherical coordinate pairs in storage order, with φ varying fastest.

# Arguments

  - `s::SphericalSampling`: The `SphericalSampling` object to enumerate.

# Returns

  - An iterator yielding a linear index and its `(θ, φ)` coordinate pair.
"""
function Base.enumerate(s::SphericalSampling)
    Iterator = [(θindex, φindex) for φindex in Sampling.φs(s), θindex in Sampling.θs(s)]
    return Base.Iterators.enumerate(Iterator)
end

"""
    size(s::SphericalSampling)

Return the array shape used by `MLFMA.arraytype(s)` for a function sampled on `s`.

# Arguments

  - `s::SphericalSampling`: The `SphericalSampling` object.

# Returns

  - A tuple `(Nφ, Nθ)`.
"""
function Base.size(s::SphericalSampling)
    return (Sampling.Nφ(s), Sampling.Nθ(s))
end

"""
    arraytype(::Type{SphericalSampling{T}}) where {T}

Return the array type used to store function values on `SphericalSampling`.

# Arguments

  - `::Type{SphericalSampling{T}}`: Type of `SphericalSampling`.

# Returns

The array type for `SphericalSampling{T}`.
"""
function arraytype(::Type{SphericalSampling{T}}) where {T}
    return Matrix
end

function arraytype(::Type{SphericalSampling})
    return Matrix
end

function arraytype(sampling::SphericalSampling)
    return arraytype(typeof(sampling))
end

"""
    Base.eltype(::Type{SphericalSampling{T}}) where {T}

Return the sampling point element type of `SphericalSampling{T}`.

# Arguments

  - `T`: The type parameter of `SphericalSampling`.

# Returns

The element type `T`.
"""
function Base.eltype(::Type{SphericalSampling{T}}) where {T}
    return T
end

function _findθclosest(s::SphericalSampling{T}, θ::T, lastθindex::Integer=1) where {T}
    θindex = searchsortedlast(Sampling.θs(s)[lastθindex:end], θ)
    iszero(θindex) && (θindex = 1)

    θindex = θindex + lastθindex - 1

    if θindex < Nθ(s)
        abs(Sampling.θs(s)[θindex] - θ) > abs(Sampling.θs(s)[θindex + 1] - θ) &&
            (θindex += 1)
    end

    return θindex
end

function _findθfirst(s::SphericalSampling{T}, θ::T, lastθindex::Integer=1) where {T}
    θindex = searchsortedlast(view(Sampling.θs(s), lastθindex:Nθ(s)), θ)
    iszero(θindex) && (θindex = 1)

    θindex = θindex + lastθindex - 1

    θindex == Nθ(s) && (θindex -= 1)

    return θindex
end

function _findφ(s::SphericalSampling, φ)
    φindex = floor(Int, φ / (2 * pi) * Sampling.Nφ(s)) + 1
    return φindex
end

function _θsφs(Nθ::Integer, Nφ::Integer, T::DataType)
    return Sampling._θsφs(Nθ, Nφ, Val(T))
end

function _θsφs(Nθ::Integer, Nφ::Integer, ::Val{T}) where {T}
    θs, wθ = FastGaussQuadrature.gausslegendre(Nθ)
    θs = pi / 2 * (θs .+ 1)

    # Exclude the repeated endpoint at 2π.
    φs = Array(range(T(0), T(2) * pi; length=(Nφ + 1)))
    φs = φs[1:(end - 1)]
    return T.(θs), T.(wθ), φs
end

function _integrationweight(sinθ::T, wθ::T, Nφ::Integer) where {T}
    return sinθ * wθ * T(pi * pi / Nφ)
end

function _φblockindex(s::Sampling.SphericalSampling, φindex::Int)
    Nφ = Sampling.Nφ(s)
    return ((φindex - Int(1)) * Nφ + Int(1)):((φindex) * Nφ)
end

"""
    θs(s::SphericalSampling{T}) -> Vector{T}

The polar sample positions, in radians on `[0, π]`.

These are Gauss-Legendre nodes mapped from `[-1, 1]` to `[0, π]`, not nodes in `cos θ`.
The integration weights include the `sin θ` Jacobian separately. The φ nodes are equispaced
and periodic (see `φs`).

Returned in the sampling's own element type, and computed on each call rather than stored.
"""
function θs(s::SphericalSampling{T}) where {T}
    θs, _ = FastGaussQuadrature.gausslegendre(Nθ(s))
    θs = pi / 2 * (θs .+ 1)
    return T.(θs)
end

function φs(s::SphericalSampling{T}) where {T}
    φs = Array(range(T(0), T(2) * pi; length=(Nφ(s) + 1)))
    φs = φs[1:(end - 1)]
    return φs
end

# Spherical sampling rates for a requested cutoff parameter.
"""
    NθNφ(cutoffparemeter::Int)

Calculate the number of samples in the θ and φ directions for a given cutoff parameter.

# Arguments

  - `cutoffparemeter::Int`: The cutoff parameter.

# Returns

A tuple `(Nθ, Nφ)` representing the number of samples in the θ and φ directions, respectively.

# Example

```julia
julia> NθNφ(2)
(6, 12)
```
"""
function NθNφ(cutoffparameter::I) where {I<:Integer}
    Nθ = I(2) * I(cutoffparameter + I(1))
    Nφ = I(2) * Nθ

    return Nθ, Nφ
end
