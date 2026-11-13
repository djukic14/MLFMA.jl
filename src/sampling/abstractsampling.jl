# `AbstractSampling` describes directions and integration weights without assuming a grid layout.
# `AbstractSphericalSampling` adds θ×φ product structure, including `length = Nθ·Nφ`.
# Circular and non-product grids implement the base interface directly.
# `cartesiansamplepoints` returns directions and integration weights together. These weights are
# distinct from the midpoint weights named `quadratureweights` inside `MLFMA.Equidistant`.

"""
    AbstractSampling{T<:Number}

A set of sampling directions on which a far field is represented, together with the quadrature weights
that integrate over them.

The layout may be a θ×φ product grid (see [`AbstractSphericalSampling`](@ref)), a set of rings,
or a one-angle circle. Generic far-field containers, phases and collection builders use this interface.

# Type parameters

  - `T`: the element type of the sampling points. Real even when the operator is complex -- the grid
    is geometry, not field data.

# Required interface

A concrete sampling must provide:

  - `Base.length(s)`: the number of directions.
  - `Base.size(s)`: the shape of an array holding one scalar per direction. May be any tuple length --
    the far-field constructors in `farfield/farfield.jl` derive their array rank from it, so a grid
    reporting a 1-tuple gets a `Vector` for free.
  - `Base.enumerate(s)`: the coordinates in storage order, which must agree with the linear
    order of `cartesiansamplepoints`.
  - [`arraytype`](@ref): the container a function sampled on this grid lives in.
  - [`cartesiansamplepoints`](@ref): `(k̂, weights)`, both shaped like `size(s)`.
  - [`mirrorindices`](@ref): the linear-index permutation realizing `k̂ → -k̂`.
  - [`cutoffparameter`](@ref): the degree this grid integrates exactly, i.e. its bandwidth.

`Base.eltype` comes free from the type parameter.

# See also

[`AbstractSphericalSampling`](@ref), [`SphericalSampling`](@ref).
"""
abstract type AbstractSampling{T<:Number} end

"""
    AbstractSphericalSampling{T<:Number} <: AbstractSampling{T}

An [`AbstractSampling`](@ref) whose directions form a product grid over the two spherical
coordinates: `Nθ` values of θ crossed with `Nφ` values of φ, stored `[φ, θ]` with φ contiguous.

`Nθ`, `Nφ`, two-index `getindex`, global-index arithmetic and `length = Nθ·Nφ` require this
rectangular structure. Non-product grids do not inherit these methods.

The concrete grid chooses the θ nodes and quadrature rule:
[`SphericalSampling`](@ref) places Gauss-Legendre nodes in θ, `MLFMA.Equidistant.EquidistantSampling`
places midpoint nodes. Code that relies on a particular rule, such as local θ interpolation,
must name the concrete type.

# Required interface

Everything [`AbstractSampling`](@ref) requires, plus:

  - [`Nθ`](@ref MLFMA.Sampling.Nθ) / `Nφ`: the two rates. Read off `s.Nθ` / `s.Nφ` by default.
  - [`θs`](@ref MLFMA.Sampling.θs) / [`φs`](@ref): the node vectors.
  - `θ(s, i)` / `φ(s, j)`: single nodes, with whatever out-of-range continuation the grid defines.
  - [`sphericalsamplepoints`](@ref): `(k̂, weights)` with `k̂` in spherical coordinates.

`length`, `θsφs`, the two-index `getindex` and [`θφindicesfromglobalindex`](@ref) then follow from
`Nθ`/`Nφ` and are provided here.
"""
abstract type AbstractSphericalSampling{T<:Number} <: AbstractSampling{T} end

function Base.eltype(s::Sampling.AbstractSampling{T}) where {T<:Number}
    return T
end

function Base.eltype(::Type{<:Sampling.AbstractSampling{T}}) where {T<:Number}
    return T
end

# --- the product-grid layer -----------------------------------------------------------------------

"""
    Nθ(s::AbstractSphericalSampling)
    Nθ(ff::AbstractFarField)

The number of polar (θ) samples on a product grid.

`Sampling` reads the grid's field; `FarFields` reads the stored array's second dimension.
Both methods use the same generic for the same polar count.

See [`Nφ`](@ref MLFMA.Sampling.Nφ) for the azimuthal count and [`θs`](@ref MLFMA.Sampling.θs) for the
sample positions themselves. `length(s)` is `Nθ * Nφ`, which holds only because the grid is a
rectangle.
"""
function Nθ(s::AbstractSphericalSampling)
    return s.Nθ
end

"""
    Nφ(s::AbstractSphericalSampling)
    Nφ(ff::AbstractFarField)

The number of azimuthal (φ) samples on a product grid.

The counterpart of [`Nθ`](@ref MLFMA.Sampling.Nθ). A far field is stored as `(Nφ, Nθ)` with φ
fastest, so its method reads the first array dimension while `Nθ` reads the second.
"""
function Nφ(s::AbstractSphericalSampling)
    return s.Nφ
end

function θsφs(s::AbstractSphericalSampling)
    return θs(s), φs(s)
end

function Base.getindex(
    s::Sampling.AbstractSphericalSampling, θindex::Integer, φindex::Integer
)
    return θ(s, θindex), φ(s, φindex)
end

# `Nθ · Nφ` counts directions only on product grids, not on every `AbstractSampling`.
function Base.length(s::Sampling.AbstractSphericalSampling)
    return Sampling.Nθ(s) * Sampling.Nφ(s)
end

# function Base.isequal(s::A, s2::A) where {A<:MLFMA.AbstractSampling}
#     return Nθ(s) == Nθ(s2) && Nφ(s) == Nφ(s2)
# end

# function Base.:(==)(s::A, s2::A) where {A<:MLFMA.AbstractSampling}
#     return Nθ(s) == Nθ(s2) && Nφ(s) == Nφ(s2)
# end

function Base.getindex(
    vector::AbstractArray{<:Number,1},
    globalindex::Integer,
    s::Sampling.AbstractSphericalSampling,
)
    !(0 < globalindex <= Sampling.Nθ(s) * Sampling.Nφ(s)) &&
        throw(BoundsError(vector, globalindex))
    return vector[globalindex]
end
