# Extend Sampling's accessors so assembly and level transfers can use this grid through the
# sampling interface. Generic `Nθ`, `Nφ`, `θsφs`, and `length` methods apply through AbstractSampling.
#
# No `wθs` method is provided. Use `exactquadratureweights` for the full array of spherical weights;
# these already account for the Jacobian and must not be multiplied by another `sinθ` factor.

"""
    EquidistantSamplingBuilder{T}()

Builds an [`EquidistantSampling`](@ref) once a rate `(Nθ, Nφ)` has been derived, mirroring
`MLFMA.SphericalSamplingBuilder` so the two grids are interchangeable at a construction site.
"""
struct EquidistantSamplingBuilder{T<:Number} end

(::EquidistantSamplingBuilder{T})(Nθ::Int, Nφ::Int) where {T} =
    EquidistantSampling{T}(Nθ, Nφ)

"""
    samplingrate(::EquidistantSamplingBuilder, cutoffparameter)

Use the same `NθNφ` rate rule as the Gauss-Legendre builder. The grids have the same angular counts
but different θ nodes.
"""
Sampling.samplingrate(::EquidistantSamplingBuilder, cutoffparameter::Integer) =
    Sampling.NθNφ(cutoffparameter)

"""
    cutoffparameter(s::EquidistantSampling) -> Int

The design degree inferred from the θ count, using the inverse of the standard rate rule.

`MLFMA.NθNφ(L)` returns `(2(L+1), 4(L+1))` to resolve products of degree-`L` patterns, whose
bandwidth reaches `2L`. `exactquadratureweights` integrates θ modes through `Nθ - 1`, so the θ
requirement is `2L ≤ Nθ - 1`. A hand-built grid must also provide enough φ samples; this accessor
checks neither its φ count nor the field's bandwidth.
"""
Sampling.cutoffparameter(s::EquidistantSampling) = ceil(Int, Sampling.Nθ(s) / 2) - 1

"""
    cartesiansamplepoints(s::EquidistantSampling) -> (k̂, weights)

The unit direction and the integration weight at every grid point, both `Nφ × Nθ`.

The weights come from [`exactquadratureweights`](@ref), not the midpoint rule. They account
analytically for the torus Jacobian `|sin θ|`; sampling that factor would give only algebraic convergence.
"""
function Sampling.cartesiansamplepoints(s::EquidistantSampling{T}) where {T}
    k̂ = Matrix{SVector{3,T}}(undef, size(s))
    @inbounds for (i, θ) in enumerate(s.θs), (j, φ) in enumerate(s.φs)
        k̂[j, i] = SVector{3,T}(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
    end
    return k̂, exactquadratureweights(s)
end

"""
    sphericalsamplepoints(s::EquidistantSampling) -> (k̂, weights)

Unit-sphere coordinates `(1, θ, φ)` and exact integration weights, both `Nφ × Nθ`.
The linear ordering matches [`cartesiansamplepoints`](@ref), with φ varying fastest.
Weights include the spherical Jacobian; no additional `sinθ` factor is needed.
"""
function Sampling.sphericalsamplepoints(s::EquidistantSampling{T}) where {T}
    k̂ = Matrix{SVector{3,T}}(undef, size(s))
    @inbounds for (i, θ) in enumerate(s.θs), (j, φ) in enumerate(s.φs)
        k̂[j, i] = SVector{3,T}(one(T), θ, φ)
    end
    return k̂, exactquadratureweights(s)
end

"""
    arraytype(::EquidistantSampling) -> Matrix

The container a function sampled on this grid lives in.
"""
Sampling.arraytype(::Type{EquidistantSampling{T}}) where {T} = Matrix
Sampling.arraytype(::Type{EquidistantSampling}) = Matrix
Sampling.arraytype(s::EquidistantSampling) = Sampling.arraytype(typeof(s))

"""
    Base.enumerate(s::EquidistantSampling)

Enumerate `(θ, φ)` pairs in storage order, with φ varying fastest, as for `SphericalSampling`.
"""
function Base.enumerate(s::EquidistantSampling)
    return Base.Iterators.enumerate([(θ, φ) for φ in Sampling.φs(s), θ in Sampling.θs(s)])
end

"""
    supportedsymmetry(s::EquidistantSampling)

The same symmetry policy as `SphericalSampling`: the 16-element subgroup preserving the polar axis
when `Nφ` is divisible by four, and the smaller antipodal group otherwise.

The action requires exact index maps for `θ → π - θ` and `φ → σφ + β`. The uniform φ grid supports
quarter-turns when `Nφ` is divisible by four. The midpoint θ grid supports reversal because

    θᵢ + θ_{Nθ+1-i} = (i - ½)π/Nθ + (Nθ - i + ½)π/Nθ = π

Thus `π - θᵢ == θ_{Nθ+1-i}` in exact arithmetic. The action reverses indices rather than subtracting
stored angles. Combined with `φ → φ+π`, this is the antipodal map, not continuation across a pole.

The full 48-element lattice group is not supported: elements that move the polar axis into the
equatorial plane need not map this product grid onto itself.
"""
Sampling.supportedsymmetry(s::EquidistantSampling) = Sampling._axispreservingpolicy(s.Nφ)

"""
    angularpermutation(s::EquidistantSampling, symmetry)

The index permutation realizing `symmetry` on this grid; see
[`supportedsymmetry(::EquidistantSampling)`](@ref MLFMA.Sampling.supportedsymmetry(::MLFMA.Equidistant.EquidistantSampling))
for why the midpoint θ rule admits one, and
`MLFMA.angularpermutation(::MLFMA.SphericalSampling, _)` for the derivation of the θ/φ factorization.

Storage order is `(Nφ, Nθ)` with φ fastest -- the same order `Base.enumerate` and
`MLFMA.cartesiansamplepoints` use here, and the same one the spherical grid uses, which is what lets
the two share the index arithmetic.
"""
function Sampling.angularpermutation(
    s::EquidistantSampling, symmetry::H2Trees.LatticeSymmetry{3}
)
    return Sampling._θφpermutation(s.Nθ, s.Nφ, symmetry, s)
end
