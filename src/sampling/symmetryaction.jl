# How a lattice symmetry acts on an angular sampling.
#
# `H2Trees` canonicalizes translation directions up to a symmetry orbit and records which symmetry
# maps the stored representative back onto the direction an interaction needs. H2Factory carries
# the tag; this file defines its action on each angular grid. Reduction requires an exact sample
# permutation, not interpolation.

"""
    supportedsymmetry(sampling)

The largest lattice symmetry policy `sampling` can apply by an exact index permutation.

Use this policy with `H2Trees.SymmetryDirectionInvariancePerLevel`. A larger policy can choose
representatives that the sampling cannot map to the required offset. Aligned offsets may hide
that mismatch.
"""
function supportedsymmetry end

# Missing angular actions raise an error. Returning `NoSymmetry()` would hide the missing
# implementation behind a valid collection with no storage reduction.
function supportedsymmetry(sampling::Sampling.AbstractSampling)
    return error(
        "supportedsymmetry is not defined for $(typeof(sampling)). A lattice symmetry reduces " *
        "translation storage only if this sampling can apply the symmetry EXACTLY, by an index " *
        "permutation, and that is a per-sampling fact this package does not have for this grid " *
        "yet. Define `supportedsymmetry` and `angularpermutation` for it, or use the unreduced " *
        "`MLFMA.TranslationCollection`.",
    )
end

"""
    supportedsymmetry(s::CircularSampling)

`φ → φ + π/2` lands on a grid point only when `Nφ` is divisible by four, so the full square
symmetry is available exactly then. Otherwise only the antipodal map survives, which `Nφ` being
even already guarantees.
"""
function supportedsymmetry(s::CircularSampling)
    return if iszero(mod(s.Nφ, 4))
        H2Trees.FullLatticeSymmetry()
    else
        H2Trees.OppositeSymmetry()
    end
end

"""
    supportedsymmetry(s::SphericalSampling)

The order-16 subgroup fixing the polar axis, for a symmetric rule in θ and a uniform rule in φ
whose rate supports quarter turns.

The full order-48 group includes elements that carry the polar axis into the equatorial plane.
The θ×φ grid is not closed under those rotations: θ uses Gauss-Legendre nodes and φ is uniform.

Within the subgroup the two factors are independent, so the action is a
permutation. Such a symmetry is `blockdiag(A, ±1)` with `A` a 2x2 signed permutation, so it sends
`θ → θ` or `π - θ` and, separately, `φ → σφ + β`. `sin(π - θ) == sin θ`, so the `φ` factor does not
depend on which `θ` case holds.

The `φ` quarter turns need `Nφ % 4 == 0`, which [`NθNφ`](@ref) always delivers (`Nφ = 4(L+1)`), so
the fallback below is unreachable through the rate rule. It exists so a hand-built sampling gets a
smaller group rather than a wrong one.
"""
supportedsymmetry(s::SphericalSampling) = _axispreservingpolicy(s.Nφ)

# Shared by `SphericalSampling` and `Equidistant.EquidistantSampling`: both have a θ rule symmetric
# about π/2 and a uniform φ rule. The policy depends on those properties, not the θ node positions.
# It is not a method on `AbstractSphericalSampling`, which does not require symmetric quadrature.
function _axispreservingpolicy(Nφ::Int)
    return if iszero(mod(Nφ, 4))
        H2Trees.AxisPreservingSymmetry(3)
    else
        # Eight sign flips are supported, but H2Trees has no policy for that subgroup.
        # Use the smaller antipodal group. The standard rate rule never needs this fallback.
        H2Trees.OppositeSymmetry()
    end
end

"""
    angularpermutation(sampling, symmetry) -> Vector{Int32}

The index permutation `π` with `k̂[π[j]] == symmetry * k̂[j]` for every sample of `sampling`.

Materialized rather than computed inline in the translation kernel. Measured at 0.82-0.83x the
cost of the equivalent inline index arithmetic in an identical loop, and the storage is one vector
per (symmetry, level), shared by every translation at that level -- roughly 1% of what the
translation reduction saves. `Int32` because no angular grid this is used on approaches 2^31.

This is the permutation of `symmetry` itself. A translation kernel holding the
canonical translation for `q == applysymmetry(S, canonical)` needs `T(k̂, q) == T(S⁻¹k̂, canonical)`
and therefore wants the permutation of `inversesymmetry(S)`. The two differ for every element that
is not its own inverse; the antipodal map is its own inverse and cannot detect the confusion.

Raises rather than approximating when `symmetry` does not map the grid onto itself.
"""
function angularpermutation end

# A signed permutation of a plane sends φ -> σφ + β, with σ the determinant of its 2x2 block and β
# the angle that block sends (1,0) to. Read both from the symmetry images.
# The grid is preserved exactly when β is a multiple of 2π/Nφ.
#
# Shared by both dimensions. In three, the equatorial block of a symmetry fixing the polar axis is a
# two-dimensional signed permutation, so the φ action is the same function of the same two images.
function _φaction(e₁, e₂, Nφ::Int, symmetry, sampling)
    σ = e₁[1] * e₂[2] - e₁[2] * e₂[1]
    β = atan(float(e₁[2]), float(e₁[1]))

    shift = β * Nφ / (2π)
    rounded = round(Int, shift)
    isapprox(shift, rounded; atol=1e-9) || error(
        "angularpermutation: $(symmetry) rotates a $(typeof(sampling)) by $(β) radians in φ, " *
        "which is not a multiple of the sample spacing 2π/$(Nφ). Applying it would require " *
        "interpolation, which this reduction exists to avoid. Use " *
        "`supportedsymmetry(sampling)` to choose a policy this grid can act with.",
    )
    return σ, rounded
end

_φindex(j::Int, σ, rounded::Int, Nφ::Int) = mod(σ * (j - 1) + rounded, Nφ) + 1

function angularpermutation(s::CircularSampling, symmetry::H2Trees.LatticeSymmetry{2})
    N = s.Nφ
    e₁ = H2Trees.applysymmetry(symmetry, (1, 0))
    e₂ = H2Trees.applysymmetry(symmetry, (0, 1))
    σ, rounded = _φaction(e₁, e₂, N, symmetry, s)

    permutation = Vector{Int32}(undef, N)
    for j in 1:N
        permutation[j] = _φindex(j, σ, rounded, N)
    end
    return permutation
end

"""
    angularpermutation(s::SphericalSampling, symmetry)

A symmetry fixing the polar axis factors into an independent action on `θ` and on `φ`, and this is
the product of the two index maps.

`blockdiag(A, s₃)` sends `k̂ = (sinθ cosφ, sinθ sinφ, cosθ)` to
`(sinθ · A(cosφ, sinφ), s₃ cosθ)`. The third component gives `θ → θ` for `s₃ = +1` and `θ → π - θ`
for `s₃ = -1`; the first two give `φ → σφ + β` exactly as in two dimensions. The two do not
interact, because `sin(π - θ) == sin θ` leaves the equatorial factor's magnitude alone.

`θ → π - θ` is an exact index reversal because Gauss-Legendre nodes are symmetric about the
midpoint: `θᵢ` and `θ_{Nθ+1-i}` are reflections of each other. Without that quadrature symmetry,
these elements would require interpolation and could not be used for reduction.
"""
function angularpermutation(s::SphericalSampling, symmetry::H2Trees.LatticeSymmetry{3})
    return _θφpermutation(s.Nθ, s.Nφ, symmetry, s)
end

# The action shared by every θ x φ product grid with a θ rule symmetric about π/2 (see
# `_axispreservingpolicy` for why this is a shared helper and not an `AbstractSphericalSampling`
# method). `sampling` is carried only so the two error messages name the concrete grid.
function _θφpermutation(Nθ::Int, Nφ::Int, symmetry::H2Trees.LatticeSymmetry{3}, sampling)
    e₁ = H2Trees.applysymmetry(symmetry, (1, 0, 0))
    e₂ = H2Trees.applysymmetry(symmetry, (0, 1, 0))
    e₃ = H2Trees.applysymmetry(symmetry, (0, 0, 1))

    # Callers can supply any element, so reject polar-axis mixing here as well as in the policy.
    iszero(e₃[1]) && iszero(e₃[2]) || error(
        "angularpermutation: $(symmetry) carries the polar axis to $(e₃), i.e. into the " *
        "equatorial plane. A $(typeof(sampling)) is a tensor product of a θ rule with a uniform " *
        "rule in φ and is not closed under that -- the images land between θ samples. Use " *
        "`supportedsymmetry(sampling)`, which returns the order-16 subgroup fixing the polar axis.",
    )

    σ, rounded = _φaction(e₁, e₂, Nφ, symmetry, sampling)
    reverseθ = e₃[3] < 0

    permutation = Vector{Int32}(undef, Nθ * Nφ)
    for i in 1:Nθ
        θimage = reverseθ ? Nθ + 1 - i : i
        for j in 1:Nφ
            # `size(sampling) == (Nφ, Nθ)`, φ fastest -- the same linear order `globalindex` uses.
            permutation[j + (i - 1) * Nφ] = _φindex(j, σ, rounded, Nφ) + (θimage - 1) * Nφ
        end
    end
    return permutation
end

"""
    angularpermutations(sampling, group) -> Vector{Vector{Int32}}

[`angularpermutation`](@ref) for every element of `group`, indexed by symmetry ID.

Built once per level and shared by every translation at that level, which is what makes
materializing them cheap: the count is the group order, not the number of translations.
"""
function angularpermutations(sampling, group::H2Trees.SymmetryGroup)
    return [angularpermutation(sampling, symmetry) for symmetry in group]
end
