"""
    dotu(u, v)

Return the unconjugated dot product, `sum(u .* v)`. The phase `exp(γ k̂ · r)` uses a bilinear
form. Using `⋅` would conjugate `u` instead.

Both `phasefactor` and the far-field kernel in `beastfarfieldassembler.jl` use this helper.
"""
function dotu(u::Array{<:Real,1}, v::Array{<:Real,1})
    return u ⋅ v
end

function dotu(u, v)
    return sum(u .* v)
end

# The diagonal phase re-references a moment between box centres. `PhaseShifter` builds one per
# (level, sector); `InterpolationPhaseTransfer` combines it with interpolation.
# `_phase_k` selects `exp(-γ k̂·r)` for radiation and `exp(+γ k̂·r)` for receive fields.
# Transpose alone cannot supply the change of sign between the two sides.
# Leaf centering is handled separately in `assembly/beast/beastfarfieldassembler.jl`.

struct PhaseFunctor{C,K}
    center::C
    k::K
end

function (f::PhaseFunctor)(k̂)
    return cis(f.k * Phase.dotu(k̂, f.center))
end

# Radiation and receive conventions differ only in this sign. Dispatch accepts either the abstract
# collection family or a concrete subtype, including `phasefactor(..., typeof(collection))`.
# Unsupported types raise a `MethodError` rather than selecting a convention implicitly.
_phase_k(γ, ::Type{<:AbstractRadiationFieldCollection}) = im * γ
_phase_k(γ, ::Type{<:AbstractReceiveFieldCollection}) = -im * γ

# The output element type is `typeof(γ)`, without promotion against the direction or centre types.
#
#   * `eltype(k̂)` is `SVector{3,<:Real}` (the direction vectors), so promoting against it yields
#     `Any` -- `promote_type(ComplexF64, SVector{3,Float64}, Float64) === Any`.
#     `test/FarField/typestability.jl` checks that phase arrays retain a concrete element type.
#   * a real `γ` still works, because `k = ±im*γ` is then purely imaginary and
#     `cis(im*x) == exp(-x) + 0.0im` has an imaginary part of exactly zero, so the convert into a
#     real array is exact. Keying the array on `typeof(γ)` therefore preserves the caller's
#     precision (`Float32` in, `Float32` out) instead of widening it to complex.
function phasefactor(γ::Tγ, k̂, center, fieldtype) where {Tγ}
    array = Array{Tγ}(undef, size(k̂))
    return phasefactor!(array, γ, k̂, center, fieldtype)
end

function phasefactor!(array, γ, k̂, center, fieldtype)
    map!(PhaseFunctor(center, _phase_k(γ, fieldtype)), array, k̂)
    return array
end
