# The two operations that sit between a synthesised CARTESIAN pattern and the TANGENTIAL far field
# MLFMA actually stores: the frame projection, and the antipodal mirror.
#
# θ̂ and φ̂ carry the spherical frame's pole singularity, so tangential components need not have
# rapidly decaying scalar harmonic coefficients even when the vector field is smooth. SHE therefore
# expands the three Cartesian components (Eibert 2005 §II) and projects onto (θ̂, φ̂) after synthesis.
# Measured cost of that choice: it pays a third component and still compresses 5.3x-12x (see
# `benchmark/interpolation/shecalibration.jl`), and the projection itself is ~6 flops per direction
# against the synthesis GEMM's 3(P+1)², i.e. about 2.5 % at P = 9.
#
# In the sampled representation the Galerkin receive
# side reuses the radiation patterns through `MLFMA.mirrorindices` -- a permutation of the NΩ samples
# realising k̂ -> -k̂ (θ -> π-θ, φ -> φ+π), plus the ±1 component table `mirrorrule`. In COEFFICIENT
# space that same map is DIAGONAL:
#
#     Y_lm(-k̂) = (-1)^l Y_lm(k̂)
#
# one sign per DEGREE. So the receive side needs no coefficients, no index array and no mirror pass of
# its own -- only a sign, foldable straight into a second synthesis matrix that every group at a level
# shares. `parity` below is that sign vector and `mirrorsynthesis` is that matrix.
#
# The frame also supplies the component rule, so it needs no separate application. With k̂ at
# (θ, φ) and -k̂ at (π-θ, φ+π):
#
#     θ̂(π-θ, φ+π) = ( cosθcosφ,  cosθsinφ, -sinθ) =  θ̂(θ, φ)
#     φ̂(π-θ, φ+π) = ( sinφ,     -cosφ,      0   ) = -φ̂(θ, φ)
#
# i.e. exactly `mirrorrule(TangentialSphericalFarField) == (ComponentMap{1,1}(1), ComponentMap{2,2}(-1))`.
# Projecting AFTER synthesis reproduces the component signs for free. The tests assert this against
# MLFMA's own `mirrorindices` and rule table rather than restating the algebra.

"""
    sphericalframe(sampling) -> (θ̂, φ̂)

The tangential unit vectors at every direction of `sampling`, each as an `ndirections × 3` matrix of
Cartesian components.

This is the frame used by `StoreCartesianAsTangentialSpherical` during assembly
(`assembly/store/store.jl`). Here it is built from the sampling's angles for projection at
synthesis time, where the input is a matrix rather than one value at a time.
"""
function sphericalframe(sampling::Sampling.AbstractSphericalSampling)
    k̂, _ = Sampling.sphericalsamplepoints(sampling)
    θs = [p[2] for p in vec(k̂)]
    φs = [p[3] for p in vec(k̂)]
    return sphericalframe(θs, φs)
end

function sphericalframe(θs::AbstractVector, φs::AbstractVector)
    length(θs) == length(φs) || throw(DimensionMismatch("θs and φs must have equal length"))
    T = float(promote_type(eltype(θs), eltype(φs)))
    θ̂ = Matrix{T}(undef, length(θs), 3)
    φ̂ = Matrix{T}(undef, length(θs), 3)
    @inbounds for q in eachindex(θs)
        sθ, cθ = sincos(θs[q])
        sφ, cφ = sincos(φs[q])
        θ̂[q, 1] = cθ * cφ
        θ̂[q, 2] = cθ * sφ
        θ̂[q, 3] = -sθ
        φ̂[q, 1] = -sφ
        φ̂[q, 2] = cφ
        φ̂[q, 3] = zero(T)
    end
    return θ̂, φ̂
end

"""
    parity(P) -> Vector{Int}

`(-1)^l` at every coefficient index of an order-`P` expansion -- the antipodal map `k̂ -> -k̂` in
coefficient space.

Scaling an expansion by this and synthesising gives the pattern evaluated at the ANTIPODE of every
direction, which is what the Galerkin receive side needs from the radiation coefficients. See this
file's header for why that replaces both `MLFMA.mirrorindices` and the `mirrorrule` component table.
"""
function parity(P::Integer)
    signs = Vector{Int}(undef, numcoefficients(P))
    @inbounds for q in eachindex(signs)
        signs[q] = isodd(degreeof(q)) ? -1 : 1
    end
    return signs
end

"""
    mirrorsynthesis(t) -> Matrix

`t`'s synthesis matrix with the [`parity`](@ref) sign folded in: `S⁻[q, lm] = (-1)^l S[q, lm]`, so
`S⁻ a` evaluates the expansion `a` at the antipode of each of `t`'s directions.

One matrix per LEVEL, shared by every group there -- the same sharing that makes the forward synthesis
worth batching. Precomputing it costs `(P+1)²·NΩ` complex numbers (2.4 MB at P = 9, NΩ = 1568) and
saves applying the sign per group per matvec.
"""
function mirrorsynthesis(t::SphericalHarmonicTransform)
    S = copy(t.synthesis)
    signs = parity(order(t))
    @inbounds for lm in axes(S, 2), q in axes(S, 1)
        signs[lm] == 1 || (S[q, lm] = -S[q, lm])
    end
    return S
end

"""
    project!(F, C, θ̂, φ̂) -> F

Project a synthesised Cartesian pattern onto the spherical frame, keeping the two TANGENTIAL
components: `F[:, 2g-1] = θ̂ ⋅ C[:, 3g-2:3g]` and `F[:, 2g] = φ̂ ⋅ C[:, 3g-2:3g]`.

`C` is `(ndirections, 3 * ngroups)` as a batched synthesis produces -- group-major, component-minor --
and `F` receives `(ndirections, 2 * ngroups)` in the matching tangential layout, which is also the
order consecutive `TangentialSphericalFarField` moments occupy in H2Factory's moment arena.

Single-threaded; [`project`](@ref MLFMA.Harmonics.SphericalHarmonics.projectthreaded!) threads over groups.
Which to call is a size decision, not a preference -- see `benchmark/interpolation/shewrite.jl`.
"""
function project!(
    F::AbstractMatrix, C::AbstractMatrix, θ̂::AbstractMatrix, φ̂::AbstractMatrix
)
    ngroups = _checkprojection(F, C, θ̂, φ̂)
    @inbounds for g in 1:ngroups
        _projectgroup!(F, C, θ̂, φ̂, g)
    end
    return F
end

"""
    projectthreaded!(F, C, θ̂, φ̂) -> F

[`project!`](@ref) with the group loop threaded. Groups write disjoint columns of `F`, so this needs
no synchronisation.

Threading is useful for large batches. The synthesis GEMM uses BLAS threads, leaving projection as
the serial part of the leaf multiply. Threading it measured a 2.1x speedup at 272 groups.
At one group the spawn overhead alone costs 0.824 ms against 0.122 ms here, so the caller must choose by size.
"""
function projectthreaded!(
    F::AbstractMatrix, C::AbstractMatrix, θ̂::AbstractMatrix, φ̂::AbstractMatrix
)
    ngroups = _checkprojection(F, C, θ̂, φ̂)
    Threads.@threads for g in 1:ngroups
        _projectgroup!(F, C, θ̂, φ̂, g)
    end
    return F
end

@inline function _projectgroup!(F, C, θ̂, φ̂, g)
    @inbounds for q in axes(F, 1)
        x = C[q, 3g - 2]
        y = C[q, 3g - 1]
        z = C[q, 3g]
        F[q, 2g - 1] = θ̂[q, 1] * x + θ̂[q, 2] * y + θ̂[q, 3] * z
        F[q, 2g] = φ̂[q, 1] * x + φ̂[q, 2] * y + φ̂[q, 3] * z
    end
    return nothing
end

function _checkprojection(F, C, θ̂, φ̂)
    nd = size(F, 1)
    size(C, 1) == nd ||
        throw(DimensionMismatch("C has $(size(C, 1)) directions, F has $nd"))
    size(θ̂) == (nd, 3) || throw(DimensionMismatch("θ̂ is $(size(θ̂)), expected $((nd, 3))"))
    size(φ̂) == (nd, 3) || throw(DimensionMismatch("φ̂ is $(size(φ̂)), expected $((nd, 3))"))
    size(C, 2) % 3 == 0 ||
        throw(DimensionMismatch("C has $(size(C, 2)) columns, not a multiple of 3"))
    ngroups = size(C, 2) ÷ 3
    size(F, 2) == 2ngroups || throw(
        DimensionMismatch(
            "F has $(size(F, 2)) columns; $(size(C, 2)) Cartesian columns is $ngroups " *
            "groups, needing $(2ngroups)",
        ),
    )
    return ngroups
end
