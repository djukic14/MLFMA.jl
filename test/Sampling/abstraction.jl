# The sampling abstraction has TWO layers, and this pins which methods live in which.
#
# WHY A TEST AND NOT JUST THE TYPES. The split is only worth anything if the upper layer stays
# genuinely free of θ×φ assumptions, and nothing in the existing suite can notice it drifting back:
# every sampling the package ships IS a product grid, so a `Nθ` quietly reintroduced on
# `AbstractSampling` would be inherited correctly by both of them and by every test. The only way to
# catch it is to exercise the upper layer with a grid that is NOT a product grid -- which is what
# `TestRingSampling` below is for.
#
# IT IS ALSO THE 2-D PREVIEW, at zero risk. A 2-D Helmholtz problem radiates into a CIRCLE: one angle,
# `SVector{2}` directions, `N = O(L)` rather than `O(L²)`. `TestRingSampling` is exactly that shape.
# Nothing in `src/` knows about it and nothing ships it -- it exists so the claim "the far-field
# containers and the phase layer are dimension-blind" is asserted rather than believed. Whatever a real
# `CircularSampling` ends up looking like, the assertions here are the floor it has to clear.

using LinearAlgebra
using MLFMA
using StaticArrays
using Test

# A ring of `N` equidistant directions in the xy-plane. Implements the `AbstractSampling` interface as
# documented in `src/sampling/abstractsampling.jl` and NOTHING ELSE -- no `Nθ`, no `Nφ`, no θ at all.
struct TestRingSampling{T<:Number} <: MLFMA.AbstractSampling{T}
    N::Int
    function TestRingSampling{T}(N::Int) where {T}
        iseven(N) ||
            error("TestRingSampling needs an even N so φ → φ+π is an index shift; got $N")
        return new{T}(N)
    end
end

_φs(s::TestRingSampling{T}) where {T} = T[2 * π * (j - 1) / s.N for j in 1:(s.N)]

Base.length(s::TestRingSampling) = s.N
# A ONE-tuple, not a rectangle. This is the property the far-field constructors are asserted against
# below: they derive their array rank from `size`, so this grid must yield a `Vector`.
Base.size(s::TestRingSampling) = (s.N,)
Base.enumerate(s::TestRingSampling) = Base.Iterators.enumerate(_φs(s))
MLFMA.arraytype(::Type{TestRingSampling{T}}) where {T} = Vector
MLFMA.arraytype(s::TestRingSampling) = MLFMA.arraytype(typeof(s))

function MLFMA.cartesiansamplepoints(s::TestRingSampling{T}) where {T}
    φs = _φs(s)
    k̂ = [SVector{2,T}(cos(φ), sin(φ)) for φ in φs]
    weights = fill(T(2) * T(π) / s.N, s.N)
    return k̂, weights
end

# `φ → φ + π`, which on an even-`N` ring is the exact index shift `j → j + N÷2`. Simpler than the
# spherical case, which has to reflect θ as well.
MLFMA.mirrorindices(s::TestRingSampling) = [mod1(j + s.N ÷ 2, s.N) for j in 1:(s.N)]

MLFMA.cutoffparameter(s::TestRingSampling) = s.N ÷ 2 - 1

@testset "The two sampling layers" begin
    spherical = MLFMA.SphericalSampling{Float64}(6, 12)
    equidistant = MLFMA.Equidistant.EquidistantSampling{Float64}(6, 12)
    ring = TestRingSampling{Float64}(12)

    @testset "type layer" begin
        @test MLFMA.AbstractSphericalSampling <: MLFMA.AbstractSampling
        @test MLFMA.AbstractSphericalSampling{Float64} <: MLFMA.AbstractSampling{Float64}

        # Every grid the package ships is a product grid, so nothing about the existing path changes.
        for s in (spherical, equidistant)
            @test s isa MLFMA.AbstractSphericalSampling
            @test s isa MLFMA.AbstractSampling
        end

        # The test ring is the case the split exists for: an `AbstractSampling` that is NOT one.
        @test ring isa MLFMA.AbstractSampling
        @test !(ring isa MLFMA.AbstractSphericalSampling)

        # And the SHIPPING type of that shape lands on the same side. Asserted here, in the test
        # that owns the split, so the two cannot drift apart -- `test/Sampling/circularsampling.jl`
        # covers the type's own contract.
        circular = MLFMA.CircularSampling{Float64}(12)
        @test circular isa MLFMA.AbstractSampling
        @test !(circular isa MLFMA.AbstractSphericalSampling)
        @test_throws MethodError MLFMA.Nθ(circular)
    end

    @testset "the product-grid layer is not inherited by a non-product grid" begin
        # Each of these was, before the split, defined on `AbstractSampling` and would have been
        # inherited here -- silently, and computed from an `Nθ`/`Nφ` this type does not have.
        @test_throws MethodError MLFMA.Nθ(ring)
        @test_throws MethodError MLFMA.Nφ(ring)
        @test_throws MethodError MLFMA.θsφs(ring)
        @test_throws MethodError ring[1, 1]
        @test_throws MethodError MLFMA.θφindicesfromglobalindex(1, ring)

        # `length` is the sharpest one: it USED to be `Nθ·Nφ` on the abstraction, so a grid that did
        # not override it reported a wrong direction count everywhere the pipeline sizes a buffer.
        # Now it is the ring's own, and the product formula belongs to the product layer.
        @test length(ring) == 12
        @test length(spherical) == MLFMA.Nθ(spherical) * MLFMA.Nφ(spherical)
    end

    @testset "the product-grid layer still works for the grids that have it" begin
        for s in (spherical, equidistant)
            @test MLFMA.Nθ(s) == 6
            @test MLFMA.Nφ(s) == 12
            @test length(s) == 72
            @test size(s) == (12, 6)

            θvalues, φvalues = MLFMA.θsφs(s)
            @test length(θvalues) == 6
            @test length(φvalues) == 12
            @test s[1, 2] == (MLFMA.θ(s, 1), MLFMA.φ(s, 2))

            # `θφindicesfromglobalindex` is pure arithmetic on `Nφ` and the `[φ, θ]` storage order,
            # so it is the same formula for EVERY product grid and belongs to this layer. Checked
            # against that order written out, not against `globalindex` -- see below for why those
            # are different questions.
            for θindex in 1:6, φindex in 1:12
                linear = φindex + (θindex - 1) * MLFMA.Nφ(s)
                @test MLFMA.θφindicesfromglobalindex(linear, s) == (θindex, φindex)
            end
        end

        # `globalindex` is NOT on the product layer and deliberately so: its out-of-range branch
        # resolves through `θindex`/`φindex`, whose wraparound and pole-reflection rules are the
        # Gauss-Legendre grid's own. It is defined for `SphericalSampling` only, and the equidistant
        # grid genuinely does not have it -- asserted here so that stays a decision rather than a gap
        # someone closes by widening the dispatch.
        for θindex in 1:6, φindex in 1:12
            @test MLFMA.globalindex(θindex, φindex, spherical) ==
                φindex + (θindex - 1) * MLFMA.Nφ(spherical)
        end
        @test_throws MethodError MLFMA.globalindex(1, 1, equidistant)
    end

    @testset "the dimension-agnostic layer really is dimension-agnostic" begin
        # Everything below runs on the RING, whose directions are `SVector{2}` and whose sampled
        # arrays are `Vector`s. None of it is reached through a θ or a φ.
        k̂, weights = MLFMA.cartesiansamplepoints(ring)
        @test size(k̂) == (12,)
        @test eltype(k̂) === SVector{2,Float64}
        @test all(≈(1.0), norm.(k̂))
        @test sum(weights) ≈ 2π

        # The antipodal map, which the receive side is built on.
        mirror = MLFMA.mirrorindices(ring)
        @test length(mirror) == 12
        @test all(k̂[mirror[j]] ≈ -k̂[j] for j in 1:12)

        # THE FAR-FIELD CONTAINERS. `farfield/farfield.jl` sizes off `size(sampling)` and derives the
        # array RANK from the tuple's length, which is what the header there claims makes it
        # dimension-generic. A 1-tuple must therefore give a `Vector`, not a `Matrix`.
        for constructor in (MLFMA.ScalarFarField, MLFMA.TangentialSphericalFarField)
            undeffield = constructor(ComplexF64, ring)
            zerofield = constructor(ComplexF64, ring, Val(:zero))
            for field in (undeffield, zerofield)
                @test field[1] isa Vector{ComplexF64}
                @test size(field[1]) == (12,)
            end
            @test all(iszero, zerofield[1])
        end

        # THE PHASE LAYER. `phasefactor` is the level-transfer primitive, and `dotu` is the only thing
        # in it that touches a direction -- so a 2-component direction and a 2-component centre are
        # all it needs. This is what makes the whole `phase/` layer free for 2-D.
        γ = 2.0im
        centre = SVector{2,Float64}(0.3, -0.7)
        phase = MLFMA.phasefactor(γ, k̂, centre, MLFMA.AbstractRadiationFieldCollection)
        @test phase isa Vector{ComplexF64}
        @test size(phase) == (12,)
        @test all(phase[j] ≈ cis(im * γ * (k̂[j] ⋅ centre)) for j in 1:12)

        # The receive convention is the conjugate sign, and that is the ONLY difference between the
        # two sides.
        receivephase = MLFMA.phasefactor(γ, k̂, centre, MLFMA.AbstractReceiveFieldCollection)
        @test all(receivephase[j] ≈ conj(phase[j]) for j in 1:12)

        # `getweights` is the map-construction helper that pulls the per-level weights; it reads
        # nothing but `cartesiansamplepoints`.
        weightsvector = MLFMA.getweights([ring, TestRingSampling{Float64}(24)])
        @test length(weightsvector) == 2
        @test sum(weightsvector[1]) ≈ 2π
        @test sum(weightsvector[2]) ≈ 2π

        # Bandwidth. The 3-D rule `NθNφ` does not apply here and is not used -- `cutoffparameter` is
        # the grid's own statement about itself, which is the whole reason it is required interface.
        @test MLFMA.cutoffparameter(ring) == 5
    end
end
