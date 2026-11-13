# `CircularSampling`: the one-angle grid, and the first REAL type to subtype `AbstractSampling`
# without being a product grid.
#
# `test/Sampling/abstraction.jl` already drives the generic layer with a fake ring, and deliberately
# so -- that test must keep working whether or not this type exists. What is asserted here is the
# type's own contract: that its quadrature is exact to the order it claims, that its antipodal map is a
# genuine permutation, and that `cutoffparameter` and `NφCircular` are actually inverse. Those are the
# three properties everything downstream will assume without rechecking.

using BEAST
using CompScienceMeshes
using H2Trees
using LinearAlgebra
using MLFMA
using StaticArrays
using Test

@testset "CircularSampling" begin
    @testset "construction" begin
        s = MLFMA.CircularSampling{Float64}(12)
        @test s isa MLFMA.AbstractSampling{Float64}
        # THE POINT OF THE WHOLE b1 SPLIT: a circle is not a product grid, and now says so.
        @test !(s isa MLFMA.AbstractSphericalSampling)
        @test eltype(s) === Float64

        @test MLFMA.CircularSampling(12) == MLFMA.CircularSampling{Float64}(12)

        # An odd Nφ would put the half-turn φ → φ+π BETWEEN samples, making `mirrorindices` lossy.
        # Rejected at construction rather than rounded.
        @test_throws ArgumentError MLFMA.CircularSampling{Float64}(11)
        @test_throws ArgumentError MLFMA.CircularSampling{Float64}(0)
        @test_throws ArgumentError MLFMA.CircularSampling{Float64}(-4)

        @test MLFMA.CircularSamplingBuilder{Float64}()(12) ==
            MLFMA.CircularSampling{Float64}(12)
    end

    @testset "the AbstractSampling interface" begin
        for Nφ in (4, 12, 30)
            s = MLFMA.CircularSampling{Float64}(Nφ)

            @test length(s) == Nφ
            # A ONE-tuple. This is what makes a sampled function a `Vector`.
            @test size(s) == (Nφ,)
            @test MLFMA.arraytype(s) === Vector

            φvalues = MLFMA.φs(s)
            @test length(φvalues) == Nφ
            @test φvalues[1] == 0
            # `2π` itself is excluded -- it is the same direction as `0`.
            @test φvalues[end] ≈ 2π * (Nφ - 1) / Nφ
            @test all(diff(φvalues) .≈ 2π / Nφ)

            k̂, w = MLFMA.cartesiansamplepoints(s)
            @test k̂ isa Vector{SVector{2,Float64}}
            @test length(k̂) == Nφ
            @test all(≈(1.0), norm.(k̂))
            # The circle's full measure, the way the spherical weights sum to 4π.
            @test sum(w) ≈ 2π
            @test all(≈(2π / Nφ), w)

            # `enumerate` must agree with the linear order of `cartesiansamplepoints`, since indices
            # produced against one are used to read the other.
            for (j, φ) in enumerate(s)
                @test k̂[j] ≈ SVector{2,Float64}(cos(φ), sin(φ))
            end
        end
    end

    @testset "angles wrap periodically, with no pole to reflect through" begin
        s = MLFMA.CircularSampling{Float64}(8)
        φvalues = MLFMA.φs(s)
        for j in 1:8
            @test MLFMA.φ(s, j) ≈ φvalues[j]
        end
        # Plain periodicity: no sign flip, no partner index, unlike the spherical grids.
        for j in (-9, -1, 0, 9, 17)
            @test cos(MLFMA.φ(s, j)) ≈ cos(φvalues[mod1(j, 8)])
            @test sin(MLFMA.φ(s, j)) ≈ sin(φvalues[mod1(j, 8)])
        end
    end

    @testset "mirrorindices is the exact antipodal permutation" begin
        for Nφ in (4, 12, 30)
            s = MLFMA.CircularSampling{Float64}(Nφ)
            k̂, _ = MLFMA.cartesiansamplepoints(s)
            mirror = MLFMA.mirrorindices(s)

            @test sort(mirror) == collect(1:Nφ)          # a permutation, so nothing is lost
            @test all(k̂[mirror[j]] ≈ -k̂[j] for j in 1:Nφ)
            @test mirror[mirror] == collect(1:Nφ)        # antipodal twice is the identity
        end
    end

    @testset "the bandwidth claim is exact, not asymptotic" begin
        for L in (1, 3, 7, 12)
            Nφ = MLFMA.NφCircular(L)
            s = MLFMA.CircularSampling{Float64}(Nφ)

            @test iseven(Nφ)
            @test Nφ == 2 * (L + 1)
            # The two must be exact inverses, or every level's rate drifts against its own cutoff.
            @test MLFMA.cutoffparameter(s) == L

            # `cutoffparameter` claims the grid integrates modes exactly. The trapezoidal rule on a
            # periodic interval is exact for `|m| < Nφ`, so check the claim directly rather than
            # trusting the formula: ∫e^{imφ}dφ is 2π at m = 0 and exactly 0 otherwise.
            _, w = MLFMA.cartesiansamplepoints(s)
            φvalues = MLFMA.φs(s)
            for m in 0:(Nφ - 1)
                integral = sum(w .* cis.(m .* φvalues))
                if m == 0
                    @test integral ≈ 2π
                else
                    @test abs(integral) < 1e-12
                end
            end

            # And the reason 2L matters: a PRODUCT of two degree-L patterns reaches 2L, which must
            # still be inside the exact range.
            @test 2L < Nφ
        end
    end

    @testset "the generic layer accepts it unchanged" begin
        s = MLFMA.CircularSampling{Float64}(16)
        k̂, w = MLFMA.cartesiansamplepoints(s)

        # A 1-tuple `size` must give a `Vector`, not a `Matrix` -- the property `farfield/farfield.jl`
        # claims makes it dimension-generic.
        field = MLFMA.ScalarFarField(ComplexF64, s, Val(:zero))
        @test field[1] isa Vector{ComplexF64}
        @test length(field[1]) == 16
        @test all(iszero, field[1])

        # The phase layer, on SVector{2} directions and an SVector{2} centre.
        γ = 2.0im
        centre = SVector{2,Float64}(0.4, -0.1)
        phase = MLFMA.phasefactor(γ, k̂, centre, MLFMA.AbstractRadiationFieldCollection)
        @test phase isa Vector{ComplexF64}
        @test all(phase[j] ≈ cis(im * γ * (k̂[j] ⋅ centre)) for j in 1:16)

        weightsvector = MLFMA.getweights([s, MLFMA.CircularSampling{Float64}(32)])
        @test length(weightsvector) == 2
        @test all(sum(v) ≈ 2π for v in weightsvector)
    end
end

# The accuracy-driven construction path: box size and target accuracy in, a grid out. Two hardcoded
# 3-D constants used to sit between those, and this pins both -- including that removing them left
# the 3-D answer bit-for-bit unchanged, which is the only reason the change is safe.
@testset "the rate rules follow the geometry" begin
    @testset "diameter is a box diagonal in D dimensions" begin
        for halfsize in (0.25, 1.0, 3.7)
            @test MLFMA.diameter(halfsize, 3) ≈ 2 * halfsize * sqrt(3)   # a cube
            @test MLFMA.diameter(halfsize, 2) ≈ 2 * halfsize * sqrt(2)   # a square
            @test MLFMA.diameter(halfsize, 1) ≈ 2 * halfsize             # a segment
        end
        # The old hardcoded form, kept as an explicit regression: 3-D must not have moved.
        @test MLFMA.diameter(0.25, 3) == 0.25 * sqrt(3) * 2
    end

    @testset "each builder states its own rate" begin
        for L in (1, 5, 13)
            # A product grid needs TWO rates, a circle ONE. That difference is the whole reason this
            # dispatches rather than being written out at the call site.
            @test MLFMA.samplingrate(MLFMA.SphericalSamplingBuilder{Float64}(), L) ==
                MLFMA.NθNφ(L)
            @test length(
                MLFMA.samplingrate(MLFMA.SphericalSamplingBuilder{Float64}(), L)
            ) == 2

            # NOT `NφCircular(L)` itself: the circular builder rounds UP to a multiple of four so
            # the quarter turn `φ → φ + π/2` is a grid point, which is what makes the full square
            # symmetry group available to the translation reduction. Asserted as "at least
            # exactness, and divisible by four" rather than a formula, so this states the two
            # properties that matter instead of restating the implementation.
            circularrate = MLFMA.samplingrate(MLFMA.CircularSamplingBuilder{Float64}(), L)
            @test length(circularrate) == 1
            @test circularrate[1] >= MLFMA.NφCircular(L)
            @test iszero(mod(circularrate[1], 4))
            # And it never overshoots by more than the rounding needs.
            @test circularrate[1] - MLFMA.NφCircular(L) in (0, 2)

            @test MLFMA.samplingrate(
                MLFMA.Equidistant.EquidistantSamplingBuilder{Float64}(), L
            ) == MLFMA.NθNφ(L)

            # Splatting the rate into its own builder must round-trip through `cutoffparameter`.
            grid = MLFMA.SphericalSamplingBuilder{Float64}()(
                MLFMA.samplingrate(MLFMA.SphericalSamplingBuilder{Float64}(), L)...
            )
            @test MLFMA.cutoffparameter(grid) == L

            # The circle round-trips to `>= L`, not `== L`, and that is the price of rounding the
            # rate up to a multiple of four: an even-`L` grid carries one degree more than asked
            # for. Safe -- exactness needs `2(L+1) < 2L+4` -- and it means the translation operator
            # is built one degree higher there. Bounded, so a rounding change cannot drift.
            circulargrid = MLFMA.CircularSamplingBuilder{Float64}()(
                MLFMA.samplingrate(MLFMA.CircularSamplingBuilder{Float64}(), L)...
            )
            @test MLFMA.cutoffparameter(circulargrid) in (L, L + 1)
        end
    end

    @testset "a circle gets a circular grid, at a square's rate" begin
        γ = 2π * im
        ϵ = 1e-4
        mesh = CompScienceMeshes.meshcircle(1.0, 0.1)
        X = lagrangec0d1(mesh)
        tree = H2Trees.TwoNTree(
            X; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.15, minvalues=0)
        )

        for level in H2Trees.levels(tree)
            grid, L = MLFMA.sampling(
                γ,
                ϵ,
                tree;
                level=level,
                samplingbuilder=MLFMA.CircularSamplingBuilder{Float64}(),
            )
            halfsize = H2Trees.halfsize(tree, H2Trees.LevelIterator(tree, level)[begin])

            @test grid isa MLFMA.CircularSampling
            @test length(grid) ==
                MLFMA.samplingrate(MLFMA.CircularSamplingBuilder{Float64}(), L)[1]
            @test length(grid) >= MLFMA.NφCircular(L)
            @test iszero(mod(length(grid), 4))
            @test MLFMA.cutoffparameter(grid) in (L, L + 1)

            # The rate follows the SQUARE's diagonal, read off the tree's own 2-component centres --
            # not the cube's, which would silently oversample.
            @test L == MLFMA.cutoffparameter(γ, ϵ, MLFMA.diameter(halfsize, 2))
            @test L < MLFMA.cutoffparameter(γ, ϵ, MLFMA.diameter(halfsize, 3))
        end
    end
end
