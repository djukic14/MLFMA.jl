using Test
using MLFMA, H2Trees
using LinearAlgebra: norm
using StaticArrays: SVector

@testset "Angular action of a lattice symmetry" begin
    # A quarter turn: phi -> phi + pi/2. Off-grid unless Nphi is divisible by four.
    quarterturn = H2Trees.LatticeSymmetry((2, 1), (-1, 1))

    @testset "supported policy follows the grid" begin
        # Nphi is always even (CircularSampling enforces it), so the antipodal map is always
        # available; the quarter turns need one more factor of two.
        for N in (8, 16, 64, 252)
            @test MLFMA.supportedsymmetry(MLFMA.CircularSampling{Float64}(N)) ===
                H2Trees.FullLatticeSymmetry()
        end
        for N in (6, 10, 14, 18)
            @test MLFMA.supportedsymmetry(MLFMA.CircularSampling{Float64}(N)) ===
                H2Trees.OppositeSymmetry()
        end
    end

    @testset "permutations are exact" begin
        # The defining property: k[perm[j]] == S * k[j], for every element of the policy the
        # sampling advertises. Checked against the sample points themselves rather than against
        # a reimplementation of the index arithmetic.
        for N in (8, 10, 14, 16, 64)
            sampling = MLFMA.CircularSampling{Float64}(N)
            group = H2Trees.symmetrygroup(MLFMA.supportedsymmetry(sampling), Val(2))
            k, _ = MLFMA.cartesiansamplepoints(sampling)

            permutations = MLFMA.angularpermutations(sampling, group)
            @test length(permutations) == length(group)

            for (id, symmetry) in enumerate(group)
                permutation = permutations[id]
                @test permutation == MLFMA.angularpermutation(sampling, symmetry)
                @test sort(permutation) == 1:N          # it is a permutation, not just a map
                for j in 1:N
                    image = H2Trees.applysymmetry(symmetry, (k[j][1], k[j][2]))
                    @test norm(k[permutation[j]] - [image...]) < 1e-12
                end
            end

            # Composition: applying S then its inverse returns every index to itself. This is
            # what a translation kernel relies on when it uses the INVERSE permutation, and it
            # is the property a self-inverse-only test (antipodal) cannot check.
            for symmetry in group
                forward = MLFMA.angularpermutation(sampling, symmetry)
                backward = MLFMA.angularpermutation(
                    sampling, H2Trees.inversesymmetry(symmetry)
                )
                @test [backward[forward[j]] for j in 1:N] == 1:N
            end
        end
    end

    @testset "off-grid rotations raise instead of approximating" begin
        # Interpolating here would reintroduce exactly the approximation the reduction exists to
        # avoid, so an unusable symmetry is an error rather than a quietly resampled answer.
        for N in (6, 10, 14, 18)
            @test_throws "not a multiple of the sample spacing" MLFMA.angularpermutation(
                MLFMA.CircularSampling{Float64}(N), quarterturn
            )
        end
        for N in (8, 16, 64)
            @test MLFMA.angularpermutation(
                MLFMA.CircularSampling{Float64}(N), quarterturn
            ) isa Vector{Int32}
        end
    end

    @testset "the translation identity the reduction rests on" begin
        # T(k, S*d) == T(S^-1 k, d) on the grid, for a kernel depending on k.d alone. Verified
        # against the sample points with an explicit exponential rather than through a
        # translation operator, so a failure localizes to the angular action.
        for N in (8, 16, 64)
            sampling = MLFMA.CircularSampling{Float64}(N)
            group = H2Trees.symmetrygroup(MLFMA.supportedsymmetry(sampling), Val(2))
            k, _ = MLFMA.cartesiansamplepoints(sampling)
            κ = 2.3
            d = [0.7, -1.3]

            for symmetry in group
                Sd = [H2Trees.applysymmetry(symmetry, (d[1], d[2]))...]
                inverse = MLFMA.angularpermutation(
                    sampling, H2Trees.inversesymmetry(symmetry)
                )
                for j in 1:N
                    lhs = exp(im * κ * (k[j][1] * Sd[1] + k[j][2] * Sd[2]))
                    kinv = k[inverse[j]]
                    rhs = exp(im * κ * (kinv[1] * d[1] + kinv[2] * d[2]))
                    @test abs(lhs - rhs) < 1e-12
                end
            end
        end
    end

    # ------------------------------------------------------------------------------------------
    # THREE DIMENSIONS.
    #
    # The same four claims, on the grid where the reduction actually pays. Tolerances rather than
    # exact index-arithmetic comparisons throughout: `θ → π - θ` is an exact index reversal only
    # because the Gauss-Legendre nodes are symmetric, and their symmetry holds to round-off, not to
    # the last bit. So `cos θ_{Nθ+1-i} == -cos θ_i` is a 1e-16 statement, not a `==` one.
    @testset "three dimensions: the polar axis is what may not move" begin
        # Carries z into the equatorial plane -- the defining exclusion from the order-16 subgroup.
        polarmover = H2Trees.LatticeSymmetry((3, 2, 1), (1, 1, 1))
        # Fixes it: swaps x and y, flips z. In the subgroup, and chiral, so it is not its own
        # inverse and can detect a reversed convention.
        equatorialswap = H2Trees.LatticeSymmetry((2, 1, 3), (1, 1, -1))

        @testset "supported policy is the order-16 subgroup" begin
            # `NθNφ` gives `Nφ = 4(L+1)`, so every grid the rate rule produces takes the full
            # subgroup; the antipodal fallback is only reachable by hand.
            for (Nθ, Nφ) in ((4, 8), (6, 12), (8, 16), (12, 24))
                sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
                @test MLFMA.supportedsymmetry(sampling) ===
                    H2Trees.AxisPreservingSymmetry(3)
                @test length(
                    H2Trees.symmetrygroup(MLFMA.supportedsymmetry(sampling), Val(3))
                ) == 16
            end
            for (Nθ, Nφ) in ((4, 6), (6, 10), (8, 14))
                @test MLFMA.supportedsymmetry(MLFMA.SphericalSampling{Float64}(Nθ, Nφ)) ===
                    H2Trees.OppositeSymmetry()
            end
        end

        @testset "permutations are exact" begin
            # Odd `Nθ` is in the list on purpose: it puts a node exactly at the equator, which must
            # map to itself under the θ reversal. An even-only sweep never exercises that.
            for (Nθ, Nφ) in ((4, 8), (5, 12), (6, 12), (8, 16))
                sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
                group = H2Trees.symmetrygroup(MLFMA.supportedsymmetry(sampling), Val(3))
                k, _ = MLFMA.cartesiansamplepoints(sampling)
                flat = vec(k)
                N = Nθ * Nφ

                permutations = MLFMA.angularpermutations(sampling, group)
                @test length(permutations) == 16

                for (id, symmetry) in enumerate(group)
                    permutation = permutations[id]
                    @test permutation == MLFMA.angularpermutation(sampling, symmetry)
                    @test sort(permutation) == 1:N
                    for j in 1:N
                        image = H2Trees.applysymmetry(
                            symmetry, (flat[j][1], flat[j][2], flat[j][3])
                        )
                        @test norm(flat[permutation[j]] - [image...]) < 1e-12
                    end
                end

                for symmetry in group
                    forward = MLFMA.angularpermutation(sampling, symmetry)
                    backward = MLFMA.angularpermutation(
                        sampling, H2Trees.inversesymmetry(symmetry)
                    )
                    @test [backward[forward[j]] for j in 1:N] == 1:N
                end
            end
        end

        @testset "elements the grid cannot act with raise" begin
            sampling = MLFMA.SphericalSampling{Float64}(6, 12)
            @test_throws "into the equatorial plane" MLFMA.angularpermutation(
                sampling, polarmover
            )
            @test MLFMA.angularpermutation(sampling, equatorialswap) isa Vector{Int32}
            # An in-subgroup element can still be off-grid in φ, and that is a different message.
            @test_throws "not a multiple of the sample spacing" MLFMA.angularpermutation(
                MLFMA.SphericalSampling{Float64}(6, 10), equatorialswap
            )
        end

        @testset "the translation identity, on the real translation operator" begin
            # Unlike the two-dimensional testset above, which uses an explicit exponential to keep a
            # failure local, this one goes through `translationoperator!`. Both are wanted: the
            # exponential proves the angular action, this proves the thing the collection actually
            # stores obeys it.
            γ = ComplexF64(0, 2π)
            d = SVector{3,Float64}(2.0, -3.0, 1.0)

            for (Nθ, Nφ) in ((6, 12), (8, 16))
                sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)
                group = H2Trees.symmetrygroup(MLFMA.supportedsymmetry(sampling), Val(3))
                k, _ = MLFMA.cartesiansamplepoints(sampling)
                L = MLFMA.cutoffparameter(sampling)

                canonical = Matrix{ComplexF64}(undef, size(sampling))
                MLFMA.translationoperator!(canonical, γ, k, d, L)

                for symmetry in group
                    Sd = SVector{3,Float64}(
                        H2Trees.applysymmetry(symmetry, (d[1], d[2], d[3]))
                    )
                    actual = Matrix{ComplexF64}(undef, size(sampling))
                    MLFMA.translationoperator!(actual, γ, k, Sd, L)

                    inverse = MLFMA.angularpermutation(
                        sampling, H2Trees.inversesymmetry(symmetry)
                    )
                    reconstructed = [vec(canonical)[inverse[j]] for j in eachindex(inverse)]
                    @test norm(vec(actual) - reconstructed) / norm(vec(actual)) < 1e-12
                end
            end
        end
    end
end
