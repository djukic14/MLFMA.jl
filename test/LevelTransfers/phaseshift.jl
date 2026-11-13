using MLFMA
using Test
using CompScienceMeshes, BEAST
using StaticArrays
using ClusterTrees
using H2Factory
using H2Trees

# Independent reference for a child-to-parent re-referencing phase shift.
#
# A box-centered radiation moment is F_c(k̂) = ∫ exp(γ k̂·(r - c)) J(r) dr. Re-referencing it from
# the child center `childcenter` to the parent center `parentcenter` therefore multiplies it by
# exp(γ k̂·(childcenter - parentcenter)), and the parent-to-child direction is the same expression
# with the sign of the argument flipped. This is written out from that definition on purpose --
# it deliberately avoids `MLFMA.phasefactor` and `H2Trees.parentcenterminuschildcenter` so that it
# is a check on the physics, not a restatement of the production implementation.
function referencephaseshift(γ, k̂s, childcenter, parentcenter; toparent::Bool)
    Δ = childcenter - parentcenter
    sign = toparent ? one(γ) : -one(γ)
    return map(k̂ -> exp(sign * γ * (k̂[1] * Δ[1] + k̂[2] * Δ[2] + k̂[3] * Δ[3])), k̂s)
end

@testset "Phase Shift" begin
    λ = 0.1
    m = meshsphere(5 * λ, λ / 10)
    X = raviartthomas(m)

    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=λ / 10, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
        ),
    )

    γ = ComplexF64(1.0 * im)
    #TODO: test for an imaginary gamma
    ϵ = 1e-3

    samplingfunction = MLFMA.sampling

    samplings = Vector{MLFMA.SphericalSampling{Float64}}(
        undef, length(H2Trees.levels(tree)) - 1
    )

    for (i, level) in enumerate(H2Trees.levels(tree)[2:end])
        samplings[i] = samplingfunction(γ, ϵ, tree; level=level)[1]
    end

    p = MLFMA.PhaseShifter(γ, tree, H2Trees.levels(tree)[2:end], samplings)

    for (i, level) in enumerate(H2Trees.levels(tree))
        level <= 2 && continue
        parentsampling = samplingfunction(γ, ϵ, tree; level=level - 1)[1]
        sampling = samplingfunction(γ, ϵ, tree; level=level)[1]

        # the shift arrays live on the parent's direction grid, so the reference does too
        parentk̂, _ = MLFMA.cartesiansamplepoints(parentsampling)

        for node in H2Trees.LevelIterator(tree, level)
            childcenter = H2Trees.center(tree, node)
            parentcenter = H2Trees.center(tree, H2Trees.parent(tree, node))

            expectedtoparent = referencephaseshift(
                γ, parentk̂, childcenter, parentcenter; toparent=true
            )
            expectedtochild = referencephaseshift(
                γ, parentk̂, childcenter, parentcenter; toparent=false
            )

            ff = MLFMA.TangentialSphericalFarField(γ, parentsampling)
            ff[1] .= 1.0
            ff[2] .= 1.0
            MLFMA.shifttoparentphase!(ff, p, tree, node)

            @assert !all(iszero.(ff[1]))

            @test ff[1] ≈ expectedtoparent
            @test ff[2] ≈ expectedtoparent

            ff2 = MLFMA.TangentialSphericalFarField(γ, parentsampling)
            ff2[1] .= 1.0
            ff2[2] .= 1.0
            MLFMA.shifttochildphase!(ff2, p, tree, node)

            # parent-to-child is checked against its own reference, not merely against the
            # child-to-parent result -- an inverted direction must not be able to pass
            @test ff2[1] ≈ expectedtochild
            @test ff2[2] ≈ expectedtochild

            @test ff[1] ≈ conj(ff2[1])
            @test ff[2] ≈ conj(ff2[2])

            ff = MLFMA.TangentialSphericalFarField(γ, parentsampling)
            storagefarfield = MLFMA.TangentialSphericalFarField(γ, parentsampling)
            ff[1] .= 1.0
            ff[2] .= 1.0
            MLFMA.shifttoparentphase!(storagefarfield, ff, p, tree, node)

            @assert !all(iszero.(storagefarfield[1]))

            # the out-of-place form must agree with the same reference as the in-place one
            @test storagefarfield[1] ≈ expectedtoparent
            @test storagefarfield[2] ≈ expectedtoparent

            storagefarfield2 = MLFMA.TangentialSphericalFarField(γ, parentsampling)
            MLFMA.shifttochildphase!(storagefarfield2, ff, p, tree, node)

            @test storagefarfield2[1] ≈ expectedtochild
            @test storagefarfield2[2] ≈ expectedtochild

            @test storagefarfield[1] ≈ conj(storagefarfield2[1])
            @test storagefarfield[2] ≈ conj(storagefarfield2[2])
        end
    end
end

# THE SECTOR COUNT, AND WHAT THE BOUND ON `appendphaseshifts!`'s SEARCH IS FOR.
#
# The loop stops walking a level once every sector is recorded. That bound was the literal `8`, which
# is `2^3` -- true of an octree and false of the quadtrees MLFMA gained with 2-D support. On a
# quadtree the condition was never met, so the break never fired and the loop visited every node at
# the level instead of the four it needed.
#
# THIS IS NOT A PERFORMANCE TEST, and the measurement is worth writing down so nobody later reads it
# as one. On `meshcircle(10.0, 1/40)` the loop visited 267 nodes against 9 with the correct bound -- a
# 30x reduction worth **1.5 µs**, against a 94.9 s map construction. The body for an already-recorded
# sector is a `parent` call, a `sector` call and a dict lookup, so thirty times fewer of them is
# thirty times fewer of nothing.
#
# What is under test is that the RESULT is unchanged: an early exit that stopped too soon would drop
# sectors, and one that stopped on the wrong count would be the same bug in the other direction. So
# both dimensions are checked for a full sector table AND for every entry agreeing with an
# independently computed phase.
# The dummy is how a caller swaps the phase stage out of a matvec, so its constructor has to accept
# whatever samplings the map was built with. It was typed `Vector{SphericalSampling{T1}}` and refused
# circular ones -- the eighth and last member of the family this branch removed, and the only one
# whose cost was MEASURABILITY rather than a failed run: while it refused, the 2-D level-transfer
# phase share could not be isolated at all.
#
# Every argument is ignored, so the narrow bound was never carrying anything.
@testset "DummyPhaseShifter accepts every sampling family" begin
    γ = ComplexF64(im)
    for samplings in (
        [MLFMA.SphericalSampling{Float64}(4, 8)],
        [MLFMA.CircularSampling{Float64}(8)],
        [MLFMA.Equidistant.EquidistantSampling{Float64}(4, 8)],
    )
        @test MLFMA.DummyPhaseShifter(γ, nothing, 1:2, samplings) isa
            MLFMA.DummyPhaseShifter
    end

    # And the shape the strategy layer actually calls it through -- a partially applied type invoked
    # as a constructor, which is what `phaseshiftertype(strategy)(...)` does.
    @test MLFMA.DummyPhaseShifter() isa MLFMA.DummyPhaseShifter
end

@testset "The sector bound follows the tree's dimension" begin
    γ = ComplexF64(im)
    ϵ = 1e-3

    # Dimension-neutral, unlike `referencephaseshift` above, which unrolls a three-component dot on
    # purpose. Still independent of production: the convention is written out from the definition --
    # a radiation moment referenced to the child is re-referenced to the parent by
    # `exp(-γ k̂·(parentcentre - childcentre))` -- rather than taken from `MLFMA.phasefactor` or
    # `H2Trees.parentcenterminuschildcenter`.
    #
    # The sign is the whole content and it is easy to get backwards: writing `exp(+γ k̂·Δ)` here
    # scores a worst error of 1.0 (3-D) and 2.0 (2-D), i.e. `|e^{iθ} - e^{-iθ}|`, which is what an
    # O(1) result from a pure sign flip looks like.
    function referencephase(γ, k̂s, Δ)
        return map(k̂ -> exp(-γ * sum(k̂ .* Δ)), k̂s)
    end

    function checktree(tree, samplings, expectedsectors)
        phaseshifter = MLFMA.PhaseShifter(γ, tree, 1:(H2Trees.levels(tree)[end]), samplings)

        @test MLFMA.numberofsectors(tree) == expectedsectors
        # Every level records a FULL table, which is what makes the early exit reachable at all --
        # and is exactly what silently never happened in 2-D.
        for entries in phaseshifter.phaseshifts
            @test length(keys(entries)) == expectedsectors
        end

        worst = 0.0
        checked = 0
        for level in H2Trees.levels(tree)
            level <= 1 && continue
            k̂s, _ = MLFMA.cartesiansamplepoints(samplings[level - 1])
            for node in H2Trees.LevelIterator(tree, level)
                H2Trees.parent(tree, node) == 0 && continue
                Δ =
                    H2Trees.center(tree, H2Trees.parent(tree, node)) -
                    H2Trees.center(tree, node)
                want = referencephase(γ, k̂s, Δ)
                got = MLFMA.findphaseshift(phaseshifter, tree, node)
                worst = max(worst, maximum(abs, got .- want) / maximum(abs, want))
                checked += 1
            end
        end
        return worst, checked
    end

    @testset "octree: 8, and the phases are unchanged" begin
        λ = 0.1
        X = raviartthomas(meshsphere(5 * λ, λ / 10))
        tree = TwoNTree(
            X;
            builder=TwoNTreeBuilder(;
                minhalfsize=λ / 10, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
            ),
        )
        samplings = [
            MLFMA.sampling(γ, ϵ, tree; level=level)[1] for
            level in H2Trees.levels(tree)[2:end]
        ]

        worst, checked = checktree(tree, samplings, 8)
        # Measured 2.5e-16 over 14377 nodes. This is the fixture the earlier attempt at a 3-D control
        # got WRONG: a shallower sphere put `relevantlevels[begin]` at or above every level, so the
        # loop was never entered and the control reported zero visits -- a vacuous pass. Hence the
        # explicit count assertion below.
        @test worst < 1e-13
        @test checked > 10_000
    end

    @testset "quadtree: 4, and the phases are unchanged" begin
        X = lagrangec0d1(CompScienceMeshes.meshcircle(4.0, 1 / 20))
        tree = TwoNTree(X; builder=TwoNTreeBuilder(; minhalfsize=0.25, minvalues=0))
        samplings = [
            MLFMA.sampling(
                γ,
                ϵ,
                tree;
                level=level,
                samplingbuilder=MLFMA.CircularSamplingBuilder{Float64}(),
            )[1] for level in H2Trees.levels(tree)[2:end]
        ]

        worst, checked = checktree(tree, samplings, 4)
        # Measured 4.6e-16 over 104 nodes.
        @test worst < 1e-13
        @test checked > 50
    end
end
