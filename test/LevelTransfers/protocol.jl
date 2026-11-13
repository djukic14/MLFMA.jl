using MLFMA
using Test
using CompScienceMeshes, BEAST
using StaticArrays
using LinearAlgebra
using H2Factory
using OhMyThreads
using H2Trees

include("../support/util_test.jl")

# The two remaining checks from the plan's Phase 3 test list: that disaggregation really reaches the
# level transfer through `transpose(I2I)` on the TEST side, and an allocation contract for the hot
# apply path.
#
# The plan asks for an "O2O poisoning" test -- fill the legacy O2O slot with NaNs and confirm
# disaggregation is still correct. There is no such slot any more: disaggregation resolves an I2I
# translator and transposes it. The intent is served more directly here by showing that the composed
# `transpose(test I2I)` action equals what the deleted `o2otranslation` computed, and that swapping in
# the trial side instead gives a different answer -- so a side mix-up could not go unnoticed.

@testset "Disaggregation routes through transpose(test I2I)" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ
    ϵ = 1e-3
    T = ComplexF64

    m = meshsphere(1 * λ, λ / 8)
    X = raviartthomas(m)
    operator = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=λ / 8, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
        ),
    )

    A = MLFMA.GalerkinMLFMA(
        operator,
        X;
        tree=tree,
        strategy=stubbedstrategy(
            operator, X; samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ)
        ),
    )

    # --- structural: which side, which direction ---------------------------------------------------
    trialcollection = H2Factory.triali2itranslator(A)
    testcollection = H2Factory.testi2itranslator(A)

    # distinct objects, or the side distinction is not being made at all
    @test trialcollection !== testcollection

    # aggregation takes the trial side forward; disaggregation the test side, transposed
    @test H2Factory.i2itranslator(A, H2Factory.isForward()) === trialcollection
    @test H2Factory.disaggregationi2itranslator(A, H2Factory.isForward()) === testcollection
    @test H2Factory.disaggregationdirection(H2Factory.isForward()) ==
        H2Factory.TransposeTranslation()

    # and the sides swap for a transposed map, which is what makes the two collections load-bearing
    @test H2Factory.i2itranslator(A, H2Factory.isTranspose()) === testcollection
    @test H2Factory.disaggregationi2itranslator(A, H2Factory.isTranspose()) ===
        trialcollection

    # --- behavioural: the composed action equals the deleted o2otranslation ------------------------
    phaseshifter = MLFMA.phaseshifter(A)
    samplings = MLFMA.samplings(A)
    minrelevantlevel = H2Factory.minrelevantlevel(A)

    checked = 0
    for level in (minrelevantlevel + 1):H2Trees.levels(tree)[end]
        for child in H2Trees.LevelIterator(tree, level)
            parent = H2Trees.parent(tree, child)
            parent == 0 && continue

            childsampling = samplings[level - minrelevantlevel + 1]
            parentsampling = samplings[level - 1 - minrelevantlevel + 1]

            incoming = MLFMA.TangentialSphericalFarField(T, parentsampling)
            incoming[1] .= 1.0 + 0.25im
            incoming[2] .= 0.5 - 0.75im

            # new path: resolve the TEST side's transfer and apply it transposed, exactly as the
            # traversal does for a forward map
            testtransfer = H2Factory.leveltransfer(testcollection, tree, parent, child)
            viaprotocol = MLFMA.TangentialSphericalFarField(T, childsampling)
            context = H2Factory.TranslationContext(
                OhMyThreads.SerialScheduler(),
                MLFMA.LevelTransferScratch(
                    MLFMA.TangentialSphericalFarField(T, parentsampling),
                    MLFMA.TangentialSphericalFarField(T, childsampling),
                    nothing,
                ),
            )
            H2Factory.applytransfer!(
                viaprotocol,
                incoming,
                testtransfer,
                H2Factory.TransposeTranslation(),
                H2Factory.Overwrite(),
                context,
            )

            # old path, spelled out here as the reference: shift to the child phase, then anterpolate
            # from the parent grid down to the child grid. This is what `o2otranslation` did.
            shifted = MLFMA.TangentialSphericalFarField(T, parentsampling)
            MLFMA.shifttochildphase!(shifted, incoming, phaseshifter, tree, child)
            viaoldpath = MLFMA.TangentialSphericalFarField(T, childsampling)
            MLFMA.anterpolator(A)(
                viaoldpath,
                shifted,
                parentsampling,
                childsampling;
                interpolationtrait=MLFMA.interpolationtrait(MLFMA.interpolator(A)),
            )

            @test viaprotocol[1] ≈ viaoldpath[1]
            @test viaprotocol[2] ≈ viaoldpath[2]

            # non-vacuity: the reference is not trivially zero
            @test !all(iszero, viaoldpath[1])

            # the side matters -- applying the TRIAL transfer instead must give something else, so a
            # trial/test mix-up on the disaggregation path would be caught
            trialtransfer = H2Factory.leveltransfer(trialcollection, tree, parent, child)
            viawrongside = MLFMA.TangentialSphericalFarField(T, childsampling)
            H2Factory.applytransfer!(
                viawrongside,
                incoming,
                trialtransfer,
                H2Factory.TransposeTranslation(),
                H2Factory.Overwrite(),
                context,
            )
            @test !(viawrongside[1] ≈ viaprotocol[1])

            checked += 1
            checked >= 6 && break
        end
        checked >= 6 && break
    end
    @test checked >= 6 # the loop actually ran
end

@testset "Level transfer apply allocation contract" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ
    T = ComplexF64

    childsampling = MLFMA.SphericalSampling{Float64}(8, 16)
    parentsampling = MLFMA.SphericalSampling{Float64}(6, 12)
    trait = MLFMA.SphericalθφInterpolation()
    interp = MLFMA.SphericalLagrangeInterpolator{4}(T, Float64)
    anterp = MLFMA.SphericalLagrangeAnterpolator{4}(T, Float64)

    k̂s, _ = MLFMA.cartesiansamplepoints(parentsampling)
    phase = MLFMA.phasefactor(
        γ, k̂s, λ * SVector(0.19, -0.27, 0.11), MLFMA.AbstractRadiationFieldCollection
    )
    transfer = MLFMA.InterpolationPhaseTransfer(
        interp, anterp, phase, childsampling, parentsampling, trait
    )

    bare = H2Factory.TranslationContext(
        OhMyThreads.SerialScheduler(), MLFMA.TangentialSphericalFarField(T, parentsampling)
    )
    # A REAL `interim`, not `nothing`, and the change is deliberate. The reverse directions used to
    # stage `in .* phase` into this tuple's `parentmoment`; they now hand the diagonal to the
    # anterpolator as `prescale`, which stages into its OWN parent-grid slot so it may consume the copy
    # and skip the restoring inverse FFT. That slot lives in `interim`, so a context with
    # `interim=nothing` takes the documented allocate-a-fresh-array fallback and this contract would be
    # measuring a configuration production never builds -- the workspace always supplies a real interim
    # via `interimscratch`. Building one here keeps the OVERHEAD bound below at its original strictness
    # rather than relaxing it to accommodate a fixture artefact.
    tupled = H2Factory.TranslationContext(
        OhMyThreads.SerialScheduler(),
        MLFMA.LevelTransferScratch(
            MLFMA.TangentialSphericalFarField(T, parentsampling),
            MLFMA.TangentialSphericalFarField(T, childsampling),
            MLFMA.interimscratch(anterp, childsampling, parentsampling),
        ),
    )

    inchild = MLFMA.TangentialSphericalFarField(T, childsampling)
    inchild[1] .= 1.0
    inchild[2] .= 1.0
    inparent = MLFMA.TangentialSphericalFarField(T, parentsampling)
    inparent[1] .= 1.0
    inparent[2] .= 1.0
    outparent = MLFMA.TangentialSphericalFarField(T, parentsampling)
    outchild = MLFMA.TangentialSphericalFarField(T, childsampling)

    # warm up before measuring, so `@allocated` never counts compilation (H2Trees' performance-test
    # pattern)
    function measure(f)
        f()
        f()
        return @allocated f()
    end

    # The reference points are the BARE interpolator/anterpolator calls. Nearly all of an apply's
    # allocation happens inside them and predates the level-transfer protocol work, so the contract
    # is on the OVERHEAD this file adds -- which is what a change here could regress. Pinning the
    # absolute figure instead would break as soon as the interpolators are optimized, without any
    # defect in the transfer.
    bareinterp = measure(
        () -> interp(
            outparent, inchild, childsampling, parentsampling; interpolationtrait=trait
        ),
    )
    bareanterp = measure(
        () -> anterp(
            outchild, inparent, parentsampling, childsampling; interpolationtrait=trait
        ),
    )

    forwardoverwrite = measure(
        () -> H2Factory.applydirectedtransfer!(outparent, inchild, transfer, false, bare)
    )
    forwardaccumulate = measure(
        () -> H2Factory.applydirectedtransfer!(outparent, inchild, transfer, true, bare)
    )
    transposeoverwrite = measure(
        () -> H2Factory.applydirectedtransfer!(
            outchild, inparent, transpose(transfer), false, tupled
        ),
    )
    adjointoverwrite = measure(
        () -> H2Factory.applydirectedtransfer!(
            outchild, inparent, adjoint(transfer), false, tupled
        ),
    )

    @info "level transfer apply allocations (bytes)" bareinterp bareanterp forwardoverwrite forwardaccumulate transposeoverwrite adjointoverwrite

    # Non-vacuity: if the bare kernels ever stop allocating, these deltas stop being meaningful and
    # this contract should be replaced by an absolute pin.
    @test bareinterp > 0
    @test bareanterp > 0

    # The transfer layer must not add a per-call buffer of its own. A few hundred bytes of headroom
    # covers small bookkeeping; an accidental array materialization here would be kilobytes.
    OVERHEAD = 1024
    @test forwardoverwrite - bareinterp <= OVERHEAD
    @test forwardaccumulate - bareinterp <= OVERHEAD
    @test transposeoverwrite - bareanterp <= OVERHEAD
    @test adjointoverwrite - bareanterp <= OVERHEAD

    # `elementwisemul!`, the one operation this file performs directly on moments, is allocation-free.
    @test measure(() -> MLFMA.elementwisemul!(outparent, outparent, phase)) == 0

    # The adjoint direction must not allocate a conjugated phase per call -- that is why `conjphase`
    # is stored on the transfer.
    @test adjointoverwrite <= transposeoverwrite + OVERHEAD
end

# Adjoint-map AGGREGATION routing.
#
# HISTORY, because this testset was once used to close the very defect it now pins. It was raised in
# review that H2Factory's aggregation functor hardcoded `ForwardTranslation()` whatever the map trait,
# so an adjoint map aggregated by applying the test collection FORWARD. That was dismissed with an
# argument recorded here: "the aggregation step needs the CONJUGATED operator, not the ADJOINTED one
# ... MLFMA's test collection stores `conj(phase)` and the interpolation matrix is real, so
# forward-applying it yields exactly `conj` of the trial transfer -- which is what the adjoint map
# needs."
#
# HALF RIGHT, AND THE HALF THAT WAS WRONG COST 0.343 RELATIVE ERROR. Right: aggregation runs
# child-to-parent in every orientation, so it needs a CONJUGATED operator and not an ADJOINTED one --
# adjoining would also transpose, turning a child -> parent step into parent -> child, which is the
# wrong shape for aggregation entirely. That is exactly why the fix is `ConjugateTranslation` and not
# `AdjointTranslation`.
#
# Wrong: WHICH transfer gets conjugated. An adjoint map aggregates the TEST side, so what it needs is
# `conj(T_test)` -- the conjugate of the operator the TRANSPOSE would have applied there. The old
# argument instead compared against `conj(T_trial)`, conjugating the other side's transfer. Since the
# test collection stores `conj(phase)`, those two differ by a phase conjugation for EVERY interpolator
# family, real or not; the stated `imag(intermatrix) == 0` guard was necessary but not sufficient, and
# it was the operator half alone. Measured consequence once fixtures could actually run an adjoint
# matvec: 0.343 far-only relative error on `test/Maps/galerkin.jl` (global-φ) and a full-sweep failure
# on `test/Maps/tworects.jl` (Lagrange, real operator) -- so this was never a complex-interpolator-only
# bug, the real-operator case was wrong too.
#
# WHAT IS ASSERTED NOW: `conj(T_test)` is what adjoint aggregation applies, and for a REAL
# interpolation operator that equals the TRIAL transfer. Both routes are computed here so the identity
# is pinned rather than asserted, and the old forward application is pinned as the DIFFERENT operator
# it is, so this cannot quietly return to green if the direction is dropped again.
#
# The transfer's `conjphase` field is load-bearing for the other half too: adjoint DISaggregation
# resolves the trial side and applies `AdjointTranslation()`, which is the path that reads it.
@testset "Adjoint-map aggregation applies the conjugated transfer" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    γ = im * κ
    T = ComplexF64

    childsampling = MLFMA.SphericalSampling{Float64}(8, 16)
    parentsampling = MLFMA.SphericalSampling{Float64}(6, 12)
    trait = MLFMA.SphericalθφInterpolation()
    interp = MLFMA.SphericalLagrangeInterpolator{4}(T, Float64)
    anterp = MLFMA.SphericalLagrangeAnterpolator{4}(T, Float64)

    k̂s, _ = MLFMA.cartesiansamplepoints(parentsampling)
    phase = MLFMA.phasefactor(
        γ, k̂s, λ * SVector(0.19, -0.27, 0.11), MLFMA.AbstractRadiationFieldCollection
    )

    trialtransfer = MLFMA.InterpolationPhaseTransfer(
        interp, anterp, phase, childsampling, parentsampling, trait
    )
    # what `interpolationphasecollections` stores on the test side
    testtransfer = MLFMA.InterpolationPhaseTransfer(
        interp, anterp, conj.(phase), childsampling, parentsampling, trait
    )

    intermatrix = MLFMA._getinterpolationmatrix(interp, childsampling, parentsampling, T)
    Mtrial = Diagonal(vec(phase)) * intermatrix

    # The conjugation identity only holds because interpolation itself is real; if that ever changes,
    # forward-applying the test collection stops being the conjugate of the trial transfer.
    @test maximum(abs, imag.(intermatrix)) == 0

    # non-vacuity: with a real or near-real phase, conjugating would be a no-op and this proves nothing
    @test maximum(abs, imag.(Mtrial)) > 0.1
    @test !(conj(Mtrial) ≈ Mtrial)

    context = H2Factory.TranslationContext(
        OhMyThreads.SerialScheduler(), MLFMA.TangentialSphericalFarField(T, parentsampling)
    )

    for i in (1, 5, 17)
        u = zeros(T, size(childsampling))
        u[i] = 1.0 + 0.5im
        input = MLFMA.TangentialSphericalFarField(SVector(u, copy(u)))

        # What the traversal does for an adjoint map: test side, CONJUGATE direction
        # (`aggregationdirection(::isAdjoint)`).
        out = MLFMA.TangentialSphericalFarField(T, parentsampling)
        H2Factory.applydirectedtransfer!(out, input, conj(testtransfer), false, context)

        # `conj(T_test)`, and -- because this family's interpolation operator is real -- that is
        # exactly the TRIAL transfer. Both are asserted: the first is the definition, the second is
        # the identity the old argument half-saw.
        Mtest = Diagonal(vec(conj.(phase))) * intermatrix
        @test vec(out[1]) ≈ conj(Mtest) * vec(u)
        @test vec(out[1]) ≈ Mtrial * vec(u)

        # It is NOT what the traversal used to do -- forward-applying the test transfer. That is
        # `conj(Mtrial)`, which differs from the above by a phase conjugation, and is the 0.343 error.
        forwardout = MLFMA.TangentialSphericalFarField(T, parentsampling)
        H2Factory.applydirectedtransfer!(forwardout, input, testtransfer, false, context)
        @test vec(forwardout[1]) ≈ conj(Mtrial) * vec(u)
        @test !(vec(forwardout[1]) ≈ vec(out[1]))

        # The conjugate direction must not have written through to the traversal's input.
        @test vec(input[1]) ≈ vec(u)
    end

    # Why `AdjointTranslation()` in aggregation would be wrong rather than merely differently scaled:
    # the adjointed transfer maps parent grid -> child grid, so it cannot produce an aggregated moment
    # on the parent grid at all.
    @test size(adjoint(Mtrial)) == (length(childsampling), length(parentsampling))
    @test size(Mtrial) == (length(parentsampling), length(childsampling))
    @test size(Mtrial) != size(adjoint(Mtrial))
end
