using MLFMA
using H2Factory
using LinearAlgebra
using StaticArrays
using Test

const IDA = MLFMA.InterpolationDiagnostics

# ANTERPOLATION IS ADJOINT INTERPOLATION; TRANSPOSED INTERPOLATION IS A DIFFERENT OPERATOR.
#
# This file pins that distinction, which the standing TODO in
# `src/interpolation/anterpolator/sphericalanterpolation.jl` named and did not resolve. The two
# interpolator families disagree about it, and until `transposedinterpolator` existed nothing in the
# suite could tell:
#
#   * `SphericalLagrangeAnterpolator` -- the interpolation matrix is exactly REAL (barycentric weights),
#     so transpose and adjoint coincide and one operator serves both directions. No test built on this
#     family can distinguish the conventions, which is why the question stayed open so long.
#   * `LocalθGlobalφAnterpolator` -- the φ step is an FFT, so the matrix is complex and the anterpolator
#     is the CONJUGATE transpose, differing from the plain transpose by ~35% -- entirely in the Nyquist
#     φ order.
#
# WHY THE DISTINCTION IS HONOURED RATHER THAN AVERAGED OVER. H2Factory's direction step is
# `_directedtransfer(T, ::TransposeTranslation) = transpose(T)`, i.e. Julia's `transpose`, and its
# contract (`H2Factory/src/maps/leveltranslator.jl:169-171`) keeps transpose and adjoint distinct
# precisely so "a future complex level transfer must tell them apart without any other code changing".
# MLFMA's `InterpolationPhaseTransfer` is that transfer. So `transfer.jl`'s `transpose(M)` contract is
# authoritative, and the transpose direction now applies `transposedinterpolator(anterpolator)` while the
# adjoint direction applies the anterpolator.
#
# SCOPE, so this reads as a contract fix and not an incident: no shipped product changed. Production
# defaults to the Lagrange family (real operator), and `MLFMA.NθNφ` oversamples φ ~2.5x relative to the
# band it supports, so ε-driven sampling never places content at the Nyquist order where the operators
# differ -- an end-to-end Galerkin forward product was identical under either family both before and
# after. What the fix buys is that MLFMA's map algebra now agrees with Julia's, and that reducing φ
# oversampling (a tempting storage optimization) cannot silently make the reverse direction wrong.
@testset "Anterpolation adjointness contract" begin
    T = ComplexF64
    # Nφ must GROW from coarse to fine: the local/global interpolator zero-pads the φ spectrum, and
    # `_interpolateφFFT!` computes `Npad = floor(interN/2 - N/2)`, which goes negative and indexes out
    # of bounds when asked to shrink φ. `localglobalanterpolation.jl` skips those cases for the same
    # reason.
    coarse = MLFMA.SphericalSampling{Float64}(6, 12)
    fine = MLFMA.SphericalSampling{Float64}(8, 16)

    # EACH FAMILY BRINGS ITS OWN GRIDS AND ORDERS, because the contract is a statement about operators
    # rather than about one angular representation. The circular family joined here rather than getting
    # its own copy for exactly that reason: `anter ≈ adjoint(inter)` is the same sentence for all three,
    # and a second copy of it elsewhere is a second place for it to be weakened.
    #
    # `exactlyreal` is separate from `realmatrix` because they are different claims. `realmatrix` is
    # what `transposedinterpolator` consumes -- "transpose and adjoint coincide, so skip the wrapper" --
    # and holds in exact arithmetic; `exactlyreal` says the MEASURED imaginary part is bitwise zero. The
    # barycentric family is both. The circular family is real to round-off only: its FFT round trip
    # leaves ~1e-16 in the imaginary part, so demanding `== 0` would fail it for a reason that says
    # nothing about the contract, and loosening the Lagrange assertion to match would give up a real
    # property. Same distinction #109 drew for the equidistant FFT backend.
    families = (
        (
            name="SphericalLagrange",
            coarse=coarse,
            fine=fine,
            orders=(4, 6),
            interpolator=p -> MLFMA.SphericalLagrangeInterpolator{p}(T, Float64),
            anterpolator=p -> MLFMA.SphericalLagrangeAnterpolator{p}(T, Float64),
            realmatrix=true,
            exactlyreal=true,
        ),
        (
            name="LocalθGlobalφ",
            coarse=coarse,
            fine=fine,
            orders=(4, 6),
            interpolator=p -> MLFMA.LocalθGlobalφInterpolator{p}(T, Float64),
            anterpolator=p -> MLFMA.LocalθGlobalφAnterpolator{p}(T, Float64),
            realmatrix=false,
            exactlyreal=false,
        ),
        (
            name="Circular",
            coarse=MLFMA.CircularSampling{Float64}(12),
            fine=MLFMA.CircularSampling{Float64}(20),
            # One "order", and it is not one: the step is a global spectral resample with no stencil,
            # so `0` is the family's spelling for "every source sample contributes".
            orders=(0,),
            interpolator=p -> MLFMA.CircularInterpolator{T}(),
            anterpolator=p -> MLFMA.anterpolator(MLFMA.CircularInterpolator{T}()),
            realmatrix=true,
            exactlyreal=false,
        ),
    )

    # The testsets below the matrix relation are SPHERICAL-GRID SPECIFIC and stay that way: they fill
    # their fields with `IDA.fillexact!`, which evaluates spherical harmonics, and one of them reads
    # `MLFMA.Nφ` directly. Neither has meaning on a circle. The circular family's equivalent bilinear /
    # Hermitian pairing lives in `circularinterpolation.jl`, on a circular band, rather than being
    # forced through machinery built for a sphere.
    sphericalfamilies = families[1:2]

    # `familycoarse`/`familyfine`, NOT `coarse`/`fine`. A `@testset for` body is not its own scope for
    # names the enclosing testset already binds, so assigning `coarse = family.coarse` here would
    # REBIND the outer grids and hand every later testset in this file the last family's -- which is
    # exactly what it did: nine errors, all of them `Nφ(::CircularSampling)` and `fillexact!` on a
    # circular grid, none of them in this testset.
    @testset "matrix relation: $(family.name)" for family in families, p in family.orders
        familycoarse = family.coarse
        familyfine = family.fine
        interpolator = family.interpolator(p)
        anterpolator = family.anterpolator(p)

        # These helpers probe with a `TangentialSphericalFarField`, so both matrices are built under
        # that representation's trait. The pairing testset below uses `ScalarFarField`, hence a
        # different trait -- consistent WITHIN each testset, which is what the relation needs.
        transposed = MLFMA.transposedinterpolator(anterpolator)

        inter = MLFMA._getinterpolationmatrix(interpolator, familycoarse, familyfine, T)
        anter = MLFMA._getanterpolationmatrix(anterpolator, familyfine, familycoarse, T)
        transp = MLFMA._getanterpolationmatrix(transposed, familyfine, familycoarse, T)

        @test size(inter) == (length(familyfine), length(familycoarse))
        @test size(anter) == (length(familycoarse), length(familyfine))
        @test size(transp) == (length(familycoarse), length(familyfine))
        @test norm(inter) > 0   # non-vacuity: an all-zero matrix satisfies every relation below

        # THE CONTRACT, and it is the same statement for every family: the anterpolator is the adjoint,
        # the transposed interpolator is the transpose. What differs between families is only whether
        # those two are the same matrix.
        @test anter ≈ adjoint(inter)
        @test transp ≈ transpose(inter)

        # `isrealinterpolation` decides which branch `transposedinterpolator` takes, so it is checked
        # against the MEASURED operator rather than trusted as a declaration.
        @test MLFMA.isrealinterpolation(interpolator) == family.realmatrix
        @test MLFMA.isrealinterpolation(anterpolator) == family.realmatrix

        if family.realmatrix
            # The reason this family cannot distinguish the conventions, stated as a test rather than
            # left as a remark: if the weights ever became complex, the assertions below would start
            # disagreeing and this one says why.
            #
            # Split on `exactlyreal` -- see the note on the family table above. Bitwise zero for the
            # barycentric weights; FFTW round-off for the circular family, measured at 0 … 6.4e-17
            # relative across grid pairs and asserted three orders of magnitude looser so what is
            # pinned is "real" rather than one machine's rounding.
            if family.exactlyreal
                @test maximum(abs, imag(inter)) == 0
            else
                @test maximum(abs, imag(inter)) / maximum(abs, real(inter)) < 1e-13
            end
            @test transpose(inter) ≈ adjoint(inter)
            @test anter ≈ transpose(inter)
            # Real operator, so the two reverse directions ARE one operator and no wrapper is built --
            # wrapping would cost three conjugation passes per apply for nothing.
            @test transposed === anterpolator
        else
            # Non-vacuity: the matrix must be genuinely complex, or "adjoint but not transpose" is an
            # empty distinction and the assertions below would pass trivially.
            @test maximum(abs, imag(inter)) / maximum(abs, real(inter)) > 0.01
            @test !(transpose(inter) ≈ adjoint(inter))

            # The anterpolator alone does NOT satisfy the transpose relation -- measured ~0.35, asserted
            # with a wide margin so what is pinned is the finding and not one machine's round-off. This
            # is what makes the separate operator necessary rather than decorative.
            @test !(anter ≈ transpose(inter))
            @test norm(anter - transpose(inter)) / norm(inter) > 0.1

            @test transposed isa MLFMA.TransposedInterpolator
            @test transposed !== anterpolator
            # The wrapper holds an anterpolator, not an interpolator, so the inherited accessor has to
            # forward. Compared against what the WRAPPED anterpolator reports rather than against the
            # interpolator built above: those two are structurally equal but distinct objects, and
            # Julia's default struct equality is identity, so `=== interpolator` would fail for a
            # reason that says nothing about forwarding.
            @test MLFMA.interpolator(transposed) === MLFMA.interpolator(anterpolator)
            @test MLFMA.stencilsize(transposed) == p
        end
    end

    # THE WRAPPER RESTORES ITS INPUT, and how faithfully is a property of the operator it wraps rather
    # than of the conjugations. Pinned together with the underlying behaviour, because a bare `≈` here
    # would hide which of the two is responsible if it ever drifted:
    #
    #   * `SphericalLagrangeAnterpolator` leaves its input BITWISE unchanged.
    #   * `LocalθGlobalφAnterpolator` applies an in-place FFT round trip and perturbs its input by
    #     ~4e-16. That is pre-existing and undocumented; `transfer.jl` only ever checked `≈`, and the
    #     traversal happens to hand it scratch rather than a live moment, so it has never mattered.
    #
    # The wrapper's own conjugations are exact, so it inherits exactly that residual and adds nothing.
    @testset "input restoration, and whose round-off it is" begin
        y = MLFMA.ScalarFarField(T, fine, Val(:zero))
        IDA.fillexact!(y, IDA.RandomBandlimitedCase(; L=4, seed=0x3333), fine)
        reference = copy(y[1])

        @testset "the wrapped anterpolators themselves" begin
            for (anterpolator, bitwise) in (
                (MLFMA.SphericalLagrangeAnterpolator{6}(T, Float64), true),
                (MLFMA.LocalθGlobalφAnterpolator{6}(T, Float64), false),
            )
                field = MLFMA.ScalarFarField(T, fine, Val(:zero))
                field[1] .= reference
                out = MLFMA.ScalarFarField(T, coarse, Val(:zero))
                anterpolator(out, field, fine, coarse)

                @test (field[1] == reference) == bitwise
                @test maximum(abs, field[1] .- reference) < 1e-14
            end
        end

        @testset "the transposed wrapper adds no further drift" begin
            anterpolator = MLFMA.LocalθGlobalφAnterpolator{6}(T, Float64)
            transposed = MLFMA.transposedinterpolator(anterpolator)

            wrapped = MLFMA.ScalarFarField(T, fine, Val(:zero))
            wrapped[1] .= reference
            bare = MLFMA.ScalarFarField(T, fine, Val(:zero))
            bare[1] .= reference

            outwrapped = MLFMA.ScalarFarField(T, coarse, Val(:zero))
            outbare = MLFMA.ScalarFarField(T, coarse, Val(:zero))
            transposed(outwrapped, wrapped, fine, coarse)
            anterpolator(outbare, bare, fine, coarse)

            # The same residual to within rounding, not to the bit: the wrapped anterpolator runs on
            # CONJUGATED data, and an FFT round trip over different bit patterns rounds slightly
            # differently (measured 4.58e-16 wrapped against 4.44e-16 bare). What is pinned is that the
            # conjugations add no drift of their own -- same order of magnitude, small constant factor --
            # rather than an equality that would be asserting the FFT's rounding is conjugation-symmetric.
            wrappeddrift = maximum(abs, wrapped[1] .- reference)
            baredrift = maximum(abs, bare[1] .- reference)
            @test wrappeddrift < 1e-14
            @test wrappeddrift < 3 * baredrift

            @test maximum(abs, outwrapped[1]) > 1e-6   # non-vacuity
            @test !any(isnan, outwrapped[1])
        end

        # THE RESTORE MUST SURVIVE A THROW, or a caught error leaves the caller holding a silently
        # conjugated moment -- corruption that would surface far from its cause. The reachable trigger
        # is concrete rather than contrived: the local/global family cannot resample φ in the direction
        # that grows it, and raises `BoundsError` from its negative FFT pad width.
        @testset "the input is restored even when the wrapped apply throws" begin
            anterpolator = MLFMA.LocalθGlobalφAnterpolator{6}(T, Float64)
            transposed = MLFMA.transposedinterpolator(anterpolator)

            x = MLFMA.ScalarFarField(T, coarse, Val(:zero))
            IDA.fillexact!(x, IDA.RandomBandlimitedCase(; L=3, seed=0x77), coarse)
            before = copy(x[1])
            out = MLFMA.ScalarFarField(T, fine, Val(:zero))

            # Deliberately the unsupported direction: coarse -> fine grows Nφ.
            @test_throws BoundsError transposed(out, x, coarse, fine)
            # Bitwise, because the wrapped call throws before mutating anything, so the two
            # conjugations cancel exactly. Also checked against the conjugate, so a restore that
            # silently did nothing at all could not pass by coincidence.
            @test x[1] == before
            @test !(x[1] == conj.(before))
        end
    end

    # THE SAME STATEMENT ON ACTUAL FIELD CONTENT, at m != 0 so the φ machinery is exercised.
    #
    # For an interpolation operator I and anterpolator A:
    #   A == transpose(I)  <=>  sum(I*x .* y)  == sum(x .* A*y)     (bilinear pairing)
    #   A == adjoint(I)    <=>  dot(I*x, y)    == dot(x, A*y)       (Hermitian pairing)
    # A unit-vector matrix comparison already covers every mode; this is the readable form, and it is
    # the form that would catch a trait or seam convention that only shows up on a field with real φ
    # structure rather than on a single basis direction.
    #
    # WHY RANDOM BANDS AND NOT TWO SINGLE HARMONICS, which is what this testset was first written with
    # and which was VACUOUS. Both pairings factor a sum over an equispaced φ grid, so a single-harmonic
    # pair contributes `exp(i(m_x + m_y)φ)` to the bilinear form and `exp(i(m_y - m_x)φ)` to the
    # Hermitian one, and each sums to EXACTLY zero unless its exponent vanishes mod Nφ. Both pairings
    # are therefore ~1e-15 for any m_x != m_y, and requiring both to be nonzero at m != 0 forces
    # `2 m_x ≡ 0`, i.e. m_x = 0 -- precisely the case that does not exercise φ. A band-limited random
    # field contains every order, so both pairings are generically nonzero and φ is genuinely involved.
    @testset "pairing on exact harmonics: $(family.name)" for family in sphericalfamilies
        p = 6
        interpolator = family.interpolator(p)
        anterpolator = family.anterpolator(p)

        # Different seeds, so the two sides are independent fields and neither pairing can pass by a
        # symmetry between them.
        x = MLFMA.ScalarFarField(T, coarse, Val(:zero))
        y = MLFMA.ScalarFarField(T, fine, Val(:zero))
        IDA.fillexact!(x, IDA.RandomBandlimitedCase(; L=3, seed=0x1111), coarse)
        IDA.fillexact!(y, IDA.RandomBandlimitedCase(; L=3, seed=0x2222), fine)

        Ix = interpolator(x, coarse, fine)
        Ay = anterpolator(y, fine, coarse)

        bilinearleft = sum(Ix[1] .* y[1])
        bilinearright = sum(x[1] .* Ay[1])
        hermitianleft = dot(Ix[1], y[1])
        hermitianright = dot(x[1], Ay[1])

        # Non-vacuity: the pairings must be nonzero, and the bilinear and Hermitian forms must differ
        # from each other -- otherwise this testset cannot tell the conventions apart either.
        @test abs(bilinearleft) > 1e-6
        @test abs(hermitianleft) > 1e-6
        @test !(bilinearleft ≈ hermitianleft)

        # The HERMITIAN pairing holds for both families, at every bandwidth: anterpolation is exactly
        # adjoint interpolation. This is the relation that is unconditionally true.
        @test hermitianleft ≈ hermitianright

        # The BILINEAR pairing holds here too -- but only because an L = 3 band has no content at the
        # Nyquist φ order. See the testset below, which is where that stops being true.
        @test bilinearleft ≈ bilinearright
    end

    # WHERE THE TRANSPOSE RELATION BREAKS, and it is one mode, not a general discrepancy.
    #
    # The local/global anterpolator is exactly `adjoint(I)`. It also equals `transpose(I)` on any field
    # with no content at the NYQUIST φ order, |m| = Nφ/2 of the coarse grid -- and differs from it
    # entirely on that mode. Measured on Nφ = 12 (Nyquist order 6), relative bilinear mismatch:
    #
    #     L = 2, 3, 5 -> ~1e-16      L = 6 -> 0.090      L = 8 -> 0.088      pure Y_6^6 -> 1.0
    #
    # while the Hermitian mismatch stays at round-off throughout. That is the signature of the even-N
    # zero-padding FFT: the single Nyquist bin has to be split in half between the two padded ends to
    # keep the interpolant real, and it is not, so the operator picks up an imaginary part on exactly
    # that mode.
    #
    # WHY THIS MATTERS FOR THE BUILDER API, and why it is pinned rather than fixed here: `anterpolator`
    # is derived from the interpolator, so choosing a family chooses this behaviour. Production is
    # unaffected today -- the default family is Lagrange (real matrix, both relations hold), and
    # ε-driven sampling keeps content below the grid's Nyquist, which is why an end-to-end Galerkin
    # forward product is identical under either family. But `transfer.jl` pins the transfer's transpose
    # direction to `transpose(M)`, and a local/global pair cannot satisfy that for a field carrying
    # Nyquist φ content. Which convention is authoritative is the open question named by the standing
    # TODO in `sphericalanterpolation.jl`.
    @testset "transpose relation holds only below the Nyquist φ order" begin
        p = 6
        nyquistorder = MLFMA.Nφ(coarse) ÷ 2

        function mismatches(interpolator, anterpolator, case)
            x = MLFMA.ScalarFarField(T, coarse, Val(:zero))
            y = MLFMA.ScalarFarField(T, fine, Val(:zero))
            IDA.fillexact!(x, case, coarse)
            IDA.fillexact!(y, case, fine)

            Ix = interpolator(x, coarse, fine)
            Ay = anterpolator(y, fine, coarse)

            bilinear = (sum(Ix[1] .* y[1]), sum(x[1] .* Ay[1]))
            hermitian = (dot(Ix[1], y[1]), dot(x[1], Ay[1]))
            return (
                bilinear=abs(bilinear[1] - bilinear[2]) / abs(bilinear[1]),
                hermitian=abs(hermitian[1] - hermitian[2]) / abs(hermitian[1]),
                scale=abs(bilinear[1]),
            )
        end

        localglobal = (
            MLFMA.LocalθGlobalφInterpolator{p}(T, Float64),
            MLFMA.LocalθGlobalφAnterpolator{p}(T, Float64),
        )
        lagrange = (
            MLFMA.SphericalLagrangeInterpolator{p}(T, Float64),
            MLFMA.SphericalLagrangeAnterpolator{p}(T, Float64),
        )

        @testset "below Nyquist both relations hold" begin
            for L in (2, 3, nyquistorder - 1)
                case = IDA.RandomBandlimitedCase(; L=L, seed=0x1234)
                for pair in (localglobal, lagrange)
                    m = mismatches(pair..., case)
                    @test m.scale > 1e-6          # non-vacuity: a zero pairing proves nothing
                    @test m.bilinear < 1e-10
                    @test m.hermitian < 1e-10
                end
            end
        end

        @testset "at and above Nyquist the ANTERPOLATOR satisfies only the adjoint relation" begin
            for L in (nyquistorder, nyquistorder + 2)
                case = IDA.RandomBandlimitedCase(; L=L, seed=0x1234)

                local_ = mismatches(localglobal..., case)
                @test local_.scale > 1e-6
                @test local_.hermitian < 1e-10   # still exactly the adjoint
                # Asserted as a factor, not a measured value: ~0.09 at both bandwidths.
                @test local_.bilinear > 0.01

                # The Lagrange family keeps both, because its operator is real -- which is what makes
                # this difference a property of the FFT φ step and not of the bandwidth.
                lag = mismatches(lagrange..., case)
                @test lag.bilinear < 1e-10
                @test lag.hermitian < 1e-10
            end
        end

        # AND THIS IS WHAT THE SEPARATE OPERATOR BUYS: the transposed interpolator satisfies the
        # bilinear pairing exactly where the anterpolator does not, i.e. at and above the Nyquist φ
        # order. Without it there is no operator in the package that satisfies H2Factory's transpose
        # contract for a complex level transfer.
        @testset "the TRANSPOSED operator satisfies the bilinear pairing at Nyquist" begin
            transposedpair = (
                first(localglobal), MLFMA.transposedinterpolator(last(localglobal))
            )
            for L in (nyquistorder, nyquistorder + 2)
                case = IDA.RandomBandlimitedCase(; L=L, seed=0x1234)
                m = mismatches(transposedpair..., case)
                @test m.scale > 1e-6
                @test m.bilinear < 1e-10
            end
        end

        @testset "a pure Nyquist-order harmonic breaks the transpose relation completely" begin
            case = IDA.HarmonicCase(; l=nyquistorder, m=nyquistorder)
            local_ = mismatches(localglobal..., case)
            @test local_.scale > 1e-6
            @test local_.bilinear > 0.5

            lag = mismatches(lagrange..., case)
            @test lag.bilinear < 1e-10

            # The worst case for the anterpolator is exact for the transposed operator.
            transposedpair = (
                first(localglobal), MLFMA.transposedinterpolator(last(localglobal))
            )
            @test mismatches(transposedpair..., case).bilinear < 1e-10
        end
    end

    # TRANSFER-LEVEL BEHAVIOUR, which is where the whole question came from: every direction of an
    # `InterpolationPhaseTransfer` against the dense reference `M = Diagonal(phase) * intermatrix`.
    # `transfer.jl` already does this for the Lagrange family on grids that SHRINK Nφ; this covers both
    # families on grids that grow it, which is the direction the local/global family supports.
    @testset "transfer directions against a dense reference: $(family.name)" for family in
                                                                                 sphericalfamilies

        p = 4
        interpolator = family.interpolator(p)
        anterpolator = family.anterpolator(p)

        Δ = SVector(0.19, -0.27, 0.11)
        k̂s, _ = MLFMA.cartesiansamplepoints(fine)
        phase = MLFMA.phasefactor(im * 2π, k̂s, Δ, MLFMA.AbstractRadiationFieldCollection)
        # Non-vacuity: a weakly complex phase makes transpose and adjoint agree and proves nothing.
        @test maximum(abs, imag.(phase)) > 0.1

        transfer = MLFMA.InterpolationPhaseTransfer(
            interpolator,
            anterpolator,
            phase,
            coarse,
            fine,
            MLFMA.SphericalθφInterpolation(),
        )
        M =
            Diagonal(vec(phase)) *
            MLFMA._getinterpolationmatrix(interpolator, coarse, fine, T)
        @test !(transpose(M) ≈ adjoint(M))   # the reference must distinguish the directions

        context = H2Factory.TranslationContext(
            nothing, MLFMA.TangentialSphericalFarField(T, fine)
        )

        u = zeros(T, size(coarse))
        u[2] = 1.0 + 0.5im
        forwardin = MLFMA.TangentialSphericalFarField(SVector(u, copy(u)))
        forwardout = MLFMA.TangentialSphericalFarField(T, fine)
        H2Factory.applydirectedtransfer!(forwardout, forwardin, transfer, false, context)
        @test vec(forwardout[1]) ≈ M * vec(u)

        up = zeros(T, size(fine))
        up[2] = 1.0 + 0.5im
        reversein = MLFMA.TangentialSphericalFarField(SVector(up, copy(up)))

        transposeout = MLFMA.TangentialSphericalFarField(T, coarse)
        H2Factory.applydirectedtransfer!(
            transposeout, reversein, transpose(transfer), false, context
        )
        @test vec(transposeout[1]) ≈ transpose(M) * vec(up)
        @test !(vec(transposeout[1]) ≈ adjoint(M) * vec(up))

        adjointout = MLFMA.TangentialSphericalFarField(T, coarse)
        H2Factory.applydirectedtransfer!(
            adjointout, reversein, adjoint(transfer), false, context
        )
        @test vec(adjointout[1]) ≈ adjoint(M) * vec(up)
        @test !(vec(adjointout[1]) ≈ transpose(M) * vec(up))

        # The reverse directions must not have written through to the traversal's input.
        @test vec(reversein[1]) ≈ vec(up)

        # CONJUGATE: the fourth direction, and the only one that is not a reversal -- it runs
        # child-to-parent like the forward one, so it takes the FORWARD input and lands on the parent
        # grid. This is what an ADJOINT map's aggregation applies; without it an adjoint map
        # aggregated with the plain forward transfer, worth 0.343 far-only relative error on
        # `test/Maps/galerkin.jl`'s fixture.
        conjugateout = MLFMA.TangentialSphericalFarField(T, fine)
        H2Factory.applydirectedtransfer!(
            conjugateout, forwardin, conj(transfer), false, context
        )
        @test vec(conjugateout[1]) ≈ conj(M) * vec(u)
        # Discriminating, or this would pass on the very bug it exists to catch. For the LAGRANGE
        # family the interpolation operator is real, so `conj(M)` differs from `M` through the PHASE
        # alone -- which is why the phase non-vacuity check above carries as much weight here as the
        # operator's own complexity.
        @test !(vec(conjugateout[1]) ≈ M * vec(u))

        # The forward input is the traversal's, and this direction reaches `conj(M)` by conjugating
        # that input IN PLACE. It has to come back unchanged -- to round-off rather than bitwise,
        # since the local/global family perturbs its input on an FFT round trip.
        @test vec(forwardin[1]) ≈ vec(u)

        # An involution, like `transpose`/`adjoint` above.
        @test conj(conj(transfer)) === transfer

        # The wrapper is built only when it buys something: for a real interpolation operator
        # `conj(I) == I`, and `conjugatedinterpolator` hands the interpolator straight back instead of
        # wrapping it in two conjugation passes for nothing.
        if MLFMA.isrealinterpolation(interpolator)
            @test transfer.conjugatedinterpolator === interpolator
        else
            @test transfer.conjugatedinterpolator isa MLFMA.ConjugatedInterpolator
        end
    end

    # SCRATCH-BACKED ANTERPOLATION. Only the local/global family has interim arrays
    # (`interimscratch(::LocalθGlobalφAnterpolator, ...)`); the spherical one returns a scratch tuple
    # too, but its anterpolator takes no `interim` keyword, so there is nothing to compare there.
    #
    # `takeinterim` falls back to a fresh array when the buffer is absent or wrongly shaped, which is
    # what keeps standalone calls working -- so the two paths must agree numerically, and the
    # scratch-backed one must actually be the cheaper of the two, or the buffer is not being used.
    @testset "interim scratch does not change the result" begin
        p = 6
        anterpolator = MLFMA.LocalθGlobalφAnterpolator{p}(T, Float64)
        scratch = MLFMA.interimscratch(anterpolator, coarse, fine)
        @test scratch !== nothing
        @test haskey(scratch, :ftpadmatrix)

        y = MLFMA.ScalarFarField(T, fine, Val(:zero))
        IDA.fillexact!(y, IDA.HarmonicCase(; l=5, m=3), fine)

        fallback = MLFMA.ScalarFarField(T, coarse, Val(:zero))
        backed = MLFMA.ScalarFarField(T, coarse, Val(:zero))

        anterpolator(fallback, y, fine, coarse)
        anterpolator(backed, y, fine, coarse; interim=scratch)

        @test backed[1] ≈ fallback[1]
        @test maximum(abs, fallback[1]) > 1e-6   # non-vacuity

        # A wrongly shaped buffer must fall back rather than write out of bounds -- the property that
        # makes moving the scratch onto the workspace safe for a non-uniform tree.
        mismatched = (
            ftpadmatrix=Matrix{T}(undef, (3, 3)),
            interimmatrix=Matrix{T}(undef, (3, 3)),
            interimθvector=Vector{T}(undef, 3),
        )
        salvaged = MLFMA.ScalarFarField(T, coarse, Val(:zero))
        anterpolator(salvaged, y, fine, coarse; interim=mismatched)
        @test salvaged[1] ≈ fallback[1]
    end
end
