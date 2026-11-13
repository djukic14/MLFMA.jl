module EquidistantInterpolationTests

# The MLFMA interpolator/anterpolator PROTOCOL over the equidistant backends.
#
# `backends.jl` and `transposed.jl` already pin the numerics of `resample` and `resampleadjoint`. This
# file pins the ADAPTER: that the functor moves the right component through the right kernel with the
# right parity, that the reverse directions mean what MLFMA says they mean, and that the two diagonal
# hooks (`postscale`, `prescale`) behave as the other families' do.
#
# The central testset is deliberately the SAME statement as `test/LevelTransfers/anterpolationcontract.jl`
# makes for the Gauss-Legendre families -- `anterpolator == adjoint(interpolator)`,
# `transposedinterpolator == transpose(interpolator)` -- run through the very same matrix builders. That
# is the point: a new angular representation either satisfies the package's operator algebra or it is not
# a drop-in, and reusing the harness is what makes the claim comparable rather than merely similar.

using Test
using MLFMA
using LinearAlgebra
using MLFMA.Equidistant:
    EquidistantSampling,
    EquidistantInterpolator,
    EquidistantAnterpolator,
    componentparities,
    SPHERICAL,
    ANTISPHERICAL,
    Nθ,
    Nφ,
    resample,
    resampleadjoint,
    anterpolate,
    resamplematrix,
    exactquadratureweights,
    FFTBackend,
    LagrangeBackend

const T = ComplexF64
const COARSE = EquidistantSampling{Float64}(6, 12)
const FINE = EquidistantSampling{Float64}(12, 24)

# Both backends, with TWO separate realness facts each, because for the FFT family they differ.
#
#   `declaresreal` -- what `isrealinterpolation` reports, which is what decides whether MLFMA wraps the
#                     reverse direction in a conjugating `TransposedInterpolator`.
#   `exactlyreal`  -- whether the assembled matrix has a bitwise-zero imaginary part.
#
# BOTH BACKENDS COMPUTE A REAL OPERATOR, AND BOTH NOW DECLARE IT. Resampling maps samples of a function
# to samples of the same function; its weights are real, and the Lagrange matrix is exactly real for
# that reason. The FFT one reaches the same operator through complex transforms, so its imaginary part
# is FFTW round-off rather than zero -- real in exact arithmetic, not in floating point. Measured over
# ten shape pairs under both parities (up, down, equal, odd and even Nθ, so the Nyquist bin is
# exercised): max|imag| / max|real| <= 1.8e-16, and transpose(R) vs adjoint(R) 1.1e-16 … 3.3e-16.
#
# The FFT family declared `false` until #109. That was CONSERVATIVE rather than correct once the θ
# Nyquist bug was fixed: it made MLFMA wrap the reverse direction in a conjugating
# `TransposedInterpolator`, which costs three passes to change an answer by 1e-16.
#
# It did NOT used to be. Before the θ Nyquist fix the FFT matrix was over 1% imaginary, and this file
# asserted that as a property. It was a bug artifact: the mis-phased half of the ±N/2 split was the
# only thing making the operator complex. An interpolation operator with a genuinely complex matrix
# should have been read as a symptom, and was not.
const FAMILIES = (
    (name="FFT", backend=FFTBackend(), declaresreal=true, exactlyreal=false),
    (name="Lagrange", backend=LagrangeBackend(6), declaresreal=true, exactlyreal=true),
)

filled(sampling) = reshape(
    T[cis(0.7k) * (1 + 0.3k) for k in 1:(Nφ(sampling) * Nθ(sampling))],
    Nφ(sampling),
    Nθ(sampling),
)

@testset "trait to parity mapping" begin
    # The whole translation between MLFMA's trait vocabulary and this submodule's parity argument.
    # Tabulated rather than derived, because each entry restates what the trait's own docstring says
    # and a silent disagreement between the two is exactly the failure this table exists to catch.
    @test componentparities(MLFMA.VirtualExtension(), MLFMA.ScalarFarField) == (SPHERICAL,)
    @test componentparities(MLFMA.VirtualExtensionFlipθSign(), MLFMA.ScalarFarField) ==
        (ANTISPHERICAL,)
    @test componentparities(
        MLFMA.SphericalθφInterpolation(), MLFMA.TangentialSphericalFarField
    ) == (ANTISPHERICAL, ANTISPHERICAL)
    @test componentparities(
        MLFMA.SphericalInterpolation(), MLFMA.SphericalVectorFarField
    ) == (SPHERICAL, ANTISPHERICAL, ANTISPHERICAL)
    @test componentparities(
        MLFMA.DoubleSphericalInterpolation(), MLFMA.PairedSphericalVectorFarField
    ) == (SPHERICAL, ANTISPHERICAL, ANTISPHERICAL, SPHERICAL, ANTISPHERICAL, ANTISPHERICAL)

    # `DefaultInterpolation` must resolve through the far-field TYPE, so the same trait gives different
    # answers for different representations. A scalar field is spherical; BOTH tangential components
    # flip. That second fact is the one an earlier version of this submodule had wrong, and it is
    # asserted here from MLFMA's trait table rather than from the geometry -- `tangential.jl` derives it
    # the other way, and the two agreeing is the evidence.
    @test componentparities(MLFMA.DefaultInterpolation(), MLFMA.ScalarFarField) ==
        (SPHERICAL,)
    @test componentparities(
        MLFMA.DefaultInterpolation(), MLFMA.TangentialSphericalFarField
    ) == (ANTISPHERICAL, ANTISPHERICAL)
    @test length(
        componentparities(MLFMA.DefaultInterpolation(), MLFMA.SphericalVectorFarField)
    ) == MLFMA.fielddim(MLFMA.SphericalVectorFarField)
end

@testset "declared stencil size and realness" begin
    fft = EquidistantInterpolator{T}(FFTBackend())
    # 0 means GLOBAL, not "no stencil": every source sample contributes to every target.
    @test MLFMA.stencilsize(fft) == 0
    # `true` since #109: the operator is real to 1.8e-16 once the θ Nyquist bug is fixed. It declared
    # `false` before that, which was conservative rather than correct -- see this file's header.
    @test MLFMA.isrealinterpolation(fft) == true

    lag = EquidistantInterpolator{T}(LagrangeBackend(6))
    @test MLFMA.stencilsize(lag) == 6
    @test MLFMA.isrealinterpolation(lag) == true

    # An anisotropic stencil reports the larger of the two orders; the backend keeps both.
    @test MLFMA.stencilsize(EquidistantInterpolator{T}(LagrangeBackend(4, 8))) == 8

    # The realness declaration follows the interpolator through both reverse wrappers, since that is
    # what `transposedinterpolator` branches on.
    @test MLFMA.isrealinterpolation(MLFMA.anterpolator(fft)) == true
    @test MLFMA.isrealinterpolation(MLFMA.anterpolator(lag)) == true

    # THE BEHAVIOUR the trait exists to produce, asserted directly rather than inferred from the flag:
    # a real operator means transpose and adjoint are one matrix, so `transposedinterpolator` must hand
    # back the anterpolator ITSELF rather than a conjugating wrapper. Checking the flag alone would
    # still pass if the branch it feeds were rewired.
    for it in (fft, lag)
        ant = MLFMA.anterpolator(it)
        @test MLFMA.transposedinterpolator(ant) === ant
    end
end

@testset "adjointness contract" begin
    for family in FAMILIES
        @testset "$(family.name)" begin
            interpolator = EquidistantInterpolator{T}(family.backend)
            anterpolator = MLFMA.anterpolator(interpolator)
            transposed = MLFMA.transposedinterpolator(anterpolator)

            inter = MLFMA._getinterpolationmatrix(interpolator, COARSE, FINE, T)
            anter = MLFMA._getanterpolationmatrix(anterpolator, FINE, COARSE, T)
            transp = MLFMA._getanterpolationmatrix(transposed, FINE, COARSE, T)

            @test size(inter) == (length(FINE), length(COARSE))
            @test size(anter) == (length(COARSE), length(FINE))
            @test norm(inter) > 0   # non-vacuity: an all-zero matrix satisfies everything below

            # THE CONTRACT, stated exactly as the Gauss-Legendre families state it.
            @test anter ≈ adjoint(inter)
            @test transp ≈ transpose(inter)

            @test MLFMA.isrealinterpolation(interpolator) == family.declaresreal

            # BOTH operators are real, so for both of them transpose and adjoint coincide. Asserted on
            # the MEASURED matrix, not inferred from the declaration -- the declaration is what this is
            # here to check, and for the FFT family the two currently disagree on purpose.
            if family.exactlyreal
                @test maximum(abs, imag(inter)) == 0
            else
                @test maximum(abs, imag(inter)) / maximum(abs, real(inter)) < 1e-14
            end
            @test transpose(inter) ≈ adjoint(inter)
            @test anter ≈ transpose(inter)

            if family.declaresreal
                # Real operator, so no wrapper is built: the two reverse directions ARE one operator
                # and wrapping would buy three conjugation passes for nothing.
                @test transposed === anterpolator
            else
                # The FFT family still declares itself complex, so it still gets the wrapper. That is
                # now a CONSERVATIVE declaration rather than a true one: conjugating an operator whose
                # imaginary part is 5.6e-17 is three wasted passes, not a wrong answer. Flipping the
                # declaration is a separate change with its own measurement -- see the follow-up task --
                # because `isrealinterpolation` gates more than this wrapper.
                @test transposed isa MLFMA.TransposedInterpolator
            end
        end
    end
end

@testset "the protocol anterpolator is the BARE adjoint, not the weighted one" begin
    # A REGRESSION GUARD ON A DELIBERATE CHOICE, not a numerical check. `Equidistant.anterpolate` is
    # `W_c⁻¹ Rᴴ W_f` and is the right operator for a standalone two-level kernel; MLFMA's protocol wants
    # `Rᴴ`. Both are correct for their own caller and they are NOT interchangeable, so this pins that
    # the protocol side uses the bare one -- and that the two genuinely differ, or the distinction
    # would be untestable and someone would eventually "simplify" them together.
    interpolator = EquidistantInterpolator{T}(FFTBackend())
    anterpolator = MLFMA.anterpolator(interpolator)

    G = filled(FINE)
    up = MLFMA.ScalarFarField(T, FINE)
    up[1] .= G
    down = MLFMA.ScalarFarField(T, COARSE)
    anterpolator(down, up, FINE, COARSE)

    bare = resampleadjoint(G, COARSE, FINE, SPHERICAL, FFTBackend())
    weighted = anterpolate(G, FINE, COARSE, SPHERICAL, FFTBackend())

    @test down[1] ≈ bare
    # The two operators differ by the weight ratio. On these grids that is a large, unmistakable
    # factor; on finer grids it tends to 1, which is exactly why using the wrong one survives casual
    # testing and has to be pinned here instead.
    @test norm(bare - weighted) / norm(bare) > 0.1
end

@testset "the functor routes each component through the right parity" begin
    for family in FAMILIES
        @testset "$(family.name)" begin
            interpolator = EquidistantInterpolator{T}(family.backend)

            # Scalar: one component, SPHERICAL.
            scalarin = MLFMA.ScalarFarField(T, COARSE)
            scalarin[1] .= filled(COARSE)
            scalarout = MLFMA.ScalarFarField(T, FINE)
            interpolator(scalarout, scalarin, COARSE, FINE)
            @test scalarout[1] ≈
                resample(scalarin[1], COARSE, FINE, SPHERICAL, family.backend)

            # Tangential: two components, BOTH anti-spherical, and the two components must not be
            # mixed -- hence distinct data in each.
            vecin = MLFMA.TangentialSphericalFarField(T, COARSE)
            vecin[1] .= filled(COARSE)
            vecin[2] .= 2 .* filled(COARSE) .+ 1
            vecout = MLFMA.TangentialSphericalFarField(T, FINE)
            interpolator(vecout, vecin, COARSE, FINE)
            for component in 1:2
                @test vecout[component] ≈ resample(
                    vecin[component], COARSE, FINE, ANTISPHERICAL, family.backend
                )
            end

            # An explicitly overridden trait must actually CHANGE the operator, or the parity argument
            # is decorative. Overridden on the scalar field, where an arity-1 trait is legitimate:
            # `VirtualExtensionFlipθSign` is physically wrong for a scalar and that is the point --
            # the caller asked for it and must get it, sign flip and all.
            forced = MLFMA.ScalarFarField(T, FINE)
            interpolator(
                forced,
                scalarin,
                COARSE,
                FINE;
                interpolationtrait=MLFMA.VirtualExtensionFlipθSign(),
            )
            @test forced[1] ≈
                resample(scalarin[1], COARSE, FINE, ANTISPHERICAL, family.backend)
            @test !(forced[1] ≈ scalarout[1])

            # A trait of the WRONG ARITY is a caller error and must say so. Without the guard it
            # reaches the component loop and dies as `BoundsError` on a tuple, naming neither the
            # trait nor the far field -- which is how this assertion came to exist.
            @test_throws ErrorException interpolator(
                vecout, vecin, COARSE, FINE; interpolationtrait=MLFMA.VirtualExtension()
            )
        end
    end
end

@testset "the allocating functor preserves the representation" begin
    interpolator = EquidistantInterpolator{T}(FFTBackend())
    vecin = MLFMA.TangentialSphericalFarField(T, COARSE)
    vecin[1] .= filled(COARSE)
    vecin[2] .= filled(COARSE)

    out = interpolator(vecin, COARSE, FINE)
    # Interpolation resamples the direction grid; it does not reinterpret what the components mean.
    @test MLFMA.basefarfieldtype(out) === MLFMA.TangentialSphericalFarField
    @test size(out[1]) == (Nφ(FINE), Nθ(FINE))

    inplace = MLFMA.TangentialSphericalFarField(T, FINE)
    interpolator(inplace, vecin, COARSE, FINE)
    @test out[1] == inplace[1]
    @test out[2] == inplace[2]
end

@testset "postscale is fused interpolate-then-scale" begin
    for family in FAMILIES
        interpolator = EquidistantInterpolator{T}(family.backend)
        input = MLFMA.ScalarFarField(T, COARSE)
        input[1] .= filled(COARSE)

        diagonal = reshape(
            T[cis(0.11k) for k in 1:(Nφ(FINE) * Nθ(FINE))], Nφ(FINE), Nθ(FINE)
        )

        fused = MLFMA.ScalarFarField(T, FINE)
        interpolator(fused, input, COARSE, FINE; postscale=diagonal)

        plain = MLFMA.ScalarFarField(T, FINE)
        interpolator(plain, input, COARSE, FINE)

        # `==`, not `≈`: the fused write is `resampled[i] * postscale[i]`, the same single
        # multiplication in the same order as the separate pass, so it is bitwise identical rather than
        # merely equal to round-off. Tightening this is what stops a future "optimization" from
        # precombining factors and moving the last bit.
        @test fused[1] == plain[1] .* diagonal
    end
end

@testset "prescale stages a private copy and never mutates the input" begin
    for family in FAMILIES
        interpolator = EquidistantInterpolator{T}(family.backend)
        anterpolator = MLFMA.anterpolator(interpolator)

        input = MLFMA.ScalarFarField(T, FINE)
        input[1] .= filled(FINE)
        untouched = copy(input[1])

        diagonal = reshape(
            T[cis(0.23k) for k in 1:(Nφ(FINE) * Nθ(FINE))], Nφ(FINE), Nθ(FINE)
        )

        scaled = MLFMA.ScalarFarField(T, COARSE)
        anterpolator(scaled, input, FINE, COARSE; prescale=diagonal)

        # THE INPUT MUST COME BACK BITWISE. In disaggregation one parent moment feeds several children
        # that different chunks may process concurrently, so an apply that scaled in place would be a
        # data race -- the failure mode that showed up as run-to-run varying error in the threaded
        # suite and in nothing else.
        @test input[1] == untouched

        premultiplied = MLFMA.ScalarFarField(T, FINE)
        premultiplied[1] .= untouched .* diagonal
        reference = MLFMA.ScalarFarField(T, COARSE)
        anterpolator(reference, premultiplied, FINE, COARSE)
        @test scaled[1] ≈ reference[1]

        # `stagetransform` applies to the PRODUCT, not to the input -- that is what makes
        # `Iᵀ(D·x) = conj(Iᴴ(conj(D·x)))` work with the diagonal forwarded unconjugated.
        transformed = MLFMA.ScalarFarField(T, COARSE)
        anterpolator(
            transformed, input, FINE, COARSE; prescale=diagonal, stagetransform=conj
        )
        conjugated = MLFMA.ScalarFarField(T, FINE)
        conjugated[1] .= conj.(untouched .* diagonal)
        conjreference = MLFMA.ScalarFarField(T, COARSE)
        anterpolator(conjreference, conjugated, FINE, COARSE)
        @test transformed[1] ≈ conjreference[1]
        @test input[1] == untouched
    end
end

@testset "the transposed wrapper composes with prescale" begin
    # The FFT family is the one that needs the wrapper at all, and the `prescale` branch is the one the
    # traversal takes -- so this is the composition that actually runs in anger.
    interpolator = EquidistantInterpolator{T}(FFTBackend())
    anterpolator = MLFMA.anterpolator(interpolator)
    transposed = MLFMA.transposedinterpolator(anterpolator)

    input = MLFMA.ScalarFarField(T, FINE)
    input[1] .= filled(FINE)
    untouched = copy(input[1])
    diagonal = reshape(T[cis(0.31k) for k in 1:(Nφ(FINE) * Nθ(FINE))], Nφ(FINE), Nθ(FINE))

    got = MLFMA.ScalarFarField(T, COARSE)
    transposed(got, input, FINE, COARSE; prescale=diagonal)
    @test input[1] == untouched

    # NOT `_getinterpolationmatrix`: that probes with a `TangentialSphericalFarField` and so builds the
    # ANTI-spherical operator, while the scalar field applied above is spherical. `resamplematrix` takes
    # the parity explicitly, which is the whole reason it is the right reference here.
    want = transpose(resamplematrix(COARSE, FINE, SPHERICAL, FFTBackend()))
    @test vec(got[1]) ≈ want * vec(untouched .* diagonal)
end

end # module
