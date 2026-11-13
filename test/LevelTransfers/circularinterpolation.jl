# The circular interpolator family, checked against oracles that cost nothing to be exact about.
#
# TWO OF THE ASSERTIONS HERE ARE THE POINT AND THE REST SUPPORT THEM:
#
#   1. EXACTNESS ON BANDLIMITED DATA. A trigonometric polynomial of degree ≤ L sampled on
#      `NφCircular(L)` points and resampled onto any finer circular grid must come back to machine
#      precision -- not to a tolerance, because spectral resampling of resolved content has no
#      truncation error to trade. Measured 2.2e-16 … 3.7e-15 across L ∈ {3,5,8,12}; asserted at 1e-13,
#      which is loose enough not to pin one machine's round-off and tight enough that any real
#      truncation would fail it by orders of magnitude.
#   2. ADJOINTNESS, which lives in `anterpolationcontract.jl` with the other two families rather than
#      here -- the statement `anter ≈ adjoint(inter)` is the same statement for every family and
#      belongs in one place. This file's job is everything family-specific.
#
# AND ONE THAT EXISTS BECAUSE OF A DELIBERATE DUPLICATION. `_movecircularcoefficients!` is the
# plain-torus twin of `Equidistant.resamplecoefficients!`; core `src/interpolation/` is included before
# `src/equidistant/`, so sharing the twenty lines would mean core depending on a submodule. The
# duplication is accepted and PINNED: the two are asserted bitwise equal over a shape sweep, so a fix
# applied to one and not the other fails here instead of drifting apart silently.

using MLFMA
using BEAST
using FFTW
using H2Factory
using LinearAlgebra
using Random
using StaticArrays
using Test

include("../support/util_test.jl")

@testset "CircularInterpolator" begin
    T = ComplexF64

    @testset "the trait says what it means, and nothing more" begin
        # `PeriodicφInterpolation` exists so the 2-D path does not borrow a spherical rule about a
        # seam a circle does not have. Two things make that more than a rename: it is what the 2-D
        # operators actually report, and it is arity 1.
        @test MLFMA.PeriodicφInterpolation() isa MLFMA.InterpolationExpansionTrait
        @test MLFMA.traitarity(MLFMA.PeriodicφInterpolation()) == 1

        # THE THREE SCALAR OPERATORS, arity 1. The hypersingular one is deliberately NOT in this
        # loop: it radiates a THREE-component far field (`[n̂₁f, n̂₂f, f']`) and so carries
        # `PeriodicφInterpolation{3}()`. This loop originally included it and asserted arity 1 --
        # true only while its far field was unmapped, and the suite caught the moment that stopped
        # being so, which is what the assertion was for.
        for operator in (
            Helmholtz2D.singlelayer(; wavenumber=2.0),
            Helmholtz2D.doublelayer(; wavenumber=2.0),
            Helmholtz2D.doublelayer_transposed(; wavenumber=2.0),
        )
            @test MLFMA.interpolationtrait(operator) === MLFMA.PeriodicφInterpolation()
            @test MLFMA.traitarity(MLFMA.interpolationtrait(operator)) == 1
            # And specifically NOT the superficially-obvious one: both are arity 1 and neither flips a
            # sign, so only the identity distinguishes them, which is exactly why the distinction had
            # to be made in the type system rather than in a comment.
            @test MLFMA.interpolationtrait(operator) !== MLFMA.VirtualExtension()
        end

        # Same continuation rule, different arity -- which is the whole of what the parameter carries.
        let hyper = Helmholtz2D.hypersingular(; wavenumber=2.0)
            @test MLFMA.interpolationtrait(hyper) === MLFMA.PeriodicφInterpolation{3}()
            @test MLFMA.traitarity(MLFMA.interpolationtrait(hyper)) == 3
            @test MLFMA.interpolationtrait(hyper) !== MLFMA.PeriodicφInterpolation()
        end

        # The arities of the spherical traits, which the circular family reads to check a caller's
        # trait against the far field it was handed.
        @test MLFMA.traitarity(MLFMA.VirtualExtension()) == 1
        @test MLFMA.traitarity(MLFMA.VirtualExtensionFlipθSign()) == 1
        @test MLFMA.traitarity(MLFMA.SphericalθφInterpolation()) == 2
        @test MLFMA.traitarity(MLFMA.SphericalInterpolation()) == 3
        @test MLFMA.traitarity(MLFMA.DoubleSphericalInterpolation()) == 6
        # `DefaultInterpolation` deliberately has none -- it means "whatever this far field's default
        # is", so asking is the caller's mistake.
        @test_throws MethodError MLFMA.traitarity(MLFMA.DefaultInterpolation())

        # A SPHERICAL family handed the 2-D trait must FAIL rather than pick a pole rule. This is the
        # whole practical payoff of naming the trait, so it is asserted rather than described.
        @test_throws MethodError MLFMA.converttrait(
            MLFMA.PeriodicφInterpolation(), (1,), MLFMA.DefaultInterpolation()
        )
        @test_throws MethodError MLFMA.Equidistant.componentparities(
            MLFMA.PeriodicφInterpolation(), MLFMA.ScalarFarField
        )
    end

    @testset "exact on data the source grid resolves" begin
        # THE PRIMARY ORACLE. `NφCircular(L)` is built to carry degree L exactly, so a degree-L
        # trigonometric polynomial must survive a move to any finer grid with nothing but round-off.
        Random.seed!(11)
        for L in (3, 5, 8, 12)
            N = MLFMA.NφCircular(L)
            child = MLFMA.CircularSampling{Float64}(N)
            coefficients = Dict(m => randn(ComplexF64) for m in (-L):L)
            f(φ) = sum(c * cis(m * φ) for (m, c) in coefficients)

            for parentdegree in (L + 1, L + 4, 2L)
                parent = MLFMA.CircularSampling{Float64}(MLFMA.NφCircular(parentdegree))

                farfield = MLFMA.ScalarFarField(T, child)
                farfield[1] .= f.(MLFMA.φs(child))
                interpolated = MLFMA.ScalarFarField(T, parent)

                MLFMA.CircularInterpolator{T}()(interpolated, farfield, child, parent)

                want = f.(MLFMA.φs(parent))
                @test norm(interpolated[1] - want) / norm(want) < 1e-13
                @test norm(want) > 1e-6                       # non-vacuity
            end
        end
    end

    @testset "the pairings, in the readable form" begin
        # `anterpolationcontract.jl` pins `anter ≈ adjoint(inter)` as MATRICES for this family. This is
        # the same statement written as an inner product, which is the form that would catch a
        # convention error only visible on a field with real φ structure rather than on one basis
        # direction. It lives here rather than in the contract file because the spherical version fills
        # its fields with spherical harmonics, which mean nothing on a circle.
        #
        # For an interpolation operator I and its anterpolator A:
        #   A == transpose(I)  <=>  sum(I*x .* y) == sum(x .* A*y)     (bilinear)
        #   A == adjoint(I)    <=>  dot(I*x, y)   == dot(x, A*y)       (Hermitian)
        # This family's operator is real, so BOTH must hold -- which is the pairing form of
        # `isrealinterpolation`, arrived at independently of the matrix that declares it.
        Random.seed!(77)
        child = MLFMA.CircularSampling{Float64}(12)
        parent = MLFMA.CircularSampling{Float64}(20)
        interpolator = MLFMA.CircularInterpolator{T}()
        anterpolator = MLFMA.anterpolator(interpolator)

        # A full band, not a single mode: both pairings factor a sum over an equispaced φ grid, so a
        # single-harmonic pair sums to exactly zero unless its exponent vanishes, which would make this
        # testset vacuous. Different seeds so neither pairing can pass by a symmetry between the sides.
        L = 5
        band(N, seed) = begin
            Random.seed!(seed)
            coefficients = Dict(m => randn(ComplexF64) for m in (-L):L)
            [
                sum(c * cis(m * φ) for (m, c) in coefficients) for
                φ in MLFMA.φs(MLFMA.CircularSampling{Float64}(N))
            ]
        end

        x = MLFMA.ScalarFarField(T, child)
        x[1] .= band(12, 0x1111)
        y = MLFMA.ScalarFarField(T, parent)
        y[1] .= band(20, 0x2222)

        Ix = interpolator(x, child, parent)
        Ay = anterpolator(y, parent, child)

        bilinearleft = sum(Ix[1] .* y[1])
        bilinearright = sum(x[1] .* Ay[1])
        hermitianleft = dot(Ix[1], y[1])
        hermitianright = dot(x[1], Ay[1])

        # Non-vacuity: the pairings must be nonzero and must differ from each other, or the two
        # relations below cannot be told apart.
        @test abs(bilinearleft) > 1e-6
        @test abs(hermitianleft) > 1e-6
        @test !(bilinearleft ≈ hermitianleft)

        @test hermitianleft ≈ hermitianright
        @test bilinearleft ≈ bilinearright
    end

    @testset "every component gets the same treatment" begin
        # A multi-component far field is resampled component-wise with NO per-component sign, which is
        # the arithmetic consequence of the trait decision. Checked by resampling each component
        # separately as a scalar and demanding the results agree BITWISE -- if a parity table ever crept
        # in, one of the two components would differ.
        Random.seed!(4)
        child = MLFMA.CircularSampling{Float64}(12)
        parent = MLFMA.CircularSampling{Float64}(20)
        interpolator = MLFMA.CircularInterpolator{T}()

        tangential = MLFMA.TangentialSphericalFarField(T, child, Val(:zero))
        tangential[1] .= randn(ComplexF64, 12)
        tangential[2] .= tangential[1]                        # same content in both components
        up = MLFMA.TangentialSphericalFarField(T, parent, Val(:zero))
        interpolator(up, tangential, child, parent)

        @test up[1] == up[2]

        scalar = MLFMA.ScalarFarField(T, child)
        scalar[1] .= tangential[1]
        scalarup = MLFMA.ScalarFarField(T, parent)
        interpolator(scalarup, scalar, child, parent)
        @test scalarup[1] == up[1]
    end

    @testset "a trait of the wrong arity is refused by name" begin
        child = MLFMA.CircularSampling{Float64}(8)
        parent = MLFMA.CircularSampling{Float64}(12)
        interpolator = MLFMA.CircularInterpolator{T}()
        scalar = MLFMA.ScalarFarField(T, child)
        out = MLFMA.ScalarFarField(T, parent)

        # Without the check this is a `BoundsError` from inside a component loop naming neither the
        # trait nor the field.
        @test_throws ErrorException interpolator(
            out, scalar, child, parent; interpolationtrait=MLFMA.SphericalθφInterpolation()
        )

        # A SPHERICAL trait of the RIGHT arity is accepted, and has to be: `_getinterpolationmatrix`
        # probes through a `TangentialSphericalFarField` and so arrives carrying
        # `SphericalθφInterpolation()`, which is what makes the adjointness contract measurable for
        # this family at all. Accepting it changes no arithmetic, since there is no sign to change.
        tangential = MLFMA.TangentialSphericalFarField(T, child, Val(:zero))
        tangentialup = MLFMA.TangentialSphericalFarField(T, parent, Val(:zero))
        @test interpolator(
            tangentialup,
            tangential,
            child,
            parent;
            interpolationtrait=MLFMA.SphericalθφInterpolation(),
        ) === nothing
    end

    @testset "the level-pair constructor validates what it is handed" begin
        circular = MLFMA.CircularSampling{Float64}(8)
        spherical = MLFMA.SphericalSampling{Float64}(4, 8)

        @test MLFMA.CircularInterpolator{0,T}([circular], [circular]) isa
            MLFMA.CircularInterpolator{0,T}
        # Mismatched pair lengths: they are level PAIRS, and a silent zip would build information for
        # the wrong levels.
        @test_throws ErrorException MLFMA.CircularInterpolator{0,T}(
            [circular, circular], [circular]
        )
        # A spherical grid handed to the circular family. Without this it fails at the first apply,
        # deep inside a traversal, rather than at construction.
        @test_throws ErrorException MLFMA.CircularInterpolator{0,T}(
            [spherical], [spherical]
        )
        @test_throws ErrorException MLFMA.CircularInterpolator{0,T}([circular], [spherical])

        # The trait is STORED, which is what the map constructors' agreement check reads.
        stored = MLFMA.CircularInterpolator{0,T}(
            [circular], [circular]; interpolationtrait=MLFMA.PeriodicφInterpolation()
        )
        @test MLFMA.interpolationtrait(stored) === MLFMA.PeriodicφInterpolation()
        @test MLFMA.interpolationtrait(MLFMA.anterpolator(stored)) ===
            MLFMA.PeriodicφInterpolation()
        # Concrete and callable as a partially applied type -- the shape `interpolatortype(strategy)`
        # hands the map constructors. A trait as a type parameter would break exactly this.
        @test isconcretetype(MLFMA.CircularInterpolator{0,T})
        @test MLFMA.stencilsize(stored) == 0
    end

    @testset "the duplicated coefficient move agrees with Equidistant's, bitwise" begin
        # See this file's header for why the duplication exists. `==`, not `≈`: these are the same
        # arithmetic in the same order, and anything less would let a real divergence hide in rounding.
        Random.seed!(9)
        forwardmatches = 0
        adjointmatches = 0
        for Nsrc in 4:2:16, Ndst in 4:2:16
            v = randn(ComplexF64, Nsrc)
            @test MLFMA._movecircularcoefficients!(
                Vector{ComplexF64}(undef, Ndst), v, Nsrc, Ndst
            ) == MLFMA.Equidistant.resamplecoefficients(v, Nsrc, Ndst)
            forwardmatches += 1

            w = randn(ComplexF64, Ndst)
            @test MLFMA._movecircularcoefficientsadjoint!(
                Vector{ComplexF64}(undef, Nsrc), w, Nsrc, Ndst
            ) == MLFMA.Equidistant.resamplecoefficientsadjoint(w, Nsrc, Ndst)
            adjointmatches += 1
        end
        # Non-vacuity: the sweep has to have run, or the two assertions above are a pair of no-ops.
        @test forwardmatches == 49
        @test adjointmatches == 49

        # The move and its adjoint are exact transposes, checked as MATRICES rather than inferred from
        # the loop above -- that is the property the anterpolator's correctness rests on.
        for (Nsrc, Ndst) in ((6, 10), (10, 6), (8, 8), (8, 16), (16, 8))
            P = zeros(ComplexF64, Ndst, Nsrc)
            for k in 1:Nsrc
                e = zeros(ComplexF64, Nsrc)
                e[k] = 1
                P[:, k] = MLFMA._movecircularcoefficients!(
                    Vector{ComplexF64}(undef, Ndst), e, Nsrc, Ndst
                )
            end
            Pt = zeros(ComplexF64, Nsrc, Ndst)
            for k in 1:Ndst
                e = zeros(ComplexF64, Ndst)
                e[k] = 1
                Pt[:, k] = MLFMA._movecircularcoefficientsadjoint!(
                    Vector{ComplexF64}(undef, Nsrc), e, Nsrc, Ndst
                )
            end
            @test Pt == transpose(P)
            # Real, which is what makes `Pᵀ == Pᴴ` and hence the anterpolator the exact adjoint.
            @test all(iszero, imag(P))
            @test norm(P) > 0                                 # non-vacuity
        end
    end

    @testset "the scratch changes nothing but the allocations" begin
        Random.seed!(3)
        for (N, M) in ((12, 28), (18, 26), (26, 26), (28, 12))
            child = MLFMA.CircularSampling{Float64}(N)
            parent = MLFMA.CircularSampling{Float64}(M)
            interpolator = MLFMA.CircularInterpolator{T}()
            anterpolator = MLFMA.anterpolator(interpolator; nchunks=1)
            scratch = MLFMA.interimscratch(anterpolator, child, parent)
            @test scratch !== nothing

            input = MLFMA.TangentialSphericalFarField(T, child, Val(:zero))
            input[1] .= randn(ComplexF64, N)
            input[2] .= randn(ComplexF64, N)
            parentinput = MLFMA.TangentialSphericalFarField(T, parent, Val(:zero))
            parentinput[1] .= randn(ComplexF64, M)
            parentinput[2] .= randn(ComplexF64, M)

            plainup = MLFMA.TangentialSphericalFarField(T, parent, Val(:zero))
            bufferedup = MLFMA.TangentialSphericalFarField(T, parent, Val(:zero))
            interpolator(plainup, input, child, parent)
            interpolator(bufferedup, input, child, parent; interim=scratch)
            # BITWISE, not `≈`: the buffered path runs the same operations in the same order on the
            # same values, so anything less would be hiding a real difference in rounding.
            @test plainup[1] == bufferedup[1]
            @test plainup[2] == bufferedup[2]

            plaindown = MLFMA.TangentialSphericalFarField(T, child, Val(:zero))
            buffereddown = MLFMA.TangentialSphericalFarField(T, child, Val(:zero))
            anterpolator(plaindown, parentinput, parent, child)
            anterpolator(buffereddown, parentinput, parent, child; interim=scratch)
            @test plaindown[1] == buffereddown[1]
            @test plaindown[2] == buffereddown[2]

            # A scratch built for a DIFFERENT level pair must fall back to allocating rather than write
            # out of bounds -- the same promise `takeinterim` makes for the other families, and what
            # keeps a non-uniform tree from turning a mis-sized buffer into a corrupt moment.
            wrongscratch = MLFMA.interimscratch(
                anterpolator,
                MLFMA.CircularSampling{Float64}(N + 2),
                MLFMA.CircularSampling{Float64}(M + 2),
            )
            mismatched = MLFMA.TangentialSphericalFarField(T, parent, Val(:zero))
            interpolator(mismatched, input, child, parent; interim=wrongscratch)
            @test mismatched[1] == plainup[1]
            # A hand-built NamedTuple with no `circularscratch` field is not an error either.
            othershape = MLFMA.TangentialSphericalFarField(T, parent, Val(:zero))
            interpolator(
                othershape, input, child, parent; interim=(interimmatrix=zeros(T, 2, 2),)
            )
            @test othershape[1] == plainup[1]

            # `== 0`, not a threshold. The buffers and all four FFT plans live in the scratch, so a
            # warmed-up apply has nothing left to allocate in either direction.
            phase = cis.(randn(M))
            @test warmedallocations(
                interpolator, bufferedup, input, child, parent; interim=scratch
            ) == 0
            @test warmedallocations(
                anterpolator, buffereddown, parentinput, parent, child; interim=scratch
            ) == 0
            @test warmedallocations(
                interpolator,
                bufferedup,
                input,
                child,
                parent;
                interim=scratch,
                postscale=phase,
            ) == 0
            @test warmedallocations(
                anterpolator,
                buffereddown,
                parentinput,
                parent,
                child;
                interim=scratch,
                prescale=phase,
            ) == 0
        end
    end

    @testset "the fused diagonal is bitwise the two-pass form" begin
        # What lets the level-transfer tests compare fused and unfused paths with `==` rather than `≈`.
        Random.seed!(21)
        child = MLFMA.CircularSampling{Float64}(12)
        parent = MLFMA.CircularSampling{Float64}(28)
        interpolator = MLFMA.CircularInterpolator{T}()
        anterpolator = MLFMA.anterpolator(interpolator)

        farfield = MLFMA.ScalarFarField(T, child)
        farfield[1] .= randn(ComplexF64, 12)
        phase = cis.(randn(28))

        fused = MLFMA.ScalarFarField(T, parent)
        interpolator(fused, farfield, child, parent; postscale=phase)
        twopass = MLFMA.ScalarFarField(T, parent)
        interpolator(twopass, farfield, child, parent)
        twopass[1] .*= phase
        @test fused[1] == twopass[1]

        parentfield = MLFMA.ScalarFarField(T, parent)
        parentfield[1] .= randn(ComplexF64, 28)
        before = copy(parentfield[1])

        fuseddown = MLFMA.ScalarFarField(T, child)
        anterpolator(fuseddown, parentfield, parent, child; prescale=phase)
        scaledinput = MLFMA.ScalarFarField(T, parent)
        scaledinput[1] .= before .* phase
        twopassdown = MLFMA.ScalarFarField(T, child)
        anterpolator(twopassdown, scaledinput, parent, child)
        @test fuseddown[1] == twopassdown[1]

        # THE INPUT IS NEVER TOUCHED, bitwise -- not merely restored to round-off as the local/global
        # family's is. This family stages into its own buffer instead of transforming the caller's
        # array, which matters in disaggregation where one parent moment feeds several children that
        # different chunks may process concurrently.
        @test parentfield[1] == before

        unscaled = MLFMA.ScalarFarField(T, child)
        anterpolator(unscaled, parentfield, parent, child)
        @test parentfield[1] == before
    end

    @testset "the algebraic wrappers decline to wrap, and still work when forced" begin
        # `isrealinterpolation` is a DECISION consumed by two wrappers, so both halves are asserted:
        # that they decline, and that the operator they would have wrapped is the same one.
        Random.seed!(31)
        child = MLFMA.CircularSampling{Float64}(12)
        parent = MLFMA.CircularSampling{Float64}(20)
        interpolator = MLFMA.CircularInterpolator{T}()
        anterpolator = MLFMA.anterpolator(interpolator)

        @test MLFMA.isrealinterpolation(interpolator)
        @test MLFMA.isrealinterpolation(anterpolator)
        @test MLFMA.transposedinterpolator(anterpolator) === anterpolator
        @test MLFMA.conjugatedinterpolator(interpolator) === interpolator

        # Forced construction: a real operator makes the wrappers UNEXERCISED, not unusable, and #35
        # and #61 were both adjoint-side bugs that a forward-only check would have missed. So the
        # wrappers are built by hand and required to reproduce the bare operators.
        parentfield = MLFMA.ScalarFarField(T, parent)
        parentfield[1] .= randn(ComplexF64, 20)
        bare = MLFMA.ScalarFarField(T, child)
        anterpolator(bare, parentfield, parent, child)

        forced = MLFMA.TransposedInterpolator(anterpolator)
        wrapped = MLFMA.ScalarFarField(T, child)
        forced(wrapped, parentfield, parent, child)
        @test wrapped[1] == bare[1]
        @test MLFMA.interpolator(forced) === interpolator
        @test MLFMA.interimscratch(forced, child, parent) isa NamedTuple

        childfield = MLFMA.ScalarFarField(T, child)
        childfield[1] .= randn(ComplexF64, 12)
        bareup = MLFMA.ScalarFarField(T, parent)
        interpolator(bareup, childfield, child, parent)

        forcedconjugate = MLFMA.ConjugatedInterpolator(interpolator)
        wrappedup = MLFMA.ScalarFarField(T, parent)
        forcedconjugate(wrappedup, childfield, child, parent)
        # `≈` here rather than `==`: the wrapper conjugates the input, runs the FFT over different bit
        # patterns and conjugates back, which rounds slightly differently. Measured 1.9e-16 relative.
        @test wrappedup[1] ≈ bareup[1]
        @test norm(bareup[1]) > 1e-6                          # non-vacuity
    end

    @testset "the batched entry points say why they are absent" begin
        child = MLFMA.CircularSampling{Float64}(8)
        parent = MLFMA.CircularSampling{Float64}(12)
        interpolator = MLFMA.CircularInterpolator{T}()
        anterpolator = MLFMA.anterpolator(interpolator)

        fields = [MLFMA.ScalarFarField(T, child)]
        outs = [MLFMA.ScalarFarField(T, parent)]
        @test_throws ErrorException interpolator(outs, fields, child, parent)
        @test_throws ErrorException anterpolator(fields, outs, parent, child)
    end

    @testset "the builder resolves to the family and hands back both reverse operators" begin
        builder = MLFMA.CircularInterpolationBuilder(; T=T)
        @test builder isa MLFMA.AbstractInterpolationBuilder
        @test MLFMA.elementtype(builder) === T
        # `0`, the family's spelling for "global -- there is no stencil whose size is a dial".
        @test MLFMA.interpolationorder(builder) == 0
        @test MLFMA.interpolatortype(builder) === MLFMA.CircularInterpolator{0,T}

        child = MLFMA.CircularSampling{Float64}(10)
        parent = MLFMA.CircularSampling{Float64}(18)
        built = MLFMA.buildinterpolator(
            builder, [child], [parent]; interpolationtrait=MLFMA.PeriodicφInterpolation()
        )
        @test built isa MLFMA.CircularInterpolator{0,T}
        @test MLFMA.interpolationtrait(built) === MLFMA.PeriodicφInterpolation()

        reverse = MLFMA.buildreverseoperators(builder, built; nchunks=2)
        @test reverse.anterpolator isa MLFMA.CircularAnterpolator
        # A REAL family, so the two reverse directions are one operator and both entries are the SAME
        # object. That is the correct answer, not a shortcut -- see `transposedinterpolation.jl`.
        @test reverse.transposedinterpolator === reverse.anterpolator
        @test MLFMA.buildanterpolator(builder, built; nchunks=2) isa
            MLFMA.CircularAnterpolator
    end
end
