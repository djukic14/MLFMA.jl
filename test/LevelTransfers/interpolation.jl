using MLFMA
using FastGaussQuadrature
using Test
using StaticArrays
using CompScienceMeshes
using BEAST
using LinearAlgebra

include("../support/util_test.jl")

@testset "Interpolator" begin
    stencilsize = 10

    # sampling of far field
    Nθ = 20
    Nφ = 40
    sampling = MLFMA.SphericalSampling{Float64}(Nθ, Nφ)

    # sampling of interpolated far field
    Nθinterpolation = 40
    Nφinterpolation = 80
    samplinginterpolation = MLFMA.SphericalSampling{Float64}(
        Nθinterpolation, Nφinterpolation
    )

    fn = joinpath(pkgdir(MLFMA), "test/assets/sphere.msh")

    m = CompScienceMeshes.readmesh(fn)
    X = raviartthomas(m)
    efie = MLFMA.efie(; wavenumber=1.0)
    doublelayer = Maxwell3D.doublelayer(; wavenumber=1.0)
    rotateddoublelayer = BEAST.n × doublelayer

    for operator in [efie, doublelayer, rotateddoublelayer]
        farfieldcollection = MLFMA.buildfieldcollection(
            radiationbuilder(), operator, sampling, X; verbose=true
        )

        extendedfarfielcollection = MLFMA.buildfieldcollection(
            radiationbuilder(), operator, samplinginterpolation, X; verbose=true
        )
        for i in 1:10
            u = zeros(numfunctions(X))
            u[i] = 1.0

            fvals = reshape(MLFMA.Fφ(farfieldcollection) * u, size(sampling))
            fvalsextended = reshape(
                MLFMA.Fφ(extendedfarfielcollection) * u, size(samplinginterpolation)
            )

            interpolator = MLFMA.SphericalLagrangeInterpolator{stencilsize}(
                ComplexF64, Float64
            )
            interpolantfarfield = MLFMA.TangentialSphericalFarField(
                im * 1.0, samplinginterpolation
            )

            interpolator(
                interpolantfarfield,
                MLFMA.TangentialSphericalFarField(SVector(fvals, fvals)),
                sampling,
                samplinginterpolation,
            )

            @test interpolantfarfield[1][:] ≈ fvalsextended[:] rtol = 1e-6

            farfield = MLFMA.TangentialSphericalFarField(im * 1.0, sampling)
            LinearAlgebra.mul!(farfield, farfieldcollection, u)
            interpolatedfarfield = MLFMA.TangentialSphericalFarField(
                im * 1.0, samplinginterpolation
            )

            LinearAlgebra.mul!(interpolatedfarfield, extendedfarfielcollection, u)
            interpolator(interpolantfarfield, farfield, sampling, samplinginterpolation)

            @test MLFMA.Fθ(interpolantfarfield) ≈ MLFMA.Fθ(interpolatedfarfield) rtol = 1e-6
            @test MLFMA.Fφ(interpolantfarfield) ≈ MLFMA.Fφ(interpolatedfarfield) rtol = 1e-6
        end
    end
end

@testset "Abstract Interface" begin
    struct MyInterpolator{stencilsize,T} <: MLFMA.AbstractInterpolator{stencilsize,T} end

    sampling = MLFMA.SphericalSampling{Float64}(2, 2)
    @test_throws ErrorException MyInterpolator{2,ComplexF64}()(
        MLFMA.TangentialSphericalFarField(im * 1.0, sampling),
        MLFMA.SphericalSampling{Float64}(2, 2),
        MLFMA.SphericalSampling{Float64}(2, 2),
    )

    @test MLFMA.stencilsize(MyInterpolator{2,Float64}()) == 2
end

@testset "both families agree on one shared θ information object" begin
    # THE COUPLING TEST. `GaussLegendre{Interpolation,Anterpolation}Information` is consumed by BOTH
    # kernel families -- the barycentric-Lagrange one in `sphericalinterpolation.jl` and the global-φ
    # one in `localglobalinterpolation.jl` -- and four bugs have come from changing the type for one
    # consumer and not the other: a missed `isassigned` check, a missed empty-range sentinel, and the
    # row-versus-column `mul!` asymmetry behind the `492475d` regression.
    #
    # WHY THIS RATHER THAN SPLITTING THE TYPE PER FAMILY, which was the standing proposal. The case for
    # splitting was that the families "want different storage". They did once; they no longer do. Both
    # θ kernels now read the same fields in the same order through the same `UnitRange` indices and
    # `Vector` weights -- the only difference left is the DESTINATION (a column view against a strided
    # row), and `492475d` resolved that by staging the row-writing family through a contiguous vector
    # rather than by changing storage. A split would thread a parameter through four wrapper types and
    # every kernel signature to buy separation for a divergence that does not exist, and it would not
    # even prevent the observed bug: when both families need the SAME change, two types still means
    # changing both. What actually catches that failure is feeding one object to both kernels.
    #
    # The two write transposed layouts of the same data, which is the whole reason they are easy to
    # desynchronize -- so the assertion has to transpose, not just compare.
    source = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(6)...)
    target = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(12)...)
    T = ComplexF64

    info = MLFMA.GaussLegendreInterpolationInformation{5}(source, target)
    # Non-vacuity: the extending branch -- where the sentinel and the rotation split live, and where
    # three of the four bugs were -- must actually be reached, or this compares only the easy path.
    @test !isempty(info.nonconformiterations)

    fvals = rand(T, MLFMA.Nφ(source), MLFMA.Nθ(source))

    columnwritten = zeros(T, MLFMA.Nφ(source), MLFMA.Nθ(target))
    MLFMA._interpolateθ!(columnwritten, fvals, info)

    rowwritten = zeros(T, MLFMA.Nθ(target), MLFMA.Nφ(source))
    MLFMA._interpolateθtransposed!(rowwritten, fvals, info, zeros(T, MLFMA.Nφ(source)))

    @test size(columnwritten) == reverse(size(rowwritten))
    # EXACT, measured: both reach `mulrotatedrows!` the same way, so there is not even a reassociation
    # difference between them. Equality is the strongest statement available and the most sensitive to
    # one family drifting.
    @test transpose(rowwritten) == columnwritten
    @test maximum(abs, columnwritten) > 1e-6   # non-vacuity: not comparing two zero matrices

    # --- the anterpolation twin, same argument ---------------------------------------------------
    anterpinfo = MLFMA.GaussLegendreAnterpolationInformation{5}(target, source)
    @test !isempty(anterpinfo.nonconformindices)

    anterpfvals = rand(T, MLFMA.Nφ(source), MLFMA.Nθ(target))

    anterpcolumn = zeros(T, MLFMA.Nφ(source), MLFMA.Nθ(source))
    MLFMA._anterpolateθ!(anterpcolumn, anterpfvals, anterpinfo, zeros(T, MLFMA.Nφ(source)))

    # The Lagrange kernel reads `transpose(fvals)`, so it takes the other orientation of the SAME data
    # and writes the same result layout.
    anterprow = zeros(T, MLFMA.Nφ(source), MLFMA.Nθ(source))
    MLFMA._anterpolateθtransposed!(
        anterprow, Matrix(transpose(anterpfvals)), anterpinfo, zeros(T, MLFMA.Nφ(source))
    )

    # `≈` and not `==` here, unlike the forward pair: this family still gathers the mirrored rows
    # through `flippingφindices` while the other uses `mulrotatedrows!`, and the two sum in different
    # orders. Measured 2.1e-16, so the bound is tight enough to catch a real divergence.
    @test anterprow ≈ anterpcolumn
    @test maximum(abs, anterprow - anterpcolumn) < 1e-14 * maximum(abs, anterpcolumn)
    @test maximum(abs, anterpcolumn) > 1e-6
end

@testset "θ information storage stays on the fast mul! path" begin
    # These are TYPE assertions, which normally reads as testing the implementation rather than the
    # behaviour. They earn their place here because the storage type IS the contract: reverting
    # `posextendingindices` to `Vector{Int}` or `weights` to `SVector` costs 20x and 4.4x respectively on
    # every θ product and breaks NOTHING functionally, so no behavioural test would catch it. See the
    # `GaussLegendreInterpolationInformation` docstring for the measurements.
    source = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(6)...)
    target = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(12)...)
    info = MLFMA.GaussLegendreInterpolationInformation{5}(source, target)

    @test eltype(info.conformindices) == UnitRange{Int}
    @test eltype(info.posextendingindices) == UnitRange{Int}
    @test eltype(info.negextendingindices) == UnitRange{Int}
    @test eltype(info.weights) == Vector{Float64}
    @test eltype(info.posextendingweights) == Vector{Float64}
    @test eltype(info.negextendingweights) == Vector{Float64}

    # Zero-padding means index count and weight count must agree entry by entry -- the invariant that
    # makes the padded product equal the sparse one.
    for (ranges, weights) in (
        (info.conformindices, info.weights),
        (info.posextendingindices, info.posextendingweights),
        (info.negextendingindices, info.negextendingweights),
    )
        @test length(ranges) == length(weights)
        for i in eachindex(ranges)
            @test length(ranges[i]) == length(weights[i])
        end
    end

    # The extending branch is what the rotation split rewrote, so assert it is actually reached here --
    # otherwise the checks above would pass vacuously on a grid pair with no pole extension.
    @test !isempty(info.nonconformiterations)
end

@testset "the rotation split reproduces the permuted-row product" begin
    # `mulrotatedrows!` replaced `view(fvals, flippingφindices, indices) * weights` in both forward θ
    # kernels. This pins the decomposition against the form it replaced, directly rather than through a
    # whole apply, and with α/β exercised in the accumulating configuration the kernels use.
    Nφ, Nθ = 16, 9
    fvals = rand(ComplexF64, Nφ, Nθ)
    flipping = vcat((Nφ ÷ 2 + 1):Nφ, 1:(Nφ ÷ 2))
    indices = 3:6
    weights = rand(Float64, length(indices))

    reference = view(fvals, flipping, indices) * weights
    result = zeros(ComplexF64, Nφ)
    MLFMA.mulrotatedrows!(result, fvals, indices, weights)
    @test result ≈ reference
    # Round-off only: the split sums in a different order, so `==` would be the wrong assertion.
    @test maximum(abs, result - reference) < 1e-14 * maximum(abs, reference)

    # Accumulating form, as the interpolator's extending branch calls it.
    initial = rand(ComplexF64, Nφ)
    accumulated = copy(initial)
    MLFMA.mulrotatedrows!(accumulated, fvals, indices, weights, true, true)
    @test accumulated ≈ initial + reference

    # An empty range contributes nothing when accumulating, and zeroes when assigning.
    untouched = copy(initial)
    MLFMA.mulrotatedrows!(untouched, fvals, 1:0, Float64[], true, true)
    @test untouched == initial
    zeroed = copy(initial)
    MLFMA.mulrotatedrows!(zeroed, fvals, 1:0, Float64[])
    @test all(iszero, zeroed)
end

@testset "gap padding equals the sparse product it replaces" begin
    # The padding is only equivalent because a gap marks an index whose weight was zero. Assert that
    # directly: a gapped set and its padded span must give the same product.
    Nφ, Nθ = 12, 8
    fvals = rand(ComplexF64, Nφ, Nθ)
    sparseindices = [2, 3, 5, 6]
    sparseweights = rand(Float64, 4)

    ranges, padded = MLFMA.spanningranges([sparseindices], [sparseweights], Float64)
    @test ranges[1] == 2:6
    @test padded[1][3] == 0.0   # span is 2:6, so position 3 is the skipped index 4
    @test count(iszero, padded[1]) == 1  # exactly the one gap, nothing else zeroed
    @test length(padded[1]) == length(ranges[1])

    @test view(fvals, :, ranges[1]) * padded[1] ≈
        view(fvals, :, sparseindices) * sparseweights

    # An empty or unassigned set becomes the empty range, the sentinel the kernels test with `isempty`.
    emptyranges, emptyweights = MLFMA.spanningranges(
        Vector{Vector{Int}}(undef, 1), Vector{Vector{Float64}}(undef, 1), Float64
    )
    @test emptyranges[1] == 1:0
    @test isempty(emptyweights[1])
end

@testset "scratch-backed applies allocate nothing" begin
    # WHY AN ABSOLUTE PIN, when "Level transfer apply allocation contract" in protocol.jl deliberately
    # pins a DELTA instead. That contract measures the transfer layer's overhead RELATIVE to the bare
    # kernel, which is correct for what it guards but structurally cannot see a kernel that starts
    # allocating: both sides of the subtraction rise together and the difference stays inside budget.
    # A θ-kernel change did exactly that -- 440 allocations / 26 KB per Lagrange forward apply -- and the
    # full suite passed, because the only absolute pins covered the default family. This closes that gap
    # for both families and both directions.
    #
    # The figures are constant in grid size and stencil order (checked at Nθ 8/14/28/42, p 4 and 15), so
    # a small grid pins the same numbers a production one would.
    #
    # `interpolationtrait=trait` ON THE BUILDER IS LOAD-BEARING, and its absence is what made this
    # testset report a 224-byte forward allocation that production never paid. Both map constructors
    # build their interpolator with the OPERATOR's concrete trait; left to default, the interpolator
    # stores `DefaultInterpolation()` and every apply asking for `SphericalθφInterpolation()` makes
    # `converttrait` build a fresh `(info, info)` tuple -- 224 bytes of pure measurement artifact. The
    # defaulted path is pinned separately below rather than pretended away.
    source = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(3)...)
    target = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(6)...)
    trait = MLFMA.SphericalθφInterpolation()

    measure(f) = (f(); f(); @allocated f())   # warm up first, so compilation is never counted

    results = Dict{String,NTuple{2,Int}}()
    for (name, builder) in (
        ("lagrange", MLFMA.SphericalLagrangeInterpolationBuilder(; order=4)),
        ("localglobal", MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=4)),
    )
        interpolator = MLFMA.buildinterpolator(
            builder, [source], [target]; interpolationtrait=trait
        )
        reverse = MLFMA.buildreverseoperators(builder, interpolator; nchunks=1)
        scratch = MLFMA.interimscratch(reverse.anterpolator, source, target)

        child = MLFMA.TangentialSphericalFarField(ComplexF64, source)
        child[1] .= 1.0
        child[2] .= 1.0
        parent = MLFMA.TangentialSphericalFarField(ComplexF64, target)
        parent[1] .= 1.0
        parent[2] .= 1.0
        parentout = MLFMA.TangentialSphericalFarField(ComplexF64, target)
        childout = MLFMA.TangentialSphericalFarField(ComplexF64, source)

        forward = measure(
            () -> interpolator(
                parentout,
                child,
                source,
                target;
                interpolationtrait=trait,
                interim=scratch,
            ),
        )
        backward = measure(
            () -> reverse.anterpolator(
                childout,
                parent,
                target,
                source;
                interpolationtrait=trait,
                interim=scratch,
            ),
        )
        results[name] = (forward, backward)
    end

    @info "scratch-backed apply allocations (bytes)" results

    # ALL FOUR ARE ZERO in the configuration production uses. Equality, not an upper bound: there is no
    # remaining known cost to leave headroom for, and `<=` on a zero target only hides a regression.
    @test results["lagrange"][1] == 0
    @test results["lagrange"][2] == 0
    @test results["localglobal"][1] == 0
    @test results["localglobal"][2] == 0
end

@testset "a trait-defaulted interpolator pays for the conversion, and only there" begin
    # The companion to the pin above, and the reason it can assert zero. `converttrait(::T, v, ::T)` is
    # the identity, so an interpolator carrying the trait its applies request converts nothing. One that
    # does NOT -- `DefaultInterpolation()` from `buildinterpolator`'s default -- rebuilds its information
    # container on EVERY apply.
    #
    # This is asserted rather than merely avoided for two reasons. It is the standing explanation for a
    # 224-byte figure that sat in this file as a "known, unfixed" production cost through several
    # attempts to remove it, when the real fix was to build the interpolator the way the maps already
    # did -- so the number needs to stay attached to its actual cause. And it is a live trap: any future
    # code path that reaches an apply through a defaulted interpolator pays this per call.
    source = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(3)...)
    target = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(6)...)
    trait = MLFMA.SphericalθφInterpolation()
    builder = MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=4)

    defaulted = MLFMA.buildinterpolator(builder, [source], [target])
    matched = MLFMA.buildinterpolator(builder, [source], [target]; interpolationtrait=trait)
    @test MLFMA.interpolationtrait(defaulted) == MLFMA.DefaultInterpolation()
    @test MLFMA.interpolationtrait(matched) == trait

    scratch = MLFMA.interimscratch(
        MLFMA.buildreverseoperators(builder, matched; nchunks=1).anterpolator,
        source,
        target,
    )
    child = MLFMA.TangentialSphericalFarField(ComplexF64, source)
    child[1] .= 1.0
    child[2] .= 1.0
    out = MLFMA.TangentialSphericalFarField(ComplexF64, target)

    apply(interp) = (
        f=() -> interp(
            out, child, source, target; interpolationtrait=trait, interim=scratch
        );
        f();
        f();
        @allocated f()
    )

    # The conversion costs a fixed 224 bytes -- one `(info, info)` tuple, constant in grid size and
    # order. Bounded rather than pinned exactly, since the figure is a struct size and not a contract.
    @test 0 < apply(defaulted) <= 224
    @test apply(matched) == 0

    # ...and it is the TRAIT that makes the difference, not the two objects differing some other way:
    # the identity conversion is what the matched one gets.
    stored = matched.interpolationinfodict[(source, target)]
    @test MLFMA.converttrait(trait, stored, trait) === stored
end

@testset "fused postscale equals separate interpolation + phase shift" begin
    # THE EQUIVALENCE THE FUSION HAS TO EARN. `InterpolationPhaseTransfer` used to interpolate and then
    # walk the parent grid a second time with `elementwisemul!(out, out, phase)`; it now hands the
    # diagonal to the interpolator as `postscale`, which folds it into whatever the φ stage writes last.
    # That is only a valid optimization if the two produce the same numbers.
    #
    # `==`, NOT `≈`, and that is the point of the exercise. The fused kernels apply scale-then-diagonal
    # per element -- the same operations in the same order as the two-pass form -- so the results are
    # BITWISE identical. An `≈` here would pass on a kernel that precombined the factors
    # (`v * (scale * phase)` instead of `(v * scale) * phase`), which rounds differently and would drift
    # from the reference the rest of the suite is pinned against.
    source = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(4)...)
    target = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(8)...)
    T = ComplexF64
    trait = MLFMA.SphericalθφInterpolation()

    k̂s, _ = MLFMA.cartesiansamplepoints(target)
    phase = MLFMA.phasefactor(
        im * 2π, k̂s, SVector(0.19, -0.27, 0.11), MLFMA.AbstractRadiationFieldCollection
    )
    # Non-vacuity: a real or near-real diagonal would make the fused and unfused forms agree for
    # reasons that have nothing to do with the fusion being right.
    @test maximum(abs, imag.(phase)) > 0.1
    @test !all(isapprox(1.0), phase)

    for (name, builder) in (
        ("lagrange", MLFMA.SphericalLagrangeInterpolationBuilder(; order=4)),
        ("localglobal", MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=4)),
    )
        interpolator = MLFMA.buildinterpolator(
            builder, [source], [target]; interpolationtrait=trait
        )
        scratch = MLFMA.interimscratch(
            MLFMA.buildreverseoperators(builder, interpolator; nchunks=1).anterpolator,
            source,
            target,
        )

        child = MLFMA.TangentialSphericalFarField(T, source)
        for i in eachindex(child)
            child[i] .= reshape(
                [cis(0.31 * n) * (1 + 0.4 * cospi(n / 11)) for n in 1:length(child[i])],
                size(child[i]),
            )
        end

        # SEPARATE: interpolate, then apply the diagonal as its own pass -- exactly what the transfer
        # did before this change.
        separate = MLFMA.TangentialSphericalFarField(T, target)
        interpolator(
            separate, child, source, target; interpolationtrait=trait, interim=scratch
        )
        MLFMA.elementwisemul!(separate, separate, phase)

        # FUSED: the same diagonal handed to the interpolator.
        fused = MLFMA.TangentialSphericalFarField(T, target)
        interpolator(
            fused,
            child,
            source,
            target;
            interpolationtrait=trait,
            interim=scratch,
            postscale=phase,
        )

        for i in eachindex(fused)
            @test fused[i] == separate[i]
        end
        @test maximum(abs, separate[1]) > 1e-6   # non-vacuity: not comparing zeros

        # `postscale=nothing` must leave the interpolator exactly as it was -- the unscaled path is the
        # one every standalone caller, diagnostic and benchmark uses, and it has to stay free.
        plain = MLFMA.TangentialSphericalFarField(T, target)
        interpolator(
            plain,
            child,
            source,
            target;
            interpolationtrait=trait,
            interim=scratch,
            postscale=nothing,
        )
        implicit = MLFMA.TangentialSphericalFarField(T, target)
        interpolator(
            implicit, child, source, target; interpolationtrait=trait, interim=scratch
        )
        for i in eachindex(plain)
            @test plain[i] == implicit[i]
        end
        # ...and the fused result must actually DIFFER from the unscaled one, or the diagonal is being
        # dropped and every equality above would hold vacuously.
        @test fused[1] != plain[1]

        # Fusing must not have made the apply allocate.
        run() = interpolator(
            fused,
            child,
            source,
            target;
            interpolationtrait=trait,
            interim=scratch,
            postscale=phase,
        )
        run()
        run()
        @test (@allocated run()) == 0
    end
end

@testset "prescale equals separate phase shift + anterpolation" begin
    # The reverse twin of the `postscale` testset above, and it pins TWO properties, because the
    # `prescale` handover changed more than where the multiply happens.
    #
    #   1. EQUIVALENCE. `prescale` must give bitwise what `elementwisemul!` into a staging moment
    #      followed by a plain anterpolation gave.
    #   2. THE CALLER'S INPUT IS NOW UNTOUCHED. Without `prescale` the local/global anterpolator FFTs
    #      its input in place and undoes it, returning it perturbed by ~4e-16 -- a documented wart.
    #      With `prescale` the operator stages a private copy, so the input comes back BITWISE
    #      identical. That is what licenses skipping the restoring inverse FFT (31 µs per component),
    #      which is where the reverse direction's 1.46x actually comes from -- not from the diagonal.
    child = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(4)...)
    parent = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(8)...)
    T = ComplexF64
    trait = MLFMA.SphericalθφInterpolation()

    k̂s, _ = MLFMA.cartesiansamplepoints(parent)
    phase = MLFMA.phasefactor(
        im * 2π, k̂s, SVector(0.19, -0.27, 0.11), MLFMA.AbstractRadiationFieldCollection
    )
    @test maximum(abs, imag.(phase)) > 0.1   # non-vacuity: a real diagonal would prove nothing

    fill!(m) =
        for i in eachindex(m)
            m[i] .= reshape(
                [cis(0.29 * n) * (1 + 0.3 * sinpi(n / 13)) for n in 1:length(m[i])],
                size(m[i]),
            )
        end

    for (name, builder) in (
        ("lagrange", MLFMA.SphericalLagrangeInterpolationBuilder(; order=4)),
        ("localglobal", MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=4)),
    )
        interpolator = MLFMA.buildinterpolator(
            builder, [child], [parent]; interpolationtrait=trait
        )
        anterpolator =
            MLFMA.buildreverseoperators(builder, interpolator; nchunks=1).anterpolator
        interim = MLFMA.interimscratch(anterpolator, child, parent)

        # SEPARATE: stage the diagonal into a moment, then anterpolate it -- what the transfer did.
        parentin = MLFMA.TangentialSphericalFarField(T, parent)
        fill!(parentin)
        staged = MLFMA.TangentialSphericalFarField(T, parent)
        MLFMA.elementwisemul!(staged, parentin, phase)
        separate = MLFMA.TangentialSphericalFarField(T, child)
        anterpolator(
            separate, staged, parent, child; interpolationtrait=trait, interim=interim
        )

        # PRESCALE: the same diagonal handed to the operator, reading the untouched input.
        parentin2 = MLFMA.TangentialSphericalFarField(T, parent)
        fill!(parentin2)
        pristine = [copy(parentin2[i]) for i in eachindex(parentin2)]
        fused = MLFMA.TangentialSphericalFarField(T, child)
        anterpolator(
            fused,
            parentin2,
            parent,
            child;
            interpolationtrait=trait,
            interim=interim,
            prescale=phase,
        )

        for i in eachindex(fused)
            @test fused[i] == separate[i]
        end
        @test maximum(abs, separate[1]) > 1e-6   # non-vacuity: not comparing zeros

        # The input is the caller's and must come back EXACTLY, which is stronger than this family's
        # historical contract and is the property the restore-skip depends on being true.
        for i in eachindex(parentin2)
            @test parentin2[i] == pristine[i]
        end

        # `prescale=nothing` leaves the anterpolator exactly as it was, restore included.
        plainin = MLFMA.TangentialSphericalFarField(T, parent)
        fill!(plainin)
        plain = MLFMA.TangentialSphericalFarField(T, child)
        anterpolator(
            plain, plainin, parent, child; interpolationtrait=trait, interim=interim
        )
        implicitin = MLFMA.TangentialSphericalFarField(T, parent)
        fill!(implicitin)
        implicit = MLFMA.TangentialSphericalFarField(T, child)
        anterpolator(
            implicit,
            implicitin,
            parent,
            child;
            interpolationtrait=trait,
            interim=interim,
            prescale=nothing,
        )
        for i in eachindex(plain)
            @test plain[i] == implicit[i]
        end
        # ...and the diagonal must actually change the answer, or every equality above is vacuous.
        @test fused[1] != plain[1]

        # THE TRANSPOSE WRAPPER, where `prescale` has to be conjugated on the way in. Only the
        # local/global family builds one; for Lagrange `transposedinterpolator` returns the
        # anterpolator itself and this re-checks the same path, which is harmless.
        transposed = MLFMA.transposedinterpolator(anterpolator)
        tin = MLFMA.TangentialSphericalFarField(T, parent)
        fill!(tin)
        tstaged = MLFMA.TangentialSphericalFarField(T, parent)
        MLFMA.elementwisemul!(tstaged, tin, phase)
        tseparate = MLFMA.TangentialSphericalFarField(T, child)
        transposed(
            tseparate, tstaged, parent, child; interpolationtrait=trait, interim=interim
        )

        tin2 = MLFMA.TangentialSphericalFarField(T, parent)
        fill!(tin2)
        tfused = MLFMA.TangentialSphericalFarField(T, child)
        transposed(
            tfused,
            tin2,
            parent,
            child;
            interpolationtrait=trait,
            interim=interim,
            prescale=phase,
        )
        for i in eachindex(tfused)
            @test tfused[i] == tseparate[i]
        end

        # THE WRAPPER MUST NOT TOUCH THE CALLER'S INPUT EITHER, and this assertion exists because its
        # absence shipped a race. `TransposedInterpolator` reaches the transpose by conjugating its
        # input IN PLACE and restoring it, which was safe only while the level transfer handed it a
        # private per-chunk staging copy. Once the transfer started passing the traversal's live
        # moment, one parent moment feeding several children on different chunks meant one task
        # conjugating what another was reading: the threaded suite saw 2.2e-6 correct and up to
        # 2.6e-3 wrong from run to run, and every serial run passed.
        #
        # The equivalence checks above did NOT catch it -- both the fused and separate paths mutated a
        # copy the test itself owned, so the aliasing was invisible. Comparing outputs cannot see an
        # input being clobbered; only asserting the input's survival can.
        tpristine = [copy(tin2[i]) for i in eachindex(tin2)]
        transposed(
            tfused,
            tin2,
            parent,
            child;
            interpolationtrait=trait,
            interim=interim,
            prescale=phase,
        )
        for i in eachindex(tin2)
            @test tin2[i] == tpristine[i]
        end

        # Allocation-free with the diagonal, as without it.
        run() = anterpolator(
            fused,
            parentin2,
            parent,
            child;
            interpolationtrait=trait,
            interim=interim,
            prescale=phase,
        )
        run()
        run()
        @test (@allocated run()) == 0
    end
end
