include("../support/util_test.jl")

# `mirror`/`mirror!`/`receivefields`/`receivefields!` are now built directly on
# `mirrorrule`/`receivecouplingrule` + `transform!` (farfield modernization plan §6), so comparing them
# against `applyrule` below is a wiring/dispatch check (did this pick the right representation and
# rule for this operator?), not independent verification of the arithmetic -- that was pinned
# against the old hand-written per-operator methods when these tables were introduced (see
# 25ac666, before the rewrite). Independent ground truth for the actual physics lives in
# `farfield.jl`'s "Mirroring" testset (reversed spherical unit vectors) and `mirrored.jl`
# (compared against BEAST). Tiny deterministic synthetic matrices here, not BEAST assembly, so
# every expected value is exact and a component swap/sign slip fails immediately.

# --- LOCAL REFERENCES ---------------------------------------------------------------------------
#
# `LazyReceiveFieldCollection` is now a SCRATCH-ONLY leaf collection: it implements the two
# `H2Factory.applyleafmul!` arities the traversal calls and nothing else -- no `mul!`, no `conjmul!`,
# no allocating `receivefields`. Those used to live in `src/` as a fused fallback and as this file's
# ground truth; once the leaf channel landed they had no production caller left, so they moved here,
# where their only remaining job -- being an independent reference -- is all they are.
#
# `receivefieldsreference` is the allocating wrapper around the surviving in-place kernel.
function receivefieldsreference(
    src::F, mirrorindices, op
) where {D,T,F<:MLFMA.AbstractFarField{D,T}}
    dest = MLFMA.basefarfieldtype(F){D,eltype(T)}(size(src[1]))
    return MLFMA.receivefields!(dest, src, mirrorindices, op)
end

# THE MATERIALIZED RECEIVE COLLECTION the lazy one stands in for, built from the operator's own
# `receivecouplingrule`. Ground truth for BOTH leaf multiplies, and genuinely independent of them: an
# ordinary collection driven through H2Factory's own protocol, with no mirroring at multiply time and
# no code shared with `applyleafmul!`.
#
# Derivation. The read side must satisfy
#     result[basis] = Σ_mappings coeff * Σ_s C[dest][s,basis] * ff[src][idx[s]]
# while a collection `mul!` computes `Σ_i Σ_t R[i][t,basis] * ff[i][t]`. Substituting `t = idx[s]`
# (legal: `idx` is a permutation AND an involution, so `s = idx[t]`) and matching component `i = src`:
#     R[src][t,basis] = coeff * C[dest][idx[t],basis].
# `src` indexes R while `dest` indexes C -- the same end-swap the write-through needs, and the reason
# this is derived rather than eyeballed: on every rule shipped today (each either diagonal or a
# symmetric two-way swap) the un-swapped version agrees, so a mistake would be invisible.
#
# The same R is the write-through's ground truth without further argument: `mul!(moment, R, y)` gives
# `moment[src][s] = coeff * Σ_basis C[dest][idx[s],basis] * y[basis]`, exactly what it computes.
function materializedreceive(operator, FarFieldType, radiation, mirrorindices)
    ncomponents = MLFMA.fielddim(FarFieldType)
    components = [Matrix{ComplexF64}(undef, size(radiation[1])) for _ in 1:ncomponents]
    for mapping in MLFMA.receivecouplingrule(operator, FarFieldType)
        dest = MLFMA.destcomponent(mapping)
        src = MLFMA.srccomponent(mapping)
        coeff = MLFMA.coefficient(mapping)
        for basis in axes(radiation[dest], 2), t in eachindex(mirrorindices)
            components[src][t, basis] = coeff * radiation[dest][mirrorindices[t], basis]
        end
    end
    return MLFMA.receivecollectiontype(FarFieldType)(
        SVector{ncomponents,Matrix{ComplexF64}}(components)
    )
end

# A rule with a DIFFERENT coefficient each way, unrelated to any real operator. Every shipped rule is
# either diagonal or a symmetric two-way swap, so none of them can tell a `src`/`dest` mix-up from the
# correct convention; this one can. At top level rather than inside a testset because two testsets
# below drive it.
struct AsymmetricRuleTestOperator end

function MLFMA.receivecouplingrule(
    ::AsymmetricRuleTestOperator, ::Type{<:MLFMA.TangentialSphericalFarField}
)
    return (MLFMA.ComponentMap{1,2}(2.0 + 1.0im), MLFMA.ComponentMap{2,1}(0.3 - 0.7im))
end

# `LazyReceiveFieldCollection` validates that its collection matches the operator's representation, so
# even a synthetic operator has to declare one.
MLFMA.farfieldtype(::AsymmetricRuleTestOperator) = MLFMA.TangentialSphericalFarField

@testset "Component map rule tables match current mirror/receivefields" begin
    sampling = MLFMA.SphericalSampling{Float64}(4, 6)
    mirrorindex = MLFMA.mirrorindices(sampling)
    sz = size(sampling)

    randomcomponents(dim) = [rand(ComplexF64, sz) for _ in 1:dim]
    # Passing `components` (a `Vector{Matrix{ComplexF64}}`) *unsplatted* is what makes
    # StaticArrays build a length-`dim` SVector-of-matrices; splatting collapses at `dim == 1`
    # into "convert this one matrix's own elements into an SVector{1}", which fails since the
    # matrix has more than one entry.
    tosvector(components) = SVector{length(components),Matrix{ComplexF64}}(components)

    function applyrule(rule, components)
        dim = length(components)
        out = [zeros(ComplexF64, sz) for _ in 1:dim]
        for m in rule
            d = MLFMA.destcomponent(m)
            s = MLFMA.srccomponent(m)
            c = MLFMA.coefficient(m)
            out[d] = c .* components[s][mirrorindex]
        end
        return out
    end

    @testset "mirrorrule vs MLFMA.mirror, D=1,2,3" begin
        for FarFieldType in (
            MLFMA.ScalarFarField,
            MLFMA.TangentialSphericalFarField,
            MLFMA.SphericalVectorFarField,
        )
            dim = MLFMA.fielddim(FarFieldType)
            components = randomcomponents(dim)
            ff = FarFieldType(tosvector(components))

            expected = MLFMA.mirror(ff, sampling)
            actual = applyrule(MLFMA.mirrorrule(FarFieldType), components)

            for i in 1:dim
                @test expected[i] ≈ actual[i]
            end
        end
    end

    @testset "mirrorrule D=6 doubles the D=3 sign pattern, and MLFMA.mirror now works at D=6" begin
        components3 = randomcomponents(3)
        ff3 = MLFMA.SphericalVectorFarField(tosvector(components3))
        halfexpected = MLFMA.mirror(ff3, sampling)

        components6 = vcat(components3, randomcomponents(3))
        doubled = applyrule(
            MLFMA.mirrorrule(MLFMA.PairedSphericalVectorFarField), components6
        )

        for i in 1:3
            @test doubled[i] ≈ halfexpected[i]
        end
        # second triple uses independent data, so it must NOT match the first triple's mirror --
        # a guard against the rule table accidentally reading from the wrong half.
        for i in 1:3
            @test !(doubled[3 + i] ≈ halfexpected[i])
        end

        # `MLFMA.mirror` itself, called directly at D=6 -- this representation had no `mirror`
        # method before phase 6, so this is genuine new-capability coverage, not a wiring check.
        #
        # Checked over ALL SIX components against the rule table, not just the first triple. With
        # only `1:3` verified, "the second triple is copied from the first" and "component 6 has
        # the wrong sign" would both pass: the first triple would still be right, and the negative
        # checks above constrain only `doubled` (the rule table's own output), never `mirror`'s.
        # All six arrays below are independent, so a cross-triple copy shows up as a value
        # mismatch instead of being masked by shared data.
        allsixcomponents = randomcomponents(6)
        ffsix = MLFMA.PairedSphericalVectorFarField(tosvector(allsixcomponents))
        mirroredsix = MLFMA.mirror(ffsix, sampling)
        expectedsix = applyrule(
            MLFMA.mirrorrule(MLFMA.PairedSphericalVectorFarField), allsixcomponents
        )
        for i in 1:6
            @test mirroredsix[i] ≈ expectedsix[i]
        end

        # The two triples stay independent: with distinct inputs, no component of the second may
        # coincide with the matching component of the first. Pins the "second triple copied from
        # the first" failure mode directly rather than by implication.
        for i in 1:3
            @test !(mirroredsix[3 + i] ≈ mirroredsix[i])
        end

        # And the sign pattern is genuinely per-triple (+,+,-) rather than uniform: each triple's
        # radial component comes back unflipped while its φ component flips.
        for offset in (0, 3)
            @test mirroredsix[offset + 1] ≈ allsixcomponents[offset + 1][mirrorindex]
            @test mirroredsix[offset + 3] ≈ -allsixcomponents[offset + 3][mirrorindex]
        end
    end

    @testset "receivecouplingrule vs MLFMA.receivefields, every currently-supported operator" begin
        efie = MLFMA.efie(; wavenumber=1.3)
        doublelayer = Maxwell3D.doublelayer(; wavenumber=1.3)
        singlelayer = MLFMA.singlelayer(efie)
        hypersingular = MLFMA.hypersingular(; wavenumber=1.3)
        hh_single = Helmholtz3D.singlelayer(; wavenumber=1.3)
        hh_hyper = Helmholtz3D.hypersingular(; wavenumber=1.3)

        cases = (
            (efie, MLFMA.TangentialSphericalFarField),
            (doublelayer, MLFMA.TangentialSphericalFarField),
            (singlelayer, MLFMA.SphericalVectorFarField),
            (hypersingular, MLFMA.ScalarFarField),
            (hh_single, MLFMA.ScalarFarField),
            (hh_hyper, MLFMA.PairedSphericalVectorFarField),
        )

        for (op, FarFieldType) in cases
            dim = MLFMA.fielddim(FarFieldType)
            components = randomcomponents(dim)
            # `receivefields` operates on a far-field MOMENT (what `mul!` is called with), not
            # on a radiation collection, despite the historical parameter name.
            ff = FarFieldType(tosvector(components))

            expected = receivefieldsreference(ff, mirrorindex, op)
            actual = applyrule(MLFMA.receivecouplingrule(op, FarFieldType), components)
            actual_generic = applyrule(MLFMA.receivecouplingrule(op), components)

            for i in 1:dim
                @test expected[i] ≈ actual[i]
                @test expected[i] ≈ actual_generic[i]
            end
        end
    end

    @testset "transform! matches mirror/receivefields, and allocates nothing once warm" begin
        for FarFieldType in (
            MLFMA.ScalarFarField,
            MLFMA.TangentialSphericalFarField,
            MLFMA.SphericalVectorFarField,
            # Paired belongs here too: it is the widest rule table (6 mappings), so it is the case
            # most likely to allocate if the tuple-peeling recursion ever regresses to generic
            # iteration.
            MLFMA.PairedSphericalVectorFarField,
        )
            dim = MLFMA.fielddim(FarFieldType)
            components = randomcomponents(dim)
            src = FarFieldType(tosvector(components))
            expected = MLFMA.mirror(src, sampling)
            rule = MLFMA.mirrorrule(FarFieldType)

            dest = FarFieldType(1.0 + 0.0im, sampling, Val(:zero))
            MLFMA.transform!(dest, src, mirrorindex, rule)
            for i in 1:dim
                @test dest[i] ≈ expected[i]
            end

            @test warmedallocations(MLFMA.transform!, dest, src, mirrorindex, rule) == 0
        end

        op = MLFMA.efie(; wavenumber=1.3)
        components = randomcomponents(2)
        src = MLFMA.TangentialSphericalFarField(tosvector(components))
        expected = receivefieldsreference(src, mirrorindex, op)
        rule = MLFMA.receivecouplingrule(op, MLFMA.TangentialSphericalFarField)

        dest = MLFMA.TangentialSphericalFarField(1.0 + 0.0im, sampling, Val(:zero))
        MLFMA.transform!(dest, src, mirrorindex, rule)
        for i in 1:2
            @test dest[i] ≈ expected[i]
        end

        @test warmedallocations(MLFMA.transform!, dest, src, mirrorindex, rule) == 0
    end
end

# THE LEAF MULTIPLY, both directions, against an independently materialized `ReceiveFieldCollection`.
#
# `LazyReceiveFieldCollection` implements exactly two operations -- the two `applyleafmul!` arities
# the traversal's leaf multiply calls -- so these are the whole of its behaviour. Ground truth is
# `materializedreceive` (see the header): the collection the lazy object stands in for, built once
# from the operator's own `receivecouplingrule` and then driven through H2Factory's ordinary
# collection protocol, sharing no code with the thing under test.
#
# Tiny deterministic synthetic matrices, not BEAST assembly: every expected value is exact, so a
# component swap or sign slip fails immediately rather than hiding under a tolerance. Real
# BEAST-backed correctness for both directions lives in `mirrored.jl`'s "LazyReceiveFieldCollection
# Maxwell/Helmholtz" testsets.
@testset "Leaf multiply matches an explicitly materialized ReceiveFieldCollection" begin
    sampling = MLFMA.SphericalSampling{Float64}(4, 6)
    mirrorindex = MLFMA.mirrorindices(sampling)
    sz = size(sampling)
    nsamples = length(sampling)
    nbasis = 5

    efie = MLFMA.efie(; wavenumber=1.3)
    cases = (
        (efie, MLFMA.TangentialSphericalFarField),
        (Maxwell3D.doublelayer(; wavenumber=1.3), MLFMA.TangentialSphericalFarField),
        (MLFMA.singlelayer(efie), MLFMA.SphericalVectorFarField),
        (MLFMA.hypersingular(; wavenumber=1.3), MLFMA.ScalarFarField),
        (Helmholtz3D.singlelayer(; wavenumber=1.3), MLFMA.ScalarFarField),
        (Helmholtz3D.hypersingular(; wavenumber=1.3), MLFMA.PairedSphericalVectorFarField),
    )

    # Short adapters for the numerical comparisons below.
    runread!(scratch, result, lazy, ff, a, b) =
        H2Factory.applyleafmul!(scratch, result, lazy, ff, a, b)
    runwrite!(scratch, moment, lazy, y) = H2Factory.applyleafmul!(scratch, moment, lazy, y)

    for (op, FarFieldType) in cases
        dim = MLFMA.fielddim(FarFieldType)
        ff = FarFieldType(
            SVector{dim,Matrix{ComplexF64}}([rand(ComplexF64, sz) for _ in 1:dim])
        )
        radiationcol = MLFMA.radiationcollectiontype(FarFieldType)(
            SVector{dim,Matrix{ComplexF64}}([
                rand(ComplexF64, nsamples, nbasis) for _ in 1:dim
            ]),
        )
        lazy = MLFMA.LazyReceiveFieldCollection(op, radiationcol, mirrorindex)
        receivecol = materializedreceive(op, FarFieldType, radiationcol, mirrorindex)

        # THE SCRATCH a workspace's leaf channel holds, built exactly as `MLFMA.leafscratchchannel`
        # builds it. It must be the SAME concrete type as the incoming moment: `receivefields!`
        # constrains both sides to one `F`, which is what stops one representation being written into
        # another, and the production path relies on a pooled moment and a fresh scratch landing on
        # the identical `{D,T,P,V}`.
        scratch = FarFieldType(ComplexF64, sampling, Val(:zero))
        @test typeof(scratch) === typeof(ff)

        α, β = 2.0 + 0.5im, 3.0 - 1.0im
        start = rand(ComplexF64, nbasis)

        expected = copy(start)
        LinearAlgebra.mul!(expected, receivecol, ff, α, β)
        actual = copy(start)
        runread!(scratch, actual, lazy, ff, α, β)
        @test actual ≈ expected

        # ...and the plain (α = true, β = false) form the traversal uses for a scalar-α operator.
        expectedplain = zeros(ComplexF64, nbasis)
        LinearAlgebra.mul!(expectedplain, receivecol, ff)
        actualplain = zeros(ComplexF64, nbasis)
        runread!(scratch, actualplain, lazy, ff, true, false)
        @test actualplain ≈ expectedplain

        # PER-COMPONENT α, the path leaf disaggregation takes for a vector-α operator. The scratch
        # path has no weighted kernel of its own -- it forwards to H2Factory's own per-component
        # collection `mul!` -- so this checks that the weight lands on the component it should.
        weights = MLFMA.α(op)
        if weights isa AbstractVector
            @test length(weights) == dim
            expectedv = copy(start)
            LinearAlgebra.mul!(expectedv, receivecol, ff, weights, β)
            actualv = copy(start)
            runread!(scratch, actualv, lazy, ff, weights, β)
            @test actualv ≈ expectedv

            @test warmedallocations(
                H2Factory.applyleafmul!, scratch, actualv, lazy, ff, weights, β
            ) == 0
        end

        # A vector `β` is refused rather than silently reinterpreted: H2Factory defines result-side
        # vector-`β` only for single-component collections, so there is no general-`D` contract to
        # mirror. Skipped at `dim == 1`, where that overload genuinely exists.
        if dim > 1
            @test_throws MethodError runread!(
                scratch, copy(start), lazy, ff, true, [false for _ in 1:dim]
            )
        end

        # WRITE-THROUGH, against the SAME materialized collection: `mul!(moment, R, y)` gives
        # `moment[src][s] = coeff * Σ_basis C[dest][idx[s],basis] * y[basis]`, which is what the
        # write-through computes. The component roles are swapped relative to the read side, and
        # getting that wrong is invisible on every rule shipped today -- hence the asymmetric-rule
        # testset below.
        y = rand(ComplexF64, nbasis)
        expectedmoment = FarFieldType(ComplexF64, sz)
        LinearAlgebra.mul!(expectedmoment, receivecol, y)
        actualmoment = FarFieldType(ComplexF64, sz)
        runwrite!(scratch, actualmoment, lazy, y)
        for component in 1:dim
            @test actualmoment[component] ≈ expectedmoment[component]
        end

        # The whole point of the leaf channel: with the buffer supplied, the materialization costs no
        # allocation, so the BLAS path is strictly better than the fused kernel it replaced.
        @test warmedallocations(H2Factory.applyleafmul!, scratch, actual, lazy, ff, α, β) ==
            0
        @test warmedallocations(H2Factory.applyleafmul!, scratch, actualmoment, lazy, y) ==
            0
    end
end

# The read/write pair must be genuine transposes of each other. Driven through a rule with a DIFFERENT
# coefficient each way (`AsymmetricRuleTestOperator`, defined at the top of this file), because every
# rule in the package is either diagonal or a symmetric two-way swap and therefore cannot distinguish
# the correct `src`/`dest` convention from the swapped one.
@testset "Write-through is the genuine transpose of the read side (asymmetric rule)" begin
    sampling = MLFMA.SphericalSampling{Float64}(4, 6)
    mirrorindex = MLFMA.mirrorindices(sampling)
    sz = size(sampling)
    nsamples = length(sampling)
    nbasis = 5

    op = AsymmetricRuleTestOperator()
    radiationcol = MLFMA.TangentialSphericalRadiationFieldCollection(
        SVector{2,Matrix{ComplexF64}}([rand(ComplexF64, nsamples, nbasis) for _ in 1:2])
    )
    lazy = MLFMA.LazyReceiveFieldCollection(op, radiationcol, mirrorindex)
    farfield = MLFMA.TangentialSphericalFarField(
        SVector{2,Matrix{ComplexF64}}([rand(ComplexF64, sz) for _ in 1:2])
    )
    y = rand(ComplexF64, nbasis)

    readscratch = MLFMA.TangentialSphericalFarField(ComplexF64, sampling, Val(:zero))
    writescratch = MLFMA.TangentialSphericalFarField(ComplexF64, sampling, Val(:zero))

    result = zeros(ComplexF64, nbasis)
    H2Factory.applyleafmul!(readscratch, result, lazy, farfield, true, false)

    moment = MLFMA.TangentialSphericalFarField(ComplexF64, sz)
    H2Factory.applyleafmul!(writescratch, moment, lazy, y)

    # `<Ax, y> == <x, transpose(A) y>`, the defining property, checked directly rather than by
    # comparing the two implementations against each other -- so this still stands if both change.
    lhs = sum(result .* y)
    rhs = sum(farfield[1] .* moment[1]) + sum(farfield[2] .* moment[2])
    @test lhs ≈ rhs

    # Non-vacuity: the swapped (buggy) convention must NOT also satisfy the identity, or the test is
    # not discriminating between the two conventions at all.
    swapped = MLFMA.TangentialSphericalFarField(ComplexF64, sz)
    scatterscratch = MLFMA.TangentialSphericalFarField(ComplexF64, sampling, Val(:zero))
    LinearAlgebra.mul!(scatterscratch, radiationcol, y)
    for mapping in MLFMA.receivecouplingrule(op, MLFMA.TangentialSphericalFarField)
        c = MLFMA.coefficient(mapping)
        destarray = swapped[MLFMA.destcomponent(mapping)]  # bug: should be srccomponent
        S = scatterscratch[MLFMA.srccomponent(mapping)]    # bug: should be destcomponent
        for sampleindex in eachindex(mirrorindex)
            destarray[sampleindex] = c * S[mirrorindex[sampleindex]]
        end
    end
    swappedrhs = sum(farfield[1] .* swapped[1]) + sum(farfield[2] .* swapped[2])
    @test !(lhs ≈ swappedrhs)
end

# THE CONTRACT, stated as failures. Two different things are easy to confuse:
#
#   * `farfieldtype(op)` means the operator HAS a far-field representation. Every operator here does.
#   * `receivecouplingrule(op, F)` means lazy receive is SUPPORTED. These three have no rule.
#
# So construction succeeds -- the collection matches the operator's representation, which is all the
# constructor validates -- and the failure lands on the first multiply. Pinned so that (a) nobody
# reads "has farfieldtype" as "lazy receive works", and (b) if someone adds a rule for one of these,
# this testset fails and forces the addition to be deliberate and verified rather than silently
# turning a `MethodError` into numbers nothing checks.
@testset "Lazy receive is unsupported, loudly, for operators with no coupling rule" begin
    sampling = MLFMA.SphericalSampling{Float64}(4, 6)
    nsamples = length(sampling)
    nbasis = 3

    unsupported = (
        BEAST.n × Maxwell3D.doublelayer(; wavenumber=1.3),
        Helmholtz3D.doublelayer(; wavenumber=1.3),
        Helmholtz3D.doublelayer_transposed(; wavenumber=1.3),
    )

    for op in unsupported
        FarFieldType = MLFMA.farfieldtype(op)
        dim = MLFMA.fielddim(FarFieldType)

        # Representation side works: that is exactly why the gap is easy to miss.
        @test MLFMA.fielddim(MLFMA.radiationfieldcollectiontype(op)) == dim
        @test !hasmethod(MLFMA.receivecouplingrule, Tuple{typeof(op),Type{FarFieldType}})

        radiationcol = MLFMA.radiationcollectiontype(FarFieldType)(
            SVector{dim,Matrix{ComplexF64}}([
                rand(ComplexF64, nsamples, nbasis) for _ in 1:dim
            ]),
        )
        lazy = MLFMA.LazyReceiveFieldCollection(
            op, radiationcol, MLFMA.mirrorindices(sampling)
        )
        @test lazy isa MLFMA.LazyReceiveFieldCollection

        scratch = FarFieldType(ComplexF64, sampling, Val(:zero))
        farfield = FarFieldType(ComplexF64, sampling, Val(:zero))
        moment = FarFieldType(ComplexF64, sampling, Val(:zero))
        result = zeros(ComplexF64, nbasis)
        y = rand(ComplexF64, nbasis)

        # Both leaf multiplies fail, and fail as a missing METHOD (the absent rule) rather than as an
        # arithmetic or bounds error somewhere inside a kernel.
        @test_throws MethodError H2Factory.applyleafmul!(
            scratch, result, lazy, farfield, true, false
        )
        @test_throws MethodError H2Factory.applyleafmul!(scratch, moment, lazy, y)
        # ...and so does the in-place kernel underneath them.
        @test_throws MethodError MLFMA.receivefields!(
            scratch, farfield, MLFMA.mirrorindices(sampling), op
        )
    end

    # Non-vacuity: the same construction and calls DO work for an operator that has a rule, so the
    # failures above are about the missing rule and not about the fixture being malformed.
    efie = MLFMA.efie(; wavenumber=1.3)
    supportedtype = MLFMA.farfieldtype(efie)
    supportedcol = MLFMA.radiationcollectiontype(supportedtype)(
        SVector{2,Matrix{ComplexF64}}([rand(ComplexF64, nsamples, nbasis) for _ in 1:2])
    )
    supportedlazy = MLFMA.LazyReceiveFieldCollection(
        efie, supportedcol, MLFMA.mirrorindices(sampling)
    )
    supportedscratch = supportedtype(ComplexF64, sampling, Val(:zero))
    supportedff = supportedtype(ComplexF64, sampling, Val(:zero))
    supportedresult = zeros(ComplexF64, nbasis)
    @test H2Factory.applyleafmul!(
        supportedscratch, supportedresult, supportedlazy, supportedff, true, false
    ) === supportedresult
end

# NO LEAF SCRATCH, NO MULTIPLY -- the cost of narrowing this to a scratch-only collection, pinned so
# the failure mode is a decision rather than a surprise.
#
# `H2Factory.buildleafchannels` has a DEFAULT (`EmptyChannels()`), and `borrowbuffer` answers
# `EmptyChannels()` for any level with no pool. Before the narrowing that combination degraded
# gracefully to the fused kernel; now it cannot run at all, so it must at least say why. A bare
# `MethodError` from the generic `applyleafmul!` falling through to `LinearAlgebra.mul!` would name
# none of the three things that actually went wrong.
@testset "A lazy receive collection with no leaf scratch fails with a named error" begin
    sampling = MLFMA.SphericalSampling{Float64}(4, 6)
    nbasis = 3
    efie = MLFMA.efie(; wavenumber=1.3)
    FarFieldType = MLFMA.farfieldtype(efie)

    radiationcol = MLFMA.radiationcollectiontype(FarFieldType)(
        SVector{2,Matrix{ComplexF64}}([
            rand(ComplexF64, length(sampling), nbasis) for _ in 1:2
        ]),
    )
    lazy = MLFMA.LazyReceiveFieldCollection(
        efie, radiationcol, MLFMA.mirrorindices(sampling)
    )
    farfield = FarFieldType(ComplexF64, sampling, Val(:zero))
    moment = FarFieldType(ComplexF64, sampling, Val(:zero))
    result = zeros(ComplexF64, nbasis)
    y = rand(ComplexF64, nbasis)
    empty = H2Factory.EmptyChannels()

    for call in (
        () -> H2Factory.applyleafmul!(empty, result, lazy, farfield, true, false),
        () -> H2Factory.applyleafmul!(empty, moment, lazy, y),
    )
        err = try
            call()
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        # The message has to name the hook to implement, or it is not doing its job.
        @test occursin("buildleafchannels", err.msg)
        @test occursin("leaf scratch", err.msg)
    end
end
