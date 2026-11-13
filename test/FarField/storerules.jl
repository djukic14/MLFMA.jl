using MLFMA
using BEAST, CompScienceMeshes
using H2Factory
using SphericalScattering
using StaticArrays
using Test
using LinearAlgebra

include("../support/util_test.jl")

# THE STORE RULE IS A SEPARATE AXIS FROM THE STORED TYPE.
#
# `farfieldtype(op)` says what layout a leaf moment has; `farfieldstorerule(op)` says how BEAST's raw
# far-field value is projected into it. They used to be one table (`H2Factory.storetrait` returning a
# storer type), which could only ever encode the PAIR.
#
# Every testset below is built around the combinations where the two axes visibly come apart, because
# those are the ones a single table could not express and the ones a future edit is most likely to
# re-conflate.

@testset "farfieldtype and farfieldstorerule are independent axes" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)

    efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
    mwsinglelayer = Maxwell3D.singlelayer(; wavenumber=κ)
    mwdoublelayer = Maxwell3D.doublelayer(; wavenumber=κ)
    mwrotateddoublelayer = BEAST.n × Maxwell3D.doublelayer(; wavenumber=κ)
    mwhypersingular = MLFMA.MWHyperSingular3D(; wavenumber=κ)
    hhsinglelayer = Helmholtz3D.singlelayer(; wavenumber=κ)
    hhdoublelayer = Helmholtz3D.doublelayer(; wavenumber=κ)
    hhhypersingular = Helmholtz3D.hypersingular(; wavenumber=κ)

    # The four combinations the split exists for, pinned as (operator, type, rule) triples rather
    # than as two separate lists -- a list per axis would pass just as happily if a future edit
    # re-derived one from the other.
    for (op, F, rule) in (
        (
            efie,
            MLFMA.TangentialSphericalFarField,
            MLFMA.StoreCartesianAsTangentialSpherical(),
        ),
        (
            mwdoublelayer,
            MLFMA.TangentialSphericalFarField,
            MLFMA.StoreCartesianAsTangentialSpherical(),
        ),
        (
            mwrotateddoublelayer,
            MLFMA.TangentialSphericalFarField,
            MLFMA.StoreCartesianAsTangentialSpherical(),
        ),
        (
            mwsinglelayer,
            MLFMA.SphericalVectorFarField,
            MLFMA.StoreCartesianAsSphericalVector(),
        ),
        (mwhypersingular, MLFMA.ScalarFarField, MLFMA.StoreCartesianAsRadialScalar()),
        (hhsinglelayer, MLFMA.ScalarFarField, MLFMA.StoreAsFarFieldValue()),
        (hhdoublelayer, MLFMA.ScalarFarField, MLFMA.StoreAsFarFieldValue()),
        (
            hhhypersingular,
            MLFMA.PairedSphericalVectorFarField,
            MLFMA.StoreDoubleCartesianAsPairedSpherical(),
        ),
    )
        @test MLFMA.farfieldtype(op) === F
        @test MLFMA.farfieldstorerule(op) === rule
        @test MLFMA.farfieldstorerule(op) isa MLFMA.AbstractFarFieldStoreRule
    end

    # THE WITNESS, stated as its own assertion because it is the whole argument for the split:
    # `MWHyperSingular3D` and the Helmholtz scalar operators store the SAME representation by
    # DIFFERENT routes. Any design that derives the rule from the type gets one of these wrong.
    @test MLFMA.farfieldtype(mwhypersingular) === MLFMA.farfieldtype(hhsinglelayer)
    @test MLFMA.farfieldstorerule(mwhypersingular) !==
        MLFMA.farfieldstorerule(hhsinglelayer)

    # The other direction, so the pair above is not read as "the two tables never agree": operators
    # that genuinely share BOTH still share both. Sharing a type does not imply sharing a rule
    # (above); sharing a rule here does come with a shared type.
    @test MLFMA.farfieldtype(hhsinglelayer) === MLFMA.farfieldtype(hhdoublelayer)
    @test MLFMA.farfieldstorerule(hhsinglelayer) === MLFMA.farfieldstorerule(hhdoublelayer)

    # The rule is a VALUE, and rules are singletons -- so an assembler can store one in a concrete
    # field and dispatch on it with nothing to infer.
    @test isbitstype(typeof(MLFMA.farfieldstorerule(efie)))
    @test MLFMA.farfieldstorerule(efie) === MLFMA.StoreCartesianAsTangentialSpherical()

    # BOTH CALL FORMS, for every operator. The table this replaced mixed `Type{}` and instance arms
    # in one `Union`, so the instance form resolved for some operators via the arm and for others via
    # the `typeof` forwarder -- and the documented TYPE form recursed into that forwarder and died
    # with a `StackOverflowError` for the instance-arm operators. Asking both ways of every operator
    # is what makes that class of mistake impossible to reintroduce quietly.
    for op in (
        efie,
        mwdoublelayer,
        mwrotateddoublelayer,
        mwsinglelayer,
        mwhypersingular,
        hhsinglelayer,
        hhdoublelayer,
        hhhypersingular,
    )
        @test MLFMA.farfieldstorerule(op) === MLFMA.farfieldstorerule(typeof(op))
    end
end

@testset "Only projecting rules build a frame cache" begin
    sampling = MLFMA.SphericalSampling{Float64}(4, 6)

    # `StoreAsFarFieldValue` does no projection, so it must not pay for trig tables. On the tree path
    # this is four arrays PER LEAF, so the difference is not cosmetic for a scalar Helmholtz problem.
    @test MLFMA.framecachefor(MLFMA.StoreAsFarFieldValue(), sampling) === nothing

    for rule in (
        MLFMA.StoreCartesianAsSphericalVector(),
        MLFMA.StoreCartesianAsTangentialSpherical(),
        MLFMA.StoreCartesianAsRadialScalar(),
        MLFMA.StoreDoubleCartesianAsPairedSpherical(),
    )
        cache = MLFMA.framecachefor(rule, sampling)
        @test cache isa MLFMA.SphericalFrameCache
        @test cache.sampling === sampling
        @test length(cache.sinθs) == length(MLFMA.θs(sampling))
        @test length(cache.sinφs) == length(MLFMA.φs(sampling))
        # the cache really holds sin/cos of the sampling's own angles, not of some default grid
        @test cache.sinθs ≈ sin.(MLFMA.θs(sampling))
        @test cache.cosφs ≈ cos.(MLFMA.φs(sampling))
    end
end

@testset "Per-leaf frame caches are shared, not duplicated per leaf" begin
    coarse = MLFMA.SphericalSampling{Float64}(4, 6)
    fine = MLFMA.SphericalSampling{Float64}(6, 8)
    points(s) = MLFMA.cartesiansamplepoints(s)[1]
    centre = SVector(0.0, 0.0, 0.0)

    # Three leaves on one level plus two on another -- the non-uniform case, where sharing has to be
    # per SAMPLING rather than global. `leaffarfieldsamplings` hands every leaf at a level the same
    # sampling OBJECT, which is what makes identity the right key.
    leafsamplings = [
        MLFMA.BoxFarFieldSampling(coarse, points(coarse), centre, 11),
        MLFMA.BoxFarFieldSampling(coarse, points(coarse), centre, 12),
        MLFMA.BoxFarFieldSampling(fine, points(fine), centre, 13),
        MLFMA.BoxFarFieldSampling(coarse, points(coarse), centre, 14),
        MLFMA.BoxFarFieldSampling(fine, points(fine), centre, 15),
    ]

    caches = MLFMA.framecachefor(MLFMA.StoreCartesianAsRadialScalar(), leafsamplings)

    # one entry per leaf, as the cluster-indexed borrow path requires...
    @test length(caches) == length(leafsamplings)
    # ...but only as many distinct objects as there are distinct samplings
    @test length(unique(objectid, caches)) == 2
    @test caches[1] === caches[2] === caches[4]
    @test caches[3] === caches[5]
    @test caches[1] !== caches[3]

    # each leaf still gets the cache for ITS OWN sampling -- sharing must not mean collapsing the
    # two levels onto one grid
    @test caches[1].sampling === coarse
    @test caches[3].sampling === fine

    # CONCRETE element type: `storer.framecache[cluster]` runs per stored value, so an abstract
    # eltype here would cost a dynamic dispatch on the hot path.
    @test isconcretetype(eltype(caches))

    # A non-projecting rule builds nothing at all on the tree path either. This pair of arguments is
    # exactly the combination that used to raise an ambiguity `MethodError`.
    @test MLFMA.framecachefor(MLFMA.StoreAsFarFieldValue(), leafsamplings) === nothing
    @test MLFMA.needsframecache(MLFMA.StoreAsFarFieldValue()) == false
    @test MLFMA.needsframecache(MLFMA.StoreCartesianAsRadialScalar()) == true
end

# Each rule against an INDEPENDENT reference: BEAST's own far field, converted with
# `SphericalScattering`'s Cartesian-to-spherical helper rather than with anything in `store.jl`. That
# independence is the point -- comparing the storer against a hand-rolled copy of its own formula
# would pass even if both were wrong in the same way.
@testset "Each store rule projects the way its name says" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    m = meshsphere(1 * λ, λ / 5)
    sampling = MLFMA.SphericalSampling{Float64}(6, 8)
    points, _ = MLFMA.cartesiansamplepoints(sampling)
    quad = BEAST.SingleNumQStrat(3)

    build(op, space) = MLFMA.buildfieldcollection(
        radiationbuilder(; quadstrategy=quad), op, sampling, space
    )

    X = raviartthomas(m)
    Y = lagrangec0d1(m)

    sphericalof(op, space, u, ::Type{T}=SVector{3,ComplexF64}) where {T} =
        SphericalScattering.convertCartesian2Spherical.(
            BEAST.potential(
                MLFMA.radiationoperator(op), points, u, space; type=T, quadstrat=quad
            ),
            SphericalScattering.cart2sph.(points),
        )

    unit(space, i) = (u=zeros(numfunctions(space)); u[i]=1.0; u)

    @testset "Cartesian to tangential spherical (MWEFIE3D)" begin
        efie = MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ))
        collection = build(efie, X)
        u = unit(X, 1)
        refθ, refφ = MLFMA._removeradialcomponent(sphericalof(efie, X, u))

        @test vec(refθ) ≈ MLFMA.Fθ(collection)[:, 1]
        @test vec(refφ) ≈ MLFMA.Fφ(collection)[:, 1]
        # the radial component is DISCARDED, which is what distinguishes this rule from
        # `StoreCartesianAsSphericalVector` -- so the collection has exactly two components
        @test MLFMA.fielddim(MLFMA.farfieldtype(efie)) == 2
    end

    @testset "Cartesian to spherical vector (MWSingleLayer3D)" begin
        mwsinglelayer = Maxwell3D.singlelayer(; wavenumber=κ)
        collection = build(mwsinglelayer, X)
        u = unit(X, 1)
        spherical = sphericalof(mwsinglelayer, X, u)

        @test vec(getindex.(spherical, 1)) ≈ MLFMA.Fr(collection)[:, 1]
        @test vec(getindex.(spherical, 2)) ≈ MLFMA.Fθ(collection)[:, 1]
        @test vec(getindex.(spherical, 3)) ≈ MLFMA.Fφ(collection)[:, 1]
        # non-vacuity for "keeps the radial part": it is not ~0, so the previous testset's claim
        # that the tangential rule DROPS something real is meaningful
        @test maximum(abs, MLFMA.Fr(collection)) > 1e-8
    end

    @testset "Cartesian to radial scalar (MWHyperSingular3D)" begin
        mwhypersingular = MLFMA.MWHyperSingular3D(; wavenumber=κ)
        collection = build(mwhypersingular, X)
        u = unit(X, 1)
        spherical = sphericalof(mwhypersingular, X, u)

        # ONLY the radial component survives, into a one-component ScalarFarField.
        @test MLFMA.fielddim(MLFMA.farfieldtype(mwhypersingular)) == 1
        @test vec(getindex.(spherical, 1)) ≈ collection[1][:, 1]
    end

    @testset "Store as is (Helmholtz scalar)" begin
        hhsinglelayer = Helmholtz3D.singlelayer(; wavenumber=κ)
        collection = build(hhsinglelayer, Y)
        u = unit(Y, 1)
        reference = BEAST.potential(
            MLFMA.radiationoperator(hhsinglelayer),
            points,
            u,
            Y;
            type=ComplexF64,
            quadstrat=quad,
        )

        # No projection at all: BEAST's value IS the stored moment. Note the reference here needs no
        # spherical conversion, unlike every other rule -- that asymmetry is the rule.
        @test vec(reference) ≈ collection[1][:, 1]
    end

    @testset "Double Cartesian to paired spherical (HH3DHyperSingularFDBIO)" begin
        hhhypersingular = Helmholtz3D.hypersingular(; wavenumber=κ)
        collection = build(hhhypersingular, Y)
        u = unit(Y, 1)

        @test MLFMA.fielddim(MLFMA.farfieldtype(hhhypersingular)) == 6

        # The two halves come from DIFFERENT slices of the same 6-element value. This is what the
        # offset indexing that replaced `view(v, 1:3)` / `view(v, 4:6)` has to get right, and an
        # off-by-one in the offset would make the halves equal or shift components between them.
        @test collection[1][:, 1] != collection[4][:, 1]
        @test collection[2][:, 1] != collection[5][:, 1]
        @test collection[3][:, 1] != collection[6][:, 1]

        # ...and each half is a genuine spherical projection, checked against BEAST for the SECOND
        # half -- the one reached through the non-zero offset, so this pins the offset itself and not
        # just that two different numbers came out.
        paired = BEAST.potential(
            MLFMA.radiationoperator(hhhypersingular),
            points,
            u,
            Y;
            type=SVector{6,ComplexF64},
            quadstrat=quad,
        )
        secondhalf = [SVector{3}(p[4], p[5], p[6]) for p in paired]
        spherical = SphericalScattering.convertCartesian2Spherical.(
            secondhalf, SphericalScattering.cart2sph.(points)
        )
        @test vec(getindex.(spherical, 1)) ≈ collection[4][:, 1]
        @test vec(getindex.(spherical, 2)) ≈ collection[5][:, 1]
        @test vec(getindex.(spherical, 3)) ≈ collection[6][:, 1]
    end
end

# REGRESSION PIN for a bug the collapse to one storer exposed and fixed.
#
# The generated storers each defined `(storer::S{F,Nothing,TP,S})(v, cluster::Int, ...)` alongside a
# per-type `(storer::S)(v, cluster, ...)`. Where that second method annotated `cluster::Int` -- which
# only `CartesianToSphericalRemoveRadial` did -- the two were mutually ambiguous, so the FLAT
# (non-tree) assembler path raised `MethodError: ... is ambiguous` for that one storer. It looked
# like a quirk of one type; it was a specificity mistake that the other three merely hid by leaving
# `cluster` untyped.
#
# `FarFieldStorer` has exactly one such pair, so the mistake became unconditional and was found. The
# fix is restating `R<:AbstractFarFieldStoreRule` on the flat method (see `store.jl`).
@testset "The flat assembler path works for every rule" begin
    ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ = physicalconstants(5e7)
    m = meshsphere(1 * λ, λ / 4)
    sampling = MLFMA.SphericalSampling{Float64}(4, 6)
    quad = BEAST.SingleNumQStrat(1)

    X = raviartthomas(m)
    Y = lagrangec0d1(m)

    for (op, space) in (
        (MLFMA.efie(; wavenumber=κ, alpha=(-im * ω * μ)), X),   # the rule that used to be ambiguous
        (Maxwell3D.singlelayer(; wavenumber=κ), X),
        (MLFMA.MWHyperSingular3D(; wavenumber=κ), X),
        (Helmholtz3D.singlelayer(; wavenumber=κ), Y),
        (Helmholtz3D.hypersingular(; wavenumber=κ), Y),
    )
        collection = MLFMA.buildfieldcollection(
            radiationbuilder(; quadstrategy=quad), op, sampling, space
        )
        @test collection isa MLFMA.radiationcollectiontype(MLFMA.farfieldtype(op))
        # non-vacuity: it produced real numbers, not an empty or all-zero block
        @test maximum(abs, collection[1]) > 0
    end
end
