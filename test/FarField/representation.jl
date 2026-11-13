# Representation is carried by the concrete far-field/collection TYPE, not inferred from the
# component count. These tests pin the two consequences that were unreachable under the previous
# count-keyed scheme:
#
#   1. `SphericalVectorFarField` and `CartesianVectorFarField` both have three components but are
#      different, non-comparable types, each with its own accessors.
#   2. Asking a representation for a component it does not have is a `MethodError`, not a silent
#      read of whatever index happens to be there.
#
# The negative cases matter as much as the positive ones here: under a count-keyed scheme `Fx` on
# a spherical far field would have happily returned component 1 (the radial part), which is a
# wrong answer rather than an error.
@testset "Representation-specific accessors" begin
    sampling = MLFMA.SphericalSampling{Float64}(4, 6)

    # Distinct per-component values so an accessor reading the wrong index fails on the value,
    # not only on the type.
    function labelled(FarFieldType)
        ff = FarFieldType(ComplexF64, sampling, Val(:zero))
        for i in eachindex(ff)
            ff[i] .= i
        end
        return ff
    end

    scalar = labelled(MLFMA.ScalarFarField)
    tangential = labelled(MLFMA.TangentialSphericalFarField)
    spherical = labelled(MLFMA.SphericalVectorFarField)
    cartesian = labelled(MLFMA.CartesianVectorFarField)
    paired = labelled(MLFMA.PairedSphericalVectorFarField)

    @testset "each representation's own accessors read the right component" begin
        @test MLFMA.F(scalar) === scalar[1]

        @test MLFMA.Fθ(tangential) === tangential[1]
        @test MLFMA.Fφ(tangential) === tangential[2]

        # Note the offset relative to the tangential case: r,θ,φ puts θ at 2, not 1. Reading
        # these by count alone is exactly the ambiguity the type now resolves.
        @test MLFMA.Fr(spherical) === spherical[1]
        @test MLFMA.Fθ(spherical) === spherical[2]
        @test MLFMA.Fφ(spherical) === spherical[3]

        @test MLFMA.Fx(cartesian) === cartesian[1]
        @test MLFMA.Fy(cartesian) === cartesian[2]
        @test MLFMA.Fz(cartesian) === cartesian[3]
    end

    @testset "accessors a representation does not have are MethodErrors" begin
        # Same component count as `spherical`, opposite accessor set -- neither answers the
        # other's questions.
        @test_throws MethodError MLFMA.Fx(spherical)
        @test_throws MethodError MLFMA.Fy(spherical)
        @test_throws MethodError MLFMA.Fz(spherical)

        @test_throws MethodError MLFMA.Fr(cartesian)
        @test_throws MethodError MLFMA.Fθ(cartesian)
        @test_throws MethodError MLFMA.Fφ(cartesian)

        @test_throws MethodError MLFMA.Fθ(scalar)
        @test_throws MethodError MLFMA.Fφ(scalar)
        @test_throws MethodError MLFMA.Fr(scalar)

        # A tangential far field stores only θ,φ -- there is no radial component to read.
        @test_throws MethodError MLFMA.Fr(tangential)
        @test_throws MethodError MLFMA.F(tangential)

        # Six paired components have no single conventional name each; `component` is the
        # documented escape hatch, and the named accessors deliberately do not apply.
        @test_throws MethodError MLFMA.Fr(paired)
        @test MLFMA.component(paired, 4) === paired[4]
    end

    @testset "collections share their moment type's accessors" begin
        Fθvalues = fill(1.0 + 0im, 3, 2)
        Fφvalues = fill(2.0 + 0im, 3, 2)

        radiation = MLFMA.TangentialSphericalRadiationFieldCollection(
            SVector(Fθvalues, Fφvalues)
        )
        receive = MLFMA.TangentialSphericalReceiveFieldCollection(
            SVector(Fθvalues, Fφvalues)
        )

        @test MLFMA.Fθ(radiation) === Fθvalues
        @test MLFMA.Fφ(radiation) === Fφvalues
        @test MLFMA.Fθ(receive) === Fθvalues
        @test MLFMA.Fφ(receive) === Fφvalues

        @test_throws MethodError MLFMA.Fx(radiation)
        @test_throws MethodError MLFMA.Fr(receive)
    end

    # `mirrorrule` is a property of the representation, so it dispatches on the type. Cartesian
    # deliberately has no rule: a Cartesian vector does not pick up the same per-component signs
    # under the θ→π-θ, φ→φ+π reflection that a spherical one does, and inventing that pattern
    # would be inventing physics. The `MethodError` is the intended behaviour, not a gap.
    @testset "mirrorrule dispatches on type; Cartesian is deliberately absent" begin
        @test length(MLFMA.mirrorrule(MLFMA.ScalarFarField)) == 1
        @test length(MLFMA.mirrorrule(MLFMA.TangentialSphericalFarField)) == 2
        @test length(MLFMA.mirrorrule(MLFMA.SphericalVectorFarField)) == 3
        @test length(MLFMA.mirrorrule(MLFMA.PairedSphericalVectorFarField)) == 6

        @test_throws MethodError MLFMA.mirrorrule(MLFMA.CartesianVectorFarField)
        @test_throws MethodError MLFMA.mirror(cartesian, sampling)

        # Same reasoning for the interpolation expansion trait, which also encodes reflection
        # sign behaviour and so cannot be shared between the two three-component types.
        @test_throws MethodError MLFMA.defaultinterpolationtrait(typeof(cartesian))
        @test MLFMA.defaultinterpolationtrait(typeof(spherical)) isa
            MLFMA.SphericalInterpolation
    end

    # Mixing representations across a moment/collection pair is a MethodError rather than a
    # silently wrong product: `@linkmomenttocollection` only generates `mul!` for matching pairs.
    @testset "moment/collection representation mismatch does not multiply" begin
        nbasis = 2
        nsamples = length(sampling)
        y = rand(ComplexF64, nbasis)

        tangentialcol = MLFMA.TangentialSphericalRadiationFieldCollection(
            SVector{2,Matrix{ComplexF64}}([rand(ComplexF64, nsamples, nbasis) for _ in 1:2])
        )

        moment = MLFMA.TangentialSphericalFarField(ComplexF64, sampling)
        @test LinearAlgebra.mul!(moment, tangentialcol, y) === moment

        mismatched = MLFMA.SphericalVectorFarField(ComplexF64, sampling)
        @test_throws MethodError LinearAlgebra.mul!(mismatched, tangentialcol, y)
    end

    # A receive coupling rule permutes components within ONE representation; it never converts
    # between them. Without the single-`F` constraint, a spherical source could be written into a
    # a cartesian destination purely because both hold three components -- reinterpreting r,θ,φ as
    # x,y,z with no error anywhere.
    @testset "receivefields! will not write across representations" begin
        efie = MLFMA.efie(; wavenumber=1.3)
        mirrorindex = MLFMA.mirrorindices(sampling)

        src = MLFMA.TangentialSphericalFarField(ComplexF64, sampling, Val(:zero))
        matchingdest = MLFMA.TangentialSphericalFarField(ComplexF64, sampling, Val(:zero))
        @test MLFMA.receivefields!(matchingdest, src, mirrorindex, efie) === matchingdest

        singlelayer = MLFMA.singlelayer(efie)
        sphericalsrc = MLFMA.SphericalVectorFarField(ComplexF64, sampling, Val(:zero))
        cartesiandest = MLFMA.CartesianVectorFarField(ComplexF64, sampling, Val(:zero))
        @test_throws MethodError MLFMA.receivefields!(
            cartesiandest, sphericalsrc, mirrorindex, singlelayer
        )

        # A freshly built scratch of the same representation must resolve to the identical concrete
        # type as the moment -- which is what makes `receivefields!`'s single-`F` signature safe
        # rather than over-strict, and what lets the leaf-channel scratch be handed straight to it.
        scratch = MLFMA.farfieldtype(efie)(ComplexF64, size(src[1]))
        @test typeof(scratch) === typeof(src)
        @test typeof(MLFMA.receivefields!(scratch, src, mirrorindex, efie)) === typeof(src)
    end

    # The leaf multiplies take their rule from the operator + far-field type but read matrices from
    # the stored collection, so those must describe the same representation. Mismatches are
    # rejected at construction rather than silently producing wrong numbers in the hot loop.
    @testset "LazyReceiveFieldCollection rejects a mismatched radiation collection" begin
        efie = MLFMA.efie(; wavenumber=1.3)
        nsamples = length(sampling)
        nbasis = 3

        tangentialcol = MLFMA.TangentialSphericalRadiationFieldCollection(
            SVector{2,Matrix{ComplexF64}}([rand(ComplexF64, nsamples, nbasis) for _ in 1:2])
        )
        mirrorindex = MLFMA.mirrorindices(sampling)
        @test MLFMA.LazyReceiveFieldCollection(efie, tangentialcol, mirrorindex) isa
            MLFMA.LazyReceiveFieldCollection

        # Three components, so every tangential rule index (1 and 2) is in range -- exactly the
        # case that would have gone through silently.
        sphericalcol = MLFMA.SphericalVectorRadiationFieldCollection(
            SVector{3,Matrix{ComplexF64}}([rand(ComplexF64, nsamples, nbasis) for _ in 1:3])
        )
        @test_throws ErrorException MLFMA.LazyReceiveFieldCollection(
            efie, sphericalcol, mirrorindex
        )

        # A receive collection is not a radiation collection either, even at the right count.
        receivecol = MLFMA.TangentialSphericalReceiveFieldCollection(
            SVector{2,Matrix{ComplexF64}}([rand(ComplexF64, nsamples, nbasis) for _ in 1:2])
        )
        @test_throws ErrorException MLFMA.LazyReceiveFieldCollection(
            efie, receivecol, mirrorindex
        )

        err = try
            MLFMA.LazyReceiveFieldCollection(efie, sphericalcol, mirrorindex)
        catch e
            sprint(showerror, e)
        end
        @test occursin("TangentialSphericalRadiationFieldCollection", err)
        @test occursin("SphericalVectorRadiationFieldCollection", err)
    end
end
