using MLFMA
using StaticArrays
using CompScienceMeshes, BEAST
using Test
using H2Factory, LinearAlgebra
using H2Trees
@testset "FarField Type Stabilty" begin
    types = [BigFloat, Float64, Float32, ComplexF64, ComplexF32, Complex{BigFloat}]

    for SAMPLINGTYPES in types
        sampling = MLFMA.SphericalSampling{SAMPLINGTYPES}(10, 10)
        for T in types
            # Zero-initialized, not the (now genuinely undef) default: every op below reads an
            # entry before writing it (`conj`, `add!`, `mul!`, `elementwisemul!`, `copyto!`'s
            # source), and BigFloat/Complex{BigFloat} are boxed -- an undef entry there is a
            # `Base.undef_ref_str`-throwing unset reference, not merely unpredictable bits like
            # an undef Float64 entry. This testset checks *type* stability of the ops, not the
            # undef/zero construction policy itself (see `constructors.jl` for that), so a
            # readable value is all it needs.
            ff = MLFMA.TangentialSphericalFarField(T(1.0), sampling, Val(:zero))
            ff2 = MLFMA.TangentialSphericalFarField(T(1.0), sampling, Val(:zero))
            @test typeof(MLFMA.Nθ(ff)) == Int64
            @test typeof(MLFMA.Nφ(ff)) == Int64
            @test typeof(MLFMA.Fθ(ff)) == Matrix{T}
            @test typeof(MLFMA.Fφ(ff)) == Matrix{T}

            # Pin the moment's shape by component count and array type only. The full parameter
            # list belongs to H2Factory's `@createmoment` (currently `{D,T,P,V}`), so spelling it
            # out here would break on every H2Factory parameter addition without any MLFMA-side
            # defect -- `<:` keeps this checking what MLFMA actually relies on.
            @test ff isa MLFMA.TangentialSphericalFarField{2,Matrix{T}}

            # The type-stability claim itself: each of these must return the very same concrete
            # type it was given, whatever that type's parameters happen to be.
            @test (typeof(conj(ff))) == typeof(ff)
            @test (typeof(conj!(ff))) == typeof(ff)
            @test (typeof(MLFMA.mirror(ff, sampling))) == typeof(ff)

            @test (typeof(H2Factory.add!(ff, ff2))) == typeof(ff)
            @test (typeof(LinearAlgebra.mul!(ff, ff, T(1.0)))) == typeof(ff)
            @test (typeof(elementwisemul!(ff, ff2, ones(T, size(sampling))))) == typeof(ff)
            @test (typeof(copyto!(ff, ff2))) == typeof(ff)
        end
    end
end

@testset "RadiationFieldCollection Type Stabilty" begin
    types = [Float64, Float32, ComplexF64, ComplexF32, Complex{BigFloat}]

    mesh = CompScienceMeshes.meshsphere(0.1, 0.1)
    X = raviartthomas(mesh)
    nfunctions = numfunctions(X)

    efie = MLFMA.efie(; wavenumber=1.0)
    doublelayer = Maxwell3D.doublelayer(; wavenumber=1.0)
    rotateddoublelayer = BEAST.n × doublelayer
    for operator in [efie, doublelayer, rotateddoublelayer]
        for SAMPLINGTYPE in types
            Nθ, Nφ = MLFMA.NθNφ(1)
            @test typeof(Nθ) == Int
            @test typeof(Nφ) == Int

            sampling = MLFMA.SphericalSampling{SAMPLINGTYPE}(Nθ, Nφ)
            for T in types
                γ = T(1.0)

                Fθ = ones(T, Nθ * Nφ, nfunctions)
                Fφ = ones(T, Nθ * Nφ, nfunctions)

                ffcollection = MLFMA.TangentialSphericalRadiationFieldCollection(
                    SVector(Fθ, Fφ)
                )

                @test typeof(MLFMA.Fθ(ffcollection)) == Matrix{T}
                @test typeof(MLFMA.Fφ(ffcollection)) == Matrix{T}

                @test typeof(LinearAlgebra.mul!(ffcollection, ffcollection, T(2))) ==
                    MLFMA.TangentialSphericalRadiationFieldCollection{2,Matrix{T}}

                # Zero-initialized: `mul!` below reads `ff`'s entries, and a genuinely undef
                # BigFloat/Complex{BigFloat} entry is an unset reference, not just unpredictable
                # bits -- see the matching note in the "FarField Type Stabilty" testset above.
                ff = MLFMA.TangentialSphericalFarField(T(1.0), sampling, Val(:zero))

                @test typeof(LinearAlgebra.mul!(ones(T, nfunctions), ffcollection, ff)) ==
                    Vector{T}

                @test typeof(
                    MLFMA.phasefactor(
                        γ,
                        MLFMA.cartesiansamplepoints(sampling)[1],
                        SVector(SAMPLINGTYPE(1), SAMPLINGTYPE(0), SAMPLINGTYPE(0)),
                        MLFMA.AbstractRadiationFieldCollection,
                    ),
                ) == Matrix{T}

                phase = zeros(T, size(sampling))
                @test typeof(
                    MLFMA.phasefactor!(
                        phase,
                        γ,
                        MLFMA.cartesiansamplepoints(sampling)[1],
                        SVector(SAMPLINGTYPE(1), SAMPLINGTYPE(0), SAMPLINGTYPE(0)),
                        MLFMA.AbstractRadiationFieldCollection,
                    ),
                ) == Matrix{T}

                if !(SAMPLINGTYPE <: Complex)
                    m = convert(SAMPLINGTYPE, mesh)
                    X = raviartthomas(m)

                    points = BEAST.positions(X)
                    root_center, root_size = H2Trees.boundingbox(points)

                    tree = TwoNTree(
                        SVector(root_center...),
                        points,
                        root_size * SAMPLINGTYPE(1.1),
                        SAMPLINGTYPE(0.1 / 4),
                    )
                    minrelevantlevel = H2Trees.level(tree, first(H2Trees.leaves(tree)))
                    @test_nowarn MLFMA.leaffarfieldsamplings(
                        tree, [sampling], minrelevantlevel
                    )
                end
                result = zeros(T, nfunctions)

                if !(operator isa Maxwell3D.DoubleLayerRotatedMW3D) && (T <: Complex)
                    mirroredcollection = MLFMA.LazyReceiveFieldCollection(
                        operator, ffcollection, MLFMA.mirrorindices(sampling)
                    )
                    @test typeof(mirroredcollection) == MLFMA.LazyReceiveFieldCollection{
                        typeof(operator),typeof(ffcollection),Matrix{Int}
                    }

                    # A sampling where indices belong is refused at construction: the field is
                    # unconstrained structurally, so without this the object would build fine and
                    # fail much later inside `transform!`.
                    @test_throws ErrorException MLFMA.LazyReceiveFieldCollection(
                        operator, ffcollection, sampling
                    )

                    if !(SAMPLINGTYPE <: Complex) && !(T <: Complex)
                        leafscratch = MLFMA.farfieldtype(operator)(T, sampling, Val(:zero))
                        @test typeof(
                            H2Factory.applyleafmul!(
                                leafscratch, result, mirroredcollection, ff, true, false
                            ),
                        ) == Vector{T}
                    end
                end
            end
        end
    end
end
