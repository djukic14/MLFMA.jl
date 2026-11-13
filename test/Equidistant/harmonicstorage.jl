module EquidistantHarmonicStorageTests

using Test
using MLFMA
using MLFMA.Equidistant:
    EquidistantSampling,
    EquidistantSamplingBuilder,
    EquidistantInterpolator,
    FFTBackend,
    exactquadratureweights
using MLFMA.SphericalHarmonics:
    SphericalHarmonicTransform,
    VectorSphericalHarmonicTransform,
    numcoefficients,
    numvshcoefficients,
    synthesize,
    analyze,
    vshsynthesize,
    vshanalyze,
    ylm,
    lmindex
using BEAST
using CompScienceMeshes
using H2Trees
using H2Factory
using LinearAlgebra
using Random
using StaticArrays

@testset "Equidistant scalar and vector harmonic transforms" begin
    rng = MersenneTwister(32)
    for P in (1, 3, 5)
        s = EquidistantSampling{Float64}(2P + 2, 4P + 4)
        points, weights = MLFMA.sphericalsamplepoints(s)
        scalar = SphericalHarmonicTransform(ComplexF64, s, P)
        vector = VectorSphericalHarmonicTransform(ComplexF64, s, P)
        @test scalar.weights == vector.weights == vec(exactquadratureweights(s))

        coefficients = randn(rng, ComplexF64, numcoefficients(P), 3)
        field = synthesize(scalar, coefficients)
        @test analyze(scalar, field) ≈ coefficients atol = 2e-12
        @test sum(vec(weights) .* abs2.(field)) ≈ sum(abs2, coefficients) rtol = 2e-12
        # An independently sampled harmonic pins the transform's flattened direction order.
        harmonic = [ylm(P, -1, p[2], p[3]) for p in vec(points)]
        unit = zeros(ComplexF64, numcoefficients(P))
        unit[lmindex(P, -1)] = 1
        @test synthesize(scalar, unit) ≈ harmonic atol = 2e-14
        @test analyze(scalar, harmonic) ≈ unit atol = 2e-12

        a = randn(rng, ComplexF64, numvshcoefficients(P), 2)
        b = randn(rng, ComplexF64, numvshcoefficients(P), 2)
        fθ, fφ = vshsynthesize(vector, a, b)
        ar, br = vshanalyze(vector, fθ, fφ)
        @test ar ≈ a atol = 2e-12
        @test br ≈ b atol = 2e-12
        @test sum(vec(weights) .* (abs2.(fθ) .+ abs2.(fφ))) ≈ sum(abs2, a) + sum(abs2, b) rtol =
            2e-12

        gθ = randn(rng, ComplexF64, size(fθ))
        gφ = randn(rng, ComplexF64, size(fφ))
        ga, gb = vshanalyze(vector, gθ, gφ)
        @test sum(vec(weights) .* (conj.(fθ) .* gθ .+ conj.(fφ) .* gφ)) ≈
            dot(a, ga) + dot(b, gb) atol = 2e-12
    end
end

@testset "Equidistant MLFMA with both harmonic leaf storage policies" begin
    # Two separated square patches, one RWG each: a 2 x 2 dense control, with a real far path.
    vertices = SVector{3,Float64}[
        (x + offset, y, 0) for offset in (0.0, 3.0) for
        (x, y) in ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
    ]
    faces = SVector{3,Int}[(1, 2, 3), (1, 3, 4), (5, 6, 7), (5, 7, 8)]
    X = raviartthomas(
        CompScienceMeshes.Mesh(vertices, CompScienceMeshes.SimplexGraph.(faces))
    )
    @test numfunctions(X) == 2
    op = MLFMA.efie(; wavenumber=2π)
    tree = TwoNTree(
        X;
        builder=TwoNTreeBuilder(;
            minhalfsize=0.25, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
        ),
    )
    IT = typeof(EquidistantInterpolator{ComplexF64}(FFTBackend()))
    phase = MLFMA.phaseshiftertype(MLFMA.MLFMAStrategy(op, X))
    function mapwith(storage)
        strategy = MLFMA.MLFMAStrategy(
            op,
            X;
            samplingstrategy=MLFMA.SamplingStrategy(;
                samplingbuilder=EquidistantSamplingBuilder{Float64}(), ϵ=1e-4
            ),
            leveltransferstrategy=MLFMA.LevelTransferStrategy(IT, phase),
            farfieldstrategy=MLFMA.FarFieldStrategy(; leafstorage=storage),
        )
        return MLFMA.GalerkinMLFMA(op, X; tree=tree, strategy=strategy, verbose=false)
    end
    sampled = mapwith(MLFMA.SampledLeafStorage())
    dense = BEAST.assemble(op, X, X)
    x = ComplexF64[1 + 0.3im, -0.7 + 0.2im]
    far = similar(x)
    H2Factory.farmul!(far, sampled, x)
    @test norm(far) > 1e-6 * norm(dense * x)
    # The small fixture has a shared pipeline error, independent of leaf compression.
    # Keep the dense screen separate from the sensitive sampled and far-only comparisons.
    for orient in (identity, transpose, adjoint)
        @test orient(sampled) * x ≈ orient(dense) * x rtol = 1e-2
    end

    for (storage, radiation, receive) in (
        (
            MLFMA.SphericalHarmonicLeafStorage(),
            MLFMA.SphericalHarmonicRadiationCollection,
            MLFMA.SphericalHarmonicReceiveCollection,
        ),
        (
            MLFMA.VectorSphericalHarmonicLeafStorage(),
            MLFMA.VectorSphericalHarmonicRadiationCollection,
            MLFMA.VectorSphericalHarmonicReceiveCollection,
        ),
    )
        compressed = mapwith(storage)
        core = H2Factory.core(compressed)
        @test valtype(core.trialfieldcollections) <: radiation
        @test valtype(core.testfieldcollections) <: receive
        @test all(s -> s isa EquidistantSampling, core.samplings)
        for orient in (identity, transpose, adjoint)
            reference = orient(sampled) * x
            @test orient(compressed) * x ≈ reference rtol = 1e-5
            @test orient(compressed) * x ≈ orient(dense) * x rtol = 1e-2
            gotfar = similar(x)
            reffar = similar(x)
            H2Factory.farmul!(gotfar, orient(compressed), x)
            H2Factory.farmul!(reffar, orient(sampled), x)
            @test gotfar ≈ reffar rtol = 1e-4
        end
    end
end

end # module
