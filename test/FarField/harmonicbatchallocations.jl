function harmonicchunkallocations(run)
    run()
    run()
    return minimum(@allocated(run()) for _ in 1:3)
end

@testset "Harmonic batch allocations" begin
    P, nleaves, nbf = 2, 20, 3
    sampling = MLFMA.SphericalSampling{Float64}(8, 16)
    center = SVector(0.0, 0.0, 0.0)
    tree = FakeChunkTree(Set(1:nleaves), Dict(i => collect((3i - 2):3i) for i in 1:nleaves))
    map = FakeChunkMap(Dict(i => zeros(ComplexF64, nbf) for i in 1:nleaves))
    x = ones(ComplexF64, nbf * nleaves)
    for (make, scratchbuilder) in (
        (
            P -> MLFMA.SphericalHarmonicRadiationCollection(
                SphericalHarmonicStorage(P, center, ones(ComplexF64, 3(P + 1)^2, nbf))
            ),
            MLFMA.shescratch,
        ),
        (
            P -> MLFMA.VectorSphericalHarmonicRadiationCollection(
                MLFMA.VectorSphericalHarmonicStorage(
                    P, center, ones(ComplexF64, 2 * P * (P + 2), nbf)
                ),
            ),
            MLFMA.vshscratch,
        ),
    )
        collections = Dict(i => make(P) for i in 1:nleaves)
        moments = Dict(
            i => MLFMA.TangentialSphericalFarField{2,ComplexF64}(size(sampling)) for
            i in 1:nleaves
        )
        for capacity in (1, 7, nleaves)
            scratch = scratchbuilder(sampling, P; ngroups=capacity)
            run() = H2Factory.aggregateleafchunk!(
                map,
                tree,
                1:nleaves,
                i -> moments[i],
                i -> collections[i],
                H2Factory.LinAlgebraMulFunctor(),
                x,
                1,
                scratch,
            )
            bytes = harmonicchunkallocations(run)
            # The original loops used at most 5856 bytes on this fixture. Leave room
            # for compiler variation, but catch larger temporary allocations.
            @test bytes <= 8 * 1024
            @test all(all(isfinite, m[1]) && all(isfinite, m[2]) for m in values(moments))
        end
    end
end
