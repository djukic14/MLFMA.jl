module TestMaps

using Test

@testset verbose = true "Maps" begin
    include("galerkin.jl")
    include("workspace.jl")
    include("sharedcore_directions.jl")
    include("petrov.jl")
    include("petrovdirections.jl")
    include("aggregation.jl")
    include("tworects.jl")
    include("comparison.jl")
    # The 2-D matvec, end to end. After the 3-D map suites, because a fault it finds is most legible
    # once those are known good -- and before `operators.jl`, which is much the slower of the two
    # broad sweeps.
    include("helmholtz2d.jl")
    # Compressed 2-D leaf storage through a whole matvec -- test (d) of #138. After `helmholtz2d.jl`,
    # whose sampled maps are the baseline it compares against, and whose symmetry measurement is what
    # bounds the adjoint claim this file is allowed to make.
    include("circularharmonicstorage.jl")
    # Every operator in `MLFMA.farfieldtype`'s table, against a dense reference, and every leaf
    # storage its representation admits. Last, because it is the broadest and slowest -- and because
    # a fault it finds is most legible once the focused suites above are known good.
    include("operators.jl")
end

end
