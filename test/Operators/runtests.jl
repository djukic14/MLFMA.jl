module TestOperators

using Test

@testset verbose = true "Operators" begin
    include("efie.jl")
    include("farfields.jl")
    include("operatorutils.jl")
    # The 2-D Helmholtz mappings, last: they compose the far-field operators and the translation
    # operator that the files above cover individually.
    include("helmholtz2d.jl")
    include("hypersingular2d.jl")
end

end
