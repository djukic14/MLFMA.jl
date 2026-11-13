module TestTranslations

using Test

@testset verbose = true "Translations" begin
    include("translation.jl")
    include("translation2d.jl")
    include("i2o_collection.jl")
    include("symmetrycollection.jl")
end

end
