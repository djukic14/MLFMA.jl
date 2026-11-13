module TestFarField

using Test

@testset verbose = true "FarField" begin
    include("farfield.jl")
    include("constructors.jl")
    include("representation.jl")
    include("rules.jl")
    include("typestability.jl")
    include("assembler.jl")
    # The same assembler on a 2-D problem -- right after the 3-D assembler tests, since the
    # two lines it pins are shared with them and a 3-D fault should surface there first.
    include("farfield2d.jl")
    include("dummyassembler.jl")
    include("fieldcollections.jl")
    include("collectionbuilders.jl")
    include("storerules.jl")
    include("leaforder.jl")
    include("mirrored.jl")
    # The spherical-harmonic leaf collection, last -- it is checked AGAINST the sampled collection
    # the files above cover, so a fault in that one should surface as its own failure first.
    include("sphericalharmoniccollection.jl")
    # Building those collections from a real assembler and tree -- last, since it needs both the
    # assembler fixtures above and the collection itself to be sound.
    include("shecollectionbuilder.jl")
    # The TANGENTIAL vector backend, last: its cross-check is against the sampled collections the
    # files above cover, and it reuses `shecollectionbuilder.jl`'s fixture and
    # `sphericalharmoniccollection.jl`'s traversal fakes.
    include("vshcollection.jl")
    include("harmonicplumbing.jl")
    include("harmonicbatchallocations.jl")
end

end
