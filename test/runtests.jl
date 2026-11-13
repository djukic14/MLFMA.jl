using Test
using MLFMA

# Explicit nested testsets by module, one runner per source area, matching iFMM's `test/runtests.jl`.
# Each module runner is its own `module`, so a fixture defined in one file cannot silently satisfy a
# reference in another.
#
# Not included here, deliberately:
#   - `scripts/`, runnable exploratory scripts with no assertions, kept because they are useful by
#     hand but they verify nothing, so they do not belong in the suite.
#   - `performance/` (lowercase), allocation exploration that reports via `@info` rather than
#     asserting, and needs a GUI backend for its `@profview`. The capitalized `Performance/` below is
#     where Phase 8's real contracts live; the lowercase directory stays the scratchpad it is. That
#     capitalisation split is the convention already in use here: capitalized directories run in the
#     suite, lowercase ones (`scripts/`, `performance/`) are deliberately outside it.
@testset verbose = true "MLFMA" begin
    # First: the cheap package-shape checks. A syntax error or a lost H2Factory hook should be
    # reported as such, before an hour of physics runs on top of it.
    include("Package/runtests.jl")
    include("Sampling/runtests.jl")
    include("Operators/runtests.jl")
    include("Construction/runtests.jl")
    include("FarField/runtests.jl")
    include("LevelTransfers/runtests.jl")
    include("Translations/runtests.jl")
    include("Equidistant/runtests.jl")
    include("SphericalHarmonics/runtests.jl")
    include("CircularHarmonics/runtests.jl")
    include("Maps/runtests.jl")
    include("Performance/runtests.jl")
    include("Visualization/runtests.jl")

    @testset "Code formatting (JuliaFormatter.jl)" begin
        using JuliaFormatter
        @test JuliaFormatter.format(pkgdir(MLFMA); overwrite=false)
    end
end
