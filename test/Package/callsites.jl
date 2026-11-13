module CallSitesTest
using Test
using MLFMA

# EVERY `MLFMA.name` USED OUTSIDE `src` STILL RESOLVES.
#
# The layer split moves definitions into submodules and re-exposes them at the top level. A name
# that fails to make that trip does not break the build: `using MLFMA` still succeeds, and the
# failure only appears when something calls it. There are several hundred such call sites in
# `test`, `benchmark` and `examples`, so a check that walks them is the one that actually covers
# the promise Phase 1 makes.
#
# It has already earned its place. `FarFields` generates eighteen of its types with
# `@createmoment`/`@linkmomenttocollection`, which no source-parsing export list can see, and they
# were missing from the first version of that layer's exports.
#
# THIS PARSES RATHER THAN GREPS, and that is not fussiness. A regex over the raw text matches
# `MLFMA.jl` in prose, and `MLFMA.name` in a docstring that is describing the pattern rather than
# calling anything. Both showed up as false failures on the first version. The parser drops comments
# and string literals for free, so what is left is what the code actually references.

const SEARCHEDDIRECTORIES = ("test", "benchmark", "examples")

function walkexpr(f, ex)
    ex isa Expr || return nothing
    f(ex)
    for arg in ex.args
        walkexpr(f, arg)
    end
    return nothing
end

"""
    referencednames(root)

Every `name` reached from `MLFMA` in real code under [`SEARCHEDDIRECTORIES`](@ref), written either
as `MLFMA.name` or as `getfield(MLFMA, :name)`.

Only the leading qualifier is collected: for `MLFMA.Submodule.name` the name that has to resolve in
`MLFMA` is `Submodule`, and what it holds is that submodule's business.
"""
function namesin(source::AbstractString; filename="none")
    found = Set{Symbol}()
    walkexpr(Meta.parseall(source; filename=filename)) do ex
        # `getfield(MLFMA, :name)` IS a call site, and the dotted form is not the only spelling.
        # This cost a gate cycle: `beastinterface.jl` asked `getfield(MLFMA, :quaddata)`, the walk
        # below could not see it, so this test passed while `using BEAST` was being removed from the
        # top level, and the breakage only appeared when the suite ran. Same family as the
        # string-interpolation and qualified-reference blind spots recorded in `selfreference.jl`:
        # a scan shaped like one spelling misses the others.
        if ex.head === :call &&
            length(ex.args) == 3 &&
            ex.args[1] === :getfield &&
            ex.args[2] === :MLFMA &&
            ex.args[3] isa QuoteNode &&
            ex.args[3].value isa Symbol
            push!(found, ex.args[3].value)
            return nothing
        end
        # `length(args) == 2` matters: a broadcast operator such as `.+` parses to a `:.`
        # expression with a SINGLE argument, and indexing past it throws.
        ex.head === :. && length(ex.args) == 2 || return nothing
        ex.args[1] === :MLFMA || return nothing
        qualified = ex.args[2]
        name = qualified isa QuoteNode ? qualified.value : qualified
        return name isa Symbol && push!(found, name)
    end
    return found
end

function referencednames(root)
    found = Set{Symbol}()
    for directory in SEARCHEDDIRECTORIES
        path = joinpath(root, directory)
        isdir(path) || continue
        for (folder, _, files) in walkdir(path), file in files
            endswith(file, ".jl") || continue
            filepath = joinpath(folder, file)
            union!(found, namesin(read(filepath, String); filename=filepath))
        end
    end

    return found
end

@testset "Call sites outside src" begin
    referenced = referencednames(pkgdir(MLFMA))
    unresolved = sort([name for name in referenced if !isdefined(MLFMA, name)]; by=string)

    @test unresolved == Symbol[]

    # Not vacuous: the walk found a substantial surface, not an empty directory or a broken parse.
    @test length(referenced) > 200
    @test :GalerkinMLFMA in referenced
    @test :SphericalSampling in referenced

    # Underscore internals and Unicode names are in scope too, not just the tidy public ones. A
    # check that quietly skipped them would miss most of what the layer split can break.
    @test :_leaflevelindex in referenced
    @test :θs in referenced

    @testset "both spellings, on a fixture" begin
        # A FIXTURE RATHER THAN A REAL FILE, because the `getfield` branch has no live call site
        # left: the one that motivated it now asks `MLFMA.Operators` instead. Pinning the branch
        # against the repo would mean pinning it against zero occurrences, which passes whether or
        # not the branch works.
        found = namesin("""
            a = MLFMA.dotted(x)
            b = getfield(MLFMA, :viagetfield)
            c = MLFMA.Submodule.deeper(y)
            d = getfield(MLFMA.Operators, :notthetoplevel)
            e = getfield(Other, :notmlfma)
            f = "prose naming MLFMA.instring"
            g = x .+ y
            """)
        @test :dotted in found
        @test :viagetfield in found
        # Only the leading qualifier: what `Submodule` holds is its own business.
        @test :Submodule in found
        @test !(:deeper in found)
        # `getfield` on a LAYER is not a top-level call site, and neither is one on another module.
        @test !(:notthetoplevel in found)
        @test !(:notmlfma in found)
        # But `MLFMA.Operators` inside that expression IS one: the layer has to resolve in `MLFMA`
        # for the line to run at all, so collecting it is right, not a leak.
        @test :Operators in found
        # Prose is not code; and `.+` parses to a one-argument `:.` that must not be indexed past.
        @test !(:instring in found)
        @test length(found) == 4
    end
end
end
