module SelfReferenceTest
using Test
using MLFMA

# NO FILE UNDER `src/` MAY SAY `MLFMA.` IN CODE, `src/MLFMA.jl` ITSELF ASIDE.
#
# Inside a layer submodule the name `MLFMA` is not bound, so `MLFMA.foo` is an `UndefVarError`. It is
# not a load error, though: the reference resolves when the line RUNS, so a package with one of these
# precompiles cleanly and fails later, in whatever test happens to reach that branch.
#
# This has escaped twice, each time differently, which is why it is a test rather than a habit:
#
#   1. The migration tool rewrote `MLFMA.foo` inside `error(...)` strings, turning user-facing advice
#      into layer-internal names. Fixed by skipping string literals.
#   2. That fix then SKIPPED string INTERPOLATIONS, which are code. `$(MLFMA.elementtype(builder))`
#      in `strategies.jl` survived into `Assembly` and raised `UndefVarError` from an error path,
#      three gate-hours later.
#
# Parsing catches both: the parser puts interpolated code in the tree and drops prose.
#
# THERE ARE NO LONGER ANY EXEMPT DIRECTORIES. `equidistant/`, `interpolation/diagnostics/`,
# `sphericalharmonics/` and `circularharmonics/` predate the layer split and used to reach the parent
# with `import ...MLFMA`; each now imports the layers it actually reads (`Sampling`, `Interpolation`,
# `FarFields`) and names them directly. That is what let `Sampling` and `Interpolation` stop exporting
# `_axispreservingpolicy`, `_θφpermutation` and `_stageprescaled!`, which were exported for no reason
# but to be visible through the top level. See task #174.

function walkexpr(f, ex)
    ex isa Expr || return nothing
    f(ex)
    for arg in ex.args
        walkexpr(f, arg)
    end
    return nothing
end

"""
    filereferences(filepath)

Every `name` where `filepath` refers to `MLFMA.name` in CODE. Prose is dropped and interpolated code
is kept, because the parser draws that line rather than this function.
"""
function filereferences(filepath)
    found = Symbol[]
    walkexpr(Meta.parseall(read(filepath, String); filename=filepath)) do ex
        ex.head === :. && length(ex.args) == 2 || return nothing
        ex.args[1] === :MLFMA || return nothing
        qualified = ex.args[2]
        name = qualified isa QuoteNode ? qualified.value : qualified
        name isa Symbol && push!(found, name)
        return nothing
    end
    return found
end

"""
    selfreferences(root)

Every `(file, name)` where a file under `root/src` refers to `MLFMA.name` in code.
"""
function selfreferences(root)
    found = Tuple{String,Symbol}[]
    for (folder, _, files) in walkdir(joinpath(root, "src")), file in files
        endswith(file, ".jl") || continue
        filepath = joinpath(folder, file)
        relative = relpath(filepath, root)
        # `src/MLFMA.jl` is the top level itself, where `MLFMA.` would be redundant but not wrong.
        relative == joinpath("src", "MLFMA.jl") && continue

        for name in filereferences(filepath)
            push!(found, (relative, name))
        end
    end

    return found
end

@testset "No file under src references MLFMA" begin
    @test selfreferences(pkgdir(MLFMA)) == Tuple{String,Symbol}[]

    # NON-VACUITY, and it has to be synthetic now. It used to be "the exempt submodules really do
    # contain such references", which was a fact about those directories; once they were repointed
    # that control asserted the opposite of the truth and would have had to be deleted rather than
    # updated. A fixture cannot go stale that way, and it pins more: that prose is dropped AND that
    # interpolated code is kept, which are the two ways this check has been wrong before.
    mktemp() do path, io
        write(
            io,
            """
            f(x) = MLFMA.plaincall(x)
            g(y) = "interpolated \$(MLFMA.insidestring(y)) here"
            h() = error("advice naming MLFMA.inprose, which is not a reference")
            """,
        )
        close(io)
        found = filereferences(path)
        @test :plaincall in found            # ordinary call position
        @test :insidestring in found         # code inside a string interpolation
        @test !(:inprose in found)           # prose inside a string literal
        @test length(found) == 2
    end
end
end
