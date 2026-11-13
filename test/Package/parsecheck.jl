module ParseCheckTest
using Test
using MLFMA

# Directories whose `.jl` files are expected to stay parseable. `src` is the package itself. The
# other three are included because none of them is fully executed by the suite, so a syntax error in
# them survives a green test run.
const PARSEABLEDIRECTORIES = ("src", "test", "benchmark", "examples")

# `Meta.parseall` does not throw on bad input: it returns the failure as an `:error` (invalid
# syntax) or `:incomplete` (truncated file) expression among the toplevel arguments.
function isunparseable(parsed)
    return any(
        arg -> arg isa Expr && (arg.head === :error || arg.head === :incomplete),
        parsed.args,
    )
end

"""
    scanforsyntaxerrors(root)

Walk [`PARSEABLEDIRECTORIES`](@ref) under `root` and return `(scanned, offenders)`, both as paths
relative to `root`: every `.jl` file looked at, and the subset Julia cannot parse.
"""
function scanforsyntaxerrors(root)
    scanned = String[]
    offenders = String[]
    for directory in PARSEABLEDIRECTORIES
        path = joinpath(root, directory)
        isdir(path) || continue
        for (folder, _, files) in walkdir(path), file in files
            endswith(file, ".jl") || continue
            filepath = joinpath(folder, file)
            push!(scanned, relpath(filepath, root))
            isunparseable(Meta.parseall(read(filepath, String); filename=filepath)) &&
                push!(offenders, relpath(filepath, root))
        end
    end

    return scanned, offenders
end

@testset "Package parses" begin
    # This is a separate check rather than a stricter formatting test because the formatter cannot
    # report it. `JuliaFormatter.format` only WARNS on a file it fails to parse and still returns
    # `true`, measured on a scratch package holding a single unparseable file, so a syntax error
    # anywhere leaves the formatting testset green.
    scanned, offenders = scanforsyntaxerrors(pkgdir(MLFMA))
    @test isempty(offenders)

    # Case-only filename differences collide on common Windows and macOS filesystems.
    @test allunique(lowercase.(scanned))

    # Guard against a vacuous pass: an empty walk, a wrong root or a renamed directory would satisfy
    # the assertion above without having looked at anything.
    @test joinpath("src", "MLFMA.jl") in scanned
    @test joinpath("test", "runtests.jl") in scanned

    # The predicate itself, on inputs whose verdict is known. Without this the testset only says
    # that nothing was found, not that anything could be.
    @test isunparseable(Meta.parseall("function f(x)\n    return x +\nend\n"))
    @test isunparseable(Meta.parseall("struct Truncated\n    x::Int\n"))
    @test !isunparseable(Meta.parseall("f(x) = x + 1\n"))
end
end
