# Included by layers.jl so these checks run with the existing package tests.
module ArchitectureTest
using Test

# An explicit permitted graph, not "anything loaded earlier". Nested modules have
# their own budgets; including a child is containment, not a dependency edge.
const PERMITTED = Dict(
    :Seams => Set{Symbol}(),
    :Sampling => Set{Symbol}(),
    :Harmonics => Set{Symbol}(),
    :SphericalHarmonics => Set([:Sampling]),
    :CircularHarmonics => Set([:Sampling]),
    :FarFields =>
        Set([:Sampling, :Seams, :Harmonics, :SphericalHarmonics, :CircularHarmonics]),
    :Interpolation => Set([:Sampling, :FarFields]),
    :InterpolationDiagnostics => Set([:Sampling, :FarFields, :Interpolation]),
    :Equidistant => Set([:Sampling, :FarFields, :Interpolation]),
    :Operators => Set([
        :Sampling,
        :Seams,
        :Harmonics,
        :SphericalHarmonics,
        :CircularHarmonics,
        :FarFields,
        :Interpolation,
    ]),
    :Phase => Set([
        :Sampling,
        :Seams,
        :Harmonics,
        :SphericalHarmonics,
        :CircularHarmonics,
        :FarFields,
        :Interpolation,
        :Operators,
    ]),
    :Translations => Set([
        :Sampling,
        :Seams,
        :Harmonics,
        :SphericalHarmonics,
        :CircularHarmonics,
        :FarFields,
        :Interpolation,
        :Operators,
        :Phase,
    ]),
    :Assembly => Set([
        :Sampling,
        :Seams,
        :Harmonics,
        :SphericalHarmonics,
        :CircularHarmonics,
        :FarFields,
        :Interpolation,
        :Operators,
        :Phase,
        :Translations,
    ]),
    :TestSupport => Set([
        :Sampling,
        :Seams,
        :Harmonics,
        :SphericalHarmonics,
        :CircularHarmonics,
        :FarFields,
        :Interpolation,
        :Operators,
        :Phase,
        :Translations,
        :Assembly,
    ]),
)

const OWNERS = Dict(
    "seams" => :Seams,
    "sampling" => :Sampling,
    "harmonics" => :Harmonics,
    "sphericalharmonics" => :SphericalHarmonics,
    "circularharmonics" => :CircularHarmonics,
    "farfield" => :FarFields,
    "interpolation" => :Interpolation,
    "equidistant" => :Equidistant,
    "operators" => :Operators,
    "phase" => :Phase,
    "translation" => :Translations,
    "assembly" => :Assembly,
    "testsupport" => :TestSupport,
)

function ownerof(relative)
    parts = splitpath(relative)
    parts[1] == "interpolation" &&
        length(parts) > 1 &&
        parts[2] == "diagnostics" &&
        return :InterpolationDiagnostics
    return get(OWNERS, parts[1], nothing)
end

function rootname(ex)
    ex isa Symbol && return ex
    ex isa Expr && ex.head === :. || return nothing
    # Import paths use leading dot symbols; ordinary qualification nests Expr(:.).
    for arg in ex.args
        arg === :. && continue
        return rootname(arg)
    end
    return nothing
end

function dependencies(ex)
    found = Set{Symbol}()
    function visit(ex)
        ex isa Expr || return nothing
        if ex.head in (:using, :import)
            for arg in ex.args
                path = arg isa Expr && arg.head === :(:) ? first(arg.args) : arg
                name = rootname(path)
                (haskey(PERMITTED, name) || name === :MLFMA) && push!(found, name)
            end
        elseif ex.head === :.
            name = rootname(ex)
            (haskey(PERMITTED, name) || name === :MLFMA) && push!(found, name)
        end
        return foreach(visit, ex.args)
    end
    visit(ex)
    return found
end

forbidden(owner, deps) = setdiff(deps, union(PERMITTED[owner], Set([owner])))

function acyclic(graph)
    active, done = Set{Symbol}(), Set{Symbol}()
    function visit(node)
        node in active && return false
        node in done && return true
        push!(active, node)
        all(visit, graph[node]) || return false
        delete!(active, node)
        push!(done, node)
        return true
    end
    return all(visit, keys(graph))
end

# Whitelist declarations instead of searching for a few forbidden syntax forms.
# A docstring may wrap a declaration, but may not hide executable interpolation.
function declarationonly(ex; inmodule=false)
    ex isa LineNumberNode && return true
    ex isa Expr || return false
    if ex.head in (:toplevel, :block)
        return all(arg -> declarationonly(arg; inmodule), ex.args)
    elseif ex.head === :module
        return !inmodule &&
               ex.args[2] === :Seams &&
               declarationonly(ex.args[3]; inmodule=true)
    elseif ex.head === :macrocall
        macro_name = ex.args[1]
        isdoc =
            macro_name === Symbol("@doc") ||
            (macro_name isa GlobalRef && macro_name.name === Symbol("@doc"))
        return isdoc &&
               length(ex.args) == 4 &&
               ex.args[3] isa String &&
               declarationonly(ex.args[4]; inmodule)
    elseif inmodule && ex.head === :function
        return length(ex.args) == 1 && ex.args[1] isa Symbol
    elseif inmodule && ex.head === :export
        return all(arg -> arg isa Symbol, ex.args)
    end
    return false
end

@testset "Architecture boundaries" begin
    @testset "Explicit sibling DAG, including nested modules" begin
        @test acyclic(PERMITTED)
        seen = Set{Symbol}()
        src = joinpath(@__DIR__, "..", "..", "src")
        for (folder, _, files) in walkdir(src), file in files
            endswith(file, ".jl") || continue
            path = joinpath(folder, file)
            relative = relpath(path, src)
            owner = ownerof(relative)
            # Only the package root and plotting declaration are outside the layers.
            if owner === nothing
                @test relative in
                    ("MLFMA.jl", joinpath("visualization", "visualization.jl"))
                continue
            end
            push!(seen, owner)
            @testset "$relative" begin
                @test isempty(
                    forbidden(owner, dependencies(Meta.parseall(read(path, String))))
                )
            end
        end
        @test seen == Set(keys(PERMITTED))

        for (owner, source, expected) in [
            (:Sampling, "using ..Assembly", :Assembly),
            (:FarFields, "import ..Operators: farfieldstorerule", :Operators),
            (:Equidistant, "f() = Assembly.assemble()", :Assembly),
            (:SphericalHarmonics, "import ...Translations", :Translations),
            (:CircularHarmonics, "using ...FarFields", :FarFields),
            (
                :InterpolationDiagnostics,
                "f() = TestSupport.dummyfarfieldvalue()",
                :TestSupport,
            ),
            (:Phase, "f() = \"value: \$(Assembly.assemble())\"", :Assembly),
            (:Sampling, "import ..MLFMA: assemble", :MLFMA),
        ]
            @test expected in forbidden(owner, dependencies(Meta.parseall(source)))
        end
        @test isempty(dependencies(Meta.parseall("f() = \"Assembly.assemble()\"")))
        @test isempty(
            forbidden(
                :Equidistant,
                dependencies(
                    Meta.parseall(
                        "import ..Interpolation; f(x) = Interpolation._stageprescaled!(x)"
                    ),
                ),
            ),
        )
        @test isempty(
            forbidden(
                :SphericalHarmonics, dependencies(Meta.parseall("import ...Sampling"))
            ),
        )
        @test !acyclic(Dict(:Sampling => Set([:Assembly]), :Assembly => Set([:Sampling])))
    end

    @testset "Seams is declaration-only" begin
        @test declarationonly(
            Meta.parseall(
                read(joinpath(@__DIR__, "..", "..", "src", "seams", "Seams.jl"), String)
            ),
        )
        @test declarationonly(
            Meta.parseall(
                "\"module docs\"\nmodule Seams\n\"generic docs\"\nfunction f end\nexport f\nend",
            ),
        )
        for body in [
            "f(x) = x",
            "function f(x)\nreturn x\nend",
            "function f() end",
            "struct Policy end",
            "abstract type Policy end",
            "const POLICY = 1",
            "POLICY = Dict()",
            "using ..Operators",
            "import ..Assembly: assemble",
            "include(\"implementation.jl\")",
            "eval(:(f(x) = x))",
            "@eval function f(x)\nx\nend",
            "begin\nf(x) = x\nend",
            "module Hidden\nf(x) = x\nend",
            "\"docs \$(runpolicy())\"\nfunction f end",
        ]
            @test !declarationonly(Meta.parseall("module Seams\n$body\nend"))
        end
    end
end
end
