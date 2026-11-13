using Test
include("prepare-environment.jl")

@testset "Clean CI project" begin
    root = dirname(dirname(@__DIR__))
    runtime = TOML.parsefile(joinpath(root, "Project.toml"))
    testfile = joinpath(root, "test", "Project.toml")
    original = read(testfile, String)
    manifestfile = joinpath(root, "test", "Manifest.toml")
    manifest = isfile(manifestfile) ? read(manifestfile) : nothing
    project = ci_project(; root)
    @test !haskey(project, "sources")
    @test project["deps"] == merge(TOML.parse(original)["deps"], runtime["deps"])
    @test project["compat"]["julia"] == runtime["compat"]["julia"]
    @test project["deps"]["MLFMA"] == runtime["uuid"]
    @test all(
        name -> name == "julia" || haskey(project["deps"], name), keys(project["compat"])
    )
    @test project["compat"]["JuliaFormatter"] == "=2.12.4"
    for name in ("H2Factory", "H2FactoryTestUtils", "H2Trees")
        @test haskey(project["deps"], name)
        @test project["compat"][name] == runtime["compat"][name]
    end
    @test read(testfile, String) == original
    @test manifest === nothing || read(manifestfile) == manifest

    @testset "An explicit test bound is preserved" begin
        mktempdir() do fixture
            mkpath(joinpath(fixture, "test"))
            cp(joinpath(root, "Project.toml"), joinpath(fixture, "Project.toml"))
            testproject = TOML.parse(original)
            testproject["compat"]["BEAST"] = "2.9"
            open(joinpath(fixture, "test", "Project.toml"), "w") do io
                TOML.print(io, testproject)
            end
            @test ci_project(; root=fixture)["compat"]["BEAST"] == "2.9"
            @test !haskey(ci_project(; root=fixture), "sources")
        end
    end
end
