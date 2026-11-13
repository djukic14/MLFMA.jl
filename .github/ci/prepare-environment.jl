using Pkg
using TOML
using UUIDs

function ci_project(; root=dirname(dirname(@__DIR__)))
    project = TOML.parsefile(joinpath(root, "test", "Project.toml"))
    pop!(project, "sources", nothing)
    runtime = TOML.parsefile(joinpath(root, "Project.toml"))
    # Direct dependencies keep Pkg from borrowing paths from the package's manifest.
    merge!(project["deps"], runtime["deps"])
    compat = get!(project, "compat", Dict{String,Any}())
    for (name, bound) in runtime["compat"]
        if name == "julia" || haskey(project["deps"], name)
            get!(compat, name, bound)
        end
    end
    return project
end

function prepare_environment(; root=dirname(dirname(@__DIR__)))
    root = realpath(root)
    project = ci_project(; root)
    # Do not copy the developer manifest or sibling-checkout overrides into CI.
    environment = mktempdir(; cleanup=false)
    open(joinpath(environment, "Project.toml"), "w") do io
        return TOML.print(io, project; sorted=true)
    end
    Pkg.Registry.add("General")
    Pkg.Registry.update()
    Pkg.activate(environment)
    Pkg.develop(Pkg.PackageSpec(; path=root))
    Pkg.instantiate()

    dependencies = Pkg.dependencies()
    for info in values(dependencies)
        info.is_tracking_path &&
            info.name != "MLFMA" &&
            error("Unexpected path dependency: $(info.name) at $(info.source)")
    end
    info = dependencies[UUID(project["deps"]["MLFMA"])]
    info.is_tracking_path && realpath(info.source) == root ||
        error("MLFMA did not resolve to the intended checkout")
    println("MLFMA: ", info.source)
    for name in ("BEAST", "CompScienceMeshes", "H2Trees", "H2Factory", "H2FactoryTestUtils")
        matches = [info for info in values(dependencies) if info.name == name]
        length(matches) == 1 || error("Expected exactly one $name dependency")
        info = only(matches)
        info.is_tracking_registry || error("$name must resolve from a registry")
        println("$name: registry version ", info.version, ", tree ", info.tree_hash)
    end
    return environment
end
