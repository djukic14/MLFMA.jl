module LayersTest
using Test
using MLFMA

# The layer modules, and the one invariant that keeps splitting the package into them safe.
#
# When a layer exports a name and a file further up still defines methods for it, `using .Layer` at
# the top level is NOT enough. Julia resolves the later `function name(...)` to a NEW top-level
# binding that shadows the import. Nothing errors, the package precompiles, and the layer's methods
# simply stop being reachable through `MLFMA.name`. The fix is `import .Layer: name`, which makes
# the later definition extend the layer's function instead.
#
# Measured, not hypothetical: `Nθ`, `Nφ`, `mirrorindices`, `cutoffparameter` and `supportedsymmetry`
# each silently dropped to a single method the first time `Sampling` was split out this way.
const LAYERS = [
    MLFMA.Seams,
    MLFMA.Sampling,
    MLFMA.Harmonics,
    MLFMA.FarFields,
    MLFMA.Interpolation,
    MLFMA.Operators,
    MLFMA.Phase,
    MLFMA.Translations,
    MLFMA.Assembly,
    MLFMA.TestSupport,
]

"""
    shadowedexports(layer)

Names `layer` exports whose binding in `MLFMA` is not the layer's own. A non-empty result means the
top level forked the name instead of re-exposing it.
"""
function shadowedexports(layer::Module)
    shadowed = Symbol[]
    for name in names(layer)
        name === nameof(layer) && continue
        isdefined(MLFMA, name) || (push!(shadowed, name); continue)
        getfield(MLFMA, name) === getfield(layer, name) || push!(shadowed, name)
    end

    return shadowed
end

@testset "Layer modules" begin
    @testset "exports are not shadowed at the top level" begin
        for layer in LAYERS
            @test shadowedexports(layer) == Symbol[]
        end
    end

    @testset "split generics keep every method" begin
        # Each of these is defined in the layer AND extended further up. Both sides have to live on
        # one function, so assert the method list spans more than the layer's own files.
        for (name, layerfile) in [
            (:Nθ, "abstractsampling.jl"),
            (:Nφ, "abstractsampling.jl"),
            (:mirrorindices, "sphericalsampling.jl"),
            (:cutoffparameter, "sphericalsampling.jl"),
            (:supportedsymmetry, "symmetryaction.jl"),
        ]
            files = unique([
                basename(string(m.file)) for m in methods(getfield(MLFMA, name))
            ])
            @test layerfile in files
            @test length(files) > 1
        end
    end

    @testset "re-exported child submodules keep their identity" begin
        # `Harmonics` wraps two modules that were already standalone. Nesting them one level deeper
        # must not change what `MLFMA.SphericalHarmonics` means, and both had to have their
        # `import ..MLFMA` deepened to `import ...MLFMA` to survive the extra level.
        @test MLFMA.SphericalHarmonics === MLFMA.Harmonics.SphericalHarmonics
        @test MLFMA.CircularHarmonics === MLFMA.Harmonics.CircularHarmonics
        @test parentmodule(MLFMA.SphericalHarmonics) === MLFMA.Harmonics
    end

    @testset "explicit private dependencies retain compatibility bindings" begin
        for name in (
            :_Adjointed,
            :_aggregatecircularconjugated!,
            :_aggregateconjugated!,
            :_applycircularcoupling!,
            :_applycoupling!,
            :_chescatter!,
            :_lazyreceivescatter!,
            :_projectinto!,
            :_shescatter!,
        )
            @test getfield(MLFMA.Assembly, name) === getfield(MLFMA.FarFields, name)
            @test getfield(MLFMA, name) === getfield(MLFMA.FarFields, name)
        end
        @test MLFMA.TestSupport._dimensionvalue === MLFMA.Assembly._dimensionvalue
        @test MLFMA._dimensionvalue === MLFMA.Assembly._dimensionvalue
    end

    @testset "the check can fail" begin
        # A negative control for `shadowedexports`: a module whose export MLFMA does not have at all
        # must be reported, otherwise the assertions above prove nothing.
        @test shadowedexports(Base) != Symbol[]
    end
end
include("architecture.jl")
end
