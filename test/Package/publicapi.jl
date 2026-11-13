module PublicAPITest
using Test
using MLFMA

# The public API policy, stated as a test so it is enforced rather than described. See the header of
# `src/MLFMA.jl` for the policy itself; this file is the pin.
#
# Every name below must keep resolving as `MLFMA.name`. Removing one is allowed, but it is a
# deliberate API decision that has to be made here first, not a side effect of moving a file into a
# layer module.
#
# THIS LIST IS NOT THE WHOLE REACHABLE SURFACE, and the gap is deliberate. Many more names resolve as
# `MLFMA.name` than appear here: they are a compatibility surface, supported in practice and not
# promised. Absence from this list is therefore NOT a licence to unexport a name. Measured at the
# time of the layer split: 351 names are used as `MLFMA.name` outside `src`, of which 51 are listed
# below, and a further 82 exported names are used nowhere in the repository at all. None of those 82
# were removed, because "unused here" says nothing about whether a caller dispatches on it.
#
# Adding a name here is the way to promote it from compatibility to supported, and it should come
# with a user-facing reason for the group it belongs to rather than because it happened to be handy.

const PUBLICCONSTRUCTORS = [:GalerkinMLFMA, :PetrovGalerkinMLFMA, :assemble]

const PUBLICSTRATEGIES = [
    :AbstractMLFMAStrategy,
    :MLFMAStrategy,
    :TreeStrategy,
    :SamplingStrategy,
    :FarFieldStrategy,
    :LevelTransferStrategy,
    :I2OTranslationStrategy,
    :SparseBEASTNearInteractionsStrategy,
]

const PUBLICSAMPLING = [
    :AbstractSampling,
    :AbstractSphericalSampling,
    :SphericalSampling,
    :CircularSampling,
    :SphericalSamplingBuilder,
    :CircularSamplingBuilder,
    :Nθ,
    :Nφ,
    :θs,
    :φs,
    :cartesiansamplepoints,
    :sphericalsamplepoints,
]

const PUBLICINTERPOLATIONBUILDERS = [
    :AbstractInterpolationBuilder,
    :SphericalLagrangeInterpolationBuilder,
    :LocalθGlobalφInterpolationBuilder,
    :CircularInterpolationBuilder,
]

# Leaf storage and field side are separate axes and stay separate names, per the refactor plan.
const PUBLICLEAFSTORAGE = [
    :AbstractLeafStorage,
    :SampledLeafStorage,
    :SphericalHarmonicLeafStorage,
    :VectorSphericalHarmonicLeafStorage,
    :CircularHarmonicLeafStorage,
    :AbstractFieldCollectionSide,
    :RadiationSide,
    :ReceiveSide,
    :LazyReceiveSide,
]

const PUBLICFARFIELDS = [
    :AbstractFarField,
    :RadiationFieldCollection,
    :ReceiveFieldCollection,
    :LazyReceiveFieldCollection,
]

const PUBLICOPERATORPROPERTIES = [
    :farfieldtype,
    :farfieldstorerule,
    :receivecouplingrule,
    :radiationoperator,
    :receiveoperator,
    :interpolationtrait,
    :α,
    :γ,
]

const PUBLICSUBMODULES = [
    :Equidistant, :SphericalHarmonics, :CircularHarmonics, :InterpolationDiagnostics
]

const PUBLICNAMES = vcat(
    PUBLICCONSTRUCTORS,
    PUBLICSTRATEGIES,
    PUBLICSAMPLING,
    PUBLICINTERPOLATIONBUILDERS,
    PUBLICLEAFSTORAGE,
    PUBLICFARFIELDS,
    PUBLICOPERATORPROPERTIES,
    PUBLICSUBMODULES,
)

# A method or type defined inside `MLFMA.CircularHarmonics` is still MLFMA's own.
function belongstomlfma(mod::Module)
    mod === MLFMA && return true
    parent = parentmodule(mod)
    return parent !== mod && belongstomlfma(parent)
end

@testset "Public API" begin
    @testset "resolves" begin
        for name in PUBLICNAMES
            @test isdefined(MLFMA, name)
        end
    end

    @testset "is MLFMA's own" begin
        # A name resolving through `MLFMA.` proves nothing on its own, because the top level still
        # pulls external packages in with `using` and their exports come along. Every public name has
        # to be one MLFMA actually owns.
        for name in PUBLICNAMES
            isdefined(MLFMA, name) || continue
            @test belongstomlfma(parentmodule(getfield(MLFMA, name)))
        end

        # THE NEGATIVE CONTROL, and it has to name something the top level still imports. It used to
        # be `MLFMA.gamma`, which is SpecialFunctions'; task #175 removed nine top-level `using`
        # lines that served no call site at all, SpecialFunctions among them, so `MLFMA.gamma` no
        # longer resolves and the control asserted a name that was simply absent.
        #
        # `elementwisemul!` is H2Factory's and reaches call sites outside `src` through `MLFMA.`,
        # which is the same demonstration on a name that is load-bearing rather than incidental: it
        # cannot quietly disappear the way `gamma` did.
        @test isdefined(MLFMA, :elementwisemul!)
        @test !belongstomlfma(parentmodule(getfield(MLFMA, :elementwisemul!)))
    end

    @testset "export list" begin
        # Deliberately almost empty: the package is reached as `MLFMA.name`, so nothing above is
        # exported. `@memorydonut` is the exception, a macro whose whole point is to be written
        # unqualified at a call site.
        @test Set(names(MLFMA)) == Set([:MLFMA, Symbol("@memorydonut")])
    end

    @testset "no duplicate entries" begin
        @test length(unique(PUBLICNAMES)) == length(PUBLICNAMES)
    end
end
end
