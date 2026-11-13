module H2FactoryInterfaceTest
using Test
using MLFMA
using H2Factory

# MLFMA participates in H2Factory by adding methods to H2Factory's own functions, spread over a dozen
# files. This pins BOTH the set of functions extended and HOW MANY methods each carries.
#
# The count matters, and an earlier version of this file did not have it. Pinning only the function
# set catches a whole extension point disappearing, but not a deleted specialization: drop the
# `applyleafmul!` method for one collection type and the name is still in the set, so the test stays
# green while that collection silently falls back to a generic path. The layer split moved all 179
# of these methods between modules, which is exactly when a lost specialization would have been
# easiest to cause and hardest to notice.
#
# Both halves are derived from the loaded package rather than written by hand. The literals below are
# the reviewed snapshot they are compared against, so a deliberate change is a one-line diff here and
# an accidental one is a failure.
const EXTENDEDH2FACTORYMETHODS = Dict(
    :applydirectedtransfer! => 4,
    :presenttransfer => 1,
    :aggregateleafchunk! => 3,
    :applyconjugatedleafmul! => 10,
    :applyleafmul! => 25,
    :buildi2ichannels => 2,
    :buildi2ochannel => 2,
    :buildleafchannels => 2,
    :buildmoments => 2,
    :buildnearinteractionsassembler => 1,
    :buildo2ochannels => 2,
    :buildvectorbuffers => 2,
    :conjmul! => 84,
    :i2otranslator => 3,
    :isgalerkinsymmetric => 2,
    :leafmul => 2,
    :leafmulscheduler => 2,
    :minrelevantlevel => 2,
    :operator => 2,
    :schedulinglevels => 2,
    :buildstorer => 2,
    :testaggregationplan => 2,
    :testdisaggregationplan => 2,
    :testi2itranslator => 2,
    :testmomentcollections => 2,
    :trialaggregationplan => 2,
    :trialdisaggregationplan => 2,
    :triali2itranslator => 2,
    :trialmomentcollections => 2,
    :α => 6,
)

# A method defined inside `MLFMA.Assembly` counts as MLFMA's, so walk up to the root module rather
# than comparing against `MLFMA` directly.
function belongstomlfma(mod::Module)
    mod === MLFMA && return true
    parent = parentmodule(mod)
    return parent !== mod && belongstomlfma(parent)
end

"""
    ish2factorygeneric(name, value) -> Bool

Whether `value` is a generic function H2Factory OWNS, as opposed to one it merely re-exports.

Two owners are valid, and they are named EXPLICITLY rather than accepted by walking the module tree.
`H2Factory.Interfaces` holds the declared extension protocol, so a generic migrated there is still
H2Factory's and must keep being counted; `H2Factory` itself owns everything not yet migrated.
Accepting any H2Factory submodule would weaken this: an implementation helper in some future
submodule would then be counted as protocol.

Only the function's canonical name counts; compiler-generated aliases must not count it twice.
"""
function ish2factorygeneric(name::Symbol, value)
    value isa Function || return false
    name === nameof(value) || return false
    return parentmodule(value) in (H2Factory, H2Factory.Interfaces)
end

"""
    extendedh2factorymethods()

Every function owned by `H2Factory` that MLFMA adds methods to, mapped to how many it adds.
"""
function extendedh2factorymethods()
    extended = Dict{Symbol,Int}()
    # `imported=true` IS REQUIRED, and its absence is silent. `names(M; all=true)` lists names
    # DEFINED in M and omits ones it imported, so every generic that moves into
    # `H2Factory.Interfaces` disappears from this enumeration and reads as zero methods. The
    # ownership predicate cannot save it: the name never reaches the predicate at all.
    #
    # It also admits H2Factory's imports from other packages (`AggregatePlan` and friends from
    # H2Trees), which is exactly what `ish2factorygeneric` then discards. The two work as a pair:
    # the keyword widens the enumeration to everything reachable, the predicate narrows it to what
    # H2Factory owns.
    for name in names(H2Factory; all=true, imported=true)
        isdefined(H2Factory, name) || continue
        value = getfield(H2Factory, name)
        # Skips functions H2Factory only re-exports: an extension of those is somebody else's seam.
        ish2factorygeneric(name, value) || continue
        count = sum(method -> belongstomlfma(method.module), methods(value); init=0)
        count > 0 && (extended[name] = count)
    end

    return extended
end

@testset "H2Factory extension points" begin
    extended = extendedh2factorymethods()

    @testset "compiler aliases are not extension points" begin
        @test ish2factorygeneric(:buildleafchannels, H2Factory.buildleafchannels)
        @test !ish2factorygeneric(Symbol("#212#val"), H2Factory.buildleafchannels)
        @test !ish2factorygeneric(:buildleafchannels, nothing)
    end

    @testset "the set of extension points" begin
        # Split rather than a single equality so a failure names what changed.
        # `collect` matters: `setdiff` over two `KeySet`s returns a `Set`, which is never
        # `== Symbol[]` even when empty, so without it these two assertions fail on a green tree.
        @test sort(collect(setdiff(keys(extended), keys(EXTENDEDH2FACTORYMETHODS)))) ==
            Symbol[]
        @test sort(collect(setdiff(keys(EXTENDEDH2FACTORYMETHODS), keys(extended)))) ==
            Symbol[]
    end

    @testset "the method count of each" begin
        # This is what catches a lost specialization rather than a lost function.
        for name in sort(collect(keys(EXTENDEDH2FACTORYMETHODS)); by=string)
            @test get(extended, name, 0) == EXTENDEDH2FACTORYMETHODS[name]
        end
        @test sum(values(extended)) == sum(values(EXTENDEDH2FACTORYMETHODS))
    end

    @testset "the enumeration is not vacuous" begin
        @test !isempty(extended)
        @test :applyleafmul! in keys(extended)
        # The busiest seam, and the one where a dropped specialization would be quietest.
        @test extended[:applyleafmul!] > 10
    end

    @testset "MLFMA adds no applycoupling! method" begin
        # `H2Factory.applycoupling!` is ORCHESTRATION: one method, in H2Factory, that owns the
        # `assertnosymmetry` guard before delegating to `applycouplingkernel!`. Dispatch cannot
        # prevent a downstream method at a concrete collection type from shadowing it and silently
        # skipping the refusal -- that is exactly what the former direction-named primitives allowed
        # (#172) -- so what is enforceable is that no such method exists, and that is pinned rather
        # than assumed.
        #
        # This says nothing about whether MLFMA participates in the I2O protocol; it says that if it
        # ever does, it does so through the kernel. The set/count testsets above are what notice the
        # participation itself.
        @test !any(m -> belongstomlfma(m.module), methods(H2Factory.applycoupling!))
        @test :applycoupling! ∉ keys(extended)
    end
end
end
