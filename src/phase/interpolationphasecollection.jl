# Build one level-transfer collection per side, even for a Galerkin map with a single tree.
# H2Factory derives disaggregation by transposing/adjointing the disaggregation side's I2I family
# (`disaggregationi2itranslator(A, ::isForward) == testi2itranslator(A)`, applied with
# `TransposeTranslation()`). Transpose leaves a diagonal unchanged, so the radiation/receive
# conjugation must be stored per side. A single collection cannot serve all four
# `{trial,test}{i2i,o2o}translator` slots. `test/LevelTransfers/phaseconvention.jl` checks this convention.
#
# These collections do not declare scratch through `H2Factory.i2ichannels`/`o2ochannels`.
# The map exposes level-keyed `scratchi2ichannel`/`scratcho2ochannel` buffers shaped for `FarField`
# moments, so the collections need no separate buffer declarations.

"""
    interpolationphasecollections(phaseshifter, interpolator, anterpolator, samplings,
                                  minrelevantlevel, levels, interpolationtrait; dims=3)

The `(trial, test)` pair of `H2Factory.I2ITranslationCollection`s for MLFMA's level transfers.

Each stores one [`InterpolationPhaseTransfer`](@ref) per `(child level, child sector)`, resolved by
H2Factory's `leveltransfer` as `transfers[level(tree, child)][transferkey(selector, ...)]` -- so the
outer key is the child's own level and the inner key is its **1-based** sector (`H2Trees.sector` is
0-based; `SectorSelector`'s `transferkey` adds one).

`phaseshifter` supplies the per-level, per-sector phases, already on the parent grid. The trial
collection stores them as-is (radiation convention) and the test collection stores their conjugates
(receive convention).

Only levels strictly deeper than `minrelevantlevel` get transfers, matching `PhaseShifter`: a level
transfer needs a parent, and the shallowest relevant level has none within range.
"""
function interpolationphasecollections(
    phaseshifter,
    interpolator,
    anterpolator,
    samplings,
    minrelevantlevel::Int,
    levels,
    interpolationtrait;
    dims::Int=3,
)
    selector = H2Factory.SectorSelector{dims}()
    trait = InterpolationPhaseI2I()

    childlevels = [level for level in levels if level > minrelevantlevel]
    if isempty(childlevels)
        return (
            H2Factory.I2ITranslationCollection(nothing, selector, trait),
            H2Factory.I2ITranslationCollection(nothing, selector, trait),
        )
    end

    # Build the first transfer eagerly so both dictionaries can be concretely typed.
    firstlevel = first(childlevels)
    firstchildsampling = _samplingatlevel(samplings, firstlevel, minrelevantlevel)
    firstparentsampling = _samplingatlevel(samplings, firstlevel - 1, minrelevantlevel)
    firstphases = _phasesatlevel(phaseshifter, firstlevel, firstparentsampling)
    firstphase = firstphases[first(keys(firstphases))]
    prototype = InterpolationPhaseTransfer(
        interpolator,
        anterpolator,
        firstphase,
        firstchildsampling,
        firstparentsampling,
        interpolationtrait,
    )
    TransferType = typeof(prototype)

    trialtransfers = Dict{Int,Dict{Int,TransferType}}()
    testtransfers = Dict{Int,Dict{Int,TransferType}}()

    for level in childlevels
        childsampling = _samplingatlevel(samplings, level, minrelevantlevel)
        parentsampling = _samplingatlevel(samplings, level - 1, minrelevantlevel)
        phases = _phasesatlevel(phaseshifter, level, parentsampling)

        trialatlevel = Dict{Int,TransferType}()
        testatlevel = Dict{Int,TransferType}()

        for (sector, phase) in phases
            key = sector + 1 # `SectorSelector`'s `transferkey` convention
            trialatlevel[key] = InterpolationPhaseTransfer(
                interpolator,
                anterpolator,
                phase,
                childsampling,
                parentsampling,
                interpolationtrait,
            )
            testatlevel[key] = InterpolationPhaseTransfer(
                interpolator,
                anterpolator,
                conj.(phase),
                childsampling,
                parentsampling,
                interpolationtrait,
            )
        end

        trialtransfers[level] = trialatlevel
        testtransfers[level] = testatlevel
    end

    return (
        H2Factory.I2ITranslationCollection(trialtransfers, selector, trait),
        H2Factory.I2ITranslationCollection(testtransfers, selector, trait),
    )
end

# `PhaseShifter` indexes its per-level dictionaries by `level - minlevel` (see `_getlevelindex`) and
# keys each by the 0-based sector.
function _phasesatlevel(phaseshifter::PhaseShifter, level::Int, parentsampling)
    return phaseshifter.phaseshifts[_getlevelindex(level, phaseshifter.minlevel)]
end

# Unit phases make dummy transfers interpolation-only. Supply every sector of a `2^3`-tree.
# Tests still multiply by ones, using the same application path as real phases.
function _phasesatlevel(::DummyPhaseShifter, level::Int, parentsampling)
    identityphase = ones(complex(eltype(parentsampling)), size(parentsampling))
    return Dict(sector => identityphase for sector in 0:7)
end

# `samplings` spans `relevantlevels`, so level `l` sits at `l - minrelevantlevel + 1`.
function _samplingatlevel(samplings, level::Int, minrelevantlevel::Int)
    return samplings[level - minrelevantlevel + 1]
end
