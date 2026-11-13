# A test phase shifter that isolates interpolation from re-referencing.
# It lives under `src/` because the level-transfer implementation dispatches on this type.

"""
    DummyPhaseShifter()

A phase shifter that applies no phase at all.

Level transfers obtain unit phases through `_phasesatlevel` in `interpolationphasecollection.jl`.
This type does not implement `shifttoparentphase!` or `shifttochildphase!`.
"""
struct DummyPhaseShifter end

# Same positional signature as `PhaseShifter(γ, tree, relevantlevels, samplings)`, so
# `MLFMA.phaseshiftertype(strategy)(...)` can construct either. Every argument is ignored.
# Accept any angular sampling, including circular grids, so tests can isolate the phase cost in 2-D.
function DummyPhaseShifter(
    γ::T, tree, relevantlevels, samplings::AbstractVector{<:Sampling.AbstractSampling}
) where {T<:Number}
    return DummyPhaseShifter()
end
