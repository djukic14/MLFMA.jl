"""
    MLFMA.Phase

The phase a far field picks up when its reference point moves.

Defines phase factors, per-(level, sector) shifts, and level transfers that combine interpolation
with phase. `DummyPhaseShifter` supplies unit phases for tests through the same transfer path.
"""
module Phase

using H2Factory
using H2Trees
using LinearAlgebra

using ..FarFields
using ..Harmonics
using ..Interpolation
using ..Operators
using ..Sampling
using ..Seams

include("phasefactor.jl")
include("phaseshift.jl")
include("dummyphaseshifter.jl")

# Define transfer types before the collection builder that constructs them.
include("interpolationphasetransfer.jl")
include("interpolationphasecollection.jl")

export AdjointInterpolationPhaseTransfer,
    ConjugatedInterpolationPhaseTransfer,
    DummyPhaseShifter,
    InterpolationPhaseI2I,
    InterpolationPhaseTransfer,
    PhaseFunctor,
    PhaseShifter,
    TransposedInterpolationPhaseTransfer,
    appendphaseshifts!,
    dotu,
    findoppositephaseshift,
    findphaseshift,
    interpolationphasecollections,
    numberofsectors,
    phasefactor,
    phasefactor!,
    shifttochildphase!,
    shifttoparentphase!

end
