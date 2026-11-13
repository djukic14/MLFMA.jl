
# TODO: Consider interpolating translation operators to save memory.
# TODO: Consider storing leaf far fields at a lower level, with a maximum translation level.

# Read-only cores used by the accessors below. Only the shared scalar type `T` is named; the other
# type parameters describe different fields in the two core types.
const AbstractMLFMAMap{T} = Union{GalerkinMLFMACore{T},PetrovGalerkinMLFMACore{T}}

# Constructors return workspaces over these cores. Restrict the alias to MLFMA cores because
# the accessors below read MLFMA-specific fields, not fields shared by every H2Factory workspace.
const MLFMAWorkspace{T} = Union{
    H2Factory.GalerkinH2Workspace{T,<:GalerkinMLFMACore},
    H2Factory.PetrovGalerkinH2Workspace{T,<:PetrovGalerkinMLFMACore},
}

"""
    NEARINTERACTIONSQUADSTRATEGYRECOM

The recommended BEAST quadrature strategy for near interactions.

A `DoubleNumWiltonSauterQStrat` for singular and near-singular double integrals over 3-D surface
elements. This recommendation suits 3-D Maxwell problems, not every geometry. The explicit type
checks the constant at load time. Use [`nearinteractionsquadstrategy`](@ref) to select the
operator-dependent default, including BEAST's quadrature for 2-D Helmholtz operators.
"""
const NEARINTERACTIONSQUADSTRATEGYRECOM::BEAST.DoubleNumWiltonSauterQStrat{Int64,Int64} = BEAST.DoubleNumWiltonSauterQStrat(
    10, 11, 6, 7, 5, 5, 4, 3
)
const FARFIELDQUADSTRATEGYRECOM::BEAST.SingleNumQStrat = BEAST.SingleNumQStrat(5)
const VERBOSERECOM::Bool = false
const NEARINTERACTIONSASSEMBLERRECOM = H2Factory.SparseBEASTNearInteractionsAssembler
const FARFIELDASSEMBLERBUILDERRECOM::Assembly.BEASTFarFieldAssemblerBuilder = Assembly.BEASTFarFieldAssemblerBuilder()
# Keep sampled storage as the default: the package's tolerances were calibrated against it.
# Harmonic storage saves memory in the sphere benchmarks, but needs a broader accuracy sweep
# before becoming the default.
const LEAFSTORAGERECOM::Assembly.SampledLeafStorage = Assembly.SampledLeafStorage()
# Store one translation per lattice symmetry orbit and recover the rest by exact grid permutations.
# Results may differ in rounding from the unreduced path. Non-lattice block-tree displacements
# trigger a warning and fallback to `TranslationCollection`.
# Sphere benchmarks show less retained memory but more transient allocation per matvec;
# see `benchmark/symmetrytranslations.jl` for the measured time and memory trade-offs.
const TRANSLATIONRECOM = Translations.SymmetryTranslationCollection
# Global φ interpolation is faster at the default stencil order in the scalar/tangential benchmarks;
# the advantage does not hold at small orders. See `benchmark/interpolation/`, group 10.
# The operator is complex: transpose uses `transposedinterpolator`, whereas adjoint uses
# anterpolation. `test/LevelTransfers/anterpolationcontract.jl` checks that distinction.
const INTERPOLATORECOM = Interpolation.LocalθGlobalφInterpolator
const PHASESHIFTERRECOM::Type{Phase.PhaseShifter} = Phase.PhaseShifter
const SAMPLINGBUILDERRECOM::Sampling.SphericalSamplingBuilder{Float64} = Sampling.SphericalSamplingBuilder{
    Float64
}()
const BUFFERBOXRECOM::Int = 1
const MINHALFSIZERECOM::Float64 = 0.25 #in λ
# Stop subdivision when either `minvalues` or `minhalfsize` is reached.
# This count avoids extra levels for boxes containing only a few basis functions.
const MINVALUESRECOM::Int = 100
const AGGREGATENODES::Function = H2Trees.istranslatingnode
const TRANSLATINGNODESRECOM::Function = H2Trees.TranslatingNodesIterator

# Read the I2O collection from a core or its workspace. Directed wrappers use H2Factory's accessors.
translationcollection(A::AbstractMLFMAMap) = A.i2otranslator
translationcollection(A::MLFMAWorkspace) = translationcollection(H2Factory.core(A))

# --- MLFMA's own accessors ------------------------------------------------------------------------
#
# Generate each accessor for both the core and its workspace.
for accessor in (:phaseshifter, :samplings, :γ, :ϵ, :interpolator, :anterpolator)
    @eval begin
        $accessor(A::AbstractMLFMAMap) = A.$accessor
        $accessor(A::MLFMAWorkspace) = $accessor(H2Factory.core(A))
    end
end

# Use accessors so the same level lookups work for cores and workspaces.
function levelindex(A, level::Int)
    return level - H2Factory.minrelevantlevel(A) + 1
end

function sampling(A, level::Int)
    return samplings(A)[levelindex(A, level)]
end

function parentsampling(A, level::Int)
    return samplings(A)[levelindex(A, level - 1)]
end

function weights(A, level::Int)
    return sphericalsamplepoints(sampling(A, level))[2]
end

# H2Factory's read-only protocol methods live with each core. Mutable state and scheduling
# belong to the workspace. H2Factory handles traversal directions and I2O node ordering.
