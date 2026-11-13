"""
    MLFMA.Assembly

Build MLFMA maps from operators, spaces and trees.

This top layer owns the BEAST-facing far-field assembler, store logic, collection builders,
`MLFMAStrategy` family, Galerkin and Petrov cores, and map constructors.

Collection builders choose the assembler, collection family, side and storage. The collection
types and their multiply kernels belong to `FarFields`, which does not depend on this layer.

`dummyfarfieldassembler.jl` belongs to `TestSupport`: it uses `BoxFarFieldSampling` and `leafids`
from this layer, but this layer does not depend on it.
"""
module Assembly

# H2Factory builds the sparse near-interaction matrix; this layer does not need `SparseArrays`.
# Harmonic builders call `SphericalHarmonics` for Gauss-Legendre grids.
using BEAST
using CompScienceMeshes
using Dictionaries
using H2Factory
using H2Trees
using LinearAlgebra
using OhMyThreads
using StaticArrays

using ..FarFields
using ..Harmonics
using ..Interpolation
using ..Operators
using ..Sampling
using ..Seams
using ..Phase
using ..Translations

# Internal leaf kernels are dependencies, not part of this layer's public interface.
# Keep FarFields' exports intact for existing qualified MLFMA calls.
import ..FarFields:
    _Adjointed,
    _aggregatecircularconjugated!,
    _aggregateconjugated!,
    _applycircularcoupling!,
    _applycoupling!,
    _chescatter!,
    _lazyreceivescatter!,
    _projectinto!,
    _shescatter!

# Import functions owned by other layers so local methods extend the intended functions.
# `test/Package/layers.jl` checks these bindings.
import ..FarFields: operator
# `interpolator` and `anterpolator` are Interpolation's; `AbstractMLFMAMatrix.jl` gives the
# workspace accessors the same names, and under `using` alone those definitions fork rather
# than extend, so `MLFMA.interpolator(workspace)` raises a MethodError at matvec time.
import ..Interpolation: anterpolator, interpolator, interpolatortype
# Same for `γ`: `AbstractMLFMAMatrix.jl` adds a workspace method to Operators' generic.
import ..Operators: farfieldstorerule, γ
import ..Sampling: cutoffparameter
import ..Seams: farfieldtype

# H2Factory owns the adjoint leaf multiply; `adjointleafmul.jl` specializes it.
import H2Factory: applyconjugatedleafmul!, ConjugatedCollectionMul

# Files are grouped by responsibility:
#
#     beast/       what talks to BEAST
#     store/       the store rule machinery
#     builders/    construction policy for leaf field collections
#     strategies/  the user-facing strategy objects
#     cores/       the maps themselves and their multiply
#
# BEAST-facing assembly and the store logic.
include("beast/beastfarfieldassembler.jl")
include("store/store.jl")

# The field-collection builders. After the assembler, whose `leafids` and `BoxFarFieldSampling` they
# read, and after the operator property tables they consult to pick a collection family.
include("builders/fieldcollections.jl")
include("builders/collectionbuilders.jl")
# The compressed builders, after the sampled one whose level grouping and clamp check they reuse.
include("builders/shecollectionbuilder.jl")
include("builders/vshcollectionbuilder.jl")
include("builders/circularharmoniccollectionbuilder.jl")

# Map construction proper.
include("cores/utilsmaps.jl")
include("strategies/strategies.jl")
include("cores/GalerkinMLFMA.jl")
include("cores/PetrovGalerkinMLFMA.jl")
include("cores/AbstractMLFMAMatrix.jl")
include("cores/adjointleafmul.jl")

# `AbstractAssembledFieldCollectionBuilder` owns an assembler; the lazy builder does not.
export AbstractAssembledFieldCollectionBuilder, AbstractFieldCollectionBuilder

export AGGREGATENODES,
    BUFFERBOXRECOM,
    FARFIELDASSEMBLERBUILDERRECOM,
    FARFIELDQUADSTRATEGYRECOM,
    LEAFSTORAGERECOM,
    MINHALFSIZERECOM,
    MINVALUESRECOM,
    NEARINTERACTIONSQUADSTRATEGYRECOM,
    PHASESHIFTERRECOM,
    RadiationFieldCollection,
    RadiationFieldCollections,
    ReceiveFieldCollection,
    SAMPLINGBUILDERRECOM,
    TRANSLATINGNODESRECOM,
    phaseshifter,
    samplings

# Public construction types and helpers exposed through `MLFMA`.
export AbstractFieldCollectionSide,
    AbstractLeafStorage,
    AbstractMLFMAStrategy,
    AssembleMomentsFunctor,
    BEASTFarFieldAssembler,
    BEASTFarFieldAssemblerBuilder,
    BEASTFieldCollectionBuilder,
    BoxFarFieldSampling,
    BufferFunctor,
    CircularHarmonicFieldCollectionBuilder,
    CircularHarmonicLeafStorage,
    ConstantClusterData,
    FarFieldStorer,
    FarFieldStrategy,
    GalerkinMLFMA,
    GalerkinMLFMACore,
    I2OTranslationStrategy,
    INTERPOLATORECOM,
    InterimScratchFunctor,
    LazyReceiveFieldCollectionBuilder,
    LazyReceiveSide,
    LevelTransferStrategy,
    MLFMAStrategy,
    NEARINTERACTIONSASSEMBLERRECOM,
    PetrovGalerkinMLFMA,
    PetrovGalerkinMLFMACore,
    RadiationSide,
    ReceiveSide,
    SHEANALYSISMARGIN,
    SampledLeafStorage,
    SamplingFunctor,
    SamplingSizeFunctor,
    SamplingStrategy,
    SparseBEASTNearInteractionsStrategy,
    SphericalFrameCache,
    SphericalHarmonicFieldCollectionBuilder,
    SphericalHarmonicLeafStorage,
    StorageFieldsFunctor,
    TRANSLATIONRECOM,
    TreeStrategy,
    VSHFieldCollectionBuilder,
    VectorSphericalHarmonicLeafStorage,
    _dimensionvalue,
    _indexchunks,
    _indexinleafbox,
    _leaflevelindex,
    _leaflevels,
    _mirrorindicesbylevel,
    _removeradialcomponent,
    _samplinglevelindex,
    assemble,
    assemblerbuilder,
    assertsingleleafstorage,
    buildfarfieldassembler,
    buildfieldcollection,
    buildfieldcollections,
    checkboxsizes,
    checollections,
    cheorders,
    chescratch,
    chescratchchannel,
    collectiontype,
    defaultleafmulscheduler,
    diameter,
    executionschedulingfor,
    farfieldassemblerbuilder,
    farfieldkernelvals,
    farfieldlocal!,
    farfieldquadstrategy,
    farfieldstrategy,
    fieldcollectionconstructors,
    fieldoperator,
    fields,
    framecachefor,
    getfieldcollectionsdict,
    getsamplingsparentsamplings,
    getweights,
    i2ostrategy,
    interpolatorrecom,
    lazyreceivefieldcollectionbuilder,
    leaffarfieldsamplings,
    leafids,
    leafscratchchannel,
    leavesperchunk,
    levelindex,
    levelprotrusions,
    leveltransferstrategy,
    minhalfsize,
    minvalues,
    nearfieldstrategy,
    nearfieldstrategyfor,
    nearinteractionsquadstrategy,
    nearinteractionsstrategy,
    nearinteractionsstrategyfor,
    needsframecache,
    parentsampling,
    phaseshiftertype,
    protrudedhalfsize,
    protrusionaware,
    quadstrategy,
    radiationfieldcollectionbuilder,
    receivefieldcollectionbuilder,
    sampling,
    samplingbuilder,
    samplingbuilderrecom,
    samplingfunction,
    samplingstrategy,
    scheduler,
    scratchi2ichannel,
    scratcho2ochannel,
    sheanalysisorder,
    shecollections,
    sheorders,
    shescratch,
    shescratchchannel,
    side,
    space,
    testleafstorage,
    translationcollection,
    translationcollectiontype,
    treestrategy,
    trialleafstorage,
    vshanalysisorder,
    vshcollections,
    vshorders,
    vshscratch,
    vshscratchchannel,
    weights,
    γκ,
    λ,
    ϵ

end
