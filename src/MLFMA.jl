module MLFMA

# Keep these bindings reachable for existing qualified calls. They are not public API.
using Dictionaries
using H2Factory
import H2Factory: applyconjugatedleafmul!, ConjugatedCollectionMul

# Public names are used as `MLFMA.name`; only `@memorydonut` is exported.
# `test/Package/publicapi.jl` lists the supported API. Other reachable names may be used
# by existing callers, so removing them still needs a compatibility review.
# Names beginning with `_` are private to their layer.

# Load layers in dependency order. Each layer imports its own dependencies and exports
# the names that should remain reachable through `MLFMA`.

# Shared declarations let lower layers call methods supplied by higher ones.
include("seams/Seams.jl")
using .Seams
# Explicit imports keep later methods attached to these generics.
import .Seams: farfieldtype, radiationfieldcollectiontype, receivecouplingrule

include("sampling/Sampling.jl")
using .Sampling
import .Sampling: Nθ, Nφ, cutoffparameter, mirrorindices, supportedsymmetry

# Spherical and circular harmonics use the sampling grids.
include("harmonics/Harmonics.jl")
using .Harmonics

# Far-field representations, stored leaf patterns and multiply kernels.
# Collection builders belong to Assembly below.
include("farfield/FarFields.jl")
using .FarFields
import .FarFields: operator

# Interpolation, anterpolation and their diagnostics.
include("interpolation/Interpolation.jl")
using .Interpolation
import .Interpolation: anterpolator, interpolationtrait, interpolator, interpolatortype

# Equidistant angular grids with FFT and local-Lagrange interpolation.
include("equidistant/Equidistant.jl")

# Operator properties, far-field wrappers and storage rules.
include("operators/Operators.jl")
using .Operators
import .Seams: farfieldtype
import .Operators: farfieldstorerule, farfieldvaluetype, γ

# Phase shifts and transfers between levels.
include("phase/Phase.jl")
using .Phase

# Translations between boxes at the same level.
include("translation/Translations.jl")
using .Translations

# Assemblers, collection builders, strategies, cores and map constructors.
include("assembly/Assembly.jl")
using .Assembly
# Resolve the name explicitly: BEAST also exports `assemble`.
import .Assembly: assemble

# Test helpers depend on Assembly, not the other way around.
include("testsupport/TestSupport.jl")
using .TestSupport

# The plotting extension implements this top-level declaration.
include("visualization/visualization.jl")

export @memorydonut
end # module MLFMA
