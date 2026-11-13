# Test assembler that skips far-field quadrature. Deterministic values vary with component, sample
# point and basis function, exposing transpose, axis-swap and trial/test collection mistakes.
# Complex values also expose missing conjugation; a matrix of ones would hide these errors.

# As in `BEASTFarFieldAssembler`, derive the component count from the target type through `fielddim`.
"""
    DummyFarFieldAssembler{OperatorType,Space,QuadStrategyType,StoreRule,F,D}

A dummy far-field assembler for tests: it performs no quadrature, and instead fills every `Fθ`/`Fφ`
entry with a deterministic value from [`dummyfarfieldvalue`](@ref).

Values are deterministic and RNG-free, and vary across all three indices to expose collection errors.

# Fields

  - `operator`: Used only for its scalar type.
  - `space`: Not used.
  - `quadstrategy`: Not used.
  - `storerule`: Not used -- this assembler writes `dummyfarfieldvalue` blocks directly rather than
    going through `H2Factory.buildstorer`. Carried only so it mirrors `BEASTFarFieldAssembler`'s shape
    and can be built by the same five-argument assembler-builder call.
"""
struct DummyFarFieldAssembler{OperatorType,Space,QuadStrategyType,StoreRule,F,D}
    operator::OperatorType
    space::Space
    quadstrategy::QuadStrategyType
    storerule::StoreRule
    farfieldtype::Type{F}
    dimensionality::Val{D}
    function DummyFarFieldAssembler(
        operator::OperatorType,
        space::Space,
        quadstratrategy::QuadStrategyType,
        storerule::StoreRule,
        ::Type{F},
    ) where {OperatorType,Space,QuadStrategyType,StoreRule,F<:AbstractFarField}
        D = fielddim(F)
        return new{OperatorType,Space,QuadStrategyType,StoreRule,F,D}(
            operator, space, quadstratrategy, storerule, F, Val(D)
        )
    end
end

function farfieldtype(assembler::DummyFarFieldAssembler)
    return assembler.farfieldtype
end

"""
    DummyFarFieldAssemblerBuilder()

The test counterpart of [`BEASTFarFieldAssemblerBuilder`](@ref MLFMA.Assembly.BEASTFarFieldAssemblerBuilder).
Builds a [`DummyFarFieldAssembler`](@ref) from the same five arguments:

```julia
FarFieldStrategy(; assemblerbuilder=MLFMA.DummyFarFieldAssemblerBuilder())
```

Pass a builder instance, not the assembler type.
"""
struct DummyFarFieldAssemblerBuilder end

function (::DummyFarFieldAssemblerBuilder)(
    operator, space, quadstrategy, storerule, farfieldtype
)
    return DummyFarFieldAssembler(operator, space, quadstrategy, storerule, farfieldtype)
end

"""
    dummyfarfieldvalue(T, component, samplepoint, basisfunction)

One deterministic pattern entry that varies with all three indices.

The pattern exposes transposes, swapped sample/basis axes and trial/test collection swaps.
Complex `T` also makes missing conjugation visible; real `T` uses a real fallback.
Values stay bounded on the scale of one to avoid extreme magnitudes during leaf aggregation.
"""
function dummyfarfieldvalue(
    ::Type{T}, component::Int, samplepoint::Int, basisfunction::Int
) where {T<:Complex}
    amplitude = 1 + 0.25 * cospi((samplepoint + 2 * basisfunction + component) / 17)
    phase = 0.7 * samplepoint + 1.3 * basisfunction + 0.31 * component
    return T(amplitude * cis(phase))
end

function dummyfarfieldvalue(
    ::Type{T}, component::Int, samplepoint::Int, basisfunction::Int
) where {T<:Real}
    amplitude = 1 + 0.25 * cospi((samplepoint + 2 * basisfunction + component) / 17)
    return T(amplitude * cospi(0.7 * samplepoint + 1.3 * basisfunction + 0.31 * component))
end

# Fill one `(nsamplepoints, nbasisfunctions)` block for component `d`. `basisoffset` shifts the basis
# index so that two leaves holding different functions get different patterns -- without it every leaf
# would repeat the same block and a mixed-up leaf/node association would stay invisible.
function _dummyfarfieldblock(
    ::Type{T}, d::Int, nsamplepoints::Int, nbasisfunctions::Int, basisoffset::Int
) where {T}
    F = Matrix{T}(undef, nsamplepoints, nbasisfunctions)
    for n in 1:nbasisfunctions, p in 1:nsamplepoints
        F[p, n] = dummyfarfieldvalue(T, d, p, n + basisoffset)
    end
    return F
end

function operator(assembler::TestSupport.DummyFarFieldAssembler)
    return assembler.operator
end

function space(assembler::TestSupport.DummyFarFieldAssembler)
    return assembler.space
end

function quadstrategy(assembler::TestSupport.DummyFarFieldAssembler)
    return assembler.quadstrategy
end

function (assembler::DummyFarFieldAssembler)(sampling; verbose::Bool=false)
    D = _dimensionvalue(assembler.dimensionality)
    T = scalartype(assembler.operator)
    return SVector{D,Matrix{T}}([
        _dummyfarfieldblock(T, d, length(sampling), numfunctions(assembler.space), 0) for
        d in 1:D
    ],)
end

# Use each leaf's `BoxFarFieldSampling` to size its block for that level's grid.
# The generic method below would mistake the number of leaf samplings for the direction count.
function (assembler::DummyFarFieldAssembler)(
    tree, leafsamplings::AbstractVector{<:BoxFarFieldSampling}; verbose::Bool=false
)
    D = _dimensionvalue(assembler.dimensionality)
    # Match the production assembler's block order from `leafsamplings`.
    leaves = leafids(leafsamplings)

    Fs = Vector{SVector{D,Matrix{scalartype(assembler.operator)}}}(undef, length(leaves))

    T = scalartype(assembler.operator)
    for (i, leaf) in enumerate(leaves)
        values = H2Trees.values(tree, leaf)
        Fvector = [
            _dummyfarfieldblock(
                T, d, length(leafsamplings[i].samplingpoints), length(values), first(values)
            ) for d in 1:D
        ]
        Fs[i] = SVector{D,Matrix{T}}(Fvector)
    end

    return Fs
end

function (assembler::DummyFarFieldAssembler)(tree, sampling; verbose::Bool=false)
    D = _dimensionvalue(assembler.dimensionality)
    leaves = H2Trees.leaves(tree)

    Fs = Vector{SVector{D,Matrix{scalartype(assembler.operator)}}}(undef, length(leaves))

    T = scalartype(assembler.operator)
    for (i, leaf) in enumerate(leaves)
        values = H2Trees.values(tree, leaf)
        Fvector = [
            _dummyfarfieldblock(T, d, length(sampling), length(values), first(values)) for
            d in 1:D
        ]
        Fs[i] = SVector{D,Matrix{T}}(Fvector)
    end

    return Fs
end
