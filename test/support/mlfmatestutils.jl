module MLFMATestUtils
using LinearAlgebra, StaticArrays
using CompScienceMeshes, BEAST
using H2Factory
using MLFMA

function fixturemesh(name::AbstractString)
    return CompScienceMeshes.readmesh(
        joinpath(@__DIR__, "..", "assets", "fixed", name * ".in")
    )
end

# Measure concrete calls, not the surrounding testset's captured variables.
function warmedallocations(f::F, args::Vararg{Any,N}; kwargs...) where {F,N}
    f(args...; kwargs...)
    f(args...; kwargs...)
    return @allocated f(args...; kwargs...)
end

function physicalconstants(f::U) where {U}
    μ0 = U(4 * π * 1e-7)
    μr = U(1.0)
    μ = μ0 * μr
    ε0 = U(8.854187812e-12)
    εr = U(1)
    ε = ε0 * εr
    c = U(1) / sqrt(ε * μ)

    λ = c / f
    κ = U(2 * π / λ)
    ω = κ * c

    return ω, c, μ0, μr, μ, ε0, εr, ε, λ, κ
end

"""
    meshspherewithcenter(center, radius, h)

A sphere mesh of `radius` with element size `h`, centred on `center` rather than the origin.

Several testsets need two or three spheres at known, well-separated positions so the tree
actually produces far interactions between them; `CompScienceMeshes.meshsphere` only builds
origin-centred spheres.
"""
function meshspherewithcenter(center, radius, h)
    return CompScienceMeshes.translate(
        CompScienceMeshes.meshsphere(radius, h), SVector{3}(center...)
    )
end

struct DipoleBasis{T} <: BEAST.Space{T}
    pos::Vector{SVector{3,T}}
    fns::Vector{SVector{3,T}}
end

function positions(db::DipoleBasis)
    return db.pos
end

function directions(db::DipoleBasis)
    return db.fns
end

function dipoleblockassembler(I, J, storefn)
    for i in eachindex(I)
        for j in eachindex(J)
            storefn(ComplexF64(0), i, j)
        end
    end
    return zeros(ComplexF64, length(I), length(J))
end

function _kernel(k, x, y)
    return cis(-k * norm(x - y)) / (4 * pi * norm(x - y))
end

function _Greensdivdiv(κ, x, y, dipoledir)
    R = x - y
    d = norm(R)
    kd = (κ * d)
    kd² = kd^2
    ℓeᵣeᵣ = R * (dot(R, dipoledir)) / (d^2)

    return (
        _kernel(κ, x, y) *
        ((3.0 / kd² + 3.0im / kd - 1) * ℓeᵣeᵣ + (-1.0 / kd² - 1.0im / kd + 1) * dipoledir)[:]
    )
end

"""
    stubbedstrategy(operator, spaces...; kwargs...)

An `MLFMA.MLFMAStrategy` with the expensive-but-irrelevant parts stubbed: a dummy near block, a dummy
I2O translation, and the cheapest far-field quadrature. Everything else is the real default.

Most tests exercise one part of the machinery and pay full price for the rest -- a real
`SparseBEAST` near block at `NEARINTERACTIONSQUADSTRATEGYRECOM` costs minutes on meshes where the thing
under test costs seconds. This is the shared "stub what I am not testing" strategy, so that choice is
made once and visibly, instead of three sub-strategies being rebuilt at every call site.

!!! warning "Matvec results are UNDEFINED under this strategy"

    The stubbed I2O is `H2Factory.DummyI2OTranslationCollection`, whose coupling returns its
    output buffer unchanged -- it does not zero it -- so disaggregation reads uninitialised moment
    memory. A matvec therefore produces garbage, observed as `NaN` in places. Never assert on matvec
    VALUES with this strategy; use it for construction, accessors, plans, moments and allocation
    behaviour. Tests that need real numbers override `i2ostrategy` back to the real
    `TranslationCollection` (see test/Maps/comparison.jl, which stubs only the near block).

`kwargs...` go straight to `MLFMAStrategy`, so any of the three stubs can be overridden back to the
real thing by naming its sub-strategy:

```julia
# real far-field quadrature, near and translation still stubbed
stubbedstrategy(
    operator,
    X;
    farfieldstrategy=MLFMA.FarFieldStrategy(; quadstrategy=BEAST.SingleNumQStrat(2)),
)
```
"""
function stubbedstrategy(operator, spaces...; kwargs...)
    return MLFMA.MLFMAStrategy(
        operator,
        spaces...;
        nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
            MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
        ),
        i2ostrategy=MLFMA.I2OTranslationStrategy(;
            TranslationCollection=MLFMA.DummyTranslationCollection
        ),
        farfieldstrategy=MLFMA.FarFieldStrategy(; quadstrategy=BEAST.SingleNumQStrat(1)),
        kwargs...,
    )
end

"""
    radiationbuilder(; quadstrategy=BEAST.SingleNumQStrat(7), assemblerbuilder=...)
    receivebuilder(; quadstrategy=BEAST.SingleNumQStrat(7), assemblerbuilder=...)

A `MLFMA.BEASTFieldCollectionBuilder` for one side, for tests that build a single leaf-free field
collection with `MLFMA.buildfieldcollection(builder, op, sampling, space)`.

The `quadstrategy` default is 7 -- the value the removed
`RadiationFieldCollection(op, sampling, space; quadstrat=...)` keyword defaulted to -- so migrating
a call site off that form is a rename and not also a silent change of quadrature. It is NOT
`MLFMA.FARFIELDQUADSTRATEGYRECOM` (5): that is the recommendation for a whole map, and matching it
here would have shifted every one of these tests' reference values at the same time as the API
change.

Deliberately thin. It exists so ~20 call sites do not each spell out the same three-argument
builder, not to hide the builder protocol: tests that are ABOUT the protocol (the side/assembler
wiring in test/Construction/strategies.jl, test/FarField/collectionbuilders.jl) construct their
builders explicitly.
"""
function radiationbuilder(;
    quadstrategy=BEAST.SingleNumQStrat(7),
    assemblerbuilder=MLFMA.BEASTFarFieldAssemblerBuilder(),
)
    return MLFMA.BEASTFieldCollectionBuilder(
        MLFMA.RadiationSide(), assemblerbuilder, quadstrategy
    )
end

function receivebuilder(;
    quadstrategy=BEAST.SingleNumQStrat(7),
    assemblerbuilder=MLFMA.BEASTFarFieldAssemblerBuilder(),
)
    return MLFMA.BEASTFieldCollectionBuilder(
        MLFMA.ReceiveSide(), assemblerbuilder, quadstrategy
    )
end

export physicalconstants, meshspherewithcenter, stubbedstrategy
export fixturemesh
export warmedallocations
export radiationbuilder, receivebuilder
export DipoleBasis, positions, directions, dipoleblockassembler, _kernel, _Greensdivdiv

end # module MLFMATestUtils

using .MLFMATestUtils
