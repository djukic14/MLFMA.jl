
"""
    TranslationCollection(γ, tree, translatingplan, translationsamplings; levelscales, verbose=false)
    TranslationCollection(γ, tree, aggregationplan, disaggregationplan, translationsamplings; ...)

Build the MLFMA interpolation-to-observation translation operators for a tree.

This builder returns an `H2Factory.I2OTranslationCollection` wrapping an
`MLFMATranslator`, tagged `H2Trees.DirectionInvariancePerLevel()`. There is no
`TranslationCollection` struct to dispatch on.

The four-argument form takes a ready translating plan. The five-argument form is the
Petrov-Galerkin entry: it combines the trial-side aggregation and test-side disaggregation plans
into one translating plan and delegates, mirroring `H2Factory.I2OTranslationCollection`'s own
convenience wrapper.

With no translation samplings it delegates to `DummyTranslationCollection`.

The directions come from the sampling and keep its element type. They are not forced
to `SVector{3}`, because a `CircularSampling` yields `SVector{2}`; and not converted to `real(T)`
from γ, because the grid owns its own precision.
"""
function TranslationCollection(
    γ::T,
    tree,
    aggregationplan::H2Trees.AbstractPlan,
    disaggregationplan::H2Trees.AbstractPlan,
    translationsamplings;
    levelscales=Vector{Nothing}(undef, length(translationsamplings)),
    verbose=false,
) where {T}
    translatingplan = H2Trees.translatingplan(aggregationplan, disaggregationplan)
    return TranslationCollection(
        γ,
        tree,
        translatingplan,
        translationsamplings;
        levelscales=levelscales,
        verbose=verbose,
    )
end

function TranslationCollection(
    γ::T,
    tree,
    translatingplan,
    translationsamplings;
    levelscales=Vector{Nothing}(undef, length(translationsamplings)),
    verbose=false,
) where {T}
    isempty(translationsamplings) &&
        return DummyTranslationCollection(γ, tree, translatingplan, translationsamplings)

    # Infer the direction dimension, array rank and precision from each grid, not from γ.
    # The nonempty, concretely typed sampling vector makes this map type-stable.
    cartesiansamplingpoints = map(
        sampling -> Sampling.cartesiansamplepoints(sampling)[1], translationsamplings
    )

    translator = MLFMATranslator(
        γ,
        translationsamplings,
        cartesiansamplingpoints,
        translatingplan,
        tree;
        levelscales=levelscales,
    )

    return H2Factory.I2OTranslationCollection(
        tree,
        translatingplan,
        H2Trees.DirectionInvariancePerLevel(),
        translator;
        verbose=verbose,
    )
end

struct MLFMATranslator{T,Sampling,SamplingPoints,Scales,P,Tree}
    γ::T
    translationsamplings::Vector{Sampling}
    cartesiansamplingpoints::Vector{SamplingPoints}
    levelscales::Vector{Scales}
    plan::P
    tree::Tree
end

function MLFMATranslator(
    γ,
    translationsamplings,
    cartesiansamplingpoints,
    plan,
    tree;
    levelscales=Vector{Nothing}(undef, length(translationsamplings)),
)
    return MLFMATranslator(
        γ,
        translationsamplings,
        cartesiansamplingpoints,
        plan,
        tree,
        H2Trees.treetrait(tree);
        levelscales=levelscales,
    )
end

function MLFMATranslator(
    γ,
    translationsamplings,
    cartesiansamplingpoints,
    plan,
    tree,
    ::Any;
    levelscales=Vector{Nothing}(undef, length(translationsamplings)),
)
    return MLFMATranslator(
        γ, translationsamplings, cartesiansamplingpoints, levelscales, plan, tree
    )
end

function MLFMATranslator(
    γ,
    translationsamplings,
    cartesiansamplingpoints,
    plan,
    tree,
    ::H2Trees.isBlockTree;
    levelscales=Vector{Nothing}(undef, length(translationsamplings)),
)
    return MLFMATranslator(
        γ,
        translationsamplings,
        cartesiansamplingpoints,
        levelscales,
        plan,
        H2Trees.testtree(tree),
    )
end

function Base.eltype(translator::MLFMATranslator{T}) where {T}
    return Sampling.arraytype(translator.translationsamplings[end]){T}
end

# Use the same level index for translation grids and symmetry permutation tables.
# A mismatch can select the wrong grid without a shape error when adjacent grids have equal sizes.
function translationlevelindex(translator::MLFMATranslator, node::Int)
    return H2Trees.level(translator.tree, node) -
           H2Trees.mintranslationlevel(translator.plan) + 1
end

function (translator::MLFMATranslator)(
    receivingnode::Int, translatingnode::Int, translationdirection
)
    levelID = translationlevelindex(translator, receivingnode)

    sampling = translator.translationsamplings[levelID]

    tf = eltype(translator)(undef, size(sampling))
    Translations.translationoperator!(
        tf,
        translator.γ,
        translator.cartesiansamplingpoints[levelID],
        translationdirection,
        Sampling.cutoffparameter(sampling),
    )
    _mulscales!(tf, translator.levelscales[levelID])

    return tf
end

function _mulscales!(tf, scale::Nothing)
    return tf
end

function _mulscales!(tf, scale)
    return tf .*= scale
end

function DummyTranslationCollection(
    γ::T,
    tree,
    aggregationplan::H2Trees.AbstractPlan,
    disaggregationplan::H2Trees.AbstractPlan,
    translationsamplings;
    levelscales=nothing,
    verbose=false,
) where {T}
    return H2Factory.DummyI2OTranslationCollection(), H2Factory.EmptyChannels()
end

function DummyTranslationCollection(
    γ::T,
    tree,
    plan::H2Trees.AbstractPlan,
    translationsamplings;
    levelscales=nothing,
    verbose=false,
) where {T}
    return H2Factory.DummyI2OTranslationCollection(), H2Factory.EmptyChannels()
end
