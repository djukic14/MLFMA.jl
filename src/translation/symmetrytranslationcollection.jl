# Symmetry-reduced translations on angular grids.
# H2Trees records one geometry per cubic/square lattice orbit and tags each interaction with the
# symmetry mapping its representative to the required direction. H2Factory stores the translations
# and supplies the lazy `PermutedTranslation` view. Sampling supplies the supported subgroup and
# angular permutations; this file selects the permutation for each level.
#
# If a symmetry maps the grid onto itself, `T(k̂, Sd) == T(S⁻¹k̂, d)` allows the canonical translation
# to be read through a permutation, without another array or interpolation.
# This adds no approximation, but is not bitwise identical to the unreduced path: `k̂ⱼ · d` and
# `k̂_{π(j)} · Sd` use rotated inputs whose `cos`/`sin` values can differ in rounding.
# On the 2-D single-layer fixture, 89 of 106 couplings differed in their last bits, with a worst
# relative difference of 8.3e-15. The matvec agreed to better than 1e-14.

"""
    supportedsymmetry(samplings::AbstractVector, ::Val{D})

Return the largest symmetry policy supported exactly by every level's grid.

Per-level policies can differ -- `CircularSampling`'s quarter turns need `Nφ % 4 == 0`, and the rate
rule does not guarantee that at every level -- and the tree is canonicalized once, with one policy,
for all of them. So the common choice is the smallest group among the levels.

Choosing by group order relies on the policies forming a chain under inclusion.
[`SymmetryTranslationCollection`](@ref) validates the choice by building every element's angular
permutation at every level. `angularpermutation` raises if the grid cannot apply an element exactly.
"""
function supportedsymmetry(samplings::AbstractVector, ::Val{D}) where {D}
    policies = map(supportedsymmetry, samplings)
    orders = map(policy -> length(H2Trees.symmetrygroup(policy, Val(D))), policies)
    return policies[argmin(orders)]
end

# Petrov-Galerkin entry, mirroring `TranslationCollection`'s: combine the trial aggregation and test
# disaggregation plans into one translating plan and delegate.
function SymmetryTranslationCollection(
    γ::T,
    tree,
    aggregationplan::H2Trees.AbstractPlan,
    disaggregationplan::H2Trees.AbstractPlan,
    translationsamplings;
    levelscales=Vector{Nothing}(undef, length(translationsamplings)),
    verbose=false,
) where {T}
    translatingplan = H2Trees.translatingplan(aggregationplan, disaggregationplan)
    return SymmetryTranslationCollection(
        γ,
        tree,
        translatingplan,
        translationsamplings;
        levelscales=levelscales,
        verbose=verbose,
    )
end

"""
    SymmetryTranslationCollection(γ, tree, translatingplan, translationsamplings; levelscales, verbose)

Drop-in replacement for [`TranslationCollection`](@ref MLFMA.Translations.TranslationCollection) that builds one translation per symmetry
orbit instead of one per direction.

Select it through the strategy:

```julia
strategy = MLFMA.MLFMAStrategy(
    operator,
    space;
    i2ostrategy=MLFMA.I2OTranslationStrategy(;
        TranslationCollection=MLFMA.SymmetryTranslationCollection
    ),
)
```

Symmetry acts by an exact grid permutation, with no interpolation or additional approximation.
Rounding can differ from the unreduced collection because the inputs are rotated. The 2-D
single-layer fixture agreed to ~1e-14 relative in both couplings and matvec; see the header.
"""
function SymmetryTranslationCollection(
    γ::T,
    tree,
    translatingplan,
    translationsamplings;
    levelscales=Vector{Nothing}(undef, length(translationsamplings)),
    verbose=false,
) where {T}
    isempty(translationsamplings) &&
        return DummyTranslationCollection(γ, tree, translatingplan, translationsamplings)

    dimension = Val(length(eltype(tree)))
    policy = supportedsymmetry(translationsamplings, dimension)
    trait = H2Trees.SymmetryDirectionInvariancePerLevel(policy)

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

    # Non-lattice offsets between block-tree roots require the unreduced collection.
    # Catch only `NonLatticeTranslationError`; unrelated failures must propagate.
    # The check runs in `translationmetadata`, before any operators are computed. Fallback repeats
    # one geometry pass, measured at 0.4 to 1.6 seconds for three million pairs, versus minutes for
    # operator construction.
    canonical, channels = try
        H2Factory.I2OTranslationCollection(
            tree, translatingplan, trait, translator; verbose=verbose
        )
    catch exception
        exception isa H2Trees.NonLatticeTranslationError || rethrow()
        @warn(
            "SymmetryTranslationCollection: falling back to the unreduced " *
                "`TranslationCollection`. This tree's translation displacements are not all lattice " *
                "vectors, which happens when a block tree's two roots are offset by something other " *
                "than a multiple of the box size, and a lattice symmetry cannot be applied across " *
                "such an offset. The map is correct and complete; it simply stores one translation " *
                "per direction rather than one per symmetry orbit.",
            exception = exception,
        )
        return TranslationCollection(
            γ,
            tree,
            translatingplan,
            translationsamplings;
            levelscales=levelscales,
            verbose=verbose,
        )
    end

    return H2Factory.symmetryreducedcollection(
        canonical, _permutationsupplier(canonical, translator, translationsamplings)
    ),
    channels
end

# The stored convention is `q == applysymmetry(group[symmetryID], canonical)`.
# The tag maps canonical to actual, so angular directions need `S⁻¹`:
# `T(k̂, Sd) == T(S⁻¹k̂, d)`. The antipodal map is its own inverse and cannot detect a reversed
# convention; a quarter turn can.
#
# One table per (level, symmetry), shared by every translation at that level: the count is the group
# order times the level count, not the interaction count. Measured on the 16λ sphere this is about
# 1 % of what the translation reduction saves, and reading a materialized permutation costs 0.82-0.83x
# the equivalent inline index arithmetic.
function _inversepermutationtables(group, translationsamplings)
    return [
        [
            angularpermutation(sampling, H2Trees.inversesymmetry(symmetry)) for
            symmetry in group
        ] for sampling in translationsamplings
    ]
end

# H2Factory reads a pair's translation and symmetry from the same key, including for transpose and
# adjoint traversal. This supplier maps the receiving node and symmetry element to a permutation.
# MLFMA keeps its level indexing here: `level - mintranslationlevel(plan) + 1`.
# Captured `tables`, `idbyelement` and `translator` are assigned once and only read, avoiding boxing.
function _permutationsupplier(canonical, translator, translationsamplings)
    group = H2Factory.translationsymmetrygroup(canonical)
    group === nothing && error(
        "SymmetryTranslationCollection: H2Factory built a collection with no symmetry tags. " *
        "That means the translation trait did not canonicalize, so there is nothing to permute " *
        "and wrapping it would silently claim a reduction that did not happen.",
    )

    tables = _inversepermutationtables(group, translationsamplings)

    # Map symmetry elements to table positions once, rather than searching per pair.
    idbyelement = Dict(symmetry => id for (id, symmetry) in enumerate(group))

    # The level comes from the receiving node, which is safe whichever way round the traversal
    # passes the pair, because I2O pairs are same-level.
    return function (receivingnode::Int, symmetry)
        return tables[translationlevelindex(translator, receivingnode)][idbyelement[symmetry]]
    end
end
