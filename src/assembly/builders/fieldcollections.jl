# Construction entry points for field collections.
#
# The four names generated here (`RadiationFieldCollection`, `RadiationFieldCollections`,
# `ReceiveFieldCollection`, `ReceiveFieldCollections`) are functions, not types. Each call resolves the concrete,
# representation-specific collection type (`TangentialSphericalRadiationFieldCollection` and the
# other nine, see `farfield.jl`) from the operator, via
# `radiationfieldcollectiontype`/`receivefieldcollectiontype` in `operatorutils.jl`.
#
# That resolution needs an operator (or an assembler carrying the far-field type). It cannot be
# done from assembled data alone: a raw `SVector` of three matrices is equally consistent with a
# spherical and a Cartesian far field, and picking by component count is exactly the inference
# must be avoided. There is no `RadiationFieldCollection(Fs)` method that
# guesses: code holding bare component matrices names the concrete collection type it wants.
#
# These methods take an assembler that carries the far-field type. Higher-level construction uses
# `BEASTFieldCollectionBuilder` (`collectionbuilders.jl`) to keep the side, assembler and quadrature
# together. Its `buildfieldcollection` and `buildfieldcollections` methods call these functions.
function fieldcollectionconstructors(collection, collections, collectiontypeof)
    @eval function $collection(farfieldassembler, sampling; verbose::Bool=false)
        Fs = farfieldassembler(sampling; verbose=verbose)

        return $collectiontypeof(farfieldtype(farfieldassembler))(Fs)
    end

    @eval function $collections(farfieldassembler, tree, leafsamplings; verbose::Bool=false)
        Fs = farfieldassembler(tree, leafsamplings; verbose=verbose)

        CollectionType = $collectiontypeof(farfieldtype(farfieldassembler))
        firstcollection = CollectionType(Fs[begin])
        farfieldcollections = Vector{typeof(firstcollection)}(undef, length(Fs))
        farfieldcollections[1] = firstcollection

        for i in eachindex(farfieldcollections)
            i == 1 && continue
            farfieldcollections[i] = CollectionType(Fs[i])
        end
        return farfieldcollections
    end

    return nothing
end

fieldcollectionconstructors(
    :RadiationFieldCollection, :RadiationFieldCollections, :radiationcollectiontype
)
fieldcollectionconstructors(
    :ReceiveFieldCollection, :ReceiveFieldCollections, :receivecollectiontype
)
