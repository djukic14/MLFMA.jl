struct PhaseShifter{D<:Dict}
    phaseshifts::Vector{D}
    minlevel::Int
end

function PhaseShifter(
    γ::T, tree, relevantlevels::AbstractVector{Int}, samplings::Vector{S}
) where {T<:Number,T1<:Number,S<:Sampling.AbstractSampling{T1}}
    return PhaseShifter(γ, tree, relevantlevels, samplings, H2Trees.treetrait(tree))
end

function shifttoparentphase!(
    farfield::FarFields.AbstractFarField, P::PhaseShifter, tree, nodeid::I
) where {I<:Integer}
    return elementwisemul!(farfield, farfield, findphaseshift(P, tree, nodeid))
end

function shifttoparentphase!(
    storagefarfield::FarFields.AbstractFarField,
    farfield::FarFields.AbstractFarField,
    P::PhaseShifter,
    tree,
    nodeid::I,
) where {I<:Integer}
    return elementwisemul!(storagefarfield, farfield, findphaseshift(P, tree, nodeid))
end

function shifttochildphase!(
    farfield::FarFields.AbstractFarField, P::PhaseShifter, tree, childid::I
) where {I<:Integer}
    return elementwisemul!(farfield, farfield, findoppositephaseshift(P, tree, childid))
end

function shifttochildphase!(
    storagefarfield::FarFields.AbstractFarField,
    farfield::FarFields.AbstractFarField,
    P::PhaseShifter,
    tree,
    childid::I,
) where {I<:Integer}
    return elementwisemul!(
        storagefarfield, farfield, findoppositephaseshift(P, tree, childid)
    )
end

function PhaseShifter(
    γ::T,
    tree,
    relevantlevels::AbstractVector{Int},
    samplings::Vector{S},
    ::H2Trees.isTwoNTree,
) where {T<:Number,T1<:Number,S<:Sampling.AbstractSampling{T1}}
    if isempty(samplings)
        return PhaseShifter(Dict{Int,Matrix{T}}[], relevantlevels[begin])
    end
    minrelevantlevel = first(relevantlevels)
    MatrixType = Sampling.arraytype(samplings[end]){T}

    phaseshifts = Vector{Dict{Int,MatrixType}}(undef, length(relevantlevels) - 1)
    for i in eachindex(phaseshifts)
        phaseshifts[i] = Dict{Int,MatrixType}()
    end

    appendphaseshifts!(phaseshifts, γ, tree, minrelevantlevel, samplings)

    return PhaseShifter(phaseshifts, minrelevantlevel)
end

function PhaseShifter(
    γ::T,
    tree,
    relevantlevels::AbstractVector{Int},
    samplings::Vector{S},
    ::H2Trees.isBlockTree,
) where {T<:Number,T1<:Number,S<:Sampling.AbstractSampling{T1}}
    if isempty(samplings)
        return PhaseShifter(Dict{Int,Matrix{T}}[], relevantlevels[begin])
    end
    minrelevantlevel = first(relevantlevels)
    MatrixType = Sampling.arraytype(samplings[end]){T}

    trialtree = H2Trees.trialtree(tree)
    testtree = H2Trees.testtree(tree)

    phaseshifts = Vector{Dict{Int,MatrixType}}(undef, length(relevantlevels) - 1)
    for i in eachindex(phaseshifts)
        phaseshifts[i] = Dict{Int,MatrixType}()
    end

    appendphaseshifts!(phaseshifts, γ, trialtree, minrelevantlevel, samplings)
    appendphaseshifts!(phaseshifts, γ, testtree, minrelevantlevel, samplings)

    return PhaseShifter(phaseshifts, minrelevantlevel)
end

function _getlevelindex(level, minrelevantlevel)
    return level - minrelevantlevel
end

function findphaseshift(P::PhaseShifter, tree, nodeid::I) where {I<:Integer}
    return findphaseshift(
        P, H2Trees.sector(tree, nodeid), H2Trees.level(tree, nodeid), P.minlevel
    )
end

function findoppositephaseshift(P::PhaseShifter, tree, nodeid::I) where {I<:Integer}
    return findphaseshift(
        P, H2Trees.oppositesector(tree, nodeid), H2Trees.level(tree, nodeid), P.minlevel
    )
end

function findphaseshift(P::PhaseShifter, sector::Int, level::Int, minrelevantlevel)
    return P.phaseshifts[_getlevelindex(level, minrelevantlevel)][sector]
end

"""
    numberofsectors(tree) -> Int

Return the number of child sectors: `2^dimension`, i.e. 8 for an octree and 4 for a quadtree.

The root centre's length gives the dimension through the tree wrapper interface, as in
`MLFMA.sampling`. `appendphaseshifts!` stops visiting a level once every sector is recorded.
This avoids redundant visits but has little effect on total construction time: on
`meshcircle(10.0, 1/40)` (2514 functions, 284 nodes), visiting 9 rather than 267 nodes saved
1.5 µs out of 94.9 s.
"""
numberofsectors(tree) = 1 << length(H2Trees.center(tree))

function appendphaseshifts!(phaseshifts, γ::T, tree, minrelevantlevel, samplings) where {T}
    # The sector count is constant across levels; read the centre only once.
    nsectors = numberofsectors(tree)

    for level in H2Trees.levels(tree)
        level <= minrelevantlevel && continue

        levelindex = _getlevelindex(level, minrelevantlevel)

        length(keys(phaseshifts[levelindex])) == nsectors && continue

        parentk̂, _ = Sampling.cartesiansamplepoints(samplings[levelindex])

        for node in H2Trees.LevelIterator(tree, level)
            length(keys(phaseshifts[levelindex])) == nsectors && break
            H2Trees.parent(tree, node) == 0 && continue

            sector = H2Trees.sector(tree, node)
            sector in keys(phaseshifts[levelindex]) && continue

            phaseshifts[levelindex][sector] = Phase.phasefactor(
                γ,
                parentk̂,
                H2Trees.parentcenterminuschildcenter(tree, node),
                AbstractRadiationFieldCollection,
            )

            oppositesector = H2Trees.oppositesector(tree, node)
            oppositesector in keys(phaseshifts[levelindex]) && continue

            phaseshifts[levelindex][oppositesector] = Phase.phasefactor(
                γ,
                parentk̂,
                -H2Trees.parentcenterminuschildcenter(tree, node),
                AbstractRadiationFieldCollection,
            )
        end
    end
    return phaseshifts
end
