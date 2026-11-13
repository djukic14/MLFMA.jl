# The adjoint of `CircularInterpolator`, built as `Rᴴ` term by term rather than as a normalization
# that happens to work out.
#
# With `R = (1/N) F_Mᴴ P F_N` (see `interpolator/circularinterpolation.jl` for where that comes from)
# and `P` real,
#
#     Rᴴ = (1/N) F_Nᴴ Pᵀ F_M = IFFT_N ∘ Pᵀ ∘ FFT_M
#
# so this apply is: transform the parent moment UNNORMALIZED, gather coefficients through the
# transpose of the coefficient move, inverse-transform on the child grid -- and there is NO leftover
# scale factor. That absence is the check worth keeping in mind while reading the loop below: a stray
# `M/N` here would be a plausible-looking adjoint that fails `⟨Iu, v⟩ == ⟨u, Iᵀv⟩` by a ratio which
# tends to 1 as the grids refine, which is precisely the slow-drift failure mode of #35/#61/#104.
#
# This is the Euclidean adjoint, not the quadrature-weighted one:
# MLFMA folds the quadrature weights into the LEAF receive patterns once, so a level transfer must not
# re-apply a weight ratio. `test/LevelTransfers/anterpolationcontract.jl` asserts `anter ≈ adjoint(inter)`
# literally, which is what pins this.

"""
    CircularAnterpolator(interpolator)

The ADJOINT of a [`CircularInterpolator`](@ref) -- `Rᴴ`, bare, with no quadrature weights.

Shares the interpolator's [`CircularInterimScratch`](@ref): interpolation runs child → parent and
anterpolation parent → child, so between them they need exactly the two spectra it holds.
"""
struct CircularAnterpolator{stencilsize,T,I} <:
       Interpolation.AbstractAnterpolator{stencilsize,T}
    interpolator::I
end

function CircularAnterpolator(
    interpolator::CircularInterpolator{stencilsize,T}
) where {stencilsize,T}
    return CircularAnterpolator{stencilsize,T,typeof(interpolator)}(interpolator)
end

# Accept but ignore `nchunks` for compatibility with map constructors. This family builds its
# plans in scratch owned by each (level, chunk).
function anterpolator(interpolator::CircularInterpolator; nchunks::Int=Threads.nthreads())
    return CircularAnterpolator(interpolator)
end

"""
    (anterpolator::CircularAnterpolator)(anterpolatedfarfield, farfield, farfieldsampling,
                                         anterpolationsampling; interpolationtrait, interim,
                                         prescale, stagetransform)

Apply the adjoint of interpolation, taking `farfield` on the PARENT grid `farfieldsampling` down to
the CHILD grid `anterpolationsampling`.

`prescale` and `stagetransform` carry the same meaning as in the other families and go through the
same `Interpolation._stageprescaled!`: apply the diagonal and transform while copying into the
private parent spectrum. The caller's moment is never mutated, so concurrent children can read it.
Without `prescale`, the copy applies no `stagetransform`.
"""
function (anterpolator::CircularAnterpolator{stencilsize,T})(
    anterpolatedfarfield::F,
    farfield::F,
    farfieldsampling::Sampling.CircularSampling,
    anterpolationsampling::Sampling.CircularSampling;
    interpolationtrait=Interpolation.defaultinterpolationtrait(F),
    interim=nothing,
    prescale=nothing,
    stagetransform=identity,
) where {stencilsize,T,F<:FarFields.AbstractFarField}
    matrices = H2Factory.moments(farfield)
    resultmatrices = H2Factory.moments(anterpolatedfarfield)

    @assert length(farfieldsampling) == length(first(matrices))
    @assert length(anterpolationsampling) == length(first(resultmatrices))
    _checkcirculararity(interpolationtrait, F)

    # `N`/`M` name the FORWARD direction throughout -- child then parent -- even though the data flows
    # the other way here, so that the coefficient move and its transpose are called with the same two
    # numbers in the same two slots.
    N = length(anterpolationsampling)
    M = length(farfieldsampling)
    S = eltype(first(resultmatrices))

    scratch = _circularscratch(interim, N, M)
    childspectrum = _childvector(scratch, N, S)
    parentspectrum = _parentvector(scratch, M, S)

    for component in eachindex(matrices)
        # Transform only the private spectrum. Fuse `prescale` into the input copy when present;
        # the unscaled path is a plain `copyto!`.
        if prescale === nothing
            copyto!(parentspectrum, matrices[component])
        else
            Interpolation._stageprescaled!(
                parentspectrum, matrices[component], prescale, stagetransform
            )
        end
        _fftparent!(parentspectrum, scratch)
        _movecircularcoefficientsadjoint!(childspectrum, parentspectrum, N, M)
        _ifftchild!(childspectrum, scratch)
        copyto!(resultmatrices[component], childspectrum)
    end

    return nothing
end

# The vector-of-far-fields entry point the abstract anterpolator declares; see the interpolator's.
function (anterpolator::CircularAnterpolator)(
    anterpolatedfarfields::AbstractArray{F,1}, farfields::AbstractArray{F,1}, args...
) where {F<:FarFields.AbstractFarField}
    return error(
        "Not implemented: this anterpolator uses the special structure of CircularSampling"
    )
end
