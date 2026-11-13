# Adjoint multiplication for complex leaf collections.
#
# H2Factory realizes `A^H` by swapping sides, applying `adjoint(transfer)` at level transfers and
# conjugating the I2O coupling. Its default leaf multiply is unchanged between orientations:
# `leafmul(A, ::AbstractMapTrait) = LinAlgebraMulFunctor()`. MLFMA also needs collection conjugation
# because its leaf patterns `exp(γ·k̂·r)` are complex.
#
# For a leaf pair translating directly, without interpolation or phase shifts, the stored objects
# reconstruct the far block as
#
#     far[I,J] = α · Σ_c transpose(L_test_c[:,I]) · Diagonal(T) · L_trial_c[:,J]
#
# The stored collections carry no hidden
# conjugation, so
#
#     A^H = conj(α) · transpose(conj(L_trial)) · T^H · conj(L_test)
#
# requires conjugating both leaf collections and `α`, in addition to H2Factory's `T^H`.
# Partial conjugation is not sufficient and need not reduce error.
#
# `H2Factory.conjmul!` alone gives `C·conj(x)`, not `conj(C)·x`. Moments must not carry hidden
# conjugation between levels, since H2Factory already directs the transfers. Aggregation combines
# `conjmul!` with an output conjugation using
# `conj(C)·x == conj(C·conj(x))`.

# H2Factory supplies `ConjugatedCollectionMul` and its matrix kernels. The methods here handle
# lazy receive and compressed harmonic collections, which are not ordinary matrices.
H2Factory.leafmul(A::GalerkinMLFMACore, ::H2Factory.isAdjoint) = ConjugatedCollectionMul()
H2Factory.leafmul(A::PetrovGalerkinMLFMACore, ::H2Factory.isAdjoint) =
    ConjugatedCollectionMul()

# A Galerkin map's lazy test collection mirrors radiation data instead of storing a matrix.
# It implements scratch-taking leaf multiplication, not `mul!` or `conjmul!`.
# Only conjugated aggregation is needed: the adjoint aggregates on the test side and disaggregates
# through ordinary trial-side radiation matrices. Forward receive multiplication needs no conjugation.
#
# The effective collection is `scatter ∘ radiation`. Apply `conjmul!` into scratch, scatter through
# the mirror and coupling rule into `moment`, then conjugate the overwritten result.
function applyconjugatedleafmul!(
    scratch::AbstractFarField,
    moment::AbstractFarField,
    mirroredfcol::LazyReceiveFieldCollection,
    y::AbstractArray,
)
    H2Factory.conjmul!(scratch, farfieldcollection(mirroredfcol), y)
    _lazyreceivescatter!(
        moment,
        scratch,
        mirrorindices(mirroredfcol),
        receivecouplingrule(operator(mirroredfcol), typeof(moment)),
    )
    for i in eachindex(moment)
        conj!(H2Factory.wrapvector(moment, i))
    end
    return moment
end

# Spherical-harmonic leaves need both arities. Radiation reconstructs through an (NΩ x 3)
# intermediate, while mirrored receive collections reuse radiation data; neither uses the default
# matrix kernel.
#
# The matrices they stand for are in `farfield/sphericalharmoniccollection.jl`'s reverse-orientation
# table: with `Rec = P M C` the receive collection and `R = P S C` the radiation one, the adjoint
# aggregates through `conj(Rec)` and scatters through `Rᴴ`.

# Test-side aggregation: `moment = conj(Rec) x = P conj(M) conj(C) x`.
#
# `P` is real, so `conj(Rec) x == conj(P M C conj(x))`. `_aggregateconjugated!` handles the input
# conjugation in the coefficient GEMV; the final loop conjugates the projected moment.
#
# The final conjugation requires overwrite semantics: `upwardpass!` calls the 4-argument arity,
# which takes no `β` and retains no previous contributions.
function applyconjugatedleafmul!(
    scratch::SphericalHarmonicLeafScratch,
    moment::TangentialSphericalFarField,
    collection::SphericalHarmonicReceiveCollection,
    x::AbstractVector,
)
    @views A = scratch.coefficients[:, 1:3]
    _aggregateconjugated!(
        vec(A), SphericalHarmonics.coefficientmatrix(storage(collection)), x
    )
    @views Cart = scratch.cartesian[:, 1:3]
    LinearAlgebra.mul!(Cart, scratch.mirror, A)
    _projectinto!(moment, Cart, scratch.θ̂, scratch.φ̂)
    # Apply coupling before conjugation: the collection is `K B C`, and
    # `conj(K B C) x == conj(K (B C conj(x)))`, so `K` belongs inside the argument of `conj`.
    _applycoupling!(moment, coupling(collection))
    for i in eachindex(moment)
        conj!(H2Factory.wrapvector(moment, i))
    end
    return moment
end

# Assembled test-side aggregation: `moment = conj(Rec) x` with `Rec = P S C`.
# Petrov test collections store their own patterns, so use `scratch.synthesis` rather than
# `scratch.mirror`, with no receive coupling before conjugation.
function applyconjugatedleafmul!(
    scratch::SphericalHarmonicLeafScratch,
    moment::TangentialSphericalFarField,
    collection::SphericalHarmonicAssembledReceiveCollection,
    x::AbstractVector,
)
    @views A = scratch.coefficients[:, 1:3]
    _aggregateconjugated!(
        vec(A), SphericalHarmonics.coefficientmatrix(storage(collection)), x
    )
    @views Cart = scratch.cartesian[:, 1:3]
    LinearAlgebra.mul!(Cart, scratch.synthesis, A)
    _projectinto!(moment, Cart, scratch.θ̂, scratch.φ̂)
    for i in eachindex(moment)
        conj!(H2Factory.wrapvector(moment, i))
    end
    return moment
end

# Trial-side disaggregation: `result = α Rᴴ moment + β result`.
# `H2Factory.α(core, ::isAdjoint)` has already conjugated `α`. `_Adjointed()` selects
# `transpose(conj(R)) == Rᴴ` rather than the transposed scatter.
#
# The assembled-collection union also covers compressed test-side scatter when H2Factory swaps
# sides: `Rec = P S C` uses the same `Sᴴ`.
function applyconjugatedleafmul!(
    scratch::SphericalHarmonicLeafScratch,
    result::AbstractVector,
    collection::SphericalHarmonicAssembledCollection,
    moment::TangentialSphericalFarField,
    α::Number,
    β::Number,
)
    return _shescatter!(
        result, scratch, storage(collection), moment, scratch.synthesis, α, β, _Adjointed()
    )
end

# Circular-harmonic leaves also need both arities because neither compressed collection is a matrix.
#
# The matrices they stand for are in `farfield/circularharmoniccollection.jl`'s orientation table:
# with `Rec = K M C` the receive collection and `R = S C` the radiation one -- no `P`, because a 2-D
# far field has no frame to project through -- the adjoint aggregates through `conj(Rec)` and scatters
# through `Rᴴ`.

# Test-side aggregation: `moment = conj(Rec) x = conj(K) conj(M) conj(C) x`.
#
# `conj(Rec) x == conj(K M C conj(x))`. `_aggregatecircularconjugated!` handles the input
# conjugation in the coefficient GEMV. The final conjugation is valid because `upwardpass!` calls
# the overwriting 4-argument arity, which takes no `β`.
function applyconjugatedleafmul!(
    scratch::CircularHarmonicLeafScratch,
    moment::AbstractFarField,
    collection::CircularHarmonicReceiveCollection,
    x::AbstractVector,
)
    store = storage(collection)
    D = Harmonics.CircularHarmonics.ncomponents(store)
    @views A = scratch.coefficients[:, 1:D]
    _aggregatecircularconjugated!(
        vec(A), Harmonics.CircularHarmonics.coefficientmatrix(store), x
    )
    for d in 1:D
        @views LinearAlgebra.mul!(H2Factory.wrapvector(moment, d), scratch.mirror, A[:, d])
    end
    # Apply coupling before conjugation: the collection is `K M C`, and
    # `conj(K M C) x == conj(K (M C conj(x)))`, so `K` belongs inside the argument of `conj`.
    _applycircularcoupling!(moment, coupling(collection))
    for i in eachindex(moment)
        conj!(H2Factory.wrapvector(moment, i))
    end
    return moment
end

# Assembled test-side aggregation: `moment = conj(Rec) x` with `Rec = S C`.
# Petrov test collections store their own patterns rather than mirroring the trial leaf.
#
# Use `scratch.synthesis` rather than `scratch.mirror`, with no receive coupling.
# `conj(S C) x == conj(S (C conj(x)))`; the final conjugation relies on the overwriting
# 4-argument arity called by `upwardpass!`.
function applyconjugatedleafmul!(
    scratch::CircularHarmonicLeafScratch,
    moment::AbstractFarField,
    collection::CircularHarmonicAssembledReceiveCollection,
    x::AbstractVector,
)
    store = storage(collection)
    D = Harmonics.CircularHarmonics.ncomponents(store)
    @views A = scratch.coefficients[:, 1:D]
    _aggregatecircularconjugated!(
        vec(A), Harmonics.CircularHarmonics.coefficientmatrix(store), x
    )
    for d in 1:D
        @views LinearAlgebra.mul!(
            H2Factory.wrapvector(moment, d), scratch.synthesis, A[:, d]
        )
    end
    for i in eachindex(moment)
        conj!(H2Factory.wrapvector(moment, i))
    end
    return moment
end

# Trial-side disaggregation: `result = α Rᴴ moment + β result`.
# `H2Factory.α(core, ::isAdjoint)` has already conjugated `α`. `_Adjointed()` selects
# `transpose(conj(R)) == Rᴴ` rather than the transposed scatter.
#
# The assembled-collection union also covers compressed test-side scatter after H2Factory swaps
# sides: `Rec = S C` uses the same `Sᴴ`.
function applyconjugatedleafmul!(
    scratch::CircularHarmonicLeafScratch,
    result::AbstractVector,
    collection::CircularHarmonicAssembledCollection,
    moment::AbstractFarField,
    α,
    β::Number,
)
    return _chescatter!(
        result, scratch, storage(collection), moment, scratch.synthesis, α, β, _Adjointed()
    )
end
