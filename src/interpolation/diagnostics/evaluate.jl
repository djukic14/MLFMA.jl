# Exact angular references are continuous functions evaluated independently at source and target
# points. No mesh, assembly, translation, or interpolated reference enters the comparison.
#
# The local harmonic recurrences need no additional special-function package.
# `test/LevelTransfers/interpolationdiagnostics.jl` checks their normalization and signs against
# Gauss-Legendre orthonormality, low-degree closed forms, and the negative-order reflection.
#
# Normalization:
#
#     Y_l^m(θ, φ) = sqrt((2l+1)/(4π) ⋅ (l-m)!/(l+m)!) ⋅ P_l^m(cos θ) ⋅ exp(i m φ)
#
# with the Condon-Shortley phase, so ``Y_0^0 = 1/sqrt(4π)`` and
# ``∫ Y_l^m conj(Y_l'^m') dΩ = δ_{ll'} δ_{mm'}`` over the full sphere. `P_l^m` carries the
# ``(-1)^m`` of the Condon-Shortley convention, which is why ``Y_1^1`` comes out negative.

"""
    normalizedlegendre(l, m, θ)

``sqrt((2l+1)/(4π) ⋅ (l-m)!/(l+m)!) ⋅ P_l^m(cos θ)``, the θ-dependent factor of ``Y_l^m``.

Computed by recurrence in the fully normalized quantity rather than as an unnormalized `P_l^m` scaled
afterwards. That matters beyond style: `(l-m)!/(l+m)!` and `P_l^m` each overflow well before their
product does, so the scaled-afterwards form loses high degrees to `Inf/Inf` while the recurrence below
stays bounded.

`m` must satisfy `0 <= m <= l`; negative orders are handled by [`sphericalharmonic`](@ref).
"""
function normalizedlegendre(l::Integer, m::Integer, θ::T) where {T<:AbstractFloat}
    sinθ, cosθ = sincos(θ)

    # Q_m^m, by the m-recurrence Q_m^m = -sqrt((2m+1)/(2m)) ⋅ sinθ ⋅ Q_{m-1}^{m-1} from
    # Q_0^0 = 1/sqrt(4π). The minus sign is the Condon-Shortley phase, accumulated once per step.
    q = sqrt(one(T) / (4 * T(π)))
    for k in 1:m
        q *= -sqrt(T(2k + 1) / T(2k)) * sinθ
    end
    l == m && return q

    # Q_{m+1}^m, then the l-recurrence. Split out because its b coefficient vanishes, and writing it
    # through the general formula would evaluate sqrt(0 / (2m-1)) -- which is -0.0 at m = 0.
    qprevious = q
    q = sqrt(T(2m + 3)) * cosθ * qprevious
    for ll in (m + 2):l
        a = sqrt((T(2ll + 1) * T(2ll - 1)) / (T(ll - m) * T(ll + m)))
        b = sqrt(
            (T(2ll + 1) * T(ll + m - 1) * T(ll - m - 1)) /
            (T(2ll - 3) * T(ll + m) * T(ll - m)),
        )
        q, qprevious = a * cosθ * q - b * qprevious, q
    end
    return q
end

"""
    sphericalharmonic(l, m, θ, φ)

``Y_l^m(θ, φ)`` in the normalization documented at the top of this file.

Negative orders come from ``Y_l^{-m} = (-1)^m conj(Y_l^m)`` rather than from a second recurrence, so
there is one code path to get right.
"""
function sphericalharmonic(l::Integer, m::Integer, θ::T, φ::T) where {T<:AbstractFloat}
    order = abs(m)
    q = normalizedlegendre(l, order, θ)
    y = complex(q) * cis(order * φ)
    m >= 0 && return y
    return isodd(order) ? -conj(y) : conj(y)
end

# Linear index of the (l, m) coefficient, m ∈ -l:l, giving 1:(L+1)^2 with no gaps. The usual packing:
# every degree occupies the 2l+1 slots after the previous degree's.
_coefficientindex(l::Integer, m::Integer) = l^2 + l + m + 1

_decayweight(::Val{:none}, ::Integer, ::Type{T}) where {T} = one(T)
_decayweight(::Val{:inverse_l}, l::Integer, ::Type{T}) where {T} = one(T) / T(1 + l)
_decayweight(::Val{:inverse_l2}, l::Integer, ::Type{T}) where {T} = one(T) / T(1 + l)^2

"""
    bandcoefficients(case::RandomBandlimitedCase, ::Type{T}=Float64)

The `(L+1)^2` complex coefficients of `case`, drawn once from its seed.

Drawn here rather than inside the point loop because the coefficients define the FUNCTION: a draw per
point would evaluate a different function at every direction, and the interpolation error of noise is
not a diagnostic of anything.
"""
function bandcoefficients(case::RandomBandlimitedCase, ::Type{T}=Float64) where {T}
    rng = Random.Xoshiro(case.seed)
    coefficients = Vector{Complex{T}}(undef, (case.L + 1)^2)
    decay = Val(case.decay)
    for l in 0:(case.L)
        weight = _decayweight(decay, l, T)
        for m in (-l):l
            re = 2 * rand(rng, T) - one(T)
            im = 2 * rand(rng, T) - one(T)
            coefficients[_coefficientindex(l, m)] = weight * complex(re, im)
        end
    end
    return coefficients
end

"""
    exactvalue(case, θ, φ, coefficients=nothing)

The exact angular function of `case` at one direction.

`coefficients` is the precomputed [`bandcoefficients`](@ref) for a `RandomBandlimitedCase`, threaded
in so a grid fill draws them once rather than per point; it is ignored for a `HarmonicCase`.
"""
function exactvalue(case::HarmonicCase, θ::T, φ::T, coefficients=nothing) where {T}
    return sphericalharmonic(case.l, case.m, θ, φ)
end

# One recurrence sweep per point, accumulating every (l, m) as it goes -- O(L^2) per point. Calling
# `sphericalharmonic` for each of the (L+1)^2 pairs instead would restart the recurrence every time,
# making it O(L^3) per point and the fill O(L^5) overall on a grid sized for degree L. At L = 50 that
# is the difference between a second and several minutes.
function exactvalue(
    case::RandomBandlimitedCase, θ::T, φ::T, coefficients::AbstractVector{Complex{T}}
) where {T}
    sinθ, cosθ = sincos(θ)
    total = zero(Complex{T})

    for m in 0:(case.L)
        q = sqrt(one(T) / (4 * T(π)))
        for k in 1:m
            q *= -sqrt(T(2k + 1) / T(2k)) * sinθ
        end
        qprevious = zero(T)
        phase = cis(m * φ)

        for l in m:(case.L)
            y = complex(q) * phase
            total += coefficients[_coefficientindex(l, m)] * y
            if m > 0
                # Y_l^{-m} = (-1)^m conj(Y_l^m), the same reflection `sphericalharmonic` uses.
                total +=
                    coefficients[_coefficientindex(l, -m)] * (isodd(m) ? -conj(y) : conj(y))
            end

            l == case.L && break
            ll = l + 1
            qnext = if l == m
                sqrt(T(2m + 3)) * cosθ * q
            else
                a = sqrt((T(2ll + 1) * T(2ll - 1)) / (T(ll - m) * T(ll + m)))
                b = sqrt(
                    (T(2ll + 1) * T(ll + m - 1) * T(ll - m - 1)) /
                    (T(2ll - 3) * T(ll + m) * T(ll - m)),
                )
                a * cosθ * q - b * qprevious
            end
            qprevious = q
            q = qnext
        end
    end
    return total
end

function exactvalue(case::RandomBandlimitedCase, θ::T, φ::T, ::Nothing) where {T}
    return exactvalue(case, θ, φ, bandcoefficients(case, T))
end

"""
    fillexact!(farfield, case, sampling)

Fill a scalar far field with `case`'s exact values on `sampling`'s directions.

The array is indexed `[φindex, θindex]` -- `Base.size(::SphericalSampling)` returns `(Nφ, Nθ)`, not
`(Nθ, Nφ)`. Getting that backwards transposes the function instead of failing, which on a
`HarmonicCase` with `l == m` produces a plausible-looking error field, so the loop below is written
with φ innermost both for that reason and because φ is the fast axis.
"""
function fillexact!(
    farfield::FarFields.ScalarFarField,
    case::AnyScalarCase,
    sampling::Sampling.AbstractSphericalSampling{T},
) where {T}
    θvalues = Sampling.θs(sampling)
    φvalues = Sampling.φs(sampling)
    coefficients = case isa RandomBandlimitedCase ? bandcoefficients(case, T) : nothing
    values = farfield[1]

    for (θindex, θ) in enumerate(θvalues), (φindex, φ) in enumerate(φvalues)
        values[φindex, θindex] = exactvalue(case, θ, φ, coefficients)
    end
    return farfield
end

"""
    intermediatesampling(sourcesampling, targetsampling)

The grid that sits between a θ stage and a φ stage: `targetsampling`'s θ nodes, `sourcesampling`'s φ
nodes.

The Gauss-Legendre interpolators resample θ first, then φ. Interpolating onto this grid changes
only θ because its φ nodes match the source's. Interpolating from it to `targetsampling` changes
only φ. Use the ordinary production calls to measure these stages separately:

```julia
middle = intermediatesampling(source, target)

θonly = interpolationerror(interpolator, source, middle, cases)   # θ resampled, φ untouched
φonly = interpolationerror(interpolator, middle, target, cases)   # φ resampled, θ untouched
both  = interpolationerror([interpθ, interpφ], [source, middle, target], cases)
```

The last form measures an order split with `pθ != pφ` by passing two interpolators with different
stencil sizes. With equal orders, the chain applies the same two stages as a single interpolation.
"""
function intermediatesampling(
    sourcesampling::Sampling.SphericalSampling{T},
    targetsampling::Sampling.SphericalSampling{T},
) where {T}
    return Sampling.SphericalSampling{T}(
        Sampling.Nθ(targetsampling), Sampling.Nφ(sourcesampling)
    )
end

"""
    interpolationerror(interpolator, sourcesampling, targetsampling, cases; farfieldtype)
    interpolationerror(interpolators, samplings, cases; farfieldtype)

Interpolation error of `interpolator` taking `cases` from `sourcesampling` to `targetsampling`.

The second form measures a CHAIN: `samplings[i] -> samplings[i+1]` by `interpolators[i]`, so
`length(interpolators)` must be `length(samplings) - 1`. The exact function is evaluated on
`samplings[1]` and again on `samplings[end]`, and the stages in between pass the interpolated field
along -- so the reference is still never itself interpolated, and the reported error is the whole
chain's, including any error one stage introduces that a later stage amplifies. `rho` and
`supporteddegree` refer to `samplings[1]`, the grid the case actually starts on. See
[`intermediatesampling`](@ref) for what chains are for.

Returns an [`InterpolationErrorStudy`](@ref) -- also for a single case, which comes back as a
one-element study rather than a bare result so that [`summary`](@ref), [`rows`](@ref),
[`worstcase`](@ref) and [`pointwiseerror`](@ref) apply to whatever this function returned, without the
caller having to know which arity produced it.

The interpolator is the PRODUCTION one, called through its ordinary
`interpolator(farfield, source, target)` method. There is no reference interpolation path here on
purpose: a diagnostic that compared production against a second implementation would report the
difference between two pieces of this package rather than the error against the exact function.

# Arguments

  - `interpolator`: any `MLFMA.AbstractInterpolator`
  - `sourcesampling`: the grid the far field is known on
  - `targetsampling`: the grid to interpolate onto
  - `cases`: one case, or any iterable of cases (see [`HarmonicCase`](@ref),
    [`RandomBandlimitedCase`](@ref), [`harmoniccases`](@ref))
  - `farfieldtype`: `ScalarFarField` or `TangentialSphericalFarField`, with matching exact cases;
    other representations are rejected

# Sampling families

`AbstractSampling`, not `SphericalSampling`, so the `Equidistant` family can be measured on the same
axis as the Gauss-Legendre one -- see `benchmark/interpolation/benchmarks.jl` group 11. Nothing in the
path above is quadrature-specific: the exact function is evaluated at `MLFMA.θs`/`MLFMA.φs` and the
error is a plain norm over grid points, so no weight ever enters.

Two helpers remain specific to `SphericalSampling`:

  - [`adjointpairingerror`](@ref) checks an unweighted Hermitian pairing and its adjoint relation.
    It does not evaluate a quadrature-weighted spherical integral.
  - [`intermediatesampling`](@ref) constructs a Gauss-Legendre intermediate grid for the θ-then-φ
    interpolators. It does not construct an equidistant intermediate grid.
"""
function interpolationerror(
    interpolator,
    sourcesampling::Sampling.AbstractSphericalSampling,
    targetsampling::Sampling.AbstractSphericalSampling,
    cases;
    farfieldtype=FarFields.ScalarFarField,
)
    return interpolationerror(
        [interpolator], [sourcesampling, targetsampling], cases; farfieldtype=farfieldtype
    )
end

function interpolationerror(
    interpolators::AbstractVector,
    samplings::AbstractVector{<:Sampling.AbstractSphericalSampling},
    cases;
    farfieldtype=FarFields.ScalarFarField,
)
    _checkfarfieldtype(farfieldtype)
    _checkchain(interpolators, samplings)

    casevector = collect(_ascases(cases))
    results = [
        _errorforcase(interpolators, samplings, case, farfieldtype) for case in casevector
    ]

    stages = [
        (
            interpolator=interpolators[i],
            stencilsize=Interpolation.stencilsize(interpolators[i]),
            sourcesize=(Sampling.Nθ(samplings[i]), Sampling.Nφ(samplings[i])),
            targetsize=(Sampling.Nθ(samplings[i + 1]), Sampling.Nφ(samplings[i + 1])),
        ) for i in eachindex(interpolators)
    ]

    # Record every stage's stencil size and the trait actually used. The three-argument call uses
    # the far-field representation's default trait, not the interpolator's stored trait.
    metadata = (
        stages=stages,
        stencilsizes=[stage.stencilsize for stage in stages],
        interpolationtrait=Interpolation.defaultinterpolationtrait(farfieldtype),
        farfieldtype=farfieldtype,
        sourcesize=first(stages).sourcesize,
        targetsize=last(stages).targetsize,
        supporteddegree=supporteddegree(first(samplings)),
    )
    return InterpolationErrorStudy(results, metadata)
end

function _checkchain(interpolators, samplings)
    isempty(interpolators) &&
        throw(ArgumentError("interpolationerror: a chain needs at least one interpolator"))
    length(interpolators) == length(samplings) - 1 || throw(
        DimensionMismatch(
            "interpolationerror: got $(length(interpolators)) interpolator(s) for " *
            "$(length(samplings)) sampling(s); a chain applies interpolators[i] from " *
            "samplings[i] to samplings[i+1], so it needs exactly one fewer interpolator " *
            "than sampling.",
        ),
    )
    return nothing
end

_ascases(case::AnyCase) = (case,)
_ascases(cases) = cases

function _checkfarfieldtype(farfieldtype)
    farfieldtype === FarFields.ScalarFarField && return nothing
    farfieldtype === FarFields.TangentialSphericalFarField && return nothing
    return error(
        "InterpolationDiagnostics: only `MLFMA.ScalarFarField` and " *
        "`MLFMA.TangentialSphericalFarField` are supported, got `$farfieldtype`. The remaining " *
        "representations need exact cases that respect their own continuation and sign rules across " *
        "the θ = 0, π and φ seams; filling each component with an unrelated scalar harmonic would " *
        "run, and would report an error that means nothing physically. `SphericalVectorFarField` " *
        "additionally carries a RADIAL component, which a tangential harmonic does not supply at all.",
    )
end

function _errorforcase(interpolators, samplings, case::AnyCase, farfieldtype)
    sourcesampling = first(samplings)
    targetsampling = last(samplings)
    T = eltype(sourcesampling)

    field = farfieldtype(Complex{T}, sourcesampling, Val(:zero))
    fillexact!(field, case, sourcesampling)

    # Pass each stage's output to the next so the study includes propagated interpolation error.
    for i in eachindex(interpolators)
        field = interpolators[i](field, samplings[i], samplings[i + 1])
    end

    targetexact = farfieldtype(Complex{T}, targetsampling, Val(:zero))
    fillexact!(targetexact, case, targetsampling)

    return _compare(case, field, targetexact, sourcesampling, targetsampling)
end

function _compare(case, targetinterpolated, targetexact, sourcesampling, targetsampling)
    # Compare all components through the far-field interface, including the single scalar component.
    interpolated = [targetinterpolated[i] for i in eachindex(targetinterpolated)]
    exact = [targetexact[i] for i in eachindex(targetexact)]
    metrics = componenterrormetrics(interpolated, exact)

    # `l`/`m` come from either harmonic case type; a tangential case has them too, and dropping them
    # would leave a tangential study unable to say which mode a row is.
    l, m = if case isa HarmonicCase || case isa TangentialHarmonicCase
        (case.l, case.m)
    else
        (nothing, nothing)
    end
    L = case isa RandomBandlimitedCase ? case.L : nothing

    return InterpolationErrorResult(
        case,
        (Sampling.Nθ(sourcesampling), Sampling.Nφ(sourcesampling)),
        (Sampling.Nθ(targetsampling), Sampling.Nφ(targetsampling)),
        l,
        m,
        L,
        rho(case, sourcesampling),
        metrics.rel_l2,
        metrics.rel_linf,
        metrics.max_abs,
        metrics.component_rel_l2,
        metrics.component_rel_linf,
        metrics.vector_rel_l2,
        metrics.vector_rel_linf,
        metrics.abs_error,
        metrics.rel_error,
    )
end
