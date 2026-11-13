"""
    _relativeerror(errornorm, referencenorm)

`errornorm / referencenorm`, with the zero-reference cases defined rather than left to IEEE.

  - reference zero and error zero -> `0`. The component is identically zero and was interpolated to
    identically zero, which is exact, not undefined. This case is REAL and unavoidable: a poloidal
    `m = 0` field has `Fφ ≡ 0` everywhere, and `(l=1, m=0)` is the simplest tangential case anyone would
    reach for first.
  - reference zero and error nonzero -> `Inf`. "Relative error against a zero reference" is genuinely
    unbounded, and `Inf` says so in a way a table or a log plot can display. Raising here instead would
    take down a whole study over one component of one case.
  - otherwise the quotient.

Centralized so per-component metrics cannot each invent their own convention. Note this is deliberately
NOT what the whole-field guard in [`errormetrics`](@ref) does: a reference that is identically zero over
EVERY component means the case evaluated to nothing on the target grid, which is a broken case rather
than a zero component, and that still raises.
"""
function _relativeerror(errornorm::Real, referencenorm::Real)
    referencenorm > 0 && return errornorm / referencenorm
    iszero(errornorm) && return zero(errornorm)
    return convert(typeof(errornorm), Inf)
end

"""
    componenterrormetrics(interpolated, exact)

Error of a MULTI-COMPONENT field, as a `NamedTuple` carrying three views of the same discrepancy.

`interpolated` and `exact` are iterables of per-component matrices (e.g. `H2Factory.moments(farfield)`).

  - `rel_l2`, `rel_linf`, `max_abs` -- combined over all components, with the same definitions as in
    the single-component [`errormetrics`](@ref).
  - `component_rel_l2`, `component_rel_linf` -- per component, in component order. PRIMARY for a
    tangential field: poloidal harmonics have a `1/sinθ` factor in `Fφ`, while toroidal harmonics swap
    the component roles. Combined metrics can hide that asymmetry. Zero-reference
    components follow [`_relativeerror`](@ref) -- `0` when the error is zero too, `Inf` otherwise -- which
    a poloidal `m = 0` case needs, since its `Fφ` is identically zero.
  - `vector_rel_l2`, `vector_rel_linf` -- of the pointwise VECTOR magnitude, i.e. treating each direction's
    components as one physical vector. ATTRIBUTION ONLY: it says whether a componentwise spike is a real
    field error or the coordinate representation amplifying near a pole. Never the number that gates a
    policy decision, because production interpolates the component arrays, not the vector.

`vector_rel_l2` and `rel_l2` are mathematically equal, though rounding can differ:
`sqrt(Σ_points Σ_components |e|^2)` is the same sum either way you group it, so the combined L2 metric ALREADY
is the physical vector L2 metric. The two genuinely differ only for `linf` (a max over components and
points is not a max of vector magnitudes) and for the pointwise fields. So attribution lives in
`component_*`, `vector_rel_linf` and the θ-band profile -- not in the L2 pair.

`abs_error` and `rel_error` are pointwise fields of the VECTOR magnitude, which is what one would plot on
a sphere; for a single-component field they reduce to the scalar case unchanged.
"""
function componenterrormetrics(interpolated, exact)
    length(interpolated) == length(exact) || throw(
        DimensionMismatch(
            "componenterrormetrics: got $(length(interpolated)) interpolated components " *
            "against $(length(exact)) exact ones",
        ),
    )
    isempty(interpolated) &&
        throw(ArgumentError("componenterrormetrics: no components to compare"))

    for (i, (a, b)) in enumerate(zip(interpolated, exact))
        size(a) == size(b) || throw(
            DimensionMismatch(
                "componenterrormetrics: component $i has interpolated size $(size(a)) " *
                "against exact size $(size(b))",
            ),
        )
    end

    errors = [a .- b for (a, b) in zip(interpolated, exact)]

    componentnorms = [LinearAlgebra.norm(e) for e in errors]
    referencenorms = [LinearAlgebra.norm(b) for b in exact]
    componentmaxima = [maximum(abs, e) for e in errors]
    referencemaxima = [maximum(abs, b) for b in exact]

    # Pointwise vector magnitudes: sqrt of the summed squared components at each direction.
    squarederror = sum(abs2.(e) for e in errors)
    squaredreference = sum(abs2.(b) for b in exact)
    abs_error = sqrt.(squarederror)
    referencemagnitude = sqrt.(squaredreference)

    combinederrornorm = sqrt(sum(abs2, componentnorms))
    combinedreferencenorm = sqrt(sum(abs2, referencenorms))
    combinedreferencemax = maximum(referencemaxima)

    combinedreferencenorm > 0 && combinedreferencemax > 0 || error(
        "componenterrormetrics: the exact reference is identically zero across EVERY component, so " *
        "no relative error is defined. This means the case evaluated to nothing on the target grid " *
        "-- check the case rather than dividing by zero. (A single zero component is fine and is " *
        "reported as 0 or Inf per component.)",
    )

    floorvalue = eps(eltype(referencemagnitude))
    rel_error = abs_error ./ max.(referencemagnitude, floorvalue)

    return (
        rel_l2=_relativeerror(combinederrornorm, combinedreferencenorm),
        rel_linf=_relativeerror(maximum(componentmaxima), combinedreferencemax),
        max_abs=maximum(componentmaxima),
        component_rel_l2=[
            _relativeerror(e, r) for (e, r) in zip(componentnorms, referencenorms)
        ],
        component_rel_linf=[
            _relativeerror(e, r) for (e, r) in zip(componentmaxima, referencemaxima)
        ],
        vector_rel_l2=_relativeerror(
            LinearAlgebra.norm(abs_error), LinearAlgebra.norm(referencemagnitude)
        ),
        vector_rel_linf=_relativeerror(maximum(abs_error), maximum(referencemagnitude)),
        abs_error=abs_error,
        rel_error=rel_error,
    )
end

"""
    errormetrics(interpolated, exact)

Global and pointwise error of `interpolated` against `exact`, as a `NamedTuple`.

Fields: `rel_l2`, `rel_linf`, `max_abs`, `abs_error`, `rel_error`.

Single-component form. For a multi-component representation use [`componenterrormetrics`](@ref), which
adds per-component and vector-magnitude views while keeping these three names meaning the same thing.

TWO KINDS OF NORMALIZATION, and they are not interchangeable:

  - The GLOBAL metrics divide by a norm of the whole reference. `rel_l2` is
    `norm(err) / norm(exact)` and `rel_linf` is `maximum(abs, err) / maximum(abs, exact)`. Both are
    well posed for any nonzero reference, which is why the summary and the worst-case ranking use
    them.
  - The POINTWISE `rel_error` divides each point by its own reference magnitude, floored at `eps`.
    Useful for a plot, because it shows where the error is large relative to the local signal --
    but a harmonic has genuine zeros, and near them a pointwise relative error is a ratio of two
    small numbers that says more about the zero than about the interpolator. The floor keeps such
    points finite instead of `Inf`/`NaN`, and it is a display convenience, not a metric: never rank
    or report accuracy from `maximum(rel_error)`.

`abs_error` is stored alongside precisely so a reader has an unnormalized field to fall back on.
"""
function errormetrics(interpolated::AbstractMatrix, exact::AbstractMatrix)
    size(interpolated) == size(exact) || throw(
        DimensionMismatch(
            "errormetrics: interpolated size $(size(interpolated)) does not match exact size " *
            "$(size(exact))",
        ),
    )

    err = interpolated .- exact
    abs_error = abs.(err)
    referencemagnitude = abs.(exact)

    exactnorm = LinearAlgebra.norm(exact)
    exactmax = maximum(referencemagnitude)
    exactnorm > 0 && exactmax > 0 || error(
        "errormetrics: the exact reference is identically zero, so no relative error is " *
        "defined. This means the case evaluated to nothing on the target grid -- check the case " *
        "rather than dividing by zero.",
    )

    floorvalue = eps(eltype(referencemagnitude))
    rel_error = abs_error ./ max.(referencemagnitude, floorvalue)

    return (
        rel_l2=LinearAlgebra.norm(err) / exactnorm,
        rel_linf=maximum(abs_error) / exactmax,
        max_abs=maximum(abs_error),
        abs_error=abs_error,
        rel_error=rel_error,
    )
end
