"""
    InterpolationErrorResult

One case's interpolation error: the case, the two grid sizes, its bandwidth, the global metrics and
the pointwise error fields.

Grid sizes use `(Nθ, Nφ)`, not `Base.size(sampling)`. The pointwise arrays keep the far field's
storage shape, `(Nφ, Nθ)`, with φ as the fast axis.

`l`, `m` and `L` are `nothing` for the case kinds that do not define them, so [`rows`](@ref) can
produce uniform `NamedTuple`s across mixed studies.

THREE VIEWS OF THE SAME ERROR, and which one gates a decision matters:

  - `rel_l2`, `rel_linf`, `max_abs` -- combined over all components, with the same meaning as the
    single-component metrics.
  - `component_rel_l2`, `component_rel_linf` -- per component, in component order. PRIMARY for a
    tangential field: poloidal harmonics have a `1/sinθ` factor in `Fφ`; toroidal harmonics swap the
    component roles. Combined metrics can hide this asymmetry. A zero reference component (a poloidal `m = 0`
    field's `Fφ`) reports `0` when its error is zero too and `Inf` otherwise.
  - `vector_rel_l2`, `vector_rel_linf` -- of the pointwise tangent-vector magnitude. ATTRIBUTION ONLY:
    they say whether a componentwise spike is a real field error or the coordinate representation
    amplifying near a pole. Production interpolates the component arrays, so these must not be the
    number a policy is decided on. Note `vector_rel_l2 == rel_l2` identically -- the same sum grouped
    two ways -- so the attribution value is in `vector_rel_linf`, the per-component pair, and the θ-band
    profile, not in the L2 numbers.
"""
struct InterpolationErrorResult{Case,T,C,A}
    case::Case
    source_size::Tuple{Int,Int}
    target_size::Tuple{Int,Int}
    l::Union{Int,Nothing}
    m::Union{Int,Nothing}
    L::Union{Int,Nothing}
    rho::Float64
    rel_l2::T
    rel_linf::T
    max_abs::T
    component_rel_l2::C
    component_rel_linf::C
    vector_rel_l2::T
    vector_rel_linf::T
    abs_error::A
    rel_error::A
end

"""
    InterpolationErrorStudy

A vector of [`InterpolationErrorResult`](@ref)s plus the metadata they share.

The metadata holds what was held FIXED across the study, so a result never has to restate it and two
studies cannot be compared without their conditions being visible:

  - `stages` -- one `NamedTuple` per interpolation stage, each with that stage's `interpolator`,
    `stencilsize`, `sourcesize` and `targetsize`. A single-stage study has one entry; a chain (see
    [`intermediatesampling`](@ref)) has one per link.
  - `stencilsizes` -- the stages' stencil sizes, in order, for tabulating.
  - `interpolationtrait`, `farfieldtype` -- the trait the interpolation actually ran under, and the
    representation it ran on.
  - `sourcesize`, `targetsize` -- the ENDPOINTS, `(Nθ, Nφ)` each: the first stage's source and the last
    stage's target. Any intermediate grid is in `stages`, not here.
  - `supporteddegree` -- of the source grid, the ρ denominator.

There is deliberately no singular `interpolator`/`stencilsize` pair: a chain has no single stencil size,
and a field reporting the first stage's would label a `pθ = 20, pφ = 8` study as `{20}`. Read
`only(stages)` for a single-stage study.
"""
struct InterpolationErrorStudy{R,M}
    results::Vector{R}
    metadata::M
end

"""
    results(study)

The study's [`InterpolationErrorResult`](@ref)s, in the order their cases were given.
"""
results(study::InterpolationErrorStudy) = study.results

"""
    metadata(study)

The `NamedTuple` of conditions held fixed across `study`.
"""
metadata(study::InterpolationErrorStudy) = study.metadata

Base.length(study::InterpolationErrorStudy) = length(study.results)
Base.getindex(study::InterpolationErrorStudy, i) = study.results[i]
Base.iterate(study::InterpolationErrorStudy, state...) = iterate(study.results, state...)
Base.eltype(::Type{InterpolationErrorStudy{R,M}}) where {R,M} = R

"""
    rows(study)

One `NamedTuple` per result, holding only scalars -- ready for a table.

Deliberately drops the pointwise arrays: this is the form you tabulate, and a `DataFrame` of 20,000
element matrices is not a table. Build one outside MLFMA, which is not a dependency here:

```julia
using DataFrames
DataFrame(MLFMA.InterpolationDiagnostics.rows(study))
```

Every row carries the same field names regardless of case kind, with `nothing` where a kind has no
such parameter -- otherwise a mixed study would produce a heterogeneous vector that no table
constructor accepts.
"""
function rows(study::InterpolationErrorStudy)
    return [
        (
            casekind=casekind(result.case),
            l=result.l,
            m=result.m,
            L=result.L,
            bandwidth=casebandwidth(result.case),
            rho=result.rho,
            source_Nθ=result.source_size[1],
            source_Nφ=result.source_size[2],
            target_Nθ=result.target_size[1],
            target_Nφ=result.target_size[2],
            rel_l2=result.rel_l2,
            rel_linf=result.rel_linf,
            max_abs=result.max_abs,
        ) for result in study.results
    ]
end

"""
    worstcase(study; by=:rel_l2)

The single stored result with the largest `by` metric.

Returns the result itself, not a copy or a summary, so its pointwise fields are available for a plot
of exactly the case that was worst.

`by` must be one of `:rel_l2`, `:rel_linf`, `:max_abs` -- the global metrics. Pointwise `rel_error` is
deliberately not rankable: see [`errormetrics`](@ref) for why its floor makes it a display aid.
"""
function worstcase(study::InterpolationErrorStudy; by::Symbol=:rel_l2)
    by in (:rel_l2, :rel_linf, :max_abs) || throw(
        ArgumentError(
            "worstcase: `by` must be one of :rel_l2, :rel_linf, :max_abs, got :$by"
        ),
    )
    isempty(study.results) && throw(ArgumentError("worstcase: the study has no results"))
    return argmax(result -> getfield(result, by), study.results)
end

"""
    θbandprofile(study, i; bands=4)

Where the error sits in θ: the share of the total squared error in each band of θ rows, counted in from
each pole.

Returns a `NamedTuple` of `(north, south, rowsperband, nbands)`, where `north[k]` and `south[k]` are the
fractions of the total squared error in the `k`-th band from θ = 0 and from θ = π. The two hemispheres are
kept APART rather than summed: an asymmetry between them is what a sign error in the pole continuation
would produce, and averaging the two would hide exactly that.

The binning is chosen per call rather than stored in the result.

Tangential components can be less smooth near the poles than the physical vector field. Compare
component metrics and scalar profiles to investigate a concentration of error there; the profile
alone does not identify its cause.

`bands` is the number of θ indices per band, with all φ samples included. The last band in each
hemisphere includes its remaining interior indices. On an odd-sized θ grid, the central index is
excluded from both hemispheres but included in the total used for normalization.
"""
function θbandprofile(study::InterpolationErrorStudy, i::Integer; bands::Integer=4)
    bands >= 1 || throw(ArgumentError("θbandprofile: bands must be >= 1, got $bands"))
    field = study.results[i].abs_error
    Nφ, Nθ = size(field)
    2 * bands <= Nθ || throw(
        ArgumentError(
            "θbandprofile: $bands rows per band needs at least $(2 * bands) θ rows, grid has $Nθ",
        ),
    )

    total = sum(abs2, field)
    total > 0 || throw(
        ArgumentError(
            "θbandprofile: result $i has zero error everywhere, so no profile is defined",
        ),
    )

    half = Nθ ÷ 2
    edges = collect(1:bands:half)
    north = Float64[]
    south = Float64[]
    for (k, first_) in enumerate(edges)
        last_ = k == length(edges) ? half : min(first_ + bands - 1, half)
        push!(north, sum(abs2, view(field, :, first_:last_)) / total)
        push!(south, sum(abs2, view(field, :, (Nθ + 1 - last_):(Nθ + 1 - first_))) / total)
    end
    return (north=north, south=south, rowsperband=bands, nbands=length(edges))
end

"""
    pointwiseerror(study, i; kind=:relative)

The pointwise error field of result `i`, shaped like the target far field, i.e. `(Nφ, Nθ)`.

`kind` is `:relative` or `:absolute`. Prefer `:absolute` when the case has zeros on the target grid;
`:relative`'s per-point denominator is floored at `eps` there, so it stays finite but stops meaning
what its name suggests.
"""
function pointwiseerror(study::InterpolationErrorStudy, i::Integer; kind::Symbol=:relative)
    kind in (:relative, :absolute) || throw(
        ArgumentError("pointwiseerror: `kind` must be :relative or :absolute, got :$kind"),
    )
    result = study.results[i]
    return kind === :relative ? result.rel_error : result.abs_error
end
