# Bandwidth labels use `rho = casebandwidth / supporteddegree`.
# `supporteddegree` is the sampling's `cutoffparameter`, the inverse of the standard rate rule:
# `MLFMA.NθNφ(L)` returns `(2(L+1), 4(L+1))`, and `ceil(Nθ/2) - 1` recovers L.
#
# This design degree describes quadrature support, not the largest representable single harmonic.
# For example, `min(Nθ - 1, Nφ ÷ 2)` reports 21 on `NθNφ(10)`, rather than the design degree 10.
# Use ρ to compare diagnostic cases. Choose physical grids through the sampling strategy, which
# also accounts for box size, wavenumber, and target accuracy.

"""
    supporteddegree(sampling)

The sampling's design degree, `MLFMA.cutoffparameter(sampling)`, not its single-field aliasing limit.

A grid built with `NθNφ(L)` has `supporteddegree == L`. Gauss-Legendre and equidistant samplings use
the same inverse, `ceil(Nθ/2) - 1`, so equal angular counts give equal ρ labels.
This helper accepts spherical samplings only; it does not choose the degree a physical level needs.
"""
supporteddegree(sampling::Sampling.AbstractSphericalSampling) =
    Sampling.cutoffparameter(sampling)

"""
    rho(case, sampling)

`casebandwidth(case) / supporteddegree(sampling)`.

Values up to 1 place the case within the grid's design degree. Larger values exceed that degree,
but do not by themselves prove aliasing: the cutoff describes quadrature support, not the limit
for representing a single field. Reported as a `Float64` to label diagnostic plots.
"""
function rho(case, sampling::Sampling.AbstractSphericalSampling)
    degree = supporteddegree(sampling)
    degree > 0 || throw(
        ArgumentError(
            "rho: the sampling supports degree $degree, so ρ is not defined. A grid this " *
            "coarse (Nθ = $(Sampling.Nθ(sampling))) cannot represent any varying harmonic.",
        ),
    )
    return Float64(casebandwidth(case)) / Float64(degree)
end

"""
    summary([io], study)

A compact table of `study`: its fixed conditions, then one line per case.

Extends `Base.summary`, so `MLFMA.InterpolationDiagnostics.summary(study)` returns the `String` and
the two-argument form prints. Rows are shown in the order the cases were given, NOT sorted by error --
use [`worstcase`](@ref) for the ranking, so that a study read next to its definition lines up with it.
"""
function Base.summary(io::IO, study::InterpolationErrorStudy)
    meta = metadata(study)
    println(
        io,
        "InterpolationErrorStudy: ",
        length(study),
        " case(s), ",
        length(meta.stages),
        length(meta.stages) == 1 ? " stage, " : " stages, ",
        nameof(typeof(meta.interpolationtrait)),
    )
    println(
        io,
        "  ",
        meta.sourcesize[1],
        "x",
        meta.sourcesize[2],
        " -> ",
        meta.targetsize[1],
        "x",
        meta.targetsize[2],
        " (Nθ x Nφ), source supports degree ",
        meta.supporteddegree,
        ", ",
        nameof(meta.farfieldtype),
    )

    # Print every stage's interpolator and stencil size so chains with equal endpoints remain distinct.
    for (i, stage) in enumerate(meta.stages)
        println(
            io,
            "  stage ",
            i,
            ": ",
            rpad(
                string(nameof(typeof(stage.interpolator)), "{", stage.stencilsize, "}"), 38
            ),
            stage.sourcesize[1],
            "x",
            stage.sourcesize[2],
            " -> ",
            stage.targetsize[1],
            "x",
            stage.targetsize[2],
        )
    end
    println(io, "")
    println(io, "  case                     bandwidth      rho      rel_l2    rel_linf")

    for result in results(study)
        label = if result.case isa HarmonicCase
            "harmonic l=$(result.l) m=$(result.m)"
        else
            "random L=$(result.L) $(result.case.decay)"
        end
        println(
            io,
            "  ",
            rpad(label, 25),
            lpad(casebandwidth(result.case), 9),
            lpad(_format(result.rho, 4), 9),
            lpad(_format(result.rel_l2, 4), 12),
            lpad(_format(result.rel_linf, 4), 12),
        )
    end
    return nothing
end

function Base.summary(study::InterpolationErrorStudy)
    buffer = IOBuffer()
    Base.summary(buffer, study)
    return String(take!(buffer))
end

function Base.show(io::IO, ::MIME"text/plain", study::InterpolationErrorStudy)
    return Base.summary(io, study)
end

# Fixed-significand formatting without a Printf dependency: `round(x; sigdigits)` then `string`, which
# is enough for a diagnostic table and keeps this module dependency-free.
_format(x::Real, sigdigits::Int) = string(round(x; sigdigits=sigdigits))
