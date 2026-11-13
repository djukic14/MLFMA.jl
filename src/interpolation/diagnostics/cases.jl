"""
    HarmonicCase(; l, m)

A single spherical harmonic ``Y_l^m`` as the exact angular function to interpolate.

The sharpest diagnostic available: a single harmonic has one known bandwidth, so the error it
produces is attributable to that degree alone rather than to a mixture. `l` fixes how fast the
function oscillates and `m` fixes in which direction, which is why the [`harmoniccases`](@ref)
families vary `m` at fixed `l` -- an interpolator that is separable in θ and φ does not treat
``m = 0`` and ``m = l`` alike.

# Arguments

  - `l`: harmonic degree, `l >= 0`
  - `m`: harmonic order, `abs(m) <= l`
"""
struct HarmonicCase
    l::Int
    m::Int
end

function HarmonicCase(; l::Integer, m::Integer)
    l >= 0 || throw(ArgumentError("HarmonicCase: degree l must be >= 0, got l = $l"))
    abs(m) <= l || throw(
        ArgumentError("HarmonicCase: order m must satisfy abs(m) <= l, got l = $l, m = $m"),
    )
    return HarmonicCase(Int(l), Int(m))
end

"""
    RandomBandlimitedCase(; L, seed=0x12345678, decay=:none)

A random superposition of every harmonic up to degree `L`, as the exact angular function.

Complements [`HarmonicCase`](@ref): a single harmonic can flatter an interpolator that happens to
resolve one direction of oscillation well, while a full band exercises every `(l, m)` at once and is
closer in character to an actual far field. `seed` makes it reproducible -- two runs of the same case
compare a bit-identical function, which is what lets a study be a regression baseline rather than a
sample.

`decay` weights the coefficients by degree, since a physical far field is not white across the band:

  - `:none` -- every degree weighted equally, the harshest of the three
  - `:inverse_l` -- weight `1 / (1 + l)`
  - `:inverse_l2` -- weight `1 / (1 + l)^2`

# Arguments

  - `L`: highest degree in the band, `L >= 0`
  - `seed`: seed for the coefficient draw
  - `decay`: one of `:none`, `:inverse_l`, `:inverse_l2`
"""
struct RandomBandlimitedCase
    L::Int
    seed::UInt64
    decay::Symbol
end

const SUPPORTEDDECAYS = (:none, :inverse_l, :inverse_l2)

function RandomBandlimitedCase(; L::Integer, seed::Integer=0x12345678, decay::Symbol=:none)
    L >= 0 || throw(ArgumentError("RandomBandlimitedCase: L must be >= 0, got L = $L"))
    decay in SUPPORTEDDECAYS || throw(
        ArgumentError(
            "RandomBandlimitedCase: decay must be one of " *
            "$(join(SUPPORTEDDECAYS, ", ")), got :$decay",
        ),
    )
    return RandomBandlimitedCase(Int(L), UInt64(seed), decay)
end

"""
    TangentialHarmonicCase(; kind, l, m)

A single vector spherical harmonic as the exact TANGENTIAL field to interpolate.

`kind` is `:poloidal` (``B_l^m = ∇_Ω Y / sqrt(l(l+1))``) or `:toroidal`
(``C_l^m = r̂ × ∇_Ω Y / sqrt(l(l+1))``). Together the two families span every band-limited tangential
field, i.e. exactly what a far-field pattern is -- so these are the cases a `TangentialSphericalFarField`
actually has to carry.

Why not two scalar harmonics in the two components: the interpolator continues both with
`VirtualExtensionFlipθSign`, `f(θ,φ) = -f(-θ,φ+π)`, and unrelated scalars do not satisfy that. A genuine
tangential field does, automatically -- see `vectorharmonics.jl`. Getting this wrong would measure the
reference disagreeing with the continuation rule and call it interpolation error.

`l >= 1`: there is no degree-zero tangential harmonic.

# Arguments

  - `kind`: `:poloidal` or `:toroidal`
  - `l`: harmonic degree, `l >= 1`
  - `m`: harmonic order, `abs(m) <= l`
"""
struct TangentialHarmonicCase
    kind::Symbol
    l::Int
    m::Int
end

function TangentialHarmonicCase(; kind::Symbol, l::Integer, m::Integer)
    kind in SUPPORTEDVECTORKINDS || throw(
        ArgumentError(
            "TangentialHarmonicCase: kind must be one of " *
            "$(join(SUPPORTEDVECTORKINDS, ", ")), got :$kind",
        ),
    )
    l >= 1 || throw(
        ArgumentError(
            "TangentialHarmonicCase: degree l must be >= 1, got l = $l. The surface gradient of " *
            "a constant vanishes, so there is no degree-zero tangential harmonic.",
        ),
    )
    abs(m) <= l || throw(
        ArgumentError(
            "TangentialHarmonicCase: order m must satisfy abs(m) <= l, got l = $l, m = $m",
        ),
    )
    return TangentialHarmonicCase(kind, Int(l), Int(m))
end

"""
    RandomTangentialBandlimitedCase(; L, seed=0x12345678, decay=:none)

A random superposition of every poloidal AND toroidal harmonic up to degree `L`, as the exact tangential
field.

The tangential counterpart of [`RandomBandlimitedCase`](@ref), and wanted for the same reason: a single
vector harmonic can flatter an interpolator that happens to resolve one direction well, while a full band
mixes sectoral and non-sectoral content in one field the way a real far-field pattern does. Without it,
tangential evidence rests on single modes while the scalar evidence has both -- which is not a fair
comparison to decide an interpolator policy on.

BOTH FAMILIES IN ONE FIELD, not two studies averaged: a physical far field is a superposition of poloidal
and toroidal parts, and their `Fθ`/`Fφ` structure differs (they are component-swapped mirrors), so a
mixture exercises interactions that neither family alone can.

`L >= 1`, since there is no degree-zero tangential harmonic. `decay` weights by degree exactly as the
scalar case does.

# Arguments

  - `L`: highest degree in the band, `L >= 1`
  - `seed`: seed for the coefficient draw
  - `decay`: one of `:none`, `:inverse_l`, `:inverse_l2`
"""
struct RandomTangentialBandlimitedCase
    L::Int
    seed::UInt64
    decay::Symbol
end

function RandomTangentialBandlimitedCase(;
    L::Integer, seed::Integer=0x12345678, decay::Symbol=:none
)
    L >= 1 || throw(
        ArgumentError(
            "RandomTangentialBandlimitedCase: L must be >= 1, got L = $L. There is no " *
            "degree-zero tangential harmonic.",
        ),
    )
    decay in SUPPORTEDDECAYS || throw(
        ArgumentError(
            "RandomTangentialBandlimitedCase: decay must be one of " *
            "$(join(SUPPORTEDDECAYS, ", ")), got :$decay",
        ),
    )
    return RandomTangentialBandlimitedCase(Int(L), UInt64(seed), decay)
end

"""
    AnyTangentialCase

The case types that evaluate to a tangential `(Fθ, Fφ)` pair per direction.
"""
const AnyTangentialCase = Union{TangentialHarmonicCase,RandomTangentialBandlimitedCase}

"""
    AnyCase

The case types this module knows how to evaluate exactly.

A `Union` rather than an abstract type with subtypes: the set is closed on purpose, so adding a case type
is a deliberate edit here (and to `exactvalue` or `fillexact!`) rather than something that silently
type-checks and then fails at evaluation.

Not every case fits every representation: the scalar cases fill a `ScalarFarField` and
[`TangentialHarmonicCase`](@ref) fills a `TangentialSphericalFarField`. `fillexact!` dispatches on the
pair, so a mismatch is a `MethodError` at the fill rather than a plausible-looking number.
"""
const AnyCase = Union{
    HarmonicCase,
    RandomBandlimitedCase,
    TangentialHarmonicCase,
    RandomTangentialBandlimitedCase,
}

"""
    AnyScalarCase

The case types that evaluate to a single complex value per direction.

Exists so the scalar-only entry points can say so in their signatures rather than checking at run time.
"""
const AnyScalarCase = Union{HarmonicCase,RandomBandlimitedCase}

"""
    casebandwidth(case)

The highest harmonic degree `case` contains.

The numerator of [`rho`](@ref): what the exact function demands, against what a grid supports.
"""
casebandwidth(case::HarmonicCase) = case.l
casebandwidth(case::RandomBandlimitedCase) = case.L
casebandwidth(case::TangentialHarmonicCase) = case.l
casebandwidth(case::RandomTangentialBandlimitedCase) = case.L

"""
    casekind(case)

A short `Symbol` naming the case type, for grouping rows and curves.

Read off the type rather than stored, so it cannot disagree with the case it labels.
"""
casekind(::HarmonicCase) = :harmonic
casekind(::RandomBandlimitedCase) = :randombandlimited
# The FAMILY is part of the kind, not a detail: poloidal and toroidal are different fields with
# different component structure, and grouping them under one label would merge two curves in a plot.
casekind(case::TangentialHarmonicCase) = case.kind === :poloidal ? :poloidal : :toroidal
casekind(::RandomTangentialBandlimitedCase) = :randomtangential

"""
    harmoniccases(degrees; families=(:axisymmetric, :mixed, :sectoral))

One [`HarmonicCase`](@ref) per `(degree, family)` pair.

The families pick `m` from `l`, which is the comparison worth making: at a fixed degree, how the
error depends on the direction the function oscillates in.

  - `:axisymmetric` -- `m = 0`, no φ dependence at all, so this isolates the θ interpolation
  - `:mixed` -- `m = round(Int, l / 2)`
  - `:sectoral` -- `m = l`, the fastest φ oscillation the degree allows

Called "families" rather than "modes" deliberately: `mode` already means something physical in an
MLFMA, and these are diagnostic inputs, not modes of the operator.

# Arguments

  - `degrees`: the `l` values to cover
  - `families`: any of `:axisymmetric`, `:mixed`, `:sectoral`
"""
function harmoniccases(degrees; families=(:axisymmetric, :mixed, :sectoral))
    supported = (:axisymmetric, :mixed, :sectoral)
    for family in families
        family in supported || throw(
            ArgumentError(
                "harmoniccases: unknown family :$family, expected one of " *
                "$(join(supported, ", "))",
            ),
        )
    end

    cases = HarmonicCase[]
    for l in degrees, family in families
        m = if family === :axisymmetric
            0
        elseif family === :mixed
            round(Int, l / 2)
        else
            l
        end
        push!(cases, HarmonicCase(; l=l, m=m))
    end
    return cases
end
