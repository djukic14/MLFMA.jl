# Accuracy and runtime studies for MLFMA's angular interpolation (`src/interpolation/`).
#
# DEFINES ONLY -- including this file builds the grids, the case sets, the study definitions and
# `TIMINGS`, and runs nothing. `benchmark/interpolation/run.jl` is the entry point. Including it is
# cheap: no mesh, no gmsh, no BEAST, no tree.
#
# SEPARATE FROM `benchmark/benchmarks.jl` on purpose, and deliberately NOT named `SUITE`. That file
# measures the far-field layer and is the PkgBenchmark entry point; this one measures interpolation,
# and most of what it produces is not a `BenchmarkTools.Trial` at all. Two files defining `SUITE`
# differently in one repo would invite including both.
#
# ACCURACY AND RUNTIME ARE REPORTED SEPARATELY, and that split is the point of the file rather than a
# formatting choice. An interpolation change is only an improvement if it is one on both axes, and the
# two are measured by different machinery: accuracy by `MLFMA.InterpolationDiagnostics` against exact
# angular functions, runtime by `BenchmarkTools` over the same grids. A faster kernel that loses a
# digit is a regression, and nothing in a `Trial` would say so.
#
# NO BEAST ASSEMBLY ANYWHERE HERE. The reference is always a closed-form spherical harmonic evaluated
# independently on both grids, so every number below is interpolation error alone -- no quadrature, no
# mesh, no operator, no translation, no addition-theorem truncation, no tree. That is what makes these
# studies usable as a before/after for an interpolation refactor.
#
# HOW TO READ ρ (also in this directory's README):
#
#     ρ = casebandwidth(case) / supporteddegree(sourcesampling)
#
# `supporteddegree` is `MLFMA.cutoffparameter`, the exact inverse of `MLFMA.NθNφ`, so ρ = 1 is
# precisely where the SOURCE grid stops carrying the case. Below 1 the error is the interpolator's;
# at and above 1 the grid does not represent the function at all and a large error is arithmetic, not
# a defect. Rows past ρ = 1 are kept because they are the reference point that makes the in-band
# numbers mean something.

using BenchmarkTools
using Printf: @sprintf
# The level-transfer timing group applies transfers through H2Factory's own direction/application path
# rather than a local imitation of it, so the numbers describe what the traversal actually runs.
using H2Factory
using MLFMA
using MLFMA.Equidistant:
    EquidistantSampling, EquidistantInterpolator, FFTBackend, LagrangeBackend
using StaticArrays

const ID = MLFMA.InterpolationDiagnostics

# Grids ###########################################################################################
#
# Built from a degree through `MLFMA.NθNφ` rather than by hand, so `supporteddegree(SOURCE)` is
# exactly `SOURCEDEGREE` and ρ reads directly as a fraction of the source band. Coarse -> fine is the
# aggregation direction in a real MLFMA, and the target is the usual level-to-level doubling.

const SOURCEDEGREE = 13

sampling(degree) = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(degree)...)

const SOURCE = sampling(SOURCEDEGREE)
const TARGET = sampling(2 * SOURCEDEGREE)

# The grid between the θ stage and the φ stage: the target's θ nodes, the source's φ nodes. Both
# production interpolators resample θ and φ separably, θ first, so a study onto MIDDLE is the θ half
# alone and a study from MIDDLE is the φ half alone -- see `ID.intermediatesampling`.
const MIDDLE = ID.intermediatesampling(SOURCE, TARGET)

# Cases ###########################################################################################

# The four shapes worth telling apart at one degree: no φ dependence, mixed, the fastest φ
# oscillation the degree allows, and a full random band that exercises every (l, m) at once.
function casesatdegree(degree)
    return vcat(
        ID.harmoniccases([degree]), [ID.RandomBandlimitedCase(; L=degree, seed=0x1234)]
    )
end

# Degrees spanning both sides of ρ = 1 for the source grid (bandwidth 13), so a sweep shows the knee
# rather than only its in-band side.
const SWEEPDEGREES = 2:4:(2 * SOURCEDEGREE)

const SWEEPCASES = vcat(
    ID.harmoniccases(SWEEPDEGREES),
    [ID.RandomBandlimitedCase(; L=L, seed=0x1234) for L in SWEEPDEGREES],
)

const FIXEDDEGREE = 10
const FIXEDCASES = casesatdegree(FIXEDDEGREE)

lagrange(p) = MLFMA.SphericalLagrangeInterpolator{p}(ComplexF64, Float64)
localglobal(p) = MLFMA.LocalθGlobalφInterpolator{p}(ComplexF64, Float64)

# Reporting helpers ###############################################################################

function caselabel(case::ID.HarmonicCase)
    family = if case.m == 0
        "axisym"
    elseif case.m == case.l
        "sectoral"
    else
        "mixed"
    end
    return "Y_$(case.l)^$(case.m) $family"
end
# The decay is part of the label, not decoration: a study may hold two random cases at the same L
# differing only in it, and two rows reading `random L=10` would be indistinguishable.
function caselabel(case::ID.RandomBandlimitedCase)
    return "random L=$(case.L) $(case.decay)"
end
# The tangential cases, which group 10 labels inline because it needs a narrower column. Group 11
# reuses the shared label so its scalar and tangential tables can be read against each other.
caselabel(case::ID.TangentialHarmonicCase) = "$(case.kind) Y_$(case.l)^$(case.m)"
caselabel(case::ID.RandomTangentialBandlimitedCase) = "random tangential L=$(case.L)"

# `@sprintf` rather than `string(round(x; sigdigits=3))`, which is what this was and which BREAKS THE
# TABLE for very small values: `round(3.12e-21; sigdigits=3)` is not exactly representable, so `string`
# prints `3.1200000000000004e-21` -- 22 characters into a 13-wide column, running two columns together
# in exactly the group whose job is to be read as evidence for a decision.
_number(x::Real) = @sprintf("%.3g", x)
_number(::Nothing) = "-"

"""
    pivot(io, title, rowlabels, collabels, cells; note="")

Print one accuracy table: cases down the side, the swept parameter across the top.

A pivot rather than one row per (case, parameter) pair: the question every group here asks is how a
column changes at fixed case, and a long-format table makes that comparison a search.
"""
function pivot(io::IO, title, rowlabels, collabels, cells; note::AbstractString="")
    println(io, "\n", title)
    isempty(note) || println(io, note)

    labelwidth = maximum(length, rowlabels) + 2
    widths = [max(length(string(label)), 11) + 2 for label in collabels]

    print(io, " "^labelwidth)
    for (label, width) in zip(collabels, widths)
        print(io, lpad(string(label), width))
    end
    println(io)

    for (i, rowlabel) in enumerate(rowlabels)
        print(io, rpad(rowlabel, labelwidth))
        for (j, width) in enumerate(widths)
            print(io, lpad(_number(cells[i, j]), width))
        end
        println(io)
    end
    return nothing
end

"""
    errormatrix(cases, columns, metric=:rel_l2)

`cells[i, j]`: the `metric` of `cases[i]` under `columns[j]`.

`columns[j]` is a callable taking the case vector and returning a study, so a column is free to be a
single stage, a chain, or a different interpolator entirely -- which is what lets one printer serve
every group below.
"""
function errormatrix(cases, columns, metric::Symbol=:rel_l2)
    cells = Matrix{Float64}(undef, length(cases), length(columns))
    for (j, column) in enumerate(columns)
        study = column(cases)
        for (i, result) in enumerate(ID.results(study))
            cells[i, j] = getfield(result, metric)
        end
    end
    return cells
end

# Accuracy studies ################################################################################
#
# Each entry is `(name, note, run)`. `run(io)` prints that group's tables. Kept as thunks so this
# file stays a definition file: `run.jl` decides what to execute and in what order.

const STENCILS = (6, 8, 10, 12, 14, 16, 20, 24)

# 1. Stencil/order sweep ##########################################################################

function stencilsweep(io)
    labels = [caselabel(case) for case in SWEEPCASES]
    rholabels = [
        "$(label)  ρ=$(_number(ID.rho(case, SOURCE)))" for
        (label, case) in zip(labels, SWEEPCASES)
    ]
    columns = [
        cases -> ID.interpolationerror(lagrange(p), SOURCE, TARGET, cases) for p in STENCILS
    ]

    pivot(
        io,
        "1a. rel_l2 versus stencil size -- SphericalLagrangeInterpolator",
        rholabels,
        ["p=$p" for p in STENCILS],
        errormatrix(SWEEPCASES, columns),
    )
    pivot(
        io,
        "1b. rel_linf versus stencil size -- SphericalLagrangeInterpolator",
        rholabels,
        ["p=$p" for p in STENCILS],
        errormatrix(SWEEPCASES, columns, :rel_linf),
        ;
        note="  The worst single direction rather than the norm: a pole or seam artefact shows here first.",
    )
    return nothing
end

# 2. Interpolator-family comparison ###############################################################

function familycomparison(io)
    cases = vcat(
        ID.harmoniccases([6, 10, 20]),
        [
            ID.RandomBandlimitedCase(; L=10, seed=0x1234),
            ID.RandomBandlimitedCase(; L=10, seed=0x1234, decay=:inverse_l2),
        ],
    )
    orders = (8, 20)
    columns = vcat(
        [
            cases -> ID.interpolationerror(lagrange(p), SOURCE, TARGET, cases) for
            p in orders
        ],
        [
            cases -> ID.interpolationerror(localglobal(p), SOURCE, TARGET, cases) for
            p in orders
        ],
    )
    collabels = vcat(
        ["lagrange p=$p" for p in orders], ["localglobal p=$p" for p in orders]
    )

    pivot(
        io,
        "2. rel_l2 by interpolator family, same grids and cases",
        ["$(caselabel(case))  ρ=$(_number(ID.rho(case, SOURCE)))" for case in cases],
        collabels,
        errormatrix(cases, columns),
        ;
        note="  `localglobal` is global (FFT) in φ and local in θ, so its θ column is the same code as " *
             "`lagrange`'s.\n  Differences are therefore attributable to φ alone.",
    )
    return nothing
end

# 3. Sampling-margin sweep ########################################################################

function marginsweep(io)
    degrees = (13, 16, 20, 26, 32)
    orders = (8, 20)

    for p in orders
        columns = [
            let source = sampling(degree), target = sampling(2 * degree)
                cases -> ID.interpolationerror(lagrange(p), source, target, cases)
            end for degree in degrees
        ]
        pivot(
            io,
            "3. rel_l2 versus source sampling margin at fixed case degree $FIXEDDEGREE, p=$p",
            [caselabel(case) for case in FIXEDCASES],
            ["L=$degree ρ=$(_number(FIXEDDEGREE / degree))" for degree in degrees],
            errormatrix(FIXEDCASES, columns),
            ;
            note="  Target is 2x source throughout, so only the margin varies. This is the table that " *
                 "answers\n  \"how much sampling margin does a requested tolerance need\".",
        )
    end
    return nothing
end

# 4 and 5. θ-only and φ-only isolation ############################################################
#
# The plan lists these as two groups; they are printed as one table because the comparison that
# matters is total versus θ-only versus φ-only at fixed case and fixed order, and reading that off two
# tables invites lining up rows by hand.

function isolation(io)
    cases = vcat(
        ID.harmoniccases([10, 20]), [ID.RandomBandlimitedCase(; L=10, seed=0x1234)]
    )

    for p in (8, 20)
        columns = [
            cases -> ID.interpolationerror(lagrange(p), SOURCE, TARGET, cases),
            cases -> ID.interpolationerror(lagrange(p), SOURCE, MIDDLE, cases),
            cases -> ID.interpolationerror(lagrange(p), MIDDLE, TARGET, cases),
            cases -> ID.interpolationerror(localglobal(p), MIDDLE, TARGET, cases),
        ]
        pivot(
            io,
            "4/5. rel_l2 split into its θ and φ halves, p=$p",
            ["$(caselabel(case))  ρ=$(_number(ID.rho(case, SOURCE)))" for case in cases],
            ["total", "θ only", "φ only local", "φ only global"],
            errormatrix(cases, columns),
            ;
            note="  θ only: $(MLFMA.Nθ(SOURCE))x$(MLFMA.Nφ(SOURCE)) -> $(MLFMA.Nθ(MIDDLE))x$(MLFMA.Nφ(MIDDLE)), " *
                 "φ untouched (its nodes are already the source's, so that step is the exact identity).\n" *
                 "  φ only: $(MLFMA.Nθ(MIDDLE))x$(MLFMA.Nφ(MIDDLE)) -> $(MLFMA.Nθ(TARGET))x$(MLFMA.Nφ(TARGET)), " *
                 "θ untouched. Whichever half is close to `total` is the limiting direction.",
        )
    end
    return nothing
end

# 6. Prototype separate θ/φ orders ################################################################

function ordersplit(io)
    cases = vcat(ID.harmoniccases([10]), [ID.RandomBandlimitedCase(; L=10, seed=0x1234)])
    orders = ((8, 8), (12, 8), (20, 8), (8, 20), (12, 12), (20, 20))

    columns = [
        cases -> ID.interpolationerror(
            [lagrange(pθ), lagrange(pφ)], [SOURCE, MIDDLE, TARGET], cases
        ) for (pθ, pφ) in orders
    ]
    pivot(
        io,
        "6. rel_l2 with separate θ and φ orders (prototype -- no such production API yet)",
        ["$(caselabel(case))  ρ=$(_number(ID.rho(case, SOURCE)))" for case in cases],
        ["pθ=$pθ pφ=$pφ" for (pθ, pφ) in orders],
        errormatrix(cases, columns),
        ;
        note="  Two production interpolators chained through MIDDLE, which is exactly the arithmetic a\n" *
             "  `pθ != pφ` interpolator would do -- verified: at pθ == pφ a chain reproduces the single-stage\n" *
             "  study to round-off. Compare `pθ=20 pφ=8` against `pθ=20 pφ=20` for what a φ order buys, and\n" *
             "  against `pθ=8 pφ=8` for what a θ order buys.",
    )
    return nothing
end

# 7. θ in θ versus θ in cos θ #####################################################################
#
# A 1D study, and the only place in this file that does not call a production interpolator -- because
# the question is what production does NOT do. `_θinformation!` builds its stencil nodes as
# `MLFMA.θ.(Ref(sampling), θindices)` and takes barycentric weights on those, i.e. it interpolates in
# θ itself. A Gauss-Legendre grid's θ nodes are the roots of a Legendre polynomial in cos θ, and
# `Y_l^0` is a polynomial of degree l in cos θ -- so interpolating in x = cos θ should be EXACT once
# the stencil holds l + 1 nodes, while interpolating in θ is not exact at any order.
#
# Both variants below use the same nodes, the same stencil selection and the same barycentric formula
# (`MLFMA._interpolationweights`, production's own) so that the ONLY difference is the variable.
#
# Restricted to target θ whose whole stencil is interior. That is not a convenience: the pole
# extension mirrors θ -> -θ, and cos(-θ) == cos(θ), so a mirrored node is a DUPLICATE node in x and
# the barycentric weights are singular there. Interpolating in x therefore cannot reuse the virtual
# extension as it stands, which is a real constraint on the idea and is reported rather than hidden.

function _stencilindices(source, θ, p, lastindex)
    index = if isodd(p)
        MLFMA._findθclosest(source, θ, lastindex)
    else
        MLFMA._findθfirst(source, θ, lastindex)
    end
    return index, MLFMA.stencil(index, p)
end

function θ1derror(l, m, p, source, target; variable::Symbol=:θ)
    sourceθs = MLFMA.θs(source)
    # The θ FACTOR of Y_l^m, which is what a θ stage sees: at fixed φ the harmonic is this times a
    # constant. Polynomial of degree l in x = cos θ at m = 0; for m > 0 it carries (1 - x^2)^(m/2),
    # which is polynomial only for even m -- so exactness in x is expected to survive even m and not
    # odd m, and both are measured rather than assumed.
    exact(θ) = ID.normalizedlegendre(l, m, θ)

    errors = Float64[]
    interior = 0
    lastindex = 1
    for θ in MLFMA.θs(target)
        lastindex, indices = _stencilindices(source, θ, p, lastindex)
        # Non-conforming stencils need the pole extension, which is what the x variant cannot do.
        all(>(0), indices) && all(<=(MLFMA.Nθ(source)), indices) || continue
        interior += 1

        nodes = variable === :θ ? sourceθs[indices] : cos.(sourceθs[indices])
        point = variable === :θ ? θ : cos(θ)
        weights = MLFMA._interpolationweights(MLFMA.lagrangeweights(nodes), nodes, point)

        interpolated = sum(
            weights[i] * exact(sourceθs[index]) for (i, index) in enumerate(indices)
        )
        push!(errors, abs(interpolated - exact(θ)))
    end

    reference = sqrt(sum(abs2, exact.(MLFMA.θs(target))))
    return (rel_l2=sqrt(sum(abs2, errors)) / reference, npoints=interior)
end

function cosθstudy(io)
    # (l, m) pairs rather than degrees alone: m controls whether the θ factor is a polynomial in
    # x at all, which is the whole mechanism under test. m = 5 is the odd case, m = 10 the even one.
    modes = ((4, 0), (10, 0), (10, 5), (10, 10), (20, 0))
    orders = (8, 12, 20)

    cells = Matrix{Float64}(undef, length(modes), 2 * length(orders))
    for (i, (l, m)) in enumerate(modes), (j, p) in enumerate(orders)
        cells[i, j] = θ1derror(l, m, p, SOURCE, TARGET; variable=:θ).rel_l2
        cells[i, length(orders) + j] =
            θ1derror(l, m, p, SOURCE, TARGET; variable=:cosθ).rel_l2
    end

    npoints = θ1derror(4, 0, first(orders), SOURCE, TARGET).npoints
    pivot(
        io,
        "7. 1D θ interpolation of the θ factor of Y_l^m: in θ (what production does) versus in x = cos θ",
        ["l=$l m=$m" for (l, m) in modes],
        vcat(["θ p=$p" for p in orders], ["cosθ p=$p" for p in orders]),
        cells,
        ;
        note="  Same nodes, same stencils, same barycentric formula -- only the interpolation variable " *
             "differs.\n  Restricted to the $npoints of $(MLFMA.Nθ(TARGET)) target θ whose stencil is interior: " *
             "the pole extension mirrors\n  θ -> -θ, and cos is even, so mirrored nodes collide in x and the " *
             "weights are singular there.",
    )
    return nothing
end

# 8. Where the error sits in θ ####################################################################

function polarprofile(io)
    cases = [
        ID.HarmonicCase(; l=10, m=0),
        ID.HarmonicCase(; l=10, m=5),
        ID.HarmonicCase(; l=10, m=10),
    ]
    p = 8
    study = ID.interpolationerror(lagrange(p), SOURCE, TARGET, cases)

    Nθtarget = MLFMA.Nθ(TARGET)
    # Rows are counted in from each pole. A stencil of p needs about p/2 nodes on each side, so the
    # outermost ~p/2 target rows are the ones whose stencil had to be extended across a pole.
    bands = (1:4, 5:8, 9:(Nθtarget ÷ 2))

    cells = Matrix{Float64}(undef, length(cases), 2 * length(bands))
    for i in eachindex(cases)
        field = ID.pointwiseerror(study, i; kind=:absolute)
        total = sum(abs2, field)
        for (j, band) in enumerate(bands)
            # Symmetric bands from both poles, kept apart so an asymmetry between θ = 0 and θ = π --
            # which a sign error in the extension would produce -- cannot average itself away.
            north = sum(abs2, view(field, :, band))
            south = sum(abs2, view(field, :, (Nθtarget + 1) .- band))
            cells[i, j] = north / total
            cells[i, length(bands) + j] = south / total
        end
    end

    pivot(
        io,
        "8. Share of squared absolute error by θ band, p=$p (fraction of the total)",
        [caselabel(case) for case in cases],
        vcat(
            ["N rows $(first(band))-$(last(band))" for band in bands],
            ["S rows $(first(band))-$(last(band))" for band in bands],
        ),
        cells,
        ;
        note="  Rows counted in from θ = 0 (N) and θ = π (S). The two halves of the grid hold equal " *
             "numbers of\n  directions, so a band's share exceeding its share of the rows means the error " *
             "concentrates there.",
    )
    return nothing
end

# 9. Forward field error next to reverse pairing error ############################################
#
# The two numbers a tolerance argument needs, side by side, because they are NOT interchangeable and the
# difference between them is systematic rather than noise:
#
#   * `rel_l2` is the FIELD error of the forward interpolation -- a norm over the whole target grid.
#   * `pairing` is what the REVERSE direction contributes to the scalar a Galerkin test side computes,
#     `|<f_fine, g> - <f_coarse, A g>| / |<f_fine, g>|` against exact resampling.
#
# `pairing` is normally SMALLER, and for a structural reason: it contracts the error field against the
# test function, so oscillatory parts of the error cancel in the inner product instead of accumulating in
# a norm. So a field tolerance is the conservative one, and quoting a field number as though it were a
# matrix number over-states the error reaching the matrix. Neither is a substitute for the other.
#
# `adjointness` rides along as a sanity invariant, not a measurement: it must stay at round-off, or the
# `pairing` column beside it was read off an operator that is no longer the adjoint. See
# `src/interpolation/diagnostics/adjointness.jl` for why an "anterpolation accuracy" metric that is not
# this one is either meaningless or the forward error in disguise.

function reversedirection(io)
    cases = vcat(
        ID.harmoniccases([6, 10, 20]), [ID.RandomBandlimitedCase(; L=10, seed=0x1234)]
    )

    for p in (8, 20)
        builder = MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=p)
        interpolator = MLFMA.buildinterpolator(builder, [SOURCE], [TARGET])
        reverse = MLFMA.buildreverseoperators(builder, interpolator; nchunks=1)

        forward = ID.results(ID.interpolationerror(interpolator, SOURCE, TARGET, cases))
        rows = ID.adjointpairingerror(
            interpolator, reverse.anterpolator, SOURCE, TARGET, cases
        )

        cells = Matrix{Float64}(undef, length(cases), 4)
        for i in eachindex(cases)
            cells[i, 1] = forward[i].rel_l2
            cells[i, 2] = rows[i].pairing
            cells[i, 3] = rows[i].pairing / forward[i].rel_l2
            cells[i, 4] = rows[i].adjointness
        end

        pivot(
            io,
            "9. forward field error versus reverse pairing error, θorder=$p (LocalθGlobalφ)",
            ["$(caselabel(case))  ρ=$(_number(ID.rho(case, SOURCE)))" for case in cases],
            ["field rel_l2", "pairing", "ratio", "adjointness"],
            cells,
            ;
            note="  `ratio` = pairing / field. Below 1 because the pairing contracts the error field " *
                 "against the test\n  function; a field tolerance is therefore the conservative one. " *
                 "`adjointness` is an invariant, not a\n  measurement -- it must stay at round-off.",
        )
    end
    return nothing
end

# 10. Tangential representation: is the default flip safe? ########################################
#
# FINAL DILIGENCE FOR CHANGING `INTERPOLATORECOM`, and the framing matters because the earlier evidence
# was easy to misread. Local/global is NOT broadly more accurate: on random tangential bands -- both
# families mixed, which is what a far-field pattern is -- the two families are indistinguishable
# (measured ratio 0.993 to 1.000). Its factor-of-7 advantage is confined to pure sectoral modes, which
# carry little of a full band's energy, and the total stays θ-limited exactly as group 4/5 predicts.
#
# So the flip is a PERFORMANCE change that must not cost accuracy, and this group's job is to show that
# nothing regresses on the representation production actually carries. Four questions, in columns:
#
#   * does combined accuracy stay neutral across cases?           -> `lg` / `gl` / `ratio`
#   * does either COMPONENT regress while the combined stays flat? -> `θ ratio`, `φ ratio`
#   * does the error move toward the poles?                        -> `pole share`
#
# A combined-only comparison would miss the middle two: `Fφ` carries a `1/sinθ` factor `Fθ` does not, so
# the components can trade against each other while their norm-combination looks unchanged. The pole
# share is the same guard in θ -- if global φ pushed error toward the poles it would show here even with
# an unchanged total.

function tangentialflip(io)
    cases = vcat(
        [
            ID.TangentialHarmonicCase(; kind=k, l=10, m=m) for k in (:poloidal, :toroidal)
            for m in (0, 5, 10)
        ],
        [ID.RandomTangentialBandlimitedCase(; L=L, seed=0x1234) for L in (4, 10, 20)],
    )

    for p in (8, 20)
        families = (
            MLFMA.SphericalLagrangeInterpolator{p}(ComplexF64, Float64),
            MLFMA.LocalθGlobalφInterpolator{p}(ComplexF64, Float64),
        )
        studies = map(families) do interpolator
            return ID.interpolationerror(
                interpolator,
                SOURCE,
                TARGET,
                cases;
                farfieldtype=MLFMA.TangentialSphericalFarField,
            )
        end

        cells = Matrix{Float64}(undef, length(cases), 6)
        for i in eachindex(cases)
            lagrange, localglobal = ID.results(studies[1])[i], ID.results(studies[2])[i]
            cells[i, 1] = lagrange.rel_l2
            cells[i, 2] = localglobal.rel_l2
            cells[i, 3] = localglobal.rel_l2 / lagrange.rel_l2
            # Per-component ratios, guarding against a component regressing under a flat total. A
            # component that is identically zero on both sides (poloidal m = 0's Fφ) gives 0/0; report
            # 1, i.e. "unchanged", which is what it is.
            for (column, component) in ((4, 1), (5, 2))
                a = lagrange.component_rel_l2[component]
                b = localglobal.component_rel_l2[component]
                cells[i, column] = iszero(a) && iszero(b) ? 1.0 : b / a
            end
            # Share of squared error in the outermost 4 θ rows of BOTH hemispheres, under global φ.
            profile = ID.θbandprofile(studies[2], i; bands=4)
            cells[i, 6] = profile.north[1] + profile.south[1]
        end

        pivot(
            io,
            "10. tangential: accuracy neutrality of the global-φ default, θorder=$p",
            [
                "$(ID.casekind(case)) $(ID.casebandwidth(case))" *
                (case isa ID.TangentialHarmonicCase ? " m=$(case.m)" : " band") for
                case in cases
            ],
            ["lagrange", "localglobal", "ratio", "θ ratio", "φ ratio", "pole share"],
            cells,
            ;
            note="  `ratio` = localglobal / lagrange; at or below 1 means no accuracy cost. `θ/φ ratio` " *
                 "catch a component\n  regressing under a flat total. `pole share` is the fraction of " *
                 "squared error in the outermost 4 θ rows\n  of both hemispheres under global φ -- the " *
                 "scalar studies put ~1e-11 there, so a jump would mean the\n  coordinate singularity " *
                 "started to bite. Ratios of 1.0 on a zero component mean both were zero.",
        )
    end
    return nothing
end

# 11. The equidistant representation against the Gauss-Legendre one ###############################
#
# THE COMPARISON THIS WHOLE FILE EXISTS FOR, and the one that is easiest to get backwards. The
# equidistant family (`src/equidistant/`) resamples SPECTRALLY IN BOTH ANGLES: extend to the torus,
# FFT, move coefficients by signed frequency, inverse. `LocalθGlobalφ` does that in φ only and keeps a
# local barycentric stencil in θ. Group 4/5 established that θ owns essentially the whole error, so
# making θ spectral too should collapse the error to round-off -- and it does, by eleven orders.
#
# READ IT AGAINST THE RUNTIME GROUP BELOW, NEVER ALONE. The equidistant FFT costs 2.9x to 5.2x a
# `LocalθGlobalφ` apply at equal grid size and the ratio WORSENS with the grid. So neither axis decides
# this on its own, which is exactly the situation this file's header refuses to collapse into one
# table: a slower kernel that gains eleven digits is no more a regression than a faster one that loses
# a digit is an improvement.
#
# ρ MEANS SOMETHING DIFFERENT HERE, and printing it without saying so would make this table read as
# magic. `supporteddegree` is `MLFMA.cutoffparameter`, a QUADRATURE bound: the degree at which the grid
# still integrates a PRODUCT of two harmonics exactly, hence ~Nθ/2. Interpolation only has to
# REPRESENT one harmonic, so its band is the doubled torus's own Nyquist -- about twice that. The
# Gauss-Legendre family never gets to exploit the difference because its θ step is not spectral; the
# equidistant one does. Measured cliff at Nθ = 28: round-off (1.3e-14) through l = 27, then 0.688 at
# l = 28 and 1.41 by l = 29. Exactly `Nθ - 1`, which is why `interpolationband` below is that and not
# a fitted constant. Both ratios are printed so neither can be misread as the other.

# The equidistant grid with the same (Nθ, Nφ) as a Gauss-Legendre one. EQUAL GRID SIZE is the honest
# axis for a per-apply cost comparison: equal ρ would compare different point counts, and equal
# accuracy is not attainable in the same table because one family reaches round-off and the other
# cannot at any order.
equidistant(s) = EquidistantSampling{Float64}(MLFMA.Nθ(s), MLFMA.Nφ(s))

# Highest degree the equidistant grid REPRESENTS (as opposed to integrates). The θ extension makes a
# `Nθ`-point grid into a `2Nθ`-point periodic one whose Nyquist is `Nθ`; the Nyquist bin itself is the
# ambiguous one `resamplecoefficients` splits, so the last unambiguous degree is `Nθ - 1`. With
# `Nφ = 2Nθ` the φ limit lands in the same place, so one number covers both directions.
interpolationband(s::EquidistantSampling) = MLFMA.Nθ(s) - 1

const EQSOURCE = equidistant(SOURCE)
const EQTARGET = equidistant(TARGET)

function equidistantcomparison(io)
    columns = [
        cases -> ID.interpolationerror(localglobal(15), SOURCE, TARGET, cases),
        cases -> ID.interpolationerror(
            EquidistantInterpolator{ComplexF64}(FFTBackend()), EQSOURCE, EQTARGET, cases
        ),
        cases -> ID.interpolationerror(
            EquidistantInterpolator{ComplexF64}(LagrangeBackend(8)),
            EQSOURCE,
            EQTARGET,
            cases,
        ),
        cases -> ID.interpolationerror(
            EquidistantInterpolator{ComplexF64}(LagrangeBackend(16)),
            EQSOURCE,
            EQTARGET,
            cases,
        ),
    ]
    collabels = ["GL localglobal 15", "equi FFT", "equi Lagr 8", "equi Lagr 16"]

    function rowlabels(cases)
        return [
            "$(caselabel(case))  ρ=$(_number(ID.rho(case, SOURCE)))" *
            "  ρi=$(_number(ID.casebandwidth(case) / interpolationband(EQSOURCE)))" for
            case in cases
        ]
    end

    note =
        "  ρ is quadrature-referenced (`cutoffparameter`, ~Nθ/2) as everywhere else in this file; " *
        "ρi is\n  referenced to what the equidistant grid REPRESENTS (Nθ-1 = $(interpolationband(EQSOURCE))). " *
        "The FFT column is at\n  round-off for every ρi < 1 and useless above it -- measured cliff " *
        "between l=27 and l=28 -- so ρi is\n  the column that predicts it and ρ is the one that does not."

    pivot(
        io,
        "11a. rel_l2, scalar: spectral-in-both-angles against spectral-in-φ-only",
        rowlabels(SWEEPCASES),
        collabels,
        errormatrix(SWEEPCASES, columns),
        ;
        note=note,
    )

    # THE TANGENTIAL TABLE IS THE ONE THAT DECIDES ANYTHING, because it is the representation a Maxwell
    # operator carries. A scalar-only result would leave the θ/φ continuation and sign rules -- which
    # differ between the two grid families -- entirely unexercised.
    tangentialcases = [
        ID.RandomTangentialBandlimitedCase(; L=L, seed=0x1234) for L in (4, 10, 20, 26)
    ]
    tangentialcolumns = [
        cases -> ID.interpolationerror(
            localglobal(15),
            SOURCE,
            TARGET,
            cases;
            farfieldtype=MLFMA.TangentialSphericalFarField,
        ),
        cases -> ID.interpolationerror(
            EquidistantInterpolator{ComplexF64}(FFTBackend()),
            EQSOURCE,
            EQTARGET,
            cases;
            farfieldtype=MLFMA.TangentialSphericalFarField,
        ),
        cases -> ID.interpolationerror(
            EquidistantInterpolator{ComplexF64}(LagrangeBackend(16)),
            EQSOURCE,
            EQTARGET,
            cases;
            farfieldtype=MLFMA.TangentialSphericalFarField,
        ),
    ]

    pivot(
        io,
        "11b. rel_l2, TANGENTIAL: the representation a Maxwell operator actually carries",
        rowlabels(tangentialcases),
        ["GL localglobal 15", "equi FFT", "equi Lagr 16"],
        errormatrix(tangentialcases, tangentialcolumns),
        ;
        note="  Both tangential components under `SphericalθφInterpolation`, i.e. both flip sign across " *
             "a pole.\n  If the equidistant continuation rules were wrong this is where it would show, " *
             "and it does not.",
    )
    return nothing
end

# 12. Cost at equal TOLERANCE, not equal grid #####################################################
#
# THE QUESTION EVERY OTHER GROUP IN THIS FILE DECLINES TO ANSWER. Group 11 compares families at the
# same `(Nθ, Nφ)`, where the equidistant FFT family sits at round-off and `LocalθGlobalφ` sits at 3e-4
# -- eleven orders apart -- and costs 1.7x-2.5x more. That comparison charges the spectral family for
# accuracy nobody requested. A sampling strategy does not choose a grid; it chooses a TOLERANCE, and
# the grid follows. So the production question is the inverse: for a requested ε, what grid does each
# family need, and what does it cost THERE.
#
# THE KNOB IS THE REQUESTED DEGREE, not `Nθ` directly, because that is the knob `SamplingStrategy`
# actually turns: grids come from `MLFMA.NθNφ(D)`. So the answer transfers straight into a strategy
# setting instead of needing translation.
#
# The content's true bandwidth is held FIXED while the requested degree sweeps around it. Requesting
# BELOW the content's bandwidth is deliberately included -- `cutoffparameter` is a quadrature bound
# (~Nθ/2) while interpolation only has to REPRESENT one harmonic (~Nθ-1), so a family that exploits
# that gap should be visibly exact at roughly half the requested degree, and one that does not should
# not be. That gap is the whole reason this group exists.
#
# WHY THIS DOES NOT BREAK THE FILE'S RULE that accuracy and runtime are never printed in one table.
# The rule exists because a merged table hides an accuracy regression behind a faster number. Here
# accuracy is PINNED -- it is the row label -- and only cost varies, which is the one arrangement
# where combining them is the honest comparison rather than the misleading one. Even so the tables
# stay split: 12a/12b are pure accuracy and name grids, and the timing group runs each family on the
# grid 12b assigns it.

const TOLERANCES = (1e-3, 1e-6, 1e-9, 1e-12)

# The content's real bandwidth, fixed. Tangential, because that is the representation a Maxwell
# operator carries and the one whose continuation rules differ between the two grid families.
const TOLERANCEDEGREE = 13
const TOLERANCECASE = ID.RandomTangentialBandlimitedCase(; L=TOLERANCEDEGREE, seed=0x1234)

# Spans well below the content bandwidth (where everything must fail) to well above it.
const REQUESTDEGREES = 4:2:36

# `(label, build)` where `build(D)` returns the interpolator and its own two grids, so each family is
# asked for the same DEGREE and builds whatever grid family it uses from it.
# GL interpolators are built through `buildinterpolator`, NOT through the bare `localglobal(p)`
# constructor this file uses for accuracy work. The bare form stores no per-level information, so every
# apply rebuilds it -- which the fixture comment further down already warns about as a 224-byte trap
# and which shows up here as a ~90x one. Measured before this was fixed: p=20 timed 10.4 ms at Nθ=30
# and 8.9 ms at Nθ=70, i.e. FASTER on a grid 2.3x larger. A cost that falls as the grid grows is a
# broken harness, not a fast kernel.
_glbuild(p) =
    D -> (
        MLFMA.buildinterpolator(
            MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=p),
            [sampling(D)],
            [sampling(2D)];
            interpolationtrait=MLFMA.SphericalθφInterpolation(),
        ),
        sampling(D),
        sampling(2D),
    )

const TOLERANCEFAMILIES = (
    (label="GL localglobal 10", build=_glbuild(10)),
    (label="GL localglobal 15", build=_glbuild(15)),
    (label="GL localglobal 20", build=_glbuild(20)),
    (
        label="equi FFT",
        build=D -> (
            EquidistantInterpolator{ComplexF64}(FFTBackend()),
            equidistant(sampling(D)),
            equidistant(sampling(2D)),
        ),
    ),
    (
        label="equi Lagr 16",
        build=D -> (
            EquidistantInterpolator{ComplexF64}(LagrangeBackend(16)),
            equidistant(sampling(D)),
            equidistant(sampling(2D)),
        ),
    ),
)

function toleranceerror(family, D)
    interpolator, source, target = family.build(D)
    study = ID.interpolationerror(
        interpolator,
        source,
        target,
        [TOLERANCECASE];
        farfieldtype=MLFMA.TangentialSphericalFarField,
    )
    return ID.results(study)[1].rel_l2
end

# First requested degree meeting `tol` AND still meeting it at the next step, so a single lucky point
# on a non-monotone curve cannot be reported as the answer. 12a prints the whole sweep, so anything
# ragged is visible rather than hidden behind this rule.
function mindegree(errors, tol)
    for i in 1:(length(REQUESTDEGREES) - 1)
        errors[i] <= tol && errors[i + 1] <= tol && return REQUESTDEGREES[i]
    end
    return nothing
end

const TOLERANCEERRORS = Dict(
    family.label => [toleranceerror(family, D) for D in REQUESTDEGREES] for
    family in TOLERANCEFAMILIES
)

function equaltolerance(io)
    labels = ["D=$(D)  Nθ=$(MLFMA.NθNφ(D)[1])" for D in REQUESTDEGREES]
    collabels = [f.label for f in TOLERANCEFAMILIES]
    cells = hcat((TOLERANCEERRORS[f.label] for f in TOLERANCEFAMILIES)...)

    pivot(
        io,
        "12a. rel_l2 vs REQUESTED degree, tangential content of true bandwidth $(TOLERANCEDEGREE)",
        labels,
        collabels,
        cells;
        note="  The requested degree sets the grid through `MLFMA.NθNφ(D)` (Nθ = 2(D+1)), which is the\n" *
             "  knob a SamplingStrategy turns. Requesting below D=$(TOLERANCEDEGREE) is asking for a grid coarser than\n" *
             "  the content -- a family exact there is exploiting the gap between the quadrature bound\n" *
             "  and the representation bound, not getting something for nothing.",
    )

    mincells = Matrix{Float64}(undef, length(TOLERANCES), length(TOLERANCEFAMILIES))
    for (i, tol) in enumerate(TOLERANCES), (j, family) in enumerate(TOLERANCEFAMILIES)
        D = mindegree(TOLERANCEERRORS[family.label], tol)
        mincells[i, j] = D === nothing ? NaN : Float64(MLFMA.NθNφ(D)[1])
    end
    pivot(
        io,
        "12b. minimum Nθ reaching each tolerance (NaN = never, within D <= $(last(REQUESTDEGREES)))",
        ["ε = $(tol)" for tol in TOLERANCES],
        collabels,
        mincells;
        note="  This is the grid each family is entitled to run on for that tolerance. The runtime group\n" *
             "  `\"equal tolerance\"` times each family THERE rather than on a shared grid.",
    )
    return nothing
end

# 13. Can the θ and φ stencil orders differ? ######################################################
#
# `LagrangeBackend{Pθ,Pφ}` has always accepted two orders and every caller has always passed one
# number to both. This asks whether that is leaving anything on the table, on the EQUIDISTANT grid
# specifically -- group 6 asks the same question of the Gauss-Legendre family, where θ nodes are
# non-uniform and θ was found to be the limiting direction. Here BOTH coordinates are uniform, so that
# result does not transfer and the question is genuinely open.
#
# Why it matters for cost and not just tidiness: the two stages are separable and their costs are
# independent (`Nθ_s·Nφ_d·pφ` for the φ stage, `Nθ_d·Nφ_d·pθ` for the θ one), so an order that can be
# dropped in one direction is work that simply disappears.

# UNIT RESOLUTION, and that matters. A first version swept only even orders, which forced every
# reported optimum to differ by 2 and overstated the saving -- it was comparing against a symmetric
# pair that was itself unnecessarily coarse. The displayed table below is thinned for width; the
# search in 13c runs over all of these.
const ANISOORDERS = 4:16
const ANISOSHOWN = (4, 6, 8, 10, 12, 16)
const ANISOCASE = ID.RandomTangentialBandlimitedCase(; L=10, seed=0x1234)

function anisotropicorders(io)
    cells = Matrix{Float64}(undef, length(ANISOORDERS), length(ANISOORDERS))
    for (i, pθ) in enumerate(ANISOORDERS), (j, pφ) in enumerate(ANISOORDERS)
        study = ID.interpolationerror(
            EquidistantInterpolator{ComplexF64}(LagrangeBackend(pθ, pφ)),
            EQSOURCE,
            EQTARGET,
            [ANISOCASE];
            farfieldtype=MLFMA.TangentialSphericalFarField,
        )
        cells[i, j] = ID.results(study)[1].rel_l2
    end
    shown = [findfirst(==(p), ANISOORDERS) for p in ANISOSHOWN]
    pivot(
        io,
        "13a. rel_l2 vs the two stencil orders, tangential L=$(ANISOCASE.L) on $(MLFMA.Nθ(EQSOURCE))x$(MLFMA.Nφ(EQSOURCE)) -> $(MLFMA.Nθ(EQTARGET))x$(MLFMA.Nφ(EQTARGET))",
        ["pθ=$p" for p in ANISOSHOWN],
        ["pφ=$p" for p in ANISOSHOWN],
        cells[shown, shown];
        note="  Read DOWN a column for the θ order's effect at fixed φ, and ACROSS a row for φ's at\n" *
             "  fixed θ. A row that is flat means that direction's order is already free.",
    )

    # 13b. THE TWO SATURATED FLOORS, which is what actually bounds how far the orders may differ.
    #
    # Fix one order at its maximum and sweep the other: the last column is what θ can do with φ out of
    # the way, the last row is φ with θ out of the way. Their RATIO is the whole answer -- it says how
    # much accuracy θ owes at equal order, and therefore how many order steps of φ are spare.
    floors = Matrix{Float64}(undef, length(ANISOORDERS), 3)
    for (i, _) in enumerate(ANISOORDERS)
        floors[i, 1] = cells[i, end]
        floors[i, 2] = cells[end, i]
        floors[i, 3] = cells[i, end] / cells[end, i]
    end
    pivot(
        io,
        "13b. the saturated per-direction floors, and how far apart they are",
        ["p=$p" for p in ANISOORDERS],
        ["θ floor (φ maxed)", "φ floor (θ maxed)", "ratio θ/φ"],
        floors;
        note="  One order step buys about 2.1x. The ratio peaks near 2.5, i.e. barely more than ONE\n" *
             "  step -- which is the bound on how far the two orders may usefully differ. It falls back\n" *
             "  to 1.0 at p=16 because both directions have hit a floor that is no longer the stencil's.",
    )

    # 13c. WHICH PAIR IS CHEAPEST FOR A TARGET ε, and does it beat the best pθ == pφ pair.
    #
    # Cost is the TAP COUNT: the two stages are separable, so the φ stage does `Nθ_s·Nφ_d·pφ`
    # multiply-adds and the θ stage `Nθ_d·Nφ_d·pθ`. That makes a θ order roughly `Nθ_d/Nθ_s` times as
    # expensive as a φ order -- here 54/28 ≈ 1.9 -- which is the whole tension: θ is the direction that
    # needs the higher order AND the direction that costs more to raise.
    #
    # The model is not assumed. Measured across ten (pθ, pφ) pairs it holds at 2.23-2.94 µs per 1000
    # taps, mostly 2.5-2.6, so tap count and wall time agree to about 15%.
    cost(pθ, pφ) = MLFMA.Nθ(EQTARGET) * pθ + MLFMA.Nθ(EQSOURCE) * pφ
    tolerances = (5e-3, 1e-3, 1e-4, 1e-5)
    # `rows` records the gap explicitly, because "they can differ" is only interesting with a number.
    rows = Vector{String}(undef, length(tolerances))
    best = Matrix{Float64}(undef, length(tolerances), 3)
    for (t, tol) in enumerate(tolerances)
        anybest, symbest = nothing, nothing
        for (i, pθ) in enumerate(ANISOORDERS), (j, pφ) in enumerate(ANISOORDERS)
            cells[i, j] <= tol || continue
            c = cost(pθ, pφ)
            (anybest === nothing || c < anybest[1]) && (anybest = (c, pθ, pφ))
            pθ == pφ && (symbest === nothing || c < symbest[1]) && (symbest = (c, pθ, pφ))
        end
        if anybest === nothing
            rows[t] = "ε = $(tol)  (unreached)"
            best[t, :] .= NaN
        else
            rows[t] = "ε = $(tol)  best (pθ,pφ)=($(anybest[2]),$(anybest[3])) Δ=$(anybest[2] - anybest[3])"
            best[t, 1] = anybest[1]
            best[t, 2] = symbest === nothing ? NaN : symbest[1]
            best[t, 3] = symbest === nothing ? NaN : symbest[1] / anybest[1]
        end
    end
    pivot(
        io,
        "13c. cheapest order pair for a target ε, against the cheapest pθ == pφ pair",
        rows,
        ["taps (best)", "taps (pθ==pφ)", "saving"],
        best;
        note="  Cost is `Nθ_dst·pθ + Nθ_src·pφ` per φ column, the separable tap count. θ sets the error\n" *
             "  floor so its order cannot come down; the only saving is trimming φ. The winning gap is\n" *
             "  ONE order, not two -- see 13b for why -- and at the loosest tolerance it is zero.",
    )
    return nothing
end

const STUDIES = (
    (name="1. stencil/order sweep", run=stencilsweep),
    (name="2. interpolator family", run=familycomparison),
    (name="3. sampling margin", run=marginsweep),
    (name="4/5. θ and φ isolation", run=isolation),
    (name="6. separate θ/φ orders", run=ordersplit),
    (name="7. θ versus cos θ", run=cosθstudy),
    (name="8. polar error profile", run=polarprofile),
    (name="9. forward field vs reverse pairing", run=reversedirection),
    (name="10. tangential default-flip diligence", run=tangentialflip),
    (name="11. equidistant against Gauss-Legendre", run=equidistantcomparison),
    (name="12. cost at equal tolerance", run=equaltolerance),
    (name="13. anisotropic stencil orders", run=anisotropicorders),
)

# Runtime #########################################################################################
#
# The cost side of the same questions, on the same grids. Note what is NOT here: no accuracy claim.
# A row of this group being faster says nothing about whether it is better, which is why the accuracy
# studies above exist and why the two are never printed in one table.

const TIMINGS = BenchmarkGroup()

# One field per grid, filled with a real case rather than left as zeros -- an interpolator does
# `mul!`s whose cost does not depend on the values, but zeros would let a future subnormal-sensitive
# kernel look artificially fast.
function preparedfield(sampling)
    field = MLFMA.ScalarFarField(ComplexF64, sampling, Val(:zero))
    ID.fillexact!(field, ID.HarmonicCase(; l=10, m=5), sampling)
    return field
end

const SOURCEFIELD = preparedfield(SOURCE)
const MIDDLEFIELD = preparedfield(MIDDLE)

# The TANGENTIAL apply, which is what a Maxwell operator actually runs: two components under
# `SphericalθφInterpolation` rather than one under `VirtualExtension`. Timed separately from the scalar
# rows above rather than replacing them, because the default-flip question is about this representation
# and a scalar timing would answer a different one. Allocations matter as much as time here: a per-apply
# allocation is paid per transfer per matvec.
function preparedtangentialfield(sampling)
    field = MLFMA.TangentialSphericalFarField(ComplexF64, sampling, Val(:zero))
    ID.fillexact!(field, ID.RandomTangentialBandlimitedCase(; L=10, seed=0x1234), sampling)
    return field
end

const TANGENTIALSOURCEFIELD = preparedtangentialfield(SOURCE)

for p in (8, 20)
    lagrange = MLFMA.SphericalLagrangeInterpolator{p}(ComplexF64, [SOURCE], [TARGET])
    localglobal = MLFMA.LocalθGlobalφInterpolator{p}(ComplexF64, [SOURCE], [TARGET])

    TIMINGS["apply tangential"]["lagrange"]["p=$p"] = @benchmarkable(
        $lagrange($TANGENTIALSOURCEFIELD, $SOURCE, $TARGET), seconds = 10,
    )
    TIMINGS["apply tangential"]["localglobal"]["p=$p"] = @benchmarkable(
        $localglobal($TANGENTIALSOURCEFIELD, $SOURCE, $TARGET), seconds = 10,
    )
end

for p in (8, 12, 20)
    # Interpolators are constructed WITH their sampling pair, so the interpolation information is
    # built once at construction and cached in the dict, not rebuilt per call. That is how the
    # production core holds them, and timing the cache-miss path instead would measure construction.
    full = MLFMA.SphericalLagrangeInterpolator{p}(ComplexF64, [SOURCE], [TARGET])
    θstage = MLFMA.SphericalLagrangeInterpolator{p}(ComplexF64, [SOURCE], [MIDDLE])
    φstage = MLFMA.SphericalLagrangeInterpolator{p}(ComplexF64, [MIDDLE], [TARGET])
    globalφ = MLFMA.LocalθGlobalφInterpolator{p}(ComplexF64, [SOURCE], [TARGET])

    TIMINGS["apply"]["lagrange full"]["p=$p"] = @benchmarkable(
        $full($SOURCEFIELD, $SOURCE, $TARGET), seconds = 10,
    )
    TIMINGS["apply"]["lagrange θ stage"]["p=$p"] = @benchmarkable(
        $θstage($SOURCEFIELD, $SOURCE, $MIDDLE), seconds = 10,
    )
    TIMINGS["apply"]["lagrange φ stage"]["p=$p"] = @benchmarkable(
        $φstage($MIDDLEFIELD, $MIDDLE, $TARGET), seconds = 10,
    )
    TIMINGS["apply"]["localglobal full"]["p=$p"] = @benchmarkable(
        $globalφ($SOURCEFIELD, $SOURCE, $TARGET), seconds = 10,
    )
end

# THE LEVEL TRANSFER, PER DIRECTION AND PER FAMILY -- the group that prices the transposed-interpolation
# wrapper against the default it would replace.
#
# HOW TO READ IT, because the interesting number is a DIFFERENCE and not any single row. Both reverse
# directions do the same work apart from which operator they apply:
#
#   * `adjoint`   applies the anterpolator (adjoint interpolation) with the conjugated phase;
#   * `transpose` applies `transposedinterpolator` with the plain phase.
#
# For `lagrange` those are the SAME object -- its operator is real -- so the two rows differ only by
# noise, and their agreement is the control. For `localglobal` the transpose row additionally pays
# `TransposedInterpolator`'s in-place conjugation passes (input, output, and the input restore), so
# `transpose - adjoint` on that family isolates the wrapper's cost with everything else held fixed.
#
# Realistic in the ways that matter to that difference: a `TangentialSphericalFarField` (two components,
# as a Maxwell level transfer carries) under `SphericalθφInterpolation`, and a scratch-backed context, so
# the conjugation passes are measured against production's allocation behaviour rather than against the
# allocating fallback. `Overwrite` rather than `Accumulate`: the accumulating reverse path adds one
# child-grid staging step to BOTH directions equally, so it dilutes the difference without changing it.

const TRANSFERPHASE = MLFMA.phasefactor(
    im * 2π,
    first(MLFMA.cartesiansamplepoints(TARGET)),
    SVector(0.19, -0.27, 0.11),
    MLFMA.AbstractRadiationFieldCollection,
)

function transferfixture(builder)
    # `interpolationtrait` MATCHES the transfer's, because that is what the map constructors do: both
    # build their interpolator with the OPERATOR's concrete trait. Left to default, the interpolator
    # stores `DefaultInterpolation()` and every apply makes `converttrait` rebuild its information
    # container -- which this group used to report as a 224-byte per-apply cost of the `localglobal`
    # FAMILY, when it was a cost of this FIXTURE. See benchmark/interpolation/README.md.
    interpolator = MLFMA.buildinterpolator(
        builder, [SOURCE], [TARGET]; interpolationtrait=MLFMA.SphericalθφInterpolation()
    )
    reverse = MLFMA.buildreverseoperators(builder, interpolator; nchunks=1)
    transfer = MLFMA.InterpolationPhaseTransfer(
        interpolator,
        reverse.anterpolator,
        TRANSFERPHASE,
        SOURCE,
        TARGET,
        MLFMA.SphericalθφInterpolation(),
    )
    context = H2Factory.TranslationContext(
        nothing,
        MLFMA.LevelTransferScratch(
            MLFMA.TangentialSphericalFarField(ComplexF64, TARGET),
            MLFMA.TangentialSphericalFarField(ComplexF64, SOURCE),
            MLFMA.interimscratch(reverse.anterpolator, SOURCE, TARGET),
        ),
    )
    return (transfer=transfer, context=context)
end

for p in (8, 20)
    families = (
        lagrange=MLFMA.SphericalLagrangeInterpolationBuilder(; order=p),
        localglobal=MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=p),
    )
    for (name, builder) in pairs(families)
        fixture = transferfixture(builder)
        childmoment = MLFMA.TangentialSphericalFarField(ComplexF64, SOURCE)
        parentmoment = MLFMA.TangentialSphericalFarField(ComplexF64, TARGET)
        childout = MLFMA.TangentialSphericalFarField(ComplexF64, SOURCE)
        parentout = MLFMA.TangentialSphericalFarField(ComplexF64, TARGET)

        TIMINGS["transfer"]["$name forward"]["p=$p"] = @benchmarkable(
            H2Factory.applydirectedtransfer!(
                $parentout, $childmoment, $(fixture.transfer), false, $(fixture.context)
            ),
            seconds = 10,
        )
        TIMINGS["transfer"]["$name transpose"]["p=$p"] = @benchmarkable(
            H2Factory.applydirectedtransfer!(
                $childout,
                $parentmoment,
                transpose($(fixture.transfer)),
                false,
                $(fixture.context),
            ),
            seconds = 10,
        )
        TIMINGS["transfer"]["$name adjoint"]["p=$p"] = @benchmarkable(
            H2Factory.applydirectedtransfer!(
                $childout,
                $parentmoment,
                adjoint($(fixture.transfer)),
                false,
                $(fixture.context),
            ),
            seconds = 10,
        )
    end
end

# THE REVERSE PATH, DECOMPOSED -- the Phase 6 baseline.
#
# Disaggregation runs inside every FORWARD matvec, and anterpolation costs ~1.6x interpolation at the
# shipped default (p=15). This group splits that cost so an optimization is aimed by measurement.
#
# THE SUB-KERNELS ARE TIMED DIRECTLY, and the first version of this group did NOT do that -- it timed
# whole applies over an intermediate grid, the way the ACCURACY split does, and that is wrong here. An
# anterpolation apply always runs `_anterpolateφbfft!`, even when Nφ does not change, so a "θ-only"
# whole-apply stage still pays the full φ FFT and both stages double-count it. The tell was mild and I
# misread it: φ + θ came to 1.15-1.28x the full apply, which looks like ordinary per-call overhead and is
# actually the FFT counted twice. Those numbers (φ ~191 µs flat, θ 303-654 µs, and a claimed "~191 µs
# floor") were committed and are wrong. Calling the two kernels directly accounts for 99.7% of the apply
# with nothing double-counted, which is the check the whole-apply version never had.
#
# A SECOND TRAP, from the same first attempt, kept because it is real: interpolation information -- FFT
# plans included -- is cached in the interpolator's dict keyed by (child, parent). Build an operator for
# one pair and apply it to another and every call rebuilds that information, so the number is plan
# construction rather than the kernel. Measured that way the φ stage read 12.8-29.5 ms against a full
# apply of 0.39-0.74 ms: a part apparently costing 40x the whole, which is the only reason THAT mistake
# surfaced. Both traps point the same way -- a decomposition is not to be trusted until its parts are
# shown to sum to the whole.
#
# The corrected picture at p=15, per component, and what a fix should target:
#
#     full apply (2 components)   589.8 µs
#     _anterpolateθ!              216.3 µs   per component   -> 74% of the cost
#     _anterpolateφbfft!           77.6 µs   per component   -> 26%
#     2 x (θ + φ)                 587.9 µs   i.e. 99.7% of the apply
#
# THE θ KERNEL IS THE TARGET, and the cause is measured rather than guessed. Its `mul!` arithmetic is only
# ~11 µs of the 216 µs: 56 calls at ~199 ns. The cost is in the NONCONFORM branch, which reads
# `view(fvals, flippingφindices, indices)` -- rows indexed through a permutation `Vector{Int}`, which is
# not strided, so `mul!` takes a generic path costing 4.23 µs against 211 ns for the contiguous
# `view(fvals, :, indices)`. A 20x penalty on 28 of the 56 calls; 28 contiguous + 28 fancy accounts for
# 125 µs of the 216 µs.
#
# THE FIX THAT FOLLOWS, not yet made: `flippingφindices` is `vcat(Nφ÷2+1:Nφ, 1:Nφ÷2)` -- a circular shift.
# So the fancy view decomposes into TWO CONTIGUOUS row blocks, and one 4.23 µs call becomes two ~211 ns
# calls. The same pattern appears in the interpolator's `_interpolateθ!`
# (`view(fvals, flippingindices, negindices)`), so both directions stand to gain.

const REVERSEPARENTFIELD = preparedtangentialfield(TARGET)

for p in (8, 15, 20)
    interpolator = MLFMA.LocalθGlobalφInterpolator{p}(ComplexF64, [SOURCE], [TARGET])
    anterpolator = MLFMA.anterpolator(interpolator; nchunks=1)
    scratch = MLFMA.interimscratch(anterpolator, SOURCE, TARGET)
    childout = MLFMA.TangentialSphericalFarField(ComplexF64, SOURCE, Val(:zero))

    # Whole applies, forward and reverse, scratch-backed as production is.
    TIMINGS["reverse split"]["1 interpolation apply"]["p=$p"] = @benchmarkable(
        $interpolator($TANGENTIALSOURCEFIELD, $SOURCE, $TARGET), seconds = 10,
    )
    TIMINGS["reverse split"]["2 anterpolation apply"]["p=$p"] = @benchmarkable(
        $anterpolator($childout, $REVERSEPARENTFIELD, $TARGET, $SOURCE; interim=($scratch)),
        seconds = 10,
    )

    # The two sub-kernels the reverse apply is made of, called exactly as it calls them, ONE component
    # each -- so `2 * (θ + φ)` should reconstruct the two-component apply above. If it stops doing so,
    # the decomposition has drifted and the split numbers are not to be trusted.
    information = MLFMA.anterpolationinformation(
        anterpolator,
        TARGET,
        SOURCE,
        REVERSEPARENTFIELD[1],
        scratch.interimmatrix,
        MLFMA.defaultinterpolationtrait(typeof(REVERSEPARENTFIELD)),
    )
    φshape = (MLFMA.Nφ(SOURCE), MLFMA.Nθ(TARGET))

    TIMINGS["reverse split"]["3 anterpolate θ kernel"]["p=$p"] = @benchmarkable(
        MLFMA._anterpolateθ!(
            $(childout[1]),
            $(scratch.interimmatrix),
            $(information[1].θinformation),
            $(scratch.interimθvector),
        ),
        seconds = 10,
    )
    TIMINGS["reverse split"]["4 anterpolate φ kernel"]["p=$p"] = @benchmarkable(
        MLFMA._anterpolateφbfft!(
            $(scratch.interimmatrix),
            $(REVERSEPARENTFIELD[1]),
            $(scratch.ftpadmatrix),
            $(information[1].φinformation),
            $φshape,
        ),
        seconds = 10,
    )
end

# THE EQUIDISTANT FAMILY -- the cost side of accuracy group 11.
#
# WHAT ACCURACY GROUP 11 COSTS. That group shows the equidistant FFT at round-off where
# `LocalθGlobalφ` sits at ~3e-4 on the same grid; this one prices it. Both numbers are needed and
# NEITHER decides alone -- see the note at the top of group 11.
#
# HELD FIXED SO THE RATIO MEANS SOMETHING: identical `(Nθ, Nφ)`, the same two-component
# `TangentialSphericalFarField` under `SphericalθφInterpolation`, preallocated output, and a
# scratch-backed apply for every family that has one. The three-argument form is NOT used here even
# though the older `"apply"` group uses it: it allocates its own output field, which is 230 KiB of
# noise on top of a comparison whose whole point is a 3x-5x ratio.
#
# `interimscratch` DISPATCHES ON THE ANTERPOLATOR for every family, and the returned scratch serves
# both directions, so the reverse row is built from the same object as the forward one. The Lagrange
# backend returns `nothing` on purpose (its kernel has not been buffered -- task #107), so its rows
# measure an allocating path and say so in the `allocs` column rather than in a comment.

# INPUTS ARE FILLED WITH A REAL CASE, never left at `Val(:zero)`, for the reason `preparedfield`
# states and which this group hit in its first revision: an all-zero moment made the Lagrange backend
# report a 9x forward/reverse asymmetry that vanished once real data went in. Zeros are not a neutral
# input -- they change which arithmetic the hardware actually performs.
function equidistantfixture(interpolator, child, parent)
    anterpolator = MLFMA.anterpolator(interpolator; nchunks=1)
    childfield = MLFMA.TangentialSphericalFarField(ComplexF64, child, Val(:zero))
    parentfield = MLFMA.TangentialSphericalFarField(ComplexF64, parent, Val(:zero))
    case = ID.RandomTangentialBandlimitedCase(; L=10, seed=0x1234)
    ID.fillexact!(childfield, case, child)
    ID.fillexact!(parentfield, case, parent)
    return (
        interpolator=interpolator,
        anterpolator=anterpolator,
        scratch=MLFMA.interimscratch(anterpolator, child, parent),
        childfield=childfield,
        parentfield=parentfield,
        childout=MLFMA.TangentialSphericalFarField(ComplexF64, child, Val(:zero)),
        parentout=MLFMA.TangentialSphericalFarField(ComplexF64, parent, Val(:zero)),
    )
end

# `interpolationtrait` passed explicitly, for the reason the `"transfer"` group's fixture records at
# length: an interpolator left on `DefaultInterpolation` makes every apply rebuild its information
# container, and that cost belongs to the fixture rather than to the family.
function equidistantfamilies(child, parent, θorder)
    return (
        "1 GL localglobal p=$θorder" => (
            MLFMA.buildinterpolator(
                MLFMA.LocalθGlobalφInterpolationBuilder(; θorder=θorder),
                [child.spherical],
                [parent.spherical];
                interpolationtrait=MLFMA.SphericalθφInterpolation(),
            ),
            child.spherical,
            parent.spherical,
        ),
        "2 equi FFT" => (
            EquidistantInterpolator{ComplexF64}(
                FFTBackend(); interpolationtrait=MLFMA.SphericalθφInterpolation()
            ),
            child.equidistant,
            parent.equidistant,
        ),
        "3 equi Lagrange p=10" => (
            EquidistantInterpolator{ComplexF64}(
                LagrangeBackend(10); interpolationtrait=MLFMA.SphericalθφInterpolation()
            ),
            child.equidistant,
            parent.equidistant,
        ),
    )
end

_gridpair(s) = (spherical=s, equidistant=equidistant(s))

for (name, (interpolator, child, parent)) in
    equidistantfamilies(_gridpair(SOURCE), _gridpair(TARGET), 15)
    fixture = equidistantfixture(interpolator, child, parent)

    TIMINGS["equidistant"]["$name"]["1 interpolate"] = @benchmarkable(
        $(fixture.interpolator)(
            $(fixture.parentout),
            $(fixture.childfield),
            $child,
            $parent;
            interim=($(fixture.scratch)),
        ),
        seconds = 10,
    )
    TIMINGS["equidistant"]["$name"]["2 anterpolate"] = @benchmarkable(
        $(fixture.anterpolator)(
            $(fixture.childout),
            $(fixture.parentfield),
            $parent,
            $child;
            interim=($(fixture.scratch)),
        ),
        seconds = 10,
    )
end

# HOW THE RATIO SCALES, which is the part a single grid pair cannot show and the part that decides
# whether the equidistant family is affordable on a deep tree. The equidistant FFT transforms a
# DOUBLED-θ torus at both ends -- a full 2D transform on `Nφ x 2Nθ` -- while `LocalθGlobalφ` pays a
# local θ stencil plus 1D φ FFTs. Different asymptotics, so the gap GROWS rather than sitting at a
# constant: measured 2.9x at 18x36 -> 26x52, 3.2x at 26x52 -> 40x80, 5.2x at 40x80 -> 64x128.
#
# That is a structural cost and not a missing optimization, but it is not the floor either. Two shapes
# were measured against the current one at 56x56 -> 108x108 (raw FFTW plans, 2000 samples):
# fully separable 1D passes with the resampling interleaved -- so each dimension's INVERSE runs while
# the other is still at child length -- gives 1.14x, and dropping the θ doubling gives 2.03x. Task #107
# holds both; the doubling is the one worth having.
# Equal TOLERANCE, not equal grid ################################################################
#
# Each family runs on the grid group 12b says it is entitled to for that tolerance -- so the rows are
# not comparable in grid size, deliberately. That is the point: the accuracy is pinned by the row and
# only the cost varies, which is the arrangement the equal-grid groups above cannot express.
#
# A family that never reaches a tolerance within the swept range is ABSENT from that row rather than
# reported at its best effort. Printing its time anyway would put a number next to an accuracy it does
# not achieve, which is the exact confusion this file's accuracy/runtime split exists to prevent.

function toleranceapply(family, D)
    interpolator, src, dst = family.build(D)
    anterpolator = MLFMA.anterpolator(interpolator; nchunks=1)
    scratch = MLFMA.interimscratch(anterpolator, src, dst)
    input = MLFMA.TangentialSphericalFarField(ComplexF64, src, Val(:zero))
    ID.fillexact!(input, TOLERANCECASE, src)
    return (
        interpolator=interpolator,
        src=src,
        dst=dst,
        scratch=scratch,
        input=input,
        out=MLFMA.TangentialSphericalFarField(ComplexF64, dst, Val(:zero)),
    )
end

for tol in TOLERANCES, family in TOLERANCEFAMILIES
    D = mindegree(TOLERANCEERRORS[family.label], tol)
    D === nothing && continue
    f = toleranceapply(family, D)
    label = "$(family.label) @Nθ$(MLFMA.NθNφ(D)[1])"
    TIMINGS["equal tolerance"]["eps=$(tol)"][label] = if f.scratch === nothing
        @benchmarkable $(f.interpolator)($(f.out), $(f.input), $(f.src), $(f.dst))
    else
        @benchmarkable $(f.interpolator)(
            $(f.out), $(f.input), $(f.src), $(f.dst); interim=($(f.scratch))
        )
    end
end

const SCALINGGRIDS = (
    (MLFMA.SphericalSampling{Float64}(18, 36), MLFMA.SphericalSampling{Float64}(26, 52)),
    (MLFMA.SphericalSampling{Float64}(26, 52), MLFMA.SphericalSampling{Float64}(40, 80)),
    (MLFMA.SphericalSampling{Float64}(40, 80), MLFMA.SphericalSampling{Float64}(64, 128)),
)

for (childgrid, parentgrid) in SCALINGGRIDS
    label = "$(MLFMA.Nθ(childgrid))x$(MLFMA.Nφ(childgrid)) -> $(MLFMA.Nθ(parentgrid))x$(MLFMA.Nφ(parentgrid))"
    for (name, (interpolator, child, parent)) in
        equidistantfamilies(_gridpair(childgrid), _gridpair(parentgrid), 15)
        fixture = equidistantfixture(interpolator, child, parent)
        TIMINGS["equidistant scaling"][label][name] = @benchmarkable(
            $(fixture.interpolator)(
                $(fixture.parentout),
                $(fixture.childfield),
                $child,
                $parent;
                interim=($(fixture.scratch)),
            ),
            seconds = 10,
        )
    end
end

# Construction of the interpolation information, which happens once per (level, sampling pair) in a
# real core and so belongs in a separate group from the per-matvec apply above.
for p in (8, 20)
    TIMINGS["information"]["lagrange"]["p=$p"] = @benchmarkable(
        MLFMA.SphericalLagrangeInterpolator{$p}(ComplexF64, [$SOURCE], [$TARGET]),
        seconds = 10,
    )
    TIMINGS["information"]["localglobal"]["p=$p"] = @benchmarkable(
        MLFMA.LocalθGlobalφInterpolator{$p}(ComplexF64, [$SOURCE], [$TARGET]), seconds = 10,
    )
end
