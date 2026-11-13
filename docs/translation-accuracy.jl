using MLFMA, SHA

const ACCURACY_OFFSETS = (0.1, 0.2, 0.4, 0.8)
const ACCURACY_ORDERS = 1:24
const ACCURACY_DISTANCES = 10.0 .^ range(log10(0.5), 3; length=101)

function accuracygrid(dimension, family, L, refinement)
    L >= 1 && refinement >= 1 || throw(ArgumentError("Use positive cutoff and refinement"))
    valid = if dimension == 2
        family == "circular"
    else
        dimension == 3 && family in ("spherical", "equidistant")
    end
    valid || throw(ArgumentError("Sampling family does not match the dimension"))
    if dimension == 2
        n =
            refinement *
            first(MLFMA.samplingrate(MLFMA.CircularSamplingBuilder{Float64}(), L))
        return MLFMA.CircularSampling{Float64}(n)
    end
    nt, np = refinement .* MLFMA.NθNφ(L)
    return if family == "spherical"
        MLFMA.SphericalSampling{Float64}(nt, np)
    else
        MLFMA.Equidistant.EquidistantSampling{Float64}(nt, np)
    end
end

# Collinearity removes all azimuthal dependence in 3-D. Sum weights over φ,
# but keep exactly the production θ nodes, weights and translation kernel.
function accuracydirections(grid, dimension)
    directions, weights = MLFMA.cartesiansamplepoints(grid)
    if dimension == 2
        return directions, weights, [d[1] for d in directions]
    end
    return directions[1, :], vec(sum(weights; dims=1)), [d[3] for d in directions[1, :]]
end

function accuracygreen(dimension, k, radius)
    return if dimension == 3
        cis(-k * radius) / (4π * radius)
    else
        -im / 4 * MLFMA.Translations.SpecialFunctions.hankelh2(0, k * radius)
    end
end

function accuracysample(dimension, family, L, refinement, kd, offset)
    k = 2π
    D = kd / k
    D > abs(offset) || throw(ArgumentError("The expansion requires D > |x|"))
    grid = accuracygrid(dimension, family, L, refinement)
    directions, weights, mu = accuracydirections(grid, dimension)
    V = MLFMA.Sampling.SVector
    displacement = dimension == 3 ? V(0.0, 0.0, D) : V(D, 0.0)
    values = zeros(ComplexF64, length(directions))
    effective = MLFMA.translationoperator!(values, k * im, directions, displacement, L)
    approximation = sum(weights .* values .* exp.(-im * k * offset .* mu))
    reference = accuracygreen(dimension, k, D + offset)
    return (;
        error=abs(approximation / reference - 1),
        effective,
        directions=length(grid),
        approximation,
    )
end

# Independent analytic angular integral of the finite 3-D expansion. The power
# series is evaluated in BigFloat, avoiding the sampled integral's cancellation.
function sphericaljseries(l, z)
    term = z^l / prod(BigFloat(2j + 1) for j in 0:l)
    total = term
    for j in 1:1000
        term *= -z^2 / (2j * (2l + 2j + 1))
        total += term
        abs(term) <= eps(BigFloat) * max(abs(total), eps(BigFloat)) && return total
    end
    return error("Spherical Bessel reference did not converge")
end

function accuracyreference3d(D, offset, maxorder; bits=256)
    return setprecision(BigFloat, bits) do
        k = BigFloat(2π)
        d, x = BigFloat(D), BigFloat(offset)
        hankels = MLFMA.collectsphericalhankelsecond(maxorder, k * d)
        target = exp(-im * k * (d + x)) / (4 * big(π) * (d + x))
        accumulator = zero(Complex{BigFloat})
        errors = Float64[]
        for l in 0:maxorder
            accumulator += (-1)^l * (2l + 1) * hankels[l + 1] * sphericaljseries(l, k * x)
            push!(errors, Float64(abs((-im * k / (4 * big(π))) * accumulator / target - 1)))
        end
        return errors
    end
end

function accuracyreference2d(D, offset, maxorder)
    S = MLFMA.Translations.SpecialFunctions
    k = 2π
    hankels = MLFMA.collectcylindricalhankelsecond(maxorder, k * D)
    target = accuracygreen(2, k, D + offset)
    accumulator = hankels[1] * S.besselj(0, k * offset)
    errors = [abs((-im / 4) * accumulator / target - 1)]
    for m in 1:maxorder
        accumulator += 2 * (-1)^m * hankels[m + 1] * S.besselj(m, k * offset)
        push!(errors, abs((-im / 4) * accumulator / target - 1))
    end
    return errors
end

function translationaccuracydata()
    V = MLFMA.Sampling.SVector
    k = 2π
    families = ((3, "spherical"), (3, "equidistant"), (2, "circular"))
    curves = Dict{String,Any}()
    references = Dict{String,Any}()
    for dimension in (2, 3), offset in ACCURACY_OFFSETS
        columns = map(ACCURACY_DISTANCES) do kd
            D = kd / k
            D <= offset && return nothing
            return if dimension == 3
                accuracyreference3d(D, offset, last(ACCURACY_ORDERS))
            else
                accuracyreference2d(D, offset, last(ACCURACY_ORDERS))
            end
        end
        references["$dimension/$offset"] = columns
    end
    for (dimension, family) in families, refinement in (1, 2, 4), L in ACCURACY_ORDERS
        grid = accuracygrid(dimension, family, L, refinement)
        directions, weights, mu = accuracydirections(grid, dimension)
        values = zeros(ComplexF64, length(directions))
        effective = Int[]
        errors = Dict(string(x) => Union{Nothing,Float64}[] for x in ACCURACY_OFFSETS)
        integrated = Dict(string(x) => Union{Nothing,Float64}[] for x in ACCURACY_OFFSETS)
        phases = Dict(x => exp.(-im * k * x .* mu) for x in ACCURACY_OFFSETS)
        for (i, kd) in enumerate(ACCURACY_DISTANCES)
            D = kd / k
            displacement = dimension == 3 ? V(0.0, 0.0, D) : V(D, 0.0)
            actual = MLFMA.translationoperator!(values, k * im, directions, displacement, L)
            push!(effective, actual)
            for offset in ACCURACY_OFFSETS
                valid = D > offset
                e = if valid
                    abs(
                        sum(weights .* values .* phases[offset]) /
                        accuracygreen(dimension, k, D + offset) - 1,
                    )
                else
                    nothing
                end
                push!(errors[string(offset)], e)
                push!(
                    integrated[string(offset)],
                    valid ? references["$dimension/$offset"][i][actual + 1] : nothing,
                )
            end
        end
        curves["$family/$refinement/$L"] = (;
            errors, integrated, effective, directions=length(grid)
        )
    end
    return (;
        kd=ACCURACY_DISTANCES,
        offsets=ACCURACY_OFFSETS,
        maxorder=last(ACCURACY_ORDERS),
        curves,
    )
end

function ensuretranslationaccuracy(output)
    dependencies = (
        @__FILE__,
        joinpath(pkgdir(MLFMA), "src", "translation", "translation.jl"),
        joinpath(pkgdir(MLFMA), "src", "sampling", "sphericalsampling.jl"),
        joinpath(pkgdir(MLFMA), "src", "sampling", "circularsampling.jl"),
        joinpath(pkgdir(MLFMA), "src", "equidistant", "samplinginterface.jl"),
        joinpath(pkgdir(MLFMA), "src", "equidistant", "quadrature.jl"),
        joinpath(@__DIR__, "Manifest.toml"),
    )
    fingerprint = bytes2hex(
        sha256(join([string(VERSION); [read(p, String) for p in dependencies]]))
    )
    marker = joinpath(output, "accuracy.sha256")
    target = joinpath(output, "accuracy.js")
    isfile(target) &&
        isfile(marker) &&
        read(marker, String) == fingerprint &&
        return nothing
    @info "Generating translation-accuracy curves"
    data = translationaccuracydata()
    io = IOBuffer()
    print(io, "window.MLFMAAccuracy = ")
    PlotlyJS.JSON.print(io, data)
    println(io, ";")
    writeifchanged(target, take!(io))
    writeifchanged(marker, Vector{UInt8}(codeunits(fingerprint)))
    return nothing
end
