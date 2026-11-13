# Is there anything in doing all of a parent's I2I translations AT ONCE?
#
# THE QUESTION. `H2Factory.i2ipass!` loops over a node's children, interpolating each child's moment
# onto the parent grid and accumulating: the first child overwrites, the rest add. The interpolation
# operator is the SAME for every child at a level -- same child grid, same parent grid -- and the
# per-child phase is applied on the PARENT grid after interpolation (#105 fused it into the
# interpolator's final write). So the children COULD go through one batched interpolation instead of
# up to eight separate ones. Summing them first is NOT available: different centres, different phases.
#
# WHAT THIS MEASURES, AND WHAT IT DELIBERATELY DOES NOT. Building a genuinely batched interpolator is
# the work under evaluation, so measuring it would mean doing it first. Instead this measures the
# CEILING: the dominant kernels -- the θ stencil (a GEMM) and the φ resample (an FFT) -- run eight
# times separately against once with eight columns, plus the parent-moment accumulate traffic. No
# batched implementation can beat that ceiling, so a small ceiling settles the question outright.
#
# WHY THE ANSWER IS NOT OBVIOUS EITHER WAY. Two measured precedents point in opposite directions:
#
#   * FOR: the SHE leaf multiply gained 12x from batching, because one shared operator streamed once
#     instead of per group -- structurally the same argument as here.
#   * AGAINST: that same study put the crossover at 16-32 GROUPS, and measured batching at 8 groups
#     as 3.7x SLOWER than not batching. A parent has at most 8 children, and fewer on a non-uniform
#     tree. The mechanism differs -- SHE amortised a 2.4 MB matrix, this would amortise per-call
#     overhead -- so the crossover need not transfer, but 8 is squarely in the region where the one
#     directly relevant measurement says batching loses.
#
# THE PRIZE IS BOUNDED AND SMALL. The far path is aggregation ~17 % / I2O translation ~70 % / O2O
# disaggregation ~13 % (#120). I2I sits INSIDE the 17 %, sharing it with the leaf multiply, and its
# transpose inside the 13 %. So all I2I work in both directions is at most ~20-25 % of the far matvec,
# and any speedup here is that number times the ratio measured below.

using MLFMA
using BEAST, H2Factory
using FFTW
using LinearAlgebra
using Printf
using Random

const NCHILDREN = 8          # a full TwoNTree parent; fewer on a non-uniform tree

sampling(degree) = MLFMA.SphericalSampling{Float64}(MLFMA.NθNφ(degree)...)

griddims(s) = (length(MLFMA.θs(s)), length(MLFMA.φs(s)))

best(f; samples=5) = (f(); minimum(@elapsed(f()) for _ in 1:samples))

# --- the two kernels, separately and batched -------------------------------------------------------

# φ RESAMPLE, the global stage of `LocalθGlobalφ`: an in-place complex FFT along the φ dimension.
# Separate calls run one plan per child; the batched form uses FFTW's `howmany` over a third axis,
# which is the whole mechanism the batching would exploit.
function φstage(Nθ, Nφ)
    single = Matrix{ComplexF64}(undef, Nθ, Nφ)
    batch = Array{ComplexF64}(undef, Nθ, Nφ, NCHILDREN)
    Random.seed!(1)
    single .= randn(ComplexF64, Nθ, Nφ)
    batch .= randn(ComplexF64, Nθ, Nφ, NCHILDREN)

    psingle = FFTW.plan_fft!(single, 2; flags=FFTW.MEASURE)
    pbatch = FFTW.plan_fft!(batch, 2; flags=FFTW.MEASURE)

    separate = best(() -> for _ in 1:NCHILDREN
        psingle * single
    end)
    batched = best(() -> pbatch * batch)
    return separate, batched
end

# θ STENCIL, the local stage: a `(Nθtarget × Nθsource)` operator applied down the θ dimension. As a
# matrix product that is `S * child` per child, against `S * [child_1 … child_8]` batched.
function θstage(Nθsource, Nθtarget, Nφ)
    Random.seed!(2)
    S = randn(ComplexF64, Nθtarget, Nθsource)
    children = [randn(ComplexF64, Nθsource, Nφ) for _ in 1:NCHILDREN]
    out = Matrix{ComplexF64}(undef, Nθtarget, Nφ)

    wide = reduce(hcat, children)
    outwide = Matrix{ComplexF64}(undef, Nθtarget, NCHILDREN * Nφ)

    separate = best(() -> for c in children
        LinearAlgebra.mul!(out, S, c)
    end)
    batched = best(() -> LinearAlgebra.mul!(outwide, S, wide))
    return separate, batched
end

# PARENT ACCUMULATE, the traffic the loop structure itself costs: eight passes writing into the parent
# moment against one pass summing eight staged results. Both kernels above are memory-bound at these
# sizes, so this is not a rounding error.
function accumulatestage(Nθ, Nφ)
    Random.seed!(3)
    parent = Matrix{ComplexF64}(undef, Nθ, Nφ)
    staged = [randn(ComplexF64, Nθ, Nφ) for _ in 1:NCHILDREN]

    separate = best(function ()
        parent .= staged[1]
        for i in 2:NCHILDREN
            parent .+= staged[i]
        end
    end)
    batched = best(function ()
        @inbounds for q in eachindex(parent)
            acc = staged[1][q]
            for i in 2:NCHILDREN
                acc += staged[i][q]
            end
            parent[q] = acc
        end
    end)
    return separate, batched
end

function run(; degrees=(6, 13, 26))
    @printf(
        "%d children per parent; separate = today's loop, batched = the ceiling on doing them at once\n\n",
        NCHILDREN
    )
    println(
        "  level    child grid    parent grid |    φ FFT sep/batch |    θ GEMM sep/batch |  accumulate sep/batch",
    )
    for d in degrees
        source = sampling(d)
        target = sampling(2d)
        Nθs, Nφs = griddims(source)
        Nθt, Nφt = griddims(target)

        φsep, φbat = φstage(Nθt, Nφs)             # φ resample runs on the θ-resampled grid
        θsep, θbat = θstage(Nθs, Nθt, Nφs)
        asep, abat = accumulatestage(Nθt, Nφt)

        @printf(
            "  p=%-3d  %4d x %-5d  %4d x %-5d |  %6.1f / %6.1f µs  %6.1f / %6.1f µs  %6.1f / %6.1f µs\n",
            d,
            Nθs,
            Nφs,
            Nθt,
            Nφt,
            1e6φsep,
            1e6φbat,
            1e6θsep,
            1e6θbat,
            1e6asep,
            1e6abat
        )
        flush(stdout)
    end

    println("\n  ratios (separate / batched -- above 1 means batching wins)")
    for d in degrees
        source = sampling(d)
        target = sampling(2d)
        Nθs, Nφs = griddims(source)
        Nθt, Nφt = griddims(target)
        φsep, φbat = φstage(Nθt, Nφs)
        θsep, θbat = θstage(Nθs, Nθt, Nφs)
        asep, abat = accumulatestage(Nθt, Nφt)
        total = (φsep + θsep + asep) / (φbat + θbat + abat)
        @printf(
            "  p=%-3d   φ %.2fx   θ %.2fx   accumulate %.2fx   ALL THREE %.2fx\n",
            d,
            φsep / φbat,
            θsep / θbat,
            asep / abat,
            total
        )
        flush(stdout)
    end
    println()
    println(
        "Read the last column against the ~20-25 % of the far matvec that ALL I2I work occupies:\n" *
        "a ratio of r there is worth at most (1 - 1/r) x 0.25 of the far path.",
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
