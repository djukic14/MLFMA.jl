# Can I2O translation be batched by TRANSFER VECTOR, and is it worth it?
#
# THE QUESTION. `H2Factory.translateintonode!` loops over (receiving node, translating node) PAIRS,
# and each call streams the translation operator `T` once to do `target += T ⊙ source`. In an octree
# `T` depends only on the RELATIVE OFFSET between the two boxes -- at most 7³-3³ = 316 distinct
# offsets in a standard interaction list -- so the same `T` is re-streamed once per pair sharing an
# offset. Grouping the loop by offset would stream it once per GROUP instead.
#
# WHY THIS IS NOT THE I2I QUESTION AGAIN. Batching a parent's I2I translations lost because a parent
# has at most 8 children, below the 16-32 crossover the SHE study measured. Here the batch is the
# number of pairs sharing an offset, and PART 1 below measures what that actually is rather than
# assuming it. If it is in the hundreds the regime is the one where SHE batching won 12x; if it is
# small this dies for the same reason I2I did.
#
# AND IT TARGETS THE RIGHT STAGE. I2O translation is 70 % of the far matvec (#120), against the
# ~20-25 % all I2I work occupies.
#
# TWO PARTS, and the first can kill it on its own:
#   1. the distribution of pairs per transfer vector, on a real tree
#   2. grouped versus per-pair elementwise multiply-accumulate, at real grid sizes
#
# Part 2 measures the CEILING -- the kernel with `T` streamed once per group against once per pair --
# not an implementation. Restructuring the traversal is the work under evaluation.

using MLFMA
using BEAST, CompScienceMeshes, H2Trees, H2Factory
using LinearAlgebra
using Printf
using Random
using StaticArrays

const HT = H2Trees

function physicalconstants(f::U) where {U}
    μ0 = U(4 * π * 1e-7)
    ε0 = U(8.854187812e-12)
    c = U(1) / sqrt(ε0 * μ0)
    λ = c / f
    κ = U(2 * π / λ)
    return (; ω=κ * c, μ=μ0, λ, κ, γ=im * κ)
end

best(f; samples=5) = (f(); minimum(@elapsed(f()) for _ in 1:samples))

# --- part 1: how many pairs share a transfer vector? ----------------------------------------------
#
# The offset is computed GEOMETRICALLY -- `(centre_t - centre_r) / (2·halfsize)`, rounded -- rather
# than read off a plan, so it needs no assumption about how the interaction list is stored. On a
# regular octree those quotients are integers; the rounding residual is reported so a fixture that
# breaks that assumption is visible rather than silently binned wrong.
function transfervectorhistogram(tree, plan, level)
    counts = Dict{SVector{3,Int},Int}()
    worstresidual = 0.0
    npairs = 0
    for receivingnode in HT.LevelIterator(tree, level)
        translating = HT.translatingnodes(plan, receivingnode, level)
        translating === nothing && continue
        hs = HT.halfsize(tree, receivingnode)
        cr = HT.center(tree, receivingnode)
        for t in translating
            d = (HT.center(tree, t) - cr) ./ (2hs)
            key = SVector{3,Int}(round.(Int, d))
            worstresidual = max(worstresidual, maximum(abs.(d .- key)))
            counts[key] = get(counts, key, 0) + 1
            npairs += 1
        end
    end
    return counts, npairs, worstresidual
end

# --- part 2: the kernel, grouped against per pair --------------------------------------------------
#
# `target += T ⊙ source`, `nd` directions, `npairs` pairs sharing one `T`. The per-pair form re-reads
# `T` every iteration; the grouped form reads it once and sweeps the pairs inside. Both do exactly the
# same arithmetic and touch exactly the same source/target bytes -- the only difference is how often
# `T` crosses the memory hierarchy, which is the entire claim.
function kernelcomparison(nd, npairs)
    Random.seed!(7)
    T = randn(ComplexF64, nd)
    sources = [randn(ComplexF64, nd) for _ in 1:npairs]
    targets = [zeros(ComplexF64, nd) for _ in 1:npairs]

    perpair = best(function ()
        for p in 1:npairs
            s, t = sources[p], targets[p]
            @inbounds @simd for q in 1:nd
                t[q] += T[q] * s[q]
            end
        end
    end)

    # Grouped: `T` hoisted out of the pair loop. Blocked over directions so `T`'s working set is a
    # cache block reused across every pair, which is the form a restructured traversal would take.
    block = 1024
    grouped = best(function ()
        @inbounds for q0 in 1:block:nd
            q1 = min(q0 + block - 1, nd)
            for p in 1:npairs
                s, t = sources[p], targets[p]
                @simd for q in q0:q1
                    t[q] += T[q] * s[q]
                end
            end
        end
    end)
    return perpair, grouped
end

function run(; radius=2.0, meshfactor=8)
    p = physicalconstants(5e7)
    λ, γ = p.λ, p.γ
    X = raviartthomas(meshsphere(radius * λ, λ / meshfactor))
    efie = MLFMA.efie(; wavenumber=p.κ, alpha=(-im * p.ω * p.μ))
    tree = HT.TwoNTree(X; builder=HT.TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0))
    strategy = MLFMA.MLFMAStrategy(
        efie,
        X;
        samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
        nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
            MLFMA.NEARINTERACTIONSQUADSTRATEGYRECOM
        ),
    )
    A = MLFMA.GalerkinMLFMA(efie, X; tree=tree, strategy=strategy, verbose=false)
    core = H2Factory.core(A)
    # The DISAGGREGATION side is the one carrying the interaction lists: the trial aggregation plan
    # is a plain `AggregatePlan`, and `translatingnodes` is defined on the *Translate* plans.
    plan = core.plans.testdisaggregationplan

    @printf(
        "N = %d   sphere %.1fλ   mesh λ/%d   leaves λ/4\n\n",
        numfunctions(X),
        radius,
        meshfactor
    )
    println("PART 1 -- pairs per transfer vector, per level")
    println("  level  pairs   vectors   mean    median     max   grid nd   residual")

    levels = sort(collect(HT.levels(tree)))
    perlevel = Tuple{Int,Int,Int}[]        # (level, median pairs per vector, nd)
    for level in levels
        counts, npairs, residual = transfervectorhistogram(tree, plan, level)
        isempty(counts) && continue
        v = sort(collect(values(counts)))
        idx = MLFMA._samplinglevelindex(level, core.minrelevantlevel, core.samplings)
        (1 <= idx <= length(core.samplings)) || continue
        s = core.samplings[idx]
        nd = length(MLFMA.θs(s)) * length(MLFMA.φs(s))
        @printf(
            "  %5d  %6d  %7d  %6.1f  %7d  %6d  %8d  %.1e\n",
            level,
            npairs,
            length(v),
            npairs / length(v),
            v[(end + 1) ÷ 2],
            v[end],
            nd,
            residual
        )
        push!(perlevel, (level, v[(end + 1) ÷ 2], nd))
        flush(stdout)
    end

    println("\nPART 2 -- `target += T .* source`, T streamed per pair vs once per group")
    println("     nd   pairs   per-pair    grouped    ratio")
    for (level, medianpairs, nd) in perlevel
        medianpairs < 2 && continue
        perpair, grouped = kernelcomparison(nd, medianpairs)
        @printf(
            "  %5d  %6d  %8.1f µs %8.1f µs  %6.2fx\n",
            nd,
            medianpairs,
            1e6perpair,
            1e6grouped,
            perpair / grouped
        )
        flush(stdout)
    end
    println()
    println(
        "A ratio of r on this kernel is worth at most (1 - 1/r) x 0.70 of the far matvec,\n" *
        "since I2O translation is 70 % of it (#120).",
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
