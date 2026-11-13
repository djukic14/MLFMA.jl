# Command-line entry point for the far-field benchmark suite:
#
#     julia --project=benchmark benchmark/run.jl
#
# `benchmarks.jl` defines the fixture and `SUITE` and runs nothing, so this file is the only place
# that starts a run. Keep it that way -- see the note at the top of `benchmarks.jl`.
#
# No `tune!(SUITE)` here. Every case in the suite sets its own `seconds`/`samples`, and tuning
# rewrites the evaluations-per-sample from whatever the machine happens to be doing at that moment,
# which is the opposite of what a run-to-run comparison wants. Tune by hand in a REPL if a specific
# case needs it.

include(joinpath(@__DIR__, "benchmarks.jl"))
# The batched leaf group lives in its own file because its fixture is a real map -- a mesh plus
# three constructions, ~49 s -- which would blow `benchmarks.jl`'s include budget for REPL work.
# It appends to the same `SUITE`, so everything below is unchanged.
include(joinpath(@__DIR__, "leafbatch.jl"))

results = run(SUITE; verbose=true)

display(results)
println()

# The leaves, deepest-first, so the interesting numbers are readable without expanding a nested
# `BenchmarkGroup` by hand.
for (group, cases) in sort(collect(pairs(results)); by=first)
    println("\n", group)
    for (case, levels) in sort(collect(pairs(cases)); by=first)
        for (level, trial) in sort(collect(pairs(levels)); by=first)
            estimate = minimum(trial)
            println(
                "  ",
                rpad(string(case, " / ", level), 46),
                lpad(BenchmarkTools.prettytime(BenchmarkTools.time(estimate)), 12),
                lpad(BenchmarkTools.prettymemory(BenchmarkTools.memory(estimate)), 12),
                lpad(string(BenchmarkTools.allocs(estimate), " allocs"), 14),
            )
        end
    end
end
println()
