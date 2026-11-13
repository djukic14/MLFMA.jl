# Command-line entry point for the interpolation accuracy and runtime studies:
#
#     julia --project=benchmark benchmark/interpolation/run.jl            # accuracy only
#     julia --project=benchmark benchmark/interpolation/run.jl --timings  # accuracy and runtime
#
# `benchmarks.jl` defines and runs nothing, so this file is the only place that starts a run -- same
# arrangement as `benchmark/run.jl` for the far-field suite. See this directory's README.md.
#
# Accuracy is the default and runtime is opt-in because the two answer different questions and take
# very different times: the accuracy studies are seconds, the timing group is minutes. A developer
# checking whether a change moved the error budget should not have to wait for `BenchmarkTools`.
#
# No `tune!` here, for the reason given in `benchmark/run.jl`: every case sets its own `seconds`, and
# tuning rewrites evaluations-per-sample from whatever the machine is doing at that moment.

include(joinpath(@__DIR__, "benchmarks.jl"))

const WANTTIMINGS = "--timings" in ARGS

println("MLFMA angular interpolation studies")
println("=" ^ 96)
println()
println(
    "  source  ",
    MLFMA.Nθ(SOURCE),
    "x",
    MLFMA.Nφ(SOURCE),
    "  (Nθ x Nφ), supports degree ",
    ID.supporteddegree(SOURCE),
)
println(
    "  middle  ",
    MLFMA.Nθ(MIDDLE),
    "x",
    MLFMA.Nφ(MIDDLE),
    "  target θ nodes, source φ nodes -- the θ/φ stage boundary",
)
println(
    "  target  ",
    MLFMA.Nθ(TARGET),
    "x",
    MLFMA.Nφ(TARGET),
    "  (Nθ x Nφ), supports degree ",
    ID.supporteddegree(TARGET),
)
println()
println(
    "  ρ = case bandwidth / source supported degree. ρ = 1 is where the SOURCE grid stops"
)
println(
    "  carrying the case, so rows past it are expected to be bad and are kept as a reference.",
)
println(
    "  All numbers are relative to a closed-form harmonic evaluated on both grids: no mesh, no",
)
println("  operator, no assembly, no translation, no truncation, no tree.")
println()

println("ACCURACY")
println("-" ^ 96)
for study in STUDIES
    elapsed = @elapsed study.run(stdout)
    println("\n  [", study.name, " completed in ", round(elapsed; sigdigits=3), " s]")
end

if WANTTIMINGS
    println()
    println("RUNTIME")
    println("-" ^ 96)
    println(
        "  Runtime only. Nothing in this section says whether a configuration is more ACCURATE;",
    )
    println(
        "  that is the section above, and a change has to win in both to be an improvement."
    )
    println()

    results = run(TIMINGS; verbose=true)
    println()
    for (group, cases) in sort(collect(pairs(results)); by=first)
        println("\n", group)
        for (case, orders) in sort(collect(pairs(cases)); by=first)
            for (order, trial) in sort(collect(pairs(orders)); by=first)
                estimate = minimum(trial)
                println(
                    "  ",
                    rpad(string(case, " / ", order), 40),
                    lpad(BenchmarkTools.prettytime(BenchmarkTools.time(estimate)), 12),
                    lpad(BenchmarkTools.prettymemory(BenchmarkTools.memory(estimate)), 12),
                    lpad(string(BenchmarkTools.allocs(estimate), " allocs"), 14),
                )
            end
        end
    end
    println()
else
    println()
    println("Runtime group skipped. Re-run with `--timings` to include it.")
end
