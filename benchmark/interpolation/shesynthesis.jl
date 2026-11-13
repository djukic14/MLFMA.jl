# SLICE 2 of #122: how should the SHE synthesis kernel be shaped?
#
# RUN AS:  julia --startup-file=no --project=test benchmark/interpolation/shesynthesis.jl
# Run it BOTH ways -- the default (BLAS threaded) and with `BLAS.set_num_threads(1)` prepended. The
# single-threaded numbers are the fair kernel comparison, because real MLFMA threads the GROUP LOOP
# with serial BLAS inside; the threaded ones flatter the batched variant by letting one big GEMM use
# every core where the baseline issues many small GEMVs.
#
# ANSWER (λ/4 leaf, NΩ=1568, P=9, N_j=90): batched SHE synthesis is 0.57x single-threaded and 0.22x
# threaded -- i.e. FASTER than today's leaf aggregation, not slower. Unbatched is 1.63x SLOWER, so the
# batching is mandatory rather than an optimisation.
#
# WHY THE FLOP MODEL SAID THE OPPOSITE, and this is the lesson: SHE does 1.76x MORE arithmetic
# (3(P+1)^2 N_j + 3 NΩ (P+1)^2 against 2 NΩ N_j), and I predicted 1.84x slower from that. But the
# baseline leaf multiply is MEMORY-BOUND -- it streams a (2NΩ x N_j) = 4.5 MB matrix per group at an
# arithmetic intensity of ~0.5 flop/byte. SHE reads a 10.4x smaller coefficient block plus a shared S
# that stays resident (~8.7 flop/byte). Time follows BYTES, so compressing storage 10x speeds the
# kernel up rather than slowing it down. A flop count was simply the wrong model for this kernel.
#
# WHY THIS IS THE KERNEL THAT MATTERS. 95 % of SHE's added cost is synthesis (measured, step 1). The
# flop model says leaf aggregation goes to 1.84x at the current λ/4 leaf, but a flop model cannot see
# BLAS efficiency, and the three candidate shapes below differ enormously in it while doing identical
# arithmetic. Ring sampling died of exactly this gap between flop count and realised throughput.
#
# THE STRUCTURAL FACT THE SHAPES EXPLOIT. `S[q, lm] = Y_lm(direction q)` depends only on the LEVEL's
# sampling and the order P -- not on the group. Every leaf at a level shares the same S. So synthesis
# over all leaves is one big GEMM `S * [A_1 A_2 ... A_k]` rather than k skinny ones, and the only
# question is how much that is worth.
#
#   BASELINE   per group: F_g[2NΩ] = Fcoll[2NΩ x N_j] * x_g[N_j]         (what MLFMA does today)
#   SHE-EACH   per group: A_g = a_g * x_g ; F_g[NΩ x 3] = S * A_g        (skinny GEMM per group)
#   SHE-BATCH  per group: A_g = a_g * x_g ; then ONE  S * [A_1 ... A_k]  (BLAS-3 across groups)
#
# Coefficient aggregation is block-diagonal (each group has its own basis functions), so it stays
# per-group in all variants; only synthesis can batch.

using LinearAlgebra, Printf, Random

best(f, n=5) = (f(); minimum(@elapsed(f()) for _ in 1:n))

function study(; NΩ=1568, P=9, Nj=90, ngroups=256, D=3)
    nc = (P + 1)^2
    Random.seed!(2)

    # --- BASELINE: today's leaf aggregation. Tangential, so 2 components, stored as one tall matrix
    # per group (that is how TangentialSphericalRadiationFieldCollection holds it).
    Fcoll = [randn(ComplexF64, 2NΩ, Nj) for _ in 1:ngroups]
    xs = [randn(ComplexF64, Nj) for _ in 1:ngroups]
    Fout = [Vector{ComplexF64}(undef, 2NΩ) for _ in 1:ngroups]
    baseline() =
        for g in 1:ngroups
            mul!(Fout[g], Fcoll[g], xs[g])
        end

    # --- SHE: coefficients per basis function, Cartesian so D components. Layout [lm*D, basis] keeps
    # the coefficient aggregation a single GEMV per group.
    acoef = [randn(ComplexF64, nc * D, Nj) for _ in 1:ngroups]
    Ag = [Vector{ComplexF64}(undef, nc * D) for _ in 1:ngroups]
    S = randn(ComplexF64, NΩ, nc)

    Fg = [Matrix{ComplexF64}(undef, NΩ, D) for _ in 1:ngroups]
    function sheeach()
        for g in 1:ngroups
            mul!(Ag[g], acoef[g], xs[g])
            mul!(Fg[g], S, reshape(Ag[g], nc, D))
        end
    end

    Aall = Matrix{ComplexF64}(undef, nc, D * ngroups)
    Fall = Matrix{ComplexF64}(undef, NΩ, D * ngroups)
    function shebatch()
        for g in 1:ngroups
            mul!(Ag[g], acoef[g], xs[g])
            @views Aall[:, ((g - 1) * D + 1):(g * D)] .= reshape(Ag[g], nc, D)
        end
        return mul!(Fall, S, Aall)
    end

    tb = best(baseline)
    te = best(sheeach)
    tt = best(shebatch)

    # Coefficient aggregation alone, to see how much of SHE's time is the part that cannot batch.
    coefonly() =
        for g in 1:ngroups
            mul!(Ag[g], acoef[g], xs[g])
        end
    tc = best(coefonly)

    mem = D * nc / (2NΩ)
    @printf(
        "%5d %4d %5d %7d | %8.2f %8.2f %8.2f %8.2f | %6.2f %6.2f | %6.3f\n",
        NΩ,
        P,
        Nj,
        ngroups,
        tb * 1e3,
        te * 1e3,
        tt * 1e3,
        tc * 1e3,
        te / tb,
        tt / tb,
        mem
    )
    return nothing
end

println(
    "Dense SHE synthesis shapes vs today's leaf aggregation. Times in ms for the whole level.",
)
println("`each x` and `batch x` are SHE/baseline -- BELOW 1 means SHE is FASTER.")
println(
    "`coef` is the non-batchable coefficient-aggregation part. `mem` = D(P+1)^2 / 2NΩ.\n"
)
@printf(
    "%5s %4s %5s %7s | %8s %8s %8s %8s | %6s %6s | %6s\n",
    "NΩ",
    "P",
    "N_j",
    "groups",
    "base ms",
    "each ms",
    "batch ms",
    "coef ms",
    "each x",
    "batch x",
    "mem"
)

# The measured λ/4 operating point, then the leaf-size sweep from step 1 (e) with its own (NΩ, P, N_j).
study(; NΩ=1568, P=9, Nj=90, ngroups=256)     # λ/4, the current default
study(; NΩ=1568, P=9, Nj=90, ngroups=1024)    # same, more leaves (105k-unknown scale)
study(; NΩ=3528, P=13, Nj=604, ngroups=64)    # λ/2 leaves
study(; NΩ=800, P=7, Nj=32, ngroups=1024)     # λ/8 leaves
