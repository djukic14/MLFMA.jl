# WHERE DOES COMPRESSED LEAF STORAGE TURN FASTER IN 2-D, AND DOES IT TURN FASTER AT ALL?
#
# The 2-D counterpart of `shesynthesis.jl`, and it reaches the opposite verdict, which is the whole
# reason it exists as a separate measurement rather than an assumption carried over. Both compare the
# same two shapes:
#
#     sampled      F_g = P_g x_g                       one GEMV per group, `(Nφ × N_j)`
#     compressed   A_g = C_g x_g  then  F = S [A_1 … A_k]
#                                                      one GEMV per group, `(2M+1 × N_j)`,
#                                                      plus ONE batched GEMM for the whole level
#
# BATCHING IS NOT OPTIONAL and is not being measured here as a variable -- #122 established that
# unbatched synthesis runs 1.63x slower than the sampled path in 3-D, a regression on both axes, and
# 2-D has strictly less margin to give away. Every compressed number below is already batched.
#
# THE FREE PARAMETER IS `ncoeff`, AND GETTING IT WRONG IS WHAT MADE THE ORIGINAL ESTIMATE OPTIMISTIC.
# Task #138 was specified from a table measured at `ncoeff = 15`, i.e. `M = 7`, the MEDIAN per-leaf
# order at ε = 1e-5. The scheme does not run at the median: `leveltruncationorder` takes the MAXIMUM
# over the level, because the batched GEMM needs one `M` or it goes ragged, and the calibrated
# `τ = ε/10` (see `checalibration.jl`) puts that at `M = 9`, `ncoeff = 19`. Both orders are swept
# below so the size of that mistake is visible rather than merely corrected.
#
# ============================== RESULTS ==============================
# Nφ = 28 (L = 13, λ/4 box), 3840 basis functions in the level held FIXED as occupancy varies, so the
# rows differ only in how the same work is grouped. Ratios are compressed/sampled; below 1 is a win.
#
#   basis fns/leaf   leaves   leaf block    M = 9 (ε=1e-5)   M = 7 (ε≈1e-3)
#            8         480       3.5 KB          1.67x            1.44x
#           16         240       7.0 KB          1.34x            1.24x
#           32         120      14.0 KB          1.07x            0.97x
#           64          60      28.0 KB          0.98x            0.84x
#          256          15     112.0 KB          0.95x            0.80x
#         1024           3     448.0 KB          0.89x            0.76x
#
# AT THE ORDER THE SCHEME ACTUALLY RUNS AT, IT NEVER MEANINGFULLY WINS. Break-even needs ~64 basis
# functions per leaf and the best case beyond it is 0.89x. An ordinary `meshcircle(4.0, 1/20)` sits at
# ~8 (503 functions over 60 leaves), where it is 1.67x SLOWER. At the looser `M = 7` the crossover
# moves down to ~32 and the win reaches 0.76x -- so the scheme is not worthless, it is simply on the
# wrong side of the crossover for the tolerances and meshes 2-D maps normally use.
#
# BLAS THREADING MAKES IT WORSE, NOT BETTER, which is the sharpest contrast with the 3-D result
# (0.57x serial improving to 0.22x threaded). At `M = 9`:
#
#   basis fns/leaf    1 thread   4 threads   8 threads
#            8          1.67x      2.27x       3.25x
#           16          1.34x      2.51x       2.56x
#           32          1.07x      1.07x       1.08x
#           64          0.98x      0.98x       0.98x
#         1024          0.89x      0.81x       1.03x
#
# The per-group products at low occupancy are `(19 × 8)` by `(8)` -- far too small to amortise thread
# dispatch -- while the sampled path's `(28 × 8)` GEMV is equally small and equally unthreaded, so the
# extra dispatch lands entirely on the compressed side. Above the crossover the two converge because
# both are dominated by the same batched GEMM. There is no thread count at which 2-D behaves like 3-D.
#
# WHY, and it is the same mechanism `shesynthesis.jl` measured, read the other way: SHE wins because
# the 3-D sampled multiply is MEMORY-BOUND -- a 4.5 MB block per group at ~0.5 flop/byte -- so
# compressing bytes compresses time. A 3.5 KB 2-D leaf block is in L1 either way. Nothing is being
# saved on memory traffic, so the compressed path's extra synthesis is pure added arithmetic.
#
# SYNTHETIC MATRICES, DELIBERATELY. This file times two matrix-multiply SHAPES; the numbers in them
# never enter the timing. What order those shapes should have is a question about real patterns and is
# answered in `checalibration.jl`, which is where `ncoeff = 19` comes from.

using LinearAlgebra, Random, Printf

const NΦ = 28          # production leaf grid, L = 13, λ/4 box
const TOTAL = 3840     # basis functions in the level, held fixed as occupancy varies

best(f, n=9) = (f(); minimum(@elapsed(f()) for _ in 1:n))

function compare(ncoeff, perleaf; Nφ=NΦ, total=TOTAL)
    nleaves = total ÷ perleaf
    rng = MersenneTwister(11)
    P = [randn(rng, ComplexF64, Nφ, perleaf) for _ in 1:nleaves]      # sampled patterns
    C = [randn(rng, ComplexF64, ncoeff, perleaf) for _ in 1:nleaves]  # coefficients
    S = randn(rng, ComplexF64, Nφ, ncoeff)                            # shared synthesis
    xs = [randn(rng, ComplexF64, perleaf) for _ in 1:nleaves]

    F = Matrix{ComplexF64}(undef, Nφ, nleaves)
    A = Matrix{ComplexF64}(undef, ncoeff, nleaves)

    sampled = best() do
        @inbounds for g in 1:nleaves
            @views mul!(F[:, g], P[g], xs[g])
        end
    end
    compressed = best() do
        @inbounds for g in 1:nleaves
            @views mul!(A[:, g], C[g], xs[g])
        end
        # The whole level in ONE GEMM. See this file's header on why the per-group alternative is not
        # offered as a variant.
        return mul!(F, S, A)
    end
    return sampled, compressed, nleaves
end

function crossoversweep(;
    threadcounts=(1, 4, 8), ncoeffs=(19, 15), occupancies=(8, 16, 32, 64, 256, 1024)
)
    restore = BLAS.get_num_threads()
    try
        for nthreads in threadcounts
            BLAS.set_num_threads(nthreads)
            @printf(
                "\n=== Nφ = %d, %d basis functions in the level, %d BLAS thread(s) ===\n",
                NΦ,
                TOTAL,
                nthreads
            )
            for ncoeff in ncoeffs
                @printf(
                    "\n  ncoeff = %d  (M = %d, compression %.2fx)\n",
                    ncoeff,
                    (ncoeff - 1) ÷ 2,
                    NΦ / ncoeff
                )
                @printf(
                    "  %14s %8s %12s %14s %8s %12s\n",
                    "basis fns/leaf",
                    "leaves",
                    "sampled ms",
                    "compressed ms",
                    "ratio",
                    "leaf block"
                )
                for perleaf in occupancies
                    s, c, nl = compare(ncoeff, perleaf)
                    @printf(
                        "  %14d %8d %12.4f %14.4f %7.2fx %9.1f KB\n",
                        perleaf,
                        nl,
                        s * 1e3,
                        c * 1e3,
                        c / s,
                        NΦ * perleaf * 16 / 1024
                    )
                    flush(stdout)
                end
            end
        end
    finally
        # A shared machine and a global setting: leave it as it was found, whatever happens above.
        BLAS.set_num_threads(restore)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    crossoversweep()
end
