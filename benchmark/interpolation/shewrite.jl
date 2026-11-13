# SE-MLFMA: HOW the synthesised pattern reaches the moment arena. Two strategies, against run length.
#
# H2Factory allocates every node moment as an `unsafe_wrap` view into pooled buffers, laid out by
# (level, node) with a node's components consecutive (`moments/allocatemoments.jl`,
# `moments/unsafe_wraps.jl`). Since `PlanSchedule` chunks the SAME node iterator with contiguous
# `index_chunks` batches, a chunk's moments are already the `(NΩ x D*k)` column-major matrix a batched
# synthesis wants -- so it can write STRAIGHT INTO THE ARENA across any maximal run of leaves that is
# consecutive, same-pool and actually contiguous.
#
# Measured run lengths (scratchpad, and the reason this file sweeps `r` rather than assuming one):
#     UNIFORM   sphere 1λ leaf λ/4, level 3:  56 leaves -> ONE run of 56
#     UNIFORM   sphere 2λ leaf λ/4, level 4: 272 leaves -> ONE run of 272
#     NON-UNIF  sphere 0.8λ,        level 4:  38 leaves -> runs of 32 and 6
#                                   level 3:  15 leaves -> 6 runs, median 2.5, max 4
# Uniform trees give one run per level. Non-uniform trees fragment at SHALLOW MIXED levels, where
# leaves interleave with internal nodes in node order. (Not the stored/transient split -- every leaf
# measured was stored bar one.)
#
# ============================== THE THING THAT SPLITS THE DESIGN ==============================
# The moment is a `TangentialSphericalFarField{2}` -- TWO components. SHE synthesises THREE Cartesian
# ones, because θ̂/φ̂ carry the frame's pole singularity and do not compress (see `shepatterns.jl`). So
# something has to turn 3 into 2 at the point of write, and there are two ways:
#
#   A. FUSED. Fold the frame into the operator: S_θ[q, (c,lm)] = Y_lm(q) * θ̂_c(q), so `F_θ = S_θ a`
#      lands directly in the arena. Two GEMMs per run, each into a stride-2NΩ view (θ and φ interleave
#      per node, which BLAS handles through `ldc` at no cost). No scratch, no second pass -- 2NΩ of
#      moment traffic per leaf instead of 8NΩ. But the operator grows 2.5 MB -> 15 MB and is
#      re-streamed PER RUN, so short runs pay for it with nothing to amortise against.
#
#   B. SCRATCH. Plain `(NΩ x 3r)` synthesis into scratch, then a per-direction 3->2 pass into the
#      arena. IMMUNE TO FRAGMENTATION: the expensive GEMM writes contiguous scratch regardless of what
#      the arena looks like, and only the cheap streaming projection touches the arena node by node.
#      Costs the scratch and 4x the moment traffic; operator stays 2.5 MB.
#
# BASELINE is today's sampled leaf multiply: one `(NΩ x N_j)` GEMV per tangential component per group.
#
# ================================ VERDICT: B WINS, A IS NOT WORTH BUILDING ================================
# The reasoning that motivated A -- fold the frame in, save 4x the moment traffic, win at long runs --
# is wrong, and wrong in the same way a flop-and-bytes argument was wrong once already in this feature
# (it predicted SHE would be 1.84x SLOWER than the sampled multiply when it is 0.57x). A never wins
# serially, and once B's projection is threaded A does not win at any run length.
#
# THREADED BLAS, which IS the production configuration -- `MLFMAStrategy` schedules the traversal
# SERIALLY (`nchunks = 1`, measured), so one task runs the level and calls a multi-threaded BLAS:
#
#      r    A fused    B scratch   B' threaded    baseline    A/base   B'/base   A/B'
#      8    2.947 ms    1.569 ms    2.229 ms      0.606 ms     4.86x    3.68x    1.32x
#     16    1.287 ms    0.919 ms    1.265 ms      1.077 ms     1.19x    1.17x    1.02x
#     32    1.532 ms    1.227 ms    1.500 ms      3.899 ms     0.39x    0.38x    1.02x
#     64    2.059 ms    1.921 ms    1.632 ms     13.445 ms     0.15x    0.12x    1.26x
#    128    3.058 ms    3.077 ms    2.559 ms     25.510 ms     0.12x    0.10x    1.20x
#    272    4.750 ms    6.013 ms    3.956 ms     52.455 ms     0.09x    0.08x    1.20x
#
# So: B' is 12x faster than today's leaf multiply at a full uniform level, and 1.20x faster than A.
# Threading the projection was worth 2.1x on its own (B 6.013 -> B' 3.956 at r = 272) and is what
# turned A's apparent long-run win into a loss -- the unthreaded projection was the only serial
# stretch left in the whole leaf multiply, and it grows linearly in `r`.
#
# BREAK-EVEN IS r BETWEEN 16 AND 32, and the bracket is the honest form: across two runs r = 16 read
# 1.17x and 1.54x of baseline while r = 32 read 0.38x and 0.41x, so the crossing is real but its
# location is not pinned to better than that. Below it SHE LOSES to the sampled multiply -- 3.7x slower
# at r = 8, 9-11x at r = 1 -- because there is nothing to amortise `S` against. That is the number the
# wire-in must respect on thin levels of a non-uniform tree, where chunks that small actually occur.
#
# THREAD THE PROJECTION ONLY ABOVE THE THRESHOLD. At r = 1 the threaded projection costs 0.824 ms
# against the unthreaded 0.122 ms -- pure spawn overhead. Same branch, same threshold.
#
# WITH BLAS SERIAL the ranking is the same but flatter (B best, B' 1.48x behind it on spawn overhead,
# A 0.84x of baseline at best). Nothing changes the verdict.
#
# THE BASELINE'S BLOWUP ABOVE r = 32 IS THE PHENOMENON, NOT NOISE. At r = 272 the sampled blocks total
# 1.2 GB; they fall out of every cache and the kernel goes at memory speed. That is precisely the cost
# SHE removes, and it is why the advantage GROWS with run length rather than saturating.
#
# CONSEQUENCE FOR THE LAYOUT WORK: because B writes to contiguous SCRATCH and only the cheap streaming
# projection touches the arena, ARENA FRAGMENTATION DOES NOT MATTER. The measured run lengths above
# gate A, and A is dead. The batching unit under B is the CHUNK, not the contiguous run -- so
# reordering nodes within a level to make leaves contiguous (which existed only to serve A) is
# unnecessary, and H2Factory's moment layout needs no change at all.

using MLFMA
using MLFMA.SphericalHarmonics
using MLFMA.SphericalHarmonics: SphericalHarmonicTransform, numcoefficients
using LinearAlgebra, StaticArrays, Printf, Random

# --- the operators --------------------------------------------------------------------------------

# θ̂ and φ̂ at every direction of `sampling`, as three columns each -- the same frame
# `StoreCartesianAsTangentialSpherical` projects onto (`assembly/store/store.jl`), built here directly so
# the benchmark carries no dependency on the storer's cache.
function sphericalframe(sampling)
    k̂, _ = MLFMA.sphericalsamplepoints(sampling)
    θs = [p[2] for p in vec(k̂)]
    φs = [p[3] for p in vec(k̂)]
    θ̂ = hcat(cos.(θs) .* cos.(φs), cos.(θs) .* sin.(φs), -sin.(θs))
    φ̂ = hcat(-sin.(φs), cos.(φs), zeros(length(φs)))
    return θ̂, φ̂
end

# `S_component[q, (c-1)*nc + lm] = S[q, lm] * frame[q, c]`, i.e. the synthesis matrix replicated once
# per Cartesian component and scaled by that component's frame entry. The column order is
# COMPONENT-SLOWEST / lm-FASTEST, matching `SphericalHarmonicStorage`'s coefficient layout exactly, so
# a group's `3*(P+1)²` coefficient vector feeds this with no permutation.
function fusedoperator(S, frame)
    nc = size(S, 2)
    F = Matrix{eltype(S)}(undef, size(S, 1), 3nc)
    for c in 1:3
        @views F[:, ((c - 1) * nc + 1):(c * nc)] .= frame[:, c] .* S
    end
    return F
end

# --- the two write strategies ---------------------------------------------------------------------

# A. Two GEMMs straight into the arena. `arena` is `(NΩ, 2r)` with group `g` occupying columns
# `2g-1` (θ) and `2g` (φ) -- the interleaving `wrapbuffer` produces. The strided views below have
# column stride `2NΩ`, which is a plain `ldc` to BLAS.
function writefused!(arena, Sθf, Sφf, A)
    r = size(A, 2)
    @views mul!(arena[:, 1:2:(2r)], Sθf, A)
    @views mul!(arena[:, 2:2:(2r)], Sφf, A)
    return arena
end

# B. One GEMM into contiguous scratch, then project. `reshape(A, nc, 3r)` is FREE: `A` is `(3nc x r)`
# column-major with lm fastest, so reshaping gives columns `g1c1, g1c2, g1c3, g2c1, ...` -- exactly
# the component-per-column form `S` consumes.
function writescratch!(arena, S, scratch, A, θ̂, φ̂, nc)
    r = size(A, 2)
    mul!(scratch, S, reshape(A, nc, 3r))
    @inbounds for g in 1:r
        _project!(arena, scratch, θ̂, φ̂, g)
    end
    return arena
end

# B'. The same, with the projection threaded over groups. The GEMM above is BLAS-threaded but this
# pass is plain Julia, so on the serial-traversal/threaded-BLAS configuration MLFMA actually schedules
# (`nchunks = 1`, one task calling a multi-threaded BLAS) an unthreaded projection is the only serial
# stretch in the whole leaf multiply -- and it grows linearly in `r`, which is exactly where B was
# losing to A. Groups write disjoint arena columns, so this needs no synchronisation.
function writescratchthreaded!(arena, S, scratch, A, θ̂, φ̂, nc)
    r = size(A, 2)
    mul!(scratch, S, reshape(A, nc, 3r))
    Threads.@threads for g in 1:r
        _project!(arena, scratch, θ̂, φ̂, g)
    end
    return arena
end

@inline function _project!(arena, scratch, θ̂, φ̂, g)
    @inbounds for q in axes(arena, 1)
        x = scratch[q, 3g - 2]
        y = scratch[q, 3g - 1]
        z = scratch[q, 3g]
        arena[q, 2g - 1] = θ̂[q, 1] * x + θ̂[q, 2] * y + θ̂[q, 3] * z
        arena[q, 2g] = φ̂[q, 1] * x + φ̂[q, 2] * y + φ̂[q, 3] * z
    end
    return nothing
end

# BASELINE: today's path. One GEMV per tangential component per group, against that group's own
# `(NΩ x N_j)` sampled pattern block.
function writesampled!(arena, blocks, xs)
    for (g, (block, x)) in enumerate(zip(blocks, xs))
        @views mul!(arena[:, 2g - 1], block[1], x)
        @views mul!(arena[:, 2g], block[2], x)
    end
    return arena
end

# --- harness --------------------------------------------------------------------------------------

function best(f; samples=5)
    f()
    t = Inf
    for _ in 1:samples
        t = min(t, @elapsed f())
    end
    return t
end

function runsweep(;
    P=9,
    Nj=92,
    runs=(1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 272),
    ϵ=1e-4,
    frequency=5e7,
    leaffactor=4,
)
    μ0 = 4π * 1e-7
    ε0 = 8.854187812e-12
    c = 1 / sqrt(ε0 * μ0)
    λ = c / frequency
    γ = im * 2π / λ
    sampling, _ = MLFMA.sampling(
        γ, ϵ, MLFMA.diameter(λ / leaffactor, 3), MLFMA.SphericalSamplingBuilder{Float64}()
    )

    t = SphericalHarmonicTransform(ComplexF64, sampling, P)
    S = t.synthesis
    NΩ, nc = size(S)
    θ̂, φ̂ = sphericalframe(sampling)
    Sθf = fusedoperator(S, θ̂)
    Sφf = fusedoperator(S, φ̂)
    θ̂c = ComplexF64.(θ̂)
    φ̂c = ComplexF64.(φ̂)

    @printf("NΩ = %d   P = %d   (P+1)² = %d   N_j = %d\n", NΩ, P, nc, Nj)
    @printf(
        "operators: S %.1f MB   fused S_θ+S_φ %.1f MB   sampled block per group %.1f MB\n\n",
        sizeof(S) / 2^20,
        (sizeof(Sθf) + sizeof(Sφf)) / 2^20,
        2 * NΩ * Nj * 16 / 2^20
    )

    println(
        "     r    A fused     B scratch   B' threaded    baseline    A vs base  B' vs base  A vs B'",
    )
    for r in runs
        rng = MersenneTwister(r)
        A = randn(rng, ComplexF64, 3nc, r)
        arena = Matrix{ComplexF64}(undef, NΩ, 2r)
        scratch = Matrix{ComplexF64}(undef, NΩ, 3r)
        blocks = [
            (randn(rng, ComplexF64, NΩ, Nj), randn(rng, ComplexF64, NΩ, Nj)) for _ in 1:r
        ]
        xs = [randn(rng, ComplexF64, Nj) for _ in 1:r]

        # the two SHE paths must agree -- otherwise the timings compare two different computations
        writefused!(arena, Sθf, Sφf, A)
        fused = copy(arena)
        writescratch!(arena, S, scratch, A, θ̂c, φ̂c, nc)
        agree = norm(fused - arena) / norm(fused)
        agree < 1e-12 || error("A and B disagree at r=$r: $agree")
        fill!(arena, 0)
        writescratchthreaded!(arena, S, scratch, A, θ̂c, φ̂c, nc)
        agree = norm(fused - arena) / norm(fused)
        agree < 1e-12 || error("A and B' disagree at r=$r: $agree")

        ta = best(() -> writefused!(arena, Sθf, Sφf, A))
        tb = best(() -> writescratch!(arena, S, scratch, A, θ̂c, φ̂c, nc))
        tbt = best(() -> writescratchthreaded!(arena, S, scratch, A, θ̂c, φ̂c, nc))
        tc = best(() -> writesampled!(arena, blocks, xs))

        @printf(
            "%6d  %9.3f ms  %9.3f ms  %9.3f ms  %9.3f ms  %8.2fx  %8.2fx  %7.2fx\n",
            r,
            1e3ta,
            1e3tb,
            1e3tbt,
            1e3tc,
            ta / tc,
            tbt / tc,
            ta / tbt
        )
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    nthreads = BLAS.get_num_threads()
    println("=== BLAS threads = 1 ===")
    BLAS.set_num_threads(1)
    runsweep()
    println("\n=== BLAS threads = ", nthreads, " ===")
    BLAS.set_num_threads(nthreads)
    runsweep()
end
