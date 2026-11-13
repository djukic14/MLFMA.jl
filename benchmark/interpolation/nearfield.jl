# Is there anything left in the NEAR field?
#
# WHY LOOK. Every optimization this session has gone after the far path, but the near block is a
# large share of the actual matvec: #121 measured near 0.0365 s against far 0.0803 s at λ/4 leaves --
# about 30 % of the total, entirely unexamined.
#
# WHAT DECIDES IT. The near block is a sparse matvec, so the ceiling is memory bandwidth: a CSC SpMV
# reads, per nonzero, one 16-byte `ComplexF64` and one 8-byte row index, plus the source vector
# gathered through those indices. If it is already at bandwidth, the only lever left is FEWER
# NONZEROS -- which is tree shape, the same factor #120 and #121 already named, and not a kernel
# question. If it is far off bandwidth, the gather locality is the suspect and ordering is worth
# trying.
#
# THE ORDERING EXPERIMENT. MLFMA keys everything by the ORIGINAL mesh numbering: `H2Trees.values`
# returns indices into it, so the near matrix inherits whatever order the mesher produced. Near
# interactions are between geometric neighbours, which in tree order are contiguous and in mesh order
# need not be. Permuting rows and columns into leaf-traversal order costs one permutation per matvec
# (or none, if the whole solve is done permuted) and could turn a scattered gather into a blocked one.
# This measures the permuted matrix against the original, same nonzeros, same arithmetic.

using MLFMA
using BEAST, CompScienceMeshes, H2Trees, H2Factory
using LinearAlgebra
using Printf
using Random
using SparseArrays

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

# Bytes a CSC SpMV must move at minimum: the values, the row indices, the column pointers, the
# result, and the source vector read at least once. The gather may read the source many more times
# than once -- that is exactly the inefficiency under test -- so this is a LOWER bound on traffic and
# therefore an UPPER bound on the achieved bandwidth.
function csctraffic(S)
    n = size(S, 2)
    return nnz(S) * (sizeof(ComplexF64) + sizeof(Int)) +
           (n + 1) * sizeof(Int) +
           2 * n * sizeof(ComplexF64)
end

# The machine's achievable bandwidth, measured rather than looked up: a streaming triad over arrays
# far larger than last-level cache.
function streambandwidth(; n=1 << 24)
    a = zeros(Float64, n)
    b = rand(Float64, n)
    c = rand(Float64, n)
    t = best(function ()
        @inbounds @simd for i in eachindex(a)
            a[i] = b[i] + 3.0 * c[i]
        end
    end)
    return 3 * n * sizeof(Float64) / t / 1e9
end

# Leaf-traversal order: every leaf's basis functions listed contiguously, leaves in tree order.
function treeorder(tree, n)
    perm = Int[]
    sizehint!(perm, n)
    for leaf in sort(collect(HT.leaves(tree)))
        append!(perm, HT.values(tree, leaf))
    end
    length(perm) == n || error(
        "leaf values cover $(length(perm)) of $n unknowns; this fixture has functions outside the " *
        "leaves and the permutation would not be a permutation",
    )
    return perm
end

function run(; radius=2.0, meshfactor=8)
    p = physicalconstants(5e7)
    λ = p.λ
    X = raviartthomas(meshsphere(radius * λ, λ / meshfactor))
    efie = MLFMA.efie(; wavenumber=p.κ, alpha=(-im * p.ω * p.μ))
    quadstrat = BEAST.DoubleNumWiltonSauterQStrat(3, 3, 3, 3, 3, 3, 3, 3)
    tree = HT.TwoNTree(X; builder=HT.TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0))
    strategy = MLFMA.MLFMAStrategy(
        efie,
        X;
        samplingstrategy=MLFMA.SamplingStrategy(; ϵ=1e-4),
        nearinteractionsstrategy=MLFMA.SparseBEASTNearInteractionsStrategy(quadstrat),
    )
    A = MLFMA.GalerkinMLFMA(efie, X; tree=tree, strategy=strategy, verbose=false)

    n = numfunctions(X)
    S = SparseArrays.sparse(H2Factory.nearinteractions(A))
    Random.seed!(11)
    x = randn(ComplexF64, n)
    y = similar(x)

    bw = streambandwidth()
    @printf(
        "N = %d   sphere %.1fλ   mesh λ/%d   leaves λ/4   nnz = %d (%.1f per row)\n",
        n,
        radius,
        meshfactor,
        nnz(S),
        nnz(S) / n
    )
    @printf("measured stream bandwidth: %.1f GB/s\n\n", bw)

    tnear = best(() -> H2Factory.nearmul!(y, A, x))
    tfar = best(() -> H2Factory.farmul!(y, A, x))
    ttot = best(() -> LinearAlgebra.mul!(y, A, x))
    tspmv = best(() -> LinearAlgebra.mul!(y, S, x))

    println("  stage                 time      share of matvec")
    @printf("  near (H2Factory)   %8.4f s        %5.1f %%\n", tnear, 100tnear / ttot)
    @printf("  far                %8.4f s        %5.1f %%\n", tfar, 100tfar / ttot)
    @printf("  full matvec        %8.4f s\n\n", ttot)

    traffic = csctraffic(S)
    @printf(
        "  bare CSC SpMV      %8.4f s   -> %5.1f GB/s = %4.1f %% of stream\n",
        tspmv,
        traffic / tspmv / 1e9,
        100 * (traffic / tspmv / 1e9) / bw
    )

    # --- the ordering experiment -------------------------------------------------------------------
    perm = treeorder(tree, n)
    Sp = S[perm, perm]
    xp = x[perm]
    yp = similar(xp)
    tperm = best(() -> LinearAlgebra.mul!(yp, Sp, xp))
    @printf(
        "  tree-ordered SpMV  %8.4f s   -> %5.1f GB/s = %4.1f %% of stream   (%.2fx)\n",
        tperm,
        traffic / tperm / 1e9,
        100 * (traffic / tperm / 1e9) / bw,
        tspmv / tperm
    )

    # correctness of the permutation, so a speedup cannot come from computing the wrong thing
    @printf(
        "  permutation check: rel diff %.2e\n",
        norm((Sp * xp)[invperm(perm)] - S * x) / norm(S * x)
    )
    println()
    println(
        "If the bare SpMV is already near stream bandwidth, the only lever left is FEWER NONZEROS,\n" *
        "i.e. tree shape -- not the kernel and not the ordering.",
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
