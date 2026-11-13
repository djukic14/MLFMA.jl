# Should the equidistant FFT family replace LocalθGlobalφ as INTERPOLATORECOM?
#
# WHY THE OBVIOUS STUDY IS THE WRONG ONE. Group 12 measured 4.7-6.5x in the equidistant family's
# favour AT EQUAL TOLERANCE -- but that comparison let each family pick its own grid, and the win came
# entirely from the spectral family needing a much coarser one (Nθ 16/20/26 against GL's 34/46/70).
# In production nobody picks a grid: `SamplingStrategy(; ϵ)` maps ϵ -> L -> (2(L+1), 4(L+1)), and that
# map is FAMILY-INDEPENDENT. So at the same ϵ both families get the SAME grid, and at equal grid the
# spectral family is 1.7-2.5x SLOWER (group 11). The 4.7-6.5x is only realisable if the sampling rule
# can be lowered for the spectral family specifically.
#
# AND ON A REAL MATVEC IT MAY NOT MATTER AT ALL: test/Equidistant/pipeline.jl records the matvec error
# as 9.5e-3 / 7.2e-3 / 6.9e-3 / 7.2e-3 at ϵ = 1e-2 / 1e-3 / 1e-5 / 1e-7 -- FLAT. The end-to-end error
# is floored by the near field and the mesh, not by the angular representation, so tightening ϵ buys
# nothing and production would run at the coarsest ϵ that reaches the floor.
#
# Which makes the production question small and concrete, and it is the one measured here:
# AT THE ϵ ONE WOULD ACTUALLY RUN, WHICH FAMILY'S MATVEC IS FASTER, END TO END?
# Interpolation is one stage among near-field, aggregation, translation and disaggregation, so a
# 1.7-2.5x on that stage is diluted by an unknown factor -- which is exactly why it has to be measured
# on the whole matvec rather than inferred from group 11.

using LinearAlgebra, Printf, Random
using MLFMA, BEAST, CompScienceMeshes, H2Trees
using MLFMA.Equidistant:
    EquidistantSamplingBuilder, FFTBackend, LagrangeBackend, EquidistantInterpolator

const EQB = EquidistantSamplingBuilder{Float64}()
mesh = CompScienceMeshes.read_gmsh_mesh(
    joinpath(pkgdir(MLFMA), "test", "assets", "tworects.msh")
)
const X = raviartthomas(mesh)
const EFIE = MLFMA.efie(; wavenumber=2 * pi)
_, rootsize = H2Trees.boundingbox(BEAST.positions(X))
const TREE = TwoNTree(
    X;
    builder=TwoNTreeBuilder(;
        minhalfsize=rootsize / 5, minvalues=0, protrusion=H2Trees.NoProtrusionCheck()
    ),
)
const DENSE = BEAST.assemble(EFIE, X, X)
Random.seed!(11)
const XVEC = randn(ComplexF64, numfunctions(X))
const WANT = DENSE * XVEC
const PHASESHIFTER = MLFMA.phaseshiftertype(MLFMA.MLFMAStrategy(EFIE, X))

gausslegendre(ϵ) =
    MLFMA.MLFMAStrategy(EFIE, X; samplingstrategy=MLFMA.SamplingStrategy(; ϵ=ϵ))

function equidistant(backend, ϵ)
    IT = typeof(EquidistantInterpolator{ComplexF64}(backend))
    return MLFMA.MLFMAStrategy(
        EFIE,
        X;
        samplingstrategy=MLFMA.SamplingStrategy(; samplingbuilder=EQB, ϵ=ϵ),
        leveltransferstrategy=MLFMA.LevelTransferStrategy(IT, PHASESHIFTER),
    )
end

# Hand-rolled min-of-N rather than BenchmarkTools: the `benchmark` environment resolves a REGISTRY
# BEAST whose `blockassembler` has no `primer` keyword, so this must run in `test`, which does not
# carry BenchmarkTools. Matvecs here are milliseconds, so min-of-N after a warm-up is adequate.
function besttime(f, n=7)
    f()
    return minimum(@elapsed(f()) for _ in 1:n)
end

function study(label, strategy)
    tbuild = @elapsed A = MLFMA.GalerkinMLFMA(EFIE, X; tree=TREE, strategy=strategy)
    y = A * XVEC
    err = norm(y - WANT) / norm(WANT)
    tmv = besttime(() -> A * XVEC)
    return (label=label, build=tbuild, mv=tmv, err=err)
end

println("End-to-end matvec on the tworects fixture (κ=2π, minhalfsize=rootsize/5).")
println(
    "The dense reference floors the error near 7e-3 regardless of ϵ -- shown so that is visible,",
)
println("not because it discriminates.\n")
@printf("%-6s %-22s %10s %10s %12s\n", "ϵ", "family", "build s", "matvec ms", "rel err")

rows = []
for ϵ in (3e-1, 1e-1, 3e-2, 1e-2)
    for (label, s) in (
        ("Gauss-Legendre p=15", gausslegendre(ϵ)),
        ("equidistant FFT", equidistant(FFTBackend(), ϵ)),
    )
        r = study(label, s)
        push!(rows, (ϵ=ϵ, r...))
        @printf(
            "%-6.0e %-22s %10.2f %10.3f %12.4e\n", ϵ, r.label, r.build, r.mv * 1e3, r.err
        )
    end
end

println(
    "\nRatio to the Gauss-Legendre control at the same ϵ (>1 means the equidistant family is SLOWER):",
)
@printf("%-6s %-22s %10s %10s\n", "ϵ", "family", "build", "matvec")
for ϵ in unique(r.ϵ for r in rows)
    base = only(filter(r -> r.ϵ == ϵ && r.label == "Gauss-Legendre p=15", rows))
    for r in filter(r -> r.ϵ == ϵ && r.label != "Gauss-Legendre p=15", rows)
        @printf(
            "%-6.0e %-22s %10.2f %10.2f\n", ϵ, r.label, r.build / base.build, r.mv / base.mv
        )
    end
end
