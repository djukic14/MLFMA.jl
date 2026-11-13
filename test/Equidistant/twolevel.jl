module EquidistantTwoLevelTests

# END-TO-END: a real two-level MLFMA kernel on the equidistant grid, against the Green function.
#
# Every other testset here checks one piece in isolation. This one runs the whole chain the way the
# algorithm actually uses it -- aggregate on child grids, interpolate to the parent rate, phase-shift
# to the parent centre, translate, phase-shift to the observer child, anterpolate down, integrate --
# and compares against e^{ikR}/R summed over source/observer pairs. It is the check that says the
# representation is usable for MLFMA at all, as opposed to merely self-consistent.
#
# WHAT THE NUMBERS MEAN. The residual is entirely GEGENBAUER TRUNCATION, not the angular
# representation: doubling both grids leaves the error unchanged (1.77e-6 at L=8 either way), while
# raising L moves it four orders. The error traces the classic MLFMA U-curve -- 2.8e-3 at L=4 down to
# 2.7e-10 at L=12, then back up to 2.3e-4 at L=18 as h_l(kd) round-off overwhelms the series. That
# rise is correct behaviour, which is why the assertions below bracket a WINDOW in L rather than
# demanding monotone improvement.

using Test
using MLFMA
using LinearAlgebra
using Random
using MLFMA.Equidistant:
    EquidistantSampling,
    SPHERICAL,
    Nθ,
    Nφ,
    directions,
    resample,
    anterpolate,
    integrate,
    translationoperator,
    FFTBackend,
    LagrangeBackend

norm3(v) = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
sub(a, b) = (a[1] - b[1], a[2] - b[2], a[3] - b[3])
dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

radiation(k, r, c, s) = (u=sub(r, c); [cis(-k * dot3(k̂, u)) for k̂ in directions(s)])
receivepattern(k, r, c, s) = (u=sub(r, c); [cis(k * dot3(k̂, u)) for k̂ in directions(s)])
shiftphase(k, δ, s, sgn) = [cis(sgn * k * dot3(k̂, δ)) for k̂ in directions(s)]
exactgreen(k, ro, rs) = (R=norm3(sub(ro, rs)); cis(k * R) / R)

# Anterpolation for a BILINEAR pairing is the TRANSPOSE, not the adjoint: the reciprocity integral
# ∮ u v dΩ carries no conjugation, so the adjoint's would be one conjugation too many. Rᵀ = conj∘Rᴴ∘conj
# is the same identity the package already uses for `transposedinterpolator`.
anterpT(H, src, dst, backend) = conj(anterpolate(conj(H), src, dst, SPHERICAL, backend))

"""
Run the two-level kernel and return (relative L2 error vs the Green function, worst
anterpolation-vs-direct discrepancy).
"""
function twolevel(L, child, parent, backend; k=1.0)
    cPS, cPO = (0.0, 0.0, 0.0), (0.0, 0.0, 12.0)
    schild = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    ochild = [(-1.0, 0.0, 12.0), (1.0, 0.0, 12.0)]
    rng = MersenneTwister(7)
    a = 0.5
    inbox(c) = (
        c[1] + a * (2rand(rng) - 1),
        c[2] + a * (2rand(rng) - 1),
        c[3] + a * (2rand(rng) - 1),
    )
    sources = [(inbox(c), randn(rng, ComplexF64)) for c in schild for _ in 1:2]
    srcof = [ci for ci in 1:2 for _ in 1:2]
    observers = [inbox(c) for c in ochild for _ in 1:2]
    obsof = [ci for ci in 1:2 for _ in 1:2]

    d = sub(cPO, cPS)
    T = translationoperator(L, k * norm3(d), d ./ norm3(d), parent)

    FP = zeros(ComplexF64, Nφ(parent), Nθ(parent))
    for ci in 1:2
        Fc = zeros(ComplexF64, Nφ(child), Nθ(child))
        for (j, (r, q)) in enumerate(sources)
            srcof[j] == ci || continue
            Fc .+= q .* radiation(k, r, schild[ci], child)
        end
        FP .+=
            resample(Fc, child, parent, SPHERICAL, backend) .*
            shiftphase(k, sub(schild[ci], cPS), parent, -1)
    end

    GP = FP .* T
    got = zeros(ComplexF64, length(observers))
    gap = 0.0
    for ci in 1:2
        Hc = GP .* shiftphase(k, sub(ochild[ci], cPO), parent, +1)
        Gdown = anterpT(Hc, parent, child, backend)
        for (m, ro) in enumerate(observers)
            obsof[m] == ci || continue
            got[m] =
                (im * k / (4π)) *
                integrate(Gdown .* receivepattern(k, ro, ochild[ci], child), child)
            # The same quantity without ever leaving the parent grid. Anterpolation is the ONLY
            # difference between the two, so any gap is anterpolation's alone.
            onparent =
                (im * k / (4π)) *
                integrate(Hc .* receivepattern(k, ro, ochild[ci], parent), parent)
            gap = max(gap, abs(got[m] - onparent) / abs(onparent))
        end
    end
    want = [sum(q * exactgreen(k, ro, r) for (r, q) in sources) for ro in observers]
    return norm(got - want) / norm(want), gap
end

const CHILD = EquidistantSampling{Float64}(12, 24)
const PARENT = EquidistantSampling{Float64}(24, 48)

@testset "two-level MLFMA reproduces the Green function" begin
    err, _ = twolevel(12, CHILD, PARENT, FFTBackend())
    @test err < 1e-8
end

@testset "anterpolation is exactly consistent with staying on the parent grid" begin
    # Structural, not statistical: the weighted transpose must reproduce the parent-grid integral to
    # machine precision, because that is the identity it is defined by.
    _, gap = twolevel(12, CHILD, PARENT, FFTBackend())
    @test gap < 1e-12
end

@testset "accuracy is truncation-limited, not grid-limited" begin
    # Doubling both grids must NOT help -- if it does, the representation is the bottleneck.
    bigchild = EquidistantSampling{Float64}(24, 48)
    bigparent = EquidistantSampling{Float64}(48, 96)
    small, _ = twolevel(8, CHILD, PARENT, FFTBackend())
    big, _ = twolevel(8, bigchild, bigparent, FFTBackend())
    @test isapprox(small, big; rtol=0.05)
    # ...while raising L does help, by orders.
    better, _ = twolevel(12, CHILD, PARENT, FFTBackend())
    @test better < small / 100
end

@testset "the TRANSLATION grid needs the quadrature rate; the interpolated-from grid does not" begin
    # A DESIGN CONSTRAINT, measured rather than assumed, and it decides what a sampling strategy may do.
    #
    # Interpolation only has to REPRESENT one harmonic, so the equidistant FFT backend is exact at
    # Nθ = L+1 -- which invites the conclusion that every grid could run at that rate. It cannot.
    # Translation is a POINTWISE PRODUCT with `T_L`, and a product has twice the bandwidth of its
    # factors, so the grid the product lands on needs ~2(L+1). That is exactly what `NθNφ` already
    # returns, and this pins WHY.
    #
    # The two directions are separated deliberately: varying both grids together (the obvious sweep)
    # cannot tell which one is binding, and the answer is that only one of them is.
    L = 12
    quadrature = MLFMA.NθNφ(L)[1]        # 26
    fixedchild = EquidistantSampling{Float64}(26, 52)

    # PARENT is where the product lands. At quadrature grade the loop is truncation-limited; below it
    # the error grows by orders, and at representation grade (Nθ ≈ L+1) it is useless.
    atgrade, _ = twolevel(
        L, fixedchild, EquidistantSampling{Float64}(quadrature, 2quadrature), FFTBackend()
    )
    starved, _ = twolevel(L, fixedchild, EquidistantSampling{Float64}(13, 26), FFTBackend())
    @test atgrade < 1e-9
    @test starved > 1e-3                 # measured 1.76e-2 -- a cliff, not a slope

    # CHILD is only interpolated FROM, so it may sit far below the quadrature rate at no cost. This is
    # the half that IS free, and the reason the two must be measured separately.
    fixedparent = EquidistantSampling{Float64}(52, 104)
    rich, _ = twolevel(L, EquidistantSampling{Float64}(26, 52), fixedparent, FFTBackend())
    lean, _ = twolevel(L, EquidistantSampling{Float64}(13, 26), fixedparent, FFTBackend())
    @test lean < 1e-9
    @test isapprox(lean, rich; rtol=1e-3)
end

@testset "the local Lagrange backend also closes the loop" begin
    # Needs oversampling, as everywhere else, so the grids are larger and the stencil is high order.
    # Sized where it reaches 4.1e-9; 24×48/48×96 would buy 3.2e-10. Both are cheap now that
    # `anterpolate` runs the transposed kernel rather than materializing the operator -- this
    # configuration is kept because the accuracy is what the testset is about, not the cost.
    lagchild = EquidistantSampling{Float64}(16, 32)
    lagparent = EquidistantSampling{Float64}(32, 64)
    err, _ = twolevel(12, lagchild, lagparent, LagrangeBackend(10))
    @test err < 1e-8
end

end # module
