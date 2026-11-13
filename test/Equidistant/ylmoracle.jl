# An INDEPENDENT oracle: the unnormalized complex spherical harmonic, evaluated directly from its
# recurrences. Nothing in `MLFMA.Equidistant` is used to produce it, which is the point -- the backends
# are checked against analytically evaluated values at the TARGET points, never against each other and
# never against a resampling of themselves.

"""
    ylm(l, m, θ, φ)

`P_l^m(cos θ) e^{imφ}` in the Condon-Shortley convention, with `sqrt(1-x²) = +|sin θ|`.
"""
function ylm(l::Int, m::Int, θ::Real, φ::Real)
    am = abs(m)
    x = cos(θ)
    pmm = 1.0
    if am > 0
        s = sqrt(max(0.0, 1 - x^2))
        for i in 1:am
            pmm *= -(2i - 1) * s
        end
    end
    p = if l == am
        pmm
    else
        pmm1 = x * (2am + 1) * pmm
        if l == am + 1
            pmm1
        else
            pll = 0.0
            for ll in (am + 2):l
                pll = ((2ll - 1) * x * pmm1 - (ll + am - 1) * pmm) / (ll - am)
                pmm = pmm1
                pmm1 = pll
            end
            pll
        end
    end
    v = p * cis(am * φ)
    return m < 0 ? conj(v) * (-1)^am : v
end

"""
    bandlimitedfield(L; seed) -> (θ, φ) -> value

A fixed pseudo-random superposition `Σ a_lm Y_lm` of degree at most `L`.
"""
function bandlimitedfield(L::Int; seed=0x1234)
    rng = MersenneTwister(seed)
    c = [(l, m, randn(rng, ComplexF64) / (l + 1)) for l in 0:L for m in (-l):l]
    return (θ, φ) -> sum(a * ylm(l, m, θ, φ) for (l, m, a) in c)
end

# Arrays are stored [φ, θ] -- MLFMA's convention; see the STORAGE ORDER note in sampling.jl.
sampleon(f, s) = [f(θ, φ) for φ in MLFMA.Equidistant.φs(s), θ in MLFMA.Equidistant.θs(s)]
relL2(got, exact) = norm(vec(got) - vec(exact)) / norm(vec(exact))
