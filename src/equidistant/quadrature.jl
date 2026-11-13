# Spectrally exact spherical quadrature on the equidistant midpoint grid.
#
# `quadratureweights` uses the midpoint rule, `Δθ Δφ sin θ`.
# On a periodic band-limited integrand the trapezoidal/midpoint rule is spectrally accurate, so it is
# tempting to assume that carries over here -- but the sphere's Jacobian breaks it. Extending the θ
# integral to the full torus turns `sin θ` into `|sin θ|`, which has a KINK at θ = 0 and θ = π. A
# non-smooth factor reduces convergence to algebraic O(h²). Gauss-Legendre weights account for
# `sin θ`; this grid instead integrates that factor through its Fourier coefficients.
#
# `|sin θ|` is not band-limited, but its Fourier series is known in closed form:
#
#     |sin θ| = 2/π - (4/π) Σ_{k≥1} cos(2kθ) / (4k² - 1)
#
# so for a band-limited F the integral ∫F|sin θ|dθ = 2π Σ_m F̂_m ĉ_{-m} is a FINITE exact sum. Folding
# the DFT that produces F̂ back into node space gives per-node weights that integrate every resolved
# mode exactly -- the non-smooth part is handled analytically instead of being sampled.
#
# Exactness applies to resolved modes, not to arbitrary unresolved integrands.

"""
    exactquadratureweights(sampling) -> Matrix

Weights `W` such that `sum(W .* F)` equals `∮ F dΩ` EXACTLY for every `F` band-limited to the grid,
where `F` is stored on the physical grid with φ as the first axis and θ as the second.

Exact for θ-modes up to `Nθ - 1` and all φ-modes the grid resolves. Contrast
[`quadratureweights`](@ref), which is the plain midpoint rule and only O(h²) -- see this file's
header for why the sphere's Jacobian costs the midpoint rule its spectral accuracy.
"""
function exactquadratureweights(s::EquidistantSampling{T}) where {T}
    nθ = s.Nθ
    # Only EVEN cosine modes appear in |sin θ|, so the sum runs over k with 2k <= Nθ - 1.
    kmax = (nθ - 1) ÷ 2
    θweights = map(s.θs) do θ
        acc = T(2)
        for k in 1:kmax
            acc -= T(4) * cos(2 * k * θ) / T(4 * k^2 - 1)
        end
        return acc / nθ
    end
    # The φ rule is the plain trapezoidal rule, which IS spectrally exact: φ is genuinely periodic
    # with no Jacobian factor, so there is nothing here to spoil it.
    Δφ = 2 * T(π) / s.Nφ
    return [w * Δφ for _ in 1:(s.Nφ), w in θweights]
end

"""
    integrate(F, sampling) -> Number

`∮ F dΩ` using [`exactquadratureweights`](@ref).
"""
integrate(F::AbstractMatrix, s::EquidistantSampling) = sum(exactquadratureweights(s) .* F)
