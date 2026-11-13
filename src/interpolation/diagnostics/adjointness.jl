# Anterpolation diagnostics distinguish adjoint correctness from interpolation accuracy.
# Anterpolation spreads weights; it does not resample an exact function onto the coarse grid.
# Comparing its output with that function's coarse samples is therefore not an accuracy test.
#
# The residual `|<I f, g> - <f, A g>| / |<I f, g>|` checks whether A is the adjoint of I and should
# be near round-off when the denominator is nonzero. It does not measure approximation error.
# If I approximates exact resampling R, their adjoints have the same spectral-norm error.
#
# `adjointpairingerror` also measures interpolation error contracted against another field.
# This pairing error can differ from a field-norm error because contributions can cancel.
# The implementation uses unweighted Hermitian dot products, not spherical quadrature.

"""
    adjointpairingerror(interpolator, anterpolator, sourcesampling, targetsampling, cases;
                        farfieldtype=MLFMA.ScalarFarField, othercaseseed=0x5eed)

For each case, the error in an unweighted Hermitian pairing and the adjointness residual.

One `NamedTuple` per case, table-friendly like [`rows`](@ref):

  - `pairing` -- `|<f_fine, g> - <f_coarse, A g>| / |<f_fine, g>|`, where `f_fine` is the case evaluated
    exactly on the target grid and `f_coarse` on the source grid. The reference is exact resampling,
    not interpolation. Contracting against `g` can cancel contributions to the error.
  - `adjointness` -- `|<I f, g> - <f, A g>| / |<I f, g>|`. A correctness invariant, expected at round-off.
    Reported so that a `pairing` number is never read without knowing whether the operator it came from
    is still the adjoint at all.
  - `case`, `rho`, `bandwidth`, `casekind` -- as in [`rows`](@ref).

`g` is a scalar `RandomBandlimitedCase` at the case's bandwidth, drawn from `othercaseseed`.
Use scalar cases and `ScalarFarField`: the comparison field has no tangential counterpart here.
A full band avoids the systematic zero pairings of single harmonics with different φ orders.
The reported ratios divide by the pairing magnitudes without a zero-denominator guard.

# Arguments

  - `interpolator`, `anterpolator`: a matched pair, e.g. from `MLFMA.buildreverseoperators`
  - `sourcesampling`, `targetsampling`: coarse and fine grids, as for [`interpolationerror`](@ref)
  - `cases`: one case or any iterable of them
"""
function adjointpairingerror(
    interpolator,
    anterpolator,
    sourcesampling::Sampling.SphericalSampling,
    targetsampling::Sampling.SphericalSampling,
    cases;
    farfieldtype=FarFields.ScalarFarField,
    othercaseseed::Integer=0x5eed,
)
    _checkfarfieldtype(farfieldtype)
    T = eltype(sourcesampling)

    return map(collect(_ascases(cases))) do case
        coarse = farfieldtype(Complex{T}, sourcesampling, Val(:zero))
        fine = farfieldtype(Complex{T}, targetsampling, Val(:zero))
        fillexact!(coarse, case, sourcesampling)
        fillexact!(fine, case, targetsampling)

        other = RandomBandlimitedCase(;
            L=casebandwidth(case), seed=othercaseseed, decay=:none
        )
        g = farfieldtype(Complex{T}, targetsampling, Val(:zero))
        fillexact!(g, other, targetsampling)

        interpolated = interpolator(coarse, sourcesampling, targetsampling)
        anterpolated = anterpolator(g, targetsampling, sourcesampling)

        # `dot` conjugates its first argument. Sum over all components to test the Hermitian
        # adjoint relation; bilinear `sum(a .* b)` would instead test a transpose relation.
        pairing(a, b) = sum(LinearAlgebra.dot(a[i], b[i]) for i in eachindex(a))

        exactpairing = pairing(fine, g)
        computedpairing = pairing(coarse, anterpolated)
        interpolatedpairing = pairing(interpolated, g)

        return (
            case=case,
            casekind=casekind(case),
            bandwidth=casebandwidth(case),
            rho=rho(case, sourcesampling),
            pairing=abs(exactpairing - computedpairing) / abs(exactpairing),
            adjointness=abs(interpolatedpairing - computedpairing) /
                        abs(interpolatedpairing),
        )
    end
end
