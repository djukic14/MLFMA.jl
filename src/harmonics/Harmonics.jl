"""
    MLFMA.Harmonics

Orthonormal spherical and circular harmonics, their transforms and storage helpers.

The submodules are available as `MLFMA.SphericalHarmonics` and
`MLFMA.CircularHarmonics`, but are not exported from `MLFMA`.
"""
module Harmonics

# Spherical and vector harmonics compress stored leaf patterns.
include("../sphericalharmonics/SphericalHarmonics.jl")

# Circular harmonics `exp(imφ)/√(2π)` serve the 2-D sampling grids.
include("../circularharmonics/CircularHarmonics.jl")

export CircularHarmonics, SphericalHarmonics

end
