using MLFMA, PlotlyJS, Test

include("check-translation-accuracy.jl")

const I = MLFMA.Interpolation
const SH = MLFMA.SphericalHarmonics

@testset "Browser figures against Julia" begin
    fixtures = Dict{String,Any}()
    fixtures["sampling"] = map(((0.1, 2), (0.25, 4), (0.5, 6))) do (width, digits)
        s, L = MLFMA.sampling(2π * im, 10.0^(-digits), sqrt(3) * width)
        (; width, digits, L, theta=MLFMA.θs(s), nt=MLFMA.Nθ(s), np=MLFMA.Nφ(s))
    end
    fixtures["transfers"] = map((
        (16, 64, 3, 6), (12, 36, 8, 5), (20, 40, 2, 4)
    )) do (n, target, mode, stencil)
        child = MLFMA.SphericalSampling{Float64}(10, n)
        parent = MLFMA.SphericalSampling{Float64}(10, target)
        field = MLFMA.ScalarFarField{1,ComplexF64}(size(child))
        field[1] .= cos.(mode .* MLFMA.φs(child))
        values = Dict{String,Any}()
        for (family, constructor) in (
            ("Local Lagrange", MLFMA.SphericalLagrangeInterpolator),
            ("Fourier", MLFMA.LocalθGlobalφInterpolator),
        )
            interpolator = constructor{stencil}(ComplexF64, child, parent)
            result = MLFMA.ScalarFarField{1,ComplexF64}(size(parent))
            interpolator(result, field, child, parent)
            values[family] = real.(result[1][:, 5])
        end
        (; n, target, mode, stencil, values)
    end
    fixtures["harmonics"] = map(((3.0, 2), (8.0, 6), (12.0, 18))) do (z, P)
        theta, phi, w = SH.gausslegendregrid(max(32, P + 8))
        transform = SH.SphericalHarmonicTransform(ComplexF64, theta, phi, w, P)
        coefficients = SH.analyze(transform, exp.(-im .* z .* cos.(theta)))
        axisymmetric = [coefficients[SH.lmindex(l, 0)] * sqrt((2l + 1)/(4π)) for l in 0:P]
        (; z, P, real=real.(axisymmetric), imag=imag.(axisymmetric))
    end
    phi = collect(range(0.0, 2π; length=33))
    V = MLFMA.Sampling.SVector
    directions = [V(cos(p), sin(p), 0.0) for p in phi]
    fixtures["phases"] = map((-0.25, 0.0, 0.5)) do delta
        values = MLFMA.phasefactor(
            2π * im, directions, V(delta, 0.0, 0.0), MLFMA.AbstractRadiationFieldCollection
        )
        (; delta, phi, real=real.(values), imag=imag.(values))
    end
    # Check the inverse symmetry convention with the actual translation kernel.
    rotated = [V(v[2], -v[1], 0.0) for v in directions]
    a = zeros(ComplexF64, length(phi))
    b = similar(a)
    MLFMA.translationoperator!(a, 2π * im, directions, V(0.0, 1.0, 0.0), 8)
    MLFMA.translationoperator!(b, 2π * im, rotated, V(1.0, 0.0, 0.0), 8)
    @test a ≈ b rtol=1e-12
    mktemp() do path, io
        PlotlyJS.JSON.print(io, fixtures)
        close(io)
        run(`node $(joinpath(@__DIR__, "check-explorers.cjs")) $path`)
        @test true
    end
end
