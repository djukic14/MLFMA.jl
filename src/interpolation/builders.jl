# Builders select an angular interpolator family, its order, and its element type.
# They construct the existing parameterized operators; order remains a type parameter used in
# dispatch. `SphericalLagrangeInterpolator{15,ComplexF64}`, for example, has stencil size 15.
# `LocalθGlobalφInterpolationBuilder` names its local order explicitly as `θorder`.
#
# Reverse transfers need both the adjoint and the plain transpose. `buildreverseoperators` returns
# both, using the same object only when the interpolation operator is real.
#
# `LocalθGlobalφ` exposes only `θorder`: its global FFT φ step has no stencil order to tune.
# The Lagrange builder uses one order for both axes. To study unequal orders, pass separate
# interpolators and an intermediate grid to `ID.interpolationerror(interpolators, samplings, cases)`.

"""
    AbstractInterpolationBuilder{order,T}

Construction policy for one angular interpolator family: which family, at what order, over what element
type.

`order` is a type parameter, so [`buildinterpolator`](@ref) resolves to the same concrete types as
direct construction. Keyword constructors convert the requested order into that parameter during
construction, not during an apply.
"""
abstract type AbstractInterpolationBuilder{order,T} end

"""
    interpolationorder(builder)

The builder's stencil/interpolation order, i.e. its `order` type parameter.
"""
interpolationorder(::AbstractInterpolationBuilder{order}) where {order} = order

"""
    elementtype(builder)

The element type the built operators carry.
"""
elementtype(::AbstractInterpolationBuilder{order,T}) where {order,T} = T

"""
    SphericalLagrangeInterpolationBuilder(; order, T=ComplexF64)

Build the barycentric-Lagrange family: local interpolation in both θ and φ.

The same `order` applies to θ and φ. Its operator is exactly real, so its anterpolator serves both reverse directions and
[`buildreverseoperators`](@ref) returns the same object twice.
"""
struct SphericalLagrangeInterpolationBuilder{order,T} <:
       AbstractInterpolationBuilder{order,T} end

function SphericalLagrangeInterpolationBuilder(; order::Integer, T::DataType=ComplexF64)
    return SphericalLagrangeInterpolationBuilder{Int(order),T}()
end

"""
    LocalθGlobalφInterpolationBuilder(; θorder, T=ComplexF64)

Build the local-θ, global-φ family: local interpolation in θ, an FFT in φ.

`θorder` controls the local θ stencil. The φ step is global and has no stencil order; it evaluates
the Fourier interpolant to round-off for resolved content away from the Nyquist ambiguity.

Its operator is COMPLEX, so its anterpolator (adjoint interpolation) and its transposed interpolation are
different operators; [`buildreverseoperators`](@ref) returns both.
"""
struct LocalθGlobalφInterpolationBuilder{θorder,T} <: AbstractInterpolationBuilder{θorder,T} end

function LocalθGlobalφInterpolationBuilder(; θorder::Integer, T::DataType=ComplexF64)
    return LocalθGlobalφInterpolationBuilder{Int(θorder),T}()
end

"""
    CircularInterpolationBuilder(; T=ComplexF64)

Build the circular family: spectral resampling between two `CircularSampling` grids, for 2-D problems.

NO ORDER KEYWORD, and that is a property of the method rather than an omission. The step is a global
spectral resample, exact for anything the source grid resolves, so there is no stencil whose size
trades accuracy against cost -- its `order` parameter is fixed at `0`, MLFMA's spelling for "global:
every source sample contributes".

Its operator is REAL (see `MLFMA.isrealinterpolation`), so its anterpolator serves both reverse
directions and [`buildreverseoperators`](@ref) returns the same object twice.
"""
struct CircularInterpolationBuilder{T} <: AbstractInterpolationBuilder{0,T} end

CircularInterpolationBuilder(; T::DataType=ComplexF64) = CircularInterpolationBuilder{T}()

"""
    interpolatortype(builder)

The concrete interpolator type `builder` resolves to, partially parameterized by order and element type.

The remaining parameter, the sampling's element type, is inferred from the samplings at construction --
which is why this returns a type rather than an instance, and why it is directly usable as
`LevelTransferStrategy`'s `Interpolator` field.
"""
interpolatortype(::SphericalLagrangeInterpolationBuilder{order,T}) where {order,T} =
    Interpolation.SphericalLagrangeInterpolator{order,T}
interpolatortype(::LocalθGlobalφInterpolationBuilder{order,T}) where {order,T} =
    Interpolation.LocalθGlobalφInterpolator{order,T}
interpolatortype(::CircularInterpolationBuilder{T}) where {T} =
    Interpolation.CircularInterpolator{0,T}

"""
    buildinterpolator(builder, samplings, parentsamplings; interpolationtrait=DefaultInterpolation())

The forward interpolator for `builder`, with its interpolation information built for each
`samplings[i] -> parentsamplings[i]` level pair.

Passing the pairs at construction is what populates the information cache once, rather than on the first
apply of every level -- the same thing the map constructors do today, spelled through the builder.
"""
function buildinterpolator(
    builder::AbstractInterpolationBuilder,
    samplings::AbstractVector,
    parentsamplings::AbstractVector;
    interpolationtrait=DefaultInterpolation(),
)
    return interpolatortype(builder)(
        samplings, parentsamplings; interpolationtrait=interpolationtrait
    )
end

"""
    buildanterpolator(builder, interpolator; nchunks=Threads.nthreads())

The ADJOINT interpolation operator for `interpolator`.

`nchunks` is accepted for compatibility only. FFT planning uses representative matrices;
per-chunk apply buffers belong to workspace scratch, not operator construction.
"""
function buildanterpolator(
    ::AbstractInterpolationBuilder, interpolator; nchunks::Int=Threads.nthreads()
)
    return anterpolator(interpolator; nchunks=nchunks)
end

"""
    buildreverseoperators(builder, interpolator; nchunks=Threads.nthreads())

Both reverse-direction operators as a `NamedTuple`: `(anterpolator, transposedinterpolator)`.

`anterpolator` is adjoint interpolation; `transposedinterpolator` is plain transposed interpolation.
They coincide only for a real operator, in which case both entries are the same object.
"""
function buildreverseoperators(
    builder::AbstractInterpolationBuilder, interpolator; nchunks::Int=Threads.nthreads()
)
    anter = buildanterpolator(builder, interpolator; nchunks=nchunks)
    return (anterpolator=anter, transposedinterpolator=transposedinterpolator(anter))
end
