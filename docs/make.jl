using MLFMA
using Documenter

include("figures.jl")

const FARFIELD_SCRATCH = Set([
    MLFMA.SphericalHarmonicLeafScratch,
    MLFMA.VectorSphericalHarmonicLeafScratch,
    MLFMA.CircularHarmonicLeafScratch,
])
function farfieldarea(obj)
    obj === MLFMA.FarFields.LinearAlgebra.mul! && return :mul
    obj === MLFMA.H2Factory.Interfaces.conjmul! && return :conjmul
    obj in FARFIELD_SCRATCH && return :collections
    obj isa Type && obj <: MLFMA.AbstractFieldCollection && return :collections
    return :fields
end

DocMeta.setdocmeta!(MLFMA, :DocTestSetup, :(using MLFMA); recursive=true)

# List every layer, including nested modules, so `checkdocs` covers the whole package.
# Each module also needs an `@autodocs` block in the API reference.
const MODULES = [
    MLFMA,
    MLFMA.Sampling,
    MLFMA.Operators,
    MLFMA.FarFields,
    MLFMA.Phase,
    MLFMA.Interpolation,
    MLFMA.Interpolation.InterpolationDiagnostics,
    MLFMA.Harmonics,
    MLFMA.Harmonics.SphericalHarmonics,
    MLFMA.Harmonics.CircularHarmonics,
    MLFMA.Translations,
    MLFMA.Assembly,
    MLFMA.Equidistant,
    MLFMA.Seams,
    MLFMA.TestSupport,
]

makedocs(;
    modules=MODULES,
    authors="Danijel Jukić <danijel.jukic14@gmail.com> and contributors",
    sitename="MLFMA.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://djukic14.github.io/MLFMA.jl",
        edit_link="main",
        assets=["assets/guide.css", "assets/generated/figures.js", "assets/explorers.js"],
    ),
    # Group the API reference by layer.
    pages=[
        "Home" => "index.md",
        "Using MLFMA" => [
            "Constructing a map" => "getting-started.md",
            "Choosing parameters" => "guide/parameters.md",
            "Supported operators" => "guide/operators.md",
        ],
        "Far fields and sampling" => [
            "Patterns and components" => "fields/patterns.md",
            "Angular sampling" => "fields/sampling.md",
        ],
        "Level transfers: I2I and O2O" => "transfers/i2i.md",
        "Box translations: I2O" => "translations/i2o.md",
        "Leaf storage" => [
            "Choosing storage" => "storage/choosing.md",
            "Harmonic expansions" => "storage/harmonics.md",
        ],
        "Extending MLFMA" => [
            "Adding an operator" => "extending/operators.md",
            "Sampling, transfers and storage" => "extending/backends.md",
        ],
        "API Reference" => [
            "Overview" => "api/index.md",
            "Top level" => "api/toplevel.md",
            "Sampling" => "api/sampling.md",
            "Operators" => "api/operators.md",
            "Far fields" => "api/farfields.md",
            "Leaf collections and scratch" => "api/collections.md",
            "Moment multiplication" => "api/mul.md",
            "Conjugated moment multiplication" => "api/conjmul.md",
            "Phase" => "api/phase.md",
            "Interpolation" => "api/interpolation.md",
            "Interpolation diagnostics" => "api/diagnostics.md",
            "Harmonics" => "api/harmonics.md",
            "Spherical harmonics" => "api/sphericalharmonics.md",
            "Circular harmonics" => "api/circularharmonics.md",
            "Translations" => "api/translations.md",
            "Assembly" => "api/assembly.md",
            "Equidistant representation" => "api/equidistant.md",
            "Seams" => "api/seams.md",
            "Test support" => "api/testsupport.md",
        ],
    ],
    # Most public names are qualified, not exported, so check all docstrings.
    checkdocs=:all,
    # Keep broken references as build errors; do not enable `warnonly`.
)

deploydocs(;
    repo="github.com/djukic14/MLFMA.jl",
    target="build",
    devbranch="main",
    push_preview=true,
    forcepush=true,
)
