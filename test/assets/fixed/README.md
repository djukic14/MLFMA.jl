# Fixed Sphere Meshes

These meshes preserve the geometry used by structural regression tests. Generating
them during each run allowed platform-dependent Gmsh output to change the tree
depths and remove the cases the tests were intended to cover.

The files use CompScienceMeshes' text `.in` format. They were generated once with
`meshsphere(radius * wavelength, wavelength / meshfactor)` and saved with
`CompScienceMeshes.writemesh`. Reading them back preserves every coordinate and cell.
The wavelength is from `physicalconstants(5e7)` in the test helpers.

| File | Radius / wavelength | Mesh Factor | Intended Coverage |
| --- | ---: | ---: | --- |
| `default-nonuniform.in` | 1.0 | 8 | A default-built tree with leaves above the first aggregation level |
| `mixed-aggregation.in` | 0.7 | 6 | Populated mixed leaf depths within the aggregation range |
| `she-protruding.in` | 0.8 | 10 | Mixed depths with basis supports larger than some leaf boxes, giving `P > L` |

The tests still assert these properties; replacing a mesh must not silently remove
their coverage. No numerical tolerances depend on a particular leaf histogram.
