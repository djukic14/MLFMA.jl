using MLFMA
using StaticArrays
using ClusterTrees
using PlotlyJS
using CompScienceMeshes
using H2Factory

const P = SVector{3,Float64}

fn = joinpath(dirname(@__FILE__), "meshes/OneCubeThreeSpheres.msh")
m = CompScienceMeshes.readmesh(fn)
points = vertices(m)

rootcenter, rootsize = H2Trees.boundingbox(points)
root_center = P(rootcenter)

tree = TwoNTree(root_center, points, rootsize, 0.25)

targetlevel = 3

w = CompScienceMeshes.wireframe(skeleton(m, 1); width=1.0, color=:blue)

traces = [w]

for node in H2Trees.DepthFirstIterator(tree, H2Trees.root(tree))
    if H2Trees.level(tree, node) == targetlevel
        push!(
            traces,
            tracecube(
                H2Trees.center(tree, node),
                H2Trees.halfsize(tree, node);
                line_color=:red,
                legendgroup="near",
                name="Near Boxes",
                showlegend=false,
            ),
        )
    end
end

p = PlotlyJS.plot(traces)

PlotlyJS.display(p)
