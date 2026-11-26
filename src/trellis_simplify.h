#pragma once
// The mesh simplification of the TRELLIS.2 export (o-voxel's to_glb: mesh.simplify, CuMesh's quadric error edge collapses on the device,
// driven as cumesh's Python simplify does: the threshold starts at 1e-8 and grows tenfold while a pass removes less than 1% of the faces;
// the lambdas are 1e-2 (edge length) and 1e-3 (skinniness)). The per-pass adjacency and sorted edges are built on the host for now.

#include <cstdint>
#include <string>
#include <vector>

namespace trellis {

struct SimplifyStats {
    int passes = 0;
    int faces_before = 0, faces_after = 0;
    double seconds = 0;
};

// verts [V][3], faces [F][3]: reduced in place to at most target faces (no change when F <= target).
// manifold: the mesh is a closed, consistently wound manifold (the solid's surface, trellis_solid.h) and stays one: each edge is
// taken once from the face that runs it upwards (no sort), and a collapse must keep the link condition (sddev::simp_collapse).
bool simplify_to_faces(std::vector<float>& verts, std::vector<int32_t>& faces, int target, SimplifyStats& stats, std::string& error,
                       bool manifold = false);

}  // namespace trellis
