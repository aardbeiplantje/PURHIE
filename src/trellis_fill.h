#pragma once
// The mesh post-process of the TRELLIS.2 pipeline (mesh/base.py: fill_holes, which calls CuMesh's fill_holes): the boundary loops
// shorter than max_perimeter are closed with a fan, a new vertex at the loop's centre (the mean of its edges' midpoints) and one
// triangle per boundary edge. The index work is a host pass over the mesh (sorting the edges and walking the loops); it is not a
// model computation, so it is not a device kernel.

#include <cstdint>
#include <vector>

namespace trellis {

struct FillStats {
    int boundary_edges = 0;  // the edges used by one face
    int loops = 0;           // the boundary loops (CuMesh's rule: see trellis_fill.cpp)
    int filled = 0;          // the loops closed (perimeter below the limit)
    int too_long = 0;        // the loops left open: perimeter at or above the limit
};

// verts: [V][3] floats; faces: [F][3] int32 (consistently oriented). Appends the new vertices and faces in place.
void fill_small_holes(std::vector<float>& verts, std::vector<int32_t>& faces, float max_perimeter, FillStats& stats);

}  // namespace trellis
