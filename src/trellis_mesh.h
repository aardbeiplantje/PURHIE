#pragma once
// The mesh step of the TRELLIS.2 shape decoder (FlexiDualGridVaeDecoder's eval path: o-voxel's flexible_dual_grid_to_mesh with
// train = False) on the device. The decoded features of the voxels (the dual vertex, the intersection flags and the split
// weight) become a triangle mesh over a grid^3 grid spanning [lo, hi]^3. Plain device kernels (src/sd_gpu.hip): the vertex of
// each voxel, the 12 edge corners of each voxel looked up in a hash table, the valid edges compacted by an exclusive scan, and
// two triangles per quad, split where the split weights say.

#include <cstdint>
#include <string>
#include <vector>

namespace trellis {

// coords: [n][4] int32 (batch, x, y, z) of the voxels; feats: [n][7] (f0..2 the dual vertex before the margin sigmoid, f3..5 the
// intersection flags (> 0), f6 the split weight before the softplus). margin: the dual vertex's margin (0.5 for TRELLIS.2).
// verts: [n][3] floats, one per voxel in the input order; faces: [T][3] int32 indices into verts, two triangles per valid quad.
// The inputs may be plain host memory: they are copied into device-visible memory first. Returns false with error set on failure.
bool dual_grid_to_mesh(const int32_t* coords, const float* feats, int n, int grid, float lo, float hi, float margin,
                       std::vector<float>& verts, std::vector<int32_t>& faces, std::string& error);

}  // namespace trellis
