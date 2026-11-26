#pragma once
// A closed solid from the TRELLIS.2 mesh, for 3D printing (the STL and the GLB without texture). The mesh of the flexible dual grid
// marks where the surface crosses the voxel edges but not which side is inside: its faces point both ways, it has holes, edges of
// three or more faces and thousands of pieces. A slicer needs a closed surface around a solid. This rebuilds one on a grid:
//
//   1. the distance of each grid point to the mesh, near it (the exact point-triangle distance, `band` cells around each face);
//   2. the points nearer than `iso` cells are the wall; the distance of every point to the wall (an exact Euclidean distance
//      transform, three passes of Felzenszwalb and Huttenlocher's lower envelope);
//   3. the outside: a flood fill from the grid's border through the points at least `close` cells from the wall, so it can not
//      go through a hole or a gap narrower than about twice that. A sealed region the fill did not reach is the model's own
//      hollow (a whistle's chamber) when rays from it to the border cross an even number of walls (most of them: a hole or a
//      doubled sheet on one ray does not decide), and is kept as outside; with an odd count it is the inside of a solid,
//      filled. Then the distance of every point to that outside (a second transform);
//   4. the field f = min(distance to the mesh - iso, close + 0.5 - distance to the outside), negative inside: the surface is `iso`
//      cells out from the mesh where the outside reaches it, and a plug over the holes and gaps the fill did not go through; a
//      second fill from the outside through the positive points, and those it does not reach (a pocket behind the wall, a void
//      inside) are made inside;
//   5. its zero set by marching tetrahedra (six per cube, the Kuhn split along the main diagonal): the zero set of a piecewise
//      linear field on a simplicial grid is a closed 2-manifold, each face wound with its normal pointing to the outside.
//
// The grid has `grid` cells across the mesh's largest side, and a border of empty cells. Device kernels (sd_gpu.hip solid_*), the
// scans and the pieces on the host. The surface is then simplified to the target with the simplifier's manifold mode (the link
// condition: no collapse pinches the surface), by the caller.

#include <cstdint>
#include <string>
#include <vector>

namespace trellis {

struct SolidConfig {
    int grid = 512;        // cells across the mesh's largest side (64 to 1024)
    float iso = 0.6f;      // the surface's distance from the mesh, in cells (over half a cell: no grid edge crosses the mesh unseen)
    float close = 4.0f;    // the fill keeps this far from the wall, in cells: holes and gaps narrower than about twice it are closed
    float close_max = 16.0f;    // when the solid comes out a thin shell (the fill got in through a wider hole), close is doubled up to this
    float min_thickness = 3.0f; // ... a thin shell: a mean thickness (2 volume / area) under this many cells
    float band = 2.5f;     // the exact distance to the mesh is computed this far from each face, in cells
    double min_piece = 1e-3;  // pieces with less than this part of the largest piece's volume are dropped (specks, small voids)
    // A hollow print (cheaper, lighter): the inside further than `wall` from the surface is empty, in the mesh's units (0: solid).
    // `drain`: each hollow gets a hole this wide (the mesh's units; 0: none) from its lowest point straight down (-z, the STL's
    // down) until it is out of the solid, so resin drains and air gets in. Hollows under min_piece of the largest are filled.
    float wall = 0;
    float drain = 0;
};

struct SolidStats {
    int dims[3] = {0, 0, 0};   // the grid points along x, y, z
    double cell = 0;           // the cell's size in the mesh's units
    int flood_rounds = 0;      // the fill's rounds of six sweeps
    long outside = 0;          // grid points the fill reached
    int pieces = 0;            // closed pieces of the surface, before the small ones are dropped
    int pieces_dropped = 0;
    int faces = 0, verts = 0;  // the surface's faces and vertices (after the drop)
    double volume = 0;         // the solid's volume, in the mesh's units
    double area = 0;           // its surface's area
    double thickness = 0;      // its mean thickness, 2 volume / area, in cells
    float close = 0;           // the close of the result
    int attempts = 0;          // the runs (close doubled after a thin shell)
    int hollows = 0;           // with a wall: the empty regions inside, and those filled for being small
    int hollows_filled = 0;
    int drains = 0;            // the drain holes made
    int cavities = 0;          // sealed free regions kept as the model's own hollows (air: an even count of walls to the outside)
    int interiors = 0;         // sealed free regions filled (the inside of a solid: an odd count)
    double seconds_distance = 0, seconds_transform = 0, seconds_fill = 0, seconds_surface = 0, seconds = 0;
};

// verts [V][3], faces [F][3] (any winding, any defects) to the closed surface out_verts, out_faces (outward winding). The device path.
// A result thinner than min_thickness (on average) is a sign the fill went in through a hole wider than twice close: the run is
// repeated with close doubled, up to close_max; the first thick enough result is kept, else the first one (a thin object). The
// choice is made on the solid; a wall then hollows it in one more run with that close.
bool solidify(const std::vector<float>& verts, const std::vector<int32_t>& faces, const SolidConfig& cfg, std::vector<float>& out_verts,
              std::vector<int32_t>& out_faces, SolidStats& stats, std::string& error);

// The same steps on the host, a reference for tests/solid_verify.cpp (not a serving path). The intermediate grids, when asked for:
// udf (the distance to the mesh in cells, +inf beyond the band), d1 (squared distance to the wall), state (0 wall-near, 1 not reached,
// 2 outside), field (f).
struct SolidGrids {
    std::vector<float> udf, d1, field;
    std::vector<uint8_t> state;
};
bool solidify_reference(const std::vector<float>& verts, const std::vector<int32_t>& faces, const SolidConfig& cfg,
                        std::vector<float>& out_verts, std::vector<int32_t>& out_faces, SolidStats& stats, std::string& error,
                        SolidGrids* grids = nullptr);
// The device path's grids (the same as the reference's), for the verify tool.
bool solidify_grids(const std::vector<float>& verts, const std::vector<int32_t>& faces, const SolidConfig& cfg, SolidGrids& grids,
                    std::string& error);

// A mesh's edges: closed and manifold when every edge has two faces (boundary: one, non-manifold: three or more); misoriented: the
// two faces of a manifold edge wound the same way. Pieces: the faces joined through shared edges. volume: the signed volume.
struct MeshCheck {
    long edges = 0, boundary = 0, nonmanifold = 0, misoriented = 0, degenerate = 0;
    int pieces = 0;
    double volume = 0;
    bool closed() const { return boundary == 0 && nonmanifold == 0 && misoriented == 0; }
};
void check_mesh(const std::vector<float>& verts, const std::vector<int32_t>& faces, MeshCheck& out);

}  // namespace trellis
