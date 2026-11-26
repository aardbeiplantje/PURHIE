#pragma once
// The textured output of the TRELLIS.2 pipeline (o-voxel's postprocess.to_glb, the parts the engine has): the UV unwrap (xatlas, vendored
// in third_party/xatlas), the texture baked from the voxel attributes (the texture flow's and decoder's channels: base colour 0 to 2,
// metallic 3, roughness 4, alpha 5), the inpainting of the gaps around the charts, and a binary glTF with two PNG textures.
// The charts are CuMesh's clustering (trellis_charts.h, on the device), then xatlas packs and parametrises each chart. Not ported from the
// python path: the remeshing and simplification (off by default there), the Telea inpainting (here a dilation of the chart edges), and the BVH lookup
// of the original mesh (the unwrap keeps the vertices, so each texel's position is the interpolation of its triangle's vertices).

#include <cstdint>
#include <string>
#include <vector>

namespace trellis {

// The unwrapped mesh: xatlas splits the vertices at the chart seams; xref maps each to its original vertex.
struct UvMesh {
    std::vector<float> positions;  // [V'][3] (the original positions)
    std::vector<float> uvs;        // [V'][2] in [0, 1], the atlas (v up, as the atlas has it)
    std::vector<int32_t> faces;    // [F'][3]
    std::vector<int32_t> xref;     // [V'] the original vertex of each
    int atlas_width = 0, atlas_height = 0;
    int charts = 0;
};

// verts [V][3], faces [F][3]: the unwrap with xatlas's default charts, packed at texture_size texels.
bool unwrap_uv(const std::vector<float>& verts, const std::vector<int32_t>& faces, int texture_size, UvMesh& out, std::string& error);

// The same unwrap with the charts given (chart_ids [F]: the chart of each face, the clustering of the device): each chart is added to xatlas
// as its own mesh (as CuMesh's uv_unwrap does), then the charts are parametrised, packed and gathered back to the original vertices.
bool unwrap_uv_by_charts(const std::vector<float>& verts, const std::vector<int32_t>& faces, const std::vector<int32_t>& chart_ids,
                         int texture_size, UvMesh& out, std::string& error);

// The voxel attributes to bake: the voxels of the grid (coords [N][4], batch 0, indices in 0..res-1) and their channels (feats [N][C]).
struct VoxelAttributes {
    std::vector<int32_t> coords;
    std::vector<float> feats;
    int channels = 0;
    int res = 0;  // the grid: voxel k covers [k, k + 1) in index units; the mesh is over [-0.5, 0.5]^3 (voxel size 1 / res)
};

// The baked textures (8-bit): the base colour with alpha (RGBA), the metallic-roughness (RGB: 0, roughness, metallic, the glTF packing).
struct BakedTextures {
    int size = 0;
    std::vector<uint8_t> base_rgba;
    std::vector<uint8_t> metal_rough;
    double coverage = 0;  // the fraction of texels on a chart
};

bool bake_textures(const UvMesh& uv, const VoxelAttributes& attr, int texture_size, BakedTextures& out, std::string& error);

// The vertex normals of the mesh (area-weighted face normals), per original vertex.
std::vector<float> vertex_normals(const std::vector<float>& verts, const std::vector<int32_t>& faces);

// An 8-bit PNG (channels 3 or 4) from row-major pixels, rows from the top (zlib stored blocks, no compression).
std::vector<uint8_t> encode_png(const std::vector<uint8_t>& pixels, int width, int height, int channels);

// A binary glTF (.glb): the unwrapped mesh with the python path's axes (y and z swapped, y negated) and uv v flipped, the two PNG textures.
bool write_glb(const std::string& path, const UvMesh& uv, const std::vector<float>& normals, const BakedTextures& tex, std::string& error);
// The same GLB in memory (the server's reply).
bool encode_glb(const UvMesh& uv, const std::vector<float>& normals, const BakedTextures& tex, std::vector<uint8_t>& glb, std::string& error);

// A GLB of the mesh alone (verts [V][3], faces [F][3] in the mesh's frame, z up): the positions and vertex normals with the GLB's axes
// (as encode_glb), the faces, a plain grey material; no texture coordinates, no images.
bool encode_glb_mesh(const std::vector<float>& verts, const std::vector<int32_t>& faces, std::vector<uint8_t>& glb, std::string& error);

// A binary STL of the mesh (for 3D printing): z up as the mesh is; every face wound counter-clockwise seen from outside (when the
// mesh's signed volume is negative all are turned), its normal from that winding; scaled so the largest side of the bounding box is
// size_mm millimetres (0: the mesh's own units), centred on x and y, standing on z = 0. *flipped: the faces were turned.
bool encode_stl(const std::vector<float>& verts, const std::vector<int32_t>& faces, double size_mm, std::vector<uint8_t>& stl, std::string& error,
                bool* flipped = nullptr);

// The whole export: the charts (trellis_charts.h), the unwrap of each chart by xatlas, the texture baked from the voxel attributes at
// texture_size texels, and the GLB at path.
bool export_textured_glb(const std::vector<float>& verts, const std::vector<int32_t>& faces, const VoxelAttributes& attr, int texture_size,
                         const std::string& path, std::string& error);

}  // namespace trellis
