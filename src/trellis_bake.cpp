// The textured output on the engine (trellis_bake.h). The rasteriser and the trilinear sampler follow o-voxel's postprocess.to_glb and
// flexgemm's grid_sample_3d (trilinear over the neighbours that exist, their weights renormalised): a texel's position is in voxel
// index units, (p + 0.5) * res, and voxel k is centred at k + 0.5.
#include "trellis_bake.h"
#include "trellis_charts.h"

#include "xatlas.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace trellis {
namespace {

uint32_t crc32_of(const uint8_t* data, size_t n, uint32_t crc = 0) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

void put_be32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back((uint8_t)(v >> 24));
    out.push_back((uint8_t)(v >> 16));
    out.push_back((uint8_t)(v >> 8));
    out.push_back((uint8_t)v);
}

void put_le32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back((uint8_t)v);
    out.push_back((uint8_t)(v >> 8));
    out.push_back((uint8_t)(v >> 16));
    out.push_back((uint8_t)(v >> 24));
}

void png_chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
    put_be32(out, (uint32_t)data.size());
    std::vector<uint8_t> td(type, type + 4);
    td.insert(td.end(), data.begin(), data.end());
    out.insert(out.end(), td.begin(), td.end());
    put_be32(out, crc32_of(td.data(), td.size()));
}

// the trilinear sample of the attributes at a position in voxel index units (flexgemm's rule: the 8 integer neighbours of q +- 0.5,
// truncated as int() does; weight prod(1 - |n + 0.5 - q|) over the neighbours that exist, renormalised; zero when none exists)
void sample_trilinear(const std::unordered_map<uint64_t, int>& index, const VoxelAttributes& attr, const float q[3], double* acc) {
    const int C = attr.channels;
    for (int c = 0; c < C; ++c) acc[c] = 0;
    double wsum = 0;
    const uint64_t R = (uint64_t)attr.res;
    for (int dx = 0; dx < 2; ++dx)
        for (int dy = 0; dy < 2; ++dy)
            for (int dz = 0; dz < 2; ++dz) {
                const float off[3] = {dx ? 0.5f : -0.5f, dy ? 0.5f : -0.5f, dz ? 0.5f : -0.5f};
                int n[3];
                for (int a = 0; a < 3; ++a) n[a] = (int)(q[a] + off[a]);  // int() truncates, as torch's .int() does
                bool inside = true;
                for (int a = 0; a < 3; ++a) inside = inside && n[a] >= 0 && n[a] < attr.res;
                if (!inside) continue;
                const uint64_t key = ((uint64_t)n[0] * R + (uint64_t)n[1]) * R + (uint64_t)n[2];
                const auto it = index.find(key);
                if (it == index.end()) continue;
                double w = 1;
                for (int a = 0; a < 3; ++a) w *= 1.0 - std::fabs((double)n[a] + 0.5 - q[a]);
                const float* f = &attr.feats[(size_t)it->second * C];
                for (int c = 0; c < C; ++c) acc[c] += w * f[c];
                wsum += w;
            }
    if (wsum > 0)
        for (int c = 0; c < C; ++c) acc[c] /= wsum;
}

// The gaps around the charts: each pass gives an uncovered texel with covered neighbours the mean of those (the 4-neighbourhood).
void dilate(std::vector<uint8_t>& value, std::vector<char>& covered, int T, int K, int radius) {
    for (int pass = 0; pass < radius; ++pass) {
        std::vector<uint8_t> nv = value;
        std::vector<char> nc = covered;
        for (int y = 0; y < T; ++y)
            for (int x = 0; x < T; ++x) {
                const size_t i = (size_t)y * T + x;
                if (covered[i]) continue;
                const int dxs[4] = {1, -1, 0, 0}, dys[4] = {0, 0, 1, -1};
                int cnt = 0;
                double sum[4] = {0, 0, 0, 0};
                for (int k = 0; k < 4; ++k) {
                    const int xx = x + dxs[k], yy = y + dys[k];
                    if (xx < 0 || yy < 0 || xx >= T || yy >= T) continue;
                    const size_t j = (size_t)yy * T + xx;
                    if (!covered[j]) continue;
                    cnt++;
                    for (int c = 0; c < K; ++c) sum[c] += value[j * K + c];
                }
                if (cnt == 0) continue;
                for (int c = 0; c < K; ++c) nv[i * K + c] = (uint8_t)(sum[c] / cnt);
                nc[i] = 1;
            }
        value.swap(nv);
        covered.swap(nc);
    }
}

}  // namespace

bool unwrap_uv(const std::vector<float>& verts, const std::vector<int32_t>& faces, int texture_size, UvMesh& out, std::string& error) {
    out = UvMesh();
    if (verts.size() < 9 || faces.size() < 3) {
        error = "no mesh to unwrap";
        return false;
    }
    cumesh_xatlas::Atlas* atlas = cumesh_xatlas::Create();
    cumesh_xatlas::MeshDecl decl;
    decl.vertexPositionData = verts.data();
    decl.vertexPositionStride = sizeof(float) * 3;
    decl.vertexCount = (uint32_t)(verts.size() / 3);
    decl.indexData = faces.data();
    decl.indexCount = (uint32_t)faces.size();
    decl.indexFormat = cumesh_xatlas::IndexFormat::UInt32;
    const cumesh_xatlas::AddMeshError add = cumesh_xatlas::AddMesh(atlas, decl, 0);
    if (add != cumesh_xatlas::AddMeshError::Success) {
        error = std::string("xatlas: ") + cumesh_xatlas::StringForEnum(add);
        cumesh_xatlas::Destroy(atlas);
        return false;
    }
    cumesh_xatlas::ComputeCharts(atlas, cumesh_xatlas::ChartOptions());
    cumesh_xatlas::PackOptions pack;
    pack.resolution = (uint32_t)texture_size;
    pack.padding = 2;
    cumesh_xatlas::PackCharts(atlas, pack);
    if (atlas->meshCount == 0 || atlas->meshes[0].vertexCount == 0) {
        error = "xatlas produced no mesh";
        cumesh_xatlas::Destroy(atlas);
        return false;
    }
    const cumesh_xatlas::Mesh& m = atlas->meshes[0];
    const size_t Vn = m.vertexCount;
    out.positions.resize(Vn * 3);
    out.uvs.resize(Vn * 2);
    out.xref.resize(Vn);
    for (size_t i = 0; i < Vn; ++i) {
        const cumesh_xatlas::Vertex& vx = m.vertexArray[i];
        const int32_t o = (int32_t)vx.xref;
        out.xref[i] = o;
        for (int c = 0; c < 3; ++c) out.positions[i * 3 + c] = verts[(size_t)o * 3 + c];
        out.uvs[i * 2] = vx.uv[0] / (float)atlas->width;
        out.uvs[i * 2 + 1] = vx.uv[1] / (float)atlas->height;
    }
    out.faces.resize(m.indexCount);
    for (size_t i = 0; i < m.indexCount; ++i) out.faces[i] = (int32_t)m.indexArray[i];
    out.atlas_width = (int)atlas->width;
    out.atlas_height = (int)atlas->height;
    out.charts = (int)atlas->chartCount;
    cumesh_xatlas::Destroy(atlas);
    return true;
}

bool unwrap_uv_by_charts(const std::vector<float>& verts, const std::vector<int32_t>& faces, const std::vector<int32_t>& chart_ids,
                         int texture_size, UvMesh& out, std::string& error) {
    out = UvMesh();
    const size_t F = faces.size() / 3, V = verts.size() / 3;
    if (V == 0 || F == 0 || chart_ids.size() != F) {
        error = "no mesh, or one chart id per face is needed";
        return false;
    }
    int nc = 0;
    for (int32_t c : chart_ids) nc = std::max(nc, c + 1);
    // the faces of each chart, and the chart's vertices (global ids; local index by a map)
    std::vector<std::vector<int32_t>> cfaces(nc);
    for (size_t f = 0; f < F; ++f) cfaces[chart_ids[f]].push_back((int32_t)f);
    std::vector<int32_t> local(V, -1);
    std::vector<std::vector<int32_t>> cverts(nc), clocal(nc);  // the chart's global vertices, its faces in local indices
    for (int c = 0; c < nc; ++c) {
        if (cfaces[c].empty()) continue;
        std::vector<int32_t> used;
        for (int32_t f : cfaces[c])
            for (int k = 0; k < 3; ++k) {
                const int32_t v = faces[3 * f + k];
                if (local[v] < 0) {
                    local[v] = (int32_t)used.size();
                    used.push_back(v);
                }
            }
        for (int32_t f : cfaces[c])
            for (int k = 0; k < 3; ++k) clocal[c].push_back(local[faces[3 * f + k]]);
        for (int32_t v : used) local[v] = -1;
        cverts[c] = std::move(used);
    }
    // the chart meshes: positions per chart (copied to the chart's own arrays for xatlas)
    std::vector<std::vector<float>> cpos(nc);
    cumesh_xatlas::Atlas* atlas = cumesh_xatlas::Create();
    std::vector<int> mesh_chart;
    for (int c = 0; c < nc; ++c) {
        if (cfaces[c].empty()) continue;
        cpos[c].resize(cverts[c].size() * 3);
        for (size_t i = 0; i < cverts[c].size(); ++i)
            for (int k = 0; k < 3; ++k) cpos[c][3 * i + k] = verts[3 * (size_t)cverts[c][i] + k];
        cumesh_xatlas::MeshDecl decl;
        decl.vertexPositionData = cpos[c].data();
        decl.vertexPositionStride = sizeof(float) * 3;
        decl.vertexCount = (uint32_t)cverts[c].size();
        decl.indexData = clocal[c].data();
        decl.indexCount = (uint32_t)clocal[c].size();
        decl.indexFormat = cumesh_xatlas::IndexFormat::UInt32;
        if (cumesh_xatlas::AddMesh(atlas, decl, 0) != cumesh_xatlas::AddMeshError::Success) {
            error = "xatlas: a chart could not be added";
            cumesh_xatlas::Destroy(atlas);
            return false;
        }
        mesh_chart.push_back(c);
    }
    cumesh_xatlas::ComputeCharts(atlas, cumesh_xatlas::ChartOptions());
    cumesh_xatlas::PackOptions pack;
    pack.resolution = (uint32_t)texture_size;
    pack.padding = 2;
    cumesh_xatlas::PackCharts(atlas, pack);
    if (atlas->meshCount == 0) {
        error = "xatlas produced no meshes";
        cumesh_xatlas::Destroy(atlas);
        return false;
    }
    // gather: each chart's mesh (its vertices, faces and uvs) into one mesh with the original vertices
    for (uint32_t mi = 0; mi < atlas->meshCount; ++mi) {
        const cumesh_xatlas::Mesh& m = atlas->meshes[mi];
        const int c = mesh_chart[mi];
        const uint32_t base = (uint32_t)(out.positions.size() / 3);
        for (uint32_t i = 0; i < m.vertexCount; ++i) {
            const cumesh_xatlas::Vertex& vx = m.vertexArray[i];
            const int32_t o = cverts[c][vx.xref];
            out.xref.push_back(o);
            for (int k = 0; k < 3; ++k) out.positions.push_back(verts[3 * (size_t)o + k]);
            out.uvs.push_back(vx.uv[0] / (float)atlas->width);
            out.uvs.push_back(vx.uv[1] / (float)atlas->height);
        }
        for (uint32_t i = 0; i < m.indexCount; ++i) out.faces.push_back((int32_t)(base + m.indexArray[i]));
    }
    out.atlas_width = (int)atlas->width;
    out.atlas_height = (int)atlas->height;
    out.charts = (int)mesh_chart.size();
    cumesh_xatlas::Destroy(atlas);
    return true;
}

bool bake_textures(const UvMesh& uv, const VoxelAttributes& attr, int T, BakedTextures& out, std::string& error) {
    out = BakedTextures();
    const size_t N = attr.coords.size() / 4;
    if (T <= 0 || N == 0 || attr.channels < 6 || attr.feats.size() != N * (size_t)attr.channels || attr.res <= 0) {
        error = "no attributes to bake (six channels: base colour, metallic, roughness, alpha)";
        return false;
    }
    const size_t NT = (size_t)T * T;
    std::vector<char> covered(NT, 0);
    std::vector<float> pos(NT * 3, 0.0f);

    // the charts: each triangle's texels (the texel centres inside it), with the 3D position interpolated from its vertices
    const size_t F = uv.faces.size() / 3;
    for (size_t t = 0; t < F; ++t) {
        const int32_t ia = uv.faces[3 * t], ib = uv.faces[3 * t + 1], ic = uv.faces[3 * t + 2];
        const double ua = uv.uvs[2 * ia] * T, va = uv.uvs[2 * ia + 1] * T;
        const double ub = uv.uvs[2 * ib] * T, vb = uv.uvs[2 * ib + 1] * T;
        const double uc = uv.uvs[2 * ic] * T, vc = uv.uvs[2 * ic + 1] * T;
        const double d = (vb - vc) * (ua - uc) + (uc - ub) * (va - vc);
        if (std::fabs(d) < 1e-12) continue;
        const int x0 = std::max(0, (int)std::floor(std::min(ua, std::min(ub, uc))));
        const int x1 = std::min(T - 1, (int)std::ceil(std::max(ua, std::max(ub, uc))));
        const int y0 = std::max(0, (int)std::floor(std::min(va, std::min(vb, vc))));
        const int y1 = std::min(T - 1, (int)std::ceil(std::max(va, std::max(vb, vc))));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;
                const double l0 = ((vb - vc) * (px - uc) + (uc - ub) * (py - vc)) / d;
                const double l1 = ((vc - va) * (px - uc) + (ua - uc) * (py - vc)) / d;
                const double l2 = 1.0 - l0 - l1;
                if (l0 < -1e-6 || l1 < -1e-6 || l2 < -1e-6) continue;
                // the texel: row r from the top (v up in the atlas: row r = T - 1 - y)
                const size_t r = (size_t)(T - 1 - y), idx = r * T + x;
                for (int c = 0; c < 3; ++c)
                    pos[idx * 3 + c] = (float)(l0 * uv.positions[3 * ia + c] + l1 * uv.positions[3 * ib + c] + l2 * uv.positions[3 * ic + c]);
                covered[idx] = 1;
            }
    }

    // the voxels by their index, then the attributes at the covered texels
    std::unordered_map<uint64_t, int> index;
    index.reserve(N * 2);
    const uint64_t R = (uint64_t)attr.res;
    for (size_t i = 0; i < N; ++i) {
        const int32_t* c = &attr.coords[4 * i];
        index[((uint64_t)c[1] * R + (uint64_t)c[2]) * R + (uint64_t)c[3]] = (int)i;
    }
    std::vector<float> base(NT * 4, 0.0f), mr(NT * 2, 0.0f);  // base colour and alpha; roughness and metallic
    size_t covered_n = 0;
    double acc[64];
    for (size_t idx = 0; idx < NT; ++idx) {
        if (!covered[idx]) continue;
        covered_n++;
        float q[3];
        for (int a = 0; a < 3; ++a) q[a] = (pos[idx * 3 + a] + 0.5f) * (float)attr.res;  // (p - aabb0) / voxel_size, aabb0 = -0.5
        sample_trilinear(index, attr, q, acc);
        for (int c = 0; c < 3; ++c) base[idx * 4 + c] = (float)acc[c];
        base[idx * 4 + 3] = (float)acc[5];
        mr[idx * 2] = (float)acc[4];      // roughness
        mr[idx * 2 + 1] = (float)acc[3];  // metallic
    }

    // to 8 bits as python does (clip to [0, 1], times 255, truncated), then the gaps dilated (python's Telea inpainting, radius 3 for
    // the base colour and 1 for the others, here a mean of the neighbours)
    std::vector<uint8_t> base8(NT * 4, 0), mr8(NT * 2, 0);
    std::vector<char> cov_base = covered, cov_mr = covered;
    for (size_t i = 0; i < NT * 4; ++i) base8[i] = (uint8_t)std::min(255.0f, std::max(0.0f, base[i] * 255.0f));
    for (size_t i = 0; i < NT * 2; ++i) mr8[i] = (uint8_t)std::min(255.0f, std::max(0.0f, mr[i] * 255.0f));
    dilate(base8, cov_base, T, 4, 3);
    dilate(mr8, cov_mr, T, 2, 1);

    out.size = T;
    out.coverage = (double)covered_n / (double)NT;
    out.base_rgba = base8;
    out.metal_rough.assign(NT * 3, 0);
    for (size_t i = 0; i < NT; ++i) {
        out.metal_rough[i * 3 + 0] = 0;
        out.metal_rough[i * 3 + 1] = mr8[i * 2];      // green: roughness
        out.metal_rough[i * 3 + 2] = mr8[i * 2 + 1];  // blue: metallic
    }
    return true;
}

std::vector<float> vertex_normals(const std::vector<float>& verts, const std::vector<int32_t>& faces) {
    std::vector<float> n(verts.size(), 0.0f);
    for (size_t t = 0; t < faces.size() / 3; ++t) {
        const int32_t a = faces[3 * t], b = faces[3 * t + 1], c = faces[3 * t + 2];
        const float e1[3] = {verts[3 * b] - verts[3 * a], verts[3 * b + 1] - verts[3 * a + 1], verts[3 * b + 2] - verts[3 * a + 2]};
        const float e2[3] = {verts[3 * c] - verts[3 * a], verts[3 * c + 1] - verts[3 * a + 1], verts[3 * c + 2] - verts[3 * a + 2]};
        const float f[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        for (int k = 0; k < 3; ++k) {
            const int32_t v = faces[3 * t + k];
            for (int c2 = 0; c2 < 3; ++c2) n[(size_t)v * 3 + c2] += f[c2];
        }
    }
    for (size_t v = 0; v < n.size() / 3; ++v) {
        const float l = std::sqrt(n[3 * v] * n[3 * v] + n[3 * v + 1] * n[3 * v + 1] + n[3 * v + 2] * n[3 * v + 2]);
        if (l > 0) {
            n[3 * v] /= l;
            n[3 * v + 1] /= l;
            n[3 * v + 2] /= l;
        } else {
            n[3 * v] = 0;
            n[3 * v + 1] = 0;
            n[3 * v + 2] = 1;
        }
    }
    return n;
}

std::vector<uint8_t> encode_png(const std::vector<uint8_t>& pixels, int width, int height, int channels) {
    std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    put_be32(ihdr, (uint32_t)width);
    put_be32(ihdr, (uint32_t)height);
    ihdr.push_back(8);                             // bit depth
    ihdr.push_back(channels == 4 ? 6 : 2);         // colour type: RGBA or RGB
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    png_chunk(out, "IHDR", ihdr);
    // the scanlines (filter 0), deflated by zlib (the textures were 117 MB stored at 4096 texels)
    std::vector<uint8_t> raw;
    raw.reserve((size_t)height * (1 + (size_t)width * channels));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), pixels.begin() + (size_t)y * width * channels, pixels.begin() + (size_t)(y + 1) * width * channels);
    }
    uLongf zn = compressBound((uLong)raw.size());
    std::vector<uint8_t> z(zn);
    if (compress2(z.data(), &zn, raw.data(), (uLong)raw.size(), 6) != Z_OK) return {};
    z.resize(zn);
    png_chunk(out, "IDAT", z);
    png_chunk(out, "IEND", {});
    return out;
}

bool encode_glb(const UvMesh& uv, const std::vector<float>& normals, const BakedTextures& tex, std::vector<uint8_t>& glb, std::string& error) {
    const size_t V = uv.positions.size() / 3, F = uv.faces.size() / 3;
    if (V == 0 || F == 0 || normals.empty()) {
        error = "nothing to write";
        return false;
    }
    // the vertices and normals with the python path's axes (x, y, z) -> (x, z, -y), the uv v flipped; normals are per original vertex
    std::vector<float> pos(V * 3), nrm(V * 3), tc(V * 2);
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (size_t i = 0; i < V; ++i) {
        const float p[3] = {uv.positions[3 * i], uv.positions[3 * i + 2], -uv.positions[3 * i + 1]};
        const float nn[3] = {normals[3 * uv.xref[i]], normals[3 * uv.xref[i] + 2], -normals[3 * uv.xref[i] + 1]};
        for (int c = 0; c < 3; ++c) {
            pos[3 * i + c] = p[c];
            nrm[3 * i + c] = nn[c];
            lo[c] = std::min(lo[c], p[c]);
            hi[c] = std::max(hi[c], p[c]);
        }
        tc[2 * i] = uv.uvs[2 * i];
        tc[2 * i + 1] = 1.0f - uv.uvs[2 * i + 1];
    }
    std::vector<uint32_t> idx(F * 3);
    for (size_t i = 0; i < F * 3; ++i) idx[i] = (uint32_t)uv.faces[i];

    const std::vector<uint8_t> png_base = encode_png(tex.base_rgba, tex.size, tex.size, 4);
    const std::vector<uint8_t> png_mr = encode_png(tex.metal_rough, tex.size, tex.size, 3);

    // the binary chunk: positions, normals, uvs, indices, the two images (each 4-byte aligned)
    std::vector<uint8_t> bin;
    auto align4 = [&]() {
        while (bin.size() % 4) bin.push_back(0);
    };
    size_t off_pos = bin.size();
    bin.resize(bin.size() + pos.size() * 4);
    std::memcpy(&bin[off_pos], pos.data(), pos.size() * 4);
    align4();
    const size_t off_nrm = bin.size();
    bin.resize(bin.size() + nrm.size() * 4);
    std::memcpy(&bin[off_nrm], nrm.data(), nrm.size() * 4);
    align4();
    const size_t off_tc = bin.size();
    bin.resize(bin.size() + tc.size() * 4);
    std::memcpy(&bin[off_tc], tc.data(), tc.size() * 4);
    align4();
    const size_t off_idx = bin.size();
    bin.resize(bin.size() + idx.size() * 4);
    std::memcpy(&bin[off_idx], idx.data(), idx.size() * 4);
    align4();
    const size_t off_img0 = bin.size();
    bin.insert(bin.end(), png_base.begin(), png_base.end());
    align4();
    const size_t off_img1 = bin.size();
    bin.insert(bin.end(), png_mr.begin(), png_mr.end());
    align4();

    char json[4096 + 512];
    std::snprintf(json, sizeof json,
                  "{\"asset\":{\"version\":\"2.0\",\"generator\":\"rocm-inference trellis_bake\"},"
                  "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
                  "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2},\"indices\":3,\"material\":0}]}],"
                  "\"materials\":[{\"doubleSided\":true,\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,1],"
                  "\"baseColorTexture\":{\"index\":0},\"metallicRoughnessTexture\":{\"index\":1},\"metallicFactor\":1,\"roughnessFactor\":1}}],"
                  "\"textures\":[{\"source\":0},{\"source\":1}],"
                  "\"images\":[{\"bufferView\":4,\"mimeType\":\"image/png\"},{\"bufferView\":5,\"mimeType\":\"image/png\"}],"
                  "\"accessors\":["
                  "{\"bufferView\":0,\"componentType\":5126,\"count\":%zu,\"type\":\"VEC3\",\"min\":[%.7g,%.7g,%.7g],\"max\":[%.7g,%.7g,%.7g]},"
                  "{\"bufferView\":1,\"componentType\":5126,\"count\":%zu,\"type\":\"VEC3\"},"
                  "{\"bufferView\":2,\"componentType\":5126,\"count\":%zu,\"type\":\"VEC2\"},"
                  "{\"bufferView\":3,\"componentType\":5125,\"count\":%zu,\"type\":\"SCALAR\"}],"
                  "\"bufferViews\":["
                  "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":34963},"
                  "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu},"
                  "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu}],"
                  "\"buffers\":[{\"byteLength\":%zu}]}",
                  V, (double)lo[0], (double)lo[1], (double)lo[2], (double)hi[0], (double)hi[1], (double)hi[2], V, V, F * 3,
                  off_pos, pos.size() * 4, off_nrm, nrm.size() * 4, off_tc, tc.size() * 4, off_idx, idx.size() * 4, off_img0,
                  png_base.size(), off_img1, png_mr.size(), bin.size());
    std::string js(json);
    while (js.size() % 4) js.push_back(' ');
    glb.clear();
    const uint32_t total = (uint32_t)(12 + 8 + js.size() + 8 + bin.size());
    glb.insert(glb.end(), {'g', 'l', 'T', 'F'});
    put_le32(glb, 2);
    put_le32(glb, total);
    put_le32(glb, (uint32_t)js.size());
    glb.insert(glb.end(), {'J', 'S', 'O', 'N'});
    glb.insert(glb.end(), js.begin(), js.end());
    put_le32(glb, (uint32_t)bin.size());
    glb.insert(glb.end(), {'B', 'I', 'N', 0});
    glb.insert(glb.end(), bin.begin(), bin.end());
    return true;
}

bool encode_glb_mesh(const std::vector<float>& verts, const std::vector<int32_t>& faces, std::vector<uint8_t>& glb, std::string& error) {
    const size_t V = verts.size() / 3, F = faces.size() / 3;
    if (V == 0 || F == 0) {
        error = "nothing to write";
        return false;
    }
    for (int32_t i : faces)
        if (i < 0 || (size_t)i >= V) {
            error = "a face names a vertex the mesh does not have";
            return false;
        }
    // the axes of encode_glb: (x, y, z) -> (x, z, -y)
    const std::vector<float> normals = vertex_normals(verts, faces);
    std::vector<float> pos(V * 3), nrm(V * 3);
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (size_t i = 0; i < V; ++i) {
        const float p[3] = {verts[3 * i], verts[3 * i + 2], -verts[3 * i + 1]};
        const float nn[3] = {normals[3 * i], normals[3 * i + 2], -normals[3 * i + 1]};
        for (int c = 0; c < 3; ++c) {
            pos[3 * i + c] = p[c];
            nrm[3 * i + c] = nn[c];
            lo[c] = std::min(lo[c], p[c]);
            hi[c] = std::max(hi[c], p[c]);
        }
    }
    std::vector<uint8_t> bin(pos.size() * 4 + nrm.size() * 4 + F * 3 * 4);
    const size_t off_nrm = pos.size() * 4, off_idx = off_nrm + nrm.size() * 4;
    std::memcpy(&bin[0], pos.data(), pos.size() * 4);
    std::memcpy(&bin[off_nrm], nrm.data(), nrm.size() * 4);
    std::memcpy(&bin[off_idx], faces.data(), F * 3 * 4);  // (indices: non-negative int32, the same bits as uint32)
    char json[2048];
    std::snprintf(json, sizeof json,
                  "{\"asset\":{\"version\":\"2.0\",\"generator\":\"rocm-inference trellis_bake\"},"
                  "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
                  "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"NORMAL\":1},\"indices\":2,\"material\":0}]}],"
                  "\"materials\":[{\"doubleSided\":true,\"pbrMetallicRoughness\":{\"baseColorFactor\":[0.8,0.8,0.8,1],"
                  "\"metallicFactor\":0,\"roughnessFactor\":0.8}}],"
                  "\"accessors\":["
                  "{\"bufferView\":0,\"componentType\":5126,\"count\":%zu,\"type\":\"VEC3\",\"min\":[%.7g,%.7g,%.7g],\"max\":[%.7g,%.7g,%.7g]},"
                  "{\"bufferView\":1,\"componentType\":5126,\"count\":%zu,\"type\":\"VEC3\"},"
                  "{\"bufferView\":2,\"componentType\":5125,\"count\":%zu,\"type\":\"SCALAR\"}],"
                  "\"bufferViews\":["
                  "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":%zu,\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":34962},"
                  "{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":34963}],"
                  "\"buffers\":[{\"byteLength\":%zu}]}",
                  V, (double)lo[0], (double)lo[1], (double)lo[2], (double)hi[0], (double)hi[1], (double)hi[2], V, F * 3, pos.size() * 4,
                  off_nrm, nrm.size() * 4, off_idx, F * 3 * 4, bin.size());
    std::string js(json);
    while (js.size() % 4) js.push_back(' ');
    glb.clear();
    const uint32_t total = (uint32_t)(12 + 8 + js.size() + 8 + bin.size());
    glb.insert(glb.end(), {'g', 'l', 'T', 'F'});
    put_le32(glb, 2);
    put_le32(glb, total);
    put_le32(glb, (uint32_t)js.size());
    glb.insert(glb.end(), {'J', 'S', 'O', 'N'});
    glb.insert(glb.end(), js.begin(), js.end());
    put_le32(glb, (uint32_t)bin.size());
    glb.insert(glb.end(), {'B', 'I', 'N', 0});
    glb.insert(glb.end(), bin.begin(), bin.end());
    return true;
}

bool encode_stl(const std::vector<float>& verts, const std::vector<int32_t>& faces, double size_mm, std::vector<uint8_t>& stl, std::string& error,
                bool* flipped) {
    const size_t V = verts.size() / 3, F = faces.size() / 3;
    if (V == 0 || F == 0) {
        error = "nothing to write";
        return false;
    }
    if (F > 0xffffffffu) {
        error = "too many faces for an STL file";
        return false;
    }
    for (int32_t i : faces)
        if (i < 0 || (size_t)i >= V) {
            error = "a face names a vertex the mesh does not have";
            return false;
        }
    // the bounding box, and the scale: the largest side is size_mm
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (size_t i = 0; i < V; ++i)
        for (int c = 0; c < 3; ++c) {
            lo[c] = std::min(lo[c], (double)verts[3 * i + c]);
            hi[c] = std::max(hi[c], (double)verts[3 * i + c]);
        }
    const double side = std::max(hi[0] - lo[0], std::max(hi[1] - lo[1], hi[2] - lo[2]));
    const double s = size_mm > 0 && side > 0 ? size_mm / side : 1.0;
    const double off[3] = {-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]};  // centred on x and y, on z = 0
    auto at = [&](int32_t v, int c) { return (verts[3 * (size_t)v + c] + off[c]) * s; };
    // the winding: the signed volume (the sum of the tetrahedra to the origin) is positive for faces counter-clockwise from outside
    double vol = 0;
    for (size_t f = 0; f < F; ++f) {
        const int32_t a = faces[3 * f], b = faces[3 * f + 1], c = faces[3 * f + 2];
        const double ax = at(a, 0), ay = at(a, 1), az = at(a, 2), bx = at(b, 0), by = at(b, 1), bz = at(b, 2);
        const double cx = at(c, 0), cy = at(c, 1), cz = at(c, 2);
        vol += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
    }
    const bool flip = vol < 0;
    if (flipped) *flipped = flip;
    stl.assign(84 + 50 * F, 0);
    const char head[] = "binary STL, rocm-inference trellis_bake (TRELLIS.2), millimetres, z up";
    std::memcpy(stl.data(), head, std::min(sizeof head - 1, (size_t)80));
    const uint32_t n = (uint32_t)F;
    std::memcpy(&stl[80], &n, 4);  // (little-endian: the host's order on gfx1151 machines)
    uint8_t* p = &stl[84];
    for (size_t f = 0; f < F; ++f, p += 50) {
        const int32_t a = faces[3 * f], b = faces[3 * f + (flip ? 2 : 1)], c = faces[3 * f + (flip ? 1 : 2)];
        float v[3][3];
        for (int k = 0; k < 3; ++k) {
            v[0][k] = (float)at(a, k);
            v[1][k] = (float)at(b, k);
            v[2][k] = (float)at(c, k);
        }
        const double e1[3] = {(double)v[1][0] - v[0][0], (double)v[1][1] - v[0][1], (double)v[1][2] - v[0][2]};
        const double e2[3] = {(double)v[2][0] - v[0][0], (double)v[2][1] - v[0][1], (double)v[2][2] - v[0][2]};
        double nn[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        const double len = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
        const float fn[3] = {len > 0 ? (float)(nn[0] / len) : 0.0f, len > 0 ? (float)(nn[1] / len) : 0.0f,
                             len > 0 ? (float)(nn[2] / len) : 0.0f};
        std::memcpy(p, fn, 12);
        std::memcpy(p + 12, v, 36);  // (the attribute bytes stay 0)
    }
    return true;
}

bool write_glb(const std::string& path, const UvMesh& uv, const std::vector<float>& normals, const BakedTextures& tex, std::string& error) {
    std::vector<uint8_t> glb;
    if (!encode_glb(uv, normals, tex, glb, error)) return false;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        error = "cannot write " + path;
        return false;
    }
    const size_t wrote = std::fwrite(glb.data(), 1, glb.size(), f);
    std::fclose(f);
    if (wrote != glb.size()) {
        error = "short write to " + path;
        return false;
    }
    return true;
}

bool export_textured_glb(const std::vector<float>& verts, const std::vector<int32_t>& faces, const VoxelAttributes& attr, int texture_size,
                         const std::string& path, std::string& error) {
    // the charts (CuMesh's clustering, on the device), then xatlas on each chart
    std::vector<int32_t> chart_ids;
    ChartStats cs;
    if (!compute_charts(verts, faces, ChartConfig(), chart_ids, cs, error)) return false;
    UvMesh uv;
    if (!unwrap_uv_by_charts(verts, faces, chart_ids, texture_size, uv, error)) return false;
    BakedTextures bt;
    if (!bake_textures(uv, attr, texture_size, bt, error)) return false;
    return write_glb(path, uv, vertex_normals(verts, faces), bt, error);
}

}  // namespace trellis
