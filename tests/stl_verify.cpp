// The mesh-only exports of trellis_bake.h on a cube: encode_stl (the size, the faces turned when the mesh is wound inside out, each
// normal pointing out of the cube, the box scaled to the size asked, centred on x and y, standing on z = 0, z kept up) and
// encode_glb_mesh (the GLB's header and chunks, the accessors of the positions, normals and indices, no texture). No device needed.
// Usage: stl_verify
#include "trellis_bake.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_bad = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-64s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_bad;
}

int main() {
    // a cube of side 2 at (3, -1, 5) .. (5, 1, 7), its 12 faces wound clockwise from outside (inside out)
    const std::vector<float> v = {3, -1, 5, 5, -1, 5, 5, 1, 5, 3, 1, 5, 3, -1, 7, 5, -1, 7, 5, 1, 7, 3, 1, 7};
    const int out_ccw[12][3] = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4},
                                {2, 3, 7}, {2, 7, 6}, {1, 2, 6}, {1, 6, 5}, {3, 0, 4}, {3, 4, 7}};
    std::vector<int32_t> f_out, f_in;
    for (const auto& t : out_ccw) {
        f_out.insert(f_out.end(), {t[0], t[1], t[2]});
        f_in.insert(f_in.end(), {t[0], t[2], t[1]});
    }
    for (int inside_out = 0; inside_out < 2; ++inside_out) {
        std::printf("STL of the cube %s\n", inside_out ? "wound inside out" : "wound outward");
        std::vector<uint8_t> stl;
        std::string err;
        bool flipped = false;
        const bool ok = trellis::encode_stl(v, inside_out ? f_in : f_out, 50.0, stl, err, &flipped);
        check(ok, "encoded");
        check(stl.size() == 84 + 50 * 12, "84 bytes of header and count, 50 per face");
        uint32_t n = 0;
        std::memcpy(&n, &stl[80], 4);
        check(n == 12, "12 faces");
        check(flipped == (inside_out != 0), inside_out ? "the faces are turned" : "the faces are kept");
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        bool normals_out = true, normals_match = true;
        for (uint32_t i = 0; i < n; ++i) {
            float nn[3], t[3][3];
            std::memcpy(nn, &stl[84 + 50 * i], 12);
            std::memcpy(t, &stl[84 + 50 * i + 12], 36);
            float c[3] = {0, 0, 0};
            for (int k = 0; k < 3; ++k)
                for (int a = 0; a < 3; ++a) {
                    lo[a] = std::min(lo[a], t[k][a]);
                    hi[a] = std::max(hi[a], t[k][a]);
                    c[a] += t[k][a] / 3;
                }
            // the face's centre minus the cube's centre (0, 0, 25 after the placement) points the way the normal does
            const float d[3] = {c[0], c[1], c[2] - 25.0f};
            if (nn[0] * d[0] + nn[1] * d[1] + nn[2] * d[2] <= 0) normals_out = false;
            // and the normal is the winding's (counter-clockwise seen from outside)
            const float e1[3] = {t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]};
            const float e2[3] = {t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]};
            const float cr[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
            if (cr[0] * nn[0] + cr[1] * nn[1] + cr[2] * nn[2] <= 0) normals_match = false;
        }
        check(normals_out, "every normal points out of the cube");
        check(normals_match, "every normal is its face's winding (counter-clockwise from outside)");
        check(std::fabs(lo[0] + 25) < 1e-4 && std::fabs(hi[0] - 25) < 1e-4 && std::fabs(lo[1] + 25) < 1e-4 && std::fabs(hi[1] - 25) < 1e-4,
              "scaled to 50 mm and centred on x and y");
        check(std::fabs(lo[2]) < 1e-4 && std::fabs(hi[2] - 50) < 1e-4, "standing on z = 0, z up (its height 50)");
    }
    {
        std::printf("STL in the mesh's own units (size 0)\n");
        std::vector<uint8_t> stl;
        std::string err;
        trellis::encode_stl(v, f_out, 0.0, stl, err);
        float t[3][3];
        float hi = -1e30f;
        for (int i = 0; i < 12; ++i) {
            std::memcpy(t, &stl[84 + 50 * i + 12], 36);
            for (int k = 0; k < 3; ++k) hi = std::max(hi, t[k][2]);
        }
        check(std::fabs(hi - 2) < 1e-5, "the height stays 2 (only placed, not scaled)");
        std::vector<int32_t> bad = f_out;
        bad[5] = 99;
        check(!trellis::encode_stl(v, bad, 50, stl, err), "a face naming a vertex the mesh lacks is refused");
    }
    {
        std::printf("GLB of the mesh alone\n");
        std::vector<uint8_t> glb;
        std::string err;
        check(trellis::encode_glb_mesh(v, f_out, glb, err), "encoded");
        uint32_t magic_ver[3], json_len = 0;
        std::memcpy(magic_ver, glb.data(), 12);
        std::memcpy(&json_len, &glb[12], 4);
        check(std::memcmp(glb.data(), "glTF", 4) == 0 && magic_ver[1] == 2 && magic_ver[2] == glb.size(), "glTF 2, its length the file's");
        check(std::memcmp(&glb[16], "JSON", 4) == 0 && json_len % 4 == 0, "a JSON chunk, 4-byte aligned");
        const std::string js(reinterpret_cast<const char*>(&glb[20]), json_len);
        check(js.find("\"POSITION\":0") != std::string::npos && js.find("\"NORMAL\":1") != std::string::npos &&
                  js.find("\"indices\":2") != std::string::npos,
              "positions, normals and indices");
        check(js.find("TEXCOORD") == std::string::npos && js.find("\"images\"") == std::string::npos, "no texture coordinates, no images");
        uint32_t bin_len = 0;
        std::memcpy(&bin_len, &glb[20 + json_len], 4);
        check(std::memcmp(&glb[24 + json_len], "BIN", 3) == 0 && bin_len == 8 * 12 + 8 * 12 + 12 * 12, "a BIN chunk of the 3 buffers");
        // the GLB's axes (x, z, -y): the cube's y range [-1, 1] becomes z's, its z range [5, 7] becomes y's
        check(js.find("\"min\":[3,5,-1]") != std::string::npos && js.find("\"max\":[5,7,1]") != std::string::npos,
              "the axes of the textured GLB: (x, y, z) to (x, z, -y)");
    }
    std::printf("%s\n", g_bad ? "FAIL" : "PASS");
    return g_bad ? 1 : 0;
}
