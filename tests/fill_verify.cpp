// Check of the engine's hole filling (src/trellis_fill.cpp) against CuMesh's fill_holes (tools/fill_holes_oracle.py, on the 512 shape
// oracle's mesh at max_hole_perimeter 3e-2). The loop counts, the vertex and face counts must match; the new vertices (CuMesh's loop
// order differs from ours) are compared by position, and the triangles by their vertices' positions. Usage: fill_verify [shape_oracle]
// [fill_oracle]
#include "trellis_fill.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

static bool read_bytes(const std::string& path, std::vector<char>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)bytes);
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    return got == v.size();
}

using P = std::tuple<float, float, float>;

int main(int argc, char** argv) {
    const std::string sdir = argc > 1 ? argv[1] : "/data/trellis2/oracle_shape";
    const std::string fdir = argc > 2 ? argv[2] : "/data/trellis2/oracle_fill";
    std::vector<char> vb, fb, rvb, rfb;
    if (!read_bytes(sdir + "/mesh_vertices.f32", vb) || !read_bytes(sdir + "/mesh_faces.i32", fb) ||
        !read_bytes(fdir + "/filled_vertices.f32", rvb) || !read_bytes(fdir + "/filled_faces.i32", rfb)) {
        std::fprintf(stderr, "cannot read the oracles in %s and %s\n", sdir.c_str(), fdir.c_str());
        return 1;
    }
    std::vector<float> verts(vb.size() / sizeof(float));
    std::memcpy(verts.data(), vb.data(), vb.size());
    std::vector<int32_t> faces(fb.size() / sizeof(int32_t));
    std::memcpy(faces.data(), fb.data(), fb.size());
    const size_t V_in = verts.size() / 3;
    std::vector<float> rv(rvb.size() / sizeof(float));
    std::memcpy(rv.data(), rvb.data(), rvb.size());
    std::vector<int32_t> rf(rfb.size() / sizeof(int32_t));
    std::memcpy(rf.data(), rfb.data(), rfb.size());

    trellis::FillStats st;
    trellis::fill_small_holes(verts, faces, 3e-2f, st);
    std::printf("boundary edges %d, loops %d: filled %d, too long %d\n", st.boundary_edges, st.loops, st.filled, st.too_long);

    const size_t V = verts.size() / 3, F = faces.size() / 3;
    const size_t Vr = rv.size() / 3, Fr = rf.size() / 3;
    const bool counts = V == Vr && F == Fr;
    std::printf("vertices %zu (CuMesh %zu), faces %zu (CuMesh %zu): %s\n", V, Vr, F, Fr, counts ? "equal" : "DIFFER");
    bool ok = counts && st.boundary_edges == 30725 && st.loops == 630 && st.filled == 618;

    // the new vertices by position (both sides sorted), the triangles by their vertices' ranks among the new vertices
    std::vector<P> ours, theirs;
    for (size_t i = V_in; i < V; ++i) ours.emplace_back(verts[3 * i], verts[3 * i + 1], verts[3 * i + 2]);
    for (size_t i = V_in; i < Vr; ++i) theirs.emplace_back(rv[3 * i], rv[3 * i + 1], rv[3 * i + 2]);
    std::sort(ours.begin(), ours.end());
    std::sort(theirs.begin(), theirs.end());
    double max_pos = 0;
    if (ours.size() == theirs.size())
        for (size_t i = 0; i < ours.size(); ++i)
            max_pos = std::max(max_pos, (double)std::fabs(std::get<0>(ours[i]) - std::get<0>(theirs[i])) +
                                            std::fabs(std::get<1>(ours[i]) - std::get<1>(theirs[i])) +
                                            std::fabs(std::get<2>(ours[i]) - std::get<2>(theirs[i])));
    std::printf("new vertices: %zu (CuMesh %zu), largest position difference %.3e\n", ours.size(), theirs.size(), max_pos);
    ok = ok && ours.size() == theirs.size() && max_pos < 1e-5;

    // the triangles as sets: each vertex by its position (the input's by index, the new ones by their sorted position)
    auto canon = [&](const std::vector<float>& vs, const std::vector<int32_t>& fs, size_t Vi, const std::vector<P>& news) {
        std::vector<std::array<int64_t, 3>> tris;
        for (size_t t = 0; t < fs.size() / 3; ++t) {
            std::array<int64_t, 3> k{};
            for (int c = 0; c < 3; ++c) {
                const int32_t id = fs[3 * t + c];
                if ((size_t)id < Vi) {
                    k[c] = id;
                } else {
                    const P p(vs[3 * id], vs[3 * id + 1], vs[3 * id + 2]);
                    k[c] = (int64_t)(Vi + (size_t)(std::lower_bound(news.begin(), news.end(), p) - news.begin()));
                }
            }
            std::sort(k.begin(), k.end());
            tris.push_back(k);
        }
        std::sort(tris.begin(), tris.end());
        return tris;
    };
    const auto ta = canon(verts, faces, V_in, ours);
    const auto tb = canon(rv, rf, V_in, theirs);
    const bool same = ta == tb;
    std::printf("triangles as sets: %s\n", same ? "equal" : "DIFFER");
    ok = ok && same;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
