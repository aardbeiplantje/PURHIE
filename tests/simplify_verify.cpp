// Check of the engine's mesh simplification (src/trellis_simplify.cpp) against CuMesh's simplify (tools/simplify_oracle.py, the filled
// shape mesh to 100000 faces): the face count, the validity of the result (indices in range, finite positions), the passes, and the
// Chamfer distance between the two simplified surfaces (box units). Usage: simplify_verify [oracle_fill] [oracle_simplify] [target]
#include "trellis_simplify.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
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

static double chamfer(const float* p, size_t np, const float* q, size_t nq, size_t cap) {
    auto one_way = [&](const float* A, size_t na, const float* B, size_t nb) {
        const size_t sa = na > cap ? na / cap : 1, sb = nb > cap ? nb / cap : 1;
        double sum = 0;
        size_t cnt = 0;
        for (size_t i = 0; i < na; i += sa) {
            double best = 1e30;
            for (size_t j = 0; j < nb; j += sb) {
                const double dx = A[3 * i] - B[3 * j], dy = A[3 * i + 1] - B[3 * j + 1], dz = A[3 * i + 2] - B[3 * j + 2];
                best = std::fmin(best, dx * dx + dy * dy + dz * dz);
            }
            sum += std::sqrt(best);
            ++cnt;
        }
        return cnt ? sum / (double)cnt : 0.0;
    };
    return 0.5 * (one_way(p, np, q, nq) + one_way(q, nq, p, np));
}

int main(int argc, char** argv) {
    const std::string fdir = argc > 1 ? argv[1] : "/data/trellis2/oracle_fill";
    const std::string sdir = argc > 2 ? argv[2] : "/data/trellis2/oracle_simplify";
    const int target = argc > 3 ? std::atoi(argv[3]) : 100000;
    std::vector<char> vb, fb, rvb, rfb;
    if (!read_bytes(fdir + "/filled_vertices.f32", vb) || !read_bytes(fdir + "/filled_faces.i32", fb) ||
        !read_bytes(sdir + "/simplified_vertices.f32", rvb) || !read_bytes(sdir + "/simplified_faces.i32", rfb)) {
        std::fprintf(stderr, "cannot read the oracles in %s and %s\n", fdir.c_str(), sdir.c_str());
        return 1;
    }
    std::vector<float> verts(vb.size() / 4), rv(rvb.size() / 4);
    std::memcpy(verts.data(), vb.data(), vb.size());
    std::memcpy(rv.data(), rvb.data(), rvb.size());
    std::vector<int32_t> faces(fb.size() / 4), rf(rfb.size() / 4);
    std::memcpy(faces.data(), fb.data(), fb.size());
    std::memcpy(rf.data(), rfb.data(), rfb.size());

    trellis::SimplifyStats st;
    std::string error;
    if (!trellis::simplify_to_faces(verts, faces, target, st, error)) {
        std::fprintf(stderr, "simplify: %s\n", error.c_str());
        return 1;
    }
    const size_t V = verts.size() / 3, F = faces.size() / 3;
    bool valid = true;
    for (int32_t id : faces) valid = valid && id >= 0 && (size_t)id < V;
    for (float x : verts) valid = valid && std::isfinite(x);
    const double cham = chamfer(verts.data(), V, rv.data(), rv.size() / 3, 20000);
    std::printf("engine: %zu vertices, %zu faces in %d passes (%.2f s); CuMesh: %zu vertices, %zu faces\n", V, F, st.passes, st.seconds,
                rv.size() / 3, rf.size() / 3);
    std::printf("valid: %s, Chamfer to CuMesh's simplified surface %.4f (box units)\n", valid ? "yes" : "NO", cham);
    const bool ok = valid && F <= (size_t)target && F + 0.05 * target >= (double)target * 0.95 && cham < 0.01;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
