// Check of the engine's shape decoder sparse UNet (src/trellis_sdec.cpp) against the python decoder (tools/sdec_oracle.py):
// the level 0 output, then the final voxels (exact order) and their features. Usage: sdec_verify [oracle_dir] [safetensors]
#include "trellis_sdec.h"

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

static double rel_l2(const float* got, const float* ref, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = (double)got[i] - (double)ref[i];
        num += e * e;
        den += (double)ref[i] * ref[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1.0));
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_sparse";
    const std::string ckpt = argc > 2 ? argv[2] : "/data/trellis2/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors";
    std::vector<char> inf, inc, l0f, l0c, outf, outc;
    if (!read_bytes(dir + "/sdec_in_feats.f32", inf) || !read_bytes(dir + "/sdec_in_coords.i32", inc) ||
        !read_bytes(dir + "/sdec_l0_feats.f32", l0f) || !read_bytes(dir + "/sdec_l0_coords.i32", l0c) ||
        !read_bytes(dir + "/sdec_out_feats.f32", outf) || !read_bytes(dir + "/sdec_out_coords.i32", outc)) {
        std::fprintf(stderr, "cannot read the oracle in %s\n", dir.c_str());
        return 1;
    }
    const int n = (int)(inc.size() / (4 * sizeof(int32_t)));
    const int M_ref = (int)(outc.size() / (4 * sizeof(int32_t)));
    const int M0_ref = (int)(l0c.size() / (4 * sizeof(int32_t)));

    trellis::ShapeDecoder dec(ckpt);
    if (!dec.ok()) {
        std::fprintf(stderr, "open: %s\n", dec.error().c_str());
        return 1;
    }
    bool ok = true;
    auto stage = [&](const char* name, const float* feats, const int32_t* coords, size_t cnt, size_t C) {
        const bool same_count = (int)cnt == M0_ref && C == l0f.size() / (sizeof(float) * (size_t)M0_ref);
        const bool coords_same = same_count && std::memcmp(coords, l0c.data(), l0c.size()) == 0;
        const double rel = same_count ? rel_l2(feats, reinterpret_cast<const float*>(l0f.data()), cnt * C) : -1.0;
        std::printf("%-10s %d voxels x %zu: coords %s, feats rel L2 %.3e\n", name, (int)cnt, C,
                    coords_same ? "exact" : "DIFFER", rel);
        ok = ok && coords_same && rel >= 0 && rel < 1e-4;
    };

    const int cap = 4 * M_ref + 1024;
    std::vector<float> out_feats((size_t)cap * 7);
    std::vector<int32_t> out_coords((size_t)cap * 4);
    int M = 0;
    if (!dec.forward(reinterpret_cast<const int32_t*>(inc.data()), n, reinterpret_cast<const float*>(inf.data()),
                     out_feats.data(), out_coords.data(), &M, cap, stage)) {
        std::fprintf(stderr, "forward: %s\n", dec.error().c_str());
        return 1;
    }
    const bool same_count = M == M_ref;
    const bool coords_same = same_count && std::memcmp(out_coords.data(), outc.data(), outc.size()) == 0;
    const double rel = same_count ? rel_l2(out_feats.data(), reinterpret_cast<const float*>(outf.data()), (size_t)M * 7) : -1.0;
    std::printf("output     %d voxels (oracle %d): coords %s, feats rel L2 %.3e\n", M, M_ref,
                coords_same ? "exact" : "DIFFER", rel);
    ok = ok && coords_same && rel >= 0 && rel < 1e-4;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
