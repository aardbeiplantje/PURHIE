// Check of the engine's channel-to-spatial upsampling (sd_gpu.hip: sp_c2s_mask, sp_exclusive_scan, sp_c2s_fill,
// sp_c2s_gather) against the python SparseChannel2Spatial (tools/c2s_oracle.py). The coordinates must match exactly, the
// features to fp32 rounding. Usage: c2s_verify [oracle_dir]
#include "sd_device.h"

#include <cmath>
#include <cstdint>
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

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_sparse";
    std::vector<char> fb, cb, lb, ocb, ofb;
    if (!read_bytes(dir + "/c2s_feats.f32", fb) || !read_bytes(dir + "/c2s_coords.i32", cb) ||
        !read_bytes(dir + "/c2s_logit.f32", lb) || !read_bytes(dir + "/c2s_out_coords.i32", ocb) ||
        !read_bytes(dir + "/c2s_out_feats.f32", ofb)) {
        std::fprintf(stderr, "cannot read the oracle in %s\n", dir.c_str());
        return 1;
    }
    if (!sddev::enabled()) {
        std::fprintf(stderr, "no device\n");
        return 1;
    }
    const int N = (int)(cb.size() / (4 * sizeof(int)));
    const int C = (int)(fb.size() / (sizeof(float) * 8 * (size_t)N));
    const int M_ref = (int)(ocb.size() / (4 * sizeof(int)));

    int* coords = static_cast<int*>(sddev::alloc(cb.size(), false));
    float* logit = static_cast<float*>(sddev::alloc(lb.size(), false));
    float* feats = static_cast<float*>(sddev::alloc(fb.size(), false));
    unsigned char* bits = static_cast<unsigned char*>(sddev::alloc(N, false));
    int* cnt = static_cast<int*>(sddev::alloc(sizeof(int) * N, false));
    int* off = static_cast<int*>(sddev::alloc(sizeof(int) * (N + 1), false));
    std::memcpy(coords, cb.data(), cb.size());
    std::memcpy(logit, lb.data(), lb.size());
    std::memcpy(feats, fb.data(), fb.size());

    sddev::sp_c2s_mask(logit, N, bits, cnt);
    sddev::sp_exclusive_scan(cnt, N, off);
    const int M = off[N];  // the scan's total, read on the host after the sync
    int* new_coords = static_cast<int*>(sddev::alloc(sizeof(int) * 4 * (size_t)M, false));
    int* idx = static_cast<int*>(sddev::alloc(sizeof(int) * (size_t)M, false));
    int* subidx = static_cast<int*>(sddev::alloc(sizeof(int) * (size_t)M, false));
    float* out = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)M * C, false));
    sddev::sp_c2s_fill(coords, bits, off, N, new_coords, idx, subidx);
    sddev::sp_c2s_gather(feats, idx, subidx, M, C, 1, out);

    int coord_bad = (M != M_ref);
    if (!coord_bad) {
        const int* ref = reinterpret_cast<const int*>(ocb.data());
        for (size_t i = 0; i < (size_t)4 * M; ++i)
            if (new_coords[i] != ref[i]) {
                ++coord_bad;
                break;
            }
    }
    std::vector<float> ref_f((size_t)M_ref * C);
    std::memcpy(ref_f.data(), ofb.data(), sizeof(float) * ref_f.size());
    double num = 0, den = 0;
    for (size_t i = 0; i < ref_f.size() && M == M_ref; ++i) {
        const double e = (double)out[i] - (double)ref_f[i];
        num += e * e;
        den += (double)ref_f[i] * ref_f[i];
    }
    const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
    std::printf("channel-to-spatial (%d voxels -> %d children, %d channels): coords %s, feats rel L2 %.3e\n", N, M, C,
                coord_bad ? "DIFFER" : "exact", rel);
    const bool ok = !coord_bad && rel < 1e-6;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    sddev::release(coords, cb.size());
    sddev::release(logit, lb.size());
    sddev::release(feats, fb.size());
    sddev::release(bits, N);
    sddev::release(cnt, sizeof(int) * N);
    sddev::release(off, sizeof(int) * (N + 1));
    sddev::release(new_coords, sizeof(int) * 4 * (size_t)M);
    sddev::release(idx, sizeof(int) * (size_t)M);
    sddev::release(subidx, sizeof(int) * (size_t)M);
    sddev::release(out, sizeof(float) * (size_t)M * C);
    return ok ? 0 : 1;
}
