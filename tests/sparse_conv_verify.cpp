// Check of the engine's submanifold sparse convolution (sd_gpu.hip: sp_hash_insert, sp_neighbor_map, sp_subm_conv_f16w)
// against FlexGEMM (tools/sparse_conv_oracle.py writes the inputs and the output). Usage: sparse_conv_verify [oracle_dir]
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
    std::vector<char> xb, cb, wb, bb, yb;
    if (!read_bytes(dir + "/sc_x.f32", xb) || !read_bytes(dir + "/sc_coords.i32", cb) || !read_bytes(dir + "/sc_w.u16", wb) ||
        !read_bytes(dir + "/sc_b.f32", bb) || !read_bytes(dir + "/sc_y.f32", yb)) {
        std::fprintf(stderr, "cannot read the oracle in %s\n", dir.c_str());
        return 1;
    }
    if (!sddev::enabled()) {
        std::fprintf(stderr, "no device\n");
        return 1;
    }
    const int N = (int)(cb.size() / (4 * sizeof(int)));
    const int CI = (int)(xb.size() / (sizeof(float) * (size_t)N));
    const int CO = (int)(bb.size() / sizeof(float));
    unsigned int tsize = 1;
    while (tsize < 2u * (unsigned)N) tsize <<= 1;

    int* coords = static_cast<int*>(sddev::alloc(cb.size(), false));
    float* x = static_cast<float*>(sddev::alloc(xb.size(), false));
    uint16_t* W = static_cast<uint16_t*>(sddev::alloc(wb.size(), false));
    float* b = static_cast<float*>(sddev::alloc(bb.size(), false));
    unsigned long long* keys = static_cast<unsigned long long*>(sddev::alloc(sizeof(unsigned long long) * tsize, false));
    int* vals = static_cast<int*>(sddev::alloc(sizeof(int) * tsize, false));
    int* nb = static_cast<int*>(sddev::alloc(sizeof(int) * (size_t)N * 27, false));
    float* y = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)N * CO, false));
    std::memcpy(coords, cb.data(), cb.size());
    std::memcpy(x, xb.data(), xb.size());
    std::memcpy(W, wb.data(), wb.size());
    std::memcpy(b, bb.data(), bb.size());

    sddev::sp_hash_build(coords, N, keys, vals, tsize);
    sddev::sp_neighbor_map(coords, N, keys, vals, tsize, nb);
    sddev::sp_subm_conv_f16w(x, N, CI, nb, W, b, CO, y, 0, CO);

    std::vector<float> ref((size_t)N * CO);
    std::memcpy(ref.data(), yb.data(), sizeof(float) * ref.size());
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = (double)y[i] - (double)ref[i];
        num += e * e;
        den += (double)ref[i] * ref[i];
    }
    const double rel = std::sqrt(num / den);
    int hits = 0;
    for (int i = 0; i < N * 27; ++i) hits += nb[i] >= 0;
    std::printf("sparse submanifold conv (%d voxels, %d -> %d, table %u): rel L2 %.3e, neighbours %d of %d\n", N, CI, CO,
                tsize, rel, hits, N * 27);
    std::printf("%s\n", rel < 1e-4 ? "PASS" : "FAIL");
    sddev::release(coords, cb.size());
    sddev::release(x, xb.size());
    sddev::release(W, wb.size());
    sddev::release(b, bb.size());
    sddev::release(keys, sizeof(unsigned long long) * tsize);
    sddev::release(vals, sizeof(int) * tsize);
    sddev::release(nb, sizeof(int) * (size_t)N * 27);
    sddev::release(y, sizeof(float) * (size_t)N * CO);
    return rel < 1e-4 ? 0 : 1;
}
