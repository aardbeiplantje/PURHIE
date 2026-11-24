// Stage check of the TRELLIS.2 shape-SLAT flow on the device (src/trellis_slat.cpp) against the python model's dumps
// (tools/trellis_slat_oracle.py: the inputs and each stage as raw float32, token-major, in the model's voxel order).
// The engine computes in bf16 (linear_bf16 rounds its activations) and the oracle in fp32, so the tolerance is the bf16
// budget. Usage: trellis_slat_verify [oracle_dir] [safetensors]
#include "trellis_slat.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static bool read_raw(const std::string& path, std::vector<float>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)bytes / sizeof(float));
    const size_t got = std::fread(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return got == v.size();
}

static bool read_i32(const std::string& path, std::vector<int32_t>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)bytes / sizeof(int32_t));
    const size_t got = std::fread(v.data(), sizeof(int32_t), v.size(), f);
    std::fclose(f);
    return got == v.size();
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_slat";
    const std::string ckpt = argc > 2 ? argv[2] : "/data/trellis2/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors";
    const double tol = 1e-2;  // the bf16 budget (see the header)

    std::vector<int32_t> coords;
    std::vector<float> feats, cond, t;
    if (!read_i32(dir + "/slat_coords.i32", coords) || !read_raw(dir + "/slat_feats.f32", feats) ||
        !read_raw(dir + "/slat_cond.f32", cond) || !read_raw(dir + "/slat_t.f32", t)) {
        std::fprintf(stderr, "cannot read the oracle inputs in %s\n", dir.c_str());
        return 1;
    }
    const int n = (int)(coords.size() / 4);
    const int Lc = (int)(cond.size() / 1024);

    trellis::SlatFlow flow(ckpt);
    if (!flow.ok()) {
        std::fprintf(stderr, "open: %s\n", flow.error().c_str());
        return 1;
    }

    int bad = 0;
    auto compare = [&](const char* name, const float* got, size_t cnt) {
        std::vector<float> ref;
        if (!read_raw(dir + "/" + std::string(name) + ".f32", ref)) {
            std::printf("%-14s no oracle file\n", name);
            ++bad;
            return;
        }
        if (ref.size() != cnt) {
            std::printf("%-14s size %zu, oracle %zu: CHECK\n", name, cnt, ref.size());
            ++bad;
            return;
        }
        double num = 0, den = 0;
        for (size_t i = 0; i < cnt; ++i) {
            const double e = (double)got[i] - (double)ref[i];
            num += e * e;
            den += (double)ref[i] * ref[i];
        }
        const double rel = std::sqrt(num / (den > 0 ? den : 1.0));
        const bool ok = rel < tol;
        std::printf("%-14s rel L2 %.3e  %s\n", name, rel, ok ? "ok" : "CHECK");
        if (!ok) ++bad;
    };

    std::vector<float> out((size_t)n * 32);
    const bool ran = flow.forward(coords.data(), n, feats.data(), t[0], cond.data(), Lc, out.data(),
                                  [&](const char* name, const float* data, size_t cnt) { compare(name, data, cnt); });
    if (!ran) {
        std::fprintf(stderr, "forward: %s\n", flow.error().c_str());
        return 1;
    }
    std::printf("%s: %d check(s) over the tolerance %.0e (%d voxels)\n", bad ? "FAIL" : "PASS", bad, tol, n);
    return bad ? 1 : 0;
}
