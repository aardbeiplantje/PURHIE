// Stage check of the TRELLIS.2 sparse-structure flow on the device (src/trellis_ss.cpp) against the python model's dumps
// (tools/trellis_ss_oracle.py writes them: the inputs and each stage as raw float32; tools/trellis_ss_block0_oracle.py adds
// the block-0 taps "dbg_*"). The engine computes in bf16 (linear_bf16 rounds its activations), the oracle in fp32, so the
// tolerance is the bf16 budget. Usage:
//   trellis_ss_verify [oracle_dir] [safetensors]
// Prints the relative L2 error of every stage and of the output; exit 1 when a stage is off by more than the tolerance.
#include "trellis_ss.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static bool read_f32(const std::string& path, std::vector<float>& v) {
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

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_ss";
    const std::string ckpt = argc > 2 ? argv[2] : "/data/trellis2/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors";
    const double tol = 1e-2;  // bf16 activations in the linears; the oracle is fp32

    std::vector<float> x, t, cond, rope, out_ref;
    if (!read_f32(dir + "/x.f32", x) || !read_f32(dir + "/t.f32", t) || !read_f32(dir + "/cond.f32", cond) ||
        !read_f32(dir + "/rope_phases.f32", rope) || !read_f32(dir + "/out.f32", out_ref)) {
        std::fprintf(stderr, "cannot read the oracle inputs in %s\n", dir.c_str());
        return 1;
    }
    const int Lc = (int)(cond.size() / 1024);

    trellis::SsFlow flow(ckpt);
    if (!flow.ok()) {
        std::fprintf(stderr, "open: %s\n", flow.error().c_str());
        return 1;
    }

    int bad = 0;
    auto compare = [&](const char* name, const float* got, size_t n, const std::vector<float>& ref) {
        if (ref.size() != n) {
            std::printf("%-16s size %zu, oracle %zu: CHECK\n", name, n, ref.size());
            ++bad;
            return;
        }
        double num = 0, den = 0;
        for (size_t i = 0; i < n; ++i) {
            const double e = (double)got[i] - (double)ref[i];
            num += e * e;
            den += (double)ref[i] * ref[i];
        }
        const double rel = std::sqrt(num / (den > 0 ? den : 1.0));
        const bool ok = rel < tol;
        std::printf("%-16s rel L2 %.3e  %s\n", name, rel, ok ? "ok" : "CHECK");
        if (!ok) ++bad;
    };

    auto stage = [&](const char* name, const float* data, size_t n) {
        std::vector<float> ref;
        if (!read_f32(dir + "/" + name + ".f32", ref)) {
            std::printf("%-16s no oracle file\n", name);
            ++bad;
            return;
        }
        compare(name, data, n, ref);
    };

    std::vector<float> out((size_t)8 * 4096);
    const bool ran = flow.forward(x.data(), t[0], cond.data(), Lc, rope.data(), out.data(), stage);
    if (!ran) {
        std::fprintf(stderr, "forward: %s\n", flow.error().c_str());
        return 1;
    }
    compare("out", out.data(), out.size(), out_ref);
    std::printf("%s: %d check(s) over the tolerance %.0e\n", bad ? "FAIL" : "PASS", bad, tol);
    return bad ? 1 : 0;
}
