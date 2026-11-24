// Stage check of the TRELLIS.2 sparse-structure decoder on the device (src/trellis_ss_dec.cpp) against the python model's
// dumps (tools/trellis_ss_decoder_oracle.py: the input is the flow's output out.f32, and each stage as raw float32,
// channel-major). The engine's stages are token-major; they are transposed here before the comparison. Usage:
//   trellis_ss_dec_verify [oracle_dir] [safetensors]
// The oracle is fp32 and the engine's arithmetic is fp32 (the fp16 weights upcast), so the tolerance is tight.
#include "trellis_ss_dec.h"

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

// token-major [V][C] to channel-major [C][V]
static std::vector<float> to_channel_major(const float* tok, size_t V, size_t C) {
    std::vector<float> out(V * C);
    for (size_t v = 0; v < V; ++v)
        for (size_t c = 0; c < C; ++c) out[c * V + v] = tok[v * C + c];
    return out;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_ss";
    const std::string ckpt = argc > 2 ? argv[2] : "/data/trellis2/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors";
    const double tol = 1e-3;

    std::vector<float> z, out_ref;
    if (!read_f32(dir + "/out.f32", z) || z.size() != 8 * 4096) {
        std::fprintf(stderr, "cannot read the flow output (out.f32) in %s\n", dir.c_str());
        return 1;
    }
    trellis::SsDecoder dec(ckpt);
    if (!dec.ok()) {
        std::fprintf(stderr, "open: %s\n", dec.error().c_str());
        return 1;
    }

    int bad = 0;
    auto compare = [&](const char* name, const std::vector<float>& got, const std::vector<float>& ref) {
        if (ref.size() != got.size()) {
            std::printf("%-10s size %zu, oracle %zu: CHECK\n", name, got.size(), ref.size());
            ++bad;
            return;
        }
        double num = 0, den = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double e = (double)got[i] - (double)ref[i];
            num += e * e;
            den += (double)ref[i] * ref[i];
        }
        const double rel = std::sqrt(num / (den > 0 ? den : 1.0));
        const bool ok = rel < tol;
        std::printf("%-10s rel L2 %.3e  %s\n", name, rel, ok ? "ok" : "CHECK");
        if (!ok) ++bad;
    };

    // a stage: the engine's token-major [V][C] against the oracle's channel-major [C][V]
    auto stage = [&](const char* name, const float* data, size_t n) {
        std::vector<float> ref;
        if (!read_f32(dir + "/" + std::string(name) + ".f32", ref)) {
            std::printf("%-10s no oracle file\n", name);
            ++bad;
            return;
        }
        // the stage layout: V voxels by C channels (the oracle has the same n elements)
        size_t V = 0, Cc = 0;
        if (std::string(name) == "dec_in" || std::string(name) == "dec_mid") V = 4096, Cc = 512;
        else if (std::string(name) == "dec_b2") V = 32768, Cc = 128;
        else if (std::string(name) == "dec_b5") V = 262144, Cc = 32;
        else V = 262144, Cc = 1;
        if (V * Cc != n) {
            std::printf("%-10s stage size %zu does not match the layout: CHECK\n", name, n);
            ++bad;
            return;
        }
        compare(name, to_channel_major(data, V, Cc), ref);
    };

    std::vector<float> out((size_t)64 * 64 * 64);
    if (!dec.forward(z.data(), out.data(), stage)) {
        std::fprintf(stderr, "forward: %s\n", dec.error().c_str());
        return 1;
    }
    read_f32(dir + "/dec_out.f32", out_ref);
    compare("out", out, out_ref);
    std::printf("%s: %d check(s) over the tolerance %.0e\n", bad ? "FAIL" : "PASS", bad, tol);
    return bad ? 1 : 0;
}
