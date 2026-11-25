// Check of the engine's image encoder (src/dinov3.cpp: DINOv3 ViT-L/16 from ComfyUI's local copy of the weights) against the transformers
// reference (tools/dinov3_oracle.py: the same weights and the TRELLIS feature path, fp32). Usage: dinov3_verify [oracle_dino] [weights]
#include "dinov3.h"

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

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_dino";
    const std::string weights = argc > 2 ? argv[2] : "/comfyui/clip_vision/dino_v3_L_naf_fp32.safetensors";
    std::vector<char> xb, tb;
    if (!read_bytes(dir + "/input.f32", xb) || !read_bytes(dir + "/tokens.f32", tb)) {
        std::fprintf(stderr, "cannot read the oracle in %s\n", dir.c_str());
        return 1;
    }
    const int S = 512;
    const size_t T = trellis::DinoV3::tokens(S);
    if (xb.size() != 3 * (size_t)S * S * sizeof(float) || tb.size() != T * 1024 * sizeof(float)) {
        std::fprintf(stderr, "the oracle does not match a %d px image\n", S);
        return 1;
    }
    const float* x = reinterpret_cast<const float*>(xb.data());
    const float* ref = reinterpret_cast<const float*>(tb.data());

    trellis::DinoV3 enc(weights);
    if (!enc.ok()) {
        std::fprintf(stderr, "open: %s\n", enc.error().c_str());
        return 1;
    }
    std::vector<float> out(T * 1024);
    if (!enc.forward(x, S, out.data())) {
        std::fprintf(stderr, "forward: %s\n", enc.error().c_str());
        return 1;
    }
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double e = (double)out[i] - ref[i];
        num += e * e;
        den += (double)ref[i] * ref[i];
        mx = std::fmax(mx, std::fabs(e));
    }
    const double rel = std::sqrt(num / den);
    std::printf("image encoder: %zu tokens, rel L2 %.3e, max abs %.3e\n", T, rel, mx);
    const bool ok = rel < 1e-3;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
