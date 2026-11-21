// zimage_dit_verify: the Z-Image DiT's first stages (src/zimage_dit.cpp, host path) against the reference's stage outputs.
//
//   zimage_dit_verify <dit.safetensors> <raw dir>
//
// raw dir: the oracle as raw float32 files (export from /data/zimage-ref/dit: x.f32 [16][32][32] the latent, cap.f32 [15][2560]
// the caption's embeddings, t.f32 the step (0.5), t_embedder.f32 [256], x_embedder.f32 [256][3840], cap_embedder.f32
// [32][3840]). Each stage prints its max abs error and cosine against the reference. Build (host only):
//   clang++ -O2 -std=c++17 -Isrc tests/zimage_dit_verify.cpp src/zimage_dit.cpp -o zimage_dit_verify
#include "zimage_dit.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static bool read_raw(const std::string& path, std::vector<float>& v) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t)n / sizeof(float));
    const size_t got = fread(v.data(), 1, (size_t)n, f);
    fclose(f);
    return got == (size_t)n;
}

static int compare(const char* what, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    if (got.size() != want.size()) {
        printf("[FAIL] %s: %zu values, the reference has %zu\n", what, got.size(), want.size());
        return 1;
    }
    double maxabs = 0, dot = 0, ng = 0, nw = 0, maxref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double a = got[i], b = want[i];
        maxabs = std::fmax(maxabs, std::fabs(a - b));
        maxref = std::fmax(maxref, std::fabs(b));
        dot += a * b; ng += a * a; nw += b * b;
    }
    const double cos = dot / std::sqrt(ng * nw + 1e-30);
    const bool ok = maxabs <= tol * std::fmax(maxref, 1e-12) && cos > 0.9999;
    printf("[%s] %s: max abs %.3g (ref max %.3g, relative %.2g), cosine %.7f\n", ok ? "PASS" : "FAIL", what, maxabs,
           maxref, maxabs / std::fmax(maxref, 1e-12), cos);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s dit.safetensors raw_dir\n", argv[0]); return 2; }
    const std::string dir = argv[2];
    zimage::SafeTensors st(argv[1]);
    if (!st.ok()) { fprintf(stderr, "%s\n", st.error().c_str()); return 1; }
    std::vector<float> x, cap, t, want;
    if (!read_raw(dir + "/x.f32", x) || !read_raw(dir + "/cap.f32", cap) || !read_raw(dir + "/t.f32", t)) {
        fprintf(stderr, "cannot read the inputs in %s\n", dir.c_str());
        return 1;
    }
    int fails = 0;
    /* the reference's bf16 rounding of the timestep embedding is the expected difference: 2e-2 relative */
    std::vector<float> temb(256);
    zimage::timestep_embed(st, t[0], temb.data());
    if (read_raw(dir + "/t_embedder.f32", want)) fails += compare("t_embedder", temb, want, 2e-2);

    std::vector<float> tokens;
    zimage::patchify(x.data(), 16, 32, 32, 2, tokens);
    std::vector<float> xe;
    zimage::patch_embed(st, tokens, (int)(tokens.size() / 64), xe);
    if (read_raw(dir + "/x_embedder.f32", want)) fails += compare("x_embedder (patchify + patch embedding)", xe, want, 2e-2);

    std::vector<float> ce;
    int Lpad = 0;
    zimage::caption_embed(st, cap.data(), (int)(cap.size() / 2560), ce, Lpad);
    if (read_raw(dir + "/cap_embedder.f32", want)) fails += compare("cap_embedder (RMSNorm + linear, padded)", ce, want, 2e-2);

    printf("%s\n", fails ? "FAILED" : "all stages PASS");
    return fails ? 1 : 0;
}
