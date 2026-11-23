/* AudioLDM 2's CLAP text encoder (src/audioldm2_clap.cpp) against the diffusers oracle: tools/audioldm2_stage_clap.py writes
 * clap_ids.i32, clap_mask.i32 (512 each), clap_hidden.f32 (512 x 768), clap_pooled.f32 (768), clap_features.f32 (512) to
 * /data/audioldm2-stages. Usage: audioldm2_clap_verify [text_encoder/model.safetensors] [stages dir]. Exit 0 when all match. */
#include "audioldm2_clap.h"
#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static bool read_bytes(const std::string& path, void* out, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    const size_t got = std::fread(out, 1, bytes, f);
    std::fclose(f);
    return got == bytes;
}

static bool read_floats(const std::string& path, std::vector<float>& v) {
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

static bool compare(const char* what, const std::vector<float>& got, const std::vector<float>& ref) {
    double num = 0, den = 0, maxerr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)got[i] - (double)ref[i];
        num += d * d; den += (double)ref[i] * ref[i];
        maxerr = std::fmax(maxerr, std::fabs(d));
    }
    const double rel = std::sqrt(num / den);
    const bool ok = rel <= 1e-3;
    std::printf("%-9s rel L2 %.3e, max abs %.3e  %s\n", what, rel, maxerr, ok ? "ok" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "/data/audioldm2/text_encoder/model.safetensors";
    const std::string dir = argc > 2 ? argv[2] : "/data/audioldm2-stages";
    std::vector<int32_t> ids(512), mask(512);
    std::vector<float> hidden, pooled, features;
    if (!read_bytes(dir + "/clap_ids.i32", ids.data(), 4 * ids.size()) || !read_bytes(dir + "/clap_mask.i32", mask.data(), 4 * mask.size()) ||
        !read_floats(dir + "/clap_hidden.f32", hidden) || !read_floats(dir + "/clap_pooled.f32", pooled) ||
        !read_floats(dir + "/clap_features.f32", features)) {
        std::fprintf(stderr, "cannot read the stage files in %s\n", dir.c_str());
        return 1;
    }
    zimage::SafeTensors st(model);
    if (!st.ok() || !st.register_device()) {
        std::fprintf(stderr, "the CLAP file: %s\n", st.ok() ? "registration failed" : st.error().c_str());
        return 1;
    }
    const int T = (int)ids.size();
    std::vector<float> gh((size_t)T * 768), gp(768), gf(512);
    if (!audioldm2::clap_text_forward(st, ids.data(), mask.data(), T, gh.data(), gp.data(), gf.data())) {
        std::fprintf(stderr, "the CLAP device path failed\n");
        return 1;
    }
    std::printf("CLAP text model, %d tokens (%d valid)\n", T, [&] { int v = 0; for (int m : mask) v += m; return v; }());
    const bool a = compare("hidden", gh, hidden);
    const bool b = compare("pooled", gp, pooled);
    const bool c = compare("features", gf, features);
    return a && b && c ? 0 : 1;
}
