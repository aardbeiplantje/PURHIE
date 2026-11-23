/* AudioLDM 2's vocoder (src/audioldm2_vocoder.cpp) against the diffusers/transformers oracle: tools/audioldm2_stage_vocoder.py writes
 * voc_mel.f32 (T x 64) and voc_out.f32 (the waveform). Usage: audioldm2_vocoder_verify [vocoder/model.safetensors] [stages dir]. */
#include "audioldm2_vocoder.h"
#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

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

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "/data/audioldm2/vocoder/model.safetensors";
    const std::string dir = argc > 2 ? argv[2] : "/data/audioldm2-stages";
    std::vector<float> mel, ref;
    if (!read_floats(dir + "/voc_mel.f32", mel) || !read_floats(dir + "/voc_out.f32", ref)) {
        std::fprintf(stderr, "cannot read the stage files in %s\n", dir.c_str());
        return 1;
    }
    zimage::SafeTensors st(model);
    if (!st.ok() || !st.register_device()) {
        std::fprintf(stderr, "the vocoder file: %s\n", st.ok() ? "registration failed" : st.error().c_str());
        return 1;
    }
    std::vector<float> got;
    const int T = (int)(mel.size() / 64);
    if (!audioldm2::vocoder_forward(st, mel.data(), T, got)) {
        std::fprintf(stderr, "the vocoder device path failed\n");
        return 1;
    }
    if (got.size() != ref.size()) {
        std::printf("vocoder: %zu samples, the reference has %zu  FAIL\n", got.size(), ref.size());
        return 1;
    }
    double num = 0, den = 0, maxerr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)got[i] - ref[i];
        num += d * d; den += (double)ref[i] * ref[i];
        if (std::fabs(d) > maxerr) maxerr = std::fabs(d);
    }
    const double rel = std::sqrt(num / den);
    std::printf("vocoder, %d frames -> %zu samples: rel L2 %.3e, max abs %.3e  %s\n", T, got.size(), rel, maxerr, rel <= 1e-3 ? "ok" : "FAIL");
    return rel <= 1e-3 ? 0 : 1;
}
