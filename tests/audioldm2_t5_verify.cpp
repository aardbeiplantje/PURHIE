/* AudioLDM 2's T5 encoder (src/audioldm2_t5.cpp) against the diffusers oracle: tools/audioldm2_stage_t5.py writes t5_ids.i32 (L)
 * and t5_hidden.f32 (L x 1024) to /data/audioldm2-stages. Usage: audioldm2_t5_verify [text_encoder_2/model.safetensors] [stages dir]. */
#include "audioldm2_t5.h"
#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "/data/audioldm2/text_encoder_2/model.safetensors";
    const std::string dir = argc > 2 ? argv[2] : "/data/audioldm2-stages";
    std::vector<int32_t> ids;
    std::vector<float> ref;
    FILE* f = std::fopen((dir + "/t5_ids.i32").c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s/t5_ids.i32\n", dir.c_str()); return 1; }
    std::fseek(f, 0, SEEK_END);
    ids.resize((size_t)std::ftell(f) / 4);
    std::fseek(f, 0, SEEK_SET);
    const size_t got_ids = std::fread(ids.data(), 4, ids.size(), f);
    std::fclose(f);
    f = std::fopen((dir + "/t5_hidden.f32").c_str(), "rb");
    if (!f || got_ids != ids.size()) { std::fprintf(stderr, "cannot read the hidden states\n"); return 1; }
    std::fseek(f, 0, SEEK_END);
    ref.resize((size_t)std::ftell(f) / 4);
    std::fseek(f, 0, SEEK_SET);
    const size_t got_ref = std::fread(ref.data(), 4, ref.size(), f);
    std::fclose(f);
    if (got_ref != ref.size()) return 1;
    zimage::SafeTensors st(model);
    if (!st.ok() || !st.register_device()) {
        std::fprintf(stderr, "the T5 file: %s\n", st.ok() ? "registration failed" : st.error().c_str());
        return 1;
    }
    const int T = (int)ids.size();
    std::vector<float> got((size_t)T * 1024);
    if (!audioldm2::t5_encoder_forward(st, ids.data(), T, got.data())) {
        std::fprintf(stderr, "the T5 device path failed\n");
        return 1;
    }
    double num = 0, den = 0, maxerr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)got[i] - ref[i];
        num += d * d; den += (double)ref[i] * ref[i];
        if (std::fabs(d) > maxerr) maxerr = std::fabs(d);
    }
    const double rel = std::sqrt(num / den);
    std::printf("T5 encoder, %d tokens: hidden rel L2 %.3e, max abs %.3e  %s\n", T, rel, maxerr, rel <= 1e-3 ? "ok" : "FAIL");
    return rel <= 1e-3 ? 0 : 1;
}
