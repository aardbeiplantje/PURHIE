/* AudioLDM 2's GPT-2 (the device path, src/audioldm2_gpt2.cpp) against the diffusers oracle: /data/audioldm2-stages/gpt2_input.f32
 * (T x 768, float), gpt2_hidden6.f32 (after the sixth block) and gpt2_last.f32 (after the final norm), written by
 * /data/audioldm2_stage_gpt2.py. Usage: audioldm2_gpt2_verify [model.safetensors] [stages dir]. Exit 0 when both match. */
#include "audioldm2_gpt2.h"
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

static bool compare(const char* what, const std::vector<float>& got, const std::vector<float>& ref) {
    double maxerr = 0, maxref = 0, num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)got[i] - (double)ref[i];
        maxerr = std::fmax(maxerr, std::fabs(d));
        maxref = std::fmax(maxref, std::fabs((double)ref[i]));
        num += d * d;
        den += (double)ref[i] * ref[i];
    }
    const double rel = std::sqrt(num / den);
    const bool ok = rel <= 1e-3;
    std::printf("%-8s rel L2 %.3e, max abs %.3e (max |ref| %.3e)  %s\n", what, rel, maxerr, maxref, ok ? "ok" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "/data/audioldm2/language_model/model.safetensors";
    const std::string dir = argc > 2 ? argv[2] : "/data/audioldm2-stages";
    std::vector<float> x, h6, last;
    if (!read_floats(dir + "/gpt2_input.f32", x) || !read_floats(dir + "/gpt2_hidden6.f32", h6) || !read_floats(dir + "/gpt2_last.f32", last)) {
        std::fprintf(stderr, "cannot read the stage files in %s\n", dir.c_str());
        return 1;
    }
    const int T = (int)(x.size() / 768);
    zimage::SafeTensors st(model);
    if (!st.ok() || !st.register_device()) {
        std::fprintf(stderr, "the GPT-2 file: %s\n", st.ok() ? "registration failed" : st.error().c_str());
        return 1;
    }
    std::vector<float> got6((size_t)T * 768), gotl((size_t)T * 768);
    if (!audioldm2::gpt2_forward(st, x.data(), T, nullptr, got6.data(), gotl.data())) {
        std::fprintf(stderr, "the GPT-2 device path failed\n");
        return 1;
    }
    std::printf("GPT-2, T = %d tokens\n", T);
    const bool a = compare("hidden6", got6, h6);
    const bool b = compare("last", gotl, last);
    /* the generation (generate_language_model: 8 steps on the projection's rows, gen_proj.f32 -> gen_out.f32) */
    std::vector<float> gp, gout;
    if (!read_floats(dir + "/gen_proj.f32", gp) || !read_floats(dir + "/gen_out.f32", gout)) {
        std::fprintf(stderr, "cannot read gen_proj.f32 / gen_out.f32 in %s\n", dir.c_str());
        return 1;
    }
    const int Tp = (int)(gp.size() / 768);
    std::vector<float> gen(gout.size());
    if (!audioldm2::gpt2_generate(st, gp.data(), nullptr, Tp, (int)(gout.size() / 768), gen.data())) {
        std::fprintf(stderr, "the GPT-2 generation failed\n");
        return 1;
    }
    const bool c = compare("generate", gen, gout);
    return a && b && c ? 0 : 1;
}
