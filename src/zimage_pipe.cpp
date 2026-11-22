/* Z-Image-Turbo's text to image (zimage_pipe.h): the chat template and the tokenizer on the host (the prompt's ids are the only
 * host work besides the noise), then the device path: the text encoder, the 8-step sampler, the VAE decoder. */
#include "zimage_pipe.h"

#include "tokenizer.h"
#include "zimage_dit.h"
#include "zimage_text.h"
#include "zimage_vae.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_error;

/* the Qwen3 chat template for one user turn with thinking enabled (the pipeline's encode_prompt):
 * <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n  (the markers are special ids; the rest is encoded as text) */
constexpr int kImStart = 151644;
constexpr int kImEnd = 151645;
constexpr int kSteps = 8;
constexpr float kShift = 3.0f;

/* splitmix64: the noise's uniform draws (deterministic from the seed) */
uint64_t next_u64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* the standard normal values (Box-Muller) of the latent [16][32][32] */
void noise_latent(unsigned long long seed, float* out, size_t n) {
    uint64_t s = seed;
    for (size_t i = 0; i < n; i += 2) {
        double u1 = ((next_u64(s) >> 11) + 1.0) * (1.0 / 9007199254740993.0);
        double u2 = (next_u64(s) >> 11) * (1.0 / 9007199254740992.0);
        double r = std::sqrt(-2.0 * std::log(u1)), a = 6.283185307179586 * u2;
        out[i] = (float)(r * std::cos(a));
        if (i + 1 < n) out[i + 1] = (float)(r * std::sin(a));
    }
}

}  // namespace

struct zi_pipe {
    zimage::SafeTensors dit, vae, te;
    tokenizer_t* tok = nullptr;
    zi_pipe(const std::string& d, const std::string& v, const std::string& t) : dit(d), vae(v), te(t) {}
};

extern "C" {

zi_pipe_t* zi_open(const char* dit_path, const char* vae_path, const char* te_path, const char** vocab, size_t nv,
                   const char** merges, size_t nm) {
    g_error.clear();
    zi_pipe_t* p = new zi_pipe(dit_path, vae_path, te_path);
    if (!p->dit.ok() || !p->vae.ok() || !p->te.ok()) {
        g_error = "zimage: " + std::string(!p->dit.ok() ? p->dit.error() : !p->vae.ok() ? p->vae.error() : p->te.error());
        delete p;
        return nullptr;
    }
    if (!p->dit.register_device() || !p->vae.register_device() || !p->te.register_device()) {
        g_error = "zimage: the weights' registration failed";
        delete p;
        return nullptr;
    }
    p->tok = tokenizer_create(vocab, nv, merges, nm);
    if (!p->tok) {
        g_error = "zimage: the tokenizer does not load";
        delete p;
        return nullptr;
    }
    return p;
}

void zi_close(zi_pipe_t* pipe) {
    if (!pipe) return;
    if (pipe->tok) tokenizer_free(pipe->tok);
    delete pipe;
}

const char* zi_last_error(void) { return g_error.c_str(); }

/* the pipeline with the noise given (the latent [16][H/8][W/8]: the reference's noise in the checks) */
static int run(zi_pipe_t* pipe, const char* prompt, int height, int width, int steps, const float* noise_in, unsigned char* rgb) {
    try {
        /* the ids: <|im_start|> user\n + prompt + <|im_end|> \n <|im_start|> assistant\n (the markers as special ids) */
        std::vector<int> ids;
        auto enc = [&](const std::string& text) {
            std::vector<int> buf(text.size() + 8);
            int n = tokenizer_encode(pipe->tok, text.c_str(), buf.data(), (int)buf.size());
            if (n < 0) throw std::runtime_error("the prompt does not tokenize");
            ids.insert(ids.end(), buf.begin(), buf.begin() + n);
        };
        ids.push_back(kImStart);
        enc("user\n" + std::string(prompt));
        ids.push_back(kImEnd);
        enc("\n");
        ids.push_back(kImStart);
        enc("assistant\n");
        const int L = (int)ids.size();

        std::vector<float> cap((size_t)L * 2560);
        if (!zimage::dev_text_encode(pipe->te, ids.data(), L, cap.data())) throw std::runtime_error("the text encoder failed");

        const int gh = height / 16, gw = width / 16, LH = 2 * gh, LW = 2 * gw;
        std::vector<float> lat((size_t)16 * LH * LW), pix((size_t)3 * height * width);
        if (!zimage::dev_sample(pipe->dit, noise_in, cap.data(), L, steps, kShift, lat.data(), gh, gw))
            throw std::runtime_error("the sampler failed");
        if (!zimage::dev_vae_decode(pipe->vae, lat.data(), pix.data(), LH, LW)) throw std::runtime_error("the decoder failed");

        /* [-1, 1] floats (channel-major) to 8-bit HWC, rounded and clamped */
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c) {
                    float v = (pix[((size_t)c * height + y) * width + x] + 1.0f) * 127.5f;
                    int q = (int)std::lrint(v);
                    rgb[((size_t)y * width + x) * 3 + c] = (unsigned char)(q < 0 ? 0 : q > 255 ? 255 : q);
                }
    } catch (const std::exception& e) {
        g_error = std::string("zimage: ") + e.what();
        return -1;
    }
    return 0;
}

int zi_generate_steps(zi_pipe_t* pipe, const char* prompt, int height, int width, int steps, unsigned long long seed, unsigned char* rgb) {
    g_error.clear();
    if (!pipe || !prompt || !rgb) { g_error = "zimage: a pipeline, a prompt and an output are needed"; return -1; }
    if (height < 64 || width < 64 || height % 16 || width % 16) { g_error = "zimage: height and width: multiples of 16 from 64"; return -1; }
    if (steps < 1 || steps > 50) { g_error = "zimage: steps 1 to 50"; return -1; }
    std::vector<float> noise((size_t)16 * (height / 8) * (width / 8));
    noise_latent(seed, noise.data(), noise.size());
    return run(pipe, prompt, height, width, steps, noise.data(), rgb);
}

int zi_generate_noise_steps(zi_pipe_t* pipe, const char* prompt, int height, int width, int steps, const float* noise, unsigned char* rgb) {
    g_error.clear();
    if (!pipe || !prompt || !noise || !rgb) { g_error = "zimage: a pipeline, a prompt, a noise and an output are needed"; return -1; }
    if (height < 64 || width < 64 || height % 16 || width % 16) { g_error = "zimage: height and width: multiples of 16 from 64"; return -1; }
    if (steps < 1 || steps > 50) { g_error = "zimage: steps 1 to 50"; return -1; }
    return run(pipe, prompt, height, width, steps, noise, rgb);
}

int zi_generate(zi_pipe_t* pipe, const char* prompt, int height, int width, unsigned long long seed, unsigned char* rgb) {
    return zi_generate_steps(pipe, prompt, height, width, kSteps, seed, rgb);
}

int zi_generate_with_noise(zi_pipe_t* pipe, const char* prompt, const float* noise, unsigned char* rgb) {
    g_error.clear();
    if (!pipe || !prompt || !noise || !rgb) { g_error = "zimage: a pipeline, a prompt, a noise and an output are needed"; return -1; }
    return run(pipe, prompt, 256, 256, kSteps, noise, rgb);
}

}  // extern "C"
