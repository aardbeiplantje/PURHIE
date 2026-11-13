// MiniMax-H3-Image's text-to-image pipeline (minimax_pipe.h). The text encoder's conditioning is the layer-50 residual of the
// engine's Qwen3 path over the prompt's tokens (the M1 check: tests/mm_te_verify), the sampler is minimax_dit's, the decoder is
// minimax_vae's, and the output follows the pipeline's decode: image = decoded * std + mean (ImageNet's), clamped to [0, 1],
// truncated to 8 bits.
#include "minimax_pipe.h"

#include "inference.h"
#include "inference_internal.h"
#include "minimax_dit.h"
#include "minimax_vae.h"
#include "qwen_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string g_error;

struct Layer50 {
    std::vector<float>* out;
    int E;
};

// The prompt's hidden state after decoder layer 50 (index 49), one row per token; the prompt verbatim, no special tokens
// (the tokenizer adds none for this model), as the M1 verifier checks against transformers.
std::vector<float> encode_prompt(const char* gguf, const std::string& prompt, int* T_out) {
    inference_model* m = inference_load_model(gguf);
    if (!m) throw std::runtime_error(std::string("text encoder: ") + inference_get_last_error());
    struct Free {
        inference_model* m;
        ~Free() { inference_free_model(m); }
    } guard{m};
    if (inference_kv_cache_init(m, 1024) != 0) throw std::runtime_error(std::string("text encoder kv: ") + inference_get_last_error());
    int n = 0;
    int32_t* ids = inference_tokenize(m, prompt.c_str(), &n);
    if (!ids || n <= 0) throw std::runtime_error("text encoder: the prompt does not tokenize");
    std::vector<int> tokens(ids, ids + n);
    delete[] ids;
    if (std::getenv("MM_PIPE_DEBUG")) {
        std::fprintf(stderr, "prompt tokens %d:", n);
        for (int i = 0; i < n && i < 16; ++i) std::fprintf(stderr, " %d", tokens[i]);
        std::fprintf(stderr, "\n");
    }
    const int E = m->n_embd;
    if (E != 5120) throw std::runtime_error("text encoder: not the Qwen3-VL text width (5120)");
    qwen_model_t* qm = inference_get_qwen_model(m);
    std::vector<float> hid((size_t)n * E);
    Layer50 cap{&hid, E};
    qwen_model_set_layer_callback(qm, [](int l, int pos, const float* x, int width, void* u) {
        Layer50* c = static_cast<Layer50*>(u);
        if (l == 49) std::memcpy(c->out->data() + (size_t)pos * c->E, x, sizeof(float) * (size_t)width);
    }, &cap);
    std::vector<float> logits(m->vocab_size);
    const int rc = qwen_model_prefill(qm, tokens.data(), n, 0, logits.data());
    qwen_model_set_layer_callback(qm, nullptr, nullptr);
    if (rc != 0) throw std::runtime_error(std::string("text encoder prefill: ") + inference_get_last_error());
    *T_out = n;
    return hid;
}

// Standard normal values from a seed (a Mersenne twister and Box-Muller: not torch's sequence)
void seeded_normal(unsigned long long seed, size_t n, std::vector<float>& out) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    out.resize(n);
    for (size_t i = 0; i < n; i += 2) {
        const double r = std::sqrt(-2.0 * std::log(1.0 - u(rng))), t = 2.0 * M_PI * u(rng);
        out[i] = (float)(r * std::cos(t));
        if (i + 1 < n) out[i + 1] = (float)(r * std::sin(t));
    }
}

}  // namespace

struct mm_pipe {
    std::string text, dit, pdd, vae;
};

mm_pipe_t* mm_open(const char* text_gguf, const char* dit_gguf, const char* pdd_gguf, const char* vae_gguf) {
    try {
        if (!text_gguf || !dit_gguf || !pdd_gguf || !vae_gguf) throw std::runtime_error("mm_open: a path is missing");
        std::unique_ptr<mm_pipe> p(new mm_pipe{text_gguf, dit_gguf, pdd_gguf, vae_gguf});
        return p.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void mm_close(mm_pipe_t* pipe) { delete pipe; }

const char* mm_last_error(void) { return g_error.c_str(); }

// The canvas the released pipeline generates on (get_image_canvas_size): a request below 2048 x 2048 pixels is generated on a
// canvas scaled up to that area at the same aspect ratio, each side rounded to a multiple of 32; the result is then resized back.
void canvas_size(int height, int width, int* ch, int* cw) {
    const double scale = std::fmax(1.0, std::sqrt(2048.0 * 2048.0 / ((double)width * height)));
    *ch = std::fmax(32.0, std::nearbyint(height * scale / 32.0) * 32.0);
    *cw = std::fmax(32.0, std::nearbyint(width * scale / 32.0) * 32.0);
}

// Area-averaging resize of 8-bit RGB (each output pixel averages its source footprint, fractional at the edges)
void resize_area(const unsigned char* src, int sh, int sw, unsigned char* dst, int dh, int dw) {
    const double sy = (double)sh / dh, sx = (double)sw / dw;
    for (int y = 0; y < dh; ++y) {
        const double y0 = y * sy, y1 = (y + 1) * sy;
        for (int x = 0; x < dw; ++x) {
            const double x0 = x * sx, x1 = (x + 1) * sx;
            double acc[3] = {0, 0, 0}, wsum = 0;
            for (int iy = (int)y0; iy < (int)std::ceil(y1) && iy < sh; ++iy) {
                const double wy = std::fmin(y1, iy + 1.0) - std::fmax(y0, (double)iy);
                for (int ix = (int)x0; ix < (int)std::ceil(x1) && ix < sw; ++ix) {
                    const double wx = std::fmin(x1, ix + 1.0) - std::fmax(x0, (double)ix);
                    const double wt = wy * wx;
                    for (int c = 0; c < 3; ++c) acc[c] += wt * src[((size_t)iy * sw + ix) * 3 + c];
                    wsum += wt;
                }
            }
            for (int c = 0; c < 3; ++c)
                dst[((size_t)y * dw + x) * 3 + c] = (unsigned char)std::fmin(255.0, std::nearbyint(acc[c] / wsum));
        }
    }
}

// phase timing (MM_PIPE_DEBUG set: printed to stderr)
static double phase_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void phase(const char* name, double& t) {
    if (!std::getenv("MM_PIPE_DEBUG")) return;
    const double now = phase_now();
    std::fprintf(stderr, "phase %-16s %7.2f s\n", name, now - t);
    t = now;
}

int mm_generate(mm_pipe_t* p, const char* prompt, int height, int width, unsigned long long seed, unsigned char* rgb) {
    if (!p || !prompt || !rgb || height <= 0 || width <= 0 || height % 32 || width % 32) {
        g_error = "mm_generate: bad arguments (height and width: multiples of 32)";
        return -1;
    }
    try {
        int canvas_h = 0, canvas_w = 0;
        canvas_size(height, width, &canvas_h, &canvas_w);
        const int h = canvas_h / 16, w = canvas_w / 16;  // the latent grid (16 px per cell)
        // 1. the prompt's conditioning, then the encoder is released (it and the DiT are not resident together)
        double tp = phase_now();
        int T = 0;
        std::vector<float> text = encode_prompt(p->text.c_str(), prompt, &T);
        phase("text encoder", tp);
        // 2. the initial noise and the six-step sampler
        std::vector<float> noise, lat((size_t)24 * h * w);
        seeded_normal(seed, (size_t)24 * h * w, noise);
        {
            mmdit_t* dit = mmdit_open(p->dit.c_str(), p->pdd.c_str());
            phase("dit open", tp);
            if (!dit) throw std::runtime_error(std::string("dit: ") + mmdit_last_error());
            const int rc = mmdit_sample(dit, text.data(), T, noise.data(), h, w, lat.data());
            const std::string err = rc != 0 ? mmdit_last_error() : "";
            mmdit_close(dit);
            phase("sampler", tp);
            if (rc != 0) throw std::runtime_error("sampler: " + err);
        }
        // 3. the VAE decoder (the latents' normalization is applied inside), then the pixels
        std::vector<float> planes((size_t)3 * canvas_h * canvas_w);
        {
            mmvae_t* vae = mmvae_open(p->vae.c_str());
            phase("dit close", tp);
            if (!vae) throw std::runtime_error(std::string("vae: ") + mmvae_last_error());
            const int rc = mmvae_decode(vae, lat.data(), h, w, planes.data());
            const std::string err = rc != 0 ? mmvae_last_error() : "";
            mmvae_close(vae);
            phase("vae decode", tp);
            if (rc != 0) throw std::runtime_error("vae: " + err);
        }
        // the pipeline's output: decoded * std + mean (ImageNet's), clamped to [0, 1], truncated to 8 bits on the canvas, then
        // resized back to the requested size
        static const float kMean[3] = {0.485f, 0.456f, 0.406f}, kStd[3] = {0.229f, 0.224f, 0.225f};
        std::vector<unsigned char> canvas((size_t)canvas_h * canvas_w * 3);
        for (int y = 0; y < canvas_h; ++y)
            for (int x = 0; x < canvas_w; ++x)
                for (int c = 0; c < 3; ++c) {
                    float v = planes[((size_t)c * canvas_h + y) * canvas_w + x] * kStd[c] + kMean[c];
                    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                    canvas[((size_t)y * canvas_w + x) * 3 + c] = (unsigned char)(v * 255.0f);
                }
        if (canvas_h == height && canvas_w == width) std::memcpy(rgb, canvas.data(), canvas.size());
        else resize_area(canvas.data(), canvas_h, canvas_w, rgb, height, width);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}
