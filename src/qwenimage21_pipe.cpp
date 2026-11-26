/* Qwen-Image 2.1's text to image (qwenimage21_pipe.h): the template and the tokenizer on the host, the three models on the device,
 * the sampler's state (gh * gw x 64 values) on the host. */
#include "qwenimage21_pipe.h"

#include "qwenimage21_dit.h"
#include "qwenimage21_text.h"
#include "qwenimage21_vae.h"
#include "qwenimage_pipe.h"
#include "sd_device.h"
#include "torch_randn.h"
#include "trellis_bake.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

thread_local std::string g_error;

constexpr int kImStart = 151644;
constexpr int kImEnd = 151645;
constexpr int kText = 4096;
const char* const kSystem = "Comprehend and analyze the provided prompt.";

}  // namespace

namespace qwenimage21 {

bool prompt_ids(tokenizer_t* tok, const std::string& prompt, std::vector<int>& ids, int& drop) {
    /* <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n: the markers are
     * special ids, the rest is encoded as text between them (as the tokenizer splits on its special tokens). An empty prompt is
     * " " (the pipeline's: the encoder needs something to read). */
    ids.clear();
    auto enc = [&](const std::string& text) {
        std::vector<int> buf(text.size() + 8);
        const int n = tokenizer_encode(tok, text.c_str(), buf.data(), (int)buf.size());
        if (n < 0) return false;
        ids.insert(ids.end(), buf.begin(), buf.begin() + n);
        return true;
    };
    ids.push_back(kImStart);
    if (!enc(std::string("system\n") + kSystem)) return false;
    ids.push_back(kImEnd);
    if (!enc("\n")) return false;
    drop = (int)ids.size();
    ids.push_back(kImStart);
    if (!enc("user\n" + (prompt.empty() ? std::string(" ") : prompt))) return false;
    ids.push_back(kImEnd);
    if (!enc("\n")) return false;
    ids.push_back(kImStart);
    return enc("assistant\n");
}

void sigmas(int steps, int image_tokens, std::vector<float>& sig) { qwenimage::sigmas_for(steps, image_tokens, sig); }

bool sample(const zimage::SafeTensors& dit, const float* noise, int gh, int gw, const float* emb, int L, int steps, float* latent,
            const std::function<void(int, const float*, const float*)>& on_step) {
    const size_t n = (size_t)gh * gw * 64;
    std::vector<float> sig, x(noise, noise + n), v(n);
    sigmas(steps, gh * gw, sig);
    for (int i = 0; i < steps; ++i) {
        /* the pipeline's timestep: the scheduler's sigma * 1000 (float32), divided by 1000 for the model */
        const float t = (sig[i] * 1000.0f) / 1000.0f;
        if (!dev_dit21_forward(dit, x.data(), gh, gw, emb, L, t, v.data())) return false;
        const float dt = sig[i + 1] - sig[i];
        for (size_t k = 0; k < n; ++k) x[k] += dt * v[k];
        if (on_step) on_step(i, v.data(), x.data());
    }
    std::memcpy(latent, x.data(), sizeof(float) * n);
    return true;
}

bool decode(const zimage::SafeTensors& vae, const float* latent, int gh, int gw, float* out) {
    /* unpacked: [N][64] token-major to the decoder's channel-major [64][gh][gw] */
    const int N = gh * gw;
    std::vector<float> lat((size_t)64 * N);
    for (int p = 0; p < N; ++p)
        for (int c = 0; c < 64; ++c) lat[(size_t)c * N + p] = latent[(size_t)p * 64 + c];
    return dev_vae21_decode(vae, lat.data(), gh, gw, out);
}

}  // namespace qwenimage21

struct qi21_pipe {
    zimage::SafeTensors dit, vae, te;
    tokenizer_t* tok = nullptr;
    /* the background job (qi21_start) */
    std::mutex mu;
    std::thread th;
    int status = 2;  /* 0 running, 1 done, -1 failed, 2 none */
    int step = 0, steps = 0;
    std::string error;
    std::vector<uint8_t> png;
    std::chrono::steady_clock::time_point t0, t1;
    qi21_pipe(const std::string& d, const std::string& v, const std::string& t) : dit(d), vae(v), te(t) {}
};

namespace {

/* the options' check, shared by the two ways in */
bool check_options(qi21_pipe_t* pipe, const char* prompt, int height, int width, int steps) {
    if (!pipe || !prompt) {
        g_error = "qwen-image 2.1: a pipeline and a prompt are needed";
        return false;
    }
    if (height < 256 || width < 256 || height > 2048 || width > 2048 || height % 32 || width % 32) {
        g_error = "qwen-image 2.1: height and width: multiples of 32 from 256 to 2048";
        return false;
    }
    if (steps < 1 || steps > 100) {
        g_error = "qwen-image 2.1: steps 1 to 100";
        return false;
    }
    return true;
}

}  // namespace

extern "C" {

qi21_pipe_t* qi21_open(const char* dit_path, const char* vae_path, const char* te_path, const char** vocab, size_t nv,
                       const char** merges, size_t nm) {
    g_error.clear();
    qi21_pipe_t* p = new qi21_pipe(dit_path, vae_path, te_path);
    if (!p->dit.ok() || !p->vae.ok() || !p->te.ok()) {
        g_error = "qwen-image 2.1: " + std::string(!p->dit.ok() ? p->dit.error() : !p->vae.ok() ? p->vae.error() : p->te.error());
        delete p;
        return nullptr;
    }
    if (!p->dit.register_device() || !p->vae.register_device() || !p->te.register_device()) {
        g_error = "qwen-image 2.1: the weights' registration failed";
        delete p;
        return nullptr;
    }
    p->tok = tokenizer_create(vocab, nv, merges, nm);
    if (!p->tok) {
        g_error = "qwen-image 2.1: the tokenizer does not load";
        delete p;
        return nullptr;
    }
    return p;
}

void qi21_close(qi21_pipe_t* pipe) {
    if (!pipe) return;
    if (pipe->th.joinable()) pipe->th.join();
    if (pipe->tok) tokenizer_free(pipe->tok);
    delete pipe;
}

const char* qi21_last_error(void) { return g_error.c_str(); }

int qi21_generate(qi21_pipe_t* pipe, const char* prompt, int height, int width, int steps, unsigned long long seed, unsigned char* rgba,
                  void (*progress)(int step, int steps, void* user), void* user) {
    g_error.clear();
    if (!check_options(pipe, prompt, height, width, steps)) return -1;
    if (!rgba) {
        g_error = "qwen-image 2.1: no output";
        return -1;
    }
    try {
        std::vector<int> ids;
        int drop = 0;
        if (!qwenimage21::prompt_ids(pipe->tok, prompt, ids, drop)) throw std::runtime_error("the prompt does not tokenize");
        const int T = (int)ids.size(), L = T - drop;
        std::vector<float> all((size_t)T * kText);
        if (!qwenimage21::dev_text21_encode(pipe->te, ids.data(), T, all.data())) throw std::runtime_error("the text encoder failed");
        const float* emb = all.data() + (size_t)drop * kText;

        const int gh = height / 16, gw = width / 16, N = gh * gw;
        std::vector<float> chw((size_t)64 * N), noise((size_t)N * 64), latent((size_t)N * 64);
        torch_randn(seed, chw.data(), chw.size());  /* torch.randn(1, 1, 64, gh, gw) after manual_seed(seed) */
        for (int c = 0; c < 64; ++c)
            for (int p = 0; p < N; ++p) noise[(size_t)p * 64 + c] = chw[(size_t)c * N + p];  /* the pipeline's _pack_latents */
        auto on_step = [&](int i, const float*, const float*) {
            if (progress) progress(i + 1, steps, user);
        };
        if (!qwenimage21::sample(pipe->dit, noise.data(), gh, gw, emb, L, steps, latent.data(), on_step))
            throw std::runtime_error("the sampler failed");
        std::vector<float> pix((size_t)4 * height * width);
        if (!qwenimage21::decode(pipe->vae, latent.data(), gh, gw, pix.data())) throw std::runtime_error("the decoder failed");
        /* [-1, 1] channel-major to 8-bit RGBA, rounded (the pipeline's postprocess: x / 2 + 0.5, clamped, times 255) */
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 4; ++c) {
                    const float v = (pix[((size_t)c * height + y) * width + x] + 1.0f) * 127.5f;
                    const int q = (int)std::lrint(v);
                    rgba[((size_t)y * width + x) * 4 + c] = (unsigned char)(q < 0 ? 0 : q > 255 ? 255 : q);
                }
    } catch (const std::exception& e) {
        g_error = std::string("qwen-image 2.1: ") + e.what();
        return -1;
    }
    return 0;
}

int qi21_start(qi21_pipe_t* pipe, const char* prompt, int height, int width, int steps, unsigned long long seed) {
    g_error.clear();
    if (!check_options(pipe, prompt, height, width, steps)) return -1;
    {
        std::lock_guard<std::mutex> lk(pipe->mu);
        if (pipe->status == 0) return 1;
    }
    if (pipe->th.joinable()) pipe->th.join();
    {
        std::lock_guard<std::mutex> lk(pipe->mu);
        pipe->status = 0;
        pipe->step = 0;
        pipe->steps = steps;
        pipe->error.clear();
        pipe->png.clear();
        pipe->t0 = pipe->t1 = std::chrono::steady_clock::now();
    }
    pipe->th = std::thread([pipe, text = std::string(prompt), height, width, steps, seed]() {
        std::vector<unsigned char> rgba((size_t)height * width * 4);
        std::vector<uint8_t> png;
        std::string err;
        int rc = -1;
        {
            /* the device's other users (the image and audio generators, the 3D jobs) wait while the image is made */
            std::lock_guard<std::recursive_mutex> dev(sddev::run_mutex());
            auto progress = [](int step, int, void* user) {
                qi21_pipe_t* p = static_cast<qi21_pipe_t*>(user);
                std::lock_guard<std::mutex> lk(p->mu);
                p->step = step;
            };
            rc = qi21_generate(pipe, text.c_str(), height, width, steps, seed, rgba.data(), progress, pipe);
            if (rc != 0) err = qi21_last_error();
        }
        if (rc == 0) {
            png = trellis::encode_png(std::vector<uint8_t>(rgba.begin(), rgba.end()), width, height, 4);
            if (png.empty()) {
                rc = -1;
                err = "qwen-image 2.1: the PNG encoder failed";
            }
        }
        std::lock_guard<std::mutex> lk(pipe->mu);
        pipe->t1 = std::chrono::steady_clock::now();
        if (rc == 0) {
            pipe->png = std::move(png);
            pipe->status = 1;
        } else {
            pipe->error = err.empty() ? "qwen-image 2.1: the job failed" : err;
            pipe->status = -1;
        }
    });
    return 0;
}

int qi21_poll(qi21_pipe_t* pipe, int* step, int* steps, double* seconds) {
    if (!pipe) return 2;
    std::lock_guard<std::mutex> lk(pipe->mu);
    if (step) *step = pipe->step;
    if (steps) *steps = pipe->steps;
    if (seconds)
        *seconds = pipe->status == 2 ? 0.0
                                     : std::chrono::duration<double>((pipe->status == 0 ? std::chrono::steady_clock::now() : pipe->t1) -
                                                                     pipe->t0).count();
    return pipe->status;
}

int qi21_take(qi21_pipe_t* pipe, unsigned char** png, size_t* n) {
    if (!pipe || !png || !n) {
        g_error = "qi21_take: no pipeline or output";
        return -1;
    }
    std::lock_guard<std::mutex> lk(pipe->mu);
    if (pipe->status == -1) {
        g_error = pipe->error;
        pipe->status = 2;
        return -1;
    }
    if (pipe->status != 1) {
        g_error = pipe->status == 0 ? "qwen-image 2.1: the image is not done" : "qwen-image 2.1: no image";
        return -1;
    }
    unsigned char* out = static_cast<unsigned char*>(std::malloc(pipe->png.size() ? pipe->png.size() : 1));
    if (!out) {
        g_error = "out of memory";
        return -1;
    }
    std::memcpy(out, pipe->png.data(), pipe->png.size());
    *png = out;
    *n = pipe->png.size();
    pipe->png.clear();
    pipe->png.shrink_to_fit();
    pipe->status = 2;
    return 0;
}

void qi21_free(void* ptr) { std::free(ptr); }

}  // extern "C"
