/* Qwen-Image 2.1's text to image (qwenimage21_pipe.h): the template and the tokenizer on the host, the three models on the device,
 * the sampler's state (gh * gw x 64 values) on the host. */
#include "qwenimage21_pipe.h"

#include "image.h"
#include "mrope.h"
#include "qwenimage21_dit.h"
#include "qwenimage21_text.h"
#include "qwenimage21_vae.h"
#include "qwenimage21_vision.h"
#include "qwenimage_pipe.h"
#include "sd_device.h"
#include "torch_randn.h"
#include "trellis_bake.h"
#include "trellis_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

thread_local std::string g_error;

constexpr int kImStart = 151644;
constexpr int kImEnd = 151645;
constexpr int kVisionStart = 151652;
constexpr int kVisionEnd = 151653;
constexpr int kImagePad = 151655;
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
    Condition c;
    c.enc = emb;
    c.L = L;
    return sample(dit, noise, gh, gw, c, steps, latent, on_step);
}

bool sample(const zimage::SafeTensors& dit, const float* noise, int gh, int gw, const Condition& c, int steps, float* latent,
            const std::function<void(int, const float*, const float*)>& on_step) {
    const size_t n = (size_t)gh * gw * 64;
    std::vector<float> sig, x(noise, noise + n), v(n);
    /* the schedule's shift follows the target's tokens alone (the pipeline's latents.shape[1]) */
    sigmas(steps, gh * gw, sig);
    Dit21 model(dit);
    if (!model.ok() || !model.prefix(c, gh, gw)) return false;
    for (int i = 0; i < steps; ++i) {
        /* the pipeline's timestep: the scheduler's sigma * 1000 (float32), divided by 1000 for the model */
        const float t = (sig[i] * 1000.0f) / 1000.0f;
        if (!model.step(x.data(), t, v.data())) return false;
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

bool condition_pixels(const unsigned char* rgba, int w, int h, std::vector<float>& rows, int& gh, int& gw) {
    if (!rgba || w < 32 || h < 32 || w % 32 || h % 32) return false;
    /* white.paste(img, mask=alpha): Pillow's BLEND, (white (255 - a) + c a) / 255 rounded as DIV255 */
    std::vector<unsigned char> rgb((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        const unsigned a = rgba[i * 4 + 3];
        for (int c = 0; c < 3; ++c) {
            const unsigned t = 255u * (255u - a) + rgba[i * 4 + c] * a + 128u;
            rgb[i * 3 + c] = (unsigned char)(((t >> 8) + t) >> 8);
        }
    }
    float* out = nullptr;
    /* the processor's bounds: 65536 to 16777216 pixels (Qwen/Qwen-Image-2.1 processor/preprocessor_config.json) */
    if (image_qwen_pixels(rgb.data(), w, h, 16, 2, 2, 0.5f, 0.5f, 65536, 16777216, &out, &gh, &gw) != 0) return false;
    rows.assign(out, out + (size_t)gh * gw * 1536);
    image_free(out);
    return true;
}

void edit_size(int w, int h, int resolution, int& width, int& height) {
    const double ratio = (double)w / (double)h;
    const double fw = std::sqrt((double)resolution * resolution * ratio), fh = fw / ratio;
    width = (int)std::nearbyint(fw / 32.0) * 32;  /* Python's round: half to even (the default rounding mode) */
    height = (int)std::nearbyint(fh / 32.0) * 32;
}

std::vector<unsigned char> resize_condition(const unsigned char* rgba, int w, int h, int ow, int oh) {
    if (ow == w && oh == h) return std::vector<unsigned char>(rgba, rgba + (size_t)w * h * 4);
    /* RGBA -> RGBa (Pillow's rgba2rgbA: c a / 255 rounded as MULDIV255), the resampling, RGBa -> RGBA (rgbA2rgba: 255 c / a,
     * clipped; unchanged where a is 0 or 255) */
    std::vector<unsigned char> pm((size_t)w * h * 4);
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        const unsigned a = rgba[i * 4 + 3];
        for (int c = 0; c < 3; ++c) {
            const unsigned t = rgba[i * 4 + c] * a + 128u;
            pm[i * 4 + c] = (unsigned char)(((t >> 8) + t) >> 8);
        }
        pm[i * 4 + 3] = (unsigned char)a;
    }
    std::vector<unsigned char> r = trellis::resize_pillow8(pm.data(), w, h, 4, ow, oh, trellis::PilFilter::Lanczos);
    for (size_t i = 0; i < (size_t)ow * oh; ++i) {
        const unsigned a = r[i * 4 + 3];
        if (a == 0 || a == 255) continue;
        for (int c = 0; c < 3; ++c) r[i * 4 + c] = (unsigned char)std::min(255u, (255u * r[i * 4 + c]) / a);
    }
    return r;
}

bool edit_ids(tokenizer_t* tok, const std::string& prompt, int n_pad, std::vector<int>& ids, int& drop) {
    /* <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<image1><|vision_start|>{pads}<|vision_end|>{prompt}<|im_end|>\n
     * <|im_start|>assistant\n (the pipeline's prompt_template_ti2i, its <|image_pad|> expanded by the processor) */
    if (n_pad < 1) return false;
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
    if (!enc("user\n<image1>")) return false;
    ids.push_back(kVisionStart);
    ids.insert(ids.end(), (size_t)n_pad, kImagePad);
    ids.push_back(kVisionEnd);
    if (!enc(prompt.empty() ? std::string(" ") : prompt)) return false;
    ids.push_back(kImEnd);
    if (!enc("\n")) return false;
    ids.push_back(kImStart);
    return enc("assistant\n");
}

bool edit_positions(const std::vector<int>& ids, int gh, int gw, std::vector<int>& pos) {
    const int T = (int)ids.size();
    std::vector<int32_t> id32(ids.begin(), ids.end());
    const int32_t grid[2] = {gh, gw};
    std::vector<int64_t> p3((size_t)T * 3);
    if (mrope_positions(id32.data(), T, grid, 1, 2, kImagePad, p3.data(), nullptr) != 0) return false;
    pos.assign((size_t)3 * T, 0);
    for (int t = 0; t < T; ++t)
        for (int a = 0; a < 3; ++a) pos[(size_t)a * T + t] = (int)p3[(size_t)t * 3 + a];
    return true;
}

bool encode_condition(const zimage::SafeTensors& te, const zimage::SafeTensors& vae, tokenizer_t* tok, const std::string& prompt,
                      const unsigned char* rgba, int cw, int ch, std::vector<float>& emb, std::vector<unsigned char>& pad,
                      std::vector<float>& cond) {
    /* the vision tower's tokens of the image composited over white */
    std::vector<float> rows;
    int pgh = 0, pgw = 0;
    if (!condition_pixels(rgba, cw, ch, rows, pgh, pgw)) return false;
    const int M = pgh * pgw / 4;
    std::vector<float> merged((size_t)M * VISION_OUT), ds((size_t)VISION_DEEPSTACK * M * VISION_OUT);
    if (!dev_vision21_encode(te, rows.data(), pgh, pgw, merged.data(), ds.data())) return false;
    /* the text encoder over the template with the image's tokens at its pads */
    std::vector<int> ids, pos;
    int drop = 0;
    if (!edit_ids(tok, prompt, M, ids, drop) || !edit_positions(ids, pgh, pgw, pos)) return false;
    const int T = (int)ids.size(), L = T - drop;
    std::vector<float> all((size_t)T * kText);
    if (!dev_text21_encode_vl(te, ids.data(), T, pos.data(), merged.data(), ds.data(), M, all.data())) return false;
    emb.assign(all.begin() + (size_t)drop * kText, all.end());
    pad.resize(L);
    for (int j = 0; j < L; ++j) pad[j] = ids[drop + j] == kImagePad;
    /* the VAE encoder of the RGBA image in [-1, 1] (VaeImageProcessor: x / 255, then 2 x - 1, in float32) */
    std::vector<float> img((size_t)4 * ch * cw);
    for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x)
            for (int c = 0; c < 4; ++c)
                img[((size_t)c * ch + y) * cw + x] = 2.0f * ((float)rgba[((size_t)y * cw + x) * 4 + c] / 255.0f) - 1.0f;
    cond.resize((size_t)(ch / 16) * (cw / 16) * 64);
    return dev_vae21_encode(vae, img.data(), ch, cw, cond.data());
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

/* the latent through the decoder to 8-bit RGBA, rounded (the pipeline's postprocess: x / 2 + 0.5, clamped, times 255) */
void decode_rgba8(qi21_pipe_t* pipe, const float* latent, int height, int width, unsigned char* rgba) {
    std::vector<float> pix((size_t)4 * height * width);
    if (!qwenimage21::decode(pipe->vae, latent, height / 16, width / 16, pix.data())) throw std::runtime_error("the decoder failed");
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 4; ++c) {
                const float v = (pix[((size_t)c * height + y) * width + x] + 1.0f) * 127.5f;
                const int q = (int)std::lrint(v);
                rgba[((size_t)y * width + x) * 4 + c] = (unsigned char)(q < 0 ? 0 : q > 255 ? 255 : q);
            }
}

/* the seed's noise, packed [gh gw][64]: torch.randn(1, 1, 64, gh, gw) after manual_seed(seed), then _pack_latents */
std::vector<float> packed_noise(unsigned long long seed, int gh, int gw) {
    const int N = gh * gw;
    std::vector<float> chw((size_t)64 * N), noise((size_t)N * 64);
    torch_randn(seed, chw.data(), chw.size());
    for (int c = 0; c < 64; ++c)
        for (int p = 0; p < N; ++p) noise[(size_t)p * 64 + c] = chw[(size_t)c * N + p];
    return noise;
}

/* the edit, from the condition image's bytes to RGBA (the output's size in width, height) */
void run_edit(qi21_pipe_t* pipe, const std::string& prompt, const unsigned char* image, size_t n, int resolution, int steps,
              unsigned long long seed, std::vector<unsigned char>& rgba, int& width, int& height,
              void (*progress)(int step, int steps, void* user), void* user) {
    unsigned char* in = nullptr;
    int w = 0, h = 0, alpha = 0;
    if (image_decode_rgba(image, n, &in, &w, &h, &alpha) != 0) throw std::runtime_error("the condition image does not decode");
    std::vector<unsigned char> src(in, in + (size_t)w * h * 4);
    image_free(in);
    if (std::max(w, h) > 200 * std::min(w, h)) throw std::runtime_error("the condition image's aspect ratio is beyond 200");
    int cw = 0, ch = 0;
    qwenimage21::edit_size(w, h, resolution, cw, ch);
    if (cw < 256 || ch < 256 || cw > 2048 || ch > 2048)
        throw std::runtime_error("the condition image's size at this resolution is outside 256 to 2048");
    const std::vector<unsigned char> cond_rgba = qwenimage21::resize_condition(src.data(), w, h, cw, ch);
    std::vector<float> emb, cond;
    std::vector<unsigned char> pad;
    if (!qwenimage21::encode_condition(pipe->te, pipe->vae, pipe->tok, prompt, cond_rgba.data(), cw, ch, emb, pad, cond))
        throw std::runtime_error("the condition's encoders failed");
    /* the output: the condition's size (one image: the last image's aspect ratio) */
    width = cw;
    height = ch;
    const int gh = height / 16, gw = width / 16;
    qwenimage21::Condition c;
    c.enc = emb.data();
    c.L = (int)pad.size();
    c.pad = pad.data();
    c.cond = cond.data();
    c.grids.push_back({ch / 16, cw / 16});
    const std::vector<float> noise = packed_noise(seed, gh, gw);
    std::vector<float> latent(noise.size());
    auto on_step = [&](int i, const float*, const float*) {
        if (progress) progress(i + 1, steps, user);
    };
    if (!qwenimage21::sample(pipe->dit, noise.data(), gh, gw, c, steps, latent.data(), on_step))
        throw std::runtime_error("the sampler failed");
    rgba.resize((size_t)width * height * 4);
    decode_rgba8(pipe, latent.data(), height, width, rgba.data());
}

/* the background job: make() fills the RGBA image and its size, or returns an error; the PNG kept for qi21_take */
int start_job(qi21_pipe_t* pipe, int steps,
              std::function<std::string(std::vector<unsigned char>&, int&, int&, void (*)(int, int, void*), void*)> make) {
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
    pipe->th = std::thread([pipe, make]() {
        std::vector<unsigned char> rgba;
        std::vector<uint8_t> png;
        std::string err;
        int w = 0, h = 0;
        {
            /* the device's other users (the image and audio generators, the 3D jobs) wait while the image is made */
            std::lock_guard<std::recursive_mutex> dev(sddev::run_mutex());
            auto progress = [](int step, int, void* user) {
                qi21_pipe_t* p = static_cast<qi21_pipe_t*>(user);
                std::lock_guard<std::mutex> lk(p->mu);
                p->step = step;
            };
            err = make(rgba, w, h, progress, pipe);
        }
        if (err.empty()) {
            png = trellis::encode_png(std::vector<uint8_t>(rgba.begin(), rgba.end()), w, h, 4);
            if (png.empty()) err = "qwen-image 2.1: the PNG encoder failed";
        }
        std::lock_guard<std::mutex> lk(pipe->mu);
        pipe->t1 = std::chrono::steady_clock::now();
        if (err.empty()) {
            pipe->png = std::move(png);
            pipe->status = 1;
        } else {
            pipe->error = err;
            pipe->status = -1;
        }
    });
    return 0;
}

/* the edit's options */
bool check_edit(qi21_pipe_t* pipe, const char* prompt, const unsigned char* image, size_t n, int resolution, int steps) {
    if (!pipe || !prompt || !image || !n) {
        g_error = "qwen-image 2.1: a pipeline, a prompt and an image are needed";
        return false;
    }
    if (resolution < 256 || resolution > 2048) {
        g_error = "qwen-image 2.1: the resolution: 256 to 2048";
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
        const std::vector<float> noise = packed_noise(seed, gh, gw);
        std::vector<float> latent((size_t)N * 64);
        auto on_step = [&](int i, const float*, const float*) {
            if (progress) progress(i + 1, steps, user);
        };
        if (!qwenimage21::sample(pipe->dit, noise.data(), gh, gw, emb, L, steps, latent.data(), on_step))
            throw std::runtime_error("the sampler failed");
        decode_rgba8(pipe, latent.data(), height, width, rgba);
    } catch (const std::exception& e) {
        g_error = std::string("qwen-image 2.1: ") + e.what();
        return -1;
    }
    return 0;
}

int qi21_start(qi21_pipe_t* pipe, const char* prompt, int height, int width, int steps, unsigned long long seed) {
    g_error.clear();
    if (!check_options(pipe, prompt, height, width, steps)) return -1;
    const std::string text(prompt);
    return start_job(pipe, steps, [pipe, text, height, width, steps, seed](std::vector<unsigned char>& rgba, int& w, int& h,
                                                                           void (*progress)(int, int, void*), void* user) {
        rgba.resize((size_t)height * width * 4);
        w = width;
        h = height;
        if (qi21_generate(pipe, text.c_str(), height, width, steps, seed, rgba.data(), progress, user) != 0) {
            const std::string e = qi21_last_error();
            return e.empty() ? std::string("qwen-image 2.1: the job failed") : e;
        }
        return std::string();
    });
}

int qi21_edit(qi21_pipe_t* pipe, const char* prompt, const unsigned char* image, size_t n, int resolution, int steps,
              unsigned long long seed, unsigned char** rgba, int* width, int* height, void (*progress)(int step, int steps, void* user),
              void* user) {
    g_error.clear();
    if (!check_edit(pipe, prompt, image, n, resolution, steps)) return -1;
    if (!rgba || !width || !height) {
        g_error = "qwen-image 2.1: no output";
        return -1;
    }
    try {
        std::vector<unsigned char> out;
        int w = 0, h = 0;
        run_edit(pipe, prompt, image, n, resolution, steps, seed, out, w, h, progress, user);
        unsigned char* p = static_cast<unsigned char*>(std::malloc(out.size()));
        if (!p) throw std::runtime_error("out of memory");
        std::memcpy(p, out.data(), out.size());
        *rgba = p;
        *width = w;
        *height = h;
    } catch (const std::exception& e) {
        g_error = std::string("qwen-image 2.1: ") + e.what();
        return -1;
    }
    return 0;
}

int qi21_start_edit(qi21_pipe_t* pipe, const char* prompt, const unsigned char* image, size_t n, int resolution, int steps,
                    unsigned long long seed) {
    g_error.clear();
    if (!check_edit(pipe, prompt, image, n, resolution, steps)) return -1;
    const std::string text(prompt);
    const std::vector<unsigned char> bytes(image, image + n);
    return start_job(pipe, steps, [pipe, text, bytes, resolution, steps, seed](std::vector<unsigned char>& rgba, int& w, int& h,
                                                                               void (*progress)(int, int, void*), void* user) {
        try {
            run_edit(pipe, text, bytes.data(), bytes.size(), resolution, steps, seed, rgba, w, h, progress, user);
        } catch (const std::exception& e) {
            return std::string("qwen-image 2.1: ") + e.what();
        }
        return std::string();
    });
}

int qi21_edit_size(int w, int h, int resolution, int* width, int* height) {
    if (w < 1 || h < 1 || resolution < 1 || !width || !height) return -1;
    qwenimage21::edit_size(w, h, resolution, *width, *height);
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
