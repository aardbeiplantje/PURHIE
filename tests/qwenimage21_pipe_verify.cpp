/* Qwen-Image 2.1's text to image in the engine (src/qwenimage21_pipe.h) against the reference pipeline (diffusers 0.43.0 in float32 on
 * the ComfyUI files, the int8 weights dequantised exactly: tools/qwenimage21_oracle.py, whose outputs are in DIR). Each stage is checked
 * from the reference's own inputs, so its error is its own: the prompt's ids and the system tokens dropped (exact), the text
 * embeddings, the schedule's sigmas, the seed's noise, the first step's velocity, the whole sampler (the reference's noise and
 * embeddings), the decoder (the reference's latent); then the whole chain from the prompt and the seed (qi21_generate) against the
 * reference's image, and the engine's image written as DIR/engine.png. The vocabulary and merges are the Qwen3 tokenizer's dump
 * (/data/zimage-ref/te/vocab.txt, merges.txt: the same tokenizer as Qwen3-VL's). Run from build/co (the kernels).
 * Usage: qwenimage21_pipe_verify [DIR] [PROMPT] [WIDTH] [HEIGHT] [STEPS] [SEED] [DIT]  (the oracle's defaults: 512 x 512, 8 steps,
 * seed 0; DIT the diffusion model's file, the int8 ConvRot one by default, the BF16 one with the oracle's --dit) */
#include "qwenimage21_pipe.h"

#include "image.h"
#include "qwenimage21_dit.h"
#include "qwenimage21_text.h"
#include "torch_randn.h"
#include "trellis_bake.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static int g_fail = 0;

static std::vector<std::string> lines(const char* path) {
    std::ifstream f(path);
    std::vector<std::string> v;
    std::string s;
    while (std::getline(f, s)) v.push_back(s);
    return v;
}

template <typename T>
static bool read_all(const std::string& path, std::vector<T>& v) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t)n / sizeof(T));
    const bool ok = fread(v.data(), sizeof(T), v.size(), f) == v.size();
    fclose(f);
    return ok;
}

/* the relative L2 error and the smallest row cosine (rows of `row` values) */
static void compare(const char* what, const float* a, const float* ref, size_t n, size_t row, double max_rel, double min_cos) {
    double d2 = 0, r2 = 0, worst = 1.0;
    for (size_t i = 0; i < n; ++i) {
        d2 += (double)(a[i] - ref[i]) * (a[i] - ref[i]);
        r2 += (double)ref[i] * ref[i];
    }
    for (size_t r = 0; r + row <= n; r += row) {
        double ab = 0, aa = 0, bb = 0;
        for (size_t i = 0; i < row; ++i) {
            ab += (double)a[r + i] * ref[r + i];
            aa += (double)a[r + i] * a[r + i];
            bb += (double)ref[r + i] * ref[r + i];
        }
        if (aa > 0 && bb > 0) worst = std::min(worst, ab / std::sqrt(aa * bb));
    }
    const double rel = std::sqrt(d2 / (r2 > 0 ? r2 : 1));
    const bool ok = rel <= max_rel && worst >= min_cos;
    printf("[%s] %s: relative %.3g, min row cosine %.6f\n", ok ? "PASS" : "FAIL", what, rel, worst);
    g_fail += !ok;
}

static double secs(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/zimage-ref/qwenimage2/pipe";
    const std::string prompt = argc > 2 ? argv[2] : "a red cube on a wooden table, soft studio light";
    const int W = argc > 3 ? atoi(argv[3]) : 512, H = argc > 4 ? atoi(argv[4]) : 512;
    const int steps = argc > 5 ? atoi(argv[5]) : 8;
    const unsigned long long seed = argc > 6 ? strtoull(argv[6], nullptr, 10) : 0;
    const char* dit = argc > 7 ? argv[7] : "/comfyui/diffusion_models/qwen_image_2.1_int8_convrot.safetensors";
    const char* vae = "/comfyui/vae/qwen_image_2.1_vae_bf16.safetensors";
    const char* te = "/comfyui/text_encoders/qwen3vl_8b_int8_convrot.safetensors";
    const int gh = H / 16, gw = W / 16, N = gh * gw;

    auto vocab = lines("/data/zimage-ref/te/vocab.txt");
    auto merges = lines("/data/zimage-ref/te/merges.txt");
    std::vector<const char*> vp, mp;
    for (auto& s : vocab) vp.push_back(s.c_str());
    for (auto& s : merges) mp.push_back(s.c_str());
    tokenizer_t* tok = tokenizer_create(vp.data(), vp.size(), mp.data(), mp.size());
    {  /* the stages 1 to 7 map and register the files for their own checks; the pipeline (stage 8) registers them again, and the
     * process's registration limit (about 29 GB) would not hold both: the mappings are released here */
    zimage::SafeTensors st_dit(dit), st_vae(vae), st_te(te);
    if (!st_dit.ok() || !st_vae.ok() || !st_te.ok() || !st_dit.register_device() || !st_vae.register_device() || !st_te.register_device()) {
        printf("[FAIL] the model files\n");
        return 1;
    }

    /* 1. the prompt's ids and the system tokens dropped */
    std::vector<int32_t> ref_ids;
    std::vector<int> ids;
    int drop = -1;
    const int ref_drop = atoi(lines((dir + "/drop.txt").c_str()).at(0).c_str());
    if (!read_all(dir + "/ids.i32", ref_ids) || !qwenimage21::prompt_ids(tok, prompt, ids, drop)) {
        printf("[FAIL] the ids\n");
        return 1;
    }
    const bool ids_ok = ids.size() == ref_ids.size() && std::equal(ids.begin(), ids.end(), ref_ids.begin()) && drop == ref_drop;
    printf("[%s] the ids: %zu tokens (the reference %zu), %d system tokens dropped (the reference %d)\n", ids_ok ? "PASS" : "FAIL",
           ids.size(), ref_ids.size(), drop, ref_drop);
    g_fail += !ids_ok;

    /* 2. the text embeddings (the reference's ids) */
    std::vector<float> ref_emb;
    read_all(dir + "/emb.f32", ref_emb);
    const int T = (int)ref_ids.size(), L = T - ref_drop;
    std::vector<float> all((size_t)T * 4096);
    auto t0 = std::chrono::steady_clock::now();
    if (!qwenimage21::dev_text21_encode(st_te, ref_ids.data(), T, all.data())) {
        printf("[FAIL] the text encoder\n");
        return 1;
    }
    printf("       the text encoder: %d tokens in %.2f s\n", T, secs(t0));
    if (ref_emb.size() != (size_t)L * 4096) {
        printf("[FAIL] the reference's embeddings: %zu values, %d x 4096 expected\n", ref_emb.size(), L);
        return 1;
    }
    compare("the text embeddings", all.data() + (size_t)ref_drop * 4096, ref_emb.data(), ref_emb.size(), 4096, 2e-3, 0.9999);

    /* 3. the sigmas */
    std::vector<float> ref_sig, sig;
    read_all(dir + "/sigmas.f32", ref_sig);
    qwenimage21::sigmas(steps, N, sig);
    double sd = sig.size() == ref_sig.size() ? 0 : 1;
    for (size_t i = 0; i < sig.size() && i < ref_sig.size(); ++i) sd = std::max(sd, (double)std::fabs(sig[i] - ref_sig[i]));
    printf("[%s] the sigmas: %zu, largest difference %.3g\n", sd < 2e-6 ? "PASS" : "FAIL", sig.size(), sd);
    g_fail += !(sd < 2e-6);

    /* 4. the seed's noise, packed */
    std::vector<float> ref_noise, chw((size_t)64 * N), noise((size_t)N * 64);
    read_all(dir + "/noise.f32", ref_noise);
    torch_randn(seed, chw.data(), chw.size());
    for (int c = 0; c < 64; ++c)
        for (int q = 0; q < N; ++q) noise[(size_t)q * 64 + c] = chw[(size_t)c * N + q];
    double nd = ref_noise.size() == noise.size() ? 0 : 1;
    for (size_t i = 0; i < noise.size() && i < ref_noise.size(); ++i) nd = std::max(nd, (double)std::fabs(noise[i] - ref_noise[i]));
    printf("[%s] the noise: largest difference %.3g\n", nd < 1e-5 ? "PASS" : "FAIL", nd);
    g_fail += !(nd < 1e-5);

    /* 5. the first step's velocity, and 6. the whole sampler, from the reference's noise and embeddings */
    std::vector<float> ref_v0, ref_x1, ref_lat, lat((size_t)N * 64), v0, x1;
    read_all(dir + "/v0.f32", ref_v0);
    read_all(dir + "/x1.f32", ref_x1);
    read_all(dir + "/latent.f32", ref_lat);
    t0 = std::chrono::steady_clock::now();
    auto on_step = [&](int i, const float* v, const float* x) {
        if (i == 0) {
            v0.assign(v, v + (size_t)N * 64);
            x1.assign(x, x + (size_t)N * 64);
        }
    };
    if (!qwenimage21::sample(st_dit, ref_noise.data(), gh, gw, ref_emb.data(), L, steps, lat.data(), on_step)) {
        printf("[FAIL] the sampler\n");
        return 1;
    }
    printf("       the sampler: %d steps at %d x %d in %.2f s\n", steps, W, H, secs(t0));
    compare("the first step's velocity", v0.data(), ref_v0.data(), v0.size(), 64, 2e-2, 0.999);
    /* the DiT is a BF16 chain where it counts, as Z-Image's: the linears' outputs (the BF16 file) and the image attention's q, k and v
     * rounded to BF16. After 8 steps about 3% (the worst latent row's cosine about 0.97); the image is the gate below */
    compare("the latent after the first step", x1.data(), ref_x1.data(), x1.size(), 64, 2e-3, 0.9999);
    compare("the final latent", lat.data(), ref_lat.data(), lat.size(), 64, 5e-2, 0.95);

    /* 7. the decoder (the reference's latent) */
    std::vector<float> ref_dec, dec((size_t)4 * H * W);
    read_all(dir + "/dec.f32", ref_dec);
    t0 = std::chrono::steady_clock::now();
    if (!qwenimage21::decode(st_vae, ref_lat.data(), gh, gw, dec.data())) {
        printf("[FAIL] the decoder\n");
        return 1;
    }
    printf("       the decoder: %d x %d in %.2f s\n", W, H, secs(t0));
    compare("the decoded image", dec.data(), ref_dec.data(), dec.size(), (size_t)H * W, 1e-2, 0.9999);

    }
    qi21_pipe_t* p = qi21_open(dit, vae, te, vp.data(), vp.size(), mp.data(), mp.size());
    if (!p) {
        printf("[FAIL] qi21_open: %s\n", qi21_last_error());
        return 1;
    }
    /* 8. the whole chain from the prompt and the seed, against the reference's image */
    std::vector<unsigned char> rgba((size_t)H * W * 4), png;
    auto t0 = std::chrono::steady_clock::now();
    if (qi21_generate(p, prompt.c_str(), H, W, steps, seed, rgba.data(), nullptr, nullptr) != 0) {
        printf("[FAIL] qi21_generate: %s\n", qi21_last_error());
        return 1;
    }
    printf("       the whole chain: %.2f s\n", secs(t0));
    const auto enc = trellis::encode_png(std::vector<uint8_t>(rgba.begin(), rgba.end()), W, H, 4);
    if (FILE* f = fopen((dir + "/engine.png").c_str(), "wb")) {
        fwrite(enc.data(), 1, enc.size(), f);
        fclose(f);
    }
    if (read_all(dir + "/image.png", png)) {
        unsigned char* ref = nullptr;
        int rw = 0, rh = 0, alpha = 0;
        if (image_decode_rgba(png.data(), png.size(), &ref, &rw, &rh, &alpha) == 0 && rw == W && rh == H) {
            double se = 0, ae = 0;
            size_t within3 = 0;
            for (size_t i = 0; i < rgba.size(); ++i) {
                const int d = (int)rgba[i] - (int)ref[i];
                se += (double)d * d;
                ae += std::abs(d);
                within3 += std::abs(d) <= 3;
            }
            const double mse = se / rgba.size(), psnr = mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : 99;
            const bool ok = psnr >= 35;
            printf("[%s] the image against the reference's: PSNR %.2f dB, mean %.2f levels, %.1f%% within 3\n", ok ? "PASS" : "FAIL", psnr,
                   ae / rgba.size(), 100.0 * within3 / rgba.size());
            g_fail += !ok;
            image_free(ref);
        } else {
            printf("[FAIL] the reference's image.png\n");
            ++g_fail;
        }
    }
    tokenizer_free(tok);
    qi21_close(p);
    printf(g_fail ? "%d FAILED\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
