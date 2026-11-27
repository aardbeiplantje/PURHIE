/* Qwen-Image 2.1's image edit in the engine (src/qwenimage21_pipe.h) against the reference pipeline (diffusers 0.43.0's
 * QwenImage21Pipeline with a condition image, in float32 on the ComfyUI files: tools/qwenimage21_edit_oracle.py, whose outputs are
 * in DIR). Each stage is checked from the reference's own inputs, so its error is its own: the condition image's pixel rows for the
 * vision tower (its merged tokens and the three deepstack features), the text encoder with the image's tokens (the M-RoPE positions,
 * the deepstack features at the pads), the condition's latent (the VAE encoder), the sampler with the condition (the velocity of the
 * first step, the latent after it, the final latent), the decoder, and the whole edit from the image's bytes (qi21_edit) against the
 * reference image. Each stage takes the reference's own inputs. Run from build/co (the kernels).
 * Usage: qwenimage21_edit_verify [DIR] [IMAGE] [DIT]  (IMAGE: the condition image's file, default ../pipe/image.png; DIT: the
 * diffusion model's file, the int8 ConvRot one by default) */
#include "qwenimage21_pipe.h"

#include "image.h"
#include "qwenimage21_dit.h"
#include "qwenimage21_text.h"
#include "qwenimage21_vae.h"
#include "qwenimage21_vision.h"
#include "trellis_bake.h"
#include "tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static int g_fail = 0;

static std::vector<std::string> lines(const std::string& path) {
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
    const std::string dir = argc > 1 ? argv[1] : "/data/zimage-ref/qwenimage2/edit";
    const std::string image_path = argc > 2 ? argv[2] : "/data/zimage-ref/qwenimage2/pipe/image.png";
    /* the diffusion model: the int8 ConvRot file by default (the oracle's dequantised weights), or the BF16 file (argv 3) */
    const std::string dit = argc > 3 ? argv[3] : "/comfyui/diffusion_models/qwen_image_2.1_int8_convrot.safetensors";
    const char* te = "/comfyui/text_encoders/qwen3vl_8b_int8_convrot.safetensors";
    zimage::SafeTensors st_te(te);
    if (!st_te.ok() || !st_te.register_device()) {
        printf("[FAIL] the model files\n");
        return 1;
    }
    int cw = 0, ch = 0;
    {
        const auto l = lines(dir + "/cond_size.txt");
        if (l.empty() || sscanf(l[0].c_str(), "%d %d", &cw, &ch) != 2) {
            printf("[FAIL] %s/cond_size.txt\n", dir.c_str());
            return 1;
        }
    }

    /* 1. the condition image's pixel rows (the reference's resized image) */
    std::vector<unsigned char> crgba;
    std::vector<float> ref_px, px;
    std::vector<int32_t> grid;
    int gh = 0, gw = 0;
    if (!read_all(dir + "/cond_rgba.u8", crgba) || crgba.size() != (size_t)cw * ch * 4 || !read_all(dir + "/pixels.f32", ref_px) ||
        !read_all(dir + "/grid.i32", grid) || grid.size() != 3) {
        printf("[FAIL] the reference's condition image\n");
        return 1;
    }
    if (!qwenimage21::condition_pixels(crgba.data(), cw, ch, px, gh, gw)) {
        printf("[FAIL] condition_pixels\n");
        return 1;
    }
    {
        double md = px.size() == ref_px.size() ? 0 : 1;
        for (size_t i = 0; i < px.size() && i < ref_px.size(); ++i) md = std::max(md, (double)std::fabs(px[i] - ref_px[i]));
        const bool ok = gh == grid[1] && gw == grid[2] && md < 1e-5;
        printf("[%s] the pixel rows: grid %d x %d (the reference %d x %d), largest difference %.3g\n", ok ? "PASS" : "FAIL", gh, gw,
               grid[1], grid[2], md);
        g_fail += !ok;
    }

    /* 2. the vision tower (the reference's pixel rows) */
    {
        const int M = grid[1] * grid[2] / 4;
        std::vector<float> merged((size_t)M * 4096), ds((size_t)3 * M * 4096), ref_m, ref_d;
        auto t0 = std::chrono::steady_clock::now();
        if (!qwenimage21::dev_vision21_encode(st_te, ref_px.data(), grid[1], grid[2], merged.data(), ds.data())) {
            printf("[FAIL] the vision tower\n");
            return 1;
        }
        printf("       the vision tower: %d x %d patches in %.2f s\n", grid[1], grid[2], secs(t0));
        read_all(dir + "/vis_merged.f32", ref_m);
        if (ref_m.size() != merged.size()) {
            printf("[FAIL] the reference's merged tokens: %zu values, %zu expected\n", ref_m.size(), merged.size());
            return 1;
        }
        compare("the vision tower's merged tokens", merged.data(), ref_m.data(), merged.size(), 4096, 2e-2, 0.999);
        for (int i = 0; i < 3; ++i) {
            read_all(dir + "/vis_ds" + std::to_string(i) + ".f32", ref_d);
            if (ref_d.size() != (size_t)M * 4096) {
                printf("[FAIL] the reference's deepstack feature %d\n", i);
                return 1;
            }
            const std::string what = "the deepstack feature " + std::to_string(i);
            compare(what.c_str(), ds.data() + (size_t)i * M * 4096, ref_d.data(), ref_d.size(), 4096, 2e-2, 0.999);
        }
    }

    /* 3 to 6 need the text encoder, the VAE and the DiT mapped for the device: the stages' own registrations, released with them */
    {
        zimage::SafeTensors st_dit(dit);
        zimage::SafeTensors st_vae("/comfyui/vae/qwen_image_2.1_vae_bf16.safetensors");
        if (!st_dit.ok() || !st_vae.ok() || !st_dit.register_device() || !st_vae.register_device()) {
            printf("[FAIL] the DiT or the VAE files\n");
            return 1;
        }
        /* the text encoder over the reference's ids and positions (the image pads take the vision tower's tokens): the embeddings
         * after the system turn */
        std::vector<int32_t> ids, pos;
        std::vector<unsigned char> mask;  /* the image pads, one byte each */
        std::vector<float> ref_emb, merged, all;
        read_all(dir + "/ids.i32", ids);
        read_all(dir + "/pos.i32", pos);
        read_all(dir + "/emb.f32", ref_emb);
        read_all(dir + "/imgmask.u8", mask);
        const int drop = atoi(lines(dir + "/drop.txt").at(0).c_str());
        const int T = (int)ids.size(), L = T - drop, M = (int)std::count(mask.begin(), mask.end(), 1);
        read_all(dir + "/vis_merged.f32", merged);
        std::vector<float> ds3((size_t)3 * M * 4096);  /* the deepstack features, one file per layer */
        for (int i = 0; i < 3; ++i) {
            std::vector<float> d;
            read_all(dir + "/vis_ds" + std::to_string(i) + ".f32", d);
            std::copy(d.begin(), d.end(), ds3.begin() + (size_t)i * M * 4096);
        }
        all.resize((size_t)T * 4096);
        std::vector<int> ids_i(ids.begin(), ids.end()), pos_i(pos.begin(), pos.end());
        auto t0 = std::chrono::steady_clock::now();
        if (!qwenimage21::dev_text21_encode_vl(st_te, ids_i.data(), T, pos_i.data(), merged.data(), ds3.data(), M, all.data())) {
            printf("[FAIL] the text encoder with the image\n");
            return 1;
        }
        printf("       the text encoder: %d tokens (%d image pads) in %.2f s\n", T, M, secs(t0));
        compare("the text embeddings with the image", all.data() + (size_t)drop * 4096, ref_emb.data(), ref_emb.size(), 4096, 2e-3, 0.9999);

        /* the condition's latent: the VAE encoder over the resized condition image in [-1, 1], channel-major */
        int cw2 = 0, ch2 = 0;
        {
            const auto l = lines(dir + "/cond_size.txt");
            sscanf(l[0].c_str(), "%d %d", &cw2, &ch2);
        }
        std::vector<float> img4((size_t)4 * ch2 * cw2), cond_ref, cond((size_t)(ch2 / 16) * (cw2 / 16) * 64);
        for (int y = 0; y < ch2; ++y)
            for (int x = 0; x < cw2; ++x)
                for (int c = 0; c < 4; ++c)
                    img4[((size_t)c * ch2 + y) * cw2 + x] = 2.0f * ((float)crgba[((size_t)y * cw2 + x) * 4 + c] / 255.0f) - 1.0f;
        if (!qwenimage21::dev_vae21_encode(st_vae, img4.data(), ch2, cw2, cond.data())) {
            printf("[FAIL] the VAE encoder\n");
            return 1;
        }
        read_all(dir + "/cond.f32", cond_ref);
        compare("the condition's latent", cond.data(), cond_ref.data(), cond.size(), 64, 1e-2, 0.998);  /* the encoder's BF16 weights: products rounded */

        /* the sampler with the condition: the reference's noise and embeddings; the first velocity, the latent after it, the final */
        const int gh = ch2 / 16, gw = cw2 / 16, N = gh * gw;
        std::vector<float> noise, ref_v0, ref_x1, ref_lat, lat((size_t)N * 64), v0, x1;
        read_all(dir + "/noise.f32", noise);
        read_all(dir + "/v0.f32", ref_v0);
        read_all(dir + "/x1.f32", ref_x1);
        read_all(dir + "/latent.f32", ref_lat);
        std::vector<unsigned char> pad(mask.begin(), mask.end());
        qwenimage21::Condition c;
        c.enc = ref_emb.data();
        c.L = L;
        c.pad = pad.data();
        c.cond = cond_ref.data();
        c.grids.push_back({gh, gw});
        const int steps = 8;
        t0 = std::chrono::steady_clock::now();
        if (!qwenimage21::sample(st_dit, noise.data(), gh, gw, c, steps, lat.data(),
                                 [&](int i, const float* v, const float* xx) {
                                     if (i == 0) {
                                         v0.assign(v, v + (size_t)N * 64);
                                         x1.assign(xx, xx + (size_t)N * 64);
                                     }
                                 })) {
            printf("[FAIL] the sampler with the condition\n");
            return 1;
        }
        printf("       the sampler: %d steps at %d x %d in %.2f s\n", steps, cw2, ch2, secs(t0));
        compare("the first step's velocity (with the condition)", v0.data(), ref_v0.data(), v0.size(), 64, 2e-2, 0.999);
        compare("the latent after the first step (with the condition)", x1.data(), ref_x1.data(), x1.size(), 64, 2e-3, 0.9999);
        compare("the final latent (with the condition)", lat.data(), ref_lat.data(), lat.size(), 64, 5e-2, 0.95);

        /* the decoder (the reference's latent) */
        std::vector<float> ref_dec, dec((size_t)4 * ch2 * cw2);
        read_all(dir + "/dec.f32", ref_dec);
        if (!qwenimage21::decode(st_vae, ref_lat.data(), gh, gw, dec.data())) {
            printf("[FAIL] the decoder\n");
            return 1;
        }
        compare("the decoded image (with the condition)", dec.data(), ref_dec.data(), dec.size(), (size_t)ch2 * cw2, 1e-2, 0.9999);
    }

    /* the whole edit from the image's bytes (qi21_edit): the pipeline registers its own files, so the stages' mappings are released */
    {
        std::vector<const char*> vp, mp;
        const auto vocab = lines("/data/zimage-ref/te/vocab.txt"), merges = lines("/data/zimage-ref/te/merges.txt");
        for (auto& v : vocab) vp.push_back(v.c_str());
        for (auto& m : merges) mp.push_back(m.c_str());
        std::vector<unsigned char> bytes;
        if (!read_all(image_path, bytes) || bytes.empty()) {
            printf("[FAIL] the condition image %s\n", image_path.c_str());
            return 1;
        }
        qi21_pipe_t* p = qi21_open(dit.c_str(), "/comfyui/vae/qwen_image_2.1_vae_bf16.safetensors",
                                   "/comfyui/text_encoders/qwen3vl_8b_int8_convrot.safetensors", vp.data(), vp.size(), mp.data(), mp.size());
        if (!p) {
            printf("[FAIL] qi21_open: %s\n", qi21_last_error());
            return 1;
        }
        unsigned char* rgba = nullptr;
        int w = 0, h = 0;
        auto t0 = std::chrono::steady_clock::now();
        if (qi21_edit(p, "make the cube blue", bytes.data(), bytes.size(), 512, 8, 0, &rgba, &w, &h, nullptr, nullptr) != 0) {
            printf("[FAIL] qi21_edit: %s\n", qi21_last_error());
            qi21_close(p);
            return 1;
        }
        printf("       the whole edit: %d x %d in %.2f s\n", w, h, secs(t0));
        unsigned char* ref = nullptr;
        int rw = 0, rh = 0, ra = 0;
        std::vector<unsigned char> refbytes;
        read_all(dir + "/image.png", refbytes);
        if (refbytes.empty() || image_decode_rgba(refbytes.data(), refbytes.size(), &ref, &rw, &rh, &ra) != 0 || rw != w || rh != h) {
            printf("[FAIL] the reference's image.png\n");
            ++g_fail;
        } else {
            double se = 0;
            long within3 = 0;
            for (size_t i = 0; i < (size_t)w * h * 4; ++i) {
                const int d = (int)rgba[i] - (int)ref[i];
                se += (double)d * d;
                within3 += std::abs(d) <= 3;
            }
            const double mse = se / ((double)w * h * 4), psnr = mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : 99;
            const bool ok = psnr >= 35;
            printf("[%s] the edited image against the reference's: PSNR %.2f dB, %.1f%% within 3\n", ok ? "PASS" : "FAIL", psnr,
                   100.0 * within3 / ((double)w * h * 4));
            g_fail += !ok;
            image_free(ref);
        }
        if (FILE* f = fopen((dir + "/engine.png").c_str(), "wb")) {
            const auto enc = trellis::encode_png(std::vector<uint8_t>(rgba, rgba + (size_t)w * h * 4), w, h, 4);
            fwrite(enc.data(), 1, enc.size(), f);
            fclose(f);
        }
        qi21_free(rgba);
        qi21_close(p);
    }

    printf("%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
