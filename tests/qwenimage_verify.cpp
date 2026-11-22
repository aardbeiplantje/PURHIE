/* Qwen-Image on the device against the diffusers oracle (/data/zimage-ref/qwenimage, out/, from ref_stages.py) and a host reference.
 *   dit    the FP8 linear kernel on the DiT's img_in (E4M3 weights and bias, no scale) against a host reference;
 *   te     the text encoder's q_proj (E4M3 weights with the per-tensor scale, BF16 bias) against a host reference;
 *   enc    the whole text encoder on the oracle's 46 template tokens: the last 12 rows against te_embeds;
 *   fwd    the whole DiT on the oracle's packed latent and text embeddings at t = 0.5: the output against dit_out.
 * Each file is mapped once per process (a freed mapping's address is reused by the next file while the driver still holds the
 * old registration: hipErrorHostMemoryAlreadyRegistered). Run from build/co (the kernels). */
#include "qwenimage_dit.h"
#include "qwenimage_text.h"
#include "qwenimage21_dit.h"
#include "qwenimage21_text.h"
#include "qwenimage_pipe.h"
#include "qwenimage_vae.h"
#include "sd_device.h"
#include "zimage_dit.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char* kDitPath = "/comfyui/diffusion_models/qwen_image_fp8_e4m3fn.safetensors";
static const char* kTePath = "/comfyui/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
static const char* kDit21Int8 = "/comfyui/diffusion_models/qwen_image_2.1_int8_convrot.safetensors";
static const char* kDit21Bf16 = "/comfyui/diffusion_models/qwen_image_2.1_bf16.safetensors";
static const char* kQwen3vl21 = "/comfyui/text_encoders/qwen3vl_8b_int8_convrot.safetensors";
static const char* kVaePath = "/comfyui/vae/qwen_image_vae.safetensors";
static const std::string kOut = "/data/zimage-ref/qwenimage/out/";

/* the mapped file, registered for the device on first use */
static zimage::SafeTensors& dit_file() {
    static zimage::SafeTensors d(kDitPath);
    static const bool reg = d.ok() && d.register_device();
    if (!reg) fprintf(stderr, "the DiT mapping did not register\n");
    return d;
}
static zimage::SafeTensors& te_file() {
    static zimage::SafeTensors t(kTePath);
    static const bool reg = t.ok() && t.register_device();
    if (!reg) fprintf(stderr, "the text encoder mapping did not register\n");
    return t;
}

static zimage::SafeTensors& dit21_int8() {
    static zimage::SafeTensors d(kDit21Int8);
    static const bool reg = d.ok() && d.register_device();
    if (!reg) fprintf(stderr, "the 2.1 int8 mapping did not register\n");
    return d;
}
static zimage::SafeTensors& dit21_bf16() {
    static zimage::SafeTensors d(kDit21Bf16);
    static const bool reg = d.ok() && d.register_device();
    if (!reg) fprintf(stderr, "the 2.1 bf16 mapping did not register\n");
    return d;
}

/* the regular Hadamard of 256 (H4 Kronecker powers, normalised by 1/16) and the block rotation of x [rows][K] by it */
static std::vector<double> regular_h256() {
    static const double H4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    std::vector<double> H = {1.0}; int n = 1;
    while (n < 256) {
        std::vector<double> K((size_t)n * 4 * n * 4);
        for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
            K[(size_t)(a * 4 + i) * (n * 4) + (b * 4 + j)] = H[(size_t)a * n + b] * H4[i][j];
        H.swap(K); n *= 4;
    }
    for (double& v : H) v /= 16.0;
    return H;
}

static float e4m3_host(uint8_t v);

static zimage::SafeTensors& qwen3vl21_file() {
    static zimage::SafeTensors t(kQwen3vl21);
    static const bool reg = t.ok() && t.register_device();
    if (!reg) fprintf(stderr, "the Qwen3-VL 2.1 text mapping did not register\n");
    return t;
}

static zimage::SafeTensors& vae_file() {
    static zimage::SafeTensors v(kVaePath);
    static const bool reg = v.ok() && v.register_device();
    if (!reg) fprintf(stderr, "the VAE mapping did not register\n");
    return v;
}

static float e4m3_host(uint8_t v) {
    const int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    float mag;
    if (e == 0) mag = std::ldexp((float)m, -9);
    else if (e == 15 && m == 7) return NAN;
    else mag = std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -mag : mag;
}

static bool read_file_path(const char* path, void* dst, size_t bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    const size_t n = fread(dst, 1, bytes, f);
    fclose(f);
    return n == bytes;
}

static bool read_file(const char* name, void* dst, size_t bytes) {
    FILE* f = fopen((kOut + name).c_str(), "rb");
    if (!f) return false;
    const size_t n = fread(dst, 1, bytes, f);
    fclose(f);
    return n == bytes;
}

/* the device's values against a reference: the cosine per row (the minimum over rows) and the largest error relative to the
 * largest reference value */
struct Metrics {
    double min_cos;
    double max_rel;
};

static Metrics compare_rows(const float* dev, const float* ref, int rows, int cols) {
    double maxref = 0, maxabs = 0, mincos = 1.0;
    for (size_t i = 0; i < (size_t)rows * cols; ++i) maxref = std::fmax(maxref, std::fabs(ref[i]));
    for (int r = 0; r < rows; ++r) {
        const float* a = dev + (size_t)r * cols;
        const float* b = ref + (size_t)r * cols;
        double dot = 0, na = 0, nb = 0;
        for (int c = 0; c < cols; ++c) {
            dot += (double)a[c] * b[c];
            na += (double)a[c] * a[c];
            nb += (double)b[c] * b[c];
            maxabs = std::fmax(maxabs, std::fabs((double)a[c] - b[c]));
        }
        mincos = std::fmin(mincos, dot / std::sqrt(na * nb));
    }
    return {mincos, maxabs / maxref};
}

int main(int argc, char** argv) {
    const bool all = argc < 2;
    auto want = [&](const char* name) { return all || std::strcmp(argv[1], name) == 0; };
    bool ok = true;

    if (want("dit")) {
        /* img_in: [3072][64] E4M3 weights, E4M3 bias; the oracle's packed latent as the input */
        zimage::SafeTensors& dit = dit_file();
        if (!dit.ok()) {
            printf("[FAIL] the DiT file: %s\n", dit.error().c_str());
            return 1;
        }
        const uint8_t *W = nullptr, *B = nullptr;
        std::vector<size_t> sw, sb;
        if (!dit.fp8_in_place("img_in.weight", W, sw) || !dit.fp8_in_place("img_in.bias", B, sb) || sw.size() != 2 || sw[0] != 3072 ||
            sw[1] != 64) {
            printf("[FAIL] img_in tensors\n");
            return 1;
        }
        const int T = 256, in = 64, out = 3072;
        std::vector<float> x((size_t)T * in);
        if (!read_file("dit_lat.f32", x.data(), x.size() * 4)) {
            printf("[FAIL] dit_lat.f32 (run ref_stages.py)\n");
            return 1;
        }
        sddev::Floats xd(x.size()), yd((size_t)T * out);
        std::memcpy(xd.data(), x.data(), x.size() * 4);
        sddev::linear_f8(xd.data(), T, W, 1.0f, in, out, B, 1, yd.data());
        std::vector<float> ref((size_t)T * out);
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < out; ++o) {
                double acc = e4m3_host(B[o]);
                for (int i = 0; i < in; ++i) acc += (double)x[(size_t)t * in + i] * e4m3_host(W[(size_t)o * in + i]);
                ref[(size_t)t * out + o] = (float)acc;
            }
        const Metrics m = compare_rows(yd.data(), ref.data(), T, out);
        const bool pass = m.max_rel < 1e-5;
        ok = ok && pass;
        printf("%s img_in (256 x 64 -> 3072, E4M3 weights and bias): min row cosine %.7f, max error %.3g relative\n",
               pass ? "[PASS]" : "[FAIL]", m.min_cos, m.max_rel);
    }

    if (want("te")) {
        /* q_proj of layer 0: E4M3 weights [3584][3584] with the per-tensor scale, BF16 bias; a seeded input */
        zimage::SafeTensors& te = te_file();
        if (!te.ok()) {
            printf("[FAIL] the text encoder file: %s\n", te.error().c_str());
            return 1;
        }
        const uint8_t* W = nullptr;
        const uint16_t* Bbits = nullptr;
        const float* sc = nullptr;
        std::vector<size_t> sw, sb, ss;
        if (!te.fp8_in_place("model.layers.0.self_attn.q_proj.weight", W, sw) ||
            !te.bf16_in_place("model.layers.0.self_attn.q_proj.bias", Bbits, sb) ||
            !te.f32_in_place("model.layers.0.self_attn.q_proj.scale_weight", sc, ss) || sw.size() != 2 || sw[0] != 3584 ||
            sw[1] != 3584) {
            printf("[FAIL] q_proj tensors\n");
            return 1;
        }
        const float wscale = sc[0];
        const int T = 12, in = 3584, out = 3584;
        std::vector<float> x((size_t)T * in);
        srand(7);
        for (float& v : x) v = (float)(rand() % 2001 - 1000) / 1000.0f;
        sddev::Floats xd(x.size()), yd((size_t)T * out);
        std::memcpy(xd.data(), x.data(), x.size() * 4);
        sddev::linear_f8(xd.data(), T, W, wscale, in, out, Bbits, 2, yd.data());
        std::vector<float> ref((size_t)T * out);
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < out; ++o) {
                const uint32_t bb = (uint32_t)Bbits[o] << 16;  // BF16 bits to float
                float bf;
                std::memcpy(&bf, &bb, 4);
                double dot = 0;
                for (int i = 0; i < in; ++i) dot += (double)x[(size_t)t * in + i] * e4m3_host(W[(size_t)o * in + i]);
                ref[(size_t)t * out + o] = (float)(wscale * dot + bf);  // the kernel's definition: scale the dot, then add the bias
            }
        const Metrics m = compare_rows(yd.data(), ref.data(), T, out);
        const bool pass = m.max_rel < 1e-5;
        ok = ok && pass;
        printf("%s text q_proj (12 x 3584 -> 3584, E4M3 weights x scale %.4g, BF16 bias): min row cosine %.7f, max error %.3g relative\n",
               pass ? "[PASS]" : "[FAIL]", wscale, m.min_cos, m.max_rel);
    }

    /* Qwen-Image 2.1's ConvRot linear (int8, rotated) on one real layer: against the exact host computation of the same quantized
     * layer (rotate x, dequantise q s, multiply in double), and against the BF16 original (the quantization error only) */
    if (want("convrot")) {
        zimage::SafeTensors& q = dit21_int8();
        zimage::SafeTensors& b16 = dit21_bf16();
        const char* L = "transformer_blocks.0.attn.to_q";
        const int8_t* Wq = nullptr;
        const float* sc = nullptr;
        const uint16_t* Wb = nullptr;
        std::vector<size_t> sw, ss, sb;
        if (!q.ok() || !b16.ok() || !q.i8_in_place(std::string(L) + ".weight", Wq, sw) ||
            !q.f32_in_place(std::string(L) + ".weight_scale", sc, ss) || !b16.bf16_in_place(std::string(L) + ".weight", Wb, sb) ||
            sw.size() != 2 || sw[0] != 4096 || sw[1] != 4096) {
            printf("[FAIL] the 2.1 layer tensors (%s)\n", q.error().c_str());
            return 1;
        }
        const int T = 64, K = 4096, N = 4096;
        std::vector<float> x((size_t)T * K);
        srand(11);
        for (float& v : x) v = (float)(rand() % 2001 - 1000) / 1000.0f;
        std::vector<double> H = regular_h256();
        /* the rotation of the activations on the host (exact), then the device */
        std::vector<double> xr((size_t)T * K);
        for (int t = 0; t < T; ++t)
            for (int blk = 0; blk < K / 256; ++blk)
                for (int c = 0; c < 256; ++c) {
                    double acc = 0;
                    for (int j = 0; j < 256; ++j) acc += (double)x[(size_t)t * K + blk * 256 + j] * H[(size_t)j * 256 + c];
                    xr[(size_t)t * K + blk * 256 + c] = acc;
                }
        std::vector<double> ref_q((size_t)T * N), ref_bf((size_t)T * N);
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < N; ++o) {
                double a = 0, bsum = 0;
                const double so = sc[o];
                for (int i = 0; i < K; ++i) {
                    a += xr[(size_t)t * K + i] * (double)Wq[(size_t)o * K + i] * so;
                    const uint32_t bits = (uint32_t)Wb[(size_t)o * K + i] << 16;
                    float wf;
                    std::memcpy(&wf, &bits, 4);
                    bsum += (double)x[(size_t)t * K + i] * wf;
                }
                ref_q[(size_t)t * N + o] = a;
                ref_bf[(size_t)t * N + o] = bsum;
            }
        sddev::Floats xd((size_t)T * K), xrd((size_t)T * K), yd((size_t)T * N), Hd(256 * 256);
        std::memcpy(xd.data(), x.data(), x.size() * 4);
        std::vector<float> Hf(256 * 256);
        for (size_t i = 0; i < Hf.size(); ++i) Hf[i] = (float)H[i];
        std::memcpy(Hd.data(), Hf.data(), Hf.size() * 4);
        sddev::linear_convrot(xd.data(), T, Wq, sc, Hd.data(), K, N, xrd.data(), yd.data());
        std::vector<float> yv(yd.data(), yd.data() + (size_t)T * N);
        std::vector<float> rq(ref_q.begin(), ref_q.end()), rb(ref_bf.begin(), ref_bf.end());
        const Metrics mq = compare_rows(yv.data(), rq.data(), T, N);
        const Metrics mb = compare_rows(yv.data(), rb.data(), T, N);
        const bool pass = mq.max_rel <= 1e-4 && mb.max_rel <= 0.03;
        ok = ok && pass;
        printf("%s ConvRot linear (2.1 %s, 64 x 4096 -> 4096): against the exact quantized layer %.3g relative; against the BF16 original %.3g relative (min row cosine %.6f)\n",
               pass ? "[PASS]" : "[FAIL]", L, mq.max_rel, mb.max_rel, mb.min_cos);
    }

    /* Qwen-Image 2.1's text encoder (Qwen3-VL 8B, int8 ConvRot) on the oracle's tokens (ref_te21_fp32.py): the kept rows (the system
     * tokens dropped) against the FP32 oracle */
    if (want("te21")) {
        std::vector<int32_t> ids(4096);
        const size_t nbytes = 0;
        (void)nbytes;
        FILE* f = fopen("/data/zimage-ref/qwen3vl/out/te21_ids.i32", "rb");
        if (!f) { printf("[FAIL] te21_ids.i32 (run ref_te21_fp32.py)\n"); return 1; }
        const size_t nids = fread(ids.data(), 4, ids.size(), f);
        fclose(f);
        ids.resize(nids);
        const int T = (int)nids;
        std::vector<float> ref(15 * 4096);
        if (!read_file_path("/data/zimage-ref/qwen3vl/out/te21_emb.f32", ref.data(), ref.size() * 4)) {
            printf("[FAIL] te21_emb.f32\n");
            return 1;
        }
        const int drop = T - 15;
        std::vector<int> ids_i(ids.begin(), ids.end());
        std::vector<float> out((size_t)T * 4096);
        if (!qwenimage21::dev_text21_encode(qwen3vl21_file(), ids_i.data(), T, out.data())) {
            printf("[FAIL] dev_text21_encode\n");
            return 1;
        }
        const Metrics m = compare_rows(out.data() + (size_t)drop * 4096, ref.data(), 15, 4096);
        const bool pass = m.min_cos >= 0.9999 && m.max_rel <= 0.01;
        ok = ok && pass;
        printf("%s Qwen3-VL 8B text encoder (2.1, %d tokens, %d kept): min row cosine %.6f, max error %.3g relative\n",
               pass ? "[PASS]" : "[FAIL]", T, 15, m.min_cos, m.max_rel);
    }

    /* Qwen-Image 2.1's DiT (int8 ConvRot, the whole 32-block forward) on the FP32 oracle's inputs (ref_dit21_fp32.py): the packed
     * latent, the text embeddings, t = 0.5; the oracle dequantises the int8 weights exactly, so the reference is the exact layer math */
    if (want("dit21")) {
        std::vector<float> x(256 * 64), emb(12 * 4096), ref_joint(268 * 64), out(256 * 64);
        if (!read_file_path("/data/zimage-ref/qwenimage2/out/dit21_x.f32", x.data(), x.size() * 4) ||
            !read_file_path("/data/zimage-ref/qwenimage2/out/dit21_emb.f32", emb.data(), emb.size() * 4) ||
            !read_file_path("/data/zimage-ref/qwenimage2/out/dit21_out.f32", ref_joint.data(), ref_joint.size() * 4)) {
            printf("[FAIL] the 2.1 DiT oracle files (run ref_dit21_fp32.py)\n");
            return 1;
        }
        zimage::SafeTensors& q21 = dit21_int8();
        if (!q21.ok()) {
            printf("[FAIL] the 2.1 file: %s\n", q21.error().c_str());
            return 1;
        }
        if (!qwenimage21::dev_dit21_forward(q21, x.data(), 16, 16, emb.data(), 12, 0.5f, out.data())) {
            printf("[FAIL] dev_dit21_forward\n");
            return 1;
        }
        /* the oracle's output is the whole joint sequence: 12 text rows, then the 256 image rows */
        const std::vector<float> ref(ref_joint.begin() + 12 * 64, ref_joint.end());
        const Metrics m = compare_rows(out.data(), ref.data(), 256, 64);
        const bool pass = m.min_cos >= 0.9999 && m.max_rel <= 0.01;
        ok = ok && pass;
        printf("%s Qwen-Image 2.1 DiT (32 blocks, int8 ConvRot, t 0.5, text 12 + 256 image tokens): min row cosine %.6f, max error %.3g relative\n",
               pass ? "[PASS]" : "[FAIL]", m.min_cos, m.max_rel);
    }

    /* the oracle's template tokens and embeddings (the pipeline keeps the last 12 rows: the first 34 template tokens are dropped) */
    std::vector<int> ids;
    std::vector<float> enc(12 * 3584);
    if (want("enc") || want("fwd")) {
        std::vector<int32_t> idv(46);
        if (!read_file("te_ids.i32", idv.data(), idv.size() * 4) || !read_file("te_embeds.f32", enc.data(), enc.size() * 4)) {
            printf("[FAIL] te_ids.i32 or te_embeds.f32 (run ref_stages.py)\n");
            return 1;
        }
        ids.assign(idv.begin(), idv.end());
    }

    /* the device's text encoder on all 46 tokens (the oracle computes in BF16; this in F32) */
    if (want("enc")) {
        const int T = (int)ids.size();
        std::vector<float> out((size_t)T * 3584);
        if (!qwenimage::dev_text_encode(te_file(), ids.data(), T, out.data())) {
            printf("[FAIL] dev_text_encode\n");
            return 1;
        }
        const Metrics m = compare_rows(out.data() + (size_t)(T - 12) * 3584, enc.data(), 12, 3584);
        const bool pass = m.min_cos >= 0.999 && m.max_rel <= 0.05;
        ok = ok && pass;
        printf("%s text encoder (Qwen2.5-VL, 46 tokens, the last 12 rows): min row cosine %.6f, max error %.3g relative\n",
               pass ? "[PASS]" : "[FAIL]", m.min_cos, m.max_rel);
    }

    /* the whole DiT: the oracle's packed latent (256 x 64, the 16 x 16 grid), its text embeddings, t = 0.5 */
    if (want("fwd")) {
        std::vector<float> lat(256 * 64), ref(256 * 64);
        if (!read_file("dit_lat.f32", lat.data(), lat.size() * 4) || !read_file("dit_out.f32", ref.data(), ref.size() * 4)) {
            printf("[FAIL] dit_lat.f32 or dit_out.f32 (run ref_stages.py)\n");
            return 1;
        }
        std::vector<float> out(256 * 64);
        if (!qwenimage::dev_dit_forward(dit_file(), lat.data(), 16, 16, enc.data(), 12, 0.5f, out.data())) {
            printf("[FAIL] dev_dit_forward\n");
            return 1;
        }
        const Metrics m = compare_rows(out.data(), ref.data(), 256, 64);
        const bool pass = m.min_cos >= 0.99 && m.max_rel <= 0.05;
        ok = ok && pass;
        printf("%s DiT forward (60 blocks, t 0.5, 256 patches): min row cosine %.6f, max error %.3g relative\n",
               pass ? "[PASS]" : "[FAIL]", m.min_cos, m.max_rel);
        /* the same call against the FP32 reference (dit_out_fp32, when made: the pipeline's BF16 rounding is absent there) */
        std::vector<float> ref32(256 * 64);
        if (read_file("dit_out_fp32.f32", ref32.data(), ref32.size() * 4)) {
            const Metrics m32 = compare_rows(out.data(), ref32.data(), 256, 64);
            printf("  against the FP32 reference: min row cosine %.6f, max error %.3g relative\n", m32.min_cos, m32.max_rel);
        }
    }
    /* the decoder on the oracle's seeded latent [16][32][32] (vae_lat), against the oracle's decode (vae_dec, [3][256][256]) */
    if (want("vae")) {
        std::vector<float> lat(16 * 32 * 32), ref(3 * 256 * 256), out(3 * 256 * 256);
        if (!read_file("vae_lat.f32", lat.data(), lat.size() * 4) || !read_file("vae_dec.f32", ref.data(), ref.size() * 4)) {
            printf("[FAIL] vae_lat.f32 or vae_dec.f32 (run ref_stages.py)\n");
            return 1;
        }
        if (!vae_file().ok()) {
            printf("[FAIL] the VAE file\n");
            return 1;
        }
        if (!qwenimage::dev_vae_decode(vae_file(), lat.data(), out.data())) {
            printf("[FAIL] dev_vae_decode\n");
            return 1;
        }
        const Metrics m = compare_rows(out.data(), ref.data(), 3, 256 * 256);
        const bool pass = m.min_cos >= 0.999 && m.max_rel <= 0.05;
        ok = ok && pass;
        printf("%s VAE decode (16 x 32 x 32 -> 3 x 256 x 256): min channel cosine %.6f, max error %.3g relative\n",
               pass ? "[PASS]" : "[FAIL]", m.min_cos, m.max_rel);
    }
    /* the sampler: 8 steps from the oracle's packed noise and text embeddings (dit_lat, te_embeds), against the pipeline's final
     * packed latent (pipe_latent: the pipeline run of ref_pipeline.py, no guidance) */
    /* the sampler in FP32 (exact timesteps) against the FP32 sampler reference (sampler_fp32_latent, ref_sampler_fp32_stream.py) */
    if (want("pipe32")) {
        std::vector<float> noise(256 * 64), ref(256 * 64), out(256 * 64);
        if (!read_file("dit_lat.f32", noise.data(), noise.size() * 4) || !read_file("sampler_fp32_latent.f32", ref.data(), ref.size() * 4) ||
            !read_file("te_embeds.f32", enc.data(), enc.size() * 4)) {
            printf("[FAIL] the FP32 sampler inputs (run ref_sampler_fp32_stream.py)\n");
            return 1;
        }
        if (!qwenimage::dev_sample(dit_file(), noise.data(), 16, 16, enc.data(), 12, 8, out.data(), false)) {
            printf("[FAIL] dev_sample\n");
            return 1;
        }
        const Metrics m = compare_rows(out.data(), ref.data(), 256, 64);
        const bool pass = m.min_cos >= 0.9999 && m.max_rel <= 0.01;
        ok = ok && pass;
        printf("%s sampler (8 steps, exact timesteps) against the FP32 sampler: min row cosine %.6f, max error %.3g relative\n",
               pass ? "[PASS]" : "[FAIL]", m.min_cos, m.max_rel);
    }
    if (want("pipe")) {
        std::vector<float> noise(256 * 64), ref(256 * 64), out(256 * 64);
        if (!read_file("dit_lat.f32", noise.data(), noise.size() * 4) || !read_file("pipe_latent.f32", ref.data(), ref.size() * 4) ||
            !read_file("te_embeds.f32", enc.data(), enc.size() * 4)) {
            printf("[FAIL] dit_lat.f32, te_embeds.f32 or pipe_latent.f32 (run ref_pipeline.py)\n");
            return 1;
        }
        if (!qwenimage::dev_sample(dit_file(), noise.data(), 16, 16, enc.data(), 12, 8, out.data())) {
            printf("[FAIL] dev_sample\n");
            return 1;
        }
        const Metrics m = compare_rows(out.data(), ref.data(), 256, 64);
        printf("  (informational, against the BF16 pipeline: min row cosine %.6f, max error %.3g relative)\n", m.min_cos, m.max_rel);
    }
    /* each step on its own: the reference's latent before step i (pipe_step{i-1}, or the noise) through one engine step, against the
     * reference's latent after it (pipe_step{i}). The timestep and the schedule as the pipeline computes them. */
    if (want("steps")) {
        std::vector<float> enc2(12 * 3584), noise(256 * 64);
        if (!read_file("te_embeds.f32", enc2.data(), enc2.size() * 4) || !read_file("dit_lat.f32", noise.data(), noise.size() * 4)) {
            printf("[FAIL] te_embeds.f32 or dit_lat.f32\n");
            return 1;
        }
        std::vector<float> sig;
        qwenimage::sigmas_for(8, 256, sig);
        for (int i = 0; i < 8; ++i) {
            std::vector<float> prev(256 * 64), next(256 * 64), v(256 * 64);
            const std::string pf = i == 0 ? std::string() : "pipe_step" + std::to_string(i - 1) + ".f32";
            if (i == 0) prev = noise;
            else if (!read_file(pf.c_str(), prev.data(), prev.size() * 4)) {
                printf("[FAIL] %s\n", pf.c_str());
                return 1;
            }
            const std::string nf = "pipe_step" + std::to_string(i) + ".f32";
            if (!read_file(nf.c_str(), next.data(), next.size() * 4)) {
                printf("[FAIL] %s\n", nf.c_str());
                return 1;
            }
            /* the model's timestep: sigma * 1000 in BF16, divided by 1000 in BF16 */
            auto bf = [](float x) {
                uint32_t b;
                std::memcpy(&b, &x, 4);
                b = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
                float r;
                std::memcpy(&r, &b, 4);
                return r;
            };
            const float tm = bf(bf(sig[i] * 1000.0f) / 1000.0f);
            if (!qwenimage::dev_dit_forward(dit_file(), prev.data(), 16, 16, enc2.data(), 12, tm, v.data())) {
                printf("[FAIL] dev_dit_forward at step %d\n", i);
                return 1;
            }
            const float dt = sig[i + 1] - sig[i];
            for (size_t k = 0; k < v.size(); ++k) v[k] = prev[k] + dt * v[k];
            const Metrics m = compare_rows(v.data(), next.data(), 256, 64);
            printf("  step %d (sigma %.4f -> %.4f, model t %.4f): min row cosine %.6f, max error %.3g relative\n", i, sig[i],
                   sig[i + 1], tm, m.min_cos, m.max_rel);
        }
        /* the free run: the engine's own states from the noise, each against the reference's state after the same step */
        std::vector<float> x = noise, v(256 * 64);
        for (int i = 0; i < 8; ++i) {
            auto bf = [](float a) {
                uint32_t b;
                std::memcpy(&b, &a, 4);
                b = (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
                float r;
                std::memcpy(&r, &b, 4);
                return r;
            };
            const float tm = bf(bf(sig[i] * 1000.0f) / 1000.0f);
            if (!qwenimage::dev_dit_forward(dit_file(), x.data(), 16, 16, enc2.data(), 12, tm, v.data())) return 1;
            const float dt = sig[i + 1] - sig[i];
            for (size_t k = 0; k < x.size(); ++k) x[k] += dt * v[k];
            std::vector<float> next(256 * 64);
            read_file(("pipe_step" + std::to_string(i) + ".f32").c_str(), next.data(), next.size() * 4);
            const Metrics m = compare_rows(x.data(), next.data(), 256, 64);
            printf("  free run after step %d: min row cosine %.6f, max error %.3g relative\n", i, m.min_cos, m.max_rel);
        }
    }
    return ok ? 0 : 1;
}
