/* The vision tower (Qwen3_5VisionModel) on the host, float32: patch embedding, the bilinearly interpolated position
 * embedding, 2-D rotary positions, the blocks (LayerNorm, full attention per image, GELU MLP), the spatial merger.
 * Written to transformers' order of operations (see vision.h). Layout: a GGUF tensor [ne0, ne1] is row-major
 * [ne1][ne0] (ne0 fastest), so Linear weights are [out][in]. */
#include "vision.h"
#include "gguf_mmap.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

std::string g_err;

void set_err(const std::string& s) { g_err = s; }

struct VBlock {
    const float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    const float *qkv_w, *qkv_b, *out_w, *out_b;
    const float *up_w, *up_b, *down_w, *down_b;
};

}  // namespace

struct vision_model {
    gguf_model_t* g = nullptr;
    int depth = 0, n_embd = 0, n_ff = 0, n_head = 0, patch = 0, tpatch = 0, merge = 0, n_out = 0, n_pos = 0, n_side = 0;
    int in_dim = 0;  // C * T * P * P: a pixel row
    float eps = 1e-6f;
    bool gelu_tanh = true;  // the blocks' MLP (config.hidden_act); the merger's GELU is exact
    const float *patch_w = nullptr, *patch_b = nullptr, *pos = nullptr;
    const float *post_w = nullptr, *post_b = nullptr, *fc1_w = nullptr, *fc1_b = nullptr, *fc2_w = nullptr, *fc2_b = nullptr;
    std::vector<VBlock> blocks;
    ~vision_model() { if (g) gguf_mmap_close(g); }
};

namespace {

// The F32 tensor's data, checked for its element count (a missing or mis-shaped tensor is an error)
const float* tensor(const vision_model* vm, const std::string& name, size_t count) {
    size_t bytes = 0;
    const void* p = gguf_get_tensor_ptr(vm->g, name.c_str(), &bytes);
    if (!p) throw std::runtime_error("mmproj: no tensor " + name);
    if (bytes != count * sizeof(float))
        throw std::runtime_error("mmproj: " + name + " has " + std::to_string(bytes) + " bytes, expected " +
                                 std::to_string(count * sizeof(float)));
    return static_cast<const float*>(p);
}

// y[n][o] = b[o] + sum_i W[o][i] * x[n][i], over n rows
void linear(const float* W, const float* b, const float* x, float* y, int n, int in_dim, int out_dim) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < n; ++r) {
        const float* xr = x + (size_t)r * in_dim;
        float* yr = y + (size_t)r * out_dim;
        for (int o = 0; o < out_dim; ++o) {
            const float* wr = W + (size_t)o * in_dim;
            float acc = 0.f;
            for (int i = 0; i < in_dim; ++i) acc += wr[i] * xr[i];
            yr[o] = acc + (b ? b[o] : 0.f);
        }
    }
}

// LayerNorm over the last dimension of n rows (biased variance, eps as the config's)
void layernorm(const float* x, float* y, int n, int d, const float* w, const float* b, float eps) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < n; ++r) {
        const float* xr = x + (size_t)r * d;
        float* yr = y + (size_t)r * d;
        float mean = 0.f;
        for (int i = 0; i < d; ++i) mean += xr[i];
        mean /= (float)d;
        float var = 0.f;
        for (int i = 0; i < d; ++i) {
            const float c = xr[i] - mean;
            var += c * c;
        }
        var /= (float)d;
        const float inv = 1.f / std::sqrt(var + eps);
        for (int i = 0; i < d; ++i) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
    }
}

inline float gelu_tanh(float x) {
    return 0.5f * x * (1.f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

inline float gelu_exact(float x) { return 0.5f * x * (1.f + std::erf(x * 0.7071067811865476f)); }

// torch.linspace(0, end, steps)[i], as torch computes it (from the start for the first half, the end for the rest)
float linspace_at(int i, int steps, float end) {
    const float step = end / (float)(steps - 1);
    return i < steps / 2 ? (float)i * step : end - (float)(steps - 1 - i) * step;
}

}  // namespace

extern "C" {

const char* vision_last_error(void) { return g_err.c_str(); }

vision_model_t* vision_open(const char* path) {
    std::unique_ptr<vision_model> vm(new vision_model());
    try {
        vm->g = gguf_mmap_open(path);
        if (!vm->g) { set_err(std::string("cannot open ") + path); return nullptr; }
        const char* arch = gguf_arch(vm->g);
        if (!arch || std::strcmp(arch, "qwen3_5_vision") != 0) {
            set_err(std::string(path) + ": not an mmproj of qwen3_5_vision (architecture " + (arch ? arch : "?") + ")");
            return nullptr;  // (vm closes its mapping)
        }
        vm->depth = (int)gguf_arch_int(vm->g, "block_count", 0);
        vm->n_embd = (int)gguf_arch_int(vm->g, "embedding_length", 0);
        vm->n_ff = (int)gguf_arch_int(vm->g, "feed_forward_length", 0);
        vm->n_head = (int)gguf_arch_int(vm->g, "attention.head_count", 0);
        vm->patch = (int)gguf_arch_int(vm->g, "patch_size", 0);
        vm->tpatch = (int)gguf_arch_int(vm->g, "temporal_patch_size", 0);
        vm->merge = (int)gguf_arch_int(vm->g, "spatial_merge_size", 0);
        vm->n_out = (int)gguf_arch_int(vm->g, "projection_dim", 0);
        vm->n_pos = (int)gguf_arch_int(vm->g, "num_position_embeddings", 0);
        vm->eps = gguf_arch_float(vm->g, "layer_norm_epsilon", 1e-6f);
        if (vm->depth <= 0 || vm->n_embd <= 0 || vm->n_ff <= 0 || vm->n_head <= 0 || vm->patch <= 0 || vm->tpatch <= 0 ||
            vm->merge <= 0 || vm->n_out <= 0 || vm->n_pos <= 0 || vm->n_embd % vm->n_head != 0)
            throw std::runtime_error("mmproj: missing or bad qwen3_5_vision hyperparameters");
        vm->n_side = (int)std::lround(std::sqrt((double)vm->n_pos));
        if (vm->n_side * vm->n_side != vm->n_pos) throw std::runtime_error("mmproj: num_position_embeddings is not a square");
        vm->in_dim = 3 * vm->tpatch * vm->patch * vm->patch;
        const int D = vm->n_embd, F = vm->n_ff, M = D * vm->merge * vm->merge, O = vm->n_out;
        vm->patch_w = tensor(vm.get(), "v.patch_embd.weight", (size_t)D * vm->in_dim);
        vm->patch_b = tensor(vm.get(), "v.patch_embd.bias", D);
        vm->pos = tensor(vm.get(), "v.position_embd.weight", (size_t)vm->n_pos * D);
        vm->blocks.resize(vm->depth);
        for (int l = 0; l < vm->depth; ++l) {
            const std::string b = "v.blk." + std::to_string(l) + ".";
            VBlock& k = vm->blocks[l];
            k.ln1_w = tensor(vm.get(), b + "ln1.weight", D);
            k.ln1_b = tensor(vm.get(), b + "ln1.bias", D);
            k.ln2_w = tensor(vm.get(), b + "ln2.weight", D);
            k.ln2_b = tensor(vm.get(), b + "ln2.bias", D);
            k.qkv_w = tensor(vm.get(), b + "attn_qkv.weight", (size_t)3 * D * D);
            k.qkv_b = tensor(vm.get(), b + "attn_qkv.bias", 3 * D);
            k.out_w = tensor(vm.get(), b + "attn_out.weight", (size_t)D * D);
            k.out_b = tensor(vm.get(), b + "attn_out.bias", D);
            k.up_w = tensor(vm.get(), b + "ffn_up.weight", (size_t)F * D);
            k.up_b = tensor(vm.get(), b + "ffn_up.bias", F);
            k.down_w = tensor(vm.get(), b + "ffn_down.weight", (size_t)D * F);
            k.down_b = tensor(vm.get(), b + "ffn_down.bias", D);
        }
        vm->post_w = tensor(vm.get(), "v.post_ln.weight", D);
        vm->post_b = tensor(vm.get(), "v.post_ln.bias", D);
        vm->fc1_w = tensor(vm.get(), "v.merger.fc1.weight", (size_t)M * M);
        vm->fc1_b = tensor(vm.get(), "v.merger.fc1.bias", M);
        vm->fc2_w = tensor(vm.get(), "v.merger.fc2.weight", (size_t)O * M);
        vm->fc2_b = tensor(vm.get(), "v.merger.fc2.bias", O);
        return vm.release();
    } catch (const std::exception& e) {
        set_err(e.what());
        return nullptr;
    }
}

void vision_close(vision_model_t* vm) { delete vm; }

int vision_n_out(const vision_model_t* vm) { return vm ? vm->n_out : 0; }
int vision_merge(const vision_model_t* vm) { return vm ? vm->merge : 0; }
int vision_patch(const vision_model_t* vm) { return vm ? vm->patch : 0; }

int vision_encode(vision_model_t* vm, const float* pixels, int h, int w, float* out, vision_stage_fn stage, void* ud) {
    if (!vm || !pixels || !out) { set_err("vision_encode: NULL argument"); return -1; }
    const int m = vm->merge, D = vm->n_embd, H = vm->n_head, hd = D / H;
    if (h <= 0 || w <= 0 || h % m || w % m) { set_err("vision_encode: the grid must be a positive multiple of the merge"); return -1; }
    try {
        const int N = h * w;
        // rows in the order the merger takes them: merged blocks (hb, wb), then the m x m patches inside each (ir, ic)
        std::vector<int> row_of(N), col_of(N);
        for (int hb = 0; hb < h / m; ++hb)
            for (int wb = 0; wb < w / m; ++wb)
                for (int ir = 0; ir < m; ++ir)
                    for (int ic = 0; ic < m; ++ic) {
                        const int t = ((hb * (w / m) + wb) * m + ir) * m + ic;
                        row_of[t] = hb * m + ir;
                        col_of[t] = wb * m + ic;
                    }

        // patch embedding: a Linear over the pixel rows. The image processor writes them in merge order already
        // (Qwen2VLImageProcessor's patch reshape), the order of row_of / col_of above.
        std::vector<float> x((size_t)N * D), tmp((size_t)N * D);
        linear(vm->patch_w, vm->patch_b, pixels, x.data(), N, vm->in_dim, D);
        if (stage) stage("patch_embed", x.data(), N, D, ud);

        // position embedding: the learned n_side x n_side table bilinearly interpolated to h x w, as
        // fast_pos_embed_interpolate: four corners weighted, summed in that order
        const int S = vm->n_side;
        for (int t = 0; t < N; ++t) {
            const int hh = row_of[t], ww = col_of[t];
            const float hi = linspace_at(hh, h, (float)(S - 1)), wi = linspace_at(ww, w, (float)(S - 1));
            const int hf = (int)hi, wf = (int)wi;
            const int hc = std::min(hf + 1, S - 1), wc = std::min(wf + 1, S - 1);
            const float dh = hi - (float)hf, dw = wi - (float)wf;
            const float wts[4] = {(1.f - dh) * (1.f - dw), (1.f - dh) * dw, dh * (1.f - dw), dh * dw};
            const int idx[4] = {hf * S + wf, hf * S + wc, hc * S + wf, hc * S + wc};
            float* xr = &x[(size_t)t * D];
            for (int c = 0; c < D; ++c) {
                const float p0 = vm->pos[(size_t)idx[0] * D + c] * wts[0];
                const float p1 = vm->pos[(size_t)idx[1] * D + c] * wts[1];
                const float p2 = vm->pos[(size_t)idx[2] * D + c] * wts[2];
                const float p3 = vm->pos[(size_t)idx[3] * D + c] * wts[3];
                xr[c] = xr[c] + (((p0 + p1) + p2) + p3);
            }
        }

        // 2-D rotary: the frequencies of the row and the column (hd / 4 each), concatenated and doubled as
        // rotate_half needs: cos / sin per token and head dimension
        const int half = hd / 2, nf = hd / 4;  // half: the rotate_half split; nf: frequencies per axis
        std::vector<float> cosv((size_t)N * hd), sinv((size_t)N * hd);
        std::vector<float> inv(nf);
        for (int j = 0; j < nf; ++j) inv[j] = 1.f / std::pow(10000.f, (float)(2 * j) / (float)half);
        for (int t = 0; t < N; ++t) {
            const float fr = (float)row_of[t], fc = (float)col_of[t];
            for (int j = 0; j < nf; ++j) {
                const float a = fr * inv[j], b = fc * inv[j];
                // emb = [row (nf), col (nf), row (nf), col (nf)]
                cosv[(size_t)t * hd + j] = std::cos(a);
                sinv[(size_t)t * hd + j] = std::sin(a);
                cosv[(size_t)t * hd + nf + j] = std::cos(b);
                sinv[(size_t)t * hd + nf + j] = std::sin(b);
                cosv[(size_t)t * hd + 2 * nf + j] = cosv[(size_t)t * hd + j];
                sinv[(size_t)t * hd + 2 * nf + j] = sinv[(size_t)t * hd + j];
                cosv[(size_t)t * hd + 3 * nf + j] = cosv[(size_t)t * hd + nf + j];
                sinv[(size_t)t * hd + 3 * nf + j] = sinv[(size_t)t * hd + nf + j];
            }
        }

        std::vector<float> h1((size_t)N * D), qkv((size_t)N * 3 * D), att((size_t)N * D), mlp((size_t)N * vm->n_ff);
        const float scale = 1.f / std::sqrt((float)hd);
        for (int l = 0; l < vm->depth; ++l) {
            const VBlock& k = vm->blocks[l];
            layernorm(x.data(), h1.data(), N, D, k.ln1_w, k.ln1_b, vm->eps);
            linear(k.qkv_w, k.qkv_b, h1.data(), qkv.data(), N, D, 3 * D);
            // rotary on q and k (the qkv rows are [q (H x hd), k, v]), in place
            for (int t = 0; t < N; ++t) {
                float* row = &qkv[(size_t)t * 3 * D];
                for (int which = 0; which < 2; ++which)
                    for (int hh = 0; hh < H; ++hh) {
                        float* v = row + which * D + hh * hd;
                        float r[256];
                        for (int d = 0; d < hd; ++d) r[d] = d < half ? -v[d + half] : v[d - half];
                        for (int d = 0; d < hd; ++d) {
                            const float c = cosv[(size_t)t * hd + d], s = sinv[(size_t)t * hd + d];
                            v[d] = v[d] * c + r[d] * s;
                        }
                    }
            }
            // full attention within the image, per head
#pragma omp parallel for schedule(static) collapse(2)
            for (int hh = 0; hh < H; ++hh)
                for (int i = 0; i < N; ++i) {
                    std::vector<float> sc(N);
                    const float* q = &qkv[(size_t)i * 3 * D + hh * hd];
                    float mx = -INFINITY;
                    for (int j = 0; j < N; ++j) {
                        const float* kk = &qkv[(size_t)j * 3 * D + D + hh * hd];
                        float s = 0.f;
                        for (int d = 0; d < hd; ++d) s += q[d] * kk[d];
                        sc[j] = s * scale;
                        mx = std::max(mx, sc[j]);
                    }
                    float sum = 0.f;
                    for (int j = 0; j < N; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                    float* o = &att[(size_t)i * D + hh * hd];
                    for (int d = 0; d < hd; ++d) o[d] = 0.f;
                    for (int j = 0; j < N; ++j) {
                        const float p = sc[j] / sum;
                        const float* vv = &qkv[(size_t)j * 3 * D + 2 * D + hh * hd];
                        for (int d = 0; d < hd; ++d) o[d] += p * vv[d];
                    }
                }
            linear(k.out_w, k.out_b, att.data(), tmp.data(), N, D, D);
            for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];

            layernorm(x.data(), h1.data(), N, D, k.ln2_w, k.ln2_b, vm->eps);
            linear(k.up_w, k.up_b, h1.data(), mlp.data(), N, D, vm->n_ff);
            for (float& v : mlp) v = vm->gelu_tanh ? gelu_tanh(v) : gelu_exact(v);
            linear(k.down_w, k.down_b, mlp.data(), tmp.data(), N, vm->n_ff, D);
            for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
            if (l == 0 && stage) stage("block0", x.data(), N, D, ud);
        }

        // the merger: LayerNorm per patch, then the m x m patches of a block concatenated and projected
        layernorm(x.data(), h1.data(), N, D, vm->post_w, vm->post_b, vm->eps);
        const int M = D * m * m, G = N / (m * m);
        std::vector<float> cat((size_t)G * M), f1((size_t)G * M);
        std::memcpy(cat.data(), h1.data(), sizeof(float) * (size_t)N * D);  // rows of a block are consecutive: same bytes
        linear(vm->fc1_w, vm->fc1_b, cat.data(), f1.data(), G, M, M);
        for (float& v : f1) v = gelu_exact(v);
        linear(vm->fc2_w, vm->fc2_b, f1.data(), out, G, M, vm->n_out);
        if (stage) stage("merged", out, G, vm->n_out, ud);
        return 0;
    } catch (const std::exception& e) {
        set_err(e.what());
        return -1;
    }
}

}  // extern "C"
