// MiniMax-H3-Image's diffusion transformer (minimax_dit.h; diffusers' MiniMaxH3ImageTransformer2DModel for text-to-image, no
// reference rows, no guidance). Per velocity: the prompt rows through condition_proj and the 2-block token refiner (plain
// pre-norm, no modulation, no rotary), the image rows through video_patch_proj, the packed sequence (text, then image) of
// 50 blocks (AdaLN modulation per row from the (timestep, modality) table; fused QKV with RMS-normed heads and a 3-axis
// rotary on 96 of the 128 channels; SwiGLU), the final layer's AdaLN, the step's PDD head, and the unpatchify.
#include "minimax_dit.h"

#include "gguf_mmap.h"
#include "kernel_loader.h"
#include "minimax_vae.h"  // (the MiniMax kernels' module: minimax_vae.co)
#include "sd_device.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string g_error;

// per-operation wall time (MMDIT_PROF set: printed at each velocity's end): the linears, the attention, the rest
struct Prof {
    double lin = 0, att = 0, rest = 0;
    bool on = std::getenv("MMDIT_PROF") != nullptr;
};
Prof g_prof;
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}


constexpr int kH = 5376;        // hidden size (the sequence's width)
constexpr int kInner = 7168;    // heads * head_dim (QKV's output width)
constexpr int kHeads = 56, kD = 128;
constexpr int kFF = 14336;      // the SwiGLU's inner width (fc1 projects to 2 * kFF)
constexpr int kTemb = 2688, kFreq = 256, kTembH = 5376;
constexpr int kText = 5120, kPatch = 96, kLayers = 50, kRefiner = 2, kSteps = 6;
constexpr int kRot = 96;        // rotary channels per head (3 axes x 16 frequencies, doubled)
constexpr float kEps = 1e-5f;
constexpr int kThreads = 256;

struct Kernels {
    hipModule_t mod = nullptr;
    hipFunction_t rmsnorm = nullptr, rope = nullptr, swiglu = nullptr, mod_rows = nullptr, gate_rows = nullptr,
                  mod_cols = nullptr, split_qkv = nullptr;
};
Kernels g_k;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax dit: ") + what + ": " + hipGetErrorString(e));
}

void load_kernels() {
    if (g_k.mod) return;
    if (kernel_module_load(&g_k.mod, "minimax_vae.co") != hipSuccess) throw std::runtime_error("minimax dit: minimax_vae.co does not load");
    hipFunction_t* fs[] = {&g_k.rmsnorm, &g_k.rope, &g_k.swiglu, &g_k.mod_rows, &g_k.gate_rows, &g_k.mod_cols, &g_k.split_qkv};
    const char* names[] = {"mv_rmsnorm", "mv_rope", "mm_swiglu_native", "mm_mod_rows", "mm_gate_rows", "mm_mod_cols", "mm_split_qkv"};
    for (int i = 0; i < 7; ++i) check(hipModuleGetFunction(fs[i], g_k.mod, names[i]), names[i]);
}

void launch(hipFunction_t f, long n, void** args) {
    check(hipModuleLaunchKernel(f, (unsigned)((n + kThreads - 1) / kThreads), 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr), "launch");
    check(hipStreamSynchronize(nullptr), "stream");
}

}  // namespace

struct mmdit {
    gguf_model_t* g = nullptr;
    const void* dev_base = nullptr;
    gguf_model_t* pg = nullptr;

    struct Lin {
        const uint16_t* w = nullptr;  // BF16 [out][in], in place (or null: F32 in wf)
        const float* wf = nullptr;    // F32 [out][in] in place, for the checkpoint's F32 linears
        sddev::Floats b;              // the bias (F32; the BF16 ones converted), or empty
        int in = 0, out = 0;
    };
    struct BlockW {
        sddev::Floats norm1, norm2, qn, kn;
        const uint16_t *qkv = nullptr, *out = nullptr, *fc1 = nullptr, *fc2 = nullptr;
        Lin adaln;  // (blocks only: 2688 -> 6 * 3 * 5376)
    };
    std::vector<BlockW> refiner, blocks;
    Lin cond, pvp, t1, t2, fadaln;
    sddev::Floats refnorm, fnorm;
    sddev::Floats pdd_w, pdd_b;   // [6][96][5376] and [6][96], F32
    std::vector<float> sigmas;    // steps + 1
};

namespace {

// a tensor's values as F32: an F32 tensor is copied, a BF16 one converted (the checkpoint's own values either way)
sddev::Floats floats_of(gguf_model_t* g, const std::string& name, size_t numel) {
    size_t bytes = 0;
    const void* p = gguf_get_tensor_ptr(g, name.c_str(), &bytes);
    if (!p) throw std::runtime_error("minimax dit: no tensor " + name);
    sddev::Floats out(numel);
    if (bytes == 4 * numel) {
        std::memcpy(out.data(), p, 4 * numel);
    } else if (bytes == 2 * numel) {
        const uint16_t* h = static_cast<const uint16_t*>(p);
        for (size_t i = 0; i < numel; ++i) {
            const uint32_t bits = (uint32_t)h[i] << 16;
            std::memcpy(&out[i], &bits, 4);
        }
    } else {
        throw std::runtime_error("minimax dit: " + name + " is not F32 or BF16 of its size");
    }
    return out;
}

const uint16_t* bf16_of(gguf_model_t* g, const std::string& name, size_t numel) {
    size_t bytes = 0;
    const void* p = gguf_get_tensor_ptr(g, name.c_str(), &bytes);
    if (!p) throw std::runtime_error("minimax dit: no tensor " + name);
    if (bytes != 2 * numel) throw std::runtime_error("minimax dit: " + name + " is not BF16 of its size");
    return static_cast<const uint16_t*>(p);
}

mmdit::Lin lin_of(gguf_model_t* g, const std::string& prefix, int in, int out, bool bias) {
    mmdit::Lin L;
    size_t bytes = 0;
    const void* wp = gguf_get_tensor_ptr(g, (prefix + ".weight").c_str(), &bytes);
    if (!wp) throw std::runtime_error("minimax dit: no tensor " + prefix + ".weight");
    if (bytes == 4 * (size_t)in * out) L.wf = static_cast<const float*>(wp);
    else L.w = bf16_of(g, prefix + ".weight", (size_t)in * out);
    if (bias) L.b = floats_of(g, prefix + ".bias", (size_t)out);
    L.in = in;
    L.out = out;
    return L;
}

// y = x W^T + b for the linear L (its weight in place, BF16 or F32), T token rows
void lin_apply(const mmdit::Lin& L, const float* x, int T, float* y) {
    const double t0 = g_prof.on ? now_s() : 0;
    if (L.wf) sddev::linear_f32(x, T, L.wf, L.in, L.out, L.b.data(), y);
    else sddev::linear_bf16(x, T, L.w, L.in, L.out, L.b.data(), y);
    if (g_prof.on) g_prof.lin += now_s() - t0;
}

// y = RMS norm of each of `rows` rows of n values (w: the affine weight, or null), eps
void rms(const float* x, long rows, int n, const float* w, float* y) {
    const float eps = kEps;
    void* args[] = {(void*)&x, (void*)&n, (void*)&w, (void*)&eps, (void*)&y};
    // one block per row (the kernel's loop is over the row's n values), not one thread per element
    check(hipModuleLaunchKernel(g_k.rmsnorm, (unsigned)rows, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr), "rmsnorm");
    check(hipStreamSynchronize(nullptr), "stream");
}

void rope(float* x, int S, const float* cs, const float* sn) {
    const int C = kInner, heads = kHeads, D = kD, rd = kRot;
    const long total = (long)S * heads * (rd / 2);
    void* args[] = {(void*)&x, (void*)&S, (void*)&C, (void*)&heads, (void*)&D, (void*)&rd, (void*)&cs, (void*)&sn};
    launch(g_k.rope, total, args);
}

void swiglu(const float* pj, int S, float* g) {
    const int F = kFF;
    void* args[] = {(void*)&pj, (void*)&S, (void*)&F, (void*)&g};
    launch(g_k.swiglu, (long)S * kFF, args);
}

// the head-grouped QKV rows [S][heads * 3 * D] into q, k, v [S][heads * D]
void split_qkv(const float* in, int S, float* q, float* k, float* v) {
    const int heads = kHeads, D = kD;
    void* args[] = {(void*)&in, (void*)&S, (void*)&heads, (void*)&D, (void*)&q, (void*)&k, (void*)&v};
    launch(g_k.split_qkv, (long)S * kInner, args);
}

void mod_rows(const float* x, const float* rows, const float* m, int S, int jsh, int js, float* y) {
    const int H = kH;
    void* args[] = {(void*)&x, (void*)&rows, (void*)&m, (void*)&S, (void*)&H, (void*)&jsh, (void*)&js, (void*)&y};
    launch(g_k.mod_rows, (long)S * kH, args);
}

void gate_rows(float* x, const float* o, const float* rows, const float* m, int S, int jg) {
    const int H = kH;
    void* args[] = {(void*)&x, (void*)&o, (void*)&rows, (void*)&m, (void*)&S, (void*)&H, (void*)&jg};
    launch(g_k.gate_rows, (long)S * kH, args);
}

void mod_cols(const float* x, const float* sc, const float* sh, int S, float* y) {
    const int H = kH;
    void* args[] = {(void*)&x, (void*)&sc, (void*)&sh, (void*)&S, (void*)&H, (void*)&y};
    launch(g_k.mod_cols, (long)S * kH, args);
}

// The attention block (or the refiner's plain block when mod is null): hid [S][H] updated in place. Rows: the modality row
// of each token (0 image, 1 text) for the AdaLN table; cs/sn: the rotary tables of the S rows (null: no rotary).
void dbg(const char* name, const float* p, size_t n);  // (defined below)
int g_block_layer = -1;  // the block being run (for the per-op dumps)

void block(const mmdit::BlockW& B, sddev::Floats& hid, int S, const sddev::Floats* mod, const sddev::Floats* rows,
           const float* cs, const float* sn) {
    const char* dl = std::getenv("MMDIT_DEBUG_LAYER");
    const bool dump_ops = dl && g_block_layer == std::atoi(dl);  // (debugging: this layer's per-op activations)
    char nm[32];
    const int C = kInner, H = kH;
    sddev::Floats n1((size_t)S * H), qkv((size_t)S * 3 * C), q((size_t)S * C), k((size_t)S * C), v((size_t)S * C), qn((size_t)S * C),
        kn((size_t)S * C), a((size_t)S * C), o((size_t)S * H), n2((size_t)S * H), pj((size_t)S * 2 * kFF),
        g((size_t)S * kFF), f((size_t)S * H);
    rms(hid.data(), S, H, B.norm1.data(), n1.data());
    if (mod) mod_rows(n1.data(), rows->data(), mod->data(), S, 0, 1, n1.data());  // shift_msa, scale_msa
    if (dump_ops) { std::snprintf(nm, sizeof nm, "L%d_n1", g_block_layer); dbg(nm, n1.data(), (size_t)S * H); }
    double tl = g_prof.on ? now_s() : 0;
    sddev::linear_bf16(n1.data(), S, B.qkv, H, 3 * C, nullptr, qkv.data());  // head-grouped rows: (q, k, v) per head
    split_qkv(qkv.data(), S, q.data(), k.data(), v.data());
    if (g_prof.on) g_prof.lin += now_s() - tl;
    rms(q.data(), (long)S * kHeads, kD, B.qn.data(), qn.data());
    rms(k.data(), (long)S * kHeads, kD, B.kn.data(), kn.data());
    if (cs) {
        rope(qn.data(), S, cs, sn);
        rope(kn.data(), S, cs, sn);
    }
    double t0 = g_prof.on ? now_s() : 0;
    sddev::attention(qn.data(), S, kn.data(), v.data(), S, C, kHeads, a.data());
    if (g_prof.on) g_prof.att += now_s() - t0;
    if (dump_ops) { std::snprintf(nm, sizeof nm, "L%d_a", g_block_layer); dbg(nm, a.data(), (size_t)S * C); }
    tl = g_prof.on ? now_s() : 0;
    sddev::linear_bf16(a.data(), S, B.out, C, H, nullptr, o.data());
    if (g_prof.on) g_prof.lin += now_s() - tl;
    if (dump_ops) { std::snprintf(nm, sizeof nm, "L%d_o", g_block_layer); dbg(nm, o.data(), (size_t)S * H); }
    if (mod) gate_rows(hid.data(), o.data(), rows->data(), mod->data(), S, 2);  // gate_msa
    else
        for (size_t i = 0; i < (size_t)S * H; ++i) hid[i] += o[i];
    rms(hid.data(), S, H, B.norm2.data(), n2.data());
    if (mod) mod_rows(n2.data(), rows->data(), mod->data(), S, 3, 4, n2.data());  // shift_mlp, scale_mlp
    if (dump_ops) { std::snprintf(nm, sizeof nm, "L%d_n2", g_block_layer); dbg(nm, n2.data(), (size_t)S * H); }
    tl = g_prof.on ? now_s() : 0;
    sddev::linear_bf16(n2.data(), S, B.fc1, H, 2 * kFF, nullptr, pj.data());
    swiglu(pj.data(), S, g.data());
    sddev::linear_bf16(g.data(), S, B.fc2, kFF, H, nullptr, f.data());
    if (g_prof.on) g_prof.lin += now_s() - tl;
    if (dump_ops) { std::snprintf(nm, sizeof nm, "L%d_f", g_block_layer); dbg(nm, f.data(), (size_t)S * H); }
    if (mod) gate_rows(hid.data(), f.data(), rows->data(), mod->data(), S, 5);  // gate_mlp
    else
        for (size_t i = 0; i < (size_t)S * H; ++i) hid[i] += f[i];
}

// (debugging: MMDIT_DEBUG=<dir> writes the activations to <dir>/dbg_<name>.f32, as mm_dit_ref.py's hooks do)
bool g_dbg_step0 = false;  // the dumps are of the first step only
void dbg(const char* name, const float* p, size_t n) {
    const char* dir = std::getenv("MMDIT_DEBUG");
    if (!dir || !g_dbg_step0) return;
    if (FILE* f = std::fopen((std::string(dir) + "/dbg_" + name + ".f32").c_str(), "wb")) {
        std::fwrite(p, sizeof(float), n, f);
        std::fclose(f);
    }
}

// The sinusoidal timestep embedding (diffusers' Timesteps: flip_sin_to_cos, downscale shift 0, scale 1): cos first
void sinusoid(float t, float* out) {
    const int half = kFreq / 2;
    for (int i = 0; i < half; ++i) {
        const double ang = (double)t * std::exp(-std::log(10000.0) * i / half);
        out[i] = (float)std::cos(ang);
        out[half + i] = (float)std::sin(ang);
    }
}

}  // namespace

mmdit_t* mmdit_open(const char* dit_gguf, const char* pdd_gguf) {
    try {
        std::unique_ptr<mmdit> m(new mmdit());
        m->g = gguf_mmap_open(dit_gguf);
        if (!m->g) throw std::runtime_error(std::string("cannot open ") + (dit_gguf ? dit_gguf : "(null)"));
        if (std::string(gguf_arch(m->g)) != "minimax_h3_dit") throw std::runtime_error("not a MiniMax-H3 DiT GGUF");
        if (gguf_arch_int(m->g, "hidden_size", -1) != kH || gguf_arch_int(m->g, "layers", -1) != kLayers ||
            gguf_arch_int(m->g, "heads", -1) != kHeads || gguf_arch_int(m->g, "head_dim", -1) != kD ||
            gguf_arch_int(m->g, "ffn_size", -1) != kFF)
            throw std::runtime_error("minimax dit: the GGUF's configuration is not the one this engine has");
        if (!sddev::enabled()) throw std::runtime_error("minimax dit: needs the device (SD_DEVICE unset or gpu)");
        size_t bytes = 0;
        const void* base = gguf_map_base(m->g, &bytes);
        if (gguf_n_maps(m->g) != 1 || !sddev::register_weights(base, bytes))
            throw std::runtime_error("minimax dit: the weights cannot be registered for the device");
        m->dev_base = base;

        m->cond = lin_of(m->g, "condition_proj", kText, kH, true);
        m->pvp = lin_of(m->g, "video_patch_proj", kPatch, kH, true);
        m->t1 = lin_of(m->g, "time_embedder.proj_in", kFreq, kTembH, true);
        m->t2 = lin_of(m->g, "time_embedder.proj_out", kTembH, kTemb, true);
        m->fadaln = lin_of(m->g, "final_layer.adaln_proj.linear", kTemb, 2 * kH, true);
        m->fnorm = floats_of(m->g, "final_layer.norm.weight", kH);
        m->refnorm = floats_of(m->g, "token_refiner.final_norm.weight", kH);
        for (int l = 0; l < kRefiner; ++l) {
            const std::string p = "token_refiner.blocks." + std::to_string(l) + ".";
            mmdit::BlockW B;
            B.norm1 = floats_of(m->g, p + "norm1.weight", kH);
            B.norm2 = floats_of(m->g, p + "norm2.weight", kH);
            B.qn = floats_of(m->g, p + "attn.q_norm.weight", kD);
            B.kn = floats_of(m->g, p + "attn.k_norm.weight", kD);
            B.qkv = bf16_of(m->g, p + "attn.qkv_proj.weight", (size_t)3 * kInner * kH);
            B.out = bf16_of(m->g, p + "attn.out_proj.weight", (size_t)kH * kInner);
            B.fc1 = bf16_of(m->g, p + "mlp.fc1.weight", (size_t)2 * kFF * kH);
            B.fc2 = bf16_of(m->g, p + "mlp.fc2.weight", (size_t)kH * kFF);
            m->refiner.push_back(std::move(B));
        }
        for (int l = 0; l < kLayers; ++l) {
            const std::string p = "blocks." + std::to_string(l) + ".";
            mmdit::BlockW B;
            B.norm1 = floats_of(m->g, p + "norm1.weight", kH);
            B.norm2 = floats_of(m->g, p + "norm2.weight", kH);
            B.qn = floats_of(m->g, p + "attn.q_norm.weight", kD);
            B.kn = floats_of(m->g, p + "attn.k_norm.weight", kD);
            B.qkv = bf16_of(m->g, p + "attn.qkv_proj.weight", (size_t)3 * kInner * kH);
            B.out = bf16_of(m->g, p + "attn.out_proj.weight", (size_t)kH * kInner);
            B.fc1 = bf16_of(m->g, p + "mlp.fc1.weight", (size_t)2 * kFF * kH);
            B.fc2 = bf16_of(m->g, p + "mlp.fc2.weight", (size_t)kH * kFF);
            B.adaln = lin_of(m->g, p + "adaln_proj.linear", kTemb, 3 * 6 * kH, true);
            m->blocks.push_back(std::move(B));
        }
        // the PDD heads and the sigmas
        m->pg = gguf_mmap_open(pdd_gguf);
        if (!m->pg) throw std::runtime_error(std::string("cannot open ") + (pdd_gguf ? pdd_gguf : "(null)"));
        if (std::string(gguf_arch(m->pg)) != "minimax_h3_pdd") throw std::runtime_error("not a MiniMax-H3 PDD heads GGUF");
        m->pdd_w = floats_of(m->pg, "video_out.weight", (size_t)kSteps * kPatch * kH);
        m->pdd_b = floats_of(m->pg, "video_out.bias", (size_t)kSteps * kPatch);
        {
            sddev::Floats s = floats_of(m->pg, "sigmas", (size_t)kSteps + 1);
            m->sigmas.assign(s.data(), s.data() + s.size());
        }
        load_kernels();
        return m.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void mmdit_close(mmdit_t* model) {
    if (!model) return;
    if (model->dev_base) sddev::unregister_weights(model->dev_base);
    if (model->g) gguf_mmap_close(model->g);
    if (model->pg) gguf_mmap_close(model->pg);
    delete model;
}

const char* mmdit_last_error(void) { return g_error.c_str(); }

int mmdit_steps(const mmdit_t* model) { return model ? kSteps : 0; }

float mmdit_sigma(const mmdit_t* model, int i) {
    if (!model || i < 0 || i > kSteps) return -1.0f;
    return model->sigmas[(size_t)i];
}

int mmdit_velocity(mmdit_t* m, const float* text, int T, const float* sample, int h, int w, int step, float* out) {
    if (!m || !text || !sample || !out || T <= 0 || h <= 0 || w <= 0 || h % 2 || w % 2 || step < 0 || step >= kSteps) {
        g_error = "mmdit_velocity: bad arguments";
        return -1;
    }
    try {
        std::lock_guard<std::recursive_mutex> lock(sddev::run_mutex());
        g_dbg_step0 = step == 0;
        const int hp = h / 2, wp = w / 2, N = hp * wp, S = T + N;
        // the image rows: the latent patchified (token row-major over the patch grid; (c, ph, pw) per token)
        sddev::Floats patches((size_t)N * kPatch);
        for (int i = 0; i < hp; ++i)
            for (int j = 0; j < wp; ++j)
                for (int c = 0; c < 24; ++c)
                    for (int ph = 0; ph < 2; ++ph)
                        for (int pw = 0; pw < 2; ++pw)
                            patches[((size_t)(i * wp + j)) * kPatch + c * 4 + ph * 2 + pw] =
                                sample[((size_t)c * h + 2 * i + ph) * w + 2 * j + pw];
        sddev::Floats img((size_t)N * kH);
        lin_apply(m->pvp, patches.data(), N, img.data());
        // the prompt rows: condition_proj, the token refiner, its final norm
        sddev::Floats prompt((size_t)T * kText);  // the caller's array (host memory) copied to the device-visible one
        std::memcpy(prompt.data(), text, sizeof(float) * (size_t)T * kText);
        sddev::Floats tx((size_t)T * kH);
        lin_apply(m->cond, prompt.data(), T, tx.data());
        for (auto& B : m->refiner) block(B, tx, T, nullptr, nullptr, nullptr, nullptr);
        sddev::Floats tn((size_t)T * kH);
        rms(tx.data(), T, kH, m->refnorm.data(), tn.data());
        dbg("tn", tn.data(), (size_t)T * kH);
        // the packed sequence: the text rows (tag 1), then the image rows (tag 0)
        sddev::Floats hid((size_t)S * kH), rows((size_t)S);
        std::memcpy(hid.data(), tn.data(), sizeof(float) * (size_t)T * kH);
        std::memcpy(hid.data() + (size_t)T * kH, img.data(), sizeof(float) * (size_t)N * kH);
        for (int t = 0; t < S; ++t) rows[t] = t < T ? 1.0f : 0.0f;
        // the rotary positions (t, h, w): text (index, 0, 0); the image rows on the frame's clock after the text, with the
        // spatial grid of the reference (_spatial_position_grid: aspect-normalized, cell centres, scaled by 32)
        std::vector<double> pt((size_t)S * 3, 0.0);
        for (int t = 0; t < T; ++t) pt[(size_t)t * 3] = (double)t;
        {
            const double sq = std::sqrt((double)h * w);
            auto grid = [&](int dim, int n, std::vector<double>& v) {
                const double ratio = dim / sq, left = (1.0 - ratio) / 2.0;
                v.resize(n);
                for (int i = 0; i < n; ++i) v[i] = 32.0 * (left + i * ratio / n);
            };
            std::vector<double> gh, gw;
            grid(h, hp, gh);
            grid(w, wp, gw);
            for (int i = 0; i < hp; ++i)
                for (int j = 0; j < wp; ++j) {
                    double* p = &pt[(size_t)(T + i * wp + j) * 3];
                    p[0] = (double)T;
                    p[1] = gh[i];
                    p[2] = gw[j];
                }
        }
        std::vector<float> cs((size_t)S * kRot), sn((size_t)S * kRot);
        for (int t = 0; t < S; ++t)
            for (int d = 0; d < kRot; ++d) {
                const int e = d % 48, axis = e / 16, k = e % 16;
                const double ang = pt[(size_t)t * 3 + axis] * std::pow(10000.0, -k / 16.0);
                cs[(size_t)t * kRot + d] = (float)std::cos(ang);
                sn[(size_t)t * kRot + d] = (float)std::sin(ang);
            }
        sddev::Floats dcs((size_t)S * kRot), dsn((size_t)S * kRot);
        std::memcpy(dcs.data(), cs.data(), sizeof(float) * cs.size());
        std::memcpy(dsn.data(), sn.data(), sizeof(float) * sn.size());
        // the timestep: t = 1 - sigma (unscaled), the sinusoid, the embedder, SiLU; the AdaLN input is SiLU(temb)
        float sin_t[kFreq];
        sinusoid(1.0f - m->sigmas[(size_t)step], sin_t);
        sddev::Floats te(kFreq), h1(kTembH), sil(kTemb);
        std::memcpy(te.data(), sin_t, sizeof(sin_t));
        lin_apply(m->t1, te.data(), 1, h1.data());
        for (int i = 0; i < kTembH; ++i) h1[i] = h1[i] / (1.0f + std::exp(-h1[i]));
        lin_apply(m->t2, h1.data(), 1, sil.data());
        for (int i = 0; i < kTemb; ++i) sil[i] = sil[i] / (1.0f + std::exp(-sil[i]));
        // the 50 blocks, each with its (timestep, modality) table
        dbg("b0_in", hid.data(), (size_t)S * kH);
        for (size_t l = 0; l < m->blocks.size(); ++l) {
            const mmdit::BlockW& B = m->blocks[l];
            g_block_layer = (int)l;
            sddev::Floats mod((size_t)3 * 6 * kH);
            sddev::linear_bf16(sil.data(), 1, B.adaln.w, kTemb, 3 * 6 * kH, B.adaln.b.data(), mod.data());
            block(B, hid, S, &mod, &rows, dcs.data(), dsn.data());
            if (l == 0) dbg("b0_out", hid.data(), (size_t)S * kH);
            if (std::getenv("MMDIT_DEBUG_ALL")) {  // (debugging: every block's output, named blk<NN>)
                char nm[16];
                std::snprintf(nm, sizeof nm, "blk%02zu", l);
                dbg(nm, hid.data(), (size_t)S * kH);
            }
            if (l + 1 == m->blocks.size()) dbg("b49_out", hid.data(), (size_t)S * kH);
        }
        // the final layer: norm, then the AdaLN (shift, then scale) of the one timestep
        sddev::Floats n((size_t)S * kH), fm((size_t)2 * kH), n2((size_t)S * kH);
        rms(hid.data(), S, kH, m->fnorm.data(), n.data());
        sddev::linear_bf16(sil.data(), 1, m->fadaln.w, kTemb, 2 * kH, m->fadaln.b.data(), fm.data());
        mod_cols(n.data(), fm.data() + kH, fm.data(), S, n2.data());
        dbg("norm_out", n2.data(), (size_t)S * kH);
        if (g_prof.on) {
            std::fprintf(stderr, "MMDIT_PROF linears %.2f s, attention %.2f s (of which the velocity's total is measured by the caller)\n",
                         g_prof.lin, g_prof.att);
            g_prof = Prof{};
            g_prof.on = true;
        }
        // the step's PDD head on the image rows, then the unpatchify (the inverse of the patchify)
        sddev::Floats velo((size_t)N * kPatch);
        sddev::linear_f32(n2.data() + (size_t)T * kH, N, m->pdd_w.data() + (size_t)step * kPatch * kH, kH, kPatch,
                          m->pdd_b.data() + (size_t)step * kPatch, velo.data());
        for (int i = 0; i < hp; ++i)
            for (int j = 0; j < wp; ++j)
                for (int c = 0; c < 24; ++c)
                    for (int ph = 0; ph < 2; ++ph)
                        for (int pw = 0; pw < 2; ++pw)
                            out[((size_t)c * h + 2 * i + ph) * w + 2 * j + pw] =
                                velo[((size_t)(i * wp + j)) * kPatch + c * 4 + ph * 2 + pw];
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int mmdit_sample(mmdit_t* m, const float* text, int T, const float* noise, int h, int w, float* out) {
    if (!m || !noise || !out) {
        g_error = "mmdit_sample: bad arguments";
        return -1;
    }
    try {
        const size_t n = (size_t)24 * h * w;
        std::vector<float> x(noise, noise + n), v(n);
        for (int s = 0; s < kSteps; ++s) {
            if (mmdit_velocity(m, text, T, x.data(), h, w, s, v.data()) != 0) return -1;
            // the velocity Euler step (MiniMaxH3Scheduler, velocity form): x += (sigma - sigma_next) * v
            const float dt = m->sigmas[(size_t)s] - m->sigmas[(size_t)s + 1];
            for (size_t i = 0; i < n; ++i) x[i] += dt * v[i];
        }
        std::memcpy(out, x.data(), sizeof(float) * n);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}
