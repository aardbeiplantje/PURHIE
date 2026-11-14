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
#include "minimax_weights.h"
#include "quant_types.h"
#include "safetensors_mmap.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
    WeightFile w, pw;  // the DiT's and the PDD heads' weights
    std::vector<const void*> dev_bases;  // the mappings registered for the device

    struct Lin {
        const uint16_t* w = nullptr;  // BF16 [out][in], in place (or null: F32 in wf)
        const float* wf = nullptr;    // F32 [out][in] in place, for the checkpoint's F32 linears
        sddev::Floats b;              // the bias (F32; the BF16 ones converted), or empty
        int in = 0, out = 0;
        int qt = -1;                  // a quantized weight: its ggml type id, q its first row, row_bytes per row (else -1)
        const void* q = nullptr;
        size_t row_bytes = 0;
    };
    struct BlockW {
        sddev::Floats norm1, norm2, qn, kn;
        Lin qkv, out, fc1, fc2;  // (no bias)
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
sddev::Floats floats_of(const WeightFile& w, const std::string& name, size_t numel) {
    size_t bytes = 0;
    const void* p = w.tensor(name, &bytes);
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

// a value's BF16 bits (round to nearest, ties to even; inf and NaN truncated)
uint16_t bf16_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// the rows of a quantized [N][K] matrix decoded to BF16 (the engine's CPU decoders, on up to 8 threads)
void dequant_bf16(int qt, const uint8_t* src, size_t row_bytes, int N, int K, uint16_t* dst) {
    const int nt = std::max(1, std::min(8, (int)std::thread::hardware_concurrency()));
    std::vector<char> ok((size_t)nt, 1);
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            std::vector<float> row((size_t)K);
            for (int r = t; r < N; r += nt) {
                if (qt_to_float(qt, src + (size_t)r * row_bytes, row.data(), K) != 0) {
                    ok[(size_t)t] = 0;
                    return;
                }
                for (int i = 0; i < K; ++i) dst[(size_t)r * K + i] = bf16_bits(row[(size_t)i]);
            }
        });
    for (auto& x : th) x.join();
    for (char c : ok)
        if (!c) throw std::runtime_error("minimax dit: a quantized weight does not decode");
}

mmdit::Lin lin_of(const WeightFile& w, const std::string& prefix, int in, int out, bool bias) {
    mmdit::Lin L;
    size_t bytes = 0;
    const void* wp = w.tensor(prefix + ".weight", &bytes);
    if (!wp) throw std::runtime_error("minimax dit: no tensor " + prefix + ".weight");
    if (bytes == 4 * (size_t)in * out) {
        L.wf = static_cast<const float*>(wp);
    } else if (bytes == 2 * (size_t)in * out) {
        L.w = static_cast<const uint16_t*>(wp);
    } else {  // a quantized GGUF tensor (rocm-convert's --type), decoded to BF16 for each call (lin_apply)
        const int qt = w.gguf_type(prefix + ".weight");
        if (qt < 0 || !qt_get_traits(qt) || bytes != (size_t)out * qt_row_size(qt, in))
            throw std::runtime_error("minimax dit: " + prefix + ".weight is not BF16, F32 or a quantized type of its size");
        L.qt = qt;
        L.q = wp;
        L.row_bytes = qt_row_size(qt, in);
    }
    if (bias) L.b = floats_of(w, prefix + ".bias", (size_t)out);
    L.in = in;
    L.out = out;
    return L;
}

// the token rows per GPU launch (MMDIT_UBATCH, default 256): one launch over a 2k-row matmul holds the device for long
// stretches, and the screen stalls with it; the rows are independent, so the batches give the same numbers
int ubatch() {
    static const int u = [] {
        const char* e = std::getenv("MMDIT_UBATCH");
        const int v = e ? std::atoi(e) : 256;
        return v > 0 ? v : 256;
    }();
    return u;
}

// y = x W^T + b for the linear L, T token rows: its weight in place (BF16 or F32), or a quantized one decoded to BF16 once
// for the call, then the BF16 kernels with the activations rounded on load as always (so a run differs from the BF16 one
// by the weights' decoded values alone). The rows go in ubatch-row launches; the rows are independent
void lin_apply(const mmdit::Lin& L0, const float* x, int T, float* y) {
    const double t0 = g_prof.on ? now_s() : 0;
    static sddev::Floats* scr = new sddev::Floats();  // (the decoded weights; leaked, so no static teardown of device memory)
    static size_t scr_n = 0;                          // its capacity, in BF16 values
    mmdit::Lin Ld;
    const mmdit::Lin* Lp = &L0;
    const bool control = L0.w && !L0.wf && std::getenv("MMDIT_SCRATCH_BF16") != nullptr;  // (debugging: BF16 through the scratch)
    if (L0.qt >= 0 || control) {
        const size_t nw = (size_t)L0.out * L0.in;
        if (scr_n < nw) {
            *scr = sddev::Floats((nw + 1) / 2);
            scr_n = nw;
        }
        uint16_t* wb = reinterpret_cast<uint16_t*>(scr->data());
        if (L0.qt >= 0) dequant_bf16(L0.qt, static_cast<const uint8_t*>(L0.q), L0.row_bytes, L0.out, L0.in, wb);
        else std::memcpy(wb, L0.w, nw * sizeof(uint16_t));
        Ld = L0;
        Ld.qt = -1;
        Ld.w = wb;
        Lp = &Ld;
    }
    const mmdit::Lin& L = *Lp;
    for (int r = 0; r < T; r += ubatch()) {  // the rows in ubatch-row launches
        const int n = std::min(ubatch(), T - r);
        const float* xr = x + (size_t)r * L.in;
        float* yr = y + (size_t)r * L.out;
        if (L.wf) sddev::linear_f32(xr, n, L.wf, L.in, L.out, L.b.data(), yr);
        else sddev::linear_bf16(xr, n, L.w, L.in, L.out, L.b.data(), yr);
    }
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
    lin_apply(B.qkv, n1.data(), S, qkv.data());  // head-grouped rows: (q, k, v) per head
    split_qkv(qkv.data(), S, q.data(), k.data(), v.data());
    if (g_prof.on) g_prof.lin += now_s() - tl;
    rms(q.data(), (long)S * kHeads, kD, B.qn.data(), qn.data());
    rms(k.data(), (long)S * kHeads, kD, B.kn.data(), kn.data());
    if (cs) {
        rope(qn.data(), S, cs, sn);
        rope(kn.data(), S, cs, sn);
    }
    double t0 = g_prof.on ? now_s() : 0;
    for (int r = 0; r < S; r += ubatch()) {  // the queries in ubatch-row launches (each row's softmax is its own)
        const int n = std::min(ubatch(), S - r);
        sddev::attention(qn.data() + (size_t)r * C, n, kn.data(), v.data(), S, C, kHeads, a.data() + (size_t)r * C);
    }
    if (g_prof.on) g_prof.att += now_s() - t0;
    if (dump_ops) { std::snprintf(nm, sizeof nm, "L%d_a", g_block_layer); dbg(nm, a.data(), (size_t)S * C); }
    tl = g_prof.on ? now_s() : 0;
    lin_apply(B.out, a.data(), S, o.data());
    if (g_prof.on) g_prof.lin += now_s() - tl;
    if (dump_ops) { std::snprintf(nm, sizeof nm, "L%d_o", g_block_layer); dbg(nm, o.data(), (size_t)S * H); }
    if (mod) gate_rows(hid.data(), o.data(), rows->data(), mod->data(), S, 2);  // gate_msa
    else
        for (size_t i = 0; i < (size_t)S * H; ++i) hid[i] += o[i];
    rms(hid.data(), S, H, B.norm2.data(), n2.data());
    if (mod) mod_rows(n2.data(), rows->data(), mod->data(), S, 3, 4, n2.data());  // shift_mlp, scale_mlp
    if (dump_ops) { std::snprintf(nm, sizeof nm, "L%d_n2", g_block_layer); dbg(nm, n2.data(), (size_t)S * H); }
    tl = g_prof.on ? now_s() : 0;
    lin_apply(B.fc1, n2.data(), S, pj.data());
    swiglu(pj.data(), S, g.data());
    lin_apply(B.fc2, g.data(), S, f.data());
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

// A path of a GGUF (rocm-convert's output), else the original checkpoint: a directory (transformer/) or a safetensors file
static bool is_gguf_path(const char* path) {
    const std::string s = path ? path : "";
    return s.size() >= 5 && s.compare(s.size() - 5, 5, ".gguf") == 0;
}

// The PDD sigmas of a checkpoint's pdd_config.json (the directory of the heads' file): "sigmas": [ ... ]
static std::vector<float> sigmas_of_config(const std::string& dir) {
    std::vector<float> out;
    char* text = st_read_text(dir.c_str(), "pdd_config.json");
    if (!text) return out;
    const char* k = std::strstr(text, "\"sigmas\"");
    const char* o = k ? std::strchr(k, '[') : nullptr;
    const char* c = o ? std::strchr(o, ']') : nullptr;
    if (o && c) {
        for (const char* p = o + 1; p < c;) {
            char* e = nullptr;
            const double v = std::strtod(p, &e);
            if (e == p) { ++p; continue; }
            out.push_back((float)v);
            p = e;
        }
    }
    std::free(text);
    return out;
}

mmdit_t* mmdit_open(const char* dit_path, const char* pdd_path) {
    try {
        std::unique_ptr<mmdit> m(new mmdit());
        if (is_gguf_path(dit_path)) {
            m->w.g = gguf_mmap_open(dit_path);
            if (!m->w.g) throw std::runtime_error(std::string("cannot open ") + dit_path);
            if (std::string(gguf_arch(m->w.g)) != "minimax_h3_dit") throw std::runtime_error("not a MiniMax-H3 DiT GGUF");
            if (gguf_arch_int(m->w.g, "hidden_size", -1) != kH || gguf_arch_int(m->w.g, "layers", -1) != kLayers ||
                gguf_arch_int(m->w.g, "heads", -1) != kHeads || gguf_arch_int(m->w.g, "head_dim", -1) != kD ||
                gguf_arch_int(m->w.g, "ffn_size", -1) != kFF)
                throw std::runtime_error("minimax dit: the GGUF's configuration is not the one this engine has");
        } else {
            // the original checkpoint: its tensors are the names and the layouts the engine reads (the converter copies them)
            m->w.st = st_open(dit_path);
            if (!m->w.st) throw std::runtime_error(std::string("minimax dit: ") + st_last_error());
        }
        if (!sddev::enabled()) throw std::runtime_error("minimax dit: needs the device (SD_DEVICE unset or gpu)");
        if (m->w.g) {
            size_t bytes = 0;
            const void* base = gguf_map_base(m->w.g, &bytes);
            if (gguf_n_maps(m->w.g) != 1 || !sddev::register_weights(base, bytes))
                throw std::runtime_error("minimax dit: the weights cannot be registered for the device");
            m->dev_bases.push_back(base);
        } else {
            for (int i = 0; i < st_n_shards(m->w.st); ++i) {
                size_t bytes = 0;
                const void* base = st_shard_base(m->w.st, i, &bytes);
                if (!sddev::register_weights(base, bytes))
                    throw std::runtime_error("minimax dit: the weights cannot be registered for the device");
                m->dev_bases.push_back(base);
            }
        }
        const WeightFile& w = m->w;
        m->cond = lin_of(w, "condition_proj", kText, kH, true);
        m->pvp = lin_of(w, "video_patch_proj", kPatch, kH, true);
        m->t1 = lin_of(w, "time_embedder.proj_in", kFreq, kTembH, true);
        m->t2 = lin_of(w, "time_embedder.proj_out", kTembH, kTemb, true);
        m->fadaln = lin_of(w, "final_layer.adaln_proj.linear", kTemb, 2 * kH, true);
        m->fnorm = floats_of(w, "final_layer.norm.weight", kH);
        m->refnorm = floats_of(w, "token_refiner.final_norm.weight", kH);
        for (int l = 0; l < kRefiner; ++l) {
            const std::string p = "token_refiner.blocks." + std::to_string(l) + ".";
            mmdit::BlockW B;
            B.norm1 = floats_of(w, p + "norm1.weight", kH);
            B.norm2 = floats_of(w, p + "norm2.weight", kH);
            B.qn = floats_of(w, p + "attn.q_norm.weight", kD);
            B.kn = floats_of(w, p + "attn.k_norm.weight", kD);
            B.qkv = lin_of(w, p + "attn.qkv_proj", kH, 3 * kInner, false);
            B.out = lin_of(w, p + "attn.out_proj", kInner, kH, false);
            B.fc1 = lin_of(w, p + "mlp.fc1", kH, 2 * kFF, false);
            B.fc2 = lin_of(w, p + "mlp.fc2", kFF, kH, false);
            m->refiner.push_back(std::move(B));
        }
        for (int l = 0; l < kLayers; ++l) {
            const std::string p = "blocks." + std::to_string(l) + ".";
            mmdit::BlockW B;
            B.norm1 = floats_of(w, p + "norm1.weight", kH);
            B.norm2 = floats_of(w, p + "norm2.weight", kH);
            B.qn = floats_of(w, p + "attn.q_norm.weight", kD);
            B.kn = floats_of(w, p + "attn.k_norm.weight", kD);
            B.qkv = lin_of(w, p + "attn.qkv_proj", kH, 3 * kInner, false);
            B.out = lin_of(w, p + "attn.out_proj", kInner, kH, false);
            B.fc1 = lin_of(w, p + "mlp.fc1", kH, 2 * kFF, false);
            B.fc2 = lin_of(w, p + "mlp.fc2", kFF, kH, false);
            B.adaln = lin_of(w, p + "adaln_proj.linear", kTemb, 3 * 6 * kH, true);
            m->blocks.push_back(std::move(B));
        }
        // the PDD heads and the sigmas: the GGUF's tensors, or the checkpoint's heads file and pdd_config.json
        if (is_gguf_path(pdd_path)) {
            m->pw.g = gguf_mmap_open(pdd_path);
            if (!m->pw.g) throw std::runtime_error(std::string("cannot open ") + pdd_path);
            if (std::string(gguf_arch(m->pw.g)) != "minimax_h3_pdd") throw std::runtime_error("not a MiniMax-H3 PDD heads GGUF");
            m->pdd_w = floats_of(m->pw, "video_out.weight", (size_t)kSteps * kPatch * kH);
            m->pdd_b = floats_of(m->pw, "video_out.bias", (size_t)kSteps * kPatch);
            sddev::Floats s = floats_of(m->pw, "sigmas", (size_t)kSteps + 1);
            m->sigmas.assign(s.data(), s.data() + s.size());
        } else {
            m->pw.st = st_open(pdd_path);
            if (!m->pw.st) throw std::runtime_error(std::string("minimax dit: ") + st_last_error());
            m->pdd_w = floats_of(m->pw, "video_out.weight", (size_t)kSteps * kPatch * kH);
            m->pdd_b = floats_of(m->pw, "video_out.bias", (size_t)kSteps * kPatch);
            const std::string pdir = std::string(pdd_path).substr(0, std::string(pdd_path).rfind('/'));
            m->sigmas = sigmas_of_config(pdir);
            if (m->sigmas.size() != (size_t)kSteps + 1)
                throw std::runtime_error("minimax dit: pdd_config.json has no sigmas of " + std::to_string(kSteps + 1));
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
    for (const void* base : model->dev_bases) sddev::unregister_weights(base);
    delete model;  // (the weight files close with their WeightFile)
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
            if (const char* rin = std::getenv("MMDIT_REPLACE_IN")) {  // (debugging: this layer's input from a file, the same
                const char* rl = std::getenv("MMDIT_DEBUG_LAYER");  // input as another run's; with MMDIT_DEBUG_LAYER)
                if (rl && (int)l == std::atoi(rl)) {
                    if (FILE* f = std::fopen(rin, "rb")) {
                        const size_t got = std::fread(hid.data(), sizeof(float), (size_t)S * kH, f);
                        std::fclose(f);
                        if (got != (size_t)S * kH) { g_error = "MMDIT_REPLACE_IN: wrong size"; return -1; }
                    }
                }
            }
            sddev::Floats mod((size_t)3 * 6 * kH);
            lin_apply(B.adaln, sil.data(), 1, mod.data());
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
        lin_apply(m->fadaln, sil.data(), 1, fm.data());
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
