// Stable Diffusion in the engine (sd.h): the GGUF's tables (the names of the checkpoint's tensors by component, the configs,
// the CLIP vocabulary and merges), the CLIP tokenizer (byte-level BPE with the word-end marker), and the text encoder
// (CLIPTextModel: token and position embeddings, causal self-attention with GELU MLPs, the final layer norm). The weights
// are F16 read in place; the arithmetic float32, the order of the operations of transformers' CLIP.
#include "sd.h"

#include "gguf_mmap.h"
#include "sd_device.h"

#include <zlib.h>

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::string g_error;

// binary16 to float (the hardware conversion when the build has F16C, else the bit arithmetic)
inline float f16(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, out;
    if (e == 0) {
        if (m == 0) {
            out = s;
        } else {
            e = 1;
            while (!(m & 0x400)) { m <<= 1; --e; }
            m &= 0x3ff;
            out = s | ((e + 127 - 15) << 23) | (m << 13);
        }
    } else if (e == 31) {
        out = s | 0x7f800000u | (m << 13);
    } else {
        out = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
#endif
}

// A tensor of the checkpoint in the GGUF: F16 values in place, and its shape (the GGUF's ne: innermost first)
struct T16 {
    const uint16_t* p = nullptr;
    std::vector<int64_t> ne;
    size_t n() const {
        size_t c = 1;
        for (int64_t v : ne) c *= (size_t)v;
        return c;
    }
};

// The device path (sd_device.h): the primitives below run on the device when it is on and the memory is device-visible: the
// UNet's and the VAE's Floats. The text encoder runs on the host (HostOnly: its buffers are plain vectors).
using Floats = sddev::Floats;
thread_local int t_host_only = 0;
struct HostOnly {
    HostOnly() { ++t_host_only; }
    ~HostOnly() { --t_host_only; }
};
bool use_device() { return t_host_only == 0 && sddev::enabled(); }

// y[t][o] = b[o] + sum_i x[t][i] W[o][i] for T rows; W [out][in] F16 in place
void linear16(const float* x, int T, const T16& W, const T16* b, float* y) {
    if (use_device()) {
        sddev::linear_tok(x, T, W.p, (int)W.ne[0], (int)W.ne[1], b ? b->p : nullptr, y);
        return;
    }
    const int in = (int)W.ne[0], out = (int)W.ne[1];
#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; ++o) {
        const uint16_t* wr = W.p + (size_t)o * in;
        const float bias = b ? f16(b->p[o]) : 0.0f;
        for (int t = 0; t < T; ++t) {
            const float* xr = x + (size_t)t * in;
            float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
            for (int i = 0; i < in; ++i) acc += xr[i] * f16(wr[i]);
            y[(size_t)t * out + o] = bias + acc;
        }
    }
}

void layer_norm(const float* x, int T, int C, const T16& w, const T16& b, float eps, float* y) {
    if (use_device()) {
        sddev::layer_norm(x, T, C, w.p, b.p, eps, y);
        return;
    }
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t)t * C;
        double m = 0.0, v = 0.0;
        for (int i = 0; i < C; ++i) m += xr[i];
        m /= C;
        for (int i = 0; i < C; ++i) v += (xr[i] - m) * (xr[i] - m);
        v /= C;
        const double inv = 1.0 / std::sqrt(v + eps);
        for (int i = 0; i < C; ++i) y[(size_t)t * C + i] = (float)((xr[i] - m) * inv * f16(w.p[i]) + f16(b.p[i]));
    }
}

// the GPT-2 byte-to-unicode table of CLIP's tokenizer (printable bytes map to themselves, the others to 256 and up)
std::string byte_to_unicode(unsigned char c) {
    static std::vector<std::string> table;
    if (table.empty()) {
        std::vector<int> bs, cs;
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        cs = bs;
        int n = 0;
        std::vector<int> mapv(256, -1);
        for (int b : bs) mapv[b] = b;
        for (int b = 0; b < 256; ++b)
            if (mapv[b] < 0) {
                mapv[b] = 256 + n++;
            }
        table.resize(256);
        for (int b = 0; b < 256; ++b) {
            const int cp = mapv[b];
            std::string u;
            if (cp < 0x80) {
                u += (char)cp;
            } else {
                u += (char)(0xc0 | (cp >> 6));
                u += (char)(0x80 | (cp & 0x3f));
            }
            table[b] = u;
        }
    }
    return table[c];
}

// the CLIP word pattern on lowercased text (ASCII; the bytes of other characters count as letters): contractions, runs
// of letters, single digits, runs of other symbols; whitespace separates
std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    const size_t n = s.size();
    auto is_letter = [](unsigned char c) { return (c >= 'a' && c <= 'z') || c >= 0x80; };
    auto is_digit = [](unsigned char c) { return c >= '0' && c <= '9'; };
    auto is_space = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
    while (i < n) {
        const unsigned char c = (unsigned char)s[i];
        if (is_space(c)) { ++i; continue; }
        if (c == '\'' && i + 1 < n) {
            const std::string two = s.substr(i, 2);
            if (two == "'s" || two == "'t" || two == "'m" || two == "'d") { out.push_back(two); i += 2; continue; }
            if (i + 2 < n) {
                const std::string three = s.substr(i, 3);
                if (three == "'re" || three == "'ve" || three == "'ll") { out.push_back(three); i += 3; continue; }
            }
        }
        if (is_letter(c)) {
            size_t j = i;
            while (j < n && is_letter((unsigned char)s[j])) ++j;
            out.push_back(s.substr(i, j - i));
            i = j;
            continue;
        }
        if (is_digit(c)) { out.push_back(std::string(1, (char)c)); ++i; continue; }
        // the reference tokenizer (transformers' fast CLIP) makes each '!' a word of its own, without the word-end marker
        if (c == '!') { out.push_back("!"); ++i; continue; }
        size_t j = i;
        while (j < n && !is_space((unsigned char)s[j]) && !is_letter((unsigned char)s[j]) && !is_digit((unsigned char)s[j]) && s[j] != '!' &&
               !(s[j] == '\'' && j + 1 < n && (s[j + 1] == 's' || s[j + 1] == 't' || s[j + 1] == 'm' || s[j + 1] == 'd' ||
                                              s.compare(j + 1, 2, "re") == 0 || s.compare(j + 1, 2, "ve") == 0 ||
                                              s.compare(j + 1, 2, "ll") == 0)))
            ++j;
        if (j == i) j = i + 1;
        out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

}  // namespace

struct sd_model {
    gguf_model_t* g = nullptr;
    const void* dev_base = nullptr;  // the GGUF mapping registered for the device (NULL: none)
    size_t dev_bytes = 0;
    std::unordered_map<std::string, T16> t;  // the checkpoint's tensors by their diffusers names
    int te_ctx = 77, te_dim = 1024, te_layers = 23, te_heads = 16, te_ff = 4096, te_vocab = 49408, te_bos = 49406, te_eos = 49407,
        te_pad = 49407;
    std::vector<int> vae_ch;                        // the VAE's block_out_channels
    int vae_layers = 2, vae_groups = 32, vae_latent = 4;
    int sched_train = 1000;
    float sched_beta_start = 0.00085f, sched_beta_end = 0.012f;
    float vae_scale = 0.18215f;
    std::vector<std::string> vocab;                 // by id
    std::unordered_map<std::string, int> token_id;  // vocab string -> id
    std::unordered_map<std::string, int> merge_rank;  // "a b" -> rank
    ~sd_model() {
        if (g) gguf_mmap_close(g);
    }
};

namespace {

// the tensor of a diffusers name, or throws
const T16& need(const sd_model* m, const std::string& name) {
    auto it = m->t.find(name);
    if (it == m->t.end()) throw std::runtime_error("sd: no tensor " + name);
    return it->second;
}

// BPE of one word: the symbols (the word's bytes as CLIP's byte-level symbols, the last one with the word-end marker),
// merged by rank until no pair ranks
std::vector<std::string> bpe(const sd_model* m, const std::string& word) {
    if (word == "!") return {byte_to_unicode('!')};  // (a word of its own: no merges, no word-end marker)
    std::vector<std::string> sym;
    for (size_t i = 0; i < word.size(); ++i) sym.push_back(byte_to_unicode((unsigned char)word[i]));
    if (sym.empty()) return sym;
    sym.back() += "</w>";
    while (sym.size() > 1) {
        int best = -1, best_rank = INT32_MAX;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = m->merge_rank.find(sym[i] + " " + sym[i + 1]);
            if (it != m->merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best = (int)i;
            }
        }
        if (best < 0) break;
        sym[best] += sym[best + 1];
        sym.erase(sym.begin() + best + 1);
    }
    return sym;
}

void clip_forward(const sd_model* m, const int* ids, float* out) {
    HostOnly host_only;
    const int T = m->te_ctx, C = m->te_dim, H = m->te_heads, D = C / H;
    const T16& tok = need(m, "text_model.embeddings.token_embedding.weight");
    const T16& pos = need(m, "text_model.embeddings.position_embedding.weight");
    std::vector<float> x((size_t)T * C), h((size_t)T * C), q((size_t)T * C), k((size_t)T * C), v((size_t)T * C),
        att((size_t)T * C), tmp((size_t)T * C), ff((size_t)T * m->te_ff);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < C; ++i)
            x[(size_t)t * C + i] = f16(tok.p[(size_t)ids[t] * C + i]) + f16(pos.p[(size_t)t * C + i]);
    const std::string base = "text_model.encoder.layers.";
    for (int l = 0; l < m->te_layers; ++l) {
        const std::string p = base + std::to_string(l) + ".";
        layer_norm(x.data(), T, C, need(m, p + "layer_norm1.weight"), need(m, p + "layer_norm1.bias"), 1e-5f, h.data());
        linear16(h.data(), T, need(m, p + "self_attn.q_proj.weight"), &need(m, p + "self_attn.q_proj.bias"), q.data());
        linear16(h.data(), T, need(m, p + "self_attn.k_proj.weight"), &need(m, p + "self_attn.k_proj.bias"), k.data());
        linear16(h.data(), T, need(m, p + "self_attn.v_proj.weight"), &need(m, p + "self_attn.v_proj.bias"), v.data());
        // causal attention, per head: the scores of the positions up to t
        const float scale = 1.0f / std::sqrt((float)D);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t) {
            std::vector<float> sc(T);
            for (int hd = 0; hd < H; ++hd) {
                const float* qr = &q[(size_t)t * C + hd * D];
                float mx = -INFINITY;
                for (int s = 0; s <= t; ++s) {
                    const float* kr = &k[(size_t)s * C + hd * D];
                    float d = 0.0f;
                    for (int i = 0; i < D; ++i) d += qr[i] * kr[i];
                    sc[s] = d * scale;
                    mx = std::max(mx, sc[s]);
                }
                double z = 0.0;
                for (int s = 0; s <= t; ++s) {
                    sc[s] = std::exp(sc[s] - mx);
                    z += sc[s];
                }
                float* o = &att[(size_t)t * C + hd * D];
                for (int i = 0; i < D; ++i) o[i] = 0.0f;
                for (int s = 0; s <= t; ++s) {
                    const float pw = (float)(sc[s] / z);
                    const float* vr = &v[(size_t)s * C + hd * D];
                    for (int i = 0; i < D; ++i) o[i] += pw * vr[i];
                }
            }
        }
        linear16(att.data(), T, need(m, p + "self_attn.out_proj.weight"), &need(m, p + "self_attn.out_proj.bias"), tmp.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        layer_norm(x.data(), T, C, need(m, p + "layer_norm2.weight"), need(m, p + "layer_norm2.bias"), 1e-5f, h.data());
        linear16(h.data(), T, need(m, p + "mlp.fc1.weight"), &need(m, p + "mlp.fc1.bias"), ff.data());
        for (float& f : ff) f = 0.5f * f * (1.0f + std::erf(f / std::sqrt(2.0f)));  // GELU (exact)
        linear16(ff.data(), T, need(m, p + "mlp.fc2.weight"), &need(m, p + "mlp.fc2.bias"), tmp.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
    }
    layer_norm(x.data(), T, C, need(m, "text_model.final_layer_norm.weight"), need(m, "text_model.final_layer_norm.bias"), 1e-5f, out);
}


// ---- the VAE decoder (AutoencoderKL.decoder, diffusers 0.35: post_quant_conv, conv_in, the mid block (a resnet, one
// attention head over the positions, a resnet), the four up blocks (resnets, nearest upsampling and a convolution), the
// output norm, SiLU, conv_out). Channel-major planes [C][H][W]; the weights F16 in place.

// 2-D convolution with stride st and zero padding p: y [cout][H'][W'], H' = (H + 2p - k) / st + 1. The inner loop runs
// along the output row (stride 1: contiguous).
void conv2d_s(const float* x, int cin, int H, int W, const T16& w, const T16* b, int k, int p, int st, float* y, int* Ho_out = nullptr,
              int* Wo_out = nullptr) {
    const int cout = (int)w.ne[3];
    const int Ho = (H + 2 * p - k) / st + 1, Wo = (W + 2 * p - k) / st + 1;
    if (Ho_out) *Ho_out = Ho;
    if (Wo_out) *Wo_out = Wo;
    if (use_device()) {
        sddev::conv2d(x, cin, H, W, w.p, cout, b ? b->p : nullptr, k, p, st, y, Ho, Wo);
        return;
    }
#pragma omp parallel for schedule(static)
    for (int o = 0; o < cout; ++o) {
        float* yp = y + (size_t)o * Ho * Wo;
        const float bias = b ? f16(b->p[o]) : 0.0f;
        for (size_t i = 0; i < (size_t)Ho * Wo; ++i) yp[i] = bias;
        for (int c = 0; c < cin; ++c) {
            const float* xp = x + (size_t)c * H * W;
            for (int ky = 0; ky < k; ++ky)
                for (int kx = 0; kx < k; ++kx) {
                    const float wv = f16(w.p[(((size_t)o * cin + c) * k + ky) * k + kx]);
                    // output xx reads input column xx * st + kx - p: the valid output columns
                    const int dx = kx - p;
                    const int x0 = dx >= 0 ? 0 : (-dx + st - 1) / st;
                    const int x1 = std::min(Wo, (W - dx + st - 1) / st);
                    for (int yy = 0; yy < Ho; ++yy) {
                        const int sy = yy * st + ky - p;
                        if (sy < 0 || sy >= H) continue;
                        float* yr = yp + (size_t)yy * Wo;
                        const float* xr = xp + (size_t)sy * W;
                        if (st == 1) {
#pragma omp simd
                            for (int xx = x0; xx < x1; ++xx) yr[xx] += wv * xr[xx + dx];
                        } else {
                            for (int xx = x0; xx < x1; ++xx) yr[xx] += wv * xr[xx * st + dx];
                        }
                    }
                }
        }
    }
}

void conv2d(const float* x, int cin, int H, int W, const T16& w, const T16* b, int k, int p, float* y) {
    conv2d_s(x, cin, H, W, w, b, k, p, 1, y);
}

// GroupNorm over `groups` groups of channels (biased variance), per-channel affine: y = (x - mean) / sqrt(var + eps) * w + b
void group_norm(const float* x, int C, int HW, int groups, const T16& gw, const T16& gb, float eps, float* y) {
    if (use_device()) {
        sddev::group_norm(x, C, HW, groups, gw.p, gb.p, eps, y);
        return;
    }
    const int per = C / groups;
#pragma omp parallel for schedule(static)
    for (int g = 0; g < groups; ++g) {
        const size_t n = (size_t)per * HW;
        const float* base = x + (size_t)g * per * HW;
        double m = 0.0, v = 0.0;
        for (size_t i = 0; i < n; ++i) m += base[i];
        m /= n;
        for (size_t i = 0; i < n; ++i) v += (base[i] - m) * (base[i] - m);
        v /= n;
        const double inv = 1.0 / std::sqrt(v + eps);
        for (int c = 0; c < per; ++c) {
            const int ch = g * per + c;
            const float sc = (float)inv * f16(gw.p[ch]), sh = f16(gb.p[ch]);
            for (int i = 0; i < HW; ++i) y[(size_t)ch * HW + i] = (base[(size_t)c * HW + i] - (float)m) * sc + sh;
        }
    }
}

void silu_in(float* x, size_t n) {
    if (use_device()) {
        sddev::silu(x, n);
        return;
    }
#pragma omp simd
    for (size_t i = 0; i < n; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

// y[t][o] = b[o] + sum_c x[c][t] W[o][c]: the input in channel planes [C][T], the output in token rows [T][out]
void linear16_cm(const float* x, int T, const T16& W, const T16* b, float* y) {
    if (use_device()) {
        sddev::linear_cm(x, T, W.p, (int)W.ne[0], (int)W.ne[1], b ? b->p : nullptr, y);
        return;
    }
    const int in = (int)W.ne[0], out = (int)W.ne[1];
#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; ++o) {
        const uint16_t* wr = W.p + (size_t)o * in;
        const float bias = b ? f16(b->p[o]) : 0.0f;
        for (int t = 0; t < T; ++t) {
            float acc = 0.0f;
            for (int c = 0; c < in; ++c) acc += x[(size_t)c * T + t] * f16(wr[c]);
            y[(size_t)t * out + o] = bias + acc;
        }
    }
}

// y[o][t] (+=) b[o] + sum_i x[t][i] W[o][i]: the input in token rows [T][in], the output in channel planes [out][T]
void linear16_to_cm(const float* x, int T, const T16& W, const T16* b, float* y, bool accumulate) {
    if (use_device()) {
        sddev::linear_to_cm(x, T, W.p, (int)W.ne[0], (int)W.ne[1], b ? b->p : nullptr, y, accumulate);
        return;
    }
    const int in = (int)W.ne[0], out = (int)W.ne[1];
#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; ++o) {
        const uint16_t* wr = W.p + (size_t)o * in;
        const float bias = b ? f16(b->p[o]) : 0.0f;
        for (int t = 0; t < T; ++t) {
            const float* xr = x + (size_t)t * in;
            float acc = 0.0f;
            for (int i = 0; i < in; ++i) acc += xr[i] * f16(wr[i]);
            float& dst = y[(size_t)o * T + t];
            dst = (accumulate ? dst : 0.0f) + bias + acc;
        }
    }
}

// out += x over n floats
void add_into(float* out, const float* x, size_t n) {
    if (use_device()) {
        sddev::add(out, x, n);
        return;
    }
#pragma omp simd
    for (size_t i = 0; i < n; ++i) out[i] += x[i];
}

// c[ch][i] += t[ch] over the channel planes [C][HW]
void add_channel(float* c, const float* t, int C, int HW) {
#pragma omp parallel for schedule(static)
    for (int ch = 0; ch < C; ++ch)
        for (int i = 0; i < HW; ++i) c[(size_t)ch * HW + i] += t[ch];
}

// GEGLU: g[t][j] = pj[t][j] * gelu(pj[t][ff + j]) for the projection rows pj [rows][2 ff]
void geglu(const float* pj, int rows, int ff, float* g) {
    if (use_device()) {
        sddev::geglu(pj, rows, ff, g);
        return;
    }
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < ff; ++j) {
            const float hid = pj[(size_t)i * 2 * ff + j], gate = pj[(size_t)i * 2 * ff + ff + j];
            g[(size_t)i * ff + j] = hid * (0.5f * gate * (1.0f + std::erf(gate / std::sqrt(2.0f))));
        }
}

void attend(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out);

struct VaeCfg {
    std::vector<int> ch;  // block_out_channels
    int layers = 2, groups = 32, latent = 4;
    float scale = 0.18215f, eps = 1e-6f;
};

// one resnet: norm1, SiLU, conv1, norm2, SiLU, conv2, plus the input (through a 1x1 conv when the channels change)
void vae_resnet(const sd_model* m, const std::string& p, const float* x, int cin, int cout, int H, int W, const VaeCfg& cfg,
                float* out) {
    const int HW = H * W;
    Floats h((size_t)cin * HW), g((size_t)cin * HW), c1((size_t)cout * HW);
    group_norm(x, cin, HW, cfg.groups, need(m, p + "norm1.weight"), need(m, p + "norm1.bias"), cfg.eps, h.data());
    silu_in(h.data(), h.size());
    conv2d(h.data(), cin, H, W, need(m, p + "conv1.weight"), &need(m, p + "conv1.bias"), 3, 1, c1.data());
    group_norm(c1.data(), cout, HW, cfg.groups, need(m, p + "norm2.weight"), need(m, p + "norm2.bias"), cfg.eps, g.data());
    silu_in(g.data(), (size_t)cout * HW);
    conv2d(g.data(), cout, H, W, need(m, p + "conv2.weight"), &need(m, p + "conv2.bias"), 3, 1, out);
    if (cin == cout) {
        add_into(out, x, (size_t)cout * HW);
    } else {
        Floats sc((size_t)cout * HW);
        conv2d(x, cin, H, W, need(m, p + "conv_shortcut.weight"), &need(m, p + "conv_shortcut.bias"), 1, 0, sc.data());
        add_into(out, sc.data(), (size_t)cout * HW);
    }
}

// the mid block's attention: one head of width C over the HW positions, the residual added
// the mid block's attention: one head of width C over the HW positions, the residual added (the projections read the planes)
void vae_mid_attention(const sd_model* m, const std::string& p, float* x, int C, int H, int W, const VaeCfg& cfg) {
    const int HW = H * W;
    Floats n((size_t)C * HW);
    group_norm(x, C, HW, cfg.groups, need(m, p + "group_norm.weight"), need(m, p + "group_norm.bias"), cfg.eps, n.data());
    Floats q((size_t)HW * C), k((size_t)HW * C), v((size_t)HW * C), att((size_t)HW * C);
    linear16_cm(n.data(), HW, need(m, p + "to_q.weight"), &need(m, p + "to_q.bias"), q.data());
    linear16_cm(n.data(), HW, need(m, p + "to_k.weight"), &need(m, p + "to_k.bias"), k.data());
    linear16_cm(n.data(), HW, need(m, p + "to_v.weight"), &need(m, p + "to_v.bias"), v.data());
    attend(q.data(), HW, k.data(), v.data(), HW, C, 1, att.data());
    linear16_to_cm(att.data(), HW, need(m, p + "to_out.0.weight"), &need(m, p + "to_out.0.bias"), x, true);
}

// nearest-neighbour 2x upsampling of [C][H][W] into [C][2H][2W]
void upsample2(const float* x, int C, int H, int W, float* y) {
    if (use_device()) {
        sddev::upsample2(x, C, H, W, y);
        return;
    }
#pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < 2 * H; ++h)
            for (int w = 0; w < 2 * W; ++w) y[((size_t)c * 2 * H + h) * 2 * W + w] = x[((size_t)c * H + h / 2) * W + w / 2];
}

// the decoder: [4][h][w] latents (already divided by the scaling factor) to [3][8h][8w]
// the decoder: [4][h][w] latents (already divided by the scaling factor) to [3][8h][8w]
void vae_decode_planes(const sd_model* m, const VaeCfg& cfg, const float* z, int h, int w, float* out) {
    // post_quant_conv (1x1, 4 -> 4)
    Floats q((size_t)cfg.latent * h * w);
    conv2d(z, cfg.latent, h, w, need(m, "post_quant_conv.weight"), &need(m, "post_quant_conv.bias"), 1, 0, q.data());
    int H = h, W = w;
    int C = cfg.ch.back();
    Floats x((size_t)C * H * W);
    conv2d(q.data(), cfg.latent, H, W, need(m, "decoder.conv_in.weight"), &need(m, "decoder.conv_in.bias"), 3, 1, x.data());
    // the mid block
    {
        Floats r((size_t)C * H * W);
        vae_resnet(m, "decoder.mid_block.resnets.0.", x.data(), C, C, H, W, cfg, r.data());
        x.swap(r);
        vae_mid_attention(m, "decoder.mid_block.attentions.0.", x.data(), C, H, W, cfg);
        vae_resnet(m, "decoder.mid_block.resnets.1.", x.data(), C, C, H, W, cfg, r.data());
        x.swap(r);
    }
    // the up blocks: the channels in reverse order; a block's resnets (the first takes the previous block's channels)
    const int nb = (int)cfg.ch.size();
    int prev = C;
    for (int b = 0; b < nb; ++b) {
        const int oc = cfg.ch[nb - 1 - b];
        for (int j = 0; j <= cfg.layers; ++j) {
            const int in_c = j == 0 ? prev : oc;
            Floats r((size_t)oc * H * W);
            vae_resnet(m, "decoder.up_blocks." + std::to_string(b) + ".resnets." + std::to_string(j) + ".", x.data(), in_c, oc, H,
                       W, cfg, r.data());
            x.swap(r);
        }
        prev = oc;
        if (b + 1 < nb) {  // the upsampler: nearest 2x, then a 3x3 convolution
            Floats u((size_t)oc * 2 * H * 2 * W), r((size_t)oc * 2 * H * 2 * W);
            upsample2(x.data(), oc, H, W, u.data());
            H *= 2;
            W *= 2;
            conv2d(u.data(), oc, H, W, need(m, "decoder.up_blocks." + std::to_string(b) + ".upsamplers.0.conv.weight"),
                   &need(m, "decoder.up_blocks." + std::to_string(b) + ".upsamplers.0.conv.bias"), 3, 1, r.data());
            x.swap(r);
        }
    }
    // the output: norm, SiLU, conv_out (128 -> 3)
    const int HW = H * W, c0 = cfg.ch[0];
    Floats n((size_t)c0 * HW);
    group_norm(x.data(), c0, HW, cfg.groups, need(m, "decoder.conv_norm_out.weight"), need(m, "decoder.conv_norm_out.bias"),
               cfg.eps, n.data());
    silu_in(n.data(), n.size());
    conv2d(n.data(), c0, H, W, need(m, "decoder.conv_out.weight"), &need(m, "decoder.conv_out.bias"), 3, 1, out);
}


// ---- the UNet (UNet2DConditionModel, SD 2.x: sinusoidal time embedding, ResnetBlock2D with the time projection,
// Transformer2DModel with linear projections (self-attention, cross-attention over the prompt, GEGLU feed-forward), the
// down, mid and up blocks with the skip connections; heads of width 64).

// the sinusoidal embedding of t (flip_sin_to_cos: cos first; frequencies exp(-ln(10000) i / half))
void time_embedding(float t, int dim, float* out) {
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const double f = std::exp(-std::log(10000.0) * i / half);
        out[i] = (float)std::cos(t * f);
        out[half + i] = (float)std::sin(t * f);
    }
}

// the channels of a conv weight [cout][cin][k][k]
int conv_cin(const T16& w) { return (int)w.ne[2]; }
int conv_cout(const T16& w) { return (int)w.ne[3]; }

// ResnetBlock2D: norm1, SiLU, conv1, the time projection added per channel, norm2, SiLU, conv2, the input (through a 1x1
// shortcut when the channels change), added
void unet_resnet(const sd_model* m, const std::string& p, const float* x, int H, int W, const float* temb_act, float* out) {
    const T16& c1 = need(m, p + "conv1.weight");
    const int cin = conv_cin(c1), cout = conv_cout(c1), HW = H * W;
    Floats h((size_t)cin * HW), c((size_t)cout * HW), g((size_t)cout * HW);
    group_norm(x, cin, HW, 32, need(m, p + "norm1.weight"), need(m, p + "norm1.bias"), 1e-5f, h.data());
    silu_in(h.data(), h.size());
    conv2d(h.data(), cin, H, W, c1, &need(m, p + "conv1.bias"), 3, 1, c.data());
    // the time embedding's projection: [cout] per channel
    const T16& tw = need(m, p + "time_emb_proj.weight");
    const T16& tb = need(m, p + "time_emb_proj.bias");
    Floats tp(cout);
    linear16(temb_act, 1, tw, &tb, tp.data());
    add_channel(c.data(), tp.data(), cout, HW);
    group_norm(c.data(), cout, HW, 32, need(m, p + "norm2.weight"), need(m, p + "norm2.bias"), 1e-5f, g.data());
    silu_in(g.data(), g.size());
    conv2d(g.data(), cout, H, W, need(m, p + "conv2.weight"), &need(m, p + "conv2.bias"), 3, 1, out);
    if (cin == cout) {
        add_into(out, x, (size_t)cout * HW);
    } else {
        Floats sc((size_t)cout * HW);
        conv2d(x, cin, H, W, need(m, p + "conv_shortcut.weight"), &need(m, p + "conv_shortcut.bias"), 1, 0, sc.data());
        add_into(out, sc.data(), (size_t)cout * HW);
    }
}

// attention over token rows: q [Tq][C], k and v [Tk][C] (C = heads * 64): out [Tq][C], per row streaming softmax
void attend(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out) {
    if (use_device()) {
        sddev::attention(q, Tq, k, v, Tk, C, heads, out);
        return;
    }
    const int D = C / heads;
    const float scale = 1.0f / std::sqrt((float)D);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < Tq; ++i) {
        std::vector<float> row(Tk);
        for (int hd = 0; hd < heads; ++hd) {
            const float* qr = q + (size_t)i * C + hd * D;
            float mx = -INFINITY;
            for (int j = 0; j < Tk; ++j) {
                const float* kr = k + (size_t)j * C + hd * D;
                float d = 0.0f;
                for (int c = 0; c < D; ++c) d += qr[c] * kr[c];
                row[j] = d * scale;
                mx = std::max(mx, row[j]);
            }
            double z = 0.0;
            for (int j = 0; j < Tk; ++j) {
                row[j] = std::exp(row[j] - mx);
                z += row[j];
            }
            float* o = out + (size_t)i * C + hd * D;
            for (int c = 0; c < D; ++c) o[c] = 0.0f;
            for (int j = 0; j < Tk; ++j) {
                const float pw = (float)(row[j] / z);
                const float* vr = v + (size_t)j * C + hd * D;
                for (int c = 0; c < D; ++c) o[c] += pw * vr[c];
            }
        }
    }
}

// Transformer2DModel: GroupNorm, proj_in (linear), one BasicTransformerBlock (self-attention, cross-attention over the
// context [Tc][1024], GEGLU feed-forward, each with its LayerNorm and residual), proj_out (linear), the input added
void unet_transformer(const sd_model* m, const std::string& p, float* x, int C, int H, int W, const float* ctx, int Tc, int heads) {
    const int HW = H * W;
    Floats n((size_t)C * HW);
    group_norm(x, C, HW, 32, need(m, p + "norm.weight"), need(m, p + "norm.bias"), 1e-6f, n.data());
    // the token rows (HW, C) of the normalized planes: proj_in reads the planes
    Floats h((size_t)HW * C);
    linear16_cm(n.data(), HW, need(m, p + "proj_in.weight"), &need(m, p + "proj_in.bias"), h.data());
    const std::string b = p + "transformer_blocks.0.";
    Floats ln((size_t)HW * C), q((size_t)HW * C), k((size_t)HW * C), v((size_t)HW * C), a((size_t)HW * C), tmp((size_t)HW * C);
    auto norm = [&](const std::string& name) {
        layer_norm(h.data(), HW, C, need(m, b + name + ".weight"), need(m, b + name + ".bias"), 1e-5f, ln.data());
    };
    // self-attention
    norm("norm1");
    linear16(ln.data(), HW, need(m, b + "attn1.to_q.weight"), nullptr, q.data());
    linear16(ln.data(), HW, need(m, b + "attn1.to_k.weight"), nullptr, k.data());
    linear16(ln.data(), HW, need(m, b + "attn1.to_v.weight"), nullptr, v.data());
    attend(q.data(), HW, k.data(), v.data(), HW, C, heads, a.data());
    linear16(a.data(), HW, need(m, b + "attn1.to_out.0.weight"), &need(m, b + "attn1.to_out.0.bias"), tmp.data());
    add_into(h.data(), tmp.data(), h.size());
    // cross-attention over the prompt
    norm("norm2");
    Floats kc((size_t)Tc * C), vc((size_t)Tc * C);
    linear16(ln.data(), HW, need(m, b + "attn2.to_q.weight"), nullptr, q.data());
    linear16(ctx, Tc, need(m, b + "attn2.to_k.weight"), nullptr, kc.data());
    linear16(ctx, Tc, need(m, b + "attn2.to_v.weight"), nullptr, vc.data());
    attend(q.data(), HW, kc.data(), vc.data(), Tc, C, heads, a.data());
    linear16(a.data(), HW, need(m, b + "attn2.to_out.0.weight"), &need(m, b + "attn2.to_out.0.bias"), tmp.data());
    add_into(h.data(), tmp.data(), h.size());
    // GEGLU feed-forward: the projection's halves, the hidden times the GELU of the gate
    norm("norm3");
    const int ff = (int)need(m, b + "ff.net.2.weight").ne[0];  // the inner width (4 C)
    Floats pj((size_t)HW * 2 * ff), g2((size_t)HW * ff);
    linear16(ln.data(), HW, need(m, b + "ff.net.0.proj.weight"), &need(m, b + "ff.net.0.proj.bias"), pj.data());
    geglu(pj.data(), HW, ff, g2.data());
    linear16(g2.data(), HW, need(m, b + "ff.net.2.weight"), &need(m, b + "ff.net.2.bias"), tmp.data());
    add_into(h.data(), tmp.data(), h.size());
    // proj_out (into the planes, added to the input)
    linear16_to_cm(h.data(), HW, need(m, p + "proj_out.weight"), &need(m, p + "proj_out.bias"), x, true);
}


// the UNet's forward: sample [4][h][w] at the timestep t, the context [77][1024]; out [4][h][w]
void unet_forward(const sd_model* m, const float* sample, int h, int w, float t, const float* ctx, int Tc, float* out) {
    // the time embedding: the sinusoid (320), linear, SiLU, linear (1280); the resnets take its SiLU
    Floats te(320), e1(1280), e2(1280);
    time_embedding(t, 320, te.data());
    linear16(te.data(), 1, need(m, "time_embedding.linear_1.weight"), &need(m, "time_embedding.linear_1.bias"), e1.data());
    silu_in(e1.data(), e1.size());
    linear16(e1.data(), 1, need(m, "time_embedding.linear_2.weight"), &need(m, "time_embedding.linear_2.bias"), e2.data());
    Floats ta = e2;
    silu_in(ta.data(), ta.size());
    // the sample and the context in device-visible memory
    Floats sx((size_t)4 * h * w), cx((size_t)Tc * m->te_dim);
    std::memcpy(sx.data(), sample, sizeof(float) * sx.size());
    std::memcpy(cx.data(), ctx, sizeof(float) * cx.size());
    const float* cxp = cx.data();
    // conv_in
    int H = h, W = w;
    Floats x((size_t)320 * H * W);
    conv2d(sx.data(), 4, H, W, need(m, "conv_in.weight"), &need(m, "conv_in.bias"), 3, 1, x.data());
    // the down blocks, each skip recorded: the input, every resnet's output, every downsampler's output
    std::vector<Floats> skips;
    skips.push_back(x);
    for (int b = 0; b < 4; ++b) {
        const bool attn = b < 3;
        for (int j = 0; j < 2; ++j) {
            const std::string rp = "down_blocks." + std::to_string(b) + ".resnets." + std::to_string(j) + ".";
            const int cout = conv_cout(need(m, rp + "conv1.weight"));
            Floats y((size_t)cout * H * W);
            unet_resnet(m, rp, x.data(), H, W, ta.data(), y.data());
            x.swap(y);
            if (attn) unet_transformer(m, "down_blocks." + std::to_string(b) + ".attentions." + std::to_string(j) + ".", x.data(),
                                       cout, H, W, cxp, Tc, cout / 64);
            skips.push_back(x);
        }
        if (b < 3) {  // the downsampler: a 3x3 convolution with stride 2
            const std::string dp = "down_blocks." + std::to_string(b) + ".downsamplers.0.conv.";
            const T16& dw = need(m, dp + "weight");
            const int C = conv_cout(dw), cin = conv_cin(dw);
            int H2 = 0, W2 = 0;
            Floats y((size_t)C * ((H + 2 - 3) / 2 + 1) * ((W + 2 - 3) / 2 + 1));
            conv2d_s(x.data(), cin, H, W, dw, &need(m, dp + "bias"), 3, 1, 2, y.data(), &H2, &W2);
            x.swap(y);
            H = H2;
            W = W2;
            skips.push_back(x);
        }
    }
    // the mid block: a resnet, a transformer over the 1280 channels, a resnet
    {
        const int C = 1280;
        Floats y((size_t)C * H * W);
        unet_resnet(m, "mid_block.resnets.0.", x.data(), H, W, ta.data(), y.data());
        x.swap(y);
        unet_transformer(m, "mid_block.attentions.0.", x.data(), C, H, W, cxp, Tc, C / 64);
        y = Floats((size_t)C * H * W);
        unet_resnet(m, "mid_block.resnets.1.", x.data(), H, W, ta.data(), y.data());
        x.swap(y);
    }
    // the up blocks: three resnets each over the concatenated skips (the last ones first), then the upsampler
    for (int b = 0; b < 4; ++b) {
        const bool attn = b > 0;
        for (int j = 0; j < 3; ++j) {
            Floats skip = std::move(skips.back());
            skips.pop_back();
            const int xc = (int)(x.size() / ((size_t)H * W)), sc = (int)(skip.size() / ((size_t)H * W));
            Floats cat((size_t)(xc + sc) * H * W);
            std::memcpy(cat.data(), x.data(), sizeof(float) * x.size());
            std::memcpy(cat.data() + x.size(), skip.data(), sizeof(float) * skip.size());
            const std::string rp = "up_blocks." + std::to_string(b) + ".resnets." + std::to_string(j) + ".";
            const int cout = conv_cout(need(m, rp + "conv1.weight"));
            Floats y((size_t)cout * H * W);
            unet_resnet(m, rp, cat.data(), H, W, ta.data(), y.data());
            x.swap(y);
            if (attn) unet_transformer(m, "up_blocks." + std::to_string(b) + ".attentions." + std::to_string(j) + ".", x.data(),
                                       cout, H, W, cxp, Tc, cout / 64);
        }
        if (b < 3) {  // the upsampler: nearest 2x, then a 3x3 convolution
            const std::string up = "up_blocks." + std::to_string(b) + ".upsamplers.0.conv.";
            const T16& uw = need(m, up + "weight");
            const int C = conv_cout(uw);
            Floats u((size_t)C * 2 * H * 2 * W), y((size_t)C * 2 * H * 2 * W);
            upsample2(x.data(), C, H, W, u.data());
            H *= 2;
            W *= 2;
            conv2d(u.data(), C, H, W, uw, &need(m, up + "bias"), 3, 1, y.data());
            x.swap(y);
        }
    }
    // the output: norm, SiLU, conv_out (320 -> 4)
    const int HW = H * W;
    Floats n((size_t)320 * HW);
    group_norm(x.data(), 320, HW, 32, need(m, "conv_norm_out.weight"), need(m, "conv_norm_out.bias"), 1e-5f, n.data());
    silu_in(n.data(), n.size());
    Floats o((size_t)4 * H * W);
    conv2d(n.data(), 320, H, W, need(m, "conv_out.weight"), &need(m, "conv_out.bias"), 3, 1, o.data());
    std::memcpy(out, o.data(), sizeof(float) * o.size());
}

}  // namespace

namespace {

// The noise level of the one step (EulerDiscreteScheduler, trailing spacing, one step: t = 999): sigma = sqrt((1 - a) / a)
// with the scaled-linear schedule's cumulative product a (beta from sqrt(beta_start) to sqrt(beta_end), 1000 steps)
double sigma_at(const sd_model* m, int t) {
    const double b0 = std::sqrt((double)m->sched_beta_start), b1 = std::sqrt((double)m->sched_beta_end);
    double a = 1.0;
    for (int i = 0; i <= t; ++i) {
        const double b = (b0 + (b1 - b0) * i / (m->sched_train - 1)) * (b0 + (b1 - b0) * i / (m->sched_train - 1));
        a *= 1.0 - b;
    }
    return std::sqrt((1.0 - a) / a);
}

// PNG: IHDR, one IDAT of the zlib-compressed rows (filter 0), IEND; CRCs by zlib's crc32
void put_be32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24));
    v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}

void png_chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
    put_be32(out, (uint32_t)data.size());
    std::vector<uint8_t> td(type, type + 4);
    td.insert(td.end(), data.begin(), data.end());
    out.insert(out.end(), td.begin(), td.end());
    put_be32(out, (uint32_t)crc32(0L, td.data(), (uInt)td.size()));
}

}  // namespace

extern "C" {

sd_model_t* sd_open(const char* gguf_path) {
    try {
        std::unique_ptr<sd_model> m(new sd_model());
        m->g = gguf_mmap_open(gguf_path);
        if (!m->g) {
            g_error = std::string("cannot open ") + (gguf_path ? gguf_path : "(null)");
            return nullptr;
        }
        if (std::string(gguf_arch(m->g)) != "sd") throw std::runtime_error("not a Stable Diffusion GGUF");
        auto kv = [&](const char* key) { return (int)gguf_arch_int(m->g, key, -1); };
        m->te_ctx = kv("te.n_ctx");
        m->te_dim = kv("te.n_embd");
        m->te_layers = kv("te.n_layer");
        m->te_heads = kv("te.n_head");
        m->te_ff = kv("te.n_ff");
        m->te_vocab = kv("te.vocab");
        m->te_bos = kv("te.bos");
        m->te_eos = kv("te.eos");
        m->te_pad = kv("te.pad");
        m->vae_layers = kv("vae.layers_per_block");
        m->sched_train = kv("sched.num_train_timesteps") > 0 ? kv("sched.num_train_timesteps") : 1000;
        m->sched_beta_start = gguf_arch_float(m->g, "sched.beta_start", 0.00085f);
        m->sched_beta_end = gguf_arch_float(m->g, "sched.beta_end", 0.012f);
        m->vae_groups = kv("vae.norm_groups");
        m->vae_latent = kv("vae.latent_channels");
        m->vae_scale = gguf_arch_float(m->g, "vae.scaling_factor", 0.18215f);
        {
            int32_t type = 0;
            const void* data = nullptr;
            size_t count = 0;
            if (gguf_kv_get_array(m->g, "sd.vae.block_out_channels", &type, &data, &count) == 0)
                for (size_t i = 0; i < count; ++i) m->vae_ch.push_back((int)static_cast<const uint32_t*>(data)[i]);
        }
        if (m->te_pad < 0) m->te_pad = m->te_eos;
        if (m->te_ctx <= 0 || m->te_dim <= 0 || m->te_layers <= 0 || m->te_heads <= 0 || m->te_ff <= 0 || m->te_dim % m->te_heads)
            throw std::runtime_error("sd: missing or bad text encoder hyperparameters");
        // the tensors: the name table of each component (sd.names.<prefix>), the GGUF's tensor of that index (<prefix>NNNN)
        std::unordered_map<std::string, int64_t> index_of;
        const int64_t n_tensors = gguf_n_tensors(m->g);
        for (int64_t i = 0; i < n_tensors; ++i) index_of[gguf_tensor_name(m->g, i)] = i;
        for (const char* prefix : {"te", "un", "va"}) {
            const std::string key = std::string("sd.names.") + prefix;
            const char** names = nullptr;
            size_t count = 0;
            if (gguf_kv_get_string_array(m->g, key.c_str(), &names, &count) != 0) continue;
            for (size_t i = 0; i < count; ++i) {
                char gname[16];
                std::snprintf(gname, sizeof gname, "%s%04zu", prefix, i);
                auto it = index_of.find(gname);
                if (it == index_of.end()) throw std::runtime_error(std::string("sd: no tensor ") + gname);
                const int64_t idx = it->second;
                T16 T;
                size_t bytes = 0;
                T.p = static_cast<const uint16_t*>(gguf_get_tensor_ptr(m->g, gname, &bytes));
                const int dims = gguf_tensor_dims(m->g, idx);
                int64_t* ne = gguf_tensor_ne(m->g, idx);
                for (int d = 0; d < dims; ++d) T.ne.push_back(ne[d]);
                if (!T.p || bytes != T.n() * 2) throw std::runtime_error(std::string("sd: ") + gname + " is not F16 of its shape");
                m->t[names[i]] = T;
            }
            gguf_free_string_array(names, count);
        }
        // the weights for the device: the GGUF mapping registered read-only (one mapping; SD_DEVICE=cpu registers nothing)
        if (sddev::enabled()) {
            if (gguf_n_maps(m->g) != 1) throw std::runtime_error("sd: the GGUF is split; no device mapping");
            size_t bytes = 0;
            const void* base = gguf_map_base(m->g, &bytes);
            if (!sddev::register_weights(base, bytes)) throw std::runtime_error("sd: cannot register the weights for the device");
            m->dev_base = base;
            m->dev_bytes = bytes;
        }
        // the CLIP tokenizer
        const char** vocab = nullptr;
        size_t nv = 0;
        if (gguf_kv_get_string_array(m->g, "sd.tok.vocab", &vocab, &nv) != 0) throw std::runtime_error("sd: no sd.tok.vocab");
        m->vocab.assign(vocab, vocab + nv);
        gguf_free_string_array(vocab, nv);
        for (size_t i = 0; i < m->vocab.size(); ++i) m->token_id[m->vocab[i]] = (int)i;
        const char** merges = nullptr;
        size_t nm = 0;
        if (gguf_kv_get_string_array(m->g, "sd.tok.merges", &merges, &nm) != 0) throw std::runtime_error("sd: no sd.tok.merges");
        for (size_t i = 0; i < nm; ++i) {
            std::string line(merges[i]);
            const size_t sp = line.find(' ');
            if (sp != std::string::npos) m->merge_rank[line] = (int)i;
        }
        gguf_free_string_array(merges, nm);
        return m.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void sd_close(sd_model_t* model) {
    if (model) sddev::unregister_weights(model->dev_base);
    delete model;
}

const char* sd_last_error(void) { return g_error.c_str(); }

int sd_text_ctx(const sd_model_t* m) { return m ? m->te_ctx : 0; }
int sd_text_dim(const sd_model_t* m) { return m ? m->te_dim : 0; }

int sd_tokenize(sd_model_t* m, const char* prompt, int* ids) {
    if (!m || !prompt || !ids) { g_error = "sd_tokenize: bad arguments"; return -1; }
    try {
        std::string s(prompt);
        for (char& c : s)
            if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        // collapse the whitespace (the tokenizer's cleanup)
        std::string clean;
        for (char c : s) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                if (!clean.empty() && clean.back() != ' ') clean += ' ';
            } else {
                clean += c;
            }
        }
        while (!clean.empty() && clean.back() == ' ') clean.pop_back();
        std::vector<int> toks;
        for (const std::string& w : split_words(clean))
            for (const std::string& piece : bpe(m, w)) {
                auto it = m->token_id.find(piece);
                toks.push_back(it == m->token_id.end() ? m->te_eos : it->second);
            }
        const int T = m->te_ctx;
        if ((int)toks.size() > T - 2) toks.resize(T - 2);
        int n = 0;
        ids[n++] = m->te_bos;
        for (int t : toks) ids[n++] = t;
        ids[n++] = m->te_eos;
        const int real = n;
        while (n < T) ids[n++] = m->te_pad;  // (the tokenizer's pad token)
        return real;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int sd_vae_decode(sd_model_t* m, const float* latents, int h, int w, float* image) {
    if (!m || !latents || !image || h <= 0 || w <= 0) { g_error = "sd_vae_decode: bad arguments"; return -1; }
    try {
        std::lock_guard<std::recursive_mutex> lock(sddev::run_mutex());
        if (m->vae_ch.empty()) throw std::runtime_error("sd: no VAE channels in the GGUF");
        VaeCfg cfg;
        cfg.ch = m->vae_ch;
        cfg.layers = m->vae_layers;
        cfg.groups = m->vae_groups;
        cfg.latent = m->vae_latent;
        cfg.scale = m->vae_scale;
        Floats z((size_t)cfg.latent * h * w);
        for (size_t i = 0; i < z.size(); ++i) z[i] = latents[i] / cfg.scale;
        Floats o((size_t)3 * (8 * h) * (8 * w));
        vae_decode_planes(m, cfg, z.data(), h, w, o.data());
        std::memcpy(image, o.data(), sizeof(float) * o.size());
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int unet_call(sd_model_t* m, const float* sample, int h, int w, float t, const float* ctx, float* out) {
    try {
        unet_forward(m, sample, h, w, t, ctx, m->te_ctx, out);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int sd_unet(sd_model_t* m, const float* sample, int h, int w, float t, const float* ctx, float* out) {
    if (!m || !sample || !ctx || !out || h <= 0 || w <= 0 || h % 8 || w % 8) { g_error = "sd_unet: bad arguments"; return -1; }
    try {
        std::lock_guard<std::recursive_mutex> lock(sddev::run_mutex());
        unet_forward(m, sample, h, w, t, ctx, m->te_ctx, out);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}


int sd_random_latents(uint64_t seed, int h, int w, float* out) {
    if (!out || h <= 0 || w <= 0 || h % 8 || w % 8) { g_error = "sd_random_latents: bad arguments"; return -1; }
    // standard normal values from a Mersenne Twister (the seed fixes the image; this is not torch's sequence)
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t n = (size_t)4 * (h / 8) * (w / 8);
    for (size_t i = 0; i < n; ++i) out[i] = nd(rng);
    return 0;
}

int sd_init_sigma(sd_model_t* m, float* sigma) {
    if (!m || !sigma) { g_error = "sd_init_sigma: bad arguments"; return -1; }
    *sigma = (float)sigma_at(m, m->sched_train - 1);
    return 0;
}

int sd_generate(sd_model_t* m, const char* prompt, const float* init, int h, int w, uint8_t* rgb) {
    if (!m || !prompt || !init || !rgb || h <= 0 || w <= 0 || h % 8 || w % 8) { g_error = "sd_generate: bad arguments"; return -1; }
    try {
        std::lock_guard<std::recursive_mutex> lock(sddev::run_mutex());
        const int C = 4, lh = h / 8, lw = w / 8, HW = lh * lw;
        std::vector<float> ctx((size_t)m->te_ctx * m->te_dim);
        if (sd_encode_prompt(m, prompt, ctx.data()) != 0) throw std::runtime_error(g_error);
        // one Euler step from the noise level sigma to 0 (epsilon prediction): the model input is x / sqrt(sigma^2 + 1)
        const double sigma = sigma_at(m, m->sched_train - 1);
        std::vector<float> x(init, init + (size_t)C * HW), in(x.size()), eps(x.size());
        for (size_t i = 0; i < x.size(); ++i) in[i] = (float)(x[i] / std::sqrt(sigma * sigma + 1.0));
        if (unet_call(m, in.data(), lh, lw, (float)(m->sched_train - 1), ctx.data(), eps.data()) != 0)
            throw std::runtime_error(g_error);
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float)(x[i] - sigma * eps[i]);
        // the VAE: the image planes in [-1, 1], then diffusers' postprocess: (v / 2 + 0.5) clamped to [0, 1], times 255, rounded
        std::vector<float> img((size_t)3 * h * w);
        if (sd_vae_decode(m, x.data(), lh, lw, img.data()) != 0) throw std::runtime_error(g_error);
        for (int y = 0; y < h; ++y)
            for (int xx = 0; xx < w; ++xx)
                for (int c = 0; c < 3; ++c) {
                    float v = img[((size_t)c * h + y) * w + xx] / 2.0f + 0.5f;
                    v = std::min(1.0f, std::max(0.0f, v));
                    rgb[((size_t)y * w + xx) * 3 + c] = (uint8_t)std::lround(v * 255.0f);
                }
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int sd_encode_png(const uint8_t* rgb, int w, int h, uint8_t** out, size_t* n) {
    if (!rgb || !out || !n || w <= 0 || h <= 0) { g_error = "sd_encode_png: bad arguments"; return -1; }
    try {
        std::vector<uint8_t> raw((size_t)h * (1 + 3 * w));
        for (int y = 0; y < h; ++y) {
            raw[(size_t)y * (1 + 3 * w)] = 0;  // filter: none
            std::memcpy(&raw[(size_t)y * (1 + 3 * w) + 1], rgb + (size_t)y * 3 * w, (size_t)3 * w);
        }
        uLongf zlen = compressBound(raw.size());
        std::vector<uint8_t> z(zlen);
        if (compress2(z.data(), &zlen, raw.data(), raw.size(), 6) != Z_OK) throw std::runtime_error("sd_encode_png: zlib");
        z.resize(zlen);
        std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
        std::vector<uint8_t> ihdr;
        put_be32(ihdr, (uint32_t)w);
        put_be32(ihdr, (uint32_t)h);
        ihdr.push_back(8);  // bit depth
        ihdr.push_back(2);  // RGB
        ihdr.push_back(0);
        ihdr.push_back(0);
        ihdr.push_back(0);
        png_chunk(png, "IHDR", ihdr);
        png_chunk(png, "IDAT", z);
        png_chunk(png, "IEND", {});
        uint8_t* buf = static_cast<uint8_t*>(std::malloc(png.size()));
        if (!buf) throw std::runtime_error("sd_encode_png: no memory");
        std::memcpy(buf, png.data(), png.size());
        *out = buf;
        *n = png.size();
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

void sd_free_png(uint8_t* png) { std::free(png); }

int sd_encode_prompt(sd_model_t* m, const char* prompt, float* out) {
    if (!m || !prompt || !out) { g_error = "sd_encode_prompt: bad arguments"; return -1; }
    try {
        std::vector<int> ids(m->te_ctx);
        if (sd_tokenize(m, prompt, ids.data()) < 0) return -1;
        clip_forward(m, ids.data(), out);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

}  // extern "C"
