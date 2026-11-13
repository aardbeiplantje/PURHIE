// Stable Diffusion in the engine (sd.h): the GGUF's tables (the names of the checkpoint's tensors by component, the configs,
// the CLIP vocabulary and merges), the CLIP tokenizer (byte-level BPE with the word-end marker), and the text encoder
// (CLIPTextModel: token and position embeddings, causal self-attention with GELU MLPs, the final layer norm). The weights
// are F16 read in place; the arithmetic float32, the order of the operations of transformers' CLIP.
#include "sd.h"

#include "gguf_mmap.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
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

// y[t][o] = b[o] + sum_i x[t][i] W[o][i] for T rows; W [out][in] F16 in place
void linear16(const float* x, int T, const T16& W, const T16* b, float* y) {
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
    std::unordered_map<std::string, T16> t;  // the checkpoint's tensors by their diffusers names
    int te_ctx = 77, te_dim = 1024, te_layers = 23, te_heads = 16, te_ff = 4096, te_vocab = 49408, te_bos = 49406, te_eos = 49407,
        te_pad = 49407;
    std::vector<int> vae_ch;                        // the VAE's block_out_channels
    int vae_layers = 2, vae_groups = 32, vae_latent = 4;
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

// 2-D convolution, stride 1, zero padding p: y [cout][H'][W'] with H' = H + 2p - k + 1. The inner loop runs along the row.
void conv2d(const float* x, int cin, int H, int W, const T16& w, const T16* b, int k, int p, float* y) {
    const int cout = (int)w.ne[3];
    const int Ho = H + 2 * p - k + 1, Wo = W + 2 * p - k + 1;
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
                    const int dx = kx - p;
                    const int x0 = std::max(0, -dx), x1 = std::min(Wo, W - dx);
                    for (int yy = 0; yy < Ho; ++yy) {
                        const int sy = yy + ky - p;
                        if (sy < 0 || sy >= H) continue;
                        float* yr = yp + (size_t)yy * Wo;
                        const float* xr = xp + (size_t)sy * W + dx;
#pragma omp simd
                        for (int xx = x0; xx < x1; ++xx) yr[xx] += wv * xr[xx];
                    }
                }
        }
    }
}

// GroupNorm over `groups` groups of channels (biased variance), per-channel affine: y = (x - mean) / sqrt(var + eps) * w + b
void group_norm(const float* x, int C, int HW, int groups, const T16& gw, const T16& gb, float eps, float* y) {
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
#pragma omp simd
    for (size_t i = 0; i < n; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

struct VaeCfg {
    std::vector<int> ch;  // block_out_channels
    int layers = 2, groups = 32, latent = 4;
    float scale = 0.18215f, eps = 1e-6f;
};

// one resnet: norm1, SiLU, conv1, norm2, SiLU, conv2, plus the input (through a 1x1 conv when the channels change)
void vae_resnet(const sd_model* m, const std::string& p, const float* x, int cin, int cout, int H, int W, const VaeCfg& cfg,
                float* out) {
    const int HW = H * W;
    std::vector<float> h((size_t)cin * HW), g((size_t)cin * HW), c1((size_t)cout * HW);
    group_norm(x, cin, HW, cfg.groups, need(m, p + "norm1.weight"), need(m, p + "norm1.bias"), cfg.eps, h.data());
    silu_in(h.data(), h.size());
    conv2d(h.data(), cin, H, W, need(m, p + "conv1.weight"), &need(m, p + "conv1.bias"), 3, 1, c1.data());
    group_norm(c1.data(), cout, HW, cfg.groups, need(m, p + "norm2.weight"), need(m, p + "norm2.bias"), cfg.eps, g.data());
    silu_in(g.data(), (size_t)cout * HW);
    conv2d(g.data(), cout, H, W, need(m, p + "conv2.weight"), &need(m, p + "conv2.bias"), 3, 1, out);
    if (cin == cout) {
        for (size_t i = 0; i < (size_t)cout * HW; ++i) out[i] += x[i];
    } else {
        std::vector<float> sc((size_t)cout * HW);
        conv2d(x, cin, H, W, need(m, p + "conv_shortcut.weight"), &need(m, p + "conv_shortcut.bias"), 1, 0, sc.data());
        for (size_t i = 0; i < (size_t)cout * HW; ++i) out[i] += sc[i];
    }
}

// the mid block's attention: one head of width C over the HW positions, the residual added
void vae_mid_attention(const sd_model* m, const std::string& p, float* x, int C, int H, int W, const VaeCfg& cfg) {
    const int HW = H * W;
    std::vector<float> n((size_t)C * HW);
    group_norm(x, C, HW, cfg.groups, need(m, p + "group_norm.weight"), need(m, p + "group_norm.bias"), cfg.eps, n.data());
    // (HW, C) token rows
    std::vector<float> t((size_t)HW * C), q((size_t)HW * C), k((size_t)HW * C), v((size_t)HW * C), o((size_t)HW * C),
        att((size_t)HW * C), sc((size_t)HW);
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < HW; ++i) t[(size_t)i * C + c] = n[(size_t)c * HW + i];
    linear16(t.data(), HW, need(m, p + "to_q.weight"), &need(m, p + "to_q.bias"), q.data());
    linear16(t.data(), HW, need(m, p + "to_k.weight"), &need(m, p + "to_k.bias"), k.data());
    linear16(t.data(), HW, need(m, p + "to_v.weight"), &need(m, p + "to_v.bias"), v.data());
    const float scale = 1.0f / std::sqrt((float)C);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < HW; ++i) {
        std::vector<float> row(HW);
        const float* qr = &q[(size_t)i * C];
        float mx = -INFINITY;
        for (int j = 0; j < HW; ++j) {
            const float* kr = &k[(size_t)j * C];
            float d = 0.0f;
            for (int c = 0; c < C; ++c) d += qr[c] * kr[c];
            row[j] = d * scale;
            mx = std::max(mx, row[j]);
        }
        double z = 0.0;
        for (int j = 0; j < HW; ++j) {
            row[j] = std::exp(row[j] - mx);
            z += row[j];
        }
        float* orow = &att[(size_t)i * C];
        for (int c = 0; c < C; ++c) orow[c] = 0.0f;
        for (int j = 0; j < HW; ++j) {
            const float pw = (float)(row[j] / z);
            const float* vr = &v[(size_t)j * C];
            for (int c = 0; c < C; ++c) orow[c] += pw * vr[c];
        }
    }
    linear16(att.data(), HW, need(m, p + "to_out.0.weight"), &need(m, p + "to_out.0.bias"), o.data());
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < HW; ++i) x[(size_t)c * HW + i] += o[(size_t)i * C + c];
}

// nearest-neighbour 2x upsampling of [C][H][W] into [C][2H][2W]
void upsample2(const float* x, int C, int H, int W, float* y) {
#pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < 2 * H; ++h)
            for (int w = 0; w < 2 * W; ++w) y[((size_t)c * 2 * H + h) * 2 * W + w] = x[((size_t)c * H + h / 2) * W + w / 2];
}

// the decoder: [4][h][w] latents (already divided by the scaling factor) to [3][8h][8w]
void vae_decode_planes(const sd_model* m, const VaeCfg& cfg, const float* z, int h, int w, float* out) {
    // post_quant_conv (1x1, 4 -> 4)
    std::vector<float> q((size_t)cfg.latent * h * w);
    conv2d(z, cfg.latent, h, w, need(m, "post_quant_conv.weight"), &need(m, "post_quant_conv.bias"), 1, 0, q.data());
    int H = h, W = w;
    int C = cfg.ch.back();
    std::vector<float> x((size_t)C * H * W);
    conv2d(q.data(), cfg.latent, H, W, need(m, "decoder.conv_in.weight"), &need(m, "decoder.conv_in.bias"), 3, 1, x.data());
    // the mid block
    {
        std::vector<float> r((size_t)C * H * W);
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
            std::vector<float> r((size_t)oc * H * W);
            vae_resnet(m, "decoder.up_blocks." + std::to_string(b) + ".resnets." + std::to_string(j) + ".", x.data(), in_c, oc, H,
                       W, cfg, r.data());
            x.swap(r);
        }
        prev = oc;
        if (b + 1 < nb) {  // the upsampler: nearest 2x, then a 3x3 convolution
            std::vector<float> u((size_t)oc * 2 * H * 2 * W), r((size_t)oc * 2 * H * 2 * W);
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
    std::vector<float> n((size_t)c0 * HW);
    group_norm(x.data(), c0, HW, cfg.groups, need(m, "decoder.conv_norm_out.weight"), need(m, "decoder.conv_norm_out.bias"),
               cfg.eps, n.data());
    silu_in(n.data(), n.size());
    conv2d(n.data(), c0, H, W, need(m, "decoder.conv_out.weight"), &need(m, "decoder.conv_out.bias"), 3, 1, out);
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

void sd_close(sd_model_t* model) { delete model; }

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
        if (m->vae_ch.empty()) throw std::runtime_error("sd: no VAE channels in the GGUF");
        VaeCfg cfg;
        cfg.ch = m->vae_ch;
        cfg.layers = m->vae_layers;
        cfg.groups = m->vae_groups;
        cfg.latent = m->vae_latent;
        cfg.scale = m->vae_scale;
        std::vector<float> z((size_t)cfg.latent * h * w);
        for (size_t i = 0; i < z.size(); ++i) z[i] = latents[i] / cfg.scale;
        vae_decode_planes(m, cfg, z.data(), h, w, image);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

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
