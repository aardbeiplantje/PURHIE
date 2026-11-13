// Qwen3-TTS's talker and code predictor on the host in float32 (qwen3tts.h): the prefill of the in-context voice clone
// (Qwen3TTSForConditionalGeneration.generate and generate_icl_prompt, qwen-tts 0.1.1), the talker's decoder (Qwen3 blocks:
// RMS norms, per-head q and k norms, GQA, SiLU-gated MLP, causal attention), and the code predictor (the same blocks,
// 16 positions per frame). The rotary is the standard one: the talker's M-RoPE with its three position axes equal (text
// only, no padding) is the same rotation.
#include "qwen3tts.h"

#include "gguf_mmap.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string g_error;

struct Lin {
    const float* w = nullptr;  // [out][in], row-major
    const float* b = nullptr;  // [out] or NULL
    int in = 0, out = 0;
};

struct Block {
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;  // q/k norm: [head_dim]
    Lin q, k, v, o, gate, up, down;
};

const float* tensor(const gguf_model_t* g, const std::string& name, size_t count, bool optional = false) {
    size_t bytes = 0;
    const void* p = gguf_get_tensor_ptr(g, name.c_str(), &bytes);
    if (!p) {
        if (optional) return nullptr;
        throw std::runtime_error("qwen3tts: no tensor " + name);
    }
    if (bytes != count * sizeof(float))
        throw std::runtime_error("qwen3tts: " + name + " has " + std::to_string(bytes) + " bytes, expected " +
                                 std::to_string(count * sizeof(float)));
    return static_cast<const float*>(p);
}

// y[t][o] = x[t][:] . w[o][:] + b[o], parallel over the (t, o) pairs (a decode step has one row: the outputs are the work)
void linear(const float* x, int T, const Lin& L, float* y) {
    const long long n = (long long)T * L.out;
#pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < n; ++idx) {
        const int t = (int)(idx / L.out), o = (int)(idx % L.out);
        const float* xr = x + (size_t)t * L.in;
        const float* wr = L.w + (size_t)o * L.in;
        float acc = L.b ? L.b[o] : 0.0f;
        for (int i = 0; i < L.in; ++i) acc += xr[i] * wr[i];
        y[(size_t)t * L.out + o] = acc;
    }
}

// RMSNorm over the last n values of each of the T rows: y = x / sqrt(mean(x^2) + eps) * w (the weight has n values)
void rms_norm(const float* x, int T, int n, float eps, const float* w, float* y) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t)t * n;
        float* yr = y + (size_t)t * n;
        double ss = 0.0;
        for (int c = 0; c < n; ++c) ss += (double)xr[c] * xr[c];
        const float inv = 1.0f / std::sqrt((float)(ss / n) + eps);
        for (int c = 0; c < n; ++c) yr[c] = xr[c] * inv * w[c];
    }
}

// The rotary of q [T][heads][D] and k [T][kv_heads][D] at the positions pos[t] (the standard rotate-half form)
void rope(float* x, int T, int heads, int D, const int* pos, double theta) {
    const int half = D / 2;
    std::vector<double> inv(half);
    for (int i = 0; i < half; ++i) inv[i] = std::pow(theta, -2.0 * i / D);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < heads; ++h) {
            float* v = x + ((size_t)t * heads + h) * D;
            for (int i = 0; i < half; ++i) {
                const double f = (double)pos[t] * inv[i];
                const float c = (float)std::cos(f), s = (float)std::sin(f);
                const float x1 = v[i], x2 = v[i + half];
                v[i] = x1 * c - x2 * s;
                v[i + half] = x2 * c + x1 * s;
            }
        }
}

struct Dims {
    int C, H, Hkv, D, FF;
    float eps;
    double theta;
};

// One pre-norm block over the n rows x [n][C] (in place) at the absolute positions pos0 .. pos0 + n - 1: their keys and
// values go to the cache rows kc / vc (indexed by absolute position: [pos0 + n][Hkv * D]), and each row attends
// causally to the cache up to its own position (GQA), then the gated MLP
void block_step(const Block& B, const Dims& d, float* x, int n, int pos0, float* kc, float* vc) {
    const int C = d.C, HD = d.H * d.D, KD = d.Hkv * d.D;
    std::vector<float> h((size_t)n * C), q((size_t)n * HD), k((size_t)n * KD), v((size_t)n * KD), a((size_t)n * HD),
        o((size_t)n * C), g((size_t)n * d.FF), u((size_t)n * d.FF);
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = pos0 + i;
    rms_norm(x, n, C, d.eps, B.attn_norm, h.data());
    linear(h.data(), n, B.q, q.data());
    linear(h.data(), n, B.k, k.data());
    linear(h.data(), n, B.v, v.data());
    rms_norm(q.data(), n * d.H, d.D, d.eps, B.q_norm, q.data());  // per head
    rms_norm(k.data(), n * d.Hkv, d.D, d.eps, B.k_norm, k.data());
    rope(q.data(), n, d.H, d.D, pos.data(), d.theta);
    rope(k.data(), n, d.Hkv, d.D, pos.data(), d.theta);
    std::memcpy(kc + (size_t)pos0 * KD, k.data(), sizeof(float) * (size_t)n * KD);
    std::memcpy(vc + (size_t)pos0 * KD, v.data(), sizeof(float) * (size_t)n * KD);
    // causal attention: row i (position p) sees the cache's positions 0..p; query head hq uses kv head hq / (H / Hkv)
    const int rep = d.H / d.Hkv;
    const float scale = 1.0f / std::sqrt((float)d.D);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int p = pos0 + i;
        std::vector<float> sc(p + 1);
        for (int hq = 0; hq < d.H; ++hq) {
            const float* qr = q.data() + ((size_t)i * d.H + hq) * d.D;
            const int hk = hq / rep;
            float mx = -INFINITY;
            for (int j = 0; j <= p; ++j) {
                const float* kr = kc + ((size_t)j * d.Hkv + hk) * d.D;
                float dot = 0.0f;
                for (int c = 0; c < d.D; ++c) dot += qr[c] * kr[c];
                sc[j] = dot * scale;
                mx = std::max(mx, sc[j]);
            }
            double sum = 0.0;
            for (int j = 0; j <= p; ++j) {
                sc[j] = std::exp(sc[j] - mx);
                sum += sc[j];
            }
            float* orow = a.data() + ((size_t)i * d.H + hq) * d.D;
            for (int c = 0; c < d.D; ++c) orow[c] = 0.0f;
            for (int j = 0; j <= p; ++j) {
                const float pr = (float)(sc[j] / sum);
                const float* vr = vc + ((size_t)j * d.Hkv + hk) * d.D;
                for (int c = 0; c < d.D; ++c) orow[c] += pr * vr[c];
            }
        }
    }
    linear(a.data(), n, B.o, o.data());
    for (size_t i = 0; i < (size_t)n * C; ++i) x[i] += o[i];
    rms_norm(x, n, C, d.eps, B.ffn_norm, h.data());
    linear(h.data(), n, B.gate, g.data());
    linear(h.data(), n, B.up, u.data());
    for (size_t i = 0; i < (size_t)n * d.FF; ++i) {
        const float gi = g[i];
        g[i] = gi / (1.0f + std::exp(-gi)) * u[i];  // SiLU(gate) * up
    }
    linear(g.data(), n, B.down, o.data());
    for (size_t i = 0; i < (size_t)n * C; ++i) x[i] += o[i];
}

}  // namespace

struct qwen3tts_model {
    gguf_model_t* g = nullptr;
    int n_layer = 0, C = 0, H = 0, Hkv = 0, D = 0, FF = 0, text_vocab = 0, text_C = 0, codec_vocab = 0, ncb = 0;
    float eps = 1e-6f;
    double theta = 1e6;
    int pn_layer = 0, pC = 0, pH = 0, pHkv = 0, pD = 0, pFF = 0, pvocab = 0;
    int tts_bos = 0, tts_eos = 0, tts_pad = 0, codec_pad = 0, codec_bos = 0, codec_eos = 0, codec_think = 0,
        codec_nothink = 0, codec_think_bos = 0, codec_think_eos = 0;
    std::vector<std::string> languages;
    std::vector<int> language_ids;
    const float* text_embd = nullptr;   // [text_vocab][text_C]
    const float* codec_embd = nullptr;  // [codec_vocab][C]
    Lin fc1, fc2;                       // the text projection: text_C -> text_C -> C (SiLU between)
    std::vector<Block> blk;
    const float* out_norm = nullptr;
    Lin codec_head;                     // C -> codec_vocab, no bias
    std::vector<const float*> pred_embd;  // ncb - 1 tables [pvocab][C]
    std::vector<Block> pblk;
    const float* pred_out_norm = nullptr;
    std::vector<Lin> pred_head;         // ncb - 1 heads [pvocab][pC]
    ~qwen3tts_model() {
        if (g) gguf_mmap_close(g);
    }
};

namespace {

Dims talker_dims(const qwen3tts_model* m) { return Dims{m->C, m->H, m->Hkv, m->D, m->FF, m->eps, m->theta}; }
Dims pred_dims(const qwen3tts_model* m) { return Dims{m->pC, m->pH, m->pHkv, m->pD, m->pFF, m->eps, m->theta}; }

// The text projection of T text rows (text_C each) into C values: fc2(SiLU(fc1(x)))
void text_project(const qwen3tts_model* m, const float* x, int T, float* y) {
    std::vector<float> h((size_t)T * m->text_C);
    linear(x, T, m->fc1, h.data());
    for (float& v : h) v = v / (1.0f + std::exp(-v));
    linear(h.data(), T, m->fc2, y);
}

// The rows of a text table for the ids: [n][text_C] -> projected [n][C]
void text_rows_projected(const qwen3tts_model* m, const std::vector<int>& ids, float* out) {
    const int n = (int)ids.size();
    std::vector<float> raw((size_t)n * m->text_C);
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= m->text_vocab) throw std::runtime_error("qwen3tts: text id out of range");
        std::memcpy(&raw[(size_t)i * m->text_C], m->text_embd + (size_t)ids[i] * m->text_C, sizeof(float) * m->text_C);
    }
    text_project(m, raw.data(), n, out);
}

void codec_rows(const qwen3tts_model* m, const std::vector<int>& ids, float* out) {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= m->codec_vocab) throw std::runtime_error("qwen3tts: codec id out of range");
        std::memcpy(out + i * m->C, m->codec_embd + (size_t)ids[i] * m->C, sizeof(float) * m->C);
    }
}

}  // namespace

extern "C" {

qwen3tts_model_t* qwen3tts_open(const char* gguf_path) {
    try {
        std::unique_ptr<qwen3tts_model> m(new qwen3tts_model());
        m->g = gguf_mmap_open(gguf_path);
        if (!m->g) {
            g_error = std::string("cannot open ") + (gguf_path ? gguf_path : "(null)");
            return nullptr;
        }
        if (std::string(gguf_arch(m->g)) != "qwen3tts") throw std::runtime_error("not a qwen3tts GGUF");
        // (the qwen3tts.* keys by their suffix: gguf_arch_int reads "<architecture>.<suffix>")
        auto kv = [&](const char* key) { return (int)gguf_arch_int(m->g, key, -1); };
        m->n_layer = kv("talker.n_layer");
        m->C = kv("talker.n_embd");
        m->H = kv("talker.n_head");
        m->Hkv = kv("talker.n_head_kv");
        m->D = kv("talker.head_dim");
        m->FF = kv("talker.n_ff");
        m->theta = gguf_arch_float(m->g, "talker.rope_theta", 1e6f);
        m->eps = gguf_arch_float(m->g, "talker.rms_eps", 1e-6f);
        m->text_vocab = kv("talker.text_vocab");
        m->text_C = kv("talker.text_n_embd");
        m->codec_vocab = kv("talker.codec_vocab");
        m->ncb = kv("n_codebooks");
        m->pn_layer = kv("pred.n_layer");
        m->pC = kv("pred.n_embd");
        m->pH = kv("pred.n_head");
        m->pHkv = kv("pred.n_head_kv");
        m->pD = kv("pred.head_dim");
        m->pFF = kv("pred.n_ff");
        m->pvocab = kv("pred.vocab");
        m->tts_bos = kv("tts_bos_token");
        m->tts_eos = kv("tts_eos_token");
        m->tts_pad = kv("tts_pad_token");
        m->codec_pad = kv("codec_pad_id");
        m->codec_bos = kv("codec_bos_id");
        m->codec_eos = kv("codec_eos_id");
        m->codec_think = kv("codec_think_id");
        m->codec_nothink = kv("codec_nothink_id");
        m->codec_think_bos = kv("codec_think_bos_id");
        m->codec_think_eos = kv("codec_think_eos_id");
        if (m->n_layer <= 0 || m->C <= 0 || m->H <= 0 || m->Hkv <= 0 || m->D <= 0 || m->FF <= 0 || m->text_vocab <= 0 ||
            m->text_C <= 0 || m->codec_vocab <= 0 || m->ncb <= 1 || m->pn_layer <= 0 || m->pC <= 0 || m->pH <= 0 ||
            m->pHkv <= 0 || m->pD <= 0 || m->pFF <= 0 || m->pvocab <= 0 || m->H % m->Hkv != 0 || m->pH % m->pHkv != 0)
            throw std::runtime_error("qwen3tts: missing or bad hyperparameters");
        if (m->pC != m->C) throw std::runtime_error("qwen3tts: the code predictor's width differs from the talker's");
        // the languages (the names and their codec ids)
        {
            const char** names = nullptr;
            size_t n = 0;
            if (gguf_kv_get_string_array(m->g, "qwen3tts.languages", &names, &n) == 0)
                for (size_t i = 0; i < n; ++i) m->languages.push_back(names[i]);
            int32_t type = 0;
            const void* data = nullptr;
            size_t count = 0;
            if (gguf_kv_get_array(m->g, "qwen3tts.language_ids", &type, &data, &count) == 0 && count == m->languages.size())
                for (size_t i = 0; i < count; ++i) m->language_ids.push_back((int)static_cast<const uint32_t*>(data)[i]);
        }

        m->text_embd = tensor(m->g, "talker.text_embd.weight", (size_t)m->text_vocab * m->text_C);
        m->codec_embd = tensor(m->g, "talker.codec_embd.weight", (size_t)m->codec_vocab * m->C);
        m->fc1 = {tensor(m->g, "talker.text_proj.fc1.weight", (size_t)m->text_C * m->text_C),
                  tensor(m->g, "talker.text_proj.fc1.bias", m->text_C), m->text_C, m->text_C};
        m->fc2 = {tensor(m->g, "talker.text_proj.fc2.weight", (size_t)m->C * m->text_C),
                  tensor(m->g, "talker.text_proj.fc2.bias", m->C), m->text_C, m->C};
        m->out_norm = tensor(m->g, "talker.out_norm.weight", m->C);
        m->codec_head = {tensor(m->g, "talker.codec_head.weight", (size_t)m->codec_vocab * m->C), nullptr, m->C, m->codec_vocab};
        auto lin = [&](const std::string& base, int in, int out, bool bias) {
            Lin L;
            L.in = in;
            L.out = out;
            L.w = tensor(m->g, base + ".weight", (size_t)in * out);
            L.b = bias ? tensor(m->g, base + ".bias", out) : nullptr;
            return L;
        };
        auto block_of = [&](const std::string& p, int C, int H, int Hkv, int D, int FF) {
            Block B;
            B.attn_norm = tensor(m->g, p + "attn_norm.weight", C);
            B.ffn_norm = tensor(m->g, p + "ffn_norm.weight", C);
            B.q_norm = tensor(m->g, p + "attn_q_norm.weight", D);
            B.k_norm = tensor(m->g, p + "attn_k_norm.weight", D);
            B.q = lin(p + "attn_q", C, H * D, false);
            B.k = lin(p + "attn_k", C, Hkv * D, false);
            B.v = lin(p + "attn_v", C, Hkv * D, false);
            B.o = lin(p + "attn_o", H * D, C, false);
            B.gate = lin(p + "ffn_gate", C, FF, false);
            B.up = lin(p + "ffn_up", C, FF, false);
            B.down = lin(p + "ffn_down", FF, C, false);
            return B;
        };
        for (int l = 0; l < m->n_layer; ++l) m->blk.push_back(block_of("talker.blk." + std::to_string(l) + ".", m->C, m->H, m->Hkv, m->D, m->FF));
        for (int i = 0; i < m->ncb - 1; ++i) {
            m->pred_embd.push_back(tensor(m->g, "pred.embd." + std::to_string(i) + ".weight", (size_t)m->pvocab * m->C));
            m->pred_head.push_back(lin("pred.head." + std::to_string(i), m->pC, m->pvocab, false));
        }
        for (int l = 0; l < m->pn_layer; ++l)
            m->pblk.push_back(block_of("pred.blk." + std::to_string(l) + ".", m->pC, m->pH, m->pHkv, m->pD, m->pFF));
        m->pred_out_norm = tensor(m->g, "pred.out_norm.weight", m->pC);
        return m.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void qwen3tts_close(qwen3tts_model_t* model) { delete model; }

const char* qwen3tts_last_error(void) { return g_error.c_str(); }

int qwen3tts_n_embd(const qwen3tts_model_t* m) { return m ? m->C : 0; }
int qwen3tts_n_codebooks(const qwen3tts_model_t* m) { return m ? m->ncb : 0; }
int qwen3tts_codec_vocab(const qwen3tts_model_t* m) { return m ? m->codec_vocab : 0; }
int qwen3tts_pred_vocab(const qwen3tts_model_t* m) { return m ? m->pvocab : 0; }

int qwen3tts_language_id(const qwen3tts_model_t* m, const char* name) {
    if (!m || !name) return -1;
    std::string lower;
    for (const char* p = name; *p; ++p) lower += (char)std::tolower((unsigned char)*p);
    for (size_t i = 0; i < m->languages.size(); ++i)
        if (m->languages[i] == lower) return m->language_ids[i];
    return -1;
}

int qwen3tts_build_prefill(qwen3tts_model_t* m, const int* text_ids, int n_text, const int* ref_ids, int n_ref,
                           const int* ref_codes, int R, const float* spk, int language_id, float* out, int max_positions,
                           int* positions, float* trailing, int max_trailing, int* n_trailing, float* pad) {
    if (!m || !text_ids || !ref_ids || !ref_codes || !spk || !out || !positions || !trailing || !n_trailing || !pad) {
        g_error = "qwen3tts_build_prefill: null argument";
        return -1;
    }
    if (n_text < 9 || n_ref < 8 || R < 1) { g_error = "qwen3tts_build_prefill: the text, the reference or its frames are too short"; return -1; }
    try {
        const int C = m->C;
        // the assistant template's parts: the role (the first three ids), the text (after them, before the last five)
        std::vector<int> role(text_ids, text_ids + 3);
        std::vector<int> text(text_ids + 3, text_ids + n_text - 5);
        std::vector<int> ref_text(ref_ids + 3, ref_ids + n_ref - 2);
        // the TTS special embeddings: bos, eos, pad (projected)
        std::vector<int> specials = {m->tts_bos, m->tts_eos, m->tts_pad};
        std::vector<float> sp((size_t)3 * C);
        text_rows_projected(m, specials, sp.data());
        const float* tts_bos = sp.data();
        const float* tts_eos = sp.data() + C;
        const float* tts_pad = sp.data() + 2 * C;
        // the codec prefix: the language (or no-think) tags, the speaker, the bos
        std::vector<int> cprefix = {m->codec_think, m->codec_think_bos, language_id, m->codec_think_eos};
        std::vector<float> codec_in((size_t)7 * C);  // 4 tags + speaker + pad, bos
        codec_rows(m, cprefix, codec_in.data());
        std::memcpy(&codec_in[(size_t)4 * C], spk, sizeof(float) * C);
        std::vector<int> cpad = {m->codec_pad, m->codec_bos};
        codec_rows(m, cpad, &codec_in[(size_t)5 * C]);
        // the talker's input for the role and the codec prefix: role (3), then pad * 5 and bos, plus codec_in[:6]
        std::vector<float> role_e((size_t)3 * C);
        text_rows_projected(m, role, role_e.data());
        std::vector<float> talker_pre((size_t)9 * C);
        std::memcpy(talker_pre.data(), role_e.data(), sizeof(float) * 3 * C);
        for (int j = 0; j < 6; ++j) {
            const float* add = j < 5 ? tts_pad : tts_bos;
            for (int c = 0; c < C; ++c) talker_pre[(size_t)(3 + j) * C + c] = add[c] + codec_in[(size_t)j * C + c];
        }
        // the in-context reference: the reference text and the text with the eos, the reference's codec frames with the bos
        std::vector<int> both = ref_text;
        both.insert(both.end(), text.begin(), text.end());
        std::vector<float> tproj((size_t)(both.size() + 1) * C);
        text_rows_projected(m, both, tproj.data());
        std::memcpy(&tproj[both.size() * C], tts_eos, sizeof(float) * C);
        const int text_len = (int)both.size() + 1;
        // codec: the bos, then each frame's sum over its 16 codebooks
        const int codec_len = R + 1;
        std::vector<float> codec((size_t)codec_len * C, 0.0f);
        {
            std::vector<int> bos = {m->codec_bos};
            codec_rows(m, bos, codec.data());
            for (int r = 0; r < R; ++r) {
                float* dst = &codec[(size_t)(r + 1) * C];
                for (int gi = 0; gi < m->ncb; ++gi) {
                    const int id = ref_codes[(size_t)r * m->ncb + gi];
                    if (id < 0) throw std::runtime_error("qwen3tts: negative code");
                    const float* table = gi == 0 ? m->codec_embd : m->pred_embd[gi - 1];
                    if (id >= (gi == 0 ? m->codec_vocab : m->pvocab)) throw std::runtime_error("qwen3tts: code out of range");
                    const float* row = table + (size_t)id * C;
                    for (int c = 0; c < C; ++c) dst[c] += row[c];
                }
            }
        }
        // the ICL block: text and codec frames side by side (the shorter padded with the pad), the text left over trailing
        std::vector<float> icl;
        std::vector<float> trail;
        if (text_len > codec_len) {
            icl.resize((size_t)codec_len * C);
            for (size_t i = 0; i < icl.size(); ++i) icl[i] = tproj[i] + codec[i];
            trail.assign(tproj.begin() + (size_t)codec_len * C, tproj.begin() + (size_t)text_len * C);
        } else {
            icl.resize((size_t)codec_len * C);
            for (int i = 0; i < codec_len; ++i)
                for (int c = 0; c < C; ++c) {
                    const float tv = i < text_len ? tproj[(size_t)i * C + c] : tts_pad[c];
                    icl[(size_t)i * C + c] = tv + codec[(size_t)i * C + c];
                }
            trail.assign(tts_pad, tts_pad + C);
        }
        const int P = 9 + codec_len;
        if (P > max_positions) { g_error = "qwen3tts_build_prefill: out is too small"; return -1; }
        std::memcpy(out, talker_pre.data(), sizeof(float) * 9 * C);
        std::memcpy(out + (size_t)9 * C, icl.data(), sizeof(float) * icl.size());
        *positions = P;
        const int nt = (int)trail.size() / C;
        if (nt > max_trailing) { g_error = "qwen3tts_build_prefill: trailing is too small"; return -1; }
        std::memcpy(trailing, trail.data(), sizeof(float) * trail.size());
        *n_trailing = nt;
        std::memcpy(pad, tts_pad, sizeof(float) * C);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int qwen3tts_frame_input(qwen3tts_model_t* m, const int* codes, const float* trailing, float* out) {
    if (!m || !codes || !out) { g_error = "qwen3tts_frame_input: null argument"; return -1; }
    const int C = m->C;
    std::fill(out, out + C, 0.0f);
    for (int gi = 0; gi < m->ncb; ++gi) {
        const int id = codes[gi];
        const float* table = gi == 0 ? m->codec_embd : m->pred_embd[gi - 1];
        if (id < 0 || id >= (gi == 0 ? m->codec_vocab : m->pvocab)) { g_error = "qwen3tts_frame_input: code out of range"; return -1; }
        const float* row = table + (size_t)id * C;
        for (int c = 0; c < C; ++c) out[c] += row[c];
    }
    if (trailing)
        for (int c = 0; c < C; ++c) out[c] += trailing[c];
    return 0;
}

int qwen3tts_talker_forward(qwen3tts_model_t* m, const float* X, int L, float* hidden, float* logits) {
    if (!m || !X || L <= 0 || !hidden || !logits) { g_error = "qwen3tts_talker_forward: bad arguments"; return -1; }
    try {
        const int C = m->C;
        std::vector<float> x(X, X + (size_t)L * C);
        const Dims d = talker_dims(m);
        std::vector<float> kc((size_t)L * d.Hkv * d.D), vc((size_t)L * d.Hkv * d.D);
        for (const Block& B : m->blk) block_step(B, d, x.data(), L, 0, kc.data(), vc.data());
        rms_norm(x.data(), L, C, m->eps, m->out_norm, hidden);
        linear(hidden, L, m->codec_head, logits);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int qwen3tts_code_predictor(qwen3tts_model_t* m, const float* H, const int* codes, int T, float* logits) {
    if (!m || !H || !codes || T <= 0 || !logits) { g_error = "qwen3tts_code_predictor: bad arguments"; return -1; }
    try {
        const int C = m->C, ncb = m->ncb;
        const int rows = ncb;  // H, the talker's embedding of c0, then the predictor's of c1 .. c_{ncb-2} (c_{ncb-1} is never an input)
        const Dims d = pred_dims(m);
        std::vector<float> x((size_t)rows * C);
        for (int t = 0; t < T; ++t) {
            // the inputs of frame t: [H_t, E0(c0), E1(c1) .. E_{ncb-2}(c_{ncb-2})]; the talker's table for c0, the code
            // predictor's table j for codebook j + 1
            std::memcpy(x.data(), H + (size_t)t * C, sizeof(float) * C);
            const int* c = codes + (size_t)t * ncb;
            if (c[0] < 0 || c[0] >= m->codec_vocab) { g_error = "qwen3tts_code_predictor: code out of range"; return -1; }
            std::memcpy(x.data() + C, m->codec_embd + (size_t)c[0] * C, sizeof(float) * C);
            for (int j = 0; j + 2 < rows; ++j) {
                const int id = c[j + 1];
                if (id < 0 || id >= m->pvocab) { g_error = "qwen3tts_code_predictor: code out of range"; return -1; }
                std::memcpy(x.data() + (size_t)(j + 2) * C, m->pred_embd[j] + (size_t)id * C, sizeof(float) * C);
            }
            std::vector<float> kc((size_t)rows * d.Hkv * d.D), vc((size_t)rows * d.Hkv * d.D);
            for (const Block& B : m->pblk) block_step(B, d, x.data(), rows, 0, kc.data(), vc.data());
            std::vector<float> hn((size_t)rows * C);
            rms_norm(x.data(), rows, C, m->eps, m->pred_out_norm, hn.data());
            // codebook k + 1 from the position k + 1 through head k
            for (int k = 0; k < ncb - 1; ++k) {
                const float* hr = hn.data() + (size_t)(k + 1) * C;
                float* lr = logits + ((size_t)t * (ncb - 1) + k) * m->pvocab;
                const Lin& hd = m->pred_head[k];
                for (int o = 0; o < m->pvocab; ++o) {
                    const float* wr = hd.w + (size_t)o * C;
                    float acc = 0.0f;
                    for (int i = 0; i < C; ++i) acc += hr[i] * wr[i];
                    lr[o] = acc;
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int qwen3tts_generate_greedy(qwen3tts_model_t* m, const int* text_ids, int n_text, const int* ref_ids, int n_ref,
                             const int* ref_codes, int R, const float* spk, int language_id, float repetition_penalty,
                             int max_frames, int* out_codes) {
    if (!m || !text_ids || !ref_ids || !ref_codes || !spk || !out_codes || max_frames <= 0 || repetition_penalty <= 0.0f) {
        g_error = "qwen3tts_generate_greedy: bad arguments";
        return -1;
    }
    try {
        const int C = m->C, V = m->codec_vocab, ncb = m->ncb;
        const Dims d = talker_dims(m);
        const int KD = d.Hkv * d.D;
        const int maxP = 128;
        std::vector<float> prefill((size_t)maxP * C), trailing(8 * (size_t)C), pad(C);
        int P = 0, nt = 0;
        if (qwen3tts_build_prefill(m, text_ids, n_text, ref_ids, n_ref, ref_codes, R, spk, language_id, prefill.data(), maxP,
                                   &P, trailing.data(), 8, &nt, pad.data()) != 0)
            return -1;
        const int maxL = P + max_frames + 1;
        std::vector<std::vector<float>> kc(m->n_layer, std::vector<float>((size_t)maxL * KD)), vc(m->n_layer, std::vector<float>((size_t)maxL * KD));
        // the prefill
        std::vector<float> x(prefill.begin(), prefill.begin() + (size_t)P * C);
        for (int l = 0; l < m->n_layer; ++l) block_step(m->blk[l], d, x.data(), P, 0, kc[l].data(), vc[l].data());
        std::vector<float> hid(C), logits(V), xrow(C);
        auto head = [&](const float* last) {
            rms_norm(last, 1, C, m->eps, m->out_norm, hid.data());
            linear(hid.data(), 1, m->codec_head, logits.data());
        };
        head(x.data() + (size_t)(P - 1) * C);
        int pos = P;
        std::vector<int> history;
        const int eos = m->codec_eos;
        int n_out = 0;
        std::vector<int> frame(ncb);
        for (int t = 0; t < max_frames; ++t) {
            // the talker's logits, processed as qwen-tts does (repetition penalty, the suppressed range, the minimum)
            std::vector<float> lg(logits);
            // (once per distinct token, as the transformers' processor: its scatter over the unique ids)
            std::vector<int> seen(history);
            std::sort(seen.begin(), seen.end());
            seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
            for (int h : seen) {
                float& l = lg[h];
                l = l < 0.0f ? l * repetition_penalty : l / repetition_penalty;
            }
            for (int v = V - 1024; v < V; ++v)
                if (v != eos) lg[v] = -INFINITY;
            if (t < 2) lg[eos] = -INFINITY;
            int c0 = 0;
            for (int v = 1; v < V; ++v)
                if (lg[v] > lg[c0]) c0 = v;
            if (c0 == eos) break;
            history.push_back(c0);
            // the code predictor, greedy, with its own cache: the rows H, E(c0), then each new codebook's embedding; the row
            // k's output gives codebook k through head k - 1 (causal: one new row per codebook)
            std::fill(frame.begin(), frame.end(), 0);
            frame[0] = c0;
            {
                const int pKD = m->pHkv * m->pD;
                const Dims pd = pred_dims(m);
                std::vector<std::vector<float>> pk(m->pn_layer, std::vector<float>((size_t)(ncb + 1) * pKD)),
                    pv(m->pn_layer, std::vector<float>((size_t)(ncb + 1) * pKD));
                std::vector<float> px((size_t)2 * C), pn((size_t)C);
                std::memcpy(px.data(), hid.data(), sizeof(float) * C);
                std::memcpy(px.data() + C, m->codec_embd + (size_t)c0 * C, sizeof(float) * C);
                for (int l = 0; l < m->pn_layer; ++l) block_step(m->pblk[l], pd, px.data(), 2, 0, pk[l].data(), pv[l].data());
                std::vector<float> row(C);
                for (int k = 1; k < ncb; ++k) {
                    // the output row k (the last one processed) predicts codebook k through head k - 1
                    const float* last = k == 1 ? px.data() + C : row.data();
                    rms_norm(last, 1, C, m->eps, m->pred_out_norm, pn.data());
                    const Lin& hd = m->pred_head[k - 1];
                    int best = 0;
                    float bv = -INFINITY;
                    for (int v = 0; v < m->pvocab; ++v) {
                        const float* wr = hd.w + (size_t)v * C;
                        float acc = 0.0f;
                        for (int i = 0; i < C; ++i) acc += pn[i] * wr[i];
                        if (acc > bv) { bv = acc; best = v; }
                    }
                    frame[k] = best;
                    if (k + 1 < ncb) {
                        // the next row: codebook k's embedding in the table k - 1, at position k + 1
                        std::memcpy(row.data(), m->pred_embd[k - 1] + (size_t)best * C, sizeof(float) * C);
                        for (int l = 0; l < m->pn_layer; ++l) block_step(m->pblk[l], pd, row.data(), 1, k + 1, pk[l].data(), pv[l].data());
                    }
                }
            }
            std::memcpy(out_codes + (size_t)n_out * ncb, frame.data(), sizeof(int) * ncb);
            ++n_out;
            if (t + 1 == max_frames) break;
            // the next input: the frame's sum plus the trailing text (the pad after the trailing rows)
            const float* tr = t < nt ? &trailing[(size_t)t * C] : pad.data();
            if (qwen3tts_frame_input(m, frame.data(), tr, xrow.data()) != 0) return -1;
            for (int l = 0; l < m->n_layer; ++l) block_step(m->blk[l], d, xrow.data(), 1, pos, kc[l].data(), vc[l].data());
            ++pos;
            head(xrow.data());
        }
        return n_out;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

}  // extern "C"
