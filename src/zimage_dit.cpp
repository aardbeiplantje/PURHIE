/* Z-Image-Turbo's diffusion transformer, host path: see zimage_dit.h. The formulas are the reference's
 * (diffusers' ZImageTransformer2DModel, the timestep embedder, the RMS norm, the patchify and the padding). */
#include "zimage_dit.h"
#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace zimage {

namespace {

/* a minimal reader for the safetensors header: an object of {name: {dtype, shape, data_offsets}} and __metadata__ */
struct Json {
    const char* p;
    const char* end;
    bool ok = true;

    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    bool eat(char c) { ws(); if (p < end && *p == c) { ++p; return true; } return false; }
    std::string str() {
        std::string s;
        if (!eat('"')) { ok = false; return s; }
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) { ++p; }
            s.push_back(*p++);
        }
        if (p < end) ++p; else ok = false;
        return s;
    }
    double num() {
        ws();
        char* e = nullptr;
        double v = strtod(p, &e);
        if (e == p) { ok = false; return 0; }
        p = e;
        return v;
    }
    void skip() {  /* any value */
        ws();
        if (p >= end) { ok = false; return; }
        if (*p == '"') { str(); return; }
        if (*p == '{' || *p == '[') {
            char close = *p == '{' ? '}' : ']';
            ++p;
            ws();
            if (eat(close)) return;
            while (ok) {
                if (close == '}') { str(); eat(':'); }
                skip();
                if (eat(close)) return;
                if (!eat(',')) { ok = false; return; }
            }
            return;
        }
        if (strncmp(p, "true", 4) == 0) { p += 4; return; }
        if (strncmp(p, "false", 5) == 0) { p += 5; return; }
        if (strncmp(p, "null", 4) == 0) { p += 4; return; }
        num();
    }
};

inline float bf16_to_f32(uint16_t v) {
    const uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

SafeTensors::SafeTensors(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { error_ = path + ": cannot open"; return; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 8) { close(fd); error_ = path + ": too short"; return; }
    size_ = (size_t)st.st_size;
    void* m = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { error_ = path + ": cannot map"; return; }
    base_ = (const uint8_t*)m;
    uint64_t n = 0;
    memcpy(&n, base_, 8);
    if (n == 0 || n > size_ - 8) { error_ = path + ": bad header length"; return; }
    data_ = 8 + (size_t)n;
    Json j{(const char*)base_ + 8, (const char*)base_ + data_};
    if (!j.eat('{')) { error_ = path + ": header is not an object"; return; }
    while (j.ok && !j.eat('}')) {
        const std::string name = j.str();
        j.eat(':');
        if (name == "__metadata__") { j.skip(); }
        else {
            Entry e;
            if (!j.eat('{')) { j.ok = false; break; }
            while (j.ok && !j.eat('}')) {
                const std::string key = j.str();
                j.eat(':');
                if (key == "dtype") e.dtype = j.str();
                else if (key == "shape") {
                    if (!j.eat('[')) { j.ok = false; break; }
                    while (j.ok && !j.eat(']')) { e.shape.push_back((size_t)j.num()); j.eat(','); }
                } else if (key == "data_offsets") {
                    if (!j.eat('[')) { j.ok = false; break; }
                    e.begin = (size_t)j.num(); j.eat(',');
                    e.end = (size_t)j.num();
                    j.eat(']');
                } else j.skip();
                j.eat(',');
            }
            t_[name] = e;
        }
        j.eat(',');
    }
    if (!j.ok) { error_ = path + ": bad header"; munmap(m, size_); base_ = nullptr; t_.clear(); }
}

SafeTensors::~SafeTensors() {
    if (base_) munmap((void*)base_, size_);
}

bool SafeTensors::get(const std::string& name, std::vector<float>& out, std::vector<size_t>& shape) const {
    auto it = t_.find(name);
    if (it == t_.end()) return false;
    const Entry& e = it->second;
    const size_t n = e.end - e.begin;
    const uint8_t* src = base_ + data_ + e.begin;
    if (e.dtype == "BF16") {
        out.resize(n / 2);
        for (size_t i = 0; i < n / 2; ++i) { uint16_t v; memcpy(&v, src + 2 * i, 2); out[i] = bf16_to_f32(v); }
    } else if (e.dtype == "F32") {
        out.resize(n / 4);
        memcpy(out.data(), src, n);
    } else {
        return false;
    }
    shape = e.shape;
    return true;
}

bool SafeTensors::register_device() const {
    return base_ != nullptr && sddev::register_weights(base_, size_);
}

bool SafeTensors::bf16_in_place(const std::string& name, const uint16_t*& ptr, std::vector<size_t>& shape) const {
    auto it = t_.find(name);
    if (it == t_.end() || it->second.dtype != "BF16") return false;
    ptr = reinterpret_cast<const uint16_t*>(base_ + data_ + it->second.begin);
    shape = it->second.shape;
    return true;
}

bool SafeTensors::f32_in_place(const std::string& name, const float*& ptr, std::vector<size_t>& shape) const {
    auto it = t_.find(name);
    if (it == t_.end() || it->second.dtype != "F32") return false;
    ptr = reinterpret_cast<const float*>(base_ + data_ + it->second.begin);
    shape = it->second.shape;
    return true;
}

void linear(const float* x, int T, int in, const std::vector<float>& W, int out, const std::vector<float>& b, float* y) {
    for (int t = 0; t < T; ++t) {
        const float* xt = x + (size_t)t * in;
        float* yt = y + (size_t)t * out;
        for (int o = 0; o < out; ++o) {
            const float* w = W.data() + (size_t)o * in;
            double acc = b.empty() ? 0.0 : (double)b[o];
            for (int i = 0; i < in; ++i) acc += (double)xt[i] * (double)w[i];
            yt[o] = (float)acc;
        }
    }
}

void rms_norm(const float* x, int T, int C, const std::vector<float>& w, float eps, float* y) {
    for (int t = 0; t < T; ++t) {
        const float* xt = x + (size_t)t * C;
        double ss = 0;
        for (int i = 0; i < C; ++i) ss += (double)xt[i] * (double)xt[i];
        const double inv = 1.0 / std::sqrt(ss / C + eps);
        for (int i = 0; i < C; ++i) y[(size_t)t * C + i] = (float)(xt[i] * inv * w[i]);
    }
}

void timestep_embed(const SafeTensors& st, float t, float* out256) {
    /* the reference scales the step by t_scale (1000) before the embedding; frequencies over 128 pairs, max period 10000 */
    const double arg_scale = (double)t * 1000.0;
    float freq_emb[256];
    for (int i = 0; i < 128; ++i) {
        const double f = std::exp(-std::log(10000.0) * (double)i / 128.0);
        freq_emb[i] = (float)std::cos(arg_scale * f);
        freq_emb[128 + i] = (float)std::sin(arg_scale * f);
    }
    std::vector<float> w0, b0, w2, b2;
    std::vector<size_t> s;
    if (!st.get("t_embedder.mlp.0.weight", w0, s) || !st.get("t_embedder.mlp.0.bias", b0, s) ||
        !st.get("t_embedder.mlp.2.weight", w2, s) || !st.get("t_embedder.mlp.2.bias", b2, s)) {
        fprintf(stderr, "zimage: t_embedder weights missing\n");
        return;
    }
    std::vector<float> h(1024);
    linear(freq_emb, 1, 256, w0, 1024, b0, h.data());
    for (float& v : h) v = v / (1.0f + std::exp(-v));  /* SiLU */
    linear(h.data(), 1, 1024, w2, 256, b2, out256);
}

void patchify(const float* latent, int C, int H, int W, int p, std::vector<float>& tokens) {
    const int Ht = H / p, Wt = W / p;
    tokens.assign((size_t)Ht * Wt * p * p * C, 0.0f);
    for (int ht = 0; ht < Ht; ++ht)
        for (int wt = 0; wt < Wt; ++wt) {
            float* tok = tokens.data() + (size_t)(ht * Wt + wt) * p * p * C;
            for (int ph = 0; ph < p; ++ph)
                for (int pw = 0; pw < p; ++pw)
                    for (int c = 0; c < C; ++c)
                        tok[(ph * p + pw) * C + c] = latent[((size_t)c * H + ht * p + ph) * W + wt * p + pw];
        }
}

void caption_embed(const SafeTensors& st, const float* cap, int L, std::vector<float>& out, int& Lpad) {
    const int D = 2560, dim = 3840;
    Lpad = L + (32 - L % 32) % 32;
    std::vector<float> padded((size_t)Lpad * D);
    for (int t = 0; t < Lpad; ++t) memcpy(&padded[(size_t)t * D], cap + (size_t)(t < L ? t : L - 1) * D, D * sizeof(float));
    std::vector<float> nw, W, b;
    std::vector<size_t> s;
    if (!st.get("cap_embedder.0.weight", nw, s) || !st.get("cap_embedder.1.weight", W, s) ||
        !st.get("cap_embedder.1.bias", b, s)) {
        fprintf(stderr, "zimage: cap_embedder weights missing\n");
        out.clear();
        return;
    }
    std::vector<float> normed((size_t)Lpad * D);
    rms_norm(padded.data(), Lpad, D, nw, 1e-5f, normed.data());
    out.assign((size_t)Lpad * dim, 0.0f);
    linear(normed.data(), Lpad, D, W, dim, b, out.data());
}

void patch_embed(const SafeTensors& st, const std::vector<float>& tokens, int N, std::vector<float>& out) {
    const int in = 64, dim = 3840;
    std::vector<float> W, b;
    std::vector<size_t> s;
    if (!st.get("x_embedder.weight", W, s) || !st.get("x_embedder.bias", b, s)) {
        fprintf(stderr, "zimage: x_embedder weights missing\n");
        out.clear();
        return;
    }
    out.assign((size_t)N * dim, 0.0f);
    linear(tokens.data(), N, in, W, dim, b, out.data());
}


/* ---- device path ---- */
namespace {
/* a BF16 bias as floats in sddev::alloc memory (small: one row of the output) */
bool bias_f32(const SafeTensors& st, const std::string& name, int n, float*& out) {
    const uint16_t* p = nullptr;
    std::vector<size_t> s;
    if (!st.bf16_in_place(name, p, s) || (int)s.size() != 1 || (int)s[0] != n) {
        fprintf(stderr, "zimage: %s missing or not [%d]\n", name.c_str(), n);
        return false;
    }
    out = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)n, false));
    for (int i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
    return true;
}
bool weight_in_place(const SafeTensors& st, const std::string& name, int out_dim, int in_dim, const uint16_t*& w) {
    std::vector<size_t> s;
    if (!st.bf16_in_place(name, w, s) || s.size() != 2 || (int)s[0] != out_dim || (int)s[1] != in_dim) {
        fprintf(stderr, "zimage: %s missing or not [%d, %d]\n", name.c_str(), out_dim, in_dim);
        return false;
    }
    return true;
}
}  // namespace

bool dev_timestep_embed(const SafeTensors& st, float t, float* out256) {
    if (!sddev::enabled()) { fprintf(stderr, "zimage: no device\n"); return false; }
    const uint16_t *w0 = nullptr, *w2 = nullptr;
    float *b0 = nullptr, *b2 = nullptr;
    if (!weight_in_place(st, "t_embedder.mlp.0.weight", 1024, 256, w0) ||
        !weight_in_place(st, "t_embedder.mlp.2.weight", 256, 1024, w2) ||
        !bias_f32(st, "t_embedder.mlp.0.bias", 1024, b0) || !bias_f32(st, "t_embedder.mlp.2.bias", 256, b2))
        return false;
    /* the frequencies (t x 1000, cos then sin of 128 frequencies), written into shared memory */
    float* freq = static_cast<float*>(sddev::alloc(sizeof(float) * 256, false));
    const double arg_scale = (double)t * 1000.0;
    for (int i = 0; i < 128; ++i) {
        const double f = std::exp(-std::log(10000.0) * (double)i / 128.0);
        freq[i] = (float)std::cos(arg_scale * f);
        freq[128 + i] = (float)std::sin(arg_scale * f);
    }
    float* h = static_cast<float*>(sddev::alloc(sizeof(float) * 1024, false));
    sddev::linear_bf16(freq, 1, w0, 256, 1024, b0, h);
    sddev::silu(h, 1024);
    sddev::linear_bf16(h, 1, w2, 1024, 256, b2, out256);
    return true;
}

bool dev_patch_embed(const SafeTensors& st, const float* latent, int C, int H, int W, float** out, int& N) {
    if (!sddev::enabled()) { fprintf(stderr, "zimage: no device\n"); return false; }
    const uint16_t* w = nullptr;
    float* b = nullptr;
    const int in = 2 * 2 * C, dim = 3840;
    if (!weight_in_place(st, "x_embedder.weight", dim, in, w) || !bias_f32(st, "x_embedder.bias", dim, b)) return false;
    N = (H / 2) * (W / 2);
    /* the patches written straight into shared memory (the GEMM reads them there) */
    float* tokens = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)N * in, false));
    for (int ht = 0; ht < H / 2; ++ht)
        for (int wt = 0; wt < W / 2; ++wt) {
            float* tok = tokens + (size_t)(ht * (W / 2) + wt) * in;
            for (int ph = 0; ph < 2; ++ph)
                for (int pw = 0; pw < 2; ++pw)
                    for (int c = 0; c < C; ++c)
                        tok[(ph * 2 + pw) * C + c] = latent[((size_t)c * H + ht * 2 + ph) * W + wt * 2 + pw];
        }
    if (!*out) *out = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)N * dim, false));  /* (or the caller's rows) */
    sddev::linear_bf16(tokens, N, w, in, dim, b, *out);
    return true;
}


bool dev_caption_embed(const SafeTensors& st, const float* cap, int L, float** out, int& Lpad) {
    if (!sddev::enabled()) { fprintf(stderr, "zimage: no device\n"); return false; }
    const int D = 2560, dim = 3840;
    const uint16_t *wn = nullptr, *w = nullptr;
    float* b = nullptr;
    std::vector<size_t> s;
    if (!st.bf16_in_place("cap_embedder.0.weight", wn, s) || s.size() != 1 || (int)s[0] != D ||
        !weight_in_place(st, "cap_embedder.1.weight", dim, D, w) || !bias_f32(st, "cap_embedder.1.bias", dim, b))
        return false;
    Lpad = L + (32 - L % 32) % 32;
    /* the caption rows padded to a multiple of 32 by repeating the last row, written into shared memory */
    float* padded = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)Lpad * D, false));
    for (int t = 0; t < Lpad; ++t) memcpy(padded + (size_t)t * D, cap + (size_t)(t < L ? t : L - 1) * D, sizeof(float) * D);
    float* normed = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)Lpad * D, false));
    sddev::rms_norm(padded, Lpad, D, wn, 1e-5f, normed);
    if (!*out) *out = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)Lpad * dim, false));  /* (or the caller's rows) */
    sddev::linear_bf16(normed, Lpad, w, D, dim, b, *out);
    return true;
}


/* ---- one noise refiner block on the device ---- */
namespace {
/* a device buffer of count floats, released at its end */
struct DevBuf {
    float* p = nullptr;
    size_t n = 0;
    explicit DevBuf(size_t count, bool zero = false) : n(count) { p = static_cast<float*>(sddev::alloc(sizeof(float) * count, zero)); }
    ~DevBuf() { if (p) sddev::release(p, sizeof(float) * n); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};
bool named(const SafeTensors& st, const std::string& name, std::vector<size_t> shape, const uint16_t*& ptr) {
    std::vector<size_t> s;
    if (!st.bf16_in_place(name, ptr, s) || s != shape) {
        fprintf(stderr, "zimage: %s missing or of the wrong shape\n", name.c_str());
        return false;
    }
    return true;
}
}  // namespace

void rope_table_ids(const int* ids, int N, float* fr) {
    const int dims[3] = {32, 48, 48};
    for (int n = 0; n < N; ++n) {
        float* row = fr + (size_t)n * 64 * 2;
        int p = 0;
        for (int ax = 0; ax < 3; ++ax)
            for (int k = 0; k < dims[ax] / 2; ++k, ++p) {
                const double f = 1.0 / std::pow(256.0, (double)(2 * k) / (double)dims[ax]);
                const double ang = (double)ids[n * 3 + ax] * f;
                row[p * 2] = (float)std::cos(ang);
                row[p * 2 + 1] = (float)std::sin(ang);
            }
    }
}

void rope_table_image(int Ht, int Wt, int start, float* fr) {
    std::vector<int> ids((size_t)Ht * Wt * 3);
    for (int ht = 0; ht < Ht; ++ht)
        for (int wt = 0; wt < Wt; ++wt) {
            int* t = ids.data() + (size_t)(ht * Wt + wt) * 3;
            t[0] = start; t[1] = ht; t[2] = wt;
        }
    rope_table_ids(ids.data(), Ht * Wt, fr);
}

bool dev_refiner_block(const SafeTensors& st, const std::string& pre, float* x, int N, const float* t256, const float* fr,
                       bool modulated) {
    if (!sddev::enabled()) { fprintf(stderr, "zimage: no device\n"); return false; }
    const int dim = 3840, H = 30, D = 128, F = 10240;
    const uint16_t *wada = nullptr, *wn1 = nullptr, *wn2 = nullptr, *wfn1 = nullptr, *wfn2 = nullptr, *wqkv = nullptr,
                   *wqn = nullptr, *wkn = nullptr, *wout = nullptr, *w1 = nullptr, *w2 = nullptr, *w3 = nullptr;
    if ((modulated && !named(st, pre + ".adaLN_modulation.0.weight", {4 * dim, 256}, wada)) ||
        !named(st, pre + ".attention_norm1.weight", {dim}, wn1) || !named(st, pre + ".attention_norm2.weight", {dim}, wn2) ||
        !named(st, pre + ".ffn_norm1.weight", {dim}, wfn1) || !named(st, pre + ".ffn_norm2.weight", {dim}, wfn2) ||
        !named(st, pre + ".attention.qkv.weight", {3 * dim, dim}, wqkv) ||
        !named(st, pre + ".attention.q_norm.weight", {D}, wqn) || !named(st, pre + ".attention.k_norm.weight", {D}, wkn) ||
        !named(st, pre + ".attention.out.weight", {dim, dim}, wout) ||
        !named(st, pre + ".feed_forward.w1.weight", {F, dim}, w1) || !named(st, pre + ".feed_forward.w2.weight", {dim, F}, w2) ||
        !named(st, pre + ".feed_forward.w3.weight", {F, dim}, w3))
        return false;
    /* the modulation: scale_msa = 1 + m0, gate_msa = tanh(m1), scale_mlp = 1 + m2, gate_mlp = tanh(m3); unmodulated
     * blocks (the context refiners) use 1 for all four (exact: x * 1, x + 1 * y) */
    DevBuf sc_msa(dim), g_msa(dim), sc_mlp(dim), g_mlp(dim);
    if (modulated) {
        std::vector<size_t> sb;
        std::vector<float> bada_h;
        if (!st.get(pre + ".adaLN_modulation.0.bias", bada_h, sb) || sb.size() != 1 || (int)sb[0] != 4 * dim) {
            fprintf(stderr, "zimage: %s.adaLN_modulation.0.bias missing\n", pre.c_str());
            return false;
        }
        DevBuf bada(4 * dim);
        for (int i2 = 0; i2 < 4 * dim; ++i2) bada.p[i2] = bada_h[i2];
        DevBuf mod(4 * dim);
        sddev::linear_bf16(t256, 1, wada, 256, 4 * dim, bada.p, mod.p);
        for (int c = 0; c < dim; ++c) {
            sc_msa.p[c] = 1.0f + mod.p[c];
            g_msa.p[c] = std::tanh(mod.p[dim + c]);
            sc_mlp.p[c] = 1.0f + mod.p[2 * dim + c];
            g_mlp.p[c] = std::tanh(mod.p[3 * dim + c]);
        }
    } else {
        for (int c = 0; c < dim; ++c) sc_msa.p[c] = g_msa.p[c] = sc_mlp.p[c] = g_mlp.p[c] = 1.0f;
    }
    DevBuf zeros(3 * dim, true);  /* zeros: the biases of the GEMMs without one (11520 >= every output here) */
    const float eps = 1e-5f;
    /* attention */
    DevBuf h(static_cast<size_t>(N) * dim);
    sddev::rms_norm(x, N, dim, wn1, eps, h.p);
    sddev::modulate(h.p, N, dim, sc_msa.p);
    const size_t sq = (size_t)dim * dim;
    DevBuf q(static_cast<size_t>(N) * dim), k(static_cast<size_t>(N) * dim), v(static_cast<size_t>(N) * dim);
    sddev::linear_bf16(h.p, N, wqkv, dim, dim, zeros.p, q.p);
    sddev::linear_bf16(h.p, N, wqkv + sq, dim, dim, zeros.p, k.p);
    sddev::linear_bf16(h.p, N, wqkv + 2 * sq, dim, dim, zeros.p, v.p);
    sddev::rms_norm(q.p, N * H, D, wqn, eps, q.p);   /* the per-head QK norms: rows of 128 */
    sddev::rms_norm(k.p, N * H, D, wkn, eps, k.p);
    sddev::rope(q.p, N, H, fr);
    sddev::rope(k.p, N, H, fr);
    DevBuf att(static_cast<size_t>(N) * dim);
    sddev::attention(q.p, N, k.p, v.p, N, dim, H, att.p);
    DevBuf o(static_cast<size_t>(N) * dim);
    sddev::linear_bf16(att.p, N, wout, dim, dim, zeros.p, o.p);
    sddev::rms_norm(o.p, N, dim, wn2, eps, o.p);
    sddev::gate_add(x, N, dim, g_msa.p, o.p);
    /* feed-forward (SwiGLU) */
    DevBuf h2(static_cast<size_t>(N) * dim);
    sddev::rms_norm(x, N, dim, wfn1, eps, h2.p);
    sddev::modulate(h2.p, N, dim, sc_mlp.p);
    DevBuf zerosF(F, true);
    DevBuf a(static_cast<size_t>(N) * F), b(static_cast<size_t>(N) * F);
    sddev::linear_bf16(h2.p, N, w1, dim, F, zerosF.p, a.p);
    sddev::linear_bf16(h2.p, N, w3, dim, F, zerosF.p, b.p);
    sddev::swiglu(a.p, b.p, a.p, (size_t)N * F);
    DevBuf f(static_cast<size_t>(N) * dim);
    sddev::linear_bf16(a.p, N, w2, F, dim, zeros.p, f.p);
    sddev::rms_norm(f.p, N, dim, wfn2, eps, f.p);
    sddev::gate_add(x, N, dim, g_mlp.p, f.p);
    return true;
}


bool dev_caption_pad_rows(const SafeTensors& st, float* cap, int L, int Lpad) {
    const uint16_t* pt = nullptr;
    std::vector<size_t> s2;
    if (!st.bf16_in_place("cap_pad_token", pt, s2) || s2.size() != 2 || s2[0] != 1 || s2[1] != 3840) {
        fprintf(stderr, "zimage: cap_pad_token missing\n");
        return false;
    }
    for (int t = L; t < Lpad; ++t)
        for (int c = 0; c < 3840; ++c) cap[(size_t)t * 3840 + c] = bf16_to_f32(pt[c]);
    return true;
}


bool dev_output(const SafeTensors& st, const float* J, const float* t256, float* lat) {
    if (!sddev::enabled()) { fprintf(stderr, "zimage: no device\n"); return false; }
    const int dim = 3840, Npix = 256, C = 64;
    const uint16_t *wada = nullptr, *wlin = nullptr;
    if (!named(st, "final_layer.adaLN_modulation.1.weight", {dim, 256}, wada) ||
        !named(st, "final_layer.linear.weight", {C, dim}, wlin))
        return false;
    float *bada = nullptr, *blin = nullptr;
    if (!bias_f32(st, "final_layer.adaLN_modulation.1.bias", dim, bada) || !bias_f32(st, "final_layer.linear.bias", C, blin))
        return false;
    /* the modulation: scale = 1 + Linear(SiLU(t)) */
    DevBuf silu_t(256);
    for (int i = 0; i < 256; ++i) silu_t.p[i] = t256[i] / (1.0f + std::exp(-t256[i]));
    DevBuf scale(dim);
    sddev::linear_bf16(silu_t.p, 1, wada, 256, dim, bada, scale.p);
    for (int c = 0; c < dim; ++c) scale.p[c] += 1.0f;
    /* norm_final (LayerNorm without affine), the scale, then the linear to 64 values per token */
    DevBuf y(static_cast<size_t>(Npix) * dim);
    sddev::layer_norm_plain(J, Npix, dim, 1e-6f, y.p);
    sddev::modulate(y.p, Npix, dim, scale.p);
    DevBuf tok(static_cast<size_t>(Npix) * C);
    sddev::linear_bf16(y.p, Npix, wlin, dim, C, blin, tok.p);
    /* unpatchify (the inverse of patchify): tok [16 x 16 tokens][(ph, pw, c)] -> lat [c][32][32] */
    const int H = 32, W = 32, Wt = 16;
    for (int ht = 0; ht < 16; ++ht)
        for (int wt = 0; wt < Wt; ++wt) {
            const float* t = tok.p + (size_t)(ht * Wt + wt) * C;
            for (int ph = 0; ph < 2; ++ph)
                for (int pw = 0; pw < 2; ++pw)
                    for (int c = 0; c < 16; ++c)
                        lat[((size_t)c * H + ht * 2 + ph) * W + wt * 2 + pw] = t[(ph * 2 + pw) * 16 + c];
        }
    return true;
}


bool dev_dit_forward(const SafeTensors& st, const float* lat, const float* cap, int L, float t, float* vel) {
    if (!sddev::enabled()) { fprintf(stderr, "zimage: no device\n"); return false; }
    const int NI = 256, LC = 32, S = NI + LC, DIM = 3840;
    if (L < 1 || L > LC) { fprintf(stderr, "zimage: the caption has %d rows (1 to 32)\n", L); return false; }
    /* the latent [16][1][32][32] as the patch embedding reads it (the sampler's latent, shared memory) */
    float* J = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)S * DIM, false));
    float* jx = J;
    int nx = 0;
    if (!dev_patch_embed(st, lat, 16, 32, 32, &jx, nx)) return false;
    float* jt = static_cast<float*>(sddev::alloc(sizeof(float) * 256, false));
    if (!dev_timestep_embed(st, t, jt)) return false;
    float* fri = static_cast<float*>(sddev::alloc(sizeof(float) * NI * 64 * 2, false));
    rope_table_image(16, 16, LC + 1, fri);
    if (!dev_refiner_block(st, "noise_refiner.0", J, NI, jt, fri) || !dev_refiner_block(st, "noise_refiner.1", J, NI, jt, fri))
        return false;
    float* jc = J + (size_t)NI * DIM;
    int lc = 0;
    if (!dev_caption_embed(st, cap, L, &jc, lc)) return false;
    /* the caption rows past L take the pad token (the reference's padding to 32) */
    dev_caption_pad_rows(st, jc, L, LC);
    int ids[LC * 3];
    for (int r = 0; r < LC; ++r) { ids[r * 3] = 1 + r; ids[r * 3 + 1] = 0; ids[r * 3 + 2] = 0; }
    float* frc = static_cast<float*>(sddev::alloc(sizeof(float) * LC * 64 * 2, false));
    rope_table_ids(ids, LC, frc);
    if (!dev_refiner_block(st, "context_refiner.0", jc, LC, nullptr, frc, false) ||
        !dev_refiner_block(st, "context_refiner.1", jc, LC, nullptr, frc, false))
        return false;
    int idj[S * 3];
    for (int r = 0; r < NI; ++r) { idj[r * 3] = LC + 1; idj[r * 3 + 1] = r / 16; idj[r * 3 + 2] = r % 16; }
    for (int r = 0; r < LC; ++r) { idj[(NI + r) * 3] = 1 + r; idj[(NI + r) * 3 + 1] = 0; idj[(NI + r) * 3 + 2] = 0; }
    float* frj = static_cast<float*>(sddev::alloc(sizeof(float) * S * 64 * 2, false));
    rope_table_ids(idj, S, frj);
    for (int layer = 0; layer < 30; ++layer)
        if (!dev_refiner_block(st, "layers." + std::to_string(layer), J, S, jt, frj, true)) return false;
    return dev_output(st, J, jt, vel);
}

bool dev_sample(const SafeTensors& st, const float* noise, const float* cap, int L, int steps, float shift, float* out) {
    /* the sigmas: linspace(1, 1/steps, steps), shifted (shift * s / (1 + (shift - 1) s)), then 0 */
    std::vector<double> sig(steps + 1);
    for (int i = 0; i < steps; ++i) {
        const double s0 = steps == 1 ? 1.0 : 1.0 + (1.0 / steps - 1.0) * i / (steps - 1);
        sig[i] = shift * s0 / (1.0 + (shift - 1.0) * s0);
    }
    sig[steps] = 0.0;
    std::vector<float> x(noise, noise + 16 * 32 * 32), vel(16 * 32 * 32);
    for (int i = 0; i < steps; ++i) {
        /* the model's timestep: (1000 - t) / 1000 with t = sigma * 1000, i.e. 1 - sigma */
        if (!dev_dit_forward(st, x.data(), cap, L, (float)(1.0 - sig[i]), vel.data())) return false;
        const float dt = (float)(sig[i + 1] - sig[i]);
        for (size_t k = 0; k < x.size(); ++k) x[k] += dt * (-vel[k]);  /* the pipeline negates the output */
    }
    std::copy(x.begin(), x.end(), out);
    return true;
}

}  // namespace zimage
