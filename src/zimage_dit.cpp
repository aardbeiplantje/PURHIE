/* Z-Image-Turbo's diffusion transformer, host path: see zimage_dit.h. The formulas are the reference's
 * (diffusers' ZImageTransformer2DModel, the timestep embedder, the RMS norm, the patchify and the padding). */
#include "zimage_dit.h"

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

}  // namespace zimage
