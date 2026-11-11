// glm5_ref.cpp -- GLM-5.3-Flash on the CPU (glm5_ref.h). The formulas are upstream llama.cpp's glm5-next
// (src/models/glm5-next.cpp and the ggml-cpu ops it runs: dsv4_hc_comb / _post, gated_delta_net in its KDA form,
// swiglu_clamp, build_moe_ffn), in float32 as those compute.

#include "glm5_ref.h"

#include "inference_internal.h"
#include "quant_types.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
float silu(float x) { return x / (1.0f + expf(-x)); }

// y[N] = W x[K]: each row decoded to float, a float dot product
void matvec(const QuantWeight& w, const float* x, float* y, int row0 = 0, int nrows = -1) {
    if (nrows < 0) nrows = w.N - row0;
    std::vector<float> r(w.K);
    for (int n = 0; n < nrows; ++n) {
        qt_to_float(w.type, w.row(row0 + n), r.data(), w.K);
        float s = 0.0f;
        for (int k = 0; k < w.K; ++k) s += r[k] * x[k];
        y[n] = s;
    }
}

std::vector<float> matvec(const QuantWeight& w, const float* x, int row0 = 0, int nrows = -1) {
    std::vector<float> y(nrows < 0 ? w.N - row0 : nrows);
    matvec(w, x, y.data(), row0, nrows);
    return y;
}

void rmsnorm(const float* x, const float* g, float* y, int n, float eps) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(s / n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * (g ? g[i] : 1.0f);
}

void layernorm(const float* x, const float* g, const float* b, float* y, int n, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
    const float r = 1.0f / sqrtf(var / n + eps);
    for (int i = 0; i < n; ++i) y[i] = (x[i] - mean) * r * g[i] + b[i];
}

// x / sqrt(sum x^2 + eps) (llama.cpp build_gdn_l2_norm)
void l2norm(float* x, int n, float eps) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(s + eps);
    for (int i = 0; i < n; ++i) x[i] *= r;
}

float swiglu(float gate, float up, float limit) {
    if (limit > 0.0f) {
        gate = std::min(gate, limit);
        up = std::clamp(up, -limit, limit);
    }
    return silu(gate) * up;
}

// SwiGLU FFN on rows [r0, r0 + ff) of gate / up and [d0, d0 + n_embd) of down
void ffn(const QuantWeight& wg, const QuantWeight& wu, const QuantWeight& wd, const float* x, float* out, int ff,
         int r0, int d0, int n_embd, float limit) {
    std::vector<float> g = matvec(wg, x, r0, ff), u = matvec(wu, x, r0, ff), h(ff);
    for (int i = 0; i < ff; ++i) h[i] = swiglu(g[i], u[i], limit);
    matvec(wd, h.data(), out, d0, n_embd);
}

// The mHC mixer (llama.cpp glm5-next build_hc_pre and ggml dsv4_hc_comb): the block input from the streams, the
// output's weights into each stream (post) and the streams' mixing (comb[dst + hc * src])
void hc_pre(const inference_model* m, const HcSinkhorn& h, const float* streams, float* x, float* post, float* comb) {
    const int E = m->n_embd, HC = m->hc_mult;
    std::vector<float> flat(HC * E);
    rmsnorm(streams, nullptr, flat.data(), HC * E, (float)m->eps);
    std::vector<float> mix = matvec(h.fn, flat.data());
    const float eps = m->hc_eps;
    float pre[8];
    for (int i = 0; i < HC; ++i) {
        pre[i] = sigmoid(mix[i] * h.scale[0] + h.base[i]) + eps;
        post[i] = 2.0f * sigmoid(mix[HC + i] * h.scale[1] + h.base[HC + i]);
    }
    for (int s = 0; s < HC; ++s) {  // softmax over dst for each src, then + eps
        float mx = -INFINITY;
        for (int d = 0; d < HC; ++d) {
            const int idx = d + HC * s;
            comb[idx] = mix[2 * HC + idx] * h.scale[2] + h.base[2 * HC + idx];
            mx = std::max(mx, comb[idx]);
        }
        float sum = 0.0f;
        for (int d = 0; d < HC; ++d) sum += comb[d + HC * s] = expf(comb[d + HC * s] - mx);
        const float inv = 1.0f / sum;
        for (int d = 0; d < HC; ++d) comb[d + HC * s] = comb[d + HC * s] * inv + eps;
    }
    auto cols = [&]() {
        for (int d = 0; d < HC; ++d) {
            float sum = eps;
            for (int s = 0; s < HC; ++s) sum += comb[d + HC * s];
            const float inv = 1.0f / sum;
            for (int s = 0; s < HC; ++s) comb[d + HC * s] *= inv;
        }
    };
    auto rows = [&]() {
        for (int s = 0; s < HC; ++s) {
            float sum = eps;
            for (int d = 0; d < HC; ++d) sum += comb[d + HC * s];
            const float inv = 1.0f / sum;
            for (int d = 0; d < HC; ++d) comb[d + HC * s] *= inv;
        }
    };
    cols();
    for (int i = 1; i < m->hc_sinkhorn_iters; ++i) rows(), cols();
    for (int e = 0; e < E; ++e) {
        float v = 0.0f;
        for (int i = 0; i < HC; ++i) v += pre[i] * streams[i * E + e];
        x[e] = v;
    }
}

// The streams after a block: out[dst] = x * post[dst] + sum_src streams[src] * comb[dst, src]
void hc_post(const inference_model* m, const float* x, float* streams, const float* post, const float* comb) {
    const int E = m->n_embd, HC = m->hc_mult;
    std::vector<float> out(HC * E);
    for (int d = 0; d < HC; ++d)
        for (int e = 0; e < E; ++e) {
            float v = x[e] * post[d];
            for (int s = 0; s < HC; ++s) v += streams[s * E + e] * comb[d + HC * s];
            out[d * E + e] = v;
        }
    std::copy(out.begin(), out.end(), streams);
}

}  // namespace

Glm5Ref::Glm5Ref(const inference_model* model) : m(model) { reset(); }

void Glm5Ref::reset() {
    st.assign(m->n_layer, LayerState());
    const int DI = m->n_head * m->kda_head_dim, HD = m->kda_head_dim;
    for (int l = 0; l < m->n_layer; ++l)
        if (m->layers[l].kind == LAYER_KDA) {
            for (auto& c : st[l].conv) c.assign((size_t)(m->ssm_d_conv - 1) * DI, 0.0f);
            st[l].S.assign((size_t)m->n_head * HD * HD, 0.0f);
        }
}

void Glm5Ref::embed(int token, float* streams) const {
    const int E = m->n_embd;
    qt_to_float(m->token_embd.type, m->token_embd.row(token), streams, E);
    for (int i = 1; i < m->hc_mult; ++i) std::copy(streams, streams + E, streams + i * E);
}

void Glm5Ref::logits(const float* streams, float* out) const {
    const int E = m->n_embd, HC = m->hc_mult;
    std::vector<float> x(E, 0.0f), y(E);
    for (int i = 0; i < HC; ++i)
        for (int e = 0; e < E; ++e) x[e] += streams[i * E + e];
    for (int e = 0; e < E; ++e) x[e] *= 1.0f / HC;
    rmsnorm(x.data(), m->output_norm, y.data(), E, (float)m->eps);
    matvec(m->output, y.data(), out);
}

void Glm5Ref::forward(int token, int pos, float* out) {
    std::vector<float> s(m->hc_mult * m->n_embd);
    embed(token, s.data());
    for (int l = 0; l < m->n_layer; ++l) layer(l, pos, s.data());
    if (out) logits(s.data(), out);
}

void Glm5Ref::layer(int l, int pos, float* streams, Glm5LayerTrace* tr) {
    const LayerWeights& L = m->layers[l];
    LayerState& S = st[l];
    const int E = m->n_embd, HC = m->hc_mult, H = m->n_head;
    const float eps = (float)m->eps;
    std::vector<float> x(E), xn(E), out(E, 0.0f);
    float post[8], comb[64];

    // ---- token mixer
    hc_pre(m, L.hcs_attn, streams, x.data(), post, comb);
    if (tr) tr->attn_in = x;
    rmsnorm(x.data(), L.attn_norm, xn.data(), E, eps);
    if (L.kind == LAYER_KDA) {
        const int HD = m->kda_head_dim, DI = H * HD, DC = m->ssm_d_conv;
        const QuantWeight* W[3] = {&L.Wq, &L.Wk, &L.Wv};
        std::vector<float> qkv[3];
        for (int c = 0; c < 3; ++c) {  // projection, causal conv1d over the last d_conv inputs, SiLU
            std::vector<float> p = matvec(*W[c], xn.data());
            std::vector<float>& win = S.conv[c];
            qkv[c].resize(DI);
            for (int ch = 0; ch < DI; ++ch) {
                const float* w = L.kda_conv[c] + (size_t)ch * DC;
                float v = w[DC - 1] * p[ch];
                for (int k = 0; k < DC - 1; ++k) v += w[k] * win[(size_t)k * DI + ch];
                qkv[c][ch] = silu(v);
            }
            for (int k = 0; k + 1 < DC - 1; ++k)  // the window moves on
                std::copy(win.begin() + (size_t)(k + 1) * DI, win.begin() + (size_t)(k + 2) * DI, win.begin() + (size_t)k * DI);
            if (DC > 1) std::copy(p.begin(), p.end(), win.begin() + (size_t)(DC - 2) * DI);
        }
        for (int h = 0; h < H; ++h) {
            l2norm(qkv[0].data() + h * HD, HD, 1e-6f);
            l2norm(qkv[1].data() + h * HD, HD, 1e-6f);
        }
        std::vector<float> fa = matvec(L.W_f_a, xn.data()), g = matvec(L.W_f_b, fa.data());
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < HD; ++i) {  // the per-channel log decay
                const float g1 = g[h * HD + i] + L.ssm_dt[h * HD + i], A = L.ssm_a[h];
                g[h * HD + i] = m->kda_gate_lower_bound > -INFINITY ? m->kda_gate_lower_bound * sigmoid(-(g1 * A))
                                                                    : (g1 > 20.0f ? g1 : log1pf(expf(g1))) * A;
            }
        std::vector<float> beta = matvec(L.W_beta, xn.data());
        std::vector<float> o(DI), delta(HD);
        const float scale = 1.0f / sqrtf((float)HD);
        for (int h = 0; h < H; ++h) {
            float* M = S.S.data() + (size_t)h * HD * HD;  // M[i * HD + j] = S[i][j]
            const float* q = qkv[0].data() + h * HD;
            const float* k = qkv[1].data() + h * HD;
            const float* v = qkv[2].data() + h * HD;
            const float b = sigmoid(beta[h]);
            for (int i = 0; i < HD; ++i) {
                const float d = expf(g[h * HD + i]);
                for (int j = 0; j < HD; ++j) M[i * HD + j] *= d;
            }
            for (int j = 0; j < HD; ++j) {
                float s = 0.0f;
                for (int i = 0; i < HD; ++i) s += M[i * HD + j] * k[i];
                delta[j] = (v[j] - s) * b;
            }
            for (int i = 0; i < HD; ++i)
                for (int j = 0; j < HD; ++j) M[i * HD + j] += k[i] * delta[j];
            for (int j = 0; j < HD; ++j) {
                float s = 0.0f;
                for (int i = 0; i < HD; ++i) s += M[i * HD + j] * q[i];
                o[h * HD + j] = s * scale;
            }
        }
        std::vector<float> ga = matvec(L.W_g_a, xn.data()), g2 = matvec(L.W_g_b, ga.data()), gated(DI);
        for (int h = 0; h < H; ++h) {
            rmsnorm(o.data() + h * HD, L.ssm_norm, gated.data() + h * HD, HD, eps);
            for (int i = 0; i < HD; ++i) gated[h * HD + i] *= sigmoid(g2[h * HD + i]);
        }
        matvec(L.Wo, gated.data(), out.data());
    } else {  // MLA over the latent, the positions the DSA indexer picks
        const int QL = m->q_lora, KL = m->kv_lora, KM = m->k_mla, VM = m->v_mla;
        std::vector<float> qa = matvec(L.W_q_a, xn.data()), qr(QL);
        rmsnorm(qa.data(), L.q_a_norm, qr.data(), QL, eps);
        std::vector<float> q = matvec(L.W_q_b, qr.data());
        std::vector<float> kv = matvec(L.W_kv_a, xn.data());
        S.latent.resize((size_t)(pos + 1) * KL);
        rmsnorm(kv.data(), L.kv_a_norm, S.latent.data() + (size_t)pos * KL, KL, eps);
        if (L.ix_full) {
            const int IH = m->ix_heads, ID = m->ix_dim, KP = m->ix_kpool;
            std::vector<float> iq = matvec(L.W_ix_q_b, qr.data());
            std::vector<float> ik = matvec(L.W_ix_k, xn.data()), ig = matvec(L.W_ix_gate, xn.data());
            S.ik.resize((size_t)(pos + 1) * ID);
            S.ig.resize((size_t)(pos + 1) * ID);
            layernorm(ik.data(), L.ix_k_norm, L.ix_k_norm_b, S.ik.data() + (size_t)pos * ID, ID, m->ln_eps);
            std::copy(ig.begin(), ig.end(), S.ig.begin() + (size_t)pos * ID);
            if ((pos + 1) % KP == 0) {  // a pool completed: its pooled key, softmax(gate + ape) over its positions
                const int p0 = pos + 1 - KP;
                std::vector<float> ape((size_t)KP * ID);
                for (int j = 0; j < KP; ++j) qt_to_float(L.W_ix_ape.type, L.W_ix_ape.row(j), ape.data() + (size_t)j * ID, ID);
                S.pooled.resize((size_t)(pos + 1) / KP * ID);
                float* pk = S.pooled.data() + (size_t)(pos / KP) * ID;
                for (int d = 0; d < ID; ++d) {
                    float lg[64], mx = -INFINITY, sum = 0.0f, v = 0.0f;
                    for (int j = 0; j < KP; ++j) mx = std::max(mx, lg[j] = S.ig[(size_t)(p0 + j) * ID + d] + ape[(size_t)j * ID + d]);
                    for (int j = 0; j < KP; ++j) sum += lg[j] = expf(lg[j] - mx);
                    for (int j = 0; j < KP; ++j) v += lg[j] / sum * S.ik[(size_t)(p0 + j) * ID + d];
                    pk[d] = v;
                }
            }
            std::vector<float> w = matvec(L.W_ix_proj, xn.data());
            const float ws = 1.0f / sqrtf((float)(ID * IH));
            const int nv = (pos + 1) / KP;  // the complete pools, all at or before pos
            std::vector<float> score(nv);
            for (int j = 0; j < nv; ++j) {
                float sc = 0.0f;
                for (int h = 0; h < IH; ++h) {
                    float d = 0.0f;
                    for (int c = 0; c < ID; ++c) d += iq[(size_t)h * ID + c] * S.pooled[(size_t)j * ID + c];
                    sc += std::max(d, 0.0f) * w[h] * ws;
                }
                score[j] = sc;
            }
            std::vector<int> order(nv);
            std::iota(order.begin(), order.end(), 0);
            const int ntop = std::min(nv, m->ix_topk / KP);
            // ties to the lower pool index: a deterministic choice, the same in decode and prefill
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });
            if (tr) {
                tr->scores = score;
                tr->tie = ntop < nv && score[order[ntop - 1]] == score[order[ntop]];
            }
            S.sel.clear();
            for (int t = 0; t < ntop; ++t)
                for (int k = 0; k < KP; ++k) S.sel.push_back(order[t] * KP + k);
            if (m->ix_select_tail)
                for (int k = 0; k < (pos + 1) % KP; ++k) S.sel.push_back(pos - k);
            std::sort(S.sel.begin(), S.sel.end());
        } else {  // the previous MLA layer's selection
            for (int j = l - 1; j >= 0; --j)
                if (m->layers[j].kind == LAYER_MLA_DSA) { S.sel = st[j].sel; break; }
        }
        if (tr) tr->selected = S.sel;
        std::vector<float> o(H * VM), qabs(KL), olat(KL), p(S.sel.size());
        const float scale = 1.0f / sqrtf((float)KM);
        for (int h = 0; h < H; ++h) {
            matvec(L.W_k_b, q.data() + (size_t)h * KM, qabs.data(), h * KL, KL);  // q absorbed into the latent
            float mx = -INFINITY, sum = 0.0f;
            for (size_t t = 0; t < S.sel.size(); ++t) {
                const float* c = S.latent.data() + (size_t)S.sel[t] * KL;
                float d = 0.0f;
                for (int i = 0; i < KL; ++i) d += qabs[i] * c[i];
                mx = std::max(mx, p[t] = d * scale);
            }
            for (auto& v : p) sum += v = expf(v - mx);
            std::fill(olat.begin(), olat.end(), 0.0f);
            for (size_t t = 0; t < S.sel.size(); ++t) {
                const float* c = S.latent.data() + (size_t)S.sel[t] * KL;
                for (int i = 0; i < KL; ++i) olat[i] += p[t] / sum * c[i];
            }
            matvec(L.W_v_b, olat.data(), o.data() + (size_t)h * VM, h * VM, VM);
        }
        matvec(L.Wo, o.data(), out.data());
    }
    if (tr) tr->mixer_out = out;
    hc_post(m, out.data(), streams, post, comb);
    if (tr) tr->attn_post.assign(streams, streams + HC * E);

    // ---- FFN
    hc_pre(m, L.hcs_ffn, streams, x.data(), post, comb);
    if (tr) tr->ffn_in = x;
    rmsnorm(x.data(), L.ffn_norm, xn.data(), E, eps);
    std::fill(out.begin(), out.end(), 0.0f);
    if (!L.moe) {
        ffn(L.W_gate, L.W_up, L.W_down, xn.data(), out.data(), L.W_gate.N, 0, 0, E, L.swiglu_limit_sh);
    } else {
        const int NE = m->n_expert, NU = m->n_expert_used, FE = m->n_ff_exp;
        std::vector<float> lg = matvec(L.W_router, xn.data()), prob(NE), selp(NE);
        for (int e = 0; e < NE; ++e) selp[e] = (prob[e] = sigmoid(lg[e])) + L.exp_probs_b[e];
        std::vector<int> order(NE);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return selp[a] > selp[b]; });
        float wsum = 0.0f;
        for (int k = 0; k < NU; ++k) wsum += prob[order[k]];
        wsum = std::max(wsum, 6.103515625e-5f);
        std::vector<float> y(E);
        for (int k = 0; k < NU; ++k) {
            const int e = order[k];
            const float w = (m->expert_weights_norm ? prob[e] / wsum : prob[e]) * m->expert_weights_scale;
            ffn(L.We_gate, L.We_up, L.We_down, xn.data(), y.data(), FE, e * FE, e * E, E, L.swiglu_limit);
            for (int i = 0; i < E; ++i) out[i] += w * y[i];
        }
        ffn(L.W_gate, L.W_up, L.W_down, xn.data(), y.data(), L.W_gate.N, 0, 0, E, L.swiglu_limit_sh);
        for (int i = 0; i < E; ++i) out[i] += y[i];
    }
    if (tr) tr->ffn_out = out;
    hc_post(m, out.data(), streams, post, comb);
}
