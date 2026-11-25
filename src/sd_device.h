// The image engine's device path (sd.cpp; PLAN.md I6): the activations are float32 in registered UMA memory (Floats), the
// F16 weights are read in place from the GGUF mapping (register_weights), and each op is one of sd_gpu.hip's kernels. The
// host code of sd.cpp is the fallback (SD_DEVICE=cpu, or no device): the same functions, on the same memory.
#ifndef SD_DEVICE_H
#define SD_DEVICE_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace sddev {

// The device path is on when SD_DEVICE is unset or "gpu", a device is present and the kernels load.
bool enabled();

// The GGUF mapping [base, base + bytes) registered read-only for the device (the weights are read in place). Returns false
// when the registration fails. unregister_weights is the mapping's release (a no-op for an unregistered base).
bool register_weights(const void* base, size_t bytes);
void unregister_weights(const void* base);

// One sd_model at a time on the device: the callers hold this for a whole UNet or VAE call (the streams sync per op).
std::recursive_mutex& run_mutex();

// Float memory (zeroed unless zero is false): pinned and device-visible when the device path is on, else calloc. Pooled.
void* alloc(size_t bytes, bool zero = true);  // zero = false: the caller writes every byte (no memset: a CPU pass)
void release(void* p, size_t bytes);
// The pinned pool's bytes handed out (in use) and released but kept for reuse (cached, at most 16 GiB; the largest go back beyond it).
void pinned_bytes(size_t& in_use, size_t& cached);
// Device-only memory (hipMalloc, pooled): for buffers the host never reads or writes (the GEMM inputs in BF16). Not
// host-visible: the host code must not touch these pointers.
void* dalloc(size_t bytes);
void drelease(void* p, size_t bytes);

// Float32 activations: the memory of alloc, zeroed. Copies are deep (the image's tensors are small next to the weights).
class Floats {
   public:
    Floats() = default;
    explicit Floats(size_t n) : n_(n), p_(n ? static_cast<float*>(alloc(n * sizeof(float))) : nullptr) {}
    ~Floats() { reset(); }
    Floats(const Floats& o) : Floats(o.n_) {
        if (n_) std::memcpy(p_, o.p_, n_ * sizeof(float));
    }
    Floats(Floats&& o) noexcept : n_(o.n_), p_(o.p_) {
        o.n_ = 0;
        o.p_ = nullptr;
    }
    Floats& operator=(const Floats& o) {
        if (this != &o) {
            Floats tmp(o);
            swap(tmp);
        }
        return *this;
    }
    Floats& operator=(Floats&& o) noexcept {
        if (this != &o) {
            reset();
            n_ = o.n_;
            p_ = o.p_;
            o.n_ = 0;
            o.p_ = nullptr;
        }
        return *this;
    }
    float* data() { return p_; }
    const float* data() const { return p_; }
    size_t size() const { return n_; }
    float& operator[](size_t i) { return p_[i]; }
    const float& operator[](size_t i) const { return p_[i]; }
    void swap(Floats& o) noexcept {
        size_t n = n_;
        float* p = p_;
        n_ = o.n_;
        p_ = o.p_;
        o.n_ = n;
        o.p_ = p;
    }

   private:
    void reset() {
        release(p_, n_ * sizeof(float));
        p_ = nullptr;
        n_ = 0;
    }
    size_t n_ = 0;
    float* p_ = nullptr;
};

// The ops. Activations: float32 in alloc'd memory; weights: F16 (uint16_t) in the registered mapping; biases F16 or null.
// Conv: x [cin][H][W] -> y [cout][Ho][Wo], H' = (H + 2p - k) / st + 1. Linear: token rows x[t][in] -> y[t][out]; the
// channel-plane input x[c][t] (linear_cm); the channel-plane output y[o][t] (linear_to_cm, accumulating into y when set).
void conv2d(const float* x, int cin, int H, int W, const uint16_t* w, int cout, const uint16_t* b, int k, int p, int st, float* y,
            int Ho, int Wo);
void linear_tok(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y);
void linear_cm(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y);
// The same with float32 weights and bias (b may be null): y[t][o] = b[o] + sum_i x[t][i] W[o][i]
void linear_f32(const float* x, int T, const float* W, int in, int out, const float* b, float* y);
// conv2d with F32 weights [cout][cin][k][k] (read in place) and F32 bias; the same im2col + GEMM as conv2d
void conv2d_f32(const float* x, int cin, int H, int W, const float* w, int cout, const float* b, int k, int p, int st, float* y, int Ho, int Wo);
/* the same convolution in FP32 throughout (no BF16 route): the accuracy-checked layers (AudioLDM 2's VAE) */
void conv2d_f32_exact(const float* x, int cin, int H, int W, const float* w, int cout, const float* b, int k, int p, int st, float* y, int Ho, int Wo);
// Qwen-Image's FP8 E4M3 linear (sd_gpu.hip qi_linear_f8): W [out][in] raw bytes read in place, times wscale; the bias is
// bias_kind 0 none, 1 E4M3, 2 BF16 bits, 3 F32. y[t][o] = wscale sum_i x[t][i] W[o][i] + b[o]
void linear_f8(const float* x, int T, const uint8_t* W, float wscale, int in, int out, const void* b, int bias_kind, float* y);
// Qwen-Image's norms and elementwise ops (sd_gpu.hip): the RMS norm with E4M3 weights; x[t][c] = x[t][c] (1 + s[c]) + b[c] in
// place; GELU (tanh approximation) in place
void rms_norm_f8(const float* x, int T, int C, const uint8_t* w, float eps, float* y);
void affine(float* x, int T, int C, const float* s, const float* b);
void gelu_tanh(float* x, size_t n);
// Qwen-Image's VAE decoder ops (sd_gpu.hip), token-major x[h][w][c] (H x W pixels, C channels): the channel RMS norm (gamma
// BF16); the convolution patches for the GEMM (mode 1: 2D 3 x 3, K = 9C; mode 3: causal 3 x 3 x 3 on one frame, K = 27C) of the
// pixel rows p0..p0+n-1 into P [n][K]; a BF16 bias added to rows of C; the nearest 2x upsampling
void rms_tok(const float* x, int HW, int C, const uint16_t* g, float* y);
void patches(const float* x, int H, int W, int C, int mode, int p0, int n, float* P);
void bias_tok(float* y, long rows, int C, const uint16_t* b);
void upsample2_tok(const float* x, int H, int W, int C, float* y);
// Qwen-Image 2.1's ConvRot linear (sd_gpu.hip qi_hadamard256 and qi_linear_i8): int8 weights W [out][in] in place with per-row
// scales, the input rotated by H (the regular Hadamard of 256, normalised) per block; xr: T * in floats of scratch (sddev::alloc)
void linear_convrot(const float* x, int T, const int8_t* W, const float* wscale, const float* H, int in, int out, float* xr, float* y);
// Qwen-Image 2.1 ops (sd_gpu.hip): the zero-centred RMS norm (BF16 weights, scale w + 1); the per-token modulation gather from the
// two rows [2][C] (sel[j] > 0.5: row 1, the t = 0 row); x *= 1 + s; h += tanh(g) y; the attention with the text causal
void zc_rms(const float* x, int T, int C, const uint16_t* w, float eps, float* y);
void gather_mod(const float* mod, const float* sel, int J, int C, float* out);
void mul_1p(float* x, const float* s, size_t n);
void gated_add(float* h, const float* g, const float* y, size_t n);
void linear_bf16_f32(const float* x, int T, const uint16_t* W, int in, int out, float* y);
void dupup(const float* x, int H, int W, int in, int out, int factor_t, float* y);
void attn21_text(const float* q, const float* k, const float* v, int J, int L, int heads, float* out);
// The same with bfloat16 weights (the checkpoint's BF16, raw bits): y[t][o] = b[o] + sum_i x[t][i] W[o][i]
void linear_bf16(const float* x, int T, const uint16_t* W, int in, int out, const float* b, float* y);
// the same with the input already BF16 (xb: T x in, row-major; device-only memory from dalloc or alloc): no conversion
void linear_bf16_xb(const uint16_t* xb, int T, const uint16_t* W, int in, int out, const float* b, float* y);
// the same with the output in BF16 (hipBLASLt's BF16 output, the bias as its epilogue): false when the device path does not take
// the shape (out >= 256, T >= 1024, in % 64 == 0, hipBLASLt on): the caller takes the FP32 path
bool linear_bf16_out16(const uint16_t* xb, int T, const uint16_t* W, int in, int out, const float* b, uint16_t* y);
// the same with the output always BF16 (the FP32 product rounded when hipBLASLt does not take the shape)
bool linear_bf16_xb16(const uint16_t* xb, int T, const uint16_t* W, int in, int out, const float* b, uint16_t* y16);
// the DiT's BF16 chain: rows of 128 (per token and head): y = rope(rms(x) w), BF16 in and out (y may be x)
void head_norm_rope_bf16(const uint16_t* x, int T, int H, const uint16_t* w, float eps, const float* fr, uint16_t* y);
// x[t] += g (rms(o[t]) w) for BF16 o: the RMS norm and the gated residual, in place on the FP32 x
void rms_gate_add_bf16(float* x, int T, int C, const uint16_t* o, const uint16_t* w, float eps, const float* g);
// y = BF16(x) (n values): an FP32 activation as the next GEMM's BF16 input
void to_bf16(const float* x, uint16_t* y, size_t n);
/* x[i] = BF16(x[i]) in place (the values stay FP32 containers) */
void round_bf16_inplace(float* x, size_t n);
/* AudioLDM 2's GPT-2 (fp32): y = x W + b with x [T][in], W [in][out] (the GPT-2 Conv1D layout), b [out] */
void audio_gemm_f32(const float* x, int T, int in, const float* W, const float* b, int out, float* y);
/* LayerNorm with weight and bias over the last dimension, x [T][C] */
void audio_layer_norm_f32(const float* x, int T, int C, const float* w, const float* b, float eps, float* y);
/* causal attention with 64-wide heads (scale 1/8): qkv [T][3 H 64] -> out [T][H 64] */
void audio_causal_attn_f32(const float* qkv, int T, int H, float* out);
/* multi-head attention (heads of D <= 128 wide, C = H * D): q [Tq][C], k and v [Tk][C] with row stride ld; causal (Tq = Tk) or not; kmask
 * (Tk floats, 0 for a padding key, device-accessible) or null; scores * scale + bias [H][Tq][Tk] (device-accessible) or none; out [Tq][C] */
void audio_attention_f32(const float* q, const float* k, const float* v, int ld, int Tq, int Tk, int H, int D, bool causal, const float* kmask,
                         float scale, const float* bias, float* out);
/* exact GELU (erf), in place */
void audio_gelu_erf(float* x, size_t n);
/* x = RMS norm (T5's: no mean, no bias) over the last dimension: y = x / sqrt(mean(x^2) + eps) * w, x [T][C] */
void audio_rms_norm_f32(const float* x, int T, int C, const float* w, float eps, float* y);
/* a = gelu_new(a) * b, in place (T5's gated GELU) */
void audio_gelu_tanh_mul(float* a, const float* b, size_t n);
/* h[c][p] += t[c] over the HW positions of each channel (h [C][HW]), in place */
void audio_add_channel(float* h, const float* t, int C, int HW);
/* dst[0..n) = src[0..n) (device-accessible buffers) */
void audio_copy(float* dst, const float* src, size_t n);
/* 1-D convolution, stride 1: x [cin][T] -> y [cout][Tout], Tout = T + 2 pad - dil (k - 1); w [cout][cin][k] */
void audio_conv1d(const float* x, int cin, int T, const float* w, const float* b, int cout, int k, int pad, int dil, float* y, int Tout);
/* 1-D transposed convolution: x [cin][T] -> y [cout][Tout], Tout = (T - 1) stride - 2 pad + k; w [cin][cout][k] */
void audio_conv_transpose1d(const float* x, int cin, int T, const float* w, const float* b, int cout, int k, int stride, int pad, float* y,
                            int Tout);
void audio_leaky_relu(float* x, float slope, size_t n);
void audio_scale(float* x, float s, size_t n);
void audio_tanh(float* x, size_t n);
/* nearest resampling of x [C][H][W] to [C][Ho][Wo] (PyTorch's nearest: the source index floor(dst * H / Ho)) */
void audio_upsample_nearest(const float* x, int C, int H, int W, int Ho, int Wo, float* y);
/* the device's BF16 values copied to host memory as float (debugging dumps; not a serving path) */
void download_bf16_as_f32(const uint16_t* d, float* h, size_t n);
// g = BF16(silu(a) b) for BF16 a, b
void swiglu_bf16_in(const uint16_t* a, const uint16_t* b, uint16_t* g, size_t n);
// attention on BF16 q, k, v (T x C row-major, the head width 128): the output in FP32
void attention_bf16(const uint16_t* q, int Tq, const uint16_t* k, const uint16_t* v, int Tk, int C, int heads, float* out);
// y = BF16(rms(x) w s): the RMS norm (w BF16) and the AdaLN scale s (per channel), the GEMM input in BF16
void rms_mod_bf16(const float* x, int T, int C, const uint16_t* w, float eps, const float* s, uint16_t* y);
// g = BF16(silu(a) b), the SwiGLU's gate as the down projection's BF16 input
void swiglu_bf16(const float* a, const float* b, uint16_t* g, size_t n);
void linear_to_cm(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y, bool accumulate);
void group_norm(const float* x, int C, int HW, int groups, const uint16_t* gw, const uint16_t* gb, float eps, float* y);
void silu(float* x, size_t n);
void add(float* out, const float* x, size_t n);
void upsample2(const float* x, int C, int H, int W, float* y);
void layer_norm(const float* x, int T, int C, const uint16_t* w, const uint16_t* b, float eps, float* y);
void rms_norm(const float* x, int T, int C, const uint16_t* w, float eps, float* y);
// Z-Image's block ops (sd_gpu.hip): x[t][c] *= s[c]; out[t][c] += g[c] * y[t][c]; g = silu(a) * b; the RoPE rotation of
// q [T][heads][128] by fr [T][64][2] = (cos, sin) per pair
void modulate(float* x, int T, int C, const float* s);
// LayerNorm of each row without weights: y = (x - mean) rsqrt(var + eps)
void layer_norm_plain(const float* x, int T, int C, float eps, float* y);
// GroupNorm with F32 affine weights on the channel-major [C][HW] x: y = groupnorm(x) * gamma + beta (per channel)
// transposes between channel-major [C][HW] and token-major [HW][C]
// Qwen3's text encoder ops (sd_gpu.hip): RoPE by rotate-half (x [T][heads][128], cos/sin [T][64]); causal attention with
// grouped KV heads (T <= 64): out [T][H][128] from q [T][H][128], k and v [T][KVH][128]
void rope_half(float* x, int T, int heads, const float* cos, const float* sin);
void causal_attn(const float* q, const float* k, const float* v, int T, int H, int KVH, float* out);
void to_tokens(const float* x, int C, int HW, float* y);
void to_chw(const float* x, int C, int HW, float* y);
void group_norm_f32(const float* x, int C, int HW, int groups, const float* gamma, const float* beta, float eps, float* y);
void gate_add(float* out, int T, int C, const float* g, const float* y);
void swiglu(const float* a, const float* b, float* g, size_t n);
void rope(float* q, int T, int heads, const float* fr);  // y = x rsqrt(mean(x^2) + eps) w
void geglu(const float* pj, long rows, int ff, float* g);
// The attention over token rows: q [Tq][C], k and v [Tk][C] (heads of width C / heads), out [Tq][C]; scores scaled by 1/sqrt(D).
void attention(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out);
// The same in FP32 throughout (no BF16 rounding of Q, K, V; the device path of attention for head widths other than 128, and the
// Qwen-Image DiT, whose 1 % error in the BF16 flash path was measured against the FP32 reference)
void attention_f32(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out);
// The same with row strides: q has Tq rows of ldq floats, k and v have Tk rows of ldk and ldv (the heads are columns of width C /
// heads); out is [Tq][C] (contiguous). The trellis flows take their Q, K, V from one packed [T][3C] projection.
void attention_f32_strided(const float* q, long ldq, int Tq, const float* k, long ldk, const float* v, long ldv, int Tk, int C,
                           int heads, float* out);

// TRELLIS.2 sparse-structure flow (sd_gpu.hip): x = x * s + sh per channel over [T][C]
void ss_affine(float* x, int T, int C, const float* s, const float* sh);
// h += g * y per channel over [T][C]
void ss_gate_add(float* h, int T, int C, const float* g, const float* y);
// per token and head (rows of D = 128): x = normalize(x) * sqrt(D) * gamma, then the RoPE pairs by phases [T][D/2][2] when
// phases is not null; x has row stride ld (in place)
void ss_head_norm_rope(float* x, int T, int ld, int H, int D, const float* gamma, const float* phases);

// TRELLIS.2 sparse-structure decoder (sd_gpu.hip), token-major activations [V][C], V = n^3:
// the 3x3x3 im2col of x [V][C] (padding 1) into P [V][C * 27]
void ss_im2col3(const float* x, int C, int n, float* P);
// y[v][o] = b[o] + sum_k P[v][k] W[o][k] over v in [v0, v0 + nv), o in [o0, o0 + no); W fp16 in place, P rows of K
void ss_gemm_f16w(const float* P, long K, const uint16_t* W, const float* b, int Cout, long Vtot, float* y, int v0, int nv, int o0, int no);
// the 3D pixel shuffle: x [(n)^3][Cout * 8] -> y [(2n)^3][Cout]
void ss_pixel_shuffle3(const float* x, int Cout, int n, float* y);

// sparse voxels (sd_gpu.hip, plain C-style kernels): the active voxels as int32 coords [N][4] = (batch, x, y, z).
// sp_hash_build: a hash table of tsize (a power of two) 64-bit keys and values; sp_neighbor_map: nb[i][t] for the 27 offsets;
// sp_subm_conv_f16w: the submanifold 3x3x3 convolution with the fp16 weights (the layout as the flag says), y [N][Cout]
void sp_hash_build(const int* coords, int n, unsigned long long* keys, int* vals, unsigned int tsize);
void sp_neighbor_map(const int* coords, int n, const unsigned long long* keys, const int* vals, unsigned int tsize, int* nb);
void sp_subm_conv_f16w(const float* x, int n, int cin, const int* nb, const uint16_t* W, const float* b, int cout, float* y,
                       int o0, int no, int layout);  // layout 0: [Cout][Cin][27]; 1: [Cout][27][Cin] (FlexGEMM's)

// channel-to-spatial upsampling (factor 2): the children of each voxel where logit > 0, in increasing child index;
// sp_c2s_mask: bits[i] (child bits) and cnt[i]; sp_exclusive_scan: off[0..n] (off[n] the total); sp_c2s_fill: the new coords
// [M][4], idx[p] and subidx[p]; sp_c2s_gather: out[p][j] = src[(idx[p] * 8 + subidx[p]) * cc + j / rep]
void sp_c2s_mask(const float* logit, int n, unsigned char* bits, int* cnt);
void sp_exclusive_scan(const int* cnt, int n, int* off);
void sp_c2s_fill(const int* coords, const unsigned char* bits, const int* off, int n, int* new_coords, int* idx, int* subidx);
void sp_c2s_gather(const float* src, const int* idx, const int* subidx, long M, int cc, int rep, float* out);

// the mesh step of the shape decoder (sd_gpu.hip): the vertices of the voxels; the four voxels of each edge (n * 12 coords);
// a hash lookup (-1 when absent); the valid edges (quads); the quads' voxels in the edges' order; two triangles per quad
void mesh_vertices(const int* coords, const float* feats, int n, float vox, float aabb0, float margin, float* v);
void mesh_edge_queries(const int* coords, int n, int* q);
void sp_lookup(const int* q, int nq, const unsigned long long* keys, const int* vals, unsigned int tsize, int* out);
void mesh_edge_valid(const float* feats, int n, const int* look, int* valid);
void mesh_quads(const int* valid, const int* pos, const int* look, int ne, int* quads);
void mesh_triangles(const int* quads, int L, const float* feats, int* faces);

// DINOv3's rotary embedding on the patch tokens (sd_gpu.hip dino_rope): x [T][C] with the heads of width D, the first `prefix` tokens
// untouched; cs, sn [P][D] (P = T - prefix) are the cosines and sines of each patch's angles.
void dino_rope(float* x, int T, int prefix, int C, int H, int D, const float* cs, const float* sn);

// mesh simplification (sd_gpu.hip simp_*): the quadric of each vertex (qem [V][10]); the cost of each edge collapse (cost [E]); the
// cheapest collapse around each face (prop [F], packed); the collapses under thresh (vkept [V], fkept [F] set to 0 for the removed); the
// compaction by the maps of an exclusive scan of the kept flags (vmap [V+1], fmap [F+1]).
void simp_qem(const float* v, const int* f, const int* v2f, const int* v2f_off, int V, float* qem);
void simp_cost(const float* v, const int* f, const int* v2f, const int* v2f_off, const unsigned long long* edges, const unsigned char* vb,
               const float* qem, int E, float lam_len, float lam_sk, float* cost);
void simp_propagate(const unsigned long long* edges, const int* v2f, const int* v2f_off, const float* cost, int E, unsigned long long* prop);
void simp_collapse(float* v, int* f, const unsigned long long* edges, const int* v2f, const int* v2f_off, const float* cost,
                   const unsigned long long* prop, const unsigned char* vb, int E, float thresh, int* vkept, int* fkept);
void simp_compress_v(const int* vmap, const float* oldv, int V, float* newv);
void simp_compress_f(const int* fmap, const int* vmap, const int* oldf, int F, int* newf);

// mesh charts (sd_gpu.hip chart_*; trellis_charts.cpp drives them): the face areas and normals; the cone of each chart (cn [C][4]: axis, half
// angle) and area from its faces (coff, cface); the perimeter of each chart from its chart-graph edges (c2e_off, c2e); the cost of each
// chart-graph edge (ek: min << 32 | max; elen its length); the cheapest edge of each chart (prop); the collapse (cmap, flag); the refinement.
void chart_face_geom(const float* v, const int* f, int F, float* fn, float* fa);
void chart_cones(const int* coff, const int* cface, const float* fn, const float* fa, int C, float* cn, float* carea);
void chart_perims(const int* c2e_off, const int* c2e, const float* elen, int C, float* perim);
void chart_costs(const unsigned long long* ek, const float* elen, const float* cn, const float* carea, const float* perim, float aw, float pw, int E,
                 float* cost);
void chart_propagate(const int* c2e_off, const int* c2e, const float* cost, int C, unsigned long long* prop);
void chart_collapse(const unsigned long long* ek, const float* cost, const unsigned long long* prop, float thresh, int E, int* cmap, float* cn,
                    int* flag);
void chart_refine(const float* cn, const float* fn, const float* v, const unsigned long long* ek, const int* f2e, const int* e2f,
                  const int* e2f_off, int F, float lam, const int* ids, int* out);

}  // namespace sddev

#endif  // SD_DEVICE_H
