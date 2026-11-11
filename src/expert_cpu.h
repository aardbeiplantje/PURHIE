// expert_cpu.h -- an expert's FFN on the CPU, read in place from the mapped model: for the experts of a model
// larger than memory that are not registered with the GPU (expert_pager.h). ROCmFP4_FAST weights x Q8_0
// activations, as the GPU's int8 kernels (matvec_q.hip): exact integer dot products per 32-value block times the
// block's weight scale and activation scale; the blocks' float sum is in another order than the GPU's, and SiLU's
// exp is the CPU's, so the result equals the GPU's to rounding, not bit for bit. AVX-512 VNNI (Zen 4 / 5), with a
// scalar path for other CPUs.

#ifndef EXPERT_CPU_H
#define EXPERT_CPU_H

#include <cstddef>
#include <cstdint>
#include <functional>

// x[K] as Q8_0, as the GPU's producers quantize (q8_device.h): per 32 values d = F16(amax / 127),
// q = round-half-even(x * 127 / amax). K % 32 == 0
void cpu_q8_0_quantize(const float* x, int K, int8_t* q, float* d);

// y[N] = W x: N rows of K ROCmFP4_FAST values (row_bytes each; K % 64 == 0), x as Q8_0 (q[K], d[K / 32])
void cpu_matvec_fp4fast_q8(const void* W, size_t row_bytes, int N, int K, const int8_t* q, const float* d, float* y);

// One expert on one token: out[E] = down . (SiLU(min(gate . x, L)) * clamp(up . x, -L, L)) (L 0: no clamp).
// gate / up: FE rows of E values at row_bytes_gu; down: E rows of FE values at row_bytes_d; x as Q8_0 (q[E],
// d[E / 32]). work: 3 * FE floats + FE bytes
void cpu_expert_ffn(const void* gate, const void* up, size_t row_bytes_gu, const void* down, size_t row_bytes_d,
                    int FE, int E, const int8_t* q, const float* d, float limit, float* out, float* work);

// fn(i) for i in 0 .. n on the pool's threads (INFERENCE_CPU_THREADS, default the cores less one, at most 16) and
// the caller's; returns when all are done
void cpu_parallel_for(int n, const std::function<void(int)>& fn);

#endif
