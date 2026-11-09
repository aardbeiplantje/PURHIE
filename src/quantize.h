// quantize.h -- weight quantization on the CPU (the encoders; quant_types.h has the decoders).
//
// Rows of any type quant_types.h decodes are re-encoded as F32, F16, BF16, Q8_0, Q4_0_ROCMFP4 or
// Q4_0_ROCMFP4_FAST. Every encoder reproduces its reference bit for bit (tests/quantize_verify.cpp):
// ggml's quantize_row_q8_0_ref and fp32 -> fp16 / bf16 rounding, and ROCmFPX's rocmfp4.c (the exhaustive
// UE4M3 scale search without an importance matrix). No ggml / llama.cpp code is linked.
// Compiled with -ffp-contract=off: the scale search compares sums of squares, and a fused multiply-add
// would round them differently from the reference and pick another scale on a tie.

#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1 when rows can be encoded as dst_type
int qz_supported(int dst_type);

// Encode nrows rows of ncols values: src (src_type, rows back to back) -> dst (dst_type, rows back to back,
// qt_row_size(dst_type, ncols) bytes each). src_type == dst_type copies. n_threads <= 0: all cores.
// Returns 0, or -1 with a message in err.
int qz_quantize(int dst_type, void *dst, int src_type, const void *src, int64_t nrows, int64_t ncols,
                int n_threads, char *err, size_t err_len);

// One block / row encoders (the reference's semantics; exposed for the tests)
void qz_row_q8_0(const float *x, void *y, int64_t n);
void qz_row_rocmfp4(const float *x, void *y, int64_t n);
void qz_row_rocmfp4_fast(const float *x, void *y, int64_t n);
uint16_t qz_fp32_to_fp16(float f);
uint16_t qz_fp32_to_bf16(float f);

#ifdef __cplusplus
}
#endif

#endif // QUANTIZE_H
