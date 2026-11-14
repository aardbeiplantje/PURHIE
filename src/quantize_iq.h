/* quantize_iq.h -- the IQ encoders (ggml's reference algorithms, no llama.cpp linked): IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
 * IQ3_S, IQ1_S, IQ1_M. Each super-block of 256 values is searched against the codebook (its neighbour table built on first
 * use). The reference needs an importance matrix for some of them: qw holds one weight per value of the row (uniform
 * weights when the caller has none). */
#ifndef QUANTIZE_IQ_H
#define QUANTIZE_IQ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when qz_iq_encode encodes t */
int qz_iq_supported(int t);

/* n values (a multiple of 256) of x -> one row of t's blocks in y (qt_row_size(t, n) bytes); qw: n weights */
void qz_iq_encode(int t, const float *x, void *y, int64_t n, const float *qw);

#ifdef __cplusplus
}
#endif

#endif /* QUANTIZE_IQ_H */
