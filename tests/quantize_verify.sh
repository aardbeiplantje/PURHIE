#!/bin/sh
# quantize_verify.sh <model.gguf> [dir]: the engine's quantizer (rocm-convert with a GGUF in) against the reference's, byte for byte.
# 1. build/quantize_verify: the encoders (src/quantize.cpp) on synthetic rows and the model's tensors.
# 2. bin/rocm-convert -r plain (a GGUF in) (every matrix the target type, as llama-quantize writes it) against ROCmFPX's llama-quantize (llama.cpp/build-cpu, the test oracle) on the whole model,
#    as Q4_0_ROCMFP4_FAST, Q4_0_ROCMFP4 (llama-quantize --pure: its Q4_0_ROCMFP4 file type mixes in K-quants) and
#    Q8_0: the same tensors (types, shapes, bytes) and metadata (tests/gguf_compare.pl).
# Needs the oracle (make quantize_verify) and the Perl module built (perl/ROCm-Inference: perl Makefile.PL && make).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE/..
REF=${REF_LLAMA_B:-$ROOT/llama.cpp/build-cpu}/bin/llama-quantize
PL="perl -I$ROOT/perl/ROCm-Inference/blib/lib -I$ROOT/perl/ROCm-Inference/blib/arch"
DIR=${2:-/dev/shm/quantize_verify.$$}
mkdir -p "$DIR"
rc=0
"$ROOT/build/quantize_verify" "$1" || rc=1
for t in q4_0_rocmfp4_fast:Q4_0_ROCMFP4_FAST: q4_0_rocmfp4:Q4_0_ROCMFP4:--pure q8_0:Q8_0:; do
    ours=${t%%:*}; rest=${t#*:}; ref=${rest%%:*}; flag=${rest#*:}
    $PL "$ROOT/perl/ROCm-Inference/bin/rocm-convert" -q -f -r plain "$1" "$DIR/ours.gguf" "$ours" || rc=1
    "$REF" $flag --allow-requantize "$1" "$DIR/ref.gguf" "$ref" 16 > "$DIR/ref.log" 2>&1 || { tail -3 "$DIR/ref.log"; rc=1; }
    printf '%s: ' "$ours"
    $PL "$HERE/gguf_compare.pl" "$DIR/ours.gguf" "$DIR/ref.gguf" || rc=1
    find "$DIR" -name '*.gguf' -delete
done
find "$DIR" -delete
[ $rc = 0 ] && echo "PASS: rocm-convert (GGUF in) = llama-quantize, byte for byte" || echo "FAIL: rocm-convert (GGUF in)"
exit $rc
