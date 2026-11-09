#!/bin/sh
# convert_verify.sh [dir]: bin/rocm-convert against llama.cpp's convert_hf_to_gguf.py, byte for byte.
# 1. Qwen3-0.6B (huggingface.co/Qwen/Qwen3-0.6B, BF16 safetensors), a tiny random GLM-5.3-Flash checkpoint
#    (tools/gen_glm5next_ckpt.py: FP8 weights with 128x128 block scales, KDA + sparse MLA + MoE + MTP layers) and a
#    tiny random DeepSeek-V4-Flash (tools/gen_dsv4_ckpt.py: DeepSeek's layout, FP8 with E8M0 scales, packed FP4
#    experts, hash routing, KV compressors), converted as BF16 by both: the same tensors (names, types, shapes,
#    bytes) and metadata (model-card keys aside).
# 2. The tiny GLM to FAST directly = to F32, then re-encoded from that GGUF: FP8 decoded and encoded in one step, rounded once.
# Needs: CONVERT_UPSTREAM (a llama.cpp checkout with convert_hf_to_gguf.py and gguf-py; default
# /data/conv/upstream), CONVERT_PYTHON (a Python with torch, transformers >= 5 and safetensors; default
# /data/hf/venv/bin/python), CONVERT_PYTHONPATH (modules ahead of it: a transformers that knows glm5_next and
# deepseek_v4, 5.22 here; default /data/conv/pylib), CONVERT_DATA (default /data/conv: qwen3-0.6b/ a clone of
# Qwen/Qwen3-0.6B, glm53-meta/ and dsv4-meta/ GLM-5.3-Flash's and DeepSeek-V4-Flash's config and tokenizer files),
# the Perl module built.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE/..
UP=${CONVERT_UPSTREAM:-/data/conv/upstream}
PY=${CONVERT_PYTHON:-/data/hf/venv/bin/python}
DATA=${CONVERT_DATA:-/data/conv}
PYP=${CONVERT_PYTHONPATH-/data/conv/pylib}
PL="perl -I$ROOT/perl/ROCm-Inference/blib/lib -I$ROOT/perl/ROCm-Inference/blib/arch"
DIR=${1:-$DATA/convert_verify.$$}
mkdir -p "$DIR"
rc=0
up() { PYTHONPATH="$PYP:$UP:$UP/gguf-py" "$PY" "$UP/convert_hf_to_gguf.py" "$1" --outtype bf16 --outfile "$2" > "$2.log" 2>&1 || { tail -5 "$2.log"; rc=1; }; }
"$PY" -I "$ROOT/tools/gen_glm5next_ckpt.py" "$DATA/glm53-meta" "$DIR/glm-tiny"
"$PY" -I "$ROOT/tools/gen_dsv4_ckpt.py" "$DATA/dsv4-meta" "$DIR/ds-tiny"
for m in "$DATA/qwen3-0.6b:qwen3" "$DIR/glm-tiny:glm" "$DIR/ds-tiny:deepseek4"; do
    src=${m%%:*}; tag=${m##*:}
    up "$src" "$DIR/$tag-ref.gguf"
    $PL "$ROOT/perl/ROCm-Inference/bin/rocm-convert" -q -f "$src" "$DIR/$tag-ours.gguf" bf16-ref || rc=1
    printf '%s, BF16: ' "$tag"
    $PL "$HERE/gguf_compare.pl" --ignore-card "$DIR/$tag-ours.gguf" "$DIR/$tag-ref.gguf" || rc=1
done
$PL "$ROOT/perl/ROCm-Inference/bin/rocm-convert" -q -f "$DIR/glm-tiny" "$DIR/glm-f32.gguf" f32 || rc=1
$PL "$ROOT/perl/ROCm-Inference/bin/rocm-convert" -q -f "$DIR/glm-f32.gguf" "$DIR/glm-f32-fast.gguf" fast || rc=1
$PL "$ROOT/perl/ROCm-Inference/bin/rocm-convert" -q -f "$DIR/glm-tiny" "$DIR/glm-fast.gguf" fast || rc=1
printf 'glm, FP8 -> FAST directly vs F32 -> GGUF re-encoded: '
$PL "$HERE/gguf_compare.pl" "$DIR/glm-fast.gguf" "$DIR/glm-f32-fast.gguf" || rc=1
find "$DIR" -delete
[ $rc = 0 ] && echo "PASS: rocm-convert = convert_hf_to_gguf.py, byte for byte" || echo "FAIL: rocm-convert"
exit $rc
