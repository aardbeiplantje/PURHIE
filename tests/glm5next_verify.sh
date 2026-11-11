#!/bin/sh
# glm5next_verify.sh <glm53-meta dir> [dir]: GLM-5.3-Flash (glm5-next) on a tiny random checkpoint
# (tools/gen_glm5next_ckpt.py from the release's config and tokenizer in <glm53-meta dir>), converted to FAST by
# rocm-convert (the Perl module built): tests/glm5next_verify.cpp, the loader, the GPU forward pass (prefill = token
# by token; then on the float path = the CPU one, src/glm5_ref.cpp), and with the oracle the CPU forward pass
# against upstream llama.cpp's glm5-next on that file decoded to F32 (its activations, ref_dump acts on a 50-token
# prompt). The oracle: build/ref_dump_stock linked to an upstream CPU build (make
# ref_dump_stock REF_STOCK=<llama.cpp checkout> REF_STOCK_B=<its build>; PLAN.md GLM-5.3-Flash), REF_DUMP_STOCK to
# name another; without it only the loader is checked. Needs CONVERT_PYTHON (a Python with numpy; default
# /data/hf/venv/bin/python).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE/..
PY=${CONVERT_PYTHON:-/data/hf/venv/bin/python}
DIR=${2:-/dev/shm/glm5next_verify.$$}
mkdir -p "$DIR"
"$PY" -I "$ROOT/tools/gen_glm5next_ckpt.py" "$1" "$DIR/GLM-tiny" > /dev/null
perl -I"$ROOT/perl/ROCm-Inference/blib/lib" -I"$ROOT/perl/ROCm-Inference/blib/arch" \
    "$ROOT/perl/ROCm-Inference/bin/rocm-convert" -q -f "$DIR/GLM-tiny" "$DIR/glm-tiny-fast.gguf"
ORACLE=${REF_DUMP_STOCK:-$ROOT/build/ref_dump_stock}
REFS=
if [ -x "$ORACLE" ]; then
    perl -I"$ROOT/perl/ROCm-Inference/blib/lib" -I"$ROOT/perl/ROCm-Inference/blib/arch" \
        "$ROOT/perl/ROCm-Inference/bin/rocm-convert" -q -f "$DIR/glm-tiny-fast.gguf" "$DIR/glm-tiny-f32.gguf" f32
    mkdir -p "$DIR/acts"
    "$ORACLE" acts "$DIR/glm-tiny-f32.gguf" "The quick brown fox jumps over the lazy dog while the five boxing wizards \
jump quickly. Pack my box with five dozen liquor jugs, and then sphinx of black quartz judge my vow as the night \
falls over the quiet old town by the river." "$DIR/acts" 8 > "$DIR/acts.log" 2>&1 || { tail -3 "$DIR/acts.log"; exit 1; }
    REFS=$DIR/acts
else
    echo "(no oracle at $ORACLE: the loader only)"
fi
rc=0
QMATVEC_NO_WMMA=1 "$ROOT/build/glm5next_verify" "$DIR/glm-tiny-fast.gguf" $REFS | grep -v '^\[LOAD\]\|^\[KV_INIT\]' || rc=1
QWEN_MATVEC_FP32=1 "$ROOT/build/glm5next_verify" "$DIR/glm-tiny-fast.gguf" | grep -v '^\[LOAD\]\|^\[KV_INIT\]' || rc=1
find "$DIR" -delete
[ $rc = 0 ] && echo "PASS: glm5-next tiny" || echo "FAIL: glm5-next tiny"
exit $rc
