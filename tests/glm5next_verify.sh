#!/bin/sh
# glm5next_verify.sh <glm53-meta dir> [dir]: GLM-5.3-Flash (glm5-next) on a tiny random checkpoint
# (tools/gen_glm5next_ckpt.py from the release's config and tokenizer in <glm53-meta dir>), converted to FAST by
# rocm-convert (the Perl module built): tests/glm5next_verify.cpp (now the loader; the forward pass against
# upstream llama.cpp's glm5-next once it is written, PLAN.md GLM-5.3-Flash). Needs CONVERT_PYTHON (a Python with
# numpy; default /data/hf/venv/bin/python).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE/..
PY=${CONVERT_PYTHON:-/data/hf/venv/bin/python}
DIR=${2:-/dev/shm/glm5next_verify.$$}
mkdir -p "$DIR"
"$PY" -I "$ROOT/tools/gen_glm5next_ckpt.py" "$1" "$DIR/GLM-tiny" > /dev/null
perl -I"$ROOT/perl/ROCm-Inference/blib/lib" -I"$ROOT/perl/ROCm-Inference/blib/arch" \
    "$ROOT/perl/ROCm-Inference/bin/rocm-convert" -q -f "$DIR/GLM-tiny" "$DIR/glm-tiny-fast.gguf"
rc=0
"$ROOT/build/glm5next_verify" "$DIR/glm-tiny-fast.gguf" | grep -v '^\[LOAD\]' || rc=1
find "$DIR" -delete
[ $rc = 0 ] && echo "PASS: glm5-next tiny" || echo "FAIL: glm5-next tiny"
exit $rc
