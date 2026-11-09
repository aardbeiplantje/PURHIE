#!/bin/sh
# mtp_own_verify.sh <qwen4exp.gguf> [dir]: a model's own MTP block as its draft head (tests/mtp_own_verify.cpp), on
# tiny random models tools/gen_qwen4exp_tiny.py writes from the header of <qwen4exp.gguf> (any Qwen3.8-Flash-Next
# GGUF: only its tokenizer is read): one with an MTP block and the same block as a sidecar, one without; the model
# as mapped, then staged.
set -e
HERE=$(dirname "$0")
BIN=$HERE/../build/mtp_own_verify
DIR=${2:-/dev/shm/mtp_own_verify.$$}
mkdir -p "$DIR/stage"
TINY_MTP="$DIR/head.gguf" python3 "$HERE/../tools/gen_qwen4exp_tiny.py" "$1" "$DIR/mtp.gguf" 3
python3 "$HERE/../tools/gen_qwen4exp_tiny.py" "$1" "$DIR/plain.gguf" 4
rc=0
"$BIN" "$DIR/mtp.gguf" "$DIR/head.gguf" "$DIR/plain.gguf" || rc=1
MTP_OWN_STAGE="$DIR/stage" "$BIN" "$DIR/mtp.gguf" "$DIR/head.gguf" || rc=1
find "$DIR" -delete
[ $rc = 0 ] && echo "PASS: own MTP block, mapped and staged" || echo "FAIL: own MTP block"
exit $rc
