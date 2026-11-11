#!/bin/sh
# glm5next_head_verify.sh <slice.gguf>: GLM-5.3-Flash on real weights - the first blocks of the release as a model
# (tools/gguf_head.pl /models/GLM-5.3-Flash-FAST.gguf <slice.gguf> 8: 3 dense KDA blocks, then MoE blocks, two
# of them MLA with the indexer; 21.5 GB), since the whole file does not fit the memory. tests/glm5next_verify.cpp:
# prefill = token by token bit for bit (QMATVEC_NO_WMMA=1), and on the float path the GPU = the CPU forward pass
# (src/glm5_ref.cpp) at every step. Minutes: the CPU pass takes ~4 s a token.
rc=0
LOG=$(mktemp)
run() {  # the program's own exit status (not the filter's)
    env "$@" > "$LOG" 2>&1 || rc=1
    grep -v '^\[LOAD\]\|^\[KV_INIT\]' "$LOG"
}
run QMATVEC_NO_WMMA=1 "$(dirname "$0")/../build/glm5next_verify" "$1"
run QWEN_MATVEC_FP32=1 "$(dirname "$0")/../build/glm5next_verify" "$1"
rm -f "$LOG"
[ $rc = 0 ] && echo "PASS: glm5-next head" || echo "FAIL: glm5-next head"
exit $rc
