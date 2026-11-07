#!/bin/bash
# profile_wmma.sh -- Quick profiling script for WMMA kernels
#
# Usage:
#   ./profile_wmma.sh <model.gguf> <prompt> [output_dir]
#
# Example:
#   ./profile_wmma.sh /models/Qwen3.8-27B.gguf "Hello world" ./profile_out

set -e

MODEL="${1:?Usage: $0 <model.gguf> <prompt> [output_dir]}"
PROMPT="${2:?Missing prompt}"
OUTPUT_DIR="${3:-./profile_output}"

echo "=== ROCmFP4_FAST WMMA Kernel Profiling ==="
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_DIR"
echo

# Check rocprofv3
if ! command -v rocprofv3 &> /dev/null; then
    echo "ERROR: rocprofv3 not found in PATH"
    echo "Install from: https://github.com/ROCm/rocprofiler-core"
    exit 1
fi

# Check for qwen_demo
DEMO=$(find . -name "qwen_demo" -type f 2>/dev/null | head -1)
if [ -z "$DEMO" ]; then
    echo "ERROR: qwen_demo not found"
    echo "Build with: make"
    exit 1
fi

echo "Using: $DEMO"
echo

# Run with profiling
mkdir -p "$OUTPUT_DIR"

export QWEN_PROFILE=1

echo "Running profiling..."
echo

# Profile the run
rocprofv3 --kernel-trace -d "$OUTPUT_DIR" -- \
    "$DEMO" "$MODEL" "$PROMPT"

echo
echo "Profiling complete. Analyzing..."
echo

# Run analysis
python3 "$(dirname "$0")/rocminfo_wmma.py" "$OUTPUT_DIR" --all

echo
echo "=== Per-section profile (QWEN_PROFILE) ==="
echo "(Check stdout above for [profile] lines)"
