#!/bin/bash

# Usage:
#   app.sh [subcommand] [args...]
#   app.sh                          # server (default)
#   app.sh rocm-server [model.gguf] # ROCm::Inference only, at /rocm/v1/ (no model: every model in /models)

# --- Environment Setup ---
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export LD_LIBRARY_PATH=${ROCM_PATH}/lib
export PATH=${ROCM_PATH}/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HIP_VISIBLE_DEVICES=0
# HIP_LAUNCH_BLOCKING=1 / AMD_SERIALIZE_KERNEL=1 serialize every kernel launch (debugging, or a workaround for
# GPU hangs): off unless set (run.sh passes them through)
export ROCM_METADATA_WAIT_TIMEOUT=1000
export ROCM_ALLOW_INT8_MIXED_PRECISION=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_CONCURRENT_EXEC=1
# export HSA_FORCE_FINE_GRAIN_PCIE=1  # Can cause GPU lockups on some firmware
export AMD_LOG_LEVEL=0
export HSA_LOG_LEVEL=0
export HSA_DISABLE_CRASH_DUMP=1
export HSA_ENABLE_SDMA=0
export XDG_CACHE_HOME=/hf

# --- Mode-specific overrides ---
export ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT:-0}
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}

# --- ROCm::Inference (rocm-hip): rocm-inference-server, OpenAI-compatible, on /tmp/rocm-inference.sock, reached
# through nginx at /rocm/v1/... ROCM_MODEL=<model.gguf>: that model, loaded at start; else every model in
# ROCM_MODELS_DIR (default /models), listed at /rocm/v1/models and loaded when a request names it (nothing held
# until then) ---
ROCM_HIP=${ROCM_HIP:-/opt/rocm-hip}
start_rocm_inference() {
    local which
    if [ -n "${ROCM_MODEL:-}" ]; then
        local model=$ROCM_MODEL
        [ -f "$model" ] || model=${M_DIR:-/models}/$model
        which=(--model "$model")
    else
        which=(--models-dir "${ROCM_MODELS_DIR:-${M_DIR:-/models}}")
    fi
    # the large outputs (the 3D models) in the scratch tmpfs run.sh mounts at /scratch; nginx sends them (nginx.conf:
    # /rocm-files/). Without a writable /scratch the server keeps its own directory in /dev/shm
    local files=${ROCM_FILES:-/scratch}
    [ -d "$files" ] && [ -w "$files" ] || files=
    # never with the kernel-serializing debug switches
    env -u HIP_LAUNCH_BLOCKING -u AMD_SERIALIZE_KERNEL \
        PERL5LIB="$ROCM_HIP/perl5/lib/perl5${PERL5LIB:+:$PERL5LIB}" \
        perl "$ROCM_HIP/perl5/bin/rocm-inference-server" \
            "${which[@]}" \
            --listen /tmp/rocm-inference.sock \
            ${ROCM_CTX:+--ctx "$ROCM_CTX"} \
            ${ROCM_DRAFT:+--draft "$ROCM_DRAFT"} \
            ${ROCM_STORE:+--store "$ROCM_STORE"} \
            ${files:+--files "$files"} \
            ${ROCM_ARGS:-} &
    ROCM_PID=$!
}

# nginx in the foreground of this shell, which stays PID 1: docker stop's SIGTERM reaches only PID 1, and the
# processes left when it exits are SIGKILLed. So the signal goes on to nginx and the servers next to it
# (rocm-inference-server writes the KV store snapshots it holds in memory on SIGTERM), and this waits for them
serve_nginx() {
    set -x
    /usr/sbin/nginx -c /nginx.conf -g "daemon off;" &
    { set +x; } 2>/dev/null
    local nginx=$! rc
    trap 'kill -TERM $nginx ${ROCM_PID:-} 2>/dev/null' TERM INT
    wait $nginx; rc=$?
    while kill -0 $nginx 2>/dev/null; do wait $nginx; rc=$?; done  # (a signal ends a wait early)
    kill -TERM ${ROCM_PID:-} 2>/dev/null  # nginx is gone: the servers too
    wait
    exit $rc
}

# --- Subcommand dispatch ---
SUBCMD="${1:-rocm-server}"

case "$SUBCMD" in
    # ---- server (default) ----
    bash|sh)
        exec "$@"
        ;;

    # ---- rocm-server: ROCm::Inference only, at /rocm/v1/... ----
    rocm-server)
        shift
        ROCM_MODEL=${1:-${ROCM_MODEL:-}}
        start_rocm_inference
        serve_nginx
        ;;

    # ---- help ----
    --help|-h|help)
        echo "Usage: app.sh [subcommand] [args...]"
        echo ""
        echo "Subcommands:"
        echo "  rocm-server       ROCm::Inference only, at /rocm/v1/ (e.g. app.sh rocm-server Qwen3.5-4B-ROCMFP4.gguf;"
        echo "                    no model: every model in /models, loaded on request)"
        echo "  help              Show this help"
        echo ""
                ;;

    *)
        echo "Unknown subcommand: $SUBCMD" >&2
        echo "Run 'app.sh help' for usage." >&2
        exit 1
        ;;
esac
