#!/bin/bash
DOCKER_IMAGE=${DOCKER_IMAGE:-local/ai/purhie-gfx1151:latest}
HERE="$BASH_SOURCE"
HERE="${HERE%/*}"

# Switch case?
DOCKER_OPTS=${DOCKER_OPTS:-""}
case "${1:-}" in
    rocm-server)
        D=purhie
        DOCKER_OPTS="$DOCKER_OPTS -d"
        ;;
    *)
        D=purhie
        DOCKER_OPTS="$DOCKER_OPTS -d"
        ;;
esac

# Clean up any existing container
docker stop $D >/dev/null 2>&1 || true
while [ "$(docker ps -a -q -f name=^$D$)" ]; do
    docker rm $D >/dev/null 2>&1 || true
    sleep 1
done

# Now run
# ZIMAGE_DIR set: the Z-Image files are mounted at /zimage and the server gets --zimage /zimage (unless ROCM_ARGS has one)
if [ -n "${ZIMAGE_DIR:-}" ] && [[ " ${ROCM_ARGS:-} " != *" --zimage "* ]]; then
    ROCM_ARGS="${ROCM_ARGS:-} --zimage /zimage"
fi
exec docker run \
    --restart=always \
    --name $D \
    --network=host \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    --ulimit nofile=65536:65536 \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --group-add=109 \
    --group-add=986 \
    --group-add=992 \
    --device /dev/kfd \
    --device /dev/dri \
    --tmpfs /tmp:rw,suid,exec,size=1G \
    --tmpfs /var/tmp:rw,suid,exec,size=1G \
    $([ -n "$MODELS_DIR"    ] && echo "-v $MODELS_DIR:/models:rw") \
    $([ -n "$ROCM_STORE_DIR" ] && echo "-v $ROCM_STORE_DIR:/kvstore:rw") \
    $([ -n "$HF_HOME" ] && echo "-v $HF_HOME:/hf:rw") \
    $([ -n "$ZIMAGE_DIR" ] && echo "-v $ZIMAGE_DIR:/zimage:ro") \
    -e XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm} \
    -e HF_HOME=${HF_HOME:-/hf} \
    -e HF_TOKEN=${HF_TOKEN:-} \
    -e HF_HUB_CACHE=${HF_HUB_CACHE:-/hf/hub} \
    -e MODEL=${MODEL:-} \
    -e MODEL_PATH=${MODEL_PATH:-/models} \
    -e MODELS_DIR=${MODELS_DIR:-} \
    -e MODELS=${MODELS:-} \
    -e LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm/lib:/app/bin} \
    -e HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.5.1} \
    -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=${GGML_CUDA_ENABLE_UNIFIED_MEMORY:-1} \
    -e GGML_HIP_FORCE_RS_GPU=${GGML_HIP_FORCE_RS_GPU:-1} \
    -e GGML_HIP_FORCE_KV_GPU=${GGML_HIP_FORCE_KV_GPU:-1} \
    -e GGML_HIP_ALLOC_GRAPH_RESERVE=${GGML_HIP_ALLOC_GRAPH_RESERVE:-2048} \
    $([ -n "$HSA_FORCE_FINE_GRAIN_PCIE" ] && echo "-e HSA_FORCE_FINE_GRAIN_PCIE=$HSA_FORCE_FINE_GRAIN_PCIE") \
    $([ -n "$HIP_LAUNCH_BLOCKING" ] && echo "-e HIP_LAUNCH_BLOCKING=$HIP_LAUNCH_BLOCKING") \
    $([ -n "$AMD_SERIALIZE_KERNEL" ] && echo "-e AMD_SERIALIZE_KERNEL=$AMD_SERIALIZE_KERNEL") \
    $([ -n "$QWEN3TTS_PROFILE" ] && echo "-e QWEN3TTS_PROFILE=$QWEN3TTS_PROFILE") \
    -e HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0} \
    -e ROCM_MODEL=${ROCM_MODEL:-} \
    -e ROCM_MODELS_DIR=${ROCM_MODELS_DIR:-} \
    -e ROCM_INFERENCE=${ROCM_INFERENCE:-1} \
    -e ROCM_CTX=${ROCM_CTX:-} \
    -e ROCM_DRAFT=${ROCM_DRAFT:-} \
    -e ROCM_STORE=${ROCM_STORE:-$([ -n "$ROCM_STORE_DIR" ] && echo /kvstore)} \
    -e ROCM_ARGS="${ROCM_ARGS:-}" \
    -e INFERENCE_EXPERT_PIN_MB=${INFERENCE_EXPERT_PIN_MB:-65536} \
    $([ -n "$QMATVEC_PREFILL_SPLIT_VARIANT" ] && echo "-e QMATVEC_PREFILL_SPLIT_VARIANT=$QMATVEC_PREFILL_SPLIT_VARIANT") \
    $DOCKER_OPTS \
        $DOCKER_IMAGE \
            "$@"
