#!/bin/bash
LLAMA_DOCKER_IMAGE=${LLAMA_DOCKER_IMAGE:-local/ai/llama.cpp-gfx1151:latest}
HERE="$BASH_SOURCE"
HERE="${HERE%/*}"

# Switch case?
DOCKER_OPTS=${DOCKER_OPTS:-""}
case "${1:-}" in
    rocm-server)
        D=llama
        DOCKER_OPTS="$DOCKER_OPTS -d"
        ;;
    quantize|cli|bench)
        D=$LOGNAME-${1}
        DOCKER_OPTS="$DOCKER_OPTS -it"
        ;;
    convert)
        D=$LOGNAME-${1}
        DOCKER_OPTS="$DOCKER_OPTS -it"
        ;;
    bash)
        D=$LOGNAME-${1}
        DOCKER_OPTS="$DOCKER_OPTS -it"
        ;;
    server)
        D=llama
        if [ "$*" != "" ]; then
            DOCKER_OPTS="$DOCKER_OPTS -it"
        else
            DOCKER_OPTS="$DOCKER_OPTS -d"
        fi
        ;;
    tail)
        # Just tail the logs, don't start the container
        docker logs -f llama
        exit $?
        ;;
    *)
        D=llama
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
    $([ -n "$LLAMA_PRESETS" ] && echo "-v $LLAMA_PRESETS:/llamacpp_presets.ini:ro") \
    $([ -n "$ROCM_STORE_DIR" ] && echo "-v $ROCM_STORE_DIR:/kvstore:rw") \
    -v llama.cpp-data:/llama.cpp:rw \
    $([ -n "$HF_HOME" ] && echo "-v $HF_HOME:/hf:rw") \
    -e XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm} \
    -e HF_HOME=${HF_HOME:-/hf} \
    -e HF_TOKEN=${HF_TOKEN:-} \
    -e HF_HUB_CACHE=${HF_HUB_CACHE:-/hf/hub} \
    -e MODEL=${MODEL:-} \
    -e MODEL_PATH=${MODEL_PATH:-/models} \
    -e MODELS_DIR=${MODELS_DIR:-} \
    -e LLAMA_PRESETS=${LLAMA_PRESETS:-} \
    -e PRESET_FILE=${PRESET_FILE:-/llamacpp_presets.ini} \
    -e MODELS=${MODELS:-} \
    -e LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/rocm/lib:/llama/bin} \
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
    $([ -n "$QMATVEC_PREFILL_SPLIT_VARIANT" ] && echo "-e QMATVEC_PREFILL_SPLIT_VARIANT=$QMATVEC_PREFILL_SPLIT_VARIANT") \
    $DOCKER_OPTS \
        $LLAMA_DOCKER_IMAGE \
            "$@"
