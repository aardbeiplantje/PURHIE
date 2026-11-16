#!/bin/bash
# serve.sh: this checkout's container (run.sh), restarted when the image changes (image-watch.sh). Run it from any place:
# run.sh and image-watch.sh are the ones next to this script.
#
#   ./serve.sh                  the defaults below, the container in the foreground; Ctrl-C stops it
#   DRY_RUN=1 ./serve.sh        print the command that would run, and exit
#
# Defaults for this host, each set in the environment to change it:
#   ROCM_IMAGE      the image to watch: what the build tags (default local/ai/purhie-gfx1151:latest)
#   INTERVAL        seconds between the image checks (default 10)
#   PULL            1: docker pull before each check, for a registry image (default 0: the local build)
#   ROCM_ARGS       the rocm-inference-server's options (default: --store-max 32 --parallel 4)
#   ROCM_STORE_DIR  the KV block store on the host (default /mnt/data3/rocm-kvstore)
#   MODELS_DIR      the models on the host (default /mnt/data3/models)
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE=${ROCM_IMAGE:-local/ai/purhie-gfx1151:latest}
INTERVAL=${INTERVAL:-10}
PULL=${PULL:-0}
export ROCM_ARGS=${ROCM_ARGS:---store-max 32 --parallel 4}
export ROCM_STORE_DIR=${ROCM_STORE_DIR:-/mnt/data3/rocm-kvstore}
export MODELS_DIR=${MODELS_DIR:-/mnt/data3/models}

pull_opt=
[ "$PULL" = 1 ] && pull_opt=--pull

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "$HERE/image-watch.sh -i $INTERVAL $pull_opt $IMAGE -- bash $HERE/run.sh"
    exit 0
fi
exec "$HERE/image-watch.sh" -i "$INTERVAL" $pull_opt "$IMAGE" -- bash "$HERE/run.sh"
