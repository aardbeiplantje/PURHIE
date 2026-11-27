#!/bin/bash
# serve.sh: this checkout's container (run.sh), replaced when the image changes. image-watch.sh stops its command and
# returns when a new image appears (a build of this checkout tags one); then this script runs again from the file on
# disk: run.sh runs again, which stops and replaces the container with the new image, and an edited serve.sh takes
# effect with the next build. It runs no docker commands itself: the container is run.sh's, the watch is image-watch.sh's.
# Run it from any place: the scripts next to it are used. Run this, not image-watch.sh alone: nothing else starts the
# container again after an image change.
#
#   ./serve.sh                  the defaults below, in the foreground; Ctrl-C stops it
#   DRY_RUN=1 ./serve.sh        print the commands that would run, and exit
#
# Defaults for this host, each set in the environment to change it:
#   ROCM_IMAGE      the image to watch: what the build tags (default local/ai/purhie-gfx1151:latest)
#   INTERVAL        seconds between the image checks (default 10)
#   PULL            1: docker pull before each check, for a registry image (default 0: the local build)
#   ROCM_ARGS       the rocm-inference-server's options (default: --store-max 32 --parallel 4)
#   ROCM_STORE_DIR  the KV block store on the host (default /mnt/data3/rocm-kvstore)
#   MODELS_DIR      the models on the host (default /mnt/data3/models)
#   RETRY           seconds before running again when image-watch fails (default 30)
# The body is one function: bash reads all of it before it runs anything, so editing this file while a run is
# waiting on the watcher cannot corrupt that run (the run ends with exec of this file, which reads the new version).
main() {
    set -u

    HERE="$(cd "$(dirname "$0")" && pwd)"
    IMAGE=${ROCM_IMAGE:-local/ai/purhie-gfx1151:latest}
    INTERVAL=${INTERVAL:-10}
    PULL=${PULL:-0}
    RETRY=${RETRY:-30}
    export ROCM_ARGS=${ROCM_ARGS:---store-max 32 --parallel 4}
    export ROCM_STORE_DIR=${ROCM_STORE_DIR:-/mnt/data3/rocm-kvstore}
    export MODELS_DIR=${MODELS_DIR:-/mnt/data3/models}

    pull_opt=
    [ "$PULL" = 1 ] && pull_opt=--pull

    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "$HERE/image-watch.sh -i $INTERVAL $pull_opt $IMAGE -- bash $HERE/run.sh"
        echo "then: bash $HERE/serve.sh"
        exit 0
    fi

    # Ctrl-C ends the watch: this script stops, instead of running again below
    trap 'exit 130' INT TERM

    "$HERE/image-watch.sh" -i "$INTERVAL" $pull_opt "$IMAGE" -- bash "$HERE/run.sh"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "serve: image-watch exited $rc, running again in ${RETRY}s"
        sleep "$RETRY"
    fi
    # the image changed: run this script again from the file on disk
    exec bash "$HERE/serve.sh"
}

main "$@"
