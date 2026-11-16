#!/bin/bash
# Watches the sha of a docker image. When it changes, the command is restarted
# with the new image. The command runs in its own process group, so a restart
# (and Ctrl-C) also stops the processes it started. If the command exits by
# itself, it is not restarted until the image changes.
#
#   ./image-watch.sh IMAGE -- COMMAND [ARGS...]
#   ./image-watch.sh -i 60 IMAGE -- COMMAND [ARGS...]   check every 60 seconds (default 30)
#   ./image-watch.sh --pull IMAGE -- COMMAND ...        docker pull before each check (a registry image)
#
#   ./image-watch.sh aicli:dev -- ./aicli.sh
#
# The sha is the image ID (sha256 of the image config) that "docker image inspect"
# gives for the local tag. Stop with Ctrl-C.
interval=30
pull=0
while [ $# -gt 0 ]; do
    case $1 in
        -i) interval=$2; shift 2 ;;
        --pull) pull=1; shift ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        --) shift; break ;;
        -*) echo "image-watch: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
if [ $# -lt 2 ]; then
    echo "usage: $0 [-i SECONDS] [--pull] IMAGE -- COMMAND [ARGS...]" >&2
    exit 2
fi
image=$1
shift
[ "$1" = "--" ] && shift
if [ $# -eq 0 ]; then
    echo "image-watch: no command given" >&2
    exit 2
fi

pid=
stop() {
    if [ -n "$pid" ]; then
        kill -TERM -- "-$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        pid=
    fi
}
start() {
    echo "== $(date '+%F %T') start: $*"
    set -m
    "$@" &
    pid=$!
    set +m
}
trap 'stop; echo; echo "image-watch: stopped"; exit 130' INT TERM

sha_of() {
    docker image inspect --format '{{.Id}}' "$image" 2>/dev/null
}

sha=
while :; do
    [ "$pull" = 1 ] && docker pull -q "$image" >/dev/null 2>&1
    new=$(sha_of)
    if [ -z "$new" ]; then
        if [ -n "$sha" ]; then
            echo "== $(date '+%F %T') image $image is gone, command kept running"
        fi
        sha=
    elif [ "$new" != "$sha" ]; then
        if [ -z "$sha" ]; then
            echo "== $(date '+%F %T') image $image: $new"
        else
            echo "== $(date '+%F %T') image $image changed: $sha -> $new, restarting"
            stop
        fi
        sha=$new
        start "$@"
    fi
    # the command ended on its own: report it once, keep watching the image
    state=
    [ -n "$pid" ] && state=$(ps -o stat= -p "$pid" 2>/dev/null)
    if [ -n "$pid" ] && { [ -z "$state" ] || [ "${state:0:1}" = Z ]; }; then
        wait "$pid"; rc=$?
        pid=
        echo "== $(date '+%F %T') command exited, exit $rc (restart on the next image change)"
    fi
    sleep "$interval" &
    wait $!
done
