#!/bin/bash
# image-watch.sh [-i SECONDS] [--pull] IMAGE -- COMMAND [ARGS...]
#
# Runs COMMAND in its own process group and watches the image ID that "docker image inspect" gives for IMAGE (the
# local tag). When the image changes, COMMAND is stopped and this exits 0: the caller runs it again (serve.sh does, so
# its edited version takes effect with the new image). A COMMAND that ends on its own is reported once, and the watch
# goes on until the image changes. Ctrl-C stops both.
#
#   ./image-watch.sh local/ai/purhie-gfx1151:latest -- bash run.sh
#   ./image-watch.sh -i 60 IMAGE -- COMMAND ...     check every 60 seconds (default 30)
#   ./image-watch.sh --pull IMAGE -- COMMAND ...    docker pull before each check (a registry image)
interval=30
pull=0
while [ $# -gt 0 ]; do
    case $1 in
        -i) interval=$2; shift 2 ;;
        --pull) pull=1; shift ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
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
    if [ -n "$new" ] && [ "$new" != "$sha" ]; then
        if [ -n "$sha" ]; then
            echo "== $(date '+%F %T') image $image changed: $sha -> $new, stopping the command"
            stop
            exit 0
        fi
        echo "== $(date '+%F %T') image $image: $new"
        sha=$new
        start "$@"
    fi
    # the command ended on its own: reported once; it runs again when the image changes
    if [ -n "$pid" ]; then
        state=$(ps -o stat= -p "$pid" 2>/dev/null)
        if [ -z "$state" ] || [ "${state:0:1}" = Z ]; then
            wait "$pid"; rc=$?
            pid=
            echo "== $(date '+%F %T') command exited, exit $rc (this watch does not run it again: it exits when the image changes, and serve.sh runs it then)"
        fi
    fi
    sleep "$interval" &
    wait $!
done
