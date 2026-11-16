#!/bin/bash
# Waits for a trigger file, then runs "docker buildx bake --pull". The output is
# shown here and appended to a log file, so it can be read with tail -f. The
# trigger file is removed when a build starts, so a touch during a build starts
# the next one. Stop with Ctrl-C.
#
#   ./build-watch.sh                        watches ./.build-trigger, logs to ./build-watch.log
#   ./build-watch.sh TRIGGER LOG            other files
#   touch .build-trigger                    (from another shell) starts a build
#   tail -f build-watch.log                 follows the output
trigger=${1:-.build-trigger}
log=${2:-build-watch.log}
echo "build-watch: waiting for $trigger (touch it to build, Ctrl-C to stop), log: $log"
while :; do
    if [ -e "$trigger" ]; then
        rm -f "$trigger"
        {
            echo "== $(date '+%F %T') docker buildx bake --pull"
            PROJECT_VERSION="$(git log -1 --date=iso-strict --format='%H %cd %s' 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')" \
            CACHEBUST=$(date +%s) docker buildx bake --pull 2>&1
            rc=$?
            echo "== $(date '+%F %T') done, exit $rc"
        } 2>&1 | tee -a "$log"
    fi
    sleep 2
done
