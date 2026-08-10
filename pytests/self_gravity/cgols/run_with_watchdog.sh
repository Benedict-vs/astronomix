#!/usr/bin/env bash
#
# Launch a cgols run under a stall watchdog.
#
#     ./run_with_watchdog.sh <logfile> <command...>
#
# WHY: JAX preallocates ~95% of every visible GPU at initialisation, so a run
# that hangs AFTER init and BEFORE the first step keeps two H200s reserved at 0%
# utilisation for as long as nobody notices. That happened on 2026-08-02: the
# A-series timing run deadlocked in the NCCL clique rendezvous
# ("Acquire clique: devices=2:[0,1] ... not all of them arrived on time") and
# sat there for 21h45m on 5m42s of CPU time, holding 184 GB across GPUs 0 and 2.
# XLA's own warn-stuck path fired at 10 s but its terminate path never did, and
# there is no scheduler on this node to enforce a wall-time limit.
#
# The watchdog polls the log's modification time. A cgols run writes a [diag]
# line every step, so a log that stops growing means a stalled run - whatever
# the cause (NCCL deadlock, silent XLA stall, a wedged host callback). On a
# stall the whole process GROUP is killed, which releases the GPU memory.
#
# STALL_TIMEOUT must comfortably exceed the longest legitimate silence, which is
# XLA compilation at the start of a run (minutes at 512^2x1024). Default 30 min.
# Waiting for free GPUs is NOT a stall: autocvd prints a spinner to the log
# every 30 s, so the log keeps growing and the watchdog stays quiet however long
# the queue takes.

set -uo pipefail

STALL_TIMEOUT=${STALL_TIMEOUT:-1800}   # seconds without log growth before killing
POLL_INTERVAL=${POLL_INTERVAL:-60}

if [ $# -lt 2 ]; then
    echo "usage: $0 <logfile> <command...>" >&2
    exit 2
fi

LOG="$1"; shift
mkdir -p "$(dirname "$LOG")"

# setsid puts the run in its own process group so the watchdog can take down
# the whole tree (python plus any NCCL/compile helper threads) in one signal.
setsid "$@" > "$LOG" 2>&1 &
RUN_PID=$!
PGID=$(ps -o pgid= -p "$RUN_PID" | tr -d ' ')

echo "run pid $RUN_PID (pgid $PGID) -> $LOG"
echo "watchdog: kills the run after ${STALL_TIMEOUT}s without log growth"

(
    last_size=-1
    last_change=$(date +%s)
    while kill -0 "$RUN_PID" 2>/dev/null; do
        sleep "$POLL_INTERVAL"
        size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
        now=$(date +%s)
        if [ "$size" != "$last_size" ]; then
            last_size=$size
            last_change=$now
            continue
        fi
        stalled=$((now - last_change))
        if [ "$stalled" -ge "$STALL_TIMEOUT" ]; then
            echo "[watchdog] no log growth for ${stalled}s - killing pgid $PGID" | tee -a "$LOG"
            kill -TERM -"$PGID" 2>/dev/null
            sleep 20
            kill -KILL -"$PGID" 2>/dev/null
            echo "[watchdog] killed; GPUs released" | tee -a "$LOG"
            exit 1
        fi
    done
    echo "[watchdog] run exited on its own; standing down"
) &
WATCHDOG_PID=$!

echo "watchdog pid $WATCHDOG_PID"
echo
echo "watch progress with:  tail -f $(readlink -f "$LOG")"
echo "stop everything with: kill -TERM -$PGID; kill $WATCHDOG_PID"
