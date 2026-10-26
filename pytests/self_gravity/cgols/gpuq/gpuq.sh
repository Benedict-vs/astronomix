#!/bin/bash
# GPU hold queue for the SHARED compgpu12 node (8x H200): run a series of
# single-GPU jobs back to back on ONE fully free GPU without losing the card
# to another user between jobs.
#
# HOUSE RULE: we normally do NOT reserve GPUs. Use this only when the user has
# explicitly authorised holding a GPU for a defined set of jobs, take only a
# FULLY free card (no compute process, <1 GB used; this script refuses
# otherwise), and release it as soon as the jobs are done (the default: the
# queue exits and frees the GPU when pending/ is empty). Never piggyback on a
# busy card. See the memory note gpu-hold-queue.
#
# How it works: a tiny idle JAX process (holder.py, ~0.5 GB, preallocation
# off) keeps a compute context on the GPU for the whole lifetime of the
# queue, so "no compute process" checks (autocvd and friends) see the card as
# busy between our jobs. Jobs are bash scripts dropped into $GPUQ_DIR/pending/
# and run one at a time in name order with CUDA_VISIBLE_DEVICES=$GPU and
# GPUQ_GPU=$GPU; their output goes to $GPUQ_DIR/logs/<job>.log, and they move
# to done/ or failed/ by exit code. The event log is $GPUQ_DIR/gpuq.log.
#
#   GPU=2 setsid nohup bash gpuq.sh > /dev/null 2>&1 < /dev/null &
#   cp myjob.sh $GPUQ_DIR/pending/010_myjob.sh      # before or while it runs
#
#   GPU          index of the (fully free) GPU to hold          (required)
#   GPUQ_DIR     queue state dir, default $HOME/.gpuq/gpu$GPU  (NOT in the repo)
#   HOLD=1       keep the GPU while pending/ is empty (discouraged; pair it with
#                idle_release.sh, and remove $GPUQ_DIR/HOLD to release)
#   $1           optional PID of OUR process that currently uses the GPU: wait
#                for it to exit instead of requiring a free card (handover)
#
# Jobs inside the queue must NOT select a GPU themselves: autocvd would wait
# forever on our own holder. Put autocvd_shim/ first on PYTHONPATH (no-op
# autocvd) - CUDA_VISIBLE_DEVICES is already set. 512^3 CGOLS runs near the
# H200 limit also want XLA_PYTHON_CLIENT_PREALLOCATE=false
# XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.98.
# Several queues on several GPUs can share a job list safely if each job takes
# an atomic lock first:  mkdir "$LOCKS/<run>" 2>/dev/null || exit 0
HERE=$(dirname "$(readlink -f "$0")")
GPU=${GPU:?set GPU=<index>}; WAITPID=${1:-}
Q=${GPUQ_DIR:-$HOME/.gpuq/gpu$GPU}
mkdir -p "$Q"/{pending,running,done,failed,logs}
[ "${HOLD:-0}" = 1 ] && touch "$Q/HOLD"
log(){ echo "$(date -Is) $*" >> "$Q/gpuq.log"; }

if [ -z "$WAITPID" ] && ! "$HERE/free_gpus.sh" | grep -qx "$GPU"; then
    log "GPU $GPU is not fully free - refusing to start"; echo "GPU $GPU is not fully free" >&2; exit 1
fi
start_holder(){ CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  setsid python "$HERE/holder.py" >> "$Q/logs/holder.log" 2>&1 < /dev/null & HOLDER=$!; log "holder started pid $HOLDER on GPU $GPU"; }
start_holder
trap 'kill $HOLDER 2>/dev/null; log "gpuq exit, holder stopped, GPU $GPU released"' EXIT
if [ -n "$WAITPID" ]; then log "waiting for pid $WAITPID to free GPU $GPU"; while kill -0 "$WAITPID" 2>/dev/null; do sleep 1; done; log "pid $WAITPID exited"; fi
while true; do
  kill -0 $HOLDER 2>/dev/null || { log "holder died, restarting"; start_holder; }
  job=$(ls "$Q/pending" 2>/dev/null | sort | head -1)
  if [ -n "$job" ]; then
    mv "$Q/pending/$job" "$Q/running/$job" || continue
    log "start $job"
    CUDA_VISIBLE_DEVICES=$GPU GPUQ_GPU=$GPU bash "$Q/running/$job" > "$Q/logs/${job%.sh}.log" 2>&1; rc=$?
    if [ $rc -eq 0 ]; then mv "$Q/running/$job" "$Q/done/"; else mv "$Q/running/$job" "$Q/failed/"; fi
    log "end $job rc=$rc"
  elif [ -e "$Q/HOLD" ]; then sleep 5
  else log "queue empty, releasing GPU $GPU"; exit 0; fi
done
