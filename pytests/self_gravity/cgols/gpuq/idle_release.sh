#!/bin/bash
# Safety net for HOLD=1 queues: once pending/ and running/ have been empty for
# IDLE_MIN minutes in a row (counted after the first job started), remove HOLD
# so gpuq.sh releases the GPU.   GPU=2 [GPUQ_DIR=...] [IDLE_MIN=45] bash idle_release.sh &
GPU=${GPU:?set GPU=<index>}; Q=${GPUQ_DIR:-$HOME/.gpuq/gpu$GPU}; IDLE_MIN=${IDLE_MIN:-45}; idle=0
until grep -q " start " "$Q/gpuq.log" 2>/dev/null; do sleep 60; done
while [ -e "$Q/HOLD" ]; do
  if [ -z "$(ls "$Q/pending" "$Q/running" 2>/dev/null | grep '\.sh$')" ]; then idle=$((idle+1)); else idle=0; fi
  if [ $idle -ge "$IDLE_MIN" ]; then rm -f "$Q/HOLD"; echo "$(date -Is) idle_release: queue empty ${IDLE_MIN} min, HOLD removed" >> "$Q/gpuq.log"; exit 0; fi
  sleep 60
done
