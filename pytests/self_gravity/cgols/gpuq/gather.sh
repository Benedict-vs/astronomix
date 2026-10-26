#!/bin/bash
# Collect NEED fully free GPUs of the SHARED compgpu12 node for one multi-GPU
# run, keeping every collected card busy with real single-GPU work meanwhile.
#
# HOUSE RULES (memory notes local-gpus-free-only / gpu-hold-queue): only FULLY
# free cards (no compute process, <1 GB), never piggyback, and a collected card
# must never sit idle at 0% - so a card is only taken while the filler queue
# has work for it, and its worker releases it when the queue runs dry.
#
# How it works: every POLL seconds (30 = autocvd's cadence) free cards are
# taken on first sighting, each with a worker that runs the filler jobs of
# $GDIR/pending/ (bash scripts, name order, atomic claim) one after another
# with CUDA_VISIBLE_DEVICES set to its card. For the whole time a card is held
# it also carries the tiny holder context of gpuq.sh (holder.py, ~0.5 GB), so
# the start-up gap between two fillers never shows it as free. Once NEED cards
# are held, long fillers are stopped at once and re-queued, short ones (names
# starting with "9", a few minutes) may finish; then the
# FINAL script runs with CUDA_VISIBLE_DEVICES=<the NEED cards> and gather exits
# after it. A short filler still running after MAX_WAIT seconds is stopped too.
#
#   GDIR=~/.gpuq/gather_fv1024 NEED=4 FINAL=/path/final.sh \
#       setsid nohup bash gather.sh > /dev/null 2>&1 < /dev/null &
#
#   GDIR       state dir (pending/ running/ done/ failed/ logs/, gather.log)
#   NEED       number of cards the FINAL run needs
#   FINAL      script of the multi-GPU run (FINAL_ARG: optional argument)
#   POLL       seconds between nvidia-smi checks (default 30)
#   MAX_WAIT   seconds to wait for short fillers once NEED cards are held
#              (default 900)
#   DEADLINE   "YYYY-MM-DD HH:MM": give up (stop fillers, release) after it
#
# While $GDIR/NO_FINAL exists the FINAL launch is on hold: the cards only run
# fillers (and are released when the queue runs dry). Remove it to arm FINAL.
#
#   REFILL     optional script (+ REFILL_ARG): while FINAL is armed and fewer
#              than NEED cards are held, a worker that finds the queue empty
#              runs it to add more fillers instead of releasing its card
#
# Filler jobs must not select a GPU themselves (autocvd would wait on our own
# card): put gpuq/autocvd_shim first on PYTHONPATH.
HERE=$(dirname "$(readlink -f "$0")")
GDIR=${GDIR:?set GDIR}; NEED=${NEED:?set NEED}; FINAL=${FINAL:?set FINAL}
POLL=${POLL:-30}; MAX_WAIT=${MAX_WAIT:-900}
DEADLINE_S=$([ -n "${DEADLINE:-}" ] && date -d "$DEADLINE" +%s || echo 99999999999)
mkdir -p "$GDIR"/{pending,running,done,failed,logs}
log(){ echo "$(date -Is) $*" >> "$GDIR/gather.log"; }
declare -A WORKER   # card -> worker pid
declare -A HOLDER   # card -> holder pid (lives as long as the card is held)

worker(){   # $1 = card; runs fillers until the queue is empty or STOP is set
  local g=$1 job rc
  while true; do
    [ -e "$GDIR/STOP" ] && return 0
    job=$(ls "$GDIR/pending" 2>/dev/null | sort | head -1)
    if [ -z "$job" ] && [ -n "${REFILL:-}" ] && [ ! -e "$GDIR/NO_FINAL" ]; then
      if mkdir "$GDIR/refill.lock" 2>/dev/null; then   # one worker refills at a time
        bash "$REFILL" ${REFILL_ARG:-} >> "$GDIR/logs/refill.log" 2>&1
        rmdir "$GDIR/refill.lock"; log "GPU $g queue empty -> refilled: $(ls "$GDIR/pending" | paste -sd' ')"
      else
        sleep 2
      fi
      continue
    fi
    [ -z "$job" ] && return 0
    mv "$GDIR/pending/$job" "$GDIR/running/$job" 2>/dev/null || continue
    log "GPU $g start $job"
    CUDA_VISIBLE_DEVICES=$g setsid bash "$GDIR/running/$job" > "$GDIR/logs/${job%.sh}.log" 2>&1 < /dev/null &
    echo $! > "$GDIR/running/$job.pid"; wait $!; rc=$?
    rm -f "$GDIR/running/$job.pid"
    if [ -e "$GDIR/running/$job.stopped" ]; then   # stopped by gather: re-queue
      rm -f "$GDIR/running/$job.stopped"; mv "$GDIR/running/$job" "$GDIR/pending/"
      log "GPU $g stopped + re-queued $job"
    else
      if [ $rc -eq 0 ]; then mv "$GDIR/running/$job" "$GDIR/done/"; else mv "$GDIR/running/$job" "$GDIR/failed/"; fi
      log "GPU $g end $job rc=$rc"
    fi
  done
}

stop_fillers(){   # stop running fillers (whole process groups); their workers re-queue them
  local f pid pattern=${1:-}   # $1 = name pattern to stop (default: all)
  for f in "$GDIR"/running/*.pid; do
    [ -e "$f" ] || continue
    [ -n "$pattern" ] && ! basename "$f" | grep -q "$pattern" && continue
    touch "${f%.pid}.stopped"; pid=$(cat "$f"); kill -- -"$pid" 2>/dev/null
  done
  sleep 20
  for f in "$GDIR"/running/*.pid; do
    [ -e "$f" ] || continue
    [ -n "$pattern" ] && ! basename "$f" | grep -q "$pattern" && continue
    kill -9 -- -"$(cat "$f")" 2>/dev/null
  done
}
start_holder(){ CUDA_VISIBLE_DEVICES=$1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
  setsid python "$HERE/holder.py" >> "$GDIR/logs/holder.log" 2>&1 < /dev/null & HOLDER[$1]=$!; }
stop_holders(){ local g; for g in "${!HOLDER[@]}"; do kill "${HOLDER[$g]}" 2>/dev/null; done; sleep 5; }
trap 'log "gather interrupted"; touch "$GDIR/STOP"; stop_fillers; stop_holders; exit 1' INT TERM

rm -f "$GDIR/STOP"
log "gather start pid $$: NEED=$NEED FINAL=$FINAL POLL=$POLL"
while true; do
  if [ "$(date +%s)" -ge "$DEADLINE_S" ]; then
    log "deadline - stopping fillers, releasing cards"; touch "$GDIR/STOP"; stop_fillers; stop_holders; exit 0
  fi
  for g in "${!WORKER[@]}"; do
    kill -0 "${HOLDER[$g]}" 2>/dev/null || { start_holder "$g"; log "GPU $g holder restarted (${HOLDER[$g]})"; }
    kill -0 "${WORKER[$g]}" 2>/dev/null && continue
    # drain phase: the holder keeps the finished card until FINAL
    [ -e "$GDIR/STOP" ] && continue
    # the queue ran dry: release the card
    kill "${HOLDER[$g]}" 2>/dev/null; unset "WORKER[$g]" "HOLDER[$g]"; log "GPU $g released (queue empty)"
  done
  if [ ${#WORKER[@]} -lt "$NEED" ] && [ ! -e "$GDIR/STOP" ]; then
    for g in $("$HERE/free_gpus.sh"); do
      [ ${#WORKER[@]} -ge "$NEED" ] && break
      [ -n "${WORKER[$g]+x}" ] && continue
      ls "$GDIR/pending" 2>/dev/null | grep -q . || break   # no work -> don't take cards
      start_holder "$g"; worker "$g" & WORKER[$g]=$!; log "GPU $g taken (holder ${HOLDER[$g]}, worker ${WORKER[$g]})"
    done
  fi
  if [ ${#WORKER[@]} -ge "$NEED" ] && [ ! -e "$GDIR/STOP" ] && [ ! -e "$GDIR/NO_FINAL" ]; then
    touch "$GDIR/STOP"; STOP_T=$(date +%s); CARDS=$(echo "${!WORKER[@]}" | tr ' ' '\n' | sort -n | paste -sd,)
    log "NEED=$NEED cards held ($CARDS) - stopping long fillers, letting short (9*) ones finish"
    stop_fillers '^[0-8]'
  fi
  if [ -e "$GDIR/STOP" ] && [ -n "${STOP_T:-}" ]; then
    busy=$(ls "$GDIR/running" 2>/dev/null | grep -c '\.sh$')
    if [ "$busy" -eq 0 ] || [ $(( $(date +%s) - STOP_T )) -ge "$MAX_WAIT" ]; then
      [ "$busy" -gt 0 ] && { log "MAX_WAIT reached"; stop_fillers; }
      for g in "${!WORKER[@]}"; do wait "${WORKER[$g]}" 2>/dev/null; done
      stop_holders
      log "launching FINAL on cards $CARDS"
      CUDA_VISIBLE_DEVICES=$CARDS bash "$FINAL" ${FINAL_ARG:-} > "$GDIR/logs/final.log" 2>&1 < /dev/null; rc=$?
      log "FINAL exited rc=$rc - gather done"; exit 0
    fi
  fi
  sleep "$POLL"
done
