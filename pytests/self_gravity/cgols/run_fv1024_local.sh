#!/bin/bash
# FV-1024 (VL2 + HLLC) on 4 local H200s, collected with gpuq/gather.sh while
# the collected cards run the FD/FV seed-scatter runs (CGOLS_IC_SEED) as
# fillers: 256^3 first, 128^3 last (the short "9*" jobs that may finish once
# all four cards are held). HoreKa job $HKJOB (the queued twin) is kept until
# the local run is stepping, then cancelled with one squeue + scancel.
#
#   bash run_fv1024_local.sh start      # fill the queue, start gather detached
#   bash run_fv1024_local.sh final      # (called by gather on the 4 cards)
#
# Queue state and logs: $GDIR (gather.log, logs/<job>.log, logs/final.log).
# Seed runs land as cgols_snapshots_<tag> / cgols_logs/cgols_diag_<tag>.log
# (tags _A<dim>s<k> = FD, _A<dim>fvs<k> = FV); move them to the diagnostics
# layout once done. More fillers (e.g. FV-PPM) = more scripts in $GDIR/pending.
C=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
GDIR=${GDIR:-$HOME/.gpuq/gather_fv1024}
HKJOB=106337
SEEDS="1 2 3"

job(){   # $1 = queue name, $2 = solver (fd|fv), $3 = dim, $4 = seed
  local tag="_A$3$([ "$2" = fv ] && echo fv)s$4"
  cat > "$GDIR/pending/$1.sh" <<EOF
#!/bin/bash
cd "$C" || exit 1
PYTHONPATH=$C/gpuq/autocvd_shim XLA_PYTHON_CLIENT_PREALLOCATE=false \\
CGOLS_SOLVER=$2 CGOLS_DIM=$3 CGOLS_IC_SEED=$4 CGOLS_RUN_TAG=$tag \\
CGOLS_IC_MODE=insitu CGOLS_CHECKPOINT_EVERY=0 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" \\
  python cgols.py > cgols_logs/run${tag}_console.log 2>&1
EOF
}

case "${1:-}" in
start)
  mkdir -p "$GDIR/pending"
  n=100; for s in $SEEDS; do job $((n+1))_A256fvs$s fv 256 $s; job $((n+2))_A256s$s fd 256 $s; n=$((n+2)); done
  n=900; for s in $SEEDS; do job $((n+1))_A128fvs$s fv 128 $s; job $((n+2))_A128s$s fd 128 $s; n=$((n+2)); done
  ls "$GDIR/pending"
  GDIR=$GDIR NEED=4 FINAL="$C/run_fv1024_local.sh" FINAL_ARG=final DEADLINE="2026-11-02 12:00" \
    setsid nohup bash "$C/gpuq/gather.sh" > /dev/null 2>&1 < /dev/null &
  echo "gather started (pid $!), log $GDIR/gather.log"
  ;;
final)
  SSH=(ssh -o ControlPath=$HOME/.ssh/cm-horeka -o BatchMode=yes -o ConnectTimeout=20 hd_ah321@hk2-x86.scc.kit.edu)
  st=$("${SSH[@]}" "squeue -h -j $HKJOB -o %T" 2>/dev/null); rc=$?
  if [ $rc -ne 255 ] && [ "$st" != PENDING ]; then
    echo "$(date -Is) HoreKa $HKJOB is '${st:-gone}' - not running FV-1024 locally"; exit 0
  fi
  [ $rc -eq 255 ] && echo "$(date -Is) HoreKa unreachable before launch - launching anyway"
  cd "$C" || exit 1
  CONSOLE=cgols_logs/run_A1024fv_console.log
  PYTHONPATH=$C/gpuq/autocvd_shim \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 \
  JAX_USE_SHARDY_PARTITIONER=false XLA_FLAGS=--xla_gpu_shard_autotuning=false PYTHONUNBUFFERED=1 \
  CGOLS_SOLVER=fv CGOLS_RIEMANN=hllc CGOLS_DIM=1024 CGOLS_RUN_TAG=_A1024fv CGOLS_IC_MODE=insitu \
  CGOLS_SHARD_SPLIT="(1, 2, 2, 1)" CGOLS_CHECKPOINT_EVERY=1 CGOLS_CHECKPOINT_KEEP=2 CGOLS_CHECKPOINT_ARCHIVE_EVERY=24 \
    python cgols.py > $CONSOLE 2>&1 < /dev/null &
  RUN=$!; echo "$(date -Is) FV-1024 launched pid $RUN on cards $CUDA_VISIBLE_DEVICES"
  until grep -q "\[diag\]" $CONSOLE 2>/dev/null || ! kill -0 $RUN 2>/dev/null; do sleep 30; done
  if kill -0 $RUN 2>/dev/null; then
    echo "$(date -Is) local FV-1024 is stepping"
    st=$("${SSH[@]}" "squeue -h -j $HKJOB -o %T" 2>/dev/null); rc=$?
    if [ $rc -eq 255 ]; then echo "$(date -Is) HoreKa UNREACHABLE - $HKJOB NOT cancelled (scancel by hand)"
    elif [ "$st" = PENDING ]; then "${SSH[@]}" "scancel $HKJOB" && echo "$(date -Is) HoreKa $HKJOB cancelled"
    else echo "$(date -Is) HoreKa $HKJOB is '${st:-gone}' - left alone"; fi
  else
    echo "$(date -Is) local FV-1024 died before its first step - HoreKa $HKJOB kept; see $CONSOLE"
  fi
  wait $RUN; echo "$(date -Is) FV-1024 exited rc=$?"
  ;;
*) echo "usage: $0 start|final" >&2; exit 2 ;;
esac
