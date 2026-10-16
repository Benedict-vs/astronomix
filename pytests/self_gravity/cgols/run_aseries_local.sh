#!/usr/bin/env bash
#
# Adiabatic A-series reruns on the local node, one GPU each, one after the
# other, on the current code (two-sided HSE initial conditions, sharded-stencil
# and boundary fixes). Tags follow the A-series naming: _A<dim>.
#
#   nohup ./run_aseries_local.sh > cgols_logs/aseries_local.log 2>&1 &   # 128 256 512
#   nohup ./run_aseries_local.sh 512 > cgols_logs/aseries_local.log 2>&1 &
#
# CGOLS_IC_MODE=insitu builds the initial conditions in-process: the
# data/initial/*_d<dim>.npy files predate the two-sided HSE fix (ef6b3f0) and
# must not be reused. CGOLS_CHECKPOINT_KEEP=0 keeps the full state of every
# frame (61 per run: 5 GB at 128, 41 GB at 256, 330 GB at 512). autocvd
# waits until a GPU is completely free, so a busy node just delays the start.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p cgols_logs

for dim in "${@:-128 256 512}"; do
    for d in $dim; do
        log="cgols_logs/run_A${d}_console.log"
        echo "$(date -Is) A-${d}: start -> $log"
        CGOLS_DIM="$d" CGOLS_RUN_TAG="_A${d}" CGOLS_IC_MODE=insitu \
        CGOLS_CHECKPOINT_KEEP=0 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" \
            python cgols.py > "$log" 2>&1 && rc=0 || rc=$?
        echo "$(date -Is) A-${d}: exit $rc"
    done
done
