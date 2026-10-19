#!/usr/bin/env bash
#
# CGOLS series reruns on the local node, one GPU each, one after the other, on
# the current code (two-sided HSE initial conditions, sharded-stencil and
# boundary fixes). Tags follow the naming _<series><dim> (_A512, _B256, ...).
#
#   nohup ./run_series_local.sh > cgols_logs/series_local.log 2>&1 &            # A: 128 256 512
#   CGOLS_SERIES=B nohup ./run_series_local.sh > cgols_logs/series_local_B.log 2>&1 &
#   nohup ./run_series_local.sh 512 > cgols_logs/series_local.log 2>&1 &        # one size
#
# A = adiabatic (the paper's A-series). B = radiative: cooling on with the
# settings of the B-512 runs (run_bseries_local.sh), the 10% cooling-dt rule
# off. Both use cgols.py's positivity-preserving WENO + dual energy.
#
# CGOLS_IC_MODE=insitu builds the initial conditions in-process: files from
# before the two-sided HSE fix (ef6b3f0) must not be reused. CGOLS_CHECKPOINT_KEEP=0
# keeps the full state of every frame (61 per run: 5 GB at 128, 41 GB at 256,
# 330 GB at 512). autocvd waits until a GPU is completely free, so a busy node
# just delays the start.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p cgols_logs

SERIES="${CGOLS_SERIES:-A}"
case "$SERIES" in
    A) ;;
    B) export CGOLS_COOLING=1 CGOLS_COOLING_DT_LIMIT=0 CGOLS_COOLING_SUBCYCLES=32 ;;
    *) echo "CGOLS_SERIES must be A or B, got '$SERIES'" >&2; exit 1 ;;
esac

for dim in "${@:-128 256 512}"; do
    for d in $dim; do
        log="cgols_logs/run_${SERIES}${d}_console.log"
        echo "$(date -Is) ${SERIES}-${d}: start -> $log"
        CGOLS_DIM="$d" CGOLS_RUN_TAG="_${SERIES}${d}" CGOLS_IC_MODE=insitu \
        CGOLS_CHECKPOINT_KEEP=0 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" \
            python cgols.py > "$log" 2>&1 && rc=0 || rc=$?
        echo "$(date -Is) ${SERIES}-${d}: exit $rc"
    done
done
