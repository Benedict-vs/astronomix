#!/bin/bash
#SBATCH --job-name=fvmn_smoke
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=gpu-h200
#SBATCH --time=00:30:00
#SBATCH --exclusive

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_fvmn_%j.out
#SBATCH --error=cgols_fvmn_%j.err

# First multi-node (one process per GPU, srun) run of the FV (VL2) scheme, and
# the measurements that set the FV-2048 time limit (run_horeka_fv2048.sh).
# 2 nodes x 4 H200 on gpu-h200 (NOT dev-gpu-h100: its node breaks every
# multi-GPU run). Steps, each under its own timeout, in priority order:
#
#   v128_s1 / v128_s8   correctness: model A 128^3 to 7.5 Myr (past the 5 Myr
#                       onset), 1 GPU vs 8 GPUs over 2 nodes split (1, 8, 1, 1)
#                       - the 8-way split that needs 4 ghost cells (auto)
#   b1024_s421          bench, 1024^3 split (1, 4, 2, 1) over 2 nodes: the SAME
#                       per-device kernel layout as 2048 at (1, 8, 4, 1)
#                       (x 257 + halo, y 514 + halo) with half the z extent, so
#                       2048 s/step ~ 2 x this and 2048 memory ~ 2 x this
#                       compiled size (exactly 2x in the CPU AOT compile)
#   b1024_s8            bench, 1024^3 split (1, 8, 1, 1): the 4-ghost-cell path
#                       on the GPU/Triton backend
#   ckpt_io             Orbax save + restore of 2048-sized shards (43 GB per
#                       node per checkpoint, as in the 2048 run)
#   b1024_s221_1n       bench, 1 node (1, 2, 2, 1) multi-process: the FV-1024
#                       layout, the baseline for the multi-node overhead and
#                       for the bench -> production factor (FV-1024 0.523 s/step)
#   v128_s421           correctness for the (1, 4, 2, 1) split (2 ghost cells)
#
# Benches run CGOLS_BENCH_STEPS fixed steps at a CFL-safe dt (5e-4 code, ~1/3
# of the initial CFL step at 1024) via CGOLS_END_FRACTION. The default bench
# (END_FRACTION 1, dt = t_end / steps ~ 1000x CFL) makes every stage fail
# positivity and re-run the first-order flux correction up to 3 times: on CPU
# 4.0x the time per step - an FV bench with the default dt is not a timing.
#
# Run from the code COPY of the fv-ppm commit (PYTHONPATH=$REPO shadows the
# editable install; the preflight's astronomix: line shows which was used):
#
#   cd ~/astronomix-ppm/pytests/self_gravity/cgols && sbatch run_horeka_fv_multinode_smoke.sh
#
# Everything lands in cgols_logs/fvmn_<jobid>/ (summary.txt = the short answer,
# rewritten after every step so a timeout still leaves one).

export CGOLS_SITE="${CGOLS_SITE:-horeka2}"
source "$SLURM_SUBMIT_DIR/_site_env.sh" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cgols_preflight
[ -f "$REPO/COMMIT" ] && echo "code copy of: $(cat "$REPO/COMMIT")"
cgols_check_workspace || exit 1
echo "Running on nodes: $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION)"

cd "$CGOLS_DIR" || exit 1
OUT=cgols_logs/fvmn_${SLURM_JOB_ID}
mkdir -p "$OUT"
export CGOLS_SOLVER=fv CGOLS_RIEMANN=hllc CGOLS_FV_RECON=plm CGOLS_IC_MODE=insitu \
       CGOLS_CHECKPOINT_EVERY=0 CGOLS_RESTART_FROM=""
BENCH_STEPS="${SMOKE_BENCH_STEPS:-50}"
# dt = t_end / steps = 5e-4 code; t_end(75 Myr) = 7.670 code.
BENCH_FRACTION=$(awk -v n="$BENCH_STEPS" 'BEGIN{printf "%.6g", 5e-4 * n / 7.670}')

run(){   # $1 name, $2 time limit, $3 nodes, $4 tasks, rest = env assignments
  local name=$1 limit=$2 nodes=$3 tasks=$4; shift 4
  echo "=== $(date +%T) $name: $tasks task(s) on $nodes node(s), $*"
  # nvidia-smi on every node of the step (--overlap: next to the solver step).
  srun --overlap --nodes="$nodes" --ntasks-per-node=1 --gpu-bind=none bash -c \
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 1000 \
     > $OUT/$name.smi_\$(hostname)" &
  local smi=$!
  local cmd=(python cgols_vl2_probe.py cgols.py)
  [ -n "${SCRIPT:-}" ] && cmd=(python "$SCRIPT")
  env "$@" timeout "$limit" srun --nodes="$nodes" --ntasks="$tasks" \
      --ntasks-per-node=$(( (tasks + nodes - 1) / nodes )) --gpu-bind=none --kill-on-bad-exit=1 \
      "${cmd[@]}" > "$OUT/$name.log" 2>&1
  local rc=$?
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  echo "rc=$rc" >> "$OUT/$name.log"
  echo "    rc=$rc  $(grep -m1 -A3 'Compiled memory usage PER DEVICE' "$OUT/$name.log" | grep 'Total size')"
  summary
}

summary(){
  {
    echo "== fvmn_${SLURM_JOB_ID}: FV multi-node smoke, $SLURM_JOB_NODELIST, bench $BENCH_STEPS steps at dt 5e-4"
    printf "%-15s %-6s %-30s %13s %13s %9s  %s\n" run rc "probe (on_shard/unsharded)" compiled_MiB smi_peak_MiB s/step ghost
    for f in "$OUT"/*.log; do
      [ -e "$f" ] || continue
      local n; n=$(basename "$f" .log)
      [ "$n" = ckpt_io ] && continue
      local probe elapsed sps ghost peak
      probe=$(grep -m1 '\[probe\]' "$f" | sed -n "s/.*on_shard': \([0-9]*\), 'pallas_local_builds_unsharded': \([0-9]*\).*/\1\/\2/p")
      elapsed=$(grep -m1 'Time elapsed' "$f" | sed -n 's/.*elapsed: \([0-9.]*\) seconds.*/\1/p')
      sps=$([ -n "$elapsed" ] && awk -v e="$elapsed" -v n="$BENCH_STEPS" 'BEGIN{printf "%.3f", e / n}')
      ghost=$(grep -m1 '^\[fv\] [0-9]* ghost cells' "$f" | awk '{print $2}')
      peak=$(cat "$OUT/$n".smi_* 2>/dev/null | awk -F', ' '$2>m{m=$2}END{print m}')
      printf "%-15s %-6s %-30s %13s %13s %9s  %s\n" "$n" "$(grep '^rc=' "$f" | tail -1 | cut -d= -f2)" \
        "${probe:--}" \
        "$(grep -m1 -A3 'Compiled memory usage PER DEVICE' "$f" | sed -n 's/.*Total size: \([0-9.]*\).*/\1/p')" \
        "${peak:--}" "${sps:--}" "${ghost:-min}"
    done
    echo
    echo "== correctness (8 GPUs over 2 nodes vs 1 GPU, 128^3, 7.5 Myr)"
    [ -f "$OUT/compare.txt" ] && cat "$OUT/compare.txt"
    echo
    echo "== checkpoint I/O (2048-sized shards, 43 GB per node)"
    [ -f "$OUT/ckpt_io.log" ] && grep '\[ckpt-io\]' "$OUT/ckpt_io.log"
    echo
    echo "== FV-2048 at (1, 8, 4, 1) from b1024_s421 (same per-device x/y layout, 2x z):"
    echo "   memory ~ 2 x compiled_MiB(b1024_s421); s/step ~ 2 x s/step(b1024_s421) + extra inter-node"
    echo "   halo (every x face crosses nodes at 8 nodes, half of them here); production/bench factor"
    echo "   = 0.523 / s/step(b1024_s221_1n) (FV-1024 production incl. FOFC, frames, checkpoints)."
  } > "$OUT/summary.txt"
}

V128="CGOLS_DIM=128 CGOLS_END_FRACTION=0.1 CGOLS_NUM_SNAPSHOTS=10"
B1024="CGOLS_DIM=1024 CGOLS_BENCH_STEPS=$BENCH_STEPS CGOLS_END_FRACTION=$BENCH_FRACTION CGOLS_RUN_TAG=_fvmn_bench"

# shellcheck disable=SC2086
{
run v128_s1 6m 1 1 $V128 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" CGOLS_RUN_TAG=_fvmn_v128_s1
run v128_s8 6m 2 8 $V128 CGOLS_SHARD_SPLIT="(1, 8, 1, 1)" CGOLS_RUN_TAG=_fvmn_v128_s8
python cgols_fv_validate_compare.py --pair _fvmn_v128_s1 _fvmn_v128_s8 > "$OUT/compare.txt" 2>&1
summary
run b1024_s421 7m 2 8 $B1024 CGOLS_SHARD_SPLIT="(1, 4, 2, 1)"
run b1024_s8 7m 2 8 $B1024 CGOLS_SHARD_SPLIT="(1, 8, 1, 1)"
SCRIPT=cgols_ckpt_io_bench.py run ckpt_io 6m 2 8 CKPT_BENCH_SPLIT="(1, 4, 2, 1)"
run b1024_s221_1n 6m 1 4 $B1024 CGOLS_SHARD_SPLIT="(1, 2, 2, 1)"
run v128_s421 5m 2 8 $V128 CGOLS_SHARD_SPLIT="(1, 4, 2, 1)" CGOLS_RUN_TAG=_fvmn_v128_s421
python cgols_fv_validate_compare.py --pair _fvmn_v128_s1 _fvmn_v128_s8 _fvmn_v128_s1 _fvmn_v128_s421 \
  > "$OUT/compare.txt" 2>&1
summary
}
cat "$OUT/summary.txt"
