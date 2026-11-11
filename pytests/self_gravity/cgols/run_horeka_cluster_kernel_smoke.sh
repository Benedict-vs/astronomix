#!/bin/bash
#SBATCH --job-name=ck_smoke
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev-gpu-h100
#SBATCH --time=01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_ck_%j.out
#SBATCH --error=cgols_ck_%j.err

# GPU test of the local-box cluster kernel (CGOLS_CLUSTER_KERNEL=local) before
# the C-1024 run with 1000 clusters. The kernel was validated on the CPU only
# (cell by cell against the full-grid kernel, also split over host devices and
# over two processes) plus a single-GPU timing of the kernel alone; this is the
# first run of the whole solver with it on GPUs and with one process per GPU
# (srun), the layout of the production run. Dev queue: 1 node, 4 x H100, 1 h.
# Steps, each under its own timeout, in priority order:
#
#   v256_full_s1 / v256_local_s1   model C at 256^3 to 8 Myr (3 Myr of
#                       feedback), N = 16, one GPU: full-grid vs local kernel
#   v256_full_s4 / v256_local_s4   the same with 4 processes, split
#                       (1, 2, 2, 1): the local kernel inside shard_map with
#                       one process per GPU (s2 / s8 with SMOKE_TASKS)
#   b512_local_n1000    bench, 512^3 split (1, 2, 2, 1), 1000 clusters
#                       (R = 30 pc = 1.5 dx, 7^3-cell cubes)
#   b512_full_n16       bench, the production kernel and cluster number so far
#   b512_local_n16      bench, same clusters with the local kernel
#   b512_local_n128     bench, 128 clusters: R = 3.05 dx and 10^3-cell cubes,
#                       the cube of the C-1024 run with 1000 clusters
#
# The benches run CGOLS_BENCH_STEPS fixed steps of dt = 1e-3 code (below the
# initial CFL step at 512) with the clusters alive from t = 0
# (CGOLS_CLUSTER_START_MYR=0): the injection rates are still zero there, but
# the kernel does all of its work.
#
# Run from a code COPY, never from a checkout that queued jobs import:
#
#   cd ~/astronomix-c1024n1000/pytests/self_gravity/cgols && sbatch run_horeka_cluster_kernel_smoke.sh
#
# The dev queue has ONE node and a job that asks for all 4 GPUs for an hour
# only starts once it is the top job and the node is empty (job 128481 waited
# > 6.5 h on 2026-11-11). Knobs for a lighter job that fits into gaps, read
# from the environment at sbatch time:
#
#   SMOKE_TASKS       processes (= GPUs) of the split runs: 4 (default), 2, 8
#   SMOKE_STEPS       steps to run, default "v256 n1000 full16 local16 n128"
#   SMOKE_TAG         extra run-tag part, so two smoke jobs never share frames
#   SMOKE_STEP_LIMIT  time limit per step, default 8m
#
#   SMOKE_TASKS=2 SMOKE_TAG=g2 SMOKE_STEPS="v256 n1000" SMOKE_STEP_LIMIT=6m \
#     sbatch --gres=gpu:2 --ntasks-per-node=2 --time=00:35:00 run_horeka_cluster_kernel_smoke.sh
#
# Everything lands in cgols_logs/ck_<jobid>/ (summary.txt = the short answer,
# rewritten after every step so a timeout still leaves one).

export CGOLS_SITE="${CGOLS_SITE:-horeka2}"
# Under sbatch this file runs as Slurm's spool copy, and with --chdir
# SLURM_SUBMIT_DIR is where sbatch was called, so ask Slurm for the script.
HERE="${SLURM_SUBMIT_DIR:-.}"
if [ ! -f "$HERE/_site_env.sh" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    _cmd="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=//p')"
    case "$_cmd" in /*) ;; *) _cmd="${SLURM_SUBMIT_DIR:-.}/$_cmd" ;; esac
    HERE="$(cd "$(dirname "$_cmd")" 2>/dev/null && pwd)"
fi
source "$HERE/_site_env.sh" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cgols_preflight
[ -f "$REPO/COMMIT" ] && echo "code copy of: $(cat "$REPO/COMMIT")"
cgols_check_workspace || exit 1
echo "Running on node: $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION)"

cd "$CGOLS_DIR" || exit 1
OUT=cgols_logs/ck_${SLURM_JOB_ID}
mkdir -p "$OUT"
export CGOLS_FEEDBACK=clusters CGOLS_IC_MODE=insitu CGOLS_CHECKPOINT_EVERY=0 \
       CGOLS_RESTART_FROM="" CGOLS_EMISSION_FRAMES=0
BENCH_STEPS="${SMOKE_BENCH_STEPS:-100}"
NT="${SMOKE_TASKS:-4}"
case "$NT" in
  2) SPLIT='(1, 2, 1, 1)' ;;
  4) SPLIT='(1, 2, 2, 1)' ;;
  8) SPLIT='(1, 4, 2, 1)' ;;
  *) echo "SMOKE_TASKS must be 2, 4 or 8 (got $NT)"; exit 1 ;;
esac
STEPS=" ${SMOKE_STEPS:-v256 n1000 full16 local16 n128} "
LIMIT="${SMOKE_STEP_LIMIT:-8m}"
T="_ck${SMOKE_TAG:-}"
# srun does not inherit --cpus-per-task from sbatch (Slurm >= 22.05).
[ -n "${SLURM_CPUS_PER_TASK:-}" ] && export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
want(){ case "$STEPS" in *" $1 "*) return 0 ;; esac; return 1; }
# dt = t_end / steps = 1e-3 code; t_end(75 Myr) = 7.670 code.
BENCH_FRACTION=$(awk -v n="$BENCH_STEPS" 'BEGIN{printf "%.6g", 1e-3 * n / 7.670}')

run(){   # $1 name, $2 time limit, $3 tasks, rest = env assignments
  local name=$1 limit=$2 tasks=$3; shift 3
  echo "=== $(date +%T) $name: $tasks task(s), $*"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 1000 > "$OUT/$name.smi" &
  local smi=$!
  env "$@" timeout "$limit" srun --nodes=1 --ntasks="$tasks" --gpu-bind=none --kill-on-bad-exit=1 \
      python cgols.py > "$OUT/$name.log" 2>&1
  local rc=$?
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  echo "rc=$rc" >> "$OUT/$name.log"
  echo "    rc=$rc  $(grep -m1 -A3 'Compiled memory usage PER DEVICE' "$OUT/$name.log" | grep 'Total size')"
  summary
}

summary(){
  {
    echo "== ck_${SLURM_JOB_ID}: cluster kernel smoke, $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION), bench $BENCH_STEPS steps at dt 1e-3"
    printf "%-18s %-5s %-34s %13s %13s %9s %7s %9s\n" run rc "kernel" compiled_MiB smi_peak_MiB s/step steps wall_s
    for f in "$OUT"/*.log; do
      [ -e "$f" ] || continue
      local n; n=$(basename "$f" .log)
      local kernel elapsed sps steps peak wall
      kernel=$(grep -m1 'Model-C clusters' "$f" | sed -n 's/.*): \([0-9]* x R = [0-9.]* pc\).*kernel \(.*\)/\1, \2/p')
      elapsed=$(grep -m1 'Time elapsed' "$f" | sed -n 's/.*elapsed: \([0-9.]*\) seconds.*/\1/p')
      steps=$(grep -c '\[diag\]' "$f")
      case "$n" in b*) steps=$BENCH_STEPS ;; esac
      sps=$([ -n "$elapsed" ] && [ "$steps" -gt 0 ] && awk -v e="$elapsed" -v n="$steps" 'BEGIN{printf "%.4f", e / n}')
      peak=$(awk -F', ' '$2>m{m=$2}END{print m}' "$OUT/$n.smi" 2>/dev/null)
      # s/step: post-compile for the benches only; wall_s includes the compile.
      wall=$(grep -m1 'time_integration wall time' "$f" | sed -n 's/.*wall time: \([0-9.]*\) s.*/\1/p')
      printf "%-18s %-5s %-34s %13s %13s %9s %7s %9s\n" "$n" "$(grep '^rc=' "$f" | tail -1 | cut -d= -f2)" \
        "${kernel:--}" \
        "$(grep -m1 -A3 'Compiled memory usage PER DEVICE' "$f" | sed -n 's/.*Total size: \([0-9.]*\).*/\1/p')" \
        "${peak:--}" "${sps:--}" "$steps" "${wall:--}"
    done
    echo
    echo "== full-grid vs local kernel at the last common frame and in the 3D final state (256^3, 8 Myr);"
    echo "   s$NT vs s1 of the full-grid kernel is the scale: the same kernel on $NT processes vs 1"
    [ -f "$OUT/compare.txt" ] && cat "$OUT/compare.txt"
  } > "$OUT/summary.txt"
}

V256="CGOLS_DIM=256 CGOLS_END_FRACTION=0.107 CGOLS_NUM_SNAPSHOTS=4"
B512="CGOLS_DIM=512 CGOLS_BENCH_STEPS=$BENCH_STEPS CGOLS_END_FRACTION=$BENCH_FRACTION CGOLS_CLUSTER_START_MYR=0 CGOLS_RUN_TAG=${T}_bench"
S1='CGOLS_SHARD_SPLIT=(1, 1, 1, 1)'
SN="CGOLS_SHARD_SPLIT=$SPLIT"

# shellcheck disable=SC2086
{
if want v256; then
  run v256_full_s1 "$LIMIT" 1 $V256 "$S1" CGOLS_CLUSTER_KERNEL=full CGOLS_RUN_TAG=${T}_v256_full_s1
  run v256_local_s1 "$LIMIT" 1 $V256 "$S1" CGOLS_CLUSTER_KERNEL=local CGOLS_RUN_TAG=${T}_v256_local_s1
  python cgols_fv_validate_compare.py --pair ${T}_v256_full_s1 ${T}_v256_local_s1 > "$OUT/compare.txt" 2>&1
  summary
  run v256_full_s$NT "$LIMIT" "$NT" $V256 "$SN" CGOLS_CLUSTER_KERNEL=full CGOLS_RUN_TAG=${T}_v256_full_s$NT
  run v256_local_s$NT "$LIMIT" "$NT" $V256 "$SN" CGOLS_CLUSTER_KERNEL=local CGOLS_RUN_TAG=${T}_v256_local_s$NT
  python cgols_fv_validate_compare.py --pair ${T}_v256_full_s1 ${T}_v256_local_s1 \
    ${T}_v256_full_s$NT ${T}_v256_local_s$NT ${T}_v256_full_s1 ${T}_v256_full_s$NT > "$OUT/compare.txt" 2>&1
  summary
fi
want n1000 && run b512_local_n1000 "$LIMIT" "$NT" $B512 "$SN" CGOLS_CLUSTER_KERNEL=local CGOLS_NUM_CLUSTERS=1000
want full16 && run b512_full_n16 "$LIMIT" "$NT" $B512 "$SN" CGOLS_CLUSTER_KERNEL=full
want local16 && run b512_local_n16 "$LIMIT" "$NT" $B512 "$SN" CGOLS_CLUSTER_KERNEL=local
want n128 && run b512_local_n128 "$LIMIT" "$NT" $B512 "$SN" CGOLS_CLUSTER_KERNEL=local CGOLS_NUM_CLUSTERS=128
}
cat "$OUT/summary.txt"
