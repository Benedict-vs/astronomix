#!/bin/bash
#SBATCH --job-name=c2048_pre
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=gpu-h200
#SBATCH --time=01:30:00
#SBATCH --exclusive

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_pre_%j.out
#SBATCH --error=cgols_pre_%j.err

# Test before the first model-C run at 2048^3 (8 nodes x 4 H200, 48 h legs).
# Model C has so far only run on ONE node (C-1024: 8 processes on one 8-GPU
# node, a single leg). C-2048 needs three things that never ran with clusters:
# several nodes, a restart from a checkpoint, and the per-GPU block of the
# 2048 run. Two nodes x 4 H200, steps in priority order, each under its own
# timeout:
#
#   v256_full_s1    model C at 256^3 to 8 Myr (3 Myr of feedback), N = 16,
#                   full-grid kernel, one process: the reference
#   v256_full_s8    the same on 8 processes over BOTH nodes, split (1, 4, 2, 1)
#   v256_full_s8r   restart of v256_full_s8 from its checkpoint 3 (6 Myr, the
#                   clusters are active) to 8 Myr, 8 processes over both nodes
#   v256_local_s8   v256_full_s8 with the local-box kernel (the kernel of the
#                   runs with 1000 and more clusters)
#   b1024_full_s4   bench, 1024^3 on the 4 GPUs of one node: 512 x 512 x 2048
#                   cells per GPU, as many as the 256 x 512 x 4096 of the 2048
#                   run -> the memory read-out for C-2048
#   b1024_full_s8   bench, 1024^3 on 8 processes over both nodes: step time
#                   against the single-node C-1024 run (1.34 s/step)
#   b1024_local_n1000_s4   as b1024_full_s4 with 1000 clusters and the local
#                   kernel: memory for the later high-N runs at 2048
#
# The v256 runs all checkpoint every frame, so they share the same time-step
# sequence (the checkpoint driver cuts the run into segments). Expected:
#   full_s8  vs full_s1   ~1e-6 median (the same kernel on 8 processes vs 1)
#   full_s8r vs full_s8   0 (a resumed run is bit-identical)
#   local_s8 vs full_s8   a few 1e-7 median
#
# Run from the code COPY of the 2048 run, never from a checkout that queued
# jobs import:
#
#   cd ~/astronomix-c2048/pytests/self_gravity/cgols && sbatch run_horeka_c2048_pretest.sh
#
# Everything lands in cgols_logs/pre_<jobid>/ (summary.txt = the short answer,
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
echo "Running on nodes: $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION)"

cd "$CGOLS_DIR" || exit 1
OUT=cgols_logs/pre_${SLURM_JOB_ID}
mkdir -p "$OUT"
export CGOLS_FEEDBACK=clusters CGOLS_IC_MODE=insitu CGOLS_RESTART_FROM="" CGOLS_EMISSION_FRAMES=0
STEP_LIMIT="${PRE_STEP_LIMIT:-6m}"
BENCH_LIMIT="${PRE_BENCH_LIMIT:-18m}"
BENCH_STEPS="${PRE_BENCH_STEPS:-20}"
STEPS="${PRE_STEPS:-v256 restart local b4 b8 bn1000}"
# dt = t_end / steps = 1e-3 code (below the initial CFL step at 1024, 1.16e-3);
# t_end(75 Myr) = 7.670 code.
BENCH_FRACTION=$(awk -v n="$BENCH_STEPS" 'BEGIN{printf "%.6g", 1e-3 * n / 7.670}')
want(){ case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }

run(){   # $1 name, $2 time limit, $3 nodes, $4 tasks, rest = env assignments
  local name=$1 limit=$2 nodes=$3 tasks=$4; shift 4
  echo "=== $(date +%T) $name: $tasks task(s) on $nodes node(s), $*"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 1000 > "$OUT/$name.smi" &
  local smi=$!
  env "$@" timeout "$limit" srun --nodes="$nodes" --ntasks="$tasks" --gpu-bind=none --kill-on-bad-exit=1 \
      python cgols.py > "$OUT/$name.log" 2>&1
  local rc=$?
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  echo "rc=$rc" >> "$OUT/$name.log"
  echo "    rc=$rc  $(grep -m1 -A3 'Compiled memory usage PER DEVICE' "$OUT/$name.log" | grep 'Total size')"
  summary
}

summary(){
  {
    echo "== pre_${SLURM_JOB_ID}: model C before 2048, $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION), bench $BENCH_STEPS steps at dt 1e-3"
    printf "%-22s %-5s %-36s %13s %13s %9s %7s %9s\n" run rc "kernel" compiled_MiB smi_peak_MiB s/step steps wall_s
    for f in "$OUT"/*.log; do
      [ -e "$f" ] || continue
      local n; n=$(basename "$f" .log)
      local kernel elapsed sps steps peak wall
      kernel=$(grep -m1 'Model-C clusters' "$f" | sed -n 's/.*): \([0-9]* x R = [0-9.]* pc\).*kernel \(.*\)/\1, \2/p')
      elapsed=$(grep -m1 'Time elapsed' "$f" | sed -n 's/.*elapsed: \([0-9.]*\) seconds.*/\1/p')
      steps=$(grep -c '\[diag\]' "$f")
      case "$n" in b*) steps=$BENCH_STEPS ;; esac
      sps=$([ -n "$elapsed" ] && [ "$steps" -gt 0 ] && awk -v e="$elapsed" -v n="$steps" 'BEGIN{printf "%.4f", e / n}')
      # nvidia-smi runs on the batch node only: the peak is over that node's GPUs.
      peak=$(awk -F', ' '$2>m{m=$2}END{print m}' "$OUT/$n.smi" 2>/dev/null)
      # s/step: post-compile for the benches only; wall_s includes the compile.
      wall=$(grep -m1 'time_integration wall time' "$f" | sed -n 's/.*wall time: \([0-9.]*\) s.*/\1/p')
      printf "%-22s %-5s %-36s %13s %13s %9s %7s %9s\n" "$n" "$(grep '^rc=' "$f" | tail -1 | cut -d= -f2)" \
        "${kernel:--}" \
        "$(grep -m1 -A3 'Compiled memory usage PER DEVICE' "$f" | sed -n 's/.*Total size: \([0-9.]*\).*/\1/p')" \
        "${peak:--}" "${sps:--}" "$steps" "${wall:--}"
    done
    echo
    echo "== restart: where v256_full_s8r resumed, and its last step next to the uninterrupted run's"
    grep -m1 -i 'restart' "$OUT/v256_full_s8r.log" 2>/dev/null
    for n in v256_full_s8 v256_full_s8r; do
      [ -f "$OUT/$n.log" ] && echo "   $n: $(grep '\[diag\]' "$OUT/$n.log" | tail -1)"
    done
    echo
    echo "== REF vs TEST at the last common frame and in the 3D final state (256^3, 8 Myr)"
    echo "   expected: full_s8 vs full_s1 ~1e-6 median; full_s8r vs full_s8 exactly 0; local_s8 vs full_s8 a few 1e-7"
    [ -f "$OUT/compare.txt" ] && cat "$OUT/compare.txt"
  } > "$OUT/summary.txt"
}

compare(){
  local pairs=()
  [ -f "$OUT/v256_full_s8.log" ] && pairs+=(_pre_v256_full_s1 _pre_v256_full_s8)
  [ -f "$OUT/v256_full_s8r.log" ] && pairs+=(_pre_v256_full_s8 _pre_v256_full_s8r)
  [ -f "$OUT/v256_local_s8.log" ] && pairs+=(_pre_v256_full_s8 _pre_v256_local_s8)
  [ "${#pairs[@]}" -gt 0 ] && python cgols_fv_validate_compare.py --pair "${pairs[@]}" > "$OUT/compare.txt" 2>&1
  summary
}

# One checkpoint per frame, all kept, in every v256 run: the same segments and
# so the same time steps everywhere, and checkpoint 3 for the restart.
V256="CGOLS_DIM=256 CGOLS_END_FRACTION=0.107 CGOLS_NUM_SNAPSHOTS=4 CGOLS_CHECKPOINT_EVERY=1 CGOLS_CHECKPOINT_KEEP=0"
B1024="CGOLS_DIM=1024 CGOLS_BENCH_STEPS=$BENCH_STEPS CGOLS_END_FRACTION=$BENCH_FRACTION CGOLS_CLUSTER_START_MYR=0 CGOLS_CHECKPOINT_EVERY=0 CGOLS_RUN_TAG=_pre_bench"
S1='CGOLS_SHARD_SPLIT=(1, 1, 1, 1)'
S4='CGOLS_SHARD_SPLIT=(1, 2, 2, 1)'
S8='CGOLS_SHARD_SPLIT=(1, 4, 2, 1)'
CKPT_S8="$CGOLS_DIR/data/cgols_checkpoints_pre_v256_full_s8"

# shellcheck disable=SC2086
{
if want v256; then
  run v256_full_s1 "$STEP_LIMIT" 1 1 $V256 "$S1" CGOLS_CLUSTER_KERNEL=full CGOLS_RUN_TAG=_pre_v256_full_s1
  run v256_full_s8 "$STEP_LIMIT" 2 8 $V256 "$S8" CGOLS_CLUSTER_KERNEL=full CGOLS_RUN_TAG=_pre_v256_full_s8
  compare
fi
if want restart; then
  # Into a tag of its own, so the uninterrupted run stays untouched; the restore
  # code is the one CGOLS_RESTART_FROM=latest uses.
  run v256_full_s8r "$STEP_LIMIT" 2 8 $V256 "$S8" CGOLS_CLUSTER_KERNEL=full CGOLS_RUN_TAG=_pre_v256_full_s8r \
      "CGOLS_RESTART_FROM=$CKPT_S8" CGOLS_RESTART_STEP=3
  compare
fi
if want local; then
  run v256_local_s8 "$STEP_LIMIT" 2 8 $V256 "$S8" CGOLS_CLUSTER_KERNEL=local CGOLS_RUN_TAG=_pre_v256_local_s8
  compare
fi
want b4 && run b1024_full_s4 "$BENCH_LIMIT" 1 4 $B1024 "$S4" CGOLS_CLUSTER_KERNEL=full
want b8 && run b1024_full_s8 "$BENCH_LIMIT" 2 8 $B1024 "$S8" CGOLS_CLUSTER_KERNEL=full
want bn1000 && run b1024_local_n1000_s4 "$BENCH_LIMIT" 1 4 $B1024 "$S4" CGOLS_CLUSTER_KERNEL=local CGOLS_NUM_CLUSTERS=1000
}
cat "$OUT/summary.txt"
