#!/bin/bash
#SBATCH --job-name=c_detlnch
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev-gpu-h100
#SBATCH --time=00:58:00
#SBATCH --exclusive

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_detl_%j.out
#SBATCH --error=cgols_detl_%j.err

# Why do two launches of the same model-C run differ at rounding level, and
# does a compiler option remove it?
#
# Job 134305 (run_horeka_restart_determinism.sh) showed that the restart is
# exact: a resumed run was bit-identical to the uninterrupted one in all four
# 4-process pairs. What differs is the LAUNCH. The total mass in the [diag]
# line, a sum over the whole grid, came out 1 ulp different in about a third
# of the steps between two launches whose states were bit-identical, in a way
# fixed per launch. Model A and the local-box cluster kernel never feed a
# grid-wide sum back into the gas, so their states stayed bit-identical. The
# full-grid cluster kernel normalises every cluster by a grid-wide sum
# (jnp.sum(_sphere_weights(slot)) in _cgols_wind.py), so there the launches
# part as soon as the feedback is on, and the difference then grows.
#
# This job measures how often that happens and tries the XLA options that
# should remove it. A base run (256^3, model C, full-grid kernel) goes to
# 6 Myr with a checkpoint every 1.2 Myr. Every launch then resumes its
# checkpoint 4 (4.8 Myr) and runs to 6 Myr (feedback from 5 Myr); launches of a
# group differ in nothing. Per launch: md5 of the 3D final state and md5 of the
# sequence of [diag] masses.
#
#   d4  4 processes, default flags                       -> how often they part
#   o4  4 processes, --xla_gpu_deterministic_ops=true
#   d1, o1  the same two on one process
#   f4  4 processes, --xla_gpu_experimental_enable_fusion_autotuner=false
#   t4  4 processes, --xla_gpu_autotune_level=0
#   a1  1 process, default flags, autotune results dumped per launch
#
# Result (job 134433): 8 default launches on 4 processes gave 5 different final
# states; with each of the three options every launch was bit-identical (22 of
# 22, one common state). The autotune dump shows why: XLA times two code
# generators for every fusion that holds a reduction (NATIVE_EMITTER and
# BLOCK_LEVEL_EMITTER) and keeps the faster, and two launches chose differently
# for 2 of 20; the two add in a different order.
#
# Second set of groups (DETL_GROUPS=ad4:at4:ld4:lt4:lo4:bench), the models that
# job 134305 tested with one pair only, and the cost of the options:
#
#   ad4, at4       model A (central injection): default, autotuning off
#   ld4, lt4, lo4  model C with the local-box kernel: default, autotuning off,
#                  deterministic ops
#   bench          512^3 on 4 processes, fixed steps, s/step after compilation:
#                  full-grid kernel (16 clusters) and local-box kernel (1000
#                  clusters), default against the two options
#
# Dev queue: 1 node, 4 x H100, at most 1 h. Run from a code COPY that no queued
# job imports:
#
#   cd ~/astronomix-c1024n1000/pytests/self_gravity/cgols && sbatch run_horeka_launch_determinism.sh
#   ... && sbatch --export=ALL,DETL_GROUPS=ad4:at4:ld4:lt4:lo4:bench run_horeka_launch_determinism.sh
#
# Everything lands in cgols_logs/detl_<jobid>/ (summary.txt = the answer,
# rewritten after every launch so a timeout still leaves one).

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
OUT=cgols_logs/detl_${SLURM_JOB_ID}
mkdir -p "$OUT"
export CGOLS_IC_MODE=insitu CGOLS_RESTART_FROM="" CGOLS_EMISSION_FRAMES=0
# One checkpoint per frame (every 1.2 Myr), all kept: checkpoint 4 = 4.8 Myr.
V256="CGOLS_DIM=256 CGOLS_END_FRACTION=0.08 CGOLS_NUM_SNAPSHOTS=5 CGOLS_CHECKPOINT_EVERY=1 CGOLS_CHECKPOINT_KEEP=0"
C="CGOLS_FEEDBACK=clusters CGOLS_CLUSTER_KERNEL=full"
L="CGOLS_FEEDBACK=clusters CGOLS_CLUSTER_KERNEL=local"
A="CGOLS_FEEDBACK=central"
LIMIT="${DETL_STEP_LIMIT:-4m}"
BENCH_LIMIT="${DETL_BENCH_LIMIT:-6m}"
BENCH_STEPS="${DETL_BENCH_STEPS:-100}"
# dt = t_end / steps = 1e-3 code (below the initial CFL step at 512, 2.3e-3);
# t_end(75 Myr) = 7.670 code. The clusters exist from t = 0, so the kernel runs.
BENCH_FRACTION=$(awk -v n="$BENCH_STEPS" 'BEGIN{printf "%.6g", 1e-3 * n / 7.670}')
B512="CGOLS_DIM=512 CGOLS_BENCH_STEPS=$BENCH_STEPS CGOLS_END_FRACTION=$BENCH_FRACTION CGOLS_CLUSTER_START_MYR=0 CGOLS_CHECKPOINT_EVERY=0 CGOLS_RUN_TAG=_det_bench"
GROUPS_WANTED="${DETL_GROUPS:-d4 o4 d1 o1 f4 t4 a1}"
GROUPS_WANTED="${GROUPS_WANTED//:/ }"
BASE_XLA="${XLA_FLAGS:-}"
echo "XLA_FLAGS of every launch: $BASE_XLA (+ the group's option)"
want(){ case " $GROUPS_WANTED " in *" $1 "*) return 0 ;; esac; return 1; }
ckpt(){ echo "$CGOLS_DIR/data/cgols_checkpoints_det_$1"; }
final(){ echo "$CGOLS_DIR/data/final/cgols_final_state_det_$1.npy"; }
S1='CGOLS_SHARD_SPLIT=(1, 1, 1, 1)'
S4='CGOLS_SHARD_SPLIT=(1, 2, 2, 1)'

run(){   # $1 name (tag _det_<name>), $2 tasks, $3 extra XLA option(s), rest = env assignments
  local name=$1 tasks=$2 xla=${3//@NAME@/$1}; shift 3
  echo "=== $(date +%T) $name: $tasks task(s)${xla:+, $xla}"
  env "$@" "XLA_FLAGS=$BASE_XLA${xla:+ $xla}" CGOLS_RUN_TAG=_det_$name timeout "$LIMIT" srun --nodes=1 --ntasks="$tasks" \
      --gpu-bind=none --kill-on-bad-exit=1 python cgols.py > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "rc=$rc" >> "$OUT/$name.log"
  local state=missing mass wall
  [ -f "$(final "$name")" ] && state=$(md5sum "$(final "$name")" | cut -c1-12)
  mass=$(grep '\[diag\]' "$OUT/$name.log" | sed 's/.*mass=\([^ ]*\).*/\1/' | md5sum | cut -c1-12)
  wall=$(grep -m1 'time_integration wall time' "$OUT/$name.log" | sed -n 's/.*wall time: \([0-9.]*\) s.*/\1/p')
  printf "%-7s %-4s %-6s %-7s %-13s %-13s\n" "$name" "$rc" "$(grep -c '\[diag\]' "$OUT/$name.log")" "${wall:--}" \
      "$state" "$mass" >> "$OUT/runs.txt"
  echo "    rc=$rc  state $state  masses $mass"
  summary
  return "$rc"
}

bench(){   # $1 name, $2 extra XLA option(s), rest = env assignments
  local name=$1 xla=$2; shift 2
  echo "=== $(date +%T) $name: bench, $BENCH_STEPS steps${xla:+, $xla}"
  # shellcheck disable=SC2086
  env $B512 "$@" "$S4" "XLA_FLAGS=$BASE_XLA${xla:+ $xla}" timeout "$BENCH_LIMIT" srun --nodes=1 --ntasks=4 --gpu-bind=none \
      --kill-on-bad-exit=1 python cgols.py > "$OUT/$name.log" 2>&1
  local rc=$? elapsed mem sps=-
  echo "rc=$rc" >> "$OUT/$name.log"
  elapsed=$(grep -m1 'Time elapsed' "$OUT/$name.log" | sed -n 's/.*elapsed: \([0-9.]*\) seconds.*/\1/p')
  mem=$(grep -m1 -A3 'Compiled memory usage PER DEVICE' "$OUT/$name.log" | sed -n 's/.*Total size: \([0-9.]*\).*/\1/p')
  [ -n "$elapsed" ] && sps=$(awk -v e="$elapsed" -v n="$BENCH_STEPS" 'BEGIN{printf "%.4f", e / n}')
  printf "%-7s %-4s %-8s %-13s %-34s %s\n" "$name" "$rc" "$sps" "${mem:--}" \
      "$(grep -m1 'Model-C clusters' "$OUT/$name.log" | sed -n 's/.*): \([0-9]* x R = [0-9.]* pc\).*kernel \([a-z]*\).*/\1, \2/p')" \
      "${xla:-default}" >> "$OUT/bench.txt"
  echo "    rc=$rc  $sps s/step"
  summary
}

# Size of the difference between two final states (the tool of job 134305).
cat > "$OUT/compare.py" <<'EOF'
import os
import sys

import numpy as np

cgols_dir, ref, test, note = sys.argv[1:5]
names = ("rho", "vx", "vy", "vz", "P", "e_int")
path = lambda tag: os.path.join(cgols_dir, "data", "final", f"cgols_final_state_det_{tag}.npy")
a, b = np.load(path(ref), mmap_mode="r"), np.load(path(test), mmap_mode="r")
parts = []
for i in range(a.shape[0]):
    x, y = np.asarray(a[i]), np.asarray(b[i])
    n = int(np.count_nonzero(x != y))
    if n:
        scale = np.abs(x).max() if names[i].startswith("v") else np.maximum(np.abs(x), 1e-30)
        rel = np.abs(y - x) / scale
        parts.append(f"{names[i]} {n / x.size:.1%} of cells, median {np.median(rel):.1e}, max {rel.max():.1e}")
print(f"{test} vs {ref} ({note}): " + ("BIT-IDENTICAL" if not parts else "; ".join(parts)))
EOF

summary(){
  {
    echo "== detl_${SLURM_JOB_ID}: launch-to-launch bit-identity, $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION), 256^3"
    echo "   b4 / b1 (model C, full-grid kernel), bl4 (local-box kernel), ba4 (model A): base runs to 6 Myr."
    echo "   Every other launch resumes its base at 4.8 Myr and runs to 6 Myr (feedback from 5 Myr)."
    echo
    [ -f "$OUT/groups.txt" ] && cat "$OUT/groups.txt" && echo
    echo "== per group: launches that finished, distinct 3D final states, distinct [diag] mass sequences"
    awk '$2 == 0 && $5 != "missing" && $1 ~ /_[0-9]+$/ {
           g = $1; sub(/_[0-9]+$/, "", g); n[g]++
           if (!((g, $5) in s)) { s[g, $5] = 1; ns[g]++ }
           if (!((g, $6) in m)) { m[g, $6] = 1; nm[g]++ }
           w[g] += $4
         }
         END { for (g in n) printf "%-4s %2d launches: %d distinct final state(s), %d distinct mass sequence(s), mean wall %.1f s\n", g, n[g], ns[g], nm[g], w[g] / n[g] }' \
        "$OUT/runs.txt" 2>/dev/null | sort
    echo
    printf "%-7s %-4s %-6s %-7s %-13s %-13s\n" run rc steps wall_s state_md5 masses_md5
    cat "$OUT/runs.txt" 2>/dev/null
    echo
    [ -f "$OUT/compare.txt" ] && cat "$OUT/compare.txt"
    if [ -f "$OUT/bench.txt" ]; then
      echo
      echo "== bench: 512^3 on 4 processes, $BENCH_STEPS fixed steps, s/step after compilation"
      printf "%-7s %-4s %-8s %-13s %-34s %s\n" run rc s/step compiled_MiB kernel option
      cat "$OUT/bench.txt"
    fi
  } > "$OUT/summary.txt"
}

group(){   # $1 name, $2 base run, $3 tasks, $4 launches, $5 extra XLA option(s), $6 note, rest = the model's env
  local g=$1 base=$2 tasks=$3 n=$4 xla=$5 note=$6 k split=$S1
  shift 6
  [ "$tasks" = 4 ] && split=$S4
  [ -d "$(ckpt "$base")" ] || { echo "$g: no checkpoints of $base, skipped"; return; }
  printf "%-4s %s\n" "$g" "$note" >> "$OUT/groups.txt"
  for k in $(seq 1 "$n"); do
    # shellcheck disable=SC2086
    run "${g}_$k" "$tasks" "$xla" $V256 "$@" "$split" "CGOLS_RESTART_FROM=$(ckpt "$base")" CGOLS_RESTART_STEP=4
    # An option the compiler rejects fails every launch: stop after the first.
    if [ $? -ne 0 ] && [ "$k" = 1 ]; then
      echo "    first launch of $g failed: $(grep -v -e sharding_propagation -e '^rc=' "$OUT/${g}_1.log" | tail -1 | cut -c1-300)"
      break
    fi
  done
  # How large is the difference where a launch did not reproduce the first one (at most two per group)?
  local first h ncmp=0
  first=$(awk -v p="${g}_1" '$1 == p {print $5}' "$OUT/runs.txt")
  for k in $(seq 2 "$n"); do
    h=$(awk -v p="${g}_$k" '$1 == p {print $5}' "$OUT/runs.txt")
    if [ -n "$h" ] && [ "$h" != missing ] && [ -n "$first" ] && [ "$first" != missing ] && [ "$h" != "$first" ] && [ "$ncmp" -lt 2 ]; then
      python "$OUT/compare.py" "$CGOLS_DIR" "${g}_1" "${g}_$k" "$note" >> "$OUT/compare.txt" 2>&1
      ncmp=$((ncmp + 1))
    fi
  done
  summary
}

DET="--xla_gpu_deterministic_ops=true"
NOTUNE="--xla_gpu_autotune_level=0"
# shellcheck disable=SC2086
{
# --- model C, full-grid kernel (job 134433) ---
if want d4 || want o4 || want f4 || want t4; then run b4 4 "" $V256 $C "$S4"; fi
want d4 && group d4 b4 4 "${DETL_N_D4:-8}" "" "model C full-grid kernel, 4 processes, default flags" $C
want o4 && group o4 b4 4 "${DETL_N_O4:-10}" "$DET" "model C full-grid kernel, 4 processes, $DET" $C
if want d1 || want o1 || want a1; then run b1 1 "" $V256 $C "$S1"; fi
want d1 && group d1 b1 1 "${DETL_N_D1:-4}" "" "model C full-grid kernel, 1 process, default flags" $C
want o1 && group o1 b1 1 "${DETL_N_O1:-4}" "$DET" "model C full-grid kernel, 1 process, $DET" $C
want f4 && group f4 b4 4 "${DETL_N_F4:-6}" "--xla_gpu_experimental_enable_fusion_autotuner=false" \
    "model C full-grid kernel, 4 processes, --xla_gpu_experimental_enable_fusion_autotuner=false" $C
want t4 && group t4 b4 4 "${DETL_N_T4:-6}" "$NOTUNE" "model C full-grid kernel, 4 processes, $NOTUNE" $C
want a1 && group a1 b1 1 "${DETL_N_A1:-2}" "--xla_gpu_dump_autotune_results_to=$CGOLS_DIR/$OUT/@NAME@.autotune.txt" \
    "model C full-grid kernel, 1 process, default flags, autotune results dumped to <launch>.autotune.txt" $C
# --- model A ---
if want ad4 || want at4; then run ba4 4 "" $V256 $A "$S4"; fi
want ad4 && group ad4 ba4 4 "${DETL_N_AD4:-6}" "" "model A, 4 processes, default flags" $A
want at4 && group at4 ba4 4 "${DETL_N_AT4:-4}" "$NOTUNE" "model A, 4 processes, $NOTUNE" $A
# --- model C, local-box kernel ---
if want ld4 || want lt4 || want lo4; then run bl4 4 "" $V256 $L "$S4"; fi
want ld4 && group ld4 bl4 4 "${DETL_N_LD4:-6}" "" "model C local-box kernel, 4 processes, default flags" $L
want lt4 && group lt4 bl4 4 "${DETL_N_LT4:-4}" "$NOTUNE" "model C local-box kernel, 4 processes, $NOTUNE" $L
want lo4 && group lo4 bl4 4 "${DETL_N_LO4:-3}" "$DET" "model C local-box kernel, 4 processes, $DET" $L
# --- speed: default against the options ---
if want bench; then
  bench bf_d1 "" $C
  bench bf_t1 "$NOTUNE" $C
  bench bf_o1 "$DET" $C
  bench bl_d1 "" $L CGOLS_NUM_CLUSTERS=1000
  bench bl_t1 "$NOTUNE" $L CGOLS_NUM_CLUSTERS=1000
  bench bl_o1 "$DET" $L CGOLS_NUM_CLUSTERS=1000
  bench bf_d2 "" $C
  bench bf_t2 "$NOTUNE" $C
  bench bf_o2 "$DET" $C
  bench bl_d2 "" $L CGOLS_NUM_CLUSTERS=1000
  bench bf_d3 "" $C
fi
}
summary
cat "$OUT/summary.txt"
