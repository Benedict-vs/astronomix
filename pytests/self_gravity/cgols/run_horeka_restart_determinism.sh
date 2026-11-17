#!/bin/bash
#SBATCH --job-name=c_detrst
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev-gpu-h100
#SBATCH --time=00:55:00
#SBATCH --exclusive

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_det_%j.out
#SBATCH --error=cgols_det_%j.err

# Why is a resumed model-C run not bit-identical to the uninterrupted one?
# The test before C-2048 (job 133267: 8 processes on two nodes, restart at
# 6 Myr) resumed at the right time and took the same steps, but its final state
# differed from the uninterrupted run's by 2.3e-7 (median, density): rounding
# level, not the bit-identity cgols.py documents for a resumed run. This job
# separates the possible causes with 256^3 runs to 8 Myr (feedback from 5 Myr,
# one checkpoint per 2 Myr frame), every pair compared BIT BY BIT:
#
#   c4a, c4b   model C (16 clusters, full-grid kernel), 4 processes, twice
#              -> are two identical runs bit-identical at all?
#   c4r3       c4a resumed from checkpoint 3 (6 Myr, feedback on)
#   c4r2       c4a resumed from checkpoint 2 (4 Myr, before the feedback)
#   c1a, c1b   model C on one process, twice;  c1r3 = c1a resumed at 6 Myr
#              -> one process against several
#   a4a, a4r3  model A (central injection), 4 processes, resumed at 6 Myr
#              -> the cluster model against the solver
#   l4a, l4r3  model C with the local-box kernel, 4 processes, resumed at 6 Myr
#
# Result (job 134305): the restart is exact. All four 4-process resumed runs
# were bit-identical to their uninterrupted run. What differed was the launch:
# c4b against c4a from 6 Myr on, c1r3 against c1a. The cause and the fix are in
# run_horeka_launch_determinism.sh; _site_env.sh now sets
# --xla_gpu_autotune_level=0, with which two launches no longer differ. To see
# the difference again, submit with
# --export=ALL,XLA_FLAGS=--xla_gpu_autotune_level=4 (XLA's default).
#
# Dev queue: 1 node, 4 x H100, at most 1 h. Run from a code COPY that no queued
# job imports:
#
#   cd ~/astronomix-c1024n1000/pytests/self_gravity/cgols && sbatch run_horeka_restart_determinism.sh
#
# Everything lands in cgols_logs/det_<jobid>/ (summary.txt = the answer,
# rewritten after every comparison so a timeout still leaves one).

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
OUT=cgols_logs/det_${SLURM_JOB_ID}
mkdir -p "$OUT"
export CGOLS_IC_MODE=insitu CGOLS_RESTART_FROM="" CGOLS_EMISSION_FRAMES=0
export CGOLS_DIM=256 CGOLS_END_FRACTION=0.107 CGOLS_NUM_SNAPSHOTS=4 CGOLS_CHECKPOINT_EVERY=1 CGOLS_CHECKPOINT_KEEP=0
LIMIT="${DET_STEP_LIMIT:-5m}"
STEPS="${DET_STEPS:-c4 c1 a4 c4r2 l4 c1b}"
want(){ case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
ckpt(){ echo "$CGOLS_DIR/data/cgols_checkpoints_det_$1"; }

run(){   # $1 name (tag _det_<name>), $2 tasks, rest = env assignments
  local name=$1 tasks=$2; shift 2
  echo "=== $(date +%T) $name: $tasks task(s), $*"
  env "$@" CGOLS_RUN_TAG=_det_$name timeout "$LIMIT" srun --nodes=1 --ntasks="$tasks" --gpu-bind=none \
      --kill-on-bad-exit=1 python cgols.py > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "rc=$rc" >> "$OUT/$name.log"
  echo "    rc=$rc  $(grep -c '\[diag\]' "$OUT/$name.log") steps  $(grep -m1 'Restarting from' "$OUT/$name.log" | sed 's/.* at t/at t/')"
}

# Bit-by-bit comparison of two runs: the 3D final state, and (with "frames")
# every frame both runs wrote, to see WHEN they part.
cat > "$OUT/compare.py" <<'EOF'
import glob
import os
import sys

import numpy as np

cgols_dir, ref, test, what, note = sys.argv[1:6]
names = ("rho", "vx", "vy", "vz", "P", "e_int")


def final(tag):
    return os.path.join(cgols_dir, "data", "final", f"cgols_final_state_det_{tag}.npy")


print(f"{test} vs {ref}: {note}")
if not (os.path.exists(final(ref)) and os.path.exists(final(test))):
    print("  final state missing: " + ", ".join(t for t in (ref, test) if not os.path.exists(final(t))))
else:
    a, b = np.load(final(ref), mmap_mode="r"), np.load(final(test), mmap_mode="r")
    if a.shape != b.shape:
        print(f"  final state shapes differ: {a.shape} vs {b.shape}")
    else:
        differing, parts = 0, []
        for i in range(a.shape[0]):
            x, y = np.asarray(a[i]), np.asarray(b[i])
            n = int(np.count_nonzero(x != y))
            differing += n
            if n:
                scale = np.abs(x).max() if names[i].startswith("v") else np.maximum(np.abs(x), 1e-30)
                rel = np.abs(y - x) / scale
                parts.append(f"{names[i]} {n / x.size:.1%} of cells, median {np.median(rel):.1e}, max {rel.max():.1e}")
        if differing == 0:
            print(f"  final state: BIT-IDENTICAL ({a.size} values)")
        else:
            print("  final state: DIFFERS: " + "; ".join(parts))
if what == "frames":
    frames = lambda tag: {os.path.basename(f) for f in glob.glob(os.path.join(cgols_dir, f"cgols_snapshots_det_{tag}", "frame_*.npz"))}
    for name in sorted(frames(ref) & frames(test)):
        fa = np.load(os.path.join(cgols_dir, f"cgols_snapshots_det_{ref}", name))
        fb = np.load(os.path.join(cgols_dir, f"cgols_snapshots_det_{test}", name))
        bad = []
        for key in sorted(set(fa.files) & set(fb.files)):
            x, y = np.asarray(fa[key]), np.asarray(fb[key])
            if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                rel = np.abs(y - x).max() / max(np.abs(x).max(), 1e-30) if x.shape == y.shape else np.nan
                bad.append(f"{key} (max diff / max {rel:.1e})")
        t = float(fa["time_myr"]) if "time_myr" in fa.files else np.nan
        print(f"  {name} t = {t:5.2f} Myr: " + ("bit-identical" if not bad else "DIFFERS: " + ", ".join(bad)))
EOF

summary(){
  {
    echo "== det_${SLURM_JOB_ID}: restart and run-to-run bit-identity, $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION), 256^3 to 8 Myr"
    printf "%-6s %-4s %-6s %-9s %s\n" run rc steps wall_s resumed
    for f in "$OUT"/*.log; do
      [ -e "$f" ] || continue
      printf "%-6s %-4s %-6s %-9s %s\n" "$(basename "$f" .log)" "$(grep '^rc=' "$f" | tail -1 | cut -d= -f2)" \
        "$(grep -c '\[diag\]' "$f")" \
        "$(grep -m1 'time_integration wall time' "$f" | sed -n 's/.*wall time: \([0-9.]*\) s.*/\1/p')" \
        "$(grep -m1 'Restarting from' "$f" | sed 's/.* at t = //')"
    done
    echo
    [ -f "$OUT/compare.txt" ] && cat "$OUT/compare.txt"
  } > "$OUT/summary.txt"
}

compare(){   # $1 ref, $2 test, $3 final|frames, $4 note
  python "$OUT/compare.py" "$CGOLS_DIR" "$1" "$2" "$3" "$4" >> "$OUT/compare.txt" 2>&1
  summary
}

S1='CGOLS_SHARD_SPLIT=(1, 1, 1, 1)'
S4='CGOLS_SHARD_SPLIT=(1, 2, 2, 1)'
C="CGOLS_FEEDBACK=clusters CGOLS_CLUSTER_KERNEL=full"
L="CGOLS_FEEDBACK=clusters CGOLS_CLUSTER_KERNEL=local"
A="CGOLS_FEEDBACK=central"

# shellcheck disable=SC2086
{
if want c4; then
  run c4a 4 $C "$S4"
  run c4b 4 $C "$S4"
  compare c4a c4b frames "model C, 4 processes, the same run twice"
  run c4r3 4 $C "$S4" "CGOLS_RESTART_FROM=$(ckpt c4a)" CGOLS_RESTART_STEP=3
  compare c4a c4r3 final "model C, 4 processes, resumed at 6 Myr (feedback on)"
fi
if want c1; then
  run c1a 1 $C "$S1"
  run c1r3 1 $C "$S1" "CGOLS_RESTART_FROM=$(ckpt c1a)" CGOLS_RESTART_STEP=3
  compare c1a c1r3 final "model C, 1 process, resumed at 6 Myr"
fi
if want a4; then
  run a4a 4 $A "$S4"
  run a4r3 4 $A "$S4" "CGOLS_RESTART_FROM=$(ckpt a4a)" CGOLS_RESTART_STEP=3
  compare a4a a4r3 final "model A, 4 processes, resumed at 6 Myr (injection ramping up)"
fi
if want c4r2; then
  run c4r2 4 $C "$S4" "CGOLS_RESTART_FROM=$(ckpt c4a)" CGOLS_RESTART_STEP=2
  compare c4a c4r2 final "model C, 4 processes, resumed at 4 Myr (before the feedback)"
fi
if want l4; then
  run l4a 4 $L "$S4"
  run l4r3 4 $L "$S4" "CGOLS_RESTART_FROM=$(ckpt l4a)" CGOLS_RESTART_STEP=3
  compare l4a l4r3 final "model C, local-box kernel, 4 processes, resumed at 6 Myr"
fi
if want c1b; then
  run c1b 1 $C "$S1"
  compare c1a c1b frames "model C, 1 process, the same run twice"
fi
}
summary
cat "$OUT/summary.txt"
