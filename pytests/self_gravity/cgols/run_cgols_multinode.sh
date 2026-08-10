#!/bin/bash
#SBATCH --job-name=cgols_mn
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h200
#SBATCH --time=48:00:00

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_%j.out
#SBATCH --error=cgols_%j.err

# Multi-node CGOLS: ONE PROCESS PER GPU under srun.
#
# This is the launcher for every grid that no longer fits on a single node.
# --nodes / --ntasks-per-node above are the only things to change between
# 8/16/32-node jobs; the shard split is derived from them automatically (a
# near-square (1, gx, gy, 1) over jax.device_count(), z unsharded), so one file
# covers 4- and 8-GPU nodes and every machine _site_env.sh knows about.
#
#   sbatch run_cgols_multinode.sh                       # defaults below
#   sbatch --nodes=16 run_cgols_multinode.sh            # 64 GPUs
#   sbatch --export=ALL,CGOLS_DIM=1024 run_cgols_multinode.sh
#
# PRODUCTION 2048 TARGET (see run_horeka_2048.sh, which just sets these):
#   accelerated-h200, 8 nodes x 4 H200 = 32 GPUs, split (1, 8, 4, 1),
#   ~127-129 GB/device, ~135k steps at ~3.1-3.3 s/step => ~120-130 h wall,
#   i.e. three 48 h legs, ~3,900-4,150 GPU-h.
#
# LEGS. A leg that TIMEOUTs is resumed from the previous leg's Orbax checkpoint
# directory, with a FRESH run tag (the startup cleanup wipes checkpoints and
# frames carrying the same tag):
#
#   sbatch --export=ALL,CGOLS_RESTART_FROM=data/cgols_checkpoints,CGOLS_RUN_TAG=_leg2 \
#          run_cgols_multinode.sh
#
# Budget the legs from the LATE-run throughput decay, not from leg 1: at 1024
# the rate went 2.6 %/h -> 0.9 -> 1.2 as max|v| climbed and the CFL step
# tightened, so 48 h banks ~84% from a cold start, not 100%.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_site_env.sh" || exit 1
cd "$CGOLS_DIR" || exit 1

# --- run configuration ------------------------------------------------------
export CGOLS_DIM="${CGOLS_DIM:-2048}"
# One process per GPU. Auto-detected by cgols.py from SLURM_NTASKS, but set
# explicitly so a single-task debug launch can still be forced into this mode.
export CGOLS_DISTRIBUTED=1
# Leave CGOLS_SHARD_SPLIT unset to let cgols.py pick the near-square split of
# jax.device_count(); set it to pin a specific decomposition.
export CGOLS_SHARD_SPLIT="${CGOLS_SHARD_SPLIT:-}"
# Empty unless overridden, so a bare sbatch is a fresh run and a continuation
# leg needs no edit to this file.
export CGOLS_RESTART_FROM="${CGOLS_RESTART_FROM:-}"
export CGOLS_RUN_TAG="${CGOLS_RUN_TAG:-}"

echo "=== job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-localhost}"
echo "    ${SLURM_NNODES:-1} node(s) x ${SLURM_NTASKS_PER_NODE:-?} GPU(s) = ${SLURM_NTASKS:-?} processes"
echo "    CGOLS_DIM=$CGOLS_DIM split='${CGOLS_SHARD_SPLIT:-auto}' tag='${CGOLS_RUN_TAG}'"
cgols_preflight

cgols_check_workspace || exit 1

if [ -n "$CGOLS_RESTART_FROM" ]; then
    echo "Continuation leg: restarting from '$CGOLS_RESTART_FROM' (tag '$CGOLS_RUN_TAG')"
else
    echo "Fresh run from t = 0"
fi

start_gpu_logger
trap 'stop_gpu_logger' EXIT

# --- launch -----------------------------------------------------------------
# --gpu-bind=none: every task sees all of its node's GPUs, so intra-node NCCL
#   peer-to-peer works and each rank selects its device via SLURM_LOCALID.
#   --gpus-per-task=1 instead cgroup-binds each task to a single GPU that shows
#   up as ordinal 0, which breaks NCCL P2P and DEADLOCKS the topology exchange
#   with "invalid device ordinal".
# no --ntasks: srun inherits the full allocation; a mismatched count hangs the
#   rendezvous.
# --kill-on-bad-exit=1: one crashed rank tears the whole step down instead of
#   leaving the other 31 blocked forever inside a collective.
SRUN_ARGS=(--gpu-bind=none --kill-on-bad-exit=1)

# WATCHDOG. Job 4901231 deadlocked ~1 min in at the NCCL clique rendezvous. XLA
# only warns ("may be stuck"), it never aborts, so a hung job silently burns its
# whole walltime. The hang strikes before any frame or checkpoint exists, so
# relaunching inside the same allocation is safe and costs ~30 min instead of
# another multi-day queue wait. Healthy runs pass untouched: warnings that
# resolve print a matching "unstuck" line.
#
# NB "giving up" has more than one cause - check the .err for RESOURCE_EXHAUSTED
# before blaming NCCL (job 4905854 was a rank-0 OOM).
#
# We background the SRUN, not python: killing srun forwards SIGTERM to every
# task, whereas killing a local python would leave 31 orphaned ranks holding
# their GPUs. The settle window is longer than the 1024 run's 15 min because 32
# ranks take correspondingly longer to form the clique and to compile.
ERR_FILE="cgols_${SLURM_JOB_ID:-local}.err"
SETTLE="${CGOLS_WATCHDOG_SETTLE:-1800}"   # 30 min
GRACE="${CGOLS_WATCHDOG_GRACE:-300}"      # 5 min for an in-flight warning to clear

for attempt in 1 2 3; do
    stuck0=$(grep -c 'may be stuck' "$ERR_FILE" 2>/dev/null || true)
    unstuck0=$(grep -c 'unstuck' "$ERR_FILE" 2>/dev/null || true)

    srun "${SRUN_ARGS[@]}" python cgols.py &
    SRUN_PID=$!

    sleep "$SETTLE"
    if kill -0 "$SRUN_PID" 2>/dev/null; then
        sleep "$GRACE"
        stuck=$(( $(grep -c 'may be stuck' "$ERR_FILE" 2>/dev/null || true) - stuck0 ))
        unstuck=$(( $(grep -c 'unstuck' "$ERR_FILE" 2>/dev/null || true) - unstuck0 ))
        if [ "$stuck" -gt "$unstuck" ]; then
            echo "WATCHDOG: attempt $attempt hung at NCCL init, relaunching" >&2
            kill "$SRUN_PID" 2>/dev/null; sleep 60
            kill -9 "$SRUN_PID" 2>/dev/null || true
            wait "$SRUN_PID" 2>/dev/null || true
            sleep 60   # let the drivers reclaim ~130 GB on every GPU
            continue
        fi
    fi

    wait "$SRUN_PID"
    rc=$?
    # Propagate srun's exit code - never mask a failed run with an
    # unconditional "DONE", which reports success and buries the real error.
    echo "=== srun exit code: $rc ==="
    exit $rc
done

echo "WATCHDOG: giving up after 3 hung attempts" >&2
exit 1
