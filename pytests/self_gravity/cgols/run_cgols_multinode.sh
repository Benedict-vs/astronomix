#!/bin/bash
#SBATCH --job-name=cgols_mn
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=gpu-h200
#SBATCH --time=48:00:00
# HoreKa 2 GPU partitions are SHARED by default; a multi-node SPMD job wants
# whole nodes (all 4 GPUs visible to every rank for --gpu-bind=none, no
# co-tenant on the NICs), so ask for them explicitly.
#SBATCH --exclusive

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
#   HoreKa 2 Ruby gpu-h200, 8 nodes x 4 H200 = 32 GPUs, split (1, 8, 4, 1),
#   ~127-129 GB/device, ~135k steps at ~3.1-4 s/step => ~120-150 h wall,
#   i.e. three to four 48 h legs, ~3,900-4,800 GPU-h.
#
# Submit from THIS directory (Slurm writes the .out/.err relative to the
# submit dir, and the watchdog reads the .err).
#
# LEGS. A leg that TIMEOUTs is resumed into the SAME run tag from its newest
# checkpoint; checkpoints, frames and the diag log then continue where the
# previous leg stopped (only a fresh start clears them):
#
#   sbatch --export=ALL,CGOLS_RUN_TAG=_A1024,CGOLS_RESTART_FROM=latest \
#          run_cgols_multinode.sh
#
# Budget the legs from the LATE-run throughput decay, not from leg 1: at 1024
# the rate went 2.6 %/h -> 0.9 -> 1.2 as max|v| climbed and the CFL step
# tightened, so 48 h banks ~84% from a cold start, not 100%.

set -u

# Under sbatch this file runs as Slurm's spool copy
# (/var/spool/slurmd/job<id>/slurm_script), so BASH_SOURCE does not lead back to
# the repo; ask Slurm for the path that was submitted instead.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$HERE/_site_env.sh" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    _cmd="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=//p')"
    if [ -n "$_cmd" ]; then
        case "$_cmd" in /*) ;; *) _cmd="${SLURM_SUBMIT_DIR:-.}/$_cmd" ;; esac
        HERE="$(cd "$(dirname "$_cmd")" 2>/dev/null && pwd)"
    else
        HERE="${SLURM_SUBMIT_DIR:-}"
    fi
fi
if [ ! -f "$HERE/_site_env.sh" ]; then
    echo "ERROR: cannot locate the cgols directory (got '$HERE')" >&2
    exit 1
fi
source "$HERE/_site_env.sh" || exit 1
# Import astronomix from THIS tree: a code copy (e.g. ~/astronomix-ppm) then
# shadows the editable install, and for the main checkout it is the same code.
# Set after the site env, whose ~/.bashrc / micromamba activation could reset it.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$REPO/COMMIT" ] && echo "code copy of: $(cat "$REPO/COMMIT")"
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
# Ask Slurm where this job's stderr goes: the bench wrapper names it
# cgols_bench_%j.err, and a hardcoded name would leave the watchdog blind.
ERR_FILE=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/^ *StdErr=//p')
ERR_FILE="${ERR_FILE:-cgols_${SLURM_JOB_ID:-local}.err}"
SETTLE="${CGOLS_WATCHDOG_SETTLE:-1800}"   # 30 min
GRACE="${CGOLS_WATCHDOG_GRACE:-300}"      # 5 min for an in-flight warning to clear

# Sleep up to $1 s, but return as soon as srun exits: a run shorter than the
# settle window (every dev-queue test) must not idle out the rest of SETTLE.
_wait_or_exit() {
    local deadline=$((SECONDS + $1))
    while [ "$SECONDS" -lt "$deadline" ] && kill -0 "$SRUN_PID" 2>/dev/null; do
        sleep 10
    done
}

# PROGRESS WATCHDOG. Job 96656 (A-2048 leg 2) hung for 24 h on 32 GPUs after
# an InfiniBand error one minute in ("NET/IB ... posted incorrect receive info"
# -> "Aborting communicator"): no "may be stuck" warning, GPUs pinned at 100%,
# no output. A healthy run prints a [diag] line every step (seconds apart at
# 2048) and a checkpoint takes minutes, so a .out untouched for STALL seconds
# is a hang. The run is killed and relaunched from its newest checkpoint.
# An "Aborting communicator" line shortens the wait to STALL_ABORT.
OUT_FILE=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/^ *StdOut=//p')
OUT_FILE="${OUT_FILE:-cgols_${SLURM_JOB_ID:-local}.out}"
STALL="${CGOLS_WATCHDOG_STALL:-2700}"              # 45 min without output
STALL_ABORT="${CGOLS_WATCHDOG_STALL_ABORT:-600}"   # 10 min after a NCCL abort
POLL="${CGOLS_WATCHDOG_POLL:-60}"
KILL_WAIT="${CGOLS_WATCHDOG_KILL_WAIT:-60}"
MAX_ATTEMPTS="${CGOLS_WATCHDOG_ATTEMPTS:-4}"
CKPT_DIR="$CGOLS_DIR/data/cgols_checkpoints${CGOLS_RUN_TAG}"

_kill_srun() {
    kill "$SRUN_PID" 2>/dev/null; sleep "$KILL_WAIT"
    kill -9 "$SRUN_PID" 2>/dev/null || true
    wait "$SRUN_PID" 2>/dev/null || true
    sleep "$KILL_WAIT"   # let the drivers reclaim ~130 GB on every GPU
}

# Returns 0 (after killing srun) if the run stalls, 1 once srun has exited.
_watch_progress() {
    local now age aborts
    while kill -0 "$SRUN_PID" 2>/dev/null; do
        sleep "$POLL"
        kill -0 "$SRUN_PID" 2>/dev/null || return 1
        now=$(date +%s)
        age=$(( now - $(stat -c %Y "$OUT_FILE" 2>/dev/null || echo "$now") ))
        aborts=$(( $(grep -c 'Aborting communicator' "$ERR_FILE" 2>/dev/null || true) - aborts0 ))
        if [ "$age" -gt "$STALL" ] || { [ "$aborts" -gt 0 ] && [ "$age" -gt "$STALL_ABORT" ]; }; then
            echo "WATCHDOG: attempt $attempt stalled (no output for ${age} s," \
                 "$aborts NCCL abort(s)) at $(date -Is), relaunching" >&2
            _kill_srun
            return 0
        fi
    done
    return 1
}

# A relaunch after a stall must resume, never cold-start: a fresh start wipes
# the checkpoints the stalled attempt wrote. Without any checkpoint yet there is
# nothing to lose, so the original setting stands.
_resume_if_possible() {
    if ls "$CKPT_DIR" 2>/dev/null | grep -qx '[0-9][0-9]*'; then
        [ "$CGOLS_RESTART_FROM" = latest ] || \
            echo "WATCHDOG: relaunching from the newest checkpoint in $CKPT_DIR" >&2
        export CGOLS_RESTART_FROM=latest
    fi
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    stuck0=$(grep -c 'may be stuck' "$ERR_FILE" 2>/dev/null || true)
    unstuck0=$(grep -c 'unstuck' "$ERR_FILE" 2>/dev/null || true)
    aborts0=$(grep -c 'Aborting communicator' "$ERR_FILE" 2>/dev/null || true)

    srun "${SRUN_ARGS[@]}" python cgols.py &
    SRUN_PID=$!

    _wait_or_exit "$SETTLE"
    if kill -0 "$SRUN_PID" 2>/dev/null; then
        _wait_or_exit "$GRACE"
        stuck=$(( $(grep -c 'may be stuck' "$ERR_FILE" 2>/dev/null || true) - stuck0 ))
        unstuck=$(( $(grep -c 'unstuck' "$ERR_FILE" 2>/dev/null || true) - unstuck0 ))
        # Only a run that is STILL alive can be hung; one that exited during
        # the grace window is reported below with its real exit code.
        if [ "$stuck" -gt "$unstuck" ] && kill -0 "$SRUN_PID" 2>/dev/null; then
            echo "WATCHDOG: attempt $attempt hung at NCCL init, relaunching" >&2
            _kill_srun
            _resume_if_possible
            continue
        fi
    fi

    if _watch_progress; then
        _resume_if_possible
        continue
    fi

    wait "$SRUN_PID"
    rc=$?
    # Propagate srun's exit code - never mask a failed run with an
    # unconditional "DONE", which reports success and buries the real error.
    echo "=== srun exit code: $rc ==="
    exit $rc
done

echo "WATCHDOG: giving up after $MAX_ATTEMPTS hung attempts" >&2
exit 1
