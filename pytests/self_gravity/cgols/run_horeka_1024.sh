#!/bin/bash
#SBATCH --job-name=cgols_1024
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h200,accelerated-h200-8
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_%j.out
#SBATCH --error=cgols_%j.err

# Production 1024^3 (512x512x1024 -> 1024x1024x2048 cells) CGOLS run on HoreKa:
# 4x H200 on one node, 75 Myr, split (1, 2, 2, 1). Defaults here ARE the
# production configuration - a bare `sbatch run_horeka_1024.sh` starts a fresh
# run from t = 0; every other CGOLS_* knob defaults to the production values.
#
# Usage info:
# - run with: sbatch run_horeka_1024.sh   (from pytests/self_gravity/cgols/)
# - check status with: squeue -u $USER
# - estimate start time: squeue --start -j <job_id>
# - read output: tail -f cgols_<job_id>.out
# - cancel job: scancel <job_id>
#
# ---------------------------------------------------------------------------
# MEASURED COST (completed run, 2026-07-15 -> 2026-07-29)
# ---------------------------------------------------------------------------
# 75 Myr does NOT fit in HoreKa's 24 h walltime cap: the run took 3 legs, each
# resumed from the last rolling Orbax checkpoint (a resumed run is
# bit-identical to an uninterrupted one, so the legs are only a scheduling
# artifact).
#
#   leg  job      sim time         progress   wall clock   state
#   1    4830581   0.0 -> 47.3 Myr   0 -> 63%   24:00:29    TIMEOUT
#   2    4910533  45.0 -> 63.0 Myr  60 -> 84%   24:00:27    TIMEOUT
#   3    4916508  63.0 -> 75.0 Myr  84 -> 100%  13:26:12    COMPLETED
#                                              ---------
#                                    TOTAL      61:27:08   (~61.5 h)
#
# => ~246 GPU-hours on H200, ~146 kWh (sum of the three jobs' reported energy).
# Leg 3's solver-only time was 48245 s (13.40 h), i.e. setup/compile/IO is a
# couple of minutes on top of the integration.
#
# Note the throughput is NOT linear in wall time - it degrades as the wind
# develops and max|v| climbs (~12 -> ~25-50 code units), tightening the CFL
# step: leg 1 banked ~2.6 %/h, leg 2 ~0.9 %/h, leg 3 ~1.2 %/h. Budget the
# 24 h legs accordingly rather than extrapolating from leg 1.
#
# Leg 2 re-did ~2.3 Myr (63% -> restart at 60%) because the newest checkpoint
# lagged the last diag line; that is the expected CGOLS_CHECKPOINT_EVERY=3 loss.
# Not counted above: two earlier aborted attempts (4675223 BFC-allocator OOM,
# 4901231 restart-path rank-0 OOM misread as an NCCL hang) - both fixed, see
# the allocator settings below and the watchdog at the bottom.
#
# To CONTINUE after a TIMEOUT, point the run at the previous leg's checkpoint
# directory and give it a fresh tag (so the startup cleanup cannot wipe the
# predecessor's frames/checkpoints):
#
#   sbatch --export=ALL,CGOLS_RESTART_FROM=data/cgols_checkpoints,CGOLS_RUN_TAG=_leg2 \
#          run_horeka_1024.sh
#   sbatch --export=ALL,CGOLS_RESTART_FROM=data/cgols_checkpoints_leg2,CGOLS_RUN_TAG=_leg3 \
#          run_horeka_1024.sh
#
# CGOLS_RESTART_FROM names the Orbax checkpoint *directory* (newest step by
# default; CGOLS_RESTART_STEP picks a specific one).

# run configuration; every other CGOLS_* knob defaults to the production values
export CGOLS_DIM=1024
PROD_SPLIT="(1, 2, 2, 1)"   # (var,x,y,z) shard split, product = #GPUs

# Fresh run by default. Both are honoured from the environment, so a
# continuation leg only needs `sbatch --export=ALL,CGOLS_RESTART_FROM=...,
# CGOLS_RUN_TAG=...` (see the header) - no edit to this file.
export CGOLS_RESTART_FROM="${CGOLS_RESTART_FROM:-}"
export CGOLS_RUN_TAG="${CGOLS_RUN_TAG:-}"

# Load system CUDA
module purge
module load devel/cuda/12.9

# activate env
source ~/.bashrc
micromamba activate astro

# linker fixes: every pip-installed nvidia lib (incl. nccl, needed for
# multi-GPU), independent of the env's python version
PYSITE=$(python -c 'import site; print(site.getsitepackages()[0])')
for d in "$PYSITE"/nvidia/*/lib; do export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done

# jax-specific. cuda_async instead of the default BFC allocator: at 1024 the
# compiled step needs a ~112 GB CONTIGUOUS temp arena, which BFC could not
# place next to the resident arguments (job 4675223 OOM'd with the memory
# nominally free); cuda_async has no contiguity requirement. 0.98 because the
# program totals ~134 GB of the H200's ~151 GB.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

# bulk data (ICs/final/checkpoints, ~300 GB at 1024) must live on a workspace:
#   ws_allocate cgols 60 && ln -s "$(ws_find cgols)" data
cd "$SLURM_SUBMIT_DIR"
[ -L data ] && [ -d data ] || { echo "ERROR: link ./data to a workspace first" >&2; exit 1; }

# debugging
echo "Running on node: $SLURM_JOB_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# logging
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv -l 30 > "gpu_usage_${SLURM_JOB_ID}.csv" &
trap 'kill %1 2>/dev/null || true' EXIT

# build the ICs once, then run production
IC_FILE="data/initial/cgols_initial_state_d${CGOLS_DIM}.npy"
[ -f "$IC_FILE" ] || CGOLS_CREATE_IC=1 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" python cgols.py

if [ -n "$CGOLS_RESTART_FROM" ]; then
    echo "Continuation leg: restarting from '$CGOLS_RESTART_FROM' (tag '$CGOLS_RUN_TAG')"
else
    echo "Fresh run from t = 0 (expect a TIMEOUT at ~63%; continue with a leg-2 sbatch)"
fi

# Watchdog wrapper: job 4901231 deadlocked ~1 min in at the NCCL clique
# rendezvous - a startup timing race (leg 1 passed the same spot after a 37 s
# wobble). XLA only warns ("may be stuck"), it never aborts, so a hung job
# silently burns its whole walltime. The hang strikes before any frame or
# checkpoint exists, so killing and relaunching inside the same allocation is
# safe and costs ~20 min instead of another multi-day queue wait. Healthy runs
# pass untouched: warnings that resolve print a matching "unstuck" line.
ERR_FILE="cgols_${SLURM_JOB_ID}.err"
for attempt in 1 2 3 4; do
    stuck0=$(grep -c 'may be stuck' "$ERR_FILE" 2>/dev/null || true)
    unstuck0=$(grep -c 'unstuck' "$ERR_FILE" 2>/dev/null || true)
    CGOLS_SHARD_SPLIT="$PROD_SPLIT" python cgols.py &
    SOLVER_PID=$!
    sleep 900   # the racy rendezvous fires ~1 min in; 15 min is ample slack
    if kill -0 "$SOLVER_PID" 2>/dev/null; then
        sleep 180   # grace so an in-flight stuck warning can still resolve
        stuck=$(( $(grep -c 'may be stuck' "$ERR_FILE" 2>/dev/null || true) - stuck0 ))
        unstuck=$(( $(grep -c 'unstuck' "$ERR_FILE" 2>/dev/null || true) - unstuck0 ))
        if [ "$stuck" -gt "$unstuck" ]; then
            echo "WATCHDOG: attempt $attempt hung at NCCL init, relaunching"
            kill "$SOLVER_PID" 2>/dev/null; sleep 30
            kill -9 "$SOLVER_PID" 2>/dev/null || true
            wait "$SOLVER_PID" 2>/dev/null || true
            sleep 30   # let the driver reclaim the ~134 GB on each GPU
            continue
        fi
    fi
    wait "$SOLVER_PID"
    exit $?
done
echo "WATCHDOG: giving up after 4 hung attempts" >&2
exit 1
