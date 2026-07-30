#!/bin/bash
#SBATCH --job-name=cgols_1024
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h200,accelerated-h200-8
#SBATCH --time=48:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_%j.out
#SBATCH --error=cgols_%j.err

# Production 1024^3 CGOLS run on HoreKa: 4x H200 on one node, 75 Myr,
# split (1, 2, 2, 1). A bare sbatch starts a fresh run from t = 0; every
# CGOLS_* knob not set here defaults to the production value.
#
#   sbatch run_ic_horeka.sh       # build the ICs first (dev queue)
#   sbatch run_horeka_1024.sh
#
# The run needs ~61.5 h, so it does not fit one job. Continue after a TIMEOUT
# from the previous leg's Orbax checkpoint directory, with a fresh tag (the
# startup cleanup wipes checkpoints/frames carrying the same tag):
#
#   sbatch --time=16:00:00 \
#          --export=ALL,CGOLS_RESTART_FROM=data/cgols_checkpoints,CGOLS_RUN_TAG=_leg2 \
#          run_horeka_1024.sh
#
# MEASURED COST of the completed run (jobs 4830581 / 4910533 / 4916508, at
# --time=24:00:00, hence three legs; 48 h should need only two):
#
#   leg  sim time          progress    wall clock   state
#   1     0.0 -> 47.3 Myr   0 -> 63%    24:00:29     TIMEOUT
#   2    45.0 -> 63.0 Myr  60 -> 84%    24:00:27     TIMEOUT
#   3    63.0 -> 75.0 Myr  84 -> 100%   13:26:12     COMPLETED
#                                       --------
#                          TOTAL        61:27:08     ~246 GPU-h, ~146 kWh
#
# Throughput decays as the wind develops and max|v| climbs (~12 -> ~25-50),
# tightening the CFL step: 2.6 %/h, then 0.9, then 1.2. Budget legs from that
# decay, not from leg 1 - 48 h banks ~84% from a cold start, not 100%.

export CGOLS_DIM=1024
PROD_SPLIT="(1, 2, 2, 1)"   # (var,x,y,z) shard split, product = #GPUs

# Empty unless overridden, so the default is a fresh run and a continuation
# leg needs no edit to this file.
export CGOLS_RESTART_FROM="${CGOLS_RESTART_FROM:-}"
export CGOLS_RUN_TAG="${CGOLS_RUN_TAG:-}"

module purge
module load devel/cuda/12.9

source ~/.bashrc
micromamba activate astro

# every pip-installed nvidia lib (incl. nccl), independent of python version
PYSITE=$(python -c 'import site; print(site.getsitepackages()[0])')
for d in "$PYSITE"/nvidia/*/lib; do export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done

# cuda_async instead of the default BFC allocator: the compiled step needs a
# ~112 GB CONTIGUOUS temp arena that BFC could not place next to the resident
# arguments (job 4675223 OOM'd with the memory nominally free). 0.98 because
# the program totals ~134 GB of the H200's ~151 GB.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

# bulk data (~300 GB at 1024) must live on a workspace:
#   ws_allocate cgols 60 && ln -s "$(ws_find cgols)" data
cd "$SLURM_SUBMIT_DIR"
[ -L data ] && [ -d data ] || { echo "ERROR: link ./data to a workspace first" >&2; exit 1; }

echo "Running on node: $SLURM_JOB_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv -l 30 > "gpu_usage_${SLURM_JOB_ID}.csv" &
trap 'kill %1 2>/dev/null || true' EXIT

# Fallback only - pre-build with run_ic_horeka.sh. Uses the CPU path, never the
# eager GPU one (~90 GB transient), so a missing IC costs allocation time
# rather than the whole job. Both files are checked: a build interrupted
# between the two writes passes a state-only guard, then fails in the solver.
IC_STATE="data/initial/cgols_initial_state_d${CGOLS_DIM}.npy"
IC_POT="data/initial/cgols_initial_potential_d${CGOLS_DIM}.npy"
if [ ! -f "$IC_STATE" ] || [ ! -f "$IC_POT" ]; then
    echo "WARNING: d${CGOLS_DIM} ICs missing, building them here - pre-build" >&2
    echo "         with 'sbatch run_ic_horeka.sh' to avoid this." >&2
    JAX_PLATFORMS=cpu CGOLS_CREATE_IC=1 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" python cgols.py \
        || { echo "ERROR: IC build failed, aborting before the solver" >&2; exit 1; }
fi

if [ -n "$CGOLS_RESTART_FROM" ]; then
    echo "Continuation leg: restarting from '$CGOLS_RESTART_FROM' (tag '$CGOLS_RUN_TAG')"
else
    echo "Fresh run from t = 0 (48 h banks ~84%; expect a TIMEOUT, then a ~16 h leg)"
fi

# Watchdog: job 4901231 deadlocked ~1 min in at the NCCL clique rendezvous.
# XLA only warns ("may be stuck"), it never aborts, so a hung job silently
# burns its whole walltime. The hang strikes before any frame or checkpoint
# exists, so relaunching inside the same allocation is safe and costs ~20 min
# instead of another multi-day queue wait. Healthy runs pass untouched:
# warnings that resolve print a matching "unstuck" line.
# NB "giving up after 4" has more than one cause - check .err for
# RESOURCE_EXHAUSTED before blaming NCCL (job 4905854 was a rank-0 OOM).
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
