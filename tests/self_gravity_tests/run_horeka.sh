#!/bin/bash
# =============================================================================
# HoreKa (NHR@KIT) batch script for the CGOLS wind run (cgols.py).
#   docs: https://www.nhr.kit.edu/userdocs/horeka/batch/
#
#   submit:        sbatch run_horeka.sh          (from tests/self_gravity_tests/)
#   queue:         squeue --me                   (ST: PD=pending, R=running)
#   start est.:    squeue --start -j <job_id>
#   follow log:    tail -f cgols_horeka_<job_id>.out
#   cancel:        scancel <job_id>
#   after the run: sacct -j <job_id> --format=JobID,Elapsed,State,MaxRSS
#
# Where the outputs land (all relative to the submit dir = this script's dir):
#   cgols_horeka_<job_id>.out/.err  - stdout/stderr (status lines, prints)
#   cgols_logs/cgols_diag.log       - full per-step diagnostics history
#   cgols_snapshots/                - streamed 2D frames (~25 MB/frame at 1024)
#   figures/cgols/                  - IC diagnostic figures
#   data/initial|final|checkpoints  - multi-GB states -> MUST be a workspace,
#                                     see the check below
#
# Log volume: astronomix detects that stdout is not a terminal and prints the
# [x%] [diag] status line only once per percent (ASTRONOMIX_STATUS_EVERY_PCT,
# default 1 -> ~100 lines/run). The full per-step history still goes to
# cgols_logs/cgols_diag.log (ASTRONOMIX_DIAG_LOG, set by cgols.py; export it
# to "" to disable).
#
# astronomix multi-GPU is SINGLE-PROCESS jax sharding (NOT MPI), so this is
# --ntasks=1 with --gres=gpu:N; one python process grabs all N GPUs on the node
# and shards the state over them. That caps us at ONE node = max 4 H100 here
# (cgols.py does not yet do multi-node jax.distributed). See the memory note
# cgols-horeka-setup.md for the scaling ceiling and what reaching 2048^3 needs.
# =============================================================================

#SBATCH --job-name=cgols_wind
#SBATCH --account=hk-project-pai00101      # VERIFY: your HoreKa project account
#SBATCH --partition=accelerated-h100        # 4x H100 80GB/node, 2-day max walltime
#SBATCH --time=24:00:00                      # raise toward 48:00:00 for full 1024^3

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4                          # must be >= product(CGOLS_SHARD_SPLIT) below
#SBATCH --cpus-per-task=32
#SBATCH --mem=200gb                           # host RAM: checkpoint offload is a ~43 GB
                                              # host copy per frame at 1024 (VERIFY node cap)

#SBATCH --output=cgols_horeka_%j.out
#SBATCH --error=cgols_horeka_%j.err

set -euo pipefail

# ---- Run configuration (the cgols.py env knobs) -----------------------------
# RES         : cells along x=y; z is always 2*RES (so 1024 -> 1024x1024x2048).
# IC tag      : defaults to _d<RES> inside cgols.py, so resolutions never
#               clobber each other; no need to set CGOLS_IC_TAG anymore.
# SHARD_SPLIT : (var,x,y,z) device split; product = #GPUs for the PRODUCTION run.
#               Per-device slice must stay divisible by pallas_block_shape (4,4,8):
#               1024^3 with (1,2,2,1) -> x=512,y=512,z=2048 all OK.
# All other knobs (CGOLS_CFL=0.9, CGOLS_TMAX_K=5e9, CGOLS_STEP_ONSET=1,
# CGOLS_BOUNDARY=diode, CGOLS_SMOOTHING_SIGMA=1.5, checkpoints every 3rd frame
# keep 4) default to the validated production values - a bare `python cgols.py`
# is the production run, so nothing else needs exporting here.
export CGOLS_DIM=1024
IC_TAG="_d${CGOLS_DIM}"
PROD_SPLIT="(1, 2, 2, 1)"          # 4 GPUs; use "(1, 1, 1, 1)" for a single-GPU run

# ---- FIRST TIME ON HOREKA: validate the environment cheaply first -----------
# Before committing to the long run, submit a throwaway 30-min single-GPU job:
#   change --gres=gpu:1, --time=00:30:00, and replace the two python calls below
#   with:   CGOLS_BENCH_STEPS=5 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" CGOLS_DIM=512 \
#           python cgols.py
# That compiles the solver on an H100 and prints per-device memory + sec/step
# without a multi-hour commit. Once it works, switch back to this production form.

# ---- Modules ----------------------------------------------------------------
module purge
module load devel/cuda/12.9        # VERIFY name on Horeka; may be droppable since
                                   # jax's pip cuda12 wheels bundle their own CUDA.

# ---- Environment ------------------------------------------------------------
source ~/.bashrc
micromamba activate astro          # VERIFY: env name on Horeka (astronomix must be
                                   # `pip install -e .` so the working tree is used)

# Point the dynamic linker at ALL pip-installed nvidia/*/lib dirs (cublas, cusparse,
# cufft, cudnn, nvjitlink, AND nccl). NCCL is required for the multi-GPU sharded
# run. Deriving the path avoids hardcoding the python version (the template's
# python3.13 path is wrong if the Horeka env is a different python).
PYSITE=$(python -c 'import site; print(site.getsitepackages()[0])')
for d in "$PYSITE"/nvidia/*/lib; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done

# ---- JAX runtime knobs ------------------------------------------------------
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   # ~76 GB of each 80 GB H100

# ---- Bulk-data location guard -----------------------------------------------
cd "$SLURM_SUBMIT_DIR"   # the tests/self_gravity_tests dir you sbatch from; cgols.py
                         # anchors all I/O to its own dir regardless, but cd keeps
                         # the .out/.err/.csv logs next to it.

# cgols.py writes everything multi-GB under ./data/ (IC ~43 GB, final ~43 GB,
# rolling checkpoints 4 x ~43 GB at 1024 -> plan ~300 GB). The symlink is
# MACHINE-LOCAL and gitignored: on the home cluster it points at
# /export/scratch/...; on HoreKa you create a fresh one pointing at a
# workspace, NOT $HOME (run these ONCE on HoreKa, in this directory):
#   ws_allocate cgols 60
#   ln -s "$(ws_find cgols)" data
# Without the link cgols.py would silently mkdir a real data/ in your home.
# -d also catches a *dangling* link (e.g. the home cluster's scratch link
# copied over verbatim, or an expired workspace).
if [ ! -L data ] || [ ! -d data ]; then
    echo "ERROR: ./data must be a symlink to an existing workspace directory." >&2
    echo "  ws_allocate cgols 60 && ln -s \"\$(ws_find cgols)\" data" >&2
    ls -la data 2>/dev/null >&2 || true
    exit 1
fi

# ---- Diagnostics ------------------------------------------------------------
echo "Node:  ${SLURM_JOB_NODELIST:-?}"
echo "GPUs:  ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# Background GPU usage logger.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv -l 30 > "cgols_gpu_usage_${SLURM_JOB_ID}.csv" &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

# ---- Run --------------------------------------------------------------------
# NOTE on autocvd: cgols.py calls autocvd(num_gpus=NUM_GPUS) at import, which picks
# free GPUs by nvidia-smi. Under SLURM --gres the job is cgroup-isolated to its
# allocated GPUs, so autocvd selects those. If it ever hangs ("waiting for N free
# GPUs"), add the CGOLS_NO_AUTOCVD guard described in cgols-horeka-setup.md.

# Step 1: build + save the initial conditions if not already present (single GPU;
# CGOLS_CREATE_IC=1 builds, saves the IC .npy to data/initial/, and EXITS without
# integrating).
IC_FILE="data/initial/cgols_initial_state${IC_TAG}.npy"
if [ ! -f "$IC_FILE" ]; then
    echo "Building initial conditions ($IC_FILE)..."
    CGOLS_CREATE_IC=1 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" python cgols.py
fi

# Step 2: the production run (loads the IC, shards over PROD_SPLIT GPUs, integrates,
# streams snapshot frames to cgols_snapshots/ + rolling checkpoints to
# data/cgols_checkpoints/, saves the final state to data/final/cgols_final_state.npy;
# set CGOLS_RUN_TAG to suffix all per-run outputs if several runs must coexist).
echo "Starting production run at ${CGOLS_DIM}x${CGOLS_DIM}x$((2*CGOLS_DIM)), split ${PROD_SPLIT}..."
CGOLS_CREATE_IC=0 CGOLS_SHARD_SPLIT="$PROD_SPLIT" python cgols.py

echo "Done."
