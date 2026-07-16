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

# Usage info:
# - run with: sbatch run_horeka.sh   (from tests/self_gravity_tests/)
# - check status with: squeue -u $USER
# - estimate start time: squeue --start -j <job_id>
# - read output: tail -f cgols_<job_id>.out
# - cancel job: scancel <job_id>

# run configuration; every other CGOLS_* knob defaults to the production values
export CGOLS_DIM=1024
PROD_SPLIT="(1, 2, 2, 1)"   # (var,x,y,z) shard split, product = #GPUs

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

# Leg 2: job 4830581 banked 0-63% in 24h (TIMEOUT); continue from the last
# rolling checkpoint (~60%). The fresh RUN_TAG keeps leg 1's frames/checkpoints
# from being wiped by the startup cleanup. For a fresh full run, drop the two
# CGOLS_RESTART_FROM / CGOLS_RUN_TAG variables again.
CGOLS_RESTART_FROM="data/cgols_checkpoints/checkpoint_0036.npz" CGOLS_RUN_TAG="_leg2" \
    CGOLS_SHARD_SPLIT="$PROD_SPLIT" python cgols.py
