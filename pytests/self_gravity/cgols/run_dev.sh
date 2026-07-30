#!/bin/bash
#SBATCH --job-name=cgols_dev
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=00:30:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200gb

#SBATCH --output=cgols_dev_%j.out
#SBATCH --error=cgols_dev_%j.err

# Pipeline validation on the dev queue (~15 min): builds the 512 ICs if
# missing, then runs a 5-step fixed-dt benchmark through the production
# sharding path. Exercises everything the real run needs (modules, env,
# linker paths, autocvd under slurm, Pallas compile on Hopper, multi-GPU
# NCCL, workspace I/O) without the multi-hour commit.
# - run with: sbatch run_dev.sh   (from pytests/self_gravity/cgols/)
# - success: per-device memory analysis + elapsed-time printout, exit 0
#   (bench steps are fixed-dt, physically meaningless; no state is saved)

export CGOLS_DIM=512
BENCH_SPLIT="(1, 2, 2, 1)"   # same shard layout as production, product = #GPUs

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

# jax-specific
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# ICs go to the workspace, same as production:
#   ws_allocate cgols 60 && ln -s "$(ws_find cgols)" data
cd "$SLURM_SUBMIT_DIR"
[ -L data ] && [ -d data ] || { echo "ERROR: link ./data to a workspace first" >&2; exit 1; }

# debugging
echo "Running on node: $SLURM_JOB_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# build the 512 ICs once, then a 5-step benchmark on the sharded path
IC_FILE="data/initial/cgols_initial_state_d${CGOLS_DIM}.npy"
[ -f "$IC_FILE" ] || CGOLS_CREATE_IC=1 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" python cgols.py
CGOLS_BENCH_STEPS=5 CGOLS_SHARD_SPLIT="$BENCH_SPLIT" python cgols.py \
    && echo "DEV VALIDATION PASSED" || echo "DEV VALIDATION FAILED (see .err)"
