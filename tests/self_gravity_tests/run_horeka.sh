#!/bin/bash
# =============================================================================
# HoreKa (NHR@KIT) batch script for the CGOLS wind run (cgols.py).
#   docs: https://www.nhr.kit.edu/userdocs/horeka/batch/
#
#   submit:        sbatch run_horeka.sh
#   queue:         squeue -u $USER
#   start est.:    squeue --start -j <job_id>
#   follow log:    tail -f cgols_horeka.out
#   cancel:        scancel <job_id>
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

#SBATCH --output=cgols_horeka.out
#SBATCH --error=cgols_horeka.err

set -euo pipefail

# ---- Run configuration (the cgols.py env knobs) -----------------------------
# RES         : cells along x=y; z is always 2*RES (so 1024 -> 1024x1024x2048).
# IC_TAG      : suffix on the saved IC .npy so different resolutions don't clobber.
# SHARD_SPLIT : (var,x,y,z) device split; product = #GPUs for the PRODUCTION run.
#               Per-device slice must stay divisible by pallas_block_shape (4,4,8):
#               1024^3 with (1,2,2,1) -> x=512,y=512,z=2048 all OK.
export CGOLS_DIM=1024
export CGOLS_IC_TAG="_horeka1024"
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

# ---- Diagnostics ------------------------------------------------------------
echo "Node:  ${SLURM_JOB_NODELIST:-?}"
echo "GPUs:  ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# Background GPU usage logger.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv -l 30 > cgols_gpu_usage.csv &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

# ---- Run --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"   # the tests/self_gravity_tests dir you sbatch from; cgols.py
                         # anchors all I/O to its own dir regardless, but cd keeps
                         # the .out/.err/.csv logs next to it.

# NOTE on autocvd: cgols.py calls autocvd(num_gpus=NUM_GPUS) at import, which picks
# free GPUs by nvidia-smi. Under SLURM --gres the job is cgroup-isolated to its
# allocated GPUs, so autocvd selects those. If it ever hangs ("waiting for N free
# GPUs"), add the CGOLS_NO_AUTOCVD guard described in cgols-horeka-setup.md.

# Step 1: build + save the initial conditions if not already present (single GPU;
# CGOLS_CREATE_IC=1 builds, saves the IC .npy, and EXITS without integrating).
IC_FILE="cgols_initial_state${CGOLS_IC_TAG}.npy"
if [ ! -f "$IC_FILE" ]; then
    echo "Building initial conditions ($IC_FILE)..."
    CGOLS_CREATE_IC=1 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" python cgols.py
fi

# Step 2: the production run (loads the IC, shards over PROD_SPLIT GPUs, integrates,
# streams snapshots to cgols_snapshots/, saves cgols_final_state${IC_TAG}? -> note
# cgols.py saves cgols_final_state.npy untagged; rename per-resolution if needed).
echo "Starting production run at ${CGOLS_DIM}x${CGOLS_DIM}x$((2*CGOLS_DIM)), split ${PROD_SPLIT}..."
CGOLS_CREATE_IC=0 CGOLS_SHARD_SPLIT="$PROD_SPLIT" python cgols.py

echo "Done."
