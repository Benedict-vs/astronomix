#!/bin/bash
#SBATCH --job-name=cgols_ic1024
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --output=cgols_ic_%j.out
#SBATCH --error=cgols_ic_%j.err

# Pre-builds the 1024 ICs on the host CPU so the production job skips its own
# IC-build step (run_horeka.sh only builds when the file is missing). Needed
# because the eager single-GPU IC build transiently holds ~90 GB at 1024 -
# fine in host RAM (192 GB with gpu:1), OOMs an 80 GB H100. A GPU must still
# be allocated: cgols.py runs autocvd (nvidia-smi) before JAX_PLATFORMS=cpu
# takes effect, so this cannot go on the cpuonly partition.

# activate env
source ~/.bashrc
micromamba activate astro

cd "$SLURM_SUBMIT_DIR"
[ -L data ] && [ -d data ] || { echo "ERROR: link ./data to a workspace first" >&2; exit 1; }

JAX_PLATFORMS=cpu CGOLS_DIM=1024 CGOLS_CREATE_IC=1 CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" python cgols.py \
    && echo "IC BUILD DONE" || echo "IC BUILD FAILED (see .err)"
