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

# Pre-builds the ICs on the host CPU, on the dev queue, so the production job
# (run_horeka_1024.sh) skips its own IC-build step and spends its whole
# allocation integrating. ALWAYS run this before a fresh production run.
#
# Needed because the eager single-device IC build transiently holds ~90 GB at
# 1024 - fine in host RAM (192 GB with gpu:1), OOMs an 80 GB H100. A GPU must
# still be allocated: cgols.py runs autocvd (nvidia-smi) before
# JAX_PLATFORMS=cpu takes effect, so this cannot go on the cpuonly partition.
#
# Writes data/initial/cgols_initial_{state,potential}_d${CGOLS_DIM}.npy
# (41 GB + 8.1 GB at 1024) - exactly the pair run_horeka_1024.sh checks for.
#
# Resolution is overridable, so this covers any grid, not just 1024:
#   sbatch run_ic1024.sh                                  # d1024 (production)
#   sbatch --export=ALL,CGOLS_DIM=512 run_ic1024.sh       # d512
# The dev queue caps at 1 h, which the 1024 build fits; a larger grid would
# need the production partition instead.

# activate env
source ~/.bashrc
micromamba activate astro

cd "$SLURM_SUBMIT_DIR"
[ -L data ] && [ -d data ] || { echo "ERROR: link ./data to a workspace first" >&2; exit 1; }

CGOLS_DIM="${CGOLS_DIM:-1024}"

# Skip if both files are already there, so re-submitting is harmless (the build
# overwrites in place and rewrites ~49 GB at 1024).
IC_STATE="data/initial/cgols_initial_state_d${CGOLS_DIM}.npy"
IC_POT="data/initial/cgols_initial_potential_d${CGOLS_DIM}.npy"
if [ -f "$IC_STATE" ] && [ -f "$IC_POT" ]; then
    echo "IC BUILD SKIPPED: d${CGOLS_DIM} ICs already present"
    exit 0
fi

JAX_PLATFORMS=cpu CGOLS_DIM="$CGOLS_DIM" CGOLS_CREATE_IC=1 \
    CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" python cgols.py \
    && echo "IC BUILD DONE" || { echo "IC BUILD FAILED (see .err)"; exit 1; }
