#!/bin/bash
#SBATCH --job-name=cgols_ic
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev_accelerated-h100
#SBATCH --time=01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --output=cgols_ic_%j.out
#SBATCH --error=cgols_ic_%j.err

# Builds the cgols ICs on the dev queue so run_horeka_1024.sh spends its whole
# allocation integrating. Run before a fresh production run.
#
#   sbatch run_ic_horeka.sh                             # d1024 (production)
#   sbatch --export=ALL,CGOLS_DIM=512 run_ic_horeka.sh  # any other grid
#
# JAX_PLATFORMS=cpu is required, not a preference: the eager single-device
# build transiently holds ~90 GB at 1024, which OOMs an 80 GB H100 but fits
# host RAM. A GPU must still be allocated because cgols.py runs autocvd
# (nvidia-smi) before JAX_PLATFORMS takes effect, so cpuonly will not work.

source ~/.bashrc
micromamba activate astro

cd "$SLURM_SUBMIT_DIR"
[ -L data ] && [ -d data ] || { echo "ERROR: link ./data to a workspace first" >&2; exit 1; }

CGOLS_DIM="${CGOLS_DIM:-1024}"
IC_STATE="data/initial/cgols_initial_state_d${CGOLS_DIM}.npy"
IC_POT="data/initial/cgols_initial_potential_d${CGOLS_DIM}.npy"

if [ -f "$IC_STATE" ] && [ -f "$IC_POT" ]; then
    echo "IC BUILD SKIPPED: d${CGOLS_DIM} ICs already present"
    exit 0
fi

JAX_PLATFORMS=cpu CGOLS_DIM="$CGOLS_DIM" CGOLS_CREATE_IC=1 \
    CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" python cgols.py \
    || { echo "IC BUILD FAILED (see .err)" >&2; exit 1; }
echo "IC BUILD DONE"
