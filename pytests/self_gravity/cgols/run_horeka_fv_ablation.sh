#!/bin/bash
#SBATCH --job-name=cgols_fvabl
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=gpu-h100,gpu-h200
#SBATCH --time=04:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --output=cgols_%j.out
#SBATCH --error=cgols_%j.err

# Single-GPU FV (VL2) ablation runs of model A on HoreKa 2. The partition list
# lets Slurm start the job on whichever of Teal (gpu-h100, 94 GB) or Ruby
# (gpu-h200, 141 GB) frees a GPU first; 512^3 FV needs ~40 GB, so both fit.
#
# Run from a code COPY of the repo (not ~/astronomix, whose queued production
# legs import it at start): PYTHONPATH=$REPO shadows the editable install, and
# the preflight banner's "astronomix:" line proves which copy was imported.
#
#   sbatch --export=ALL,CGOLS_DIM=256,CGOLS_RIEMANN=hlle,CGOLS_RUN_TAG=_A256fvhlle \
#          --time=01:30:00 run_horeka_fv_ablation.sh
#   sbatch --export=ALL,CGOLS_DIM=512,CGOLS_RIEMANN=hlle,CGOLS_RUN_TAG=_A512fvhlle \
#          run_horeka_fv_ablation.sh
#
# Local reference timings (1x H200, fp32, HLLC): 256^3 23 min, 512^3 60 min.

export CGOLS_SOLVER=fv
export CGOLS_DIM="${CGOLS_DIM:?set CGOLS_DIM}"
export CGOLS_RUN_TAG="${CGOLS_RUN_TAG:?set CGOLS_RUN_TAG}"
export CGOLS_RIEMANN="${CGOLS_RIEMANN:-hllc}"
# in-situ IC: no pre-built d256/d512 .npy needed on the workspace
export CGOLS_IC_MODE="${CGOLS_IC_MODE:-insitu}"
# every 4th frame kept (5, 10, ..., 75 Myr) for later 3D analysis, as the
# A512fv precision set
export CGOLS_CHECKPOINT_EVERY="${CGOLS_CHECKPOINT_EVERY:-4}"
export CGOLS_CHECKPOINT_KEEP="${CGOLS_CHECKPOINT_KEEP:-0}"
export CGOLS_SHARD_SPLIT="(1, 1, 1, 1)"
export CGOLS_SITE="${CGOLS_SITE:-horeka2}"

# Under sbatch BASH_SOURCE is the spool copy; the job is submitted from the
# cgols directory of the code copy.
source "$SLURM_SUBMIT_DIR/_site_env.sh" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cgols_preflight
[ -f "$REPO/COMMIT" ] && echo "code copy of: $(cat "$REPO/COMMIT")"
cgols_check_workspace || exit 1

echo "Running on node: $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION)"
nvidia-smi -L

cd "$CGOLS_DIR" || exit 1
python cgols.py
