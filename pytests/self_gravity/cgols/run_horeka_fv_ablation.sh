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
# 1024^3 (2.1e9 cells, ~205 B/cell measured at 512 => ~110 GB/device on 4)
# fits only 4x H200, split 1-2-2-1 (sbatch --export splits on commas, hence the
# dash form). Sharded VL2 matches the single-device run to rounding (CPU check,
# 2026-10-26). In-situ IC (no d1024 .npy on the workspace; the FD A-1024 run
# 93951 built its IC in situ too). Keep two rolling checkpoints (~43 GB each)
# plus frames 24 and 48 (30, 60 Myr):
#
#   sbatch --partition=gpu-h200 --gres=gpu:4 --time=16:00:00 \
#          --export=ALL,CGOLS_DIM=1024,CGOLS_RIEMANN=hllc,CGOLS_RUN_TAG=_A1024fv,SPLIT=1-2-2-1,CGOLS_CHECKPOINT_KEEP=2,CGOLS_CHECKPOINT_ARCHIVE_EVERY=24 \
#          run_horeka_fv_ablation.sh
#
# Resume after a TIMEOUT into the same tag: add CGOLS_RESTART_FROM=latest.
#
# Local reference timings (1x H200, fp32, HLLC): 256^3 23 min, 512^3 60 min.
# CGOLS_CHECKPOINT_EVERY is only an on/off gate: checkpoints are written every
# frame, and CGOLS_CHECKPOINT_KEEP=0 keeps ALL of them (512^3: 60 x 4.5 GB).

export CGOLS_SOLVER=fv
export CGOLS_DIM="${CGOLS_DIM:?set CGOLS_DIM}"
export CGOLS_RUN_TAG="${CGOLS_RUN_TAG:?set CGOLS_RUN_TAG}"
export CGOLS_RIEMANN="${CGOLS_RIEMANN:-hllc}"
# in-situ IC: no pre-built d256/d512 .npy needed on the workspace
export CGOLS_IC_MODE="${CGOLS_IC_MODE:-insitu}"
export CGOLS_CHECKPOINT_EVERY="${CGOLS_CHECKPOINT_EVERY:-1}"
export CGOLS_CHECKPOINT_KEEP="${CGOLS_CHECKPOINT_KEEP:-0}"
export CGOLS_CHECKPOINT_ARCHIVE_EVERY="${CGOLS_CHECKPOINT_ARCHIVE_EVERY:-0}"
export CGOLS_RESTART_FROM="${CGOLS_RESTART_FROM:-}"
# SPLIT=v-x-y-z (comma-free for sbatch --export), product = #GPUs
SPLIT="${SPLIT:-1-1-1-1}"
export CGOLS_SHARD_SPLIT="(${SPLIT//-/, })"
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
