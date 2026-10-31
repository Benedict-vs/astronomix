#!/bin/bash
#SBATCH --job-name=cgols_conv3d
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=cpu
#SBATCH --time=01:30:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0

#SBATCH --output=cgols_conv3d_%j.out
#SBATCH --error=cgols_conv3d_%j.err

# cgols_convergence_3d.py on HoreKa, CPU only: measures every Orbax checkpoint
# of a run where it lies on the workspace, so only the small cache
# (data/studies/convergence3d_v3<TAG>.npz, ~10 MB) has to be pulled - not the
# checkpoints (FV-1024: 60 x 36 GB). Resumable: the cache is saved after every
# checkpoint and a resubmit measures only the missing ones.
#
# Run from a code copy (git archive) whose ./data links to the workspace:
#
#   cd ~/astronomix-ana/pytests/self_gravity/cgols
#   sbatch --export=ALL,CONV3D_TAG=_A1024fvr run_horeka_convergence3d.sh
#
# CONV3D_WORKERS: one checkpoint per worker, single-threaded, ~6.5 GB peak RSS.
# Measured locally on an FV-1024 checkpoint: 17 min per checkpoint, so 60
# checkpoints = ~17 CPU-h = two rounds of 30 workers (~35-55 min).

export CGOLS_SITE="${CGOLS_SITE:-horeka2}"
source "$SLURM_SUBMIT_DIR/_site_env.sh" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -f "$REPO/COMMIT" ] && echo "code copy of: $(cat "$REPO/COMMIT")"
cd "$CGOLS_DIR" || exit 1
[ -L data ] && [ -d data ] || { echo "ERROR: link ./data to the workspace first" >&2; exit 1; }
: "${CONV3D_TAG:?set CONV3D_TAG, e.g. _A1024fvr}"
echo "node $SLURM_JOB_NODELIST: $(nproc) CPUs, $(free -g | awk '/Mem:/{print $2}') GB"

python cgols_convergence_3d.py --tag "$CONV3D_TAG" --workers "${CONV3D_WORKERS:-30}"
