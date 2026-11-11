#!/bin/bash
#SBATCH --job-name=cgols_render_extract
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=cpu
#SBATCH --time=08:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0

#SBATCH --output=cgols_render_extract_%j.out
#SBATCH --error=cgols_render_extract_%j.err

# cgols_render_extract.py on HoreKa, CPU only: reduces checkpoints on the
# workspace to float16 render volumes (log n, log T, v_r) in
# data/renders/<tag>_f<factor>/<step>/, which are small enough to pull
# (A-2048 at factor 2: 12.9 GB per checkpoint instead of 365 GB). Resumable:
# finished checkpoints (meta.json present) are skipped.
#
# EXTRACT_RUNS = space-separated <tag>:<factor>:<steps>, e.g.
#
#   cd ~/astronomix-ana/pytests/self_gravity/cgols
#   sbatch --export=ALL,EXTRACT_RUNS="_A2048:2:28,60,16,40 _C1024:1:28,60,16,40" run_horeka_render_extract.sh
#   sbatch --export=ALL,EXTRACT_RUNS="_A2048:2:1-60",EXTRACT_FIELDS=logn,logT run_horeka_render_extract.sh  # movie

export CGOLS_SITE="${CGOLS_SITE:-horeka2}"
# the job starts in --chdir (or the submit dir); SLURM_SUBMIT_DIR ignores --chdir
source "$PWD/_site_env.sh" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

[ -f "$REPO/COMMIT" ] && echo "code copy of: $(cat "$REPO/COMMIT")"
cd "$CGOLS_DIR" || exit 1
[ -L data ] && [ -d data ] || { echo "ERROR: link ./data to the workspace first" >&2; exit 1; }
: "${EXTRACT_RUNS:?set EXTRACT_RUNS, e.g. _A2048:2:28,60}"
echo "node $SLURM_JOB_NODELIST: $(nproc) CPUs, $(free -g | awk '/Mem:/{print $2}') GB"

for run in $EXTRACT_RUNS; do
    IFS=: read -r tag factor steps <<< "$run"
    python cgols_render_extract.py --tag "$tag" --factor "$factor" --steps "$steps" \
        --workers "${EXTRACT_WORKERS:-64}" --fields "${EXTRACT_FIELDS:-logn,logT,vr}" || echo "WARNING: $tag failed"
done
