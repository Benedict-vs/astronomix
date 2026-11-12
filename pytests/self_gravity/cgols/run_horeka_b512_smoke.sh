#!/bin/bash
#SBATCH --job-name=b512_smoke
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=gpu-h200
#SBATCH --time=01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2

#SBATCH --output=cgols_b512smoke_%j.out
#SBATCH --error=cgols_b512smoke_%j.err

# GPU smoke test of the KI02 model-B speedups (c56c6dd) before B-1024.
#
# The production B-512 run (job 127150, 2 x H200, split 1-2-1-1) slowed from
# ~1.1 s/step before 50 Myr to 4.0-4.8 s/step after 52.5 Myr, because
# jax.scipy.special.exp1 in the disk-FUV slab factor ran data-dependent loops
# (fixed in cd1be44). This job restarts from that run's checkpoint 43
# (53.75 Myr, in the slow phase) on the same hardware and layout, and measures
# s/step with the fix. It only READS the production checkpoint directory; it
# writes under its own run tags.
#
#   default   default settings (the fix is always on), tag _B512smoke, with
#             checkpoints on, so the state at 55 Myr can be compared with
#             production checkpoint 44
#   pallas    CGOLS_COOLING_PALLAS=1, tag _B512smokep: first GPU (Triton)
#             compile of the fused cooling kernel, plus its s/step
#
# H200 rather than the H100 dev queue: the s/step is then directly comparable
# with job 127150, and the per-device memory is the production one.
#
#   cd ~/astronomix-bseries/pytests/self_gravity/cgols && sbatch run_horeka_b512_smoke.sh
#
# Output: cgols_logs/b512smoke_<jobid>/ (summary.txt; rate_<step>.tsv =
# "unix time, diag lines", one diag line per step).

export CGOLS_SOLVER=fd CGOLS_DIM=512 CGOLS_COOLING=1 CGOLS_COOLING_DT_LIMIT=0 \
       CGOLS_COOLING_SUBCYCLES=32 CGOLS_IC_MODE=insitu \
       CGOLS_CHECKPOINT_KEEP=0 CGOLS_CHECKPOINT_ARCHIVE_EVERY=0
export CGOLS_SHARD_SPLIT="(1, 2, 1, 1)"
export CGOLS_SITE="${CGOLS_SITE:-horeka2}"
SOURCE_CKPT="${SOURCE_CKPT:-/hfs2/work/workspace/scratch/hd_ah321-cgols/cgols_checkpoints_B512}"
SOURCE_STEP="${SOURCE_STEP:-43}"
LIMIT_DEFAULT="${LIMIT_DEFAULT:-34m}"
LIMIT_PALLAS="${LIMIT_PALLAS:-16m}"

source "$SLURM_SUBMIT_DIR/_site_env.sh" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cgols_preflight
[ -f "$REPO/COMMIT" ] && echo "code copy of: $(cat "$REPO/COMMIT")"
cgols_check_workspace || exit 1
echo "Running on node: $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION)"
nvidia-smi -L

cd "$CGOLS_DIR" || exit 1
OUT=cgols_logs/b512smoke_${SLURM_JOB_ID}
mkdir -p "$OUT"
[ -d "$SOURCE_CKPT/$SOURCE_STEP" ] || { echo "missing $SOURCE_CKPT/$SOURCE_STEP"; exit 1; }

run() {  # run <name> <limit> <tag> [VAR=value ...]
    local name=$1 limit=$2 tag=$3; shift 3
    local diag=cgols_logs/cgols_diag${tag}.log
    echo "=== $name ($tag, limit $limit) $(date +%T)" | tee -a "$OUT/summary.txt"
    (while true; do
        echo "$(date +%s) $(cat "$diag" 2>/dev/null | wc -l)" >> "$OUT/rate_${name}.tsv"
        sleep 60
    done) &
    local sampler=$!
    env CGOLS_RUN_TAG="$tag" CGOLS_RESTART_FROM="$SOURCE_CKPT" CGOLS_RESTART_STEP="$SOURCE_STEP" \
        "$@" timeout --signal=TERM --kill-after=30s "$limit" python cgols.py \
        > "$OUT/${name}.log" 2>&1
    local rc=$?
    kill $sampler 2>/dev/null
    echo "$(date +%s) $(cat "$diag" 2>/dev/null | wc -l)" >> "$OUT/rate_${name}.tsv"
    {
        echo "exit $rc (124 = hit the time limit, expected)"
        grep -E "Restarting from|Error|error:|OOM|RESOURCE_EXHAUSTED|Traceback" "$OUT/${name}.log" | head -5
        tail -1 "$diag" 2>/dev/null | cut -c1-120
        # s/step over the run, excluding the first 5 minutes (compile + warm-up)
        awk 'NR==1{t0=$1} {t[NR]=$1; n[NR]=$2} END{for(i=1;i<=NR;i++) if(t[i]-t0>=300){a=i;break}
             if(a && NR>a && n[NR]>n[a]) printf "s/step after warm-up: %.3f (%d steps in %d s)\n",
             (t[NR]-t[a])/(n[NR]-n[a]), n[NR]-n[a], t[NR]-t[a]; else print "s/step: not enough samples"}' \
            "$OUT/rate_${name}.tsv"
        ls -d "$(dirname "$SOURCE_CKPT")/cgols_checkpoints${tag}"/[0-9]* 2>/dev/null | tail -2
    } | tee -a "$OUT/summary.txt"
}

run default "$LIMIT_DEFAULT" _B512smoke CGOLS_CHECKPOINT_EVERY=1
run pallas "$LIMIT_PALLAS" _B512smokep CGOLS_CHECKPOINT_EVERY=0 CGOLS_COOLING_PALLAS=1
echo "done $(date +%T)" | tee -a "$OUT/summary.txt"
