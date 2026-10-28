#!/bin/bash
#SBATCH --job-name=fvval
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=gpu-h200
#SBATCH --time=01:30:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_fvval_%j.out
#SBATCH --error=cgols_fvval_%j.err

# GPU validation of the sharded VL2 Pallas path (ghost cells + mesh) and of PPM
# before FV-1024 is committed. The fix was only checked on CPU (pallas_interpret);
# this is the first Triton compile. Two parts, all on 4x H200:
#
#   1. correctness: model A 128^3 to 7.5 Myr (past the 5 Myr wind onset), PLM and
#      PPM, unsharded vs split (1,2,2,1); cgols_vl2_probe.py reports which VL2
#      stage path the sharded runs traced (a silent native fallback shows up
#      there), and
#      cgols_fv_validate_compare.py compares the frames.
#   2. memory: CGOLS_BENCH_STEPS fixed steps at (1,2,2,1) for FV PLM / FV PPM / FD
#      at 512^3 and 1024^3 - compiled per-device size (printer says "MB", means
#      MiB) plus the nvidia-smi peak per card, and the post-compile s/step.
#      Decides 4 vs 8 GPUs for FV-1024; FD at the same split is the like-for-like
#      reference (FD-1024 July: 133724 MiB/device).
#
# Run from a code COPY (git archive of the fv-ppm commit), never ~/astronomix:
#
#   cd ~/astronomix-ppm/pytests/self_gravity/cgols && sbatch run_horeka_fv_validate.sh
#
# Quick version on the dev queue (1 node, 4x H100 94 GB, 1 h max): correctness +
# 512^3 only, since FD-1024 at (1,2,2,1) needs ~140 GB/device:
#
#   sbatch --partition=dev-gpu-h100 --time=01:00:00 --export=ALL,FVVAL_DIMS=512 run_horeka_fv_validate.sh
#
# Everything lands in cgols_logs/fvval_<jobid>/ (summary.txt = the short answer).

export CGOLS_SITE="${CGOLS_SITE:-horeka2}"
source "$SLURM_SUBMIT_DIR/_site_env.sh" || exit 1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

cgols_preflight
[ -f "$REPO/COMMIT" ] && echo "code copy of: $(cat "$REPO/COMMIT")"
cgols_check_workspace || exit 1
echo "Running on node: $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION)"
nvidia-smi -L

cd "$CGOLS_DIR" || exit 1
OUT=cgols_logs/fvval_${SLURM_JOB_ID}
mkdir -p "$OUT"
export CGOLS_IC_MODE=insitu CGOLS_CHECKPOINT_EVERY=0 CGOLS_RESTART_FROM="" CGOLS_RIEMANN=hllc

run(){   # $1 = name, $2 = time limit, rest = env assignments
  local name=$1 limit=$2; shift 2
  echo "=== $name ($*)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 1000 > "$OUT/$name.smi" &
  local smi=$!
  # The probe imports astronomix (hence jax) before cgols' autocvd picks the
  # card(s), so it is only used where all 4 cards are wanted anyway.
  local probe=cgols_vl2_probe.py
  case "$*" in *"(1, 1, 1, 1)"*) probe= ;; esac
  env "$@" timeout "$limit" python $probe cgols.py > "$OUT/$name.log" 2>&1
  local rc=$?
  kill $smi 2>/dev/null
  echo "rc=$rc" >> "$OUT/$name.log"
  echo "    rc=$rc  $(grep -m1 -A3 'Compiled memory usage PER DEVICE' "$OUT/$name.log" | grep 'Total size')"
}

# 1. correctness (128^3, ~1-2 min each)
for recon in plm ppm; do
  for split in "1, 1, 1, 1" "1, 2, 2, 1"; do
    n=v128${recon}_s$(echo "$split" | tr -d ' ,' | sed 's/^1//;s/1$//')
    run "$n" 15m CGOLS_SOLVER=fv CGOLS_FV_RECON=$recon CGOLS_DIM=128 CGOLS_END_FRACTION=0.1 \
        CGOLS_SHARD_SPLIT="($split)" CGOLS_RUN_TAG=_fvval_$n
  done
done
python cgols_fv_validate_compare.py _fvval_v128plm_s11 _fvval_v128plm_s22 _fvval_v128ppm_s11 _fvval_v128ppm_s22 \
  > "$OUT/compare.txt" 2>&1
cat "$OUT/compare.txt"

# 2. memory + s/step at (1,2,2,1)
for dim in ${FVVAL_DIMS:-512 1024}; do
  run b${dim}fvplm 20m CGOLS_SOLVER=fv CGOLS_FV_RECON=plm CGOLS_DIM=$dim CGOLS_BENCH_STEPS=10 \
      CGOLS_SHARD_SPLIT="(1, 2, 2, 1)" CGOLS_RUN_TAG=_bench
  run b${dim}fvppm 20m CGOLS_SOLVER=fv CGOLS_FV_RECON=ppm CGOLS_DIM=$dim CGOLS_BENCH_STEPS=10 \
      CGOLS_SHARD_SPLIT="(1, 2, 2, 1)" CGOLS_RUN_TAG=_bench
  run b${dim}fd 20m CGOLS_SOLVER=fd CGOLS_DIM=$dim CGOLS_BENCH_STEPS=5 \
      CGOLS_SHARD_SPLIT="(1, 2, 2, 1)" CGOLS_RUN_TAG=_bench
done

{
  echo "== correctness (128^3, 7.5 Myr)"; cat "$OUT/compare.txt"
  echo; echo "== VL2 stage path (probe)"
  for f in "$OUT"/*.log; do echo "$(basename "$f" .log): $(grep -m1 '\[probe\]' "$f" | sed 's/.*counts): //')"; done
  echo; printf "%-12s %-6s %14s %14s %12s\n" run rc compiled_MiB smi_peak_MiB wall
  for f in "$OUT"/b*.log; do
    n=$(basename "$f" .log)
    printf "%-12s %-6s %14s %14s %12s\n" "$n" "$(tail -1 "$f")" \
      "$(grep -m1 -A3 'Compiled memory usage PER DEVICE' "$f" | sed -n 's/.*Total size: \([0-9.]*\).*/\1/p')" \
      "$(awk -F', ' '$2>m{m=$2}END{print m}' "$OUT/$n.smi")" \
      "$(grep -m1 'time_integration wall time' "$f" | sed 's/.*wall time: \([0-9.]* s\).*/\1/')"
  done
} > "$OUT/summary.txt"
cat "$OUT/summary.txt"
