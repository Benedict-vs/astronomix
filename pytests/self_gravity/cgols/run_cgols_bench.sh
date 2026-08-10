#!/bin/bash
#SBATCH --job-name=cgols_bench
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=accelerated-h200
#SBATCH --time=01:00:00

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_bench_%j.out
#SBATCH --error=cgols_bench_%j.err

# Fixed-step calibration bench: CGOLS_BENCH_STEPS equal-sized steps, no CFL, no
# snapshots, no checkpoints, no progress bar - just the per-device memory report
# and the post-compile wall time. This is the GO/NO-GO measurement before any
# multi-day slot is committed.
#
#   sbatch run_cgols_bench.sh                                   # 2048 on 32 GPUs
#   sbatch --nodes=2 --export=ALL,CGOLS_DIM=1024 run_cgols_bench.sh
#   sbatch --nodes=4 --export=ALL,CGOLS_DIM=1024 run_cgols_bench.sh
#   sbatch --nodes=8 --export=ALL,CGOLS_DIM=1024 run_cgols_bench.sh
#
# The three 1024 points at 8 / 16 / 32 GPUs are the calibration set: at fixed
# global size they give the true per-device memory law INCLUDING the halo, plus
# the strong-scaling efficiency across the node boundary. FIT THEM AND
# EXTRAPOLATE - do not trust a single-measurement estimate. In particular, check
# whether the ~112 GB contiguous temp arena scales linearly with per-device
# cells or carries a fixed component: if it does, the 32-GPU H200 option is the
# first to die and the 64-GPU H100 option has to absorb it.
#
# READING THE OUTPUT: the memory_analysis printer labels MiB as "MB"
# (time_integration.py divides bytes by 1024^2). The 1024 run's
# "Total size: 133723.81 MB" was really 140.2 GB/device against a 147.7 GB
# cuda_async pool. Multiply s/step by ~135,000 for the 2048 wall-clock estimate,
# then size the legs from the LATE-run throughput decay, not from leg 1.

export CGOLS_BENCH_STEPS="${CGOLS_BENCH_STEPS:-20}"
export CGOLS_DIM="${CGOLS_DIM:-2048}"
# Leave the split unset to get the near-square auto-split of the allocation.
export CGOLS_SHARD_SPLIT="${CGOLS_SHARD_SPLIT:-}"
export CGOLS_SITE="${CGOLS_SITE:-horeka}"
# A bench must never resume or write into a production tag.
export CGOLS_RESTART_FROM=""
export CGOLS_RUN_TAG="${CGOLS_RUN_TAG:-_bench}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_cgols_multinode.sh"
