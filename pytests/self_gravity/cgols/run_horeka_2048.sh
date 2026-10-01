#!/bin/bash
#SBATCH --job-name=cgols_2048
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=gpu-h200
#SBATCH --time=48:00:00
#SBATCH --exclusive

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_%j.out
#SBATCH --error=cgols_%j.err

# Production 2048^2 x 4096 CGOLS run on HoreKa 2 Ruby (gpu-h200): 32x H200
# across 8 of its 13 nodes, one process per GPU, 75 Myr, adiabatic A-series
# (pure hydro, no cooling), split (1, 8, 4, 1).
#
#   sbatch run_horeka_2048.sh                                      # leg 1, fresh
#   sbatch --export=ALL,CGOLS_RESTART_FROM=data/cgols_checkpoints,CGOLS_RUN_TAG=_leg2 \
#          run_horeka_2048.sh                                      # leg 2
#
# This file is only the production PARAMETER SET; all of the launch mechanics
# (environment, preflight banner, workspace check, srun flags, NCCL watchdog)
# live in run_cgols_multinode.sh + _site_env.sh, so a bench run or a different
# node count reuses exactly the same code path.
#
# SIZING (extrapolated from the completed 1024/4x H200 run: 140.2 GB/device,
# ~3.13 s/step, 70,576 steps in 61.45 h):
#   per-device slice   256 x 512 x 4096   = 1.013x the 1024 run's padded cells
#   memory             ~127-129 GB/device vs the 147.7 GB cuda_async pool
#   steps              ~133-140k          (step count scales with dim)
#   wall               ~120-150 h         => 3-4 legs of 48 h, ~3,900-4,800 GPU-h
#                      (the per-device load equals the 1024 run's, so the
#                      unknown is the inter-node halo cost: Leonard's GH200
#                      weak scaling lost ~20% going 1 -> 8 nodes, i.e. ~4 s/step)
#   disk               ~344 GB/checkpoint; KEEP=4 + one in flight ~1.7 TB
#
# BEFORE COMMITTING THE SLOT, run the ladder (see the plan's section 5) - in
# particular the 20-step 2048 bench (run_cgols_bench.sh at these node counts)
# and READ ITS memory read-out. Note the memory_analysis printer labels MiB as
# "MB": the 1024 run's "133,723.81 MB" was 140.2 GB. The estimate above rests on
# a single measurement and sits only ~12 GB under the proven program size.
#
# Alternative if the Ruby queue is unobtainable: gpu-h100 (Teal), 16 of 21 nodes /
# 64 GPUs, split (1, 8, 8, 1) - ~65 of 94 GB/device, more headroom and fewer
# legs, but slower cards and only 22 such nodes exist.
#
# NOT usable: 48 GPUs / 12 nodes. Every shard count must divide 512 (2048/gx
# divisible by the Pallas block 4, 4096/gz by 8), i.e. must be a power of two.

export CGOLS_DIM=2048
export CGOLS_SHARD_SPLIT="(1, 8, 4, 1)"
export CGOLS_SITE="${CGOLS_SITE:-horeka2}"

# Empty unless overridden, so a bare sbatch is a fresh leg.
export CGOLS_RESTART_FROM="${CGOLS_RESTART_FROM:-}"
export CGOLS_RUN_TAG="${CGOLS_RUN_TAG:-}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_cgols_multinode.sh"
