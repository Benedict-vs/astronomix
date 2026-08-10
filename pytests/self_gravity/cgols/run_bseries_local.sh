#!/usr/bin/env bash
#
# Radiative B-series (Schneider & Robertson 2018b, arXiv:1803.01005) at
# 512^2x1024, split over two local GPUs. Run it ON the machine with the free
# GPUs (home and scratch are NFS-shared, so any compgpu node sees the same tree
# and writes to the same outputs):
#
#     cd /export/home/bschuber/astronomix/pytests/self_gravity/cgols
#     ./run_bseries_local.sh
#
# Outputs are tagged _B throughout (snapshots, checkpoints, diag log, figures),
# so nothing collides with the adiabatic A-series production run.
#
# WHY THE COOLING dt LIMIT IS OFF HERE
# ------------------------------------
# The paper states a 10%-thermal-energy-per-step rule. Measured on 2026-07-31
# (2xH200): with it on, dt pins at ~1.15e-6 code (~11 yr) -> 6.7e6 steps ~ 44
# days for 75 Myr. The binding cells are the dense disk core, and a sharp
# (CGOLS_SMOOTHING_SIGMA=0) disk does NOT fix it: the CIE fit is zero only
# strictly BELOW 1e4 K, and at exactly 1e4 K it returns 5.2e-24, so the denser
# sharp disk (n_c = 196 cm^-3) still has t_cool = 64 yr.
#
# Turning the limit off is also the more faithful choice: the 10% rule appears
# only in the paper text and is absent from both public Cholla branches, while
# the 1%-per-sub-cycle loop IS in the Cholla source and handles the stiffness on
# its own. Where the B-series science lives (shocked shell, T ~ 1e6 K, n ~ 10)
# t_cool ~ 6000 yr against a ~830 yr CFL step, i.e. dt/t_cool ~ 0.14 - well
# inside what the sub-cycles resolve. Only the disk transient is unresolved, and
# there the floor clip lands the gas at 1e4 K, which is where it belongs.
#
# To reproduce the paper's stated scheme instead, set CGOLS_COOLING_DT_LIMIT=1
# and expect the run to take weeks at this resolution.

set -euo pipefail
cd "$(dirname "$0")"

# NCCL NVLink-SHARP multicast fails to initialise on a shared node when other
# jobs hold the fabric ("Failed to bind NVLink SHARP (NVLS) Multicast memory").
# Disabling NVLS costs a little collective bandwidth and avoids the crash.
export NCCL_NVLS_ENABLE=0

export CGOLS_COOLING=1
export CGOLS_COOLING_DT_LIMIT=0     # see the note above
export CGOLS_COOLING_SUBCYCLES=32   # cheap insurance for intermediate dt/t_cool
# REQUIRED for the radiative series: FCT positivity flux limiter. Cooling
# collapses the wind/disk contact to a ~1-cell 1e5:1 contrast where the raw
# WENO flux overshoots catastrophically (the 2026-07-31 blow-up); the blend
# makes the offending interfaces locally diffusive instead. Costs the fused
# WENO+divergence Pallas path (see cgols.py notes).
export CGOLS_PRESERVING_FLUX=1
export CGOLS_DIM=512
export CGOLS_SHARD_SPLIT="(1, 2, 1, 1)"   # x-split over 2 GPUs; 512/2 = 256, divisible by the (4,4,8) Pallas block
export CGOLS_NUM_SNAPSHOTS=60       # matches the 61-frame A-series reference

# Checkpointing stays at its default (every 3rd frame, keep 4) so a crash or a
# preemption costs at most a few frames; resume with CGOLS_RESTART_FROM=latest.

LOG="cgols_logs/run_512_B_console.log"
mkdir -p cgols_logs
echo "launching B-series -> $LOG"
nohup python cgols.py > "$LOG" 2>&1 &
echo "pid $!"
echo
echo "watch progress with:  tail -f $(pwd)/$LOG"
echo "resume after a crash: CGOLS_RESTART_FROM=latest ./run_bseries_local.sh"
