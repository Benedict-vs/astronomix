#!/usr/bin/env bash
#
# Adiabatic A-series at 512^2x1024 on two local GPUs, run with the CURRENT
# driver and the EXACT snapshot cadence of the radiative B-series, so that the
# two wall times are directly comparable.
#
# WHY: the A-vs-B cost comparison rests on an A wall time of "~4.5 h" that was
# never logged (the pre-Orbax npz driver printed no timing line), while B has a
# logged 20.85 h on 2 GPUs. A 20-step fixed-step bench at 256^2x512 puts the
# per-step cost ratio at 2.38x (essentially all of it the FCT flux limiter -
# the cooling operator itself is ~2%) and B took 1.27x more steps, which
# predicts ~3x, not the ~4.6x the wall times imply. This run closes that gap
# with a measured number instead of a remembered one.
#
# Everything below mirrors run_bseries_local.sh except the two physics knobs:
# cooling OFF and the FCT limiter OFF (the bare paper A-series).
#
# CAVEAT: the bare adiabatic run is only marginally stable in the 14-19 Myr
# high-mass-loading window and may clip-lock into a density runaway before
# reaching 75 Myr (see the 2026-08-01 _Avr blow-up). That does NOT invalidate
# the measurement: sec/step from however far it gets is the quantity we want,
# and the diag log records the step count. Divide the elapsed time by the diag
# line count if it dies early.

set -euo pipefail
cd "$(dirname "$0")"

# Same NCCL workaround as the B-series: NVLink-SHARP multicast fails to
# initialise on a shared node when other jobs hold the fabric.
export NCCL_NVLS_ENABLE=0

export CGOLS_COOLING=0
export CGOLS_PRESERVING_FLUX=0            # bare A-series numerics, as in the paper
export CGOLS_DIM=512
export CGOLS_SHARD_SPLIT="(1, 2, 1, 1)"   # identical to the B-series run
export CGOLS_NUM_SNAPSHOTS=60             # identical to the B-series run
export CGOLS_RUN_TAG="_Atime"             # keeps outputs off the _prod_tmax5e9 A-series

# Ask XLA to terminate itself on a stuck collective rather than waiting forever
# (the 2026-08-02 rendezvous deadlock). The warn path fires by default; whether
# the terminate path covers the first-call rendezvous in this build is unclear,
# hence the belt-and-braces watchdog below.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_first_collective_call_terminate_timeout_seconds=600 --xla_gpu_nccl_termination_timeout_seconds=600"

LOG="cgols_logs/run_512_Atime_console.log"
echo "launching A-series timing run -> $LOG"
echo "(autocvd blocks until two GPUs are completely free; this is the queue)"
exec ./run_with_watchdog.sh "$LOG" python cgols.py
