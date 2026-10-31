#!/bin/bash
#SBATCH --job-name=cgols_fv2048
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=gpu-h200
#SBATCH --time=20:00:00
#SBATCH --exclusive

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --output=cgols_%j.out
#SBATCH --error=cgols_%j.err

# FV (VL2 + PLM + HLLC) CGOLS model A at 2048^2 x 4096 on HoreKa 2 Ruby
# (gpu-h200): 32x H200 over 8 nodes, one process per GPU, split (1, 8, 4, 1),
# in-situ IC, 75 Myr, tag _A2048fv. The FV twin of run_horeka_2048.sh.
#
#   cd ~/astronomix-ppm/pytests/self_gravity/cgols
#   sbatch run_horeka_fv2048.sh                                     # leg 1, fresh
#   sbatch --dependency=afterany:<leg-1 id> \
#          --export=ALL,CGOLS_RESTART_FROM=latest run_horeka_fv2048.sh   # resume leg
#
# A resume leg continues the same tag from its newest checkpoint (frames,
# checkpoint numbering and diag log continue), and exits right after the
# restore if leg 1 already reached 75 Myr, so it can be queued up front as
# insurance. NEVER resubmit WITHOUT CGOLS_RESTART_FROM=latest once leg 1 has
# written checkpoints: a fresh start clears data/cgols_checkpoints_A2048fv.
#
# 4 GHOST CELLS. The VL2 stages and the padded helper data are split along the
# ghost-padded extent N + 2g. With PLM's 2 ghost cells 2048 + 4 = 4 x 513 does
# not split 8 ways (the old code stops at startup with an IndivisibleError
# placing helper_data.r); cgols' CGOLS_FV_GHOST_CELLS=auto picks 4 here
# (2056 = 8 x 257 = 4 x 514, the padded layout FD always uses). The extra
# layer is inert and costs 0.4% more padded cells. The .out shows
# "[fv] 4 ghost cells (scheme minimum 2; CGOLS_FV_GHOST_CELLS=auto, ...)".
#
# MEMORY. Per device the state shard (5, 256, 512, 4096) equals FV-1024's at
# (1, 2, 2, 1); the kernel's halo-padded block is 268 x 524 x 4096 vs
# 524 x 524 x 2048 (+2.3%). FV-1024 (job 106337) compiled to 118.5 GiB/device,
# so ~120-121 GiB/device here; the CPU AOT compile of the bench program gives
# 2048@(1,8,4,1) / 1024@(1,2,2,1) = 1.013. The cuda_async pool is
# 0.98 x 140.4 GiB = 137.6 GiB, ~17 GiB of headroom. Exactly 2x the
# (1, 4, 2, 1) 1024 smoke bench (run_horeka_fv_multinode_smoke.sh).
#
# TIME (set --time from the smoke job's numbers; the default is the arithmetic):
#   steps     FV-512 16,281 -> FV-1024 33,359 (x2.05)  =>  2048 ~68,400
#   s/step    FV-1024 production 0.523 s (1 node, incl. FOFC, frames, 60 x 43 GB
#             checkpoints), same per-device work here; inter-node halo: FD went
#             2.10 (1024, 1 node) -> 2.54 s (2048, 8 nodes) = +0.44 s, and VL2
#             moves ~1/4 of FD's halo bytes per step (2 stages x 2-cell halos vs
#             5 LSRK stages x 3-cell WENO halos) => +0.1-0.2 s => 0.65-0.75 s
#   stepping  68,400 x 0.65-0.75 s = 12.3-14.3 h
#   ckpts     60 x 344 GB = 43 GB per node each, as at 1024 (inside the 0.523 s);
#             if hfs2 caps the aggregate at 5 (2) GB/s: +1.2 (+2.9) h
#   setup     in-situ IC + compile + NCCL init ~0.3 h
#   total     12.6-17.5 h  =>  --time=20:00:00, ~420-560 GPU-h
# Tighten to  steps x (2 x s/step(b1024_s421) x 0.523 / s/step(b1024_s221_1n))
#             + 60 x save time(ckpt_io) + 0.5 h  from the smoke summary.txt.
#
# CHECKPOINTS (344 GB each, sharded, one per frame = every 1.25 Myr):
#   CGOLS_CHECKPOINT_KEEP=2           rolling: the newest 2 (crash safety net)
#   CGOLS_CHECKPOINT_ARCHIVE_EVERY=2  never prune every 2nd frame's (2.5 Myr)
#   => 30 archived states (2.5 ... 75 Myr) = ~10.3 TB on the workspace, plus the
#   rolling pair and one in flight: ~11 TB peak. ARCHIVE_EVERY=1 keeps all 60
#   (~21 TB), =4 keeps 15 (~5 TB), =0 only the rolling pair.
#
# BEFORE SUBMITTING (HOME has a quota; frames and figures are written next to
# this script, checkpoints into data/ = the workspace):
#   W=/hfs2/work/workspace/scratch/hd_ah321-cgols
#   mkdir -p $W/frames/cgols_snapshots_A2048fv $W/figures/A2048fv
#   ln -s $W/frames/cgols_snapshots_A2048fv cgols_snapshots_A2048fv
#   ln -s $W/figures/A2048fv figures/cgols/A2048fv
# The runner refuses to start while either is a plain directory or missing.
#
# Launch mechanics (environment, preflight, workspace check, srun flags, NCCL
# watchdog) are run_cgols_multinode.sh + _site_env.sh, as for FD-2048; the
# launcher puts this tree first on PYTHONPATH, so a code copy shadows the
# editable install (the preflight's "astronomix:" line shows which was used).

export CGOLS_DIM=2048
export CGOLS_SHARD_SPLIT="(1, 8, 4, 1)"
export CGOLS_SOLVER=fv
export CGOLS_RIEMANN=hllc
export CGOLS_FV_RECON=plm
export CGOLS_FV_GHOST_CELLS="${CGOLS_FV_GHOST_CELLS:-auto}"
export CGOLS_IC_MODE=insitu
export CGOLS_RUN_TAG="${CGOLS_RUN_TAG:-_A2048fv}"
export CGOLS_CHECKPOINT_EVERY=1
export CGOLS_CHECKPOINT_KEEP="${CGOLS_CHECKPOINT_KEEP:-2}"
export CGOLS_CHECKPOINT_ARCHIVE_EVERY="${CGOLS_CHECKPOINT_ARCHIVE_EVERY:-2}"
export CGOLS_SITE="${CGOLS_SITE:-horeka2}"
# Empty unless overridden, so a bare sbatch is a fresh leg.
export CGOLS_RESTART_FROM="${CGOLS_RESTART_FROM:-}"

# Under sbatch this file runs as Slurm's spool copy, so BASH_SOURCE does not
# lead back to the repo; ask Slurm for the path that was submitted instead.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$HERE/_site_env.sh" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    _cmd="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=//p')"
    if [ -n "$_cmd" ]; then
        case "$_cmd" in /*) ;; *) _cmd="${SLURM_SUBMIT_DIR:-.}/$_cmd" ;; esac
        HERE="$(cd "$(dirname "$_cmd")" 2>/dev/null && pwd)"
    else
        HERE="${SLURM_SUBMIT_DIR:-}"
    fi
fi
if [ ! -f "$HERE/_site_env.sh" ]; then
    echo "ERROR: cannot locate the cgols directory (got '$HERE')" >&2
    exit 1
fi

for _out in "cgols_snapshots${CGOLS_RUN_TAG}" "figures/cgols/${CGOLS_RUN_TAG#_}"; do
    if [ ! -L "$HERE/$_out" ]; then
        echo "ERROR: $HERE/$_out must be a symlink into the workspace (HOME quota)," >&2
        echo "       see the header of this script." >&2
        exit 1
    fi
done

echo "FV-2048: tag $CGOLS_RUN_TAG, checkpoints keep $CGOLS_CHECKPOINT_KEEP + every" \
     "${CGOLS_CHECKPOINT_ARCHIVE_EVERY}. frame, restart '${CGOLS_RESTART_FROM}'"
exec "$HERE/run_cgols_multinode.sh"
