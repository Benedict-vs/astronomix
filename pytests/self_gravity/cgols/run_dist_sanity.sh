#!/bin/bash
#SBATCH --job-name=cgols_distsanity
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev-gpu-h100
#SBATCH --exclusive
#SBATCH --time=00:15:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --output=distsanity_%j.out
#SBATCH --error=distsanity_%j.err

# Ladder rungs 1 and 2: prove the multi-process rendezvous and a cross-process
# collective BEFORE committing anything larger. Minutes, on the dev queue.
#
#   rung 1  (intra-node NCCL P2P under --gpu-bind=none), HoreKa 2 dev queue
#           (dev-gpu-h100: 1 node, 1 h, one job at a time; there is no
#           dev-gpu-h200, but the x86 env and the jax/NCCL stack are the same):
#       sbatch run_dist_sanity.sh
#   rung 2  (inter-node IB rendezvous - dev partitions cap at ONE node, so the
#           2-node test goes to a short production-partition job, which
#           backfills quickly):
#       sbatch --nodes=2 --partition=gpu-h200 --time=00:10:00 run_dist_sanity.sh
#
# Expect, from rank 0:   allgather = [0. 1. 2. 3.]   then   PASS
#
# Rung 2 is where a jax-version surprise would surface: the multi-node skill was
# validated on jax 0.9.2 and the `astro` env runs 0.10. The completed 1024
# 4-GPU run proves 0.10 + Pallas + GSPMD works INTRA-node; this is the
# inter-node half of that claim.

set -u

# Under sbatch this file runs as Slurm's spool copy
# (/var/spool/slurmd/job<id>/slurm_script), so BASH_SOURCE does not lead back to
# the repo; ask Slurm for the path that was submitted instead.
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
source "$HERE/_site_env.sh" || exit 1

cgols_preflight

# --gpu-bind=none and no --ntasks: see the note in run_cgols_multinode.sh.
srun --gpu-bind=none --kill-on-bad-exit=1 python "$REPO/pytests/_dist_sanity.py"
rc=$?
echo "=== srun exit code: $rc ==="
exit $rc
