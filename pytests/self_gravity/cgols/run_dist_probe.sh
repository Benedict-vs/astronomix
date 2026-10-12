#!/bin/bash
#SBATCH --job-name=cgols_probe
#SBATCH --account=hk-project-pai00101
#SBATCH --partition=dev-gpu-h100
#SBATCH --exclusive
#SBATCH --time=00:58:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

#SBATCH --output=probe_%j.out
#SBATCH --error=probe_%j.err

# Localise the silent hang of job 67285 (D2: cgols, 4 processes, TO_DISK) in
# one dev-queue slot. Each stage of pytests/_dist_callback_probe.py is its own
# srun step with a faulthandler timeout, so a hung stage dumps every rank's
# Python stack into the .err and the job moves on to the next stage. Last, a
# tiny cgols run under `timeout` (SIGTERM makes cgols dump its stacks too),
# as committed and with CGOLS_CALLBACK_PARTITIONED=1.
#
#   sbatch run_dist_probe.sh
#
# Read the result with:  grep -E "PASS|exit:|callback fired|Thread|File" probe_<id>.*

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
cd "$CGOLS_DIR" || exit 1

cgols_preflight
cgols_check_workspace || exit 1

for stage in cb_noarg cb_rep cb_part todisk; do
    echo "=== probe $stage ===" | tee /dev/stderr
    PROBE_TIMEOUT=$([ "$stage" = todisk ] && echo 600 || echo 180) \
        srun --gpu-bind=none --kill-on-bad-exit=1 \
        python "$REPO/pytests/_dist_callback_probe.py" "$stage"
    echo "=== probe $stage exit: $? ===" | tee /dev/stderr
done

# cgols twice: as committed, then with the per-device frame callbacks.
for part in 0 1; do
    echo "=== probe cgols (DIM=128, CGOLS_CALLBACK_PARTITIONED=$part) ===" | tee /dev/stderr
    CGOLS_DIM=128 CGOLS_DISTRIBUTED=1 CGOLS_END_FRACTION=0.01 CGOLS_NUM_SNAPSHOTS=2 \
    CGOLS_RUN_TAG=_probe128 CGOLS_CALLBACK_PARTITIONED=$part \
        timeout -s TERM 600 srun --gpu-bind=none --kill-on-bad-exit=1 python cgols.py
    echo "=== probe cgols partitioned=$part exit: $? ===" | tee /dev/stderr
    ls -l cgols_snapshots_probe128 data/cgols_checkpoints_probe128 2>/dev/null
done
