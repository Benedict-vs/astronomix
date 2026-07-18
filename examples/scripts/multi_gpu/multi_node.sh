#!/bin/bash
# Slurm runner for the sharded multi-node astronomix example (multi_node.py).
#
#   sbatch examples/scripts/multi_gpu/multi_node.sh
#
# Model: ONE process per GPU. Set --ntasks-per-node to the GPUs-per-node of your
# partition (4 on typical H100 nodes, 8 on an 8-GPU node) and --nodes to scale.
# JAX's distributed init auto-detects the coordinator / ranks from Slurm.

#SBATCH --job-name=astronomix_multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4          # one task (process) per GPU
#SBATCH --gres=gpu:4                 # GPUs per node — match --ntasks-per-node
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
## Cluster-specific — uncomment/adjust for your site:
## SBATCH --account=<your-account>
## SBATCH --partition=<your-gpu-partition>

# No `set -e`: module / conda shell functions misbehave under it.
set -u

# Resolve the repo root from this script's location (works under Slurm, where
# $0 is a spool copy — use BASH_SOURCE, not $0).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"    # examples/scripts/multi_gpu -> repo root
cd "$REPO"

# --- environment (ADAPT to your cluster) ---------------------------------
# module load devel/cuda/12.4
# Activate your environment with micromamba (or conda). Prefer an editable
# install (`pip install -e .`) so edits are visible when running from the repo.
# micromamba activate astronomix
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# --- JAX runtime knobs ---------------------------------------------------
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # grow GPU memory on demand
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92   # leave headroom on shared queues
# JAX >= 0.10 defaults to the Shardy partitioner, which rejects the integer
# mesh-axis names / with_sharding_constraint this code uses — use GSPMD.
export JAX_USE_SHARDY_PARTITIONER=false

echo "=== job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-localhost}:"
echo "    ${SLURM_NNODES:-1} node(s) x ${SLURM_NTASKS_PER_NODE:-?} GPU(s) = ${SLURM_NTASKS:-?} processes ==="
nvidia-smi -L || true

# CRITICAL: --gpu-bind=none makes all of a node's GPUs visible to every task, so
# intra-node NCCL peer-to-peer works and multi_node.py can select its device via
# SLURM_LOCALID. Using --gpus-per-task=1 instead cgroup-binds each task to a
# single GPU that appears as ordinal 0, which breaks NCCL P2P and DEADLOCKS the
# topology exchange ("invalid device ordinal"). No --ntasks here: srun inherits
# the full allocation (matching it avoids a rendezvous hang).
srun --gpu-bind=none python examples/scripts/multi_gpu/multi_node.py
rc=$?

# Propagate srun's exit code — never mask a failed run with an unconditional
# "DONE" (that silently reports success and hides the real error in the .err).
echo "=== srun exit code: $rc ==="
exit $rc
