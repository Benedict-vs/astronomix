#!/bin/bash
# Example gpuq job: a single-GPU CGOLS run. Copy into $GPUQ_DIR/pending/<NNN>_<name>.sh.
# The optional lock lets several queues share one job list without running it twice.
C=/export/home/bschuber/astronomix/pytests/self_gravity/cgols
LOCKS=${GPUQ_LOCKS:-$HOME/.gpuq/locks}; mkdir -p "$LOCKS"
mkdir "$LOCKS/C512_example" 2>/dev/null || { echo "already claimed"; exit 0; }
cd "$C"
PYTHONPATH=$C/gpuq/autocvd_shim \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 \
CGOLS_DIM=512 CGOLS_RUN_TAG=_C512_example CGOLS_IC_MODE=insitu CGOLS_SHARD_SPLIT="(1, 1, 1, 1)" \
CGOLS_CHECKPOINT_KEEP=1 CGOLS_CHECKPOINT_ARCHIVE_EVERY=4 CGOLS_FEEDBACK=clusters \
  python cgols.py > cgols_logs/run_C512_example_console.log 2>&1
