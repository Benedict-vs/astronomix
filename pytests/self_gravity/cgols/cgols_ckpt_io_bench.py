"""
Time an Orbax save + restore of a state-sized sharded array, the way the cgols
TO_DISK driver writes its per-frame checkpoints.

Every process (one per GPU, launched by srun like cgols.py) holds one shard of
shape (5, LX, LY, LZ), default the per-device state shard of the 2048 run at
split (1, 8, 4, 1): 5 x 256 x 512 x 4096 float32 = 10.7 GB. On the
(1, gx, gy, 1) mesh of CKPT_BENCH_SPLIT the global array is
(5, gx LX, gy LY, LZ), so every node writes as much as it will in the 2048
run (4 x 10.7 GB = 43 GB per node per checkpoint), only on fewer nodes. The
data is random (a constant field would compress away). Saved CKPT_BENCH_SAVES
times through astronomix's own checkpoint writer into
data/ckpt_io_bench_<jobid>/ (data/ = the workspace), restored once onto the
same sharding, then removed again (CKPT_BENCH_CLEANUP=0 keeps it).

    srun --gpu-bind=none python cgols_ckpt_io_bench.py

Env: CKPT_BENCH_SPLIT (default "(1, 4, 2, 1)"), CKPT_BENCH_SHARD (default
"(5, 256, 512, 4096)"), CKPT_BENCH_SAVES (default 2), CKPT_BENCH_CLEANUP (1).
"""

# general
import ast
import os
import shutil
import sys
import time

import jax

# jax.distributed.initialize() must run before anything creates the backend,
# as in cgols.py (and with the same device selection).
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    _visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    _local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    jax.distributed.initialize(local_device_ids=[_local_rank] if len(_visible) > 1 else [0])

# ruff: noqa: E402
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

from astronomix._snapshotting._orbax_storage import (
    load_loop_checkpoint,
    loop_checkpointer,
    save_loop_checkpoint,
)
from astronomix.parallel.rank import barrier

RANK0 = jax.process_index() == 0
SPLIT = ast.literal_eval(os.environ.get("CKPT_BENCH_SPLIT", "(1, 4, 2, 1)"))
SHARD = ast.literal_eval(os.environ.get("CKPT_BENCH_SHARD", "(5, 256, 512, 4096)"))
SAVES = int(os.environ.get("CKPT_BENCH_SAVES", "2"))
CLEANUP = os.environ.get("CKPT_BENCH_CLEANUP", "1") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", f"ckpt_io_bench_{os.environ.get('SLURM_JOB_ID', 'local')}")


def say(*args):
    """Print on rank 0 only."""
    if RANK0:
        print(*args, flush=True)


if SPLIT[0] * SPLIT[1] * SPLIT[2] * SPLIT[3] != jax.device_count():
    sys.exit(f"CKPT_BENCH_SPLIT={SPLIT} needs {SPLIT[0] * SPLIT[1] * SPLIT[2] * SPLIT[3]} devices, "
             f"have {jax.device_count()}")

mesh = jax.make_mesh(SPLIT, ("v", "x", "y", "z"), axis_types=(AxisType.Auto,) * 4)
sharding = NamedSharding(mesh, P("v", "x", "y", "z"))
global_shape = tuple(extent * shards for extent, shards in zip(SHARD, SPLIT, strict=True))
nbytes = 4 * global_shape[0] * global_shape[1] * global_shape[2] * global_shape[3]
nodes = int(os.environ.get("SLURM_NNODES", "1"))
say(f"[ckpt-io] {jax.process_count()} processes, {jax.device_count()} devices, {nodes} node(s); "
    f"global {global_shape} float32 = {nbytes / 1e9:.1f} GB, {nbytes / 1e9 / nodes:.1f} GB per node, "
    f"shard {SHARD}; -> {OUT}")

state = jax.jit(
    lambda key: jax.random.uniform(key, global_shape, jnp.float32),
    out_shardings=sharding,
)(jax.random.key(0))
state.block_until_ready()
# The loop carry's scalars, made inside a jit as the driver's are (a host
# device_put onto a multi-process sharding is not allowed).
_time, _key, _iterations = jax.jit(
    lambda: (jnp.asarray(1.0, jnp.float32), jax.random.key(1), jnp.asarray(0, jnp.int32)),
    out_shardings=NamedSharding(mesh, P()),
)()
scalars = dict(time=_time, key=_key, num_iterations=_iterations)

if RANK0:
    os.makedirs(OUT, exist_ok=True)
barrier("ckpt-io:dir")

with loop_checkpointer(OUT) as writer:
    for step in range(1, SAVES + 1):
        barrier(f"ckpt-io:pre-save-{step}")
        t0 = time.perf_counter()
        save_loop_checkpoint(writer, step, primitive_state=state, forcing=None, **scalars)
        barrier(f"ckpt-io:post-save-{step}")
        dt = time.perf_counter() - t0
        say(f"[ckpt-io] save {step}: {dt:.1f} s = {nbytes / 1e9 / dt:.2f} GB/s aggregate, "
            f"{nbytes / 1e9 / nodes / dt:.2f} GB/s per node")

del state
barrier("ckpt-io:pre-restore")
t0 = time.perf_counter()
restored = load_loop_checkpoint(OUT, SAVES, sharding=sharding)
restored.primitive_state.block_until_ready()
barrier("ckpt-io:post-restore")
dt = time.perf_counter() - t0
say(f"[ckpt-io] restore: {dt:.1f} s = {nbytes / 1e9 / dt:.2f} GB/s aggregate "
    f"(shard {restored.primitive_state.addressable_shards[0].data.shape})")

del restored
barrier("ckpt-io:pre-cleanup")
if CLEANUP and RANK0:
    shutil.rmtree(OUT, ignore_errors=True)
    say(f"[ckpt-io] removed {OUT}")
barrier("ckpt-io:done")
