"""Staged multi-process probe: host callbacks and the TO_DISK driver under srun.

Job 67285 (cgols D2, 4 processes x 1 H100) hung silently after setup, right
where the TO_DISK driver emits its first frame through a jitted
``jax.debug.callback``. Multi-process CPU (gloo) runs of the same path pass, so
this isolates the pieces ON GPU, one srun step per stage::

    srun --gpu-bind=none --kill-on-bad-exit=1 \
        python pytests/_dist_callback_probe.py <stage>

Stages:
  cb_noarg  jitted debug.callback with no operands (no collective in the program)
  cb_rep    all-gather (with_sharding_constraint P()) + callback on the result,
            exactly the shape of cgols' frame callable (fires on rank 0 only)
  cb_part   the same with ``partitioned=True`` (fires on every rank locally)
  todisk    the library TO_DISK driver end to end on a small NATIVE_JAX blast:
            frames, Orbax saves, a restart leg, the final process_allgather

Every rank arms faulthandler: after PROBE_TIMEOUT seconds it dumps all Python
thread stacks to stderr and exits 1, so a hang reports WHERE it hangs instead
of burning the allocation. Rank 0 prints ``PASS <stage>`` on success.
"""

# general
import faulthandler
import os
import shutil
import sys

# third-party (raw jax only until jax.distributed.initialize() has run)
import jax

STAGE = sys.argv[1]
TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "300"))


def _local_device_ids():
    """This rank's local GPU (see pytests/_dist_sanity.py)."""
    visible = [
        x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x
    ]
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    return [local_rank] if len(visible) > 1 else [0]


jax.distributed.initialize(local_device_ids=_local_device_ids())
jax.config.update("jax_use_shardy_partitioner", False)  # as in cgols
faulthandler.dump_traceback_later(TIMEOUT, exit=True)

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P  # noqa: E402

RANK = jax.process_index()


def log(msg):
    print(f"[rank {RANK}] {msg}", flush=True)


def _callback_stage(with_operand, partitioned):
    n = jax.device_count()
    mesh = jax.make_mesh((n,), ("x",), axis_types=(AxisType.Auto,))
    sharding = NamedSharding(mesh, P("x"))
    data = np.arange(n * 8, dtype=np.float32)
    x = jax.make_array_from_callback(data.shape, sharding, lambda idx: data[idx])

    def _hello():
        log("callback fired (no operands)")

    def _record(t, y):
        log(f"callback fired: t={float(t)} sum={float(np.sum(y))} (expect {2 * data.sum()})")

    # The shape of the TO_DISK driver's _frame_jit: no outputs, only effects.
    @jax.jit
    def frame(t, x):
        if with_operand:
            y = jax.lax.with_sharding_constraint(x * 2, NamedSharding(mesh, P()))
            jax.debug.callback(_record, t, y, partitioned=partitioned)
        else:
            jax.debug.callback(_hello, partitioned=partitioned)

    for i in range(3):
        with mesh:
            frame(float(i), x)
        log(f"frame {i} dispatched")
        jax.effects_barrier()
        log(f"frame {i} effects_barrier done")
    multihost_utils.sync_global_devices(f"probe:{STAGE}")


def _todisk_stage():
    from astronomix import (
        FORWARDS,
        PERIODIC_BOUNDARY,
        TO_DISK,
        BoundarySettings,
        BoundarySettings1D,
        SimulationConfig,
        SimulationParams,
        construct_primitive_state,
        finalize_config,
        get_registered_variables,
        restart_from_latest_checkpoint,
        time_integration,
    )
    from astronomix.option_classes.simulation_config import (
        NATIVE_JAX,
        VARAXIS,
        XAXIS,
        YAXIS,
        ZAXIS,
        BackendConfig,
    )
    from astronomix.parallel import barrier

    out = os.environ.get("PROBE_OUT", os.path.abspath("data/_probe_todisk"))
    ck, fr = os.path.join(out, "ckpt"), os.path.join(out, "frames")
    if RANK == 0:
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(fr)
        os.makedirs(ck)
    barrier("probe:todisk:setup")

    N = 64
    pb = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        progress_bar=False,
        dimensionality=3,
        box_size=1.0,
        num_cells=N,
        differentiation_mode=FORWARDS,
        boundary_settings=BoundarySettings(pb, pb, pb),
        backend_config=BackendConfig(backend=NATIVE_JAX),
        snapshot_storage_mode=TO_DISK,
        snapshot_storage_path=ck,
        num_snapshots=3,
        activate_snapshot_callback=True,
    )
    rv = get_registered_variables(config)
    split = (1, 2, jax.device_count() // 2, 1) if jax.device_count() > 1 else (1, 1, 1, 1)
    mesh = jax.make_mesh(
        split, (VARAXIS, XAXIS, YAXIS, ZAXIS), axis_types=(AxisType.Auto,) * 4
    )
    sharding = NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))
    x = (jnp.arange(N) + 0.5) / N
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    st = construct_primitive_state(
        config=config,
        registered_variables=rv,
        density=1 + 0 * X,
        velocity_x=0.1 * jnp.sin(2 * jnp.pi * Y),
        velocity_y=0 * X,
        velocity_z=0 * X,
        gas_pressure=0.1 + 10 * (r2 < 0.01),
    )
    config = finalize_config(config, st.shape)
    st = jax.device_put(st, sharding)
    g = config.num_ghost_cells  # 0 with PERIODIC_ROLL: slice g:g+N, never g:-g
    count = {"i": 0}

    def save(t, rho_xz, mdot):
        if RANK != 0:
            return
        np.savez(
            os.path.join(fr, f"frame_{count['i']:02d}.npz"),
            t=np.float32(t),
            rho_xz=rho_xz,
            mdot=mdot,
        )
        log(f"frame {count['i']} written t={float(t):.4f}")
        count["i"] += 1

    def prune():
        if RANK == 0:
            log(f"prune sees steps {sorted(os.listdir(ck))}")

    def rep(a):
        return jax.lax.with_sharding_constraint(a, NamedSharding(mesh, P()))

    def snap(t, s, rv):
        rho = s[rv.density_index]
        vz = s[rv.velocity_index.z]
        jax.debug.callback(prune)
        jax.debug.callback(
            save,
            t,
            rep(rho[g : g + N, g + N // 2, g : g + N]),
            rep(jnp.sum(rho[g : g + N, g : g + N, g : g + N] * vz[g : g + N, g : g + N, g : g + N], axis=(0, 1))),
        )

    log("leg 1")
    time_integration(st, config, SimulationParams(t_end=0.03, C_cfl=0.4), rv, snap, sharding=sharding)
    barrier("probe:todisk:leg1")
    log("restart leg")
    rs, params, restart = restart_from_latest_checkpoint(
        ck, SimulationParams(t_end=0.06, C_cfl=0.4), sharding=sharding
    )
    final = time_integration(
        rs, config, params, rv, snap, sharding=sharding, restart_state=restart
    )
    final = multihost_utils.process_allgather(final, tiled=True)
    if RANK == 0:
        log(f"final gathered {final.shape}, finite={bool(np.isfinite(final).all())}, "
            f"frames on disk={len(os.listdir(fr))} (expect 7)")
    barrier("probe:todisk:done")


log(f"stage {STAGE}: {jax.process_count()} processes, local={jax.local_devices()}")
if STAGE == "cb_noarg":
    _callback_stage(with_operand=False, partitioned=False)
elif STAGE == "cb_rep":
    _callback_stage(with_operand=True, partitioned=False)
elif STAGE == "cb_part":
    _callback_stage(with_operand=True, partitioned=True)
elif STAGE == "todisk":
    _todisk_stage()
else:
    raise SystemExit(f"unknown stage {STAGE!r}")
faulthandler.cancel_dump_traceback_later()
if RANK == 0:
    print(f"PASS {STAGE}", flush=True)
