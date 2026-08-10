"""
CGOLS replication.

Sets up the initial conditions for a Cholla-Galactic-OutfLow-Simulations-style
disk + hot-halo galaxy (Schneider & Robertson 2018, arXiv:1803.01008) and runs
the full 75 Myr adiabatic nuclear-outflow simulation with astronomix. A bare
``python cgols.py`` reproduces the validated production configuration; the
CGOLS_* environment knobs below only exist for experiments and scaling.

By default this reproduces the *adiabatic* A-series setup of that paper, which by
design uses NO radiative cooling. ``CGOLS_COOLING=1`` switches on the companion
paper's radiative B-series (arXiv:1803.01005): the same ICs and wind injection
plus the operator-split, sub-cycled CIE cooling described in the cooling section
at the end of this file (outputs are tagged ``_B`` so the two series never
collide).

Fidelity vs Schneider & Robertson 2018
--------------------------------------
Everything the paper specifies about the *problem* is matched: the static
Miyamoto-Nagai + NFW potential (verified numerically against paper Fig. 4),
the isothermal exponential gas disk + adiabatic hot-halo ICs, adiabatic EOS
with gamma = 5/3 and mu = 0.6 throughout (including the plotted n), the CC85
feedback rates and 300 pc injection radius, the schedule with its hard step
onset at 5 Myr, diode boundaries on all six faces, the 10x10x20 kpc box and
the 75 Myr duration.

Four deliberate deviations remain:
  1. Injection momentum: the paper adds mass momentum-free (new mass at rest);
     astronomix (astronomix/_modules/_cgols_wind/) rescales momentum by
     sqrt(rho_new/rho_old) so exactly Edot enters as thermal energy. Only
     matters inside the 300 pc sphere; the emergent CC85 wind is the same.
  2. Outer disk truncation beyond 4.5 kpc: a Gaussian ramp (see the ``cutoff``
     expression in build_initial_conditions) vs the paper's "exponential"
     truncation - affects only the sparse disk edge.
  3. IC smoothing (CGOLS_SMOOTHING_SIGMA, default 1.5 cells + vertical-HSE
     pressure rebuild; the paper's ICs are sharp): central disk n_c ~ 115 vs
     ~200 cm^-3, T_c ~ 3e4 vs 1e4 K, disk-halo transition ~ 147 vs ~80 pc.
     Forced by resolution - the disk scale height is 0.96 cells at dz = 19.5 pc.
  4. Positivity backstops (rho >= 1e-4, P >= 1e-5 code, |v| <= 5000 km/s,
     T <= 5e9 K; see the params blocks) that DO engage in extreme cells (~5% of
     production steps peg the T ceiling). The paper does not document Cholla's
     equivalents.

Structurally different numerics (equivalent physics, never bit-identical):
WENO5 finite-difference + RK4_LSRK at CFL 0.9 in float32 on the PALLAS backend
vs Cholla's VL + PLMC + HLLC finite-volume (2nd order) at CFL 0.3 in double.
CGOLS_CFL=0.3 gives the closer CFL match at ~3x cost. At this resolution the
right comparison target is the paper's own A-512 run (Fig. 11), not the A-2048
flagship (Fig. 6): expect smoother late-time panels (less resolved contact
turbulence/mixing) and a slightly warmer, puffier central disk.
"""

# TODO: move all cgols related code and artifacts into tests/self_gravity_tests/cgols/
# with seperate folder for benchmarks etc. everything neatly organised.

# ==== JAX runtime knobs (must precede EVERY jax import) ====
# At 512x512x1024 the compiled solver peaks at ~28.5 GB/device, which fits on a
# 40 GB A100 but exceeds JAX's default 0.75 preallocation (~30 GB) once BFC
# fragmentation is accounted for. Raise the fraction so a ~20 GB intermediate
# buffer can be placed. This fraction is applied PER DEVICE, so in a multi-GPU
# (SHARD_SPLIT) run every visible GPU independently preallocates 0.95 of its own
# memory - it is not a shared budget. setdefault lets a command-line override win.
import ast
import math
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
# ruff: noqa: E402

# ==== GPU selection / multi-GPU domain decomposition ====
# Two launch models, auto-detected:
#
#   ONE PROCESS, N DEVICES (the default; `python cgols.py`). autocvd picks
#     NUM_GPUS free GPUs on the node and jax drives all of them from one
#     process. This is what the validated 512 / 1024 production runs used.
#
#   ONE PROCESS PER GPU (multi-node; `srun --gpu-bind=none python cgols.py`).
#     Required from 2048 on, which no single node holds. Detected from
#     SLURM_NTASKS > 1, or forced with CGOLS_DISTRIBUTED=1. autocvd is skipped
#     (Slurm already assigned the GPUs) and jax.distributed.initialize() runs
#     BEFORE astronomix is imported - see the _init_distributed note below.
#
# SHARD_SPLIT is the number of shards along each of the four state axes
# (variables, x, y, z); its product is the number of GPUs used.
#   (1, 1, 1, 1) is the original single-GPU path (no sharding)
#   (1, 2, 1, 1) splits the x-axis across 2 GPUs, roughly halving the ~28 GB/device peak
#   (1, 1, 1, 2) splits z instead
# Only split spatial axes (leave VARAXIS = 1), and KEEP Z UNSHARDED: the IC's
# vertical HSE integral is a cumsum along z, and a z-split lowers it to a
# cross-device scan. Two constraints, both asserted in _validate_shard_split():
# the split axis's cell count must divide evenly by its shard count, AND the
# per-device slice must stay divisible by pallas_block_shape (4, 4, 8). Since
# 2048/gx must be divisible by 4 and 4096/gz by 8, every shard count has to
# divide 512 - i.e. it must be a power of two. That rules out 48 GPUs (the
# accelerated-h200 12-node cap) and any 12/24/192-GPU configuration.
#
# Sizing for CGOLS_DIM=2048 (2048 x 2048 x 4096), A-series, no cooling; the
# 1024/4xH200 run measured 140.2 GB/device and ~3.13 s/step:
#
#   GPUs  nodes    split          per-device slice   per-device cells   est. GB/device
#     32  8 x 4    (1, 8, 4, 1)   256 x 512 x 4096   1.013x (padded)      ~127-129
#     64  16 x 4   (1, 8, 8, 1)   256 x 256 x 4096   0.51x                 ~64-65
#    128  32 x 4   (1, 16, 8, 1)  128 x 256 x 4096   0.26x                 ~32-34
#
# Production target: accelerated-h200, 8 nodes / 32 GPUs, (1, 8, 4, 1) - the
# per-device memory and per-device work then nearly match the already-validated
# 1024/4-GPU run. Prefer near-square gx ~ gy: the halo overhead
# (nx+8)(ny+8)(nz+8)/(nx ny nz) is 4.9% at 256x512 but 19.8% at 64x128, so ~512
# GPUs is the practical ceiling.
#
# All of the knobs below default to their literal values, so a plain
# `python cgols.py` is unchanged; the scaling driver (cgols_scaling.py) sets the
# CGOLS_* environment variables to sweep resolution / sharding / a short
# fixed-step benchmark without editing this file.

#: Per-device Pallas block shape; the per-device slice must stay a multiple
#: of it on every axis (see _validate_shard_split and build_config).
PALLAS_BLOCK_SHAPE = (4, 4, 8)

# One process per GPU? Auto-detected, so a bare `python cgols.py` is unchanged
# and any srun-launched run does the right thing without an extra knob.
DISTRIBUTED = (
    os.environ.get("CGOLS_DISTRIBUTED", "") == "1"
    or int(os.environ.get("SLURM_NTASKS", "1")) > 1
)

_SHARD_SPLIT_ENV = os.environ.get("CGOLS_SHARD_SPLIT", "")


def _auto_shard_split(num_devices):
    """A near-square (1, gx, gy, 1) split of ``num_devices``, z unsharded.

    Used when a distributed run does not set CGOLS_SHARD_SPLIT, so the runner
    only has to choose nodes x tasks-per-node and the script is portable across
    machines. ``num_devices`` must be a power of two (see the note above);
    the halves are handed to x first, so 32 -> (1, 8, 4, 1).
    """
    gx = gy = 1
    for _ in range(num_devices.bit_length()):
        if gx * gy == num_devices:
            break
        if gx <= gy:
            gx *= 2
        else:
            gy *= 2
    return (1, gx, gy, 1)


if DISTRIBUTED:
    # jax.distributed.initialize() must run before ANY call that creates the
    # JAX backend - and `import astronomix` creates it (some option NamedTuples
    # have jnp.array defaults). So bootstrap with RAW jax here, inline, rather
    # than via astronomix.parallel.init_distributed: importing that goes
    # through astronomix/__init__.py, after which initialize() fails with
    # "must be called before any JAX calls". (The same warning is in
    # astronomix/parallel/distributed.py's own NOTE.)
    import jax

    def _init_distributed():
        """Bootstrap multi-process mode; a no-op when launched single-process.

        Robust to both Slurm GPU-binding modes: with --gpu-bind=none all of a
        node's GPUs are visible and this rank selects its device by node-local
        rank (SLURM_LOCALID); with a single cgroup-bound GPU it falls back to
        the only visible device (ordinal 0). Launch with --gpu-bind=none:
        --gpus-per-task=1 breaks intra-node NCCL P2P and deadlocks the topology
        exchange with "invalid device ordinal".
        """
        multiprocess = (
            int(os.environ.get("SLURM_NTASKS", "1")) > 1
            or int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1
        )
        if not multiprocess:
            return
        visible = [
            d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d
        ]
        local_rank = int(
            os.environ.get("SLURM_LOCALID")
            or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
            or "0"
        )
        local_device_ids = [local_rank] if len(visible) > 1 else [0]
        jax.distributed.initialize(local_device_ids=local_device_ids)

    _init_distributed()

    # The allocation, not the split, decides how many devices there are; an
    # explicit CGOLS_SHARD_SPLIT is validated against it below.
    SHARD_SPLIT = (
        ast.literal_eval(_SHARD_SPLIT_ENV)
        if _SHARD_SPLIT_ENV
        else _auto_shard_split(jax.device_count())
    )
else:
    SHARD_SPLIT = ast.literal_eval(_SHARD_SPLIT_ENV or "(1, 1, 1, 1)")

    # By default autocvd waits (indefinitely) for completely free GPUs. On a busy
    # shared node that can block forever; CGOLS_GPU_LEAST_USED=1 instead takes the
    # least-used GPU(s) right away. Meant for small runs (e.g. the 256^2x512 A/B
    # experiments, ~4 GB) that comfortably fit next to other jobs - don't use it to
    # squeeze the 512 production run (~28.5 GB) onto a partially occupied card.
    from autocvd import autocvd

    autocvd(
        num_gpus=SHARD_SPLIT[0] * SHARD_SPLIT[1] * SHARD_SPLIT[2] * SHARD_SPLIT[3],
        least_used=os.environ.get("CGOLS_GPU_LEAST_USED", "0") == "1",
    )
    import jax

NUM_GPUS = SHARD_SPLIT[0] * SHARD_SPLIT[1] * SHARD_SPLIT[2] * SHARD_SPLIT[3]

# astronomix names its mesh axes with integers (VARAXIS=0, XAXIS=1, ...). jax
# 0.10.1 defaults to the Shardy partitioner, whose sdy.MeshAxisAttr.get(name,
# size) requires name to be a *str* and raises a TypeError on the integer axis
# names. The older GSPMD partitioner accepts them, so disable Shardy for the
# sharded run. (Harmless on the single-GPU path, which never builds a mesh.)
# Set AFTER the distributed init - touching jax.config beforehand can create the
# backend early. The runner also exports JAX_USE_SHARDY_PARTITIONER=false so this
# holds regardless of import ordering.
if NUM_GPUS > 1:
    jax.config.update("jax_use_shardy_partitioner", False)

if NUM_GPUS != jax.device_count():
    raise SystemExit(
        f"CGOLS_SHARD_SPLIT={SHARD_SPLIT} asks for NUM_GPUS={NUM_GPUS} devices but "
        f"jax.device_count()={jax.device_count()} (global, across all "
        f"{jax.process_count()} process(es)). Under srun the split's product must "
        f"equal nodes x ntasks-per-node."
    )

#: True on the one process allowed to do host-side I/O (prints, figures, file
#: cleanup). Always True in a single-process run, so nothing changes there.
#: Never guard a COLLECTIVE with this (an Orbax save/restore, a barrier) - that
#: deadlocks every other rank; guard only this script's own side effects.
RANK0 = jax.process_index() == 0


def rprint(*args, **kwargs):
    """``print`` on rank 0 only.

    Every rank runs this whole script, so an unguarded print is emitted
    ``process_count`` times into one interleaved job log. Everything printed
    here is either a global reduction or a config echo, i.e. identical on all
    ranks, so rank 0's copy is the complete record. Single-process runs are
    unaffected.
    """
    if RANK0:
        print(*args, **kwargs)


import glob
import shutil

import jax.numpy as jnp
from jax.sharding import AxisType, PartitionSpec as P
import numpy as np

import astropy.constants as const
from astropy import units as u

import matplotlib

# Headless by construction: batch ranks have no display, and importing pyplot
# with an interactive backend under srun can hang on the X/Qt probe.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astronomix import (
    CARTESIAN,
    CodeUnits,
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    get_registered_variables,
    restart_from_latest_checkpoint,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    OPEN_BOUNDARY,
    FINITE_DIFFERENCE,
    OPEN_BOUNDARY_DIODE,
    PALLAS,
    RK4_LSRK,
    SIMPLE_SOURCE,
    TO_DISK,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    PositivityConfig,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
)
from astronomix._modules._cgols_wind.cgols_wind_options import (
    CGOLSWindConfig,
    CGOLSWindParams,
)
from astronomix._modules._cooling._cooling_tables import cie_parabolic_cooling
from astronomix._modules._cooling.cooling_options import (
    CIE_PARABOLIC,
    COOLING_FLOOR_CLIP,
    COOLING_OPERATOR_SPLIT,
    SUBCYCLED_EXPLICIT_COOLING,
    TOTAL_NUMBER_DENSITY,
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
)
from astronomix.parallel import barrier

jax.config.update("jax_enable_x64", False)

if jax.process_count() > 1:
    rprint(
        f"[distributed] processes={jax.process_count()} "
        f"global_devices={jax.device_count()} "
        f"local_devices/proc={jax.local_device_count()} "
        f"split={SHARD_SPLIT}"
        + ("" if _SHARD_SPLIT_ENV else " (auto)"),
        flush=True,
    )


def _validate_shard_split(split, num_cells):
    """Check SHARD_SPLIT against the grid and the Pallas block shape.

    ``num_cells`` is the (nx, ny, nz) global cell count. Two hard constraints
    per spatial axis: the axis must divide evenly by its shard count, and the
    resulting per-device slice must stay a multiple of ``PALLAS_BLOCK_SHAPE`` -
    the Pallas kernels are compiled for whole blocks, and a ragged remainder
    silently produces wrong halo values rather than an error. Together these
    force every shard count to be a power of two at the production grid sizes,
    which is why 48 GPUs (the accelerated-h200 12-node cap) cannot be used.

    Raises SystemExit with the offending axis named; called before any device
    memory is committed so a bad launch fails in seconds, not after an hour of
    queueing plus a compile.
    """
    if split[0] != 1:
        raise SystemExit(
            f"CGOLS_SHARD_SPLIT={split}: the variable axis must not be split "
            "(the solver's per-variable kernels assume all variables are local)."
        )
    if split[3] != 1:
        raise SystemExit(
            f"CGOLS_SHARD_SPLIT={split}: keep z unsharded - the IC's vertical "
            "HSE pressure integral is a cumsum along z, which a z-split lowers "
            "to a cross-device scan."
        )
    for axis, name in enumerate("xyz"):
        shards = split[axis + 1]
        cells = num_cells[axis]
        block = PALLAS_BLOCK_SHAPE[axis]
        if shards < 1 or cells % shards:
            raise SystemExit(
                f"CGOLS_SHARD_SPLIT={split}: {name} has {cells} cells, which is "
                f"not divisible by its {shards} shards."
            )
        if (cells // shards) % block:
            raise SystemExit(
                f"CGOLS_SHARD_SPLIT={split}: the per-device {name} slice is "
                f"{cells // shards} cells, not a multiple of the Pallas block "
                f"size {block} (pallas_block_shape={PALLAS_BLOCK_SHAPE}). "
                f"{name} shard counts must divide {cells // block}."
            )


# ---------------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------------
# Anchor every input/output file to the directory THIS script lives in,
# regardless of the working directory the run was launched from (relative paths
# would otherwise scatter outputs into the cwd). Outputs are sorted by kind:
#   data/initial/, data/final/  - multi-GB .npy states (data/ is a scratch symlink)
#   data/cgols_checkpoints*/    - rolling full-state checkpoints (also scratch)
#   figures/cgols/              - all rendered figures / animations
#   cgols_logs/                 - per-step diag logs (and driver console logs)
#   cgols_snapshots<RUN_TAG>/   - streamed per-frame .npz (small, stays here)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _here(name):
    """Resolve ``name`` against the script directory; pass-through if absolute."""
    return os.path.join(SCRIPT_DIR, name)


_CREATED_DIRS = set()


def _in_dir(subdir, name):
    """Resolve ``name`` against ``SCRIPT_DIR/subdir`` (created on demand);
    pass-through if ``name`` is absolute.

    The ``makedirs`` runs on rank 0 only, and at most once per directory: every
    rank calls these path helpers, so an unguarded version fires 32 simultaneous
    ``mkdir``s per directory at every call site on a Lustre filesystem. Rank 0
    is also the only rank that writes into these directories (see RANK0 /
    rprint), so there is nothing for the others to wait for.
    """
    out_dir = _here(subdir)
    if RANK0 and out_dir not in _CREATED_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        _CREATED_DIRS.add(out_dir)
    return os.path.join(out_dir, name)


def _data(name):
    """Resolve ``name`` against the bulk-data directory (multi-GB files).

    ``data/`` next to this script is a symlink to scratch storage
    (/export/scratch/...), keeping the 5 GB state files off the 100 GB home
    quota - a full 512 replay campaign overran it on 2026-07-03. Figures,
    frames and logs are small and stay in the script directory.
    """
    return _in_dir("data", name)


def _data_initial(name):
    """Bulk-data path for initial conditions (data/initial/ on scratch)."""
    return _in_dir(os.path.join("data", "initial"), name)


def _data_final(name):
    """Bulk-data path for final states (data/final/ on scratch)."""
    return _in_dir(os.path.join("data", "final"), name)


def _fig(name):
    """Resolve ``name`` against the cgols figures directory (figures/cgols/)."""
    return _in_dir(os.path.join("figures", "cgols"), name)


def _log(name):
    """Resolve ``name`` against the cgols log directory (cgols_logs/)."""
    return _in_dir("cgols_logs", name)


# ---------------------------------------------------------------------------
# Tunable knobs
# ---------------------------------------------------------------------------
# Optional HSE-preserving smoothing of the initial conditions, in units of cells.
# 0.0 disables it (the run then matches the reference setup exactly). It smooths
# the sharp disk-halo contact so the grid can hold it: the density is
# Gaussian-smoothed and the pressure is then REBUILT from vertical hydrostatic
# equilibrium (so dP/dz = -rho dPhi/dz is satisfied and the disk does not "ring").
# This is a mitigation for the under-resolved vertical disk (intrinsic scale
# height H = 18.8 pc = 0.96 cells at 512^2x1024), not a substitute for resolving it.
#
# Sigma trades IC fidelity against contact sharpness (1D central-column study,
# dz = 19.5 pc; paper Fig. 4 targets: n_c ~ 200 cm^-3, T_disk = 1e4 K,
# disk-halo transition ~80 pc):
#   sigma = 0.0: n_c = 196, T_c = 1.0e4, transition  88 pc  (exact but ~1-cell peak)
#   sigma = 1.5: n_c = 115, T_c = 3.0e4, transition 147 pc  (peak FWHM ~4 cells)
#   sigma = 3.0: n_c =  67, T_c = 8.3e4, transition 225 pc  (previous default)
# 1.5 is the default: half the central-disk distortion of 3.0 while keeping the
# contact wide enough for the WENO5 stencil. Changing sigma requires an IC
# rebuild (CGOLS_CREATE_IC=1); watch cgols_vz_diagnostic.png / the static check
# on the next run for disk ringing.
SMOOTHING_SIGMA_CELLS = float(os.environ.get("CGOLS_SMOOTHING_SIGMA", "1.5"))

# Total integration time. The paper runs 75 Myr total; bump this up for a real
# stability test.
TOTAL_TIME = 75 * u.Myr

# Fraction of TOTAL_TIME to actually integrate. 1 is the full production run;
# smaller values give cheap partial runs for stability/A-B experiments. The
# wind schedule uses absolute time, so a partial run is simply a truncation,
# not a rescaling.
END_TIME_FRACTION = float(os.environ.get("CGOLS_END_FRACTION", "1"))
END_TIME = END_TIME_FRACTION * TOTAL_TIME

# Injection radius
INJ_RAD = 300 # pc

# Number of evenly-spaced intermediate snapshots to offload to the host during the
# run (for the animation and the wind time-series). These are cheap 2D slices /
# 1D reductions, not full 3D states (full states would be ~5 GB each and OOM the
# device), so this can be fairly large without a memory cost. The snapshot grid
# spans [t_start, t_end], so a restarted run (CGOLS_RESTART_FROM) distributes the
# same count over just the replayed window - crank it up for dense forensic frames.
NUM_SNAPSHOTS = int(os.environ.get("CGOLS_NUM_SNAPSHOTS", "60"))

CREATE_IC = os.environ.get("CGOLS_CREATE_IC", "0") == "1"  # build the initial conditions and save them to .npy files
# if set to False, the run will load the .npy files and run simulation from them

# Resolution
RESOLUTION = int(os.environ.get("CGOLS_DIM", "512"))  # cells along the x/y axes; z is always 2x this

# Now that the grid is known, check the split. Done here rather than at parse
# time so the error names the actual cell counts.
_validate_shard_split(SHARD_SPLIT, (RESOLUTION, RESOLUTION, 2 * RESOLUTION))

# How the initial conditions are obtained:
#   "file"   - build once with CGOLS_CREATE_IC=1, then jnp.load the .npy pair.
#              The validated path for every run up to 1024.
#   "insitu" - build them in-process, inside a jit with out_shardings, every
#              run. At 2048 the .npy pair would be 344 + 69 GB, jnp.load
#              materialises them on ONE host, there is no sharded loader, and
#              the eager builder would need ~720 GB of host RAM. The build is
#              cheap next to a 48 h leg.
# Defaults to insitu exactly where the file path stops working, so DIM <= 1024,
# run_ic_horeka.sh and every analysis script are untouched.
IC_MODE = os.environ.get(
    "CGOLS_IC_MODE", "insitu" if (RESOLUTION >= 2048 or DISTRIBUTED) else "file"
)
if IC_MODE not in ("file", "insitu"):
    raise SystemExit(f"CGOLS_IC_MODE must be 'file' or 'insitu', got {IC_MODE!r}")
INSITU_IC = IC_MODE == "insitu"

# Suffix for the IC .npy filenames. Defaults to the resolution (_d512, _d1024,
# ...), so runs at different resolutions never overwrite each other's initial
# state: changing CGOLS_DIM automatically selects (or, with CGOLS_CREATE_IC=1,
# builds) the matching IC files. Override only for special IC variants.
IC_TAG = os.environ.get("CGOLS_IC_TAG", f"_d{RESOLUTION}")

# Benchmark mode: when CGOLS_BENCH_STEPS > 0 the run does exactly that many
# fixed-size timesteps (no CFL — physically meaningless, but the per-step compute
# work is identical), prints the per-device memory + elapsed time, and skips all
# snapshot I/O. Used by cgols_scaling.py to measure sec/step and peak memory
# without a multi-hour run. 0 = normal production run.
BENCH_STEPS = int(os.environ.get("CGOLS_BENCH_STEPS", "0"))

# Feedback switch-on shape. The paper turns the wind on as a step into the low
# state at 5 Myr; that is the default and is production-proven with the
# positivity clips. CGOLS_STEP_ONSET=0 softens it to a 1 Myr linear ramp
# (5 -> 6 Myr) for experiments (see build_cgols_wind_params).
STEP_ONSET = os.environ.get("CGOLS_STEP_ONSET", "1") == "1"

# Outer boundary condition: "diode" (outflow-only, the paper's choice and the
# default) or "open" (plain zero-gradient copy, which lets the static potential
# pull ghost-reservoir gas back into the box - slow artificial accretion off
# every face). Switchable per run for A/B experiments.
BOUNDARY = os.environ.get("CGOLS_BOUNDARY", "diode")
if BOUNDARY not in ("diode", "open"):
    raise ValueError(f"CGOLS_BOUNDARY must be 'diode' or 'open', got {BOUNDARY!r}")
_BOUNDARY_TYPE = OPEN_BOUNDARY_DIODE if BOUNDARY == "diode" else OPEN_BOUNDARY

# Radiative cooling. Default off: a bare `python cgols.py` stays the adiabatic
# A-series of Schneider & Robertson 2018 (arXiv:1803.01008). CGOLS_COOLING=1
# switches on the companion paper's radiative B-series setup (arXiv:1803.01005):
# the piecewise-parabolic solar-metallicity CIE curve (Eq. A4 of CGOLS I,
# rendered for reference in cooling_lambda_cgs at the bottom of this file),
# applied as an operator-split, sub-cycled source term after each hydro step
# with a 10^4 K floor and n^2 Lambda(T) at mu = 0.6 - the Cholla convention.
COOLING = os.environ.get("CGOLS_COOLING", "0") == "1"
# Cooling temperature floor in Kelvin. Below it Lambda = 0 anyway (the curve's
# own cutoff); this is the hard clamp on the sub-cycled update.
COOLING_FLOOR_K = float(os.environ.get("CGOLS_COOLING_FLOOR_K", "1e4"))
# Fixed sub-cycle trip count. Each sub-cycle changes T by at most 1%; the loop
# always runs all of them (it is a static-bound fori_loop, for
# differentiability), so this is a direct cost knob. 32 is the
# production-validated value for the pure-CFL scheme (the default, see
# CGOLS_COOLING_DT_LIMIT below); any residual dt is consumed in one clipped
# Euler step that lands stiff cells exactly on the floor, Cholla-like.
COOLING_SUBCYCLES = int(os.environ.get("CGOLS_COOLING_SUBCYCLES", "32"))
# Largest fraction of a cell's thermal energy that may be radiated in one hydro
# step (paper: "no cell loses more than 10% of its thermal energy in a given
# hydrodynamic time step").
COOLING_DT_FRACTION = float(os.environ.get("CGOLS_COOLING_DT_FRACTION", "0.1"))

# Whether that rule constrains the hydro timestep at all. The 10% rule is the
# paper's STATED scheme, but it is absent from both public Cholla branches,
# unaffordable at 512^2x1024 (see below), and - after the FCT resolution below
# - unnecessary: the default is now OFF (pure CFL, sub-cycling absorbs the
# stiffness), which is the production-validated configuration. Set
# CGOLS_COOLING_DT_LIMIT=1 to reproduce the paper's stated scheme and expect
# the run to take weeks at 512.
#
# MEASURED (2026-07-31, 512^2x1024 on 2xH200): with the limit ON the run is
# NOT AFFORDABLE at this resolution. dt pins at 1.15e-6 code (~11 yr) -> 6.7e6
# steps ~ 44 days for 75 Myr, versus ~90k steps for the adiabatic A-series.
# The binding cells are the dense disk core: n_c = 112 cm^-3 at T_c = 3.1e4 K
# gives t_cool = 18 yr. This is a direct cost of deviation 3 in the fidelity
# list above - the paper's SHARP disk sits at exactly 1e4 K, where Lambda = 0
# and the rule is free, while our CGOLS_SMOOTHING_SIGMA disk starts a factor ~3
# above the floor and cools furiously. It is also resolution-coupled: at
# CGOLS_DIM=128 the smeared core is only 29 cm^-3 and the limited run is fine.
# Where the B-series science actually lives - the shocked shell, T ~ 1e6 K,
# n ~ 10 - t_cool ~ 6000 yr against a ~830 yr CFL step, i.e. dt/t_cool ~ 0.14,
# comfortably inside what 16 sub-cycles resolve.
#
# *** WARNING: RUNNING WITH THE LIMIT OFF BLEW UP AT 512^2x1024 (2026-07-31). ***
# The argument above - that switching the limit off costs accuracy only in the
# disk transient - is WRONG, and the run that tested it diverged at t = 0.593
# code (5.8 Myr), 0.8 Myr after the wind steps on. Density ran away at ~400-430
# pc from centre (just outside the 300 pc injection sphere, i.e. the wind/disk
# contact), 1e5 -> 1e12 in ~20 steps, ending with max_rho ~ 1e35, |v| pinned at
# 50*sqrt(3) in all three components and T at its ceiling on 98% of steps.
# NO NaN EVER APPEARED - the positivity clips masked the divergence, so watch
# max_rho, not the NaN flag. Control: the adiabatic A-series peaked at
# max_rho = 3.5e4 over the full 75 Myr.
# Forensics: cgols_logs/archive_B_blowup_20260731/.
#
# *** RESOLVED (2026-07-31): the blow-up was NOT the cooling operator and NOT
# the missing dt limit - it was the un-blended WENO interface flux at the
# cooling-sharpened contact. Radiative cooling collapses the shocked shell to
# a ~1-cell, 1e5:1 contrast contact; there the 5th-order flux reconstruction
# overshoots so hard that a single step can move more mass out of the donor
# cell than it contains. The donor is clamped back up by the density floor
# (mass created from nothing), the receiver's density jumps ~100x in ONE step
# (2.6e3 -> 2.9e5 seen at CGOLS_DIM=128), and the floors + velocity/temperature
# clips then lock the region into a mass-pumping runaway. The chain of fixes
# that did NOT work, for the record: sigma=0 sharp ICs (worse - the 1-cell disk
# dies pre-onset), CFL 0.3 (blew up anyway), a 3.2e4 K cooling floor (delayed
# it to the high state only). The fix that works is CGOLS_PRESERVING_FLUX=1,
# the Hu-Adams-Shu/Zalesak FCT positivity flux limiter - with it the full
# 75 Myr B-series at 128 runs with ZERO temperature-ceiling and ZERO
# velocity-clip engagements (vs 24k/24.5k steps before) and peak max_rho
# ~3.2e4, same scale as the adiabatic A-series. NB the limiter was silently
# a no-op on the fused Pallas RK4_LSRK path until the matching astronomix fix
# (_finite_difference/_time_integrators/_ssprk.py) that forces the explicit
# flux path when blending is requested.
# Raise CGOLS_COOLING_SUBCYCLES when running without the limit.
COOLING_DT_LIMIT = os.environ.get("CGOLS_COOLING_DT_LIMIT", "0") == "1"

# Positivity-preserving flux blending (see _finite_difference/_interface_fluxes/
# _flux_blending.py). PRESERVING_FLUX is the Hu-Adams-Shu / Zalesak-FCT limiter:
# per interface, blend the WENO flux toward first-order LLF by the smallest
# weight that keeps the LF-updated density AND pressure above their floors.
# DEEPVOID_BLEND ramps toward LLF near the density floor.
# PRESERVING_FLUX defaults ON whenever cooling is on - the radiative B-series
# is NOT viable without it (see the blow-up notes above: the brute-force clips
# rectify WENO oscillations at the cooling-sharpened contact into a density
# runaway, while the FCT blend makes the offending interface locally diffusive
# instead; costs the fused Pallas path, ~2.2x per step). It defaults OFF for an
# adiabatic run, so a bare `python cgols.py` keeps the production A-series
# numerics bit-identically.
PRESERVING_FLUX = os.environ.get("CGOLS_PRESERVING_FLUX", "1" if COOLING else "0") == "1"
DEEPVOID_BLEND = os.environ.get("CGOLS_DEEPVOID_BLEND", "0") == "1"

# Optional suffix for per-run outputs (the snapshots directory and the final
# state .npy) so several experiment runs can coexist without clobbering each
# other or the production outputs. Defaults to "_B" for a radiative run, so the
# A- and B-series never share snapshot / checkpoint / log / figure filenames.
RUN_TAG = os.environ.get("CGOLS_RUN_TAG", "_B" if COOLING else "")

# Full-state checkpointing, via astronomix's Orbax TO_DISK mode: the run is
# split into num_snapshots segments and after each one the full loop carry is
# written (each device streams its own shard - no host gather) to
# data/cgols_checkpoints<RUN_TAG>/<step>/ (bulk-data dir on scratch), keeping
# only the newest CGOLS_CHECKPOINT_KEEP step dirs - a bounded, rolling safety
# net that always brackets a potential crash. The frame callback keeps running
# alongside (the driver invokes it at every segment end). CGOLS_CHECKPOINT_EVERY
# is now only the on/off gate (0 = off, plain in-memory run); the write cadence
# is one checkpoint per snapshot frame, dictated by the TO_DISK driver.
# A run is RESUMED with CGOLS_RESTART_FROM=latest (this run-tag's checkpoint
# dir) or =<path to a checkpoint dir>, optionally CGOLS_RESTART_STEP=<n> for a
# specific step (default: newest). params.t_start continues from the stored
# time; the wind schedule uses absolute time, so nothing else shifts. Resuming
# into the same tag continues the checkpoint numbering; a resumed run is
# bit-identical to one left uninterrupted.
CHECKPOINT_EVERY = int(os.environ.get("CGOLS_CHECKPOINT_EVERY", "3"))
CHECKPOINT_KEEP = int(os.environ.get("CGOLS_CHECKPOINT_KEEP", "4"))
RESTART_FROM = os.environ.get("CGOLS_RESTART_FROM", "")
RESTART_STEP = os.environ.get("CGOLS_RESTART_STEP", "")

# Runtime numerics knobs (defaults are the production values).
# CGOLS_CFL: 0.9 is production-proven; the paper's Cholla setup ran 0.3, which
# is the closer numerics match at ~3x the step count.
# CGOLS_TMAX_K: the positivity temperature ceiling in Kelvin (becomes
# positivity_max_pressure_over_density). It must sit WELL above the hottest
# physical phase: at 5e8 K the box's hottest cell exceeded it on 27% of steps,
# and that continuous thermal-energy deletion at sharp contacts fed a density
# runaway at 43.8 Myr under CFL 0.9. 5e9 K (default) leaves real transients
# untouched while still catching the 1e21+ K near-vacuum spikes.
CFL = float(os.environ.get("CGOLS_CFL", "0.9"))
TMAX_K = float(os.environ.get("CGOLS_TMAX_K", "5e9"))


def _ic_state_path():
    """Path to the saved initial primitive state (IC_TAG-suffixed)."""
    return _data_initial(f"cgols_initial_state{IC_TAG}.npy")


def _ic_potential_path():
    """Path to the saved external gravitational potential (IC_TAG-suffixed)."""
    return _data_initial(f"cgols_initial_potential{IC_TAG}.npy")


# ---------------------------------------------------------------------------
# Code units
# ---------------------------------------------------------------------------
# code_time is derived automatically by CodeUnits as code_length / code_velocity.
code_length = 1 * u.kiloparsec
code_mass = 1e6 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G = const.G
k_B = const.k_B
m_p = const.m_p

# --- Stellar disk + disk gas ---
M_disk = 1e10 * u.M_sun
R_disk = 0.8 * u.kiloparsec
z_disk = 0.15 * u.kiloparsec
R_trunc = 4.5 * u.kiloparsec

R_gas = 1.6 * u.kiloparsec  # = 2 * R_disk
M_gas = 2.5e9 * u.M_sun
T_disk = 1e4 * u.Kelvin

# --- Dark-matter halo + halo gas ---
M_halo = 5e10 * u.M_sun
R_vir = 53 * u.kiloparsec
c = 10  # NFW concentration
R_halo = R_vir / c

rho_0h = 3e3 * u.M_sun / u.kiloparsec**3
T_halo = 1e6 * u.Kelvin  # at r = 100 kpc

mu = 0.6
gamma = 5 / 3

# Code-units-to-Kelvin factor for temperature: T = (P / rho) * T_factor, where
# P / rho is in code_velocity^2. Computing it as a single well-scaled constant
# (~7e5 K) avoids single precision: the individual code-unit constants k_B_code
# (~7e-70) and m_p_code (~8e-64) both underflow to 0.0 in float32, which turns
# T = P * mu * m_p_code / (rho * k_B_code) into 0/0 = NaN. Their ratio does not.
T_factor = (mu * m_p / k_B * code_units.code_velocity**2).to(u.K).value


def _kelvin_to_code_temperature(T_K):
    """Convert a temperature in Kelvin to the cooling module's rescaled units.

    The cooling module works in ``T~ = T k_B / m_p`` (code_energy / code_mass),
    NOT in Kelvin and not in this file's ``P / rho`` code temperature - the two
    differ by mu, i.e. ``T[K] = T~ * T_factor / mu``. 10^4 K comes out as
    ~8.3e-3, so passing a literal 1e4 to ``CoolingParams.floor_temperature``
    would put the floor at ~1.2e10 K and silently disable cooling everywhere.
    Done via astropy so the conversion is auditable rather than a magic number.
    """
    return float(
        (T_K * u.K * k_B / m_p).to(code_units.code_energy / code_units.code_mass).value
    )


# Cooling-curve parameters in code units (the published Eq. A4 coefficients plus
# the two log10 unit shifts). Cheap to build, so do it unconditionally.
_CIE_COOLING_PARAMS = cie_parabolic_cooling(code_units)


def _cooling_params_kwargs():
    """The ``cooling_params=`` kwarg for SimulationParams, or {} when adiabatic.

    Kept as a conditional kwarg so the adiabatic run's params are byte-identical
    to what they were before cooling existed.
    """
    if not COOLING:
        return {}
    return dict(
        cooling_params=CoolingParams(
            hydrogen_mass_fraction=0.76,
            metal_mass_fraction=0.02,
            # Pin mu = 0.6 as the paper prescribes ("we take mu = 0.6
            # throughout the calculation"); X, Z alone would give 0.590, a 3.4%
            # error in the cooling rate and an inconsistency with T_factor and
            # every diagnostic in this file.
            mean_molecular_weight=mu,
            floor_temperature=_kelvin_to_code_temperature(COOLING_FLOOR_K),
            max_thermal_energy_fraction=COOLING_DT_FRACTION,
            cooling_curve_params=_CIE_COOLING_PARAMS,
        )
    )


# ---------------------------------------------------------------------------
# CGOLS central starburst wind (Chevalier & Clegg 1985 feedback)
# ---------------------------------------------------------------------------
# Schneider & Robertson 2018, Section 2: mass and thermal energy are injected
# at uniform volumetric rates into a 300 pc sphere at the galaxy center, with
#   Mdot = beta * SFR              and   Edot = alpha * 3e41 erg/s * SFR.
# Low state (SFR = 5 Msun/yr, beta = 0.3, alpha = 1.0):
#   Mdot = 1.5 Msun/yr, Edot = 1.5e42 erg/s.
# High state (SFR = 20 Msun/yr, beta = 0.6, alpha = 0.9):
#   Mdot = 12 Msun/yr,  Edot = 5.4e42 erg/s.
# Schedule: no feedback for the first 5 Myr (equilibration), then the low
# state, a 5 Myr linear ramp up, 30 Myr in the high state, a 5 Myr ramp back
# down, and the low state for the remaining 30 Myr (75 Myr total).
def build_cgols_wind_params():
    mdot_to_code = lambda x: (x * u.M_sun / u.yr).to(
        code_units.code_mass / code_units.code_time
    ).value
    edot_to_code = lambda x: (x * u.erg / u.s).to(
        code_units.code_energy / code_units.code_time
    ).value
    myr_to_code = (1 * u.Myr).to(code_units.code_time).value

    mdot_low, mdot_high = mdot_to_code(1.5), mdot_to_code(12.0)
    edot_low, edot_high = edot_to_code(1.5e42), edot_to_code(5.4e42)

    # Feedback onset. Default is the paper-faithful step into the low state at
    # 5 Myr (duplicate 5 Myr knot, so jnp.interp jumps 0 -> low there);
    # CGOLS_STEP_ONSET=0 softens it to a 1 Myr linear ramp (5 -> 6 Myr). All
    # other transitions are the paper's 5 Myr linear ramps via interpolation
    # either way.
    onset_myr = 5.0 if STEP_ONSET else 6.0
    knot_times_myr = jnp.array([0.0, 5.0, onset_myr, 10.0, 40.0, 45.0])
    return CGOLSWindParams(
        # The paper's 300 pc gain region. Near-vacuum cells at the hot contacts
        # it drives are handled by the positivity_temperature_clip ceiling (see
        # build_config), so the physical radius is used directly.
        injection_radius=(INJ_RAD * u.pc).to(code_units.code_length).value,
        schedule_times=knot_times_myr * myr_to_code,
        schedule_mass_rates=jnp.array(
            [0.0, 0.0, mdot_low, mdot_high, mdot_high, mdot_low]
        ),
        schedule_energy_rates=jnp.array(
            [0.0, 0.0, edot_low, edot_high, edot_high, edot_low]
        ),
    )


# ---------------------------------------------------------------------------
# Gravitational potentials
# ---------------------------------------------------------------------------
M_disk_code = M_disk.to(code_units.code_mass).value
R_disk_code = R_disk.to(code_units.code_length).value
z_disk_code = z_disk.to(code_units.code_length).value
G_code = G.to(
    code_units.code_length**3 / (code_units.code_mass * code_units.code_time**2)
).value

M_halo_code = M_halo.to(code_units.code_mass).value
R_halo_code = R_halo.to(code_units.code_length).value


def Phi_disk_function(r, z):
    """Miyamoto-Nagai disk gravitational potential.

    Parameters
    ----------
    r : Cylindrical radius in code units.
    z : Vertical coordinate in code units.
    """
    return -(G_code * M_disk_code) / jnp.sqrt(
        r**2 + (R_disk_code + jnp.sqrt(z**2 + z_disk_code**2)) ** 2
    )


def Phi_halo_function(r):
    """NFW dark-matter halo gravitational potential.

    Parameters
    ----------
    r : Spherical radius in code units.
    """
    return (
        -(G_code * M_halo_code)
        / (r * (jnp.log(1 + c) - c / (1 + c)))
        * jnp.log(1 + r / R_halo_code)
    )


def vertical_hse_pressure(rho, Phi, dz_, P_top):
    """Pressure in discrete vertical HSE: integrate dP/dz = -rho dPhi/dz downward.

    Trapezoidal integration along each (x, y) column from the top boundary,
    where P = P_top. Returns a pressure consistent with the vertical force
    balance for the supplied density, so a (re)constructed IC does not ring
    vertically. (Matches the analytic isothermal solution to the discretisation
    error of the integrator.)
    """
    dPhi_dz = jnp.gradient(Phi, dz_, axis=2)
    q = rho * dPhi_dz  # equals -dP/dz
    incr = 0.5 * (q[:, :, :-1] + q[:, :, 1:]) * dz_  # P[k] - P[k+1]
    rev = jnp.cumsum(incr[:, :, ::-1], axis=2)[:, :, ::-1]
    return jnp.concatenate([P_top + rev, P_top], axis=2)


def _gaussian_smooth_3d(field, sigma):
    """Separable 3D Gaussian filter with edge-replicate padding, native JAX.

    Three sequential 1D convolutions on the GPU (one per axis); no GPU-host
    round-trip. Numerically equivalent (within float rounding) to
    scipy.ndimage.gaussian_filter(..., mode="nearest", truncate=4.0) and uses
    the same 4*sigma kernel radius. Returns a field of the same shape as the
    input.
    """
    if sigma <= 0:
        return field
    radius = int(4.0 * sigma + 0.5)
    xs = jnp.arange(-radius, radius + 1, dtype=field.dtype)
    k1d = jnp.exp(-0.5 * (xs / sigma) ** 2)
    k1d = k1d / k1d.sum()

    for axis in range(field.ndim):
        pad = [(0, 0)] * field.ndim
        pad[axis] = (radius, radius)
        padded = jnp.pad(field, pad, mode="edge")[None, None]
        kshape = [1, 1, 1]
        kshape[axis] = 2 * radius + 1
        kernel = k1d.reshape(kshape)[None, None]
        field = jax.lax.conv_general_dilated(
            padded, kernel,
            window_strides=(1, 1, 1),
            padding="VALID",
            dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
        )[0, 0]
    return field


# ---------------------------------------------------------------------------
# Initial-condition build
# ---------------------------------------------------------------------------
def build_config():
    """Build the SimulationConfig + registered_variables.

    Cheap: no 3D IC fields are materialised here. Factored out of
    `build_initial_conditions` so post-processing scripts (e.g. cgols_analyse.py)
    can reconstruct the grid/config without re-running the full IC build.
    """
    # ---- Grid / box ----
    bx_size_x = 10 * u.kiloparsec
    bx_size_y = 10 * u.kiloparsec
    bx_size_z = 20 * u.kiloparsec

    L_x = bx_size_x.to(code_units.code_length).value
    L_y = bx_size_y.to(code_units.code_length).value
    L_z = bx_size_z.to(code_units.code_length).value

    dim_x = dim_y = RESOLUTION
    dim_z = dim_x * 2

    rprint(f"Rendering in {dim_x} x {dim_y} x {dim_z} dimensions")

    if COOLING:
        rprint(
            f"Radiative cooling ON (B-series): CIE parabolic curve, n^2 Lambda at "
            f"mu = {mu}, floor {COOLING_FLOOR_K:.3g} K "
            f"(= {_kelvin_to_code_temperature(COOLING_FLOOR_K):.3e} code), "
            f"{COOLING_SUBCYCLES} sub-cycles, "
            + (
                f"dt <= {COOLING_DT_FRACTION:.2g} t_cool; "
                if COOLING_DT_LIMIT
                else "cooling dt limit OFF (pure CFL); "
            )
            + (
                "FCT preserving flux ON; "
                if PRESERVING_FLUX
                else "FCT preserving flux OFF - EXPECT THE CONTACT RUNAWAY; "
            )
            + f"outputs tagged {RUN_TAG!r}"
        )
    else:
        rprint("Radiative cooling OFF (adiabatic A-series); set CGOLS_COOLING=1 for the B-series")

    config = SimulationConfig(
        memory_analysis=True,
        # Build the padded helper data (here: only `r`, the cell-to-box-center
        # distance the CGOLS wind needs) in host RAM and transfer it sharded.
        # Built eagerly on GPU 0 instead, the (3, 1032, 1032, 2056) meshgrid +
        # its moveaxis copy were 2 x 24.5 GiB on ONE device before the solver
        # even started - that OOM'd the 1024 production run on a 141 GB H200
        # (job 4635570).
        #
        # In the insitu regime this flips to False: `r` is then built inside a
        # jit with out_shardings, so each PROCESS only materialises its own
        # shard. At 2048 the host path would put a full 69 GB `r` in every
        # rank's host RAM (~276 GB per 4-GPU node) - and the meshgrid blow-up
        # is gone anyway, since get_helper_data now derives `r` from broadcast
        # 1D axes. Left True below 2048 so every validated run keeps its exact
        # (eagerly built, bit-identical) geometry.
        host_helper_data=not INSITU_IC,
        geometry=CARTESIAN,
        solver_mode=FINITE_DIFFERENCE,
        # SSPRK4 (RK4_SSP) would be more robust to the strong wind-driven shocks,
        # but it carries one more full-state register and OOMs at 512x512x1024.
        # Staying on the memory-lean 2N-storage RK4_LSRK; stability at the sharp
        # wind/disk contacts is handled by the positivity backstops below.
        time_integrator=RK4_LSRK,
        backend_config=BackendConfig(
            backend=PALLAS,
            pallas_block_shape=PALLAS_BLOCK_SHAPE,
            pallas_use_triton=True,
            pallas_interpret=False,
        ),
        dimensionality=3,
        box_size=StaticFloatVector(L_x, L_y, L_z),
        num_cells=StaticIntVector(dim_x, dim_y, dim_z),
        # Positivity backstops, all required for this setup (values in the
        # params blocks):
        #   default_positivity_protection - per-stage + per-step HARD_FLOOR
        #     density/pressure floors.
        #   vacuum_rest - zeros a floored cell's momentum so the recovered
        #     velocity is 0 rather than momentum/rho_floor.
        #   nan_safe - resets any non-finite cell to a valid floor state.
        #   velocity_clip - caps |v| in EVERY cell (density-independent): the
        #     v = m/rho runaway forms in the near-floor band just ABOVE the
        #     density floor, out of reach of the density-keyed safeguards.
        #   temperature_clip - caps P/rho so c_s (and the CFL timestep) stays
        #     finite when a WENO energy overshoot lands in a near-vacuum cell.
        # All honoured bit-identically by the PALLAS kernels.
        positivity_config=PositivityConfig(
            default_positivity_protection=True,
            vacuum_rest=True,
            nan_safe=True,
            velocity_clip=True,
            temperature_clip=True,
            # FCT positivity flux limiter / deep-void LLF blending, env-gated
            # (CGOLS_PRESERVING_FLUX / CGOLS_DEEPVOID_BLEND); default OFF so the
            # bare adiabatic A-series numerics are unchanged.
            preserving_flux=PRESERVING_FLUX,
            deepvoid_blend=DEEPVOID_BLEND,
        ),
        # Benchmark mode: run exactly BENCH_STEPS equal-sized steps and print the
        # elapsed (post-compile) time so cgols_scaling.py can derive sec/step.
        # progress_bar / monitor add per-step host syncs, so drop them when timing.
        fixed_timestep=bool(BENCH_STEPS),
        num_timesteps=(BENCH_STEPS if BENCH_STEPS else 1000),
        print_elapsed_time=bool(BENCH_STEPS),
        # NB deliberately NOT `and RANK0`: config is a static argument to the
        # jitted solver, so a rank-dependent flag would compile a *different*
        # program on rank 0 than on the other ranks - invalid under
        # multi-process pjit. The duplicate output is suppressed inside the
        # library instead (time_stepping/_progress_bar.py rank-guards the host
        # side of both callbacks), which keeps every rank's program identical.
        progress_bar=(not BENCH_STEPS),
        monitor_diagnostics=(not BENCH_STEPS),
        # Outflow-only ("diode") boundaries on all six faces, as in the paper:
        # "transmissive boundaries with a 'diode' condition applied to the
        # velocities". A plain OPEN_BOUNDARY lets the static potential pull the
        # ghost-cell gas back into the box (slow artificial accretion off every
        # face, re-entrant recompression where the wind crosses +/-z). The
        # CGOLS_BOUNDARY knob switches back to "open" for A/B experiments.
        boundary_settings=BoundarySettings(
            BoundarySettings1D(left_boundary=_BOUNDARY_TYPE, right_boundary=_BOUNDARY_TYPE),
            BoundarySettings1D(left_boundary=_BOUNDARY_TYPE, right_boundary=_BOUNDARY_TYPE),
            BoundarySettings1D(left_boundary=_BOUNDARY_TYPE, right_boundary=_BOUNDARY_TYPE),
        ),
        gravity_config=GravityConfig(
            self_gravity=False,
            self_gravity_version=SIMPLE_SOURCE,
            external_potential=True,
        ),
        donate_state=True,
        cgols_wind_config=CGOLSWindConfig(cgols_wind=True),
        # Intermediate output for the animation / wind time-series. We use the
        # host-offload callback path (activate_snapshot_callback) rather than the
        # on-device snapshot store (return_snapshots + return_states): the latter
        # would keep num_snapshots full 3D states resident on the GPU (~5 GB each)
        # and OOM. The callback offloads only thin 2D slices + 1D reductions per
        # snapshot (see make_snapshot_callable). num_snapshots sets the cadence:
        # one frame every t_end / num_snapshots.
        activate_snapshot_callback=(not BENCH_STEPS),
        num_snapshots=NUM_SNAPSHOTS,
        # Full-state checkpointing (see the CGOLS_CHECKPOINT_* knobs): Orbax
        # TO_DISK mode writes the loop carry after every snapshot segment,
        # sharded per device. The frame callback above keeps working - the
        # driver invokes it at each segment end. Off in benchmark mode.
        **(
            dict(
                snapshot_storage_mode=TO_DISK,
                snapshot_storage_path=_data(f"cgols_checkpoints{RUN_TAG}"),
            )
            if CHECKPOINT_EVERY and not BENCH_STEPS
            else {}
        ),
        # Radiative cooling (CGOLS_COOLING=1, the paper's B-series). Conditional
        # kwarg so the adiabatic static config stays byte-identical - it is a
        # static_argnames member, so any change forces a recompile.
        #   SUBCYCLED_EXPLICIT_COOLING + COOLING_OPERATOR_SPLIT reproduce
        #   Cholla's scheme: forward Euler, <=1% dT per sub-cycle, applied once
        #   per step to the post-hydro state.
        #   TOTAL_NUMBER_DENSITY selects n^2 Lambda (n = rho / (mu m_p)) over
        #   the module's default n_e n_H - a factor mu_e mu_H / mu^2 ~ 4.30.
        #   COOLING_FLOOR_CLIP clamps to the 10^4 K floor rather than reverting.
        **(
            dict(
                cooling_config=CoolingConfig(
                    cooling=True,
                    cooling_method=SUBCYCLED_EXPLICIT_COOLING,
                    cooling_placement=COOLING_OPERATOR_SPLIT,
                    floor_mode=COOLING_FLOOR_CLIP,
                    max_subcycles=COOLING_SUBCYCLES,
                    max_fractional_temperature_change=0.01,
                    cooling_timestep_limit=COOLING_DT_LIMIT,
                    cooling_curve_config=CoolingCurveConfig(
                        cooling_curve_type=CIE_PARABOLIC,
                        density_convention=TOTAL_NUMBER_DENSITY,
                    ),
                )
            )
            if COOLING
            else {}
        ),
    )
    # Derive the scalar grid spacing HERE. finalize_config does it too, but only
    # once the state shape is known - until then the field still holds the
    # SimulationConfig default (0.0025), which is nothing like this box's
    # 10 kpc / dim. Everything in this file that reads config.grid_spacing (the
    # analytic IC coordinates in _ic_axes, the IC diagnostics) necessarily runs
    # BEFORE finalize_config, and silently building the IC on a 0.0025 grid
    # produces a plausible-looking but completely wrong initial state.
    config = config._replace(grid_spacing=(config.box_size / config.num_cells).x)

    registered_variables = get_registered_variables(config)
    return config, registered_variables


def _replicate(x, sharding):
    """Pin ``x`` to a fully replicated sharding (no-op without a sharding).

    Used on the small 1D/scalar outputs of the jitted IC build: the all-gather
    then happens INSIDE the jit, where it is legal, and the host gets the
    global array rather than one device's shard.
    """
    if sharding is None:
        return x
    return jax.lax.with_sharding_constraint(x, jax.NamedSharding(sharding.mesh, P()))


def _ic_axes(config):
    """The three box-centered 1D cell-center axes, in code units.

    Reproduces the unpadded (``ngc = 0``) 3D-Cartesian branch of astronomix's
    ``_build_helper_data``, so an in-situ IC stays bit-comparable with one built
    from ``get_helper_data(config).geometric_centers``. These arrays are tiny
    (``N``, not ``N^3``) and stay replicated; every diagnostic line below is
    taken from them rather than by indexing a sharded 3D field.
    """
    gs = config.grid_spacing
    nx, ny, nz = config.num_cells.x, config.num_cells.y, config.num_cells.z
    L_x, L_y, L_z = config.box_size.x, config.box_size.y, config.box_size.z
    # Guard against an un-normalised config reaching here: SimulationConfig's
    # grid_spacing default is 0.0025, and a wrong spacing does not fail loudly -
    # it silently builds the disk on the wrong grid. build_config() sets it, and
    # finalize_config re-derives the same value.
    if not math.isclose(gs, L_x / nx, rel_tol=1e-6):
        raise ValueError(
            f"config.grid_spacing={gs} does not match box_size/num_cells="
            f"{L_x / nx} - the config was not normalised (see build_config)."
        )
    return (
        jnp.linspace(gs / 2, L_x + gs / 2, nx, endpoint=False) - L_x / 2,
        jnp.linspace(gs / 2, L_y + gs / 2, ny, endpoint=False) - L_y / 2,
        jnp.linspace(gs / 2, L_z + gs / 2, nz, endpoint=False) - L_z / 2,
    )


def _ic_coordinates(config, sharding):
    """The box-centered ``(X_c, Y_c, Z_c)`` meshgrid, sharded like the state.

    The sharding constraint is applied to the stacked coordinates *immediately*,
    before any field is evaluated, so the global meshgrid is never formed on one
    device and (multi-process) each process only materialises its own shard.
    This is the ``test_setups/hydrodynamics/sound_wave3D.py`` pattern.
    """
    x1, y1, z1 = _ic_axes(config)
    X_c, Y_c, Z_c = jnp.meshgrid(x1, y1, z1, indexing="ij")
    if sharding is not None:
        # Leading stack axis replicated; X/Y/Z onto the same mesh axes as the
        # primitive state (drop its leading vars entry).
        X_c, Y_c, Z_c = jax.lax.with_sharding_constraint(
            jnp.stack([X_c, Y_c, Z_c]),
            jax.NamedSharding(sharding.mesh, P(None, *sharding.spec[1:4])),
        )
    return X_c, Y_c, Z_c


def _external_potential_fields(config, sharding):
    """``Phi_total`` on the full grid — the static Miyamoto-Nagai + NFW potential."""
    X_c, Y_c, Z_c = _ic_coordinates(config, sharding)
    dx = config.grid_spacing
    R_cyl = jnp.maximum(jnp.sqrt(X_c**2 + Y_c**2), 0.25 * dx)
    r_sph = jnp.maximum(jnp.sqrt(X_c**2 + Y_c**2 + Z_c**2), 0.25 * dx)
    return Phi_disk_function(R_cyl, Z_c) + Phi_halo_function(r_sph)


def build_external_potential(config, sharding=None):
    """The external potential alone, built (optionally) straight into its shards.

    This is all a RESTART leg needs: the state comes from the Orbax checkpoint,
    so building the full IC only to throw it away costs ~344 GB of pointless
    work at 2048.
    """
    if sharding is None:
        return _external_potential_fields(config, None)
    spatial = jax.NamedSharding(sharding.mesh, P(*sharding.spec[1:4]))
    return jax.jit(
        lambda: _external_potential_fields(config, sharding), out_shardings=spatial
    )()


def _build_ic_fields(config, registered_variables, Phi_total, sharding):
    """Assemble the initial primitive state and the IC diagnostics.

    ``Phi_total`` is passed in rather than recomputed so that the potential a
    RESTART leg builds (via :func:`build_external_potential` alone) is
    bit-identical to the one a fresh leg used: two different jits computing the
    same expression get fused differently by XLA and agree only to ~1e-7, which
    would silently break "a resumed run reproduces an uninterrupted one".
    Splitting the build in two also keeps either jit's live set smaller.

    Pure and jittable: no prints, no plots, no host syncs — every diagnostic is
    returned as a (replicated) array instead, so this whole function can run
    inside one ``jax.jit`` with ``out_shardings`` and each device builds only
    its own shard of a 344 GB state. See :func:`build_initial_conditions` for
    the driver and :func:`plot_ic_diagnostics` for the eager, rank-0 rendering.

    Memory layout: at any moment only a handful of 3D fields are alive.
    Intermediates (cutoff, Sigma, exp_factor, bracket, Phi_sph, the pressure
    components, the pressure gradients, ...) are deleted as soon as their last
    consumer is done — which matters on the eager path and is a no-op under jit,
    where XLA does its own liveness analysis. The 3D Phi_disk / Phi_halo arrays
    are never materialised.

    Sharding notes for the operations that are not plain elementwise maps:
      * ``vertical_hse_pressure``'s ``jnp.gradient`` / ``jnp.cumsum`` run along
        z, which is why z must stay UNSHARDED (a z-split lowers the cumsum to a
        cross-device scan).
      * ``_gaussian_smooth_3d`` needs a 6-cell halo on the split x/y axes;
        GSPMD inserts it, the same machinery as the solver's WENO halo. It is
        the deepest stencil in the codebase.
      * ``Phi_total[:, :, mid_z]`` indexes only the unsharded z axis - safe.
      * every 1D diagnostic line is built from the replicated axis arrays, not
        by indexing into a sharded 3D field.
    """
    L_x = config.box_size.x
    L_y = config.box_size.y
    L_z = config.box_size.z
    dim_x = config.num_cells.x
    dim_y = config.num_cells.y
    dim_z = config.num_cells.z

    x1, y1, z1 = _ic_axes(config)
    X_c, Y_c, Z_c = _ic_coordinates(config, sharding)

    # The exact fp64 cell size, not an fp32 difference of two coordinates.
    # finalize_config asserts the spacing is equal on all three axes in 3D
    # (simulation_config.py), and reading it here avoids a host sync plus a
    # gather on what is now a sharded axis.
    dx = dy = dz = config.grid_spacing

    # Cylindrical and spherical radii (code units), floored to avoid divide-by-zero.
    R_cyl = jnp.maximum(jnp.sqrt(X_c**2 + Y_c**2), 0.25 * dx)
    r_sph = jnp.maximum(jnp.sqrt(X_c**2 + Y_c**2 + Z_c**2), 0.25 * dx)

    mid_x = dim_x // 2
    mid_y = dim_y // 2
    mid_z = dim_z // 2

    # ---- Constants needed for both gas components ----
    k_B_code = k_B.to(code_units.code_energy / u.K).value
    T_disk_code = T_disk.to(u.K).value
    m_p_code = m_p.to(code_units.code_mass).value
    c_s_d = jnp.sqrt(k_B_code * T_disk_code / (mu * m_p_code))  # isothermal disk

    T_halo_code = T_halo.to(u.K).value
    rho_0h_code = rho_0h.to(code_units.code_density).value
    c_s_h = jnp.sqrt(k_B_code * T_halo_code / (mu * m_p_code))
    K_const = c_s_h**2 * rho_0h_code ** (1 - gamma) / gamma  # adiabatic P = K rho^gamma

    # ---- Disk gas: surface density (exponential with smooth truncation) ----
    M_gas_code = M_gas.to(code_units.code_mass).value
    R_gas_code = R_gas.to(code_units.code_length).value
    sigma_0 = M_gas_code / (2 * jnp.pi * R_gas_code**2)
    R_trunc_code = R_trunc.to(code_units.code_length).value
    delta = (0.15 * u.kiloparsec).to(code_units.code_length).value  # ramp width

    cutoff = jnp.where(
        R_cyl < R_trunc_code,
        1.0,
        jnp.exp(-((R_cyl - R_trunc_code) / delta) ** 2),
    )
    Sigma = sigma_0 * jnp.exp(-R_cyl / R_gas_code) * cutoff
    del cutoff

    # ---- Disk gas: vertical density structure (isothermal, Eq. 4) ----
    #   rho(z) = rho_0d * exp[-(Phi(z) - Phi_0d) / c_s_d^2]
    # Note: c_s_d is the *isothermal* sound speed, so P_disk = rho * c_s_d**2 is the
    # true thermal pressure rho * k_B * T / (mu * m_p). The paper writes
    # P = rho c_s^2 / gamma in Eq. 6, but there "c_s" is the adiabatic sound speed;
    # with the isothermal c_s_d defined in Eq. 4 the 1/gamma must be dropped.
    Phi_0d = Phi_total[:, :, mid_z]
    exp_factor = jnp.exp(-(Phi_total - Phi_0d[:, :, None]) / c_s_d**2)
    integral = jnp.sum(exp_factor, axis=2) * dz  # shape (dim_x, dim_y)
    rho_0d = Sigma[:, :, mid_z] / integral
    rho_disk = rho_0d[:, :, None] * exp_factor
    del exp_factor, Sigma, integral, rho_0d, Phi_0d

    # ---- Halo gas: hydrostatic, adiabatic atmosphere ----
    #   rho(r) = rho_0h * [1 - (gamma-1) (Phi - Phi_0h) / c_s_h^2]^(1/(gamma-1))
    # Spherically symmetric potential (disk evaluated on the sphere). We recompute
    # Phi_halo on r_sph here rather than caching the 3D field.
    Phi_sph = Phi_disk_function(0, r_sph) + Phi_halo_function(r_sph)
    r_ref_code = (100 * u.kiloparsec).to(code_units.code_length).value
    Phi_0h = Phi_disk_function(0, r_ref_code) + Phi_halo_function(r_ref_code)

    # Correct hydrostatic-equilibrium solution. Integrating dP/dr = -rho dPhi/dr for
    # an adiabatic gas (P = K rho^gamma) gives rho ~ [1 + (gamma-1)(Phi_0h - Phi)/c_s_h^2],
    # i.e. the bracket below. Inside r = 100 kpc, Phi < Phi_0h so the bracket is > 1
    # and density rises inward (matching the paper's Figure 4). The printed Eq. 7 reads
    # with the opposite sign on (Phi - Phi_0h), but this physical form is the one that
    # reproduces their profiles, so we keep it. Floored at 0 for safety outside the box.
    bracket = 1 - (gamma - 1) * (Phi_sph - Phi_0h) / c_s_h**2
    del Phi_sph
    rho_halo = rho_0h_code * jnp.maximum(bracket, 0.0) ** (1 / (gamma - 1))
    del bracket, r_sph

    # ---- Vertical resolution diagnostic (only needs Phi_total + scalars) ----
    # Recomputed analytically on the central column rather than sliced out of
    # Phi_total: Phi_total[mid_x, mid_y, :] is an integer index into TWO sharded
    # axes. Identical arithmetic on identical inputs, so identical values - the
    # central column sits at (dx/2, dy/2), half a cell off the axis, exactly as
    # the 3D field has it.
    R_cyl_col = jnp.maximum(jnp.sqrt(x1[mid_x] ** 2 + y1[mid_y] ** 2), 0.25 * dx)
    r_sph_col = jnp.maximum(
        jnp.sqrt(x1[mid_x] ** 2 + y1[mid_y] ** 2 + z1**2), 0.25 * dx
    )
    phi_col = Phi_disk_function(R_cyl_col, z1) + Phi_halo_function(r_sph_col)
    d2Phi_dz2_mid = jnp.gradient(jnp.gradient(phi_col, dz), dz)[mid_z]
    H_gas = c_s_d / jnp.sqrt(jnp.maximum(d2Phi_dz2_mid, 1e-30))
    del phi_col, R_cyl_col, r_sph_col

    # ---- Branch on smoothing: build (rho_total, P_total, v_phi, ux, uy, uz) ----
    # Smoothed path: linear smoothing of rho_total (= smooth(rho_disk)+smooth(rho_halo)),
    # then REBUILD the pressure in vertical HSE with the smoothed density. Using the
    # *total* P and rho means the pressure-supported halo gets a_phi ~ 0 automatically,
    # so no disk-fraction weighting is needed.
    # Unsmoothed path: keep P_disk for the pressure-gradient term and weight v_phi by
    # rho_disk / rho_total so the static hot halo is not spun up.
    if SMOOTHING_SIGMA_CELLS > 0:
        # Top-boundary halo pressure for HSE: a (Nx, Ny, 1) slice, not full 3D.
        P_top_2d = K_const * rho_halo[:, :, -1:] ** gamma
        rho_total = rho_disk + rho_halo
        del rho_disk, rho_halo

        rho_total = _gaussian_smooth_3d(rho_total, SMOOTHING_SIGMA_CELLS)
        P_total = vertical_hse_pressure(rho_total, Phi_total, dz, P_top_2d)
        del P_top_2d

        # Rotation from total fields.
        dPhi_dx = jnp.gradient(Phi_total, dx, axis=0)
        dPhi_dy = jnp.gradient(Phi_total, dy, axis=1)
        dPhi_dR = (X_c * dPhi_dx + Y_c * dPhi_dy) / R_cyl
        del dPhi_dx, dPhi_dy

        dP_dx = jnp.gradient(P_total, dx, axis=0)
        dP_dy = jnp.gradient(P_total, dy, axis=1)
        dP_dR = (X_c * dP_dx + Y_c * dP_dy) / R_cyl
        del dP_dx, dP_dy

        a_phi = dPhi_dR + dP_dR / rho_total
        del dPhi_dR, dP_dR
        v_phi = jnp.sqrt(jnp.maximum(a_phi * R_cyl, 0.0))
        del a_phi
    else:
        rho_total = rho_disk + rho_halo
        # Build P_total without ever holding both 3D pressure components.
        P_total = rho_disk * c_s_d**2 + K_const * rho_halo**gamma
        del rho_halo

        dPhi_dx = jnp.gradient(Phi_total, dx, axis=0)
        dPhi_dy = jnp.gradient(Phi_total, dy, axis=1)
        dPhi_dR = (X_c * dPhi_dx + Y_c * dPhi_dy) / R_cyl
        del dPhi_dx, dPhi_dy

        # Pressure gradient uses the *disk* pressure only (P_disk = rho_disk * c_s_d**2).
        # We recompute it inside the gradient call rather than caching a 3D P_disk.
        P_disk_for_grad = rho_disk * c_s_d**2
        dP_dx = jnp.gradient(P_disk_for_grad, dx, axis=0)
        dP_dy = jnp.gradient(P_disk_for_grad, dy, axis=1)
        del P_disk_for_grad
        dP_dR = (X_c * dP_dx + Y_c * dP_dy) / R_cyl
        del dP_dx, dP_dy

        # Centripetal acceleration. Gravity (dPhi/dR > 0) drives the rotation; the
        # outward pressure gradient (dP/dR < 0) reduces it. The disk density in the
        # pressure term is floored so the 1/rho term does not blow up where the disk
        # gas vanishes far from the plane.
        safe_rho_disk = jnp.maximum(rho_disk, 1e-12)
        a_phi = dPhi_dR + dP_dR / safe_rho_disk
        del safe_rho_disk, dPhi_dR, dP_dR
        v_phi = jnp.sqrt(jnp.maximum(a_phi * R_cyl, 0.0))
        del a_phi

        # Mass-weight the rotation by the disk fraction so the static hot halo is not
        # spun up: halo-dominated cells smoothly go to v -> 0, while the disk midplane
        # (rho_disk >> rho_halo) keeps the full circular speed.
        v_phi = v_phi * (rho_disk / rho_total)
        del rho_disk

    # ---- Velocity components from v_phi (Eq. 6 chain rule):
    # dPhi/dR = (x/R) dPhi/dx + (y/R) dPhi/dy, v_x = -v_phi y/R, v_y = v_phi x/R.
    ux = -v_phi * Y_c / R_cyl
    uy = v_phi * X_c / R_cyl
    uz = jnp.zeros_like(ux)
    del v_phi, R_cyl, X_c, Y_c, Z_c

    # ---- Diagnostics: 1D profile lines + reduction scalars ----------------
    # Returned instead of printed/plotted, so this function stays jittable.
    # Each is forced to a replicated sharding, i.e. the all-gather happens here,
    # inside the jit, rather than being an illegal gather on the host.
    code_density_cgs = (code_mass / code_length**3).to(u.g / u.cm**3).value
    m_p_cgs = m_p.to(u.g).value
    n_factor = code_density_cgs / (mu * m_p_cgs)

    rho_mid = rho_total[mid_x:, mid_y, mid_z]
    rho_zax = rho_total[mid_x, mid_y, mid_z:]
    P_mid = P_total[mid_x:, mid_y, mid_z]
    P_zax = P_total[mid_x, mid_y, mid_z:]

    diagnostics = {
        "H_gas": H_gas,
        "P_total_min": P_total.min(),
        "P_total_max": P_total.max(),
        "density_min": rho_total.min(),
        "density_max": rho_total.max(),
        "velocity_x_min": ux.min(),
        "velocity_x_max": ux.max(),
        "velocity_y_min": uy.min(),
        "velocity_y_max": uy.max(),
        "velocity_z_min": uz.min(),
        "velocity_z_max": uz.max(),
        "pressure_min": P_total.min(),
        "pressure_max": P_total.max(),
        "n_midplane": rho_mid * n_factor,
        "n_zaxis": rho_zax * n_factor,
        "T_midplane": P_mid / rho_mid * T_factor,
        "T_zaxis": P_zax / rho_zax * T_factor,
    }
    diagnostics = {k: _replicate(v, sharding) for k, v in diagnostics.items()}
    del rho_mid, rho_zax, P_mid, P_zax

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho_total,
        velocity_x=ux,
        velocity_y=uy,
        velocity_z=uz,
        gas_pressure=P_total,
        sharding=sharding,
    )
    del rho_total, ux, uy, uz, P_total

    return initial_state, diagnostics


def plot_ic_diagnostics(config, diagnostics):
    """Print the IC summary and render the two IC figures. Rank 0, eager.

    Takes the (already gathered, host-resident) diagnostics returned by
    :func:`_build_ic_fields`; the rotation curve is analytic and is simply
    recomputed here rather than carried out of the jit.
    """
    if not RANK0:
        return

    x1, _, z1 = _ic_axes(config)
    dx = dz = config.grid_spacing
    mid_x = config.num_cells.x // 2
    mid_z = config.num_cells.z // 2
    pc_per_code = code_length.to(u.pc).value
    H_gas = float(diagnostics["H_gas"])

    rprint(f"Cell size dz               = {dz * pc_per_code:8.1f} pc")
    rprint(f"Central gas scale height H = {H_gas * pc_per_code:8.1f} pc")
    rprint(f"Cells per gas scale height = {H_gas / dz:8.2f}")
    if SMOOTHING_SIGMA_CELLS > 0:
        rprint(
            f"Applied HSE-preserving smoothing: sigma = {SMOOTHING_SIGMA_CELLS} cells"
        )
    rprint(f"P_total min: {float(diagnostics['P_total_min']):.3e} code units")
    rprint(f"P_total max: {float(diagnostics['P_total_max']):.3e} code units")
    for name in ("density", "velocity_x", "velocity_y", "velocity_z", "pressure"):
        rprint(
            f"Initial {name + ' min:':<15} {float(diagnostics[name + '_min']):.3e}"
        )
        rprint(
            f"Initial {name + ' max:':<15} {float(diagnostics[name + '_max']):.3e}"
        )

    # ---- Diagnostic plot: rotation curve (midplane), from 1D potential slices ----
    x_full = x1
    r_line = jnp.maximum(jnp.abs(x_full), 0.25 * dx)
    z_zero = jnp.zeros_like(r_line)
    Phi_disk_line = Phi_disk_function(r_line, z_zero)
    Phi_halo_line = Phi_halo_function(r_line)

    dPhi_disk_dx = jnp.gradient(Phi_disk_line, dx)
    dPhi_halo_dx = jnp.gradient(Phi_halo_line, dx)
    dPhi_total_dx = dPhi_disk_dx + dPhi_halo_dx

    # v_circ = sign(x) * sqrt(|x * dPhi/dx|)
    v_disk_rot = jnp.sign(x_full) * jnp.sqrt(jnp.maximum(jnp.abs(x_full * dPhi_disk_dx), 0))
    v_halo_rot = jnp.sign(x_full) * jnp.sqrt(jnp.maximum(jnp.abs(x_full * dPhi_halo_dx), 0))
    v_total_rot = jnp.sign(x_full) * jnp.sqrt(jnp.maximum(jnp.abs(x_full * dPhi_total_dx), 0))

    v_to_kms = code_velocity.to(u.km / u.s).value

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x_full, v_total_rot * v_to_kms, "b-", linewidth=1.5, label="combined")
    ax.plot(x_full, v_disk_rot * v_to_kms, color="orange", linestyle="-.", linewidth=1.5, label="disk")
    ax.plot(x_full, v_halo_rot * v_to_kms, "g--", linewidth=1.5, label="halo")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-200, 200)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel(r"$v_{\rm circ}$ [km s$^{-1}$]")
    ax.legend()
    ax.set_title("Rotation curve (midplane)")
    plt.tight_layout()
    plt.savefig(_fig(f"cgols_rotation_curve{IC_TAG}.png"), dpi=300)
    plt.close(fig)

    # ---- Diagnostic plot: density and temperature profiles ----
    y_mid = _ic_axes(config)[1][config.num_cells.y // 2]
    R_kpc = jnp.sqrt(x1[mid_x:] ** 2 + y_mid**2)
    z_kpc = z1[mid_z:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    ax1.plot(R_kpc, diagnostics["n_midplane"], "b--", linewidth=2, label="xy-plane")
    ax1.plot(z_kpc, diagnostics["n_zaxis"], "r:", linewidth=2, label="z-axis")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(1e-2, 1e1)
    ax1.set_ylim(1e-4, 1e3)
    ax1.set_xlabel("r [kpc]")
    ax1.set_ylabel(r"n [cm$^{-3}$]")
    ax1.legend()
    ax1.set_title("Density profiles")

    ax2.plot(R_kpc, diagnostics["T_midplane"], "b--", linewidth=2, label="xy-plane")
    ax2.plot(z_kpc, diagnostics["T_zaxis"], "r:", linewidth=2, label="z-axis")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1e-2, 1e1)
    ax2.set_ylim(1e3, 1e7)
    ax2.set_xlabel("r [kpc]")
    ax2.set_ylabel("T [K]")
    ax2.legend()
    ax2.set_title("Temperature profiles")

    plt.tight_layout()
    plt.savefig(_fig(f"cgols_initial_profiles{IC_TAG}.png"), dpi=300)
    plt.close(fig)


def build_simulation_params(Phi_total):
    """The SimulationParams shared by the in-situ and file IC paths."""
    return SimulationParams(
        t_end=END_TIME.to(code_units.code_time).value,
        C_cfl=CFL,
        gamma=gamma,
        # Floors ~2 orders below the ambient box minimums (min_rho~8e-3,
        # min_P~3e-3), not the dynamically-zero 1e-14 default: a wind-cavity
        # cell floored to 1e-14 next to a 3e-3 neighbour is a ~1e11 pressure
        # ratio, whose flux evacuates the cell in one step -> NaN. These floors
        # cap that gradient while leaving the rarefied cavity room to form.
        minimum_density=1e-4,
        minimum_pressure=1e-5,
        # Global velocity ceiling for positivity_velocity_clip (50 code = 5000
        # km/s: well above any physical galactic wind, far below the near-floor
        # v = m/rho runaway it exists to stop). Caps |v| in EVERY cell, not just
        # sub-floor ones.
        positivity_max_velocity=50.0,
        # Temperature ceiling for positivity_temperature_clip. The solver caps
        # P/rho in code units, so convert the physical T_max via T_factor
        # (T = (P/rho)*T_factor). Default 5e9 K: far above the ~2e7 K CC85 hot
        # wind and its transient shock heating, far below the 1e21+ K
        # near-vacuum spikes that collapse the timestep. Do NOT tighten it
        # toward the physical temperatures - see the CGOLS_TMAX_K knob comment.
        positivity_max_pressure_over_density=TMAX_K / T_factor,
        gravitational_potential=Phi_total,
        cgols_wind_params=build_cgols_wind_params(),
        **_cooling_params_kwargs(),
    )


def build_initial_conditions(sharding=None, save_potential=None):
    """Build the simulation IC and return the four objects time_integration needs.

    With a ``sharding`` the whole field build runs inside ONE jit with
    ``out_shardings``, so every device (and, multi-process, every process) only
    ever materialises its own shard: at 2048^2 x 4096 the state is 344 GB and the
    potential 69 GB, neither of which fits on a device or in a node's RAM.
    Without a sharding this is the original eager single-device build.

    ``save_potential`` defaults to the file-IC mode (CGOLS_IC_MODE=file), where
    the .npy pair is the artefact the solver run later loads.
    """
    config, registered_variables = build_config()
    if save_potential is None:
        save_potential = not INSITU_IC

    # Two jits, not one: see _build_ic_fields on why the potential must come
    # out of its own (identically-compiled) build.
    Phi_total = build_external_potential(config, sharding)

    def _build(Phi):
        return _build_ic_fields(config, registered_variables, Phi, sharding)

    if sharding is None:
        initial_state, diagnostics = _build(Phi_total)
    else:
        replicated = jax.NamedSharding(sharding.mesh, P())
        struct = jax.eval_shape(_build, Phi_total)
        out_shardings = (sharding, jax.tree.map(lambda _: replicated, struct[1]))
        initial_state, diagnostics = jax.jit(_build, out_shardings=out_shardings)(
            Phi_total
        )

    plot_ic_diagnostics(config, diagnostics)

    params = build_simulation_params(Phi_total)
    if save_potential:
        jnp.save(_ic_potential_path(), Phi_total)
    del Phi_total

    config = finalize_config(config, initial_state.shape)

    return initial_state, config, params, registered_variables


def load_initial_conditions():
    """Load the initial state and config from disk, for post-processing without re-running the IC build."""
    config, registered_variables = build_config()
    try:
        initial_state = jnp.load(_ic_state_path())
        Phi_total = jnp.load(_ic_potential_path())
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No initial conditions for CGOLS_DIM={RESOLUTION} (missing {e.filename}). "
            f"Build them once with: CGOLS_CREATE_IC=1 CGOLS_DIM={RESOLUTION} python cgols.py"
        ) from e
    expected = (RESOLUTION, RESOLUTION, 2 * RESOLUTION)
    if initial_state.shape[1:] != expected:
        raise ValueError(
            f"IC file {_ic_state_path()} has grid {initial_state.shape[1:]} but "
            f"CGOLS_DIM={RESOLUTION} expects {expected} - wrong CGOLS_IC_TAG?"
        )
    config = finalize_config(config, initial_state.shape)
    return initial_state, config, build_simulation_params(Phi_total), registered_variables


# ---------------------------------------------------------------------------
# Intermediate snapshots (host-offloaded, for the animation / wind time-series)
# ---------------------------------------------------------------------------
SNAPSHOTS_DIR = _here(f"cgols_snapshots{RUN_TAG}")


def make_snapshot_callable(
    config, registered_variables, out_dir=SNAPSHOTS_DIR, sharding=None
):
    """Build the snapshot callable that streams frames to disk during the run.

    Returns the callable; pass it as the 5th argument to ``time_integration``.
    Pass the run's ``sharding`` so the operands handed to the host callback are
    explicitly all-gathered (see ``_replicated``) and only rank 0 writes.
    Each frame is written to ``out_dir/frame_NNNN.npz`` *immediately* inside the
    callback - it is NOT accumulated in host RAM and flushed at the end. Two
    reasons:
      1. Memory: an end-of-run flush would hold all ``num_snapshots`` frames in
         host RAM until the run finishes (~5 MB/frame). Streaming keeps only one
         frame alive at a time.
      2. Crash-robustness: if a run dies (blow-up, OOM, node failure), an
         end-of-run flush would lose every snapshot; streaming means each frame
         already on disk survives, so the evolution up to the crash can still
         be analysed and animated.

    What crosses to the host is only thin 2D planes (edge-on / face-on density and
    temperature) and a 1D vertical mass-flux profile - never the full 3D state
    (~5 GB; keeping ``num_snapshots`` of those on-device via ``return_snapshots`` +
    ``return_states`` would OOM, which is why we use the callback path).

    The state handed to the callable is the *padded* state (the callback path does
    not unpad, unlike ``return_snapshots``); we slice the physical interior out of
    it directly with the ghost offset, which also avoids a full-volume unpad copy.
    """
    out_dir = _here(out_dir)
    g = config.num_ghost_cells
    dim_x, dim_y, dim_z = config.num_cells.x, config.num_cells.y, config.num_cells.z

    # Physical mid-plane indices in *padded* coordinates, and an upper-slice bound
    # that also works when g == 0 (periodic roll).
    my = g + dim_y // 2
    mz = g + dim_z // 2
    hi = -g if g > 0 else None

    code_density_cgs = (code_mass / code_length**3).to(u.g / u.cm**3).value
    m_p_cgs = m_p.to(u.g).value
    n_factor = code_density_cgs / (mu * m_p_cgs)
    code_time_myr = (1 * code_units.code_time).to(u.Myr).value
    dx = config.box_size.x / dim_x
    dy = config.box_size.y / dim_y
    code_mdot_to_msun_per_yr = (code_mass / code_units.code_time).to(u.M_sun / u.yr).value

    # Radial-velocity slice support (the middle panels of Figs 1/2 in
    # arXiv:1803.01005): unit vectors r_hat on the edge-on plane. The y = 0
    # plane sits half a cell off the axis, so the y-contribution to v_r there
    # is O(dy/2r) and is dropped; r is clamped to half a cell so the origin
    # cell stays finite.
    code_velocity_kms = (1 * code_units.code_velocity).to(u.km / u.s).value
    x_c = (jnp.arange(dim_x) + 0.5) * dx - config.box_size.x / 2
    z_c = (jnp.arange(dim_z) + 0.5) * (config.box_size.z / dim_z) - config.box_size.z / 2
    X_plane = x_c[:, None]
    Z_plane = z_c[None, :]
    R_plane = jnp.maximum(jnp.sqrt(X_plane**2 + Z_plane**2), 0.5 * dx)

    # Orbax full-state checkpoints are multi-GB, so they go to the bulk-data
    # dir (scratch), separate from the small frame files.
    ckpt_dir = _data(f"cgols_checkpoints{RUN_TAG}")

    # Fresh output directories: drop any frames from a previous run so they
    # cannot be mixed into this run's animation / forensics. On a cold start
    # also clear old checkpoint step dirs (and legacy npz checkpoints):
    # leftover steps would make the TO_DISK driver continue their numbering and
    # skip the t=0 frame. On a resume (CGOLS_RESTART_FROM) the history must
    # stay - the numbering continues from it.
    #
    # Rank 0 only, with a barrier after: unguarded, rank 5 would rmtree the
    # checkpoint directory rank 0 is already writing into. The barrier is a
    # collective, so it sits OUTSIDE the RANK0 branch - every rank must reach
    # it - and it must happen before the TO_DISK driver's first latest_step()
    # read, which is itself a collective over this directory.
    if RANK0:
        os.makedirs(out_dir, exist_ok=True)
        for f in glob.glob(os.path.join(out_dir, "frame_*.npz")):
            os.remove(f)
        if CHECKPOINT_EVERY:
            os.makedirs(ckpt_dir, exist_ok=True)
            if not RESTART_FROM:
                for entry in glob.glob(os.path.join(ckpt_dir, "*")):
                    if os.path.isdir(entry):
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        os.remove(entry)
    barrier("cgols:output-dirs-ready")

    # Host-side frame counter for the filename. The callback may fire unordered, so
    # filenames are not assumed to be time-ordered; each file stores its own time
    # and the loader sorts by it.
    counter = {"i": 0}

    def _save_frame(time, n_xz, T_xz, n_xy, mdot_z, vr_xz):
        # jax.debug.callback fires on every process with the same (explicitly
        # all-gathered, see _replicated below) global planes; one writer only,
        # or 32 ranks race on frame_NNNN.npz.
        if not RANK0:
            return
        i = counter["i"]
        counter["i"] += 1
        # np.savez (uncompressed) keeps the per-frame write cheap; the per-run
        # total is ~300 MB of disk, trivial next to the .npy states.
        np.savez(
            os.path.join(out_dir, f"frame_{i:04d}.npz"),
            time_myr=np.float32(float(time) * code_time_myr),
            n_xz=np.asarray(n_xz, dtype=np.float32),
            T_xz=np.asarray(T_xz, dtype=np.float32),
            n_xy=np.asarray(n_xy, dtype=np.float32),
            mdot_z=np.asarray(mdot_z, dtype=np.float32),
            vr_xz=np.asarray(vr_xz, dtype=np.float32),
        )

    # Rolling retention for the Orbax checkpoints (CGOLS_CHECKPOINT_EVERY > 0):
    # the TO_DISK driver writes one step dir per frame and never deletes; keep
    # only the newest CHECKPOINT_KEEP steps. Runs as a host callback after each
    # frame - the driver saves the checkpoint synchronously BEFORE invoking the
    # callable, so the newest step is always complete when we prune.
    def _prune_checkpoints():
        # Rank 0 only. NB this prunes *older* steps while the newest one has
        # already been committed by the (collective) Orbax save, so it never
        # races that save. It must stay outside any barrier pairing, since it
        # runs inside a debug callback whose ordering across ranks is not
        # guaranteed.
        if not RANK0:
            return
        steps = sorted(int(d) for d in os.listdir(ckpt_dir) if d.isdigit())
        for step in steps[:-CHECKPOINT_KEEP]:
            shutil.rmtree(os.path.join(ckpt_dir, str(step)), ignore_errors=True)

    def _replicated(x):
        """Force an all-gather so the host callback sees the GLOBAL array.

        ``jax.debug.callback`` fires once per device, and what the host
        receives for a *sharded* operand is that device's shard, not the global
        array - a frame would silently contain a half-plane of real data next
        to garbage. An explicit replication constraint makes the all-gather
        happen inside the jit, where it is legal. The planes are small (three
        2048x4096 @ 33.5 MB + one 2048x2048 @ 16.8 MB ~ 117 MB per frame,
        ~7 GB over a 61-frame run), so the gather is not a cost concern.
        """
        if sharding is None:
            return x
        return jax.lax.with_sharding_constraint(
            x, jax.NamedSharding(sharding.mesh, P())
        )

    def snapshot_callable(time, state, registered_variables):
        rho = state[registered_variables.density_index]
        P_gas = state[registered_variables.pressure_index]
        vx = state[registered_variables.velocity_index.x]
        vz = state[registered_variables.velocity_index.z]

        if CHECKPOINT_EVERY:
            jax.debug.callback(_prune_checkpoints)

        # Edge-on (x-z, y=0) and face-on (x-y, z=0) physical slices.
        rho_xz = rho[g:hi, my, g:hi]   # (dim_x, dim_z)
        P_xz = P_gas[g:hi, my, g:hi]
        rho_xy = rho[g:hi, g:hi, mz]   # (dim_x, dim_y)

        n_xz = rho_xz * n_factor
        T_xz = P_xz / rho_xz * T_factor
        n_xy = rho_xy * n_factor

        # Edge-on radial velocity v_r = v . r_hat in km/s (paper Figs 1/2
        # middle panels).
        vr_xz = (
            (vx[g:hi, my, g:hi] * X_plane + vz[g:hi, my, g:hi] * Z_plane)
            / R_plane
            * code_velocity_kms
        )

        # Net vertical mass flux per z-plane, Mdot(z) = sum_xy(rho*v_z)*dx*dy, in
        # Msun/yr. Reduces the interior over (x, y) -> a length-dim_z line; XLA
        # fuses the slice*multiply*reduce so no full 3D temporary is materialised.
        # Already a full reduction over both split axes, but constrained too so
        # every callback operand is explicitly replicated.
        mdot_z = (
            jnp.sum(rho[g:hi, g:hi, g:hi] * vz[g:hi, g:hi, g:hi], axis=(0, 1))
            * (dx * dy)
            * code_mdot_to_msun_per_yr
        )

        jax.debug.callback(
            _save_frame,
            time,
            _replicated(n_xz),
            _replicated(T_xz),
            _replicated(n_xy),
            _replicated(mdot_z),
            _replicated(vr_xz),
        )

    return snapshot_callable


def load_snapshots(out_dir=SNAPSHOTS_DIR):
    """Load the streamed per-frame ``.npz`` files into time-sorted stacked arrays.

    Returns a dict with ``time_myr`` and the stacked ``n_xz``/``T_xz``/``n_xy``/
    ``mdot_z`` arrays (snapshot axis first), or ``None`` if no frames exist. Runs
    in the analysis process (no GPU), so stacking all frames in host RAM is fine.
    """
    files = sorted(glob.glob(os.path.join(_here(out_dir), "frame_*.npz")))
    if not files:
        return None
    frames = [np.load(f) for f in files]
    order = np.argsort([float(fr["time_myr"]) for fr in frames])
    frames = [frames[i] for i in order]
    out = {
        "time_myr": np.array([float(fr["time_myr"]) for fr in frames], dtype=np.float32),
        "n_xz": np.stack([fr["n_xz"] for fr in frames]),
        "T_xz": np.stack([fr["T_xz"] for fr in frames]),
        "n_xy": np.stack([fr["n_xy"] for fr in frames]),
        "mdot_z": np.stack([fr["mdot_z"] for fr in frames]),
    }
    # Frames written since the callable also stores the edge-on radial
    # velocity carry vr_xz; older frame sets simply do not have the key.
    if all("vr_xz" in fr.files for fr in frames):
        out["vr_xz"] = np.stack([fr["vr_xz"] for fr in frames])
    return out


# ---------------------------------------------------------------------------
# Post-simulation analysis
# ---------------------------------------------------------------------------
def analyse_results(final_state, config, registered_variables, initial_state=None):
    """Static-equilibrium check + vertical-velocity diagnostic.

    Wrapped in a function so the recomputed grid and the plot scratch arrays
    are released once the figures are written.

    If `initial_state` is None (e.g. when `time_integration` was called with
    `donate_state=True` and the buffer was donated), the function skips the
    initial-vs-final comparisons and shows final-state-only diagnostics.
    """
    have_initial = initial_state is not None
    
    rprint(jax.devices()[0].memory_stats())
    rprint({k: v.shape for k, v in zip(['rho','P'], (final_state[registered_variables.density_index], 
                                                    final_state[registered_variables.pressure_index]))})

    # ---- Recompute the grid we need for the plots ----
    L_x = config.box_size.x
    L_y = config.box_size.y
    L_z = config.box_size.z

    # Same coordinates get_helper_data would build, without the (N, N, 2N, 3)
    # meshgrid + its moveaxis copy (208 GB each at 2048).
    X_c, Y_c, Z_c = _ic_coordinates(config, None)
    R_cyl = jnp.sqrt(X_c**2 + Y_c**2)

    dim_x, dim_y, dim_z = final_state.shape[1:4]
    mid_x = dim_x // 2
    mid_y = dim_y // 2
    mid_z = dim_z // 2

    code_density_cgs = (code_mass / code_length**3).to(u.g / u.cm**3).value
    m_p_cgs = m_p.to(u.g).value
    v_to_kms = code_velocity.to(u.km / u.s).value

    R_kpc = R_cyl[mid_x:, mid_y, mid_z]
    z_kpc = Z_c[mid_x, mid_y, mid_z:]
    del R_cyl

    # ---- Static-equilibrium check: compare the final state to the initial state ----
    def density_temperature_profiles(state):
        """Return (n_midplane, n_zaxis, T_midplane, T_zaxis) for a primitive state.

        Slice the 1-D midplane / z-axis lines out of rho and P *first*, then derive
        n and T on those lines, so the full-3D T and n arrays are never materialised.
        """
        rho = state[registered_variables.density_index]
        P = state[registered_variables.pressure_index]
        rho_mid, rho_z = rho[mid_x:, mid_y, mid_z], rho[mid_x, mid_y, mid_z:]
        P_mid, P_z = P[mid_x:, mid_y, mid_z], P[mid_x, mid_y, mid_z:]
        n_factor = code_density_cgs / (mu * m_p_cgs)
        return (
            rho_mid * n_factor,
            rho_z * n_factor,
            P_mid / rho_mid * T_factor,
            P_z / rho_z * T_factor,
        )

    n_mid_f, n_z_f, T_mid_f, T_z_f = density_temperature_profiles(final_state)
    jax.block_until_ready((n_mid_f, n_z_f, T_mid_f, T_z_f))

    if have_initial:
        n_mid_i, n_z_i, T_mid_i, T_z_i = density_temperature_profiles(initial_state)
        jax.block_until_ready((n_mid_i, n_z_i, T_mid_i, T_z_i))
        rho_i = initial_state[registered_variables.density_index]
        rho_f = final_state[registered_variables.density_index]
        rel_drift = jnp.abs(rho_f - rho_i) / jnp.maximum(jnp.abs(rho_i), 1e-30)
        rprint(f"Density drift over {END_TIME}: max = {rel_drift.max():.3e}, mean = {rel_drift.mean():.3e}")
        del rho_i, rho_f, rel_drift

    def _style_loglog(ax, xlabel, ylabel, ylim, title):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-2, 1e1)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    if have_initial:
        ax1.plot(R_kpc, n_mid_i, "b-", linewidth=2, label="initial")
    ax1.plot(R_kpc, n_mid_f, "co", markersize=3, markevery=2, label="final")
    _style_loglog(ax1, "R [kpc]", r"n [cm$^{-3}$]", (1e-4, 1e3), "Density - midplane")

    if have_initial:
        ax2.plot(z_kpc, n_z_i, "r-", linewidth=2, label="initial")
    ax2.plot(z_kpc, n_z_f, "mo", markersize=3, markevery=3, label="final")
    _style_loglog(ax2, "z [kpc]", r"n [cm$^{-3}$]", (1e-4, 1e3), "Density - z-axis")

    if have_initial:
        ax3.plot(R_kpc, T_mid_i, "b-", linewidth=2, label="initial")
    ax3.plot(R_kpc, T_mid_f, "co", markersize=3, markevery=2, label="final")
    _style_loglog(ax3, "R [kpc]", "T [K]", (1e3, 1e7), "Temperature - midplane")

    if have_initial:
        ax4.plot(z_kpc, T_z_i, "r-", linewidth=2, label="initial")
    ax4.plot(z_kpc, T_z_f, "mo", markersize=3, markevery=3, label="final")
    _style_loglog(ax4, "z [kpc]", "T [K]", (1e3, 1e7), "Temperature - z-axis")

    suptitle = (
        f"Static check: initial vs final after t = {END_TIME}"
        if have_initial
        else f"Final state after t = {END_TIME}"
    )
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(_fig(f"cgols_static_check{RUN_TAG}.png"), dpi=300)
    plt.close(fig)
    del n_mid_f, n_z_f, T_mid_f, T_z_f
    if have_initial:
        del n_mid_i, n_z_i, T_mid_i, T_z_i

    # ---- Diagnostic: vertical velocity (v_z) ----
    #
    # v_z starts at 0 everywhere. If the disk is out of (discrete) vertical
    # equilibrium it bounces/puffs, which shows up directly as growing |v_z|. The
    # left panel is the xy-plane rms of v_z vs height; the right panel is an x-z
    # slice of the final v_z (blue/red = down/up flows) so vertical outflow off the
    # disk is visible.
    vz_f = final_state[registered_variables.velocity_index.z]
    rprint(f"Final   |v_z| max: {float(jnp.abs(vz_f).max()) * v_to_kms:.3e} km/s")
    rprint(f"Final   |v_z| mean: {float(jnp.abs(vz_f).mean()) * v_to_kms:.3e} km/s")

    vz_rms_f = jnp.sqrt(jnp.mean(vz_f**2, axis=(0, 1))) * v_to_kms
    z_line = Z_c[mid_x, mid_y, :]

    if have_initial:
        vz_i = initial_state[registered_variables.velocity_index.z]
        rprint(f"Initial |v_z| max: {float(jnp.abs(vz_i).max()) * v_to_kms:.3e} km/s")
        vz_rms_i = jnp.sqrt(jnp.mean(vz_i**2, axis=(0, 1))) * v_to_kms

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))

    if have_initial:
        axa.plot(z_line, vz_rms_i, "k-", linewidth=1.5, label="initial")
    axa.plot(z_line, vz_rms_f, "r-", linewidth=1.5, label="final")
    axa.set_xlabel("z [kpc]")
    axa.set_ylabel(r"rms $v_z$ over xy-plane [km s$^{-1}$]")
    axa.set_title("Vertical velocity growth")
    axa.legend()

    vz_slice = np.asarray(vz_f[:, mid_y, :].T) * v_to_kms  # (dim_z, dim_x)
    extent = [
        float(X_c[0, mid_y, mid_z]),
        float(X_c[-1, mid_y, mid_z]),
        float(Z_c[mid_x, mid_y, 0]),
        float(Z_c[mid_x, mid_y, -1]),
    ]
    vmax = float(np.abs(vz_slice).max()) or 1.0
    im = axb.imshow(
        vz_slice, origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    axb.set_xlabel("x [kpc]")
    axb.set_ylabel("z [kpc]")
    axb.set_title("Final $v_z$ (x-z slice, y=0)")
    plt.colorbar(im, ax=axb, label=r"$v_z$ [km s$^{-1}$]")

    plt.tight_layout()
    plt.savefig(_fig(f"cgols_vz_diagnostic{RUN_TAG}.png"), dpi=300)
    plt.close(fig)

    # ---- Final-state morphology, phase diagram, vertical mass flux ----
    #
    # Four panels:
    #   (A) Edge-on (x-z) log density - shows disk puffing, fountains, halo asymmetries.
    #   (B) Face-on (x-y) log density - checks the disk stays axisymmetric (grid-aligned
    #       m=4 patterns would show here).
    #   (C) Mass-weighted log n - log T phase diagram - shows whether the disk and
    #       halo still occupy their two intended loci or have collapsed onto a single
    #       phase.
    #   (D) Net vertical mass flux Mdot(z) = sum_xy(rho * v_z) * dx * dy, in M_sun/yr -
    #       positive = upward mass transport. Quantifies any outflow / fountain.
    from matplotlib.colors import LogNorm

    rho_f = final_state[registered_variables.density_index]
    P_f = final_state[registered_variables.pressure_index]
    T_f = P_f / rho_f * T_factor
    n_f = rho_f * code_density_cgs / (mu * m_p_cgs)

    dx = L_x / dim_x
    dy = L_y / dim_y
    dz_cell = L_z / dim_z
    cell_volume = dx * dy * dz_cell  # code length^3
    code_mass_to_msun = code_mass.to(u.M_sun).value
    code_mdot_to_msun_per_yr = (code_mass / code_units.code_time).to(u.M_sun / u.yr).value

    rho_xz = np.asarray(rho_f[:, mid_y, :].T)  # (dim_z, dim_x)
    rho_xy = np.asarray(rho_f[:, :, mid_z].T)  # (dim_y, dim_x)
    extent_xz = [
        float(X_c[0, mid_y, mid_z]),
        float(X_c[-1, mid_y, mid_z]),
        float(Z_c[mid_x, mid_y, 0]),
        float(Z_c[mid_x, mid_y, -1]),
    ]
    extent_xy = [
        float(X_c[0, mid_y, mid_z]),
        float(X_c[-1, mid_y, mid_z]),
        float(Y_c[mid_x, 0, mid_z]),
        float(Y_c[mid_x, -1, mid_z]),
    ]

    # Mass-weighted phase histogram in solar masses per bin. Flattening 512*512*1024
    # cells is ~1 GB per array in float32; bump `stride` to 2 if RAM is tight.
    stride = 1
    sl = (slice(None, None, stride),) * 3
    cell_mass_msun = np.asarray(rho_f[sl]).ravel() * cell_volume * code_mass_to_msun * stride**3
    log_n_arr = np.log10(np.maximum(np.asarray(n_f[sl]).ravel(), 1e-30))
    log_T_arr = np.log10(np.maximum(np.asarray(T_f[sl]).ravel(), 1e-30))
    h, x_edges, y_edges = np.histogram2d(
        log_n_arr, log_T_arr,
        bins=(120, 120),
        range=[[-6, 4], [2, 8]],
        weights=cell_mass_msun,
    )
    del log_n_arr, log_T_arr, cell_mass_msun
    h_plot = np.where(h > 0, h, np.nan)

    mdot_z_f = np.asarray(jnp.sum(rho_f * vz_f, axis=(0, 1))) * float(dx * dy) * code_mdot_to_msun_per_yr
    mdot_z_i = None
    if have_initial:
        rho_i = initial_state[registered_variables.density_index]
        mdot_z_i = np.asarray(jnp.sum(rho_i * vz_i, axis=(0, 1))) * float(dx * dy) * code_mdot_to_msun_per_yr
        del rho_i
    z_full = np.asarray(Z_c[mid_x, mid_y, :])

    rho_max = float(rho_f.max())
    rho_floor = max(float(rho_f.min()), rho_max * 1e-6)
    del rho_f, P_f, T_f, n_f

    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(13, 11))

    im_a = axA.imshow(
        rho_xz, origin="lower", extent=extent_xz, aspect="auto",
        cmap="magma", norm=LogNorm(vmin=rho_floor, vmax=rho_max),
    )
    axA.set_xlabel("x [kpc]")
    axA.set_ylabel("z [kpc]")
    axA.set_title(r"Final $\rho$ - edge-on (x-z, y=0)")
    plt.colorbar(im_a, ax=axA, label=r"$\rho$ [code units]")

    im_b = axB.imshow(
        rho_xy, origin="lower", extent=extent_xy, aspect="equal",
        cmap="magma", norm=LogNorm(vmin=rho_floor, vmax=rho_max),
    )
    axB.set_xlabel("x [kpc]")
    axB.set_ylabel("y [kpc]")
    axB.set_title(r"Final $\rho$ - face-on (x-y, z=0)")
    plt.colorbar(im_b, ax=axB, label=r"$\rho$ [code units]")

    im_c = axC.imshow(
        h_plot.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(),
    )
    axC.set_xlabel(r"$\log_{10} n$ [cm$^{-3}$]")
    axC.set_ylabel(r"$\log_{10} T$ [K]")
    axC.set_title("Mass-weighted phase diagram (final)")
    plt.colorbar(im_c, ax=axC, label=r"M [$M_\odot$ / bin]")

    axD.axhline(0, color="0.6", linewidth=0.8)
    if mdot_z_i is not None:
        axD.plot(z_full, mdot_z_i, "k-", linewidth=1.0, label="initial")
    axD.plot(z_full, mdot_z_f, "r-", linewidth=1.5, label="final")
    axD.set_xlabel("z [kpc]")
    axD.set_ylabel(r"net $\dot M(z)$ [$M_\odot$ yr$^{-1}$]")
    axD.set_title("Net vertical mass flux through z-planes")
    axD.legend()

    plt.tight_layout()
    plt.savefig(_fig(f"cgols_extras{RUN_TAG}.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Intermediate-snapshot analysis: animation + wind time-series
#
# These read the host-offloaded cgols_snapshots.npz (written by save_snapshots
# during the run) and need only matplotlib + numpy - no GPU. Call them from
# cgols_analyse.py alongside analyse_results.
# ---------------------------------------------------------------------------
def _snapshot_grid(config):
    """Centered cell-center coordinate lines (kpc) and imshow extents from config.

    code_length is 1 kpc, so code-unit box sizes are already kpc.
    """
    L_x, L_y, L_z = config.box_size.x, config.box_size.y, config.box_size.z
    dim_x, dim_y, dim_z = config.num_cells.x, config.num_cells.y, config.num_cells.z
    x = (np.arange(dim_x) + 0.5) * (L_x / dim_x) - L_x / 2
    y = (np.arange(dim_y) + 0.5) * (L_y / dim_y) - L_y / 2
    z = (np.arange(dim_z) + 0.5) * (L_z / dim_z) - L_z / 2
    extent_xz = [x[0], x[-1], z[0], z[-1]]
    extent_xy = [x[0], x[-1], y[0], y[-1]]
    return x, y, z, extent_xz, extent_xy


def animate_wind_snapshots(config, snapshots_dir=SNAPSHOTS_DIR, out=f"cgols_wind_animation{RUN_TAG}.gif", fps=12):
    """Animate the intermediate snapshots: edge-on n, edge-on T, face-on n.

    Builds a GIF with matplotlib's FuncAnimation + PillowWriter (the codebase's
    colab tutorial uses imageio, which is not installed here; this produces the
    same result with no extra dependency). Color norms are fixed across frames so
    the animation does not flicker.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LogNorm

    snapshots_dir, out = _here(snapshots_dir), _fig(out)
    data = load_snapshots(snapshots_dir)
    if data is None:
        rprint(f"No frames in {snapshots_dir}/ - run cgols.py first to produce snapshots.")
        return

    t = data["time_myr"]
    n_xz, T_xz, n_xy = data["n_xz"], data["T_xz"], data["n_xy"]
    n_frames = len(t)

    _, _, _, extent_xz, extent_xy = _snapshot_grid(config)
    r_inj = (INJ_RAD * u.pc).to(code_length).value  # injection-region radius, kpc

    # Fixed color ranges (≈6 decades, adapted to the data) so frames are comparable.
    n_max = float(np.nanmax(n_xz))
    n_norm = LogNorm(vmin=max(n_max * 1e-6, 1e-6), vmax=n_max)
    T_max = float(np.nanmax(T_xz))
    T_norm = LogNorm(vmin=max(1e3, T_max * 1e-5), vmax=T_max)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 5.5))

    im_a = axA.imshow(n_xz[0].T, origin="lower", extent=extent_xz, aspect="auto",
                      cmap="magma", norm=n_norm)
    axA.set_xlabel("x [kpc]"); axA.set_ylabel("z [kpc]")
    axA.set_title(r"$n$ - edge-on (x-z, y=0)")
    plt.colorbar(im_a, ax=axA, label=r"n [cm$^{-3}$]")

    im_b = axB.imshow(T_xz[0].T, origin="lower", extent=extent_xz, aspect="auto",
                      cmap="inferno", norm=T_norm)
    axB.set_xlabel("x [kpc]"); axB.set_ylabel("z [kpc]")
    axB.set_title("T - edge-on (x-z, y=0)")
    plt.colorbar(im_b, ax=axB, label="T [K]")

    im_c = axC.imshow(n_xy[0].T, origin="lower", extent=extent_xy, aspect="equal",
                      cmap="magma", norm=n_norm)
    axC.set_xlabel("x [kpc]"); axC.set_ylabel("y [kpc]")
    axC.set_title(r"$n$ - face-on (x-y, z=0)")
    plt.colorbar(im_c, ax=axC, label=r"n [cm$^{-3}$]")

    # Mark the central wind-injection region on the spatial panels.
    for ax in (axA, axC):
        ax.add_patch(plt.Circle((0, 0), r_inj, fill=False, color="cyan", lw=0.8, ls="--"))

    suptitle = fig.suptitle(f"CGOLS wind - t = {t[0]:6.1f} Myr")
    plt.tight_layout()

    def _update(i):
        im_a.set_data(n_xz[i].T)
        im_b.set_data(T_xz[i].T)
        im_c.set_data(n_xy[i].T)
        suptitle.set_text(f"CGOLS wind - t = {t[i]:6.1f} Myr")
        return im_a, im_b, im_c, suptitle

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    rprint(f"Wrote {out} ({n_frames} frames, t = {t[0]:.1f} -> {t[-1]:.1f} Myr)")


def plot_wind_timeseries(config, snapshots_dir=SNAPSHOTS_DIR, out=f"cgols_wind_timeseries{RUN_TAG}.png"):
    """Quantitative wind diagnostics from the 1D vertical mass-flux snapshots.

    Left: net mass OUTflow rate through z = +/-H planes vs time (outward = +z above
    the disk, -z below, so total = Mdot(+H) - Mdot(-H)), for two fiducial heights.
    This is the quantity that sets the mass-loading factor eta = Mdot_out / SFR.

    Right: a (z, t) kymograph of the net vertical mass flux, which shows the wind
    front propagating away from the disk and the bipolar (anti)symmetry of the flow.
    """
    from matplotlib.colors import TwoSlopeNorm

    snapshots_dir, out = _here(snapshots_dir), _fig(out)
    data = load_snapshots(snapshots_dir)
    if data is None:
        rprint(f"No frames in {snapshots_dir}/ - run cgols.py first to produce snapshots.")
        return

    t = data["time_myr"]
    mdot_z = data["mdot_z"]  # (n_frames, dim_z), Msun/yr

    _, _, z, _, _ = _snapshot_grid(config)

    def outflow_through(H):
        """Total outward mass flux through the +/-H planes vs time, Msun/yr."""
        ip = int(np.argmin(np.abs(z - H)))
        im = int(np.argmin(np.abs(z + H)))
        return mdot_z[:, ip] - mdot_z[:, im]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    for H, style in ((5.0, "b-"), (8.0, "r--")):
        axL.plot(t, outflow_through(H), style, lw=1.5, label=f"|z| = {H:.0f} kpc")
    axL.axhline(0, color="0.6", lw=0.8)
    axL.set_xlabel("t [Myr]")
    axL.set_ylabel(r"net outflow rate $\dot M_{\rm out}$ [$M_\odot$ yr$^{-1}$]")
    axL.set_title("Mass outflow rate vs time")
    axL.legend()

    vmax = float(np.nanmax(np.abs(mdot_z))) or 1.0
    im = axR.imshow(
        mdot_z.T, origin="lower", aspect="auto",
        extent=[t[0], t[-1], z[0], z[-1]],
        cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
    )
    axR.set_xlabel("t [Myr]")
    axR.set_ylabel("z [kpc]")
    axR.set_title(r"net vertical mass flux $\dot M(z, t)$")
    plt.colorbar(im, ax=axR, label=r"$\dot M$ [$M_\odot$ yr$^{-1}$]")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    peak = float(np.nanmax(np.abs(outflow_through(5.0))))
    rprint(f"Wrote {out} (peak |outflow| through |z|=5 kpc: {peak:.2f} Msun/yr)")


def plot_paper_slices(
    config,
    snapshots_dir=SNAPSHOTS_DIR,
    target_times_myr=(10.0, 25.0, 50.0, 60.0),
    out=f"cgols_paper_slices{RUN_TAG}.png",
    n_range=(1e-4, 1e3),
    T_range=(1e3, 10 ** 7.5),
    x_half=5.0,
    z_half=10.0,
    scalebar_kpc=1.0,
):
    """Replicate the paper's x-z density & temperature slices at fixed times.

    Schneider & Robertson 2018 (arXiv:1803.01008) show edge-on (x-z) hydrogen
    number-density and temperature slices of the central wind at a sequence of
    times. This builds the same layout from the streamed snapshots: a 2-row (n_H on
    top, T on bottom) by N-column (one per requested time) grid.

    The snapshots store the number density n = rho/(mu m_p), which is exactly
    the quantity the paper maps: despite the "n_h" colorbar label, Schneider &
    Robertson state "when converting between mass density rho and number density
    n, we take mu = 0.6 throughout" (their quoted rho_0,h = 3e3 Msun/kpc^3 <->
    n ~ 1e-3.5 cm^-3 confirms it). No hydrogen-fraction factor is applied - an
    earlier X_H * mu = 0.44 conversion here made every density panel 0.35 dex
    darker than the paper's. T is already in K.

    Defaults match the paper's colorbars exactly: log10(n_H [cm^-3]) in [-4, 3] and
    log10(T [K]) in [3.0, 7.5]. Pass ``n_range`` / ``T_range`` = (vmin, vmax) in
    linear units to override.

    For each target time the NEAREST available snapshot is used. Because this run is
    capped at END_TIME (~45 Myr), targets beyond the end (50, 60 Myr) fall back to
    the last available frame; that column's title flags the substitution and shows
    the actual frame time, so the comparison stays honest.
    """
    from matplotlib.colors import LogNorm

    snapshots_dir, out = _here(snapshots_dir), _fig(out)
    data = load_snapshots(snapshots_dir)
    if data is None:
        rprint(f"No frames in {snapshots_dir}/ - run cgols.py first to produce snapshots.")
        return

    t = data["time_myr"]
    n_xz = data["n_xz"]  # already n = rho/(mu m_p), the paper's plotted quantity
    T_xz = data["T_xz"]
    _, _, _, extent_xz, _ = _snapshot_grid(config)
    t_max = float(t[-1])

    # Nearest available frame per requested time.
    idxs = [int(np.argmin(np.abs(t - tt))) for tt in target_times_myr]

    # Shared per-row color scales (paper colorbars by default; adapt if None).
    if n_range is None:
        n_max = max(float(np.nanmax(n_xz[i])) for i in idxs)
        n_norm = LogNorm(vmin=max(n_max * 1e-7, 1e-7), vmax=n_max)
    else:
        n_norm = LogNorm(vmin=n_range[0], vmax=n_range[1])
    if T_range is None:
        T_max = max(float(np.nanmax(T_xz[i])) for i in idxs)
        T_norm = LogNorm(vmin=max(1e3, T_max * 1e-5), vmax=T_max)
    else:
        T_norm = LogNorm(vmin=T_range[0], vmax=T_range[1])

    from matplotlib.ticker import MultipleLocator

    ncols = len(target_times_myr)
    # Frame the FULL domain to match Schneider & Robertson 2018: their Fig. 6 panels
    # span the whole 10 x 20 kpc box (x in [-5, 5], z in [-10, 10]), exactly our
    # box_size, so with x_half=5 / z_half=10 the frame is 1:1 with the paper -- no
    # crop. Panels come out 1:2 (x:z), the paper's aspect. Ticks are drawn like the
    # paper: unlabelled inward marks, with a scale bar carrying the physical scale
    # instead of numeric axis labels.
    fig, axes = plt.subplots(2, ncols, figsize=(2.7 * ncols, 9.5), squeeze=False,
                             constrained_layout=True)

    def _annotate(ax, label):
        """Paper-style overlays: time text (top-left) + scale bar (top-right).

        Both are placed in axes-fraction coordinates so they sit consistently
        regardless of the crop, and share one height so they read as a single
        row. The scale-bar length is converted to a fraction from the physical
        x-range (2*x_half kpc across the panel); the "1 kpc" label sits to the
        right of the bar, matched in size to the time label.
        """
        y = 0.95
        ax.text(0.09, y, label, transform=ax.transAxes, color="white",
                fontsize=8.5, ha="left", va="center")
        bar_frac = scalebar_kpc / (2 * x_half)  # 1 kpc as a fraction of panel width
        x0 = 0.695  # scale-bar cluster nudged ~0.75 kpc right of its previous 0.62
        x1 = x0 + bar_frac  # bar sits to the LEFT of its "1 kpc" label
        ax.plot([x0, x1], [y, y], "-", color="white", lw=0.8, transform=ax.transAxes)
        ax.text(x1 + 0.02, y, f"{scalebar_kpc:.0f} kpc", transform=ax.transAxes,
                color="white", fontsize=8.5, ha="left", va="center")

    for col, (tt, i) in enumerate(zip(target_times_myr, idxs)):
        axn, axT = axes[0, col], axes[1, col]
        im_n = axn.imshow(n_xz[i].T, origin="lower", extent=extent_xz, aspect="equal",
                          cmap="viridis", norm=n_norm)
        im_T = axT.imshow(T_xz[i].T, origin="lower", extent=extent_xz, aspect="equal",
                          cmap="inferno", norm=T_norm)

        # Paper-style in-panel time label (actual frame time). If the requested
        # target is past the (capped) run, note it on a smaller second line so the
        # comparison stays honest.
        label = f"{t[i]:.0f} Myr"
        if tt > t_max + 1e-6:
            label += f"\n(req. {tt:.0f})"

        for ax in (axn, axT):
            ax.set_xlim(-x_half, x_half)
            ax.set_ylim(-z_half, z_half)
            # Paper look: unlabelled inward tick marks on all four edges, no titles.
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.yaxis.set_major_locator(MultipleLocator(1.0))
            ax.tick_params(which="both", direction="in", length=3, color="black",
                           top=True, right=True, labelbottom=False, labelleft=False)
            _annotate(ax, label)

    fig.colorbar(im_n, ax=axes[0, :].tolist(), label=r"$n$ [cm$^{-3}$]", shrink=0.85)
    fig.colorbar(im_T, ax=axes[1, :].tolist(), label="T [K]", shrink=0.85)
    fig.suptitle("CGOLS wind - x-z slices (cf. Schneider & Robertson 2018)")
    plt.savefig(out, dpi=200)
    plt.close(fig)
    used = ", ".join(f"{tt:.0f}->{t[i]:.1f}" for tt, i in zip(target_times_myr, idxs))
    rprint(f"Wrote {out} (requested->frame Myr: {used})")


# ---------------------------------------------------------------------------
# Reference implementation: the radiative cooling curve
#
# By default (CGOLS_COOLING=0) the run above reproduces the paper's ADIABATIC
# A-series, which uses no cooling; the paper reports those initial conditions are
# stable for >1 Gyr AT THEIR RESOLUTION (dx ~ 5 pc, i.e. ~30 cells per 0.15 kpc
# disk scale height). At the default 512^2x1024 (dx ~ 19.5 pc) the central gas
# scale height is ~1 cell, which is why the ICs are smoothed
# (CGOLS_SMOOTHING_SIGMA) and the central disk sits slightly warmer/puffier than
# Fig. 4 - a resolution issue, not a missing-cooling one.
#
# Cooling is used in the radiative B/C series (companion paper arXiv:1803.01005),
# which CGOLS_COOLING=1 activates: astronomix's CIE_PARABOLIC curve, applied
# operator-split and sub-cycled after each hydro step with a 10^4 K floor and
# n^2 Lambda(T) at mu = 0.6 (see build_config / _cooling_params_kwargs above).
#
# The function below is the exact analytic rendering of the published fit
# (Appendix A.3, Eq. A4 of CGOLS I) in plain Kelvin/cgs. It is the REFERENCE
# IMPLEMENTATION the wired-in CIE_PARABOLIC curve is validated against - see
# pytests/cooling/cie_cooling.py, which imports it as ground truth - not dead
# code, and not what the run itself calls.
#
# Interactions with the positivity clips, for reading the B-series diagnostics:
#   - temperature_clip (CGOLS_TMAX_K, 5e9 K) never conflicts with cooling, which
#     only lowers T. Fidelity note: paper-era Cholla caps T at 1e9 K INSIDE its
#     cooling kernel (unmentioned in either paper). Our 5e9 was chosen to fix the
#     adiabatic per-stage-clipping runaway; CGOLS_TMAX_K=1e9 is a legitimate
#     maximally-Cholla-faithful experiment. With cooling on the hottest gas
#     self-limits, so the difference should be small - the default is unchanged.
#   - minimum_pressure = 1e-5 is a hard per-step/per-stage floor. The 10^4 K
#     cooling floor is P = 1.4e-2 * rho, so for rho >~ 7e-4 the cooling floor
#     binds first; in the most rarefied cavity cells (rho -> 1e-4) the pressure
#     floor is the hotter of the two and parks them near ~7e4 K. Harmless, but
#     do not mistake it for a cooling bug.
#   - The disk starts at 10^4 K, i.e. already at the floor, so cooling is a no-op
#     there by construction. The B-series signal appears at the wind/shell
#     contact and in the shocked shell.
#   - Expect a smaller dt: the 10% thermal-energy constraint
#     (CGOLS_COOLING_DT_FRACTION) can beat CFL at the dense hot shell.
# ---------------------------------------------------------------------------
def cooling_lambda_cgs(T):
    """CIE cooling function Lambda(T) [erg s^-1 cm^3], Schneider & Robertson 2018 Eq. A4.

    T in Kelvin. Returns 0 below the 10^4 K temperature floor.
    """
    logT = jnp.log10(T)
    lam = jnp.where(
        logT < 5.9,
        10.0 ** (-1.3 * (logT - 5.25) ** 2 - 21.25),
        jnp.where(
            logT < 7.4,
            10.0 ** (0.7 * (logT - 7.1) ** 2 - 22.8),
            10.0 ** (0.45 * logT - 26.065),
        ),
    )
    return jnp.where(T < 1e4, 0.0, lam)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    if CREATE_IC:
        initial_state, config, params, registered_variables = build_initial_conditions(
            save_potential=True
        )

        # Save BEFORE integration: donate_state=True consumes the initial buffer in-place,
        # so this device->host copy must happen while the buffer is still valid.
        if RANK0:
            jnp.save(_ic_state_path(), initial_state)
        barrier("cgols:ic-state-saved")
    else:
        # Per-step diagnostics history (the in-place [diag] status line keeps no
        # scrollback): default to cgols_logs/cgols_diag<RUN_TAG>.log, fresh per
        # run. Set ASTRONOMIX_DIAG_LOG to override the path, or to the empty
        # string to disable logging. The env var is set on every rank (only rank
        # 0 ever writes the file - see _progress_bar), but only rank 0 truncates
        # it, or 32 ranks race to unlink the file the others just opened.
        _default_diag_log = _log(f"cgols_diag{RUN_TAG}.log")
        if os.environ.setdefault("ASTRONOMIX_DIAG_LOG", _default_diag_log) == _default_diag_log:
            if RANK0 and os.path.exists(_default_diag_log):
                os.remove(_default_diag_log)
        barrier("cgols:diag-log-truncated")

        # ---- Multi-GPU domain decomposition ----
        # When SHARD_SPLIT asks for more than one GPU, distribute the state over a
        # (var, x, y, z) device mesh and hand the sharding to time_integration, which
        # shards its helper data the same way and runs the inter-device halo exchange
        # itself. Only the initial state has to be device_put onto the sharding here.
        # SHARD_SPLIT == (1, 1, 1, 1) keeps the original single-GPU path (sharding=None).
        # Built BEFORE the IC and the restart below so both the in-situ build and a
        # checkpoint restore land directly on the mesh (no single-device staging).
        # jax.make_mesh uses the GLOBAL device list, so this is unchanged
        # multi-process.
        if NUM_GPUS > 1:
            # axis_types=Auto is required: jax.make_mesh defaults to Explicit sharding
            # mode on jax >= 0.10, under which the ghost-cell padding inside the solver
            # (jnp.pad(mode="edge"), see time_stepping/_utils._pad) cannot resolve an
            # output sharding on a sharded axis and raises a ShardingTypeError. Auto mode
            # restores implicit GSPMD (collectives auto-inserted for pad/slice), which is
            # what the solver's with_sharding_constraint / shard_map halo paths assume.
            mesh = jax.make_mesh(
                SHARD_SPLIT, (VARAXIS, XAXIS, YAXIS, ZAXIS),
                axis_types=(AxisType.Auto,) * 4,
            )
            sharding = jax.NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))
        else:
            sharding = None

        # ---- Initial conditions ----
        # CGOLS_IC_MODE=file: load the .npy pair built by a CGOLS_CREATE_IC run.
        # CGOLS_IC_MODE=insitu: build them here, inside one jit with
        #   out_shardings, so no host ever holds the global 344 GB state.
        # A restart leg needs only the external potential - the state comes from
        # the checkpoint - so at 2048 skip the state build entirely rather than
        # constructing 344 GB and immediately discarding it.
        if not INSITU_IC:
            initial_state, config, params, registered_variables = (
                load_initial_conditions()
            )
        elif RESTART_FROM:
            config, registered_variables = build_config()
            params = build_simulation_params(
                build_external_potential(config, sharding)
            )
            _expected_shape = (
                registered_variables.num_vars,
                RESOLUTION,
                RESOLUTION,
                2 * RESOLUTION,
            )
            config = finalize_config(config, _expected_shape)
            initial_state = None
            rprint(
                "In-situ IC skipped on a restart leg: only the external "
                "potential was built (the state comes from the checkpoint)."
            )
        else:
            initial_state, config, params, registered_variables = (
                build_initial_conditions(sharding=sharding)
            )

        # Resume from an Orbax checkpoint (CGOLS_RESTART_FROM=latest for this
        # run-tag's checkpoint dir, or an explicit dir; CGOLS_RESTART_STEP picks
        # a step, default newest): replace the t=0 initial state with the
        # restored carry and start the clock at its time. The wind schedule and
        # the snapshot grid both use absolute time, so the resumed window
        # continues exactly where the checkpointing run left off - only the
        # external potential and the other params are still taken from the IC
        # files.
        restart_state = None
        if RESTART_FROM:
            _ckpt_dir = (
                _data(f"cgols_checkpoints{RUN_TAG}")
                if RESTART_FROM == "latest"
                else (RESTART_FROM if os.path.isabs(RESTART_FROM) else _here(RESTART_FROM))
            )
            _restored, params, restart_state = restart_from_latest_checkpoint(
                _ckpt_dir,
                params,
                step=(int(RESTART_STEP) if RESTART_STEP else None),
                sharding=sharding,
            )
            _want_shape = (
                _expected_shape if initial_state is None else initial_state.shape
            )
            if _restored.shape != tuple(_want_shape):
                raise ValueError(
                    f"checkpoint state {_restored.shape} does not match the "
                    f"configured grid {tuple(_want_shape)} - check CGOLS_DIM / CGOLS_IC_TAG"
                )
            initial_state = _restored
            _t0 = float(params.t_start)
            _t0_myr = (_t0 * code_units.code_time).to(u.Myr).value
            rprint(f"Restarting from {_ckpt_dir} at t = {_t0:.6f} code ({_t0_myr:.2f} Myr)")

        if NUM_GPUS > 1:
            # No-op data-movement-wise when the state was already restored onto
            # the sharding above.
            initial_state = jax.device_put(initial_state, sharding)
            # Report the layout without indexing into the array: integer-indexing a
            # sharded axis (e.g. state[0, :, :, 0] when z is split) is a gather that
            # JAX cannot assign an output sharding to, so it raises. Printing the
            # sharding spec + per-device shard shape is safe for any SHARD_SPLIT.
            rprint(f"Sharding {initial_state.shape} state over {NUM_GPUS} GPUs, split {SHARD_SPLIT}")
            rprint(f"  sharding:    {initial_state.sharding}")
            rprint(f"  shard shape: {initial_state.addressable_shards[0].data.shape}")

        # Stream intermediate snapshots (2D slices + 1D vertical-flux profile) to disk
        # during the run, for the animation and the wind time-series. Each frame is
        # written immediately inside the callback (see make_snapshot_callable), so no
        # host RAM accumulates and the frames survive a blow-up. config is already
        # finalized here (build/load both call finalize_config), so num_ghost_cells is
        # set for the in-callback slicing.
        # In benchmark mode the snapshot callback is off (config.activate_snapshot_callback
        # is False), so skip building the callable and skip saving the garbage
        # fixed-dt final state; we only care about the timing/memory printout.
        snapshot_callable = (
            None
            if BENCH_STEPS
            else make_snapshot_callable(
                config, registered_variables, sharding=sharding
            )
        )

        # Wall-clock timing of the full run (includes JAX compile time). JAX
        # dispatches asynchronously, so block_until_ready forces the device to
        # finish before we stop the timer - otherwise we'd only time dispatch.
        # Pure host-side timing: no extra device memory.
        _t0 = time.perf_counter()
        final_state = time_integration(
            initial_state, config, params, registered_variables, snapshot_callable,
            sharding=sharding, restart_state=restart_state,
        )
        jax.block_until_ready(final_state)
        _elapsed = time.perf_counter() - _t0
        rprint(
            f"time_integration wall time: {_elapsed:.1f} s "
            f"({_elapsed / 60:.1f} min, {_elapsed / 3600:.2f} h) "
            f"on {NUM_GPUS} GPU(s), split {SHARD_SPLIT}"
        )
        # Dumping the final state gathers the WHOLE state to every process's
        # host: 5 GB at 512, 43 GB at 1024, 344 GB at 2048 - times one copy per
        # rank. From 2048 on the Orbax checkpoint (already written sharded, one
        # shard per device) is the final artefact and this dump is skipped.
        if not BENCH_STEPS and RESOLUTION <= 1024:
            if RANK0:
                jnp.save(_data_final(f"cgols_final_state{RUN_TAG}.npy"), final_state)
            barrier("cgols:final-state-saved")
        elif not BENCH_STEPS:
            rprint(
                f"Final .npy dump skipped at CGOLS_DIM={RESOLUTION} "
                f"(the state is {final_state.size * 4 / 1024**3:.0f} GB); the "
                "final artefact is the newest Orbax checkpoint in "
                f"{config.snapshot_storage_path or '<checkpointing DISABLED - '
                                                   'set CGOLS_CHECKPOINT_EVERY>'}."
            )

        # Analysis runs in a separate process (cgols_analyse.py) on a fresh, empty GPU.
        # Doing it here would OOM: XLA still holds the sim's ~32 GB pool.
        # print(
        #     "Saved cgols_initial_state.npy / cgols_final_state.npy. "
        #     "Run `python cgols_analyse.py` to produce the figures."
        # )