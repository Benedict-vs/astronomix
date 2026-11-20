"""Solver steps of every cell of a CGOLS chemistry run: a health check of the reaction.

Reads the state of a run of ``cgols.py`` with ``CGOLS_CHEMISTRY=1`` (an Orbax
checkpoint, or the same array saved as ``.npy``) and reacts every cell once over
a hydro step with the per-cell kernel, which also returns the trial steps of
every cell (``solver.with_steps``). Nothing is written back; the state is only
read. It answers two questions:

* What does a reaction cost, and in which gas? A cell costs its trial steps, and
  on a GPU a group of 32 cells runs as long as its slowest cell.
* Does every cell reach the end of its step? A cell that does not (a negative
  count: out of ``max_steps``, or stalled) comes back with its last accepted
  state, so it has not reacted or cooled as it should, and a cell that runs to
  ``max_steps`` holds up every launch for that long. In the first 75 Myr attempt
  (2026-11-19) 1838 cells of 33.5 million did that from about 11 Myr on, with a
  Jacobian that was not finite in gas without H and H2 (fixed since); the run
  was seven times slower and wrong in those cells, and nothing but the step
  time showed it. Run this on a checkpoint whenever the step time of a run
  changes without a reason.

Usage (from this directory; needs the chemistry environment, see
``run_chemistry_local.sh``, and the network file and KROME data of the run)::

    python cgols_chemistry_steps.py <checkpoint dir>/<step> --step-kyr 9.5
    python cgols_chemistry_steps.py state.npy --step-kyr 9.5 --frame cgols_snapshots_chem256/frame_0009.npz
    python cgols_chemistry_steps.py state.npy --step-kyr 9.5 --cpu --stride 4      # every 4th cell per axis, on the CPU

A run with a radiation field per cell (``CGOLS_CHEM_FUV=1``) is checked with
one field for all cells, ``--radiation-scale <Draine fields>``, and the network
file of that run (``--network``); the field of the cells is in the frames
(``fuv_xz``), so run it for the range found there.

``--step-kyr`` is the hydro step of the run at that time (the difference of two
``t=`` values of the diag log times 9778 kyr). The layout of the state is the one
of ``cgols.py`` in three dimensions: density first, pressure fifth, the species
of the network last. The units are the code units of ``cgols.py`` (kpc, 1e6
solar masses, 100 km/s); with ``--frame`` (a frame of the same time) they are
checked against the plane that the frame stores.

A full 256 x 256 x 512 state takes about a minute on one H200 after the kernel
is compiled (two to ten minutes, once per code version). Reading an Orbax
checkpoint or a multi-GB ``.npy`` from the network scratch can take ten minutes
when the machine is busy.
"""

import argparse
import os
import time

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("state", help="an Orbax checkpoint directory of the run (.../<step>) or the state as .npy")
parser.add_argument("--step-kyr", type=float, required=True, help="the hydro step over which every cell is reacted [kyr]")
parser.add_argument("--frame", default=None, help="a frame of the same time, to check the units against")
parser.add_argument("--stride", type=int, default=1, help="take every n-th cell along every axis (a quick look)")
parser.add_argument("--cpu", action="store_true", help="run on the CPU (default: one free GPU through autocvd)")
parser.add_argument("--out", default=None, help="write the steps and the gas state of every cell to this .npz")
parser.add_argument("--network", default=os.environ.get("CGOLS_CHEM_NETWORK", "/export/scratch/bschuber/krome_networks/react_galaxy_ism_plus_cr"))
parser.add_argument("--krome-data", default=os.environ.get("CGOLS_CHEM_KROME_DATA", "/export/scratch/lbranca/krome/data"))
parser.add_argument("--rtol", type=float, default=1.0e-4, help="as the run (CGOLS_CHEM_RTOL)")
parser.add_argument("--atol", type=float, default=1.0e-12, help="as the run (CGOLS_CHEM_ATOL)")
parser.add_argument("--no-free-free", action="store_true", help="as a run with CGOLS_CHEM_FREE_FREE=0")
parser.add_argument("--first-step", type=float, default=1.0, help="as the run (CGOLS_CHEM_FIRST_STEP)")
parser.add_argument("--radiation-scale", type=float, default=None,
                    help="a run with a radiation field per cell (CGOLS_CHEM_FUV=1): react every cell in this many Draine fields "
                         "(the field of a cell is in the frames as fuv_xz; try the smallest and the largest value)")
arguments = parser.parse_args()

if arguments.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    from autocvd import autocvd

    autocvd(num_gpus=1)
    # the kernel allocates outside the XLA pool
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ruff: noqa: E402
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import cgols_chemistry

# The code units of cgols.py: the density unit is 1e6 solar masses per kpc^3 and
# the number densities of the species are in units of it over the proton mass;
# the temperature of a cell is its pressure over its particles in units of
# (100 km/s)^2 times the proton mass over the Boltzmann constant.
NUMBER_DENSITY_UNIT_CGS = 4.04629e-2
TEMPERATURE_UNIT_KELVIN = 1.21148e6
PRESSURE_INDEX = 4
SECONDS_PER_KYR = 3.15576e10
HYDROGEN_ATOMS = {"H": 1, "H+": 1, "H-": 1, "H2": 2, "H2+": 2}


def read_state(path):
    """The primitive state of the run, shape ``(variables, x, y, z)``."""
    if path.endswith(".npy"):
        # row by row through a memory map: a single read of a multi-GB file can stall on a network file system
        mapped = np.load(path, mmap_mode="r")
        return np.stack([np.ascontiguousarray(row) for row in mapped])
    from astronomix._snapshotting._orbax_storage import load_loop_checkpoint

    path = path.rstrip("/")
    return np.asarray(load_loop_checkpoint(os.path.dirname(path), int(os.path.basename(path))).primitive_state)


tic = time.time()
state = read_state(arguments.state)
print(f"state {state.shape} {state.dtype}, read in {time.time() - tic:.0f} s", flush=True)

setup = cgols_chemistry.build_chemistry(
    arguments.network,
    arguments.krome_data,
    # the kernel takes cm^-3, K and seconds; the conversion is done here
    number_density_unit_cgs=1.0,
    temperature_unit_kelvin=1.0,
    time_unit_seconds=1.0,
    adiabatic_index=5.0 / 3.0,
    solver="kernel",
    relative_tolerance=arguments.rtol,
    absolute_tolerance=arguments.atol,
    free_free=not arguments.no_free_free,
    first_step_fraction=arguments.first_step,
    radiation_field_per_cell=arguments.radiation_scale is not None,
)
kernel = setup.config.stiff_core_fn
names = list(setup.species)
number_of_species = len(names)

species_code = state[-number_of_species:]
pressure_code = state[PRESSURE_INDEX]
hydrogen_code = sum(count * species_code[names.index(name)] for name, count in HYDROGEN_ATOMS.items() if name in names)
temperature_code = pressure_code / np.maximum(np.maximum(species_code, 0.0).sum(axis=0), 1.0e-300)

if arguments.frame is not None:
    frame = np.load(arguments.frame)
    plane = state.shape[2] // 2
    ratio_n = float(np.median(frame["n_H_xz"] / hydrogen_code[:, plane, :])) / NUMBER_DENSITY_UNIT_CGS
    ratio_t = float(np.median(frame["T_xz"] / temperature_code[:, plane, :])) / TEMPERATURE_UNIT_KELVIN
    print(f"units against the frame (plane y = {plane}): density {ratio_n:.6f}, temperature {ratio_t:.6f} of the expected")
    if abs(ratio_n - 1.0) > 1.0e-3 or abs(ratio_t - 1.0) > 1.0e-3:
        raise SystemExit("the frame does not fit the state: not the same time, or other units")

pick = (slice(None, None, arguments.stride),) * 3
shape = hydrogen_code[pick].shape
hydrogen = (hydrogen_code[pick] * NUMBER_DENSITY_UNIT_CGS).ravel()
# the reaction works between its floor and 1e9 K
temperature = np.clip(temperature_code[pick].ravel() * TEMPERATURE_UNIT_KELVIN, 10.0, 1.0e9)
cells = np.concatenate(
    [(np.maximum(species_code[(slice(None),) + pick], 0.0) * NUMBER_DENSITY_UNIT_CGS).reshape(number_of_species, -1).T, temperature[:, None]],
    axis=1,
)
number_of_cells = cells.shape[0]
interval = arguments.step_kyr * SECONDS_PER_KYR

# launch in the order of the run: bins of the particle density, then the temperature
order = np.lexsort((temperature, np.round(4.0 * np.log10(np.maximum(cells[:, :-1].sum(axis=1), 1.0e-30)))))
steps = np.zeros(number_of_cells, dtype=np.int32)
temperature_after = np.zeros(number_of_cells)
chunk = 1 << 22
tic = time.time()
for start in range(0, number_of_cells, chunk):
    index = order[start : start + chunk]
    radiation = () if arguments.radiation_scale is None else (jnp.full(len(index), arguments.radiation_scale),)
    reacted, taken = kernel.with_steps(jnp.asarray(cells[index]), jnp.full(len(index), interval), *radiation)
    steps[index] = np.asarray(taken)
    temperature_after[index] = np.asarray(reacted)[:, -1]
print(f"{number_of_cells} cells reacted over {arguments.step_kyr:g} kyr in {time.time() - tic:.0f} s", flush=True)

unfinished = steps < 0
taken = np.abs(steps)
print(
    f"\ntrial steps per cell: mean {taken.mean():.2f}, median {int(np.median(taken))}, 99.9% {int(np.quantile(taken, 0.999))}, "
    f"largest {taken.max()} (limit {kernel.max_steps})"
)
print(f"cells that did NOT reach the end of the step: {int(unfinished.sum())}" + ("" if unfinished.any() else "  (all cells finished)"))


def group_cost(sequence):
    padded = np.pad(sequence, (0, (-len(sequence)) % 32))
    return padded.reshape(-1, 32).max(axis=1).sum() * 32


print(
    f"what a GPU pays (groups of 32 cells run as long as their slowest): {group_cost(taken[order]) / max(taken.sum(), 1):.2f} "
    f"x the sum of the steps in the order of the run"
)

print("\nby gas phase (the twelve that take the most steps):")
print(f"{'log n_H':>10s}{'log T':>9s}{'cells':>10s}{'cells %':>9s}{'median':>8s}{'mean':>8s}{'max':>7s}{'steps %':>9s}{'unfinished':>12s}")
log_n, log_t = np.log10(np.maximum(hydrogen, 1.0e-30)), np.log10(temperature)
rows = []
for n_lo in range(-6, 4):
    for t_lo in range(1, 9):
        mask = (log_n >= n_lo) & (log_n < n_lo + 1) & (log_t >= t_lo) & (log_t < t_lo + 1)
        if mask.any():
            rows.append((taken[mask].sum() / max(taken.sum(), 1), n_lo, t_lo, mask.sum(), np.median(taken[mask]), taken[mask].mean(), taken[mask].max(), unfinished[mask].sum()))
for share, n_lo, t_lo, count, median, mean, largest, bad in sorted(rows, reverse=True)[:12]:
    print(f"{n_lo:6d}..{n_lo + 1:<2d}{t_lo:5d}..{t_lo + 1:<2d}{count:10d}{100 * count / number_of_cells:9.2f}{median:8.0f}{mean:8.1f}{largest:7d}{100 * share:9.1f}{bad:12d}")

worst = np.argsort(np.where(unfinished, np.iinfo(np.int32).max, taken))[::-1][:15]
print("\nthe cells with the most steps (unfinished cells first):")
print(f"{'steps':>8s}   {'(ix, iy, iz)':<16s}{'n_H [cm^-3]':>12s}{'T [K]':>12s}{'T after [K]':>13s}   species that are exactly zero")
for cell in worst:
    position = tuple(int(i) * arguments.stride for i in np.unravel_index(cell, shape))
    zeros = [name for name, value in zip(names, cells[cell, :-1]) if value == 0.0]
    print(f"{steps[cell]:8d}   {str(position):<16s}{hydrogen[cell]:12.3e}{temperature[cell]:12.3e}{temperature_after[cell]:13.3e}   {' '.join(zeros) or '-'}")

if arguments.out is not None:
    np.savez(
        arguments.out,
        steps=steps.reshape(shape),
        n_H=hydrogen.reshape(shape).astype(np.float32),
        temperature=temperature.reshape(shape).astype(np.float32),
        temperature_after=temperature_after.reshape(shape).astype(np.float32),
        step_kyr=arguments.step_kyr,
        stride=arguments.stride,
    )
    print(f"\nwritten: {arguments.out}")
