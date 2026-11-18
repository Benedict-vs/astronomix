"""A network in the KROME format at the gas conditions of a CGOLS run.

``examples/chemistry_krome_network.py`` covers cold, dense gas (50 K, 1 to 1e4
hydrogen nuclei per cm^3). A CGOLS box is mostly something else: a warm disk
(about 7000 K at about 1 cm^-3) under a hot, thin wind and halo (1e5 to 1e8 K at
1e-4 to 1e-1 cm^-3). This script runs the network there, with the vmapped Kvaerno5
and the tabulated metal cooling, and compares it with the cooling of model B (KI02
+ Gnat & Ferland 2012, ``ki02_gf12_cooling``):

1. the cooling function at ionisation equilibrium, from 30 K to 1e8 K;
2. the temperature at which disk gas settles (heating against cooling);
3. one step of the length of a hydro step from equilibrium over the (n_H, T)
   plane: does every cell converge, and in how many solver steps;
4. the wall time of ``update_chemistry`` per cell (``--timing-cells``).

Run it from the repository root; the network file and the ``data`` folder of a
KROME installation are not part of this repository::

    python pytests/chemistry/krome_cgols_conditions.py --network /path/to/react_galaxy_ism_plus \\
        --krome-data /path/to/krome/data --cpu --figure cooling.png
    python pytests/chemistry/krome_cgols_conditions.py --network ... --krome-data ... \\
        --timing-cells 4096 32768 262144          # on one free GPU (autocvd)
    python pytests/chemistry/krome_cgols_conditions.py --network ... --krome-data ... \\
        --timing-cells 4096 32768 262144 --kernel # the per-cell kernel instead of Kvaerno5

Checked on 2026-11-18 on the CPU with react_galaxy_ism_plus under the Draine
field with dust shielding (``--cpu``):

* cooling at equilibrium over cooling of model B: 0.7 to 1.9 below 6000 K, 0.8
  to 1.0 from 1.8e4 K to 3e6 K, 0.014 at 1e4 K (few free electrons at
  equilibrium, hence little Lyman alpha) and falling to 0.2 at 1e8 K (the port
  has no bremsstrahlung of H and He; with KROME's ``cooling_ff`` added the ratio
  is 0.86 to 1.5 from 1e5 K up);
* disk gas: with the cosmic-ray block of the file commented out (as the file
  stands) gas of 1 cm^-3 cools to 360 K, with the block on it settles at 4200 K
  (model B: 5000 K);
* one step of 1e-3 Myr: all 159 cells converge, in 18 solver steps (median) to
  120 (10 cm^-3 at 1.8e5 K, which cools tenfold within the step);
* cost on the CPU: 9 ms per cell and step for the cells that take at most 30
  solver steps, 28 to 55 ms with the whole plane in one call (lockstep).

On one H200, which another job was using at full load at the same time (the
timings are upper limits):

* the tables of 1 and 2 are those of the CPU; the one-step results agree with
  the CPU to 2e-6 dex in the species above 1e-6 per H and 2e-6 in the
  temperature, and to 1e-3 dex (the absolute tolerance) in the trace species;
* one cell differs after the 10 Gyr at fixed temperature: at exactly 1e6 K,
  the upper temperature limit of the three Si+ / Si++ reactions of the file,
  rounding decides whether they act (Si+ on the CPU, Si++ on the GPU);
* cost of the vmapped Kvaerno5: 50 to 90 microseconds per cell and step for
  the calm cells, 270 to 450 with the whole plane in one call, linear in the
  number of cells from 4096 to 262144; 8.4 GiB for 262144 cells.

The per-cell kernel (``--kernel``) on the same H200, the metal table looked up
(``--kernel-table lookup``), 262144 cells:

* built in 98 s; after the step of 1e-3 Myr it agrees with Kvaerno5 to 2e-3 dex
  in the species and 6e-5 in the temperature;
* 13 microseconds per cell and step for the calm cells and 69 with the whole
  plane at a relative tolerance of 1e-6, 6 and 15 at 1e-4 (``--rtol 1e-4``,
  the default of KROME); 0.3 GiB;
* one solver step of the kernel costs about 0.2 microseconds per cell, so the
  cost is the number of steps: without the metal table (``--kernel-table
  none``), where the hot gas cools more slowly, 4.7 and 23 at 1e-6;
* a first trial step of the whole step (``--kernel-first-step 1``) changes
  nothing: 5.4 and 24 without the table at 1e-6.
"""

import argparse
import os
import time
import warnings

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--network", required=True, help="network file in the KROME format")
parser.add_argument("--krome-data", required=True, help="the data folder of a KROME installation")
parser.add_argument("--cpu", action="store_true", help="run on the CPU (default: one free GPU through autocvd)")
parser.add_argument("--gpu-timeout", type=int, default=None, help="give up waiting for a free GPU after this many seconds")
parser.add_argument("--step-myr", type=float, default=1.0e-3, help="length of the hydro-sized step [Myr]")
parser.add_argument("--timing-cells", type=int, nargs="*", default=[], help="cell counts for the timing of update_chemistry")
parser.add_argument("--chunk", type=int, default=0, help="reaction_chunk_size of the timing runs")
parser.add_argument("--kernel", action="store_true", help="part 4 with the per-cell Warp kernel (gradsolve) instead of the "
                    "vmapped Kvaerno5, and its one-step result against that of Kvaerno5")
parser.add_argument("--kernel-table", choices=("lookup", "weights", "none"), default="lookup", help="metal cooling table of "
                    "the kernel: looked up as by the other solvers (a generated table function), interpolated without a "
                    "look-up (every node with its weight: the same cooling, much more arithmetic), or left out")
parser.add_argument("--kernel-first-step", type=float, default=0.01, help="first trial step of the kernel as a fraction of the step")
parser.add_argument("--figure", default=None, help="write the cooling functions to this image file")
parser.add_argument("--out", default=None, help="write the results to this npz file")
parser.add_argument("--compare", default=None, help="npz of another run (e.g. on the CPU): compare the one-step results")
parser.add_argument("--equilibrium-from", default=None, help="npz of an earlier run (--out): take its ionisation equilibrium and "
                    "skip parts 1 and 2")
parser.add_argument("--rtol", type=float, default=1.0e-6, help="relative tolerance of the solvers (KROME's default is 1e-4)")
arguments = parser.parse_args()

if arguments.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    from autocvd import autocvd

    autocvd(num_gpus=1, timeout=arguments.gpu_timeout)

# ruff: noqa: E402
import numpy as np

import jax

# the chemistry needs double precision
jax.config.update("jax_enable_x64", True)

import diffrax as dx
import jax.numpy as jnp
import optimistix as optx

import astronomix
from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
)
from astronomix._modules._chemistry._chemistry import make_vector_field, update_chemistry
from astronomix._modules._chemistry._krome_photo import photo_spectrum
from astronomix._modules._cooling._cooling import (
    get_pressure_from_temperature,
    get_temperature_from_pressure,
)
from astronomix._modules._cooling._cooling_tables import TIGRESS_GAMMA_0, _KI02GF12Model
from astronomix.option_classes.simulation_config import FINITE_VOLUME, NATIVE_JAX, BackendConfig
from astronomix.setup_helpers import build_chemistry_from_network_file

# code units of the examples: m_p cm^-3, K, Myr
GAMMA = 5.0 / 3.0
MYR = 3.15576e13
RTOL, ATOL, MAX_STEPS = arguments.rtol, 1.0e-12, 8192
ON_GPU = not arguments.cpu

print(f"astronomix from {astronomix.__file__}")
print(f"devices: {jax.devices()}")
print(f"network: {arguments.network}")


def build(thermochemistry, chunk=0, metal_table=True):
    """The chemistry of the network, with the temperature evolved or held."""
    with warnings.catch_warnings():
        # lines that KROME skips too (react_galaxy_ism_plus has three)
        warnings.simplefilter("ignore", UserWarning)
        chemistry_config, chemistry_params, names = build_chemistry_from_network_file(
            arguments.network,
            "krome",
            number_density_unit_cgs=1.0,
            temperature_unit_kelvin=1.0,
            time_unit_seconds=MYR,
            cosmic_ray_rate=3.0e-17,
            thermochemistry=thermochemistry,
            cooling_courant=1.0e3,  # no limiter: the step shows what the solver does
            floor_temperature=3.0,
            absolute_tolerance=ATOL,
            relative_tolerance=RTOL,
            max_steps=MAX_STEPS,
            krome_h2_formation_on_dust=True,
            krome_data_folder=arguments.krome_data,
            krome_metal_table=metal_table,
            krome_photo_bins=photo_spectrum("draine", 5.0, 13.6, 10),
            krome_shield_habing_dust=True,
        )
    return chemistry_config._replace(reaction_chunk_size=chunk), chemistry_params, tuple(names)


CONFIG_THERMO, CHEMISTRY_PARAMS, NAMES = build(True)
CONFIG_ISOTHERMAL, _, _ = build(False)
NETWORK = CHEMISTRY_PARAMS.network
X, Z = CHEMISTRY_PARAMS.hydrogen_mass_fraction, CHEMISTRY_PARAMS.metal_mass_fraction
print(f"species ({len(NAMES)}): {' '.join(NAMES)}")
print(f"cosmic-ray reactions in the file: {len(NETWORK.cosmic_ray_terms)}")

# Half ionised at the start, so that ionisation and recombination both run from
# the first step; the electrons balance the charge.
START = {"H": 0.5, "H+": 0.5, "H2": 1.0e-6, "He": 8.5e-2, "C+": 1.4e-4, "O": 3.2e-4, "Si+": 1.7e-6}
START = {name: START.get(name, 1.0e-20) for name in NAMES if name != "E"}
START["E"] = sum((name.count("+") - name.count("-")) * value for name, value in START.items())


def cells(chemistry_config, hydrogen_nuclei, temperature, abundances=None):
    """A row of cells of ``n_H`` [cm^-3], ``T`` [K] and abundances per H nucleus ``[species, cells]``."""
    number = len(hydrogen_nuclei)
    config = SimulationConfig(
        dimensionality=1,
        solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        box_size=1.0,
        num_cells=number,
        progress_bar=False,
        chemistry_config=chemistry_config,
    )
    params = SimulationParams(t_end=1.0, gamma=GAMMA, chemistry_params=CHEMISTRY_PARAMS)
    registered_variables = get_registered_variables(config)
    hydrogen_nuclei = jnp.asarray(hydrogen_nuclei)
    density = hydrogen_nuclei / X
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=jnp.zeros(number),
        gas_pressure=get_pressure_from_temperature(density, jnp.asarray(temperature), X, Z),
    )
    if abundances is None:
        abundances = np.array([[START[name]] * number for name in NAMES])
    start = registered_variables.chemistry_species_index
    state = state.at[start : start + len(NAMES)].set(jnp.asarray(abundances) * hydrogen_nuclei)
    return state, finalize_config(config, state.shape), params, registered_variables


def read(state, registered_variables):
    """Number densities ``[species, cells]`` and temperatures [K] of a state."""
    start = registered_variables.chemistry_species_index
    temperature = get_temperature_from_pressure(
        state[registered_variables.density_index], state[registered_variables.pressure_index], X, Z
    )
    return np.asarray(state[start : start + len(NAMES)]), np.asarray(temperature)


MODEL_B = _KI02GF12Model(zprime=1.0)

# -------------------------------------------------------------
# ======== ↓ 1. Cooling at ionisation equilibrium ↓ ===========
# -------------------------------------------------------------

LOG_T = np.linspace(1.5, 8.0, 53)
DENSITIES = np.array([1.0e-3, 1.0e-1, 1.0e1])
SHAPE = (len(LOG_T), len(DENSITIES))
grid_t, grid_n = (a.ravel() for a in np.meshgrid(10.0**LOG_T, DENSITIES, indexing="ij"))

if arguments.equilibrium_from:
    equilibrium = np.load(arguments.equilibrium_from)["equilibrium"]
    print(f"\nparts 1 and 2 skipped: ionisation equilibrium of {arguments.equilibrium_from}")
else:
    state, config, params, rv = cells(CONFIG_ISOTHERMAL, grid_n, grid_t)
    # 10 Gyr at fixed temperature: long enough for the recombination at 1e-3 cm^-3
    for step in (1.0, 10.0, 100.0, 1.0e3, 3.0e3, 6.0e3):
        state = update_chemistry(state, rv, config.chemistry_config, params, step)
    equilibrium = read(state, rv)[0] / grid_n  # per hydrogen nucleus


    @jax.jit
    def heating_and_cooling(number_densities, temperature):
        """The terms of the network [erg cm^-3 s^-1] for cells ``[cells, species]``."""
        cooling = jax.vmap(NETWORK.cooling_terms)(number_densities, temperature)
        heating = jax.vmap(
            lambda n, t: NETWORK.heating_terms(n, t, CHEMISTRY_PARAMS.cosmic_ray_rate, CHEMISTRY_PARAMS.fuv_field)
        )(number_densities, temperature)
        return cooling, heating


    cooling, heating = heating_and_cooling(jnp.asarray((equilibrium * grid_n).T), jnp.asarray(grid_t))
    lambda_network = (sum(np.asarray(term) for term in cooling.values()) / grid_n**2).reshape(SHAPE)
    heating_per_nucleus = (sum(np.asarray(term) for term in heating.values()) / grid_n).reshape(SHAPE)
    lambda_model_b = MODEL_B.cooling_lambda(10.0**LOG_T)
    of = lambda name: equilibrium[NAMES.index(name)].reshape(SHAPE) if name in NAMES else np.zeros(SHAPE)
    # KROME's cooling_ff (-cooling=FF, mean Gaunt factor 1.5), which the port does not have
    free_free = 1.42e-27 * 1.5 * np.sqrt(10.0**LOG_T)[:, None] * (of("H+") + of("He+") + 4.0 * of("He++")) * of("E")

    print("\n1. Cooling at ionisation equilibrium, Lambda = cooling / n_H^2 [erg cm^3 / s]")
    print("   The last three columns are for n_H = 0.1: network over model B, the same with bremsstrahlung, electrons per H.")
    print(
        "       T [K]"
        + "".join(f"{f'n_H = {density:g}':>13s}" for density in DENSITIES)
        + f"{'model B':>13s}{'ratio':>9s}{'with ff':>9s}{'x(E)':>11s}"
    )
    for i in range(len(LOG_T)):
        if i % 4 == 0 or 5.0e3 < 10.0 ** LOG_T[i] < 2.0e4:
            print(
                f"   {10.0 ** LOG_T[i]:9.3g}"
                + "".join(f"{lambda_network[i, j]:13.3e}" for j in range(len(DENSITIES)))
                + f"{lambda_model_b[i]:13.3e}{lambda_network[i, 1] / lambda_model_b[i]:9.3f}"
                f"{(lambda_network[i, 1] + free_free[i, 1]) / lambda_model_b[i]:9.3f}{of('E')[i, 1]:11.2e}"
            )

    if arguments.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        surface, text, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
        blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
        temperature = 10.0**LOG_T
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(9.0, 7.6), sharex=True, height_ratios=(2.0, 1.0), facecolor=surface)
        for axis in (top, bottom):
            axis.set_facecolor(surface)
            axis.grid(True, which="major", color=grid, linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xscale("log")
            axis.set_yscale("log")
        with_free_free = lambda_network[:, 1] + free_free[:, 1]
        top.plot(temperature, lambda_model_b, color=blue, linewidth=2.0, label="model B curve (KI02 + Gnat & Ferland)")
        top.plot(temperature, lambda_network[:, 1], color=orange, linewidth=2.0, label="network")
        top.plot(temperature, with_free_free, color=aqua, linewidth=2.0, linestyle=(0, (5, 2)), label="network + H and He bremsstrahlung")
        top.set_ylim(5e-29, 2e-21)
        top.set_ylabel(r"$\Lambda$ = cooling rate / $n_{\rm H}^2$   [erg cm$^3$ s$^{-1}$]", color=text)
        top.legend(loc="upper left", frameon=False, labelcolor=text)
        bottom.axhline(1.0, color=muted, linewidth=0.8)
        bottom.plot(temperature, lambda_network[:, 1] / lambda_model_b, color=orange, linewidth=2.0)
        bottom.plot(temperature, with_free_free / lambda_model_b, color=aqua, linewidth=2.0, linestyle=(0, (5, 2)))
        bottom.set_ylim(0.008, 4.0)
        bottom.set_ylabel("network / model B", color=text)
        bottom.set_xlabel("temperature [K]", color=text)
        bottom.set_xlim(temperature[0], temperature[-1])
        figure.suptitle(
            f"Cooling of {os.path.basename(arguments.network)} at ionisation equilibrium against the curve of model B",
            x=0.07, ha="left", fontsize=13, color=text,
        )  # fmt: skip
        figure.text(
            0.07, 0.925, r"$n_{\rm H}$ = 0.1 cm$^{-3}$, solar metallicity, Draine field with dust shielding, "
            f"{len(NETWORK.cosmic_ray_terms)} cosmic-ray reactions", fontsize=9.5, color=muted, ha="left",
        )  # fmt: skip
        figure.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.08, hspace=0.08)
        figure.savefig(arguments.figure, dpi=160, facecolor=surface)
        print(f"   figure: {arguments.figure}")

    # -------------------------------------------------------------
    # ============ ↓ 2. Where disk gas settles ↓ ==================
    # -------------------------------------------------------------

    disk_n = np.logspace(-2.0, 2.0, 9)
    state, config, params, rv = cells(CONFIG_THERMO, disk_n, np.full(len(disk_n), 8000.0))
    print("\n2. Disk gas from 8000 K, thermochemistry on: temperature [K] after")
    print("     time [Myr] " + "".join(f"{n:>9.3g}" for n in disk_n) + "   <- n_H [cm^-3]")
    elapsed = 0.0
    for step in (1.0, 10.0, 100.0, 1000.0, 3000.0):
        state = update_chemistry(state, rv, config.chemistry_config, params, step)
        elapsed += step
        print(f"     {elapsed:10.4g} " + "".join(f"{t:9.0f}" for t in read(state, rv)[1]), flush=True)
    settled = read(state, rv)[1]
    # model B: Gamma_0 (1 - x_HII) = n_H Lambda(T); the warmest balance below 2e4 K
    fine = np.logspace(1.0, 4.3, 4000)
    model_b_settled = []
    for n in disk_n:
        net = n * MODEL_B.cooling_lambda(fine) - TIGRESS_GAMMA_0 * (1.0 - MODEL_B.x_HII(fine))
        roots = fine[1:][np.sign(net[1:]) != np.sign(net[:-1])]
        model_b_settled.append(roots.max() if len(roots) else np.nan)
    print("        model B " + "".join(f"{t:9.0f}" for t in model_b_settled) + "   (heating 2e-26 erg/s per H against its curve)")

# -------------------------------------------------------------
# ====== ↓ 3. One hydro-sized step from equilibrium ↓ =========
# -------------------------------------------------------------



def right_hand_side(network):
    """The right-hand side ``f(t, state, args)`` of a network, state = abundances [cm^-3] and temperature [K]."""
    return make_vector_field(
        network, CHEMISTRY_PARAMS.rate_modifier_a, CHEMISTRY_PARAMS.rate_modifier_b, CHEMISTRY_PARAMS.cosmic_ray_rate,
        CHEMISTRY_PARAMS.fuv_field, CHEMISTRY_PARAMS.visual_extinction, True, GAMMA, CHEMISTRY_PARAMS.dust_to_gas_ratio,
        CHEMISTRY_PARAMS.hydrogen_molecule_formation_rate_coefficient, CONFIG_THERMO.hydrogen_index,
        CONFIG_THERMO.molecular_hydrogen_index, CONFIG_THERMO.electron_index, CONFIG_THERMO.atomic_oxygen_index,
        CONFIG_THERMO.ionized_hydrogen_index, CONFIG_THERMO.helium_index, CONFIG_THERMO.ionized_carbon_index,
        CONFIG_THERMO.co_cooling, CONFIG_THERMO.carbon_monoxide_index, CHEMISTRY_PARAMS.co_cooling_table,
        CHEMISTRY_PARAMS.co_cooling_bounds,
    )  # fmt: skip


vector_field = right_hand_side(NETWORK)
solver = dx.Kvaerno5(root_finder=dx.with_stepsize_controller_tols(optx.Newton)())


def kvaerno5(field):
    """The Diffrax call of ``_advance_single_cell`` for all cells, with its statistics."""

    def solve_cell(initial_state, step_seconds):
        solution = dx.diffeqsolve(
            dx.ODETerm(field), solver, t0=0.0, t1=step_seconds, dt0=jnp.minimum(1e-6, step_seconds * 1e-3),
            y0=initial_state, args=None, stepsize_controller=dx.PIDController(rtol=RTOL, atol=ATOL),
            saveat=dx.SaveAt(t1=True), max_steps=MAX_STEPS, throw=False,
        )  # fmt: skip
        return solution.ys[-1], solution.result == dx.RESULTS.successful, solution.stats["num_steps"]

    return jax.jit(jax.vmap(solve_cell, in_axes=(0, None)))


solve_cells = kvaerno5(vector_field)
initial_state = jnp.asarray(np.concatenate([(equilibrium * grid_n).T, grid_t[:, None]], axis=1))
print("\n3. One step from equilibrium over the (n_H, T) plane, thermochemistry on")
for step_myr in (arguments.step_myr, 10.0 * arguments.step_myr):
    final, succeeded, solver_steps = (np.asarray(a) for a in solve_cells(initial_state, step_myr * MYR))
    print(
        f"   step of {step_myr:g} Myr: {int(succeeded.sum())} of {succeeded.size} cells converged; solver steps "
        f"{solver_steps.min()} to {solver_steps.max()}, median {int(np.median(solver_steps))}"
    )
    for j, density in enumerate(DENSITIES):
        steps_j = solver_steps.reshape(SHAPE)[:, j]
        ratio = (final[:, -1] / grid_t).reshape(SHAPE)[:, j]
        print(
            f"      n_H = {density:g}: up to {steps_j.max()} solver steps (at {10.0 ** LOG_T[np.argmax(steps_j)]:.3g} K), "
            f"temperature after / before {ratio.min():.3f} to {ratio.max():.3f}"
        )
    if step_myr == arguments.step_myr:
        one_step = dict(final=final, succeeded=succeeded, solver_steps=solver_steps)

if arguments.compare:
    other = np.load(arguments.compare)
    significant = other["step_final"][:, :-1] > 1e-10 * grid_n[:, None]
    ratio = np.maximum(one_step["final"][:, :-1], 1e-300) / np.maximum(other["step_final"][:, :-1], 1e-300)
    print(
        f"   against {os.path.basename(arguments.compare)}: species above 1e-10 per H "
        f"{np.abs(np.log10(ratio))[significant].max():.2e} dex, temperature "
        f"{np.max(np.abs(one_step['final'][:, -1] / other['step_final'][:, -1] - 1.0)):.2e} (relative), solver steps "
        f"identical: {bool(np.array_equal(one_step['solver_steps'], other['step_solver_steps']))}"
    )

# -------------------------------------------------------------
# ============ ↓ 4. Cost of update_chemistry ↓ ================
# -------------------------------------------------------------

# Kvaerno5 is vmapped in lockstep: every cell of a call takes as many solver steps
# as the slowest one. "all" is the whole plane, with its few cells that cool
# strongly within the step; "calm" leaves out the cells that need more than 30
# solver steps, which is what most of a CGOLS box looks like.
calm = one_step["solver_steps"] <= 30
timings = {}
kernel = None
if arguments.kernel and arguments.timing_cells:
    from astronomix._modules._chemistry._warp_stiff import build_fused_core_solver

    with_table = arguments.kernel_table != "none"
    _, kernel_params, _ = build(True, metal_table={"lookup": True, "weights": "kernel", "none": False}[arguments.kernel_table])
    table_label = {"lookup": "looked up", "weights": "interpolated without a look-up", "none": "left out"}[arguments.kernel_table]
    print(
        f"\nBuilding the per-cell kernel, metal table {table_label}, first trial step {arguments.kernel_first_step:g} of "
        "the step (cached afterwards)",
        flush=True,
    )
    begin = time.perf_counter()
    kernel_field = right_hand_side(kernel_params.network)
    kernel = build_fused_core_solver(
        kernel_field, len(NAMES) + 1, rtol=RTOL, atol=ATOL, max_steps=MAX_STEPS,
        cache_key=("krome-cgols-conditions", arguments.network, arguments.krome_data, arguments.kernel_table),
        first_step_fraction=arguments.kernel_first_step,
    )  # fmt: skip
    generated = time.perf_counter() - begin
    begin = time.perf_counter()
    kernel_final = np.asarray(kernel(initial_state, jnp.full(len(grid_t), arguments.step_myr * MYR)))
    print(f"   code generated in {generated:.0f} s, compiled and run on the {len(grid_t)} cells in {time.perf_counter() - begin:.0f} s")
    # two references: Kvaerno5 on the very right-hand side of the kernel (what the generated code must reproduce), and
    # Kvaerno5 with the table look-up (part 3: what the kernel is meant to replace)
    same_field = np.asarray(kvaerno5(kernel_field)(initial_state, arguments.step_myr * MYR)[0])
    references = [("Kvaerno5 on the same right-hand side", same_field)]
    if with_table:
        references.append(("Kvaerno5 with the table look-up", one_step["final"]))
    for label, reference in references:
        significant = reference[:, :-1] > 1e-10 * grid_n[:, None]
        ratio = np.maximum(kernel_final[:, :-1], 1e-300) / np.maximum(reference[:, :-1], 1e-300)
        print(
            f"   kernel against {label}, step of {arguments.step_myr:g} Myr: species above 1e-10 per H "
            f"{np.abs(np.log10(ratio))[significant].max():.2e} dex, temperature "
            f"{np.max(np.abs(kernel_final[:, -1] / reference[:, -1] - 1.0)):.2e} (relative)"
        )
if arguments.timing_cells:
    print(
        f"\n4. Wall time of update_chemistry with {'the per-cell Warp kernel' if kernel else 'the vmapped Kvaerno5'}, "
        f"step of {arguments.step_myr:g} Myr. all: the {calm.size} cells of the plane, "
        f"repeated (lockstep {one_step['solver_steps'].max()} solver steps); calm: the {int(calm.sum())} of them with at "
        f"most 30 solver steps"
    )
too_slow = set()
for number in arguments.timing_cells:
    for mix, mask in (("calm", calm), ("all", np.ones_like(calm))):
        if mix in too_slow:
            print(f"   {number:8d} cells, {mix}: skipped, the smaller run took more than two minutes per call")
            continue
        repeat = int(np.ceil(number / mask.sum()))
        n_tiled, t_tiled = (np.tile(a[mask], repeat)[:number] for a in (grid_n, grid_t))
        chemistry_config, _, _ = build(True, chunk=arguments.chunk)
        if kernel is not None:
            chemistry_config = chemistry_config._replace(stiff_core_solver="warp", stiff_core_fn=kernel)
        state, config, params, rv = cells(chemistry_config, n_tiled, t_tiled, np.tile(equilibrium[:, mask], (1, repeat))[:, :number])
        begin = time.perf_counter()
        after = jax.block_until_ready(update_chemistry(state, rv, config.chemistry_config, params, arguments.step_myr))
        first = time.perf_counter() - begin
        laps = []
        for _ in range(3):
            begin = time.perf_counter()
            after = jax.block_until_ready(update_chemistry(state, rv, config.chemistry_config, params, arguments.step_myr))
            laps.append(time.perf_counter() - begin)
            if laps[-1] > 60.0:
                break
        if min(laps) > 120.0:
            too_slow.add(mix)
        (species_after, temperature_after), (species_before, _) = read(after, rv), read(state, rv)
        # a cell whose solve fails keeps its state (throw=False in the module)
        unchanged = np.all(species_after == species_before, axis=0) & (temperature_after == t_tiled)
        memory = ""
        if ON_GPU:
            peak = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)
            memory = f", peak device memory so far {peak / 2**30:.2f} GiB"
        timings[(number, mix)] = min(laps)
        print(
            f"   {number:8d} cells, {mix:4s}: {min(laps):9.3f} s per call = {min(laps) / number * 1e6:10.2f} microseconds per "
            f"cell (first call {first:.0f} s); cells left unchanged: {int(unchanged.sum())}; finite: "
            f"{bool(np.isfinite(species_after).all())}{memory}",
            flush=True,
        )

if arguments.out:
    first_parts = {}
    if not arguments.equilibrium_from:
        first_parts = dict(
            lambda_network=lambda_network,
            lambda_free_free=free_free,
            lambda_model_b=lambda_model_b,
            heating_per_nucleus=heating_per_nucleus,
            disk_densities=disk_n,
            disk_temperature=settled,
            disk_temperature_model_b=np.array(model_b_settled),
        )
    np.savez(
        arguments.out,
        log_t=LOG_T,
        densities=DENSITIES,
        names=np.array(NAMES),
        equilibrium=equilibrium,
        **first_parts,
        timing_cells=np.array([key[0] for key in timings]),
        timing_mix=np.array([key[1] for key in timings]),
        timing_seconds=np.array(list(timings.values())),
        **{f"step_{name}": value for name, value in one_step.items()},
    )
    print(f"\nsaved {arguments.out}")
