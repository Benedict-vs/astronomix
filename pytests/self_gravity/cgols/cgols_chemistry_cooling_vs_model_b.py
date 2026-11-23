"""Cooling and heating of the chemistry of the CGOLS runs against the cooling curve of model B.

usage: python cgols_chemistry_cooling_vs_model_b.py [out.png] [--cache curves.npz]   (CPU, from this directory)

Chemistry: the network of the run with the field of every cell (react_galaxy_ism_plus with the cosmic-ray block,
the Lyman-Werner flux of the field with H2 self-shielding and the corrected H2 + H+ rate; bremsstrahlung, the KROME
metal table, H2 formation on dust, cosmic rays 3e-17 /s), built by ``cgols_chemistry.build_chemistry`` as
``cgols.py`` builds it. The field J / J_0 (Draine) is set by hand; dust between source and gas is part of J.

Model B: ki02gf12 of PR #8 as model B uses it (``_KI02GF12Model``): de/dt = n_H Gamma - n_H^2 Lambda(T), Lambda the
corrected Koyama & Inutsuka (2002) below 10^4.2 K blended into Gnat & Ferland (2012) CIE, and
Gamma = 2e-26 erg/s x J/J_0 x (1 - x_HII(T)).

Panels:
  1. Lambda = cooling / n_H^2 against T: the network with its species at ionisation equilibrium at that T
     (held 10 Gyr at fixed T), for n_H = 0.1, 1, 100 cm^-3 at J/J_0 = 1, and model B's curve;
  2. heating per hydrogen nucleus against T at n_H = 1 cm^-3 for J/J_0 = 1, 100 (same equilibrium species);
  3. the temperature gas of fixed density settles at, for J/J_0 = 1, 10, 100, 1000: the network with the
     thermochemistry from 10^4 K and from 100 K after 1 Gyr, model B the stable root of n_H Lambda = Gamma
     reached from the same starts.
"""
import argparse
import os
import sys
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cgols_chemistry  # noqa: E402

from astronomix import SimulationConfig, SimulationParams, construct_primitive_state, finalize_config, get_registered_variables  # noqa: E402
from astronomix._modules._chemistry._chemistry import update_chemistry  # noqa: E402
from astronomix._modules._cooling._cooling_tables import TIGRESS_GAMMA_0, _KI02GF12Model  # noqa: E402
from astronomix.option_classes.simulation_config import FINITE_VOLUME, NATIVE_JAX, BackendConfig  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("out", nargs="?", default=os.path.join(HERE, "figures/cgols/chem256_v2/cooling_chemistry_vs_model_b.png"))
parser.add_argument("--cache", default=None, help="store the curves here, and read them from here if the file exists")
parser.add_argument("--network", default="/export/scratch/bschuber/krome_networks/react_galaxy_ism_plus_cr_fuv_savin")
parser.add_argument("--krome-data", default="/export/scratch/lbranca/krome/data")
arguments = parser.parse_args()

MYR = 3.15576e13
LOG_T = np.linspace(1.3, 8.5, 73)
LAMBDA_DENSITIES = (0.1, 1.0, 100.0)
HEATING_FIELDS = (1.0, 100.0)
EQUILIBRIUM_FIELDS = (1.0, 10.0, 100.0, 1000.0)
LOG_N = np.linspace(-2.0, 3.0, 26)
model_b = _KI02GF12Model()


def model_b_equilibrium(n_h, field, start):
    """Stable temperature reached from ``start`` [K] under de/dt = n_H Gamma - n_H^2 Lambda."""
    t = np.logspace(0.5, 9.0, 40000)
    net = n_h * model_b.cooling_lambda(t) - TIGRESS_GAMMA_0 * field * (1.0 - model_b.x_HII(t))  # > 0: cools
    i = np.searchsorted(t, start)
    if net[i] > 0:  # cools down to the first root below
        below = np.nonzero(net[:i] <= 0)[0]
        return t[below[-1]] if len(below) else t[0]
    above = np.nonzero(net[i:] >= 0)[0]  # heats up to the first root above
    return t[i + above[0]] if len(above) else t[-1]


def compute():
    def build(field):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cgols_chemistry.build_chemistry(
                arguments.network, arguments.krome_data, number_density_unit_cgs=1.0, temperature_unit_kelvin=1.0,
                time_unit_seconds=MYR, adiabatic_index=5.0 / 3.0, solver="kernel", relative_tolerance=1.0e-6,
                absolute_tolerance=1.0e-14, radiation_field_per_cell=True, draine_field=field,
            )  # fmt: skip

    setup = build(1.0)
    config, params_chem, names = setup.config, setup.params, setup.species
    network = params_chem.network
    print("species:", " ".join(names), flush=True)

    def cells(chemistry_config, n_h, temperature, abundances, chemistry_params=None):
        """A row of cells: n_H [cm^-3], T [K], abundances per H nucleus [cells, species]."""
        number = len(n_h)
        sim = SimulationConfig(dimensionality=1, solver_mode=FINITE_VOLUME, backend_config=BackendConfig(backend=NATIVE_JAX),
                               box_size=1.0, num_cells=number, progress_bar=False, chemistry_config=chemistry_config)
        params = SimulationParams(t_end=1.0, gamma=5.0 / 3.0, chemistry_params=chemistry_params or params_chem)
        rv = get_registered_variables(sim)
        species = np.asarray(abundances) * np.asarray(n_h)[:, None]
        pressure = species.sum(axis=1) * np.asarray(temperature)  # temperature from the particles of the network
        state = construct_primitive_state(config=sim, registered_variables=rv, density=jnp.asarray(n_h * setup.mass_per_hydrogen),
                                          velocity_x=jnp.zeros(number), gas_pressure=jnp.asarray(pressure))
        start = rv.chemistry_species_index
        state = state.at[start : start + len(names)].set(jnp.asarray(species.T))
        return state, finalize_config(sim, state.shape), params, rv

    def read(state, rv):
        start = rv.chemistry_species_index
        species = np.asarray(state[start : start + len(names)]).T
        temperature = np.asarray(state[rv.pressure_index]) / species.sum(axis=1)
        return species, temperature

    start_abundances = np.asarray(setup.start_abundances)
    out = {"log_t": LOG_T, "log_n": LOG_N}

    # 1 + 2: ionisation equilibrium at fixed temperature. The reaction takes a field per cell only with the
    # thermochemistry, so here one set-up per field, built as the run builds it (no local dust shielding: the
    # dust is in the field) and reacted with that field everywhere.
    begin = time.perf_counter()
    for field, densities in ((1.0, LAMBDA_DENSITIES), (100.0, (1.0,))):
        held = setup if field == 1.0 else build(field)
        held_config = held.config._replace(thermochemistry=False, stiff_core_fn=None, radiation_field_per_cell=False)
        grid_t = np.tile(10.0**LOG_T, len(densities))
        grid_n = np.repeat(densities, len(LOG_T))
        state, sim, params, rv = cells(held_config, grid_n, grid_t, np.tile(start_abundances, (len(grid_t), 1)), held.params)
        for step in (1.0, 10.0, 100.0, 1.0e3, 3.0e3, 6.0e3):
            state = update_chemistry(state, rv, sim.chemistry_config, params, step)
            print(f"   J/J0 = {field:g}, fixed temperature: {step:g} Myr done ({time.perf_counter() - begin:.0f} s)", flush=True)
        species, _ = read(state, rv)
        held_network = held.params.network
        cooling = jax.vmap(held_network.cooling_terms)(jnp.asarray(species), jnp.asarray(grid_t))
        heating = jax.vmap(lambda n, t: held_network.heating_terms(n, t, held.params.cosmic_ray_rate, held.params.fuv_field))(
            jnp.asarray(species), jnp.asarray(grid_t))
        total_cooling = np.asarray(sum(cooling.values())).reshape(len(densities), -1)
        total_heating = np.asarray(sum(heating.values())).reshape(len(densities), -1)
        for k, n in enumerate(densities):
            out[f"lambda_n{n:g}_j{field:g}"] = total_cooling[k] / n**2
            out[f"heating_n{n:g}_j{field:g}"] = total_heating[k] / n
            out[f"xe_n{n:g}_j{field:g}"] = species.reshape(len(densities), len(LOG_T), -1)[k, :, names.index("E")] / n

    # 3: thermal equilibrium with the thermochemistry (kernel, as in the run)
    n_h = 10.0**LOG_N
    for start_t, label in ((1.0e4, "warm"), (1.0e2, "cold")):
        grid_n = np.tile(n_h, len(EQUILIBRIUM_FIELDS))
        grid_j = np.repeat(EQUILIBRIUM_FIELDS, len(n_h))
        state, sim, params, rv = cells(config, grid_n, np.full(len(grid_n), start_t), np.tile(start_abundances, (len(grid_n), 1)))
        elapsed = 0.0
        for step in (1.0, 9.0, 65.0, 25.0, 900.0):
            state = update_chemistry(state, rv, sim.chemistry_config, params, step, radiation_scale=jnp.asarray(grid_j))
            elapsed += step
            if elapsed in (75.0, 1000.0):
                species, temperature = read(state, rv)
                out[f"teq_{label}_{elapsed:g}"] = temperature.reshape(len(EQUILIBRIUM_FIELDS), -1)
                out[f"h2_{label}_{elapsed:g}"] = (2 * species[:, names.index("H2")] / grid_n).reshape(len(EQUILIBRIUM_FIELDS), -1)
            print(f"   thermochemistry from {start_t:g} K: {elapsed:g} Myr ({time.perf_counter() - begin:.0f} s)", flush=True)
    return out


if arguments.cache and os.path.exists(arguments.cache):
    curves = dict(np.load(arguments.cache))
else:
    curves = compute()
    if arguments.cache:
        np.savez(arguments.cache, **curves)

for j in EQUILIBRIUM_FIELDS:
    curves[f"b_warm_{j:g}"] = np.array([model_b_equilibrium(10.0**n, j, 1.0e4) for n in LOG_N])
    curves[f"b_cold_{j:g}"] = np.array([model_b_equilibrium(10.0**n, j, 1.0e2) for n in LOG_N])

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
B, CHEM = "#2f6fb3", "#d9772b"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": TEXT, "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE})
t = 10.0 ** curves["log_t"]
n = 10.0 ** curves["log_n"]
figure, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw=dict(wspace=0.27))

axis = axes[0]
axis.plot(t, model_b.cooling_lambda(t), color=B, lw=2.6, label="model B: KI02 + Gnat & Ferland (any n$_H$)")
for density, style in zip(LAMBDA_DENSITIES, ((0, (1, 1.5)), "-", "--")):
    axis.plot(t, curves[f"lambda_n{density:g}_j1"], color=CHEM, lw=1.8, ls=style, label=f"chemistry, n$_H$ = {density:g} cm$^{{-3}}$")
axis.set_ylim(1e-29, 1e-20)
axis.set_ylabel(r"$\Lambda$ = cooling rate / n$_H^2$  [erg cm$^3$ s$^{-1}$]")
axis.set_title("Cooling, J / J$_0$ = 1 (network: species at ionisation\nequilibrium at each T)", fontsize=10, loc="left")

axis = axes[1]
for field, style in zip(HEATING_FIELDS, ("-", "--")):
    axis.plot(t, TIGRESS_GAMMA_0 * field * (1.0 - model_b.x_HII(t)), color=B, lw=2.2, ls=style, label=f"model B, J / J$_0$ = {field:g}")
    axis.plot(t, curves[f"heating_n1_j{field:g}"], color=CHEM, lw=1.8, ls=style, label=f"chemistry, J / J$_0$ = {field:g}")
axis.set_ylim(1e-28, 1e-22)
axis.set_ylabel("heating rate per H nucleus [erg s$^{-1}$]")
axis.set_title("Heating at n$_H$ = 1 cm$^{-3}$", fontsize=10, loc="left")
axis.text(1.5e4, 3e-28, "network above 10$^4$ K: dust photoelectric heating\n(Bakes & Tielens 1994) at every T; in the hot\nwind of the runs (n$_e$ ~ 1e-3) it is 2-5 % of the cooling",
          fontsize=8, color=MUTED, va="bottom")

axis = axes[2]
shades = {1.0: "#9ecae1", 10.0: "#4a90c8", 100.0: "#1f5d99", 1000.0: "#0b2a4d"}
oranges = {1.0: "#f6c08f", 10.0: "#ec9550", 100.0: "#d9772b", 1000.0: "#8a3f0c"}
for field in EQUILIBRIUM_FIELDS:
    axis.plot(n, curves[f"b_warm_{field:g}"], color=shades[field], lw=2.2)
    axis.plot(n, curves[f"b_cold_{field:g}"], color=shades[field], lw=2.2, ls=(0, (1, 1.5)))
    axis.plot(n, curves[f"teq_warm_1000"][EQUILIBRIUM_FIELDS.index(field)], color=oranges[field], lw=1.8)
    axis.plot(n, curves[f"teq_cold_1000"][EQUILIBRIUM_FIELDS.index(field)], color=oranges[field], lw=1.8, ls=(0, (1, 1.5)))
    axis.annotate(f"{field:g}", (n[2], curves[f"teq_warm_1000"][EQUILIBRIUM_FIELDS.index(field)][2]), xytext=(3, 3),
                  textcoords="offset points", fontsize=8.5, color=oranges[field])
axis.plot([], [], color=B, lw=2.2, label="model B (stable root)")
axis.plot([], [], color=CHEM, lw=1.8, label="chemistry, after 1 Gyr")
axis.plot([], [], color=MUTED, lw=1.5, label="from 10$^4$ K")
axis.plot([], [], color=MUTED, lw=1.5, ls=(0, (1, 1.5)), label="from 100 K")
axis.set_xlabel(r"hydrogen nuclei n$_H$ [cm$^{-3}$]")
axis.set_ylabel("temperature [K]")
axis.set_ylim(10, 3e4)
axis.set_title("Temperature gas of fixed density settles at,\nJ / J$_0$ = 1, 10, 100, 1000 (dark = strong)", fontsize=10, loc="left")
for axis in axes:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(color=GRID, lw=0.6)
    axis.legend(fontsize=8, frameon=False, loc="best")
for axis in axes[:2]:
    axis.set_xlabel("temperature [K]")
    axis.set_xlim(20, 3e8)
figure.text(0.125, -0.04, "Chemistry: react_galaxy_ism_plus + cosmic rays (3e-17 /s) + Lyman-Werner flux of the field with self-shielding + "
            "corrected H$_2$ + H$^+$ rate, bremsstrahlung, KROME metal table, solar metallicity, as the run with the field of every cell.\n"
            "Model B: ki02gf12 (PR #8), $\\Gamma$ = 2e-26 erg/s J/J$_0$ (1 - x$_{HII}$), as the B-series uses it. Same J / J$_0$ (Draine) for both; "
            "dust between source and gas is in J.", fontsize=8.5, color=MUTED)
figure.savefig(arguments.out, dpi=130, bbox_inches="tight")
print("written", arguments.out)
