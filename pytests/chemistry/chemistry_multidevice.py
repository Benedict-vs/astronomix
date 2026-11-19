"""Chemistry on a multi-device mesh against the same run on one device.

The 16-species carbox network (``external/carbox/data/simple_latent_tgas.csv``)
with thermochemistry in a small periodic box with a moving over-dense blob,
once on one device and once split along x over ``N`` devices with the mesh
``time_integration`` is given through ``sharding`` (the mesh context of
``_pallas_helpers``; the reaction then runs in a ``shard_map`` over that mesh,
see ``_iteration_level_updates``). ``--solver fv`` (default) is the
finite-volume solver, ``--solver fd`` the finite-difference solver, which moves
the species as abundances per unit mass (``_species_transport.py``; its
stencils cross the device boundaries through the same halo exchange as the
hydro).

Two comparisons, each for the chosen solver:

* one fixed step: the chemistry of both runs starts from bit-identical cells,
  so the difference is what the split itself introduces;
* a short adaptive run: differences grow to the tolerance of the stiff solver,
  because the gas cools strongly and the solver takes adaptive steps.

With ``--krome-network`` the network is one in the KROME format (read by
astronomix, with KROME's heating and cooling; ``--krome-data`` is the ``data``
folder of a KROME installation) and the gas is warm (8000 K, 0.3 to 30 hydrogen
nuclei per cm^3), where its cooling acts within a few steps, with the tabulated
metal cooling in both solvers.

Run it from the repository root (needs carbox, diffrax, and for ``--kernel``
gradsolve and warp-lang, see ``requirements-chemistry.txt``)::

    python pytests/chemistry/chemistry_multidevice.py --cpu 2 --kernel   # 2 emulated CPU devices
    python pytests/chemistry/chemistry_multidevice.py 2 --kernel         # 2 free GPUs (autocvd)
    python pytests/chemistry/chemistry_multidevice.py --cpu 2 --kernel --solver fd
    python pytests/chemistry/chemistry_multidevice.py --cpu 2 --cells 8 \\
        --krome-network /path/to/react_galaxy_ism_plus --krome-data /path/to/krome/data

Checked on 2026-11-17 with emulated CPU devices: one step agrees to rounding
with the Warp kernel on 2 and on 4 devices (1.8e-15 dex in the species, 6e-16
in the pressure) and to 4e-7 dex / 3e-7 with the vmapped Kvaerno5 on 2 devices
(the tolerance of the solver is 1e-6). On 2026-11-18, the KROME network
react_galaxy_ism_plus with the vmapped Kvaerno5 on 2 emulated devices and 8^3
cells: 4e-9 dex and 2e-13 in the pressure after one step, 1.5e-10 dex after the
short run; the same network with the Warp kernel (``--kernel``, the metal table
handed to the kernel as an array): 9e-16 dex and 6e-16 after one step, 2e-10 dex
after the short run. On 2026-11-19 with the finite-difference solver
(``--solver fd --kernel``, 2 emulated devices): the carbox network 1.8e-15 dex
after one step and 7e-15 dex after the short run, react_galaxy_ism_plus with
its cosmic-ray block (8^3 cells) 1.8e-15 dex and 2.5e-10 dex. Not yet run on
GPUs.
"""

import argparse
import os
import warnings

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("devices", type=int, help="number of devices the box is split over (along x)")
parser.add_argument("--cpu", action="store_true", help="emulate the devices on the CPU")
parser.add_argument("--kernel", action="store_true", help="gradsolve Warp kernel instead of the vmapped Kvaerno5")
parser.add_argument("--cells", type=int, default=16, help="cells per axis")
parser.add_argument("--solver", choices=("fv", "fd"), default="fv", help="finite-volume (default) or finite-difference solver")
parser.add_argument("--krome-network", default=None, help="a network file in the KROME format instead of the carbox network")
parser.add_argument("--krome-data", default=None, help="the data folder of a KROME installation (with --krome-network)")
arguments = parser.parse_args()

if arguments.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={arguments.devices}"
else:
    from autocvd import autocvd

    autocvd(num_gpus=arguments.devices)

# ruff: noqa: E402
import numpy as np

import jax

# carbox switches double precision on at import; do it up front (the Warp kernel
# takes float64 arrays only).
jax.config.update("jax_enable_x64", True)
# jax 0.10: the Shardy partitioner cannot take the integer axis names of the
# state mesh (see cgols.py).
jax.config.update("jax_use_shardy_partitioner", False)

import jax.numpy as jnp
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

import astronomix
from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)
from astronomix._modules._cooling._cooling import get_pressure_from_temperature
from astronomix.option_classes.simulation_config import (
    DOUBLE_PRECISION,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
)
from astronomix.setup_helpers import attach_per_cell_kernel, build_chemistry_from_network_file

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NETWORK = os.path.join(REPOSITORY_ROOT, "external", "carbox", "data", "simple_latent_tgas.csv")
GAMMA = 5.0 / 3.0
NUM_DEVICES = arguments.devices
NUM_CELLS = arguments.cells

print(f"astronomix from {astronomix.__file__}")
print(f"devices: {jax.devices()}")
if len(jax.devices()) < NUM_DEVICES:
    raise SystemExit(f"{NUM_DEVICES} devices requested, {len(jax.devices())} visible")

# code units as in examples/chemistry_cells.py: m_p cm^-3, K, Myr
KROME = arguments.krome_network is not None
if KROME:
    from astronomix._modules._chemistry._krome_photo import photo_spectrum

    with warnings.catch_warnings():
        # lines that KROME skips too (react_galaxy_ism_plus has three)
        warnings.simplefilter("ignore", UserWarning)
        chemistry_config, chemistry_params, species_names = build_chemistry_from_network_file(
            arguments.krome_network,
            "krome",
            number_density_unit_cgs=1.0,
            temperature_unit_kelvin=1.0,
            time_unit_seconds=3.15576e13,
            cosmic_ray_rate=3.0e-17,
            thermochemistry=True,
            cooling_courant=10.0,
            absolute_tolerance=1.0e-12,
            relative_tolerance=1.0e-6,
            max_steps=2048,
            krome_h2_formation_on_dust=True,
            krome_data_folder=arguments.krome_data,
            krome_metal_table=True,
            krome_photo_bins=(
                None if arguments.krome_data is None else photo_spectrum("draine", 5.0, 13.6, 10)
            ),
            krome_shield_habing_dust=True,
        )
else:
    chemistry_config, chemistry_params, species_names = build_chemistry_from_network_file(
        NETWORK,
        "latent_tgas",
        number_density_unit_cgs=1.0,
        temperature_unit_kelvin=1.0,
        time_unit_seconds=3.15576e13,
        cosmic_ray_rate=3.0e-17,
        fuv_field=1.0e-4,
        visual_extinction=2.0,
        thermochemistry=True,
        cooling_courant=10.0,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-6,
        max_steps=2048,
    )
species_names = tuple(species_names)
# the gas of the box: cold and dense for the carbox network, warm for a KROME network
TEMPERATURE = 8000.0 if KROME else 100.0
HYDROGEN_NUCLEI = 0.3 if KROME else 30.0
if arguments.kernel:
    chemistry_config = attach_per_cell_kernel(
        chemistry_config,
        chemistry_params,
        GAMMA,
        cache_key=("chemistry-multidevice", arguments.krome_network, arguments.krome_data),
    )

periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
HYDROGEN_MASS_FRACTION = chemistry_params.hydrogen_mass_fraction


def setup(t_end, one_step):
    """The box, its configuration and parameters."""
    config = SimulationConfig(
        dimensionality=3,
        box_size=1.0,
        num_cells=NUM_CELLS,
        solver_mode=FINITE_DIFFERENCE if arguments.solver == "fd" else FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        numerical_precision=DOUBLE_PRECISION,
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        progress_bar=False,
        chemistry_config=chemistry_config,
        **(dict(fixed_timestep=True, num_timesteps=1) if one_step else {}),
    )
    params = SimulationParams(t_end=t_end, gamma=GAMMA, chemistry_params=chemistry_params)
    registered_variables = get_registered_variables(config)

    x = (np.arange(NUM_CELLS) + 0.5) / NUM_CELLS
    gx, gy, gz = np.meshgrid(x, x, x, indexing="ij")
    # a density contrast of 100 (30 to 3000 hydrogen nuclei per cm^3 for the
    # carbox network), the blob off the centre so that the devices hold different gas
    r2 = (gx - 0.3) ** 2 + (gy - 0.5) ** 2 + (gz - 0.5) ** 2
    hydrogen_nuclei = jnp.asarray(HYDROGEN_NUCLEI * (1.0 + 99.0 * np.exp(-r2 / 0.02)))
    density = hydrogen_nuclei / HYDROGEN_MASS_FRACTION
    pressure = get_pressure_from_temperature(
        density,
        jnp.full(density.shape, TEMPERATURE),
        HYDROGEN_MASS_FRACTION,
        chemistry_params.metal_mass_fraction,
    )
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=jnp.full(density.shape, 2.0),
        velocity_y=jnp.asarray(np.sin(2 * np.pi * gx)),
        velocity_z=jnp.zeros(density.shape),
        gas_pressure=pressure,
    )
    start = registered_variables.chemistry_species_index
    initial = {"H": 1.0, "C+": 1.4e-4, "O": 3.2e-4, "E": 1.4e-4}
    if KROME:
        # half ionised, with helium and silicon; the electrons balance the charge
        initial = {"H": 0.5, "H+": 0.5, "He": 8.5e-2, "C+": 1.4e-4, "O": 3.2e-4, "Si+": 1.7e-6}
        initial = {name: initial.get(name, 1.0e-12) for name in species_names if name != "E"}
        initial["E"] = sum((name.count("+") - name.count("-")) * value for name, value in initial.items())
    for index, name in enumerate(species_names):
        state = state.at[start + index].set(initial.get(name, 1.0e-12) * hydrogen_nuclei)
    config = finalize_config(config, state.shape)
    return state, config, params, registered_variables


def compare(label, t_end, one_step):
    """Run on one device and on the mesh; print the differences."""
    state, config, params, registered_variables = setup(t_end, one_step)
    single = np.asarray(time_integration(state, config, params, registered_variables))

    # Auto axis types, as cgols.py and pytests/_benchmark_utils.py build the mesh
    mesh = jax.make_mesh(
        (1, NUM_DEVICES, 1, 1),
        (VARAXIS, XAXIS, YAXIS, ZAXIS),
        axis_types=(AxisType.Auto,) * 4,
        devices=jax.devices()[:NUM_DEVICES],
    )
    sharding = jax.NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))
    sharded = np.asarray(
        time_integration(
            jax.device_put(state, sharding), config, params, registered_variables, sharding=sharding
        )
    )

    start = registered_variables.chemistry_species_index
    pressure_index = registered_variables.pressure_index
    initial = np.asarray(state)
    hydrogen_nuclei = single[registered_variables.density_index] * HYDROGEN_MASS_FRACTION
    x_single = single[start : start + len(species_names)] / hydrogen_nuclei
    x_sharded = sharded[start : start + len(species_names)] / hydrogen_nuclei
    # the solver works to rtol 1e-6 and atol 1e-12 cm^-3: species far below the
    # absolute tolerance are noise, so compare those above 1e-10 per H nucleus
    significant = x_single > 1e-10
    dlog = np.abs(np.log10(np.maximum(x_sharded, 1e-300)) - np.log10(np.maximum(x_single, 1e-300)))

    def relative(index):
        return np.max(np.abs(sharded[index] - single[index]) / np.abs(single[index]))

    tracer = "H+" if KROME else "H2"  # the species whose change shows that the chemistry acted
    row = start + species_names.index(tracer)
    print(f"\n{label}")
    print(f"   finite: one device {bool(np.isfinite(single).all())}, {NUM_DEVICES} devices {bool(np.isfinite(sharded).all())}")
    print(
        f"   the chemistry acted: x({tracer}) {np.min(initial[row] / (initial[0] * HYDROGEN_MASS_FRACTION)):.1e} to "
        f"{np.max(initial[row] / (initial[0] * HYDROGEN_MASS_FRACTION)):.1e}"
        f" -> {np.min(x_single[species_names.index(tracer)]):.3e} to {np.max(x_single[species_names.index(tracer)]):.3e}, P/rho "
        f"{np.max(initial[pressure_index] / initial[0]):.1f} -> "
        f"{np.min(single[pressure_index] / single[0]):.1f} to {np.max(single[pressure_index] / single[0]):.1f}"
    )
    print(
        f"   one device against {NUM_DEVICES}: species above 1e-10 per H {dlog[significant].max():.2e} dex, "
        f"density {relative(registered_variables.density_index):.2e}, pressure {relative(pressure_index):.2e} (relative)"
    )


solver_label = "gradsolve Warp kernel" if arguments.kernel else "vmapped Kvaerno5"
solver_label += ", finite-difference solver" if arguments.solver == "fd" else ", finite-volume solver"
if KROME:
    # the sound speed is nine times that of the cold box: steps and run are that much shorter
    solver_label += f", KROME network {os.path.basename(arguments.krome_network)}"
    compare(f"one fixed step, {solver_label}", t_end=0.0002, one_step=True)
    compare(f"short adaptive run, {solver_label}", t_end=0.002, one_step=False)
else:
    compare(f"one fixed step, {solver_label}", t_end=0.002, one_step=True)
    compare(f"short adaptive run, {solver_label}", t_end=0.02 if arguments.kernel else 0.01, one_step=False)
