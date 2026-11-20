"""
A radiation field that differs from cell to cell (``ChemistryConfig.radiation_field_per_cell``).

Run from the repository root (CPU, a few minutes)::

    pytest pytests/chemistry/test_radiation_field.py

1. The network: a factor on its radiation at a call is the network built with that
   many times the radiation (photo-reactions, their heating, the Habing flux of the
   dust terms, the Lyman-Werner flux that dissociates H2).
2. The reaction of a grid of cells, each with its own factor, with Kvaerno5 and with
   the per-cell kernel: every cell is the cell of a run with that factor everywhere.
3. The field: ``fuv_radiation_field`` is the field the cooling module scales its
   photoelectric heating with, term by term.
4. A run through ``time_integration`` in the finite-difference solver: with a field
   that is the same everywhere it is the run of a network built with that field, and
   with a point source the gas near the source is the more ionised.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("diffrax")

from astronomix._modules._chemistry import _krome_photo as photo
from astronomix._modules._chemistry._krome_network import build_krome_network

# Carbon between C and C+ (photoionisation against recombination), H2 against the
# Lyman-Werner flux of a user variable with self-shielding, recombination on dust
# (which sees the Habing flux), a photo-reaction below the hydrogen edge and a
# cosmic-ray ionisation. The cross sections are expressions, so no data base is needed.
NETWORK = """# test network with radiation
@common: user_crate
@common: user_myfluxLW
@format:idx,R,R,P,P,rate
1,H2,H+,H2+,H,6.4d-10
@format:idx,R,R,P,rate
2,H+,E,H,H_recombination_on_dust(n,Tgas)
3,C+,E,C,4.67d-12*(Tgas/3d2)**(-0.6)
4,H+,E,H,3.5d-12*(Tgas/3d2)**(-0.75)
@format:idx,R,P,P,rate
5,H2,H,H,H2_solomonLW(user_myfluxLW)*krome_fshield(n,Tgas)
@photo_start
@format:idx,R,P,P,Tmin,Tmax,rate
6,C,C+,E,>11.26d0,<1.d2,1.2d-17*(energy_eV/11.26d0)**(-2)
7,H2+,H+,H,>2.65d0,<1.d2,1.0d-18
@photo_stop
@CR_start
@format:idx,R,P,P,rate
8,H,H+,E,4.6d-1*user_crate
@CR_stop
"""
#: The Lyman-Werner intensity of the Draine field at 12.87 eV [eV s^-1 cm^-2 sr^-1 Hz^-1]
DRAINE_LYMAN_WERNER = float(photo.draine_intensity(12.87))
FRACTIONS = {"H": 0.9, "H+": 1.0e-3, "H2": 5.0e-2, "H2+": 1.0e-9, "C": 1.0e-4, "C+": 4.0e-5}
COSMIC_RAY_RATE = 3.0e-17


@pytest.fixture(scope="module")
def network_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("krome_radiation") / "react_radiation"
    path.write_text(NETWORK)
    return str(path)


def _abundances(species, hydrogen_nuclei):
    fractions = dict(FRACTIONS)
    fractions["E"] = fractions["H+"] + fractions["H2+"] + fractions["C+"]
    return jnp.asarray([fractions[name] * hydrogen_nuclei for name in species])


def _network(path, field):
    """The network in ``field`` times the Draine field: its photo-reactions, its Habing flux and its Lyman-Werner flux."""
    return build_krome_network(
        path,
        photo_bins=photo.photo_spectrum("draine", 5.0, 13.6, 10, scale=field),
        user_variables={"user_myfluxLW": field * DRAINE_LYMAN_WERNER},
        radiation_user_variables=("user_myfluxLW",),
        shielding="WG11",
        h2_formation_on_dust=True,
    )


# ---------------------------------------------------------------------------
# 1. The network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [60.0, 8.0e3])
def test_a_factor_on_the_radiation_is_the_network_in_that_field(network_file, temperature):
    factor = 37.5
    base, species = _network(network_file, 1.0)
    strong, _ = _network(network_file, factor)
    n = _abundances(species, 20.0)

    plain = np.asarray(base.get_rates(temperature, COSMIC_RAY_RATE, 0.0, 0.0, n))
    scaled = np.asarray(base.get_rates(temperature, COSMIC_RAY_RATE, 0.0, 0.0, n, radiation_scale=factor))
    wanted = np.asarray(strong.get_rates(temperature, COSMIC_RAY_RATE, 0.0, 0.0, n))
    np.testing.assert_allclose(scaled, wanted, rtol=1e-13)
    # the Draine field at 12.87 eV dissociates unshielded H2 at 5.2e-11 per second (Draine & Bertoldi 1996: 5.6e-11)
    assert 1.38e9 * DRAINE_LYMAN_WERNER * 1.60217646e-12 == pytest.approx(5.2e-11, rel=0.02)
    # reactions 5 (Lyman-Werner), 6 and 7 (photo) are proportional to the field; 2 (recombination on dust) depends
    # on it through the charge of the grains; 1, 3, 4 and 8 do not see it
    np.testing.assert_allclose(scaled[[4, 5, 6]], factor * plain[[4, 5, 6]], rtol=1e-13)
    assert plain[4] > 0.0 and plain[5] > 0.0 and plain[6] > 0.0
    assert scaled[1] != plain[1]
    np.testing.assert_array_equal(scaled[[0, 2, 3, 7]], plain[[0, 2, 3, 7]])
    # no factor is the network as it was built; a factor of one gives the same numbers
    np.testing.assert_array_equal(np.asarray(base.get_rates(temperature, COSMIC_RAY_RATE, 0.0, 0.0, n, radiation_scale=None)), plain)
    np.testing.assert_allclose(np.asarray(base.get_rates(temperature, COSMIC_RAY_RATE, 0.0, 0.0, n, radiation_scale=1.0)), plain, rtol=1e-15)

    # the heating: the photoelectric effect and the photo-reactions follow the field
    heating_plain = base.heating_terms(n, temperature, COSMIC_RAY_RATE, 0.0)
    heating_scaled = base.heating_terms(n, temperature, COSMIC_RAY_RATE, 0.0, radiation_scale=factor)
    heating_wanted = strong.heating_terms(n, temperature, COSMIC_RAY_RATE, 0.0)
    for name in heating_wanted:
        assert float(heating_scaled[name]) == pytest.approx(float(heating_wanted[name]), rel=1e-12, abs=1e-60), name
    assert float(heating_scaled["heating_photo"]) == pytest.approx(factor * float(heating_plain["heating_photo"]), rel=1e-12)
    assert float(heating_scaled["heating_photoelectric"]) > float(heating_plain["heating_photoelectric"]) > 0.0
    assert float(heating_scaled["heating_cosmic_rays"]) == float(heating_plain["heating_cosmic_rays"])

    # the right-hand side and the temperature equation
    np.testing.assert_allclose(
        np.asarray(base(0.0, n, temperature, COSMIC_RAY_RATE, 0.0, 0.0, 1.0, 0.0, radiation_scale=factor)),
        np.asarray(strong(0.0, n, temperature, COSMIC_RAY_RATE, 0.0, 0.0, 1.0, 0.0)),
        rtol=1e-12,
    )
    assert float(base.temperature_derivative(n, temperature, 5.0 / 3.0, COSMIC_RAY_RATE, 0.0, radiation_scale=factor)) == pytest.approx(
        float(strong.temperature_derivative(n, temperature, 5.0 / 3.0, COSMIC_RAY_RATE, 0.0)), rel=1e-12
    )


def test_a_radiation_variable_must_be_a_variable_of_the_network(network_file):
    with pytest.raises(ValueError, match="radiation_user_variables"):
        build_krome_network(network_file, user_variables={"user_myfluxLW": 1.0}, radiation_user_variables=("user_other",))


# ---------------------------------------------------------------------------
# 2. A grid of cells, each with its own field
# ---------------------------------------------------------------------------

FIELDS = (0.0024, 1.0, 30.0, 900.0)
GAMMA = 5.0 / 3.0
TIME_STEP_SECONDS = 3.0e11  # ten thousand years


def _chemistry(path, field, per_cell=False, kernel=False, time_unit_seconds=TIME_STEP_SECONDS):
    from astronomix.setup_helpers import attach_per_cell_kernel, build_chemistry_from_network_file

    chemistry_config, chemistry_params, species = build_chemistry_from_network_file(
        path, "krome",
        number_density_unit_cgs=1.0, temperature_unit_kelvin=1.0, time_unit_seconds=time_unit_seconds,
        cosmic_ray_rate=COSMIC_RAY_RATE, thermochemistry=True, cooling_courant=1e3, floor_temperature=10.0,
        absolute_tolerance=1e-14, relative_tolerance=1e-8, max_steps=4096,
        krome_h2_formation_on_dust=True, krome_shielding="WG11",
        krome_photo_bins=photo.photo_spectrum("draine", 5.0, 13.6, 10, scale=field),
        krome_user_variables={"user_myfluxLW": field * DRAINE_LYMAN_WERNER},
        krome_radiation_user_variables=("user_myfluxLW",),
        temperature_from_species=True,
    )  # fmt: skip
    chemistry_config = chemistry_config._replace(radiation_field_per_cell=per_cell)
    if kernel:
        chemistry_config = attach_per_cell_kernel(
            chemistry_config, chemistry_params, GAMMA, cache_key=("pytest-radiation", path, field, per_cell)
        )
    return chemistry_config, chemistry_params, tuple(species)


def _row_of_cells(chemistry_config, chemistry_params, species, number_of_cells):
    """Identical cells in a row: density, velocity, pressure and the species (30 nuclei per cm^3 at 300 K)."""
    from astronomix import SimulationParams
    from astronomix.variable_registry.registered_variables import RegisteredVariables

    registered_variables = RegisteredVariables(
        density_index=0, velocity_index=1, pressure_index=2, num_vars=3 + len(species),
        chemistry_species_index=3, num_chemical_species=len(species), chemistry_species_active=True,
    )  # fmt: skip
    abundances = _abundances(species, 30.0)
    particles = float(abundances.sum())
    cell = jnp.concatenate([jnp.asarray([30.0 * 1.4, 0.0, particles * 300.0]), abundances])
    state = jnp.repeat(cell[:, None], number_of_cells, axis=1)
    return state, registered_variables, SimulationParams(gamma=GAMMA, chemistry_params=chemistry_params)


@pytest.mark.parametrize("kernel", [False, True], ids=["kvaerno5", "kernel"])
def test_every_cell_reacts_in_its_own_field(network_file, kernel):
    from astronomix._modules._chemistry._chemistry import update_chemistry

    if kernel:
        pytest.importorskip("gradsolve")
        pytest.importorskip("warp")
    chemistry_config, chemistry_params, species = _chemistry(network_file, 1.0, per_cell=True, kernel=kernel)
    state, registered_variables, params = _row_of_cells(chemistry_config, chemistry_params, species, len(FIELDS))
    reacted = np.asarray(update_chemistry(state, registered_variables, chemistry_config, params, 1.0, radiation_scale=jnp.asarray(FIELDS)))
    assert np.isfinite(reacted).all()

    for cell, field in enumerate(FIELDS):
        # the same cell in a run whose network is built in that field, the same in every cell
        uniform_config, uniform_params, _ = _chemistry(network_file, field, kernel=kernel)
        one_state, _, uniform = _row_of_cells(uniform_config, uniform_params, species, 1)
        wanted = np.asarray(update_chemistry(one_state, registered_variables, uniform_config, uniform, 1.0))[:, 0]
        significant = np.abs(wanted) > 1e-12 * 30.0
        np.testing.assert_allclose(reacted[significant, cell], wanted[significant], rtol=2e-7, err_msg=f"field {field}")

    # the field matters: more of it leaves less H2, more C+ and warmer gas
    x_h2 = reacted[3 + species.index("H2")] / 30.0
    carbon_ions = reacted[3 + species.index("C+")] / (reacted[3 + species.index("C+")] + reacted[3 + species.index("C")])
    temperature = reacted[2] / reacted[3:].sum(axis=0)
    assert (np.diff(x_h2) < 0.0).all() and x_h2[0] > 30.0 * x_h2[-1]
    assert (np.diff(carbon_ions) > 0.0).all() and carbon_ions[-1] > 0.99 > carbon_ions[0]
    assert (np.diff(temperature) > 0.0).all()


def test_a_kernel_is_built_with_or_without_the_field_of_the_cell(network_file):
    """A kernel built for a field per cell cannot be called without one, and the other way round."""
    pytest.importorskip("gradsolve")
    pytest.importorskip("warp")
    from astronomix._modules._chemistry._chemistry import update_chemistry

    with_field, params_with, species = _chemistry(network_file, 1.0, per_cell=True, kernel=True)
    without_field, params_without, _ = _chemistry(network_file, 1.0, per_cell=False, kernel=True)
    assert with_field.stiff_core_fn.per_cell_parameter and not without_field.stiff_core_fn.per_cell_parameter
    state, registered_variables, params = _row_of_cells(with_field, params_with, species, 2)
    with pytest.raises(TypeError, match="with a parameter per cell"):
        update_chemistry(state, registered_variables, with_field, params, 1.0)
    with pytest.raises(TypeError, match="without a parameter per cell"):
        update_chemistry(state, registered_variables, without_field, params, 1.0, radiation_scale=jnp.ones(2))
    # one field everywhere through the kernel with the factor is the kernel without it
    np.testing.assert_allclose(
        np.asarray(update_chemistry(state, registered_variables, with_field, params, 1.0, radiation_scale=jnp.ones(2))),
        np.asarray(update_chemistry(state, registered_variables, without_field, params, 1.0)),
        rtol=1e-12,
    )


def test_what_is_not_implemented_says_so(network_file):
    from astronomix._modules._chemistry._chemistry import update_chemistry

    chemistry_config, chemistry_params, species = _chemistry(network_file, 1.0, per_cell=True)
    state, registered_variables, params = _row_of_cells(chemistry_config, chemistry_params, species, 2)
    with pytest.raises(NotImplementedError, match="radiation field per cell"):
        update_chemistry(state, registered_variables, chemistry_config, params, 1.0, jnp.ones(2), radiation_scale=jnp.ones(2))


# ---------------------------------------------------------------------------
# 3. The field
# ---------------------------------------------------------------------------


def _sources(**changes):
    from astronomix._modules._cooling.cooling_options import FUVSourceParams

    radius, height = np.meshgrid(np.arange(6) * 0.25, -1.0 + np.arange(9) * 0.25, indexing="ij")
    return FUVSourceParams(
        source_positions=jnp.asarray([[0.1, -0.05, 0.02]]),
        source_weights=jnp.ones((1,)),
        softening_length=0.3,
        luminosity_times=jnp.asarray([0.0, 1.0]),
        luminosity_values=jnp.asarray([40.0, 160.0]),
        background=0.0024,
        dust_opacity=2.0,
        los_tau_table=jnp.asarray(3.0 * np.exp(-((height / 0.2) ** 2)) * radius),
        los_r_spacing=0.25,
        los_z_min=-1.0,
        los_z_spacing=0.25,
        disk_sfr_coefficient=0.7,
        disk_sfr_index=1.4,
    )._replace(**changes)


class _Grid:
    """The cell centres of a small box, as the helper data carries them."""

    def __init__(self, shape, box=2.0):
        centres = [(np.arange(n) + 0.5) / n * box - 0.5 * box for n in shape]
        self.cell_centers_x, self.cell_centers_y, self.cell_centers_z = (jnp.asarray(c) for c in centres)


def test_the_field_is_the_field_of_the_cooling_module():
    from astronomix._modules._cooling._cooling import (
        fuv_attenuation_factor,
        fuv_disk_field,
        fuv_heating_factor,
        fuv_line_of_sight_factor,
        fuv_radiation_field,
    )
    from astronomix._modules._cooling.cooling_options import (
        FUV_ATTENUATION_LINE_OF_SIGHT,
        FUV_ATTENUATION_NONE,
        FUV_ATTENUATION_SLAB,
    )

    shape = (6, 5, 8)
    grid, sources, time = _Grid(shape), _sources(), 0.4
    rng = np.random.default_rng(3)
    density = jnp.asarray(10.0 ** rng.uniform(-2, 1, size=shape))
    pressure = jnp.asarray(10.0 ** rng.uniform(-1, 1, size=shape))
    centres = (grid.cell_centers_x, grid.cell_centers_y, grid.cell_centers_z)
    spacing = grid.cell_centers_z[1] - grid.cell_centers_z[0]
    diluted = fuv_heating_factor(sources, *centres, time)

    # the point source alone, diluted with distance: 4 pi (d^2 + r_s^2) J / J_0 is its luminosity at that time
    plain = np.asarray(fuv_radiation_field(density, pressure, sources, FUV_ATTENUATION_NONE, False, grid, time))
    np.testing.assert_array_equal(plain, np.asarray(diluted))
    distance2 = (
        (np.asarray(centres[0]) - 0.1)[:, None, None] ** 2
        + (np.asarray(centres[1]) + 0.05)[None, :, None] ** 2
        + (np.asarray(centres[2]) - 0.02)[None, None, :] ** 2
    )
    np.testing.assert_allclose((plain - 0.0024) * 4.0 * np.pi * (distance2 + 0.09), 40.0 + 0.4 * 120.0, rtol=1e-12)

    # attenuated along the line of sight (the background is not), plus the disk
    transmitted = fuv_line_of_sight_factor(sources, *centres)
    wanted = (diluted - sources.background) * transmitted + sources.background + fuv_disk_field(density, sources, spacing)
    np.testing.assert_array_equal(
        np.asarray(fuv_radiation_field(density, pressure, sources, FUV_ATTENUATION_LINE_OF_SIGHT, True, grid, time)), np.asarray(wanted)
    )
    assert float(jnp.min(transmitted)) < 0.2 and float(jnp.max(transmitted)) > 0.999
    # the slab factor of the column of the cell
    slab = fuv_attenuation_factor(density, pressure, sources, FUV_ATTENUATION_SLAB, spacing)
    np.testing.assert_array_equal(
        np.asarray(fuv_radiation_field(density, pressure, sources, FUV_ATTENUATION_SLAB, False, grid, time)),
        np.asarray((diluted - sources.background) * slab + sources.background),
    )


# ---------------------------------------------------------------------------
# 4. A run in the finite-difference solver
# ---------------------------------------------------------------------------


RUN_TIME_UNIT_SECONDS = 1.0e13


def _run(network_file, field, per_cell, sources=None, attenuation=None, shape=(8, 8, 8), t_end=0.012, steps=3):
    """Three steps of 1.3e3 years each. The sound speed is 18 in the units of the box (P = n T, rho = 1.4 n), so a step of
    0.004 is a Courant number of 0.3 on cells of 0.25."""
    from astronomix import (
        SimulationConfig,
        SimulationParams,
        construct_primitive_state,
        finalize_config,
        get_registered_variables,
        time_integration,
    )
    from astronomix._modules._cooling.cooling_options import FUV_ATTENUATION_NONE
    from astronomix.option_classes.simulation_config import (
        FINITE_DIFFERENCE,
        NATIVE_JAX,
        PERIODIC_BOUNDARY,
        BackendConfig,
        BoundarySettings,
        BoundarySettings1D,
    )

    chemistry_config, chemistry_params, species = _chemistry(network_file, field, per_cell=per_cell, time_unit_seconds=RUN_TIME_UNIT_SECONDS)
    if per_cell:
        chemistry_config = chemistry_config._replace(
            radiation_attenuation=FUV_ATTENUATION_NONE if attenuation is None else attenuation
        )
        chemistry_params = chemistry_params._replace(fuv_source_params=sources)
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        dimensionality=3, solver_mode=FINITE_DIFFERENCE, backend_config=BackendConfig(backend=NATIVE_JAX),
        box_size=2.0, num_cells=shape[0], boundary_settings=BoundarySettings(periodic, periodic, periodic),
        progress_bar=False, chemistry_config=chemistry_config, fixed_timestep=True, num_timesteps=steps,
    )  # fmt: skip
    params = SimulationParams(t_end=t_end, gamma=GAMMA, chemistry_params=chemistry_params)
    registered_variables = get_registered_variables(config)
    abundances = _abundances(species, 30.0)
    density = jnp.full(shape, 30.0 * 1.4)
    state = construct_primitive_state(
        config=config, registered_variables=registered_variables, density=density,
        velocity_x=jnp.full(shape, 0.3), velocity_y=jnp.zeros(shape), velocity_z=jnp.zeros(shape),
        gas_pressure=jnp.full(shape, float(abundances.sum()) * 300.0),
    )  # fmt: skip
    start = registered_variables.chemistry_species_index
    for index in range(len(species)):
        state = state.at[start + index].set(abundances[index])
    config = finalize_config(config, state.shape)
    return np.asarray(time_integration(state, config, params, registered_variables)), registered_variables, species


def test_a_run_with_the_field_of_every_cell(network_file):
    from astronomix._modules._cooling.cooling_options import FUVSourceParams

    # the same field everywhere (no source, a background of 12): the run of a network built in 12 Draine fields
    everywhere = FUVSourceParams(luminosity_times=jnp.zeros((1,)), luminosity_values=jnp.zeros((1,)), background=12.0)
    per_cell, registered_variables, species = _run(network_file, 1.0, True, sources=everywhere)
    uniform, _, _ = _run(network_file, 12.0, False)
    assert np.isfinite(per_cell).all()
    np.testing.assert_allclose(per_cell, uniform, rtol=1e-9, atol=1e-30)

    # a point source in the middle of the box (the box runs from 0 to 2): the gas next to it is in a field of 270
    # Draine fields, the gas in the corners in one of 10
    source = FUVSourceParams(
        source_positions=jnp.ones((1, 3)), source_weights=jnp.ones((1,)), softening_length=0.2,
        luminosity_times=jnp.zeros((1,)), luminosity_values=jnp.asarray([300.0]), background=0.0024,
    )  # fmt: skip
    lit, _, _ = _run(network_file, 1.0, True, sources=source)
    assert np.isfinite(lit).all()
    start = registered_variables.chemistry_species_index
    h2 = lit[start + species.index("H2")]
    temperature = lit[registered_variables.pressure_index] / lit[start:].sum(axis=0)
    near, far = (slice(3, 5),) * 3, (0, 0, 0)
    # less H2 is left next to the source (about half as much after 4000 years), and the gas there is warmer
    assert h2[near].mean() < 0.7 * h2[far] and h2.max() < 1.5
    assert temperature[near].mean() > 1.03 * temperature[far]
    # the source is mirror symmetric in y and z (the gas moves along x)
    np.testing.assert_allclose(h2, h2[:, ::-1, :], rtol=1e-9)
    np.testing.assert_allclose(h2, h2[:, :, ::-1], rtol=1e-9)
