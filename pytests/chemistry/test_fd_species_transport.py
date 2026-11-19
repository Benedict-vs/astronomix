"""
Tests for the chemistry in the finite-difference solver: the transport of the
species (``astronomix/_modules/_chemistry/_species_transport.py``), the reaction
step in that solver, and the temperature from the particles of the network.

1. The registry: the species are the last block of the finite-difference state,
   behind the dual-energy row and the passive scalars.
2. Transport without reactions on a periodic box with an over-dense blob in a
   shear flow: gas of one composition keeps it to round-off; with fronts of He,
   He+ and He++ in the same cells the helium atoms and the charge per unit mass
   stay what they were to round-off, and without the restoration they do not
   (which is why the restoration exists).
3. The restoration on its own, with a species of two elements.
4. Transport and reaction together: a linear network against its exact
   solution, with and without chemistry sub-cycling, and a two-body network
   against the finite-volume solver on the same problem.
5. Outflow through open and through "diode" boundaries (ghost cells).
6. The temperature the chemistry reads and writes: from the particles of the
   network (``temperature_from_species``) and from the fixed mean molecular
   weight, on a recombining, cooling gas with an exact solution; gas hotter
   than the range of the rates keeps its excess; under dual energy the carried
   internal energy follows the reaction.
7. The density floor of the finite-difference solver carries the species.
8. The Pallas backend (interpret mode) against native JAX.

Everything is small and runs on the CPU unless a platform is chosen in the
environment: the file neither waits for nor lands on a shared GPU. Needs
diffrax (the reactions use the vmapped Kvaerno5).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# ruff: noqa: E402

import numpy as np
import pytest

pytest.importorskip("diffrax")

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp

from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)
from astronomix._modules._chemistry._chemistry import (
    chemistry_temperature_kelvin,
    update_chemistry,
)
from astronomix._modules._chemistry._species_transport import (
    conserved_totals,
    restore_conserved_totals,
)
from astronomix._modules._chemistry.chemistry_options import (
    ChemistryConfig,
    ChemistryParams,
)
from astronomix._modules._iteration_level_updates import _apply_density_floor
from astronomix.option_classes.simulation_config import (
    DOUBLE_PRECISION,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    MUSCL,
    NATIVE_JAX,
    OPEN_BOUNDARY,
    OPEN_BOUNDARY_DIODE,
    PALLAS,
    PERIODIC_BOUNDARY,
    SPLIT,
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
)
from astronomix.setup_helpers.chemistry_setup import species_composition

HELIUM = ("He", "He+", "He++", "E")
HELIUM_PER_MASS = 0.1


# -----------------------------------------------------------------------------
# ============================ ↓ Toy networks ↓ ================================
# -----------------------------------------------------------------------------

class Frozen(eqx.Module):
    """No reactions: the species are only transported."""

    def __call__(self, time, abundances, temperature, cr_rate, fuv, av, modifier_a, modifier_b):
        return jnp.zeros_like(abundances)


class Linear(eqx.Module):
    """A <-> B at constant rates; B / (A + B) has an exact solution."""

    forward: jax.Array
    backward: jax.Array

    def __call__(self, time, abundances, temperature, cr_rate, fuv, av, modifier_a, modifier_b):
        rate = self.forward * abundances[0] - self.backward * abundances[1]
        return jnp.stack([-rate, rate])


class TwoBody(eqx.Module):
    """A + A -> B: the rate depends on the density, so the transport matters."""

    coefficient: jax.Array

    def __call__(self, time, abundances, temperature, cr_rate, fuv, av, modifier_a, modifier_b):
        rate = self.coefficient * abundances[0] ** 2
        return jnp.stack([-2.0 * rate, rate])


class RecombiningCoolingGas(eqx.Module):
    """H+ + E -> H at a constant coefficient, and dT/dt = -T / cooling_time.

    Both have exact solutions; the particle number falls as the gas recombines,
    which is what tells the two temperature conventions apart.
    """

    coefficient: jax.Array
    cooling_time: jax.Array
    own_thermochemistry: bool = eqx.field(static=True, default=True)

    def __call__(self, time, abundances, temperature, cr_rate, fuv, av, modifier_a, modifier_b):
        rate = self.coefficient * abundances[1] * abundances[2]
        return jnp.stack([rate, -rate, -rate])

    def temperature_derivative(self, abundances, temperature, adiabatic_index, cr_rate, fuv):
        return -temperature / self.cooling_time


# -----------------------------------------------------------------------------
# ============================ ↑ Toy networks ↑ ================================
# -----------------------------------------------------------------------------


def _chemistry(species, network, composition=True, **overrides):
    """The chemistry containers of a toy network (code units = CGS units)."""
    chemistry_config = ChemistryConfig(
        chemistry=True,
        number_of_chemical_species=len(species),
        species_names=tuple(species),
        number_of_reactions=1,
        electron_index=species.index("E") if "E" in species else -1,
        **(species_composition(species) if composition else {}),
    )._replace(**overrides)
    chemistry_params = ChemistryParams(
        network=network,
        atol=1e-12,
        rtol=1e-9,
        rate_modifier_a=jnp.ones(1),
        rate_modifier_b=jnp.zeros(1),
    )
    return chemistry_config, chemistry_params


def _boundaries(kind):
    edge = {"periodic": PERIODIC_BOUNDARY, "open": OPEN_BOUNDARY, "diode": OPEN_BOUNDARY_DIODE}[kind]
    one_axis = BoundarySettings1D(left_boundary=edge, right_boundary=edge)
    return BoundarySettings(one_axis, one_axis, one_axis)


def _blob_in_shear(cells, dimensionality):
    """Density, velocities and the cell coordinates of an over-dense blob that
    crosses a periodic unit box in a shear flow."""
    axis = (np.arange(cells) + 0.5) / cells
    coordinates = np.meshgrid(*([axis] * dimensionality), indexing="ij")
    radius_squared = sum((coordinate - 0.5) ** 2 for coordinate in coordinates)
    density = 1.0 + 2.0 * np.exp(-radius_squared / 0.02)
    velocities = [np.full_like(density, 0.5), 0.3 * np.sin(2.0 * np.pi * coordinates[0])]
    if dimensionality == 3:
        velocities.append(np.zeros_like(density))
    return density, velocities, coordinates


def _simulate(
    solver,
    chemistry_config,
    chemistry_params,
    abundances_per_mass,
    *,
    cells=48,
    dimensionality=2,
    t_end=0.1,
    boundaries="periodic",
    velocities=None,
    **config_overrides,
):
    """Run the blob problem and return ``(initial, final, registered_variables)``.

    ``abundances_per_mass(coordinates)`` gives the initial ``n_i / rho`` of
    every species; the states are returned as the solver holds them (with ghost
    cells for the open boundaries).
    """
    density, blob_velocities, coordinates = _blob_in_shear(cells, dimensionality)
    velocities = blob_velocities if velocities is None else velocities(coordinates)
    scheme = (
        dict(solver_mode=FINITE_DIFFERENCE)
        if solver == "fd"
        else dict(solver_mode=FINITE_VOLUME, split=SPLIT, time_integrator=MUSCL)
    )
    config = SimulationConfig(
        dimensionality=dimensionality,
        box_size=1.0,
        num_cells=cells,
        progress_bar=False,
        numerical_precision=DOUBLE_PRECISION,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        boundary_settings=_boundaries(boundaries),
        chemistry_config=chemistry_config,
        **scheme,
    )._replace(**config_overrides)
    params = SimulationParams(t_end=t_end, chemistry_params=chemistry_params)
    registered_variables = get_registered_variables(config)
    velocity_arguments = {
        f"velocity_{name}": jnp.asarray(velocity) for name, velocity in zip("xyz", velocities)
    }
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.asarray(density),
        gas_pressure=jnp.ones_like(jnp.asarray(density)),
        **velocity_arguments,
    )
    start = registered_variables.chemistry_species_index
    species = np.asarray(abundances_per_mass(coordinates)) * density[None, ...]
    state = state.at[start : start + species.shape[0]].set(jnp.asarray(species))
    config = finalize_config(config, state.shape)
    initial = np.asarray(state)
    final = np.asarray(time_integration(state, config, params, registered_variables))
    return initial, final, registered_variables


def _per_mass(state, registered_variables):
    """The abundances per unit mass ``n_i / rho`` of a state."""
    start = registered_variables.chemistry_species_index
    stop = start + registered_variables.num_chemical_species
    return state[start:stop] / state[registered_variables.density_index][None, ...]


def _helium_fronts(coordinates):
    """He, He+ and He++ with fronts in the same cells, neutral, at a constant
    number of helium atoms per unit mass."""
    x, y = coordinates[0], coordinates[1]
    neutral = 0.5 * (1.0 + np.tanh((0.3 - np.hypot(x - 0.5, y - 0.5)) / 0.03))
    doubly = (1.0 - neutral) * 0.5 * (1.0 + np.tanh((x - 0.6) / 0.03))
    singly = 1.0 - neutral - doubly
    return HELIUM_PER_MASS * np.stack([neutral, singly, doubly, singly + 2.0 * doubly])


# -----------------------------------------------------------------------------
# ============================== ↓ 1. Registry ↓ ===============================
# -----------------------------------------------------------------------------

def test_species_are_the_last_block_of_the_finite_difference_state():
    chemistry_config, _ = _chemistry(HELIUM, Frozen())
    config = SimulationConfig(
        dimensionality=3,
        solver_mode=FINITE_DIFFERENCE,
        dual_energy=True,
        num_passive_scalars=2,
        chemistry_config=chemistry_config,
    )
    registered_variables = get_registered_variables(config)
    # density, three velocities, pressure | g | two scalars | four species
    assert registered_variables.internal_energy_index == 5
    assert registered_variables.passive_scalar_index == 6
    assert registered_variables.chemistry_species_index == 8
    assert registered_variables.num_chemical_species == 4
    assert registered_variables.chemistry_species_active
    assert registered_variables.num_vars == 12

    without = get_registered_variables(config._replace(chemistry_config=ChemistryConfig()))
    assert not without.chemistry_species_active
    assert without.num_vars == 8


# -----------------------------------------------------------------------------
# ======================= ↓ 2. Transport, no reactions ↓ =======================
# -----------------------------------------------------------------------------

def test_gas_of_one_composition_keeps_it():
    # the restoration is off: this is the advection alone
    chemistry_config, chemistry_params = _chemistry(
        HELIUM, Frozen(), conserve_elements_in_transport=False
    )
    composition = np.array([0.02, 0.05, 0.03, 0.11])

    def uniform(coordinates):
        return composition.reshape(-1, 1, 1) * np.ones_like(coordinates[0])[None, ...]

    initial, final, registered_variables = _simulate("fd", chemistry_config, chemistry_params, uniform)
    density = registered_variables.density_index
    assert np.abs(final[density] - initial[density]).max() > 0.1  # the blob has moved
    per_mass = _per_mass(final, registered_variables)
    np.testing.assert_allclose(per_mass, composition.reshape(-1, 1, 1) * np.ones_like(per_mass), rtol=1e-12)


@pytest.fixture(scope="module")
def helium_fronts():
    """The helium fronts after the transport, with and without the restoration
    of the conserved totals."""
    results = {}
    for restore in (True, False):
        chemistry_config, chemistry_params = _chemistry(
            HELIUM, Frozen(), conserve_elements_in_transport=restore
        )
        results[restore] = _simulate("fd", chemistry_config, chemistry_params, _helium_fronts)
    return results


def test_fronts_move_and_stay_within_their_bounds(helium_fronts):
    initial, final, registered_variables = helium_fronts[True]
    before = _per_mass(initial, registered_variables)
    after = _per_mass(final, registered_variables)
    assert np.isfinite(final).all()
    assert np.abs(after - before).max() > 0.2 * HELIUM_PER_MASS  # the fronts have moved
    assert after.min() >= 0.0
    # a maximum principle per species, up to the restoration (a few per cent of a front)
    assert (after[:3].max(axis=(1, 2)) <= 1.05 * HELIUM_PER_MASS).all()


def test_atoms_and_charge_per_unit_mass_are_kept(helium_fronts):
    _, final, registered_variables = helium_fronts[True]
    helium, singly, doubly, electrons = _per_mass(final, registered_variables)
    np.testing.assert_allclose(helium + singly + doubly, HELIUM_PER_MASS, rtol=1e-12)
    np.testing.assert_allclose(electrons, singly + 2.0 * doubly, rtol=1e-11, atol=1e-13)


def test_without_the_restoration_the_atoms_do_not_add_up(helium_fronts):
    _, final, registered_variables = helium_fronts[False]
    helium, singly, doubly, electrons = _per_mass(final, registered_variables)
    error = np.abs(helium + singly + doubly - HELIUM_PER_MASS).max() / HELIUM_PER_MASS
    charge_error = np.abs(electrons - singly - 2.0 * doubly).max() / HELIUM_PER_MASS
    # three species with fronts in the same cells: each has its own WENO weights
    assert error > 1e-4
    assert charge_error > 1e-4


# -----------------------------------------------------------------------------
# ============================ ↓ 3. Restoration ↓ ==============================
# -----------------------------------------------------------------------------

def test_restoration_with_a_species_of_two_elements():
    species = ("C", "C+", "O", "CO", "E")
    chemistry_config, _ = _chemistry(species, Frozen())
    assert chemistry_config.species_element_names == ("C", "O")
    assert chemistry_config.species_element_matrix == ((1, 1, 0, 1, 0), (0, 0, 1, 1, 0))
    assert chemistry_config.species_charges == (0, 1, 0, 0, -1)

    key = jax.random.PRNGKey(0)
    abundances = jax.random.uniform(key, (5, 6, 7), minval=0.1, maxval=1.0)
    abundances = abundances.at[4].set(abundances[1])  # neutral
    elements, charge = conserved_totals(abundances, chemistry_config)
    np.testing.assert_allclose(elements[0], abundances[0] + abundances[1] + abundances[3], rtol=1e-14)
    np.testing.assert_allclose(charge, 0.0, atol=1e-15)

    # species that drifted by up to 10 per cent go back to the totals
    drift = 1.0 + 0.1 * (jax.random.uniform(jax.random.PRNGKey(1), abundances.shape) - 0.5)
    restored = restore_conserved_totals(abundances * drift, elements, charge, chemistry_config)
    restored_elements, restored_charge = conserved_totals(restored, chemistry_config)
    drifted_elements, _ = conserved_totals(abundances * drift, chemistry_config)
    # CO couples carbon and oxygen: the element swept last is exact, the other
    # one keeps a small part of its mismatch
    np.testing.assert_allclose(restored_elements[1], elements[1], rtol=1e-13)
    mismatch_before = float(jnp.abs(drifted_elements[0] / elements[0] - 1.0).max())
    mismatch_after = float(jnp.abs(restored_elements[0] / elements[0] - 1.0).max())
    assert mismatch_before > 0.02
    assert mismatch_after < mismatch_before / 100.0
    np.testing.assert_allclose(restored_charge, 0.0, atol=1e-14)
    assert float(restored.min()) > 0.0

    # species that carry the totals already are left as they are
    untouched = restore_conserved_totals(abundances, elements, charge, chemistry_config)
    np.testing.assert_allclose(untouched, abundances, rtol=1e-13)

    # an element that a cell does not have, and species that are zero, stay zero
    without_oxygen = abundances.at[2].set(0.0).at[3].set(0.0)
    elements, charge = conserved_totals(without_oxygen, chemistry_config)
    restored = restore_conserved_totals(without_oxygen, elements, charge, chemistry_config)
    assert float(jnp.abs(restored[2:4]).max()) == 0.0
    assert bool(jnp.isfinite(restored).all())


# -----------------------------------------------------------------------------
# ====================== ↓ 4. Transport and reaction ↓ =========================
# -----------------------------------------------------------------------------

def _all_in_the_first_species(coordinates):
    ones = np.ones_like(coordinates[0])
    return np.stack([ones, 1e-12 * ones])


@pytest.mark.parametrize("step_target", [0.0, 0.02])
def test_linear_network_follows_its_exact_solution(step_target):
    forward, backward, t_end = 30.0, 10.0, 0.05
    chemistry_config, chemistry_params = _chemistry(
        ("A", "B"),
        Linear(jnp.asarray(forward), jnp.asarray(backward)),
        composition=False,
        chemistry_step_target=step_target,
    )
    initial, final, registered_variables = _simulate(
        "fd", chemistry_config, chemistry_params, _all_in_the_first_species, t_end=t_end
    )
    first, second = _per_mass(final, registered_variables)
    exact = forward / (forward + backward) * (1.0 - np.exp(-(forward + backward) * t_end))
    np.testing.assert_allclose(second / (first + second), exact, rtol=1e-6)
    # the species still add up to the density
    np.testing.assert_allclose(first + second, 1.0 + 1e-12, rtol=1e-11)


def test_two_body_network_agrees_with_the_finite_volume_solver():
    results = {}
    for solver in ("fd", "fv"):
        chemistry_config, chemistry_params = _chemistry(
            ("A", "B"), TwoBody(jnp.asarray(4.0)), composition=False
        )
        _, final, registered_variables = _simulate(
            solver, chemistry_config, chemistry_params, _all_in_the_first_species, cells=64, t_end=0.2
        )
        first, second = _per_mass(final, registered_variables)
        results[solver] = (second, final[registered_variables.density_index])
    fraction_fd, density_fd = results["fd"]
    fraction_fv, density_fv = results["fv"]
    # the reaction is faster in the blob: the product is not uniform
    assert fraction_fd.max() - fraction_fd.min() > 0.05
    # two schemes of different order on 64^2 cells: per cent level
    assert np.abs(fraction_fd - fraction_fv).mean() < 0.01 * fraction_fv.mean()
    assert np.abs(fraction_fd - fraction_fv).max() < 0.1 * fraction_fv.max()
    # total amount of the product
    total_fd, total_fv = (fraction_fd * density_fd).sum(), (fraction_fv * density_fv).sum()
    assert abs(total_fd - total_fv) < 5e-3 * total_fv


# -----------------------------------------------------------------------------
# ========================= ↓ 5. Open boundaries ↓ =============================
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["open", "diode"])
def test_outflow_through_open_boundaries(kind):
    chemistry_config, chemistry_params = _chemistry(HELIUM, Frozen())

    def outward(coordinates):
        # the gas leaves the box on all sides
        return [1.5 * (coordinate - 0.5) for coordinate in coordinates]

    initial, final, registered_variables = _simulate(
        "fd", chemistry_config, chemistry_params, _helium_fronts,
        boundaries=kind, velocities=outward, t_end=0.15,
    )
    assert np.isfinite(final).all()
    density = registered_variables.density_index
    assert final[density].sum() < 0.9 * initial[density].sum()  # gas has left
    # the totals hold in every cell the solver carries, the ghost cells included
    helium, singly, doubly, electrons = _per_mass(final, registered_variables)
    np.testing.assert_allclose(helium + singly + doubly, HELIUM_PER_MASS, rtol=1e-12)
    np.testing.assert_allclose(electrons, singly + 2.0 * doubly, rtol=1e-11, atol=1e-13)
    assert min(helium.min(), singly.min(), doubly.min()) >= 0.0


# -----------------------------------------------------------------------------
# =========================== ↓ 6. Temperature ↓ ===============================
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("from_species", [True, False])
def test_temperature_the_chemistry_reads_and_writes(from_species):
    species = ("H", "H+", "E")
    coefficient, cooling_time, step = 0.5, 4.0, 1.0
    chemistry_config, chemistry_params = _chemistry(
        species,
        RecombiningCoolingGas(jnp.asarray(coefficient), jnp.asarray(cooling_time)),
        thermochemistry=True,
        temperature_from_species=from_species,
    )
    # no limit on the temperature change of the step, no floor in the way
    chemistry_params = chemistry_params._replace(cooling_courant=1.0, floor_temperature=1.0)
    config = SimulationConfig(
        dimensionality=1,
        num_cells=4,
        solver_mode=FINITE_DIFFERENCE,
        numerical_precision=DOUBLE_PRECISION,
        chemistry_config=chemistry_config,
    )
    registered_variables = get_registered_variables(config)
    params = SimulationParams(chemistry_params=chemistry_params)

    density = jnp.asarray([1.0, 1.0, 2.0, 3.0])
    ionised = jnp.asarray([0.0, 0.5, 0.9, 0.2])  # fraction of the hydrogen
    hydrogen = density  # one nucleus per unit of mass
    neutral, ions = hydrogen * (1.0 - ionised), hydrogen * ionised
    particles = neutral + 2.0 * ions
    temperature = jnp.asarray([8000.0, 9000.0, 20000.0, 300.0])

    mean_molecular_weight = 1.0 / (2 * 0.76 + 3 * (1 - 0.76 - 0.02) / 4 + 0.02 / 2)
    read_particles = particles if from_species else density / mean_molecular_weight
    state = jnp.zeros((registered_variables.num_vars, 4))
    state = state.at[registered_variables.density_index].set(density)
    state = state.at[registered_variables.pressure_index].set(read_particles * temperature)
    start = registered_variables.chemistry_species_index
    state = state.at[start : start + 3].set(jnp.stack([neutral, ions, ions]))

    np.testing.assert_allclose(
        chemistry_temperature_kelvin(state, registered_variables, chemistry_config, chemistry_params),
        temperature,
        rtol=1e-13,
    )

    reacted = update_chemistry(state, registered_variables, chemistry_config, params, step)
    # H+ + E -> H with n(H+) = n(E): n(t) = n0 / (1 + k n0 t); T(t) = T0 exp(-t / tau)
    ions_after = ions / (1.0 + coefficient * ions * step)
    neutral_after = hydrogen - ions_after
    temperature_after = temperature * np.exp(-step / cooling_time)
    np.testing.assert_allclose(reacted[start + 1], ions_after, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(reacted[start], neutral_after, rtol=1e-7)
    np.testing.assert_allclose(reacted[start + 2], ions_after, rtol=1e-7, atol=1e-12)

    particles_after = neutral_after + 2.0 * ions_after
    written_particles = particles_after if from_species else density / mean_molecular_weight
    np.testing.assert_allclose(
        reacted[registered_variables.pressure_index], written_particles * temperature_after, rtol=1e-7
    )
    np.testing.assert_allclose(
        chemistry_temperature_kelvin(reacted, registered_variables, chemistry_config, chemistry_params),
        temperature_after,
        rtol=1e-7,
    )
    if from_species:
        # gas that recombined has fewer particles: less pressure than the
        # temperature alone would give
        assert float(reacted[registered_variables.pressure_index][2]) < 0.8 * float(
            particles[2] * temperature_after[2]
        )


def test_gas_above_the_temperature_range_keeps_its_excess():
    """Gas at 5e9 K is reacted as gas at 1e9 K (the upper end of the rates) and
    must not be written back at 1e9 K: it keeps the factor five."""
    species = ("H", "H+", "E")
    cooling_time, step = 4.0, 1.0
    chemistry_config, chemistry_params = _chemistry(
        species,
        RecombiningCoolingGas(jnp.asarray(0.0), jnp.asarray(cooling_time)),
        thermochemistry=True,
        temperature_from_species=True,
    )
    chemistry_params = chemistry_params._replace(cooling_courant=1.0, floor_temperature=1.0)
    config = SimulationConfig(
        dimensionality=1,
        num_cells=2,
        solver_mode=FINITE_DIFFERENCE,
        numerical_precision=DOUBLE_PRECISION,
        chemistry_config=chemistry_config,
    )
    registered_variables = get_registered_variables(config)
    params = SimulationParams(chemistry_params=chemistry_params)
    temperature = jnp.asarray([5.0e9, 5.0e8])
    ions = jnp.asarray([1.0, 1.0])
    state = jnp.zeros((registered_variables.num_vars, 2))
    state = state.at[registered_variables.density_index].set(ions)
    state = state.at[registered_variables.pressure_index].set(2.0 * ions * temperature)
    start = registered_variables.chemistry_species_index
    state = state.at[start : start + 3].set(jnp.stack([0.0 * ions, ions, ions]))

    reacted = update_chemistry(state, registered_variables, chemistry_config, params, step)
    np.testing.assert_allclose(
        chemistry_temperature_kelvin(reacted, registered_variables, chemistry_config, chemistry_params),
        temperature * np.exp(-step / cooling_time),
        rtol=1e-7,
    )


def test_dual_energy_row_follows_the_reaction():
    """The reaction lowers the pressure; the internal-energy density that the
    dual-energy formalism carries has to be lowered with it, or the cells whose
    pressure is recovered from it (a fast, cold flow) get the old pressure back."""
    species = ("H", "H+", "E")
    cooling_time = 0.02
    chemistry_config, chemistry_params = _chemistry(
        species,
        RecombiningCoolingGas(jnp.asarray(0.0), jnp.asarray(cooling_time)),
        thermochemistry=True,
        temperature_from_species=True,
    )
    chemistry_params = chemistry_params._replace(cooling_courant=1.0, floor_temperature=1e-6)
    adiabatic_index = 5.0 / 3.0

    def uniform_ions(coordinates):
        ones = np.ones_like(coordinates[0])
        return np.stack([1e-12 * ones, ones, ones])

    def supersonic(coordinates):
        # Mach 100 and more: the internal energy is below 1e-3 of the total
        # (``dual_energy_eta``), so the pressure is recovered from the carried
        # internal energy in every cell
        return [np.full_like(coordinates[0], 130.0), np.zeros_like(coordinates[0])]

    t_end = 0.02
    initial, final, registered_variables = _simulate(
        "fd", chemistry_config, chemistry_params, uniform_ions,
        velocities=supersonic, t_end=t_end, dual_energy=True,
    )
    assert registered_variables.internal_energy_active
    pressure = final[registered_variables.pressure_index]
    internal_energy = final[registered_variables.internal_energy_index]
    np.testing.assert_allclose(internal_energy, pressure / (adiabatic_index - 1.0), rtol=1e-12)
    # the gas has cooled by exp(-t / tau) on top of what the flow does to it:
    # compare with the same run without cooling
    slow_config, slow_params = _chemistry(
        species,
        RecombiningCoolingGas(jnp.asarray(0.0), jnp.asarray(1e30)),
        thermochemistry=True,
        temperature_from_species=True,
    )
    slow_params = slow_params._replace(cooling_courant=1.0, floor_temperature=1e-6)
    _, adiabatic, _ = _simulate(
        "fd", slow_config, slow_params, uniform_ions,
        velocities=supersonic, t_end=t_end, dual_energy=True,
    )
    ratio = pressure / adiabatic[registered_variables.pressure_index]
    np.testing.assert_allclose(ratio.mean(), np.exp(-t_end / cooling_time), rtol=0.05)
    assert ratio.max() < 0.6


# -----------------------------------------------------------------------------
# =========================== ↓ 7. Density floor ↓ =============================
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("solver_mode", [FINITE_DIFFERENCE, FINITE_VOLUME])
def test_density_floor_and_the_species(solver_mode):
    chemistry_config, _ = _chemistry(HELIUM, Frozen())
    config = SimulationConfig(
        dimensionality=1,
        num_cells=4,
        solver_mode=solver_mode,
        numerical_precision=DOUBLE_PRECISION,
        chemistry_config=chemistry_config,
    )
    registered_variables = get_registered_variables(config)
    density = jnp.asarray([1.0, 0.01, 0.5, 0.002])
    state = jnp.ones((registered_variables.num_vars, 4))
    state = state.at[registered_variables.density_index].set(density)
    start = registered_variables.chemistry_species_index
    composition = jnp.asarray([0.02, 0.05, 0.03, 0.11])
    state = state.at[start : start + 4].set(composition[:, None] * density[None, :])

    floored = _apply_density_floor(state, 0.1, 5.0 / 3.0, config, registered_variables)
    np.testing.assert_allclose(floored[registered_variables.density_index], [1.0, 0.1, 0.5, 0.1])
    per_mass = floored[start : start + 4] / floored[registered_variables.density_index][None, :]
    if solver_mode == FINITE_DIFFERENCE:
        # the added mass has the composition of its cell
        np.testing.assert_allclose(per_mass, composition[:, None] * np.ones((1, 4)), rtol=1e-14)
        # cells above the floor are untouched, bit for bit
        assert bool((floored[:, [0, 2]] == state[:, [0, 2]]).all())
    else:
        # the finite-volume solver carries the species as densities of their own
        assert bool((floored[start : start + 4] == state[start : start + 4]).all())


# -----------------------------------------------------------------------------
# ============================ ↓ 8. Pallas backend ↓ ===========================
# -----------------------------------------------------------------------------

def test_pallas_backend_gives_the_native_result():
    def fronts(coordinates):
        return _helium_fronts(coordinates)

    results = {}
    for backend in ("native", "pallas"):
        chemistry_config, chemistry_params = _chemistry(HELIUM, Frozen())
        backend_config = (
            BackendConfig(backend=NATIVE_JAX)
            if backend == "native"
            else BackendConfig(backend=PALLAS, pallas_block_shape=(4, 4, 8), pallas_interpret=True)
        )
        _, final, registered_variables = _simulate(
            "fd", chemistry_config, chemistry_params, fronts,
            cells=16, dimensionality=3, t_end=0.02, backend_config=backend_config,
        )
        results[backend] = final
    assert np.isfinite(results["pallas"]).all()
    np.testing.assert_allclose(results["pallas"], results["native"], rtol=1e-10, atol=1e-13)
    helium, singly, doubly, electrons = _per_mass(results["pallas"], registered_variables)
    np.testing.assert_allclose(helium + singly + doubly, HELIUM_PER_MASS, rtol=1e-12)
