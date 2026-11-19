"""
Tests for the per-cell implicit chemistry solver
(``astronomix/_modules/_chemistry/_warp_stiff.py``): an adaptive Rosenbrock23
integrator compiled as one NVIDIA Warp kernel with one thread per cell, built on
gradsolve's jaxpr -> Warp translator.

1. The kernel on the Robertson problem against Diffrax Kvaerno5: accuracy with a
   per-cell step length, the linear invariant, a zero step, ``jax.jit`` and the
   solver cache.
2. The translator extension on a small temperature-dependent reaction network
   written with the primitives of a carbox right-hand side (closed-over constant
   arrays, gathers and scatter-adds with constant indices, a product over an
   axis, a matrix product, ``where``, clipping): one linearly implicit step
   against the same step from the JAX field and its ``jacfwd`` Jacobian, and the
   full solve against Kvaerno5. The network repeats indices in its scatter-add
   and has as many reactions as state variables on purpose: both were translated
   wrongly before 2026-11-16 (one update kept per repeated index; a scalar times
   a vector as long as the state paired element-wise in the Jacobian).
3. The chemistry module on a carbox network: ``update_chemistry`` with the kernel
   in every cell against ``update_chemistry`` with the vmapped Kvaerno5. Skipped
   when carbox or its ``data/simple_latent_tgas.csv`` is not available; set
   ASTRONOMIX_TEST_NETWORK_CSV to point at the file.

The problems are tiny, so the file runs on the CPU unless a platform is chosen in
the environment (Warp compiles the kernel for the CPU as well): it neither waits
for nor lands on a shared GPU. Needs gradsolve, warp-lang and diffrax.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# ruff: noqa: E402

import numpy as np
import pytest

pytest.importorskip("gradsolve")
pytest.importorskip("warp")
dx = pytest.importorskip("diffrax")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from astronomix._modules._chemistry._warp_stiff import build_fused_core_solver

RTOL = 1.0e-6
ATOL = 1.0e-12


def _reference(vector_field, initial_state, end_time, rtol=1.0e-10, atol=1.0e-14):
    """Diffrax Kvaerno5 at a tight tolerance, one independent solve per cell."""

    def one_cell(state, t_end):
        solution = dx.diffeqsolve(
            dx.ODETerm(vector_field),
            dx.Kvaerno5(),
            0.0,
            t_end,
            dt0=None,
            y0=state,
            stepsize_controller=dx.PIDController(rtol=rtol, atol=atol),
            saveat=dx.SaveAt(t1=True),
            max_steps=100_000,
        )
        return solution.ys[0]

    return np.asarray(jax.jit(jax.vmap(one_cell))(initial_state, end_time))


# ---------------------------------------------------------------------------
# 1. Robertson
# ---------------------------------------------------------------------------

ROBERTSON_STEPS = np.array([0.0, 4.0e-2, 4.0e-1, 4.0, 4.0e1, 4.0e2, 4.0e3, 4.0e4])


def _robertson(time, state, args):
    """Robertson's three-species kinetics, rate constants 0.04, 3e7 and 1e4."""
    slow = 0.04 * state[0]
    fast = 3.0e7 * state[1] * state[1]
    back = 1.0e4 * state[1] * state[2]
    return jnp.stack([-slow + back, slow - fast - back, fast])


@pytest.fixture(scope="module")
def robertson():
    """The solver, the initial states and the kernel's result, one cell per step length."""
    solver = build_fused_core_solver(
        _robertson, 3, rtol=RTOL, atol=ATOL, max_steps=4096, cache_key="pytest-robertson"
    )
    initial = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (ROBERTSON_STEPS.size, 1))
    steps = jnp.asarray(ROBERTSON_STEPS)
    return solver, initial, steps, np.asarray(solver(initial, steps))


def test_robertson_matches_kvaerno5(robertson):
    _, initial, steps, result = robertson
    reference = _reference(_robertson, initial[1:], steps[1:])
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[1:], reference, rtol=1.0e-4, atol=1.0e-10)


def test_robertson_keeps_the_linear_invariant(robertson):
    # a Rosenbrock scheme with the exact Jacobian conserves y1 + y2 + y3 to round-off
    _, _, _, result = robertson
    np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)


def test_step_counts(robertson):
    """``with_steps`` returns the same states and the trial steps of every cell: more for a longer
    interval, none for a zero one, and the limit for a cell that runs out of steps."""
    solver, initial, end_time, result = robertson
    states, steps = solver.with_steps(initial, end_time)
    np.testing.assert_array_equal(np.asarray(states), result)
    steps = np.asarray(steps)
    assert steps.dtype == np.int32 and steps.shape == (initial.shape[0],)
    interval = np.asarray(end_time)
    assert (steps[interval == 0.0] == 0).all()
    # the first trial step is a hundredth of the interval and grows fivefold at most
    assert steps[interval > 0.0].min() >= 4
    assert steps.max() < solver.max_steps
    assert (np.diff(steps[np.argsort(interval)]) >= 0).all()  # a longer interval takes more steps
    _, none = solver.with_steps(initial, jnp.zeros(initial.shape[0]))
    assert int(np.asarray(none).max()) == 0
    short = build_fused_core_solver(_robertson, 3, rtol=RTOL, atol=ATOL, max_steps=5, cache_key="pytest-robertson-short")
    _, exhausted = short.with_steps(initial, end_time)
    assert int(np.asarray(exhausted).max()) == 5 == short.max_steps


def _decay_under_a_root(time, state, args):
    """A fast linear decay and a slow variable that sees its square root. The Rosenbrock stages overshoot a
    stiff decay to below zero (the stage value is -0.65 of the start for a step of 100 decay times), where
    the root is not a number."""
    return jnp.stack([-state[0] * jnp.sqrt(state[1]), -1.0e3 * state[1]])


def test_a_step_without_an_error_estimate_is_shortened():
    """A trial step whose stages leave the domain of the field has a not-a-number error. It has to be
    rejected and shortened; lengthened (what a not-a-number did to the controller before 2026-11-19), the
    cell repeats it until it has used all its steps and comes back unchanged."""
    solver = build_fused_core_solver(
        _decay_under_a_root, 2, rtol=1.0e-6, atol=1.0e-12, max_steps=4096, cache_key="pytest-root", first_step_fraction=1.0
    )
    interval = jnp.asarray([1.0e-4, 1.0e-2, 0.3])  # a tenth of a decay time, ten of them, three hundred
    initial = jnp.ones((3, 2))
    states, steps = solver.with_steps(initial, interval)
    states, steps = np.asarray(states), np.asarray(steps)
    assert np.isfinite(states).all()
    assert steps.max() < solver.max_steps
    decayed = np.exp(-1.0e3 * np.asarray(interval))
    slow = np.exp(-(2.0 / 1.0e3) * (1.0 - np.exp(-0.5e3 * np.asarray(interval))))
    np.testing.assert_allclose(states[:, 0], slow, rtol=1.0e-5)
    np.testing.assert_allclose(states[:, 1], decayed, rtol=1.0e-3, atol=1.0e-9)


def test_zero_step_returns_the_state(robertson):
    # update_chemistry relies on it under the density gate: a cell that is not due is untouched
    _, initial, _, result = robertson
    assert ROBERTSON_STEPS[0] == 0.0
    np.testing.assert_array_equal(result[0], np.asarray(initial[0]))


def test_solver_is_a_jax_primitive_under_jit(robertson):
    solver, initial, steps, result = robertson
    jitted = jax.jit(lambda state, step: 2.0 * solver(state, step))
    np.testing.assert_array_equal(np.asarray(jitted(initial, steps)), 2.0 * result)


def test_solver_cache(robertson):
    solver = robertson[0]
    again = build_fused_core_solver(
        _robertson, 3, rtol=RTOL, atol=ATOL, max_steps=4096, cache_key="pytest-robertson"
    )
    assert again is solver


# ---------------------------------------------------------------------------
# 2. A reaction network written like a carbox right-hand side
# ---------------------------------------------------------------------------

# Five species and the temperature; index 5 of the padded abundances holds 1 (the
# second "reactant" of a one-body reaction). Species 0, 1 and 2 are produced by
# several reactions each (repeated scatter-add indices), and there are six
# reactions for six state variables (rate vectors as long as the state).
_REACTANTS = np.array([[0, 1], [2, 5], [2, 3], [4, 5], [0, 4], [1, 5]])
_PRODUCT_SPECIES = np.array([2, 0, 1, 4, 3, 2, 1, 2, 0])
_PRODUCT_REACTION = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5])
_LOSS = np.zeros((5, 6))
for _reaction, (_first, _second) in enumerate(_REACTANTS):
    for _species in (_first, _second):
        if _species < 5:
            _LOSS[_species, _reaction] += 1.0
_PREFACTOR = np.array([1.0e-2, 1.0, 1.0e2, 5.0e-1, 1.0e3, 2.0e-3])
_EXPONENT = np.array([0.5, 0.0, 0.0, 0.0, -0.7, 0.0])
_BARRIER = np.array([0.0, 300.0, 0.0, 0.0, 0.0, 40.0])
_COLD_SWITCH = np.array([False, False, False, True, False, False])


def _network(time, state, args):
    """Mass-action kinetics with Arrhenius rates and a heating/cooling balance."""
    abundances = jnp.maximum(state[:-1], 0.0)
    temperature = jnp.clip(state[-1], 3.0, 1.0e9)
    coefficients = (
        jnp.asarray(_PREFACTOR)
        * (temperature / 300.0) ** jnp.asarray(_EXPONENT)
        * jnp.exp(-jnp.asarray(_BARRIER) / temperature)
    )
    # one reaction is ten times slower in cold gas (a mask of constants and a comparison)
    coefficients = coefficients * jnp.where(jnp.asarray(_COLD_SWITCH) & (temperature <= 100.0), 0.1, 1.0)
    padded = jnp.concatenate([abundances, jnp.ones(1)])
    rates = coefficients * jnp.prod(padded[jnp.asarray(_REACTANTS)], axis=1)
    gain = jnp.zeros(5).at[jnp.asarray(_PRODUCT_SPECIES)].add(rates[jnp.asarray(_PRODUCT_REACTION)])
    loss = jnp.asarray(_LOSS) @ rates
    total = jnp.sum(abundances)
    heating = 2.0e1 * total + 5.0e1 * rates[2]
    cooling = 1.0e-1 * abundances[2] * temperature + 2.0e-3 * total * temperature
    temperature_rate = (heating - cooling) / (1.5 * total)
    return jnp.concatenate([gain - loss, jnp.reshape(temperature_rate, (1,))])


def _network_states(cells, seed):
    generator = np.random.default_rng(seed)
    abundances = 10.0 ** generator.uniform(-3.0, 0.0, size=(cells, 5))
    temperature = 10.0 ** generator.uniform(np.log10(20.0), np.log10(800.0), size=(cells, 1))
    steps = 10.0 ** generator.uniform(-2.0, 2.0, size=cells)
    return np.concatenate([abundances, temperature], axis=1), steps


def test_translated_field_and_jacobian_match_jax():
    # With one step allowed and the error control off, the kernel takes exactly one linearly
    # implicit step, a hundredth of the interval long: y + h k2, with W = 1 - gamma h J,
    # W k1 = f(y) and W (k2 - k1) = f(y + h k1 / 2) - k1. The same step from the JAX field and
    # its jacfwd Jacobian must agree to round-off; an error in either translation shows here.
    import gradsolve.warp._warp_rosenbrock as scheme

    gamma = float(scheme._D_COEF)
    step = 1.0e-2
    initial, _ = _network_states(16, seed=11)
    solver = build_fused_core_solver(
        _network, 6, rtol=1.0e6, atol=1.0e6, max_steps=1, cache_key="pytest-network-one-step"
    )
    result = np.asarray(solver(jnp.asarray(initial), jnp.full(initial.shape[0], 100.0 * step)))

    field = jax.jit(lambda state: _network(0.0, state, None))
    jacobian = jax.jit(jax.jacfwd(lambda state: _network(0.0, state, None)))
    expected = np.empty_like(initial)
    for cell, state in enumerate(initial):
        matrix = np.eye(6) - gamma * step * np.asarray(jacobian(jnp.asarray(state)))
        k1 = np.linalg.solve(matrix, np.asarray(field(jnp.asarray(state))))
        k2 = np.linalg.solve(matrix, np.asarray(field(jnp.asarray(state + 0.5 * step * k1))) - k1) + k1
        expected[cell] = state + step * k2

    increment = np.abs(expected - initial)
    assert np.max(increment / np.abs(initial)) > 1.0e-3  # the step moves the state
    np.testing.assert_allclose(result - initial, expected - initial, rtol=1.0e-9, atol=1.0e-12 * np.abs(initial).max())


def test_carbox_like_network_matches_kvaerno5():
    cells = 48
    initial, steps = _network_states(cells, seed=7)
    initial, steps = jnp.asarray(initial), jnp.asarray(steps)

    solver = build_fused_core_solver(
        _network, 6, rtol=RTOL, atol=ATOL, max_steps=4096, cache_key="pytest-network"
    )
    result = np.asarray(solver(initial, steps))
    reference = _reference(_network, initial, steps)

    assert np.all(np.isfinite(result))
    assert np.all(result[:, :5] > -1.0e-12)
    # the kernel runs at rtol 1e-6; its global error over these stiff solves reaches 1e-4
    np.testing.assert_allclose(result, reference, rtol=1.0e-3, atol=1.0e-9)
    # the kernel moved the state: the comparison is not between two copies of the input
    assert np.max(np.abs(np.log10(reference[:, -1] / np.asarray(initial)[:, -1]))) > 0.1


# ---------------------------------------------------------------------------
# 2b. A table looked up at an index computed in the kernel
# ---------------------------------------------------------------------------

TABLE_NODES = np.linspace(1.0, 5.0, 33)  # log10 of the temperature
# log10 of a decay rate on five densities and the temperature nodes, with structure between the nodes
TABLE = (
    -2.0
    + 1.5 * np.sin(1.7 * TABLE_NODES)[None, :]
    + 0.2 * TABLE_NODES[None, :]
    + 0.3 * np.linspace(-1.0, 1.0, 5)[:, None] * np.cos(TABLE_NODES)[None, :]
)


def _tabulated_decay(indexing):
    """dT/dt = -rate(n, T) T with the rate interpolated in a table, as the cooling tables of the chemistry are."""
    nodes, table = jnp.asarray(TABLE_NODES), jnp.asarray(TABLE)
    spacing = TABLE_NODES[1] - TABLE_NODES[0]

    def field(time, state, args):
        density, temperature = state[0], jnp.clip(state[1], 10.0, 1.0e5)
        x = jnp.log10(temperature)
        j = jnp.clip(jnp.floor((x - nodes[0]) / spacing).astype(int), 0, nodes.shape[0] - 2)
        i = jnp.clip(jnp.floor(2.0 * jnp.log10(density) + 2.0).astype(int), 0, table.shape[0] - 1)
        if indexing == "scalars":  # table[i, j]: a dynamic_slice in the jaxpr
            low, high, node = table[i, j], table[i, j + 1], nodes[j]
        else:  # index arrays: a gather
            pair = table[jnp.stack([i, i]), jnp.stack([j, j + 1])]
            low, high, node = pair[0], pair[1], nodes[jnp.stack([j])][0]
        rate = 10.0 ** (low + (x - node) / spacing * (high - low))
        return jnp.stack([0.0 * density, -rate * state[1]])

    return field


@pytest.mark.parametrize("indexing", ["scalars", "arrays"])
def test_table_lookup_at_a_run_time_index(indexing):
    # The generated code is straight-line and scalar: a table of constants indexed in the kernel becomes a Warp
    # function (a decision tree over the flat index). Without it the tabulated metal cooling of a KROME network
    # could not be part of the kernel.
    cells = 40
    generator = np.random.default_rng(3)
    initial = np.stack([10.0 ** generator.uniform(-1.0, 1.0, cells), 10.0 ** generator.uniform(2.0, 4.8, cells)], axis=1)
    field = _tabulated_decay(indexing)
    rate = -np.asarray(jax.vmap(lambda state: field(0.0, state, None)[1])(jnp.asarray(initial))) / initial[:, 1]
    steps = jnp.asarray(1.5 / rate)  # the temperature falls by a factor of a few, across several nodes of the table
    solver = build_fused_core_solver(field, 2, rtol=RTOL, atol=ATOL, max_steps=4096, cache_key=f"pytest-table-{indexing}")
    result = np.asarray(solver(jnp.asarray(initial), steps))
    reference = _reference(_tabulated_decay("scalars"), jnp.asarray(initial), steps)

    assert np.all(np.isfinite(result))
    # the temperatures start between 100 K and 6e4 K; where the rate rises on the way down they decay to nothing
    np.testing.assert_allclose(result, reference, rtol=1.0e-4, atol=1.0e-3)
    crossed = np.log10(initial[:, 1] / np.maximum(reference[:, 1], 10.0)) / (TABLE_NODES[1] - TABLE_NODES[0])
    assert np.median(crossed) > 2.0  # the solve crosses table intervals: the look-up is exercised, not one constant


# ---------------------------------------------------------------------------
# 3. update_chemistry on a carbox network: the kernel against the vmapped Kvaerno5
# ---------------------------------------------------------------------------

NUMBER_DENSITY_UNIT_CGS = 597.6
TEMPERATURE_UNIT_KELVIN = 1.0e2
TIME_UNIT_SECONDS = 3.1557e13
NUM_CELLS = 24
TIME_STEP = 2.0e-3  # 2000 yr, of the order of a hydrodynamic step of a cloud simulation
# fraction of the hydrogen nuclei in each species; every other species starts at 1e-12
INITIAL_FRACTIONS = {"H": 0.5, "H2": 0.25, "C+": 1.4e-4, "O": 3.2e-4, "H+": 1.0e-6, "E": 1.41e-4}


def _network_file():
    path = os.environ.get("ASTRONOMIX_TEST_NETWORK_CSV")
    if path:
        return path if os.path.exists(path) else None
    try:
        import carbox
    except ImportError:
        return None
    # carbox keeps the network outside its Python package: next to it in a clone or an editable
    # install, else in the pinned submodule of this repository (external/carbox).
    repository_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = (
        os.path.join(os.path.dirname(os.path.dirname(carbox.__file__)), "data", "simple_latent_tgas.csv"),
        os.path.join(repository_root, "external", "carbox", "data", "simple_latent_tgas.csv"),
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _react(use_kernel, network_csv, order_cells=True):
    """One call of ``update_chemistry`` over a 1-D box of ``NUM_CELLS`` different cells."""
    from astronomix import (
        SimulationConfig,
        SimulationParams,
        construct_primitive_state,
        finalize_config,
        get_helper_data,
        get_registered_variables,
    )
    from astronomix._modules._chemistry._chemistry import make_vector_field, update_chemistry
    from astronomix.option_classes.simulation_config import (
        FINITE_VOLUME,
        NATIVE_JAX,
        BackendConfig,
    )
    from astronomix.setup_helpers import build_chemistry_from_network_file

    gamma = 5.0 / 3.0
    chemistry_config, chemistry_params, species_names = build_chemistry_from_network_file(
        network_csv,
        "latent_tgas",
        number_density_unit_cgs=NUMBER_DENSITY_UNIT_CGS,
        temperature_unit_kelvin=TEMPERATURE_UNIT_KELVIN,
        time_unit_seconds=TIME_UNIT_SECONDS,
        cosmic_ray_rate=3.0e-17,
        fuv_field=1.0e-4,
        visual_extinction=2.0,
        thermochemistry=True,
        cooling_courant=10.0,
        absolute_tolerance=ATOL,
        relative_tolerance=RTOL,
        max_steps=2048,
    )
    if use_kernel:
        cc, cp = chemistry_config, chemistry_params
        vector_field = make_vector_field(
            cp.network, cp.rate_modifier_a, cp.rate_modifier_b, cp.cosmic_ray_rate, cp.fuv_field,
            cp.visual_extinction, cc.thermochemistry, gamma, cp.dust_to_gas_ratio,
            cp.hydrogen_molecule_formation_rate_coefficient, cc.hydrogen_index,
            cc.molecular_hydrogen_index, cc.electron_index, cc.atomic_oxygen_index,
            cc.ionized_hydrogen_index, cc.helium_index, cc.ionized_carbon_index, cc.co_cooling,
            cc.carbon_monoxide_index, cp.co_cooling_table, cp.co_cooling_bounds,
        )
        solver = build_fused_core_solver(
            vector_field, cc.number_of_chemical_species + 1,
            rtol=float(cp.rtol), atol=float(cp.atol), max_steps=cc.max_steps,
            cache_key="pytest-carbox",
        )
        chemistry_config = chemistry_config._replace(
            stiff_core_solver="warp", stiff_core_fn=solver, order_cells_for_kernel=order_cells
        )

    config = SimulationConfig(
        dimensionality=1,
        solver_mode=FINITE_VOLUME,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        box_size=1.0,
        num_cells=NUM_CELLS,
        progress_bar=False,
        chemistry_config=chemistry_config,
    )
    params = SimulationParams(t_end=1.0, gamma=gamma, chemistry_params=chemistry_params)
    registered_variables = get_registered_variables(config)
    position = get_helper_data(config).geometric_centers

    # density from ~0.2 to ~20 code units (100 to 10^4 cm^-3) and a pressure that makes the
    # temperature differ from cell to cell, so every cell is a different stiff problem
    density = 10.0 ** (-0.7 + 2.0 * position)
    pressure = density * (0.3 + 1.5 * position)
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=jnp.zeros_like(position),
        gas_pressure=pressure,
    )
    start = registered_variables.chemistry_species_index
    count = registered_variables.num_chemical_species
    for index in range(count):
        fraction = INITIAL_FRACTIONS.get(species_names[index], 1.0e-12)
        state = state.at[start + index].set(fraction * density)

    config = finalize_config(config, state.shape)
    reacted = update_chemistry(state, registered_variables, config.chemistry_config, params, TIME_STEP)
    species = slice(start, start + count)
    pressure_index = registered_variables.pressure_index
    return (
        np.asarray(state[species]), np.asarray(reacted[species]),
        np.asarray(state[pressure_index]), np.asarray(reacted[pressure_index]),
        tuple(species_names),
    )


def test_update_chemistry_kernel_matches_kvaerno5():
    network_csv = _network_file()
    if network_csv is None:
        pytest.skip("carbox network file not found (set ASTRONOMIX_TEST_NETWORK_CSV)")

    initial, kvaerno5, initial_pressure, kvaerno5_pressure, names = _react(False, network_csv)
    _, kernel, _, kernel_pressure, _ = _react(True, network_csv)

    assert np.all(np.isfinite(kernel)) and np.all(np.isfinite(kernel_pressure))
    # the step is long enough for the chemistry to act, so the agreement below is not trivial
    assert np.max(np.abs(np.log10(kvaerno5[names.index("C+")] / initial[names.index("C+")]))) > 1.0e-2

    # species that carry more than 1e-10 of the hydrogen: within 1e-3 dex of the reference solver
    hydrogen = initial[names.index("H")] + 2.0 * initial[names.index("H2")]
    relevant = kvaerno5 > 1.0e-10 * hydrogen
    difference = np.abs(np.log10(np.maximum(kernel, 1.0e-300) / np.maximum(kvaerno5, 1.0e-300)))
    assert np.max(difference[relevant]) < 1.0e-3, (
        f"largest difference {np.max(difference[relevant]):.2e} dex"
    )
    np.testing.assert_allclose(kernel_pressure, kvaerno5_pressure, rtol=1.0e-3)


def test_order_of_the_cells_in_the_kernel_launch_does_not_change_them():
    """``update_chemistry`` launches the kernel with the cells ordered by gas phase
    (``order_cells_for_kernel``, so that the threads of a GPU warp need about the
    same number of steps) and puts them back: the same cells bit for bit."""
    network_csv = _network_file()
    if network_csv is None:
        pytest.skip("carbox network file not found (set ASTRONOMIX_TEST_NETWORK_CSV)")

    initial, ordered, _, ordered_pressure, _ = _react(True, network_csv, order_cells=True)
    _, in_grid_order, _, grid_pressure, _ = _react(True, network_csv, order_cells=False)
    assert np.max(np.abs(ordered - initial) / initial) > 1.0e-3  # the chemistry acted
    np.testing.assert_array_equal(ordered, in_grid_order)
    np.testing.assert_array_equal(ordered_pressure, grid_pressure)


def test_cost_key_groups_cells_by_density_then_temperature():
    from astronomix._modules._chemistry._chemistry import _kernel_cost_key

    # two species per cell; particle densities 1e-3, 1e-3, 30, 30, 1e-3 and temperatures as below
    species = jnp.asarray([[5e-4, 5e-4], [6e-4, 4e-4], [10.0, 20.0], [15.0, 15.0], [5e-4, 5e-4]])
    temperature = jnp.asarray([1.0e6, 1.0e4, 100.0, 5000.0, 1.0e6])
    step = jnp.asarray([1.0, 1.0, 1.0, 1.0, 0.0])
    key = np.asarray(_kernel_cost_key(species, temperature, step))
    # thin gas before dense gas, cold before hot within a density bin, a zero step last
    assert list(np.argsort(key)) == [1, 0, 2, 3, 4]
    assert np.isinf(key[4])

