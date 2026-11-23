"""
Cells whose reaction does not reach the end of its step (``ChemistryConfig.count_unfinished_cells``).

Run from the repository root (CPU, a few minutes)::

    pytest pytests/chemistry/test_unfinished_cells.py

A cell that runs out of solver steps keeps its last accepted state (per-cell kernel) or
its pre-step state (Kvaerno5), and nothing in the result says so. With the option,
``update_chemistry`` returns ``(state, unfinished)`` and a run prints a line for a call
that left cells unfinished. Without it, the return value is the state as before, and the
states are the same bit for bit either way.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("diffrax")

import test_radiation_field as radiation
from test_radiation_field import _row_of_cells, network_file  # noqa: F401  (fixture)

from astronomix._modules._chemistry._chemistry import update_chemistry
from astronomix._modules._chemistry.chemistry_options import EMULATOR

FIELDS = (0.5, 1.0, 4.0, 30.0)


def _chemistry(path, kernel, max_steps, per_cell=False, count=False):
    """The test network of ``test_radiation_field`` with a step budget, with or without the count."""
    from astronomix.setup_helpers import attach_per_cell_kernel

    chemistry_config, chemistry_params, species = radiation._chemistry(path, 1.0, per_cell=per_cell)
    chemistry_config = chemistry_config._replace(max_steps=max_steps, count_unfinished_cells=count)
    if kernel:
        chemistry_config = attach_per_cell_kernel(
            chemistry_config, chemistry_params, radiation.GAMMA, cache_key=("pytest-unfinished", path, max_steps, per_cell)
        )
    return chemistry_config, chemistry_params, species


def _skip_without_kernel(kernel):
    if kernel:
        pytest.importorskip("gradsolve")
        pytest.importorskip("warp")


@pytest.mark.parametrize("kernel", [False, True], ids=["kvaerno5", "kernel"])
@pytest.mark.parametrize("max_steps", [4096, 2], ids=["enough-steps", "two-steps"])
def test_the_count_and_the_states(network_file, kernel, max_steps):  # noqa: F811
    """With enough steps no cell is unfinished, with two steps every cell is; the states are those without the count."""
    _skip_without_kernel(kernel)
    plain, chemistry_params, species = _chemistry(network_file, kernel, max_steps)
    counted, _, _ = _chemistry(network_file, kernel, max_steps, count=True)
    state, registered_variables, params = _row_of_cells(plain, chemistry_params, species, 3)

    reacted = update_chemistry(state, registered_variables, plain, params, 1.0)
    assert not isinstance(reacted, tuple)
    reacted_counted, unfinished = update_chemistry(state, registered_variables, counted, params, 1.0)
    np.testing.assert_array_equal(np.asarray(reacted_counted), np.asarray(reacted))
    assert unfinished.dtype == jnp.int32
    assert int(unfinished) == (0 if max_steps > 2 else 3)


@pytest.mark.parametrize("kernel", [False, True], ids=["kvaerno5", "kernel"])
def test_the_count_with_the_field_of_every_cell(network_file, kernel):  # noqa: F811
    _skip_without_kernel(kernel)
    plain, chemistry_params, species = _chemistry(network_file, kernel, 2, per_cell=True)
    counted, _, _ = _chemistry(network_file, kernel, 2, per_cell=True, count=True)
    state, registered_variables, params = _row_of_cells(plain, chemistry_params, species, len(FIELDS))
    field = jnp.asarray(FIELDS)
    reacted = update_chemistry(state, registered_variables, plain, params, 1.0, radiation_scale=field)
    reacted_counted, unfinished = update_chemistry(state, registered_variables, counted, params, 1.0, radiation_scale=field)
    np.testing.assert_array_equal(np.asarray(reacted_counted), np.asarray(reacted))
    assert int(unfinished) == len(FIELDS)


def test_the_count_in_chunks(network_file):  # noqa: F811
    """Kvaerno5 in chunks of cells (``reaction_chunk_size``) counts the same."""
    counted, chemistry_params, species = _chemistry(network_file, False, 2, count=True)
    state, registered_variables, params = _row_of_cells(counted, chemistry_params, species, 5)
    whole, unfinished = update_chemistry(state, registered_variables, counted, params, 1.0)
    chunked, unfinished_chunked = update_chemistry(
        state, registered_variables, counted._replace(reaction_chunk_size=2), params, 1.0
    )
    np.testing.assert_array_equal(np.asarray(chunked), np.asarray(whole))
    assert int(unfinished) == int(unfinished_chunked) == 5


def test_only_cells_with_a_step_are_counted(network_file):  # noqa: F811
    """With the density gate a cell whose step is zero this call keeps its state and is not counted."""
    counted, chemistry_params, species = _chemistry(network_file, False, 2, count=True)
    # 42 per cm^3 of gas: 30 hydrogen nuclei; the gate at 10 makes every cell dense, at 1e3 none
    state, registered_variables, params = _row_of_cells(counted, chemistry_params, species, 4)
    dense = counted._replace(chemistry_subcycle_density_threshold_cgs=10.0)
    _, unfinished = update_chemistry(state, registered_variables, dense, params, 1.0, jnp.zeros(4))
    assert int(unfinished) == 4
    thin = counted._replace(chemistry_subcycle_density_threshold_cgs=1.0e3)
    kept, unfinished = update_chemistry(state, registered_variables, thin, params, 1.0, jnp.zeros(4))
    np.testing.assert_allclose(np.asarray(kept), np.asarray(state), rtol=1e-14)
    assert int(unfinished) == 0


def test_not_for_the_emulator(network_file):  # noqa: F811
    counted, chemistry_params, species = _chemistry(network_file, False, 2, count=True)
    state, registered_variables, params = _row_of_cells(counted, chemistry_params, species, 2)
    with pytest.raises(NotImplementedError, match="not for the emulator"):
        update_chemistry(state, registered_variables, counted._replace(solver=EMULATOR), params, 1.0)


def test_a_run_prints_unfinished_cells(network_file, monkeypatch, capfd):  # noqa: F811
    """Through ``time_integration``: the run is the run without the count, and says that cells were unfinished."""
    build = radiation._chemistry

    def with_budget(count):
        def chemistry(*arguments, **keywords):
            config, params, species = build(*arguments, **keywords)
            return config._replace(max_steps=2, count_unfinished_cells=count), params, species

        return chemistry

    monkeypatch.setattr(radiation, "_chemistry", with_budget(False))
    plain, _, _ = radiation._run(network_file, 1.0, False, shape=(4, 4, 4), steps=2)
    capfd.readouterr()
    monkeypatch.setattr(radiation, "_chemistry", with_budget(True))
    counted, _, _ = radiation._run(network_file, 1.0, False, shape=(4, 4, 4), steps=2)
    printed = capfd.readouterr().out
    np.testing.assert_array_equal(counted, plain)
    assert "[chemistry] 64 cells did not reach the end of their reaction step" in printed
