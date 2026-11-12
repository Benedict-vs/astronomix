"""
``SimulationConfig.monitor_diagnostics_interval`` test.

Runs the 1D Sod shock tube with the per-step diagnostics on every step and on
every 4th step: the final states must be bit-identical and the diagnostics log
must hold a quarter of the lines.
"""

from autocvd import autocvd

autocvd(num_gpus=1)

# ruff: noqa: E402
import os
import tempfile

import numpy as np

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.test_setups.hydrodynamics.shock_tube1D import setup_sod_shock_tube


def test_diagnostics_interval(interval=4):
    """Compare the run with ``interval`` against the every-step run."""
    results = {}
    for n in (1, interval):
        log_path = os.path.join(tempfile.mkdtemp(), "diag.log")
        os.environ["ASTRONOMIX_DIAG_LOG"] = log_path
        config = SimulationConfig(
            num_cells=200,
            monitor_diagnostics=True,
            monitor_diagnostics_interval=n,
        )
        registered_variables = get_registered_variables(config)
        state, config, params = setup_sod_shock_tube(
            config, registered_variables, SimulationParams(), get_helper_data(config)
        )
        final_state = np.asarray(
            time_integration(state, config, params, registered_variables)
        )
        with open(log_path) as log_file:
            num_lines = sum(1 for line in log_file if line.startswith("[diag]"))
        results[n] = (final_state, num_lines)

    identical = np.array_equal(results[1][0], results[interval][0])
    ratio = results[1][1] / results[interval][1]
    print(
        f"\nfinal states bit-identical: {identical}; [diag] lines "
        f"{results[1][1]} (every step) vs {results[interval][1]} "
        f"(every {interval}th), ratio {ratio:.2f}"
    )
    assert identical
    assert abs(ratio - interval) < 0.2


if __name__ == "__main__":
    test_diagnostics_interval()
