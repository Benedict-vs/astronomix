"""3D circularly-polarized Alfvén-wave convergence — paper figure (double precision).

Runs the L1-convergence + runtime sweep over N = 8..128 in *double precision*
for the two MHD methods-paper configurations — FV (JAX) and FD (Pallas) — against
the analytic CP-Alfvén-wave solution, with the AthenaPK reference overlaid, and
writes

    figures/alfven_wave3D_dp_convergence.svg   (average L1 error vs N, paper figure)
    figures/alfven_wave3D_dp_runtime.svg        (error/runtime diagnostics)

The fast correctness version (single low resolution) lives in
``pytests/mhd/alfven_wave3D.py``; this script is the full double-precision sweep
that produces the paper figure. It reuses the shared benchmark harness in
``pytests/_benchmark_utils.py`` and the AthenaPK reference NPZ shipped under
``pytests/mhd/data/athenapk/``.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/alfven_convergence.py
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import sys
from pathlib import Path

# jax
import jax

# Double precision: the paper's Alfvén-wave convergence figure is the x64 sweep.
jax.config.update("jax_enable_x64", True)

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    NATIVE_JAX,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SnapshotSettings,
)
from astronomix.option_classes.simulation_config import StaticFloatVector

# astronomix functions
from astronomix.test_setups.mhd.alfven_wave3D import (
    setup_cp_alfven_wave,
    cp_alfven_wave_solution,
)


# The shared benchmark harness and the AthenaPK reference both live under
# pytests/; put that directory on the path so this example reuses the exact
# sweep driver the fast correctness test also builds on.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
_PYTESTS_DIR = str(_REPO / "pytests")
if _PYTESTS_DIR not in sys.path:
    sys.path.insert(0, _PYTESTS_DIR)

from _benchmark_utils import (  # noqa: E402
    BenchmarkSpec,
    run_convergence_and_runtime,
)

DATA_DIR = str(_HERE / "data" / "alfven_wave3D")
FIG_DIR = str(_HERE / "figures")
ATHENAPK_NPZ = str(
    _REPO / "pytests" / "mhd" / "data" / "athenapk" / "athenapk_alfven_convergence_dp.npz"
)

# The resolution sweep for the convergence figure.
N_VALUES = [8, 16, 32, 64, 128]

# Options shared by every benchmark configuration; only the backend, solver
# mode and CFL number differ between them.
_common_kwargs = dict(
    box_size=StaticFloatVector(3.0, 1.5, 1.5),
    mhd=True,
    dimensionality=3,
    progress_bar=False,
    memory_analysis=True,
    print_elapsed_time=True,
    return_snapshots=True,
    snapshot_settings=SnapshotSettings(return_final_state=True),
)

BENCHMARKS = [
    BenchmarkSpec(
        label="FV (JAX)",
        base_config=SimulationConfig(
            backend=NATIVE_JAX,
            solver_mode=FINITE_VOLUME,
            **_common_kwargs,
        ),
        cfl=0.4,
    ),
    BenchmarkSpec(
        label="FD (Pallas)",
        base_config=SimulationConfig(
            **_common_kwargs,
        ),
        cfl=1.5,
    ),
]


def _error_indices(registered_variables):
    """State-array indices entered into the L1 error norm (rho, v, p, B)."""
    return (
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
        registered_variables.pressure_index,
        registered_variables.magnetic_index.x,
        registered_variables.magnetic_index.y,
        registered_variables.magnetic_index.z,
    )


if __name__ == "__main__":
    run_convergence_and_runtime(
        BENCHMARKS,
        N_values=N_VALUES,
        setup_fn=setup_cp_alfven_wave,
        analytic_fn=cp_alfven_wave_solution,
        error_var_indices_fn=_error_indices,
        name="alfven_wave3D_dp",
        title="3D CP Alfvén wave (double precision)",
        data_dir=DATA_DIR,
        figure_dir=FIG_DIR,
        athenapk_npz=ATHENAPK_NPZ,
    )
