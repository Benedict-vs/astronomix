"""1D Sod shock-tube — paper figure (``shock_tube1D_test.svg``).

Runs the classic Sod shock tube on three solver configurations (finite-volume
HLLC with the Minmod / Superbee limiters and the finite-difference solver) at
200 cells, overlays each on a high-resolution analytic reference, and writes
``figures/shock_tube1D_test.svg`` (density / velocity / pressure with a zoomed
inset around the contact discontinuity).

The same simulation runs as a fast correctness check in
``pytests/hydrodynamics/shock_tube1D.py``; this script is the figure generator.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/hydro/shock_tube1D.py
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLLC,
    MINMOD,
)
from astronomix.option_classes.simulation_config import SUPERBEE

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    time_integration,
    get_registered_variables,
)
from astronomix.option_classes.simulation_config import config_to_string
from astronomix.plotting_helpers.inset_box import add_inset_box
from astronomix.test_setups.hydrodynamics.shock_tube1D import (
    setup_sod_shock_tube,
    sod_shock_tube_solution,
)


# figures are written to the local figures/ directory
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

NUM_CELLS = 200

# The configurations shown: two finite-volume HLLC runs differing only in their
# slope limiter, plus the finite-difference solver.
CONFIG_LIST = [
    SimulationConfig(
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        num_cells=NUM_CELLS,
    ),
    SimulationConfig(
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=SUPERBEE,
        num_cells=NUM_CELLS,
    ),
    SimulationConfig(
        num_cells=NUM_CELLS,
    ),
]


def run_shock_tube1D():
    """Run every configuration and write the Sod shock-tube overview figure."""

    # -------------------------------------------------------------
    # =========== ↓ High-resolution reference solution ↓ ==========
    # -------------------------------------------------------------

    fig, (ax_density, ax_velocity, ax_pressure) = plt.subplots(1, 3, figsize=(15, 5))

    # A finely resolved analytic solution serves as the visual reference the
    # coarser runs are plotted against.
    config_high_res = SimulationConfig(
        dimensionality=1,
        num_cells=1000,
    )

    registered_variables = get_registered_variables(config_high_res)
    helper_data_high_res = get_helper_data(config_high_res)
    _, config_high_res, params_high_res = setup_sod_shock_tube(
        config_high_res,
        registered_variables,
        SimulationParams(),
        helper_data_high_res,
    )
    reference_solution_high_res = sod_shock_tube_solution(
        config_high_res,
        registered_variables,
        params_high_res,
        helper_data_high_res,
    )

    ax_density.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.density_index],
        label="Reference",
        color="black",
    )
    ax_velocity.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.velocity_index],
        label="Reference",
        color="black",
    )
    ax_pressure.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.pressure_index],
        label="Reference",
        color="black",
    )
    ax_density.set_xlabel("x")
    ax_density.set_ylabel("Density")
    ax_velocity.set_xlabel("x")
    ax_velocity.set_ylabel("Velocity")
    ax_pressure.set_xlabel("x")
    ax_pressure.set_ylabel("Pressure")
    ax_density.set_title("Density")
    ax_velocity.set_title("Velocity")
    ax_pressure.set_title("Pressure")

    # -------------------------------------------------------------
    # =========== ↑ High-resolution reference solution ↑ ==========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Per-configuration run and plot ↓ =============
    # -------------------------------------------------------------

    for config in CONFIG_LIST:
        registered_variables = get_registered_variables(config)
        helper_data = get_helper_data(config)

        # Build the initial state. The finite-difference solver tolerates a
        # larger CFL number than the finite-volume solver here.
        initial_state, config, params = setup_sod_shock_tube(
            config,
            registered_variables,
            SimulationParams(
                C_cfl=1.5 if config.solver_mode == FINITE_DIFFERENCE else 0.8,
            ),
            helper_data,
        )

        final_state = time_integration(
            initial_state,
            config,
            params,
            registered_variables,
        )

        ax_density.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.density_index],
            label=config_to_string(config),
        )
        ax_velocity.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.velocity_index],
            label=config_to_string(config),
        )
        ax_pressure.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.pressure_index],
            label=config_to_string(config),
        )

    # -------------------------------------------------------------
    # ============ ↑ Per-configuration run and plot ↑ =============
    # -------------------------------------------------------------

    # A zoomed inset around the contact discontinuity highlights how the
    # limiters differ where the solutions are hardest to resolve.
    add_inset_box(ax_density, x1=0.62, x2=0.72, y1=0.20, y2=0.30, connect_loc1=1)

    handles, labels = ax_density.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = FIG_DIR / "shock_tube1D_test.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    run_shock_tube1D()
