"""
Output options: Orbax checkpointing and restarting.

With ``snapshot_storage_mode = TO_DISK`` the run is split into segments and the
loop state is written to disk via Orbax after each one. A later process can pick
up the latest checkpoint with ``restart_from_latest_checkpoint`` and continue.
Here we run the 2D Kelvin-Helmholtz test to half time, then restart and finish.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import shutil

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    PERIODIC_BOUNDARY,
    FORWARDS,
    TO_DISK,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
)

# astronomix functions
from astronomix import (
    time_integration,
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
    restart_from_latest_checkpoint,
)


# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# configure the 2D Kelvin-Helmholtz test case writing checkpoints to disk
box_size = 1.0
num_cells = 256
checkpoint_path = "/tmp/khi_checkpoints"
shutil.rmtree(checkpoint_path, ignore_errors=True)

config = SimulationConfig(
    progress_bar = True,
    dimensionality = 2,
    box_size = box_size,
    num_cells = num_cells,
    differentiation_mode = FORWARDS,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    snapshot_storage_mode = TO_DISK,
    snapshot_storage_path = checkpoint_path,
    num_snapshots = 4,
)

registered_variables = get_registered_variables(config)

# two counter-streaming shear layers with a small sinusoidal perturbation
grid_spacing = box_size / num_cells
x = jnp.linspace(grid_spacing / 2, box_size - grid_spacing / 2, num_cells)
X, Y = jnp.meshgrid(x, x, indexing="ij")

rho = jnp.where((Y > 0.25) & (Y < 0.75), 2.0, 1.0)
u_x = jnp.where((Y > 0.25) & (Y < 0.75), -0.5, 0.5)
u_y = 0.01 * jnp.sin(2 * jnp.pi * X)
p = 2.5 * jnp.ones_like(X)

initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    gas_pressure = p,
)

config = finalize_config(config, initial_state.shape)

# run the first half, writing checkpoints to disk along the way
params = SimulationParams(t_end = 1.0, C_cfl = 0.4)
time_integration(initial_state, config, params, registered_variables)

# restart from the latest checkpoint and continue to the final time
params = SimulationParams(t_end = 2.0, C_cfl = 0.4)
restart_state, params, restart = restart_from_latest_checkpoint(checkpoint_path, params)
final_state = time_integration(
    restart_state, config, params, registered_variables, restart_state=restart
)

# plot the final density
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(final_state[registered_variables.density_index].T, origin="lower", cmap="viridis")
ax.set_axis_off()
fig.savefig(figures_dir / "khi_from_checkpoint.png", dpi=200, bbox_inches="tight")
