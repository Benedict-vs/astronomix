"""
2D Kelvin-Helmholtz Instability with an ad-hoc shear-layer initialization.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import PERIODIC_BOUNDARY

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
)


# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# configure the simulation
box_size = 1.0
num_cells = 512

config = SimulationConfig(
    progress_bar = True,
    dimensionality = 2,
    box_size = box_size,
    num_cells = num_cells,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
)

registered_variables = get_registered_variables(config)

params = SimulationParams(
    t_end = 2.0,
    C_cfl = 0.4,
)

# set up two counter-streaming layers with a small sinusoidal perturbation
grid_spacing = box_size / num_cells
x = jnp.linspace(grid_spacing / 2, box_size - grid_spacing / 2, num_cells)
X, Y = jnp.meshgrid(x, x, indexing="ij")

rho = jnp.ones_like(X)
u_x = 0.5 * jnp.ones_like(X)
u_y = 0.01 * jnp.sin(2 * jnp.pi * X)

shear_layer = (Y > 0.25) & (Y < 0.75)
rho = jnp.where(shear_layer, 2.0, rho)
u_x = jnp.where(shear_layer, -0.5, u_x)

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

# run the simulation
final_state = time_integration(initial_state, config, params, registered_variables)

# plot the final density
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(final_state[registered_variables.density_index].T, origin="lower", cmap="viridis")
ax.set_axis_off()
fig.savefig(figures_dir / "khi_density.png", dpi=300, bbox_inches="tight")
