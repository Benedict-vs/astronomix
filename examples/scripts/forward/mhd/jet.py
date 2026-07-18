"""
3D MHD Jet Simulation Example
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    OPEN_BOUNDARY,
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
    setup_magnetic_fields_from_vector_potential,
)


# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# configure the simulation

gamma = 5/3
box_size = 24.0
num_cells = 256
grid_spacing = box_size / num_cells
x_center = box_size / 2.0
y_center = box_size / 2.0
z_center = box_size / 2.0

config = SimulationConfig(
    grid_spacing = grid_spacing,
    mhd = True,
    progress_bar = True,
    dimensionality = 3,
    box_size = box_size,
    num_cells = num_cells,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        )
    ),
)

registered_variables = get_registered_variables(config)

# set up the initial state

rho_0 = 1.0
p_0 = 1.0

def jet_vector_potential(X, Y, Z):
    r = jnp.sqrt((X - x_center)**2 + (Y - y_center)**2 + (Z - z_center)**2)
    A0 = 20.0
    
    A_x = -jnp.exp(-r ** 2) * (Y - y_center)
    A_y = jnp.exp(-r ** 2) * (X - x_center)
    A_z = 0.5 * A0 * jnp.exp(-r ** 2)
    
    return A_x, A_y, A_z

B_x, B_y, B_z, bxb, byb, bzb = setup_magnetic_fields_from_vector_potential(
    config = config,
    vector_potential_func = jet_vector_potential,
)

rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0
u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0

params = SimulationParams(
    C_cfl = 1.5,
    t_end = 5.0,
    gamma = gamma,
)

initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    gas_pressure = p,
    magnetic_field_x = B_x,
    magnetic_field_y = B_y,
    magnetic_field_z = B_z,
    interface_magnetic_field_x = bxb,
    interface_magnetic_field_y = byb,
    interface_magnetic_field_z = bzb,
)

config = finalize_config(config, initial_state.shape)

# run the simulation
final_state = time_integration(initial_state, config, params, registered_variables)

# plot the results
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
y_index = num_cells // 2
ax.imshow(final_state[registered_variables.density_index, :, y_index, :].T, cmap="YlOrRd")
fig.savefig(figures_dir / "jet_density.png", dpi=400)
