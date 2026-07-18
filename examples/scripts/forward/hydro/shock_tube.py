"""
1D Sod Shock Tube — the simplest astronomix simulation.
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
from astronomix import CARTESIAN

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    time_integration,
    get_helper_data,
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
)


# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# configure the simulation
config = SimulationConfig(
    geometry = CARTESIAN,
    box_size = 1.0,
    num_cells = 400,
)

params = SimulationParams(
    t_end = 0.2,
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

# set up the shock initial state: (rho, u, p) left / right of x = 0.5
shock_position = 0.5
x = helper_data.geometric_centers
rho = jnp.where(x < shock_position, 1.0, 0.125)
u = jnp.zeros_like(x)
p = jnp.where(x < shock_position, 1.0, 0.1)

initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = u,
    gas_pressure = p,
)

config = finalize_config(config, initial_state.shape)

# run the simulation
final_state = time_integration(initial_state, config, params, registered_variables)

# plot the results
rho_final = final_state[registered_variables.density_index]
u_final = final_state[registered_variables.velocity_index]
p_final = final_state[registered_variables.pressure_index]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, initial, final, title in zip(
    axs,
    (rho, u, p),
    (rho_final, u_final, p_final),
    ("Density", "Velocity", "Pressure"),
):
    ax.plot(x, initial, label="initial")
    ax.plot(x, final, label="final")
    ax.set_title(title)
    ax.legend()

fig.savefig(figures_dir / "shock_tube.png", dpi=200)
