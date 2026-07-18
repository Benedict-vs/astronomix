"""
Output options: build a movie on the fly via a snapshot callback.

The callback fires at each snapshot time during the run. Only the data actually
needed for the movie (a single 2D density slice) is moved to the host inside a
``jax.debug.callback`` — the full state stays on the GPU.
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
from matplotlib.animation import FuncAnimation

# astronomix constants
from astronomix import (
    PERIODIC_BOUNDARY,
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
)


# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# configure the 2D Kelvin-Helmholtz test case, recording 60 snapshots
box_size = 1.0
num_cells = 256

config = SimulationConfig(
    progress_bar = True,
    dimensionality = 2,
    box_size = box_size,
    num_cells = num_cells,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    activate_snapshot_callback = True,
    num_snapshots = 60,
)

registered_variables = get_registered_variables(config)

params = SimulationParams(t_end = 2.0, C_cfl = 0.4)

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

# collect frames on the host as the simulation runs
frames = []

def record_frame(time, state, registered_variables):
    # move ONLY the density slice to the host, not the full state
    density = state[registered_variables.density_index]
    jax.debug.callback(lambda d: frames.append(d), density)

# run the simulation, firing the callback at each snapshot
time_integration(initial_state, config, params, registered_variables, record_frame)

# assemble the collected frames into a movie
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_axis_off()
image = ax.imshow(frames[0].T, origin="lower", cmap="viridis")

def update(frame):
    image.set_data(frame.T)
    return (image,)

animation = FuncAnimation(fig, update, frames=frames, interval=80, blit=True)
animation.save(str(figures_dir / "khi_movie.gif"), dpi=100)
