"""
Hydrostatic gas in a fixed 3D Plummer external potential stays near equilibrium (FD).

Sets up an isothermal atmosphere that exactly balances a Plummer external
potential, integrates it in time with the finite-difference solver, and checks
that the core stays close to hydrostatic equilibrium (small residual Mach number
and density drift). The z-midplane diagnostics are written to ``figures/``.
"""

# general
from pathlib import Path

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# astronomix constants
from astronomix import (
    CARTESIAN,
    OPEN_BOUNDARY,
)
from astronomix.option_classes.simulation_config import SIMPLE_SOURCE

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    SimulationConfig,
    SimulationParams,
)
from astronomix.option_classes.simulation_config import (
    StaticFloatVector,
    StaticIntVector,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)


# This is an equilibrium test where the residual we measure is tiny, so we run in
# double precision to keep round-off from masking the physical drift.
jax.config.update("jax_enable_x64", True)

# Figures are written to the local figures/ directory next to this test.
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------
# =============== ↓ Problem constants ↓ =======================
# -------------------------------------------------------------
N = 64
BOX = 8.0
A_PLUMMER = 1.0     # Plummer softening radius
GM = 1.0            # G * M
C2 = 1.0            # isothermal stratification constant (P = C2 * rho)
RHO0 = 1.0          # far-field density
GAMMA = 5 / 3
T_END = 1.0
R_CHECK = 2.0       # core measurement radius (away from boundaries)
C_S = (GAMMA * C2) ** 0.5   # uniform sound speed
# -------------------------------------------------------------
# =============== ↑ Problem constants ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Configuration and initial state ↓ =========
# -------------------------------------------------------------

# Open boundaries on every axis let the stratified atmosphere sit against a
# far-field density without reflecting waves back into the core.
open_boundary = BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY)
config = SimulationConfig(
    gravity_config=GravityConfig(
        self_gravity_version=SIMPLE_SOURCE,
        external_potential=True,
    ),
    dimensionality=3,
    geometry=CARTESIAN,
    box_size=StaticFloatVector(BOX, BOX, BOX),
    num_cells=StaticIntVector(N, N, N),
    boundary_settings=BoundarySettings(x=open_boundary, y=open_boundary, z=open_boundary),
    progress_bar=True,
)

registered_variables = get_registered_variables(config)
helper_data = get_helper_data(config)
params = SimulationParams(t_end=T_END, gamma=GAMMA)

# Radius from the box center, used both for the initial stratification and the
# core measurement mask below.
centers = helper_data.geometric_centers            # (N, N, N, 3)
box_center = BOX / 2.0
radius = jnp.sqrt(
    (centers[..., 0] - box_center) ** 2
    + (centers[..., 1] - box_center) ** 2
    + (centers[..., 2] - box_center) ** 2
)

# Plummer potential and the exact isothermal hydrostatic stratification that
# balances it, so an ideal solver would keep the state stationary.
phi = -GM / jnp.sqrt(radius ** 2 + A_PLUMMER ** 2)
rho_init = RHO0 * jnp.exp(-phi / C2)
p_init = C2 * rho_init
zero = jnp.zeros_like(radius)

state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho_init,
    velocity_x=zero,
    velocity_y=zero,
    velocity_z=zero,
    gas_pressure=p_init,
)

# The external potential lives on the bare grid (state-field shape, no ghosts).
params = params._replace(gravitational_potential=phi)

config = finalize_config(config, state.shape)
assert config.gravity_config.gravity, "finalize_config did not turn on the master gravity flag"

# -------------------------------------------------------------
# =============== ↑ Configuration and initial state ↑ =========
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Run and core diagnostics ↓ ================
# -------------------------------------------------------------

final_state = time_integration(state, config, params, registered_variables)

velocity_x = final_state[registered_variables.velocity_index.x]
velocity_y = final_state[registered_variables.velocity_index.y]
velocity_z = final_state[registered_variables.velocity_index.z]
mach = jnp.sqrt(velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2) / C_S
rho_final = final_state[registered_variables.density_index]
drho_rel = jnp.abs(rho_final - rho_init) / rho_init

# Restrict the equilibrium check to the core, away from the open boundaries where
# the far-field cut-off inevitably introduces some drift.
core = radius < R_CHECK
mach_core = float(jnp.max(jnp.where(core, mach, 0.0)))
drho_core = float(jnp.max(jnp.where(core, drho_rel, 0.0)))
print(f"max core Mach number: {mach_core:.3e}")
print(f"max core |drho|/rho:  {drho_core:.3e}")

# -------------------------------------------------------------
# =============== ↑ Run and core diagnostics ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Plotting ↓ ================================
# -------------------------------------------------------------

# z-midplane slices; the dashed circle marks the core measurement region.
kz = N // 2
extent = [0.0, BOX, 0.0, BOX]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

im0 = axes[0].imshow(
    np.asarray(rho_final[:, :, kz]).T,
    origin="lower",
    extent=extent,
    cmap="viridis",
)
axes[0].set_title("final density")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

drho_slice = np.maximum(np.asarray(drho_rel[:, :, kz]), 1e-16)
im1 = axes[1].imshow(
    drho_slice.T,
    origin="lower",
    extent=extent,
    cmap="magma",
    norm=LogNorm(),
)
axes[1].set_title(r"$|\Delta\rho|/\rho$")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

mach_slice = np.maximum(np.asarray(mach[:, :, kz]), 1e-16)
im2 = axes[2].imshow(
    mach_slice.T,
    origin="lower",
    extent=extent,
    cmap="inferno",
    norm=LogNorm(),
)
axes[2].set_title("Mach number")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

for ax in axes:
    ax.add_patch(plt.Circle((box_center, box_center), R_CHECK, fill=False, ec="cyan", ls="--", lw=0.8))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
fig.suptitle(f"3D Plummer hydrostatic equilibrium (FD, N={N}, max core Mach={mach_core:.1e})")
fig.tight_layout()
fig.savefig(figures_dir / "external_potential.png", dpi=150)

# -------------------------------------------------------------
# =============== ↑ Plotting ↑ ================================
# -------------------------------------------------------------
