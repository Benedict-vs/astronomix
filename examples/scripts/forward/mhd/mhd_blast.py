"""3D MHD blast-wave test 1 (Seo & Ryu 2023, Sec. 3.7) — paper figures.

Two figures, both built from blast-wave simulations run with the *current*
astronomix solver and cached under ``data/`` (the old ``arena/results`` cache
was deleted, so we re-run the simulations here — pass ``--rerun`` to force a
recompute even when a cache exists):

1. ``mhd_blast_oscillations_comparison.png`` — a resolution-by-scheme grid of
   central density slices.  It shows how the finite-volume Lax/HLL schemes
   develop spurious post-shock oscillations while the finite-difference WENO
   scheme stays clean.

2. ``mhd_blast_test1_<N>cells.png`` — the finite-difference result at ``N``^3 in
   the first two columns (density / kinetic-energy / magnetic-pressure /
   pressure slices) and, in the third (profile) column, the ``|B|^2`` and
   pressure profiles along the box diagonal with the finite-volume (HLL,
   midpoint) result at the same resolution overlaid for comparison.

Run with the repo on PYTHONPATH:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/mhd/mhd_blast.py [--rerun]
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# jax
import jax
# The blast is heavy at high resolution; single precision (x32) roughly halves
# compute and memory and matches the precision the paper's arena runs used.
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    FINITE_DIFFERENCE,
    HLL,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import (
    LAX_FRIEDRICHS,
    IMPLICIT_MIDPOINT,
    IMPLICIT_EULER,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    get_registered_variables,
    finalize_config,
    time_integration,
    initialize_interface_fields,
    construct_primitive_state,
)


# The dimensionality-aware interface-field initializer for the constrained-
# transport (finite-difference) magnetic update.

# shared figure helpers
from _common import DATA_DIR
from _common import FIG_DIR
from _common import rerun_requested
from _common import FD_LABEL
from _common import FV_HLL_LABEL
from _common import FD_COLOR
from _common import FV_HLL_COLOR
from _common import mhd_registered_variables


# -------------------------------------------------------------
# ================= ↓ Test-wide constants ↓ ==================
# -------------------------------------------------------------

BOX_SIZE = 1.0        # periodic box of size 1.0 (Seo & Ryu setup)
T_END = 0.02          # end time of the blast-wave test
B0 = 10.0             # background field strength, B = (B0/√2, B0/√2, 0)
R0 = 0.125            # inner blast radius (p = 100 inside)
R1 = 1.1 * R0         # outer blast radius (linear pressure ramp between R0/R1)

# Resolutions down the rows of the comparison grid.  256^3 MHD is very heavy
# (see the cost note at the bottom of this file), so the default is the lighter
# [64, 128]; use [128, 256] to match the paper figure.
import sys as _sys
# Resolutions down the rows of the comparison grid. Default is the lighter
# [64, 128]; pass e.g. ``--res 128 256`` to match the paper figure
# (mhd_blast_test1_256cells / oscillation comparison at 256^3).
if "--res" in _sys.argv:
    _i = _sys.argv.index("--res")
    RESOLUTIONS = [int(a) for a in _sys.argv[_i + 1:] if a.isdigit()] or [64, 128]
else:
    RESOLUTIONS = [64, 128]

# Solver-scheme columns: cache/config name -> plot label.  Each name resolves
# to a concrete SimulationConfig via ``SCHEME_CONFIGS`` below.
CONFIGS = [
    ("fv_mhd_lax_mid", "FV (Lax)"),
    ("fv_mhd_hll_mid", "FV (HLL)"),
    ("fv_mhd_hll_eul", "FV (HLL, Euler)"),
    ("fd_mhd", "FD"),
]

# Per-scheme solver settings, extracted from the original arena run scripts
# (fv_mhd_lax_mid / fv_mhd_hll_mid / fv_mhd_hll_eul / fd_mhd).  ``config`` holds
# SimulationConfig overrides and ``params`` holds SimulationParams overrides.
SCHEME_CONFIGS = {
    "fv_mhd_lax_mid": {
        "config": dict(
            solver_mode=FINITE_VOLUME,
            riemann_solver=LAX_FRIEDRICHS,
            fv_magnetic_integrator=IMPLICIT_MIDPOINT,
        ),
        "params": dict(C_cfl=0.8),
    },
    "fv_mhd_hll_mid": {
        "config": dict(
            solver_mode=FINITE_VOLUME,
            riemann_solver=HLL,
            fv_magnetic_integrator=IMPLICIT_MIDPOINT,
        ),
        "params": dict(C_cfl=0.8),
    },
    "fv_mhd_hll_eul": {
        "config": dict(
            solver_mode=FINITE_VOLUME,
            riemann_solver=HLL,
            fv_magnetic_integrator=IMPLICIT_EULER,
        ),
        "params": dict(C_cfl=0.8),
    },
    "fd_mhd": {
        "config": dict(
            donate_state=False,
        ),
        "params": dict(C_cfl=1.5, minimum_density=1e-3, minimum_pressure=1e-3),
    },
}

# 3D periodic boundaries; ``finalize_config`` promotes this to the faster
# PERIODIC_ROLL handling automatically for a fully-periodic 3D box.
PERIODIC_3D = BoundarySettings(
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)

# Variable registries so we index final_state by name (rv.density_index etc.)
# rather than by hard-coded integers; the finite-volume and finite-difference
# state layouts differ, so each backend gets its own registry.
RV_FV = mhd_registered_variables(FINITE_VOLUME)
RV_FD = mhd_registered_variables(FINITE_DIFFERENCE)

# -------------------------------------------------------------
# ================= ↑ Test-wide constants ↑ ==================
# -------------------------------------------------------------


def run_and_cache(config_name, num_cells):
    """Run the blast-wave test for one scheme and cache the final state.

    Builds the Seo & Ryu blast initial condition at ``num_cells``^3, configures
    the solver according to ``SCHEME_CONFIGS[config_name]``, evolves it to
    ``T_END`` with the current astronomix ``time_integration`` and writes the
    final state to ``data/mhd_blast_<config_name>_<N>cells.npz``.

    Args:
        config_name: The scheme key (e.g. ``"fd_mhd"``), a key of
            ``SCHEME_CONFIGS``.
        num_cells: The per-dimension resolution of the run.

    Returns:
        The final ``primitive_state`` array as a numpy array.
    """

    # -------------------------------------------------------------
    # =============== ↓ Solver configuration ↓ ===================
    # -------------------------------------------------------------

    scheme = SCHEME_CONFIGS[config_name]
    config = SimulationConfig(
        mhd=True,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        boundary_settings=PERIODIC_3D,
        progress_bar=True,
        **scheme["config"],
    )
    registered_variables = get_registered_variables(config)

    # -------------------------------------------------------------
    # =============== ↑ Solver configuration ↑ ===================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Blast-wave initial condition ↓ ==============
    # -------------------------------------------------------------

    # Cell-centre coordinates of the periodic box and the radial distance from
    # the box centre.  (helper_data.r is only populated for curvilinear
    # geometries, so the radius is built explicitly here.)
    dx = BOX_SIZE / num_cells
    x = jnp.linspace(dx / 2, BOX_SIZE - dx / 2, num_cells)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    centre = BOX_SIZE / 2
    r = jnp.sqrt((X - centre) ** 2 + (Y - centre) ** 2 + (Z - centre) ** 2)

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r)
    P = jnp.where(r <= R0, 100.0, P)
    P = jnp.where((r > R0) & (r <= R1), 1.0 + 99.0 * (R1 - r) / (R1 - R0), P)
    P = jnp.where(r > R1, 1.0, P)

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    B_x = jnp.ones_like(r) * (B0 / jnp.sqrt(2))
    B_y = jnp.ones_like(r) * (B0 / jnp.sqrt(2))
    B_z = jnp.zeros_like(r)

    # The finite-difference (constrained-transport) solver additionally needs
    # the staggered interface magnetic fields; the finite-volume solver does not.
    if config.solver_mode == FINITE_DIFFERENCE:
        bxb, byb, bzb = initialize_interface_fields(
            B_x, B_y, B_z, config.dimensionality
        )
    else:
        bxb, byb, bzb = None, None, None

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=V_x,
        velocity_y=V_y,
        velocity_z=V_z,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
        gas_pressure=P,
    )

    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, **scheme["params"])

    # -------------------------------------------------------------
    # ============ ↑ Blast-wave initial condition ↑ ==============
    # -------------------------------------------------------------

    print(f"running MHD blast {config_name} at {num_cells}^3 ...")
    final_state = np.asarray(time_integration(
        initial_state, config, params, registered_variables
    ))
    path = DATA_DIR / f"mhd_blast_{config_name}_{num_cells}cells.npz"
    np.savez(path, final_state=final_state)
    print(f"  cached -> {path}")
    return final_state


def load_final_state(config_name, num_cells, rerun=False):
    """Return a blast-wave final state, from cache or by (re-)running.

    Args:
        config_name: The scheme key (e.g. ``"fd_mhd"``).
        num_cells: The per-dimension resolution of the run.
        rerun: When True, ignore any cache and re-run the simulation.

    Returns:
        The ``final_state`` array for that run.
    """
    path = DATA_DIR / f"mhd_blast_{config_name}_{num_cells}cells.npz"
    if path.exists() and not rerun:
        return np.load(path)["final_state"]
    return run_and_cache(config_name, num_cells)


def make_oscillation_comparison(rerun=False):
    """Build the resolution-by-scheme oscillation-comparison figure.

    A grid of central density slices with resolution down the rows and solver
    scheme across the columns.  It shows the finite-volume Lax/HLL schemes
    developing spurious post-shock oscillations while the finite-difference
    WENO scheme stays clean.  Saves ``mhd_blast_oscillations_comparison.png``.

    Args:
        rerun: When True, force the underlying simulations to be re-run.
    """

    # -------------------------------------------------------------
    # ================= ↓ Panel grid layout ↓ ====================
    # -------------------------------------------------------------

    resolutions = RESOLUTIONS
    configs = CONFIGS

    panel_size = 3
    fig, axs = plt.subplots(
        len(resolutions),
        len(configs),
        figsize=(panel_size * len(configs), panel_size * len(resolutions)),
    )

    # -------------------------------------------------------------
    # ================= ↑ Panel grid layout ↑ ====================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============= ↓ Density slices per panel ↓ =================
    # -------------------------------------------------------------

    # Fix the colour scale across every panel so the schemes are directly
    # comparable by eye.
    density_min, density_max = 0.0, 1.5

    for row_index, num_cells in enumerate(resolutions):
        for column_index, (config_name, title) in enumerate(configs):
            final_state = load_final_state(config_name, num_cells, rerun)
            registered_variables = RV_FD if config_name == "fd_mhd" else RV_FV
            density_slice = final_state[
                registered_variables.density_index, :, :, num_cells // 2
            ]
            image = axs[row_index, column_index].imshow(
                density_slice,
                vmin=density_min,
                vmax=density_max,
                cmap="jet",
            )
            if row_index == 0:
                axs[row_index, column_index].set_title(title)
            axs[row_index, column_index].set_xticks([])
            axs[row_index, column_index].set_yticks([])

    for row_index, num_cells in enumerate(resolutions):
        axs[row_index, 0].set_ylabel(f"{num_cells}$^3$ cells", fontsize=12)

    # -------------------------------------------------------------
    # ============= ↑ Density slices per panel ↑ =================
    # -------------------------------------------------------------

    # A single shared horizontal colour bar underneath the grid.
    fig.subplots_adjust(bottom=0.18)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.035])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("density", rotation=0, labelpad=8, fontsize=11)
    cbar.ax.xaxis.set_label_position("bottom")

    out = FIG_DIR / "mhd_blast_oscillations_comparison.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def make_fd_with_profiles(num_cells=RESOLUTIONS[-1], rerun=False):
    """Build the finite-difference slices + diagonal-profile figure.

    Columns 0-1 show the finite-difference result at ``num_cells``^3 (density,
    kinetic energy, magnetic pressure and pressure slices through the box
    centre). Column 2 shows the ``|B|^2`` and pressure profiles along the box
    diagonal with the finite-volume (HLL, midpoint) result overlaid for
    comparison. Saves ``mhd_blast_test1_{num_cells}cells.png``.

    Args:
        num_cells: The per-dimension resolution of the runs to plot.
        rerun: When True, force the underlying simulations to be re-run.
    """

    # -------------------------------------------------------------
    # ============ ↓ Load states and extract slices ↓ ============
    # -------------------------------------------------------------

    fd_final_state = load_final_state("fd_mhd", num_cells, rerun)
    fv_final_state = load_final_state("fv_mhd_hll_mid", num_cells, rerun)

    mid = num_cells // 2
    extent = (0, BOX_SIZE, 0, BOX_SIZE)

    def slices(state, registered_variables):
        """Return the central (density, pressure, |B|^2, v^2) slices of a
        state, indexed through ``registered_variables``."""
        magnetic_index = registered_variables.magnetic_index
        velocity_index = registered_variables.velocity_index
        density = state[registered_variables.density_index, :, :, mid]
        pressure = state[registered_variables.pressure_index, :, :, mid]
        b_sq = (
            state[magnetic_index.x] ** 2
            + state[magnetic_index.y] ** 2
            + state[magnetic_index.z] ** 2
        )[:, :, mid]
        v_sq = (
            state[velocity_index.x] ** 2
            + state[velocity_index.y] ** 2
            + state[velocity_index.z] ** 2
        )[:, :, mid]
        return density, pressure, b_sq, v_sq

    density, pressure, b_sq, v_sq = slices(fd_final_state, RV_FD)

    # -------------------------------------------------------------
    # ============ ↑ Load states and extract slices ↑ ============
    # -------------------------------------------------------------

    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))

    def imshow_panel(ax, field, label):
        """Draw one field slice as an image panel with an attached colour bar."""
        image = ax.imshow(field.T, origin="lower", extent=extent, cmap="jet")
        cbar = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        fig.colorbar(image, cax=cbar, label=label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # -------------------------------------------------------------
    # ============= ↓ Columns 0-1: FD slices ↓ ===================
    # -------------------------------------------------------------

    imshow_panel(axs[0, 0], density, "density")
    imshow_panel(axs[0, 1], v_sq, r"$v^2$")
    imshow_panel(axs[1, 0], b_sq, r"$B^2$")
    imshow_panel(axs[1, 1], pressure, "pressure")

    # -------------------------------------------------------------
    # ============= ↑ Columns 0-1: FD slices ↑ ===================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ======== ↓ Column 2: diagonal profiles, FD vs FV ↓ =========
    # -------------------------------------------------------------

    diagonal = np.arange(num_cells)
    r_diagonal = np.sqrt(2.0) * diagonal * (BOX_SIZE / num_cells)

    _, fv_pressure_slice, fv_b_sq, _ = slices(fv_final_state, RV_FV)

    b_diagonal_fd = b_sq[diagonal, diagonal]
    b_diagonal_fv = fv_b_sq[diagonal, diagonal]
    axs[0, 2].plot(r_diagonal, b_diagonal_fd, color=FD_COLOR, label=FD_LABEL)
    axs[0, 2].plot(
        r_diagonal,
        b_diagonal_fv,
        color=FV_HLL_COLOR,
        ls="--",
        label=FV_HLL_LABEL,
    )
    axs[0, 2].set_ylabel(r"$|B|^2$")
    axs[0, 2].set_xlabel("diagonal")
    axs[0, 2].legend(fontsize=8, loc="lower right")

    pressure_diagonal_fd = pressure[diagonal, diagonal]
    pressure_diagonal_fv = fv_pressure_slice[diagonal, diagonal]
    axs[1, 2].plot(r_diagonal, pressure_diagonal_fd, color=FD_COLOR, label=FD_LABEL)
    axs[1, 2].plot(
        r_diagonal,
        pressure_diagonal_fv,
        color=FV_HLL_COLOR,
        ls="--",
        label=FV_HLL_LABEL,
    )
    axs[1, 2].set_ylabel("pressure")
    axs[1, 2].set_xlabel("diagonal")
    axs[1, 2].legend(fontsize=8, loc="lower right")

    # -------------------------------------------------------------
    # ======== ↑ Column 2: diagonal profiles, FD vs FV ↑ =========
    # -------------------------------------------------------------

    plt.tight_layout()
    out = FIG_DIR / f"mhd_blast_test1_{num_cells}cells.png"
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    rerun = rerun_requested()
    make_oscillation_comparison(rerun)
    make_fd_with_profiles(RESOLUTIONS[-1], rerun)
