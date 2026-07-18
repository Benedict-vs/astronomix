"""Fixed-timestep convergence of the energy error for the mild Evrard collapse.

Companion to ``evrard_energy_convergence`` (which varies the spatial resolution):
here the resolution is fixed at 32^3 and the *timestep* is varied, using the
fixed-timestep driver (``dt = t_end / num_timesteps``). It isolates the temporal
component of the energy error.

The non-conservative simple source has a purely spatial error, so its energy
error is flat in dt (pinned at the ~12% spatial floor at 32^3). The conservative
flux schemes conserve energy exactly in the continuous limit, so their energy
error is the temporal-integration error and converges at the time integrator's
order (RK4_SSP -> ~dt^4) until it reaches the round-off floor. In float32 that
floor is so high that the conservative error is instead dominated by round-off,
which *grows* with the number of steps (~1/dt); float64 exposes the true dt^4
convergence down to ~1e-13.

Run from the repo root:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/self_gravity/evrard_timestep_convergence.py --rerun --double
"""

# ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import sys

# jax
import jax

# ``--double`` reruns in float64 to expose the dt^4 convergence. x64 must be set
# before any array is created, hence before jax.numpy is imported below.
DOUBLE = "--double" in sys.argv
if DOUBLE:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    NATIVE_JAX,
    PALLAS,
)
from astronomix.option_classes.simulation_config import (
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)

# astronomix containers
from astronomix import SimulationParams

# astronomix functions
from astronomix import (
    get_helper_data,
    get_registered_variables,
    time_integration,
    construct_primitive_state,
    finalize_config,
)


# figure helpers
from _collapse import (
    GAMMA,
    collapse_config,
)
from _common import (
    DATA_DIR,
    FIG_DIR,
    SCHEMES,
    apply_paper_style,
)

RUN_BACKEND = NATIVE_JAX if DOUBLE else PALLAS
_SUFFIX = "_fp64" if DOUBLE else ""
_PREC_LABEL = "float64" if DOUBLE else "float32"

N = 32
T_END = 1.2
INITIAL_ENERGY = 0.20
NUM_TIMESTEPS = [250, 500, 1000, 2000, 4000, 8000, 16000]

DATA_FILE = DATA_DIR / f"evrard_timestep_convergence{_SUFFIX}.npz"
FIG_FILE = FIG_DIR / f"evrard_timestep_convergence{_SUFFIX}.svg"


def _run(version, num_timesteps):
    """Run one fixed-timestep collapse and return its final energy error.

    Args:
        version: The self-gravity source-term scheme.
        num_timesteps: The number of fixed steps to integrate ``T_END`` with
            (``dt = T_END / num_timesteps``).

    Returns:
        The final relative total-energy error ``|E(t_end) - E(0)| / |E(0)|``.
    """
    config = collapse_config(N, version, want_states=False, backend=RUN_BACKEND)
    config = config._replace(fixed_timestep=True, num_timesteps=int(num_timesteps))
    params = SimulationParams(
        t_end=T_END, dt_max=jnp.inf, minimum_density=1e-5, minimum_pressure=3e-6
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    R = 1.0
    M = 1.0
    rho = jnp.where(
        helper_data.r <= R, M / (2 * jnp.pi * R**2 * helper_data.r), 1e-4
    )
    v = jnp.zeros_like(rho)
    p = (GAMMA - 1) * rho * INITIAL_ENERGY
    p = jnp.where(p < params.minimum_pressure, params.minimum_pressure, p)
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=v,
        velocity_y=v,
        velocity_z=v,
        gas_pressure=p,
    )
    config = finalize_config(config, state.shape)
    snapshots = jax.block_until_ready(
        time_integration(state, config, params, registered_variables)
    )
    total_energy = snapshots.total_energy
    return float(
        jnp.abs(total_energy[-1] - total_energy[0]) / jnp.abs(total_energy[0])
    )


def run_and_cache():
    """Sweep the fixed timestep for every scheme and cache the energy errors."""
    timesteps = np.array([T_END / num_timesteps for num_timesteps in NUM_TIMESTEPS])
    errors = {scheme.self_gravity_version: [] for scheme in SCHEMES}
    for num_timesteps in NUM_TIMESTEPS:
        for scheme in SCHEMES:
            error = _run(scheme.self_gravity_version, num_timesteps)
            errors[scheme.self_gravity_version].append(error)
            print(
                f"  nts={num_timesteps:6d} dt={T_END / num_timesteps:.2e} "
                f"{scheme.label}: {error:.4e}",
                flush=True,
            )
    save_kwargs = {"dt": timesteps, "num_timesteps": np.array(NUM_TIMESTEPS)}
    for version, version_errors in errors.items():
        save_kwargs[f"energy_error_{version}"] = np.array(version_errors)
    np.savez(DATA_FILE, **save_kwargs)
    print(f"Saved -> {DATA_FILE}")


def plot():
    """Draw the two-panel fixed-timestep convergence figure from the cache.

    The left panel shows the (dt-flat) non-conservative simple source; the right
    panel shows the flux-based conservative schemes together with a reference
    slope (``dt^4`` in float64, ``dt^-1`` round-off growth in float32).
    """
    data = np.load(DATA_FILE)
    dt = data["dt"]

    apply_paper_style()
    by_version = {scheme.self_gravity_version: scheme for scheme in SCHEMES}
    panels = [
        ("non-conservative source", [SIMPLE_SOURCE]),
        (
            "flux-based (conservative) sources",
            [SECOND_ORDER_CONSERVATIVE, FOURTH_ORDER_CONSERVATIVE],
        ),
    ]

    order = np.argsort(dt)
    sorted_timesteps = dt[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, (subtitle, versions) in zip(axes, panels):
        for version in versions:
            scheme = by_version[version]
            ax.plot(
                dt,
                data[f"energy_error_{version}"],
                marker=scheme.marker,
                linestyle=scheme.linestyle,
                color=scheme.color,
                linewidth=2.0,
                label=scheme.label,
            )

        # Reference slope, on the conservative panel only. In float64 the error
        # converges at the RK4 order (dt^4) until the round-off floor; in float32
        # it is round-off-limited and grows ~1/dt.
        if FOURTH_ORDER_CONSERVATIVE in versions:
            conservative_sorted = data[
                f"energy_error_{FOURTH_ORDER_CONSERVATIVE}"
            ][order]
            if DOUBLE:
                # Anchor the guide only on the converging branch (dt >= dt_min).
                min_index = int(np.argmin(conservative_sorted))
                guide_timesteps = sorted_timesteps[min_index:]
                guide = conservative_sorted[-1] * (
                    guide_timesteps / sorted_timesteps[-1]
                ) ** 4
                ax.plot(
                    guide_timesteps,
                    guide,
                    color="0.4",
                    linestyle=":",
                    linewidth=1.5,
                    label=r"$\propto \Delta t^4$ (RK4)",
                )
            else:
                guide = conservative_sorted[-1] * (
                    sorted_timesteps / sorted_timesteps[-1]
                ) ** (-1)
                ax.plot(
                    sorted_timesteps,
                    guide,
                    color="0.4",
                    linestyle=":",
                    linewidth=1.5,
                    label=r"$\propto \Delta t^{-1}$ (round-off)",
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        # The simple source is essentially flat in dt; pad its y-range so the
        # near-constant line reads as flat rather than auto-zooming onto round-off.
        if versions == [SIMPLE_SOURCE]:
            simple_errors = data[f"energy_error_{SIMPLE_SOURCE}"]
            ax.set_ylim(float(simple_errors.min()) / 5, float(simple_errors.max()) * 5)
        ax.set_xlabel(r"timestep $\Delta t$")
        ax.set_title(subtitle)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel(r"final relative total energy error $|\Delta E| / |E_0|$")

    fig.suptitle(
        f"Mild Evrard collapse ($N = {N}^3$, $e_0 = {INITIAL_ENERGY}$, {_PREC_LABEL}): "
        "energy conservation vs fixed timestep"
    )
    fig.tight_layout()
    fig.savefig(FIG_FILE)
    print(f"Saved -> {FIG_FILE}")


def main():
    """Parse arguments, (re)run or load the cache, then draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="re-run simulations")
    parser.add_argument(
        "--double", action="store_true",
        help="run in float64 (native backend) to expose the dt^4 convergence",
    )
    args = parser.parse_args()
    if args.rerun or not DATA_FILE.exists():
        run_and_cache()
    else:
        print(f"Using cached results in {DATA_FILE} (pass --rerun to recompute).")
    plot()


if __name__ == "__main__":
    main()
