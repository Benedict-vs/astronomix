"""
First-order-flux-correction and floor census of a finite-volume CGOLS run.

The VL2 scheme of ``CGOLS_SOLVER=fv`` keeps positivity with AthenaPK's
first-order flux correction (FOFC: LLF fluxes around every cell whose
high-order update would go non-positive) and the density / pressure floors.
Neither reports how often it acts, so this script measures it: it restores the
run's Orbax checkpoints (one per frame, kept with ``CGOLS_CHECKPOINT_KEEP=0``),
advances each by a few steps on the native backend (bit-equivalent to the
Pallas stages up to rounding) with the correction wrapped by a counter, and
reports per checkpoint

* the cells whose uncorrected stage update has a non-positive density
  (density failures) or only a non-positive pressure (pressure failures), per
  stage, and
* the cells sitting at the density / pressure floor after the steps, with the
  mass the density floor holds there.

Run on the CPU with the same CGOLS_* environment as the run (CGOLS_SOLVER=fv,
CGOLS_DIM, CGOLS_RUN_TAG)::

    JAX_PLATFORMS=cpu CGOLS_SOLVER=fv CGOLS_DIM=128 CGOLS_RUN_TAG=_A128fv \\
        python cgols_fv_fofc_census.py [--every 6] [--steps 3]
"""

# ruff: noqa: E402

# general
import argparse
import os

os.environ.setdefault("CGOLS_SHARD_SPLIT", "(1, 1, 1, 1)")
os.environ.setdefault("CGOLS_IC_MODE", "insitu")

# numerics
import numpy as np
import jax
import jax.numpy as jnp

# cgols (module constants and builders only; the run is guarded by __main__)
import cgols
from cgols import (
    RUN_TAG,
    _data,
    build_config,
    build_external_potential,
    build_simulation_params,
    code_units,
)

# astronomix
from astronomix import restart_from_latest_checkpoint, time_integration
from astronomix.option_classes.simulation_config import (
    NATIVE_JAX,
    ON_DEVICE,
    finalize_config,
)
import astronomix._finite_volume._state_evolution._van_leer_integrator as vl2

# -------------------------------------------------------------
# ================= ↓ Counting FOFC wrapper ↓ =================
# -------------------------------------------------------------

#: Per-stage (density failures, pressure failures) of the current census.
STAGE_FAILURES = []

_original_correction = vl2._first_order_flux_correction


def _record_failures(density_failures, pressure_failures):
    """Host side of the counter."""
    STAGE_FAILURES.append((int(density_failures), int(pressure_failures)))


def _counting_correction(
    fluxes_per_axis,
    stage_primitive_state,
    base_conserved_state,
    stage_time_step,
    gamma,
    cleaning_speed,
    config,
    registered_variables,
):
    """The FOFC, after counting the failures of the uncorrected update."""
    uncorrected = base_conserved_state - stage_time_step * vl2._flux_divergence(
        fluxes_per_axis,
        config,
    )
    failure_codes = vl2._positivity_failure_code(
        list(uncorrected),
        config,
        registered_variables,
    )
    failure_codes = jnp.where(
        vl2._interior_mask(stage_primitive_state, config),
        failure_codes,
        vl2.POSITIVE,
    )
    jax.debug.callback(
        _record_failures,
        jnp.sum(failure_codes == vl2.DENSITY_FAILURE),
        jnp.sum(failure_codes == vl2.PRESSURE_FAILURE),
    )
    return _original_correction(
        fluxes_per_axis,
        stage_primitive_state,
        base_conserved_state,
        stage_time_step,
        gamma,
        cleaning_speed,
        config,
        registered_variables,
    )


vl2._first_order_flux_correction = _counting_correction

# -------------------------------------------------------------
# ================= ↑ Counting FOFC wrapper ↑ =================
# -------------------------------------------------------------


def main():
    """Restore every ``--every``-th checkpoint and count over ``--steps`` steps."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--every", type=int, default=6)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="the run's typical time step in code units (from its diag log)",
    )
    arguments = parser.parse_args()

    if cgols.SOLVER != "fv":
        raise SystemExit("set CGOLS_SOLVER=fv")

    config, registered_variables = build_config()
    # Native stages (the counter sits in the native FOFC), no frames, no
    # checkpoints, no progress output.
    config = config._replace(
        backend_config=config.backend_config._replace(backend=NATIVE_JAX),
        activate_snapshot_callback=False,
        snapshot_storage_mode=ON_DEVICE,
        snapshot_storage_path=None,
        progress_bar=False,
        monitor_diagnostics=False,
        memory_analysis=False,
        donate_state=False,
    )
    resolution = cgols.RESOLUTION
    config = finalize_config(
        config,
        (registered_variables.num_vars, resolution, resolution, 2 * resolution),
    )
    params = build_simulation_params(build_external_potential(config))
    cell_volume = config.grid_spacing**3

    checkpoint_dir = _data(f"cgols_checkpoints{RUN_TAG}")
    steps = sorted(int(entry) for entry in os.listdir(checkpoint_dir) if entry.isdigit())
    print(f"{len(steps)} checkpoints in {checkpoint_dir}; census every {arguments.every}")
    print(
        "  t[Myr]  stages  dens.fail/stage  pres.fail/stage  max dens.fail  "
        "max pres.fail  n@rho_floor  n@P_floor  floor mass"
    )
    for step in steps[:: arguments.every]:
        state, step_params, restart_state = restart_from_latest_checkpoint(
            checkpoint_dir,
            params,
            step=step,
        )
        time = float(step_params.t_start)
        # About --steps CFL steps.
        step_params = step_params._replace(t_end=time + arguments.steps * arguments.dt)
        STAGE_FAILURES.clear()
        final_state = time_integration(
            state,
            config,
            step_params,
            registered_variables,
            restart_state=restart_state,
        )
        final_state = np.asarray(final_state)
        density = final_state[registered_variables.density_index]
        pressure = final_state[registered_variables.pressure_index]
        at_density_floor = density <= params.minimum_density * (1 + 1e-6)
        at_pressure_floor = pressure <= params.minimum_pressure * (1 + 1e-6)
        failures = np.array(STAGE_FAILURES) if STAGE_FAILURES else np.zeros((1, 2))
        time_myr = (time * code_units.code_time).to("Myr").value
        print(
            f"  {time_myr:6.2f}  {len(failures):6d}  {failures[:, 0].mean():15.2f}  "
            f"{failures[:, 1].mean():15.2f}  {int(failures[:, 0].max()):13d}  "
            f"{int(failures[:, 1].max()):13d}  {at_density_floor.sum():11d}  "
            f"{at_pressure_floor.sum():9d}  "
            f"{(density * at_density_floor).sum() * cell_volume:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
