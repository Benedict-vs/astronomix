"""
Cross-checks of the model-B (ki02gf12) cooling operator variants.

Applies ``update_pressure_by_cooling`` once to the same state with

  * the native sub-cycled update (the reference),
  * the fused Pallas kernel (``CoolingConfig.pallas_subcycling``; interpreted
    on CPU, compiled with Triton on a GPU),
  * the cached FUV geometry (``CoolingConfig.cache_fuv_geometry``),
  * the tabulated exact integrator (``TABULATED_EXACT_COOLING``), against the
    native update converged with many sub-cycles,

and prints the differences in the new pressure. The state is a synthetic one
spanning the density / temperature range of the wind (default) or a sub-box of
an Orbax checkpoint (``--checkpoint DIR --step K``).

CPU:  JAX_PLATFORMS=cpu CGOLS_PALLAS_INTERPRET=1 CGOLS_COOLING=1 CGOLS_DIM=32
      CGOLS_IC_MODE=insitu CGOLS_RUN_TAG=_Bcheck python cgols_cooling_operator_check.py
"""

# ruff: noqa: E402
import argparse
import os
import time
from typing import NamedTuple

os.environ.setdefault("CGOLS_COOLING", "1")
os.environ.setdefault("CGOLS_IC_MODE", "insitu")
os.environ.setdefault("CGOLS_RUN_TAG", "_Bcheck")

import cgols as cg
import jax
import jax.numpy as jnp
import numpy as np

from astronomix import SimulationParams
from astronomix._modules._cooling import _cooling as cooling
from astronomix._modules._cooling import cooling_options
from astronomix.option_classes.simulation_config import NATIVE_JAX


class CellCenters(NamedTuple):
    """The three 1D cell-centre axes the FUV field needs."""

    cell_centers_x: jnp.ndarray
    cell_centers_y: jnp.ndarray
    cell_centers_z: jnp.ndarray


def synthetic_state(registered_variables, shape, cooling_params, seed=0):
    """Log-uniform density and temperature over the range of the wind."""
    key_density, key_temperature = jax.random.split(jax.random.PRNGKey(seed))
    density = 10 ** jax.random.uniform(key_density, shape, minval=-4.5, maxval=4.5)
    floor = cooling_params.floor_temperature
    temperature = floor * 10 ** jax.random.uniform(
        key_temperature, shape, minval=0.0, maxval=7.5
    )
    pressure = cooling.get_pressure_from_temperature(
        density,
        temperature,
        cooling_params.hydrogen_mass_fraction,
        cooling_params.metal_mass_fraction,
        cooling_params.mean_molecular_weight,
    )
    state = jnp.zeros((registered_variables.num_vars,) + shape)
    state = state.at[registered_variables.density_index].set(density)
    return state.at[registered_variables.pressure_index].set(pressure)


def checkpoint_subbox(directory, step, shape):
    """The central ``shape`` sub-box of a checkpoint and its first-cell offsets."""
    from astronomix._snapshotting._orbax_storage import load_loop_checkpoint

    checkpoint = load_loop_checkpoint(directory, step)
    full = checkpoint.primitive_state
    offsets = tuple((n - m) // 2 for n, m in zip(full.shape[1:], shape))
    box = full[
        :,
        offsets[0]: offsets[0] + shape[0],
        offsets[1]: offsets[1] + shape[1],
        offsets[2]: offsets[2] + shape[2],
    ]
    return box, offsets, float(checkpoint.time), full.shape[1]


def report(name, new, reference, seconds=None):
    """Print the relative pressure differences of ``new`` to ``reference``."""
    new, reference = np.asarray(new, np.float64), np.asarray(reference, np.float64)
    relative = np.abs(new - reference) / np.abs(reference)
    timing = "" if seconds is None else f"  [{seconds:.3f} s/call]"
    print(
        f"{name:<46s} max rel dP {relative.max():.3e}  mean {relative.mean():.3e}  "
        f"cells > 1e-5: {(relative > 1e-5).mean():.4%}  > 1e-2: {(relative > 1e-2).mean():.4%}"
        f"{timing}"
    )
    return relative.max()


def main():
    """Run the cross-checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--shape", type=int, nargs=3, default=(32, 32, 64))
    parser.add_argument("--dt", type=float, default=7.5e-4)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    shape = tuple(arguments.shape)

    config, registered_variables = cg.build_config()
    params = SimulationParams(gamma=cg.gamma, **cg._cooling_params_kwargs())
    cooling_params = params.cooling_params

    if arguments.checkpoint:
        state, offsets, t, resolution = checkpoint_subbox(
            arguments.checkpoint, arguments.step, shape
        )
        dx = 10.0 / resolution  # the box is 10 code lengths (kpc) wide
        print(f"checkpoint step {arguments.step} (t = {t:.4f}), sub-box {shape} at {offsets}")
    else:
        dx = float(config.grid_spacing)
        offsets = tuple((n - m) // 2 for n, m in zip(config.num_cells, shape))
        t = 5.6
        state = synthetic_state(registered_variables, shape, cooling_params)
        print(f"synthetic state {shape}")
    centers = CellCenters(
        *((offset + jnp.arange(n) + 0.5) * dx for offset, n in zip(offsets, shape))
    )
    pressure_index = registered_variables.pressure_index
    native_backend = config.backend_config._replace(backend=NATIVE_JAX)

    def operator(cooling_config, backend_config, operator_params=params):
        def call():
            return jax.block_until_ready(
                cooling.update_pressure_by_cooling(
                    state,
                    registered_variables,
                    cooling_config,
                    operator_params,
                    arguments.dt,
                    grid_spacing=dx,
                    helper_data=centers,
                    time=t,
                    backend_config=backend_config,
                )[pressure_index]
            )

        result = call()
        tic = time.perf_counter()
        for _ in range(arguments.repeats):
            result = call()
        return result, (time.perf_counter() - tic) / arguments.repeats

    base = config.cooling_config
    print(f"backend {jax.default_backend()}, pallas_interpret={config.backend_config.pallas_interpret}, "
          f"dt = {arguments.dt:g}, {base.max_subcycles} sub-cycles")
    reference, seconds = operator(base, native_backend)
    report("native sub-cycled (reference) vs itself", reference, reference, seconds)

    worst = {}
    result, seconds = operator(base._replace(pallas_subcycling=True), config.backend_config)
    worst["pallas"] = report("Pallas kernel vs native", result, reference, seconds)

    # Reverse-mode gradient through the Pallas dispatch (native tangent rule).
    def summed_pressure(cooling_config, backend_config):
        def loss(input_state):
            return jnp.sum(
                cooling.update_pressure_by_cooling(
                    input_state, registered_variables, cooling_config, params,
                    arguments.dt, grid_spacing=dx, helper_data=centers, time=t,
                    backend_config=backend_config,
                )[pressure_index]
            )
        return jax.grad(loss)(state)

    gradient_native = summed_pressure(base, native_backend)
    gradient_pallas = summed_pressure(
        base._replace(pallas_subcycling=True), config.backend_config
    )
    gradient_scale = float(jnp.max(jnp.abs(gradient_native)))
    gradient_difference = float(jnp.max(jnp.abs(gradient_pallas - gradient_native)))
    print(f"{'grad of sum(P) w.r.t. the state, Pallas vs native':<46s} max abs diff "
          f"{gradient_difference:.3e} (max |grad| {gradient_scale:.3e})")
    worst["gradient"] = gradient_difference / gradient_scale

    fuv_params = cooling_params.fuv_source_params
    cached = params._replace(
        cooling_params=cooling_params._replace(
            fuv_source_params=fuv_params._replace(
                los_factor_cache=cooling.fuv_line_of_sight_factor(fuv_params, *centers)
            )
        )
    )
    result, seconds = operator(base._replace(cache_fuv_geometry=True), native_backend, cached)
    worst["cache"] = report("cached FUV geometry vs native", result, reference, seconds)

    if hasattr(cooling_options, "TABULATED_EXACT_COOLING"):
        converged, _ = operator(base._replace(max_subcycles=2048), native_backend)
        report("native 32 sub-cycles vs 2048 sub-cycles", reference, converged)
        result, seconds = operator(
            base._replace(cooling_method=cooling_options.TABULATED_EXACT_COOLING),
            native_backend,
        )
        worst["tabulated"] = report(
            "tabulated exact integrator vs 2048 sub-cycles", result, converged, seconds
        )

    assert worst["pallas"] < 1e-4, "Pallas kernel deviates from the native update"
    assert worst["cache"] < 1e-6, "cached FUV geometry deviates"
    assert worst["gradient"] < 1e-5, "gradient through the Pallas dispatch deviates"


if __name__ == "__main__":
    main()
