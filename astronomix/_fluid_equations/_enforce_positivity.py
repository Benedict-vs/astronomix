"""
Here we protect the density and pressure from going negative.

In my view this is a bit of a shady practice, hiding unphysical
updates under the rug. However, it is common practice.
"""

import itertools

import jax.numpy as jnp
import jax
from functools import partial

from jaxtyping import Array, Float

from typing import Union

from astronomix._pallas_helpers import diffable_pallas_call_n
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    IDEAL_GAS,
    POSITIVITY_HARD_FLOOR,
    POSITIVITY_REDISTRIBUTE,
    STATE_TYPE,
    SimulationConfig,
)


def _enforce_positivity_native(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    return _enforce_positivity_native_impl(
        conserved_state, config, gamma,
        minimum_density, minimum_pressure, registered_variables,
    )


@partial(
    jax.jit, static_argnames=["registered_variables", "config"]
)
def _enforce_positivity(
    conserved_state: STATE_TYPE,
    config: SimulationConfig,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    if _enforce_positivity_pallas_supported(conserved_state, config):
        pallas = lambda s, g, mr, mp: _enforce_positivity_pallas(  # noqa: E731
            s, config, g, mr, mp, registered_variables,
        )
        native = lambda s, g, mr, mp: _enforce_positivity_native(  # noqa: E731
            s, g, mr, mp, config, registered_variables,
        )
        return diffable_pallas_call_n(
            (conserved_state, gamma, minimum_density, minimum_pressure),
            pallas_branch=pallas, native_branch=native,
        )
    return _enforce_positivity_native(
        conserved_state, gamma, minimum_density, minimum_pressure,
        config, registered_variables,
    )


def _enforce_positivity_native_impl(
    conserved_state: STATE_TYPE,
    config: SimulationConfig,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    # Optional NaN/inf backstop: jnp.maximum(NaN, floor) = NaN, so a non-finite
    # cell would otherwise survive the floor and propagate. Reset any non-finite
    # entry to zero first; the density/pressure floors below then turn it into a
    # valid floor state (rho -> minimum_density, momentum 0, pressure floored).
    if config.positivity_nan_safe:
        conserved_state = jnp.nan_to_num(
            conserved_state, nan=0.0, posinf=0.0, neginf=0.0
        )

    rho = conserved_state[registered_variables.density_index]

    # Vacuum-rest: cells below the density floor are effectively vacuum; zero
    # their momentum so the recovered velocity is 0 rather than
    # momentum / minimum_density (which spikes and triggers high-Mach blow-up).
    # Done before the floor + energy recovery so the kinetic energy is consistent.
    if config.positivity_vacuum_rest:
        floored = rho < minimum_density
        mom = registered_variables.momentum_index
        if config.dimensionality == 1:
            mom_indices = (mom,)
        elif config.dimensionality == 2:
            mom_indices = (mom.x, mom.y)
        else:
            mom_indices = (mom.x, mom.y, mom.z)
        for mi in mom_indices:
            conserved_state = conserved_state.at[mi].set(
                jnp.where(floored, 0.0, conserved_state[mi])
            )

    # enforce minimum density
    rho = jnp.maximum(rho, minimum_density)

    # the energy only needs to be updated in the ideal gas case
    if config.equation_of_state == IDEAL_GAS:

        if config.dimensionality == 1:
            v_x = conserved_state[registered_variables.momentum_index] / rho
        else:
            v_x = conserved_state[registered_variables.momentum_index.x] / rho

        if config.dimensionality == 2:
            v_y = conserved_state[registered_variables.momentum_index.y] / rho
            v_z = 0.0
        elif config.dimensionality == 3:
            v_y = conserved_state[registered_variables.momentum_index.y] / rho
            v_z = conserved_state[registered_variables.momentum_index.z] / rho

        energy = conserved_state[registered_variables.energy_index]

        if config.mhd:
            B_x = conserved_state[registered_variables.magnetic_index.x]
            B_y = conserved_state[registered_variables.magnetic_index.y]
            B_z = conserved_state[registered_variables.magnetic_index.z]

            b2 = B_x**2 + B_y**2 + B_z**2
        
        if config.dimensionality == 1:
            v2 = v_x**2
        elif config.dimensionality == 2:
            v2 = v_x**2 + v_y**2
        elif config.dimensionality == 3:
            v2 = v_x**2 + v_y**2 + v_z**2

        # calculate pressure
        if config.mhd:
            pressure = (gamma - 1.0) * (energy - 0.5 * rho * v2 - 0.5 * b2)
        else:
            pressure = (gamma - 1.0) * (energy - 0.5 * rho * v2)
        
        pressure = jnp.maximum(pressure, minimum_pressure)

        # redefine energy with new pressure
        if config.mhd:
            energy = pressure / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * b2
        else:
            energy = pressure / (gamma - 1.0) + 0.5 * rho * v2

        # reconstruct conserved state
        conserved_state = conserved_state.at[registered_variables.energy_index].set(energy)
    
    # for both the ideal gas and isothermal case, we need to update the density
    conserved_state = conserved_state.at[registered_variables.density_index].set(rho)

    return conserved_state

def _momentum_indices(config, registered_variables):
    """Conserved-state momentum component indices for the active dimensionality."""
    if config.dimensionality == 1:
        return [registered_variables.momentum_index]
    if config.dimensionality == 2:
        return [registered_variables.momentum_index.x,
                registered_variables.momentum_index.y]
    return [registered_variables.momentum_index.x,
            registered_variables.momentum_index.y,
            registered_variables.momentum_index.z]


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _redistribute_positivity(
    conserved_state: STATE_TYPE,
    threshold: Union[float, Float[Array, ""]],
    max_velocity: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Dispatch: Pallas neighbour-redistribution kernel when supported, else the
    native-JAX stencil. AD (jvp/vjp) routes through the native branch."""
    if _redistribute_positivity_pallas_supported(conserved_state, config):
        pallas = lambda s, thr, vmax, g, pmin: _redistribute_positivity_pallas(  # noqa: E731
            s, thr, vmax, g, pmin, config, registered_variables,
        )
        native = lambda s, thr, vmax, g, pmin: _redistribute_positivity_native(  # noqa: E731
            s, thr, vmax, g, pmin, config, registered_variables,
        )
        return diffable_pallas_call_n(
            (conserved_state, threshold, max_velocity, gamma, minimum_pressure),
            pallas_branch=pallas, native_branch=native,
        )
    return _redistribute_positivity_native(
        conserved_state, threshold, max_velocity, gamma, minimum_pressure,
        config, registered_variables,
    )


def _redistribute_positivity_native(
    conserved_state: STATE_TYPE,
    threshold: Union[float, Float[Array, ""]],
    max_velocity: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Neighbour redistribution of sub-threshold (vacuum) cells, a conserved-state
    generalisation of the HOW-MHD *isothermal* ``prot.f``.

    For every cell with ``rho <= threshold`` the density and momentum (and, for
    ideal gas, the total energy) are replaced by the average over the valid
    (``rho > threshold``) cells in its 3x3x3 neighbourhood; the patched velocity
    is the mass-weighted neighbour mean, clipped to ``max_velocity``. Isolated
    vacuum cells (no valid neighbour) fall back to ``rho = threshold`` keeping
    their momentum.

    vs HARD_FLOOR: a hard floor leaves a sharp ``threshold`` cell next to ~O(1)
    neighbours (a strong artificial gradient that the high-order WENO update then
    has to digest); this replaces it with a smooth, physically-plausible
    neighbour-averaged value, which is markedly gentler at strong shocks. NOTE: it
    is NOT strictly mass-conserving — like ``prot.f`` it copies the neighbour
    average without debiting the donors, so a tiny amount of mass is added at the
    (rare) patched cells; the win is smoothness/stability, not exact conservation.
    Native-only (3x3x3 neighbour stencil; no Pallas sibling).
    """
    ndim = config.dimensionality
    di = registered_variables.density_index
    rho = conserved_state[di]
    mom_idx = _momentum_indices(config, registered_variables)
    mom = [conserved_state[i] for i in mom_idx]

    is_invalid = rho <= threshold
    valid_f = (~is_invalid).astype(rho.dtype)

    axes = tuple(range(ndim))

    def sum_neighbors(arr):
        out = jnp.zeros_like(arr)
        for shift in itertools.product((-1, 0, 1), repeat=ndim):
            out = out + jnp.roll(arr, shift=shift, axis=axes)
        return out

    rho_sum = sum_neighbors(rho * valid_f)
    count_sum = sum_neighbors(valid_f)
    mom_sum = [sum_neighbors(m * valid_f) for m in mom]

    has = count_sum > 0
    count_safe = jnp.where(has, count_sum, 1.0)
    rho_sum_safe = jnp.where(has, rho_sum, 1.0)

    rho_patched = jnp.where(has, rho_sum / count_safe, threshold)
    mom_patched = []
    for m, ms in zip(mom, mom_sum):
        # mass-weighted neighbour mean velocity = sum(mom) / sum(rho);
        # isolated cell -> keep momentum at the floored density (v = mom/threshold)
        v = jnp.where(has, ms / rho_sum_safe, m / threshold)
        v = jnp.clip(v, -max_velocity, max_velocity)
        mom_patched.append(rho_patched * v)

    out = conserved_state.at[di].set(jnp.where(is_invalid, rho_patched, rho))
    for i, m, mp in zip(mom_idx, mom, mom_patched):
        out = out.at[i].set(jnp.where(is_invalid, mp, m))

    if config.equation_of_state == IDEAL_GAS:
        # redistribute total energy the same way, then re-floor pressure
        ei = registered_variables.energy_index
        E = conserved_state[ei]
        E_patched = jnp.where(has, sum_neighbors(E * valid_f) / count_safe, E)
        out = out.at[ei].set(jnp.where(is_invalid, E_patched, E))
        out = _enforce_positivity_native_impl(
            out, config, gamma, threshold, minimum_pressure, registered_variables,
        )

    return out


def _apply_stage_positivity(
    conserved_state: STATE_TYPE,
    mode: int,
    config: SimulationConfig,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    positivity_max_velocity: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Dispatch a positivity mode onto a conserved state (``mode`` is static)."""
    if mode == POSITIVITY_HARD_FLOOR:
        return _enforce_positivity(
            conserved_state, config, gamma,
            minimum_density, minimum_pressure, registered_variables,
        )
    if mode == POSITIVITY_REDISTRIBUTE:
        return _redistribute_positivity(
            conserved_state, minimum_density, positivity_max_velocity, gamma,
            minimum_pressure, config, registered_variables,
        )
    return conserved_state


# Bottom-of-file Pallas import (avoids circular import — see guide §2.4).
from astronomix._fluid_equations._enforce_positivity_pallas import (  # noqa: E402
    _enforce_positivity_pallas,
    _enforce_positivity_pallas_supported,
    _redistribute_positivity_pallas,
    _redistribute_positivity_pallas_supported,
)
