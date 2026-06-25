
from astronomix._modules._cnn_mhd_corrector._cnn_mhd_corrector import _cnn_mhd_corrector
from astronomix._modules._cooling._cooling import update_pressure_by_cooling
from astronomix._modules._cosmic_rays.cr_injection import inject_crs_at_strongest_shock
from astronomix._modules._frame_tracking._frame_tracking import _frame_tracking
from astronomix._modules._neural_net_force._neural_net_force import _neural_net_force
from astronomix._modules._stellar_wind.stellar_wind import _wind_injection
from astronomix._modules._turbulent_forcing._turbulent_forcing import _apply_forcing, _apply_ou_forcing, _vacuum_protection
from astronomix._modules._viscosity._viscosity import fv_viscosity_update
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE, FINITE_VOLUME, IDEAL_GAS, POSITIVITY_HARD_FLOOR, POSITIVITY_REDISTRIBUTE, STATE_TYPE, SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.shock_finder.shock_finder import shock_criteria
from astronomix.variable_registry.registered_variables import RegisteredVariables


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from typing import Union

from functools import partial


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _iteration_level_updates(
    primitive_state: STATE_TYPE,
    key,
    forcing,
    dt: Float[Array, ""],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
) -> STATE_TYPE:
    """
    Here go updates that are applied before each hydro-iteration.
    The counterpart of this are the _time_integrator_sources, which
    are handled as RHS source terms in the hydro-integrator.

    Args:
        primitive_state: The primitive state array.
        key: The PRNG key.
        forcing: The persistent Ornstein-Uhlenbeck forcing field (or ``None``
            when OU forcing is inactive). Carried across steps in ``LoopState``.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.
        registered_variables: The registered variables.
        current_time: The current simulation time.
    Returns:
        ``(key, forcing, primitive_state)`` with the iteration-level updates
        applied.
    """

    # stellar wind
    # in the finite difference case handled as a source term in the hydro-integrator
    if config.wind_config.stellar_wind and config.solver_mode == FINITE_VOLUME:
        primitive_state = _wind_injection(
            primitive_state, dt, config, params, helper_data, registered_variables
        )
    
    if config.cosmic_ray_config.diffusive_shock_acceleration:
        shock_crit = shock_criteria(
            primitive_state, config, registered_variables, helper_data
        )

        # injecting cosmic rays only after a certain amount of time
        # is an ad-hoc fix to problems that come about when a shock
        # has not yet properly formed
        primitive_state = jax.lax.cond(
            jnp.logical_and(
                current_time
                >= params.cosmic_ray_params.diffusive_shock_acceleration_start_time,
                jnp.any(shock_crit),
            ),
            lambda primitive_state: inject_crs_at_strongest_shock(
                primitive_state,
                params.gamma,
                helper_data,
                params.cosmic_ray_params,
                config,
                registered_variables,
                dt,
            ),
            lambda primitive_state: primitive_state,
            primitive_state,
        )

    # cooling
    # in the finite difference case handled as a source term in the hydro-integrator
    if config.cooling_config.cooling and config.solver_mode == FINITE_VOLUME:
        primitive_state = update_pressure_by_cooling(
            primitive_state,
            registered_variables,
            config.cooling_config,
            params,
            dt,
        )

    if config.neural_net_force_config.neural_net_force:
        primitive_state = _neural_net_force(
            primitive_state,
            config,
            registered_variables,
            params,
            helper_data,
            dt,
            current_time,
        )

    if config.cnn_mhd_corrector_config.cnn_mhd_corrector:
        primitive_state = _cnn_mhd_corrector(
            primitive_state, config, registered_variables, params, dt
        )

    # viscosity
    # in the finite difference case handled as a source term in the hydro-integrator
    if config.diffusion and config.solver_mode == FINITE_VOLUME:
        primitive_state = fv_viscosity_update(primitive_state, params, config, registered_variables, dt)

    # turbulence forcing. The PRNG key and (for OU forcing) the persistent
    # forcing field are threaded through the loop in ``LoopState`` (see
    # astronomix/time_stepping/time_integration.py).
    if config.turbulent_forcing_config.turbulent_forcing:
        if config.turbulent_forcing_config.ou_forcing:
            # OU forcing carries a persistent solenoidal field ``forcing``;
            # bundle it with the key into the (key, field) state the OU update
            # expects and unpack the advanced state back out.
            (key, forcing), primitive_state = _apply_ou_forcing(
                (key, forcing),
                primitive_state,
                dt,
                params.turbulent_forcing_params,
                config,
                registered_variables,
            )
        else:
            key, primitive_state = _apply_forcing(
                key,
                primitive_state,
                dt,
                params.turbulent_forcing_params,
                config,
                registered_variables,
            )

    # PRELIMINARY
    # Frame tracking, currently very specialized
    # CURRENTLY ONLY 3D
    if config.frame_tracking:
        primitive_state = _frame_tracking(
            primitive_state,
            config,
            params,
            registered_variables,
            helper_data,
        )

    # per-step positivity (primitive state). HARD_FLOOR clamps density (and
    # pressure for ideal gas); REDISTRIBUTE applies the conservative `prot`
    # neighbour redistribution, but is skipped when turbulent forcing already
    # runs `prot` each step (vacuum_protection) to avoid a redundant pass.
    if config.positivity_per_step_mode == POSITIVITY_HARD_FLOOR:
        primitive_state = primitive_state.at[registered_variables.density_index].set(
            jnp.maximum(
                primitive_state[registered_variables.density_index], params.minimum_density
            )
        )
        if config.equation_of_state == IDEAL_GAS:
            primitive_state = primitive_state.at[registered_variables.pressure_index].set(
                jnp.maximum(
                    primitive_state[registered_variables.pressure_index], params.minimum_pressure
                )
            )
    elif config.positivity_per_step_mode == POSITIVITY_REDISTRIBUTE:
        _forcing_runs_prot = (
            config.turbulent_forcing_config.turbulent_forcing
            and config.turbulent_forcing_config.vacuum_protection
        )
        if not _forcing_runs_prot:
            primitive_state = _vacuum_protection(
                primitive_state,
                params.minimum_density,
                params.positivity_max_velocity,
                config,
                registered_variables,
            )

    return key, forcing, primitive_state