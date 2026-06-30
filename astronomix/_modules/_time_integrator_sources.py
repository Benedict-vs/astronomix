# TODO: streamline FD and FV codes

from astronomix._fluid_equations._equations_mhd import conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
from astronomix._fluid_equations._equations import conserved_state_from_primitive, primitive_state_from_conserved
from astronomix._modules._cooling._cooling import update_pressure_by_cooling
from astronomix._modules._cooling._simple_mixing_cooling import update_pressure_by_cooling_mixing
from astronomix._modules._cooling.cooling_options import SIMPLE_MIXING_LAYER_COOLING
from astronomix._modules._gravity._gravity import _compute_total_potential, _fd_gravity_source, _gravitational_source_term_along_axis
from astronomix._modules._stellar_wind.stellar_wind import _wind_ei3D_source
from astronomix._modules._cgols_wind._cgols_wind import _cgols_wind_source
from astronomix._modules._viscosity._viscosity import fd_viscosity_source
from astronomix._modules._conduction._conduction import fd_conduction_source
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE, FINITE_VOLUME, STATE_TYPE, SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from typing import Union


from functools import partial


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _time_integrator_sources(
    conserved_state: STATE_TYPE,
    density_fluxes,
    drho,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]] = 0.0,
) -> STATE_TYPE:
    """
    Compute the physics source terms for the given **conserved** state.

    Args:
        conserved_state: The conserved state array.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.
        registered_variables: The registered variables.
        current_time: The current simulation time (start-of-step), used by
            modules with time-dependent sources (CGOLS wind schedule).
    Returns:
        The physics source terms for the conserved state.
    """

    S = jnp.zeros_like(conserved_state)

    # The finite-difference source terms below operate on the primitive
    # state. The finite-volume path only enters the gravity branch, which
    # derives its own (gas-mode) primitives, so the conversion here is
    # FD-only. This also avoids applying the MHD conversion to the
    # B-stripped gas substate the FV scheme passes in.
    if config.solver_mode == FINITE_DIFFERENCE:
        if config.mhd:
            primitive_state = primitive_state_from_conserved_mhd(
                conserved_state,
                params.minimum_density,
                params.minimum_pressure,
                gamma,
                config,
                registered_variables
            )
        else:
            primitive_state = primitive_state_from_conserved(
                conserved_state, gamma, config, registered_variables
            )

    if config.wind_config.stellar_wind and config.solver_mode == FINITE_DIFFERENCE:
        S += (
            _wind_ei3D_source(
                params.wind_params,
                conserved_state,
                dt,
                config,
                helper_data,
                config.wind_config.num_injection_cells,
                registered_variables,
            )
        )

    if config.cgols_wind_config.cgols_wind and config.solver_mode == FINITE_DIFFERENCE:
        S += _cgols_wind_source(
            params.cgols_wind_params,
            conserved_state,
            dt,
            current_time,
            config,
            helper_data,
            registered_variables,
        )

    if config.cooling_config.cooling and config.solver_mode == FINITE_DIFFERENCE:
        if not config.cooling_config.cooling_curve_config.cooling_curve_type == SIMPLE_MIXING_LAYER_COOLING:
            primitive_state = update_pressure_by_cooling(
                primitive_state,
                registered_variables,
                config.cooling_config,
                params,
                dt,
            )
        else:
            primitive_state = update_pressure_by_cooling_mixing(
                primitive_state,
                registered_variables,
                config.cooling_config,
                params,
                dt,
            )

        if config.mhd:
            final_conserved_state = conserved_state_from_primitive_mhd(
                primitive_state, gamma, registered_variables
            )
        else:
            final_conserved_state = conserved_state_from_primitive(
                primitive_state, gamma, config, registered_variables
            )
        S += (final_conserved_state - conserved_state)

    # simplest self-gravity
    # TODO: maybe only one Poisson solve per RK step?
    if config.gravity_config.gravity and config.solver_mode == FINITE_DIFFERENCE:
        S += _fd_gravity_source(
            primitive_state,
            density_fluxes,
            drho,
            dt,
            config,
            params,
            registered_variables,
        )

    # finite-volume self-gravity: the source is built from the (pre-hydro)
    # state passed in and added operator-split to the post-hydro state by the
    # caller. Matches the former _apply_self_gravity scheme, which likewise
    # evaluated the source on the gas substate with the hydro conversion.
    if config.gravity_config.gravity and config.solver_mode == FINITE_VOLUME:
        fv_primitive_state = primitive_state_from_conserved(
            conserved_state, gamma, config, registered_variables
        )

        gravitational_potential = _compute_total_potential(
            fv_primitive_state[registered_variables.density_index],
            config.grid_spacing,
            config,
            params,
            registered_variables,
            params.gravitational_constant,
        )

        source_term = jnp.zeros_like(fv_primitive_state)
        for axis in range(1, config.dimensionality + 1):
            source_term = source_term + _gravitational_source_term_along_axis(
                gravitational_potential,
                fv_primitive_state,
                config.grid_spacing,
                registered_variables,
                dt,
                gamma,
                config,
                params,
                helper_data,
                axis,
            )

        S += dt * source_term

    if config.diffusion and config.solver_mode == FINITE_DIFFERENCE:
        S += fd_viscosity_source(primitive_state, params, config, registered_variables) * dt

    # thermal conduction: kappa * laplacian(T) added to the energy equation
    if config.thermal_conduction and config.solver_mode == FINITE_DIFFERENCE:
        S += fd_conduction_source(primitive_state, params, config, registered_variables) * dt

    return S