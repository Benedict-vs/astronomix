# general imports
import jax.numpy as jnp
import jax
from functools import partial


# type checking imports
from jaxtyping import Array, Float
from beartype import beartype as typechecker
from typing import Union

# general astronomix imports
from astronomix._fluid_equations._equations_mhd import conserved_state_from_primitive_isothermal, conserved_state_from_primitive_mhd, primitive_state_from_conserved_isothermal, primitive_state_from_conserved_mhd
from astronomix._finite_difference._magnetic_update._constrained_transport import update_cell_center_fields
from astronomix._finite_difference._time_integrators._ssprk import (
    _lsrk4_hydro,
    _lsrk4_with_ct,
    _ssprk4_hydro,
    _ssprk4_with_ct,
)
from astronomix._fluid_equations._equations import conserved_state_from_primitive, primitive_state_from_conserved
from astronomix._geometry.boundaries import _boundary_handler
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    GHOST_CELLS,
    IDEAL_GAS,
    ISOTHERMAL,
    RK4_LSRK,
    STATE_TYPE,
    SimulationConfig,
)

from astronomix.option_classes.simulation_params import SimulationParams

@partial(jax.jit, static_argnames=["config", "registered_variables"], donate_argnames=["primitive_state"])
def _evolve_state_fd(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]] = 0.0,
) -> STATE_TYPE:

    if config.mhd:
        # NOTE: here we assume the magnetic field at interfaces
        # is stored in the last three indices of the state array
        if config.equation_of_state == IDEAL_GAS:
            conserved_state = conserved_state_from_primitive_mhd(
                primitive_state[:-3], gamma, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            conserved_state = conserved_state_from_primitive_isothermal(
                primitive_state[:-3], config, registered_variables
            )

        # extract interface magnetic fields
        bxb = primitive_state[registered_variables.interface_magnetic_field_index.x]
        byb = primitive_state[registered_variables.interface_magnetic_field_index.y]
        bzb = primitive_state[registered_variables.interface_magnetic_field_index.z]

        # update conserved state and interface magnetic fields — RK4_LSRK
        # selects the 2N-storage Carpenter-Kennedy LSRK4 variant (saves one
        # conserved + three interface-B carry registers vs SSPRK4).
        if config.time_integrator == RK4_LSRK:
            mhd_integrator = _lsrk4_with_ct
        else:
            mhd_integrator = _ssprk4_with_ct

        conserved_state, bxb, byb, bzb = mhd_integrator(
            conserved_state,
            bxb,
            byb,
            bzb,
            gamma,
            config.grid_spacing,
            dt,
            params,
            helper_data,
            config,
            registered_variables,
            current_time,
        )

        # back to primitive state
        if config.equation_of_state == IDEAL_GAS:
            primitive_state = primitive_state_from_conserved_mhd(
                conserved_state, params.minimum_density, params.minimum_pressure, gamma, config, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            primitive_state = primitive_state_from_conserved_isothermal(
                conserved_state, params.minimum_density, config, registered_variables
            )

        # append updated interface magnetic fields
        # NOTE: same assumption as above
        primitive_state = jnp.concatenate(
            [primitive_state, bxb[None, :], byb[None, :], bzb[None, :]], axis=0
        )
    else:
        conserved_state = conserved_state_from_primitive(
            primitive_state, gamma, config, registered_variables
        )

        # Dispatch to the requested time integrator.  RK4_LSRK is the
        # Carpenter-Kennedy 2N-storage low-storage RK4 (one fewer full-state
        # register than SSPRK4, at the cost of a smaller stability CFL).
        if int(config.time_integrator) == RK4_LSRK:
            integrator = _lsrk4_hydro
        else:
            integrator = _ssprk4_hydro

        conserved_state = integrator(
            conserved_state,
            gamma,
            config.grid_spacing,
            dt,
            params,
            helper_data,
            config,
            registered_variables,
            current_time,
        )

        primitive_state = primitive_state_from_conserved(
            conserved_state,
            gamma,
            config,
            registered_variables,
            params.minimum_density,
            params.minimum_pressure,
        )
    
    # handle the boundary conditions
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(
            primitive_state, config, registered_variables, params
        )

    return primitive_state