"""
Fourier-based Poisson solver and simple source term handling
of self gravity. To be improved to an energy-conserving scheme.
"""

# general
from functools import partial
import jax.numpy as jnp
import jax

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple, Union

# astronomix data classes
from astronomix._modules._gravity._poisson_solver import (
    _compute_gravitational_potential,
)
from astronomix._modules._gravity._utils import _pad_external_potential
from astronomix._stencil_operations._stencil_operations import _shift, _stencil_add
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
    FOURTH_ORDER_CONSERVATIVE,
    SimulationConfig,
)

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
)

from astronomix.option_classes.simulation_params import SimulationParams

@partial(jax.jit, static_argnames=["grid_spacing", "config", "registered_variables"])
def _compute_total_potential(
    gas_density: FIELD_TYPE,
    grid_spacing: float,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    G: Union[float, Float[Array, ""]] = 1.0,
) -> FIELD_TYPE:
    """
    Compute the total gravitational potential, including contributions from self-gravity and any external potentials.

    Args:
        gas_density: The gas density field (ghost-cell padded, i.e. the
            shape of a single state field).
        grid_spacing: The grid spacing.
        config: The simulation configuration.
        params: The simulation parameters (provides the external potential).
        registered_variables: The registered variables.
        G: The gravitational constant.
    Returns:
        The total gravitational potential, with the same shape as gas_density.
    """
    total_potential = jnp.zeros_like(gas_density)

    # self-gravity contribution from the Poisson solve
    if config.gravity_config.self_gravity:
        total_potential = total_potential + _compute_gravitational_potential(
            gas_density, grid_spacing, config, G
        )

    # external potential contribution
    if config.gravity_config.external_potential:
        # the external potential is provided on the bare grid cells, so it gets
        # ghost cells matching the (here padded) density field, filled per BC
        external_potential = _pad_external_potential(
            params.gravitational_potential, gas_density, config, registered_variables, params
        )
        total_potential = total_potential + external_potential

    return total_potential

def _fd_gravity_source(
    primitive_state: STATE_TYPE,
    density_fluxes,
    drho,
    dt,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):

    S = jnp.zeros_like(primitive_state)

    gravitational_potential = _compute_total_potential(
        primitive_state[registered_variables.density_index],
        config.grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant
    )

    if config.gravity_config.self_gravity_version == SIMPLE_SOURCE:

        for axis in range(1, config.dimensionality + 1):
            rho = primitive_state[registered_variables.density_index]
            v_axis = primitive_state[axis]

            # flux = fluxes[axis - 1]
            # rho_v_avg = 0.5 * (
            #     flux[registered_variables.density_index] + _shift(flux[registered_variables.density_index], 1, axis = axis-1)
            # )

            # # TODO: use higher-order finite difference
            # # a_i = - (phi_{i+1} - phi_{i-1}) / (2 * dx)
            # acceleration = -_stencil_add(
            #     gravitational_potential, indices=(1, -1), factors=(1.0, -1.0), axis=axis - 1
            # ) / (2 * config.grid_spacing)
            # # it is axis - 1 because the axis is 1-indexed as usually the zeroth axis are the different
            # # fields in the state vector not the spatial dimensions, but here we only have the spatial dimensions

            # 6th-order finite difference for gravitational acceleration
            # a_i = - (phi_{i+3} - 9*phi_{i+2} + 45*phi_{i+1} - 45*phi_{i-1} + 9*phi_{i-2} - phi_{i-3}) / (60 * dx)
            acceleration = -_stencil_add(
                gravitational_potential,
                indices=(3, 2, 1, -1, -2, -3),
                factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
                axis=axis - 1
            ) / (60.0 * config.grid_spacing)

            S_axis = jnp.zeros_like(primitive_state)
            S_axis = S_axis.at[axis].set(rho * acceleration)
            S_axis = S_axis.at[registered_variables.pressure_index].set(
                rho * v_axis * acceleration
            )

            S += S_axis * dt
    elif config.gravity_config.self_gravity_version == SECOND_ORDER_CONSERVATIVE:

        for axis in range(1, config.dimensionality + 1):
            rho = primitive_state[registered_variables.density_index]
            phi_cell = gravitational_potential

            # ── Momentum source: 6th-order centered gradient ──────────────────────
            acceleration = -_stencil_add(
                gravitational_potential,
                indices=(3,  2,   1,   -1,  -2,  -3),
                factors=(1., -9., 45., -45., 9., -1.),
                axis=axis - 1
            ) / (60.0 * config.grid_spacing)

            S_axis = jnp.zeros_like(primitive_state)
            S_axis = S_axis.at[axis].set(rho * acceleration)

            # ── Energy source: flux-compatible, no drho term needed ───────────────
            # phi at right face i+1/2  (6th-order symmetric interpolation)
            phi_face = _stencil_add(
                gravitational_potential,
                indices=(-2,  -1,    0,    1,   2,   3),
                factors=( 3., -25., 150., 150., -25., 3.),
                axis=axis - 1
            ) / 256.0

            F_right = density_fluxes[axis - 1]                       # F at i+1/2
            F_left  = _shift(density_fluxes[axis - 1], 1, axis=axis-1)  # F at i-1/2
            phi_face_left = _shift(phi_face, 1, axis=axis - 1)       # phi at i-1/2

            # W_i = -[F_right*(phi_right - phi_i) + F_left*(phi_i - phi_left)] / dx
            # Equivalent to: -div(F*phi) + phi*div(F) = -rho*v*grad(phi)
            energy_source = -(
                F_right * (phi_face      - phi_cell)
            + F_left  * (phi_cell - phi_face_left)
            ) / config.grid_spacing

            S_axis = S_axis.at[registered_variables.energy_index].set(energy_source)

            S += S_axis * dt

        # for axis in range(1, config.dimensionality + 1):

        #     rho = primitive_state[registered_variables.density_index]
        #     v_axis = primitive_state[axis]

        #     # 6th-order interpolation of the 
        #     # potential to the cell faces
        #     phi_face = _stencil_add(
        #         gravitational_potential, 
        #         indices=(-2, -1, 0, 1, 2, 3), 
        #         factors=(3.0, -25.0, 150.0, 150.0, -25.0, 3.0), 
        #         axis=axis - 1
        #     ) / 256.0

        #     # 6th-order finite difference for gravitational acceleration
        #     # a_i = - (phi_{i+3} - 9*phi_{i+2} + 45*phi_{i+1} - 45*phi_{i-1} + 9*phi_{i-2} - phi_{i-3}) / (60 * dx)
        #     acceleration = -_stencil_add(
        #         gravitational_potential, 
        #         indices=(3, 2, 1, -1, -2, -3), 
        #         factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0), 
        #         axis=axis - 1
        #     ) / (60.0 * config.grid_spacing)

        #     S_axis = jnp.zeros_like(primitive_state)

        #     # momentum source
        #     S_axis = S_axis.at[axis].set(rho * acceleration)

        #     # energy source
        #     S_axis = S_axis.at[registered_variables.pressure_index].set(
        #         -1.0 / config.grid_spacing * (density_fluxes[axis - 1] * phi_face - _shift(density_fluxes[axis - 1] * phi_face, 1, axis=axis - 1))
        #     )

        #     S += S_axis * dt

        # S = S.at[registered_variables.energy_index].add(
        #     -drho * gravitational_potential
        # )
    elif config.gravity_config.self_gravity_version == FOURTH_ORDER_CONSERVATIVE:
        for axis in range(1, config.dimensionality + 1):
            ax = axis - 1

            rho = primitive_state[registered_variables.density_index]
            v_axis = primitive_state[axis]
            dx = config.grid_spacing

            # 6th-order potential at faces (already have this)
            phi_face = _stencil_add(
                gravitational_potential,
                indices=(-2, -1, 0, 1, 2, 3),
                factors=(3.0, -25.0, 150.0, 150.0, -25.0, 3.0),
                axis=ax
            ) / 256.0

            # 6th-order gravitational acceleration at centers (already have this)
            acceleration = -_stencil_add(
                gravitational_potential,
                indices=(3, 2, 1, -1, -2, -3),
                factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
                axis=ax
            ) / (60.0 * dx)

            S_axis = jnp.zeros_like(primitive_state)
            S_axis = S_axis.at[axis].set(rho * acceleration)

            # --- corrected product form for energy ---

            f = rho * v_axis                      # (rho v) at cell centers
            dPhi = -acceleration                   # Phi' at cell centers (6th order, free)

            d2Phi = (                              # Phi'' at cell centers (2nd order, sufficient)
                _shift(gravitational_potential, -1, axis=ax)
                - 2.0 * gravitational_potential
                + _shift(gravitational_potential, 1, axis=ax)
            ) / dx**2

            df = (                                 # f' at cell centers (2nd order, sufficient)
                _shift(f, -1, axis=ax) - _shift(f, 1, axis=ax)
            ) / (2.0 * dx)

            # correction at cell centers
            corr_cc = d2Phi * f + 2.0 * dPhi * df

            # average to faces
            corr_face = 0.5 * (corr_cc + _shift(corr_cc, -1, axis=ax))

            # corrected product flux
            q_hat = density_fluxes[axis - 1] * phi_face - (dx**2 / 24.0) * corr_face

            # energy source: -div(q_hat)
            S_energy = -1.0 / dx * (q_hat - _shift(q_hat, 1, axis=ax))

            S_axis = S_axis.at[registered_variables.pressure_index].set(S_energy)
            S += S_axis * dt

        # Phi * drho term (unchanged)
        S = S.at[registered_variables.energy_index].add(
            -drho * gravitational_potential
        )
    else:
        raise NotImplementedError(
            "This scheme is not implemented."
        )
    
    return S

# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["axis", "grid_spacing", "registered_variables", "config"]
)
def _gravitational_source_term_along_axis(
    gravitational_potential: FIELD_TYPE,
    primitive_state: STATE_TYPE,
    grid_spacing: float,
    registered_variables: RegisteredVariables,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    axis: int,
) -> STATE_TYPE:
    """
    Compute the source term for the self-gravity solver along a single axis.
    Currently, simply density * gravitational_acceleration for the momentum
    and density * velocity * gravitational_acceleration for the energy.

    Args:
        gravitational_potential: The gravitational potential.
        primitive_state: The primitive state.
        grid_spacing: The grid spacing.
        registered_variables: The registered variables.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.
        axis: The axis along which to compute the source term.

    Returns:
        The source term.

    """

    rho = primitive_state[registered_variables.density_index]
    v_axis = primitive_state[axis]

    # a_i = - (phi_{i+1} - phi_{i-1}) / (2 * dx)
    acceleration = -_stencil_add(
        gravitational_potential, indices=(1, -1), factors=(1.0, -1.0), axis=axis - 1
    ) / (2 * grid_spacing)
    # it is axis - 1 because the axis is 1-indexed as usually the zeroth axis are the different
    # fields in the state vector not the spatial dimensions, but here we only have the spatial dimensions

    source_term = jnp.zeros_like(primitive_state)

    # set momentum source
    source_term = source_term.at[axis].set(rho * acceleration)

    # finite-volume self-gravity supports only the SIMPLE_SOURCE coupling
    # (the FD-only conservative flux schemes live in _fd_gravity_source).
    source_term = source_term.at[registered_variables.pressure_index].set(
        rho * v_axis * acceleration
    )

    return source_term