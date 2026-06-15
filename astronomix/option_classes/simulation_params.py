from typing import NamedTuple
import jax.numpy as jnp

from astronomix._modules._cnn_mhd_corrector._cnn_mhd_corrector_options import CNNMHDconfig
from astronomix._modules._cooling.cooling_options import CoolingParams
from astronomix._modules._cosmic_rays.cosmic_ray_options import CosmicRayParams
from astronomix._modules._neural_net_force._neural_net_force_options import NeuralNetForceParams
from astronomix._modules._stellar_wind.stellar_wind_options import WindParams
from astronomix._modules._cgols_wind.cgols_wind_options import CGOLSWindParams
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingParams

class FixedBoundaryState1D(NamedTuple):
    #: Left state of shape (num_variables,).
    left_state: jnp.ndarray = jnp.array([])
    #: Right state of shape (num_variables,).
    right_state: jnp.ndarray = jnp.array([])

class FixedBoundaryState(NamedTuple):
    x: FixedBoundaryState1D = FixedBoundaryState1D()
    y: FixedBoundaryState1D = FixedBoundaryState1D()
    z: FixedBoundaryState1D = FixedBoundaryState1D()

class SimulationParams(NamedTuple):
    """
    Different from the simulation configuration, the simulation parameters
    do not require recompilation when changed. The simulation can be 
    differentiated with respect to them.
    """

    #: The Courant-Friedrichs-Lewy number, a factor
    #: in the time step calculation.
    C_cfl: float = 0.4

    #: Gravitational constant.
    gravitational_constant: float = 1.0

    #: External, static gravitational potential evaluated at the cell
    #: centers of the grid (without ghost cells). Only used when
    #: config.external_potential is True; it is added on top of the
    #: self-gravity potential in _compute_total_potential and padded to the
    #: ghost-cell-extended shape of a state field internally.
    gravitational_potential: jnp.array = jnp.array([])

    #: Dynamic or kinematic viscosity depending 
    #: on the viscosity_type in SimulationConfig.
    viscosity: float = 0.0

    #: The isothermal sound speed used when
    #: config.equation_of_state is ISOTHERMAL.
    #: NOTE: CURRENTLY ONLY IMPLEMENTED FOR 
    #: FINITE DIFFERENCE MODE.
    isothermal_sound_speed: float = 1.0

    #: The adiabatic index of the gas.
    gamma: float = 5/3

    #: Minimum allowed density.
    #: NOTE: CURRENTLY ONLY USED IN 
    #: FINITE DIFFERENCE MODE IF
    #: config.enforce_positivity IS TRUE.
    minimum_density: float = 1e-14

    #: Minimum allowed pressure.
    #: NOTE: CURRENTLY ONLY USED IN 
    #: FINITE DIFFERENCE MODE IF
    #: config.enforce_positivity IS TRUE.
    minimum_pressure: float = 1e-14

    #: The maximum time step.
    dt_max: float = jnp.inf

    #: The final time of the simulation.
    t_end: float = 0.2

    #: Snapshot timepoints
    snapshot_timepoints: jnp.array = jnp.array([0.0])

    #: The fixed boundary state if the boundary type
    #: of the specific boundary is set to FIXED_BOUNDARY.
    fixed_boundary_state: FixedBoundaryState = FixedBoundaryState()

    # parameters of physics modules

    #: The parameters of the turbulent forcing module.
    turbulent_forcing_params: TurbulentForcingParams = TurbulentForcingParams()

    #: The parameters of the stellar wind module.
    wind_params: WindParams = WindParams()

    #: The parameters of the CGOLS-style central starburst wind module.
    cgols_wind_params: CGOLSWindParams = CGOLSWindParams()

    #: Cosmic ray parameters
    cosmic_ray_params: CosmicRayParams = CosmicRayParams()

    #: The parameters of the cooling module.
    cooling_params: CoolingParams = CoolingParams()

    #: The parameters of the neural network force module.
    neural_net_force_params: NeuralNetForceParams = NeuralNetForceParams()

    #: The parameters of the CNN MHD corrector module.
    cnn_mhd_corrector_params: CNNMHDconfig = CNNMHDconfig()