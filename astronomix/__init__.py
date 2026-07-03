# data structures
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams

# constants
from astronomix.option_classes.simulation_config import (
    FORWARDS,
    BACKWARDS,
    MINMOD,
    OSHER,
    HLL,
    HLLC,
    HLLC_LM,
    OPEN_BOUNDARY,
    OPEN_BOUNDARY_DIODE,
    REFLECTIVE_BOUNDARY,
    PERIODIC_BOUNDARY,
    CARTESIAN,
    CYLINDRICAL,
    SPHERICAL,
    ON_DEVICE,
    TO_DISK,
)

# initialization functions
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)

# module-setup
from astronomix._modules._stellar_wind.stellar_wind_options import WindParams

# run
from astronomix.time_stepping.time_integration import time_integration

# setup helpers (disk-checkpoint restart). The restart path depends on Orbax,
# which is optional: if it is missing or incompatible with the installed JAX,
# keep the base package importable (runs that don't use disk checkpointing,
# e.g. host-offload snapshots, must not be blocked) and defer the failure to
# the point of use with a clear message.
try:
    from astronomix.setup_helpers import restart_from_latest_checkpoint
except (ImportError, AttributeError) as _orbax_exc:
    # ImportError: orbax-checkpoint not installed. AttributeError: installed
    # orbax references a JAX symbol the current JAX no longer exposes (version
    # skew, e.g. jax.experimental.layout.DeviceLocalLayout on JAX 0.10+).
    _orbax_import_error = _orbax_exc

    def restart_from_latest_checkpoint(*_args, **_kwargs):
        raise ImportError(
            "restart_from_latest_checkpoint requires a working Orbax install "
            "(orbax-checkpoint, compatible with the installed JAX). Original "
            f"import error: {_orbax_import_error}"
        )

# units
from astronomix.units import CodeUnits
