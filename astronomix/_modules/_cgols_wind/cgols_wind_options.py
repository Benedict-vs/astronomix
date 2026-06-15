from typing import NamedTuple

import jax.numpy as jnp


class CGOLSWindConfig(NamedTuple):
    """Static configuration of the CGOLS-style central starburst wind.

    Mass and thermal energy are injected at (time-dependent) constant
    volumetric rates into a sphere at the box center, following the
    Chevalier & Clegg (1985) model as implemented in the CGOLS suite
    (Schneider & Robertson 2018, arXiv:1803.01008). Finite-difference
    solver mode only.
    """

    cgols_wind: bool = False


class CGOLSWindParams(NamedTuple):
    """Dynamic parameters of the CGOLS wind, all in code units.

    The injection rates follow a piecewise-linear schedule: at simulation
    time t the rates are ``jnp.interp(t, schedule_times, schedule_*_rates)``
    (clamped to the first/last value outside the knots). The CGOLS
    low -> ramp -> high -> ramp -> low history is expressed by choosing the
    knots accordingly; duplicate-adjacent knots give step changes.
    """

    #: Radius of the spherical injection ("gain") region, code length.
    #: CGOLS uses 300 pc.
    injection_radius: float = 0.3 # since code units in cgols.py is 1 * u.kiloparsec

    #: Schedule knots, code time, must be non-decreasing.
    schedule_times: jnp.ndarray = jnp.array([0.0])

    #: Mass injection rate at each knot, code mass / code time.
    schedule_mass_rates: jnp.ndarray = jnp.array([0.0])

    #: Thermal energy injection rate at each knot, code energy / code time.
    schedule_energy_rates: jnp.ndarray = jnp.array([0.0])
