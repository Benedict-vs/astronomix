"""
Configuration and parameter containers for radiative cooling.

Defines the integer tags that select a cooling-curve type and a cooling method
(explicit / implicit), together with the NamedTuples that carry the parameters
of each cooling curve and the overall cooling configuration.
"""

# typing
from typing import NamedTuple, Union
from types import NoneType
from jaxtyping import PyTree

# jax
import jax.numpy as jnp

# Cooling-curve type tags (select which Lambda(T) model is used).
SIMPLE_POWER_LAW = 1
PIECEWISE_POWER_LAW = 2
NEURAL_NET_COOLING = 3
NEURAL_NET_COOLING_WITH_DENSITY = 4
SIMPLE_MIXING_LAYER_COOLING = 5
CIE_PARABOLIC = 6

# Cooling-method tags (how the temperature update is integrated in time).
EXPLICIT_COOLING = 1
IMPLICIT_COOLING = 2
SUBCYCLED_EXPLICIT_COOLING = 3

# Volumetric-density convention for the cooling rate n_a * n_b * Lambda(T).
ELECTRON_HYDROGEN_DENSITY = 0  # n_e * n_H * Lambda, i.e. a mu / (mu_e mu_H) prefactor
TOTAL_NUMBER_DENSITY = 1  # n^2 * Lambda with n = rho / (mu m_p), i.e. a 1 / mu prefactor

# Where in the time step the cooling operator is applied.
COOLING_IN_RK_STAGES = 0  # inside every RK stage of the hydro integrator (FD)
COOLING_OPERATOR_SPLIT = 1  # once per step, after the hydro update (Lie splitting)

# What to do when a cooling step would undershoot the temperature floor.
COOLING_FLOOR_REVERT = 0  # keep the old temperature in those cells
COOLING_FLOOR_CLIP = 1  # clamp the new temperature to the floor


class SimplePowerLawParams(NamedTuple):
    """Parameters of a single power-law cooling curve Lambda(T)."""

    factor: float = 1.0
    exponent: float = 1.0
    reference_temperature: float = 1e8


class PiecewisePowerLawParams(NamedTuple):
    """Tabulated parameters of a piecewise power-law cooling curve.

    The tables hold, per temperature bin, the curve value and slope plus the
    Townsend temporal-evolution coefficients (``Y_table``).
    """

    log10_T_table: jnp.ndarray = jnp.array([])
    log10_Lambda_table: jnp.ndarray = jnp.array([])
    alpha_table: jnp.ndarray = jnp.array([])
    Y_table: jnp.ndarray = jnp.array([])
    reference_temperature: float = 1e8


class CoolingNetConfig(NamedTuple):
    """Static configuration of a neural-network cooling curve."""

    network_static: Union[PyTree, NoneType] = None


class CoolingNetParams(NamedTuple):
    """Trainable parameters of a neural-network cooling curve."""

    network_params: Union[PyTree, NoneType] = None


class CIEParabolicParams(NamedTuple):
    """Piecewise-parabolic fit to a solar-metallicity CIE cooling curve.

    This is the fit used by the CGOLS simulations (Schneider & Robertson 2018,
    Eq. A4 of arXiv:1803.01008, identical to Cholla's ``CIE_cool``):

    ::

        log10 T < 4.0          Lambda = 0
        4.0 <= log10 T < 5.9   Lambda = 10^(-1.3 (log10 T - 5.25)^2 - 21.25)
        5.9 <= log10 T < 7.4   Lambda = 10^( 0.7 (log10 T - 7.10)^2 - 22.8 )
        log10 T >= 7.4         Lambda = 10^( 0.45 log10 T - 26.065)

    with ``T`` in Kelvin and ``Lambda`` in erg s^-1 cm^3. The defaults below are
    literally those published coefficients; the conversion to the rescaled code
    units the cooling module works in enters as the two *additive log10 shifts*
    at the top, so the fit itself stays readable and directly testable against
    the paper. Build the shifted variant with
    :func:`astronomix._modules._cooling._cooling_tables.cie_parabolic_cooling`.
    """

    #: log10 T[K] = log10 T_code + this shift.
    log10_temperature_to_kelvin: float = 0.0
    #: log10 Lambda_code = log10 Lambda_cgs + this shift.
    log10_lambda_cgs_to_code: float = 0.0

    #: Temperature floor and the two breakpoints of the fit, in log10 T[K].
    log10_T_floor: float = 4.0
    log10_T_break_1: float = 5.9
    log10_T_break_2: float = 7.4

    #: Low branch: a_low (log10 T - t_low)^2 + c_low.
    a_low: float = -1.3
    t_low: float = 5.25
    c_low: float = -21.25

    #: Middle branch: a_mid (log10 T - t_mid)^2 + c_mid.
    a_mid: float = 0.7
    t_mid: float = 7.10
    c_mid: float = -22.8

    #: High branch: s_high log10 T + c_high.
    s_high: float = 0.45
    c_high: float = -26.065


class MixingCoolingParams(NamedTuple):
    """Parameters of the simple mixing-layer cooling model (Lancaster 2026)."""

    xi: float = 0.5  # xi = t_sh / t_coolmin
    mach_number: float = 0.5
    density_contrast: float = 10.0


# Union of every cooling-curve parameter container; the active variant is
# selected by the cooling-curve type tag in CoolingCurveConfig.
COOLING_CURVE_TYPE = Union[
    SimplePowerLawParams,
    PiecewisePowerLawParams,
    CoolingNetParams,
    MixingCoolingParams,
    CIEParabolicParams,
]


class CoolingCurveConfig(NamedTuple):
    """Static configuration selecting the cooling-curve model."""

    cooling_curve_type: int = SIMPLE_POWER_LAW

    #: In case of neural the cooling the network architecture
    cooling_net_config: CoolingNetConfig = CoolingNetConfig()

    #: Which pair of number densities multiplies Lambda(T). The default
    #: ELECTRON_HYDROGEN_DENSITY is the historical n_e n_H convention;
    #: TOTAL_NUMBER_DENSITY is the n^2 convention used by Cholla / CGOLS. The
    #: two differ by mu_e mu_H / mu^2 (~4.30 for X = 0.76, Z = 0.02), so this
    #: is not a cosmetic choice.
    density_convention: int = ELECTRON_HYDROGEN_DENSITY


class CoolingConfig(NamedTuple):
    """Top-level cooling configuration (activation, method and curve)."""

    cooling: bool = False
    cooling_method: int = IMPLICIT_COOLING
    cooling_curve_config: CoolingCurveConfig = CoolingCurveConfig()

    #: Where the cooling operator is applied: inside the RK stages of the hydro
    #: integrator (the historical behaviour) or once per step after the hydro
    #: update (true Lie splitting, what CGOLS / Cholla do).
    cooling_placement: int = COOLING_IN_RK_STAGES

    #: Whether an undershooting cooling step reverts or clips to the floor.
    floor_mode: int = COOLING_FLOOR_REVERT

    #: Fixed sub-cycle trip count of SUBCYCLED_EXPLICIT_COOLING. The loop is a
    #: static-bound fori_loop (so it stays reverse-mode differentiable), hence
    #: this many full array passes are always executed.
    max_subcycles: int = 16

    #: Largest fractional temperature change allowed per sub-cycle (Cholla's 1%
    #: rule).
    max_fractional_temperature_change: float = 0.01

    #: Whether the hydro timestep is additionally limited by the cooling time
    #: (see ``CoolingParams.max_thermal_energy_fraction``).
    cooling_timestep_limit: bool = False


class CoolingParams(NamedTuple):
    """Runtime cooling parameters (composition, temperature floor, curve)."""

    # NOTE: CURRENTLY ONLY POWER LAW COOLING
    hydrogen_mass_fraction: float = 0.76
    metal_mass_fraction: float = 0.02

    #: Temperature floor, in the module's **rescaled** temperature
    #: ``T~ = T[K] * k_B / m_p`` (code_energy / code_mass) — NOT in Kelvin,
    #: despite the Kelvin-looking default. In cgols code units (1 kpc,
    #: 1e6 Msun, 100 km/s) a 10^4 K floor is T~ ~ 8.3e-3, so passing a literal
    #: ``1e4`` there would put the floor at ~1.2e10 K and silently disable
    #: cooling everywhere. Convert with
    #: ``(T_K * u.K * c.k_B / c.m_p).to(code_energy / code_mass).value``.
    floor_temperature: float = 1e4

    #: If > 0, overrides the mean molecular weight derived from the hydrogen and
    #: metal mass fractions. CGOLS uses mu = 0.6 throughout, whereas
    #: X = 0.76, Z = 0.02 give mu = 0.590.
    mean_molecular_weight: float = 0.0

    #: Largest fraction of the thermal energy any cell may lose in one hydro
    #: step when ``CoolingConfig.cooling_timestep_limit`` is on.
    max_thermal_energy_fraction: float = 0.1

    cooling_curve_params: COOLING_CURVE_TYPE = SimplePowerLawParams()
