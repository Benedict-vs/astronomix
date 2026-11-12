"""
Configuration and parameter containers for radiative cooling.

Defines the integer tags that select a cooling-curve type and a cooling method
(explicit / implicit), together with the NamedTuples that carry the parameters
of each cooling curve and the overall cooling configuration.
"""

# typing
from typing import (
    NamedTuple,
    Union,
)
from types import NoneType
from jaxtyping import PyTree

# jax
import jax.numpy as jnp

# astronomix containers
from astronomix._modules._cooling._cooling_tabulated import TabulatedCoolingParams

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
#: Exact integration of cooling plus heating on precomputed tables
#: (``_cooling_tabulated.py``): fixed cost per cell, no sub-cycles. Needs
#: ``CoolingParams.tabulated_cooling`` (``tabulated_cooling_from_curve``).
TABULATED_EXACT_COOLING = 4

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

    #: Value of Lambda at the reference temperature.
    factor: float = 1.0

    #: Power-law exponent of Lambda(T) = factor * (T / T_ref)^exponent.
    exponent: float = 1.0

    #: Reference temperature T_ref (rescaled units).
    reference_temperature: float = 1e8


class PiecewisePowerLawParams(NamedTuple):
    """
    Tabulated parameters of a piecewise power-law cooling curve.

    The tables hold, per temperature bin, the curve value and slope plus the
    Townsend temporal-evolution coefficients (``Y_table``).
    """

    #: log10 of the (rescaled) bin-edge temperatures T_k, ascending.
    log10_T_table: jnp.ndarray = jnp.array([])

    #: log10 of the (rescaled) cooling rate Lambda_k at the bin edges.
    log10_Lambda_table: jnp.ndarray = jnp.array([])

    #: Power-law slope alpha_k of each bin,
    #: Lambda(T) = Lambda_k * (T / T_k)^alpha_k for T_k <= T < T_{k+1}.
    alpha_table: jnp.ndarray = jnp.array([])

    #: Townsend (2009) temporal-evolution coefficients Y_k at the bin edges.
    Y_table: jnp.ndarray = jnp.array([])

    #: Reference temperature of the temporal evolution function (the top of
    #: the table).
    reference_temperature: float = 1e8


class CoolingNetConfig(NamedTuple):
    """Static configuration of a neural-network cooling curve."""

    #: The static part of the (equinox) network, i.e. its architecture.
    network_static: Union[PyTree, NoneType] = None


class CoolingNetParams(NamedTuple):
    """Trainable parameters of a neural-network cooling curve."""

    #: The trainable part of the (equinox) network, i.e. its weights.
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

    #: Ratio of the shear time to the minimum cooling time, t_sh / t_coolmin.
    xi: float = 0.5

    #: Shear Mach number v_rel / c_s with respect to the hot medium.
    mach_number: float = 0.5

    #: Density contrast chi between the cold and the hot phase (the hot to
    #: cold temperature ratio).
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

    #: The cooling-curve type tag (``SIMPLE_POWER_LAW``,
    #: ``PIECEWISE_POWER_LAW``, ``NEURAL_NET_COOLING``,
    #: ``NEURAL_NET_COOLING_WITH_DENSITY`` or ``SIMPLE_MIXING_LAYER_COOLING``).
    cooling_curve_type: int = SIMPLE_POWER_LAW

    #: The network architecture of a neural-network cooling curve.
    cooling_net_config: CoolingNetConfig = CoolingNetConfig()

    #: Which pair of number densities multiplies Lambda(T). The default
    #: ELECTRON_HYDROGEN_DENSITY is the historical n_e n_H convention;
    #: TOTAL_NUMBER_DENSITY is the n^2 convention used by Cholla / CGOLS. The
    #: two differ by mu_e mu_H / mu^2 (~4.30 for X = 0.76, Z = 0.02), so this
    #: is not a cosmetic choice.
    density_convention: int = ELECTRON_HYDROGEN_DENSITY

    #: Look up ``PIECEWISE_POWER_LAW`` bins (and the heating-factor table) by
    #: direct indexing instead of a binary search. Only valid when both tables
    #: are evenly spaced in log10 T, as ``ki02_gf12_cooling`` builds them; the
    #: bins, and so the rates, are the same as with the search.
    uniform_log_tables: bool = False


class CoolingConfig(NamedTuple):
    """Top-level cooling configuration (activation, method and curve)."""

    #: Switch for radiative cooling.
    cooling: bool = False

    #: Time integration of the temperature update, ``EXPLICIT_COOLING``
    #: (forward Euler) or ``IMPLICIT_COOLING`` (backward Euler, solved by a
    #: safeguarded Newton iteration).
    cooling_method: int = IMPLICIT_COOLING

    #: The cooling-curve model.
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

    #: Scale the heating by a per-cell, time-dependent FUV field from point
    #: sources, diluted with distance (see ``FUVSourceParams``). Off by
    #: default: the heating is then the spatially uniform ``heating_rate``.
    #: Not seen by the cooling-time step limiters, which keep the uniform rate.
    fuv_point_source_heating: bool = False

    #: Dust attenuation of the point-source FUV field (the background is not
    #: attenuated): ``FUV_ATTENUATION_NONE`` (default), ``FUV_ATTENUATION_SLAB``
    #: ``FUV_ATTENUATION_LOCAL_JEANS`` (see ``fuv_attenuation_factor``) or
    #: ``FUV_ATTENUATION_LINE_OF_SIGHT`` (see ``fuv_line_of_sight_factor``).
    fuv_attenuation: int = 0

    #: Add the FUV of the disk's own star formation (see ``fuv_disk_field``):
    #: Kennicutt-Schmidt on each column's gas surface density, attenuated by
    #: the slab factor. Needs ``fuv_point_source_heating``.
    fuv_disk_source: bool = False

    #: Evaluate the static part of the FUV field once per integration instead
    #: of in every step: the line-of-sight attenuation factor
    #: (``FUV_ATTENUATION_LINE_OF_SIGHT``; a square root, four table gathers
    #: and an exponential per cell) depends only on the geometry. It is
    #: stored in ``FUVSourceParams.los_factor_cache`` by the time integrator,
    #: at the cost of one scalar field of memory. Same values as the per-step
    #: evaluation.
    cache_fuv_geometry: bool = False

    #: Run the sub-cycled update with heating as a fused Pallas kernel
    #: (``_cooling_pallas.py``) when the backend is ``PALLAS``: the whole
    #: update of a block of cells in registers, with the sub-cycle loop of a
    #: block stopping once none of its cells has time left. Same result as
    #: the native loop to rounding. Opt-in; needs the piecewise power-law
    #: curve with ``uniform_log_tables`` and the heating factor table.
    pallas_subcycling: bool = False


#: No attenuation of the point-source FUV field.
FUV_ATTENUATION_NONE = 0
#: Slab attenuation (Ostriker, McKee & Leroy 2010):
#: min(1, (1 - E_2(tau/2)) / tau) with tau = kappa Sigma of the cell's column.
FUV_ATTENUATION_SLAB = 1
#: Local shielding exp(-kappa rho L_J), L_J the Jeans length at the cell's
#: sound speed capped at ``jeans_sound_speed_squared_cap`` (Safranek-Shrader
#: et al. 2017).
FUV_ATTENUATION_LOCAL_JEANS = 2
#: exp(-tau) along the straight line from the (single) source, tau read from
#: the tabulated (R, z) map ``FUVSourceParams.los_tau_table`` around it.
FUV_ATTENUATION_LINE_OF_SIGHT = 3


class FUVSourceParams(NamedTuple):
    """FUV point sources for distance-diluted photoelectric heating.

    With ``CoolingConfig.fuv_point_source_heating`` the heating rate of every
    cell is multiplied by the FUV intensity in units of the solar-neighbourhood
    value J_0 (Kim & Ostriker 2017, eq. heat, with Sigma_FUV / Sigma_FUV,0
    replaced by the optically thin field of the sources),

        J / J_0 = sum_k w_k A(t) / (4 pi (d_k^2 + r_s^2)) + background,

    where A(t) = L_FUV(t) / (4 pi J_0) is the total source luminosity over the
    reference flux (an area, in code length^2), d_k the distance of the cell
    to source k and r_s a softening length (the source radius). There is no
    attenuation.
    """

    #: Source positions in box coordinates, shape (num_sources, 3).
    source_positions: jnp.ndarray = jnp.zeros((1, 3))

    #: Fraction of the total luminosity emitted by each source (sums to 1).
    source_weights: jnp.ndarray = jnp.ones((1,))

    #: Softening length r_s (code length).
    softening_length: float = 1.0

    #: Times (ascending, code time) and values of A(t) = L_FUV(t) / (4 pi J_0)
    #: in code length^2; linearly interpolated, held constant beyond the ends.
    luminosity_times: jnp.ndarray = jnp.zeros((1,))
    luminosity_values: jnp.ndarray = jnp.zeros((1,))

    #: Uniform background J / J_0 (the metagalactic 0.0024 of Kim & Ostriker
    #: 2017).
    background: float = 0.0024

    #: FUV dust opacity kappa per unit gas mass (code length^2 / code mass),
    #: used by ``CoolingConfig.fuv_attenuation``.
    dust_opacity: float = 0.0

    #: Cap on the squared sound speed in the Jeans length of the local
    #: shielding (code velocity^2).
    jeans_sound_speed_squared_cap: float = 0.0

    #: G in code units, for the Jeans length of the local shielding.
    gravitational_constant: float = 0.0

    #: Line-of-sight optical depth of the source, tau(R, z) on a uniform grid
    #: in cylindrical radius R >= 0 and height z, both relative to the first
    #: source: tau[i, j] at R = i * los_r_spacing, z = los_z_min + j *
    #: los_z_spacing (``FUV_ATTENUATION_LINE_OF_SIGHT``).
    los_tau_table: jnp.ndarray = jnp.zeros((2, 2))
    los_r_spacing: float = 1.0
    los_z_min: float = 0.0
    los_z_spacing: float = 1.0

    #: Disk FUV of ``CoolingConfig.fuv_disk_source``: J / J_0 =
    #: disk_sfr_coefficient * Sigma^disk_sfr_index (Sigma in code surface
    #: density) before the slab attenuation, i.e. Sigma_SFR(Sigma) / Sigma_SFR,0.
    disk_sfr_coefficient: float = 0.0
    disk_sfr_index: float = 1.4

    #: The line-of-sight attenuation factor on the (padded) grid, filled in by
    #: the time integrator with ``CoolingConfig.cache_fuv_geometry``; empty
    #: (the default) means it is evaluated in every step.
    los_factor_cache: jnp.ndarray = jnp.zeros((0,))


class CoolingParams(NamedTuple):
    """
    Runtime cooling parameters: composition, temperature floor, the
    cooling-resolution limiter, the floor handling, the per-step cap, the
    heating and the cooling curve.
    """

    #: Hydrogen mass fraction X.
    hydrogen_mass_fraction: float = 0.76

    #: Metal mass fraction Z.
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

    #: Cooling-resolution limiter: the minimum number of grid cells over which
    #: the cooling length ``l_cool = c_s * t_cool`` must be resolved (0 = off).
    #: Where ``l_cool`` falls below this many cells, the cooling rate is
    #: suppressed by ``min(1, (l_cool / (resolution_limiter_alpha * dx))^2)``.
    #: An unresolved radiative shock otherwise collapses into a cell-scale cold
    #: dense layer with no pressure support, which the ram pressure then
    #: crushes without bound; suppressing the unrepresentable cooling keeps
    #: such layers at the resolved adiabatic solution, while resolved cooling
    #: regions are untouched. Applied by both the finite-difference and the
    #: finite-volume solver (not by the mixing-layer cooling model).
    resolution_limiter_alpha: float = 0.0

    #: How the temperature floor is enforced after the cooling update.
    #:
    #: ``False`` (default): a cell whose update would take it below
    #: ``floor_temperature`` keeps its original temperature, i.e. the whole
    #: update is discarded. ``True``: it is clamped onto the floor instead.
    #: Cells that start below the floor are left alone either way.
    #:
    #: Clamping is the more defensible numerics: reverting makes cooling
    #: non-monotone in dt, and a cell at 1e5 K that would cross the floor stays
    #: at 1e5 K rather than reaching 1e4 K. Clamping is also what makes the
    #: explicit update usable for stiff cooling: with the revert, a stiff
    #: forward step always overshoots the floor and is discarded, so
    #: ``EXPLICIT_COOLING`` applies no cooling at all in those cells.
    #:
    #: NOTE: The revert stays the default because it suppresses cooling
    #: exactly in the unresolved cells that would otherwise collapse, which
    #: stabilises unresolved radiative shocks under ram pressure. Enable
    #: clamping only together with a crush control (e.g. the
    #: cooling-resolution limiter or the per-step cap) that keeps such layers
    #: stable.
    clamp_to_floor: bool = False

    #: Cap on the fractional temperature drop applied in one hydro step
    #: (0 = off; 0.3 means a cell may lose at most 30% of its temperature per
    #: step). Backward Euler is unconditionally stable for the cooling ODE, but
    #: the operator splitting is not: a cell taken from its post-shock
    #: temperature to the floor in a single step loses its pressure support
    #: before the hydrodynamics ever sees an intermediate state, and the
    #: neighbours ram into it. Capping the per-step drop reaches the same
    #: equilibrium (over several steps instead of one) and, unlike a cooling
    #: CFL condition, is purely local: one crushing cell does not throttle the
    #: global time step.
    max_cooling_fraction: float = 0.0

    #: Constant volumetric ISM heating (0 = off), as the effective
    #: rescaled-temperature rate: dT~/dt = (gamma - 1) * heating_rate.
    #: Physically this represents a heating rate Gamma per particle (e.g.
    #: photoelectric heating, de/dt = n * Gamma), whose T~ rate is
    #: density-independent; the conversion from a cgs Gamma [erg/s] belongs in
    #: the problem setup. Together with a cooling curve that extends below
    #: 1e4 K (``athenak_ism_cooling``) this enables the classic two-phase
    #: thermal-instability equilibrium (Lambda = Gamma).
    heating_rate: float = 0.0

    #: Optional temperature dependence of the heating, Gamma(T) =
    #: heating_rate * f(T~), tabulated as log10 of the rescaled temperature
    #: (ascending) and the dimensionless factor f. Empty tables (default) mean
    #: f = 1, i.e. the old constant heating. Needed for TIGRESS-style
    #: photoelectric heating that switches off in ionised gas: without the
    #: cut-off, constant heating spuriously heats a hot rarefied wind
    #: (n_H Gamma ~ n_H^2 Lambda already at n_H ~ 1e-3 cm^-3, T ~ 1e6-1e7 K).
    heating_log10_T_table: jnp.ndarray = jnp.array([])

    #: The heating factor f at ``heating_log10_T_table`` (linear in log10 T~).
    heating_factor_table: jnp.ndarray = jnp.array([])

    #: FUV sources of the distance-diluted heating (used only with
    #: ``CoolingConfig.fuv_point_source_heating``).
    fuv_source_params: FUVSourceParams = FUVSourceParams()

    #: The parameters of the cooling curve selected by
    #: ``CoolingCurveConfig.cooling_curve_type``.
    cooling_curve_params: COOLING_CURVE_TYPE = SimplePowerLawParams()

    #: Tables of ``TABULATED_EXACT_COOLING`` (placeholders unless that method
    #: is used; build them with ``tabulated_cooling_from_curve``).
    tabulated_cooling: TabulatedCoolingParams = TabulatedCoolingParams()
