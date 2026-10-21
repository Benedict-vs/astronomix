"""
Synthetic soft X-ray and H-alpha emission of the simulated gas.

Pure, jnp-based kernels that turn (density, pressure) fields into volume
emissivities, plus line-of-sight projection, minor-axis profile and luminosity
helpers. The kernels trace under ``jit`` (and so run in-situ inside a snapshot
callback, sharded or not) and accept plain numpy arrays for post-processing.

Three emission models are provided:

* **X-ray, CGOLS recipe** (Schneider & Robertson 2018, arXiv:1803.01008,
  Sec. IV, Eq. 17): ``n^2 Lambda_A4(T)`` with ``n = rho / (mu m_p)``, the
  bolometric Eq. A4 CIE cooling fit (the one the cooling module uses, see
  :class:`CIEParabolicParams`), counted only for cells whose ``(3/2) k_B T``
  lies in the 0.3-2.0 keV band, i.e. 2.32e6 K <= T <= 1.55e7 K.
* **X-ray, AtomDB band emissivity**: ``n_e n_H epsilon_{0.3-2 keV}(T, Z)`` from
  a pre-computed APEC table (``data/xray_emissivity_0p3_2keV_atomdb.npz``,
  built once by ``data/generate_xray_emissivity_table.py``; no runtime
  dependency on pyatomdb). ``epsilon = epsilon_HHe + sum_el A_el epsilon_el``,
  so the metallicity / abundance pattern is a free choice. "Solar" means
  Asplund et al. (2009) by default (``SOLAR_ABUNDANCE_SET``); Anders &
  Grevesse (1989), the set the table is tabulated in, can be selected instead.
* **H-alpha** (Knoell et al. 2026, arXiv:2606.10762, Eq. 3):
  ``j = 2.63e-33 n_e n_H / T^0.9 erg cm^-3 s^-1 arcsec^-2`` for fully ionised
  gas, restricted to a temperature window, plus the plain emission measure
  ``n_e n_H`` in the same window.

Composition. The electron and hydrogen number densities of the last two models
use the ionised X = 0.71, Y = 0.27, Z = 0.02 mixture of Eq. 3:
``n_H = X rho / m_p = 0.71 rho / m_p`` and
``n_e = (X + Y/2 + Z/2) rho / m_p = 0.855 rho / m_p``, rounded to 0.86 as
written in Eq. 3. The CGOLS recipe keeps its own ``n = rho / (0.6 m_p)``
convention (``n_e n_H = 0.22 n^2``); the temperature is always the simulation's
``T = mu m_p P / (k_B rho)`` with ``mu = 0.6``.

Float32 safety. Volume emissivities of order 1e-30 erg cm^-3 s^-1 and below
flush to zero in single precision, so every kernel returns its emissivity in an
explicit unit (``XRAY_EMISSIVITY_UNIT``, ``HALPHA_EMISSIVITY_UNIT``) and the
projection multiplies the unit back in only after the line-of-sight sum.
Luminosities (~1e40 erg/s) overflow float32 and are therefore summed on the
host in float64.
"""

# general
import math
import os

# typing
from typing import NamedTuple, Optional

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# units and constants
import astropy.constants as const
from astropy import units as u

# astronomix containers
from astronomix._modules._cooling.cooling_options import CIEParabolicParams

# astronomix functions
from astronomix._modules._cooling._cooling import cooling_rate_cie_parabolic


# -------------------------------------------------------------
# =============== ↓ Constants ↓ ===============================
# -------------------------------------------------------------

#: Mean molecular weight of the CGOLS simulations ("we take mu = 0.6 throughout").
CGOLS_MEAN_MOLECULAR_WEIGHT = 0.6

#: Ionised composition of Eq. 3 of arXiv:2606.10762 (mass fractions).
HYDROGEN_MASS_FRACTION = 0.71
HELIUM_MASS_FRACTION = 0.27
METAL_MASS_FRACTION = 0.02

#: Free electrons and hydrogen nuclei per proton mass of gas, ``n = f rho / m_p``.
#: X + Y/2 + Z/2 = 0.855, rounded to 0.86 as written in Eq. 3.
ELECTRONS_PER_PROTON_MASS = 0.86
HYDROGEN_NUCLEI_PER_PROTON_MASS = HYDROGEN_MASS_FRACTION

#: The CGOLS X-ray "band": cells with (3/2) k_B T in 0.3-2.0 keV.
CGOLS_XRAY_BAND_KEV = (0.3, 2.0)
CGOLS_XRAY_TEMPERATURE_MIN_K = float(
    (CGOLS_XRAY_BAND_KEV[0] * u.keV / (1.5 * const.k_B)).to(u.K).value
)
CGOLS_XRAY_TEMPERATURE_MAX_K = float(
    (CGOLS_XRAY_BAND_KEV[1] * u.keV / (1.5 * const.k_B)).to(u.K).value
)

#: Eq. 3 of arXiv:2606.10762 (fit to Osterbrock's case-B table).
HALPHA_COEFFICIENT = 2.63e-33  # erg cm^3 s^-1 arcsec^-2 K^0.9
HALPHA_TEMPERATURE_EXPONENT = -0.9
#: Default H-alpha / emission-measure temperature window [K]. Above it the
#: ionised gas still emits (proportional to T^-0.9 n^2) but negligibly; below it
#: the gas is not ionised by any assumption we would defend.
HALPHA_TEMPERATURE_WINDOW_K = (5.0e3, 5.0e4)

#: Units of the volume emissivities returned by the kernels (see the module
#: docstring, "Float32 safety").
XRAY_EMISSIVITY_UNIT = 1e-30  # erg cm^-3 s^-1
HALPHA_EMISSIVITY_UNIT = 1e-40  # erg cm^-3 s^-1 arcsec^-2

#: Solid-angle conversions.
ARCSEC2_PER_STERADIAN = float((1.0 * u.sr).to(u.arcsec**2).value)
ARCMIN2_PER_STERADIAN = float((1.0 * u.sr).to(u.arcmin**2).value)

CENTIMETERS_PER_PARSEC = float((1.0 * u.pc).to(u.cm).value)

#: The AtomDB 0.3-2.0 keV band-emissivity table shipped with astronomix.
DEFAULT_XRAY_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "xray_emissivity_0p3_2keV_atomdb.npz",
)

#: Above the table (AtomDB's CIE grid ends at 1e9 K) the band emissivity is
#: extrapolated as free-free emission far below kT, epsilon ~ T^-1/2.
HIGH_TEMPERATURE_LOG_SLOPE = -0.5

ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
)  # fmt: skip

#: The solar metal abundances "Z = Z_sun" refers to: "A09" (Asplund et al.
#: 2009, represented by the Grevesse & Asplund 2010 set AtomDB ships) or "AG89"
#: (Anders & Grevesse 1989, the set the table rows are tabulated in).
SOLAR_ABUNDANCE_SET = "A09"

#: M82 hot-wind abundances of Lopez et al. 2020 (arXiv:2006.08623, Table 2),
#: by number relative to Asplund et al. 2009: the mean over the eight wind
#: regions at |z| >= 0.9 kpc (N2-N5, S2-S5), frozen values excluded and upper
#: limits taken at their limit. Undetected metals were fixed to solar in the
#: paper and are solar here.
LOPEZ2020_M82_WIND_ABUNDANCES = {
    "O": 0.76,
    "Ne": 1.49,
    "Mg": 0.94,
    "Si": 0.92,
    "S": 0.53,
    "Fe": 0.37,
}

# -------------------------------------------------------------
# =============== ↑ Constants ↑ ===============================
# -------------------------------------------------------------


class EmissionConversions(NamedTuple):
    """Code-unit -> cgs factors the emission kernels need (all plain floats)."""

    #: ``n = rho_code * this`` [cm^-3], n = rho / (mu m_p).
    total_number_density_per_code_density: float
    #: ``n_e = rho_code * this`` [cm^-3].
    electron_density_per_code_density: float
    #: ``n_H = rho_code * this`` [cm^-3].
    hydrogen_density_per_code_density: float
    #: ``T = (P_code / rho_code) * this`` [K].
    kelvin_per_code_specific_energy: float
    #: One code length in cm.
    centimeters_per_code_length: float


def emission_conversions(
    code_units,
    mean_molecular_weight: float = CGOLS_MEAN_MOLECULAR_WEIGHT,
) -> EmissionConversions:
    """
    Build the code-unit conversion factors for the emission kernels.

    Args:
        code_units: The simulation's :class:`astronomix.CodeUnits`.
        mean_molecular_weight: The mu entering ``T = mu m_p P / (k_B rho)`` and
            the CGOLS ``n = rho / (mu m_p)``.

    Returns:
        An :class:`EmissionConversions`.
    """
    code_density_cgs = (1.0 * code_units.code_density).to(u.g / u.cm**3).value
    proton_mass_cgs = const.m_p.to(u.g).value
    nucleons_per_code_density = code_density_cgs / proton_mass_cgs
    kelvin_per_code_specific_energy = (
        mean_molecular_weight * const.m_p / const.k_B * code_units.code_velocity**2
    ).to(u.K).value
    return EmissionConversions(
        total_number_density_per_code_density=float(
            nucleons_per_code_density / mean_molecular_weight
        ),
        electron_density_per_code_density=float(
            nucleons_per_code_density * ELECTRONS_PER_PROTON_MASS
        ),
        hydrogen_density_per_code_density=float(
            nucleons_per_code_density * HYDROGEN_NUCLEI_PER_PROTON_MASS
        ),
        kelvin_per_code_specific_energy=float(kelvin_per_code_specific_energy),
        centimeters_per_code_length=float((1.0 * code_units.code_length).to(u.cm).value),
    )


# -------------------------------------------------------------
# =============== ↓ X-ray: CGOLS recipe ↓ =====================
# -------------------------------------------------------------


def cgols_xray_window(temperature):
    """Boolean mask of the CGOLS 0.3-2.0 keV temperature window (T in K)."""
    return (temperature >= CGOLS_XRAY_TEMPERATURE_MIN_K) & (
        temperature <= CGOLS_XRAY_TEMPERATURE_MAX_K
    )


def cgols_xray_emissivity(
    number_density,
    temperature,
    unit: float = XRAY_EMISSIVITY_UNIT,
):
    """
    CGOLS I Eq. 17 integrand ``n^2 Lambda_A4(T)``, windowed to 0.3-2.0 keV.

    Args:
        number_density: Total particle number density n = rho / (mu m_p) [cm^-3].
        temperature: Gas temperature [K].
        unit: The unit of the returned emissivity [erg cm^-3 s^-1].

    Returns:
        The volume emissivity in units of ``unit``; zero outside the window.
    """
    # Reuse the cooling module's Eq. A4 implementation: with no temperature
    # shift it takes T in Kelvin, and the cgs -> "unit" rescaling enters as the
    # additive log10 shift inside its exponent (no 1e-23 intermediate).
    curve_in_unit = CIEParabolicParams(log10_lambda_cgs_to_code=-math.log10(unit))
    cooling_function_in_unit = cooling_rate_cie_parabolic(temperature, curve_in_unit)
    return jnp.where(
        cgols_xray_window(temperature),
        number_density * number_density * cooling_function_in_unit,
        0.0,
    )


# -------------------------------------------------------------
# =============== ↑ X-ray: CGOLS recipe ↑ =====================
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ X-ray: AtomDB band emissivity ↓ ===========
# -------------------------------------------------------------


class XrayEmissivityTable(NamedTuple):
    """The AtomDB 0.3-2.0 keV band-emissivity table (see the module docstring).

    The temperature grid is uniform in log10 T, which lets the lookup be a
    gather-free index computation (see :func:`_interpolate_uniform_grid`).
    """

    #: First grid point and spacing of the uniform log10 T [K] grid.
    log10_temperature_start: float
    log10_temperature_step: float
    #: log10 epsilon of hydrogen + helium [erg cm^3 s^-1 per n_e n_H].
    log10_emissivity_hydrogen_helium: np.ndarray
    #: log10 epsilon of each metal at its AG89 solar abundance, (n_metals, n_T).
    log10_emissivity_metals: np.ndarray
    #: Atomic numbers of the metal rows.
    metal_atomic_numbers: tuple
    #: Solar abundances (number, relative to H) of the metal rows.
    solar_abundance_ag89: np.ndarray
    solar_abundance_ga10: np.ndarray


def load_xray_emissivity_table(path: str = DEFAULT_XRAY_TABLE_PATH) -> XrayEmissivityTable:
    """
    Load the AtomDB band-emissivity table.

    Args:
        path: The ``.npz`` table written by
            ``data/generate_xray_emissivity_table.py``.

    Returns:
        An :class:`XrayEmissivityTable` (host numpy arrays).
    """
    with np.load(path) as data:
        log10_temperature = np.asarray(data["log10_temperature"], dtype=np.float64)
        steps = np.diff(log10_temperature)
        if not np.allclose(steps, steps[0], rtol=0, atol=1e-9):
            raise ValueError(f"{path}: the log10 T grid must be uniform")
        return XrayEmissivityTable(
            log10_temperature_start=float(log10_temperature[0]),
            log10_temperature_step=float(steps[0]),
            log10_emissivity_hydrogen_helium=np.asarray(
                data["log10_emissivity_hydrogen_helium"], dtype=np.float64
            ),
            log10_emissivity_metals=np.asarray(
                data["log10_emissivity_metals"], dtype=np.float64
            ),
            metal_atomic_numbers=tuple(int(z) for z in data["metal_atomic_numbers"]),
            solar_abundance_ag89=np.asarray(data["solar_abundance_ag89"], dtype=np.float64),
            solar_abundance_ga10=np.asarray(data["solar_abundance_ga10"], dtype=np.float64),
        )


def table_log10_temperature(table: XrayEmissivityTable) -> np.ndarray:
    """The table's log10 T [K] grid."""
    num_points = table.log10_emissivity_hydrogen_helium.size
    return table.log10_temperature_start + table.log10_temperature_step * np.arange(num_points)


def metal_abundance_weights(
    table: XrayEmissivityTable,
    abundances: Optional[dict] = None,
    reference: str = SOLAR_ABUNDANCE_SET,
) -> np.ndarray:
    """
    Per-metal weights turning the table's AG89-solar rows into a pattern.

    Args:
        table: The emissivity table.
        abundances: ``{element symbol: abundance relative to solar}``; metals
            not listed are solar. ``None`` is the solar pattern.
        reference: The solar table ``abundances`` is quoted against: ``"A09"``
            (Asplund et al. 2009, represented by the Grevesse & Asplund 2010
            set AtomDB ships; the default) or ``"AG89"`` (Anders & Grevesse
            1989). The rows are rescaled by the ratio of the two sets.

    Returns:
        The weights ``A_el`` (n_metals,) of ``sum_el A_el epsilon_el``.
    """
    abundances = abundances or {}
    unknown = set(abundances) - set(ELEMENT_SYMBOLS)
    if unknown:
        raise ValueError(f"unknown element symbols {sorted(unknown)}")
    if reference == "AG89":
        reference_to_ag89 = np.ones_like(table.solar_abundance_ag89)
    elif reference == "A09":
        reference_to_ag89 = table.solar_abundance_ga10 / table.solar_abundance_ag89
    else:
        raise ValueError(f"reference must be 'AG89' or 'A09', got {reference!r}")
    relative_to_reference = np.array(
        [abundances.get(ELEMENT_SYMBOLS[z - 1], 1.0) for z in table.metal_atomic_numbers]
    )
    return relative_to_reference * reference_to_ag89


def combined_log10_emissivity(
    table: XrayEmissivityTable,
    metal_weights,
    include_hydrogen_helium: bool = True,
) -> np.ndarray:
    """
    log10 of ``[epsilon_HHe +] sum_el w_el epsilon_el`` on the table grid.

    Combining on the grid (in linear space) before interpolating keeps the
    per-cell work to a single lookup however many elements contribute.

    Args:
        table: The emissivity table.
        metal_weights: Per-metal weights on the table rows, which are at AG89
            solar abundance (see :func:`metal_abundance_weights`, which also
            converts to A09); a scalar applies the same factor to every row (a
            metallicity in units of AG89 solar).
        include_hydrogen_helium: Add the H + He part.

    Returns:
        log10 epsilon [erg cm^3 s^-1] on the grid.
    """
    weights = np.broadcast_to(
        np.asarray(metal_weights, dtype=np.float64),
        (len(table.metal_atomic_numbers),),
    )
    emissivity = weights @ (10.0**table.log10_emissivity_metals)
    if include_hydrogen_helium:
        emissivity = emissivity + 10.0**table.log10_emissivity_hydrogen_helium
    return np.log10(np.maximum(emissivity, 1e-60))


def _interpolate_uniform_grid(log10_temperature, grid_start, grid_step, grid_values):
    """Linear interpolation on a uniform grid without a search.

    ``jnp.interp`` lowers to a searchsorted that materialises full-size index
    arrays; on a uniform grid the bracketing index is a floor, and the lookup
    of a ~200-entry table fuses into the surrounding elementwise kernel.
    Clamped to the grid ends (the callers handle out-of-range temperatures).
    """
    num_points = grid_values.shape[-1]
    position = jnp.clip((log10_temperature - grid_start) / grid_step, 0.0, num_points - 1.0)
    lower_index = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, num_points - 2)
    fraction = position - lower_index
    values = jnp.asarray(grid_values, dtype=jnp.result_type(log10_temperature))
    return values[lower_index] * (1.0 - fraction) + values[lower_index + 1] * fraction


def band_emissivity_coefficient(
    temperature,
    table: XrayEmissivityTable,
    log10_emissivity_grid,
    unit: float = XRAY_EMISSIVITY_UNIT,
):
    """
    The band emissivity per ``n_e n_H``, ``epsilon(T)``, in units of ``unit``.

    Args:
        temperature: Gas temperature [K].
        table: The emissivity table (supplies the temperature grid).
        log10_emissivity_grid: log10 epsilon on that grid, e.g. from
            :func:`combined_log10_emissivity`.
        unit: The unit of ``epsilon n_e n_H`` [erg cm^-3 s^-1] (the densities
            are in cm^-3, so ``epsilon`` comes out in ``unit`` cm^6).

    Returns:
        ``epsilon / unit`` [cm^6]: zero below the table, extrapolated as
        T^-1/2 above it.
    """
    log10_grid = np.asarray(log10_emissivity_grid, dtype=np.float64)
    grid_start = table.log10_temperature_start
    grid_step = table.log10_temperature_step
    grid_end = grid_start + grid_step * (log10_grid.size - 1)

    log10_temperature = jnp.log10(jnp.maximum(temperature, 1.0))
    log10_coefficient = _interpolate_uniform_grid(
        log10_temperature,
        grid_start,
        grid_step,
        log10_grid - math.log10(unit),
    )
    beyond_table = jnp.maximum(log10_temperature - grid_end, 0.0)
    log10_coefficient = log10_coefficient + HIGH_TEMPERATURE_LOG_SLOPE * beyond_table
    return jnp.where(log10_temperature >= grid_start, 10.0**log10_coefficient, 0.0)


def atomdb_xray_emissivity(
    electron_density,
    hydrogen_density,
    temperature,
    table: XrayEmissivityTable,
    log10_emissivity_grid,
    unit: float = XRAY_EMISSIVITY_UNIT,
):
    """
    ``n_e n_H epsilon_{0.3-2 keV}(T)`` in units of ``unit`` (no temperature window).

    Args:
        electron_density: n_e [cm^-3].
        hydrogen_density: n_H [cm^-3].
        temperature: Gas temperature [K].
        table: The emissivity table.
        log10_emissivity_grid: log10 epsilon on the table grid, see
            :func:`combined_log10_emissivity`.
        unit: The unit of the returned emissivity [erg cm^-3 s^-1].

    Returns:
        The volume emissivity in units of ``unit``.
    """
    coefficient = band_emissivity_coefficient(
        temperature,
        table,
        log10_emissivity_grid,
        unit,
    )
    return electron_density * hydrogen_density * coefficient


# -------------------------------------------------------------
# =============== ↑ X-ray: AtomDB band emissivity ↑ ===========
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ H-alpha and emission measure ↓ ============
# -------------------------------------------------------------


def halpha_window(temperature, temperature_window=HALPHA_TEMPERATURE_WINDOW_K):
    """Boolean mask of the H-alpha temperature window (T in K, inclusive)."""
    return (temperature >= temperature_window[0]) & (temperature <= temperature_window[1])


def halpha_emissivity(
    electron_density,
    hydrogen_density,
    temperature,
    temperature_window=HALPHA_TEMPERATURE_WINDOW_K,
    unit: float = HALPHA_EMISSIVITY_UNIT,
):
    """
    Eq. 3 of arXiv:2606.10762, ``2.63e-33 n_e n_H T^-0.9``, in a T window.

    Args:
        electron_density: n_e [cm^-3].
        hydrogen_density: n_H [cm^-3].
        temperature: Gas temperature [K].
        temperature_window: Inclusive (T_min, T_max) [K] of the emitting gas.
        unit: The unit of the returned emissivity [erg cm^-3 s^-1 arcsec^-2].

    Returns:
        The volume emissivity per arcsec^2 in units of ``unit``.
    """
    # Clamp before the power so the (discarded) out-of-window branch stays
    # finite at T = 0.
    safe_temperature = jnp.maximum(temperature, 1.0)
    emissivity = (
        (HALPHA_COEFFICIENT / unit)
        * electron_density
        * hydrogen_density
        * safe_temperature**HALPHA_TEMPERATURE_EXPONENT
    )
    return jnp.where(halpha_window(temperature, temperature_window), emissivity, 0.0)


def emission_measure_integrand(
    electron_density,
    hydrogen_density,
    temperature,
    temperature_window=HALPHA_TEMPERATURE_WINDOW_K,
):
    """``n_e n_H`` [cm^-6] of the gas in the H-alpha temperature window."""
    return jnp.where(
        halpha_window(temperature, temperature_window),
        electron_density * hydrogen_density,
        0.0,
    )


def vertical_mask(vertical_coordinate, mask_half_height: Optional[float]):
    """``|z| >= mask_half_height`` (the gas outside a disk slab); all-True if None."""
    if mask_half_height is None:
        return jnp.ones_like(vertical_coordinate, dtype=bool)
    return jnp.abs(vertical_coordinate) >= mask_half_height


# -------------------------------------------------------------
# =============== ↑ H-alpha and emission measure ↑ ============
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Projections, profiles, luminosities ↓ =====
# -------------------------------------------------------------

#: Map name -> (physical unit of the projected map, whether it is per steradian).
#: The X-ray maps are specific intensities per steradian (Eq. 17 carries the
#: 1/4pi); Eq. 3 is already per arcsec^2; the emission measure is a column.
MAP_UNITS = {
    "xray_cgols": ("erg s^-1 cm^-2 sr^-1", True),
    "xray_atomdb_hhe": ("erg s^-1 cm^-2 sr^-1", True),
    "xray_atomdb_metals": ("erg s^-1 cm^-2 sr^-1", True),
    "xray_atomdb_lopez": ("erg s^-1 cm^-2 sr^-1", True),
    "halpha": ("erg s^-1 cm^-2 arcsec^-2", False),
    "emission_measure": ("pc cm^-6", False),
}


class EmissionMapSettings(NamedTuple):
    """Static choices for :func:`emission_maps`."""

    #: H-alpha / emission-measure temperature window [K].
    halpha_temperature_window: tuple = HALPHA_TEMPERATURE_WINDOW_K
    #: Drop H-alpha / EM of cells with |z| below this [code length]; None = off.
    #: For an edge-on view this is the same as masking the projected map rows.
    halpha_disk_mask_half_height: Optional[float] = None
    #: Line-of-sight axes of the spatial (x, y, z) field: 1 -> x-z map, 0 -> y-z.
    projection_axes: tuple = (1, 0)
    #: Viewing inclination [deg]; only edge-on (90) is implemented.
    inclination_deg: float = 90.0
    #: The solar set the "xray_atomdb_metals" map is normalised to (see
    #: SOLAR_ABUNDANCE_SET).
    solar_abundance_set: str = SOLAR_ABUNDANCE_SET


def line_of_sight_integral(field, axis: int, scale: float):
    """``scale * sum(field, axis)``; the scale (unit x path length) goes in last."""
    return jnp.sum(field, axis=axis) * scale


def emission_maps(
    density,
    pressure,
    conversions: EmissionConversions,
    table: XrayEmissivityTable,
    cell_size_code: float,
    settings: Optional[EmissionMapSettings] = None,
    vertical_coordinate=None,
):
    """
    Edge-on emission maps of a 3D (x, y, z) block, in physical units.

    Every map is a plain sum over the line of sight, so blocks of a domain can
    be processed separately and their maps added (the post-processing driver
    does this to stay within memory), and a sharded field reduces with a single
    collective. The X-ray AtomDB model is stored as two maps, H+He and the
    solar-pattern metals (``settings.solar_abundance_set``), so the metallicity
    remains a post-hoc factor: ``S = S_hhe + Z S_metals``. A third, complete
    map uses the Lopez+20 M82 wind abundance pattern
    (:data:`LOPEZ2020_M82_WIND_ABUNDANCES`), which no scalar Z reproduces.

    Args:
        density: Density field [code units], shape (nx, ny, nz).
        pressure: Pressure field [code units], same shape.
        conversions: The code -> cgs factors (:func:`emission_conversions`).
        table: The AtomDB band-emissivity table.
        cell_size_code: The (cubic) cell size [code length].
        settings: See :class:`EmissionMapSettings`; ``None`` is the default.
        vertical_coordinate: Cell-centre z [code length], broadcastable to the
            field; only needed with a disk mask.

    Returns:
        ``{f"{name}_{plane}": map}`` with ``name`` from :data:`MAP_UNITS` and
        ``plane`` "xz" (LOS along y) or "yz" (LOS along x).
    """
    if settings is None:
        settings = EmissionMapSettings()
    if settings.inclination_deg != 90.0:
        # TODO: inclined views (M82: i ~ 80 deg). Rotate about x by shearing
        # each y-plane in z, z' = z + y cot(i), before summing along y; the
        # kernels are unchanged, only this projection step needs it.
        raise NotImplementedError("only edge-on (inclination 90 deg) projections exist")

    temperature = (pressure / density) * conversions.kelvin_per_code_specific_energy
    number_density = density * conversions.total_number_density_per_code_density
    electron_density = density * conversions.electron_density_per_code_density
    hydrogen_density = density * conversions.hydrogen_density_per_code_density

    solar_metals = combined_log10_emissivity(
        table,
        metal_abundance_weights(table, reference=settings.solar_abundance_set),
        include_hydrogen_helium=False,
    )
    lopez_mixture = combined_log10_emissivity(
        table,
        metal_abundance_weights(table, LOPEZ2020_M82_WIND_ABUNDANCES, reference="A09"),
        include_hydrogen_helium=True,
    )
    integrands = {
        "xray_cgols": cgols_xray_emissivity(number_density, temperature),
        "xray_atomdb_hhe": atomdb_xray_emissivity(
            electron_density,
            hydrogen_density,
            temperature,
            table,
            table.log10_emissivity_hydrogen_helium,
        ),
        "xray_atomdb_metals": atomdb_xray_emissivity(
            electron_density,
            hydrogen_density,
            temperature,
            table,
            solar_metals,
        ),
        "xray_atomdb_lopez": atomdb_xray_emissivity(
            electron_density,
            hydrogen_density,
            temperature,
            table,
            lopez_mixture,
        ),
        "halpha": halpha_emissivity(
            electron_density,
            hydrogen_density,
            temperature,
            settings.halpha_temperature_window,
        ),
        "emission_measure": emission_measure_integrand(
            electron_density,
            hydrogen_density,
            temperature,
            settings.halpha_temperature_window,
        ),
    }
    if settings.halpha_disk_mask_half_height is not None:
        outside_disk = vertical_mask(
            vertical_coordinate,
            settings.halpha_disk_mask_half_height,
        )
        for name in ("halpha", "emission_measure"):
            integrands[name] = jnp.where(outside_disk, integrands[name], 0.0)

    path_length_cm = cell_size_code * conversions.centimeters_per_code_length
    scales = {
        "xray_cgols": XRAY_EMISSIVITY_UNIT * path_length_cm / (4.0 * math.pi),
        "xray_atomdb_hhe": XRAY_EMISSIVITY_UNIT * path_length_cm / (4.0 * math.pi),
        "xray_atomdb_metals": XRAY_EMISSIVITY_UNIT * path_length_cm / (4.0 * math.pi),
        "xray_atomdb_lopez": XRAY_EMISSIVITY_UNIT * path_length_cm / (4.0 * math.pi),
        "halpha": HALPHA_EMISSIVITY_UNIT * path_length_cm,
        "emission_measure": path_length_cm / CENTIMETERS_PER_PARSEC,
    }
    plane_of_axis = {1: "xz", 0: "yz"}
    return {
        f"{name}_{plane_of_axis[axis]}": line_of_sight_integral(
            integrands[name],
            axis,
            scales[name],
        )
        for axis in settings.projection_axes
        for name in integrands
    }


def luminosity_from_map(surface_brightness_map, pixel_area_cm2: float, name: str) -> float:
    """
    Total luminosity [erg/s] of an optically thin, isotropic emitter from one map.

    ``L = 4 pi sum(I) dA`` with ``I`` per steradian; an H-alpha map (per
    arcsec^2) is converted first. Summed in float64 on the host: ~1e40 erg/s
    overflows float32.

    Args:
        surface_brightness_map: A projected map from :func:`emission_maps`.
        pixel_area_cm2: The area of one map pixel [cm^2].
        name: The map's model name (a key of :data:`MAP_UNITS`).

    Returns:
        The luminosity [erg/s] (for the emission measure: the volume emission
        measure ``int n_e n_H dV`` [cm^-3]).
    """
    total = float(np.sum(np.asarray(surface_brightness_map, dtype=np.float64)))
    if name == "emission_measure":
        return total * CENTIMETERS_PER_PARSEC * pixel_area_cm2
    if name == "halpha":
        total = total * ARCSEC2_PER_STERADIAN
    return 4.0 * math.pi * total * pixel_area_cm2


def minor_axis_profile(
    surface_brightness_map,
    horizontal_centers,
    vertical_centers,
    horizontal_half_width: float,
    bin_width: float,
):
    """
    North / south minor-axis profiles of an edge-on map (CGOLS I Fig. 10).

    The map is averaged over the strip ``|x| <= horizontal_half_width`` and
    over ``|z|`` bins of ``bin_width`` starting at the midplane, i.e. the
    strip's summed brightness divided by its area. Pixels are assigned by
    their centre.

    Args:
        surface_brightness_map: (n_horizontal, n_vertical) map.
        horizontal_centers: Pixel-centre horizontal coordinates.
        vertical_centers: Pixel-centre z.
        horizontal_half_width: Strip half-width (same units as the centres).
        bin_width: |z| bin width (same units).

    Returns:
        ``(bin_centers, north, south)``: |z| bin centres and the mean surface
        brightness above (z > 0) and below (z < 0) the midplane, in the map's
        unit; NaN for an empty bin.
    """
    brightness = np.asarray(surface_brightness_map, dtype=np.float64)
    horizontal_centers = np.asarray(horizontal_centers, dtype=np.float64)
    vertical_centers = np.asarray(vertical_centers, dtype=np.float64)
    in_strip = np.abs(horizontal_centers) <= horizontal_half_width
    strip_mean_per_row = brightness[in_strip].mean(axis=0)

    num_bins = int(np.ceil(np.max(np.abs(vertical_centers)) / bin_width - 1e-9))
    bin_index = np.floor(np.abs(vertical_centers) / bin_width + 1e-9).astype(int)
    bin_index = np.minimum(bin_index, num_bins - 1)
    bin_centers = (np.arange(num_bins) + 0.5) * bin_width

    def side_profile(side_selector):
        sums = np.bincount(
            bin_index[side_selector],
            weights=strip_mean_per_row[side_selector],
            minlength=num_bins,
        )
        counts = np.bincount(bin_index[side_selector], minlength=num_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, sums / counts, np.nan)

    return bin_centers, side_profile(vertical_centers > 0), side_profile(vertical_centers < 0)


# -------------------------------------------------------------
# =============== ↑ Projections, profiles, luminosities ↑ =====
# -------------------------------------------------------------
