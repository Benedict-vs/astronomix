"""
Unit tests of the synthetic-emission kernels (astronomix/analysis_helpers/emission.py).

Checks, all on the CPU and against independent references:

  1. the CGOLS X-ray recipe: the 0.3-2.0 keV temperature window and the value
     n^2 Lambda_A4(T) against a direct rendering of Eq. A4;
  2. the AtomDB band-emissivity table: exact at the nodes, log-linear between
     them, zero below and T^-1/2 above the grid, the per-element decomposition,
     abundance weights and a physical sanity value;
  3. the H-alpha coefficient of Eq. 3 (arXiv:2606.10762) against case-B
     recombination (Draine 2011 eq. 14.8, Hui & Gnedin 1997 alpha_B);
  4. luminosities of a uniform box against the analytic n^2 Lambda V, for every
     model and both projections;
  5. the minor-axis profile of a synthetic exponential map;
  6. float32 vs float64 agreement of the maps for low-density gas.

    JAX_PLATFORMS=cpu python -m pytest pytests/emission/test_emission_kernels.py
"""

# general
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ruff: noqa: E402

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# units and constants
import astropy.constants as const
from astropy import units as u

# astronomix constants
from astronomix.analysis_helpers.emission import (
    ARCSEC2_PER_STERADIAN,
    CGOLS_XRAY_TEMPERATURE_MAX_K,
    CGOLS_XRAY_TEMPERATURE_MIN_K,
    HALPHA_COEFFICIENT,
    LOPEZ2020_M82_WIND_ABUNDANCES,
    SOLAR_ABUNDANCE_SET,
    XRAY_EMISSIVITY_UNIT,
)

# astronomix containers
from astronomix.analysis_helpers.emission import EmissionMapSettings
from astronomix import CodeUnits

# astronomix functions
from astronomix.analysis_helpers.emission import (
    band_emissivity_coefficient,
    cgols_xray_emissivity,
    combined_log10_emissivity,
    emission_conversions,
    emission_maps,
    halpha_emissivity,
    load_xray_emissivity_table,
    luminosity_from_map,
    metal_abundance_weights,
    minor_axis_profile,
    table_log10_temperature,
)

CODE_UNITS = CodeUnits(1 * u.kiloparsec, 1e6 * u.M_sun, 100 * u.km / u.s)


def lambda_a4_reference(temperature):
    """Eq. A4 of arXiv:1803.01008 written out independently [erg cm^3 s^-1]."""
    log_t = np.log10(temperature)
    if log_t < 4.0:
        return 0.0
    if log_t < 5.9:
        return 10.0 ** (-1.3 * (log_t - 5.25) ** 2 - 21.25)
    if log_t < 7.4:
        return 10.0 ** (0.7 * (log_t - 7.1) ** 2 - 22.8)
    return 10.0 ** (0.45 * log_t - 26.065)


@pytest.fixture(scope="module")
def table():
    return load_xray_emissivity_table()


@pytest.fixture(autouse=True)
def double_precision():
    """Run each check in float64 so tolerances are meaningful."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


# -------------------------------------------------------------
# 1. CGOLS X-ray recipe
# -------------------------------------------------------------


def test_cgols_window_edges():
    # (3/2) k_B T = 0.3 / 2.0 keV.
    kelvin_per_kev = (1 * u.keV / const.k_B).to(u.K).value
    assert CGOLS_XRAY_TEMPERATURE_MIN_K == pytest.approx(0.3 * kelvin_per_kev / 1.5, rel=1e-12)
    assert CGOLS_XRAY_TEMPERATURE_MAX_K == pytest.approx(2.0 * kelvin_per_kev / 1.5, rel=1e-12)
    assert CGOLS_XRAY_TEMPERATURE_MIN_K == pytest.approx(2.32e6, rel=1e-3)
    assert CGOLS_XRAY_TEMPERATURE_MAX_K == pytest.approx(1.55e7, rel=2e-3)


@pytest.mark.parametrize(
    "temperature, inside",
    [
        (1e4, False),
        (1e6, False),
        (2.30e6, False),
        (2.33e6, True),
        (5e6, True),
        (1.0e7, True),
        (1.54e7, True),
        (1.56e7, False),
        (1e8, False),
    ],
)
def test_cgols_window_and_value(temperature, inside):
    number_density = 3e-3
    emissivity = float(
        cgols_xray_emissivity(jnp.array(number_density), jnp.array(temperature))
    ) * XRAY_EMISSIVITY_UNIT
    expected = number_density**2 * lambda_a4_reference(temperature) if inside else 0.0
    assert emissivity == pytest.approx(expected, rel=1e-10, abs=0.0)


# -------------------------------------------------------------
# 2. AtomDB band-emissivity table
# -------------------------------------------------------------


def test_table_interpolation(table):
    log10_temperature = table_log10_temperature(table)
    grid = combined_log10_emissivity(table, metal_weights=1.0)
    unit = XRAY_EMISSIVITY_UNIT

    # Exact at the nodes.
    at_nodes = band_emissivity_coefficient(10.0**log10_temperature, table, grid, unit)
    np.testing.assert_allclose(np.asarray(at_nodes) * unit, 10.0**grid, rtol=1e-10)

    # Log-linear in between (matches np.interp in log-log).
    rng = np.random.default_rng(0)
    log10_samples = rng.uniform(log10_temperature[0], log10_temperature[-1], 1000)
    interpolated = band_emissivity_coefficient(10.0**log10_samples, table, grid, unit)
    reference = 10.0 ** np.interp(log10_samples, log10_temperature, grid)
    np.testing.assert_allclose(np.asarray(interpolated) * unit, reference, rtol=1e-10)

    # Zero below the grid, T^-1/2 above it.
    below = band_emissivity_coefficient(jnp.array([1e3, 9.9e4]), table, grid, unit)
    assert np.all(np.asarray(below) == 0.0)
    top = float(band_emissivity_coefficient(jnp.array(1e9), table, grid, unit))
    above = float(band_emissivity_coefficient(jnp.array(4e9), table, grid, unit))
    assert above / top == pytest.approx(0.5, rel=1e-10)


def test_table_composition_and_sanity(table):
    log10_temperature = table_log10_temperature(table)
    assert log10_temperature[0] == pytest.approx(5.0)
    assert log10_temperature[-1] == pytest.approx(9.0)
    assert log10_temperature.size == 201

    # AG89 (the rows' own set) solar weights are all one and reproduce the
    # plain sum; a scalar Z scales the metals only.
    solar_ag89 = metal_abundance_weights(table, reference="AG89")
    np.testing.assert_array_equal(solar_ag89, np.ones(len(table.metal_atomic_numbers)))
    full = 10.0 ** combined_log10_emissivity(table, solar_ag89)
    metals_only = 10.0 ** combined_log10_emissivity(table, 1.0, include_hydrogen_helium=False)
    hydrogen_helium = 10.0**table.log10_emissivity_hydrogen_helium
    np.testing.assert_allclose(full, hydrogen_helium + metals_only, rtol=1e-12)
    half_solar = 10.0 ** combined_log10_emissivity(table, 0.5)
    np.testing.assert_allclose(half_solar, hydrogen_helium + 0.5 * metals_only, rtol=1e-12)

    # The default solar set is Asplund 09: every row rescaled by A09 / AG89,
    # checked against the raw table arrays.
    assert SOLAR_ABUNDANCE_SET == "A09"
    solar_a09 = metal_abundance_weights(table)
    ratio = table.solar_abundance_ga10 / table.solar_abundance_ag89
    np.testing.assert_allclose(solar_a09, ratio, rtol=1e-14)
    full_a09 = 10.0 ** combined_log10_emissivity(table, solar_a09)
    expected_a09 = hydrogen_helium + np.sum(
        ratio[:, None] * 10.0**table.log10_emissivity_metals,
        axis=0,
    )
    np.testing.assert_allclose(full_a09, expected_a09, rtol=1e-12)
    # Asplund's O and Fe are ~0.6-0.7x AG89's, so the A09 metals emit less.
    metals_a09 = expected_a09 - hydrogen_helium
    i7 = int(np.argmin(np.abs(log10_temperature - 7.0)))
    assert 0.5 < metals_a09[i7] / metals_only[i7] < 0.9

    # Abundances quoted against Asplund 09 are rescaled to the table's AG89.
    lopez = metal_abundance_weights(table, LOPEZ2020_M82_WIND_ABUNDANCES, reference="A09")
    iron_row = table.metal_atomic_numbers.index(26)
    expected_iron = 0.37 * table.solar_abundance_ga10[iron_row] / table.solar_abundance_ag89[iron_row]
    assert lopez[iron_row] == pytest.approx(expected_iron)
    with pytest.raises(ValueError):
        metal_abundance_weights(table, {"Xx": 1.0})

    # Physical sanity at 1e7 K: the 0.3-2 keV band holds roughly half of the
    # total (n_e n_H) cooling. Lambda_A4 is n^2-normalised; n^2 / (n_e n_H)
    # = 1 / (0.6^2 * 0.86 * 0.71) for the composition used here.
    i = int(np.argmin(np.abs(log10_temperature - 7.0)))
    band = full[i]
    total_ne_nh = lambda_a4_reference(1e7) / (0.6**2 * 0.86 * 0.71)
    assert 0.25 < band / total_ne_nh < 0.8, (band, total_ne_nh)


# -------------------------------------------------------------
# 3. H-alpha coefficient vs case B
# -------------------------------------------------------------


@pytest.mark.parametrize("temperature", [5e3, 7.5e3, 1e4, 1.5e4, 2e4])
def test_halpha_coefficient_case_b(temperature):
    halpha_photon_energy = (const.h * const.c / (656.28 * u.nm)).to(u.erg).value
    t4 = temperature / 1e4

    # Eq. 3 is per arcsec^2; the volume emissivity 4 pi j in erg cm^3 s^-1.
    eq3_per_ne_nh = (
        float(halpha_emissivity(jnp.array(1.0), jnp.array(1.0), jnp.array(temperature)))
        * 1e-40
        * 4.0
        * math.pi
        * ARCSEC2_PER_STERADIAN
    )
    assert eq3_per_ne_nh == pytest.approx(
        4.0 * math.pi * ARCSEC2_PER_STERADIAN * HALPHA_COEFFICIENT * temperature**-0.9,
        rel=1e-10,
    )

    # Draine (2011) eq. 14.8: effective case-B H-alpha recombination coefficient.
    alpha_effective_halpha = 1.17e-13 * t4 ** (-0.942 - 0.031 * math.log(t4))
    draine = alpha_effective_halpha * halpha_photon_energy
    assert eq3_per_ne_nh == pytest.approx(draine, rel=0.05)

    # Hui & Gnedin (1997) case-B alpha_B times the H-alpha photons per case-B
    # recombination (0.45 at 1e4 K, Osterbrock & Ferland 2006 Table 4.4; weakly
    # T dependent, hence the looser tolerance).
    lam = 2.0 * 157807.0 / temperature
    alpha_b = 2.753e-14 * lam**1.5 / (1.0 + (lam / 2.740) ** 0.407) ** 2.242
    assert eq3_per_ne_nh == pytest.approx(0.45 * alpha_b * halpha_photon_energy, rel=0.12)


def test_halpha_window():
    temperature = jnp.array([1e3, 4.9e3, 5e3, 1e4, 5e4, 5.1e4, 1e6])
    emissivity = np.asarray(halpha_emissivity(jnp.ones(7), jnp.ones(7), temperature))
    np.testing.assert_array_equal(emissivity > 0, [False, False, True, True, True, False, False])


# -------------------------------------------------------------
# 4. Uniform-box luminosities
# -------------------------------------------------------------


@pytest.mark.parametrize("temperature", [5e6, 1e4])
def test_uniform_box_luminosity(table, temperature):
    conversions = emission_conversions(CODE_UNITS)
    shape = (8, 6, 12)
    cell_size_code = 0.05  # 50 pc cells
    density_code = 0.2
    pressure_code = density_code * temperature / conversions.kelvin_per_code_specific_energy
    density = jnp.full(shape, density_code)
    pressure = jnp.full(shape, pressure_code)

    maps = emission_maps(density, pressure, conversions, table, cell_size_code)
    maps_ag89 = emission_maps(
        density,
        pressure,
        conversions,
        table,
        cell_size_code,
        EmissionMapSettings(solar_abundance_set="AG89"),
    )

    cell_cm = cell_size_code * conversions.centimeters_per_code_length
    volume_cm3 = np.prod(shape) * cell_cm**3
    n = density_code * conversions.total_number_density_per_code_density
    n_e = density_code * conversions.electron_density_per_code_density
    n_h = density_code * conversions.hydrogen_density_per_code_density
    log10_temperature = table_log10_temperature(table)

    def table_value(log10_grid):
        # The table starts at 1e5 K; below it the band emissivity is zero.
        if np.log10(temperature) < log10_temperature[0]:
            return 0.0
        return 10.0 ** np.interp(np.log10(temperature), log10_temperature, log10_grid)

    in_xray_window = CGOLS_XRAY_TEMPERATURE_MIN_K <= temperature <= CGOLS_XRAY_TEMPERATURE_MAX_K
    in_halpha_window = 5e3 <= temperature <= 5e4
    analytic = {
        "xray_cgols": n**2 * lambda_a4_reference(temperature) * volume_cm3 * in_xray_window,
        "xray_atomdb_hhe": n_e * n_h * table_value(table.log10_emissivity_hydrogen_helium) * volume_cm3,
        "xray_atomdb_metals": n_e
        * n_h
        * table_value(
            combined_log10_emissivity(
                table,
                table.solar_abundance_ga10 / table.solar_abundance_ag89,
                include_hydrogen_helium=False,
            )
        )
        * volume_cm3,
        "xray_atomdb_lopez": n_e
        * n_h
        * table_value(
            combined_log10_emissivity(
                table,
                metal_abundance_weights(table, LOPEZ2020_M82_WIND_ABUNDANCES, reference="A09"),
            )
        )
        * volume_cm3,
        "halpha": 4.0 * math.pi * ARCSEC2_PER_STERADIAN * HALPHA_COEFFICIENT
        * n_e * n_h * temperature**-0.9 * volume_cm3 * in_halpha_window,
        "emission_measure": n_e * n_h * volume_cm3 * in_halpha_window,
    }
    pixel_area_cm2 = cell_cm**2
    for name, expected in analytic.items():
        for plane in ("xz", "yz"):
            luminosity = luminosity_from_map(maps[f"{name}_{plane}"], pixel_area_cm2, name)
            assert luminosity == pytest.approx(expected, rel=1e-9, abs=1e-300), (name, plane)
    assert set(maps) == {f"{name}_{plane}" for name in analytic for plane in ("xz", "yz")}

    # AG89-normalised metals: the table rows as they are.
    expected_ag89 = n_e * n_h * volume_cm3 * table_value(
        combined_log10_emissivity(table, 1.0, include_hydrogen_helium=False)
    )
    luminosity_ag89 = luminosity_from_map(maps_ag89["xray_atomdb_metals_xz"], pixel_area_cm2, "xray_atomdb_metals")
    assert luminosity_ag89 == pytest.approx(expected_ag89, rel=1e-9, abs=1e-300)

    # The (pressure, density) -> T conversion is the cgols.py one (mu = 0.6).
    mu_mp_over_kb = 0.6 * const.m_p / const.k_B * (100 * u.km / u.s) ** 2
    assert conversions.kelvin_per_code_specific_energy == pytest.approx(
        mu_mp_over_kb.to(u.K).value, rel=1e-12
    )


def test_inclined_projection_is_a_hook(table):
    conversions = emission_conversions(CODE_UNITS)
    field = jnp.ones((4, 4, 4))
    with pytest.raises(NotImplementedError):
        emission_maps(
            field,
            field,
            conversions,
            table,
            0.1,
            EmissionMapSettings(inclination_deg=80.0),
        )


# -------------------------------------------------------------
# 5. Minor-axis profile
# -------------------------------------------------------------


def test_minor_axis_profile():
    cell = 0.078125  # kpc, two cells per 0.15625 kpc bin
    x = (np.arange(128) + 0.5) * cell - 5.0
    z = (np.arange(256) + 0.5) * cell - 10.0
    scale_height = 1.2
    brightness = np.exp(-np.abs(z) / scale_height)[None, :] * np.ones((x.size, 1))
    brightness[:, z < 0] *= 2.0  # brighter south side
    centers, north, south = minor_axis_profile(brightness, x, z, 5.0, 0.15625)
    assert centers.size == 64
    assert centers[0] == pytest.approx(0.078125)
    # Bin mean of exp(-|z|/H) over the two cells in each bin.
    expected = np.exp(-np.abs(z[z > 0]) / scale_height).reshape(-1, 2).mean(axis=1)
    np.testing.assert_allclose(north, expected, rtol=1e-12)
    np.testing.assert_allclose(south, 2.0 * expected, rtol=1e-12)


# -------------------------------------------------------------
# 6. float32 vs float64
# -------------------------------------------------------------


def test_float32_matches_float64(table):
    conversions = emission_conversions(CODE_UNITS)
    rng = np.random.default_rng(1)
    shape = (16, 16, 32)
    # Densities from the 1e-4 code floor (n ~ 7e-6 cm^-3) to the disk (~1e3),
    # temperatures 3e3 - 3e8 K.
    density = 10.0 ** rng.uniform(-4.0, 4.0, shape)
    temperature = 10.0 ** rng.uniform(3.5, 8.5, shape)
    pressure = density * temperature / conversions.kelvin_per_code_specific_energy

    maps64 = emission_maps(jnp.asarray(density), jnp.asarray(pressure), conversions, table, 0.02)
    maps32 = emission_maps(
        jnp.asarray(density, dtype=jnp.float32),
        jnp.asarray(pressure, dtype=jnp.float32),
        conversions,
        table,
        0.02,
    )
    for name in maps64:
        reference = np.asarray(maps64[name])
        single = np.asarray(maps32[name], dtype=np.float64)
        assert np.all(np.isfinite(single)), name
        assert maps32[name].dtype == jnp.float32
        scale = np.max(np.abs(reference))
        assert scale > 0, name
        np.testing.assert_allclose(single, reference, rtol=2e-4, atol=1e-6 * scale, err_msg=name)
