"""
The photoelectric heating on dust only in the neutral gas (``photoelectric_neutral_only``).

Run from the repository root (CPU, a few minutes)::

    pytest pytests/chemistry/test_dust_heating.py

KROME applies the heating of Bakes & Tielens (1994) at every temperature; its fit holds
below about 1e4 K and its efficiency grows as T^0.7 above. With the option the term is
multiplied by 1 - n(H+)/n_H, as the photoelectric heating of the KI02 cooling: unchanged
in neutral gas, gone in ionised gas, and the same in the per-cell kernel as in Kvaerno5.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("diffrax")

import test_radiation_field as radiation
from test_radiation_field import network_file  # noqa: F401  (fixture)

from astronomix._modules._chemistry import _krome_photo as photo
from astronomix._modules._chemistry._chemistry import update_chemistry
from astronomix._modules._chemistry._krome_network import build_krome_network


def _network(path, neutral_only):
    return build_krome_network(
        path,
        photo_bins=photo.photo_spectrum("draine", 5.0, 13.6, 10, scale=100.0),
        user_variables={"user_myfluxLW": 100.0 * radiation.DRAINE_LYMAN_WERNER},
        radiation_user_variables=("user_myfluxLW",),
        shielding="WG11",
        h2_formation_on_dust=True,
        photoelectric_neutral_only=neutral_only,
    )


@pytest.mark.parametrize("ionised", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("temperature", [80.0, 8.0e3, 3.0e6])
def test_the_heating_follows_the_neutral_hydrogen(network_file, ionised, temperature):  # noqa: F811
    krome, species = _network(network_file, False)
    neutral_only, _ = _network(network_file, True)
    fractions = {name: 1.0e-20 for name in species}
    fractions.update({"H": 1.0 - ionised, "H+": ionised, "C+": 1.4e-4})
    fractions["E"] = ionised + 1.4e-4
    n = jnp.asarray([fractions[name] * 2.0 for name in species])  # 2 hydrogen nuclei per cm^3
    terms = krome.heating_terms(n, temperature, radiation.COSMIC_RAY_RATE, 1.0)
    kept = neutral_only.heating_terms(n, temperature, radiation.COSMIC_RAY_RATE, 1.0)
    assert float(terms["heating_photoelectric"]) != 0.0
    np.testing.assert_allclose(float(kept["heating_photoelectric"]), (1.0 - ionised) * float(terms["heating_photoelectric"]), rtol=1e-12)
    for name in terms:
        if name != "heating_photoelectric":
            np.testing.assert_array_equal(np.asarray(kept[name]), np.asarray(terms[name]), err_msg=name)


def _chemistry(path, neutral_only, kernel):
    from astronomix.setup_helpers import attach_per_cell_kernel, build_chemistry_from_network_file

    chemistry_config, chemistry_params, species = build_chemistry_from_network_file(
        path, "krome",
        number_density_unit_cgs=1.0, temperature_unit_kelvin=1.0, time_unit_seconds=radiation.TIME_STEP_SECONDS,
        cosmic_ray_rate=radiation.COSMIC_RAY_RATE, thermochemistry=True, cooling_courant=1e3, floor_temperature=10.0,
        absolute_tolerance=1e-14, relative_tolerance=1e-8, max_steps=4096,
        krome_h2_formation_on_dust=True, krome_shielding="WG11",
        krome_photo_bins=photo.photo_spectrum("draine", 5.0, 13.6, 10, scale=100.0),
        krome_user_variables={"user_myfluxLW": 100.0 * radiation.DRAINE_LYMAN_WERNER},
        krome_radiation_user_variables=("user_myfluxLW",),
        krome_photoelectric_neutral_only=neutral_only,
        temperature_from_species=True,
    )  # fmt: skip
    if kernel:
        chemistry_config = attach_per_cell_kernel(
            chemistry_config, chemistry_params, radiation.GAMMA, cache_key=("pytest-dust-heating", path, neutral_only)
        )
    return chemistry_config, chemistry_params, tuple(species)


def test_the_kernel_has_the_option_too(network_file):  # noqa: F811
    """Gas of 30 nuclei per cm^3 at 300 K, 0.1 % ionised, in 100 Draine fields: the kernel and Kvaerno5 agree with the
    option, and the option changes the result (a little: the gas is nearly neutral)."""
    pytest.importorskip("gradsolve")
    pytest.importorskip("warp")
    reacted = {}
    for neutral_only in (False, True):
        for kernel in (False, True):
            chemistry_config, chemistry_params, species = _chemistry(network_file, neutral_only, kernel)
            state, registered_variables, params = radiation._row_of_cells(chemistry_config, chemistry_params, species, 2)
            reacted[neutral_only, kernel] = np.asarray(update_chemistry(state, registered_variables, chemistry_config, params, 1.0))
    temperature = {key: value[2] / value[3:].sum(axis=0) for key, value in reacted.items()}
    np.testing.assert_allclose(temperature[True, True], temperature[True, False], rtol=1e-6)
    np.testing.assert_allclose(temperature[False, True], temperature[False, False], rtol=1e-6)
    assert not np.allclose(temperature[True, False], temperature[False, False], rtol=1e-9)
