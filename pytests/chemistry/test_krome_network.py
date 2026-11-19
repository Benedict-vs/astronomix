"""
Networks in the KROME format (``astronomix/_modules/_chemistry/_krome_network.py``).

1. The reader on a small network written here in the KROME format: layouts,
   temperature limits, ``@var`` and ``@common``, Fortran constants, the rate
   functions KROME evaluates internally, photo-reactions, the reactions KROME
   skips with a warning, and the errors for what it does not accept.
2. The right-hand side: conservation of nuclei and charge, and the rate
   coefficients against the same formulas written in NumPy.
3. The chemistry module on that network: ``update_chemistry`` with the vmapped
   Kvaerno5 against SciPy's Radau on the same right-hand side.
4. The heating and cooling of a KROME network: the terms against the same
   formulas written in NumPy, with metal line data and a cooling table written
   here in KROME's formats, and the temperature equation through
   ``update_chemistry``.
5. What KROME takes from its build options: the dust shielding of the Habing
   flux, the column densities, the H2 self-shielding and the reaction
   modifiers; the heating by the cosmic-ray reactions; the H2 line cooling and
   the bremsstrahlung against values computed with KROME; and the photo bins
   (cross sections, rates, heating and Habing flux) against the same sums
   written in NumPy, with a small database written here in KROME's formats.
6. Optionally a full KROME network file, when ASTRONOMIX_TEST_KROME_NETWORK
   points at one (for instance ``networks/react_galaxy_ism`` of KROME): it is
   read, its reactions conserve nuclei and charge, and its rates are finite;
   with ASTRONOMIX_TEST_KROME_DATA (KROME's ``data`` folder) its heating and
   cooling are finite too, also with a binned spectrum and the shielding.

The file runs on the CPU unless a platform is chosen in the environment.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from astronomix._modules._chemistry._krome_network import (
    ELECTRON_VOLT_TO_ERG,
    KELVIN_TO_ELECTRON_VOLT,
    build_krome_network,
    parse_krome_network,
)

# A hydrogen-helium-carbon network in the KROME format, written for this test.
NETWORK = """# test network
@var:T=Tgas
@common: user_crate, user_flux
@var:c1 = 2.5d0
@var:u1 = 11.26d0*invTe

# default layout: three reactants, four products, limits, rate
1,H,E,,H+,E,E,,NONE,NONE,exp(-32.7d0+13.5d0*lnTe-5.7d0*(lnTe**2))
2,H+,E,,H,g,,,NONE,.LE.5.5e3,3.92d-13*invTe**0.6353d0
3,H+,E,,H,g,,,>5.5e3,NONE,1.d-13*(T32)**(-0.7)

@format:idx,R,R,P,P,Tmin,Tmax,rate
4,HE+,H,He,H+,NONE,NONE,1.20d-15*(Tgas/3d2)**0.25d0
5,H2,H+,H2+,H,.GE.1.d2,.LT.1.d5,10**(-10.5+0.1*log10(Tgas))
6,H2+,H,H2,H+,NONE,1d4,6.0d-10

@format:idx,R,R,P,P,P,rate
7,C,E,C+,E,E,6.85d-8*u1**0.25*exp(-u1)/(0.193d0+u1)

@photo_start
@format:idx,R,P,P,rate
8,H,H+,E,auto
9,He,He+,E,auto
@photo_stop

@noTab_start
@format:idx,R,R,P,rate
10,H+,E,H,H_recombination_on_dust(n,Tgas)
11,C+,E,C,C_recombination_on_dust(n,Tgas)
@noTab_stop

@format:idx,R,P,P,rate
12,H2,H,H,H2_solomonLW(user_flux)*c1
@CR_start
13,H,H+,E,4.6d-1*user_crate
@CR_stop
"""

SPECIES = ("H", "E", "H+", "He+", "He", "H2", "H2+", "C", "C+")


@pytest.fixture(scope="module")
def network_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("krome") / "react_test"
    path.write_text(NETWORK)
    return str(path)


def _write(tmp_path, text):
    path = tmp_path / "react_broken"
    path.write_text(text)
    return str(path)


# ---------------------------------------------------------------------------
# 1. The reader
# ---------------------------------------------------------------------------


def test_reader(network_file):
    parsed = parse_krome_network(network_file)
    assert parsed["species"] == SPECIES  # order of first appearance, HE -> He, photons dropped
    assert [reaction[0] for reaction in parsed["reactions"]] == list(range(1, 14))
    assert parsed["commons"] == ("user_crate", "user_flux")
    assert [name for name, _ in parsed["variables"]] == ["t", "c1", "u1"]
    by_index = {reaction[0]: reaction for reaction in parsed["reactions"]}
    name = lambda indices: tuple(SPECIES[i] for i in indices)
    assert name(by_index[1][1]) == ("H", "E") and name(by_index[1][2]) == ("H+", "E", "E")
    assert name(by_index[2][2]) == ("H",)  # the photon is not a species
    assert by_index[2][3] is None and by_index[2][4] == ("LE", 5.5e3)
    assert by_index[3][3] == ("GT", 5.5e3) and by_index[3][4] is None
    assert by_index[5][3] == ("GE", 1e2) and by_index[5][4] == ("LT", 1e5)
    assert by_index[6][4] == ("LT", 1e4)  # a bare upper limit means "less than"
    assert [by_index[i][6] for i in (8, 9)] == [0, 1] and by_index[7][6] == -1  # photo slots
    assert parsed["photo"] == ((7, None, None, "auto"), (8, None, None, "auto"))  # position, limits, cross section
    assert parsed["cosmic_ray"] == (12,) and parsed["modifiers"] == () and parsed["skipped"] == ()


def test_reader_skips_a_malformed_reaction_as_krome(tmp_path):
    """A line that does not fit the layout in force is left out with a warning; the others are numbered in order."""
    text = (
        "@format:idx,R,R,P,P,Tmin,Tmax,rate\n"
        "1,H+,E,H,g,NONE,NONE,1d-12\n"
        "2,H,H+,E,auto\n"  # five fields under an eight-column layout, as when a @format line is commented out
        "3,He+,E,He,g,NONE,NONE,2d-12 ! a comment after the rate\n"
        "@photo_start\n@format:idx,R,P,P,rate\n4,He,He+,E,auto\n@photo_stop\n"
    )
    with pytest.warns(UserWarning) as record:
        parsed = parse_krome_network(_write(tmp_path, text))
    messages = [str(warning.message) for warning in record]
    assert any("wrong format for reaction 2" in message and "skipped as in KROME" in message for message in messages)
    assert any("not sequential" in message for message in messages)
    assert [reaction[0] for reaction in parsed["reactions"]] == [1, 2, 3]  # KROME's indices: the order of reading
    assert parsed["reactions"][1][5] == "2d-12" and len(parsed["skipped"]) == 1 and parsed["skipped"][0][0] == 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        network, species = build_krome_network(_write(tmp_path, text), photo_rates={3: 5e-11})  # by KROME index
    assert float(network.get_rates(100.0, 0.0, 0.0, 0.0, jnp.ones(len(species)))[2]) == 5e-11


def test_reader_refuses_what_it_does_not_read(tmp_path):
    with pytest.raises(ValueError, match="does not conserve"):
        parse_krome_network(_write(tmp_path, "@format:idx,R,R,P,rate\n1,H+,E,H2,1d-10\n"))
    with pytest.raises(NotImplementedError, match="@ghost"):
        parse_krome_network(_write(tmp_path, "@ghost:He\n"))
    with pytest.raises(ValueError, match="user_flux"):  # a user variable the rates need
        build_krome_network(_write(tmp_path, NETWORK))
    with pytest.raises(ValueError, match="not photo-reactions"):
        build_krome_network(_write(tmp_path, NETWORK), photo_rates={1: 1e-10}, user_variables={"user_flux": 0.0})
    with pytest.raises(ValueError, match="not both"):
        build_krome_network(
            _write(tmp_path, NETWORK), photo_rates={8: 1e-10}, user_variables={"user_flux": 0.0},
            photo_bins={"left": [5.0], "right": [13.6], "intensity": [1e-8]},
        )
    with pytest.raises(ValueError, match="krome_data_folder"):  # the cross sections of "auto" are in KROME's database
        build_krome_network(
            _write(tmp_path, NETWORK), user_variables={"user_flux": 0.0}, photo_bins={"left": [5.0], "right": [13.6], "intensity": [1e-8]}
        )
    with pytest.raises(ValueError, match="shielding"):
        build_krome_network(_write(tmp_path, NETWORK), user_variables={"user_flux": 0.0}, shielding="XYZ")
    with pytest.raises(NotImplementedError, match="modifier"):
        build_krome_network(
            _write(tmp_path, NETWORK + "@reactionModifier_start\nk(99)=k(99)*2d0\n@reactionModifier_stop\n"), user_variables={"user_flux": 0.0}
        )


# ---------------------------------------------------------------------------
# 2. The right-hand side
# ---------------------------------------------------------------------------


def _reference_rates(temperature, electrons, cosmic_ray_rate, habing, flux, photo, dust_to_gas=1.0):
    te = temperature * KELVIN_TO_ELECTRON_VOLT
    ln_te, inv_te, t32 = np.log(te), 1.0 / te, temperature / 300.0
    u1 = 11.26 * inv_te
    psi = habing * np.sqrt(temperature) / electrons
    dust = lambda c: 1e-14 * c[0] * dust_to_gas / (
        1.0 + c[1] * psi ** c[2] * (1.0 + c[3] * temperature ** c[4] * psi ** (-c[5] - c[6] * np.log(temperature)))
    )
    return np.array([
        np.exp(-32.7 + 13.5 * ln_te - 5.7 * ln_te**2),
        3.92e-13 * inv_te**0.6353 if temperature <= 5.5e3 else 0.0,
        1e-13 * t32 ** (-0.7) if temperature > 5.5e3 else 0.0,
        1.20e-15 * (temperature / 300.0) ** 0.25,
        10 ** (-10.5 + 0.1 * np.log10(temperature)) if 1e2 <= temperature < 1e5 else 0.0,
        6.0e-10 if temperature < 1e4 else 0.0,
        6.85e-8 * u1**0.25 * np.exp(-u1) / (0.193 + u1),
        photo[0],
        photo[1],
        dust((12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5)),
        dust((45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4)),
        1.38e9 * flux * ELECTRON_VOLT_TO_ERG * 2.5,
        0.46 * cosmic_ray_rate,
    ])


@pytest.mark.parametrize("temperature", [20.0, 99.9, 100.0, 5.5e3, 5.6e3, 9.9e3, 1.0e4, 2.0e5])
def test_rate_coefficients(network_file, temperature):
    network, species = build_krome_network(
        network_file, photo_rates={8: 3e-11, 9: 7e-12}, user_variables={"user_flux": 2e-10}
    )
    abundances = jnp.asarray(np.linspace(1.0, 3.0, len(species))).at[species.index("E")].set(0.3)
    rates = np.asarray(network.get_rates(temperature, 3e-17, 1.7, 0.0, abundances))
    reference = _reference_rates(temperature, 0.3, 3e-17, 1.7, 2e-10, (3e-11, 7e-12))
    np.testing.assert_allclose(rates, reference, rtol=1e-12, atol=0.0)


def test_dust_recombination_needs_electrons_and_a_field(network_file):
    network, species = build_krome_network(network_file, user_variables={"user_flux": 0.0})
    abundances = jnp.ones(len(species))
    position = [reaction[0] for reaction in network.reactions].index(10)
    assert float(network.get_rates(100.0, 0.0, 0.0, 0.0, abundances)[position]) == 0.0  # no field
    assert float(network.get_rates(100.0, 0.0, 1.7, 0.0, abundances.at[species.index("E")].set(0.0))[position]) == 0.0
    assert float(network.get_rates(100.0, 0.0, 1.7, 0.0, abundances)[position]) > 0.0


def test_right_hand_side_conserves_nuclei_and_charge(network_file):
    network, species = build_krome_network(
        network_file, photo_rates={8: 3e-11, 9: 7e-12}, user_variables={"user_flux": 2e-10}, h2_formation_on_dust=True
    )
    random = np.random.default_rng(3)
    ones, zeros = jnp.ones(network.reactions_number), jnp.zeros(network.reactions_number)
    hydrogen = np.array([1, 0, 1, 0, 0, 2, 2, 0, 0])
    helium = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0])
    carbon = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1])
    charge = np.array([0, -1, 1, 1, 0, 0, 1, 0, 1])
    for temperature in (50.0, 3.0e3, 2.0e4):
        abundances = jnp.asarray(10 ** random.uniform(-6, 2, len(species)))
        derivative = np.asarray(network(0.0, abundances, temperature, 3e-17, 1.7, 0.0, ones, zeros))
        scale = np.abs(derivative).max()
        assert np.isfinite(derivative).all() and scale > 0
        for budget in (hydrogen, helium, carbon, charge):
            assert abs(budget @ derivative) < 1e-12 * scale
    jacobian = jax.jacfwd(lambda y: network(0.0, y, 300.0, 3e-17, 1.7, 0.0, ones, zeros))(jnp.asarray(10 ** random.uniform(-6, 2, len(species))))
    assert bool(jnp.isfinite(jacobian).all())


def test_h2_formation_on_dust(network_file):
    plain, species = build_krome_network(network_file, user_variables={"user_flux": 0.0})
    dusty, _ = build_krome_network(network_file, user_variables={"user_flux": 0.0}, h2_formation_on_dust=True, dust_to_gas_relative=0.5)
    abundances = jnp.asarray(np.linspace(1.0, 3.0, len(species)))
    ones, zeros = jnp.ones(plain.reactions_number), jnp.zeros(plain.reactions_number)
    difference = np.asarray(dusty(0.0, abundances, 50.0, 0.0, 0.0, 0.0, ones, zeros) - plain(0.0, abundances, 50.0, 0.0, 0.0, 0.0, ones, zeros))
    nuclei = float(abundances[0] + abundances[2] + 2 * abundances[5] + 2 * abundances[6])  # H, H+, H2, H2+
    formation = 3.5e-17 * 0.5 * float(abundances[0]) * nuclei
    expected = np.zeros(len(species))
    expected[species.index("H")], expected[species.index("H2")] = -2 * formation, formation
    np.testing.assert_allclose(difference, expected, rtol=1e-9, atol=1e-30)


def test_single_precision_constants(network_file):
    """Constants without a ``d`` exponent can be read as a Fortran compiler reads them."""
    exact, species = build_krome_network(network_file, user_variables={"user_flux": 0.0})
    compiled, _ = build_krome_network(network_file, user_variables={"user_flux": 0.0}, single_precision_literals=True)
    abundances = jnp.ones(len(species))
    position = [reaction[0] for reaction in exact.reactions].index(3)  # 1.d-13*(T32)**(-0.7): -0.7 is single precision
    a = float(exact.get_rates(2.0e4, 0.0, 0.0, 0.0, abundances)[position])
    b = float(compiled.get_rates(2.0e4, 0.0, 0.0, 0.0, abundances)[position])
    t32 = 2.0e4 * float(np.float32(0.0033333333333333335))
    assert b == pytest.approx(1e-13 * t32 ** float(np.float32(-0.7)), rel=1e-13)
    assert a != b and a == pytest.approx(b, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. The chemistry module on the network
# ---------------------------------------------------------------------------


def test_update_chemistry_matches_scipy(network_file):
    scipy_integrate = pytest.importorskip("scipy.integrate")
    from astronomix import (
        SimulationConfig,
        SimulationParams,
        construct_primitive_state,
        finalize_config,
        get_registered_variables,
    )
    from astronomix._modules._chemistry._chemistry import update_chemistry
    from astronomix._modules._cooling._cooling import get_pressure_from_temperature
    from astronomix.option_classes.simulation_config import FINITE_VOLUME, NATIVE_JAX
    from astronomix.setup_helpers import build_chemistry_from_network_file

    num_cells, time_step_seconds, temperature = 6, 3.0e11, 300.0
    chemistry_config, chemistry_params, species = build_chemistry_from_network_file(
        network_file, "krome",
        number_density_unit_cgs=1.0, temperature_unit_kelvin=1.0, time_unit_seconds=time_step_seconds,
        cosmic_ray_rate=3e-17, fuv_field=1.7, thermochemistry=False,
        absolute_tolerance=1e-20, relative_tolerance=1e-8, max_steps=4096,
        krome_photo_rates={8: 3e-13}, krome_user_variables={"user_flux": 2e-10}, krome_h2_formation_on_dust=True,
    )
    assert species == SPECIES and chemistry_config.helium_index == SPECIES.index("He")
    config = SimulationConfig(
        dimensionality=1, solver_mode=FINITE_VOLUME, backend=NATIVE_JAX, box_size=1.0,
        num_cells=num_cells, progress_bar=False, chemistry_config=chemistry_config,
    )
    params = SimulationParams(t_end=1.0, gamma=5.0 / 3.0, chemistry_params=chemistry_params)
    registered_variables = get_registered_variables(config)
    hydrogen_nuclei = jnp.logspace(0.0, 3.0, num_cells)
    density = hydrogen_nuclei / chemistry_params.hydrogen_mass_fraction
    pressure = get_pressure_from_temperature(
        density, jnp.full(num_cells, temperature), chemistry_params.hydrogen_mass_fraction, chemistry_params.metal_mass_fraction
    )
    state = construct_primitive_state(
        config=config, registered_variables=registered_variables, density=density,
        velocity_x=jnp.zeros(num_cells), gas_pressure=pressure,
    )
    fractions = {"H": 0.99, "H+": 1e-3, "He": 0.08, "He+": 1e-5, "H2": 1e-4, "H2+": 1e-9, "C": 1e-5, "C+": 1.3e-4}
    fractions["E"] = fractions["H+"] + fractions["He+"] + fractions["H2+"] + fractions["C+"]
    start = registered_variables.chemistry_species_index
    for index, name in enumerate(species):
        state = state.at[start + index].set(fractions[name] * hydrogen_nuclei)
    config = finalize_config(config, state.shape)
    reacted = np.asarray(update_chemistry(state, registered_variables, config.chemistry_config, params, 1.0))[start : start + len(species)]

    network = chemistry_params.network
    ones, zeros = jnp.ones(network.reactions_number), jnp.zeros(network.reactions_number)
    right_hand_side = jax.jit(lambda y: network(0.0, y, temperature, 3e-17, 1.7, 2.0, ones, zeros))
    jacobian = jax.jit(jax.jacfwd(lambda y: network(0.0, y, temperature, 3e-17, 1.7, 2.0, ones, zeros)))
    initial = np.asarray(state)[start : start + len(species)]
    for cell in range(num_cells):
        solution = scipy_integrate.solve_ivp(
            lambda t, y: np.asarray(right_hand_side(jnp.asarray(y))), (0.0, time_step_seconds), initial[:, cell],
            method="Radau", jac=lambda t, y: np.asarray(jacobian(jnp.asarray(y))), rtol=1e-10, atol=1e-25,
        )
        assert solution.success
        expected = solution.y[:, -1]
        significant = expected > 1e-12 * float(hydrogen_nuclei[cell])
        np.testing.assert_allclose(reacted[significant, cell], expected[significant], rtol=2e-5)
    assert np.abs(reacted / initial - 1.0).max() > 1e-2  # the step did change the composition


# ---------------------------------------------------------------------------
# 4. Heating and cooling
# ---------------------------------------------------------------------------

# Two coolants in the format of KROME's coolZ.dat, written for this test: a
# three-level atom and a two-level ion, with a rate that changes above 500 K.
METAL_LINES = """# test coolants
@var:T2 = Tgas*1d-2
@TgasUpperLimit:1d4

metal:C
level 0: 0.e0, 1
level 1: 24.e0, 3
level 2: 63.e0, 5
1 -> 0, 7.9d-8
2 -> 1, 2.7d-7
2 -> 0, 2.1d-14
H, 1, 0, 1.6d-10*(T2)**(.14)
H, 2, 0, 9.2d-11*(T2)**(.26)
H, 2, 1, 2.9d-10*(T2)**(.26)
e, 1, 0, 2.0d-7
       if(Tgas > 5d2): 3.0d-7
para/ortho: 1/3
H2or, 2, 1, 2.9d-10
H2pa, 2, 1, 2.7D-10
endmetal

metal:C+
level 0: 0.e0, 2
level 1: 91.2e0, 4
1 -> 0, 2.4d-6
e, 1, 0, 2.8d-7*(T2)**(-.5)
H, 1, 0, 8d-10*(T2)**(.07)
endmetal

metal:Fe
level 0: 0.e0, 9
level 1: 598.43e0, 7
level 2: 1012.9e0, 5
level 3: 9968.2e0, 11
1 -> 0, 2.5d-3
endmetal
"""


def _write_cooling_data(folder):
    (folder / "coolZ.dat").write_text(METAL_LINES)
    density, temperature = np.linspace(-2.0, 4.0, 4), np.linspace(1.0, 8.0, 15)
    rows = ["# test table: log10(nH) log10(Tgas) log10(Lambda)", f"   {len(density)}, {len(temperature)}"]
    for x in density:
        rows += [f"{x:.8e} {y:.8e} {-24.0 + 0.3 * y - 0.1 * x + 0.02 * x * y:.8e}" for y in temperature] + [""]
    (folder / "coolZ_CIE2012NOUV.dat").write_text("\n".join(rows))


@pytest.fixture(scope="module")
def data_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("krome_data")
    _write_cooling_data(folder)
    return str(folder)


def _reference_thermochemistry(n, temperature, habing, cosmic_ray_rate, rates, dust_to_gas, formation):
    """The terms of ``heating_and_cooling_terms`` in NumPy; ``n`` maps a species to its density."""
    kb, ev = 1.380648e-16, 1.60217646e-12
    e, H, H2, Hp, He, Hep = n["E"], n["H"], n["H2"], n["H+"], n["He"], n["He+"]
    nuclei = H + Hp + 2 * H2 + 2 * n["H2+"]
    T, t5 = temperature, temperature / 1e5
    damp = 1.0 / (1.0 + np.sqrt(t5))
    rec = np.sqrt(T) * (T / 1e3) ** (-0.2) / (1.0 + (T / 1e6) ** 0.7)
    atomic = (
        1.27e-21 * np.sqrt(T) * damp * np.exp(-1.578091e5 / T) * e * H + 9.38e-22 * np.sqrt(T) * damp * np.exp(-2.853354e5 / T) * e * He
        + 4.95e-22 * np.sqrt(T) * damp * np.exp(-6.31515e5 / T) * e * Hep + 5.01e-27 * T ** (-0.1687) * damp * np.exp(-5.5338e4 / T) * e**2 * Hep
        + 8.7e-27 * rec * e * Hp + 1.55e-26 * T**0.3647 * e * Hep
        + 1.24e-13 * T ** (-1.5) * np.exp(-4.7e5 / T) * (1.0 + 0.3 * np.exp(-9.4e4 / T)) * e * Hep
        + 7.5e-19 * damp * np.exp(-1.18348e5 / T) * e * H + 9.1e-27 * T ** (-0.1687) * damp * np.exp(-1.3179e4 / T) * e**2 * Hep
        + 5.54e-17 * T ** (-0.397) * damp * np.exp(-4.73638e5 / T) * e * Hep
    )
    # metal lines: populations from the full rate matrix (not the closed forms of the module)
    t2, active = T / 100.0, T < 1e4
    def lines(total, energies, weights, radiative, collisional):
        size = len(energies); matrix = np.zeros((size, size))
        for (upper, lower), rate in collisional.items():
            rate = rate if active else 0.0
            matrix[lower, upper] += rate; matrix[upper, upper] -= rate
            excitation = rate * weights[upper] / weights[lower] * np.exp(-(energies[upper] - energies[lower]) / T)
            matrix[upper, lower] += excitation; matrix[lower, lower] -= excitation
        for (upper, lower), coefficient in radiative.items():
            matrix[lower, upper] += coefficient; matrix[upper, upper] -= coefficient
        matrix[0, :] = 1.0
        population = np.linalg.solve(matrix, np.eye(size)[0] * total)
        return sum(population[upper] * coefficient * (energies[upper] - energies[lower]) for (upper, lower), coefficient in radiative.items())
    carbon = lines(n["C"], [0.0, 24.0, 63.0], [1, 3, 5], {(1, 0): 7.9e-8, (2, 1): 2.7e-7, (2, 0): 2.1e-14},
                   {(1, 0): 1.6e-10 * t2**0.14 * H + (3.0e-7 if T > 5e2 else 2.0e-7) * e, (2, 0): 9.2e-11 * t2**0.26 * H,
                    (2, 1): 2.9e-10 * t2**0.26 * H + 2.9e-10 * H2 * 0.75 + 2.7e-10 * H2 * 0.25})
    ion = lines(n["C+"], [0.0, 91.2], [2, 4], {(1, 0): 2.4e-6}, {(1, 0): 2.8e-7 * t2 ** (-0.5) * e + 8e-10 * t2**0.07 * H})
    blend = 10.0 / (10.0 + np.exp(-10.0 * (np.log10(T) - 4.0)))
    x, y = np.log10(nuclei), np.log10(T)
    table = 10.0 ** (-24.0 + 0.3 * y - 0.1 * x + 0.02 * x * y) * nuclei**2 * dust_to_gas   # the test table is bilinear
    psi = habing * np.sqrt(T) / e
    efficiency = 4.9e-2 / (1.0 + 4e-3 * psi**0.73) + 3.7e-2 * (T * 1e-4) ** 0.7 / (1.0 + 2e-4 * psi)
    photoelectric = (1.3e-24 * efficiency * habing * nuclei - 4.65e-30 * T**0.94 * psi ** (0.735 * T ** (-0.068)) * e * H) * dust_to_gas
    critical = 1.0e6 * T ** (-0.5) / (1.6 * np.exp(-((4.0e2 / T) ** 2)) * H / nuclei + 1.4 * np.exp(-1.2e4 / (T + 1.2e3)) * H2 / nuclei)
    fraction = 1.0 / (1.0 + critical / nuclei)
    chemical = formation * dust_to_gas * H * nuclei * (4.2 * fraction + 0.2) + rates[6] * n["H2+"] * H * 1.83 * fraction   # reaction 6: H2+ + H -> H2 + H+
    return {
        "cooling_atomic": atomic, "cooling_compton": 5.65e-36 * (T - 2.73) * e, "cooling_metal_lines": (1.0 - blend) * (carbon + ion) * kb,
        "cooling_metal_table": blend * table, "heating_photoelectric": photoelectric, "heating_chemical": chemical * ev,
        "heating_cosmic_rays": rates[13] * H * 4.3 * 1.60217662e-12,  # reaction 13, H -> H+ + E in the cosmic-ray block
        "heating_photo": 0.0,  # no binned spectrum
    }


@pytest.mark.parametrize("temperature", [30.0, 400.0, 3.0e3, 2.0e4, 3.0e6])
def test_heating_and_cooling_terms(network_file, data_folder, temperature):
    network, species = build_krome_network(
        network_file, user_variables={"user_flux": 0.0}, h2_formation_on_dust=True, dust_to_gas_relative=0.5, data_folder=data_folder
    )
    assert [dict(metal)["name"] for metal in network.metal_lines[2]] == ["C", "C+"]  # Fe is not in the network
    fractions = {"H": 0.7, "E": 2e-3, "H+": 1.8e-3, "He+": 1e-4, "He": 0.08, "H2": 0.14, "H2+": 1e-6, "C": 1e-4, "C+": 1e-4}
    hydrogen_nuclei = 50.0
    abundances = jnp.asarray([fractions[name] * hydrogen_nuclei for name in species])
    terms = {name: float(value) for name, value in network.heating_and_cooling_terms(abundances, temperature, 3e-17, 1.7).items()}
    rates = {reaction[0]: float(rate) for reaction, rate in zip(network.reactions, network.get_rates(temperature, 3e-17, 1.7, 0.0, abundances))}
    reference = _reference_thermochemistry(
        {name: fractions[name] * hydrogen_nuclei for name in species}, temperature, 1.7, 3e-17, rates, 0.5, 3.5e-17
    )
    for name, expected in reference.items():
        assert terms[name] == pytest.approx(expected, rel=1e-9, abs=1e-60), name
    assert terms["cooling_h2"] >= 0.0 and set(terms) == set(reference) | {"cooling_h2"}
    # the temperature derivative is the net heating over the heat capacity of all particles; as in KROME the
    # cooling at the floor temperature (2.73 K) is subtracted from the cooling
    net = sum(v for k, v in terms.items() if k.startswith("heating")) - sum(v for k, v in terms.items() if k.startswith("cooling"))
    floor = sum(float(value) for value in network.cooling_terms(abundances, 2.73).values())
    assert 0.0 < floor < sum(v for k, v in terms.items() if k.startswith("cooling"))
    expected = (5.0 / 3.0 - 1.0) * (net + floor) / (1.380648e-16 * float(abundances.sum()))
    assert float(network.temperature_derivative(abundances, temperature, 5.0 / 3.0, 3e-17, 1.7)) == pytest.approx(expected, rel=1e-12)
    warm, _ = build_krome_network(
        network_file, user_variables={"user_flux": 0.0}, h2_formation_on_dust=True, dust_to_gas_relative=0.5,
        data_folder=data_folder, floor_temperature=20.0,
    )
    floor = sum(float(value) for value in warm.cooling_terms(abundances, 20.0).values())
    expected = (5.0 / 3.0 - 1.0) * (net + floor) / (1.380648e-16 * float(abundances.sum()))
    assert float(warm.temperature_derivative(abundances, temperature, 5.0 / 3.0, 3e-17, 1.7)) == pytest.approx(expected, rel=1e-12)


def test_metals_without_krome_data(network_file):
    """Without KROME's data folder the metals cool through the two lines of the chemistry module."""
    network, species = build_krome_network(network_file, user_variables={"user_flux": 0.0})
    assert network.metal_lines == () and network.metal_table == ()
    terms = network.heating_and_cooling_terms(jnp.ones(len(species)), 100.0, 3e-17, 1.7)
    assert "cooling_metal_table" not in terms and float(terms["cooling_metal_lines"]) > 0.0


def test_update_chemistry_with_heating_and_cooling(network_file, data_folder):
    """The temperature equation of a KROME network through the module, against SciPy on the same right-hand side."""
    scipy_integrate = pytest.importorskip("scipy.integrate")
    from astronomix import (
        SimulationConfig,
        SimulationParams,
        construct_primitive_state,
        finalize_config,
        get_registered_variables,
    )
    from astronomix._modules._chemistry._chemistry import update_chemistry
    from astronomix._modules._cooling._cooling import get_pressure_from_temperature, get_temperature_from_pressure
    from astronomix.option_classes.simulation_config import FINITE_VOLUME, NATIVE_JAX
    from astronomix.setup_helpers import build_chemistry_from_network_file

    num_cells, time_step_seconds, gamma = 4, 1.0e12, 5.0 / 3.0
    chemistry_config, chemistry_params, species = build_chemistry_from_network_file(
        network_file, "krome",
        number_density_unit_cgs=1.0, temperature_unit_kelvin=1.0, time_unit_seconds=time_step_seconds,
        cosmic_ray_rate=3e-17, fuv_field=1.7, thermochemistry=True, cooling_courant=1e3, floor_temperature=3.0,
        absolute_tolerance=1e-20, relative_tolerance=1e-8, max_steps=4096,
        krome_user_variables={"user_flux": 0.0}, krome_h2_formation_on_dust=True, krome_data_folder=data_folder,
    )
    config = SimulationConfig(
        dimensionality=1, solver_mode=FINITE_VOLUME, backend=NATIVE_JAX, box_size=1.0,
        num_cells=num_cells, progress_bar=False, chemistry_config=chemistry_config,
    )
    params = SimulationParams(t_end=1.0, gamma=gamma, chemistry_params=chemistry_params)
    registered_variables = get_registered_variables(config)
    hydrogen_nuclei = jnp.logspace(0.0, 3.0, num_cells)
    density = hydrogen_nuclei / chemistry_params.hydrogen_mass_fraction
    start_temperature = jnp.asarray([80.0, 300.0, 2000.0, 9000.0])
    X, Z = chemistry_params.hydrogen_mass_fraction, chemistry_params.metal_mass_fraction
    state = construct_primitive_state(
        config=config, registered_variables=registered_variables, density=density,
        velocity_x=jnp.zeros(num_cells), gas_pressure=get_pressure_from_temperature(density, start_temperature, X, Z),
    )
    fractions = {"H": 0.99, "H+": 1e-3, "He": 0.08, "He+": 1e-5, "H2": 1e-4, "H2+": 1e-9, "C": 1e-5, "C+": 1.3e-4}
    fractions["E"] = fractions["H+"] + fractions["He+"] + fractions["H2+"] + fractions["C+"]
    start = registered_variables.chemistry_species_index
    for index, name in enumerate(species):
        state = state.at[start + index].set(fractions[name] * hydrogen_nuclei)
    config = finalize_config(config, state.shape)
    reacted = update_chemistry(state, registered_variables, config.chemistry_config, params, 1.0)
    final_temperature = np.asarray(get_temperature_from_pressure(reacted[registered_variables.density_index], reacted[registered_variables.pressure_index], X, Z))
    final_species = np.asarray(reacted)[start : start + len(species)]

    network = chemistry_params.network
    ones, zeros = jnp.ones(network.reactions_number), jnp.zeros(network.reactions_number)

    def right_hand_side(y):
        abundances, temperature = y[:-1], jnp.clip(y[-1], 3.0, 1e9)
        return jnp.concatenate([
            network(0.0, abundances, temperature, 3e-17, 1.7, 2.0, ones, zeros),
            jnp.reshape(network.temperature_derivative(abundances, temperature, gamma, 3e-17, 1.7), (1,)),
        ])

    function, jacobian = jax.jit(right_hand_side), jax.jit(jax.jacfwd(right_hand_side))
    initial = np.asarray(state)[start : start + len(species)]
    for cell in range(num_cells):
        y0 = np.concatenate([initial[:, cell], [float(start_temperature[cell])]])
        solution = scipy_integrate.solve_ivp(
            lambda t, y: np.asarray(function(jnp.asarray(y))), (0.0, time_step_seconds), y0, method="Radau",
            jac=lambda t, y: np.asarray(jacobian(jnp.asarray(y))), rtol=1e-8, atol=1e-18,  # tighter, the rate jumps at the limits stall it
        )
        assert solution.success
        assert final_temperature[cell] == pytest.approx(solution.y[-1, -1], rel=1e-4)
        expected = solution.y[:-1, -1]
        significant = expected > 1e-12 * float(hydrogen_nuclei[cell])
        np.testing.assert_allclose(final_species[significant, cell], expected[significant], rtol=1e-4)
    assert np.abs(final_temperature / np.asarray(start_temperature) - 1.0).max() > 0.05  # the temperature did evolve


# ---------------------------------------------------------------------------
# 5. What KROME takes from its build options
# ---------------------------------------------------------------------------

# A network with shielded photo-reactions as KROME's networks write them, a
# cosmic-ray block and reaction modifiers.
SHIELDED_NETWORK = """# test network with shielding
@common: user_crate
@var:fsh=shield_dust(n,Tgas,3.74d0)*krome_fshield(n,Tgas)
@var:fhm=shield_dust(n,Tgas,0.5d0)
@format:idx,R,R,P,P,rate
1,H2,H+,H2+,H,6.4d-10
2,He+,H,He,H+,1.2d-15
@format:idx,R,R,P,rate
3,H+,E,H,H_recombination_on_dust(n,Tgas)
@format:idx,R,P,P,rate
4,H2,H,H,3d-11*fsh
@photo_start
@format:idx,R,P,P,Tmin,Tmax,rate
5,H2+,H+,H,>2.65d0,NONE,@xsecFile=LEIDEN
6,H2,H2+,E,>15.4d0,NONE,@xsecFile=SWRI
7,H,H+,E,>13.6d0,<1.d2,6.3d-18*(energy_eV/13.6d0)**(-3)
@format:idx,R,P,P,rate
8,He,He+,E,auto
@photo_stop
@CR_start
@format:idx,R,P,P,rate
9,H,H+,E,4.6d-1*user_crate
10,He,He+,E,5.d-1*user_crate
11,H2,H2+,E,9.3d-1*user_crate
12,H2,H,H,1d-1*user_crate
@CR_stop
@reactionModifier_start
k(5)=k(5)*fhm
k(1)=k(1)+2d0*k(2)
@reactionModifier_stop
"""
SHIELDED_SPECIES = ("H2", "H+", "H2+", "H", "He+", "He", "E")
MASSES = {  # as KROME counts them: protons, neutrons and electrons
    "H": 1.67262158e-24 + 9.10938188e-28, "H+": 1.67262158e-24, "H2": 2 * (1.67262158e-24 + 9.10938188e-28),
    "H2+": 2 * 1.67262158e-24 + 9.10938188e-28, "He": 2 * (1.67262158e-24 + 9.10938188e-28 + 1.6725e-24),
    "He+": 2 * (1.67262158e-24 + 1.6725e-24) + 9.10938188e-28, "E": 9.10938188e-28,
}


@pytest.fixture(scope="module")
def shielded_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("krome_shielded") / "react_shielded"
    path.write_text(SHIELDED_NETWORK)
    return str(path)


def _column(density, method, n, temperature):
    """KROME's num2col in NumPy; ``n`` maps a species to its density."""
    if method == "DEFAULT":
        return 1.87e21 * (max(density, 1e-40) * 1e-3) ** (2.0 / 3.0)
    mass_density = sum(MASSES[name] * value for name, value in n.items())
    mean_weight = mass_density / sum(n.values()) / 1.67262158e-24
    jeans = np.sqrt(3.14159265359 * 1.380648e-16 * temperature / mass_density / 1.67262158e-24 / 6.674e-8 / mean_weight)
    return 0.5 * max(density, 1e-40) * jeans


def _self_shielding(kind, column, temperature):
    hydrogen_mass = 1.67262158e-24 + 9.10938188e-28
    if kind == "DB96":
        return min(1.0, (column * 1e-14) ** (-0.75))
    if kind == "WG11":
        x, b5 = column * 2e-15, np.sqrt(1.380648e-16 * temperature / hydrogen_mass) * 1e-5
        return 0.965 / (1.0 + x / b5) ** 1.1 + 0.035 / np.sqrt(1.0 + x) * np.exp(-8.5e-4 * np.sqrt(1.0 + x))
    b5 = np.sqrt(1.380648e-16 * temperature / hydrogen_mass + 7.1e5**2) * 1e-5
    omega = 0.013 * (1.0 + (temperature / 2.7e3) ** 1.3) ** (1.0 / 1.3) * np.exp(-((temperature / 3.9e3) ** 14.6))
    if temperature < 3e3:
        alpha, critical = 1.4, 1.3 * (1.0 + (temperature / 6e2) ** 0.8)
    else:  # KROME uses this fit for every temperature from 3000 K on
        alpha, critical = (temperature / 4.5e3) ** (-0.8), (temperature / 4.76e3) ** (-3.8)
    x = column * 1e-14 / critical
    return (1.0 - omega) / (1.0 + x / b5) ** alpha * np.exp(-5e-7 * (1.0 + x)) + omega / np.sqrt(1.0 + x) * np.exp(-8.5e-4 * np.sqrt(1.0 + x))


@pytest.mark.parametrize("method", ["DEFAULT", "JEANS"])
@pytest.mark.parametrize("kind", ["DB96", "WG11", "R14"])
@pytest.mark.parametrize("temperature", [80.0, 5.0e3])
def test_shielding_and_reaction_modifiers(shielded_file, method, kind, temperature):
    network, species = build_krome_network(
        shielded_file, photo_rates={5: 4e-10, 6: 2e-11}, shielding=kind, shield_habing_dust=True,
        column_density_method=method, dust_to_gas_relative=0.5,
    )
    assert species == SHIELDED_SPECIES
    fractions = {"H2": 0.2, "H+": 1e-3, "H2+": 1e-7, "H": 0.6, "He+": 1e-5, "He": 0.08, "E": 1e-3 + 1e-7 + 1e-5}
    n = {name: 300.0 * value for name, value in fractions.items()}
    abundances = jnp.asarray([n[name] for name in species])
    column = lambda density: _column(density, method, n, temperature)
    hydrogen_column = column(n["H"]) + column(n["H+"]) + 2.0 * column(n["H2"])
    dust = lambda gamma: np.exp(-2e-21 * 0.5 * gamma * hydrogen_column)
    self_shielding = _self_shielding(kind, 2.0 * column(n["H2"]), temperature)

    assert float(network.column_density(n["H2"], abundances, temperature)) == pytest.approx(column(n["H2"]), rel=1e-12)
    assert float(network.dust_shielding(abundances, temperature, 3.74)) == pytest.approx(dust(3.74), rel=1e-12)
    assert float(network.h2_self_shielding(abundances, temperature)) == pytest.approx(self_shielding, rel=1e-11)
    assert float(network.habing_field(abundances, temperature, 1.7)) == pytest.approx(1.7 * dust(0.665), rel=1e-12)

    rates = np.asarray(network.get_rates(temperature, 3e-17, 1.7, 0.0, abundances))
    assert rates[3] == pytest.approx(3e-11 * dust(3.74) * self_shielding, rel=1e-11)  # the @var with both factors
    assert rates[4] == pytest.approx(4e-10 * dust(0.5), rel=1e-12)  # k(5)=k(5)*fhm
    assert rates[0] == pytest.approx(6.4e-10 + 2.0 * 1.2e-15, rel=1e-12)  # k(1)=k(1)+2d0*k(2)
    # the recombination on dust sees the attenuated Habing flux
    psi = 1.7 * dust(0.665) * np.sqrt(temperature) / n["E"]
    c = (12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5)
    expected = 1e-14 * c[0] * 0.5 / (1.0 + c[1] * psi ** c[2] * (1.0 + c[3] * temperature ** c[4] * psi ** (-c[5] - c[6] * np.log(temperature))))
    assert rates[2] == pytest.approx(expected, rel=1e-12)
    # and so does the photoelectric heating
    plain, _ = build_krome_network(shielded_file, shielding=kind, column_density_method=method, dust_to_gas_relative=0.5)
    shielded_heating = float(network.heating_terms(abundances, temperature, 3e-17, 1.7)["heating_photoelectric"])
    plain_heating = float(plain.heating_terms(abundances, temperature, 3e-17, 1.7 * dust(0.665))["heating_photoelectric"])
    assert shielded_heating == pytest.approx(plain_heating, rel=1e-12)
    jacobian = jax.jacfwd(lambda y: network(0.0, y, temperature, 3e-17, 1.7, 0.0, jnp.ones(12), jnp.zeros(12)))(abundances)
    assert bool(jnp.isfinite(jacobian).all())


def test_shielding_without_krome_fshield_warns(network_file):
    """As KROME: the self-shielding option has no effect on a network that never calls krome_fshield."""
    with pytest.warns(UserWarning, match="no effect"):
        shielded, species = build_krome_network(network_file, user_variables={"user_flux": 1e-10}, shielding="R14")
    plain, _ = build_krome_network(network_file, user_variables={"user_flux": 1e-10})
    abundances = jnp.ones(len(species))
    np.testing.assert_array_equal(
        np.asarray(shielded.get_rates(300.0, 3e-17, 1.7, 0.0, abundances)), np.asarray(plain.get_rates(300.0, 3e-17, 1.7, 0.0, abundances))
    )


@pytest.mark.parametrize("molecular_hydrogen", [1e-7, 3e-3, 40.0, 2e6, 5e9, 3e10])
def test_cosmic_ray_heating(shielded_file, molecular_hydrogen):
    """Every reaction of the cosmic-ray block deposits its energy: 4.3 eV for H and He, a fit in n(H2) for H2, 20 eV otherwise."""
    network, species = build_krome_network(shielded_file)
    n = {"H2": molecular_hydrogen, "H+": 1e-3, "H2+": 1e-7, "H": 7.0, "He+": 1e-5, "He": 0.9, "E": 1e-3}
    abundances = jnp.asarray([n[name] for name in species])
    crate = 5e-17
    logarithm = np.log10(molecular_hydrogen)
    polynomial = (9.1462198642562, -0.443120145068494, 0.601271617560071, -0.0710284101055109, 0.00242079494592405)
    if molecular_hydrogen > 1e10:
        per_h2 = sum(c * 10.0**i for i, c in enumerate(polynomial))
    elif 1.0366017494884676e-05 <= molecular_hydrogen < 1e5:
        per_h2 = 1.4510849135088 * logarithm + 7.23277032061136
    elif molecular_hydrogen < 1.0366017494884676e-05:
        per_h2 = 0.0
    else:
        per_h2 = sum(c * logarithm**i for i, c in enumerate(polynomial))
    expected = crate * 1.60217662e-12 * (0.46 * n["H"] * 4.3 + 0.5 * n["He"] * 4.3 + 0.93 * n["H2"] * per_h2 + 0.1 * n["H2"] * 20.0)
    heating = float(network.heating_terms(abundances, 100.0, crate, 0.0)["heating_cosmic_rays"])
    assert heating == pytest.approx(expected, rel=1e-12)
    assert [kind for _, _, kind in network.cosmic_ray_terms] == ["atomic", "atomic", "H2", "other"]


# H2 line cooling [erg cm^-3 s^-1] computed with KROME's cooling_H2 (the Fortran that KROME generates, compiled with
# gfortran) for n(H2) = 1 cm^-3 and one collider at 1 cm^-3: temperature, then the colliders H, H+, H2, e, He.
KROME_H2_COOLING = (
    (5, 1.486348534141e-71, 1.510375468310e-68, 0.000000000000e00, 0.000000000000e00, 1.510375468310e-68),
    (30, 1.466946657530e-34, 1.227060872448e-31, 8.203741714869e-168, 3.332560524695e-151, 3.824631229332e-33),
    (75, 8.107857381692e-29, 3.545389756292e-27, 2.943355010578e-28, 1.424010929955e-27, 1.334794305352e-28),
    (205, 1.207362734561e-26, 1.551629558766e-24, 4.547062236309e-26, 2.976216798967e-25, 2.913256471495e-26),
    (800, 8.529170549839e-25, 5.420063963285e-23, 2.683007002907e-24, 8.906087915208e-24, 1.899616781417e-24),
    (3000, 3.649425385659e-23, 4.231932821269e-22, 3.179932181213e-23, 1.638443423764e-22, 2.370106877913e-23),
    (8000, 2.142831181001e-22, 1.081976879921e-21, 1.271675266453e-22, 1.206347137361e-21, 1.012358388147e-22),
    (20000, 1.569818819554e-24, 8.626331550281e-22, 6.279255022571e-24, 1.571676552352e-24, 7.469116664584e-23),
    (100000, 0.000000000000e00, 3.855557752122e-219, 0.000000000000e00, 0.000000000000e00, 3.268836854134e-220),
)


@pytest.mark.parametrize("row", KROME_H2_COOLING, ids=lambda row: f"{row[0]}K")
def test_h2_line_cooling_against_krome(row):
    from astronomix._modules._chemistry._krome_thermochemistry import krome_h2_line_cooling

    temperature, expected = float(row[0]), row[1:]
    colliders = ("hydrogen", "ionized_hydrogen", "molecular_hydrogen", "electrons", "helium")
    for name, reference in zip(colliders, expected):
        densities = {collider: 0.0 for collider in colliders}
        densities["molecular_hydrogen"] = 1.0
        densities[name] += 1.0
        compiled = float(krome_h2_line_cooling(temperature, single_precision_literals=True, **densities))
        decimal = float(krome_h2_line_cooling(temperature, **densities))
        assert compiled == pytest.approx(reference, rel=2e-11, abs=1e-300), name  # the reference has twelve digits
        assert decimal == pytest.approx(reference, rel=3e-6, abs=1e-300), name  # constants read as decimal numbers
    for densities in ((1.0, 1.0, 1e-3, 1e-3, 0.1), (1.0, 0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 1.0)):
        slope = jax.grad(lambda t: krome_h2_line_cooling(t, *densities))(temperature)
        assert bool(jnp.isfinite(slope))


@pytest.mark.parametrize("row", KROME_H2_COOLING, ids=lambda row: f"{row[0]}K")
def test_h2_line_cooling_of_the_chemistry_module_against_krome(row):
    """The H2 line cooling that the chemistry module applies to its carbox networks is KROME's too."""
    from astronomix._modules._chemistry._thermochemistry import _glover_abel_h2_cooling

    temperature, expected = float(row[0]), row[1:]
    for collider, reference in zip((0, 2, 1, 4, 3), expected):  # densities ordered H, H2, H+, He, E
        densities = np.zeros(5)
        densities[1] = 1.0
        densities[collider] += 1.0
        value = float(_glover_abel_h2_cooling(jnp.asarray(densities), temperature, 0, 1, 2, 3, 4))
        assert value == pytest.approx(reference, rel=3e-6, abs=1e-60)  # constants read as decimal numbers


# A network with singly and doubly charged atomic ions, a molecular ion and an anion, written for the bremsstrahlung.
FREE_FREE_NETWORK = """# test network for the bremsstrahlung
@format:idx,R,R,P,P,P,rate
1,H,E,H+,E,E,1d-10
2,HE,E,HE+,E,E,1d-10
3,HE+,E,HE++,E,E,1d-10
@format:idx,R,R,P,P,rate
4,H2,H+,H2+,H,1d-10
@format:idx,R,R,P,rate
5,H,E,H-,1d-16
"""
FREE_FREE_DENSITIES = {"H": 0.3, "E": 1.7, "H+": 1.0, "He": 0.01, "He+": 0.02, "He++": 0.08, "H2": 0.1, "H2+": 0.5, "H-": 0.25}
# Bremsstrahlung [erg cm^-3 s^-1] of these densities computed with KROME's cooling_ff (its source, with the sum over the
# ions that KROME's generator writes for this network: n(H+) + n(He+) + 4 n(He++); compiled with gfortran):
# temperature, cooling.
KROME_FREE_FREE_COOLING = (
    (2.73, 8.017051006592636e-27),
    (1.0e2, 4.852140000000000e-26),
    (1.0e4, 4.852140000000000e-25),
    (1.0e6, 4.852140000000000e-24),
    (1.0e8, 4.852140000000000e-23),
    (1.0e9, 1.534381392600940e-22),
)


@pytest.mark.parametrize("row", KROME_FREE_FREE_COOLING, ids=lambda row: f"{row[0]:g}K")
def test_free_free_cooling_against_krome(tmp_path, row):
    """The bremsstrahlung (KROME's -cooling=FF): the atomic ions only, each with its charge squared."""
    temperature, expected = row
    path = _write(tmp_path, FREE_FREE_NETWORK)
    network, species = build_krome_network(path, free_free_cooling=True)
    weights = {"H+": 1.0, "He+": 1.0, "He++": 4.0}  # not the molecular ion H2+, not the anion H-, not the electrons
    assert network.free_free_charges == tuple(weights.get(name, 0.0) for name in species)
    abundances = jnp.asarray([FREE_FREE_DENSITIES[name] for name in species])
    cooling = float(network.cooling_terms(abundances, temperature)["cooling_free_free"])
    assert cooling == pytest.approx(expected, rel=1e-13)

    # Off by default: the network then has the terms, and so the right-hand side, that it had without the option.
    plain, _ = build_krome_network(path)
    assert plain.free_free_charges == () and "cooling_free_free" not in plain.cooling_terms(abundances, temperature)
    # In the temperature equation the term is part of the cooling, also of the cooling at the floor temperature
    # (2.73 K, the first row of the table) that KROME subtracts.
    with_term = float(network.temperature_derivative(abundances, temperature, 5.0 / 3.0, 3e-17, 1.7))
    without = float(plain.temperature_derivative(abundances, temperature, 5.0 / 3.0, 3e-17, 1.7))
    net_cooling = expected - KROME_FREE_FREE_COOLING[0][1]
    change = (5.0 / 3.0 - 1.0) * net_cooling / (1.380648e-16 * sum(FREE_FREE_DENSITIES.values()))
    assert without - with_term == pytest.approx(change, rel=1e-9, abs=1e-20)


# KROME's database of automatic reactions and two cross section files in the formats of the Leiden and SwRI
# databases, written for this test (wavelengths in nm and in Angstrom, cross sections in cm^2).
PHOTO_DATABASE = """###########
@photorates:yes

#test entry in the format of KROME's photoioniziation.dat
@type: photoion
@reacts: He
@prods: He+, E
@limits: 2.459d+01, 5.000d+04
@rate: sigma_v96(energy_ev, 1.361d+01, 9.492d+02, 1.469d+00, 3.188d+00, 2.039d+00, 4.434d-01, 2.136d+00)
"""
LEIDEN_FILE = "# wavelength photoabsorption photodissociation photoionisation\n" + "".join(
    f"  {wavelength:.4f}  {2e-18 + 1e-20 * wavelength:.5e}  {1e-18 * np.exp(-((wavelength - 150.0) / 60.0) ** 2):.5e}  0.00000e+00\n"
    for wavelength in np.linspace(90.0, 300.0, 43)
)
SWRI_FILE = (
    "0          References for Cross Section of H2\nLambda = 1.-900. A\n   a reference\n"
    " Lambda  Total   H/H(2)   H/H      H2+      H+H     \n"
    + "".join(
        f"  {wavelength:6.1f} {4e-18:.2E} {1e-19:.2E} {2e-19:.2E} {3e-18 * wavelength / 800.0:.2E} {5e-19:.2E}\n"
        for wavelength in np.linspace(100.0, 800.0, 36)
    )
    + "   805.0 0.00E+00 0.00E+00 0.00E+00 0.00E+00 0.00E+00\n"
)


@pytest.fixture(scope="module")
def photo_data(tmp_path_factory):
    folder = tmp_path_factory.mktemp("krome_photo_data")
    _write_cooling_data(folder)
    (folder / "database" / "leiden_xsecs").mkdir(parents=True)
    (folder / "database" / "swri_xsecs").mkdir()
    (folder / "database" / "photoioniziation.dat").write_text(PHOTO_DATABASE)
    (folder / "database" / "leiden_xsecs" / "H2+__H_H+.dat").write_text(LEIDEN_FILE)
    (folder / "database" / "swri_xsecs" / "H2.dat").write_text(SWRI_FILE)
    return str(folder)


def _table_average(left, right, energy, value):
    """KROME's average of a regular table over a bin: the intervals that start inside the bin, each times its part in it."""
    step, total = energy[1] - energy[0], 0.0
    for index, cross_section in enumerate(value):
        start = index * step + energy[0]  # KROME rebuilds the energies from the first one and the step
        if left <= start <= right:
            total += cross_section * (min(start + step, right) - start)
    return total / (right - left)


def test_photo_bins(shielded_file, photo_data):
    from astronomix._modules._chemistry import _krome_photo as photo

    # the spectra of KROME on its bins
    left, right = photo.linear_bins(5.0, 45.0, 8)
    np.testing.assert_allclose(left, 5.0 + 5.0 * np.arange(8)) and np.testing.assert_allclose(right, left + 5.0)
    low, high = photo.logarithmic_bins(1.0, 100.0, 4)
    np.testing.assert_allclose(low, [1.0, 10**0.5, 10.0, 10**1.5]) and np.testing.assert_allclose(high[-1], 100.0)
    planck, boltzmann, light = 4.135667516e-15, 8.617332478e-5, 2.99792458e10
    energy = np.array([3.0, 7.0, 12.0, 20.0])
    np.testing.assert_allclose(
        photo.draine_intensity(energy), np.where((energy > 5) & (energy < 13.6), (1.658e6 * energy - 2.152e5 * energy**2 + 6.919e3 * energy**3) * energy * planck, 0.0)
    )
    np.testing.assert_allclose(photo.black_body_intensity(energy, 3e4), 2 * energy**3 / planck**2 / light**2 / (np.exp(energy / boltzmann / 3e4) - 1))
    np.testing.assert_allclose(photo.j21_intensity(energy), 6.2415e-10 * 13.6 / energy)
    spectrum = photo.photo_spectrum("black_body", 5.0, 45.0, 8, temperature=5e4, habing=3.0)
    middle, width, intensity = 0.5 * (left + right), right - left, spectrum["intensity"]
    in_band = (middle >= 6.0) & (middle <= 13.6)
    habing = np.sum((intensity * width)[in_band]) * 4 * 3.14159265359 / 1.6e-3 / planck * 1.60217646e-12
    assert habing == pytest.approx(3.0, rel=1e-12) and photo.habing_field(left, right, intensity) == pytest.approx(3.0, rel=1e-12)

    network, species = build_krome_network(shielded_file, photo_bins=spectrum, data_folder=photo_data, dust_to_gas_relative=0.5)
    assert network.has_photo_bins and float(network.bins_habing_field) == pytest.approx(3.0, rel=1e-12)

    # cross sections per bin, written again here: tables averaged over the bin, expressions at its centre
    nm = np.linspace(90.0, 300.0, 43)[::-1]
    leiden_energy = light * planck / (nm * 1e-7)
    leiden_value = np.array([float(f"{1e-18 * np.exp(-((w - 150.0) / 60.0) ** 2):.5e}") for w in nm])
    grid = leiden_energy.min() + np.arange(5000) * (min(leiden_energy.max(), 30.0) - leiden_energy.min()) / 4999
    leiden = (np.array([float("%.12g" % x) for x in grid]), np.array([float("%.12g" % x) for x in np.interp(grid, leiden_energy, leiden_value)]))
    angstrom = np.append(np.linspace(100.0, 800.0, 36), 805.0)[::-1]
    swri_energy = light * planck / (angstrom * 1e-8)
    swri_value = np.array([0.0] + [float(f"{3e-18 * w / 800.0:.2E}") for w in np.linspace(100.0, 800.0, 36)[::-1]])
    lowest = swri_energy[1]  # the first energy with a cross section
    grid = lowest + np.arange(5000) * (30.0 - lowest) / 4999
    swri = (np.array([float("%.12g" % x) for x in grid]), np.array([float("%.12g" % x) for x in np.interp(grid, swri_energy, swri_value)]))

    def verner(e):
        e0, s0, ya, p, yw, y0, y1 = 13.61, 949.2, 1.469, 3.188, 2.039, 0.4434, 2.136
        x = e / e0 - y0
        y = np.sqrt(x**2 + y1**2)
        return 1e-18 * s0 * ((x - 1) ** 2 + yw**2) * y ** (0.5 * p - 5.5) * (1 + np.sqrt(y / ya)) ** (-p)

    thresholds = [2.65, 15.4, 13.6, 24.59]
    upper = [1e8, 1e8, 1e2, 5e4]
    sections = [
        lambda j: _table_average(left[j], right[j], *leiden), lambda j: _table_average(left[j], right[j], *swri),
        lambda j: 6.3e-18 * (middle[j] / 13.6) ** (-3), lambda j: verner(middle[j]),
    ]
    rates, heats = np.zeros(4), np.zeros(4)
    for i in range(4):
        for j in range(8):
            if thresholds[i] < middle[j] < upper[i]:
                term = sections[i](j) * intensity[j] / middle[j] * width[j]
                rates[i] += term
                heats[i] += term * (middle[j] - thresholds[i])
    rates, heats = 4 * 3.14159265359 * rates / planck, 4 * 3.14159265359 * heats / planck * 1.60217646e-12
    assert (rates > 0).all() and (heats > 0).all()
    np.testing.assert_allclose(np.asarray(network.photo_rates), rates, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(network.photo_heats), heats, rtol=1e-10)

    # the rates enter the coefficients (with the modifier of reaction 5), the heating sums over the reactants,
    # and the Habing flux of the spectrum replaces the fuv_field argument
    n = {"H2": 5.0, "H+": 1e-3, "H2+": 1e-6, "H": 7.0, "He+": 1e-5, "He": 0.9, "E": 1e-3}
    abundances = jnp.asarray([n[name] for name in species])
    coefficients = np.asarray(network.get_rates(300.0, 0.0, 99.0, 0.0, abundances))
    column = sum(a * 1.87e21 * (n[name] * 1e-3) ** (2.0 / 3.0) for name, a in (("H", 1.0), ("H+", 1.0), ("H2", 2.0)))
    np.testing.assert_allclose(coefficients[4:8], rates * np.array([np.exp(-2e-21 * 0.5 * 0.5 * column), 1.0, 1.0, 1.0]), rtol=1e-10)
    terms = network.heating_terms(abundances, 300.0, 0.0, 99.0)
    assert float(terms["heating_photo"]) == pytest.approx(heats @ np.array([n["H2+"], n["H2"], n["H"], n["He"]]), rel=1e-10)
    assert float(network.habing_field(abundances, 300.0, 99.0)) == pytest.approx(3.0, rel=1e-12)

    # through the set-up function: the Habing flux in use is reported in the parameters
    from astronomix.setup_helpers import build_chemistry_from_network_file

    _, chemistry_params, _ = build_chemistry_from_network_file(
        shielded_file, "krome", number_density_unit_cgs=1.0, temperature_unit_kelvin=1.0, time_unit_seconds=1.0,
        fuv_field=99.0, krome_photo_bins=spectrum, krome_data_folder=photo_data, krome_shield_habing_dust=True,
        krome_shielding="R14", krome_column_density_method="JEANS",
    )
    assert float(chemistry_params.fuv_field) == pytest.approx(3.0, rel=1e-12)
    assert chemistry_params.network.shield_habing_dust and chemistry_params.network.shielding == "R14"
    assert chemistry_params.network.column_density_method == "JEANS"


# ---------------------------------------------------------------------------
# 6. A full KROME network file, when one is given
# ---------------------------------------------------------------------------


def test_full_krome_network_file():
    path = os.environ.get("ASTRONOMIX_TEST_KROME_NETWORK")
    if not path or not os.path.exists(path):
        pytest.skip("no KROME network file (set ASTRONOMIX_TEST_KROME_NETWORK)")
    warnings.simplefilter("ignore")  # a file with a commented @format line is read as KROME reads it
    parsed = parse_krome_network(path)  # checks the conservation of every reaction
    users = {name: 0.0 for name in parsed["commons"] if name != "user_crate"}
    network, species = build_krome_network(path, user_variables=users, h2_formation_on_dust="H2" in parsed["species"])
    abundances = jnp.ones(len(species))
    temperatures = jnp.logspace(np.log10(2.73), 9.0, 200)
    rates = jax.vmap(lambda t: network.get_rates(t, 3e-17, 1.7, 0.0, abundances))(temperatures)
    slopes = jax.vmap(jax.jacfwd(lambda t: network.get_rates(t, 3e-17, 1.7, 0.0, abundances)))(temperatures)
    assert bool(jnp.isfinite(rates).all()) and bool(jnp.isfinite(slopes).all())
    assert int((rates < 0).sum()) == 0

    data = os.environ.get("ASTRONOMIX_TEST_KROME_DATA")
    if data and os.path.isdir(data):
        network, species = build_krome_network(path, user_variables=users, h2_formation_on_dust="H2" in parsed["species"], data_folder=data)
        state = jnp.asarray(10 ** np.random.default_rng(5).uniform(-6, 1, len(species)))
        derivative = lambda t: network.temperature_derivative(state, t, 5.0 / 3.0, 3e-17, 1.7)
        values, slopes = jax.vmap(derivative)(temperatures), jax.vmap(jax.grad(derivative))(temperatures)
        assert bool(jnp.isfinite(values).all()) and bool(jnp.isfinite(slopes).all())
        by_species = jax.vmap(jax.jacfwd(lambda y, t: network.temperature_derivative(y, t, 5.0 / 3.0, 3e-17, 1.7)), in_axes=(None, 0))(state, temperatures)
        assert bool(jnp.isfinite(by_species).all())

        # with a binned spectrum, the dust shielding of the Habing flux and the H2 self-shielding
        from astronomix._modules._chemistry._krome_photo import photo_spectrum

        if not os.path.isdir(os.path.join(data, "database")):
            return
        network, species = build_krome_network(
            path, user_variables=users, h2_formation_on_dust="H2" in parsed["species"], data_folder=data,
            photo_bins=photo_spectrum("draine", 5.0, 13.6, 10), shielding="R14", shield_habing_dust=True,
        )
        assert float(network.bins_habing_field) == pytest.approx(1.7167, rel=1e-3)  # the Draine field in Habing units
        assert float(jnp.max(network.photo_rates)) > 0.0
        ones, zeros = jnp.ones(network.reactions_number), jnp.zeros(network.reactions_number)
        for method in (jax.vmap(lambda t: network(0.0, state, t, 3e-17, 0.0, 0.0, ones, zeros)), jax.vmap(derivative)):
            assert bool(jnp.isfinite(method(temperatures)).all())
        jacobian = jax.vmap(jax.jacfwd(lambda y, t: network(0.0, y, t, 3e-17, 0.0, 0.0, ones, zeros)), in_axes=(None, 0))(state, temperatures)
        assert bool(jnp.isfinite(jacobian).all())
