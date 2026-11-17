"""
Networks in the KROME format (``astronomix/_modules/_chemistry/_krome_network.py``).

1. The reader on a small network written here in the KROME format: layouts,
   temperature limits, ``@var`` and ``@common``, Fortran constants, the rate
   functions KROME evaluates internally, photo-reactions, and the errors for
   what it does not accept.
2. The right-hand side: conservation of nuclei and charge, and the rate
   coefficients against the same formulas written in NumPy.
3. The chemistry module on that network: ``update_chemistry`` with the vmapped
   Kvaerno5 against SciPy's Radau on the same right-hand side.
4. Optionally a full KROME network file, when ASTRONOMIX_TEST_KROME_NETWORK
   points at one (for instance ``networks/react_galaxy_ism`` of KROME): it is
   read, its reactions conserve nuclei and charge, and its rates are finite.

The file runs on the CPU unless a platform is chosen in the environment.
"""

import os

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
13,H,H+,E,4.6d-1*user_crate
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


def test_reader_refuses_what_it_does_not_read(tmp_path):
    with pytest.raises(ValueError, match="fields"):  # the photo-reaction under a two-body layout
        parse_krome_network(_write(tmp_path, "@format:idx,R,R,P,P,Tmin,Tmax,rate\n1,H,H+,E,auto\n"))
    with pytest.raises(ValueError, match="does not conserve"):
        parse_krome_network(_write(tmp_path, "@format:idx,R,R,P,rate\n1,H+,E,H2,1d-10\n"))
    with pytest.raises(NotImplementedError, match="@ghost"):
        parse_krome_network(_write(tmp_path, "@ghost:He\n"))
    with pytest.raises(ValueError, match="user_flux"):  # a user variable the rates need
        build_krome_network(_write(tmp_path, NETWORK))
    with pytest.raises(ValueError, match="not photo-reactions"):
        build_krome_network(_write(tmp_path, NETWORK), photo_rates={1: 1e-10}, user_variables={"user_flux": 0.0})


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
# 4. A full KROME network file, when one is given
# ---------------------------------------------------------------------------


def test_full_krome_network_file():
    path = os.environ.get("ASTRONOMIX_TEST_KROME_NETWORK")
    if not path or not os.path.exists(path):
        pytest.skip("no KROME network file (set ASTRONOMIX_TEST_KROME_NETWORK)")
    parsed = parse_krome_network(path)  # checks the conservation of every reaction
    users = {name: 0.0 for name in parsed["commons"] if name != "user_crate"}
    network, species = build_krome_network(path, user_variables=users, h2_formation_on_dust="H2" in parsed["species"])
    abundances = jnp.ones(len(species))
    temperatures = jnp.logspace(np.log10(2.73), 9.0, 200)
    rates = jax.vmap(lambda t: network.get_rates(t, 3e-17, 1.7, 0.0, abundances))(temperatures)
    slopes = jax.vmap(jax.jacfwd(lambda t: network.get_rates(t, 3e-17, 1.7, 0.0, abundances)))(temperatures)
    assert bool(jnp.isfinite(rates).all()) and bool(jnp.isfinite(slopes).all())
    assert int((rates < 0).sum()) == 0
