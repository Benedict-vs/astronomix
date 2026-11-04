"""
Distance-diluted FUV heating and heating in the sub-cycled cooling (CPU, seconds).

Guards ``CoolingConfig.fuv_point_source_heating`` / ``FUVSourceParams`` and the
``with_heating`` path of ``update_temperature_subcycled``, on the KI02 + GF12
curve (``ki02_gf12_cooling``):

1. ``fuv_heating_factor`` equals the direct sum
   sum_k w_k A(t) / (4 pi (d_k^2 + r_s^2)) + background, with A(t) linearly
   interpolated in time.
2. The sub-cycled update with heating settles on the thermal equilibrium
   (Gamma = n_H Lambda) from above and from below, without overshooting it,
   for heat ratios 1 - 1000 and n_H = 0.1 - 100. The equilibria are the stable
   roots of the kernel's own net rate, so this tests the integrator, not the
   curve (that is test_ki02_gf12_cooling.py).
3. ``update_pressure_by_cooling`` with point-source heating in 3D: every cell
   ends at the equilibrium of its local J / J_0.

Runs in float32, the precision of the existing cooling tests.
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# testing
import pytest

# units and constants
import astropy.units as u

# astronomix constants
from astronomix._modules._cooling.cooling_options import (
    COOLING_FLOOR_CLIP,
    PIECEWISE_POWER_LAW,
    SUBCYCLED_EXPLICIT_COOLING,
)
from astronomix._modules._cooling._cooling_tables import (
    _HYDROGEN_MASS_CGS,
    TIGRESS_MU_H,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
)
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    FUVSourceParams,
)
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.units.unit_helpers import CodeUnits

# astronomix functions
from astronomix import get_registered_variables
from astronomix._modules._cooling._cooling import (
    dtemperature_dt,
    fuv_heating_factor,
    get_effective_molecular_weights,
    get_pressure_from_temperature,
    get_temperature_from_pressure,
    update_pressure_by_cooling,
    update_temperature_subcycled,
)
from astronomix._modules._cooling._cooling_tables import ki02_gf12_cooling

GAMMA = 5.0 / 3.0
X, Z = 0.76, 0.02
CODE_UNITS = CodeUnits(1 * u.pc, 1 * u.M_sun, 1 * u.km / u.s)
MYR = (1 * u.Myr).to(CODE_UNITS.code_time).value

SETUP = ki02_gf12_cooling(CODE_UNITS, X, Z)
MU, _, _ = get_effective_molecular_weights(X, Z)
RHO_U = (1 * CODE_UNITS.code_density).to(u.g / u.cm**3).value

CURVE_CONFIG = CoolingCurveConfig(cooling_curve_type=PIECEWISE_POWER_LAW)


def _as_float32(tree):
    """Cast the array leaves of a pytree to float32 (see test_ki02_gf12_cooling)."""
    return jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.float32) if isinstance(leaf, jax.Array) else leaf,
        tree,
    )


HEATING_T_TABLE = _as_float32(SETUP.heating_log10_T_table)
HEATING_F_TABLE = _as_float32(SETUP.heating_factor_table)
CURVE_PARAMS = _as_float32(SETUP.cooling_curve_params)


def _rho_code(n_H):
    """Code density of gas with hydrogen number density n_H [cm^-3]."""
    return n_H * TIGRESS_MU_H * _HYDROGEN_MASS_CGS / RHO_U


def _net_rate(rho, T, heat_ratio):
    """The kernel's net dT~/dt (heating and cooling) at code (rho, T~)."""
    return dtemperature_dt(
        jnp.asarray(rho, jnp.float32),
        jnp.asarray(T, jnp.float32),
        X,
        Z,
        GAMMA,
        CURVE_CONFIG,
        CURVE_PARAMS,
        heating_rate=jnp.asarray(SETUP.heating_rate * heat_ratio, jnp.float32),
        heating_log10_T_table=HEATING_T_TABLE,
        heating_factor_table=HEATING_F_TABLE,
    )


def _stable_equilibria(rho, heat_ratio):
    """Stable roots of the net rate (sign + -> - with rising T~), ascending."""
    T = np.logspace(np.log10(SETUP.floor_temperature) + 0.05, np.log10(SETUP.floor_temperature) + 7, 40001)
    rate = np.asarray(_net_rate(np.full_like(T, rho), T, heat_ratio), np.float64)
    i = np.nonzero((rate[:-1] > 0) & (rate[1:] <= 0))[0]
    # linear interpolation of the root inside each bracketing grid interval
    return T[i] - rate[i] * (T[i + 1] - T[i]) / (rate[i + 1] - rate[i])


def test_fuv_factor_is_the_direct_sum():
    """Two sources, A(t) interpolated between two knots, plus the background."""
    x = jnp.linspace(-3.0, 3.0, 6)
    y = jnp.linspace(-2.0, 2.0, 5)
    z = jnp.linspace(-1.0, 1.0, 4)
    params = FUVSourceParams(
        source_positions=jnp.array([[0.0, 0.0, 0.0], [1.5, -1.0, 0.5]]),
        source_weights=jnp.array([0.75, 0.25]),
        softening_length=0.3,
        luminosity_times=jnp.array([0.0, 10.0]),
        luminosity_values=jnp.array([100.0, 300.0]),
        background=0.0024,
    )
    factor = fuv_heating_factor(params, x, y, z, 2.5)

    X3, Y3, Z3 = np.meshgrid(np.asarray(x), np.asarray(y), np.asarray(z), indexing="ij")
    area = 150.0  # A(2.5) between 100 and 300 at t = 0 and 10
    expected = 0.0024
    for (px, py, pz), w in zip(np.asarray(params.source_positions), [0.75, 0.25]):
        d2 = (X3 - px) ** 2 + (Y3 - py) ** 2 + (Z3 - pz) ** 2
        expected = expected + w * area / (4 * np.pi * (d2 + 0.09))
    np.testing.assert_allclose(np.asarray(factor), expected, rtol=1e-5)


@pytest.mark.parametrize("heat_ratio", [1.0, 100.0, 1000.0])
def test_subcycled_heating_settles_on_the_equilibrium(heat_ratio):
    """From 30 K and 1e6 K to the nearest stable root, monotonically."""
    n_H = np.array([0.1, 1.0, 10.0, 100.0])
    rho = _rho_code(n_H)
    T_code_per_K = SETUP.T1_to_code / 1.27  # rough T1 ~ T / 1.27 in neutral gas
    starts = {"below": 30.0 * T_code_per_K, "above": 1e6 * SETUP.T1_to_code / 0.6}

    for side, T_start in starts.items():
        rho_cells = jnp.asarray(rho, jnp.float32)
        T = jnp.full(n_H.shape, T_start, jnp.float32)
        history = [np.asarray(T)]
        step = jax.jit(
            lambda T: update_temperature_subcycled(
                rho_cells,
                T,
                0.05 * MYR,
                X,
                Z,
                GAMMA,
                CURVE_CONFIG,
                CURVE_PARAMS,
                SETUP.floor_temperature,
                max_subcycles=32,
                with_heating=True,
                heating_rate=jnp.asarray(SETUP.heating_rate * heat_ratio, jnp.float32),
                heating_log10_T_table=HEATING_T_TABLE,
                heating_factor_table=HEATING_F_TABLE,
            )
        )
        for _ in range(600):  # 30 Myr, many cooling times even at n_H = 0.1
            T = step(T)
            history.append(np.asarray(T))
        history = np.array(history)

        for i in range(n_H.size):
            roots = _stable_equilibria(rho[i], heat_ratio)
            target = roots[-1] if side == "above" else roots[0]
            np.testing.assert_allclose(history[-1, i], target, rtol=0.01, err_msg=f"{side}, n_H={n_H[i]}")
            # the exact solution never crosses the equilibrium
            if side == "above":
                assert history[:, i].min() >= target * (1 - 0.01)
            else:
                assert history[:, i].max() <= target * (1 + 0.01)


def test_point_source_heating_in_3d():
    """Uniform n_H = 10 box around one source: each cell at its local equilibrium."""
    n = 8
    edges = jnp.linspace(-40.0, 40.0, n + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])  # pc
    fuv = FUVSourceParams(
        source_positions=jnp.zeros((1, 3)),
        source_weights=jnp.ones((1,)),
        softening_length=5.0,
        luminosity_times=jnp.zeros((1,)),
        # J/J0 ~ 250 next to the source, ~7 in the corners
        luminosity_values=jnp.array([4 * np.pi * 1e3 * 25.0]),
        background=0.0024,
    )
    cooling_config = CoolingConfig(
        cooling=True,
        cooling_method=SUBCYCLED_EXPLICIT_COOLING,
        floor_mode=COOLING_FLOOR_CLIP,
        max_subcycles=32,
        cooling_curve_config=CURVE_CONFIG,
        fuv_point_source_heating=True,
    )
    cooling_params = CoolingParams(
        hydrogen_mass_fraction=X,
        metal_mass_fraction=Z,
        floor_temperature=SETUP.floor_temperature,
        clamp_to_floor=True,
        heating_rate=SETUP.heating_rate,
        heating_log10_T_table=HEATING_T_TABLE,
        heating_factor_table=HEATING_F_TABLE,
        fuv_source_params=fuv,
        cooling_curve_params=CURVE_PARAMS,
    )
    config = SimulationConfig(dimensionality=3, cooling_config=cooling_config)
    registered_variables = get_registered_variables(config)
    params = SimulationParams(gamma=GAMMA, cooling_params=cooling_params)
    helper_data = HelperData(cell_centers_x=centres, cell_centers_y=centres, cell_centers_z=centres)

    rho = _rho_code(10.0)
    state = jnp.zeros((registered_variables.num_vars, n, n, n), jnp.float32)
    state = state.at[registered_variables.density_index].set(rho)
    state = state.at[registered_variables.pressure_index].set(
        get_pressure_from_temperature(jnp.full((n, n, n), rho), jnp.full((n, n, n), 5.0 * SETUP.T1_to_code), X, Z)
    )

    step = jax.jit(
        lambda s: update_pressure_by_cooling(
            s, registered_variables, cooling_config, params, 0.05 * MYR, helper_data=helper_data, time=0.0
        )
    )
    for _ in range(400):  # 20 Myr
        state = step(state)

    T = np.asarray(
        get_temperature_from_pressure(
            state[registered_variables.density_index], state[registered_variables.pressure_index], X, Z
        )
    )
    J = np.asarray(fuv_heating_factor(fuv, centres, centres, centres, 0.0))
    assert J.max() > 200 and J.min() < 10  # the test spans a real range of fields
    for idx in [(3, 3, 3), (0, 0, 0), (0, 3, 5), (7, 1, 4)]:
        roots = _stable_equilibria(rho, float(J[idx]))
        # started cold, so the coldest stable root is the target
        np.testing.assert_allclose(T[idx], roots[0], rtol=0.01, err_msg=str(idx))
    # a stronger field never gives a colder cell
    order = np.argsort(J.ravel())
    assert np.all(np.diff(T.ravel()[order]) >= -1e-3 * T.max())


def test_code_temperature_from_physical_pressure():
    """T1_to_code and ki02_gf12_temperature against P = (n_H + n_He + n_e) k_B T.

    Builds the pressure of gas at a physical T from the TIGRESS particle count
    (1.1 + x_e(T)) n_H, independently of ``T1_to_code``, and recovers T through
    the EOS and the post-processing inversion (review of PR #8: the rate tests
    alone do not catch a wrong ``T1_to_code``).
    """
    from astronomix._modules._cooling._cooling_tables import (
        TIGRESS_X_PART,
        _KI02GF12Model,
        ki02_gf12_temperature,
    )

    T_phys = np.logspace(1.5, 8.0, 27)
    n_H = 3.0
    x_e, _ = _KI02GF12Model().electron_fraction(T_phys)
    P_cgs = (TIGRESS_X_PART + x_e) * n_H * 1.380649e-16 * T_phys
    P_code = P_cgs / (1 * CODE_UNITS.code_pressure).to(u.erg / u.cm**3).value
    rho_code = np.full_like(T_phys, _rho_code(n_H))

    T_code = np.asarray(
        get_temperature_from_pressure(jnp.asarray(rho_code), jnp.asarray(P_code), X, Z), np.float64
    )
    T_back = ki02_gf12_temperature(T_code / SETUP.T1_to_code)
    np.testing.assert_allclose(T_back, T_phys, rtol=2e-3)


def test_slab_attenuation_factor():
    """(1 - E_2(tau/2)) / tau of each column's kappa Sigma, capped at 1."""
    from scipy.special import expn

    from astronomix._modules._cooling._cooling import fuv_attenuation_factor
    from astronomix._modules._cooling.cooling_options import FUV_ATTENUATION_SLAB

    rho = jnp.ones((3, 1, 8)) * jnp.array([1e-4, 1.0, 10.0])[:, None, None]
    params = FUVSourceParams(dust_opacity=0.5)
    factor = np.asarray(fuv_attenuation_factor(rho, rho, params, FUV_ATTENUATION_SLAB, 0.25))
    tau = 0.5 * np.array([1e-4, 1.0, 10.0]) * 8 * 0.25
    expected = np.minimum((1 - expn(2, tau / 2)) / tau, 1.0)
    np.testing.assert_allclose(factor[:, 0, :], np.repeat(expected[:, None], 8, 1), rtol=1e-5)
    assert factor[0].max() == 1.0 and factor[2].max() < 0.2  # thin capped, thick dark


def test_local_jeans_attenuation_factor():
    """exp(-kappa rho L_J) with c^2 = min(P / rho, cap)."""
    from astronomix._modules._cooling._cooling import fuv_attenuation_factor
    from astronomix._modules._cooling.cooling_options import FUV_ATTENUATION_LOCAL_JEANS

    rho = jnp.array([0.5, 2.0, 2.0])
    pressure = jnp.array([0.5, 2.0 * 0.01, 2.0 * 10.0])  # c^2 = 1, 0.01 (below cap), 10
    params = FUVSourceParams(dust_opacity=0.3, jeans_sound_speed_squared_cap=1.0,
                             gravitational_constant=2.0)
    factor = np.asarray(fuv_attenuation_factor(rho, pressure, params, FUV_ATTENUATION_LOCAL_JEANS, 1.0))
    c2 = np.array([1.0, 0.01, 1.0])
    rho_np = np.asarray(rho)
    expected = np.exp(-0.3 * rho_np * np.sqrt(np.pi * c2 / (2.0 * rho_np)))
    np.testing.assert_allclose(factor, expected, rtol=1e-6)


def test_attenuation_leaves_the_background_unattenuated():
    """Opaque gas sees only the metagalactic floor, not zero."""
    from astronomix._modules._cooling._cooling import fuv_attenuation_factor
    from astronomix._modules._cooling.cooling_options import FUV_ATTENUATION_LOCAL_JEANS

    params = FUVSourceParams(dust_opacity=1e6, jeans_sound_speed_squared_cap=1.0,
                             gravitational_constant=1.0, background=0.0024)
    rho = jnp.ones((2, 2, 2))
    f = fuv_attenuation_factor(rho, rho, params, FUV_ATTENUATION_LOCAL_JEANS, 1.0)
    field = (100.0 - params.background) * f + params.background
    np.testing.assert_allclose(np.asarray(field), 0.0024, rtol=1e-6)


def test_line_of_sight_factor_is_bilinear_in_the_table():
    """exp(-tau) with tau bilinear in the (R, z) table around the source."""
    from astronomix._modules._cooling._cooling import fuv_line_of_sight_factor

    R_t = np.arange(6) * 0.5
    z_t = -1.0 + np.arange(9) * 0.25
    table = 0.3 * R_t[:, None] + 0.7 * (z_t[None, :] + 1.0)  # bilinear, so exact
    params = FUVSourceParams(source_positions=jnp.array([[1.0, 1.0, 1.0]]),
                             los_tau_table=jnp.asarray(table), los_r_spacing=0.5,
                             los_z_min=-1.0, los_z_spacing=0.25)
    c = jnp.linspace(0.1, 1.9, 5)
    f = np.asarray(fuv_line_of_sight_factor(params, c, c, c))
    cn = np.asarray(c)
    R = np.sqrt((cn - 1)[:, None, None] ** 2 + (cn - 1)[None, :, None] ** 2)
    tau = 0.3 * R + 0.7 * ((cn - 1)[None, None, :] + 1.0)
    np.testing.assert_allclose(f, np.exp(-tau), rtol=1e-5)


def test_disk_field_is_kennicutt_schmidt_times_slab_factor():
    """coefficient * Sigma^index * min(1, (1 - E_2(tau/2)) / tau), constant along the column."""
    from scipy.special import expn

    from astronomix._modules._cooling._cooling import fuv_disk_field

    rho = jnp.ones((2, 1, 10)) * jnp.array([2.0, 20.0])[:, None, None]
    params = FUVSourceParams(dust_opacity=0.05, disk_sfr_coefficient=0.1, disk_sfr_index=1.4)
    field = np.asarray(fuv_disk_field(rho, params, 0.5))
    sigma = np.array([2.0, 20.0]) * 10 * 0.5
    tau = 0.05 * sigma
    expected = 0.1 * sigma**1.4 * np.minimum((1 - expn(2, tau / 2)) / tau, 1.0)
    np.testing.assert_allclose(field[:, 0, :], np.repeat(expected[:, None], 10, 1), rtol=1e-5)
