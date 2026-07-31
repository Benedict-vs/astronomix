"""
Radiative cooling: the CGOLS piecewise-parabolic CIE curve and its integrator.

Verifies the ``CIE_PARABOLIC`` cooling curve, the ``TOTAL_NUMBER_DENSITY``
convention, the sub-cycled explicit integrator, the cooling-time step limit and
the differentiability of the whole chain against independent references:

  1. curve identity vs the analytic Eq. A4 rendering in ``cgols.py``
  2. the volumetric prefactor (n^2 vs n_e n_H, a factor ~4.30)
  3. a single cooling cell vs a ``scipy.integrate.solve_ivp`` reference in cgs
  4. sub-cycling vs a single explicit step at dt = 5 t_cool
  5. the cooling-time timestep limit
  6. ``jax.grad`` through the sub-cycle loop
  7. a small 3D smoke run with cooling active
  8. the mixing-layer no-op regression (checksum of a short FD run)

Run it directly (no pytest):

    python pytests/cooling/cie_cooling.py                # all checks
    python pytests/cooling/cie_cooling.py --checks 1,2   # a subset
    python pytests/cooling/cie_cooling.py --cpu          # force the CPU backend
    python pytests/cooling/cie_cooling.py --cgols-smoke  # + the cgols subprocess run

Checks 1-6 run in double precision so the tolerances mean something; check 1 is
additionally repeated in float32, the precision the production runs use.
"""

# general
import argparse
import os
import subprocess
import sys
from pathlib import Path

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--checks", default="", help="comma-separated subset of 1..8")
_parser.add_argument("--cpu", action="store_true", help="run on the CPU backend")
_parser.add_argument(
    "--cgols-smoke",
    action="store_true",
    help="additionally run the cgols.py B-series subprocess smoke test (check 7b)",
)
ARGS = _parser.parse_args()

# ==== GPU selection ====
if ARGS.cpu:
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np
from scipy.integrate import solve_ivp

# plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# units and constants
import astropy.constants as c
from astropy import units as u

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    NATIVE_JAX,
    OPEN_BOUNDARY,
    PALLAS,
    PERIODIC_BOUNDARY,
    RK4_LSRK,
)
from astronomix._modules._cooling.cooling_options import (
    CIE_PARABOLIC,
    COOLING_FLOOR_CLIP,
    COOLING_IN_RK_STAGES,
    COOLING_OPERATOR_SPLIT,
    ELECTRON_HYDROGEN_DENSITY,
    EXPLICIT_COOLING,
    IMPLICIT_COOLING,
    SIMPLE_MIXING_LAYER_COOLING,
    SUBCYCLED_EXPLICIT_COOLING,
    TOTAL_NUMBER_DENSITY,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SimulationParams,
)
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    PositivityConfig,
    StaticFloatVector,
    StaticIntVector,
)
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    MixingCoolingParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)
from astronomix.units.unit_helpers import CodeUnits
from astronomix._finite_difference._timestep_estimation._timestep_estimator import (
    _cfl_time_step_fd_hydro,
)
from astronomix._modules._cooling._cooling import (
    _cooling_rate,
    _volumetric_cooling_prefactor,
    cooling_time,
    cooling_time_step_limit,
    dtemperature_dt,
    get_effective_molecular_weights,
    get_pressure_from_temperature,
    get_temperature_from_pressure,
    update_pressure_by_cooling,
)
from astronomix._modules._cooling._cooling_tables import cie_parabolic_cooling

def import_cgols_reference():
    """Import ``cooling_lambda_cgs`` from the cgols run script.

    Two things have to be neutralised first: importing ``cgols`` runs its
    module-level ``autocvd(num_gpus=...)`` (which would block waiting for a
    *second* free GPU, or for any GPU at all under ``--cpu``), and its
    module-level ``jax_enable_x64 -> False``, which would silently drop this
    script out of double precision. Stub the former, restore the latter.
    """
    import types

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "self_gravity" / "cgols")
    )

    saved_autocvd = sys.modules.get("autocvd")
    stub = types.ModuleType("autocvd")
    stub.autocvd = lambda *args, **kwargs: None
    sys.modules["autocvd"] = stub
    try:
        from cgols import cooling_lambda_cgs
    finally:
        if saved_autocvd is not None:
            sys.modules["autocvd"] = saved_autocvd
        else:
            sys.modules.pop("autocvd", None)
        jax.config.update("jax_enable_x64", True)

    return cooling_lambda_cgs

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

BACKEND = NATIVE_JAX if ARGS.cpu else PALLAS

# The cgols code-unit system: 1 kpc, 1e6 Msun, 100 km/s.
code_units = CodeUnits(1 * u.kpc, 1e6 * u.M_sun, 100 * u.km / u.s)
CIE_PARAMS = cie_parabolic_cooling(code_units)
T_SCALE = 10.0**CIE_PARAMS.log10_temperature_to_kelvin  # K per code temperature
LAMBDA_SCALE = 10.0**CIE_PARAMS.log10_lambda_cgs_to_code  # code Lambda per cgs Lambda
TIME_SCALE_S = (1.0 * code_units.code_time).to(u.s).value
DENSITY_SCALE_CGS = (1.0 * code_units.code_mass / code_units.code_length**3).to(
    u.g / u.cm**3
).value

X_H, Z_METAL, MU = 0.76, 0.02, 0.6
GAMMA = 5.0 / 3.0
FLOOR_K = 1e4
FLOOR_CODE = float(
    (FLOOR_K * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
)

CIE_CONFIG = CoolingCurveConfig(
    cooling_curve_type=CIE_PARABOLIC,
    density_convention=TOTAL_NUMBER_DENSITY,
)

_FAILURES = []


def check(condition, message):
    """Record a pass/fail line; collected into the final summary."""
    status = "PASS" if bool(condition) else "FAIL"
    if not condition:
        _FAILURES.append(message)
    print(f"  [{status}] {message}")


def kelvin_to_code(T_K):
    """Kelvin -> the cooling module's rescaled temperature."""
    return T_K / T_SCALE


def code_to_kelvin(T_code):
    """The cooling module's rescaled temperature -> Kelvin."""
    return T_code * T_SCALE


def lambda_reference_cgs(T_K):
    """Eq. A4 of CGOLS I in plain Kelvin / cgs (numpy twin of cgols'
    ``cooling_lambda_cgs``; that module is imported in check 1 to confirm the
    two agree exactly)."""
    logT = np.log10(T_K)
    lam = np.where(
        logT < 5.9,
        10.0 ** (-1.3 * (logT - 5.25) ** 2 - 21.25),
        np.where(
            logT < 7.4,
            10.0 ** (0.7 * (logT - 7.1) ** 2 - 22.8),
            10.0 ** (0.45 * logT - 26.065),
        ),
    )
    return np.where(T_K < 1e4, 0.0, lam)


# -------------------------------------------------------------
# ============ ↓ 1. curve identity vs the paper ↓ =============
# -------------------------------------------------------------
def check_curve_identity():
    """The wired-in curve reproduces Eq. A4 exactly, in x64 and in float32."""
    print("\n[1] curve identity vs the published Eq. A4")

    cooling_lambda_cgs = import_cgols_reference()

    T_K = np.logspace(3.5, 8.5, 2001)
    T_code = kelvin_to_code(T_K)

    reference = np.asarray(cooling_lambda_cgs(jnp.array(T_K)))
    check(
        np.allclose(reference, lambda_reference_cgs(T_K), rtol=1e-12, atol=0),
        "cgols.cooling_lambda_cgs matches the local Eq. A4 rendering",
    )

    lam_code = np.asarray(
        _cooling_rate(jnp.array(T_code), jnp.ones_like(jnp.array(T_code)), CIE_CONFIG, CIE_PARAMS)
    )
    lam_cgs = lam_code / LAMBDA_SCALE

    cooling = reference > 0.0
    rel_err = np.max(np.abs(lam_cgs[cooling] - reference[cooling]) / reference[cooling])
    print(f"      max relative error (x64): {rel_err:.3e}")
    check(rel_err < 1e-10, "curve matches Eq. A4 to < 1e-10 in double precision")
    check(
        np.all(lam_cgs[~cooling] == 0.0),
        f"Lambda is exactly zero below 10^4 K ({(~cooling).sum()} samples)",
    )

    # The fit is genuinely DISCONTINUOUS at log10 T = 5.9; assert the jump has
    # the analytic magnitude rather than asserting continuity, which catches an
    # off-by-one in the where-chain.
    log_low = -1.3 * (5.9 - 5.25) ** 2 - 21.25
    log_mid = 0.7 * (5.9 - 7.1) ** 2 - 22.8
    expected_jump = 10.0 ** (log_mid - log_low)
    eps = 1e-9
    below = np.asarray(
        _cooling_rate(
            jnp.array(kelvin_to_code(10.0 ** (5.9 - eps))), jnp.array(1.0), CIE_CONFIG, CIE_PARAMS
        )
    )
    above = np.asarray(
        _cooling_rate(
            jnp.array(kelvin_to_code(10.0 ** (5.9 + eps))), jnp.array(1.0), CIE_CONFIG, CIE_PARAMS
        )
    )
    jump = float(above / below)
    print(f"      jump across log10 T = 5.9: {jump:.6f} (expected {expected_jump:.6f})")
    check(
        abs(jump - expected_jump) < 1e-6,
        "the ~1.6 % discontinuity at log10 T = 5.9 has the analytic magnitude",
    )

    # Repeat in float32 - the precision the production runs use. The conversion
    # is applied inside the 10**(...) exponent precisely so no 1e-22-scale
    # intermediate is materialised.
    jax.config.update("jax_enable_x64", False)
    try:
        lam32 = np.asarray(
            _cooling_rate(
                jnp.array(T_code, dtype=jnp.float32),
                jnp.ones(T_code.shape, dtype=jnp.float32),
                CIE_CONFIG,
                CIE_PARAMS,
            )
        )
    finally:
        jax.config.update("jax_enable_x64", True)

    check(np.all(np.isfinite(lam32)), "float32 curve is finite everywhere (no over/underflow)")
    check(np.all(lam32 >= 0.0), "float32 curve is non-negative everywhere")
    # Exclude a narrow band around the 10^4 K cutoff: the curve is exactly
    # discontinuous there (Lambda jumps from 0 to 5.2e-24), so a sample sitting
    # on the boundary lands on either side depending on float32 rounding of
    # log10 T. That is the cutoff's nature, not a precision loss in the fit.
    away_from_cutoff = cooling & (np.abs(np.log10(T_K) - 4.0) > 5e-3)
    rel32 = np.max(
        np.abs(lam32[away_from_cutoff] / LAMBDA_SCALE - reference[away_from_cutoff])
        / reference[away_from_cutoff]
    )
    print(f"      max relative error (float32): {rel32:.3e}")
    check(rel32 < 1e-5, "float32 curve matches Eq. A4 to < 1e-5")
    check(np.all(lam32[~cooling] == 0.0), "float32 curve is exactly zero below 10^4 K")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(T_K[cooling], reference[cooling], "k-", lw=2, label="Eq. A4 (reference, cgs)")
    ax.loglog(T_K[cooling], lam_cgs[cooling], "r--", lw=1.2, label="astronomix CIE_PARABOLIC")
    ax.axvline(1e4, color="0.6", ls=":", lw=1)
    ax.axvline(10**5.9, color="0.8", ls=":", lw=1)
    ax.axvline(10**7.4, color="0.8", ls=":", lw=1)
    ax.set_xlabel("T [K]")
    ax.set_ylabel(r"$\Lambda$ [erg s$^{-1}$ cm$^3$]")
    ax.set_title("Solar-metallicity CIE cooling curve (CGOLS Eq. A4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cie_cooling_curve.png", dpi=200)
    plt.close(fig)
    print(f"      wrote {FIGURES_DIR / 'cie_cooling_curve.png'}")


# -------------------------------------------------------------
# ============ ↓ 2. the volumetric prefactor ↓ ================
# -------------------------------------------------------------
def check_prefactor():
    """n^2 vs n_e n_H is a factor mu_e mu_H / mu^2 ~ 4.30 - guard it."""
    print("\n[2] volumetric density convention")

    mu, mu_e, mu_H = get_effective_molecular_weights(X_H, Z_METAL, MU)
    check(abs(float(mu) - MU) < 1e-12, f"mean_molecular_weight override pins mu = {MU}")
    mu_xz, _, _ = get_effective_molecular_weights(X_H, Z_METAL)
    print(f"      mu from X,Z = {float(mu_xz):.4f}, overridden mu = {float(mu):.4f}")

    rho = jnp.array(1.0)
    T = jnp.array(kelvin_to_code(1e7))
    lam = _cooling_rate(T, rho, CIE_CONFIG, CIE_PARAMS)

    rate_total = dtemperature_dt(rho, T, X_H, Z_METAL, GAMMA, CIE_CONFIG, CIE_PARAMS, MU)
    expected = -(GAMMA - 1.0) * rho * lam / MU
    check(
        abs(float(rate_total - expected)) <= 1e-13 * abs(float(expected)),
        "TOTAL_NUMBER_DENSITY gives dT/dt = -(gamma-1) rho Lambda / mu",
    )

    eh_config = CIE_CONFIG._replace(density_convention=ELECTRON_HYDROGEN_DENSITY)
    rate_eh = dtemperature_dt(rho, T, X_H, Z_METAL, GAMMA, eh_config, CIE_PARAMS, MU)
    ratio = float(rate_total / rate_eh)
    expected_ratio = float(mu_e * mu_H / MU**2)
    print(f"      n^2 / (n_e n_H) rate ratio: {ratio:.6f} (expected {expected_ratio:.6f})")
    check(
        abs(ratio - expected_ratio) < 1e-10 * expected_ratio,
        "switching the convention changes the rate by exactly mu_e mu_H / mu^2",
    )

    check(
        abs(float(_volumetric_cooling_prefactor(CIE_CONFIG, MU, mu_e, mu_H)) - 1.0 / MU) < 1e-14,
        "TOTAL_NUMBER_DENSITY prefactor is 1 / mu",
    )
    check(
        abs(float(_volumetric_cooling_prefactor(eh_config, MU, mu_e, mu_H)) - MU / (mu_e * mu_H))
        < 1e-14,
        "ELECTRON_HYDROGEN_DENSITY prefactor is mu / (mu_e mu_H)",
    )

    # cooling_time must pick up the same convention, and be infinite where the
    # curve is exactly zero.
    t_cool = cooling_time(rho, T, X_H, Z_METAL, GAMMA, CIE_CONFIG, CIE_PARAMS, MU)
    check(
        abs(float(t_cool) - float(T / jnp.abs(rate_total))) < 1e-12 * float(t_cool),
        "cooling_time equals T / |dT/dt| for the active convention",
    )
    t_cold = cooling_time(
        rho, jnp.array(kelvin_to_code(1e3)), X_H, Z_METAL, GAMMA, CIE_CONFIG, CIE_PARAMS, MU
    )
    check(jnp.isinf(t_cold), "cooling_time is inf below the curve's 10^4 K cutoff")


# -------------------------------------------------------------
# ====== ↓ helpers for the single-cell integration checks ↓ ===
# -------------------------------------------------------------
def _single_cell_state(density_code, T_K):
    """A 1-cell primitive state (rho, vx, vy, vz, P) at the given T."""
    T_code = kelvin_to_code(T_K)
    rho = jnp.array([density_code])
    pressure = get_pressure_from_temperature(rho, jnp.array([T_code]), X_H, Z_METAL, MU)
    zero = jnp.zeros_like(rho)
    return jnp.stack([rho, zero, zero, zero, pressure])


class _CellRegistry:
    """The minimal registered-variable indices ``update_pressure_by_cooling``
    needs for the 5-row single-cell state above."""

    density_index = 0
    pressure_index = 4


def _cool_single_cell(
    state,
    dt_code,
    method,
    max_subcycles=16,
    floor_mode=COOLING_FLOOR_CLIP,
    max_fractional_temperature_change=0.01,
):
    """Apply one cooling operator step to the single-cell state."""
    cooling_config = CoolingConfig(
        cooling=True,
        cooling_method=method,
        cooling_placement=COOLING_OPERATOR_SPLIT,
        floor_mode=floor_mode,
        max_subcycles=max_subcycles,
        max_fractional_temperature_change=max_fractional_temperature_change,
        cooling_curve_config=CIE_CONFIG,
    )
    params = SimulationParams(
        gamma=GAMMA,
        cooling_params=CoolingParams(
            hydrogen_mass_fraction=X_H,
            metal_mass_fraction=Z_METAL,
            mean_molecular_weight=MU,
            floor_temperature=FLOOR_CODE,
            cooling_curve_params=CIE_PARAMS,
        ),
    )
    return update_pressure_by_cooling(
        state, _CellRegistry, cooling_config, params, dt_code
    )


def _state_temperature_K(state):
    """The Kelvin temperature of the single-cell state."""
    T_code = get_temperature_from_pressure(state[0], state[4], X_H, Z_METAL, MU)
    return float(code_to_kelvin(T_code)[0])


def _reference_temperature_K(T0_K, n_cgs, dt_code):
    """LSODA reference for dT/dt = -(gamma-1) n Lambda(T) / k_B, clipped at 10^4 K.

    Derived from d/dt [n k_B T / (gamma - 1)] = -n^2 Lambda(T) with n the TOTAL
    number density - the same convention the code uses. Integrated in pure cgs so
    it shares no code path with astronomix.
    """
    k_B = c.k_B.cgs.value
    t_end_s = dt_code * TIME_SCALE_S

    def rhs(_, y):
        T = max(y[0], 1e4)
        return [-(GAMMA - 1.0) * n_cgs * float(lambda_reference_cgs(np.array(T))) / k_B]

    def hit_floor(_, y):
        return y[0] - 1e4

    hit_floor.terminal = True
    hit_floor.direction = -1

    sol = solve_ivp(
        rhs, (0.0, t_end_s), [T0_K], method="LSODA", rtol=1e-10, atol=1e-6, events=hit_floor
    )
    return max(float(sol.y[0, -1]), 1e4)


def _density_for_number_density(n_cgs):
    """Code-unit mass density for a total number density n [cm^-3] at mu = 0.6."""
    rho_cgs = n_cgs * MU * c.m_p.cgs.value
    return rho_cgs / DENSITY_SCALE_CGS


# -------------------------------------------------------------
# ====== ↓ 3. single cell vs an independent ODE solve ↓ =======
# -------------------------------------------------------------
def check_single_cell():
    """The operator matches an LSODA reference, reaches the floor and holds it."""
    print("\n[3] single cooling cell vs a scipy LSODA reference")

    n_cgs = 1.0
    rho = _density_for_number_density(n_cgs)
    T0_K = 1e7
    print(f"      n = {n_cgs} cm^-3 -> rho = {rho:.4e} code, T0 = {T0_K:.1e} K")

    T0_code = kelvin_to_code(T0_K)
    t_cool = float(
        cooling_time(
            jnp.array(rho), jnp.array(T0_code), X_H, Z_METAL, GAMMA, CIE_CONFIG, CIE_PARAMS, MU
        )
    )
    print(f"      t_cool(T0) = {t_cool:.4e} code = {t_cool * TIME_SCALE_S / 3.156e13:.3f} Myr")

    # Step at the production cadence: dt = 0.1 t_cool, recomputed from the
    # current state each step exactly as ``cooling_time_step_limit`` does in the
    # run. That matters here - t_cool shrinks by more than an order of magnitude
    # as the cell falls from 10^7 K toward the floor (Lambda rises steeply), so
    # a dt frozen at the initial cooling time would drift out of the regime the
    # 16 sub-cycles can resolve within a handful of steps.
    state = _single_cell_state(rho, T0_K)
    T_reference = T0_K
    for step in range(6):
        T_now = get_temperature_from_pressure(state[0], state[4], X_H, Z_METAL, MU)
        dt = 0.1 * float(
            jnp.min(
                cooling_time(
                    state[0], T_now, X_H, Z_METAL, GAMMA, CIE_CONFIG, CIE_PARAMS, MU
                )
            )
        )
        state = _cool_single_cell(state, dt, SUBCYCLED_EXPLICIT_COOLING)
        T_reference = _reference_temperature_K(T_reference, n_cgs, dt)
        T_code_value = _state_temperature_K(state)
        rel = abs(T_code_value - T_reference) / T_reference
        print(
            f"      step {step + 1}: T = {T_code_value:.6e} K, reference {T_reference:.6e} K"
            f"  (rel {rel:.2e})"
        )
        # ~2.5e-4 per step, accumulating: this is the first-order truncation
        # error of the 1 % sub-cycle rule (Cholla's scheme), verified to be
        # exactly that by the convergence check below - not a coding error.
        check(rel < 5e-3, f"step {step + 1} agrees with the ODE reference to < 5e-3")

    # Tightening the per-sub-cycle fraction must shrink the error proportionally
    # - the signature of a consistent first-order integrator. If the deviation
    # above were a wrong prefactor or a wrong curve, it would not move at all.
    dt_conv = 0.1 * t_cool
    T_ref_conv = _reference_temperature_K(T0_K, n_cgs, dt_conv)
    conv_errors = {}
    for fraction in (0.01, 0.001):
        T_conv = _state_temperature_K(
            _cool_single_cell(
                _single_cell_state(rho, T0_K),
                dt_conv,
                SUBCYCLED_EXPLICIT_COOLING,
                max_subcycles=256,
                max_fractional_temperature_change=fraction,
            )
        )
        conv_errors[fraction] = abs(T_conv - T_ref_conv) / T_ref_conv
        print(f"      sub-cycle fraction {fraction:g}: rel {conv_errors[fraction]:.2e}")
    check(
        conv_errors[0.001] < 0.2 * conv_errors[0.01],
        "the error shrinks with the sub-cycle fraction (first-order convergent)",
    )

    # Sub-cycle exhaustion (Risk 7): the dt limit and max_subcycles are coupled.
    # At 16 sub-cycles the loop can only cover ~1 - 0.99^16 ~ 15 % of T, so a
    # step far beyond the 10 % rule degrades to the residual catch-all step -
    # silently, by design. Raising max_subcycles recovers the accuracy.
    state0 = _single_cell_state(rho, T0_K)
    T_ref_big = _reference_temperature_K(T0_K, n_cgs, 0.5 * t_cool)
    errors = {}
    for subcycles in (16, 64):
        T_big = _state_temperature_K(
            _cool_single_cell(state0, 0.5 * t_cool, SUBCYCLED_EXPLICIT_COOLING, subcycles)
        )
        errors[subcycles] = abs(T_big - T_ref_big) / T_ref_big
        print(f"      dt = 0.5 t_cool, {subcycles} sub-cycles: rel {errors[subcycles]:.2e}")
    check(
        errors[64] < 0.02 < errors[16],
        "an over-long step exhausts 16 sub-cycles but converges at 64 "
        "(the dt limit is what makes 16 sufficient)",
    )

    # Drive it well past the floor and confirm it lands exactly there and stays.
    state = _cool_single_cell(state, 100.0 * t_cool, SUBCYCLED_EXPLICIT_COOLING)
    T_floor = _state_temperature_K(state)
    print(f"      after a long step: T = {T_floor:.6e} K (floor {FLOOR_K:.1e} K)")
    check(abs(T_floor - FLOOR_K) < 1e-6 * FLOOR_K, "the cell lands exactly on the 10^4 K floor")

    state_again = _cool_single_cell(state, 100.0 * t_cool, SUBCYCLED_EXPLICIT_COOLING)
    check(
        abs(_state_temperature_K(state_again) - T_floor) < 1e-12 * T_floor,
        "further cooling steps at the floor are a no-op (idempotent)",
    )

    # The revert floor mode must also hold the cell at (or above) the floor.
    state_revert = _cool_single_cell(
        _single_cell_state(rho, T0_K),
        100.0 * t_cool,
        SUBCYCLED_EXPLICIT_COOLING,
        floor_mode=0,
    )
    check(
        _state_temperature_K(state_revert) >= FLOOR_K * (1 - 1e-9),
        "COOLING_FLOOR_REVERT never leaves the cell below the floor either",
    )


# -------------------------------------------------------------
# ====== ↓ 4. sub-cycling vs one big explicit step ↓ ==========
# -------------------------------------------------------------
def check_subcycling():
    """At dt = 5 t_cool the sub-cycled step is accurate and Euler is nonsense.

    Note what the comparison has to be made against: with ``COOLING_FLOOR_CLIP``
    a single explicit step at this dt is *rescued* by the floor and lands on
    10^4 K too, so comparing the floored results would hide the problem
    entirely. The Euler step itself lands at -4e7 K - the floor is doing all the
    work, and there is nothing left of the physics.
    """
    print("\n[4] sub-cycling vs a single explicit step at dt = 5 t_cool")

    n_cgs = 1.0
    rho = _density_for_number_density(n_cgs)
    T0_K = 1e7
    T0_code = kelvin_to_code(T0_K)
    t_cool = float(
        cooling_time(
            jnp.array(rho), jnp.array(T0_code), X_H, Z_METAL, GAMMA, CIE_CONFIG, CIE_PARAMS, MU
        )
    )
    dt = 5.0 * t_cool

    T_reference = _reference_temperature_K(T0_K, n_cgs, dt)

    state0 = _single_cell_state(rho, T0_K)
    T_sub = _state_temperature_K(_cool_single_cell(state0, dt, SUBCYCLED_EXPLICIT_COOLING))
    rel_sub = abs(T_sub - T_reference) / T_reference
    print(f"      reference   T = {T_reference:.6e} K")
    print(f"      sub-cycled  T = {T_sub:.6e} K  (rel {rel_sub:.2e})")
    check(rel_sub < 1e-2, "SUBCYCLED_EXPLICIT_COOLING is within 1 % of the reference")

    # The raw explicit update, i.e. what EXPLICIT_COOLING computes before the
    # floor clamps it.
    rate = dtemperature_dt(
        jnp.array(rho), jnp.array(T0_code), X_H, Z_METAL, GAMMA, CIE_CONFIG, CIE_PARAMS, MU
    )
    T_euler_raw = float(code_to_kelvin(T0_code + rate * dt))
    rel_euler = abs(T_euler_raw - T_reference) / T_reference
    print(f"      plain Euler T = {T_euler_raw:.6e} K  (rel {rel_euler:.2e}, pre-floor)")
    check(T_euler_raw < 0.0, "the unfloored explicit step goes negative at dt = 5 t_cool")
    check(
        rel_euler > 100.0 * max(rel_sub, 1e-12),
        "a single explicit step is off by orders of magnitude (why sub-cycling exists)",
    )
    check(
        abs(_state_temperature_K(_cool_single_cell(state0, dt, EXPLICIT_COOLING)) - FLOOR_K)
        < 1e-6 * FLOOR_K,
        "with the floor on, that nonsense is silently rescued to 10^4 K "
        "(so never judge the integrator by the floored result)",
    )

    # At a moderate step, where the floor masks nothing, sub-cycling is still
    # the more accurate of the two.
    dt_mid = 0.25 * t_cool
    T_ref_mid = _reference_temperature_K(T0_K, n_cgs, dt_mid)
    T_sub_mid = _state_temperature_K(
        _cool_single_cell(state0, dt_mid, SUBCYCLED_EXPLICIT_COOLING)
    )
    T_euler_mid = _state_temperature_K(_cool_single_cell(state0, dt_mid, EXPLICIT_COOLING))
    rel_sub_mid = abs(T_sub_mid - T_ref_mid) / T_ref_mid
    rel_euler_mid = abs(T_euler_mid - T_ref_mid) / T_ref_mid
    print(
        f"      at dt = 0.25 t_cool: sub-cycled rel {rel_sub_mid:.2e}, "
        f"Euler rel {rel_euler_mid:.2e}"
    )
    check(
        rel_sub_mid < rel_euler_mid,
        "sub-cycling is more accurate than one Euler step away from the floor too",
    )


# -------------------------------------------------------------
# ====== ↓ 5. the cooling-time timestep constraint ↓ ==========
# -------------------------------------------------------------
def _random_cooling_state(key, shape, n_cgs=1.0, T_K=1e7):
    """A small random 3D state around n ~ 1 cm^-3 and T ~ 1e7 K."""
    k_rho, k_T = jax.random.split(key)
    rho = _density_for_number_density(n_cgs) * (
        1.0 + 0.5 * jax.random.uniform(k_rho, shape, minval=-1.0, maxval=1.0)
    )
    T_code = kelvin_to_code(T_K) * (
        1.0 + 0.5 * jax.random.uniform(k_T, shape, minval=-1.0, maxval=1.0)
    )
    pressure = get_pressure_from_temperature(rho, T_code, X_H, Z_METAL, MU)
    zero = jnp.zeros(shape)
    return jnp.stack([rho, zero, zero, zero, pressure])


def _cooling_box_config(cooling, num_cells=16, timestep_limit=False, subcycles=16):
    """A tiny periodic FD box, optionally with the CIE cooling operator."""
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        time_integrator=RK4_LSRK,
        dimensionality=3,
        backend_config=BackendConfig(
            backend=BACKEND,
            pallas_block_shape=(4, 4, 4),
            pallas_use_triton=not ARGS.cpu,
        ),
        num_cells=StaticIntVector(num_cells, num_cells, num_cells),
        box_size=StaticFloatVector(1.0, 1.0, 1.0),
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        # The cgols positivity stack, so the smoke run exercises the same
        # interaction between the cooling floor and the pressure floor.
        positivity_config=PositivityConfig(
            default_positivity_protection=True,
            vacuum_rest=True,
            nan_safe=True,
            velocity_clip=True,
            temperature_clip=True,
        ),
        progress_bar=False,
        **(
            dict(
                cooling_config=CoolingConfig(
                    cooling=True,
                    cooling_method=SUBCYCLED_EXPLICIT_COOLING,
                    cooling_placement=COOLING_OPERATOR_SPLIT,
                    floor_mode=COOLING_FLOOR_CLIP,
                    max_subcycles=subcycles,
                    max_fractional_temperature_change=0.01,
                    cooling_timestep_limit=timestep_limit,
                    cooling_curve_config=CIE_CONFIG,
                )
            )
            if cooling
            else {}
        ),
    )
    return config


def _cooling_box_params(t_end, max_thermal_energy_fraction=0.1):
    return SimulationParams(
        t_end=t_end,
        C_cfl=0.4,
        gamma=GAMMA,
        minimum_density=1e-8,
        minimum_pressure=1e-12,
        positivity_max_velocity=1e4,
        positivity_max_pressure_over_density=kelvin_to_code(5e9) / MU,
        cooling_params=CoolingParams(
            hydrogen_mass_fraction=X_H,
            metal_mass_fraction=Z_METAL,
            mean_molecular_weight=MU,
            floor_temperature=FLOOR_CODE,
            max_thermal_energy_fraction=max_thermal_energy_fraction,
            cooling_curve_params=CIE_PARAMS,
        ),
    )


def check_timestep_limit():
    """dt is bounded by max_thermal_energy_fraction * min(t_cool), and the
    constraint is actually wired into the integrator."""
    print("\n[5] cooling-time timestep limit")

    num_cells = 16
    state = _random_cooling_state(jax.random.PRNGKey(0), (num_cells,) * 3)

    config = _cooling_box_config(cooling=True, num_cells=num_cells, timestep_limit=True)
    registered_variables = get_registered_variables(config)
    config = finalize_config(config, state.shape)
    params = _cooling_box_params(t_end=1e-3)

    dt_cool = float(cooling_time_step_limit(state, config, params, registered_variables))

    T_code = get_temperature_from_pressure(state[0], state[4], X_H, Z_METAL, MU)
    t_cool = cooling_time(
        state[0], T_code, X_H, Z_METAL, GAMMA, CIE_CONFIG, CIE_PARAMS, MU
    )
    expected = 0.1 * float(jnp.min(t_cool))
    print(f"      dt_cool = {dt_cool:.6e}, 0.1 * min(t_cool) = {expected:.6e}")
    check(abs(dt_cool - expected) < 1e-12 * expected, "dt limit is 0.1 * min(t_cool)")

    # Non-cooling gas must not constrain the step at all.
    cold = state.at[4].set(
        get_pressure_from_temperature(state[0], jnp.full_like(state[0], kelvin_to_code(5e3)),
                                      X_H, Z_METAL, MU)
    )
    check(
        jnp.isinf(cooling_time_step_limit(cold, config, params, registered_variables)),
        "an entirely non-cooling grid gives an infinite (inactive) limit",
    )

    # And the limit must be live in the integrator: with a binding constraint the
    # trajectory differs, with a non-binding one it is bit-identical to no limit.
    dt_cfl = float(
        _cfl_time_step_fd_hydro(
            state,
            config.grid_spacing,
            params.dt_max,
            params.gamma,
            config,
            params,
            registered_variables,
            params.C_cfl,
        )
    )
    # In this box the hydro CFL step is already the smaller of the two, so pick a
    # thermal-energy fraction that puts the cooling limit clearly below it.
    binding_fraction = 0.1 * (0.2 * dt_cfl / dt_cool)
    print(
        f"      dt_cfl = {dt_cfl:.6e}; binding fraction {binding_fraction:.3e} "
        f"-> dt_cool = {binding_fraction / 0.1 * dt_cool:.6e}"
    )
    check(
        binding_fraction / 0.1 * dt_cool < dt_cfl,
        "the chosen fraction makes the cooling limit the binding constraint",
    )

    t_end = 5.0 * dt_cfl

    def run(timestep_limit, fraction):
        cfg = _cooling_box_config(
            cooling=True, num_cells=num_cells, timestep_limit=timestep_limit
        )
        rv = get_registered_variables(cfg)
        cfg = finalize_config(cfg, state.shape)
        return time_integration(
            state, cfg, _cooling_box_params(t_end, fraction), rv
        )

    without = run(False, 0.1)
    binding = run(True, binding_fraction)
    loose = run(True, 1e6)

    check(
        not bool(jnp.allclose(without, binding, rtol=0, atol=0)),
        "the binding cooling-time limit changes the trajectory (the wiring is live)",
    )
    check(
        bool(jnp.array_equal(without, loose)),
        "a non-binding cooling-time limit is bit-identical to no limit",
    )


# -------------------------------------------------------------
# ====== ↓ 6. differentiability through the sub-cycles ↓ ======
# -------------------------------------------------------------
def check_differentiability():
    """jax.grad through the fori_loop sub-cycles is finite and non-zero."""
    print("\n[6] differentiability of the sub-cycled operator")

    rho = _density_for_number_density(1.0)
    t_cool = float(
        cooling_time(
            jnp.array(rho),
            jnp.array(kelvin_to_code(1e7)),
            X_H,
            Z_METAL,
            GAMMA,
            CIE_CONFIG,
            CIE_PARAMS,
            MU,
        )
    )
    def cooled_pressure(coefficient, field, state, dt):
        """sum(P) after one sub-cycled cooling step, as a function of one
        cooling-curve coefficient."""
        params = SimulationParams(
            gamma=GAMMA,
            cooling_params=CoolingParams(
                hydrogen_mass_fraction=X_H,
                metal_mass_fraction=Z_METAL,
                mean_molecular_weight=MU,
                floor_temperature=FLOOR_CODE,
                cooling_curve_params=CIE_PARAMS._replace(**{field: coefficient}),
            ),
        )
        cooling_config = CoolingConfig(
            cooling=True,
            cooling_method=SUBCYCLED_EXPLICIT_COOLING,
            cooling_placement=COOLING_OPERATOR_SPLIT,
            floor_mode=COOLING_FLOOR_CLIP,
            max_subcycles=16,
            max_fractional_temperature_change=0.01,
            cooling_curve_config=CIE_CONFIG,
        )
        out = update_pressure_by_cooling(
            state, _CellRegistry, cooling_config, params, dt
        )
        return jnp.sum(out[_CellRegistry.pressure_index])

    # Differentiate with respect to the coefficient of the branch the cell is
    # actually on: a 10^7 K cell sits on the middle parabola, a 10^5 K one on the
    # low parabola. (c_low at 10^7 K is a genuine structural zero, not a broken
    # gradient - the fit is piecewise.)
    for field, T0_K in (("c_mid", 1e7), ("c_low", 1e5)):
        state = _single_cell_state(rho, T0_K)
        t_cool_here = float(
            cooling_time(
                jnp.array(rho),
                jnp.array(kelvin_to_code(T0_K)),
                X_H,
                Z_METAL,
                GAMMA,
                CIE_CONFIG,
                CIE_PARAMS,
                MU,
            )
        )
        grad = float(
            jax.grad(cooled_pressure)(
                getattr(CIE_PARAMS, field), field, state, 0.5 * t_cool_here
            )
        )
        print(f"      T0 = {T0_K:.0e} K: d(sum P)/d({field}) = {grad:.6e}")
        check(
            np.isfinite(grad), f"the gradient w.r.t. {field} is finite (no 0/0 in the guard)"
        )
        check(grad != 0.0, f"the gradient w.r.t. {field} is non-zero (the loop is not detached)")

    # A cell parked at the floor never enters the loop body; its gradient must
    # still be a clean finite zero rather than a NaN.
    grad_floor = float(
        jax.grad(cooled_pressure)(
            CIE_PARAMS.c_mid, "c_mid", _single_cell_state(rho, FLOOR_K), 0.5 * t_cool
        )
    )
    print(f"      at the floor: d(sum P)/d(c_mid) = {grad_floor:.6e}")
    check(np.isfinite(grad_floor), "the gradient at the floor is finite (not NaN)")
    check(grad_floor == 0.0, "the gradient at the floor is exactly zero")


# -------------------------------------------------------------
# ====== ↓ 7. a small 3D smoke run with cooling on ↓ ==========
# -------------------------------------------------------------
def check_smoke_run():
    """A 32^3 FD box with the full cooling stack stays finite and floored."""
    print("\n[7] 32^3 smoke run with cooling active")

    num_cells = 32
    state = _random_cooling_state(jax.random.PRNGKey(1), (num_cells,) * 3)

    config = _cooling_box_config(cooling=True, num_cells=num_cells, timestep_limit=True)
    registered_variables = get_registered_variables(config)
    config_final = finalize_config(config, state.shape)

    dt_cool = float(
        cooling_time_step_limit(
            state, config_final, _cooling_box_params(1.0), registered_variables
        )
    )
    params = _cooling_box_params(t_end=20.0 * dt_cool)

    final_state = time_integration(state, config_final, params, registered_variables)

    check(bool(jnp.all(jnp.isfinite(final_state))), "the final state is finite everywhere")
    check(bool(jnp.all(final_state[0] > 0.0)), "density stays positive")
    check(bool(jnp.all(final_state[4] > 0.0)), "pressure stays positive")

    T_final = get_temperature_from_pressure(
        final_state[0], final_state[4], X_H, Z_METAL, MU
    )
    T_min_K = float(code_to_kelvin(jnp.min(T_final)))
    T_max_K = float(code_to_kelvin(jnp.max(T_final)))
    print(f"      T range: {T_min_K:.4e} .. {T_max_K:.4e} K")
    # The per-step pressure floor can be the hotter of the two constraints in
    # rarefied cells, so the bound is min(cooling floor, pressure-floor T).
    P_floor_T_K = float(
        code_to_kelvin(
            get_temperature_from_pressure(
                jnp.max(final_state[0]), jnp.array(params.minimum_pressure), X_H, Z_METAL, MU
            )
        )
    )
    check(
        T_min_K >= min(FLOOR_K, P_floor_T_K) * (1 - 1e-6),
        "no cell ends below the 10^4 K cooling floor",
    )

    # The box must have cooled overall. Judge that on the mean, not the maximum:
    # the hydro step still compresses gas, so individual cells can end hotter
    # than any cell started.
    T_initial = get_temperature_from_pressure(state[0], state[4], X_H, Z_METAL, MU)
    mean_initial_K = float(code_to_kelvin(jnp.mean(T_initial)))
    mean_final_K = float(code_to_kelvin(jnp.mean(T_final)))
    print(f"      mean T: {mean_initial_K:.4e} -> {mean_final_K:.4e} K")
    check(mean_final_K < mean_initial_K, "the box has cooled (mean T decreased)")
    check(
        abs(T_min_K - FLOOR_K) < 1e-6 * FLOOR_K,
        "the fastest-cooling cells have reached the 10^4 K floor exactly",
    )


def check_cgols_smoke():
    """Run cgols.py end-to-end with CGOLS_COOLING=1 at a small resolution."""
    print("\n[7b] cgols.py B-series subprocess smoke run")

    cgols_dir = Path(__file__).resolve().parents[1] / "self_gravity" / "cgols"
    env = dict(os.environ)
    env.update(
        CGOLS_DIM="64",
        CGOLS_NUM_SNAPSHOTS="2",
        CGOLS_CHECKPOINT_EVERY="0",
        CGOLS_SMOOTHING_SIGMA="1.5",
    )

    ic = subprocess.run(
        [sys.executable, "cgols.py"],
        cwd=cgols_dir,
        env={**env, "CGOLS_CREATE_IC": "1", "CGOLS_END_FRACTION": "0.001"},
        capture_output=True,
        text=True,
    )
    check(ic.returncode == 0, "CGOLS_CREATE_IC=1 at CGOLS_DIM=64 exits 0")
    if ic.returncode != 0:
        print(ic.stdout[-3000:])
        print(ic.stderr[-3000:])
        return

    run = subprocess.run(
        [sys.executable, "cgols.py"],
        cwd=cgols_dir,
        env={**env, "CGOLS_COOLING": "1", "CGOLS_END_FRACTION": "0.01"},
        capture_output=True,
        text=True,
    )
    check(run.returncode == 0, "CGOLS_COOLING=1 CGOLS_DIM=64 run exits 0")
    if run.returncode != 0:
        print(run.stdout[-3000:])
        print(run.stderr[-3000:])
        return

    check("Radiative cooling ON" in run.stdout, "the run banner reports cooling ON")

    frames = sorted((cgols_dir / "cgols_snapshots_B").glob("frame_*.npz"))
    check(len(frames) > 0, "the B-series run wrote snapshot frames")
    if frames:
        with np.load(frames[-1]) as data:
            keys = [k for k in data.files if "T" in k or "temperature" in k]
            finite = all(np.all(np.isfinite(data[k])) for k in data.files
                         if data[k].dtype.kind == "f")
            check(finite, "the last B-series frame is finite")
            for k in keys:
                positive = np.asarray(data[k])
                positive = positive[np.isfinite(positive) & (positive > 0)]
                if positive.size:
                    print(f"      {k}: min {positive.min():.3e}, max {positive.max():.3e}")


# -------------------------------------------------------------
# ====== ↓ 8. mixing-layer no-op regression ↓ =================
# -------------------------------------------------------------
# Checksum of the short mixing-layer run below, recorded from the code BEFORE
# the CIE-cooling changes: the same run was executed in a git worktree at the
# pre-change commit (d0039b2) and the two final states compared byte for byte -
# they were bit-identical, and this is their sum of squares in double precision
# on the NATIVE_JAX backend.
#
# What it guards: the mixing path (update_pressure_by_cooling_mixing plus its own
# floor block) and the finite-difference source-term rewiring -- NOT the changed
# update_pressure_by_cooling floor switch, which has no in-repo users. It is
# backend- and precision-sensitive, so it is only asserted under --cpu (x64,
# NATIVE_JAX); elsewhere the run is still exercised for finiteness.
MIXING_REFERENCE_CHECKSUM = 31411160.298206843


def _mixing_layer_final_state(num_cells=16):
    """A short 3D mixing-layer run, the only in-repo user of the cooling module."""
    density_contrast = 100.0
    xi, mach_number, P0 = 3.0, 0.5, 1.0
    rho_hot = 1.0
    rho_cold = density_contrast * rho_hot
    c_hot = (GAMMA * P0 / rho_hot) ** 0.5
    v_rel = mach_number * c_hot
    L_x = L_y = 1.0
    L_z = 1.5
    t_sh = L_x / v_rel

    config = SimulationConfig(
        positivity_config=PositivityConfig(default_positivity_protection=True),
        backend_config=BackendConfig(
            backend=BACKEND,
            pallas_block_shape=(4, 4, 4),
            pallas_use_triton=not ARGS.cpu,
        ),
        progress_bar=False,
        dimensionality=3,
        box_size=StaticFloatVector(L_x, L_y, L_z),
        num_cells=StaticIntVector(num_cells, num_cells, int(1.5 * num_cells)),
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
        cooling_config=CoolingConfig(
            cooling=True,
            cooling_method=IMPLICIT_COOLING,
            cooling_curve_config=CoolingCurveConfig(
                cooling_curve_type=SIMPLE_MIXING_LAYER_COOLING,
            ),
        ),
    )
    registered_variables = get_registered_variables(config)

    z = jnp.linspace(0, L_z, int(1.5 * num_cells), endpoint=False)
    Z = jnp.broadcast_to(z, (num_cells, num_cells, int(1.5 * num_cells)))
    smoothing_length = (L_x / num_cells) / 2
    blend = jnp.tanh((Z - L_z / 2) / smoothing_length)

    density = 0.5 * (rho_cold * (1 - blend) + rho_hot * (1 + blend))
    velocity_x = 0.5 * (-v_rel / 2 * (1 - blend) + v_rel / 2 * (1 + blend))
    zero = jnp.zeros_like(density)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity_x,
        velocity_y=zero,
        velocity_z=zero,
        gas_pressure=P0 * jnp.ones_like(density),
    )
    config = finalize_config(config, initial_state.shape)

    params = SimulationParams(
        t_end=0.05 * t_sh,
        C_cfl=1.5,
        gamma=GAMMA,
        minimum_density=rho_cold / 100,
        minimum_pressure=P0 / 100,
        cooling_params=CoolingParams(
            cooling_curve_params=MixingCoolingParams(
                xi=xi, mach_number=mach_number, density_contrast=density_contrast
            ),
            floor_temperature=P0 / rho_cold,
        ),
    )
    return time_integration(initial_state, config, params, registered_variables)


def mixing_layer_checksum():
    """A deterministic scalar summary of the short mixing-layer run."""
    final_state = _mixing_layer_final_state()
    return float(jnp.sum(final_state.astype(jnp.float64) ** 2)), final_state


def check_mixing_regression():
    """The existing cooling user is unchanged by this work."""
    print("\n[8] mixing-layer no-op regression")

    checksum, final_state = mixing_layer_checksum()
    print(f"      sum of squares: {checksum!r}")
    check(bool(jnp.all(jnp.isfinite(final_state))), "the mixing-layer run stays finite")
    check(
        config_default_cooling_placement_is_rk_stages(),
        "CoolingConfig defaults still place cooling in the RK stages",
    )

    if ARGS.cpu:
        check(
            checksum == MIXING_REFERENCE_CHECKSUM,
            "the mixing-layer result is bit-identical to the pre-change reference",
        )
    else:
        print(
            "      NOTE: the recorded reference is for --cpu (x64, NATIVE_JAX);\n"
            "            skipping the equality assertion on this backend."
        )


def config_default_cooling_placement_is_rk_stages():
    """The new fields must default to the historical behaviour."""
    default = CoolingConfig()
    return (
        default.cooling_placement == COOLING_IN_RK_STAGES
        and default.floor_mode == 0
        and not default.cooling_timestep_limit
        and CoolingCurveConfig().density_convention == ELECTRON_HYDROGEN_DENSITY
        and CoolingParams().mean_molecular_weight == 0.0
    )


# -------------------------------------------------------------
# ====================== ↓ Run ↓ ==============================
# -------------------------------------------------------------
CHECKS = {
    "1": check_curve_identity,
    "2": check_prefactor,
    "3": check_single_cell,
    "4": check_subcycling,
    "5": check_timestep_limit,
    "6": check_differentiability,
    "7": check_smoke_run,
    "8": check_mixing_regression,
}

selected = [s.strip() for s in ARGS.checks.split(",") if s.strip()] or list(CHECKS)
for name in selected:
    if name not in CHECKS:
        raise SystemExit(f"unknown check {name!r}; available: {', '.join(CHECKS)}")
    CHECKS[name]()

if ARGS.cgols_smoke:
    check_cgols_smoke()

print("\n" + "=" * 60)
if _FAILURES:
    print(f"{len(_FAILURES)} CHECK(S) FAILED:")
    for failure in _FAILURES:
        print(f"  - {failure}")
    raise SystemExit(1)
print("all checks passed")
