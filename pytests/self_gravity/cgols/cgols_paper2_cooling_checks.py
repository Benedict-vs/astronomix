"""
ASGARD Paper II: checks of the model-B cooling and heating against the literature.

Every number the Paper II Methods and Appendix A quote about the cooling curve
comes from here, so that the text uses manifest macros (writing repo, rule 1)
and a referee can rerun the checks. Nothing here reads a simulation: the
inputs are the production cooling tables of ``ki02_gf12_cooling`` (the
composition of ``cgols.py``) and published CIE tables in
``cooling_references/`` (see its README for sources and normalisations).

Checks (each prints a table; ``--strict`` turns the tolerances into failures):

  1. Kernel normalisation: the operator's de/dt equals -n_H^2 Lambda + n_H Gamma
     with n_H = rho / (1.4271 m_H), at the production X, Z.
  2. mu(T) and x_e(T) of the tables.
  3. The table's GF12 part against a rebuild from GF12's own element CIE
     columns (x_e * sum_E A_E Lambda_E,CIE, Asplund+09).
  4. Published CIE curves converted to per n_H^2 (GS07, SD93, Schure+09) and
     the CGOLS I Eq. A4 fit, as ratios to our curve.
  5. Normalisation of the CGOLS fit from its free-free regime (SD93 lambda_net
     per n_H^2 at 10^8-10^8.5 K, analytic free-free per n_H^2 at 10^9 K).
  6. Factors between the conventions (n^2 at mu = 0.6 against n_H^2, n_e n_H,
     n_e n_t), and the old over new volumetric rate at fixed rho and T.
  7. Which elements carry the cooling near the 10^5 K peak.
  8. The KI02 -> GF12 joint at 10^4.2 K.
  9. Photoelectric heating against cooling in the code, over the warm-wind
     temperatures, for J/J_0 up to the starburst field at 1 kpc.
 10. The composition against Asplund et al. (2009).

Usage (CPU is enough; no GPU job):

    JAX_PLATFORMS=cpu python cgols_paper2_cooling_checks.py [--strict] [--export]

``--export`` writes the numbers and the two figures into
``paper_exports/paper2/manifest.yaml`` (status provisional until the writing
side has checked them).
"""

# ruff: noqa: E402
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as u
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astronomix._modules._cooling import _cooling_tables as ct
from astronomix._modules._cooling._cooling import (
    dtemperature_dt,
    get_effective_molecular_weights,
)
from astronomix._modules._cooling.cooling_options import (
    ELECTRON_HYDROGEN_DENSITY,
    PIECEWISE_POWER_LAW,
    CIEParabolicParams,
    CoolingCurveConfig,
)
from astronomix.units.unit_helpers import CodeUnits

HERE = Path(__file__).resolve().parent
REF = HERE / "cooling_references"
FIG_DIR = HERE / "figures" / "cgols" / "cooling"

# Production composition of the ki02gf12 tables (cgols.py: _KI02_X, _KI02_Z) and
# the FUV calibration (CGOLS_FUV_SIGMA_SFR0, the high-state SFR of the schedule).
X_PROD, Z_PROD = 0.76, 0.02
SIGMA_SFR0 = 2.5e-3  # Msun / yr / kpc^2 (Kim & Ostriker 2017)
SFR_HIGH = 20.0  # Msun / yr

K_B = 1.380649e-16
M_P = 1.67262192e-24
M_H = ct._HYDROGEN_MASS_CGS
MU_OLD = 0.6  # CGOLS / Cholla
LOG_T_GRID = np.array([4.2, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.5, 7.0, 7.5, 8.0])

# Asplund et al. (2009) photospheric log eps (H = 12) and atomic masses.
A09 = dict(He=10.93, Li=1.05, Be=1.38, B=2.70, C=8.43, N=7.83, O=8.69, F=4.56, Ne=7.93,
           Na=6.24, Mg=7.60, Al=6.45, Si=7.51, P=5.41, S=7.12, Cl=5.50, Ar=6.40, K=5.03,
           Ca=6.34, Sc=3.15, Ti=4.95, V=3.93, Cr=5.64, Mn=5.43, Fe=7.50, Co=4.99, Ni=6.22,
           Cu=4.19, Zn=4.56)
A09_PROTOSOLAR_HE = 10.98
# Gnat & Sternberg (2007) Table 1: Asplund et al. (2005) with Ne from Drake & Testa (2005), He 1/12,
# and only these ten elements.
GS07_ABUNDANCES = dict(He=10.92, C=8.39, N=7.78, O=8.66, Ne=8.29, Mg=7.53, Si=7.51, S=7.14, Fe=7.45)
ATOMIC_MASS = dict(H=1.00794, He=4.0026, C=12.011, N=14.007, O=15.999, Ne=20.180, Na=22.99,
                   Mg=24.305, Al=26.98, Si=28.085, S=32.06, Ar=39.95, Ca=40.08, Fe=55.845,
                   Ni=58.69)
ATOMIC_NUMBER = dict(C=6, N=7, O=8, Ne=10, Na=11, Mg=12, Al=13, Si=14, S=16, Ar=18, Ca=20,
                     Fe=26, Ni=28)

RESULTS = {}
FAILURES = []


def record(key, value, unit, description, digits=3):
    RESULTS[key] = dict(value=float(value), unit=unit, description=description, digits=digits)


def expect(name, ok, detail):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def interp_log(x, xp, yp):
    return 10 ** np.interp(x, xp, np.log10(np.maximum(yp, 1e-300)))


# -------------------------------------------------------------------------------------
# Reference tables
# -------------------------------------------------------------------------------------

def load_references():
    gf12 = json.load(open(REF / "gf12_element_cie.json"))["elements"]
    gs07 = np.array(json.load(open(REF / "gs07_cie_cooling.json"))["rows"])
    sd93 = np.array(json.load(open(REF / "sd93_solar_cie.json"))["rows"])
    schure = np.array(json.load(open(REF / "schure09_table2.json"))["rows"])
    return gf12, gs07, sd93, schure


def gf12_element(gf12, element, log_t):
    rows = np.array(gf12[element])
    lt, idx = np.unique(np.log10(rows[:, 0]), return_index=True)
    return 10 ** np.interp(log_t, lt, np.log10(np.maximum(rows[idx, 1], 1e-40)))


def cgols_fit(log_t):
    """CGOLS I Eq. A4 (Cholla CIE_cool), erg cm^3 / s, as the code stores it."""
    p = CIEParabolicParams()
    lt = np.asarray(log_t, float)
    return np.where(
        lt < p.log10_T_floor, 0.0,
        np.where(lt < p.log10_T_break_1, 10 ** (p.a_low * (lt - p.t_low) ** 2 + p.c_low),
                 np.where(lt < p.log10_T_break_2, 10 ** (p.a_mid * (lt - p.t_mid) ** 2 + p.c_mid),
                          10 ** (0.45 * lt - 26.065))))


# -------------------------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------------------------

def check_kernel(model):
    print("\n1. Kernel normalisation (production X = 0.76, Z = 0.02, n_H = 3e-4 cm^-3)")
    units = CodeUnits(1 * u.kpc, 1e6 * u.M_sun, 1 * u.km / u.s)
    setup = ct.ki02_gf12_cooling(units, X_PROD, Z_PROD, log10_T1_max=10.0)
    mu, _, _ = get_effective_molecular_weights(X_PROD, Z_PROD)
    rho_u = (1 * units.code_density).to(u.g / u.cm ** 3).value
    p_u = (1 * units.code_pressure).to(u.erg / u.cm ** 3).value
    t_u = (1 * units.code_time).to(u.s).value
    config = CoolingCurveConfig(cooling_curve_type=PIECEWISE_POWER_LAW,
                                density_convention=ELECTRON_HYDROGEN_DENSITY)
    gamma = 5.0 / 3.0
    n_h = 3e-4
    rho = n_h * ct.TIGRESS_MU_H * M_H / rho_u
    worst_cool, worst_heat = 0.0, 0.0
    for lt in np.arange(3.5, 8.01, 0.25):
        t = 10 ** lt
        xe, _ = model.electron_fraction(t)
        t_code = setup.T1_to_code * t * (ct.TIGRESS_X_PART + xe) / ct.TIGRESS_MU_H
        args = (jnp.asarray([rho]), jnp.asarray([t_code]), X_PROD, Z_PROD, gamma, config,
                setup.cooling_curve_params)
        rate_cool = float(dtemperature_dt(*args)[0])
        rate_net = float(dtemperature_dt(*args, heating_rate=setup.heating_rate,
                                         heating_log10_T_table=setup.heating_log10_T_table,
                                         heating_factor_table=setup.heating_factor_table)[0])
        to_edot = float(rho / mu) / (gamma - 1) * p_u / t_u  # dT~/dt -> erg cm^-3 s^-1
        edot_cool = -rate_cool * to_edot
        edot_heat = (rate_net - rate_cool) * to_edot
        want_cool = n_h ** 2 * model.cooling_lambda(t)
        want_heat = n_h * ct.TIGRESS_GAMMA_0 * (1.0 - model.x_HII(t))
        if lt >= 4.5:
            worst_cool = max(worst_cool, abs(edot_cool / want_cool - 1))
        if want_heat > 1e-3 * n_h * ct.TIGRESS_GAMMA_0:
            worst_heat = max(worst_heat, abs(edot_heat / want_heat - 1))
    expect("cooling = n_H^2 Lambda", worst_cool < 2e-3, f"max deviation {worst_cool:.1e} (10^4.5-10^8 K)")
    expect("heating = n_H Gamma_0 (1 - x_HII)", worst_heat < 2e-2,
           f"max deviation {worst_heat:.1e} where the heating factor > 1e-3")
    record("KernelNormDeviation", 100 * worst_cool, "percent",
           "Largest deviation of the cooling operator's de/dt from n_H^2 Lambda(T), "
           "n_H = rho/(1.4271 m_H), 10^4.5-10^8 K, production composition (X = 0.76, Z = 0.02)", 2)


def check_mu(model):
    print("\n2. mu(T) = 1.4271 / (1.1 + x_e)")
    for lt in [4.0, 4.2, 4.5, 5.0, 6.0, 8.0]:
        xe, _ = model.electron_fraction(10 ** lt)
        print(f"  log T {lt:.1f}: x_e = {xe:.4f}, mu = {ct.TIGRESS_MU_H / (ct.TIGRESS_X_PART + xe):.4f}")
    xe8, _ = model.electron_fraction(1e8)
    mu_ion = ct.TIGRESS_MU_H / (ct.TIGRESS_X_PART + xe8)
    xe_neutral, _ = model.electron_fraction(10 ** 3.5)
    mu_neutral = ct.TIGRESS_MU_H / (ct.TIGRESS_X_PART + xe_neutral)
    expect("fully ionised x_e includes He and metal electrons", 1.19 < xe8 < 1.21, f"x_e(1e8 K) = {xe8:.4f}")
    record("MuIonised", mu_ion, "", "Mean molecular weight of fully ionised gas in the model-B tables, "
           "1.4271/(1.1 + x_e) at 1e8 K", 3)
    record("MuNeutral", mu_neutral, "", "Mean molecular weight of neutral gas (10^3.5 K) in the model-B tables", 3)


def check_gf12_rebuild(model, gf12):
    print("\n3. Table GF12 part against GF12's own element CIE columns (A09)")
    t = 10 ** LOG_T_GRID
    xe = model.electron_fraction(t)[0] - model.xe_floor
    rebuilt = xe * (gf12_element(gf12, "H", LOG_T_GRID)
                    + sum(10 ** (A09[e] - 12) * gf12_element(gf12, e, LOG_T_GRID) for e in A09))
    ours = model.lambda_cie(t)
    ratio = ours / rebuilt
    for lt, r in zip(LOG_T_GRID, ratio):
        print(f"  log T {lt:.2f}: table / rebuild = {r:.3f}")
    band = (LOG_T_GRID >= 4.2) & (LOG_T_GRID <= 7.5)
    expect("table matches GF12 within 15 %", np.all(np.abs(ratio[band] - 1) < 0.15),
           f"range {ratio[band].min():.3f}-{ratio[band].max():.3f} (10^4.2-10^7.5 K)")
    i5 = np.argmin(abs(LOG_T_GRID - 5.0))
    record("TableOverGFTwelvePeak", ratio[i5], "", "Model-B table (GS07 ion fractions x GF12 ion-by-ion "
           "rates, A09) over GF12's own element CIE columns with A09 abundances, at 1e5 K", 3)
    record("TableOverGFTwelveMax", np.max(np.abs(ratio[band] - 1)) * 100, "percent",
           "Largest deviation of the model-B table from the GF12 element-CIE rebuild, 10^4.2-10^7.5 K", 2)
    return rebuilt


def check_references(model, gs07, sd93, schure):
    GF12 = load_references()[0]
    print("\n4. Published curves per n_H^2 over ours (ours = KI02+GF12 as in the code)")
    t = 10 ** LOG_T_GRID
    ours = model.cooling_lambda(t)
    xe = model.electron_fraction(t)[0] - model.xe_floor
    curves = {
        "GS07": interp_log(LOG_T_GRID, np.log10(gs07[:, 0]), gs07[:, 4]) * xe,  # per n_e n_H, Z = 1
        "SD93": interp_log(LOG_T_GRID, sd93[:, 0], 10 ** sd93[:, 4]),  # lambda_net at n_H = 1
        "Schure09": interp_log(LOG_T_GRID, schure[:, 0], 10 ** schure[:, 2]),  # Lambda_hd
        "CGOLS fit": cgols_fit(LOG_T_GRID),
    }
    print("  log T  " + "  ".join(f"{k:>9s}" for k in curves))
    for i, lt in enumerate(LOG_T_GRID):
        print(f"  {lt:5.2f}  " + "  ".join(f"{curves[k][i] / ours[i]:9.2f}" for k in curves))
    i5 = np.argmin(abs(LOG_T_GRID - 5.0))
    peak = (LOG_T_GRID >= 4.5) & (LOG_T_GRID <= 5.5)
    hot = LOG_T_GRID >= 6.0
    sd_ratio = curves["SD93"] / ours
    sch_ratio = curves["Schure09"] / ours
    gs_ratio = curves["GS07"] / ours
    # Like for like: GF12 rebuilt with GS07's own abundances (A05 + Drake & Testa Ne, ten elements),
    # so that only the Cloudy version (06.02 against 08.00) differs.
    xe_gs = model.electron_fraction(t)[0] - model.xe_floor
    gf12_gs = xe_gs * (gf12_element(GF12, "H", LOG_T_GRID) + sum(
        10 ** (a - 12) * gf12_element(GF12, e, LOG_T_GRID) for e, a in GS07_ABUNDANCES.items()))
    gs_matched = curves["GS07"] / gf12_gs
    print("  GS07 over GF12 rebuilt with GS07's abundances: " + "  ".join(
        f"{a:.2f}:{b:.2f}" for a, b in zip(LOG_T_GRID, gs_matched)))
    band = (LOG_T_GRID >= 4.5) & (LOG_T_GRID <= 6.0)
    expect("GS07 and GF12 agree at matched abundances, 10^4.5-10^6 K",
           np.all(np.abs(gs_matched[band] - 1) < 0.2), f"range {gs_matched[band].min():.2f}-{gs_matched[band].max():.2f}")
    ff = LOG_T_GRID >= 7.5
    expect("all CIE tables within 1.5 in the free-free regime (>= 10^7.5 K)",
           all(np.all((curves[k][ff] / ours[ff] > 0.66) & (curves[k][ff] / ours[ff] < 1.5)) for k in curves),
           "the normalisations agree where atomic data matter least")
    record("GSMatchedMaxDev", np.max(np.abs(gs_matched[band] - 1)) * 100, "percent",
           "Largest deviation of Gnat & Sternberg (2007) from GF12 rebuilt with the same abundances, "
           "10^4.5-10^6 K (Cloudy 06.02 against 08.00)", 2)
    record("SDOverOursPeak", sd_ratio[i5], "", "Sutherland & Dopita (1993) solar CIE lambda_net (per n_H^2) "
           "over our curve at 1e5 K", 2)
    record("SDOverOursPeakMax", sd_ratio[peak].max(), "", "Largest SD93/ours ratio, 10^4.5-10^5.5 K", 2)
    record("SchureOverOursPeak", sch_ratio[i5], "", "Schure et al. (2009) Lambda_hd (per n_H^2) over our curve at 1e5 K", 2)
    record("GSOverOursMaxDev", np.max(np.abs(gs_ratio[LOG_T_GRID <= 6.0] - 1)) * 100, "percent",
           "Largest deviation of Gnat & Sternberg (2007, Z = 1, per n_e n_H times x_e) from our curve, "
           "10^4.2-10^6 K", 2)
    record("CGOLSFitOverOursPeak", curves["CGOLS fit"][i5 + 1] / ours[i5 + 1], "",
           "CGOLS I Eq. A4 fit over our curve at 10^5.25 K (shapes only, both read per n_H^2)", 2)
    return curves


def check_cgols_normalisation(sd93):
    print("\n5. Normalisation of the CGOLS fit from its free-free regime")
    lt = np.array([8.0, 8.5])
    fit = np.log10(cgols_fit(lt))
    sd = np.interp(lt, sd93[:, 0], sd93[:, 4])
    for a, b, c in zip(lt, fit, sd):
        print(f"  log T {a:.1f}: fit {b:.3f}, SD93 lambda_net/n_H^2 {c:.3f}, difference {b - c:+.3f} dex")
    # Analytic free-free per n_H^2 at 1e9 K, fully ionised, A09 abundances, Gaunt factor 1.2 (1.1-1.4).
    y_he = 10 ** (A09["He"] - 12)
    metals = {e: 10 ** (A09[e] - 12) for e in ATOMIC_NUMBER}
    ne = 1 + 2 * y_he + sum(ATOMIC_NUMBER[e] * a for e, a in metals.items())
    z2 = 1 + 4 * y_he + sum(ATOMIC_NUMBER[e] ** 2 * a for e, a in metals.items())
    ff = {g: 1.42e-27 * np.sqrt(1e9) * g * ne * z2 for g in (1.1, 1.2, 1.4)}
    fit9 = np.log10(cgols_fit(9.0))
    print(f"  log T 9.0: fit {fit9:.3f}, analytic free-free per n_H^2 {np.log10(ff[1.2]):.3f} "
          f"(g_ff 1.1-1.4: {np.log10(ff[1.1]):.3f} to {np.log10(ff[1.4]):.3f})")
    per_n2 = 2 * np.log10(ct.TIGRESS_MU_H * M_H / (MU_OLD * M_P))
    dev = np.max(np.abs(fit - sd))
    expect("fit is per n_H^2 in the free-free regime", dev < 0.1, f"|fit - SD93| <= {dev:.3f} dex")
    expect("fit is not per n^2 (mu = 0.6)", per_n2 > 0.5, f"per-n^2 reading would be {per_n2:.2f} dex lower")
    record("CGOLSFitFreeFreeDex", dev, "dex", "Largest |log difference| between the CGOLS I Eq. A4 fit and SD93 "
           "lambda_net per n_H^2 at 10^8 and 10^8.5 K (free-free regime)", 2)
    record("CGOLSFitFreeFreeAnalyticDex", abs(fit9 - np.log10(ff[1.2])), "dex",
           "|log difference| between the CGOLS fit and analytic free-free per n_H^2 at 1e9 K (A09, g_ff = 1.2)", 2)
    record("PerNsqOffsetDex", per_n2, "dex", "Offset in log Lambda between per n_H^2 and per n^2 readings "
           "of a curve, n = rho/(0.6 m_p), n_H = rho/(1.4271 m_H)", 2)


def check_conventions(model):
    print("\n6. Factors between conventions, fully ionised solar gas")
    n_over_nh = ct.TIGRESS_MU_H * M_H / (MU_OLD * M_P)
    xe, _ = model.electron_fraction(1e8)
    n_t = ct.TIGRESS_X_PART  # ions + atoms per H
    factors = {
        "n_H^2": n_over_nh ** 2,
        "n_e n_H": n_over_nh ** 2 / xe,
        "n_e n_t": n_over_nh ** 2 / (xe * n_t),
    }
    for k, v in factors.items():
        print(f"  table per {k:8s}: n^2 (mu = 0.6) gives {v:.2f} x the intended rate")
    record("OldNormFactor", factors["n_H^2"], "", "(n/n_H)^2 with n = rho/(0.6 m_p), n_H = rho/(1.4271 m_H): "
           "factor by which n^2 Lambda over-cools a curve tabulated per n_H^2", 3)
    record("OldNormFactorNeNH", factors["n_e n_H"], "", "The same for a curve tabulated per n_e n_H (fully ionised)", 3)
    record("OldNormFactorNeNt", factors["n_e n_t"], "", "The same for a curve tabulated per n_e n_t (fully ionised)", 3)
    print("  old volumetric rate (n^2 x CGOLS fit) over new (n_H^2 x KI02+GF12) at fixed rho, T:")
    lt = np.arange(4.5, 8.01, 0.25)
    ratio = factors["n_H^2"] * cgols_fit(lt) / model.cooling_lambda(10 ** lt)
    print("  " + "  ".join(f"{a:.2f}:{b:.1f}" for a, b in zip(lt, ratio)))
    band = (lt >= 4.5) & (lt <= 6.0)
    record("OldNewRateMin", ratio[band].min(), "", "Smallest old/new volumetric cooling rate at fixed rho, T, 10^4.5-10^6 K", 2)
    record("OldNewRateMax", ratio[band].max(), "", "Largest old/new volumetric cooling rate at fixed rho, T, 10^4.5-10^6 K", 2)


def check_elements(model, gf12):
    print("\n7. Elements carrying the cooling (GF12 element CIE x A09 x x_e, per n_H^2)")
    out = {}
    for lt in [4.75, 5.0, 5.25, 5.5]:
        xe = model.electron_fraction(10 ** lt)[0] - model.xe_floor
        parts = {"H": xe * gf12_element(gf12, "H", lt)}
        parts.update({e: xe * 10 ** (A09[e] - 12) * gf12_element(gf12, e, lt) for e in A09})
        total = sum(parts.values())
        top = sorted(parts.items(), key=lambda kv: -kv[1])[:5]
        print(f"  log T {lt:.2f}: " + ", ".join(f"{e} {100 * v / total:.0f} %" for e, v in top))
        out[lt] = {e: v / total for e, v in parts.items()}
    record("CoolShareCarbonPeak", 100 * out[5.0]["C"], "percent", "Share of carbon in the CIE cooling at 1e5 K (GF12, A09)", 2)
    record("CoolShareOxygenPeak", 100 * out[5.0]["O"], "percent", "Share of oxygen in the CIE cooling at 1e5 K (GF12, A09)", 2)
    record("CoolShareOxygenHigh", 100 * out[5.25]["O"], "percent", "Share of oxygen in the CIE cooling at 10^5.25 K (GF12, A09)", 2)
    return out


def check_joint(model):
    print("\n8. KI02 -> GF12 joint at 10^4.2 K")
    t = 10 ** 4.2
    jump = model.lambda_ki02(t) / model.lambda_cie(t)
    excess = model.cooling_lambda(t) / model.lambda_cie(t)
    print(f"  KI02/GF12 = {jump:.3f}; blended/GF12 = {excess:.3f}")
    record("JointJump", jump, "", "KI02 over GF12 at 10^4.2 K (the step a hard switch would make)", 3)
    record("JointExcess", excess, "", "Blended curve over GF12 at 10^4.2 K (geometric sigmoid, 0.05 dex)", 3)


def check_heating(model):
    print("\n9. Photoelectric heating over cooling in the code, n_H Gamma / (n_H^2 Lambda)")
    j_1kpc = SFR_HIGH / SIGMA_SFR0 / (4 * np.pi * 1.0 ** 2)  # unattenuated, high state
    j_5kpc = SFR_HIGH / SIGMA_SFR0 / (4 * np.pi * 5.0 ** 2)
    print(f"  J/J_0 of the unattenuated starburst (SFR 20): {j_1kpc:.0f} at 1 kpc, {j_5kpc:.0f} at 5 kpc")
    lt = np.linspace(np.log10(3e4), np.log10(5e5), 200)
    t = 10 ** lt
    gamma = ct.TIGRESS_GAMMA_0 * (1.0 - model.x_HII(t))
    lam = model.cooling_lambda(t)
    rows = []
    for n_h in (1e-4, 1e-3, 1e-2):
        ratio = j_1kpc * gamma / (n_h * lam)
        rows.append(ratio.max())
        print(f"  n_H = {n_h:.0e}: max over 3e4-5e5 K = {ratio.max():.2e} (at {t[np.argmax(ratio)]:.2e} K)")
    for tq in (2e4, 3e4, 5e4, 1e5):
        f = 1.0 - model.x_HII(tq)
        print(f"  heating factor 1 - x_HII at {tq:.0e} K: {f:.2e}")
    # Temperature above which the code's heating stays below 1 % of cooling for n_H >= 1e-3 at 1 kpc.
    ok = j_1kpc * gamma / (1e-3 * lam) < 0.01
    t_safe = t[np.argmax(ok)] if ok.any() else np.nan
    print(f"  heating < 1 % of cooling for n_H >= 1e-3 above {t_safe:.2e} K (unattenuated, 1 kpc)")
    record("HeatSafeTemperature", t_safe, "K", "Temperature above which the code's photoelectric heating stays below 1 % "
           "of cooling for n_H >= 1e-3 cm^-3 in the unattenuated starburst field at 1 kpc", 2)
    record("HeatFactorThreeE", 1.0 - model.x_HII(3e4), "", "Heating factor 1 - x_HII(CIE) at 3e4 K", 2)
    record("StarburstFieldOneKpc", j_1kpc, "", "J/J_0 of the unattenuated starburst point source at 1 kpc, SFR 20 Msun/yr, "
           "Sigma_SFR,0 = 2.5e-3 Msun/yr/kpc^2", 2)
    record("HeatOverCoolWarmMax", rows[0], "", "Largest photoelectric heating over cooling in the code for 3e4-5e5 K gas "
           "at n_H = 1e-4 cm^-3 in the unattenuated starburst field at 1 kpc", 2)
    return j_1kpc


def check_composition():
    print("\n10. Composition against Asplund et al. (2009)")
    y_he = 10 ** (A09["He"] - 12)
    y_he_proto = 10 ** (A09_PROTOSOLAR_HE - 12)
    metals = sum(10 ** (A09[e] - 12) * ATOMIC_MASS[e] for e in ATOMIC_MASS if e not in ("H", "He"))
    mu_h = 1 + y_he * ATOMIC_MASS["He"] / ATOMIC_MASS["H"] + metals / ATOMIC_MASS["H"]
    mu_h_proto = 1 + y_he_proto * ATOMIC_MASS["He"] / ATOMIC_MASS["H"] + metals / ATOMIC_MASS["H"]
    off = (ct.TIGRESS_MU_H / mu_h) ** 2 - 1
    print(f"  mu_H: tables {ct.TIGRESS_MU_H}, A09 photospheric {mu_h:.4f}, protosolar He {mu_h_proto:.4f}")
    print(f"  He/H: particle count 0.1, A09 photospheric {y_he:.4f}, protosolar {y_he_proto:.4f}")
    print(f"  n_H^2 Lambda at fixed rho: tables vs A09 photospheric {100 * -off / (1 + off):+.1f} %")
    record("MuHAZeroNine", mu_h, "", "rho/(n_H m_H) for Asplund et al. (2009) photospheric abundances", 3)
    record("NHsqOffsetAZeroNine", 100 * (1 - (mu_h / ct.TIGRESS_MU_H) ** 2), "percent",
           "How much lower n_H^2 Lambda is with mu_H = 1.4271 than with the A09 photospheric mu_H, at fixed rho", 2)


# -------------------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------------------

def figure_curves(model, gs07, sd93, schure, gf12):
    lt = np.linspace(4.0, 8.5, 451)
    t = 10 ** lt
    ours = model.cooling_lambda(t)
    xe = model.electron_fraction(t)[0] - model.xe_floor
    m = lt >= 4.0
    curves = [
        ("this work: KI02 + GF12 (A09)", ours, dict(color="k", lw=2)),
        ("GF12 element CIE, rebuilt (A09)", xe * (gf12_element(gf12, "H", lt) + sum(
            10 ** (A09[e] - 12) * gf12_element(gf12, e, lt) for e in A09)), dict(color="C0", ls="--")),
        ("Gnat & Sternberg 2007", interp_log(lt, np.log10(gs07[:, 0]), gs07[:, 4]) * xe, dict(color="C2", ls=":")),
        ("Sutherland & Dopita 1993", interp_log(lt, sd93[:, 0], 10 ** sd93[:, 4]), dict(color="C3")),
        ("Schure et al. 2009 (original)", interp_log(lt, schure[:, 0], 10 ** schure[:, 2]), dict(color="C1")),
        ("CGOLS I Eq. A4 fit", cgols_fit(lt), dict(color="C4", ls="-.")),
    ]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.6, 4.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
    for name, y, kw in curves:
        ax.plot(lt[m], np.log10(np.maximum(y[m], 1e-30)), label=name, **kw)
        if name.startswith("this work"):
            continue
        ax2.plot(lt[m], y[m] / ours[m], **kw)
    ax.set_ylabel(r"$\log\,\Lambda/n_\mathrm{H}^2$ [erg cm$^3$ s$^{-1}$]")
    ax.set_ylim(-24.6, -20.6)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    ax2.axhline(1, color="k", lw=1)
    ax2.axhspan(10 ** -0.1, 10 ** 0.1, color="0.85", zorder=0)
    ax2.set_yscale("log")
    ax2.set_ylim(0.3, 5)
    ax2.set_ylabel("ratio to this work")
    ax2.set_xlabel(r"$\log\,T$ [K]")
    for a in (ax, ax2):
        a.axvspan(np.log10(3e4), np.log10(5e5), color="C9", alpha=0.08, zorder=0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "paper2_cooling_curves_compared.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_heating(model, j_1kpc):
    lt = np.linspace(1.5, 6.0, 451)
    t = 10 ** lt
    lam = model.cooling_lambda(t)
    gamma = ct.TIGRESS_GAMMA_0 * (1.0 - model.x_HII(t))
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for j, color in ((1.0, "C0"), (j_1kpc, "C3")):
        for n_h, ls in ((1e-4, ":"), (1e-2, "--"), (1.0, "-")):
            ax.plot(lt, np.log10(np.maximum(j * gamma / (n_h * lam), 1e-12)), color=color, ls=ls,
                    label=rf"$J/J_0 = {j:.0f}$, $n_\mathrm{{H}} = 10^{{{np.log10(n_h):.0f}}}$")
    ax.axhline(0, color="k", lw=1)
    ax.axhline(-2, color="0.5", lw=0.8, ls=":")
    ax.axvspan(np.log10(3e4), np.log10(5e5), color="C9", alpha=0.08)
    ax.set_ylim(-8, 6)
    ax.set_xlabel(r"$\log\,T$ [K]")
    ax.set_ylabel(r"$\log\,n_\mathrm{H}\Gamma / n_\mathrm{H}^2\Lambda$")
    ax.legend(fontsize=5.5, frameon=False, ncol=2)
    path = FIG_DIR / "paper2_heating_over_cooling.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# -------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strict", action="store_true", help="exit 1 if a check fails")
    parser.add_argument("--export", action="store_true", help="write the paper2 manifest entries")
    args = parser.parse_args()

    model = ct._KI02GF12Model()
    gf12, gs07, sd93, schure = load_references()
    check_kernel(model)
    check_mu(model)
    check_gf12_rebuild(model, gf12)
    check_references(model, gs07, sd93, schure)
    check_cgols_normalisation(sd93)
    check_conventions(model)
    check_elements(model, gf12)
    check_joint(model)
    j_1kpc = check_heating(model)
    check_composition()
    fig_curves = figure_curves(model, gs07, sd93, schure, gf12)
    fig_heat = figure_heating(model, j_1kpc)

    out = HERE / "data" / "studies" / "paper2_cooling_checks.json"  # cache, not in git
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(RESULTS, open(out, "w"), indent=1)
    print(f"\n{len(RESULTS)} numbers -> {out.relative_to(HERE)}; figures {fig_curves.name}, {fig_heat.name}")
    print("failures:", FAILURES or "none")

    if args.export:
        sys.path.insert(0, str(HERE))
        from paper_exports.export import export_figure, export_number
        for key, r in RESULTS.items():
            export_number("paper2", key, r["value"], unit=r["unit"], status="provisional",
                          description=r["description"] + " (cgols_paper2_cooling_checks.py; no simulation)",
                          runs=["none: cooling tables only"], digits=r["digits"], data=out)
        export_figure("paper2", "cooling_curves_compared", fig_curves, runs=["none: cooling tables only"],
                      caption_hint="CIE cooling per n_H^2: this work (KI02 + GF12, A09) against the GF12 "
                                   "element-CIE rebuild, Gnat & Sternberg 2007, Sutherland & Dopita 1993, "
                                   "the original Schure et al. 2009 curve and the CGOLS I fit; lower panel "
                                   "ratios to this work, grey band +-0.1 dex, shaded the warm-wind range")
        export_figure("paper2", "heating_over_cooling", fig_heat, runs=["none: cooling tables only"],
                      caption_hint="Photoelectric heating over cooling in the model-B prescription for the "
                                   "background-free solar-neighbourhood field and the unattenuated starburst "
                                   "at 1 kpc, three densities; shaded the warm-wind range")
    if args.strict and FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
