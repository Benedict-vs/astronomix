"""Time-averaged sensitivity maps of model A (asgard-papers #16).

The instantaneous forward-mode maps of cgols_sensitivity.py carry concentric rings: a numerical ~8-cell
oscillation born at the edge of the injection region travels out with the wind, and the loading shifts its
phase, so the tangent of an instantaneous map shows the shifted ripple, growing with time. A travelling ripple
averages out over a time window of at least one period (~0.5 Myr at 128^3, ~0.27 Myr at 256^3), without a
spatial beam and without changing the model.

This script integrates to t_end - W, then K equal segments to t_end, and averages the edge-on maps (column
density Sigma, mid-plane density n, AtomDB 0.3-1.0 keV absorbed photon S_X) over the K segment ends. The
instantaneous maps at t_end are kept for comparison. The time integrator is jitted inside astronomix with the
times as runtime parameters, so all segments share one compilation; the derivative is taken eagerly around it.

Modes (CGOLS_SENS_MODE):
    jvp  forward-mode tangents d/dln Mdot and d/dln Edot of all maps   -> cgols_sensitivity_avg<TAG>.npz
    fd   the maps at (1 +- eps) for both directions and every eps         -> cgols_sensitivity_avg<TAG>_fd.npz
    growth  the tangent of the state carried through 1 Myr segments from the wind onset to t_end (chained jvps),
         logging max |d ln rho| and max |d ln P| per segment until it turns non-finite (one direction,
         CGOLS_SENS_GROWTH_DIR = M or E)                                  -> cgols_sensitivity_growth<TAG>.npz

Env knobs (besides the cgols ones, e.g. CGOLS_DIM, CGOLS_X64, CGOLS_IC_TAG):
    CGOLS_SENS_TMYR     end time [Myr] (default 30)
    CGOLS_SENS_AVG_MYR  averaging window W [Myr] (default 1)
    CGOLS_SENS_AVG_N    number of samples K in the window (default 8)
    CGOLS_SENS_WENO_Z   1 = WENO-Z, the production scheme (default); 0 = Jiang-Shu as in cgols_sensitivity.py
    CGOLS_SENS_FD_EPS   comma-separated steps of the fd mode (default 0.01,0.002,0.001)
    CGOLS_SENS_FROZEN   1 = weno_ad_frozen_weights: the tangent of the linear scheme each step used (default 0)
    CGOLS_SOLVER        fd (default) or fv, as for cgols.py; the fv runs ignore the WENO knobs

    CGOLS_DIM=128 CGOLS_IC_TAG=_d128 CGOLS_X64=1 CGOLS_SENS_MODE=jvp python cgols_sensitivity_avg.py
"""

import itertools
import os
import time

os.environ.setdefault("CGOLS_SHARD_SPLIT", "(1, 1, 1, 1)")
os.environ.setdefault("CGOLS_DIM", "128")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u
from cgols import (
    SOLVER,
    _data_study,
    _snapshot_grid,
    code_length,
    code_mass,
    code_units,
    load_initial_conditions,
    m_p,
    mu,
)

from astronomix import time_integration
from astronomix.analysis_helpers.emission import (
    XRAY_TABLE_PATHS,
    EmissionMapSettings,
    emission_conversions,
    emission_maps,
    load_xray_emissivity_table,
)
from astronomix.option_classes.simulation_config import ON_DEVICE

T_END_MYR = float(os.environ.get("CGOLS_SENS_TMYR", "30"))
AVG_MYR = float(os.environ.get("CGOLS_SENS_AVG_MYR", "1"))
AVG_N = int(os.environ.get("CGOLS_SENS_AVG_N", "8"))
WENO_Z = os.environ.get("CGOLS_SENS_WENO_Z", "1") == "1"
MODE = os.environ.get("CGOLS_SENS_MODE", "jvp")
FD_EPS = tuple(float(e) for e in os.environ.get("CGOLS_SENS_FD_EPS", "0.01,0.002,0.001").split(","))
FROZEN = os.environ.get("CGOLS_SENS_FROZEN", "0") == "1"
GROWTH_DIR = os.environ.get("CGOLS_SENS_GROWTH_DIR", "E")
WIND_ONSET_MYR = 5.0
RES = int(os.environ["CGOLS_DIM"])
SCHEME = ("_wenoz" if WENO_Z else "_wenojs") if SOLVER == "fd" else "_fv"
TAG = (f"_d{RES}_t{T_END_MYR:g}myr_w{AVG_MYR:g}n{AVG_N}" + SCHEME + ("_frozen" if FROZEN else "")
       + ("_x64" if os.environ.get("CGOLS_X64", "0") == "1" else ""))  # fmt: skip
XRAY_BAND, XRAY_NH_CM2, XRAY_METALLICITY = "0.3-1", 4e20, 1.0  # as cgols_sensitivity.py

N_FACTOR = (code_mass / code_length**3).to(u.g / u.cm**3).value / (mu * m_p.to(u.g).value)
KPC_CM = (1 * u.kpc).to(u.cm).value


def build():
    """Differentiation-clean config and the map function f(theta) -> dict of averaged and final maps."""
    initial_state, config, params, rv = load_initial_conditions()
    if initial_state.shape[0] > rv.num_vars:
        # The saved ICs carry the FD dual-energy internal energy as an extra last variable; the FV scheme
        # (no dual energy) has none. Dropping it keeps the FV runs on exactly the FD runs' initial state.
        initial_state = initial_state[: rv.num_vars]
    config = config._replace(
        donate_state=False, progress_bar=False, monitor_diagnostics=False, activate_snapshot_callback=False,
        memory_analysis=False, snapshot_storage_mode=ON_DEVICE, weno_z=WENO_Z, weno_ad_frozen_weights=FROZEN,
    )  # fmt: skip
    myr = (1 * u.Myr).to(code_units.code_time).value
    times = [T_END_MYR - AVG_MYR + k * AVG_MYR / AVG_N for k in range(AVG_N + 1)]  # times[0] = window start
    ny = config.num_cells.y
    col_factor = N_FACTOR * (config.box_size.y / ny) * KPC_CM
    table = load_xray_emissivity_table(XRAY_TABLE_PATHS[XRAY_BAND], photons=True, absorbing_column_cm2=XRAY_NH_CM2)
    conversions = emission_conversions(code_units)
    settings = EmissionMapSettings(projection_axes=(1,), solar_abundance_set="A09")
    cell = float(config.box_size.y) / ny
    base = params.cgols_wind_params

    def maps(state):
        rho = state[rv.density_index]
        sx = emission_maps(rho, state[rv.pressure_index], conversions, table, cell, settings)
        return {"sigma": jnp.sum(rho, axis=1) * col_factor, "n_mid": rho[:, ny // 2, :] * N_FACTOR,
                "sx_atomdb": sx["xray_atomdb_hhe_xz"] + XRAY_METALLICITY * sx["xray_atomdb_metals_xz"]}  # fmt: skip

    def f(theta):
        wind = base._replace(schedule_mass_rates=base.schedule_mass_rates * theta[0],
                             schedule_energy_rates=base.schedule_energy_rates * theta[1])  # fmt: skip
        p = params._replace(cgols_wind_params=wind)
        state = time_integration(initial_state, config, p._replace(t_start=0.0, t_end=times[0] * myr), rv)
        total = None
        for t0, t1 in itertools.pairwise(times):
            state = time_integration(state, config, p._replace(t_start=t0 * myr, t_end=t1 * myr), rv)
            m = maps(state)
            total = m if total is None else {k: total[k] + m[k] for k in m}
        out = {f"{k}_avg": v / AVG_N for k, v in total.items()}
        out.update({f"{k}_inst": v for k, v in m.items()})
        return out

    def segment(state, theta, t0_myr, t1_myr):
        """The state at t1 from the state at t0, for the loading multipliers theta (growth mode)."""
        wind = base._replace(schedule_mass_rates=base.schedule_mass_rates * theta[0],
                             schedule_energy_rates=base.schedule_energy_rates * theta[1])  # fmt: skip
        p = params._replace(cgols_wind_params=wind, t_start=t0_myr * myr, t_end=t1_myr * myr)
        return time_integration(state, config, p, rv)

    return f, config, times, segment, initial_state, rv


def main():
    f, config, times, segment, initial_state, rv = build()
    x, _y, z, extent_xz, _ = _snapshot_grid(config)
    grid = {"x": np.asarray(x), "z": np.asarray(z), "extent_xz": np.asarray(extent_xz),
            "sample_times_myr": np.asarray(times[1:]), "t_end_myr": T_END_MYR, "weno_z": WENO_Z,
            "frozen_weights": FROZEN, "solver": SOLVER}  # fmt: skip
    theta0 = jnp.array([1.0, 1.0])
    out, start = {}, time.perf_counter()
    if MODE == "jvp":
        for label, direction in (("M", jnp.array([1.0, 0.0])), ("E", jnp.array([0.0, 1.0]))):
            primal, tangent = jax.jvp(f, (theta0,), (direction,))
            out.update({f"{k}0": np.asarray(v) for k, v in primal.items()})
            out.update({f"d{k}_d{label}": np.asarray(v) for k, v in tangent.items()})
            finite = {k: float(np.isfinite(np.asarray(v)).mean()) for k, v in tangent.items()}
            print(f"[jvp {label}] {time.perf_counter() - start:.0f} s, finite fraction {finite}", flush=True)
        path = _data_study(f"cgols_sensitivity_avg{TAG}.npz")
    elif MODE == "fd":
        out.update({f"{k}0": np.asarray(v) for k, v in f(theta0).items()})
        for label, unit in (("M", np.array([1.0, 0.0])), ("E", np.array([0.0, 1.0]))):
            for eps in FD_EPS:
                for sign, word in ((1, "plus"), (-1, "minus")):
                    maps_ = f(jnp.asarray(1.0 + sign * eps * unit))
                    out.update({f"fd__{k}__{label}__{word}_{eps:g}": np.asarray(v) for k, v in maps_.items()})
                print(f"[fd {label} eps {eps:g}] {time.perf_counter() - start:.0f} s", flush=True)
        path = _data_study(f"cgols_sensitivity_avg{TAG}_fd.npz")
    elif MODE == "growth":
        direction = jnp.array([1.0, 0.0]) if GROWTH_DIR == "M" else jnp.array([0.0, 1.0])
        state = segment(jnp.asarray(initial_state), theta0, 0.0, WIND_ONSET_MYR)  # no wind yet: zero tangent
        tangent = jnp.zeros_like(state)
        edges = np.arange(WIND_ONSET_MYR, T_END_MYR + 1e-9, 1.0)
        log = {key: [] for key in ("t_myr", "max_dlnrho", "rms_dlnrho", "max_dlnp", "rms_dlnp", "finite")}
        for t0, t1 in itertools.pairwise(edges):
            state, tangent = jax.jvp(lambda s_, th, a=t0, b=t1: segment(s_, th, a, b), (state, theta0), (tangent, direction))
            rho, p = state[rv.density_index], state[rv.pressure_index]
            d_rho = np.asarray(tangent[rv.density_index] / rho)
            d_p = np.asarray(tangent[rv.pressure_index] / p)
            finite = bool(np.isfinite(d_rho).all() and np.isfinite(d_p).all())
            row = (t1, np.nanmax(np.abs(d_rho)), np.sqrt(np.nanmean(d_rho**2)), np.nanmax(np.abs(d_p)),
                   np.sqrt(np.nanmean(d_p**2)), finite)  # fmt: skip
            for key, value in zip(log, row):
                log[key].append(value)
            print(f"[growth {GROWTH_DIR}] t = {t1:g} Myr: max|dln rho| {row[1]:.3e}, rms {row[2]:.3e}, "
                  f"max|dln P| {row[3]:.3e}, rms {row[4]:.3e}, finite {finite} ({time.perf_counter() - start:.0f} s)",
                  flush=True)  # fmt: skip
            if not finite:
                break
        out = {key: np.asarray(v) for key, v in log.items()}
        path = _data_study(f"cgols_sensitivity_growth{TAG}_d{GROWTH_DIR}.npz")
    else:
        raise SystemExit(f"unknown CGOLS_SENS_MODE {MODE!r}")
    np.savez(path, **out, **grid)
    print(f"wrote {path} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
