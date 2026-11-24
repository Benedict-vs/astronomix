"""Differentiable sensitivity of the CGOLS surface-density projection to the
wind's mass loading and energy loading.

This is the differentiability showcase for the CGOLS replication: instead of
re-running the simulation for a grid of feedback strengths, we push a single
forward-mode tangent through the (jitted, adaptive) time integration and read
off, in one pass, how *every pixel* of the edge-on column-density map would
change if the wind's mass / energy injection rate were varied.

The two physical knobs are the CGOLS wind rates
(``params.cgols_wind_params.schedule_{mass,energy}_rates``), which enter the
solver through the native ``_cgols_wind_source`` source term. We parameterise
each as a scalar multiplier on the whole schedule, evaluated at the nominal
value 1.0:

    alpha_M  scales schedule_mass_rates    -> CGOLS mass loading   (Mdot = beta * SFR)
    alpha_E  scales schedule_energy_rates  -> CGOLS energy loading  (Edot = alpha * ... * SFR)

Because the multiplier is applied multiplicatively and evaluated at alpha = 1,
the unit-tangent derivative is exactly d/dln(rate): a clean log-sensitivity.

One scalar in -> a whole 2-D map out is the textbook case for forward-mode AD
(``jax.jvp``); reverse mode would need one pass per pixel. Forward mode works
through the PALLAS backend because every flux kernel is wrapped in
``diffable_pallas_call`` (custom_jvp: Pallas primal, native-JAX tangent), and
``differentiation_mode = FORWARDS`` (the config default) selects the
forward-differentiable ``ADAPTIVE_WHILE`` integrator.

Runs in its own process on a single GPU (like cgols_analyse.py). Nothing in
cgols.py or the solver is modified.

Validity horizon: the *primal* stays finite indefinitely (the positivity clips
keep it stable), but the forward-mode *tangent* eventually overflows to NaN once
the wind cavity gets deeply rarefied - the tangent of v = m/rho grows like
1/rho^2 there, and a single near-floor cell can overflow float32 (cgols runs in
single precision). This onset is later at coarser resolution: empirically ~128^3
is finite to >=20 Myr, whereas 256^3 already NaNs by 25 Myr. The default end time
below is chosen inside that safe window; if a run reports an all-NaN tangent,
lower CGOLS_SENS_TMYR (or CGOLS_DIM). Reverse-mode AD has the identical 1/rho^2
singularity, so it is not a workaround.

Usage:
    # smoke test: just past wind onset, cheap, with the finite-difference check
    CGOLS_DIM=128 CGOLS_SENS_TMYR=8 CGOLS_SENS_FDCHECK=1 python cgols_sensitivity.py
    # production maps: developed outflow (safe window)
    CGOLS_DIM=256 CGOLS_SENS_TMYR=18 python cgols_sensitivity.py
    CGOLS_DIM=128 CGOLS_SENS_TMYR=20 python cgols_sensitivity.py

Env knobs:
    CGOLS_DIM          resolution (needs a matching IC file; default 256 here)
    CGOLS_SENS_TMYR    integration end time in Myr (default 18; see horizon above)
    CGOLS_SENS_FDCHECK 1 = also do a finite-difference spot check (default 0)
    CGOLS_SENS_XRAY    1 = also the edge-on soft X-ray surface brightness S_X and
                       its tangents, in two models (default 0): the CGOLS I recipe
                       (n^2 Lambda_A4(T) inside the hard 0.3-2 keV temperature
                       window) and AtomDB 0.3-1.0 keV photons after Galactic
                       absorption (N_H = 4e20 cm^-2, Z = Z_sun A09; the band of
                       the M82 minor-axis comparison). The AtomDB emissivity is a
                       piecewise-linear table in log T, so its tangent is defined
                       everywhere; the CGOLS window is a step in T, so its tangent
                       misses cells that cross a window edge (the finite-difference
                       check measures how much that matters).
    CGOLS_SENS_FD_EPS  finite-difference steps, comma-separated (default 0.05)
    CGOLS_SENS_FDONLY  1 = only the finite-difference check (writes <tag>_fdcheck.npz)
    CGOLS_SENS_PLOT_ONLY 1 = redraw the figures from the npz of this tag, no run
    CGOLS_FIGURE_TITLES  0 = figures without the figure-level title (paper)
"""

import os
import time

# Single GPU: the forward + tangent working set fits comfortably on one device
# at 128/256, and sharding a jvp adds needless complexity. Set before importing
# cgols, whose autocvd(num_gpus=...) at import time reads CGOLS_SHARD_SPLIT.
os.environ.setdefault("CGOLS_SHARD_SPLIT", "(1, 1, 1, 1)")
# Default this analysis to a low resolution (the supervisor OK'd 256/128); a
# user-provided CGOLS_DIM still wins. Must be set before importing cgols, which
# freezes RESOLUTION from CGOLS_DIM at import time.
os.environ.setdefault("CGOLS_DIM", "256")

# A low-res jvp needs only a few GB, but cgols.py defaults to preallocating 0.95
# of the device (sized for the 512^3 production run) - which OOMs / races on a
# busy shared node. Allocate on demand with a modest cap instead, so this small
# run coexists with other jobs. Must be set before jax initialises its backend
# (i.e. before importing jax / astronomix / cgols); setdefault lets an explicit
# override win, and cgols's own MEM_FRACTION setdefault then sees ours first.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
# Grab the least-used GPU right away rather than blocking for a completely free
# one (this node is usually full); the run is small enough to share a card.
os.environ.setdefault("CGOLS_GPU_LEAST_USED", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
import jax
import jax.numpy as jnp
from astropy import units as u

from astronomix import time_integration
from astronomix.option_classes.simulation_config import ON_DEVICE

# Importing cgols runs autocvd + the cheap module-level constants/functions, but
# not the simulation (guarded behind __main__).
import cgols
from cgols import (
    code_length,
    code_mass,
    code_units,
    load_initial_conditions,
    mu,
    m_p,
    _study_fig,
    _data_study,
    _snapshot_grid,
)

# ruff: noqa: E402

# --- run configuration -----------------------------------------------------
T_END_MYR = float(os.environ.get("CGOLS_SENS_TMYR", "18"))
FDCHECK = os.environ.get("CGOLS_SENS_FDCHECK", "0") == "1"
XRAY = os.environ.get("CGOLS_SENS_XRAY", "0") == "1"
#: steps of the central finite differences (comma-separated), and whether to stop after the check
FD_EPS = tuple(float(e) for e in os.environ.get("CGOLS_SENS_FD_EPS", "0.05").split(","))
FD_ONLY = os.environ.get("CGOLS_SENS_FDONLY", "0") == "1"
PLOT_ONLY = os.environ.get("CGOLS_SENS_PLOT_ONLY", "0") == "1"
TITLES = os.environ.get("CGOLS_FIGURE_TITLES", "1") != "0"
#: AtomDB model of the X-ray maps: the band, absorbing column and metallicity of the
#: minor-axis S_X profiles (cgols_emission.py --xray-band 0.3-1 --xray-photons-nh 4e20).
XRAY_BAND = "0.3-1"
XRAY_NH_CM2 = 4e20
XRAY_METALLICITY = 1.0
RES = int(os.environ["CGOLS_DIM"])
TAG = f"_d{RES}_t{T_END_MYR:g}myr"
# float64 runs (CGOLS_X64=1) get their own files, never overwriting the float32 maps.
if os.environ.get("CGOLS_X64", "0") == "1":
    TAG += "_x64"
# X-ray runs, too: they recompute the Sigma maps, which must not replace the files the manifest points to.
if os.environ.get("CGOLS_SENS_XRAY", "0") == "1":
    TAG += "_xray"

# --- code-density -> physical helpers (module-level equivalents of the ones
#     built inside cgols.make_snapshot_callable) ------------------------------
_code_density_cgs = (code_mass / code_length ** 3).to(u.g / u.cm ** 3).value
_m_p_cgs = m_p.to(u.g).value
N_FACTOR = _code_density_cgs / (mu * _m_p_cgs)   # code density -> n [cm^-3]
_KPC_CM = (1 * u.kpc).to(u.cm).value             # code length (kpc) -> cm


def _sym_limit(a, q=99.0):
    """Robust symmetric color limit for a signed map: the q-th percentile of |a|
    (ignoring non-finite), so a few outlier pixels don't wash out the structure."""
    a = np.asarray(a)
    m = float(np.nanpercentile(np.abs(a[np.isfinite(a)]), q))
    return m if m > 0 else 1.0


def build_pipeline():
    """Load the IC, make the config differentiation-clean, and return everything
    the projection function needs."""
    initial_state, config, params, rv = load_initial_conditions()

    # Differentiation-clean config: no in-place donation (the buffer must survive
    # for AD), and no host-callback side effects (progress bar / diagnostic
    # monitor / snapshot callback / memory analysis) inside the differentiated
    # region. differentiation_mode is already FORWARDS and runtime_debugging
    # already False, so the ADAPTIVE_WHILE (forward-diff) integrator is used with
    # no checkify wrapper. These flags don't affect any finalize-derived field.
    config = config._replace(
        donate_state=False,
        progress_bar=False,
        monitor_diagnostics=False,
        activate_snapshot_callback=False,
        memory_analysis=False,
        # cgols.build_config() now defaults to the Orbax TO_DISK checkpoint
        # driver (host-side segmented loop) - not traceable under jvp, so force
        # the in-memory path here.
        snapshot_storage_mode=ON_DEVICE,
        # Jiang-Shu weights, although production uses WENO-Z: WENO-Z does have
        # a tangent (native and Pallas), but its exact derivative is badly
        # conditioned and turns non-finite in the developed wind (all-NaN by
        # 30 Myr at 128^3 float64; asgard-papers #16). See
        # cgols_sensitivity_avg.py for WENO-Z with frozen weights.
        weno_z=False,
    )

    # Override only the end time; keep the loaded potential, floors and the paper
    # wind schedule from load_initial_conditions().
    t_end = (T_END_MYR * u.Myr).to(code_units.code_time).value
    params = params._replace(t_end=t_end)

    dim_y = config.num_cells.y
    L_y = config.box_size.y
    col_factor = N_FACTOR * (L_y / dim_y) * _KPC_CM   # sum_y rho -> N [cm^-2]
    my = dim_y // 2

    base_wind = params.cgols_wind_params

    if XRAY:
        from astronomix.analysis_helpers.emission import (
            XRAY_TABLE_PATHS,
            EmissionMapSettings,
            emission_conversions,
            emission_maps,
            load_xray_emissivity_table,
        )

        table = load_xray_emissivity_table(XRAY_TABLE_PATHS[XRAY_BAND], photons=True,
                                           absorbing_column_cm2=XRAY_NH_CM2)
        conversions = emission_conversions(code_units)
        settings = EmissionMapSettings(projection_axes=(1,), solar_abundance_set="A09")
        cell_size = float(config.box_size.y) / dim_y

    def xray(fields):
        """Edge-on S_X(x, z) of the final (density, pressure) in the two models (see the module docstring).

        Applied outside the differentiated time integration (exact chain rule on the returned fields and their
        tangents), so it adds nothing to the integration's peak memory."""
        maps = emission_maps(fields[0], fields[1], conversions, table, cell_size, settings)
        return {"sx_cgols": maps["xray_cgols_xz"],
                "sx_atomdb": maps["xray_atomdb_hhe_xz"] + XRAY_METALLICITY * maps["xray_atomdb_metals_xz"]}

    def f(theta):
        """theta = [alpha_M, alpha_E] -> (edge-on column density Sigma(x,z) [cm^-2],
        mid-plane number-density slice n(x,z) [cm^-3]).

        Scaling the whole rate schedule keeps the paper's temporal shape and only
        varies the amplitude, so the derivative at theta = 1 is the sensitivity to
        the mass / energy loading factor."""
        wind = base_wind._replace(
            schedule_mass_rates=base_wind.schedule_mass_rates * theta[0],
            schedule_energy_rates=base_wind.schedule_energy_rates * theta[1],
        )
        p = params._replace(cgols_wind_params=wind)
        final = time_integration(initial_state, config, p, rv)  # unpadded interior
        rho = final[rv.density_index]                           # (dim_x, dim_y, dim_z)
        sigma = jnp.sum(rho, axis=1) * col_factor               # (dim_x, dim_z), cm^-2
        n_mid = rho[:, my, :] * N_FACTOR                        # (dim_x, dim_z), cm^-3
        if XRAY:
            return sigma, n_mid, (rho, final[rv.pressure_index])
        return sigma, n_mid

    return f, (xray if XRAY else None), config, params, rv


def compute_sensitivities(f, xray=None):
    theta0 = jnp.array([1.0, 1.0])
    e_M = jnp.array([1.0, 0.0])
    e_E = jnp.array([0.0, 1.0])

    # jit the directional jvp so the two directions share a single compile
    # (f is closed over -- a function can't be a traced jit argument, but a
    # closure is fine; theta / tangent are the only traced inputs).
    @jax.jit
    def jvp_dir(theta, tangent):
        return jax.jvp(f, (theta,), (tangent,))

    xray_jvp = jax.jit(lambda fields, dfields: jax.jvp(xray, (fields,), (dfields,))) if xray else None
    sx = {}

    def run(direction, label):
        """One direction; with X-ray, map the 3D fields right away and drop them before the next integration."""
        primal, tangent = jvp_dir(theta0, direction)
        if xray:
            sx0, dsx = xray_jvp(primal[2], tangent[2])
            for name in sx0:
                sx[f"{name}0"] = np.asarray(sx0[name])
                sx[f"d{name}_d{label}"] = np.asarray(dsx[name])
            primal, tangent = primal[:2], tangent[:2]
        return jax.tree_util.tree_map(np.asarray, jax.block_until_ready((primal, tangent)))

    # Direction 1 (mass loading): also yields the baseline primal.
    (sigma0, n_mid0), tan_M = run(e_M, "M")
    # Direction 2 (energy loading): reuses the compilation.
    _, tan_E = run(e_E, "E")
    return (sigma0, n_mid0, tan_M[0], tan_M[1], tan_E[0], tan_E[1], sx)


def finite_difference_check(f, xray=None, eps_list=(0.05,), k=300, z=None, halo_z_kpc=1.0):
    """Validate the mass-loading jvp against central finite differences, on the
    top-k highest-|tangent| pixels, for each step in eps_list.

    The high-*column* disk pixels are NOT a meaningful test: the projection
    Sigma = sum_y n is a sum of ~10^21 terms in float32, so the ~O(1%) perturbed
    difference sig_p - sig_m there is swamped by catastrophic cancellation
    (rounding floor ~10^16) and is pure noise. The high-|tangent| pixels near the
    injection region carry the signal well above that floor (SNR ~10^3), so the
    finite difference is trustworthy there. Mass loading (not energy) is used
    because it barely perturbs the CFL timestep, so the two forward runs share
    the adaptive step schedule and no discretisation jump contaminates the fd.

    A map that responds strongly (|dln S/dln Mdot| of several, as S_X near the
    injection region) is nonlinear over a 5 % step, so the central difference is a
    secant there; the error must then fall with eps if the tangent is right. With z
    (cell-centre z [kpc]) the metrics are also given for the halo, |z| > halo_z_kpc.

    Returns (metrics, arrays): metrics[name][region][eps] = dict of numbers; arrays =
    the tangent and the +/- maps of every eps, for the npz."""
    theta0 = jnp.array([1.0, 1.0])
    f_jit = jax.jit(f)

    xray_jit = jax.jit(xray) if xray else None
    xray_jvp = jax.jit(lambda fields, dfields: jax.jvp(xray, (fields,), (dfields,))) if xray else None

    def maps(out, sx=None):
        m = {"sigma": out[0]}
        if xray:
            m.update(sx if sx is not None else xray_jit(out[2]))
        return {name: np.asarray(v, dtype=np.float64) for name, v in m.items()}

    primal, tangent = jax.jit(lambda th: jax.jvp(f, (th,), (jnp.array([1.0, 0.0]),)))(theta0)
    tangent = maps(tangent, xray_jvp(primal[2], tangent[2])[1] if xray else None)
    del primal
    arrays = {f"fdmap__{name}__ad": v for name, v in tangent.items()}
    regions = {"all": None}
    if z is not None:
        regions["halo"] = np.broadcast_to(np.abs(np.asarray(z))[None, :] > halo_z_kpc,
                                          next(iter(tangent.values())).shape).ravel()
    metrics = {name: {region: {} for region in regions} for name in tangent}
    for eps in eps_list:
        plus = maps(f_jit(jnp.array([1.0 + eps, 1.0])))
        minus = maps(f_jit(jnp.array([1.0 - eps, 1.0])))
        for name in tangent:
            arrays[f"fdmap__{name}__plus_{eps:g}"] = plus[name]
            arrays[f"fdmap__{name}__minus_{eps:g}"] = minus[name]
            ad_all = tangent[name].ravel()
            fd_all = ((plus[name] - minus[name]) / (2 * eps)).ravel()
            for region, mask in regions.items():
                ad = ad_all if mask is None else np.where(mask, ad_all, 0.0)
                idx = np.argsort(np.abs(ad))[-k:]          # top-k by |analytic tangent| in the region
                a, d = ad_all[idx], fd_all[idx]
                rel = np.abs(a - d) / np.maximum(np.abs(a), np.abs(d))
                corr = float(np.corrcoef(a, d)[0, 1])
                slope = float(np.polyfit(a, d, 1)[0])      # fd ~ slope * ad, expect ~1
                m = {"median_rel_err": float(np.median(rel)), "corr": corr, "slope": slope}
                metrics[name][region][eps] = m
                print(f"[fd-check] {name} {region}: mass-loading jvp vs central diff (eps={eps:g}, top-{k} "
                      f"|tangent| px): median rel.err {m['median_rel_err']:.2%}, corr {corr:.4f}, "
                      f"slope(fd/ad) {slope:.3f}")
    return metrics, arrays


def plot_maps(sigma0, n_mid0, dsig_dM, dsig_dE, extent_xz, out):
    """2x3: top = column density + absolute sensitivities; bottom = mid-plane
    slice + relative (log) sensitivities."""
    # Relative sensitivity d ln Sigma / d ln(rate); guard the near-empty halo.
    floor = np.nanpercentile(sigma0[sigma0 > 0], 5) if np.any(sigma0 > 0) else 1.0
    safe = np.where(sigma0 > floor, sigma0, np.nan)
    rel_M = dsig_dM / safe
    rel_E = dsig_dE / safe

    lim_absM, lim_absE = _sym_limit(dsig_dM), _sym_limit(dsig_dE)
    lim_relM, lim_relE = _sym_limit(rel_M), _sym_limit(rel_E)

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 11), constrained_layout=True)

    def _show(a, data, *, norm=None, cmap="viridis", vmin=None, vmax=None,
              title="", cblabel=""):
        im = a.imshow(data.T, origin="lower", extent=extent_xz, aspect="equal",
                      cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        a.set_title(title, fontsize=10)
        a.set_xlabel("x [kpc]", fontsize=8)
        a.set_ylabel("z [kpc]", fontsize=8)
        a.tick_params(labelsize=7)
        fig.colorbar(im, ax=a, shrink=0.8, label=cblabel)

    _show(ax[0, 0], np.clip(sigma0, 1e-30, None),
          norm=LogNorm(vmin=max(np.nanpercentile(sigma0[sigma0 > 0], 5), 1e-30),
                       vmax=np.nanmax(sigma0)),
          title=r"baseline column density  $\Sigma=\int n\,dy$",
          cblabel=r"$N$ [cm$^{-2}$]")
    _show(ax[0, 1], dsig_dM, cmap="RdBu_r", vmin=-lim_absM, vmax=lim_absM,
          title=r"$\partial\Sigma/\partial\,$(mass loading)",
          cblabel=r"$\partial N/\partial\ln\dot M$ [cm$^{-2}$]")
    _show(ax[0, 2], dsig_dE, cmap="RdBu_r", vmin=-lim_absE, vmax=lim_absE,
          title=r"$\partial\Sigma/\partial\,$(energy loading)",
          cblabel=r"$\partial N/\partial\ln\dot E$ [cm$^{-2}$]")

    _show(ax[1, 0], np.clip(n_mid0, 1e-30, None),
          norm=LogNorm(vmin=max(np.nanpercentile(n_mid0[n_mid0 > 0], 5), 1e-30),
                       vmax=np.nanmax(n_mid0)),
          title=r"baseline mid-plane slice  $n(x,z)$",
          cblabel=r"$n$ [cm$^{-3}$]")
    _show(ax[1, 1], rel_M, cmap="RdBu_r", vmin=-lim_relM, vmax=lim_relM,
          title=r"relative: $\partial\ln\Sigma/\partial\ln\dot M$",
          cblabel="dimensionless")
    _show(ax[1, 2], rel_E, cmap="RdBu_r", vmin=-lim_relE, vmax=lim_relE,
          title=r"relative: $\partial\ln\Sigma/\partial\ln\dot E$",
          cblabel="dimensionless")

    if TITLES:
        fig.suptitle(
            f"CGOLS outflow: surface-density sensitivity to feedback loading "
            f"(edge-on, {RES}$^3$, t = {T_END_MYR:g} Myr)", fontsize=12)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_xray_maps(sx, extent_xz, out, model="sx_atomdb", rel_lim=None):
    """1x3: S_X and its two relative (log) sensitivities, for one X-ray model."""
    s0, dM, dE = sx[f"{model}0"], sx[f"d{model}_dM"], sx[f"d{model}_dE"]
    floor = np.nanpercentile(s0[s0 > 0], 50) if np.any(s0 > 0) else 1.0   # the brighter half of the emitting pixels
    safe = np.where(s0 > floor, s0, np.nan)
    rel_M, rel_E = dM / safe, dE / safe
    lim = rel_lim or max(_sym_limit(rel_M), _sym_limit(rel_E))
    label = {"sx_atomdb": r"$S_X$, absorbed 0.3-1 keV [photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]",
             "sx_cgols": r"$S_X$, CGOLS I recipe [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]"}[model]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 6.4), constrained_layout=True)
    panels = (
        (np.clip(s0, 1e-30, None), {"cmap": "magma", "norm": LogNorm(vmin=max(floor * 1e-2, 1e-30), vmax=np.nanmax(s0))},
         r"soft X-ray surface brightness $S_X$", label),
        (rel_M, {"cmap": "RdBu_r", "vmin": -lim, "vmax": lim}, r"$\partial\ln S_X/\partial\ln\dot M$", "dimensionless"),
        (rel_E, {"cmap": "RdBu_r", "vmin": -lim, "vmax": lim}, r"$\partial\ln S_X/\partial\ln\dot E$", "dimensionless"),
    )
    for a, (data, kw, title, cblabel) in zip(ax, panels):
        im = a.imshow(data.T, origin="lower", extent=extent_xz, aspect="equal", **kw)
        a.set_title(title, fontsize=10)
        a.set_xlabel("x [kpc]", fontsize=8)
        a.set_ylabel("z [kpc]", fontsize=8)
        a.tick_params(labelsize=7)
        fig.colorbar(im, ax=a, shrink=0.8, label=cblabel)
    if TITLES:
        fig.suptitle(f"CGOLS outflow: X-ray sensitivity to feedback loading ({model}, edge-on, {RES}$^3$, "
                     f"t = {T_END_MYR:g} Myr)", fontsize=12)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_profiles(n_mid0, dn_dM, dn_dE, x, z, out, x_targets=(0.0, 0.5, 1.0, 2.0)):
    """rho(z) at several x positions (baseline) + the line sensitivities
    dn/dln(rate) at the same x, tying the 1-D profiles to the gradient theme."""
    ixs = [int(np.argmin(np.abs(x - xt))) for xt in x_targets]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ixs)))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for c, ix, xt in zip(colors, ixs, x_targets):
        label = f"x = {x[ix]:.2f} kpc"
        ax[0].semilogy(z, np.clip(n_mid0[ix], 1e-30, None), color=c, lw=1.6, label=label)
        ax[1].plot(z, dn_dM[ix], color=c, lw=1.6, label=label)
        ax[2].plot(z, dn_dE[ix], color=c, lw=1.6, label=label)

    ax[0].set_title(r"baseline  $n(z)$")
    ax[0].set_ylabel(r"$n$ [cm$^{-3}$]")
    ax[1].set_title(r"$\partial n/\partial\ln\dot M$  (mass loading)")
    ax[1].set_ylabel(r"$\partial n/\partial\ln\dot M$ [cm$^{-3}$]")
    ax[2].set_title(r"$\partial n/\partial\ln\dot E$  (energy loading)")
    ax[2].set_ylabel(r"$\partial n/\partial\ln\dot E$ [cm$^{-3}$]")
    for a in ax:
        a.set_xlabel("z [kpc]")
        a.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
        a.legend(fontsize=8, frameon=False)
    for a in (ax[1], ax[2]):
        a.axhline(0.0, color="0.4", lw=0.8)

    if TITLES:
        fig.suptitle(
            f"CGOLS outflow: vertical density profiles & feedback sensitivity "
            f"(mid-plane y-slice, t = {T_END_MYR:g} Myr)", fontsize=12)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    print(f"CGOLS sensitivity run: DIM={RES}, t_end={T_END_MYR:g} Myr, "
          f"fd_check={FDCHECK}")
    npz_path = _data_study(f"cgols_sensitivity{TAG}.npz")
    if PLOT_ONLY:
        d = np.load(npz_path)
        extent_xz = tuple(d["extent_xz"])
        plot_maps(d["sigma0"], d["n_mid0"], d["dsig_dM"], d["dsig_dE"], extent_xz,
                  _study_fig(f"sensitivity_maps{TAG}.png"))
        plot_profiles(d["n_mid0"], d["dn_dM"], d["dn_dE"], d["x"], d["z"],
                      _study_fig(f"sensitivity_profiles{TAG}.png"))
        sx = {k: d[k] for k in d.files if "sx_" in k}
        for model in ("sx_atomdb", "sx_cgols"):
            if f"{model}0" in sx:
                plot_xray_maps(sx, extent_xz, _study_fig(f"sensitivity_xray_{model[3:]}{TAG}.png"), model)
        raise SystemExit(0)

    f, xray, config, params, rv = build_pipeline()
    x, y, z, extent_xz, _ = _snapshot_grid(config)

    fd_result, fd_arrays = (finite_difference_check(f, xray, FD_EPS, z=np.asarray(z)) if FDCHECK
                            else (None, {}))
    if FD_ONLY:
        extra = {f"fdcheck__{name}__{region}__eps{eps:g}__{key}": np.float64(v)
                 for name, by_region in fd_result.items() for region, by_eps in by_region.items()
                 for eps, r in by_eps.items() for key, v in r.items()}
        fd_path = _data_study(f"cgols_sensitivity{TAG}_fdcheck.npz")
        np.savez(fd_path, **fd_arrays, **extra, z=np.asarray(z), x=np.asarray(x))
        print(f"Wrote {fd_path}")
        raise SystemExit(0)

    print("Running forward-mode jvp through the solver (compiles on first pass)...")
    t_start = time.perf_counter()
    sigma0, n_mid0, dsig_dM, dn_dM, dsig_dE, dn_dE, sx = compute_sensitivities(f, xray)
    print(f"[timing] jvp in both directions incl. compile: {time.perf_counter() - t_start:.0f} s")
    for name, v in sx.items():
        print(f"  {name}: finite {np.isfinite(v).mean():.4f}, max |.| {np.nanmax(np.abs(v)):.3e}")

    # Tangent-overflow guard: if the linearisation has gone NaN everywhere the
    # cavity has out-run the safe window (see the module docstring). Fail loudly
    # with guidance rather than write all-NaN figures. The primal is still fine.
    if not np.isfinite(dsig_dM).any():
        raise SystemExit(
            f"Tangent is all-NaN at DIM={RES}, t_end={T_END_MYR:g} Myr: the wind "
            f"cavity is too rarefied for the forward-mode linearisation (v=m/rho "
            f"tangent ~1/rho^2 overflows float32). The primal is finite "
            f"(max Sigma {np.nanmax(sigma0):.2e} cm^-2). Lower CGOLS_SENS_TMYR "
            f"(128^3 is safe to ~20 Myr, 256^3 to <~20 Myr) and/or CGOLS_DIM."
        )

    # Wiring / physics sanity: with the wind off (t_end < 5 Myr) the sensitivity
    # must be ~0 everywhere; once feedback is on it is not.
    print(f"max |dSigma/dln Mdot| = {np.nanmax(np.abs(dsig_dM)):.3e} cm^-2")
    print(f"max |dSigma/dln Edot| = {np.nanmax(np.abs(dsig_dE)):.3e} cm^-2")
    if T_END_MYR < 5.0:
        print("  (t_end < 5 Myr: wind is OFF, so both should be ~0 -- AD wiring check)")

    plot_maps(sigma0, n_mid0, dsig_dM, dsig_dE, extent_xz,
              _study_fig(f"sensitivity_maps{TAG}.png"))
    plot_profiles(n_mid0, dn_dM, dn_dE, np.asarray(x), np.asarray(z),
                  _study_fig(f"sensitivity_profiles{TAG}.png"))

    for model in ("sx_atomdb", "sx_cgols"):
        if f"{model}0" in sx:
            plot_xray_maps(sx, extent_xz, _study_fig(f"sensitivity_xray_{model[3:]}{TAG}.png"), model)
    extra = {}
    if fd_result:
        extra = {f"fdcheck__{name}__{region}__eps{eps:g}__{key}": np.float64(v)
                 for name, by_region in fd_result.items() for region, by_eps in by_region.items()
                 for eps, r in by_eps.items() for key, v in r.items()}
    np.savez(
        npz_path,
        **sx, **extra,
        sigma0=sigma0, n_mid0=n_mid0,
        dsig_dM=dsig_dM, dsig_dE=dsig_dE, dn_dM=dn_dM, dn_dE=dn_dE,
        x=np.asarray(x), y=np.asarray(y), z=np.asarray(z),
        extent_xz=np.asarray(extent_xz),
        t_end_myr=np.float64(T_END_MYR), resolution=np.int32(RES),
    )
    print(f"Wrote {npz_path}")
