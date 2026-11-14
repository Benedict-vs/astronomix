"""
Cost against accuracy of the CGOLS A-series runs: which scheme (FD WENO5 / FV VL2) and resolution reaches a given
accuracy in which quantity for the fewest GPU-hours.

Reference: per quantity the mean R of FD-2048 and FV-2048; its uncertainty u = |FD-2048 - FV-2048| / 2 (the two
best runs disagree by that much, so no run can be judged more precisely). Error of a run: e = (x - R) / |R|
(percentage points for the CC85 velocity offset, percent density for the CC85 log-density offset). Two kinds of
quantity are distances instead (always >= 0, in percent): the time-resolved errors (mean |x(t) - R(t)| / mean |R(t)|
over the window, R(t) the mean 2048 series; catches timing shifts that window means average out) and the phase
structure (total-variation distance of the normalised (log n, log T) mass histogram above 1 kpc from the mean 2048
histogram). A run reaches a tolerance tol "confidently" if |e| + u <= tol, "nominally" if only |e| <= tol, and in
the family verdict only if every finer run of its scheme below 2048 reaches it too (no pass by a chance crossing).
The 2048 runs are part of the reference: their error is u by construction. A quantity with u >= tol cannot be
decided at that tolerance; the family verdict at tol uses the decidable quantities and lists the others.

Noise sigma of a run: the scatter of the seed runs at its scheme and N where there are >= 2 (measured), else the
largest relative scatter measured at <= N (extrapolated), else the relative scatter of a frame metric (borrowed, as
cgols_fvfd_synthesis does for 3D metrics); never below the time-sampling error of the window mean. A cell is starred
when |e| <= 2 sigma: one run cannot tell it from the reference.

Second reference (columns FD->inf, FV->inf): where a scheme's 512-1024-2048 values change monotonically, the first
change exceeds 2 sigma of the 1024 run and the second is at most half the first (order p >= 1), its Richardson
extrapolation v_2048 + d2 / (2^p - 1), p = log2(d1 / d2), shown as its error
against R. Both close to 0: R is near the converged value; both far: R itself is still resolution dependent.

Families (sources: cgols_fvfd_synthesis cache = frames / emission / 3D; cgols_cost_accuracy_extra cache = 3D window
means at each run's own snapshot cadence, frame slices and maps, emission profiles):
  mass outflow 5 kpc       outflow windows, peak, cumulative, side loss; time-resolved outflow
  outflow by phase 1 kpc   wind / mixed / entrained+mixed outflow at 1 kpc, mixed mass; time-resolved mixed outflow
  energy flux              Edot at 1 and 5 kpc, cumulative energy at 5 kpc; time-resolved Edot 5 kpc
  momentum flux            outward momentum flux at 1 and 5 kpc; time-resolved
  emission                 L_X1, L_X2, L_Halpha, S_X1 profile at 2-5 and 5-9 kpc; time-resolved L_X1
  observables at 156 pc    clumpiness (std / mean) of the X-ray and H-alpha maps block-averaged to 156 pc
  wind structure           cone half-angle, axis vs CC85, core share of the outflow, fast-gas fraction
  phase structure          warm share of the gas above 0.5 kpc, distance of the n-T histogram
  disk and halo            disk mass and box mass at 75 Myr, shell-breakout time
  small-scale morphology   up-down asymmetry, small-scale density variance, cavity-rim roughness, slice power at
                           100-300 pc
  mixed gas in 3D          mass and volume of the mixed gas (5.5 < log K < 6.8) in 1 < |z| < 4.5 kpc
  mixing layer and clumps  log K = 6.8 isosurface area, mixed-layer thickness V/A, entrained gas (log K < 5.5):
                           total mass, mass and number of clumps detached from the disk
  (both from cgols_structure_3d caches data/studies/structure3d_v2<TAG>.npz, 2.5 Myr checkpoint cadence)

Scale-by-scale convergence (cost_accuracy_spectra.png): the 30-45 Myr power spectrum of log n in the xz slice
(0.5 < |z| < 4.5 kpc) of every base run divided by the same scheme at twice the resolution and by the 2048
reference; the converged scale l_c is the smallest scale down to which the ratio stays within 20 %.

Cost: solver GPU-hours (H200) to 75 Myr, cgols_fvfd_synthesis.GPU_HOURS (time integration only; FD-1024 = job 93951,
8 H200, 124 GPU-h).

    python cgols_cost_accuracy.py [--tols 5,10,20] [--json OUT] [--recompute-extra]

Writes figures/cgols/comparisons/cost_accuracy_grid.png (error of every quantity for every run, runs ordered by
cost), cost_accuracy_families.png (worst error per family against GPU-hours) and cost_accuracy_spectra.png, and
prints the tables. CPU only.
"""

# ruff: noqa: E402

# general
import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# numerics
import numpy as np

# plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

# CGOLS analysis
import cgols_cost_accuracy_extra as extra
import cgols_fvfd_synthesis as syn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures", "cgols", "comparisons")
REF_N = 2048
SCHEMES = ("FD", "FV")
NS = (128, 256, 512, 1024, 2048)
SPEC_TOL = 0.2

#: family -> [(key, label)]. Keys: synthesis METRICS, extra scalars (x:<key>), time-resolved (ts:<series>:a:b),
#: histogram distance (tv:<window>), rim roughness (rim_20_35), slice power (pow_100_300),
#: 3D structure window means from cgols_structure_3d caches (s3:<scalar>:a:b, scalars of cgols_structure_3d_plots)
FAMILIES = {
    "mass outflow 5 kpc": [
        ("out5_15_40", "outflow 15-40 Myr"), ("out5_40_50", "outflow 40-50 Myr"),
        ("out5_50_75", "outflow 50-75 Myr"), ("out5_peak", "peak outflow"), ("mout5_cum", "mass out 0-75 Myr"),
        ("side_loss", "side-face mass loss"), ("ts:out5:15:40", "outflow(t) 15-40 Myr"),
        ("ts:out5:40:75", "outflow(t) 40-75 Myr")],
    "outflow by phase 1 kpc": [
        ("x:n_wind1_15_40", "wind 15-40 Myr"), ("x:n_mixed1_15_30", "mixed 15-30 Myr"),
        ("x:n_mixed1_30_45", "mixed 30-45 Myr"), ("x:n_entmixed1_50_75", "entr.+mixed 50-75 Myr"),
        ("mixedmass_t60", "mixed mass in halo, 60 Myr"), ("ts:mixed1:15:45", "mixed(t) 15-45 Myr")],
    "energy flux": [
        ("x:n_edot5_15_40", "5 kpc, 15-40 Myr"), ("x:n_edot5_50_75", "5 kpc, 50-75 Myr"),
        ("ecum5", "cumulative 5 kpc"), ("x:n_edot1_15_40", "1 kpc, 15-40 Myr"),
        ("x:n_edot1_50_75", "1 kpc, 50-75 Myr"), ("ts:edot5:15:75", "Edot 5 kpc(t) 15-75")],
    "momentum flux": [
        ("x:n_pdot5_15_40", "5 kpc, 15-40 Myr"), ("x:n_pdot5_50_75", "5 kpc, 50-75 Myr"),
        ("x:n_pdot1_15_40", "1 kpc, 15-40 Myr"), ("x:n_pdot1_50_75", "1 kpc, 50-75 Myr"),
        ("ts:pdot5:15:75", "pdot 5 kpc(t) 15-75")],
    "emission": [
        ("lx1_15_40", "L_X1 15-40 Myr"), ("lx1_60_75", "L_X1 60-75 Myr"), ("lx2_15_40", "L_X2 15-40 Myr"),
        ("lx2_60_75", "L_X2 60-75 Myr"), ("ha_15_40", "H-alpha 15-40 Myr"), ("ha_60_75", "H-alpha 60-75 Myr"),
        ("x:sx_2_5_15_40", "S_X1 at 2-5 kpc, 15-40"), ("x:sx_5_9_60_75", "S_X1 at 5-9 kpc, 60-75"),
        ("ts:lx1:15:75", "L_X1(t) 15-75 Myr")],
    "observables at 156 pc": [
        ("x:clumpx_15_40", "X-ray clumpiness 15-40"), ("x:clumpx_60_75", "X-ray clumpiness 60-75"),
        ("x:clumph_15_40", "H-alpha clumpiness 15-40"), ("x:clumph_60_75", "H-alpha clumpiness 60-75")],
    "wind structure": [
        ("x:n_cone1_20_40", "cone angle 20-40 Myr"), ("x:n_cone1_60_75", "cone angle 60-75 Myr"),
        ("cc85_dv", "axis v vs CC85 [pp]"), ("cc85_dlogn", "axis n vs CC85"),
        ("x:core1_15_40", "core share R<1 kpc, z=1"), ("x:core5_15_40", "core share R<1 kpc, z=5"),
        ("x:fast1_15_40", "fast-gas fraction 15-40"), ("x:fast1_50_75", "fast-gas fraction 50-75")],
    "phase structure": [
        ("x:warmfrac_15_40", "warm share 15-40 Myr"), ("x:warmfrac_50_75", "warm share 50-75 Myr"),
        ("tv:15_40", "n-T distribution 15-40"), ("tv:50_75", "n-T distribution 50-75")],
    "disk and halo": [
        ("x:disk_75", "disk mass at 75 Myr"), ("mass_75", "box mass at 75 Myr"),
        ("x:breakout", "shell-breakout time")],
    "small-scale morphology": [
        ("asym_50_75", "up-down asymmetry 50-75"), ("ssv256", "small-scale var. 80-300 pc"),
        ("ssv128", "small-scale var. 160-600 pc"), ("rim_20_35", "cavity-rim roughness 20-35"),
        ("pow_100_300", "slice power 100-300 pc")],
    "mixed gas in 3D, 1-4.5 kpc": [
        ("s3:mass_mixed:15:30", "mixed mass 15-30 Myr"), ("s3:mass_mixed:30:45", "mixed mass 30-45 Myr"),
        ("s3:mass_mixed:45:75", "mixed mass 45-75 Myr"), ("s3:vol_mixed:15:30", "mixed volume 15-30 Myr"),
        ("s3:vol_mixed:45:75", "mixed volume 45-75 Myr")],
    "mixing layer and clumps in 3D": [
        ("s3:area_wind:30:45", "mixing-layer area 30-45"), ("s3:area_wind:45:75", "mixing-layer area 45-75"),
        ("s3:thickness:45:75", "mixing-layer thickness 45-75"), ("s3:mass_ent:45:75", "entrained mass 45-75"),
        ("s3:mass_det:45:75", "detached-clump mass 45-75"), ("s3:n_det:45:75", "detached-clump number 45-75")],
}
#: structure3d cache tag per synthesis label where it differs from the label's frame tag (FV1024 = the
#: bit-identical rerun A1024fvr, the original kept only 4 checkpoints)
S3_TAG = {"FV1024": "_A1024fvr"}
#: relative seed scatter borrowed from these synthesis frame metrics when nothing is measured
BORROW_EARLY, BORROW_LATE = "out5_15_40", "out5_50_75"
RIM_WINDOW = (20, 35)
POW_BAND = (1 / 0.3, 1 / 0.1)  # cycles / kpc: 100-300 pc


def is_distance(key):
    return key.startswith(("ts:", "tv:"))


def window_of(key):
    """(start, end) Myr of a key, for picking the borrowed scatter."""
    if key == "pow_100_300":
        return 30.0, 45.0
    if key.startswith("s3:"):
        return float(key.split(":")[2]), float(key.split(":")[3])
    parts = key.replace("x:", "").split("_")
    try:
        return float(parts[-2]), float(parts[-1])
    except (ValueError, IndexError):
        return (50.0, 75.0) if key in ("x:disk_75", "x:breakout", "mass_75") else (15.0, 40.0)


# ---------------------------------------------------------------- collection


def members_of(runs, scheme, n):
    return [lab for lab, s in runs.items() if s[0] == scheme and s[1] == n and s[2] in ("base", "ens")]


def base_of(runs, scheme, n):
    b = [lab for lab, s in runs.items() if s[:3] == (scheme, n, "base")]
    return b[0] if b else None


def sigma_table(runs, values, st, key):
    """{(scheme, N): sigma} from member values {label: (value, se)}: measured / extrapolated / borrowed, >= se."""
    rel_meas = {}
    for scheme in SCHEMES:
        for n in NS:
            vals = [values[lab][0] for lab in members_of(runs, scheme, n) if lab in values]
            b = base_of(runs, scheme, n)
            if len(vals) >= 2 and b in values and values[b][0]:
                rel_meas[(scheme, n)] = np.std(vals, ddof=1) / abs(values[b][0])
    twin = BORROW_LATE if window_of(key)[0] >= 45 else BORROW_EARLY
    out = {}
    for scheme in SCHEMES:
        for n in NS:
            b = base_of(runs, scheme, n)
            if b not in values:
                continue
            v, se = values[b]
            if (scheme, n) in rel_meas:
                rel = rel_meas[(scheme, n)]
            elif any(m <= n for (_s, m) in rel_meas):
                rel = max(r for (_s, m), r in rel_meas.items() if m <= n)
            else:
                t = st.get(twin, {}).get((scheme, n))
                rel = t["sig_traj"] / abs(t["value"]) if t and t.get("sig_traj") and t["value"] else np.nan
            parts = [p for p in (rel * abs(v), se) if p is not None and np.isfinite(p)]
            out[(scheme, n)] = max(parts) if parts else np.nan
    return out


def rim_values(runs):
    from cgols_convergence_3d import rim_metrics

    out = {}
    for label, spec in runs.items():
        if spec[2] not in ("base", "ens") or not os.path.isdir(spec[3]):
            continue
        t, _r, scatter, _m = rim_metrics(os.path.join(SCRIPT_DIR, spec[3]))
        sel = syn.window(t, *RIM_WINDOW)
        if sel.sum() >= 3:
            out[label] = (float(np.mean(scatter[sel])), None)
    return out


def structure_values(runs, key):
    """{label: (window mean, time-sampling standard error)} of a structure3d scalar (s3:<name>:a:b)."""
    import cgols_structure_3d_plots as s3p

    _p, name, a, b = key.split(":")
    out = {}
    for label, spec in runs.items():
        if spec[2] not in ("base", "ens"):
            continue
        tag = S3_TAG.get(label, spec[4])
        recs, version = s3p.load(tag) if tag else (None, None)
        if not recs:
            continue
        member = {"tag": tag, "recs": recs, "version": version}
        sel = s3p.window_records(member, (float(a), float(b)))
        if sel is None or (name in s3p.V2_ONLY and version != "v2"):
            continue
        y = np.array([s3p.SCALARS[name][1](r) for r in sel])
        out[label] = (float(y.mean()), float(y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 1 else None)
    return out


def band_power(ex):
    spec = ex.get("spectrum")
    if spec is None:
        return None
    sel = (extra.K_MID >= POW_BAND[0]) & (extra.K_MID <= POW_BAND[1]) & (extra.K_MID <= ex["nyquist"])
    return (float(np.nansum(spec[sel] * extra.K_MID[sel])), None) if sel.any() else None


def collect():
    """{key: {"cells": {(scheme, N): {"value", "sigma"}}, "members": {label: value}}} for scalar keys, and the raw
    per-run objects for the distance keys."""
    os.chdir(SCRIPT_DIR)
    runs, data = syn.load_all(False, {})
    st = syn.stats(runs, data)
    ex = extra.load_all()
    scal = {}
    for items in FAMILIES.values():
        for key, _ in items:
            if is_distance(key):
                continue
            if key in st:
                scal[key] = {"cells": {c: {"value": v["value"], "sigma": v["sigma"]} for c, v in st[key].items()}}
                continue
            if key == "rim_20_35":
                values = rim_values(runs)
            elif key.startswith("s3:"):
                values = structure_values(runs, key)
            elif key == "pow_100_300":
                values = {lab: band_power(e) for lab, e in ex.items()}
                values = {k: v for k, v in values.items() if v}
            else:
                name = key[2:]
                values = {lab: e["scalars"][name] for lab, e in ex.items() if name in e["scalars"]}
            sig = sigma_table(runs, values, st, key)
            cells = {}
            for scheme in SCHEMES:
                for n in NS:
                    b = base_of(runs, scheme, n)
                    if b in values:
                        cells[(scheme, n)] = {"value": values[b][0], "sigma": sig.get((scheme, n), np.nan)}
            scal[key] = {"cells": cells}
    return runs, data, ex, scal


# ---------------------------------------------------------------- errors


def error(key, x, ref):
    if key == "cc85_dv":
        return 100.0 * (x - ref)
    if key == "cc85_dlogn":
        return 100.0 * (10 ** (x - ref) - 1.0)
    return 100.0 * (x - ref) / abs(ref)


def scale(key, ref):
    if key == "cc85_dv":
        return 100.0
    if key == "cc85_dlogn":
        return 100.0 * np.log(10.0)
    return 100.0 / abs(ref)


def richardson(key, cells, ref):
    """Error [%] against R of each scheme's Richardson value over 512-1024-2048 (None if not monotone)."""
    out = {}
    for scheme in SCHEMES:
        if not all((scheme, n) in cells for n in (512, 1024, 2048)):
            out[scheme] = None
            continue
        v = [cells[(scheme, n)]["value"] for n in (512, 1024, 2048)]
        d1, d2 = v[1] - v[0], v[2] - v[1]
        # the first step must be above the noise (2 sigma of the 1024 run) and the ladder at least first order
        # (|d2| <= |d1| / 2, correction <= |d2|); otherwise noise in d2 / d1 gives arbitrarily large corrections
        sig = cells[(scheme, 1024)].get("sigma", np.nan)
        real = not np.isfinite(sig) or abs(d1) > 2 * sig
        if real and np.sign(d1) == np.sign(d2) and abs(d2) <= 0.5 * abs(d1):
            p = np.log2(abs(d1) / abs(d2)) if d2 != 0 else np.inf
            out[scheme] = error(key, v[2] + (d2 / (2**p - 1) if np.isfinite(p) else 0.0), ref)
        else:
            out[scheme] = None
    return out


def series_of(runs, data, ex, label, name):
    """(t, y) of a run's time series: synthesis frame series (out5, lx1) or 3D series (mixed1, edot5, pdot5)."""
    if name in ("out5", "lx1"):
        s = data.get(label, {}).get("series", {})
        if name in s and "t" in s:
            t, y = np.asarray(s["t"], float), np.asarray(s[name], float)
            return (t, y) if len(t) == len(y) else None
        return None
    s = ex.get(label, {}).get("series", {})
    return (np.asarray(s["t"]), np.asarray(s[name])) if name in s else None


def ts_distance(a, b, ref, t0, t1):
    grid = np.arange(t0, t1 + 1e-6, 1.25)
    ya = np.interp(grid, *a)
    yr = 0.5 * (np.interp(grid, *ref[0]) + np.interp(grid, *ref[1])) if b is None else np.interp(grid, *b)
    ok = np.isfinite(ya) & np.isfinite(yr)
    return 100.0 * np.mean(np.abs(ya[ok] - yr[ok])) / np.mean(np.abs(yr[ok]))


def tv_distance(h, href):
    return 100.0 * 0.5 * np.abs(h - href).sum()


def distance_cells(key, runs, data, ex):
    """{(scheme, N): {"value": distance [%], "sigma"}}, u [%] for a time-resolved / histogram key."""
    fd2, fv2 = base_of(runs, "FD", REF_N), base_of(runs, "FV", REF_N)
    if key.startswith("ts:"):
        _, name, a, b = key.split(":")
        t0, t1 = float(a), float(b)
        ref = (series_of(runs, data, ex, fd2, name), series_of(runs, data, ex, fv2, name))
        if ref[0] is None or ref[1] is None:
            return None, None

        def dist(label):
            s = series_of(runs, data, ex, label, name)
            return ts_distance(s, None, ref, t0, t1) if s is not None and s[0].max() >= t1 - 1 else None
    else:
        w = key.split(":")[1]
        h2 = [ex.get(lab, {}).get("hists", {}).get(w) for lab in (fd2, fv2)]
        if h2[0] is None or h2[1] is None:
            return None, None
        href = 0.5 * (h2[0] + h2[1])

        def dist(label):
            h = ex.get(label, {}).get("hists", {}).get(w)
            return tv_distance(h, href) if h is not None else None
    cells = {}
    for scheme in SCHEMES:
        for n in NS:
            b = base_of(runs, scheme, n)
            if b is None:
                continue
            d = dist(b)
            if d is None:
                continue
            mem = [dist(lab) for lab in members_of(runs, scheme, n)]
            mem = [m for m in mem if m is not None]
            cells[(scheme, n)] = {"value": d, "sigma": float(np.std(mem, ddof=1)) if len(mem) >= 2 else np.nan}
    u = cells.get(("FD", REF_N), {}).get("value")
    return cells, u


def analyse(runs, data, ex, scal):
    result = {}
    for family, items in FAMILIES.items():
        for key, label in items:
            if is_distance(key):
                cells, u = distance_cells(key, runs, data, ex)
                if not cells or u is None:
                    print(f"  [skip] {key}: no 2048 pair")
                    continue
                row = {"family": family, "label": label, "ref": 0.0, "u": u, "distance": True, "runs": {},
                       "rich": {"FD": None, "FV": None}}
                for cell, c in cells.items():
                    if cell in syn.GPU_HOURS:
                        row["runs"][cell] = {"value": c["value"], "err": c["value"], "sigma": c["sigma"],
                                             "cost": syn.GPU_HOURS[cell]}
                result[key] = row
                continue
            cells = scal.get(key, {}).get("cells", {})
            if ("FD", REF_N) not in cells or ("FV", REF_N) not in cells:
                print(f"  [skip] {key}: no 2048 pair")
                continue
            fd, fv = cells[("FD", REF_N)]["value"], cells[("FV", REF_N)]["value"]
            ref = 0.5 * (fd + fv)
            k = scale(key, ref)
            row = {"family": family, "label": label, "ref": ref, "u": 0.5 * abs(fd - fv) * k, "distance": False,
                   "runs": {}, "rich": richardson(key, cells, ref)}
            for cell, c in cells.items():
                if cell[1] not in NS or cell not in syn.GPU_HOURS:
                    continue
                row["runs"][cell] = {"value": c["value"], "err": error(key, c["value"], ref),
                                     "sigma": c["sigma"] * k if np.isfinite(c["sigma"]) else np.nan,
                                     "cost": syn.GPU_HOURS[cell]}
            result[key] = row
    return result


def cheapest(result, tol):
    """Per family: the cheapest run whose worst error is <= tol, confidently and nominally, over the quantities the
    reference can decide at this tolerance (u < tol); the others are listed as undecidable. A run below REF_N only
    counts if every finer run of its scheme below REF_N passes too: a coarse run whose value happens to swing through
    the reference while finer runs are further off is not converged."""
    out = {}
    cells = sorted(syn.GPU_HOURS, key=syn.GPU_HOURS.get)
    for family in FAMILIES:
        allrows = [r for r in result.values() if r["family"] == family]
        if not allrows:
            continue
        rows = [r for r in allrows if r["u"] < tol]
        passes = {}
        for cell in cells:
            have = [r for r in rows if cell in r["runs"]]
            if not rows or len(have) < len(rows):
                continue
            worst = max(abs(r["runs"][cell]["err"]) for r in rows)
            worst_conf = max(abs(r["runs"][cell]["err"]) + r["u"] for r in rows)
            passes[cell] = (worst <= tol, worst_conf <= tol)

        def ok(cell, i):
            finer = [c for c in passes if c[0] == cell[0] and cell[1] < c[1] < REF_N]
            return passes[cell][i] and all(passes[c][i] for c in finer)

        nom = next((c for c in cells if c in passes and ok(c, 0)), None)
        conf = next((c for c in cells if c in passes and ok(c, 1)), None)
        out[family] = {"confident": conf, "nominal": nom, "ref_u_max": max(r["u"] for r in allrows),
                       "limiting": max(allrows, key=lambda r: r["u"])["label"], "decided": len(rows),
                       "undecidable": [r["label"] for r in allrows if r["u"] >= tol]}
    return out


def name(cell):
    return f"{cell[0]}-{cell[1]}"


# ---------------------------------------------------------------- figures


def fig_grid(result):
    cells = sorted({c for r in result.values() for c in r["runs"]}, key=syn.GPU_HOURS.get)
    keys = list(result)
    err = np.full((len(keys), len(cells)), np.nan)
    within = np.zeros_like(err, dtype=bool)
    for i, key in enumerate(keys):
        for j, cell in enumerate(cells):
            run = result[key]["runs"].get(cell)
            if run:
                err[i, j] = run["err"]
                within[i, j] = np.isfinite(run["sigma"]) and abs(run["err"]) <= 2 * run["sigma"]
    bounds = [0, 5, 10, 20, 50, 1e9]
    cmap = ListedColormap(["#1a9850", "#a6d96a", "#fee08b", "#fc8d59", "#d73027"])
    norm = BoundaryNorm(bounds, cmap.N)
    extra_cols = ["ref. unc.\nu [%]", "FD->inf\nvs R [%]", "FV->inf\nvs R [%]"]
    ncol = len(cells) + len(extra_cols)
    fig, ax = plt.subplots(figsize=(3.0 + 1.0 * ncol, 0.3 * len(keys) + 2.4))
    ax.imshow(np.abs(err), cmap=cmap, norm=norm, aspect="auto", extent=(-0.5, len(cells) - 0.5, len(keys) - 0.5, -0.5))
    for i, key in enumerate(keys):
        dist = result[key]["distance"]
        for j in range(len(cells)):
            if np.isfinite(err[i, j]):
                v = err[i, j]
                text = (f"{v:.0f}" if dist else f"{v:+.0f}") if abs(v) >= 9.5 else (f"{v:.1f}" if dist else f"{v:+.1f}")
                ax.text(j, i, text + ("*" if within[i, j] else ""), ha="center", va="center", fontsize=7)
        r = result[key]
        ax.text(len(cells), i, f"{r['u']:.1f}", ha="center", va="center", fontsize=7, color="0.2")
        for c, scheme in ((len(cells) + 1, "FD"), (len(cells) + 2, "FV")):
            v = r["rich"].get(scheme)
            ax.text(c, i, "-" if v is None else f"{v:+.1f}", ha="center", va="center", fontsize=7, color="0.2")
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"{name(c)}\n{syn.GPU_HOURS[c]:.3g} GPU-h" for c in cells] + extra_cols, fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([result[k]["label"] + (" (dist.)" if result[k]["distance"] else "") for k in keys],
                       fontsize=7.5)
    families = [result[k]["family"] for k in keys]
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or families[i] != families[i - 1]:
            ax.axhline(i - 0.5, color="k", lw=1.2)
            ax.text(-0.02, 1 - (start + i) / 2 / len(keys), families[i - 1], transform=ax.transAxes, ha="right",
                    va="center", fontsize=8.5, fontweight="bold",
                    bbox={"boxstyle": "round", "fc": "white", "ec": "none"})
            start = i
    ax.axvline(len(cells) - 0.5, color="k", lw=1.2)
    ax.tick_params(axis="y", pad=150)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(5)]
    labels = ["|e| < 5%", "5-10%", "10-20%", "20-50%", "> 50%"]
    ax.legend(handles + [Line2D([], [], ls="none")] * 2,
              labels + ["* within 2 sigma (run noise) of the reference",
                        "(dist.) = time-resolved / distribution distance, >= 0"],
              loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=4, fontsize=7.5, frameon=False)
    ax.set_title("CGOLS A: error e = (x - R) / |R| [%] against R = mean(FD-2048, FV-2048), runs ordered by GPU-hours "
                 "(2048 runs: |e| = u by construction); Richardson values of each scheme's 512-1024-2048 ladder "
                 "against R ('-' = not monotone)", fontsize=9, pad=30, wrap=True)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "cost_accuracy_grid.png")
    fig.savefig(path, dpi=115)
    plt.close(fig)
    print(f"wrote {path}")


#: families figure: y-axis top [%]; larger errors are drawn as up-arrows at FAM_YTOP
FAM_YMAX, FAM_YTOP = 800.0, 600.0


def fig_families(result, tols):
    fams = [f for f in FAMILIES if any(r["family"] == f for r in result.values())]
    ncol = 5
    nrow = int(np.ceil(len(fams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow), sharex=True, sharey=True, squeeze=False)
    styles = {"FD": ("C0", "o", "-"), "FV": ("C3", "s", "--")}
    tmax = max(tols)
    for ax, family in zip(axes.flat, fams):
        allrows = [r for r in result.values() if r["family"] == family]
        rows = [r for r in allrows if r["u"] < tmax]
        nund = len(allrows) - len(rows)
        for scheme in SCHEMES:
            pts = []
            for n in NS:
                cell = (scheme, n)
                if n == REF_N or not all(cell in r["runs"] for r in rows):
                    continue
                errs = [abs(r["runs"][cell]["err"]) for r in rows]
                pts.append((syn.GPU_HOURS[cell], max(errs), float(np.median(errs)), n))
            if not pts:
                continue
            x, worst, med, ns = map(np.array, zip(*pts))
            color, marker, ls = styles[scheme]
            # errors beyond the axis are drawn at its top edge as up-arrows
            off_w, off_m = worst > FAM_YMAX, med > FAM_YMAX
            worst, med = np.minimum(worst, FAM_YTOP), np.minimum(med, FAM_YTOP)
            ax.plot(x, worst, ls=ls, marker=marker, color=color, label=f"{scheme}: worst quantity")
            ax.plot(x, med, ls=ls, marker=marker, color=color, alpha=0.35, ms=4, label=f"{scheme}: median")
            ax.plot(x[off_w], worst[off_w], ls="none", marker="^", ms=9, color=color)
            ax.plot(x[off_m], med[off_m], ls="none", marker="^", ms=6, color=color, alpha=0.35)
            for xi, yi, n in zip(x, worst, ns):
                ax.annotate(f"{int(n)}", (xi, yi), textcoords="offset points", xytext=(4, -10 if yi >= FAM_YTOP else 4),
                            fontsize=7, color=color)
        u = max([r["u"] for r in rows] or [0.1])
        ax.axhspan(0.1, max(u, 0.1), color="0.85", zorder=0, label="reference uncertainty (max u)")
        if nund:
            ax.text(0.98, 0.03, f"{nund} of {len(allrows)} quantities undecidable (u >= {tmax:g}%), not shown",
                    transform=ax.transAxes, fontsize=7, va="bottom", ha="right", color="0.3")
        for cell in (("FD", REF_N), ("FV", REF_N)):
            ax.axvline(syn.GPU_HOURS[cell], color=styles[cell[0]][0], lw=0.8, ls=":")
        for tol in tols:
            ax.axhline(tol, color="0.4", lw=0.7, ls="-.")
        ax.set(xscale="log", yscale="log", ylim=(0.3, FAM_YMAX))
        ax.set_title(family, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    for ax in axes.flat[len(fams):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("GPU-hours (H200) to 75 Myr")
    for ax in axes[:, 0]:
        ax.set_ylabel("|error| vs 2048 reference [%]")
    axes.flat[0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle("CGOLS A: worst and median error per quantity family against cost; dotted verticals: cost of "
                 "FD-2048 (blue) / FV-2048 (red); dash-dot: tolerances " + ", ".join(f"{t:g}%" for t in tols)
                 + f"; up-arrows: error > {FAM_YMAX:g}%")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "cost_accuracy_families.png")
    fig.savefig(path, dpi=115)
    plt.close(fig)
    print(f"wrote {path}")


#: scales judged by the spectrum test: <= 1 kpc (a 4 kpc strip has only a handful of modes above that, which describe
#: the individual morphology of that run, not the statistics)
SPEC_LMAX_PC = 1000.0


def smooth3(y):
    """Running mean over 3 neighbouring k bins (NaN-aware), to judge the trend instead of single noisy bins."""
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(i - 1, 0):i + 2]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = np.mean(w)
    return out


def converged_scale(ratio, k, kmax):
    """Smallest scale [pc] down to which the smoothed ratio stays within SPEC_TOL of 1, starting at SPEC_LMAX_PC."""
    r = smooth3(np.where((k <= kmax) & (1e3 / k <= SPEC_LMAX_PC), ratio, np.nan))
    good = None
    for kk, rr in zip(k, r):
        if 1e3 / kk > SPEC_LMAX_PC or kk > kmax:
            continue
        if not np.isfinite(rr) or abs(rr - 1) > SPEC_TOL:
            break
        good = kk
    return None if good is None else 1e3 / good


def ensemble_spectra(runs, ex):
    """{(scheme, N): (mean spectrum over the base run and its seed members, spread (std / mean) or None, members)}."""
    out = {}
    for scheme in SCHEMES:
        for n in NS:
            labs = [lab for lab, s in runs.items() if s[:2] == (scheme, n) and s[2] in ("base", "ens")
                    and lab in ex and ex[lab]["spectrum"] is not None]
            if not labs:
                continue
            arr = np.array([ex[lab]["spectrum"] for lab in labs])
            mean = np.nanmean(arr, axis=0)
            spread = np.nanstd(arr, axis=0, ddof=1) / mean if len(labs) >= 2 else None
            out[(scheme, n)] = (mean, spread, labs)
    return out


def spectra(runs, ex):
    """Self-convergence (N vs 2N, same scheme) and agreement with the 2048 reference, seed-averaged per scheme/N."""
    k = extra.K_MID
    ens = ensemble_spectra(runs, ex)
    spec = {c: v[0] for c, v in ens.items()}
    nyq = {c: ex[ens[c][2][0]]["nyquist"] for c in spec}
    if ("FD", REF_N) not in spec or ("FV", REF_N) not in spec:
        return None
    pref = 0.5 * (spec[("FD", REF_N)] + spec[("FV", REF_N)])
    # reference spread: half the FD-2048 / FV-2048 difference relative to their mean (the "u" of the spectrum)
    uref = np.abs(spec[("FD", REF_N)] - spec[("FV", REF_N)]) / (spec[("FD", REF_N)] + spec[("FV", REF_N)])
    table = {}
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    colors = dict(zip(NS, plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(NS)))))
    kref = (k <= 0.5 * nyq[("FD", REF_N)]) & (1e3 / k <= 3000)
    axes[2].fill_between(1e3 / k[kref], (1 - uref)[kref], (1 + uref)[kref], color="tab:red", alpha=0.15, lw=0,
                         label="reference spread (FD vs FV 2048)")
    for scheme, ls in (("FD", "-"), ("FV", "--")):
        for n in NS:
            c = (scheme, n)
            if c not in spec:
                continue
            nmem = len(ens[c][2])
            tag = f"{scheme}-{n}" + (f" ({nmem} seeds)" if nmem > 1 else "")
            dx_pc = 1e4 / n
            # usable up to half the Nyquist wavenumber (4 cells per wavelength)
            kmax = 0.5 * nyq[c]
            sel = k <= kmax
            # ratio panels: scales <= 3 kpc only (the bins above hold one or two modes)
            rsel = sel & (1e3 / k <= 3000)
            axes[0].loglog(k[sel], spec[c][sel] * k[sel], ls=ls, color=colors[n], label=tag)
            ratio_ref = spec[c] / pref
            axes[2].semilogx(1e3 / k[rsel], smooth3(ratio_ref)[rsel], ls=ls, color=colors[n], label=tag)
            row = {"dx_pc": dx_pc, "members": nmem, "ref_pc": converged_scale(ratio_ref, k, kmax)}
            if ens[c][1] is not None:
                band = sel & (1e3 / k <= SPEC_LMAX_PC) & (1e3 / k >= 4 * dx_pc)
                row["seed_spread"] = float(np.nanmedian(ens[c][1][band]))
            up = (scheme, 2 * n)
            if up in spec:
                ratio_up = spec[c] / spec[up]
                axes[1].semilogx(1e3 / k[rsel], smooth3(ratio_up)[rsel], ls=ls, color=colors[n],
                                 label=f"{scheme}-{n} / {2 * n}")
                row["self_pc"] = converged_scale(ratio_up, k, kmax)
            table[name(c)] = row
    band = kref & (1e3 / k <= SPEC_LMAX_PC) & (1e3 / k >= 100)
    table["_uref_100_1000"] = float(np.nanmedian(uref[band]))
    axes[0].loglog(k, pref * k, color="k", lw=0.8, ls=":", label="reference (2048 mean)")
    axes[0].set(xlabel="k [1/kpc]", ylabel="k P(k)",
                title="log n power spectrum, xz slice, 0.5<|z|<4.5 kpc, 30-45 Myr (seed mean)")
    for ax, title in ((axes[1], "self-convergence: N / 2N (same scheme)"),
                      (axes[2], "against the 2048 reference (mean FD/FV)")):
        ax.axhspan(1 - SPEC_TOL, 1 + SPEC_TOL, color="0.88", zorder=0)
        ax.axhline(1, color="k", lw=0.7)
        ax.axvline(SPEC_LMAX_PC, color="0.5", lw=0.8, ls=":")
        ax.set(xlabel="scale 1/k [pc]", ylabel="power ratio (3-bin mean)", title=title, ylim=(0.2, 3),
               yscale="log")
        ax.invert_xaxis()
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"Scale-by-scale convergence: grey = within {SPEC_TOL:.0%}; judged below {SPEC_LMAX_PC:.0f} pc "
                 "(dotted); spectra drawn to 4 cells per wavelength")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "cost_accuracy_spectra.png")
    fig.savefig(path, dpi=115)
    plt.close(fig)
    print(f"wrote {path}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--tols", default="5,10,20", help="tolerances in percent")
    parser.add_argument("--json", default=None)
    parser.add_argument("--recompute-extra", action="store_true")
    args = parser.parse_args()
    tols = [float(t) for t in args.tols.split(",")]
    if args.recompute_extra:
        os.chdir(SCRIPT_DIR)
        extra.load_all(True)
    runs, data, ex, scal = collect()
    result = analyse(runs, data, ex, scal)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_grid(result)
    fig_families(result, tols)
    spec_table = spectra(runs, ex)

    cells = sorted(syn.GPU_HOURS, key=syn.GPU_HOURS.get)
    print("\n| quantity | " + " | ".join(name(c) for c in cells) + " | u | FD->inf | FV->inf |")
    for key, r in result.items():
        vals = [f"{r['runs'][c]['err']:+.1f}" if c in r["runs"] else "" for c in cells]
        rich = [("-" if r["rich"][s] is None else f"{r['rich'][s]:+.1f}") for s in SCHEMES]
        print(f"| {r['family']}: {r['label']} | " + " | ".join(vals) + f" | {r['u']:.1f} | " + " | ".join(rich) + " |")

    print("\nCheapest run per family (worst quantity within tol; confident = incl. reference uncertainty u):")
    print("| family | max u [%] (limiting quantity) | "
          + " | ".join(f"{t:g}% confident | {t:g}% nominal" for t in tols) + " |")
    summary = {tol: cheapest(result, tol) for tol in tols}
    fmt = lambda cell: f"{name(cell)} ({syn.GPU_HOURS[cell]:.3g})" if cell else "none"
    for family in summary[tols[0]]:
        cols = []
        for tol in tols:
            c = summary[tol][family]
            cols += [fmt(c["confident"]), fmt(c["nominal"])]
        s0 = summary[tols[0]][family]
        print(f"| {family} | {s0['ref_u_max']:.1f} ({s0['limiting']}) | " + " | ".join(cols) + " |")
    print("\nUndecidable quantities (reference uncertainty u >= tolerance; excluded from the family verdict):")
    for tol in tols:
        und = {f: d["undecidable"] for f, d in summary[tol].items() if d["undecidable"]}
        print(f"  {tol:g}%: " + "; ".join(f"{f}: {', '.join(v)}" for f, v in und.items()))
    if spec_table:
        uref = spec_table.pop("_uref_100_1000")
        print(f"\nConverged scale l_c [pc] (3-bin-mean power ratio within {SPEC_TOL:.0%} from {SPEC_LMAX_PC:.0f} pc "
              f"down to l_c; cells = l_c/dx; spectra averaged over the seed members; reference spread FD vs FV 2048 "
              f"at 100-1000 pc: median {uref:.0%}):")
        print("| run | members | dx [pc] | seed spread | vs 2N same scheme | vs 2048 reference |")
        for run, row in spec_table.items():
            f = lambda v, dx=row["dx_pc"]: "none" if v is None else f"{v:.0f} pc ({v / dx:.1f} cells)"
            sp = f"{row['seed_spread']:.0%}" if "seed_spread" in row else "-"
            print(f"| {run} | {row['members']} | {row['dx_pc']:.1f} | {sp} | "
                  f"{f(row.get('self_pc')) if 'self_pc' in row else '-'} | {f(row['ref_pc'])} |")
        spec_table["_uref_100_1000"] = uref
    if args.json:
        out = {k: {**{kk: vv for kk, vv in r.items() if kk != "runs"},
                   "runs": {name(c): v for c, v in r["runs"].items()}} for k, r in result.items()}
        summ = {str(t): {f: {kk: (name(vv) if isinstance(vv, tuple) else vv) for kk, vv in d.items()}
                         for f, d in s.items()} for t, s in summary.items()}
        with open(args.json, "w") as fh:
            json.dump({"metrics": out, "cheapest": summ, "spectra": spec_table}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
