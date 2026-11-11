"""
Extended per-run measurements for cgols_cost_accuracy.py (CGOLS A-series, FD and FV, all base and seed runs).

Everything is read from existing outputs: the frames (vertical slices n/T/v_r and projected X-ray / H-alpha maps of
every snapshot), the emission caches and the 3D caches (data/studies/convergence3d_v3<tag>.npz). Per run it stores

  scalars  {key: (value, time-sampling se)}:
           3D window means at the run's NATIVE snapshot cadence (1.25 / 2.5 / 5 Myr; the synthesis uses 5 Myr for
           all): wind / mixed / entrained+mixed outflow at 1 kpc, Edot and outward momentum flux at 1 and 5 kpc,
           cone half-angle; the core share of the net outflow (R < 1 kpc) at 1 and 5 kpc; the warm share of the
           gas above 0.5 kpc; the disk mass at 75 Myr; the shell-breakout time (halo-entropy outflow at 5 kpc first
           > 0.5 Msun/yr after 40 Myr); from the frames the mass fraction of fast gas (v_r > 600 km/s) above 1 kpc
           in the xz slice and the clumpiness (std / mean) of the X-ray and H-alpha maps block-averaged to 156 pc in
           0.5 < |z| < 4 kpc; from the emission caches the S_X1 profile at 2-5 kpc (15-40 Myr) and 5-9 kpc (60-75)
  series   time series for the time-resolved errors (3D: mixed outflow 1 kpc, Edot 5 kpc)
  hists    the window-mean mass-weighted (log n, log T) histogram above 1 kpc (15-40 and 50-75 Myr), normalised
  spectra  the window-mean (30-45 Myr) 2D power spectrum of log n in the xz slice, 0.5 < |z| < 4.5 kpc, on common
           k bins [cycles/kpc]

Cached in data/studies/cost_accuracy_extra.pkl; a run is re-measured when its frame count or its emission / 3D
cache appears or changes. CPU only, numpy.

    python cgols_cost_accuracy_extra.py [--recompute] [--only LABEL ...]
"""

# general
import argparse
import os
import pickle

# numerics
import numpy as np

# CGOLS analysis
import cgols_fvfd_synthesis as syn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRIPT_DIR, "data/studies/cost_accuracy_extra.pkl")
BOX_X, BOX_Z = 10.0, 20.0  # kpc
FAST_KMS = 600.0
COARSE = (64, 128)  # observable maps block-averaged to 10 kpc / 64 = 156 pc
CLUMP_BAND = (0.5, 4.0)  # |z| [kpc]
SPEC_BAND = (0.5, 4.5)  # |z| [kpc] of the spectrum region
K_EDGES = np.geomspace(0.1, 128.0, 49)  # cycles / kpc
K_MID = np.sqrt(K_EDGES[1:] * K_EDGES[:-1])
H1, H5 = int(np.argmin(np.abs(syn.HEIGHTS_KPC - 1))), int(np.argmin(np.abs(syn.HEIGHTS_KPC - 5)))
WINDOWS = {"15_40": (15, 40), "15_30": (15, 30), "30_45": (30, 45), "50_75": (50, 75), "60_75": (60, 75),
           "20_40": (20, 40)}


def _win(t, name):
    return syn.window(np.asarray(t), *WINDOWS[name])


def _mean_se(t, y, name):
    sel = _win(t, name) & np.isfinite(y)
    if sel.sum() < 3:
        return None
    a, b = WINDOWS[name]
    return float(np.mean(y[sel])), syn.block_se(t, y, a, b, block=5.0)


# ---------------------------------------------------------------- 3D caches


def measure_3d(tag3d):
    rec = np.load(os.path.join(SCRIPT_DIR, f"data/studies/convergence3d_v3{tag3d}.npz"), allow_pickle=True)
    rec = rec["records"].item()
    steps = sorted(rec, key=lambda s: rec[s]["time_myr"])
    t = np.array([rec[s]["time_myr"] for s in steps])
    R = [rec[s] for s in steps]

    def cone(r):
        z = r["z_kpc"]
        north = z > 0
        area = 0.5 * (r["zarea_fast"][north] + r["zarea_fast"][~north][::-1])
        k = int(np.argmin(np.abs(z[north] - 1.0)))
        return np.degrees(np.arctan(np.sqrt(area[k] / np.pi) / z[north][k]))

    ser = {
        "wind1": np.array([r["mdot_out_K"][H1, 2] for r in R]),
        "mixed1": np.array([r["mdot_out_K"][H1, 1] for r in R]),
        "entmixed1": np.array([r["mdot_out_K"][H1, 0] + r["mdot_out_K"][H1, 1] for r in R]),
        "edot1": np.array([r["edot_net"][H1].sum() / 1e42 for r in R]),
        "edot5": np.array([r["edot_net"][H5].sum() / 1e42 for r in R]),
        "pdot1": np.array([r["pdot_out"][H1].sum() / 1e3 for r in R]),
        "pdot5": np.array([r["pdot_out"][H5].sum() / 1e3 for r in R]),
        "cone": np.array([cone(r) for r in R]),
        "core1": np.array([r["mdot_net_radius"][H1, :2].sum() for r in R]),
        "all1": np.array([r["mdot_net_radius"][H1].sum() for r in R]),
        "core5": np.array([r["mdot_net_radius"][H5, :2].sum() for r in R]),
        "all5": np.array([r["mdot_net_radius"][H5].sum() for r in R]),
        "warm": np.array([r["mass_halo"][1] for r in R]),
        "halo": np.array([r["mass_halo"].sum() for r in R]),
        "halo5": np.array([r["mdot_out_K"][H5, 3] for r in R]),
    }
    out = {}
    for key, (s, w) in {
        "n_wind1_15_40": ("wind1", "15_40"), "n_mixed1_15_30": ("mixed1", "15_30"),
        "n_mixed1_30_45": ("mixed1", "30_45"), "n_entmixed1_50_75": ("entmixed1", "50_75"),
        "n_edot5_15_40": ("edot5", "15_40"), "n_edot5_50_75": ("edot5", "50_75"),
        "n_edot1_15_40": ("edot1", "15_40"), "n_edot1_50_75": ("edot1", "50_75"),
        "n_pdot5_15_40": ("pdot5", "15_40"), "n_pdot5_50_75": ("pdot5", "50_75"),
        "n_pdot1_15_40": ("pdot1", "15_40"), "n_pdot1_50_75": ("pdot1", "50_75"),
        "n_cone1_20_40": ("cone", "20_40"), "n_cone1_60_75": ("cone", "60_75"),
    }.items():
        v = _mean_se(t, ser[s], w)
        if v:
            out[key] = v
    for h in ("1", "5"):
        core, tot = _mean_se(t, ser[f"core{h}"], "15_40"), _mean_se(t, ser[f"all{h}"], "15_40")
        if core and tot:
            out[f"core{h}_15_40"] = (core[0] / tot[0], core[1] / tot[0] if core[1] is not None else None)
    for w in ("15_40", "50_75"):
        warm, halo = _mean_se(t, ser["warm"], w), _mean_se(t, ser["halo"], w)
        if warm and halo:
            out[f"warmfrac_{w}"] = (warm[0] / halo[0], warm[1] / halo[0] if warm[1] is not None else None)
    if t.max() > 74:
        out["disk_75"] = (float(R[-1]["mass_disk"].sum() / 1e9), None)
    late = (t > 40) & (ser["halo5"] > 0.5)
    if late.any():
        k = int(np.argmax(late))
        t0, t1, y0, y1 = t[k - 1], t[k], ser["halo5"][k - 1], ser["halo5"][k]
        out["breakout"] = (float(t0 + (0.5 - y0) / (y1 - y0) * (t1 - t0)) if y1 != y0 else float(t1),
                           float(t1 - t0) / np.sqrt(12))
    hists = {}
    for w in ("15_40", "50_75"):
        sel = _win(t, w)
        if sel.sum() >= 2:
            h = np.mean([R[i]["hist_nT"] for i in np.where(sel)[0]], axis=0)
            hists[w] = h / h.sum()
    series = {"t": t, "mixed1": ser["mixed1"], "edot5": ser["edot5"], "pdot5": ser["pdot5"]}
    return out, series, hists


# ---------------------------------------------------------------- frames


def _block(a, shape):
    fx, fz = a.shape[0] // shape[0], a.shape[1] // shape[1]
    return a[: shape[0] * fx, : shape[1] * fz].reshape(shape[0], fx, shape[1], fz).mean(axis=(1, 3))


def _spectrum(field, dx, dz):
    """Radially binned 2D power spectrum (continuous-FT normalisation) of a mean-subtracted, Hann-windowed field."""
    f = field - field.mean()
    win = np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1]))
    f = f * win / np.sqrt(np.mean(win**2))
    F = np.fft.fft2(f) * dx * dz
    lx, lz = f.shape[0] * dx, f.shape[1] * dz
    power = np.abs(F) ** 2 / (lx * lz)
    kx = np.fft.fftfreq(f.shape[0], dx)
    kz = np.fft.fftfreq(f.shape[1], dz)
    k = np.sqrt(kx[:, None] ** 2 + kz[None, :] ** 2)
    idx = np.digitize(k.ravel(), K_EDGES) - 1
    ok = (idx >= 0) & (idx < len(K_MID))
    sums = np.bincount(idx[ok], weights=power.ravel()[ok], minlength=len(K_MID))
    counts = np.bincount(idx[ok], minlength=len(K_MID))
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def measure_frames(frame_dir):
    files = sorted(f for f in os.listdir(frame_dir) if f.startswith("frame_") and f.endswith(".npz"))
    t, fast, cv_x, cv_h, specs, spec_t = [], [], [], [], [], []
    for name in files:
        fr = np.load(os.path.join(frame_dir, name))
        keys = set(fr.files)
        time = float(fr["time_myr"])
        n = fr["n_xz"]
        nx, nz = n.shape
        dx, dz = BOX_X / nx, BOX_Z / nz
        z = -BOX_Z / 2 + (np.arange(nz) + 0.5) * dz
        t.append(time)
        above = np.abs(z) > 1.0
        if "vr_xz" in keys:
            vr = fr["vr_xz"][:, above]
            w = n[:, above]
            fast.append(float(w[vr > FAST_KMS].sum() / w.sum()))
        else:
            fast.append(np.nan)
        zc = -BOX_Z / 2 + (np.arange(COARSE[1]) + 0.5) * BOX_Z / COARSE[1]
        band = (np.abs(zc) > CLUMP_BAND[0]) & (np.abs(zc) < CLUMP_BAND[1])
        for key, out in (("emission_xray_cgols_xz", cv_x), ("emission_halpha_xz", cv_h)):
            if key in keys and nx >= COARSE[0]:
                m = _block(fr[key].astype(np.float64), COARSE)[:, band]
                out.append(float(m.std() / m.mean()) if m.mean() > 0 else np.nan)
            else:
                out.append(np.nan)
        if 30 - syn.WINDOW_MARGIN <= time <= 45 + syn.WINDOW_MARGIN:
            logn = np.log10(np.maximum(n.astype(np.float64), 1e-12))
            north = (z > SPEC_BAND[0]) & (z < SPEC_BAND[1])
            south = (z < -SPEC_BAND[0]) & (z > -SPEC_BAND[1])
            specs.append(0.5 * (_spectrum(logn[:, north], dx, dz) + _spectrum(logn[:, south][:, ::-1], dx, dz)))
            spec_t.append(time)
    t = np.array(t)
    order = np.argsort(t)
    t, fast, cv_x, cv_h = t[order], np.array(fast)[order], np.array(cv_x)[order], np.array(cv_h)[order]
    out = {}
    for key, y, w in (("fast1_15_40", fast, "15_40"), ("fast1_50_75", fast, "50_75"),
                      ("clumpx_15_40", cv_x, "15_40"), ("clumpx_60_75", cv_x, "60_75"),
                      ("clumph_15_40", cv_h, "15_40"), ("clumph_60_75", cv_h, "60_75")):
        v = _mean_se(t, y, w)
        if v and np.isfinite(v[0]):
            out[key] = v
    spec = np.mean(specs, axis=0) if specs else None
    nyq = None
    if files:
        nyq = 0.5 / (BOX_X / np.load(os.path.join(frame_dir, files[0]))["n_xz"].shape[0])
    return out, spec, nyq, len(files)


# ---------------------------------------------------------------- emission caches


def measure_emission(emis):
    path = os.path.join(SCRIPT_DIR, f"data/studies/emission{emis}.npz")
    if not emis or not os.path.exists(path):
        return {}
    c = np.load(path, allow_pickle=True)
    t = c["per_snapshot__time_myr"]
    z = c["per_snapshot__profile_z_kpc"]
    prof = 0.5 * (c["per_snapshot__profile_xray_cgols_xz_north"] + c["per_snapshot__profile_xray_cgols_xz_south"])
    out = {}
    for key, (a, b), w in (("sx_2_5_15_40", (2, 5), "15_40"), ("sx_5_9_60_75", (5, 9), "60_75")):
        y = np.array([np.mean(p[(zz >= a) & (zz <= b)]) for p, zz in zip(prof, z)])
        v = _mean_se(t, y, w)
        if v:
            out[key] = v
    return out


# ---------------------------------------------------------------- driver


def sources(spec):
    _s, _n, _r, frames, emis, tag3d, _d = spec
    return [bool(emis and os.path.exists(os.path.join(SCRIPT_DIR, f"data/studies/emission{emis}.npz"))),
            bool(tag3d and os.path.exists(os.path.join(SCRIPT_DIR, f"data/studies/convergence3d_v3{tag3d}.npz")))]


def measure(label, spec):
    _s, _n, _r, frames, emis, tag3d, _d = spec
    scal, series, hists = {}, {}, {}
    if tag3d and os.path.exists(os.path.join(SCRIPT_DIR, f"data/studies/convergence3d_v3{tag3d}.npz")):
        scal3, series, hists = measure_3d(tag3d)
        scal.update(scal3)
    fscal, spec_k, nyq, nframes = measure_frames(os.path.join(SCRIPT_DIR, frames))
    scal.update(fscal)
    scal.update(measure_emission(emis))
    return {"scalars": scal, "series": series, "hists": hists, "spectrum": spec_k, "nyquist": nyq,
            "nframes": nframes, "sources": sources(spec)}


def load_all(recompute=False, only=()):
    cache = {}
    if os.path.exists(CACHE) and not recompute:
        with open(CACHE, "rb") as fh:
            cache = pickle.load(fh)
    changed = False
    for label, spec in syn.RUNS.items():
        if spec[2] not in ("base", "ens") or (only and label not in only):
            continue
        frames = os.path.join(SCRIPT_DIR, spec[3])
        if not os.path.isdir(frames):
            continue
        nframes = len([f for f in os.listdir(frames) if f.startswith("frame_")])
        c = cache.get(label)
        if c and c["nframes"] == nframes and c["sources"] == sources(spec):
            continue
        print(f"  measuring {label} ({spec[3]}, {nframes} frames)", flush=True)
        cache[label] = measure(label, spec)
        changed = True
    if changed:
        with open(CACHE, "wb") as fh:
            pickle.dump(cache, fh)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--only", nargs="*", default=())
    args = parser.parse_args()
    os.chdir(SCRIPT_DIR)
    cache = load_all(args.recompute, args.only)
    for label, c in cache.items():
        print(f"{label:10s} {len(c['scalars']):2d} scalars, 3D series {'yes' if c['series'] else 'no '}, "
              f"spectrum {'yes' if c['spectrum'] is not None else 'no'}")


if __name__ == "__main__":
    main()
