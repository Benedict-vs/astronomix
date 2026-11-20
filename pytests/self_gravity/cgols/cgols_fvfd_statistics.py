"""
Higher-order statistics of FD against FV at 1024^3 and 2048^3 (asgard-papers #11, item 4): where might the fifth-order
FD scheme do better than FV?

Everything is read from existing caches; nothing is re-measured:

* PDFs (cgols_convergence_3d caches, data/studies/convergence3d_v3_<TAG>.npz): the mass-weighted (log n, log T)
  histogram of the gas at |z| > 1 kpc on every checkpoint, split into the entropy phases of cgols_convergence_3d
  (log K = log T - 2/3 log n; entrained < 5.5 < mixed < 6.8 < wind < 7.9 < halo). Per run and window: the PDFs of
  log n and log T of each phase, their mass-weighted mean and rms width.
* Power spectra and clumps (cgols_structure_3d caches, structure3d_v2_<TAG>.npz, 1 < |z| < 3.5 kpc, 5 x 5 kpc):
  3D spectra of log n, log T and |v|, both hemispheres averaged; the band power at 1000-300, 300-100 and 100-30 pc;
  the mass function of all entrained clumps (log K < 5.5) and of the clumps detached from the disc.
* Map morphology (cgols_emission caches, emission_<TAG>.npz, one edge-on x-z map at 60 Myr per run): soft X-ray
  (AtomDB 0.3-2 keV) and H-alpha surface brightness at 1 < |z| < 5 kpc, at the native resolution and block-averaged
  to beams of 20, 80 and 160 pc (Chandra resolves ~10-20 pc at the distance of M82): the PDF of log S, its rms
  width, the clumpiness std(S) / mean(S), and the 2D power spectrum of log S.

Noise: the seed runs (FD-512 x2, FD-1024 x2, FV-1024 x4, FV-2048 x2). A difference FD - FV is called a scheme effect
only if it exceeds twice the seed spread (the larger of the two schemes' standard deviations at that N; FD-2048
borrows FD-1024's relative spread).

    python cgols_fvfd_statistics.py [--json OUT]

Writes figures/cgols/comparisons/fvfd_stats_{pdfs,spectra,clumps,maps}.png and prints the summary tables. CPU only.
"""

# ruff: noqa: C408 (plot keyword dicts)

# general
import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# numerics, plotting
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.join(SCRIPT_DIR, "data", "studies")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures", "cgols", "comparisons")

#: (scheme, N): [base tag, seed tags...]; the structure3d base of FV-1024 is the bit-identical rerun A1024fvr.
RUNS = {
    ("FD", 1024): ["A1024", "A1024s1"],
    ("FV", 1024): ["A1024fv", "A1024fvs1", "A1024fvs2", "A1024fvs3"],
    ("FD", 2048): ["A2048"],
    ("FV", 2048): ["A2048fv", "A2048fvs1"],
}
STRUCTURE_TAG = {"A1024fv": "A1024fvr"}
COLORS = {("FD", 1024): "tab:cyan", ("FD", 2048): "tab:blue", ("FV", 1024): "tab:pink", ("FV", 2048): "tab:red"}

WINDOWS = {"30-45": (30.0, 45.0), "45-75": (45.0, 75.0)}
# cgols_convergence_3d
LOG_N_EDGES = np.linspace(-6.0, 3.0, 91)
LOG_T_EDGES = np.linspace(3.0, 9.0, 121)
ENTROPY_EDGES = (5.5, 6.8, 7.9)
PHASES = ("entrained", "mixed", "wind", "halo")
# cgols_structure_3d
CLUMP_MASS_EDGES = np.logspace(-1.0, 8.0, 37)
K_EDGES = np.geomspace(0.1, 128.0, 49)
BANDS_PC = {"1000-300 pc": (1000.0, 300.0), "300-100 pc": (300.0, 100.0), "100-30 pc": (100.0, 30.0)}
# maps
BOX_X_KPC, BOX_Z_KPC = 10.0, 20.0
MAP_Z_RANGE_KPC = (1.0, 5.0)
BEAMS_PC = (20.0, 80.0, 160.0)
Z_SCHEME = 2.0


def window(times, bounds):
    t = np.round(np.asarray(times, dtype=float), 3)
    return (t > bounds[0]) & (t <= bounds[1])


def load_records(prefix, tag):
    path = os.path.join(STUDIES, f"{prefix}{tag}.npz")
    if not os.path.exists(path):
        return None
    recs = np.load(path, allow_pickle=True)["records"].item()
    return sorted(recs.values(), key=lambda r: float(r["time_myr"]))


# ---------------------------------------------------------------- PDFs


def phase_masks():
    """Boolean (n, T) masks of the entropy phases on the histogram bin centres."""
    log_n = 0.5 * (LOG_N_EDGES[1:] + LOG_N_EDGES[:-1])
    log_t = 0.5 * (LOG_T_EDGES[1:] + LOG_T_EDGES[:-1])
    log_k = log_t[None, :] - 2.0 / 3.0 * log_n[:, None]
    which = np.digitize(log_k, ENTROPY_EDGES)
    return {phase: which == i for i, phase in enumerate(PHASES)}, log_n, log_t


def pdf_stats(tag, bounds):
    """{phase: {pdf_n, pdf_t, mean_n, width_n, mean_t, width_t, mass}} of the window-mean histogram, or None."""
    recs = load_records("convergence3d_v3_", tag)
    if recs is None:
        return None
    sel = window([r["time_myr"] for r in recs], bounds)
    hist = np.mean([r["hist_nT"] for r, keep in zip(recs, sel, strict=True) if keep], axis=0)
    masks, log_n, log_t = phase_masks()
    out = {}
    for phase, mask in masks.items():
        h = np.where(mask, hist, 0.0)
        mass = h.sum()
        if mass <= 0:
            continue
        pn, pt = h.sum(axis=1) / mass, h.sum(axis=0) / mass
        mn, mt = np.sum(pn * log_n), np.sum(pt * log_t)
        out[phase] = {"pdf_n": pn, "pdf_t": pt, "mean_n": mn, "width_n": np.sqrt(np.sum(pn * (log_n - mn) ** 2)),
                      "mean_t": mt, "width_t": np.sqrt(np.sum(pt * (log_t - mt) ** 2)), "mass": mass}
    return out


# ---------------------------------------------------------------- spectra and clumps


def structure_window(tag, bounds):
    """Window means of the structure3d spectra (hemispheres averaged) and clump mass functions (summed)."""
    recs = load_records("structure3d_v2_", STRUCTURE_TAG.get(tag, tag))
    if recs is None:
        return None
    sel = [r for r, keep in zip(recs, window([r["time_myr"] for r in recs], bounds), strict=True) if keep]
    out = {}
    for field in ("pk_logn", "pk_logT", "pk_v"):
        out[field] = np.nanmean([0.5 * (np.asarray(r["north"][field]) + np.asarray(r["south"][field])) for r in sel],
                                axis=0)  # fmt: skip
    for field in ("clump_mass_hist", "clump_det_mass_hist"):
        out[field] = np.mean([np.asarray(r["north"][field]) + np.asarray(r["south"][field]) for r in sel], axis=0)
    out["k_nyquist"] = float(sel[0]["k_nyquist"])
    return out


def band_power(spectrum, k_nyquist, band_pc):
    """Sum of P(k) over the k bins inside a scale band (cycles / kpc = 1000 / scale_pc); NaN beyond the Nyquist k."""
    k = np.sqrt(K_EDGES[1:] * K_EDGES[:-1])
    lo, hi = 1000.0 / band_pc[0], 1000.0 / band_pc[1]
    if hi > k_nyquist:
        return np.nan
    sel = (k >= lo) & (k < hi) & np.isfinite(spectrum)
    return float(np.sum(spectrum[sel]))


# ---------------------------------------------------------------- maps


def map_stats(tag):
    """Morphology of the 60 Myr x-z maps at 1 < |z| < 5 kpc, native and beam-averaged."""
    path = os.path.join(STUDIES, f"emission_{tag}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as cache:
        maps = {"xray": np.asarray(cache["map__xray_atomdb_xz"], dtype=np.float64),
                "halpha": np.asarray(cache["map__halpha_xz"], dtype=np.float64)}  # fmt: skip
    out = {}
    for name, m in maps.items():
        nx, nz = m.shape
        dx_pc = 1000.0 * BOX_X_KPC / nx
        z = (np.arange(nz) + 0.5) * BOX_Z_KPC / nz - BOX_Z_KPC / 2
        res = {}
        for beam in (None, *BEAMS_PC):
            b = 1 if beam is None else max(1, round(beam / dx_pc))
            mb = m[: nx // b * b, : nz // b * b].reshape(nx // b, b, nz // b, b).mean(axis=(1, 3))
            zb = z[: nz // b * b].reshape(nz // b, b).mean(axis=1)
            strips = [mb[:, (zb > MAP_Z_RANGE_KPC[0]) & (zb < MAP_Z_RANGE_KPC[1])],
                      mb[:, (zb < -MAP_Z_RANGE_KPC[0]) & (zb > -MAP_Z_RANGE_KPC[1])][:, ::-1]]  # fmt: skip
            values = np.concatenate([s.ravel() for s in strips])
            floor = 1e-6 * values.max()
            log_s = np.log10(np.maximum(values, floor))
            entry = {"width": float(np.std(log_s)), "clumpiness": float(values.std() / values.mean()),
                     "pdf": np.histogram(log_s - np.median(log_s), bins=np.linspace(-4, 4, 81), density=True)[0]}
            if beam is None:
                # 2D power spectrum of log S over the two strips (each 10 kpc x 4 kpc), radially binned
                ks, ps = [], []
                for s in strips:
                    f = np.log10(np.maximum(s, floor))
                    f = f - f.mean()
                    p = np.abs(np.fft.fft2(f)) ** 2 / f.size
                    kx = np.fft.fftfreq(f.shape[0], d=dx_pc / 1000.0)
                    kz = np.fft.fftfreq(f.shape[1], d=dx_pc / 1000.0)
                    kk = np.sqrt(kx[:, None] ** 2 + kz[None, :] ** 2)
                    idx = np.digitize(kk.ravel(), K_EDGES) - 1
                    good = (idx >= 0) & (idx < len(K_EDGES) - 1)
                    sums = np.bincount(idx[good], weights=p.ravel()[good], minlength=len(K_EDGES) - 1)
                    counts = np.bincount(idx[good], minlength=len(K_EDGES) - 1)
                    ps.append(np.where(counts > 0, sums / np.maximum(counts, 1), np.nan))
                entry["pk"] = np.nanmean(ps, axis=0)
                entry["k_nyquist"] = 0.5 * 1000.0 / dx_pc
            res["native" if beam is None else f"{beam:.0f} pc"] = entry
        out[name] = res
    return out


# ---------------------------------------------------------------- comparison


def compare(values):
    """{(scheme, N): [member values]} -> per N: FD, FV (base), seed sigma, z, verdict."""
    rows = {}
    rel_fd_1024 = None
    for n in (1024, 2048):
        fd, fv = values.get(("FD", n)), values.get(("FV", n))
        if not fd or not fv or fd[0] is None or fv[0] is None or not np.isfinite(fd[0]) or not np.isfinite(fv[0]):
            continue
        sig_fd = np.std(fd, ddof=1) if len(fd) >= 2 else np.nan
        sig_fv = np.std(fv, ddof=1) if len(fv) >= 2 else np.nan
        if n == 1024 and np.isfinite(sig_fd) and fd[0]:
            rel_fd_1024 = sig_fd / abs(fd[0])
        if not np.isfinite(sig_fd) and rel_fd_1024 is not None:
            sig_fd = rel_fd_1024 * abs(fd[0])
        sig = np.nanmax([sig_fd, sig_fv])
        delta = fd[0] - fv[0]
        z = delta / sig if np.isfinite(sig) and sig > 0 else (0.0 if delta == 0 else np.inf)
        mean = 0.5 * (abs(fd[0]) + abs(fv[0]))
        rows[n] = {"FD": fd[0], "FV": fv[0], "rel": delta / mean if mean else 0.0, "sigma": sig, "z": z,
                   "verdict": "scheme" if abs(z) >= Z_SCHEME else "noise"}
    return rows


def collect():
    stats = {}
    for cell, tags in RUNS.items():
        for tag in tags:
            stats[tag] = {"pdf": {w: pdf_stats(tag, b) for w, b in WINDOWS.items()},
                          "structure": {w: structure_window(tag, b) for w, b in WINDOWS.items()},
                          "maps": map_stats(tag)}  # fmt: skip
    return stats


def scalar_table(stats):
    """{quantity: {(scheme, N): [values of base + seeds]}}."""
    table = {}

    def put(name, cell, value):
        table.setdefault(name, {}).setdefault(cell, []).append(value)

    for cell, tags in RUNS.items():
        for tag in tags:
            s = stats[tag]
            for w in WINDOWS:
                pdf = s["pdf"][w]
                if pdf:
                    for phase in ("mixed", "wind", "halo"):
                        if phase in pdf:
                            put(f"PDF {w}: {phase} width log n", cell, pdf[phase]["width_n"])
                            put(f"PDF {w}: {phase} width log T", cell, pdf[phase]["width_t"])
                            put(f"PDF {w}: {phase} mean log n", cell, pdf[phase]["mean_n"])
                            put(f"PDF {w}: {phase} mass [Msun]", cell, pdf[phase]["mass"])
                st = s["structure"][w]
                if st:
                    for field, label in (("pk_logn", "log n"), ("pk_logT", "log T"), ("pk_v", "|v|")):
                        for band, b in BANDS_PC.items():
                            put(f"spectrum {w}: {label} power {band}", cell, band_power(st[field], st["k_nyquist"], b))
                    centres = np.sqrt(CLUMP_MASS_EDGES[1:] * CLUMP_MASS_EDGES[:-1])
                    for field, label in (("clump_mass_hist", "all"), ("clump_det_mass_hist", "detached")):
                        for m_min in (1e2, 1e3, 1e4):
                            put(f"clumps {w}: {label} clumps > {m_min:.0e} Msun", cell,
                                float(st[field][centres > m_min].sum()))  # fmt: skip
            if s["maps"]:
                for name in ("xray", "halpha"):
                    for res, e in s["maps"][name].items():
                        put(f"map 60 Myr: {name} clumpiness ({res})", cell, e["clumpiness"])
                        put(f"map 60 Myr: {name} width log S ({res})", cell, e["width"])
                    e = s["maps"][name]["native"]
                    for band, b in BANDS_PC.items():
                        put(f"map 60 Myr: {name} log S power {band}", cell, band_power(e["pk"], e["k_nyquist"], b))
    return table


# ---------------------------------------------------------------- figures


def label(cell, tag, tags):
    return f"{cell[0]}-{cell[1]}" + ("" if tag == tags[0] else " seed")


def fig_pdfs(stats, w="45-75"):
    _, log_n, log_t = phase_masks()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharey="row")
    for col, phase in enumerate(("mixed", "wind", "halo")):
        for cell, tags in RUNS.items():
            for tag in tags:
                pdf = stats[tag]["pdf"][w]
                if not pdf or phase not in pdf:
                    continue
                kw = dict(color=COLORS[cell], lw=1.6 if tag == tags[0] else 0.6, alpha=1 if tag == tags[0] else 0.6,
                          label=label(cell, tag, tags) if tag == tags[0] else None)  # fmt: skip
                axes[0, col].plot(log_n, pdf[phase]["pdf_n"] / np.diff(LOG_N_EDGES), **kw)
                axes[1, col].plot(log_t, pdf[phase]["pdf_t"] / np.diff(LOG_T_EDGES), **kw)
        axes[0, col].set(title=f"{phase} gas, |z| > 1 kpc, {w} Myr", xlabel="log n [cm$^{-3}$]", yscale="log",
                         ylim=(1e-3, 10))  # fmt: skip
        axes[1, col].set(xlabel="log T [K]", yscale="log", ylim=(1e-3, 10))
    axes[0, 0].set_ylabel("mass-weighted PDF")
    axes[1, 0].set_ylabel("mass-weighted PDF")
    axes[0, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fvfd_stats_pdfs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_spectra(stats, w="45-75"):
    k = np.sqrt(K_EDGES[1:] * K_EDGES[:-1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True, gridspec_kw={"height_ratios": (2, 1)})
    for col, (field, name) in enumerate((("pk_logn", "log n"), ("pk_logT", "log T"), ("pk_v", "|v|"))):
        base = {}
        for cell, tags in RUNS.items():
            for tag in tags:
                st = stats[tag]["structure"][w]
                if not st:
                    continue
                p = np.where(k <= st["k_nyquist"], st[field], np.nan)
                if tag == tags[0]:
                    base[cell] = p
                axes[0, col].loglog(k, k * p, color=COLORS[cell], lw=1.6 if tag == tags[0] else 0.6,
                                    alpha=1 if tag == tags[0] else 0.6,
                                    label=label(cell, tag, tags) if tag == tags[0] else None)  # fmt: skip
        for n in (1024, 2048):
            if ("FD", n) in base and ("FV", n) in base:
                axes[1, col].semilogx(k, base[("FV", n)] / base[("FD", n)], color=COLORS[("FV", n)],
                                      label=f"FV / FD at {n}$^3$")  # fmt: skip
        axes[0, col].set(title=f"3D power of {name}, 1 < |z| < 3.5 kpc, {w} Myr", ylabel="k P(k)")
        axes[1, col].axhline(1.0, color="k", lw=0.8)
        axes[1, col].set(xlabel="k [cycles / kpc]", ylabel="FV / FD", ylim=(0.3, 3), yscale="log")
        for ax in axes[:, col]:
            ax.grid(alpha=0.3, which="both")
            for scale_pc in (1000, 300, 100, 30):
                ax.axvline(1000.0 / scale_pc, color="0.6", lw=0.6, ls=":")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fvfd_stats_spectra.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_clumps(stats, w="45-75"):
    centres = np.sqrt(CLUMP_MASS_EDGES[1:] * CLUMP_MASS_EDGES[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for col, (field, name) in enumerate((("clump_mass_hist", "all entrained clumps"),
                                         ("clump_det_mass_hist", "clumps detached from the disc"))):  # fmt: skip
        for cell, tags in RUNS.items():
            for tag in tags:
                st = stats[tag]["structure"][w]
                if not st:
                    continue
                cum = np.cumsum(st[field][::-1])[::-1]
                axes[col].loglog(centres, np.where(cum > 0, cum, np.nan), color=COLORS[cell],
                                 lw=1.6 if tag == tags[0] else 0.6, alpha=1 if tag == tags[0] else 0.6,
                                 label=label(cell, tag, tags) if tag == tags[0] else None)  # fmt: skip
        axes[col].set(title=f"{name}, 1 < |z| < 3.5 kpc, {w} Myr", xlabel="clump mass M [M$_\\odot$]")
        axes[col].grid(alpha=0.3, which="both")
    axes[0].set_ylabel("N(> M), mean per snapshot")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fvfd_stats_clumps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_maps(stats):
    k = np.sqrt(K_EDGES[1:] * K_EDGES[:-1])
    pdf_x = 0.5 * (np.linspace(-4, 4, 81)[1:] + np.linspace(-4, 4, 81)[:-1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for row, name in enumerate(("xray", "halpha")):
        title = "soft X-ray (0.3-2 keV)" if name == "xray" else "H-alpha"
        for cell, tags in RUNS.items():
            for tag in tags:
                m = stats[tag]["maps"]
                if not m:
                    continue
                kw = dict(color=COLORS[cell], lw=1.6 if tag == tags[0] else 0.6, alpha=1 if tag == tags[0] else 0.6,
                          label=label(cell, tag, tags) if tag == tags[0] else None)  # fmt: skip
                axes[row, 0].semilogy(pdf_x, m[name]["native"]["pdf"], **kw)
                axes[row, 1].semilogy(pdf_x, m[name]["80 pc"]["pdf"], **kw)
                e = m[name]["native"]
                axes[row, 2].loglog(k, k * np.where(k <= e["k_nyquist"], e["pk"], np.nan), **kw)
        axes[row, 0].set(title=f"{title}: PDF of log S, native, 60 Myr", xlabel="log S - median", ylim=(1e-4, 3))
        axes[row, 1].set(title=f"{title}: PDF of log S, 80 pc beam", xlabel="log S - median", ylim=(1e-4, 3))
        axes[row, 2].set(title=f"{title}: power of log S, 1 < |z| < 5 kpc", xlabel="k [cycles / kpc]",
                         ylabel="k P(k)")  # fmt: skip
        for ax in axes[row]:
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fvfd_stats_maps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    stats = collect()
    for path in (fig_pdfs(stats), fig_spectra(stats), fig_clumps(stats), fig_maps(stats)):
        print(f"wrote {path}")
    table = scalar_table(stats)
    print("\n| quantity | FD-1024 | FV-1024 | z (1024) | FD-2048 | FV-2048 | z (2048) | (FD - FV) / mean at 2048 | "
          "verdict 2048 |")  # fmt: skip
    summary = {}
    for name, values in table.items():
        rows = compare(values)
        summary[name] = {str(n): r for n, r in rows.items()}
        cols = []
        for n in (1024, 2048):
            r = rows.get(n)
            cols += [f"{r['FD']:.4g}", f"{r['FV']:.4g}", f"{r['z']:+.1f}"] if r else ["-", "-", "-"]
        r = rows.get(2048)
        cols += [f"{100 * r['rel']:+.0f} %", r["verdict"]] if r else ["-", "-"]
        print(f"| {name} | " + " | ".join(cols) + " |")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"summary": summary, "members": {k: {f"{c[0]}-{c[1]}": v for c, v in d.items()}
                                                       for k, d in table.items()}}, fh, indent=1, default=float)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
