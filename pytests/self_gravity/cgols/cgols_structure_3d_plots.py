"""
Resolution convergence of the 3D small-scale structure of the CGOLS A-series outflow (FD WENO5 vs FV VL2), from the
per-checkpoint caches of cgols_structure_3d.py (data/studies/structure3d_v2<TAG>.npz, falling back to v1).

Quantities (both hemispheres summed, slab 1 < |z| < 4.5 kpc; entropy classes of cgols_convergence_3d):
  mixed gas         mass and volume with 5.5 <= log K < 6.8
  mixing surface    area of the log K = 6.8 isosurface (mixed | wind), A = (2/3) N_faces dx^2, and the mean
                    mixed-layer thickness V_mixed / A
  entrained gas     log K < 5.5: attached to the slab face at |z| = 1 kpc (the disk surface at low N) and detached
                    clumps (v2 caches only; a v1 cache has only the total, which the disk dominates at N <= 256)
  clumps            detached clump number, mass in clumps, mass and size functions (v2)
  3D spectra        log n, log T, |v| (box |x|, |y| < 2.5 kpc, 1 < |z| < 3.5 kpc), shown as the variance per
                    ln k, 4 pi k^3 P(k); ratio to the same scheme at twice N and to the other scheme at the same N;
                    converged scale l_c = smallest scale down to which the 3-bin-mean ratio to 2N stays within 20 %
                    (judged from 500 pc down, where the bins hold >= 300 modes, to 1/2 of the 2N Nyquist k)

Ensembles: where seed members exist (FV-1024: base rerun + seeds 2, 3) the mean is shown and the member spread
(std / mean) is printed and drawn as error bars. Window means only where every even checkpoint of the window exists.

    python cgols_structure_3d_plots.py [--windows 15-30,30-45,45-75]

Writes figures/cgols/comparisons/structure3d_{timeseries,convergence,clumps,spectra}.png and prints the tables.
CPU only.
"""

# ruff: noqa: E402

# general
import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# numerics
import numpy as np

# plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CGOLS analysis
from cgols_cost_accuracy import SPEC_TOL, smooth3
from cgols_structure_3d import CLUMP_MASS_EDGES, CLUMP_RADIUS_EDGES, K_EDGES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "studies")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures", "cgols", "comparisons")
BOX_KPC = 10.0  # x extent of the A-series box (dx = BOX_KPC / N)

#: (scheme, N) -> cache tags (several = seed ensemble)
RUNS = {
    ("FD", 128): ["_A128"], ("FD", 256): ["_A256"], ("FD", 512): ["_A512"], ("FD", 1024): ["_A1024"],
    ("FD", 2048): ["_A2048"],
    ("FV", 128): ["_A128fv"], ("FV", 256): ["_A256fv"], ("FV", 512): ["_A512fvf32"],
    ("FV", 1024): ["_A1024fvr", "_A1024fvs2", "_A1024fvs3"], ("FV", 2048): ["_A2048fv"],
}
NS = (128, 256, 512, 1024, 2048)
SCHEMES = ("FD", "FV")
COLORS = dict(zip(NS, plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(NS)))))
LS = {"FD": "-", "FV": "--"}
MARK = {"FD": "o", "FV": "s"}
K_MID = np.sqrt(K_EDGES[1:] * K_EDGES[:-1])
#: largest scale judged for the 3D spectra: the box is only 2.5 kpc tall, so the bins near 1 kpc hold < 150 modes
#: and scatter by +-30 % (the slice spectra of cgols_cost_accuracy are judged from 1 kpc)
SPEC_LMAX_PC = 500.0

#: scalar quantities: name -> (label, function of a record -> value, summed over hemispheres)
SCALARS = {
    "mass_mixed": (r"mixed mass [M$_\odot$]", lambda r: _hsum(r, "mass_mixed_msun")),
    "vol_mixed": (r"mixed volume [kpc$^3$]", lambda r: _hsum(r, "vol_mixed_kpc3")),
    "area_wind": (r"area log K = 6.8 [kpc$^2$]", lambda r: _hsum(r, "area_wind_kpc2")),
    "thickness": ("mixed-layer thickness V/A [pc]",
                  lambda r: 1e3 * _hsum(r, "vol_mixed_kpc3") / max(_hsum(r, "area_wind_kpc2"), 1e-12)),
    "mass_ent": (r"entrained mass, all [M$_\odot$]", lambda r: _hsum(r, "mass_ent_msun")),
    "mass_att": (r"entrained, attached to disk [M$_\odot$]", lambda r: _hsum(r, "clump_att_mass")),
    "mass_det": (r"mass in detached clumps [M$_\odot$]", lambda r: _hsum(r, "clump_det_mass_total")),
    "n_det": ("number of detached clumps", lambda r: _hsum(r, "clump_det_n")),
    "area_det": (r"detached-clump surface [kpc$^2$]", lambda r: _hsum(r, "area_ent_det_kpc2")),
}
V2_ONLY = ("mass_att", "mass_det", "n_det", "area_det")
FIELDS = {"logn": r"log$_{10}$ n", "logT": r"log$_{10}$ T", "v": "v [km/s]"}


def _hsum(r, key):
    return float(r["north"][key] + r["south"][key])


def load(tag):
    """(records {step: record}, cache version) of one run, preferring the v2 cache."""
    for version in ("v2", "v1"):
        path = os.path.join(DATA_DIR, f"structure3d_{version}{tag}.npz")
        if os.path.exists(path):
            return np.load(path, allow_pickle=True)["records"].item(), version
    return None, None


def load_all():
    runs = {}
    for cell, tags in RUNS.items():
        members = []
        for tag in tags:
            recs, version = load(tag)
            if recs:
                members.append({"tag": tag, "recs": recs, "version": version})
        if members:
            runs[cell] = members
    return runs


def series(member, name):
    steps = sorted(member["recs"])
    if name in V2_ONLY and member["version"] != "v2":
        return None, None
    t = np.array([member["recs"][s]["time_myr"] for s in steps])
    y = np.array([SCALARS[name][1](member["recs"][s]) for s in steps])
    return t, y


def window_records(member, window):
    """Records of the even checkpoints in [t0, t1) if all of them exist (2.5 Myr cadence; FV-512: every 4th)."""
    recs = member["recs"]
    sel = [s for s in sorted(recs) if window[0] - 0.01 <= recs[s]["time_myr"] < window[1] - 0.01]
    if not sel:
        return None
    t = [recs[s]["time_myr"] for s in sel]
    cadence = np.median(np.diff(t)) if len(t) > 1 else 2.5
    expected = int(round((window[1] - window[0]) / cadence))
    return [recs[s] for s in sel] if len(sel) >= expected else None


def window_mean(members, name, window):
    """(ensemble mean, member spread std/mean or None) of the window mean of a scalar."""
    vals = []
    for m in members:
        if name in V2_ONLY and m["version"] != "v2":
            return None, None
        recs = window_records(m, window)
        if recs is None:
            return None, None
        vals.append(np.mean([SCALARS[name][1](r) for r in recs]))
    vals = np.array(vals)
    spread = float(np.std(vals, ddof=1) / abs(vals.mean())) if len(vals) > 1 and vals.mean() else None
    return float(vals.mean()), spread


def window_spectrum(members, field, window):
    """(ensemble-mean hemisphere-mean spectrum, member spread per bin or None, Nyquist k) over the window."""
    per_member = []
    nyq = None
    for m in members:
        recs = window_records(m, window)
        if recs is None:
            return None, None, None
        per_member.append(np.nanmean([0.5 * (r["north"][f"pk_{field}"] + r["south"][f"pk_{field}"])
                                      for r in recs], axis=0))
        nyq = recs[0]["k_nyquist"]
    arr = np.array(per_member)
    mean = np.nanmean(arr, axis=0)
    spread = np.nanstd(arr, axis=0, ddof=1) / mean if len(arr) > 1 else None
    return mean, spread, nyq


def window_hist(members, key, window):
    """Ensemble- and window-mean histogram (both hemispheres summed) of a v2 clump histogram."""
    out = []
    for m in members:
        if m["version"] != "v2":
            return None
        recs = window_records(m, window)
        if recs is None:
            return None
        out.append(np.mean([r["north"][key] + r["south"][key] for r in recs], axis=0))
    return np.mean(out, axis=0)


def converged_scale(ratio, k, kmax):
    """Smallest scale [pc] down to which the 3-bin-mean ratio stays within SPEC_TOL of 1, from SPEC_LMAX_PC down."""
    r = smooth3(np.where((k <= kmax) & (1e3 / k <= SPEC_LMAX_PC), ratio, np.nan))
    good = None
    for kk, rr in zip(k, r):
        if 1e3 / kk > SPEC_LMAX_PC or kk > kmax:
            continue
        if not np.isfinite(rr) or abs(rr - 1) > SPEC_TOL:
            break
        good = kk
    return None if good is None else 1e3 / good


def label(cell, members):
    s = f"{cell[0]}-{cell[1]}"
    if len(members) > 1:
        s += f" ({len(members)} seeds)"
    if members[0]["version"] == "v1":
        s += " [v1]"
    return s


def plot_timeseries(runs, path):
    names = ["mass_mixed", "vol_mixed", "area_wind", "thickness", "mass_ent", "mass_att", "mass_det", "n_det"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), sharex=True)
    for ax, name in zip(axes.ravel(), names):
        for cell, members in sorted(runs.items()):
            for i, m in enumerate(members):
                t, y = series(m, name)
                if t is None:
                    continue
                ax.plot(t, y, color=COLORS[cell[1]], ls=LS[cell[0]], lw=1.6 if i == 0 else 0.8,
                        alpha=1.0 if i == 0 else 0.6, label=label(cell, members) if i == 0 else None)
        ax.set_title(SCALARS[name][0], fontsize=10)
        if name not in ("thickness",):
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("t [Myr]")
    axes[0, 0].legend(fontsize=7.5, ncol=2)
    fig.suptitle("CGOLS A: 3D structure in the slab 1 < |z| < 4.5 kpc (both hemispheres); solid FD, dashed FV; "
                 "thin = seed members", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_convergence(runs, windows, path):
    names = ["mass_mixed", "vol_mixed", "area_wind", "thickness", "mass_ent", "mass_att", "mass_det", "n_det"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    table = {}
    wmarker = dict(zip(windows, ("v", "o", "s", "D")))
    for ax, name in zip(axes.ravel(), names):
        for scheme in SCHEMES:
            for w in windows:
                xs, ys, es = [], [], []
                for n in NS:
                    if (scheme, n) not in runs:
                        continue
                    mean, spread = window_mean(runs[(scheme, n)], name, w)
                    if mean is None:
                        continue
                    table[(name, scheme, n, w)] = (mean, spread)
                    xs.append(1e3 * BOX_KPC / n)
                    ys.append(mean)
                    es.append(0.0 if spread is None else spread * abs(mean))
                if xs:
                    ax.errorbar(xs, ys, yerr=es, ls=LS[scheme], marker=wmarker[w], ms=5, capsize=3,
                                color="tab:blue" if scheme == "FD" else "tab:orange",
                                alpha=0.35 + 0.65 * (windows.index(w) + 1) / len(windows),
                                label=f"{scheme} {w[0]:g}-{w[1]:g} Myr")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_title(SCALARS[name][0], fontsize=10)
        ax.set_xlabel("cell size dx [pc]")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Window means against resolution (error bar = seed spread where measured)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return table


def plot_clumps(runs, window, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    mlog = np.log10(CLUMP_MASS_EDGES)
    rlog = np.log10(CLUMP_RADIUS_EDGES)
    mmid = 10 ** (0.5 * (mlog[1:] + mlog[:-1]))
    rmid = 10 ** (0.5 * (rlog[1:] + rlog[:-1]))
    any_v2 = False
    for j, scheme in enumerate(SCHEMES):
        for n in NS:
            if (scheme, n) not in runs:
                continue
            members = runs[(scheme, n)]
            hm = window_hist(members, "clump_det_mass_hist", window)
            hr = window_hist(members, "clump_det_radius_hist", window)
            if hm is None:
                continue
            any_v2 = True
            lab = label((scheme, n), members)
            axes[0, j].step(mmid, hm / np.diff(mlog), where="mid", color=COLORS[n], label=lab)
            axes[1, j].step(rmid, hr / np.diff(rlog), where="mid", color=COLORS[n], label=lab)
            axes[1, j].axvline((3 / (4 * np.pi)) ** (1 / 3) * 1e3 * BOX_KPC / n, color=COLORS[n], lw=0.7, ls=":")
        axes[0, j].set_title(f"{scheme}: detached clumps (log K < 5.5), mass function, {window[0]:g}-{window[1]:g} Myr",
                             fontsize=10)
        axes[1, j].set_title(f"{scheme}: size function (dotted = radius of a 1-cell clump)", fontsize=10)
        axes[0, j].set_xlabel(r"clump mass [M$_\odot$]")
        axes[1, j].set_xlabel("equivalent radius [pc]")
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel(r"dN / dlog$_{10}$ (mean per snapshot)")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
    if not any_v2:
        plt.close(fig)
        return False
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def plot_spectra(runs, window, path):
    """Top: variance per ln k; middle: ratio to the same scheme at 2N; bottom: FV / FD at the same N."""
    fig, axes = plt.subplots(3, 3, figsize=(19, 13), sharex=True)
    scale = 1e3 / K_MID  # pc
    lc = {}
    spec = {}
    for cell, members in runs.items():
        for field in FIELDS:
            mean, spread, nyq = window_spectrum(members, field, window)
            if mean is not None:
                spec[(cell, field)] = (mean, spread, nyq)
    for i, field in enumerate(FIELDS):
        ax = axes[0, i]
        for (cell, f), (mean, spread, nyq) in sorted(spec.items()):
            if f != field:
                continue
            ok = K_MID <= nyq
            ax.plot(scale[ok], (4 * np.pi * K_MID**3 * mean)[ok], color=COLORS[cell[1]], ls=LS[cell[0]],
                    label=label(cell, runs[cell]))
            if spread is not None:
                band = 4 * np.pi * K_MID**3 * mean
                ax.fill_between(scale[ok], (band * (1 - spread))[ok], (band * (1 + spread))[ok],
                                color=COLORS[cell[1]], alpha=0.2, lw=0)
        ax.set_yscale("log")
        ax.set_title(f"{FIELDS[field]}: variance per ln k, {window[0]:g}-{window[1]:g} Myr", fontsize=10)
        ax.set_ylabel(r"4$\pi$ k$^3$ P(k)")
        # ratio to the same scheme at 2N
        ax = axes[1, i]
        ax.axhspan(1 - SPEC_TOL, 1 + SPEC_TOL, color="0.88", zorder=0)
        for scheme in SCHEMES:
            for n in NS[:-1]:
                lo, hi = ((scheme, n), field), ((scheme, 2 * n), field)
                if lo not in spec or hi not in spec:
                    continue
                kmax = 0.5 * spec[hi][2]
                ratio = spec[lo][0] / spec[hi][0]
                ok = K_MID <= spec[lo][2]
                ax.plot(scale[ok], ratio[ok], color=COLORS[n], ls=LS[scheme], label=f"{scheme}-{n} / {2 * n}")
                lc[(field, scheme, n)] = converged_scale(ratio, K_MID, min(kmax, spec[lo][2]))
        ax.set_yscale("log")
        ax.set_ylim(0.3, 3)
        ax.set_title(f"{FIELDS[field]}: N / 2N, same scheme (grey = within {SPEC_TOL:.0%})", fontsize=10)
        # FV / FD at the same N
        ax = axes[2, i]
        ax.axhspan(1 - SPEC_TOL, 1 + SPEC_TOL, color="0.88", zorder=0)
        for n in NS:
            a, b = (("FV", n), field), (("FD", n), field)
            if a not in spec or b not in spec:
                continue
            ok = K_MID <= spec[a][2]
            ax.plot(scale[ok], (spec[a][0] / spec[b][0])[ok], color=COLORS[n], label=f"FV / FD at {n}")
        ax.set_yscale("log")
        ax.set_ylim(0.3, 3)
        ax.set_title(f"{FIELDS[field]}: FV / FD, same N", fontsize=10)
        ax.set_xlabel("scale 1/k [pc]")
    for ax in axes.ravel():
        ax.set_xscale("log")
        ax.invert_xaxis() if not ax.xaxis_inverted() else None
        ax.axvline(SPEC_LMAX_PC, color="0.5", lw=0.8, ls=":")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("3D power spectra in the box |x|, |y| < 2.5 kpc, 1 < |z| < 3.5 kpc (hemisphere mean; band = seed "
                 "spread)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return lc


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--windows", default="15-30,30-45,45-75")
    args = parser.parse_args()
    windows = [tuple(float(x) for x in w.split("-")) for w in args.windows.split(",")]
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = load_all()
    for cell, members in sorted(runs.items()):
        print(f"{cell[0]}-{cell[1]}: " + ", ".join(f"{m['tag']} {m['version']} ({len(m['recs'])} ckpts)"
                                                  for m in members))
    plot_timeseries(runs, os.path.join(FIG_DIR, "structure3d_timeseries.png"))
    table = plot_convergence(runs, windows, os.path.join(FIG_DIR, "structure3d_convergence.png"))
    print("\nWindow means (seed spread in brackets where measured); change = value / value at half the resolution")
    for name in SCALARS:
        for w in windows:
            row = []
            for scheme in SCHEMES:
                prev = None
                for n in NS:
                    v = table.get((name, scheme, n, w))
                    if v is None:
                        prev = None
                        continue
                    sp = f" [{100 * v[1]:.1f}%]" if v[1] is not None else ""
                    ch = f" x{v[0] / prev:.2f}" if prev else ""
                    row.append(f"{scheme}{n} {v[0]:.3g}{sp}{ch}")
                    prev = v[0]
            if row:
                print(f"  {name:10s} {w[0]:2.0f}-{w[1]:2.0f}: " + " | ".join(row))
    late = windows[-1]
    if plot_clumps(runs, late, os.path.join(FIG_DIR, "structure3d_clumps.png")):
        print("\nwrote structure3d_clumps.png")
    else:
        print("\nno v2 caches yet: clump mass functions skipped")
    for w in windows[1:]:
        lc = plot_spectra(runs, w, os.path.join(FIG_DIR, f"structure3d_spectra_{w[0]:g}-{w[1]:g}.png"))
        print(f"\nConverged scale l_c [pc] (N vs 2N within {SPEC_TOL:.0%}, from {SPEC_LMAX_PC:.0f} pc down), "
              f"{w[0]:g}-{w[1]:g} Myr")
        for (field, scheme, n), v in sorted(lc.items()):
            dx = 1e3 * BOX_KPC / n
            s = "not within at any scale" if v is None else f"{v:5.0f} pc = {v / dx:4.1f} dx"
            print(f"  {field:5s} {scheme}-{n:4d} vs {2 * n}: {s}")
    print(f"\nfigures in {FIG_DIR}")


if __name__ == "__main__":
    main()
