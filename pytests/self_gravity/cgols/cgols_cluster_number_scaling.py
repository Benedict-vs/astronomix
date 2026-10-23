"""Outflow vs number of injection regions for model C at 256^3 (appendix figure).

Reads the frames' ``mdot_z`` of the cluster-number sweep and plots, against N:
the net outflow through |z| = 1, 5, 8 kpc (mean over the high state 15-40 Myr,
and the peak) and the cumulative mass through |z| = 5 kpc at 75 Myr, for

  S1  fixed total injection volume, R = 300 pc / N^(1/3)   (_C256n<N>, N=16: _C256)
  S2  fixed cluster radius R = 119 pc                      (_C256R119n<N>)
  S3  other random draws at N = 16 / 64                    (_C256s2/s3, _C256n64s2/s3)

with the central-injection model A256 as reference and a log2(N) fit to S1.
Runs below R = 3 dx (N >= 32 in S1 at 256^3) are marked.

    python cgols_cluster_number_scaling.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures", "cgols", "comparisons")
BOX_Z = 20.0  # kpc
DX_PC = 10_000.0 / 256  # 256^3 cell size, pc

S1 = {1: "C256n1", 2: "C256n2", 4: "C256n4", 8: "C256n8", 16: "C256",
      32: "C256n32", 64: "C256n64", 128: "C256n128"}
S2 = {1: "C256R119n1", 2: "C256R119n2", 4: "C256R119n4", 8: "C256R119n8",
      16: "C256", 32: "C256R119n32", 64: "C256R119n64"}
SEEDS = {16: ("C256", "C256s2", "C256s3"), 64: ("C256n64", "C256n64s2", "C256n64s3")}
REFERENCE = "A256"


def load(run):
    """Times [Myr] and mdot_z [Msun/yr] (frames x z-planes) of a run, or None."""
    candidates = [os.path.join(SCRIPT_DIR, f"cgols_snapshots_{run}")] + [
        os.path.join(SCRIPT_DIR, f"cgols_snapshots_{group}", run) for group in ("Nsweep", "variants")
    ]
    frame_dir = next((d for d in candidates if os.path.isdir(d)), None)
    if frame_dir is None:
        return None
    files = sorted(f for f in os.listdir(frame_dir) if f.startswith("frame_") and f.endswith(".npz"))
    frames = [np.load(os.path.join(frame_dir, f)) for f in files]
    t = np.array([float(fr["time_myr"]) for fr in frames])
    order = np.argsort(t)
    return t[order], np.stack([fr["mdot_z"] for fr in frames])[order]


def metrics(run):
    """{H: (mean 15-40 Myr, peak)} for H in 1/5/8 kpc, and M_out(5 kpc, 75 Myr)."""
    data = load(run)
    if data is None:
        return None
    t, mdot = data
    z = (np.arange(mdot.shape[1]) + 0.5) * BOX_Z / mdot.shape[1] - BOX_Z / 2
    high = (t >= 15) & (t <= 40)
    out = {}
    for H in (1.0, 5.0, 8.0):
        rate = mdot[:, np.argmin(np.abs(z - H))] - mdot[:, np.argmin(np.abs(z + H))]
        out[H] = (rate[high].mean(), rate.max())
        if H == 5.0:
            out["cumulative"] = np.trapezoid(rate, t) * 1e6  # Msun
    return out


def outflow_series(run, height=5.0):
    """Times [Myr] and net outflow [Msun/yr] through |z| = height for one run."""
    t, mdot = load(run)
    z = (np.arange(mdot.shape[1]) + 0.5) * BOX_Z / mdot.shape[1] - BOX_Z / 2
    return t, mdot[:, np.argmin(np.abs(z - height))] - mdot[:, np.argmin(np.abs(z + height))]


def main():
    fixed_volume = {n: metrics(r) for n, r in S1.items()}
    fixed_radius = {n: metrics(r) for n, r in S2.items()}
    ref = metrics(REFERENCE)
    seeds = {n: [m for m in (metrics(r) for r in runs) if m is not None] for n, runs in SEEDS.items()}

    figure = plt.figure(figsize=(15, 5.2))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    # Left: outflow through |z| = 5 kpc against time, one line per number of regions.
    axis = figure.add_subplot(grid[0])
    colormap = plt.get_cmap("viridis")
    ns = [n for n in S1 if fixed_volume[n] is not None]
    for k, n in enumerate(ns):
        t, rate = outflow_series(S1[n])
        axis.plot(t, rate, color=colormap(k / max(len(ns) - 1, 1)), lw=1.5,
                  label=f"N = {n} (R = {300 / n ** (1 / 3):.0f} pc)")
    t, rate = outflow_series(REFERENCE)
    axis.plot(t, rate, color="0.25", ls="--", lw=1.4, label="model A: one central sphere")
    axis.plot([0, 5, 5, 10, 40, 45, 75], [0, 0, 1.5, 12, 12, 1.5, 1.5], color="0.6", ls=":", lw=1,
              label=r"injected $\dot M$")
    axis.axvspan(15, 40, color="0.9", zorder=0)
    axis.text(27.5, axis.get_ylim()[1] * 0.97, "averaging window", ha="center", va="top", fontsize=8, color="0.4")
    axis.set(xlabel="t [Myr]", ylabel=r"net outflow through |z| = 5 kpc [$M_\odot$/yr]", xlim=(0, 75),
             title="Outflow over time for N injection regions")
    axis.legend(fontsize=7.5, ncol=2, loc="upper right")

    # Right: time-averaged outflow (15-40 Myr) against N, at three heights.
    axis = figure.add_subplot(grid[1])
    for height, color in ((1.0, "C1"), (5.0, "C3"), (8.0, "C0")):
        n = np.array(ns)
        axis.plot(n, [fixed_volume[k][height][0] for k in n], "o-", color=color,
                  label=f"|z| = {height:g} kpc")
        nr = np.array([k for k in S2 if fixed_radius[k] is not None])
        axis.plot(nr, [fixed_radius[k][height][0] for k in nr], "s", mfc="none", color=color, ms=7, alpha=0.7,
                  label="same, fixed radius 119 pc" if height == 1.0 else None)
        for k, runs in seeds.items():
            values = [m[height][0] for m in runs]
            axis.errorbar(k, np.mean(values), yerr=np.std(values, ddof=1), fmt="none", ecolor="k", capsize=4,
                          label="spread of 3 random cluster draws" if (height == 1.0 and k == 16) else None)
        if ref is not None:
            axis.axhline(ref[height][0], color=color, ls="--", lw=0.9, alpha=0.7)
    fit = np.polyfit(np.log2(ns), [fixed_volume[k][5.0][0] for k in ns], 1)
    axis.text(0.98, 0.03, f"|z| = 5 kpc: {fit[1]:.1f} + {fit[0]:.1f} log$_2$N $M_\\odot$/yr\ndashed: model A",
              transform=axis.transAxes, ha="right", fontsize=8)
    axis.set_xscale("log", base=2)
    axis.set(xlabel="number of injection regions N", ylabel=r"mean outflow 15–40 Myr [$M_\odot$/yr]",
             title="Time-averaged outflow vs N")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7.5, loc="upper left")

    figure.suptitle("Model C, adiabatic, 256$^3$: same total injected mass, energy and volume, split into N regions")
    figure.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "model_c_cluster_number_scaling_256.png")
    figure.savefig(out, dpi=170)
    print(f"wrote {out}")

    print(f"{'N':>4} {'R [pc]':>7} {'5kpc':>7} {'1kpc':>7} {'8kpc':>7} {'M5 [1e8 Msun]':>14}")
    for n in ns:
        m = fixed_volume[n]
        print(f"{n:>4} {300 / n ** (1 / 3):7.0f} {m[5.0][0]:7.2f} {m[1.0][0]:7.2f} {m[8.0][0]:7.2f} {m['cumulative'] / 1e8:14.2f}")
    if ref:
        print(f"   A {'300':>7} {ref[5.0][0]:7.2f} {ref[1.0][0]:7.2f} {ref[8.0][0]:7.2f} {ref['cumulative'] / 1e8:14.2f}")


if __name__ == "__main__":
    main()
