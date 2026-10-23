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


def main():
    s1 = {n: metrics(r) for n, r in S1.items()}
    s2 = {n: metrics(r) for n, r in S2.items()}
    ref = metrics(REFERENCE)
    seeds = {n: [m for m in (metrics(r) for r in runs) if m is not None] for n, runs in SEEDS.items()}

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, H in zip(axes, (1.0, 5.0, 8.0)):
        for series, label, marker, color in ((s1, "S1: fixed total volume", "o", "C3"),
                                             (s2, "S2: fixed R = 119 pc", "s", "C0")):
            n = np.array([k for k, v in series.items() if v is not None])
            mean = np.array([series[k][H][0] for k in n])
            peak = np.array([series[k][H][1] for k in n])
            axis.plot(n, mean, marker=marker, color=color, ls="-", label=f"{label} (15–40 Myr mean)")
            axis.plot(n, peak, marker=marker, color=color, ls=":", mfc="none", alpha=0.7,
                      label=f"{label} (peak)")
            if series is s1:
                under = n[(300.0 / n ** (1 / 3)) < 3 * DX_PC]
                axis.plot(under, [series[k][H][0] for k in under], "x", color="k", ms=9,
                          label="R < 3 dx" if H == 1.0 else None)
                fit = np.polyfit(np.log2(n), mean, 1)
                axis.plot(n, np.polyval(fit, np.log2(n)), color="C3", lw=0.8, alpha=0.5)
                axis.text(0.03, 0.04, f"S1 fit: {fit[1]:.1f} + {fit[0]:.2f} log$_2$N",
                          transform=axis.transAxes, fontsize=8, color="C3")
        for n, runs in seeds.items():
            values = [m[H][0] for m in runs]
            axis.errorbar(n * 1.08, np.mean(values), yerr=np.std(values, ddof=1) if len(values) > 1 else 0,
                          fmt="D", color="0.3", ms=4, capsize=3,
                          label="3 random draws (mean ± 1σ)" if (H == 1.0 and n == 16) else None)
        if ref is not None:
            axis.axhline(ref[H][0], color="0.5", ls="--", lw=1, label="A256 central (mean)" if H == 1.0 else None)
        axis.set_xscale("log", base=2)
        axis.set(xlabel="number of injection regions N", title=f"net outflow through |z| = {H:g} kpc")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\dot M_{\rm out}$ [$M_\odot$ yr$^{-1}$]")
    axes[0].legend(fontsize=7, loc="upper left")
    figure.suptitle("Model C (adiabatic, 256$^3$): outflow vs number of injection regions, same total injected mass and energy")
    figure.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "model_c_cluster_number_scaling_256.png")
    figure.savefig(out, dpi=170)
    print(f"wrote {out}")

    print(f"{'N':>4} {'S1 5kpc':>8} {'S2 5kpc':>8} {'S1 1kpc':>8} {'S1 M5 [1e8]':>12}")
    for n in S1:
        a, b = s1.get(n), s2.get(n)
        print(f"{n:>4} {a[5.0][0]:8.2f} {(b[5.0][0] if b else float('nan')):8.2f} {a[1.0][0]:8.2f} {a['cumulative'] / 1e8:12.2f}")
    if ref:
        print(f"A256 {ref[5.0][0]:8.2f} {'':8} {ref[1.0][0]:8.2f} {ref['cumulative'] / 1e8:12.2f}")


if __name__ == "__main__":
    main()
