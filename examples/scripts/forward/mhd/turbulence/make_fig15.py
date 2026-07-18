"""
HOW-MHD (Seo & Ryu 2023) Fig. 15 reproduction -- plain matplotlib style.
This is the paper `turbulence_spectra` figure.

Time-averaged power spectra of the isothermal-MHD turbulence runs, 2x3 grid:

  rows    : ISM (M_turb ~ 10) / ICM (M_turb ~ 0.5)
  columns : density P_rho, kinetic E_K, magnetic E_B

Each panel overlays every resolution present in the script-relative data/ dir
(paper: 256^3 and 512^3; a cheaper resolution such as 128^3 is picked up
automatically) so the convergence of the inertial range is visible. Spectra are
averaged over the saturated window (last third of the snapshots). A k^-5/3 guide
is drawn in each panel.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# script-relative dirs (data/ shared with paper_turbulence.py)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "data"
FIG = SCRIPT_DIR / "figures"
FIG.mkdir(exist_ok=True)

# resolutions to look for, low to high (paper runs: 256 and 512; 128 = cheap default)
RESOLUTIONS = ("128", "256", "512")


def load(tag):
    f = DATA / f"paper_{tag}.npz"
    return dict(np.load(f, allow_pickle=True)) if f.exists() else None


def series(case):
    """All resolutions present for a case, low to high."""
    out = []
    for res in RESOLUTIONS:
        d = load(f"{case}_N{res}")
        if d is not None:
            out.append((int(d["N"]), d))
    return out


def saturated_spectrum(d, key):
    """Average a spectrum over the saturated window (last third of snapshots)."""
    sp = np.asarray(d[key], dtype=np.float64)
    n = sp.shape[0]
    take = sp[max(1, n - max(1, n // 3)):]
    return np.nanmean(take, axis=0)


cases = ["ISM", "ICM"]
cols = [
    ("spec_rho", r"$P_\rho(k)$"),
    ("spec_EK", r"$E_K(k)$"),
    ("spec_EB", r"$E_B(k)$"),
]
colors = {128: "tab:green", 256: "tab:blue", 448: "tab:orange", 512: "tab:red"}
K0 = 2.0 * np.pi  # box fundamental wavenumber (box_size = 1) -> x-axis is k/k_0

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
for i, case in enumerate(cases):
    runs = series(case)
    for j, (key, ylab) in enumerate(cols):
        ax = axes[i, j]
        ymax = -np.inf
        for N, d in runs:
            k = np.asarray(d["k"], dtype=np.float64)
            sp = saturated_spectrum(d, key)
            x = k / K0
            # keep box scale .. ~85% of Nyquist (drop sub-fundamental + grid-noise tail)
            m = (sp > 0) & np.isfinite(sp) & (x >= 0.8) & (x <= 0.42 * N)
            ax.loglog(x[m], sp[m], lw=1.4, color=colors.get(N, None), label=f"$N={N}^3$")
            ymax = max(ymax, np.nanmax(sp[m]))
        # k^-5/3 guide anchored in the inertial range, lifted above the data
        if np.isfinite(ymax):
            xg = np.array([2.0, 20.0])
            yg = 1.6 * ymax * (xg / xg[0]) ** (-5.0 / 3.0)
            ax.plot(xg, yg, "k--", lw=1.0)
            ax.text(xg[0] * 1.6, yg[0], r"$k^{-5/3}$", fontsize=9, va="bottom")
            ax.set_ylim(ymax / 3e4, ymax * 4)  # ~4.5 decades, clips the noise floor
        ax.set_xlim(0.8, None)
        ax.set_xlabel(r"$k/k_0$")
        ax.set_ylabel(ylab)
        if i == 0:
            ax.set_title(["density", "kinetic", "magnetic"][j] + " spectrum")
        ax.text(0.04, 0.06, f"{case}", transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.grid(True, which="both", alpha=0.15)

fig.tight_layout()
out = FIG / "turbulence_spectra"
fig.savefig(str(out) + ".png", dpi=200)
fig.savefig(str(out) + ".svg")
print("wrote", str(out) + ".png/.svg")
for case in cases:
    for N, d in series(case):
        print(f"{case} N={N}^3: M_turb(last)={float(d['Ms_t'][-1]):.2f}, "
              f"snaps={d['spec_EK'].shape[0]}")
