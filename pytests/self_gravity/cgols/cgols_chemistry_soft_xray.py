"""Soft X-rays (AtomDB 0.3-2 keV) of the chemistry run with the field of every cell against model B, at 256^3.

usage: python cgols_chemistry_soft_xray.py [out.png]   (from this directory, CPU, seconds)

From the caches of cgols_emission.py (emission_B256, emission_chem256fuv_v3): edge-on maps at 60 Myr on one
colour scale, the luminosity against time (inside and outside |z| = 1 kpc, from the minor-axis profiles), and
the minor-axis profile averaged over 50-70 Myr with the ratio of the two runs.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures/cgols/chem256_v2/soft_xray_chemistry_vs_model_b.png")
RUNS = (
    ("emission_B256", "model B (cooling curve)", "#2f6fb3"),
    ("emission_chem256fuv_v3", "chemistry (field of every cell)", "#d9772b"),
)
SURFACE, MUTED, GRID = "#fcfcfb", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE})

data = {cache: np.load(os.path.join(HERE, "data/studies", cache + ".npz"), allow_pickle=True) for cache, _, _ in RUNS}
figure = plt.figure(figsize=(16, 6.6))
grid = figure.add_gridspec(2, 4, width_ratios=[0.8, 0.8, 1.25, 1.25], height_ratios=[1, 1], wspace=0.35, hspace=0.35)

# maps at 60 Myr, one scale
norm = LogNorm(1e-10, 1e-4)
for column, (cache, label, _) in enumerate(RUNS):
    axis = figure.add_subplot(grid[:, column])
    image = axis.imshow(np.asarray(data[cache]["map__xray_atomdb_xz"], float).T, origin="lower", extent=(-5, 5, -10, 10),
                        cmap=plt.get_cmap("inferno").with_extremes(bad="black", under="black"), norm=norm, interpolation="nearest")
    axis.set_title(f"{label}\n{float(data[cache]['meta__map_time_myr']):.0f} Myr, 0.3-2 keV", fontsize=9.5)
    axis.set_xlabel("x [kpc]")
    if column == 0:
        axis.set_ylabel("z [kpc]")
cbar = figure.colorbar(image, ax=figure.axes[:2], orientation="horizontal", fraction=0.04, pad=0.09)
cbar.set_label(r"S$_X$ [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]")

luminosity_axis = figure.add_subplot(grid[0, 2:])
profile_axis = figure.add_subplot(grid[1, 2])
ratio_axis = figure.add_subplot(grid[1, 3])
profiles = {}
for cache, label, colour in RUNS:
    d = data[cache]
    time = d["per_snapshot__time_myr"]
    luminosity = 0.5 * (d["per_snapshot__L_xray_atomdb_xz"] + d["per_snapshot__L_xray_atomdb_yz"])
    luminosity_axis.plot(time, luminosity, color=colour, lw=1.8, label=label + ", all")
    z = d["per_snapshot__profile_z_kpc"]
    z = z[0] if z.ndim > 1 else z
    stack = np.mean([d[f"per_snapshot__profile_xray_atomdb_{plane}_{side}"] for plane in ("xz", "yz") for side in ("north", "south")], axis=0)
    # the light of |z| > 1 kpc from the profiles (surface brightness x width of the box x dz, both sides): relative to all
    outside = stack[:, z > 1.0].sum(axis=1) / stack.sum(axis=1)
    luminosity_axis.plot(time, luminosity * outside, color=colour, lw=1.4, ls="--", label=label + ", |z| > 1 kpc")
    window = (time >= 50) & (time <= 70)
    profiles[cache] = stack[window].mean(axis=0)
    profile_axis.plot(z, profiles[cache], color=colour, lw=1.8, label=label)
ratio_axis.plot(z, profiles["emission_chem256fuv_v3"] / profiles["emission_B256"], color="#55524c", lw=1.8)
ratio_axis.axhline(1.0, color=MUTED, lw=0.8)
luminosity_axis.set_yscale("log")
luminosity_axis.set_xlabel("time [Myr]")
luminosity_axis.set_ylabel(r"L$_X$ 0.3-2 keV [erg s$^{-1}$]")
luminosity_axis.set_xlim(0, 75)
luminosity_axis.legend(fontsize=8, frameon=False, ncol=2, loc="lower center")
profile_axis.set_yscale("log")
profile_axis.set_xlabel("height |z| [kpc]")
profile_axis.set_ylabel("S$_X$, mean 50-70 Myr", fontsize=9)
ratio_axis.set_yscale("log")
ratio_axis.set_xlabel("height |z| [kpc]")
ratio_axis.set_ylabel("chemistry / model B")
ratio_axis.set_ylim(0.1, 2.0)
ratio_axis.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0])
ratio_axis.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
ratio_axis.minorticks_off()
for axis in (luminosity_axis, profile_axis, ratio_axis):
    axis.grid(color=GRID, lw=0.6)
figure.suptitle("Soft X-rays at 256$^3$: the light is the central kpc; above it the chemistry run is 2-6x fainter", y=0.99, fontsize=11)
figure.savefig(OUT, dpi=130, bbox_inches="tight")
print("written", OUT)
