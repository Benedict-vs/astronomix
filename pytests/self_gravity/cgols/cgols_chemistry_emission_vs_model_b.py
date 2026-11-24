"""Emission of the chemistry run with the field of every cell against model B, at 256^3.

usage: python cgols_chemistry_emission_vs_model_b.py [out.png]   (from this directory, CPU, seconds)

Reads the caches of cgols_emission.py: data/studies/emission_B256_diskmask.npz (model B, its KI02 + GF12
temperature and collisional-equilibrium ionisation, from the full states) and
emission_chem256fuv_v3_diskmask.npz (the chemistry run, maps made in the run with the network's
temperature, electrons and H+). Both with H-alpha and the emission measure masked within |z| < 0.5 kpc.
Luminosities against time (edge-on, line of sight along y and along x averaged), and the minor-axis
profiles averaged over 50-70 Myr.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures/cgols/chem256_v2/emission_chemistry_vs_model_b.png")
RUNS = (
    ("emission_B256_diskmask", "model B (cooling curve, equilibrium ionisation)", "#2f6fb3"),
    ("emission_chem256fuv_v3_diskmask", "chemistry, field of every cell (network's ionisation)", "#d9772b"),
)
QUANTITIES = (
    ("xray_atomdb", "soft X-ray 0.3-2 keV (AtomDB), L [erg/s]"),
    ("xray_cgols", "X-ray, CGOLS I recipe, L [erg/s]"),
    ("halpha", r"H$\alpha$ (|z| > 0.5 kpc), L [erg/s]"),
)
SURFACE, MUTED, GRID = "#fcfcfb", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE})

figure, axes = plt.subplots(2, 3, figsize=(16, 8.2), gridspec_kw=dict(hspace=0.32, wspace=0.28))
print("means over 50-70 Myr [erg/s]")
for cache, label, colour in RUNS:
    data = np.load(os.path.join(HERE, "data/studies", cache + ".npz"), allow_pickle=True)
    time = data["per_snapshot__time_myr"]
    z = data["per_snapshot__profile_z_kpc"]
    z = z[0] if z.ndim > 1 else z
    window = (time >= 50.0) & (time <= 70.0)
    line = []
    for column, (key, title) in enumerate(QUANTITIES):
        luminosity = 0.5 * (data[f"per_snapshot__L_{key}_xz"] + data[f"per_snapshot__L_{key}_yz"])
        axes[0, column].plot(time, luminosity, color=colour, lw=1.8, label=label)
        axes[0, column].set_title(title, fontsize=10, loc="left")
        profiles = [data[f"per_snapshot__profile_{key}_{plane}_{side}"][window] for plane in ("xz", "yz") for side in ("north", "south")]
        axes[1, column].plot(z, np.mean(profiles, axis=(0, 1)), color=colour, lw=1.8)
        line.append(f"{key} {np.mean(luminosity[window]):.2e}")
    print(f"   {label:56s}" + "  ".join(line))
for axis in axes[0]:
    axis.set_yscale("log")
    axis.set_xlabel("time [Myr]")
    axis.set_xlim(0, 75)
for axis, (key, _) in zip(axes[1], QUANTITIES):
    axis.set_yscale("log")
    axis.set_xlabel("height |z| [kpc]")
    axis.set_ylabel("surface brightness, mean 50-70 Myr\n" + ("[erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$]" if key == "halpha" else "[erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]"), fontsize=9)
for axis in axes.flat:
    axis.grid(color=GRID, lw=0.6)
axes[0, 0].legend(fontsize=8.5, frameon=False)
figure.suptitle("Emission at 256$^3$: model B against the chemistry run (minor-axis profiles below)", y=0.97, fontsize=11)
figure.savefig(OUT, dpi=130, bbox_inches="tight")
print("written", OUT)
