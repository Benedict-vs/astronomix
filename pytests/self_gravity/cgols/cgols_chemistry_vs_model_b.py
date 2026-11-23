"""The disk of model B (cooling curve) against the two chemistry runs, at the same time.

usage: python cgols_chemistry_vs_model_b.py <model B frame.npz> <chemistry frame, field per cell> <chemistry frame, one field> <out.png>
           [chemistry frame, field per cell, corrected H2 + H+ rate]

Model B has no composition, so the three are compared in P / rho, written as
T / mu [K] (for neutral atomic gas mu = 1.27, for ionised gas 0.6). Maps of the
edge-on plane around the disk and the mass-weighted mean in rings.
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

frame_b, frame_field, frame_one, out = sys.argv[1:5]
SURFACE, TEXT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
MASS_PER_HYDROGEN = 1.0 + 4 * 0.085 + 16 * 3.2e-4 + 12 * 1.4e-4 + 28 * 1.8e-6  # the network's composition [m_p]


def chemistry(data):
    """rho [m_p cm^-3] and T / mu [K] of a chemistry frame."""
    particles = sum(np.asarray(data[key], dtype=float) for key in data.files if key.startswith("x_") and key.endswith("_xz"))
    return np.asarray(data["n_H_xz"], dtype=float) * MASS_PER_HYDROGEN, np.asarray(data["T_xz"], dtype=float) * particles / MASS_PER_HYDROGEN


def cooling_curve(data):
    """The same of a frame without chemistry (n and T are stored for mu = 0.6)."""
    return np.asarray(data["n_xz"], dtype=float) * 0.6, np.asarray(data["T_xz"], dtype=float) / 0.6


runs = [
    ("model B: cooling curve, FUV field of every cell", cooling_curve(np.load(frame_b)), "#2f6fb3"),
    ("chemistry, FUV field of every cell", chemistry(np.load(frame_field)), "#d9772b"),
    ("chemistry, one Draine field everywhere", chemistry(np.load(frame_one)), "#55524c"),
]
if len(sys.argv) > 5:
    runs[1] = ("chemistry, FUV field of every cell, network file as shipped", runs[1][1], runs[1][2])
    runs.insert(2, ("chemistry, FUV field of every cell, corrected H$_2$ + H$^+$ rate", chemistry(np.load(sys.argv[5])), "#7b4fa3"))
time_myr = float(np.load(frame_field)["time_myr"])
nx, nz = runs[0][1][0].shape
x = (np.arange(nx) + 0.5 - nx / 2) * 10.0 / nx
z = (np.arange(nz) + 0.5 - nz / 2) * 20.0 / nz
radius = np.abs(x)[:, None] * np.ones((1, nz))
height = np.abs(z)[None, :] * np.ones((nx, 1))

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})
rows = len(runs)
figure = plt.figure(figsize=(13.5, 2.2 * rows), facecolor=SURFACE)
grid = figure.add_gridspec(rows, 4, width_ratios=[1.25, 0.03, 0.2, 0.9], wspace=0.05, hspace=0.32)
ramp = LinearSegmentedColormap.from_list("orange", [SURFACE, "#e8b98c", "#7a2c08", "#2b0f02"])
norm = LogNorm(3e2, 3e6)
for row, (label, (rho, t_over_mu), colour) in enumerate(runs):
    axis = figure.add_subplot(grid[row, 0])
    image = axis.imshow(t_over_mu.T, origin="lower", extent=(-5, 5, -10, 10), cmap=ramp, norm=norm, aspect="equal", interpolation="nearest")
    axis.set_ylim(-1.0, 1.0)
    axis.set_ylabel("z [kpc]")
    axis.set_title(label, fontsize=10, color=TEXT, loc="left")
    axis.plot([-4.95, -4.6], [0.85, 0.85], color=colour, lw=4, solid_capstyle="butt")
    if row == rows - 1:
        axis.set_xlabel("x [kpc]")
    else:
        axis.set_xticklabels([])
figure.colorbar(image, cax=figure.add_subplot(grid[:, 1])).set_label(r"P / $\rho$ as T / $\mu$ [K]")

edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
centres = 0.5 * (edges[1:] + edges[:-1])
top, bottom = figure.add_subplot(grid[0 : rows - rows // 2, 3]), figure.add_subplot(grid[rows - rows // 2 :, 3])
print(f"disk gas (|z| < 0.5 kpc, rho > 0.135 m_p cm^-3) at {time_myr:.2f} Myr, mass-weighted in rings")
for label, (rho, t_over_mu), colour in runs:
    mean, cold = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (height < 0.5) & (rho > 0.135) & (radius >= low) & (radius < high)
        weight = (rho * radius)[mask]
        mean.append(np.exp((weight * np.log(t_over_mu[mask])).sum() / weight.sum()))
        cold.append(100.0 * weight[t_over_mu[mask] < 1.0e3].sum() / weight.sum())
    mask = (height < 0.5) & (rho > 0.135) & (radius < 4.5)
    weight = (rho * radius)[mask]
    print(f"   {label:66s} T/mu {np.exp((weight * np.log(t_over_mu[mask])).sum() / weight.sum()):6.0f} K, below 1000 K: {100.0 * weight[t_over_mu[mask] < 1.0e3].sum() / weight.sum():4.1f} % of the mass")
    top.plot(centres, mean, color=colour, lw=2, marker="o", ms=4)
    bottom.plot(centres, cold, color=colour, lw=2, marker="o", ms=4)
top.set_yscale("log")
top.set_ylabel(r"mass-weighted T / $\mu$ [K]")
top.set_xticklabels([])
top.set_title(f"disk gas, |z| < 0.5 kpc, {time_myr:.1f} Myr", fontsize=10, color=TEXT, loc="left")
bottom.set_ylabel("mass below\nT / $\\mu$ = 1000 K [%]")
bottom.set_xlabel("radius [kpc]")
bottom.set_ylim(0, 100)
for axis in (top, bottom):
    axis.yaxis.tick_right()
    axis.yaxis.set_label_position("right")
    axis.grid(color="#e4e3df", lw=0.6)
    axis.set_xlim(0, 4.5)
figure.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
print(f"written {out}")
