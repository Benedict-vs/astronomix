"""Side-by-side comparison of model C at 1024^3 with N = 16 and N = 1000 clusters.

Same total injected mass, energy and volume, split into 16 or 1000 regions.
Top two rows (one per run): edge-on density slices at three times, the face-on
mid-plane density at the last time, and the edge-on soft X-ray surface
brightness at the last time, all from the in-situ frames. Bottom row: net
outflow through |z| = 5 kpc (frames), box L_X with the CGOLS I recipe (emission
caches) and the disc mass at |z| < 0.5 kpc (3D caches) against time.

CPU only, reads ~10 frames and three caches per run:

    python cgols_cluster_n1000_compare.py            # figures/cgols/comparisons/model_c_n16_vs_n1000.png
"""

# general
import os
import sys

# plotting
import matplotlib

# numerics
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "paper_exports"))
import paper1_numbers as p  # frame, cache and window helpers of the paper exporter

RUNS = {"C1024": "N = 16", "C1024n1000": "N = 1000"}
COLORS = {"C1024": "tab:orange", "C1024n1000": "tab:purple"}
SLICE_TIMES_MYR = (15.0, 30.0, 60.0)
MAP_TIME_MYR = 60.0
BOX_X_KPC, BOX_Z_KPC = 10.0, 20.0
FACE_ON_HALF_KPC = 3.0  # zoom of the face-on panel
X_RAY_MAP = ("emission_xray_atomdb_hhe_xz", "emission_xray_atomdb_metals_xz")  # 0.3-2 keV, Z = Z_sun (A09)
DENSITY_RANGE = (1e-4, 1e2)  # cm^-3
FIG_PATH = os.path.join(SCRIPT_DIR, "figures", "cgols", "comparisons", "model_c_n16_vs_n1000.png")
TITLES = os.environ.get("CGOLS_FIGURE_TITLES", "1") != "0"


def frame_at(tag, time_myr):
    """The in-situ frame closest to time_myr (1.25 Myr cadence)."""
    paths = sorted(p.frames_dir(tag).glob("frame_*.npz"))
    times = [float(np.load(path)["time_myr"]) for path in paths]
    return np.load(paths[int(np.argmin(np.abs(np.asarray(times) - time_myr)))])


def disc_mass(tag):
    times, recs, _ = p.records("convergence3d_v3_", tag)
    return np.asarray(times), np.array([np.sum(r["mass_disk"]) for r in recs])


def main():
    fig = plt.figure(figsize=(17, 13.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 5, height_ratios=(1.0, 1.0, 0.55))
    extent_xz = (-BOX_X_KPC / 2, BOX_X_KPC / 2, -BOX_Z_KPC / 2, BOX_Z_KPC / 2)
    extent_xy = (-BOX_X_KPC / 2, BOX_X_KPC / 2, -BOX_X_KPC / 2, BOX_X_KPC / 2)
    density_norm = LogNorm(*DENSITY_RANGE)
    xray_norm = None
    images = {}
    for row, (tag, label) in enumerate(RUNS.items()):
        for col, t in enumerate(SLICE_TIMES_MYR):
            frame = frame_at(tag, t)
            ax = fig.add_subplot(grid[row, col])
            images["n"] = ax.imshow(frame["n_xz"].T, origin="lower", extent=extent_xz, norm=density_norm,
                                    cmap="viridis", aspect="equal")  # fmt: skip
            ax.set_title(f"{label}: n, y = 0, {float(frame['time_myr']):.0f} Myr", fontsize=10)
            ax.set(xlabel="x [kpc]", ylabel="z [kpc]" if col == 0 else None)
        frame = frame_at(tag, MAP_TIME_MYR)
        ax = fig.add_subplot(grid[row, 3])
        images["nxy"] = ax.imshow(frame["n_xy"].T, origin="lower", extent=extent_xy, norm=density_norm,
                                  cmap="viridis", aspect="equal")  # fmt: skip
        ax.set(xlim=(-FACE_ON_HALF_KPC, FACE_ON_HALF_KPC), ylim=(-FACE_ON_HALF_KPC, FACE_ON_HALF_KPC),
               xlabel="x [kpc]", ylabel="y [kpc]")  # fmt: skip
        ax.set_title(f"{label}: mid-plane n, face-on, {float(frame['time_myr']):.0f} Myr", fontsize=10)
        ax = fig.add_subplot(grid[row, 4])
        sx = sum(np.asarray(frame[k], dtype=float) for k in X_RAY_MAP)
        if xray_norm is None:
            top = float(np.nanmax(sx))
            xray_norm = LogNorm(top * 1e-6, top)
        images["sx"] = ax.imshow(np.clip(sx, xray_norm.vmin, None).T, origin="lower", extent=extent_xz,
                                 norm=xray_norm, cmap="magma", aspect="equal")  # fmt: skip
        ax.set_title(f"{label}: soft X-ray $S_X$, {float(frame['time_myr']):.0f} Myr", fontsize=10)
        ax.set(xlabel="x [kpc]")
    fig.colorbar(images["n"], ax=fig.axes[0:3] + fig.axes[5:8], shrink=0.6, label=r"$n$ [cm$^{-3}$]",
                 location="bottom", pad=0.01)  # fmt: skip
    fig.colorbar(images["nxy"], ax=[fig.axes[3], fig.axes[8]], shrink=0.8, label=r"$n$ [cm$^{-3}$]",
                 location="bottom", pad=0.01)  # fmt: skip
    fig.colorbar(images["sx"], ax=[fig.axes[4], fig.axes[9]], shrink=0.8,
                 label=r"$S_X$ 0.3-2 keV [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]", location="bottom", pad=0.01)  # fmt: skip

    ax_m, ax_l, ax_d = (fig.add_subplot(grid[2, c]) for c in (slice(0, 2), slice(2, 4), 4))
    for tag, label in RUNS.items():
        t, mdot = p.plane_outflow(tag, 5.0)
        ax_m.plot(t, mdot, color=COLORS[tag], label=label)
        cache, _ = p.emission(tag)
        ax_l.semilogy(cache["per_snapshot__time_myr"], cache["per_snapshot__L_xray_cgols_xz"], color=COLORS[tag],
                      label=label)  # fmt: skip
        td, md = disc_mass(tag)
        ax_d.plot(td, md / 1e9, color=COLORS[tag], label=label)
    ax_m.set(xlabel="t [Myr]", ylabel=r"$\dot M$ through $|z| = 5$ kpc [M$_\odot$/yr]")
    ax_l.set(xlabel="t [Myr]", ylabel=r"$L_X$ (CGOLS I recipe) [erg/s]", ylim=(1e38, 5e42))
    ax_d.set(xlabel="t [Myr]", ylabel=r"disc gas, $|z| < 0.5$ kpc [$10^9$ M$_\odot$]")
    for ax in (ax_m, ax_l, ax_d):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    if TITLES:
        fig.suptitle("Model C at 1024$^3$: 16 against 1000 clusters (same total mass, energy and injection volume)",
                     fontsize=13)  # fmt: skip
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"wrote {FIG_PATH}")


if __name__ == "__main__":
    main()
