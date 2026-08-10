"""Recreate Figs 1 & 2 of the CGOLS cooling paper (arXiv:1803.01005) from our runs.

The paper shows, at t = 25 Myr, x-z slices of density, radial velocity and
temperature for the adiabatic Model A (their Fig. 1) and the radiative Model B
(their Fig. 2), with the dOmega = 60 deg bicone marked on the density panel and
(Fig. 2 only) the analytically-predicted cooling radius r_cool = 2.77 kpc as a
white dashed circle on the temperature panel.

This script reads the streamed frame .npz files directly (no GPU, no cgols
import), picks the frame nearest the target time per series, and renders
whichever of the three panels the frames provide: n_xz and T_xz always exist;
vr_xz exists only for runs made after the snapshot callable learned to store
it. Missing panels are rendered as an annotated placeholder rather than
silently dropped, so the layout always matches the paper.

Usage:
    python cgols_fig12_recreation.py                # A from prod_tmax5e9, B from _B
    CGOLS_FIG12_A=cgols_snapshots_Avr \
    CGOLS_FIG12_B=cgols_snapshots_Bvr \
        python cgols_fig12_recreation.py            # use the vr-bearing reruns
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

HERE = os.path.dirname(os.path.abspath(__file__))

A_DIR = os.environ.get("CGOLS_FIG12_A", "cgols_snapshots_prod_tmax5e9")
B_DIR = os.environ.get("CGOLS_FIG12_B", "cgols_snapshots_B")
TARGET_MYR = float(os.environ.get("CGOLS_FIG12_TIME_MYR", "25.0"))

# Paper panel specs (Figs 1 & 2 of 1803.01005).
X_HALF, Z_HALF = 5.0, 10.0          # kpc, full box
N_RANGE = (1e-4, 1e3)               # cm^-3, log
T_RANGE = (1e3, 10 ** 7.5)          # K, log
VR_RANGE = (0.0, 1200.0)            # km/s, linear
CONE_HALF_ANGLE_DEG = 30.0          # dOmega = 60 deg opening angle
R_COOL_KPC = 2.77                   # their Equation 3, high state


def nearest_frame(dirname, t_myr):
    files = sorted(glob.glob(os.path.join(HERE, dirname, "frame_*.npz")))
    if not files:
        raise SystemExit(f"no frames in {dirname}")
    frames = [np.load(f) for f in files]
    times = np.array([float(fr["time_myr"]) for fr in frames])
    i = int(np.argmin(np.abs(times - t_myr)))
    return frames[i], times[i]


def draw_cone(ax):
    t = np.tan(np.radians(CONE_HALF_ANGLE_DEG))
    for sz in (+1, -1):
        z = np.array([0.0, sz * Z_HALF])
        for sx in (+1, -1):
            ax.plot(sx * t * np.abs(z), z, color="w", lw=0.8, ls="--", alpha=0.9)


def scalebar(ax):
    ax.plot([X_HALF - 2.2, X_HALF - 1.2], [Z_HALF - 1.2] * 2, color="w", lw=1.5)
    ax.text(X_HALF - 1.7, Z_HALF - 2.0, "1 kpc", color="w", ha="center", fontsize=8)


def panel(ax, img, norm, cmap, label, t_myr, cone=False, circle=False):
    extent = [-X_HALF, X_HALF, -Z_HALF, Z_HALF]
    im = None
    if img is not None:
        im = ax.imshow(
            img.T, origin="lower", extent=extent, norm=norm, cmap=cmap,
            interpolation="nearest", aspect="equal",
        )
    else:
        ax.set_facecolor("0.15")
        ax.text(0, 0, "v_r not stored\nin these frames\n(rerun pending)",
                color="w", ha="center", va="center", fontsize=9)
    if cone:
        draw_cone(ax)
    if circle:
        th = np.linspace(0, 2 * np.pi, 256)
        ax.plot(R_COOL_KPC * np.cos(th), R_COOL_KPC * np.sin(th),
                color="w", lw=1.0, ls="--")
    ax.text(-X_HALF + 0.5, Z_HALF - 1.6, f"{t_myr:.0f} Myr", color="w", fontsize=9)
    scalebar(ax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-X_HALF, X_HALF); ax.set_ylim(-Z_HALF, Z_HALF)
    return im, label


def render(figname, dirname, title):
    fr, t_myr = nearest_frame(dirname, TARGET_MYR)
    n = np.asarray(fr["n_xz"], dtype=np.float64)
    T = np.asarray(fr["T_xz"], dtype=np.float64)
    vr = np.asarray(fr["vr_xz"], dtype=np.float64) if "vr_xz" in fr.files else None

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 7.5))
    specs = [
        (np.clip(n, *N_RANGE), LogNorm(*N_RANGE), "viridis",
         r"$n$ [cm$^{-3}$]", dict(cone=True)),
        (None if vr is None else np.clip(vr, *VR_RANGE), Normalize(*VR_RANGE),
         "plasma", r"$v_r$ [km s$^{-1}$]", {}),
        (np.clip(T, *T_RANGE), LogNorm(*T_RANGE), "inferno",
         r"$T$ [K]", dict(circle="fig2" in figname)),
    ]
    for ax, (img, norm, cmap, label, extra) in zip(axes, specs):
        im, label = panel(ax, img, norm, cmap, label, t_myr, **extra)
        if im is not None:
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            cb.set_label(label)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(HERE, "figures", "cgols", figname)
    fig.savefig(out, dpi=250)
    plt.close(fig)
    print(f"wrote {out}  (frame at {t_myr:.2f} Myr from {dirname})")


if __name__ == "__main__":
    render(
        "cgols_fig1_recreation.png", A_DIR,
        "cf. Schneider & Robertson (1803.01005) Fig. 1 — Model A (adiabatic), "
        f"512$^2\\times$1024, t = {TARGET_MYR:.0f} Myr",
    )
    render(
        "cgols_fig2_recreation.png", B_DIR,
        "cf. Schneider & Robertson (1803.01005) Fig. 2 — Model B (radiative), "
        f"512$^2\\times$1024, t = {TARGET_MYR:.0f} Myr",
    )
