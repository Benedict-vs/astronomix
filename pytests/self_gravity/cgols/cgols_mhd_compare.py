"""
MHD vs hydro comparison for the CGOLS runs (CGOLS_MHD=1).

Reads the streamed frames of an MHD run (CGOLS_RUN_TAG, e.g. _A128mhd) and of
the matching hydro run (CGOLS_REFERENCE_SNAPSHOTS, default
cgols_snapshots_A<CGOLS_DIM> next to this script) and writes, into the MHD
run's figure folder figures/cgols/<tag>/:

  * mhd_outflow_vs_hydro.png - net outflow through |z| = 5 and 8 kpc vs time;
  * mhd_slices.png           - edge-on n (hydro / MHD), T, |B| with the
                               in-plane field lines, and plasma beta at
                               25 / 50 / 75 Myr;
  * mhd_field_evolution.png  - magnetic energy, max |B|, the vertical field
                               fraction in the wind and the div B error vs time.

and prints the summary numbers (peak and phase-averaged outflow, their change
in per cent, the field geometry in the wind). CPU only, no solver state needed:

    JAX_PLATFORMS=cpu CGOLS_DIM=128 CGOLS_RUN_TAG=_A128mhd \
        CGOLS_REFERENCE_SNAPSHOTS=/path/to/cgols_snapshots_A128 \
        python cgols_mhd_compare.py
"""

# general
import os

# The comparison only reads frames; keep cgols.py from asking for more than one
# device (see cgols_analyse.py).
os.environ.setdefault("CGOLS_SHARD_SPLIT", "(1, 1, 1, 1)")

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# cgols driver
from cgols import (
    RESOLUTION,
    RUN_TAG,
    SNAPSHOTS_DIR,
    _here,
    _run_fig,
    _snapshot_grid,
    build_config,
    load_snapshots,
)

REFERENCE_DIR = os.environ.get(
    "CGOLS_REFERENCE_SNAPSHOTS", _here(f"cgols_snapshots_A{RESOLUTION}")
)
SLICE_TIMES_MYR = (25.0, 50.0, 75.0)
OUTFLOW_HEIGHTS_KPC = (5.0, 8.0)
STREAMLINE_MIN_UG = 0.01

#: Phases of the feedback schedule over which the outflow is averaged (Myr):
#: the high state including its ramps, and the late low state.
PHASES_MYR = {"10-45 Myr (high)": (10.0, 45.0), "45-75 Myr (low)": (45.0, 75.0)}


def percent_change(new, old):
    """Relative change in per cent, NaN when the reference is zero."""
    return 100.0 * (new / old - 1.0) if old != 0 else float("nan")


def outflow_through(data, z, height):
    """Net outward mass flux through the +/- height planes vs time (Msun/yr)."""
    upper = int(np.argmin(np.abs(z - height)))
    lower = int(np.argmin(np.abs(z + height)))
    return data["mdot_z"][:, upper] - data["mdot_z"][:, lower]


def phase_mean(times, values, window):
    """Time average of ``values`` over the frames inside ``window`` (Myr)."""
    inside = (times >= window[0]) & (times <= window[1])
    return float(np.trapezoid(values[inside], times[inside]) / (times[inside][-1] - times[inside][0]))


def cumulative(times, values):
    """Time-integrated outflow in 1e6 Msun (frames in Myr, rates in Msun/yr)."""
    return float(np.trapezoid(values, times))


def nearest_frame(times, target):
    """Index of the frame closest to ``target`` Myr."""
    return int(np.argmin(np.abs(times - target)))


def vertical_field_fraction(data, x, z, frame):
    """B^2-weighted vertical field fraction Bz^2 / B^2 in the wind and disk.

    Wind: the bicone |x| < |z|, 0.5 < |z| < 8 kpc of the y = 0 slice; disk:
    |z| < 0.2 kpc, |x| < 4 kpc. 0 for a purely toroidal field (B_y in this
    slice), 1 for a purely vertical one, 1/3 for an isotropic tangle.
    """
    B_x = data["Bx_xz_uG"][frame]
    B_y = data["By_xz_uG"][frame]
    B_z = data["Bz_xz_uG"][frame]
    X, Z = np.meshgrid(x, z, indexing="ij")
    total = B_x**2 + B_y**2 + B_z**2
    wind = (np.abs(X) < np.abs(Z)) & (np.abs(Z) > 0.5) & (np.abs(Z) < 8.0)
    disk = (np.abs(Z) < 0.2) & (np.abs(X) < 4.0)
    fractions = []
    for region in (wind, disk):
        weight = total[region].sum()
        fractions.append(float((B_z**2)[region].sum() / weight) if weight > 0 else np.nan)
    return fractions


def main():
    """Load both runs, write the three figures and print the summary."""
    config, _ = build_config()
    x, _, z, extent_xz, _ = _snapshot_grid(config)
    mhd = load_snapshots(SNAPSHOTS_DIR)
    hydro = load_snapshots(REFERENCE_DIR)
    if mhd is None or hydro is None:
        raise SystemExit(f"missing frames: {SNAPSHOTS_DIR} or {REFERENCE_DIR}")
    t_mhd, t_hydro = mhd["time_myr"], hydro["time_myr"]
    print(f"MHD {RUN_TAG}: {len(t_mhd)} frames to {t_mhd[-1]:.1f} Myr; hydro {REFERENCE_DIR}: "
          f"{len(t_hydro)} frames to {t_hydro[-1]:.1f} Myr")

    # -------------------------------------------------------------
    # ================ ↓ Outflow rates vs hydro ↓ =================
    # -------------------------------------------------------------

    fig, axes = plt.subplots(1, len(OUTFLOW_HEIGHTS_KPC), figsize=(12, 4.5), sharey=True)
    for ax, height in zip(axes, OUTFLOW_HEIGHTS_KPC):
        rate_mhd = outflow_through(mhd, z, height)
        rate_hydro = outflow_through(hydro, z, height)
        ax.plot(t_hydro, rate_hydro, "k-", lw=1.5, label=f"hydro A{RESOLUTION}")
        ax.plot(t_mhd, rate_mhd, "C3-", lw=1.5, label=f"MHD {RUN_TAG.lstrip('_')}")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("t [Myr]")
        ax.set_title(f"net outflow through |z| = {height:.0f} kpc")
        print(f"|z| = {height:.0f} kpc:")
        print(f"  peak:       hydro {rate_hydro.max():7.2f}  MHD {rate_mhd.max():7.2f} Msun/yr "
              f"({percent_change(rate_mhd.max(), rate_hydro.max()):+.1f}%)")
        for name, window in PHASES_MYR.items():
            if t_mhd[-1] < window[1] - 1:
                continue
            mean_hydro = phase_mean(t_hydro, rate_hydro, window)
            mean_mhd = phase_mean(t_mhd, rate_mhd, window)
            print(f"  {name}: hydro {mean_hydro:7.2f}  MHD {mean_mhd:7.2f} Msun/yr "
                  f"({percent_change(mean_mhd, mean_hydro):+.1f}%)")
        if t_mhd[-1] >= t_hydro[-1] - 1:
            total_hydro = cumulative(t_hydro, rate_hydro)
            total_mhd = cumulative(t_mhd, rate_mhd)
            print(f"  cumulative: hydro {total_hydro:7.1f}  MHD {total_mhd:7.1f} x1e6 Msun "
                  f"({percent_change(total_mhd, total_hydro):+.1f}%)")
    axes[0].set_ylabel(r"$\dot M_{\rm out}$ [$M_\odot$ yr$^{-1}$]")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(_run_fig("mhd_outflow_vs_hydro.png"), dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------
    # ===================== ↓ Edge-on slices ↓ ====================
    # -------------------------------------------------------------

    columns = ("n hydro", "n MHD", "T MHD", "|B| MHD", "beta MHD")
    fig, axes = plt.subplots(
        len(SLICE_TIMES_MYR), len(columns), figsize=(3.0 * len(columns), 4.6 * len(SLICE_TIMES_MYR))
    )
    axes = np.atleast_2d(axes)
    for row, target in enumerate(SLICE_TIMES_MYR):
        i_mhd = nearest_frame(t_mhd, target)
        i_hydro = nearest_frame(t_hydro, target)
        panels = (
            (hydro["n_xz"][i_hydro], "viridis", LogNorm(1e-4, 1e2), r"n [cm$^{-3}$]"),
            (mhd["n_xz"][i_mhd], "viridis", LogNorm(1e-4, 1e2), r"n [cm$^{-3}$]"),
            (mhd["T_xz"][i_mhd], "inferno", LogNorm(1e4, 1e8), "T [K]"),
            (mhd["B_xz_uG"][i_mhd], "magma", LogNorm(1e-2, 1e2), r"|B| [$\mu$G]"),
            (mhd["beta_xz"][i_mhd], "RdBu", LogNorm(1e-2, 1e4), r"$\beta$"),
        )
        for column, (field, cmap, norm, label) in enumerate(panels):
            ax = axes[row, column]
            image = ax.imshow(
                np.maximum(field.T, norm.vmin * 1e-3),
                origin="lower", extent=extent_xz, cmap=cmap, norm=norm,
            )
            if column == 3:
                # In-plane field lines (B_x, B_z) of the y = 0 slice, drawn
                # only where the field is above STREAMLINE_MIN_UG (the
                # direction of a round-off field is noise).
                weak = mhd["B_xz_uG"][i_mhd].T < STREAMLINE_MIN_UG
                ax.streamplot(
                    x, z,
                    np.ma.masked_where(weak, mhd["Bx_xz_uG"][i_mhd].T),
                    np.ma.masked_where(weak, mhd["Bz_xz_uG"][i_mhd].T),
                    color="w", density=1.2, linewidth=0.5, arrowsize=0.5,
                )
                ax.set_xlim(extent_xz[0], extent_xz[1])
                ax.set_ylim(extent_xz[2], extent_xz[3])
            ax.set_title(f"{columns[column]}, {t_mhd[i_mhd]:.0f} Myr", fontsize=9)
            ax.set_xlabel("x [kpc]")
            if column == 0:
                ax.set_ylabel("z [kpc]")
            plt.colorbar(image, ax=ax, label=label, fraction=0.08)
    plt.tight_layout()
    plt.savefig(_run_fig("mhd_slices.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------
    # ================= ↓ Field evolution vs time ↓ ===============
    # -------------------------------------------------------------

    fractions = np.array([vertical_field_fraction(mhd, x, z, i) for i in range(len(t_mhd))])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    axes[0].semilogy(t_mhd, mhd["magnetic_energy"], "C0-")
    axes[0].set_ylabel(r"$E_B$ [code]")
    axes[0].set_title("magnetic energy (box)")
    axes[1].semilogy(t_mhd, mhd["max_B_uG"], "C1-")
    axes[1].set_ylabel(r"max |B| [$\mu$G]")
    axes[1].set_title("peak field")
    axes[2].plot(t_mhd, fractions[:, 0], "C3-", label="wind bicone")
    axes[2].plot(t_mhd, fractions[:, 1], "C2--", label="disk |z|<0.2 kpc")
    axes[2].axhline(1 / 3, color="0.6", lw=0.8, ls=":")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel(r"$\sum B_z^2 / \sum B^2$ (y = 0 slice)")
    axes[2].set_title("vertical field fraction")
    axes[2].legend()
    axes[3].semilogy(t_mhd, mhd["divB_rel_interior"], "C4-", label="deep interior")
    axes[3].semilogy(t_mhd, mhd["divB_rel_all"], "C5--", label="incl. boundary cells")
    axes[3].set_ylabel(r"max |div B| dx / max |B|")
    axes[3].set_title("divergence error")
    axes[3].legend()
    for ax in axes:
        ax.set_xlabel("t [Myr]")
    plt.tight_layout()
    plt.savefig(_run_fig("mhd_field_evolution.png"), dpi=150)
    plt.close(fig)

    for target in (0.0,) + SLICE_TIMES_MYR:
        i = nearest_frame(t_mhd, target)
        print(f"t = {t_mhd[i]:5.1f} Myr: E_B = {mhd['magnetic_energy'][i]:.3e}, "
              f"max|B| = {mhd['max_B_uG'][i]:.1f} uG, vertical fraction wind/disk = "
              f"{fractions[i, 0]:.2f}/{fractions[i, 1]:.2f}, divB interior/all = "
              f"{mhd['divB_rel_interior'][i]:.1e}/{mhd['divB_rel_all'][i]:.1e}")
    print(f"Wrote figures to {os.path.dirname(_run_fig('mhd_slices.png'))}")


if __name__ == "__main__":
    main()
