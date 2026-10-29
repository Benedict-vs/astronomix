"""Mass outflow rate through |z| = Z_CUT against time, per resolution, A vs B.

Reads ``mdot_z`` (the net vertical mass flux through each z plane, Msun/yr) from
the streamed frames of every run and sums the two planes at +-Z_CUT, counting
upward flux above the disk and downward flux below it as outflow. The injected
rate Mdot = beta SFR of the feedback schedule is drawn for reference. Pure numpy.

Usage:
    python cgols_outflow_rates.py
    CGOLS_OUTFLOW_ZCUT=2 python cgols_outflow_rates.py
    CGOLS_OUTFLOW_SERIES=A python cgols_outflow_rates.py   # adiabatic panel only
    CGOLS_OUTFLOW_SERIES=AF python cgols_outflow_rates.py  # A-series FD vs FV
"""

import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures", "cgols", "comparisons")
Z_CUT = float(os.environ.get("CGOLS_OUTFLOW_ZCUT", "5"))  # kpc
BOX_Z = 20.0  # kpc
RESOLUTIONS = (128, 256, 512, 1024)
COLORS = {128: "#c6dbef", 256: "#6baed6", 512: "#2171b5", 1024: "#08306b"}
SERIES = (("A", "no cooling"), ("B", "with cooling"), ("F", "no cooling, FV"))
SERIES_SEL = os.environ.get("CGOLS_OUTFLOW_SERIES", "AB")  # e.g. "A" for one panel
# Frame directory (below cgols_snapshots_) of each run; "F" = the A-series FV
# (VL2 + HLLC) ladder of the *_fv comparison figures.
FV_RUNS = {128: "diagnostics/A128fv", 256: "diagnostics/A256fv", 512: "A512fvf32", 1024: "A1024fv"}


def run_name(series, res):
    """Frame directory suffix of the run of ``series`` at ``res``."""
    return FV_RUNS[res] if series == "F" else f"{series}{res}"

# Injected mass rate of the schedule (cgols.build_cgols_wind_params).
SCHEDULE_T = np.array([0.0, 5.0, 5.0, 10.0, 40.0, 45.0, 75.0])  # Myr
SCHEDULE_MDOT = np.array([0.0, 0.0, 1.5, 12.0, 12.0, 1.5, 1.5])  # Msun/yr


def outflow_rate(run):
    """Times [Myr] and outflow rate [Msun/yr] through |z| = Z_CUT, or None."""
    frame_dir = os.path.join(SCRIPT_DIR, f"cgols_snapshots_{run}")
    if not os.path.isdir(frame_dir):
        return None
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".npz"))
    t, mdot = [], []
    for f in files:
        frame = np.load(os.path.join(frame_dir, f))
        flux = frame["mdot_z"]
        z = (np.arange(flux.size) + 0.5) * BOX_Z / flux.size - BOX_Z / 2
        up, down = np.argmin(np.abs(z - Z_CUT)), np.argmin(np.abs(z + Z_CUT))
        t.append(float(frame["time_myr"]))
        mdot.append(flux[up] - flux[down])
    return np.array(t), np.array(mdot)


def main():
    series_list = [(k, lab) for k, lab in SERIES if k in SERIES_SEL]
    fig, axes = plt.subplots(len(series_list), 1, figsize=(8, 2.8 * len(series_list) + 0.4),
                             sharex=True, squeeze=False, gridspec_kw=dict(hspace=0.08))
    axes = axes[:, 0]
    for ax, (series, label) in zip(axes, series_list):
        ax.plot(SCHEDULE_T, SCHEDULE_MDOT, color="0.55", lw=1.2, ls="--",
                label=r"injected $\dot M = \beta\,$SFR")
        for res in RESOLUTIONS:
            data = outflow_rate(run_name(series, res))
            if data is not None:
                ax.plot(*data, color=COLORS[res], lw=1.6, label=f"{res}")
        if len(series_list) > 1:
            ax.text(0.98, 0.5, f"{'A' if series == 'F' else series}-series ({label})",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=11)
        ax.set_ylabel(rf"$\dot M_\mathrm{{out}}(|z| = {Z_CUT:g}$ kpc) [$M_\odot$ yr$^{{-1}}$]",
                      fontsize=9.5)
        ax.set_ylim(bottom=-0.5)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        for t in (5, 10, 40, 45):
            ax.axvline(t, color="0.8", lw=0.6, ls=":", zorder=0)
    # The first panel carries every resolution; one column fits the empty
    # low-state corner of a single panel, a row fits the two-panel layout.
    handles = axes[0].get_legend_handles_labels()
    if "F" in SERIES_SEL and len(series_list) > 1:
        # the FV panel has no empty corner (its 40-50 Myr bump): legend on top
        axes[0].legend(*handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5,
                       ncol=5, frameon=False)
    elif len(series_list) > 1:
        axes[-1].legend(*handles, loc="upper right", fontsize=8.5, ncol=5, framealpha=1,
                        edgecolor="none", title="resolution", title_fontsize=8.5)
    else:
        axes[0].legend(*handles, loc="upper right", fontsize=8.5, frameon=False,
                       title="resolution", title_fontsize=8.5)
    axes[-1].set_xlim(0, 75)
    axes[-1].set_xlabel("$t$ [Myr]")

    os.makedirs(FIG_DIR, exist_ok=True)
    tag = "".join(k for k, _ in series_list)
    out = os.path.join(FIG_DIR, f"outflow_rate_z{Z_CUT:g}kpc_{tag}_resolutions.png")
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
