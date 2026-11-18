"""
Planar outflow per cylindrical radius R for models A and C (asgard-papers issue #3).

Reads the cached 3D measurements of cgols_convergence_3d.py
(data/studies/convergence3d_v3<tag>.npz, field ``mdot_net_radius``: net mass
flux through both planes |z| = h, split into R < 0.5, 0.5-1, 1-2 and > 2 kpc)
and shows where the outflow leaves the disc region: in the wind core above the
starburst, or further out. Means over the burst (15, 40] Myr and the low state
(50, 75] Myr use every cached snapshot in the window, 15 Myr excluded (the
convention of the model A section of the status page).

    python cgols_outflow_per_radius.py            # figure + table
    python cgols_outflow_per_radius.py --export   # also write the paper-1 manifest entry

No checkpoints are read; CPU only, seconds.
"""

# general
import argparse
import os
import sys

# numerics
import numpy as np

# plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.join(SCRIPT_DIR, "data", "studies")
FIGURE = os.path.join(SCRIPT_DIR, "figures", "cgols", "comparisons", "outflow_per_R_A_C.png")

#: Must match cgols_convergence_3d.py (HEIGHTS_KPC, RADIUS_EDGES_KPC).
HEIGHTS_KPC = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
RADIUS_LABELS = ("R < 0.5", "0.5-1", "1-2", "R > 2 kpc")
WINDOWS_MYR = {"burst, 15-40 Myr": (15.0, 40.0), "low state, 50-75 Myr": (50.0, 75.0)}
SHOWN_HEIGHTS_KPC = (1.0, 5.0)

#: (label, cache tag, colour); the top of each ladder is drawn darkest.
RUNS = (
    ("A-256", "_A256", "#c6dbef"),
    ("A-512", "_A512", "#9ecae1"),
    ("A-1024", "_A1024", "#4292c6"),
    ("A-2048", "_A2048", "#08519c"),
    ("C-256", "_C256", "#fdd0a2"),
    ("C-512", "_C512", "#fd8d3c"),
    ("C-1024", "_C1024", "#d94801"),
    ("C-1024, N = 1000", "_C1024n1000", "#7f2704"),
)


def window_mean(tag, height_kpc, window):
    """Mean of mdot_net_radius [Msun/yr] at |z| = height over (t0, t1] Myr; (R bins,), n snapshots."""
    records = np.load(os.path.join(STUDIES, f"convergence3d_v3{tag}.npz"), allow_pickle=True)["records"].item()
    h = int(np.flatnonzero(np.isclose(HEIGHTS_KPC, height_kpc))[0])
    t0, t1 = window
    rows = [
        np.asarray(record["mdot_net_radius"])[h]
        for record in records.values()
        if t0 < float(record["time_myr"]) <= t1 + 1e-6
    ]
    return np.mean(rows, axis=0), len(rows)


def main():
    """Table and figure; optionally the manifest entry."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", action="store_true", help="write the figure into paper_exports/paper1")
    args = parser.parse_args()

    runs = [run for run in RUNS if os.path.exists(os.path.join(STUDIES, f"convergence3d_v3{run[1]}.npz"))]
    figure, axes = plt.subplots(
        len(WINDOWS_MYR), len(SHOWN_HEIGHTS_KPC), figsize=(11, 7.2), sharex=True, constrained_layout=True
    )
    width = 0.8 / len(runs)
    for row, (window_label, window) in enumerate(WINDOWS_MYR.items()):
        for col, height in enumerate(SHOWN_HEIGHTS_KPC):
            axis = axes[row, col]
            print(f"\n{window_label}, |z| = {height:g} kpc: net Mdot [Msun/yr] per R bin, (share of the total)")
            print(f"  {'run':18s}" + "".join(f"{label:>16s}" for label in RADIUS_LABELS) + f"{'total':>9s}  n")
            for i, (label, tag, colour) in enumerate(runs):
                mean, count = window_mean(tag, height, window)
                total = mean.sum()
                print(
                    f"  {label:18s}"
                    + "".join(f"{value:8.2f} ({value / total:4.0%})" for value in mean)
                    + f"{total:9.2f}  {count}"
                )
                axis.bar(np.arange(4) + (i - (len(runs) - 1) / 2) * width, mean, width, color=colour, label=label)
            axis.axhline(0.0, color="0.3", linewidth=0.8)
            axis.set_title(f"|z| = {height:g} kpc, {window_label}", fontsize=10)
            axis.set_xticks(np.arange(4), RADIUS_LABELS)
            if col == 0:
                axis.set_ylabel(r"net $\dot M$ through both planes [M$_\odot$ yr$^{-1}$]")
    axes[0, -1].legend(fontsize=8, ncol=2)
    figure.suptitle(
        "Where the outflow crosses |z| = 1 and 5 kpc: net mass flux per cylindrical radius (A: blue, C: orange)",
        fontsize=11,
    )
    os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
    figure.savefig(FIGURE, dpi=200)
    print(f"\nwrote {FIGURE}")

    if args.export:
        sys.path.insert(0, SCRIPT_DIR)
        from paper_exports.export import export_figure

        export_figure(
            "paper1",
            "outflow_per_R_A_C",
            FIGURE,
            caption_hint=(
                "Net mass flux through |z| = 1 and 5 kpc per cylindrical radius bin (R < 0.5, 0.5-1, 1-2, "
                "> 2 kpc), means over (15, 40] and (50, 75] Myr, models A (256-2048) and C (256-1024, "
                "N = 16 and 1000)"
            ),
            runs=[label for label, _, _ in runs],
        )


if __name__ == "__main__":
    main()
