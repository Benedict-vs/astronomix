"""Floating-point precision of the CGOLS runs: float32 vs float64 (and vs FD).

Compares runs given as LABEL=FRAMES_DIR:DIAG_LOG (the first is the reference
of the slice ratio) and prints, per run, the net outflow through |z| = 1, 5,
8 kpc (peak, 15-40 Myr and 40-75 Myr means), the minimum density / pressure
of the late cavity (from the per-step diag log; the VL2 floors are
rho = 1e-4, P = 1e-5 in code units) and the fraction of late steps whose
minimum sits at a floor. The figure shows the 5 kpc outflow, min rho and
min P against time, and the edge-on density at ``--slice-myr`` with the log
ratio of every run to the first.

CPU only (numpy + matplotlib)::

    python cgols_fv_precision.py FD=cgols_snapshots_A512:cgols_logs/cgols_diag_A512.log \\
        "FV fp64=cgols_snapshots_A512fvf64:cgols_logs/cgols_diag_A512fvf64.log" \\
        --out figures/cgols/comparisons/fv_precision_A512.png
"""

# general
import argparse
import os
import re

# numerics / plotting
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

BOX_Z = 20.0  # kpc
#: Myr per code time unit (kpc / (100 km/s)).
MYR_PER_CODE = 9.7779
#: VL2 hard floors (code units), CGOLS_SOLVER=fv.
FLOOR_RHO, FLOOR_P = 1e-4, 1e-5
DIAG = re.compile(r"t=([-+.\deE]+)\s+min_rho=([-+.\deE]+)\s+min_P=([-+.\deE]+)")


def load_frames(frame_dir):
    """Times [Myr], mdot_z (frames x z) and the frames themselves, sorted by time."""
    files = sorted(f for f in os.listdir(frame_dir) if f.startswith("frame_") and f.endswith(".npz"))
    frames = [np.load(os.path.join(frame_dir, f)) for f in files]
    t = np.array([float(fr["time_myr"]) for fr in frames])
    order = np.argsort(t)
    return t[order], np.stack([frames[i]["mdot_z"] for i in order]), [frames[i] for i in order]


def load_diag(log):
    """Times [Myr], min rho, min P of every logged step (restart overlaps deduplicated by t)."""
    rows = {}
    with open(log) as f:
        for line in f:
            m = DIAG.search(line)
            if m:
                rows[float(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    t = np.array(sorted(rows))
    values = np.array([rows[k] for k in t])
    return t * MYR_PER_CODE, values[:, 0], values[:, 1]


def net_outflow(t, mdot, height):
    """Net outflow [Msun/yr] through |z| = height."""
    z = (np.arange(mdot.shape[1]) + 0.5) * BOX_Z / mdot.shape[1] - BOX_Z / 2
    return mdot[:, np.argmin(np.abs(z - height))] - mdot[:, np.argmin(np.abs(z + height))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="LABEL=FRAMES_DIR:DIAG_LOG")
    parser.add_argument("--out", required=True)
    parser.add_argument("--slice-myr", type=float, default=60.0)
    arguments = parser.parse_args()

    runs = {}
    for spec in arguments.runs:
        label, paths = spec.split("=", 1)
        frame_dir, log = paths.split(":", 1)
        runs[label] = (load_frames(frame_dir), load_diag(log))

    print(f"{'run':>10} {'H':>3} {'peak':>6} {'15-40':>6} {'40-75':>6}   late cavity (45-75 Myr)")
    for label, ((t, mdot, _), (td, rho, p)) in runs.items():
        late = td >= 45
        floor_rho = np.mean(rho[late] <= FLOOR_RHO * 1.0001)
        floor_p = np.mean(p[late] <= FLOOR_P * 1.0001)
        for height in (1.0, 5.0, 8.0):
            rate = net_outflow(t, mdot, height)
            high, low = (t >= 15) & (t <= 40), (t >= 40) & (t <= 75)
            extra = (
                f"   min rho {rho[late].min():.2e}, min P {p[late].min():.2e}; "
                f"steps at/below the FV floor: rho {floor_rho:.0%}, P {floor_p:.0%} (of {late.sum()})"
                if height == 1.0
                else ""
            )
            print(f"{label:>10} {height:3.0f} {rate.max():6.2f} {rate[high].mean():6.2f} {rate[low].mean():6.2f}{extra}")

    n = len(runs)
    figure = plt.figure(figsize=(4.2 * max(n, 3), 9.5))
    grid = figure.add_gridspec(2, max(n, 3), height_ratios=[1, 1.25])
    colors = [f"C{k}" for k in range(n)]

    axes = [figure.add_subplot(grid[0, k]) for k in range(3)]
    for color, (label, ((t, mdot, _), (td, rho, p))) in zip(colors, runs.items()):
        axes[0].plot(t, net_outflow(t, mdot, 5.0), color=color, label=label)
        stride = max(len(td) // 4000, 1)
        axes[1].plot(td[::stride], rho[::stride], color=color, lw=0.8, label=label)
        axes[2].plot(td[::stride], p[::stride], color=color, lw=0.8, label=label)
    axes[0].set(xlabel="t [Myr]", ylabel=r"net outflow |z| = 5 kpc [$M_\odot$/yr]", xlim=(0, 75))
    axes[1].axhline(FLOOR_RHO, color="k", ls=":", lw=0.8, label="FV floor")
    axes[2].axhline(FLOOR_P, color="k", ls=":", lw=0.8, label="FV floor")
    axes[1].set(xlabel="t [Myr]", ylabel=r"min $\rho$ [code]", yscale="log", xlim=(0, 75))
    axes[2].set(xlabel="t [Myr]", ylabel="min P [code]", yscale="log", xlim=(0, 75))
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

    reference = None
    for k, (label, ((t, _, frames), _)) in enumerate(runs.items()):
        frame = frames[int(np.argmin(np.abs(t - arguments.slice_myr)))]
        density = np.asarray(frame["n_xz"], np.float64)
        axis = figure.add_subplot(grid[1, k])
        extent = [-5, 5, -10, 10]
        if reference is None:
            reference = density
            image = axis.imshow(density.T, origin="lower", extent=extent, norm=LogNorm(1e-4, 1e2), cmap="viridis")
            figure.colorbar(image, ax=axis, shrink=0.8, label=r"n [cm$^{-3}$]")
            axis.set_title(f"{label}: n at {float(frame['time_myr']):.1f} Myr", fontsize=9)
        else:
            image = axis.imshow(
                np.log10(density / reference).T, origin="lower", extent=extent, vmin=-1, vmax=1, cmap="RdBu_r"
            )
            figure.colorbar(image, ax=axis, shrink=0.8, label=f"log10 n / n({next(iter(runs))})")
            axis.set_title(f"{label} vs {next(iter(runs))}", fontsize=9)
        axis.set(xlabel="x [kpc]", ylabel="z [kpc]")

    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(arguments.out)), exist_ok=True)
    figure.savefig(arguments.out, dpi=150)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
