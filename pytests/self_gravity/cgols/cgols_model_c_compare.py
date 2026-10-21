"""Compare clustered (model C) and central (model A/B) CGOLS runs from their frames.

Reads the streamed frame .npz files directly (no GPU, no cgols import, like
cgols_fig12_recreation.py) and renders, into figures/cgols/comparisons/:

  model_c_outflow<suffix>.png   net mass outflow rate through |z| = 1, 5 and 8 kpc
                                and the cumulative outflow through |z| = 5 kpc,
                                every run on one axis;
  model_c_slices<suffix>.png    face-on n (z = 0), edge-on n and T (y = 0) at a
                                few times, one row per run, with the clusters
                                alive at that time drawn in;
  model_c_clusters<suffix>.png  the cluster schedule of every clustered run: the
                                face-on positions per epoch and the R / z
                                distributions;

and prints the outflow numbers used in the PR / notes (peak and high-state mean
rate through each height, cumulative mass).

Usage:
    python cgols_model_c_compare.py LABEL=SNAPSHOT_DIR [LABEL=SNAPSHOT_DIR ...]
    CGOLS_COMPARE_SUFFIX=_256 python cgols_model_c_compare.py \\
        C256ad=cgols_snapshots_C256ad A256=/path/to/cgols_snapshots_A256

A run is clustered if its snapshot dir holds the clusters.npz cgols.py writes.
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "figures", "cgols", "comparisons")
SUFFIX = os.environ.get("CGOLS_COMPARE_SUFFIX", "")
SLICE_TIMES_MYR = (10.0, 25.0, 50.0)

# The box of every CGOLS run: 10 x 10 x 20 kpc, centred on the disk.
L_X, L_Z = 10.0, 20.0
# Code time unit (1 kpc / 100 km/s) in Myr, for the cluster birth times.
CODE_TIME_MYR = 9.777922216807891


def load_frames(snapshot_dir):
    """Time-sorted frames of one run as a dict of stacked arrays."""
    files = sorted(glob.glob(os.path.join(snapshot_dir, "frame_*.npz")))
    if not files:
        raise SystemExit(f"no frames in {snapshot_dir}")
    frames = [np.load(f) for f in files]
    frames.sort(key=lambda fr: float(fr["time_myr"]))
    out = {"time_myr": np.array([float(fr["time_myr"]) for fr in frames])}
    for key in ("n_xz", "T_xz", "n_xy", "mdot_z"):
        out[key] = np.stack([fr[key] for fr in frames])
    return out


def load_clusters(snapshot_dir):
    """The run's cluster schedule (cgols.py clusters.npz), or None."""
    path = os.path.join(snapshot_dir, "clusters.npz")
    return dict(np.load(path)) if os.path.exists(path) else None


def clusters_at(clusters, time_myr):
    """(x, y, z) in kpc of the clusters alive at time_myr (cf. cluster_positions)."""
    births = clusters["birth_times"] * CODE_TIME_MYR
    num_born = np.sum(births <= time_myr, axis=0)
    alive = num_born > 0
    occupant = np.maximum(num_born - 1, 0)
    slots = np.arange(births.shape[1])
    pick = lambda name: clusters[name][occupant, slots]
    phase = pick("azimuths") + pick("angular_velocities") * (
        time_myr - births[occupant, slots]
    ) / CODE_TIME_MYR
    radius = pick("radii")
    xyz = np.stack(
        [radius * np.cos(phase), radius * np.sin(phase), pick("heights")], axis=-1
    )
    return xyz[alive]


def outflow_through(frames, height):
    """Net outward mass flux through the +/- height planes, Msun/yr."""
    nz = frames["mdot_z"].shape[1]
    z = (np.arange(nz) + 0.5) * (L_Z / nz) - L_Z / 2
    ip = int(np.argmin(np.abs(z - height)))
    im = int(np.argmin(np.abs(z + height)))
    return frames["mdot_z"][:, ip] - frames["mdot_z"][:, im]


def injected_mass_rate(time_myr):
    """The CGOLS Mdot schedule (Msun/yr): off, low, ramp, high, ramp, low."""
    return np.interp(time_myr, [0, 5, 5, 10, 40, 45], [0, 0, 1.5, 12, 12, 1.5])


def main(runs):
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {label: load_frames(path) for label, path in runs}
    clusters = {label: load_clusters(path) for label, path in runs}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ---- outflow rates ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    t_ref = np.linspace(0, 75, 751)
    for ax in axes[:3]:
        ax.plot(t_ref, injected_mass_rate(t_ref), color="0.6", lw=1, ls=":", label="injected")
    print(f"{'run':10s} {'H':>4s} {'peak':>7s} {'<15-40>':>8s} {'<45-75>':>8s} {'M_out(75)':>10s}")
    for k, (label, frames) in enumerate(data.items()):
        t = frames["time_myr"]
        style = dict(color=colors[k % len(colors)], lw=1.5,
                     ls="-" if clusters[label] is not None else "--")
        for ax, height in zip(axes[:3], (1.0, 5.0, 8.0)):
            rate = outflow_through(frames, height)
            ax.plot(t, rate, label=label, **style)
            high = (t >= 15) & (t <= 40)
            late = t >= 45
            cumulative = np.concatenate(
                [[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t) * 1e6)]
            )
            print(
                f"{label:10s} {height:4.0f} {np.max(rate):7.2f} "
                f"{np.mean(rate[high]) if high.any() else np.nan:8.2f} "
                f"{np.mean(rate[late]) if late.any() else np.nan:8.2f} "
                f"{cumulative[-1]:10.3e}"
            )
            if height == 5.0:
                axes[3].plot(t, cumulative, label=label, **style)
            ax.set_title(f"net outflow through |z| = {height:.0f} kpc")
    for ax in axes[:3]:
        ax.set_xlabel("t [Myr]")
        ax.set_ylabel(r"$\dot M_{\rm out}$ [M$_\odot$ yr$^{-1}$]")
        ax.axhline(0, color="0.7", lw=0.6)
    axes[3].set_xlabel("t [Myr]")
    axes[3].set_ylabel(r"$M_{\rm out}(<t)$ through |z| = 5 kpc [M$_\odot$]")
    axes[3].set_title("cumulative outflow")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"model_c_outflow{SUFFIX}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)

    # ---- slices ----
    n_rows = len(data)
    fig, axes = plt.subplots(
        n_rows, 3 * len(SLICE_TIMES_MYR), figsize=(4.0 * 3 * len(SLICE_TIMES_MYR), 4.4 * n_rows),
        squeeze=False,
    )
    n_norm = LogNorm(1e-4, 1e2)
    T_norm = LogNorm(1e3, 10**7.5)
    for row, (label, frames) in enumerate(data.items()):
        for col, target in enumerate(SLICE_TIMES_MYR):
            i = int(np.argmin(np.abs(frames["time_myr"] - target)))
            t_i = frames["time_myr"][i]
            ax_xy, ax_n, ax_T = axes[row, 3 * col : 3 * col + 3]
            ax_xy.imshow(frames["n_xy"][i].T, origin="lower", cmap="magma", norm=n_norm,
                         extent=[-L_X / 2, L_X / 2, -L_X / 2, L_X / 2])
            ax_xy.set_xlim(-2.5, 2.5)
            ax_xy.set_ylim(-2.5, 2.5)
            ax_n.imshow(frames["n_xz"][i].T, origin="lower", cmap="magma", norm=n_norm,
                        extent=[-L_X / 2, L_X / 2, -L_Z / 2, L_Z / 2])
            ax_T.imshow(frames["T_xz"][i].T, origin="lower", cmap="inferno", norm=T_norm,
                        extent=[-L_X / 2, L_X / 2, -L_Z / 2, L_Z / 2])
            if clusters[label] is not None:
                r_c = float(clusters[label]["cluster_radius"])
                for x, y, z in clusters_at(clusters[label], t_i):
                    ax_xy.add_patch(plt.Circle((x, y), r_c, fill=False, color="cyan", lw=0.6))
            ax_xy.set_title(f"{label} n face-on, {t_i:.1f} Myr", fontsize=9)
            ax_n.set_title(f"{label} n edge-on", fontsize=9)
            ax_T.set_title(f"{label} T edge-on", fontsize=9)
            for ax in (ax_xy, ax_n, ax_T):
                ax.tick_params(labelsize=7)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"model_c_slices{SUFFIX}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)

    # ---- cluster schedule ----
    clustered = [(label, cl) for label, cl in clusters.items() if cl is not None]
    if clustered:
        fig, axes = plt.subplots(len(clustered), 3, figsize=(15, 4.6 * len(clustered)), squeeze=False)
        for row, (label, cl) in enumerate(clustered):
            births = cl["birth_times"] * CODE_TIME_MYR
            r_c = float(cl["cluster_radius"])
            ax_xy, ax_R, ax_z = axes[row]
            for epoch in range(births.shape[0]):
                if births[epoch].min() >= 75:
                    continue
                x = cl["radii"][epoch] * np.cos(cl["azimuths"][epoch])
                y = cl["radii"][epoch] * np.sin(cl["azimuths"][epoch])
                color = colors[epoch % len(colors)]
                for xi, yi in zip(x, y):
                    ax_xy.add_patch(plt.Circle((xi, yi), r_c, fill=False, color=color, lw=0.8))
                ax_xy.plot(x, y, ".", color=color, label=f"born {births[epoch].min():.0f} Myr")
            ax_xy.set_aspect("equal")
            ax_xy.set_xlim(-1.2, 1.2)
            ax_xy.set_ylim(-1.2, 1.2)
            ax_xy.set_xlabel("x [kpc]")
            ax_xy.set_ylabel("y [kpc]")
            ax_xy.set_title(f"{label}: cluster positions at birth (R_c = {1e3 * r_c:.0f} pc)")
            ax_xy.legend(fontsize=7)
            live = births < 75
            ax_R.hist(cl["radii"][live] * 1e3, bins=12, color="0.4")
            ax_R.set_xlabel("R [pc]")
            ax_R.set_title("cylindrical radius")
            ax_z.hist(cl["heights"][live] * 1e3, bins=12, color="0.4")
            ax_z.set_xlabel("z [pc]")
            ax_z.set_title("height above the mid-plane")
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"model_c_clusters{SUFFIX}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    runs = []
    for arg in sys.argv[1:]:
        label, _, path = arg.partition("=")
        runs.append((label, path if os.path.isabs(path) else os.path.join(HERE, path)))
    main(runs)
