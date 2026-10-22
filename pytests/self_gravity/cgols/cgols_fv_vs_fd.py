"""
Finite-volume (VL2) vs finite-difference (WENO5) comparison of CGOLS runs.

Reads the streamed frames (``cgols_snapshots<tag>/frame_*.npz``) and the
per-step diagnostics logs (``cgols_diag<tag>.log``) of two runs at the same
resolution and prints the numbers of the comparison: the outflow rate through
|z| = 5 and 8 kpc (peak and 40-75 Myr mean), the step count, the mass budget,
and the slice differences at 25 / 50 / 75 Myr. It also renders the side-by-side
figure (outflow rates + edge-on density and temperature slices), in the layout
of ``figures/cgols/comparisons/weno_js_vs_z_A256.png``.

CPU only (numpy + matplotlib, no JAX)::

    python cgols_fv_vs_fd.py --fd-frames <dir> --fd-log <log> \\
        --fv-frames <dir> --fv-log <log> --label A256 --out <png>
"""

# general
import argparse
import glob
import os
import re

# numerics / plotting
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# -------------------------------------------------------------
# ======================= ↓ Loading ↓ =========================
# -------------------------------------------------------------

#: Box half-height in kpc (the box spans z in [-10, 10] kpc).
Z_HALF_KPC = 10.0

#: The outflow is averaged over this window (the low state after the burst).
LATE_WINDOW_MYR = (40.0, 75.0)

#: Slice comparison times.
SLICE_TIMES_MYR = (25.0, 50.0, 75.0)

#: Code time unit in Myr (kpc / (100 km/s)), to convert the diag-log times.
CODE_TIME_MYR = 9.7779222


def load_frames(frames_dir):
    """Load the frames of a run, sorted by time, into stacked arrays."""
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.npz")))
    if not files:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    frames = [np.load(path) for path in files]
    frames.sort(key=lambda frame: float(frame["time_myr"]))
    return {
        key: np.stack([frame[key] for frame in frames])
        for key in ("time_myr", "n_xz", "T_xz", "mdot_z")
    }


def load_diag_log(log_path):
    """The step times and the total masses of a diagnostics log."""
    times, masses = [], []
    pattern = re.compile(r"t=([-+0-9.eE]+).*mass=([-+0-9.eE]+)")
    with open(log_path) as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                times.append(float(match.group(1)))
                masses.append(float(match.group(2)))
    return np.array(times), np.array(masses)


def outflow_through(frames, height_kpc):
    """Total outward mass flux through the +/- height planes, Msun/yr."""
    num_z = frames["mdot_z"].shape[1]
    z = (np.arange(num_z) + 0.5) * (2.0 * Z_HALF_KPC / num_z) - Z_HALF_KPC
    upper = int(np.argmin(np.abs(z - height_kpc)))
    lower = int(np.argmin(np.abs(z + height_kpc)))
    return frames["mdot_z"][:, upper] - frames["mdot_z"][:, lower]


def frame_at(frames, time_myr):
    """Index of the frame closest to the given time."""
    return int(np.argmin(np.abs(frames["time_myr"] - time_myr)))


# -------------------------------------------------------------
# ======================= ↑ Loading ↑ =========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ Metrics ↓ =========================
# -------------------------------------------------------------


def outflow_metrics(frames):
    """Peak and late-time mean outflow through |z| = 5 and 8 kpc."""
    late = (frames["time_myr"] >= LATE_WINDOW_MYR[0]) & (frames["time_myr"] <= LATE_WINDOW_MYR[1])
    metrics = {}
    for height in (5.0, 8.0):
        rate = outflow_through(frames, height)
        metrics[f"peak{height:.0f}"] = float(np.max(rate))
        metrics[f"tpeak{height:.0f}"] = float(frames["time_myr"][np.argmax(rate)])
        metrics[f"late{height:.0f}"] = float(np.mean(rate[late]))
    return metrics


def _coarsen(field, shape):
    """Block-average a 2D field down to ``shape`` (integer factors only)."""
    factor_x = field.shape[0] // shape[0]
    factor_z = field.shape[1] // shape[1]
    return field.reshape(shape[0], factor_x, shape[1], factor_z).mean(axis=(1, 3))


def slice_metrics(frames_fd, frames_fv, time_myr):
    """
    Median and mean absolute log10 difference of the edge-on density and
    temperature slices, and the vertical extent of the hot (T > 10^6.5 K) gas
    above the disk along the minor axis, at one time.
    """
    index_fd = frame_at(frames_fd, time_myr)
    index_fv = frame_at(frames_fv, time_myr)
    result = {}
    for key in ("n_xz", "T_xz"):
        # Runs at different resolutions are compared on the coarser grid
        # (block average of log10 over the finer cells).
        log_fv = np.log10(frames_fv[key][index_fv])
        log_fd = np.log10(frames_fd[key][index_fd])
        coarse_shape = tuple(min(a, b) for a, b in zip(log_fv.shape, log_fd.shape))
        difference = np.abs(_coarsen(log_fv, coarse_shape) - _coarsen(log_fd, coarse_shape))
        result[f"{key}_median_dlog"] = float(np.median(difference))
        result[f"{key}_mean_dlog"] = float(np.mean(difference))
    # Hot-gas mass-weighted fraction of the slice (a proxy for the cavity).
    for name, frames, index in (("fd", frames_fd, index_fd), ("fv", frames_fv, index_fv)):
        temperature = frames["T_xz"][index]
        density = frames["n_xz"][index]
        hot = temperature > 10**6.5
        result[f"hot_area_{name}"] = float(np.mean(hot))
        result[f"hot_mass_{name}"] = float(np.sum(density * hot) / np.sum(density))
    return result


# -------------------------------------------------------------
# ======================= ↑ Metrics ↑ =========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ Figure ↓ ==========================
# -------------------------------------------------------------


def plot_comparison(frames_fd, frames_fv, label, out_path, fd_name, fv_name):
    """Outflow rates (left) and n / T slices of both runs (right)."""
    figure = plt.figure(figsize=(20, 12.5))
    grid = figure.add_gridspec(2, 7, width_ratios=[2.2, 1, 1, 1, 1, 1, 1])

    rate_axis = figure.add_subplot(grid[:, 0])
    for frames, color, name in ((frames_fd, "tab:blue", fd_name), (frames_fv, "tab:red", fv_name)):
        for height, style in ((5.0, "-"), (8.0, "--")):
            rate_axis.plot(
                frames["time_myr"],
                outflow_through(frames, height),
                style,
                color=color,
                label=f"{name} |z|={height:.0f}",
            )
    rate_axis.set_xlabel("t [Myr]")
    rate_axis.set_ylabel(r"net outflow [$M_\odot$/yr]")
    rate_axis.set_title(f"{label}: outflow rate")
    rate_axis.legend(fontsize=8)

    density_norm = LogNorm(1e-4, 1e2)
    temperature_norm = LogNorm(1e3, 10**7.5)
    for row, (frames, name) in enumerate(((frames_fd, fd_name), (frames_fv, fv_name))):
        column = 1
        for time_myr in SLICE_TIMES_MYR:
            index = frame_at(frames, time_myr)
            for key, norm, cmap, quantity in (
                ("n_xz", density_norm, "viridis", "n"),
                ("T_xz", temperature_norm, "inferno", "T"),
            ):
                axis = figure.add_subplot(grid[row, column])
                axis.imshow(frames[key][index].T, origin="lower", norm=norm, cmap=cmap)
                axis.set_title(f"{name} {quantity} {frames['time_myr'][index]:.0f} Myr", fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
                column += 1

    figure.suptitle(f"{label}: {fd_name} vs {fv_name}")
    figure.tight_layout()
    figure.savefig(out_path, dpi=100)
    plt.close(figure)


# -------------------------------------------------------------
# ======================= ↑ Figure ↑ ==========================
# -------------------------------------------------------------


def main():
    """Parse the arguments, print the comparison and write the figure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fd-frames", required=True)
    parser.add_argument("--fv-frames", required=True)
    parser.add_argument("--fd-log", required=True)
    parser.add_argument("--fv-log", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fd-name", default="FD WENO5")
    parser.add_argument("--fv-name", default="FV VL2")
    arguments = parser.parse_args()

    frames_fd = load_frames(arguments.fd_frames)
    frames_fv = load_frames(arguments.fv_frames)

    print(f"=== {arguments.label} ===")
    metrics_fd = outflow_metrics(frames_fd)
    metrics_fv = outflow_metrics(frames_fv)
    for key in metrics_fd:
        change = (
            100.0 * (metrics_fv[key] - metrics_fd[key]) / abs(metrics_fd[key])
            if not key.startswith("tpeak")
            else metrics_fv[key] - metrics_fd[key]
        )
        unit = "Myr" if key.startswith("tpeak") else "%"
        print(f"  {key:8s} fd {metrics_fd[key]:8.3f}  fv {metrics_fv[key]:8.3f}  ({change:+.1f} {unit})")

    for name, log_path in (("fd", arguments.fd_log), ("fv", arguments.fv_log)):
        times, masses = load_diag_log(log_path)
        positive_jumps = np.diff(masses)[np.diff(masses) > 0]
        print(
            f"  {name}: {len(times)} steps to t = {times[-1] * CODE_TIME_MYR:.2f} Myr, "
            f"mean dt {np.mean(np.diff(times)) * CODE_TIME_MYR * 1e3:.2f} kyr, "
            f"mass {masses[0]:.2f} -> {masses[-1]:.2f}, "
            f"largest single-step dM {np.max(np.abs(np.diff(masses))):.3e}"
            f" (positive steps: {len(positive_jumps)})"
        )

    for time_myr in SLICE_TIMES_MYR:
        result = slice_metrics(frames_fd, frames_fv, time_myr)
        print(
            f"  {time_myr:.0f} Myr: |dlog n| median {result['n_xz_median_dlog']:.3f} "
            f"mean {result['n_xz_mean_dlog']:.3f}; |dlog T| median "
            f"{result['T_xz_median_dlog']:.3f} mean {result['T_xz_mean_dlog']:.3f}; "
            f"hot area fd {result['hot_area_fd']:.3f} fv {result['hot_area_fv']:.3f}"
        )

    plot_comparison(
        frames_fd,
        frames_fv,
        arguments.label,
        arguments.out,
        arguments.fd_name,
        arguments.fv_name,
    )
    print(f"  wrote {arguments.out}")


if __name__ == "__main__":
    main()
