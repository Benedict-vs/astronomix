"""Magnetic-field evolution of a CGOLS MHD run: time series, animation, final 3D render.

Reads the run's frames (x-z slices of n, |B|, B_x, B_z, beta and the scalar
magnetic energy / max|B| / div B that cgols.py stores with CGOLS_MHD=1) and its
final full state, and writes into figures/cgols/<tag>/:

  bfield_evolution.png    E_B(t), max|B|(t), the B^2-weighted vertical fraction
                          of the field in the disk and above it, div B(t)
  bfield_animation.gif    x-z slices of n and |B| with in-plane field lines
  bfield_final_render.png 3D field lines traced through the final state, with
                          face-on and edge-on |B| slices

    JAX_PLATFORMS=cpu CGOLS_MHD=1 CGOLS_DIM=256 \\
        python cgols_mhd_field_evolution.py --tag _A256mhd \\
        --frames-dir <path>/cgols_snapshots_A256mhd
"""

import argparse
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import RegularGridInterpolator

import cgols

HERE = os.path.dirname(os.path.abspath(__file__))
B_NORM = LogNorm(0.1, 200)  # microgauss
N_NORM = LogNorm(1e-4, 1e2)  # cm^-3


def load_frames(frames_dir):
    """Stack the per-frame npz files into arrays (snapshot axis first)."""
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.npz")))
    frames = [np.load(f) for f in files]
    keys = ("time_myr", "n_xz", "B_xz_uG", "Bx_xz_uG", "Bz_xz_uG", "beta_xz",
            "magnetic_energy", "max_B_uG", "divB_rel_interior", "divB_rel_all")
    return {k: np.stack([fr[k] for fr in frames]) for k in keys}


def slice_axes(config, shape):
    """Centred x and z cell-centre axes (kpc) of an x-z slice."""
    nx, nz = shape
    x = (np.arange(nx) + 0.5) * config.box_size.x / nx - config.box_size.x / 2
    z = (np.arange(nz) + 0.5) * config.box_size.z / nz - config.box_size.z / 2
    return x, z


def plot_evolution(data, config, out):
    """E_B, max|B|, vertical-field fraction (disk vs |z| > 1 kpc) and div B vs time."""
    t = data["time_myr"]
    x, z = slice_axes(config, data["B_xz_uG"].shape[1:])
    above = np.abs(z) > 1.0
    in_disk = np.abs(z) < 0.3

    def vertical_fraction(mask):
        b2 = data["B_xz_uG"][:, :, mask] ** 2
        bz2 = data["Bz_xz_uG"][:, :, mask] ** 2
        return bz2.sum(axis=(1, 2)) / np.maximum(b2.sum(axis=(1, 2)), 1e-30)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    axes[0, 0].plot(t, data["magnetic_energy"] / data["magnetic_energy"][0], "k-")
    axes[0, 0].set(ylabel=r"$E_B(t) / E_B(0)$", title="total magnetic energy")
    axes[0, 1].semilogy(t, data["max_B_uG"], "C3-")
    axes[0, 1].set(ylabel=r"max |B| [$\mu$G]", title="peak field")
    axes[1, 0].plot(t, vertical_fraction(in_disk), "C0-", label="disk, |z| < 0.3 kpc")
    axes[1, 0].plot(t, vertical_fraction(above), "C1-", label="above the disk, |z| > 1 kpc")
    axes[1, 0].axhline(1 / 3, color="0.6", ls=":", label="isotropic (1/3)")
    axes[1, 0].set(ylabel=r"$\sum B_z^2 / \sum B^2$ (y = 0 slice)", xlabel="t [Myr]",
                   title="how vertical is the field?", ylim=(0, 1))
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].semilogy(t, data["divB_rel_interior"], "k-", label="interior")
    axes[1, 1].semilogy(t, data["divB_rel_all"], "C3--", label="incl. boundary cells")
    axes[1, 1].set(ylabel=r"max |div B| dx / max|B|", xlabel="t [Myr]", title="div B")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.axvspan(5, 45, color="C1", alpha=0.07)
    figure.suptitle("magnetic field evolution (shaded: high starburst state)")
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    plt.close(figure)
    print(f"wrote {out}")


def write_animation(data, config, out, z_half=6.0):
    """GIF of n and |B| x-z slices with B_x/B_z field lines, one frame per snapshot."""
    x, z = slice_axes(config, data["B_xz_uG"].shape[1:])
    keep = np.abs(z) <= z_half
    zk = z[keep]
    extent = [x[0], x[-1], zk[0], zk[-1]]
    cmap = plt.get_cmap("inferno").with_extremes(bad="black")
    figure, (ax_n, ax_b) = plt.subplots(1, 2, figsize=(9, 7.5))

    def draw(i):
        for axis in (ax_n, ax_b):
            axis.clear()
        ax_n.imshow(data["n_xz"][i][:, keep].T, origin="lower", extent=extent,
                    norm=N_NORM, cmap="viridis")
        b = data["B_xz_uG"][i][:, keep]
        ax_b.imshow(np.where(b > 0, b, np.nan).T, origin="lower", extent=extent,
                    norm=B_NORM, cmap=cmap)
        bx, bz = data["Bx_xz_uG"][i][:, keep].T, data["Bz_xz_uG"][i][:, keep].T
        strength = np.hypot(bx, bz)
        if strength.max() > 0:
            ax_b.streamplot(x, zk, np.where(strength > 0.3, bx, np.nan),
                            np.where(strength > 0.3, bz, np.nan),
                            color="white", linewidth=0.5, density=1.4, arrowsize=0.5)
        ax_n.set(title=r"n [cm$^{-3}$]", xlabel="x [kpc]", ylabel="z [kpc]")
        ax_b.set(title=r"|B| [$\mu$G] and in-plane field lines", xlabel="x [kpc]",
                 xlim=(x[0], x[-1]), ylim=(zk[0], zk[-1]))
        figure.suptitle(f"t = {data['time_myr'][i]:.1f} Myr")

    draw(0)
    figure.colorbar(plt.cm.ScalarMappable(N_NORM, "viridis"), ax=ax_n, shrink=0.7)
    figure.colorbar(plt.cm.ScalarMappable(B_NORM, cmap), ax=ax_b, shrink=0.7)
    animation = FuncAnimation(figure, draw, frames=len(data["time_myr"]))
    animation.save(out, writer=PillowWriter(fps=6), dpi=90)
    plt.close(figure)
    print(f"wrote {out}")


def trace_line(interpolators, seed, bounds, step=0.02, max_steps=1500):
    """RK4 field line along the unit B direction, traced both ways from seed."""
    def direction(point):
        b = np.array([f(point)[0] for f in interpolators])
        norm = np.linalg.norm(b)
        return b / norm if norm > 0 else np.zeros(3)

    def walk(sign):
        p, points = np.array(seed, dtype=float), []
        for _ in range(max_steps):
            k1 = direction(p)
            k2 = direction(p + 0.5 * sign * step * k1)
            k3 = direction(p + 0.5 * sign * step * k2)
            k4 = direction(p + sign * step * k3)
            increment = sign * step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if not increment.any():
                break
            p = p + increment
            if np.any(p <= bounds[0]) or np.any(p >= bounds[1]):
                break
            points.append(p.copy())
        return points

    backward, forward = walk(-1.0), walk(1.0)
    return np.array(backward[::-1] + [np.array(seed, dtype=float)] + forward)


def render_final(state_path, config, registered_variables, time_myr, out, coarsen=2, seed_count=90):
    """3D field lines of the final state plus face-on / edge-on |B| slices."""
    state = np.load(state_path, mmap_mode="r")
    to_uG = float(cgols._code_field_to_microgauss())
    mi = registered_variables.magnetic_index
    B = [np.asarray(state[i][::coarsen, ::coarsen, ::coarsen], dtype=np.float64) * to_uG
         for i in (mi.x, mi.y, mi.z)]
    rho = np.asarray(state[registered_variables.density_index][::coarsen, ::coarsen, ::coarsen])
    babs = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
    nx, ny, nz = babs.shape
    L = (config.box_size.x, config.box_size.y, config.box_size.z)
    axes1d = [(np.arange(n) + 0.5) * l / n - l / 2 for n, l in zip((nx, ny, nz), L)]
    interpolators = [RegularGridInterpolator(axes1d, b, bounds_error=False, fill_value=0.0) for b in B]
    babs_interp = RegularGridInterpolator(axes1d, babs, bounds_error=False, fill_value=0.0)
    bounds = (np.array([a[0] for a in axes1d]), np.array([a[-1] for a in axes1d]))

    # Seeds where the field is appreciable, weighted towards strong field, mixing
    # disk (|z| < 0.3 kpc) and lifted (|z| > 0.5 kpc) field.
    rng = np.random.default_rng(1)
    X, Y, Z = np.meshgrid(*axes1d, indexing="ij")
    seeds = []
    for mask, count in ((np.abs(Z) < 0.3, seed_count // 3), (np.abs(Z) > 0.5, 2 * seed_count // 3)):
        weight = np.where(mask & (babs > 1.0), babs, 0.0).ravel()
        if weight.sum() == 0:
            continue
        idx = rng.choice(weight.size, size=count, replace=False, p=weight / weight.sum())
        seeds += [(X.flat[i], Y.flat[i], Z.flat[i]) for i in idx]
    lines = [trace_line(interpolators, s, bounds) for s in seeds]

    cmap = plt.get_cmap("inferno")
    figure = plt.figure(figsize=(16, 8))
    gs = figure.add_gridspec(2, 2, width_ratios=[1.6, 1.0])
    ax3 = figure.add_subplot(gs[:, 0], projection="3d")
    for line in lines:
        if len(line) < 5:
            continue
        segments = np.stack([line[:-1], line[1:]], axis=1)
        values = np.maximum(babs_interp(line[:-1]), 0.1)
        ax3.add_collection3d(Line3DCollection(segments, colors=cmap(B_NORM(values)), linewidths=0.8, alpha=0.85))
    ax3.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(-6, 6), xlabel="x [kpc]", ylabel="y [kpc]", zlabel="z [kpc]")
    ax3.set_box_aspect((1, 1, 1.2))
    ax3.view_init(elev=12, azim=-55)
    ax3.set_title(f"field lines at t = {time_myr:.0f} Myr, coloured by |B|")
    figure.colorbar(plt.cm.ScalarMappable(B_NORM, cmap), ax=ax3, shrink=0.55, label=r"|B| [$\mu$G]")

    k_mid, j_mid = nz // 2, ny // 2
    slice_cmap = cmap.with_extremes(bad="black")
    ax = figure.add_subplot(gs[0, 1])
    ax.imshow(np.where(babs[:, :, k_mid] > 0, babs[:, :, k_mid], np.nan).T, origin="lower",
              extent=[axes1d[0][0], axes1d[0][-1], axes1d[1][0], axes1d[1][-1]], norm=B_NORM, cmap=slice_cmap)
    ax.streamplot(axes1d[0], axes1d[1], B[0][:, :, k_mid].T, B[1][:, :, k_mid].T, color="white",
                  linewidth=0.5, density=1.2, arrowsize=0.5)
    ax.set(title="face-on midplane |B|", xlabel="x [kpc]", ylabel="y [kpc]")
    ax = figure.add_subplot(gs[1, 1])
    keep = np.abs(axes1d[2]) <= 6
    ax.imshow(np.where(babs[:, j_mid, keep] > 0, babs[:, j_mid, keep], np.nan).T, origin="lower",
              extent=[axes1d[0][0], axes1d[0][-1], axes1d[2][keep][0], axes1d[2][keep][-1]],
              norm=B_NORM, cmap=slice_cmap, aspect="auto")
    ax.streamplot(axes1d[0], axes1d[2][keep], B[0][:, j_mid, keep].T, B[2][:, j_mid, keep].T,
                  color="white", linewidth=0.5, density=1.2, arrowsize=0.5)
    ax.set(title="edge-on |B| (y = 0)", xlabel="x [kpc]", ylabel="z [kpc]")
    del rho
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    plt.close(figure)
    print(f"wrote {out} ({len(lines)} field lines)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--final-state", default=None)
    args = parser.parse_args()
    tag = args.tag.lstrip("_")
    frames_dir = args.frames_dir or os.path.join(HERE, f"cgols_snapshots_{tag}")
    final_state = args.final_state or f"/export/scratch/bschuber/final/cgols_final_state_{tag}.npy"
    out_dir = os.path.join(HERE, "figures", "cgols", tag)
    os.makedirs(out_dir, exist_ok=True)

    config, registered_variables = cgols.build_config()
    data = load_frames(frames_dir)
    plot_evolution(data, config, os.path.join(out_dir, "bfield_evolution.png"))
    write_animation(data, config, os.path.join(out_dir, "bfield_animation.gif"))
    render_final(final_state, config, registered_variables, float(data["time_myr"][-1]),
                 os.path.join(out_dir, "bfield_final_render.png"))


if __name__ == "__main__":
    main()
