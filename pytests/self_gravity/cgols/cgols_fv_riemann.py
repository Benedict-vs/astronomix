"""
Riemann-solver ablation of the FV (VL2) CGOLS runs: FD vs FV-HLLC vs FV-HLLE.

Plots per resolution the 5 kpc net outflow, L_X1 (CGOLS Eq. 17, from the
cgols_emission.py caches), the axis "pinch" metric (mean log n on the z-axis at
4 < |z| < 9 kpc; high = the cavity is pinched shut) and density slices at 50 /
65 / 75 Myr, and prints the summary table (outflow and L_X1 means, pinch,
up-down mirror asymmetry rms[log n(x,z) - log n(x,-z)]).

Needs the emission caches data/studies/emission_<tag>.npz, e.g.

    CGOLS_SOLVER=fv CGOLS_RIEMANN=hlle CGOLS_DIM=256 CGOLS_RUN_TAG=_A256fvhlle \\
        python cgols_emission.py --tag _A256fvhlle --source frames \\
        --frames-dir cgols_snapshots_diagnostics/A256fvhlle

    python cgols_fv_riemann.py 256      # or 512
"""

# general
import os
import sys

# numerics
import numpy as np

# plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BOX_Z = 20.0  # kpc

#: FV-HLLC reference run per resolution (fp32, like the HLLE runs).
FV_HLLC = {
    "256": "cgols_snapshots_diagnostics/A256fv",
    "512": "cgols_snapshots_A512fvf32",
}


def load(frame_dir):
    """Frames of one run, sorted by time."""
    names = sorted(f for f in os.listdir(frame_dir) if f.startswith("frame_"))
    frames = [np.load(os.path.join(frame_dir, f)) for f in names]
    return np.array([float(f["time_myr"]) for f in frames]), frames


def net_outflow(frames, height):
    """Net outflow [Msun/yr] through |z| = height."""
    mdot = np.stack([f["mdot_z"] for f in frames])
    z = (np.arange(mdot.shape[1]) + 0.5) * BOX_Z / mdot.shape[1] - BOX_Z / 2
    return mdot[:, np.argmin(np.abs(z - height))] - mdot[:, np.argmin(np.abs(z + height))]


def pinch(frame):
    """Mean log n on the z-axis at 4 < |z| < 9 kpc (high = pinched)."""
    n = frame["n_xz"]
    nx, nz = n.shape
    z = (np.arange(nz) + 0.5) * BOX_Z / nz - BOX_Z / 2
    band = (np.abs(z) > 4) & (np.abs(z) < 9)
    w = max(1, nx // 128)
    return np.log10(n[nx // 2 - w : nx // 2 + w].mean(0))[band].mean()


def main():
    dim = sys.argv[1] if len(sys.argv) > 1 else "256"
    runs = [
        ("FD WENO5 (LF split)", f"cgols_snapshots_A{dim}", f"A{dim}", "C0"),
        ("FV VL2 HLLC", FV_HLLC[dim], os.path.basename(FV_HLLC[dim]).replace("cgols_snapshots_", ""), "C3"),
        ("FV VL2 HLLE", f"cgols_snapshots_diagnostics/A{dim}fvhlle", f"A{dim}fvhlle", "C2"),
    ]
    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(3, 5)
    ax_out, ax_x, ax_pin = (fig.add_subplot(grid[i, :2]) for i in range(3))
    ref = np.load("data/studies/emission_A1024.npz")
    ax_x.semilogy(ref["per_snapshot__time_myr"], ref["per_snapshot__L_xray_cgols_xz"],
                  color="k", lw=0.8, ls=":", label="FD 1024 (reference)")

    print(f"{'run':20s} out5 15-40  out5 50-75  LX1 15-40  LX1 60-75  pinch 60/65/70/75       asym 50/75")
    for row, (label, frame_dir, tag, color) in enumerate(runs):
        t, frames = load(frame_dir)
        out5 = net_outflow(frames, 5.0)
        pin = np.array([pinch(f) for f in frames])
        emis = np.load(f"data/studies/emission_{tag}.npz")
        te, lx = emis["per_snapshot__time_myr"], emis["per_snapshot__L_xray_cgols_xz"]
        ax_out.plot(t, out5, color=color, label=label)
        ax_x.semilogy(te, lx, color=color, label=label)
        ax_pin.plot(t, pin, color=color, label=label)

        def at(time):
            return frames[int(np.argmin(np.abs(t - time)))]

        asym = []
        for time in (50, 75):
            log_n = np.log10(at(time)["n_xz"].astype(np.float64))
            asym.append(np.sqrt(np.mean((log_n - log_n[:, ::-1]) ** 2)))
        print(
            f"{label:20s} {out5[(t >= 15) & (t <= 40)].mean():10.2f} {out5[(t >= 50) & (t <= 75)].mean():11.2f}"
            f" {lx[(te >= 15) & (te <= 40)].mean() / 1e39:10.1f} {lx[(te >= 60) & (te <= 75)].mean() / 1e39:10.1f}"
            f"  {' '.join(f'{pinch(at(T)):.2f}' for T in (60, 65, 70, 75))}  {asym[0]:.3f} {asym[1]:.3f}"
        )
        for col, time in enumerate((50, 65, 75)):
            ax = fig.add_subplot(grid[row, 2 + col])
            frame = at(time)
            ax.imshow(np.log10(frame["n_xz"]).T, origin="lower", vmin=-4.5, vmax=0.5,
                      cmap="viridis", extent=(-5, 5, -10, 10))
            ax.set_title(f"{label}\n{float(frame['time_myr']):.0f} Myr", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    ax_out.set_ylabel("net outflow |z|=5 kpc [Msun/yr]")
    ax_out.legend(fontsize=8)
    ax_x.set(ylabel="L_X1 (CGOLS Eq. 17) [erg/s]", xlim=(0, 75), ylim=(1e39, 2e42))
    ax_x.legend(fontsize=7)
    ax_pin.set(ylabel="axis <log n>, 4<|z|<9 kpc\n(high = pinched)", xlabel="t [Myr]")
    fig.suptitle(f"A{dim}: Riemann-solver ablation (FV HLLC vs HLLE vs FD)")
    fig.tight_layout()
    path = f"figures/cgols/comparisons/fv_riemann_A{dim}.png"
    fig.savefig(path, dpi=90)
    print(path)


if __name__ == "__main__":
    main()
