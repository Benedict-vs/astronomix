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

``python cgols_fv_riemann.py spectrum`` measures the small-scale structure
instead (the second dissipation channel, set by the reconstruction order):
the fraction of the x-variance of log n in 1 < |z| < 6 kpc at wavelengths
80-300 pc (slices block-averaged to the 256 grid, 39 pc) and 40-150 pc (512
grid, 19.5 pc), per frame, for FD and FV at 256-1024. It prints the 50-75 Myr
means and writes comparisons/fv_small_scale_structure.png.
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


#: Runs of the small-scale-structure measure: label, frame dir, colour, line style.
SPECTRUM_RUNS = (
    ("FD 256", "cgols_snapshots_A256", "C0", ":"),
    ("FD 512", "cgols_snapshots_A512", "C0", "--"),
    ("FD 1024", "cgols_snapshots_A1024", "C0", "-"),
    ("FV HLLC 256", "cgols_snapshots_diagnostics/A256fv", "C3", ":"),
    ("FV HLLC 512", "cgols_snapshots_A512fvf32", "C3", "--"),
    ("FV HLLC 1024", "cgols_snapshots_A1024fv", "C3", "-"),
    ("FV HLLE 512", "cgols_snapshots_diagnostics/A512fvhlle", "C2", "--"),
)

#: (common grid, wavelength band [kpc]) of the small-scale variance fraction.
SPECTRUM_BANDS = ((256, (0.08, 0.3)), (512, (0.04, 0.15)))


def small_scale_fraction(frame, grid, band):
    """Fraction of the x-variance of log n in 1 < |z| < 6 kpc at band wavelengths.

    The x-z slice is block-averaged (in n) to grid cells across the box, so runs
    at different resolutions are measured on the same grid; None if the run is
    coarser than grid.
    """
    n = frame["n_xz"].astype(np.float64)
    nx, nz = n.shape
    if nx < grid:
        return None
    f = nx // grid
    log_n = np.log10(n.reshape(grid, f, nz // f, f).mean(axis=(1, 3)))
    z = (np.arange(log_n.shape[1]) + 0.5) * BOX_Z / log_n.shape[1] - BOX_Z / 2
    rows = log_n[:, (np.abs(z) > 1) & (np.abs(z) < 6)]
    power = np.abs(np.fft.rfft(rows - rows.mean(0), axis=0)) ** 2
    k = np.fft.rfftfreq(grid, d=10.0 / grid)  # 1/kpc
    wavelength = 1.0 / np.maximum(k, 1e-12)
    in_band = (k > 0) & (wavelength >= band[0]) & (wavelength <= band[1])
    return power[in_band].sum() / power[1:].sum()


def spectrum():
    """Small-scale variance fractions of FD and FV against time (see the docstring)."""
    fig, axes = plt.subplots(1, len(SPECTRUM_BANDS), figsize=(13, 4.5))
    print("small-scale x-variance fraction of log n, 1<|z|<6 kpc, mean +- std (min-max) over frames at 50-75 Myr")
    for ax, (grid, band) in zip(axes, SPECTRUM_BANDS):
        print(f"  {grid} grid ({10e3 / grid:.0f} pc), {1e3 * band[0]:.0f}-{1e3 * band[1]:.0f} pc:")
        for label, frame_dir, color, style in SPECTRUM_RUNS:
            if not os.path.isdir(frame_dir):
                continue
            t, frames = load(frame_dir)
            values = [small_scale_fraction(f, grid, band) for f in frames]
            if values[0] is None:
                continue
            values = np.array(values)
            ax.plot(t, values, color=color, ls=style, label=label)
            late = values[(t >= 50) & (t <= 75.01)]
            print(f"    {label:13s} {late.mean():.4f} +- {late.std():.4f} ({late.min():.3f}-{late.max():.3f})")
        ax.set(
            xlabel="t [Myr]",
            ylabel="variance fraction",
            yscale="log",
            xlim=(5, 75),  # before 5 Myr there is no wind
            title=f"log n x-spectrum, {1e3 * band[0]:.0f}-{1e3 * band[1]:.0f} pc ({grid} grid), 1<|z|<6 kpc",
        )
        ax.axvspan(50, 75, color="0.93", zorder=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("A-series small-scale structure: FD WENO5 vs FV VL2 (PLM)")
    fig.tight_layout()
    path = "figures/cgols/comparisons/fv_small_scale_structure.png"
    fig.savefig(path, dpi=110)
    print(path)


def main():
    dim = sys.argv[1] if len(sys.argv) > 1 else "256"
    if dim == "spectrum":
        spectrum()
        return
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
