"""
Seed scatter of the CGOLS model-A runs: is FD (WENO5) - FV (VL2 + HLLC) larger than the run-to-run noise?

Every resolution has four FD and four FV runs that differ only in the t = 0 noise (CGOLS_IC_SEED = 1, 2, 3:
rho and P x (1 + 1e-3 N(0, 1)); seed 0 = the noise-free default IC). For each metric the script prints the
FD and FV ensemble mean +- std (and min-max), the difference FD - FV and that difference in units of the
combined single-run scatter sqrt(std_FD^2 + std_FV^2) -- the scatter of the difference of two single runs,
i.e. what a one-run-per-scheme comparison can resolve (the error of the ensemble-mean difference is half of
it for four runs each). The last column repeats the ratio with seeds 1-3 only (see the seed-0 caveat below).

Metrics (conventions of cgols_fv_riemann.py, whose helpers are reused):
  out5 15-40 / 50-75    mean net outflow through |z| = 5 kpc [Msun/yr]
  out5 peak             its maximum over the run (frame cadence 1.25 Myr)
  LX1 15-40 / 60-75     mean L_X1 (CGOLS I Eq. 17) [1e39 erg/s] from data/studies/emission_<tag>.npz
  pinch 60/65/70/75     axis <log n> at 4 < |z| < 9 kpc, nearest frame (open cavity ~ -3.95)
  pinched frac          fraction of the frames at t >= 60 Myr with pinch > PINCH_THRESHOLD
  asym 50/75            up-down mirror asymmetry rms[log n(x, z) - log n(x, -z)] of the x-z slice

Seed-0 caveat: the seed runs ran without checkpoints (CGOLS_CHECKPOINT_EVERY=0, frames at the first step past
each 1.25 Myr mark), the seed-0 baselines with them (TO_DISK segments ending exactly on the frame times). The
segment ends are one deterministic perturbation of the step sequence: the seed-0 mass history differs from the
seed runs by ~1e-4 while the seeds differ among themselves by ~1e-6 (diag logs), the noise-free FV-256 run
without checkpoints (cgols_snapshots_diagnostics/A256nockfv, no emission maps) tracks the seeds instead. The
seed-0 runs are therefore four more ensemble members with a different perturbation, not a different scheme.
The seed runs' IC noise also seeds the up-down asymmetry directly (seed 0 is mirror-symmetric to round-off).

Emission caches of the seed runs and of the FV-128 baseline come from the frames' in-situ maps, e.g.

    JAX_PLATFORMS=cpu python cgols_emission.py --tag _A256fvs1 --source frames \\
        --frames-dir cgols_snapshots_diagnostics/A256fvs1 --no-figures

    python cgols_seed_scatter.py            # both resolutions; or: python cgols_seed_scatter.py 256

Writes figures/cgols/comparisons/seed_scatter_A<dim>.png. CPU only (numpy + matplotlib, no jax).
"""

# general
import os
import sys

# plotting (Agg backend, also selected by cgols_fv_riemann on import)
import matplotlib
import matplotlib.pyplot as plt

# numerics
import numpy as np

matplotlib.use("Agg")

# shared CGOLS frame helpers
from cgols_fv_riemann import load, net_outflow, pinch

# -------------------------------------------------------------
# =============== ↓ Runs ↓ ====================================
# -------------------------------------------------------------

DIAG = "cgols_snapshots_diagnostics"

#: Axis <log n> above which a frame counts as pinched: the open cavity sits at -3.98 to -3.94 in every run,
#: pinched frames at -3.3 to -3.0; the transitions pass through -3.9 to -3.5 within 1-3 frames.
PINCH_THRESHOLD = -3.5
LATE_MYR = 60.0
SLICE_MYR = 65.0
WINDOW_MARGIN_MYR = 0.2


def runs(dim):
    """``{scheme: [(seed, frame dir, emission tag), ...]}`` for one resolution."""
    fv_seed0 = {
        # = cgols_snapshots_diagnostics/A128fv to 1e-7 in mass (same config, also checkpointed), but only the
        # JUPITER rerun carries the in-situ emission maps.
        "128": ("cgols_snapshots_A128fvf32", "_A128fvf32"),
        "256": (f"{DIAG}/A256fv", "_A256fv"),
    }[dim]
    return {
        "FD": [(0, f"cgols_snapshots_A{dim}", f"_A{dim}")]
        + [(s, f"{DIAG}/A{dim}s{s}", f"_A{dim}s{s}") for s in (1, 2, 3)],
        "FV": [(0, *fv_seed0)] + [(s, f"{DIAG}/A{dim}fvs{s}", f"_A{dim}fvs{s}") for s in (1, 2, 3)],
    }


LABELS = {"FD": "FD WENO5", "FV": "FV VL2 HLLC"}
CMAPS = {"FD": plt.get_cmap("Blues"), "FV": plt.get_cmap("Oranges")}


def color(scheme, seed):
    """One hue family per scheme, a lightness step per seed."""
    return CMAPS[scheme](0.95 - 0.17 * seed)


# -------------------------------------------------------------
# =============== ↓ Metrics ↓ =================================
# -------------------------------------------------------------


def window(t, start, end):
    """Frames in [start, end] Myr, with a margin for the seed runs' frame times.

    Without checkpoints a frame is written at the first step past each 1.25 Myr mark (up to ~0.07 Myr late
    early on, <= 0.01 Myr after 10 Myr), with them exactly on it; the margin keeps e.g. the 40.01 Myr frame
    in "15-40" so every run averages the same 21 frames.
    """
    return (t >= start - WINDOW_MARGIN_MYR) & (t <= end + WINDOW_MARGIN_MYR)


def asymmetry(frame):
    """Up-down mirror asymmetry rms[log n(x, z) - log n(x, -z)]."""
    log_n = np.log10(frame["n_xz"].astype(np.float64))
    return np.sqrt(np.mean((log_n - log_n[:, ::-1]) ** 2))


def measure(frame_dir, tag):
    """Time series and scalar metrics of one run."""
    t, frames = load(frame_dir)
    out5 = net_outflow(frames, 5.0)
    pin = np.array([pinch(f) for f in frames])
    emis = np.load(f"data/studies/emission{tag}.npz")
    te, lx = emis["per_snapshot__time_myr"], emis["per_snapshot__L_xray_cgols_xz"] / 1e39

    def at(time):
        return int(np.argmin(np.abs(t - time)))

    late = window(t, LATE_MYR, 75)
    scalars = {
        "out5 15-40 [Msun/yr]": out5[window(t, 15, 40)].mean(),
        "out5 50-75 [Msun/yr]": out5[window(t, 50, 75)].mean(),
        "out5 peak [Msun/yr]": out5.max(),
        "LX1 15-40 [1e39]": lx[window(te, 15, 40)].mean(),
        "LX1 60-75 [1e39]": lx[window(te, 60, 75)].mean(),
        **{f"pinch {T:.0f}": pin[at(T)] for T in (60, 65, 70, 75)},
        f"pinched frac t>={LATE_MYR:.0f}": np.mean(pin[late] > PINCH_THRESHOLD),
        **{f"asym {T:.0f}": asymmetry(frames[at(T)]) for T in (50, 75)},
    }
    return {
        "t": t, "out5": out5, "pinch": pin, "te": te, "lx": lx, "slice": frames[at(SLICE_MYR)],
        "scalars": scalars,
    }


def summary_table(dim, results):
    """Print the per-metric ensemble table of one resolution."""
    print(f"\n=== A{dim}: FD vs FV-HLLC over IC seeds 0-3 (pinch threshold {PINCH_THRESHOLD}) ===")
    print(f"{'metric':22s} {'FD mean+-std [min, max]':>30s} {'FV mean+-std [min, max]':>30s}"
          f" {'s0 FD/FV':>15s} {'FD-FV':>8s} {'/sig':>6s} {'/sig s1-3':>9s}")
    for metric in results["FD"][0]["scalars"]:
        values = {scheme: np.array([r["scalars"][metric] for r in results[scheme]]) for scheme in results}
        cells = []
        for scheme in ("FD", "FV"):
            v = values[scheme]
            cells.append(f"{v.mean():7.3f} +- {v.std(ddof=1):6.3f} [{v.min():6.2f}, {v.max():6.2f}]")
        diff = values["FD"].mean() - values["FV"].mean()

        def ratio(fd, fv):
            sigma, delta = np.hypot(fd.std(ddof=1), fv.std(ddof=1)), fd.mean() - fv.mean()
            if sigma == 0:
                return "-" if delta == 0 else "inf"
            return f"{delta / sigma:.1f}"

        seed0 = f"{values['FD'][0]:.2f}/{values['FV'][0]:.2f}"
        print(f"{metric:22s} {cells[0]:>30s} {cells[1]:>30s} {seed0:>15s} {diff:8.3f}"
              f" {ratio(values['FD'], values['FV']):>6s} {ratio(values['FD'][1:], values['FV'][1:]):>9s}")


def nock_check(dim):
    """FV-256 seed 0 without checkpoints (no emission maps): out5 and pinch only."""
    frame_dir = f"{DIAG}/A{dim}nockfv"
    if not os.path.isdir(frame_dir):  # only A256nockfv exists
        return
    t, frames = load(frame_dir)
    out5 = net_outflow(frames, 5.0)
    pin = np.array([pinch(f) for f in frames])
    picks = " ".join(f"{pin[int(np.argmin(np.abs(t - T)))]:.2f}" for T in (60, 65, 70, 75))
    print(f"  cross-check {frame_dir} (seed 0, no checkpoints): out5 15-40 / 50-75 / peak "
          f"{out5[window(t, 15, 40)].mean():.2f} / {out5[window(t, 50, 75)].mean():.2f} / {out5.max():.2f}, "
          f"pinch 60/65/70/75 {picks}, pinched frac {np.mean(pin[window(t, LATE_MYR, 75)] > PINCH_THRESHOLD):.2f}")


# -------------------------------------------------------------
# =============== ↓ Figure ↓ ==================================
# -------------------------------------------------------------


def figure(dim, results):
    """Outflow, L_X1 and pinch vs time (thin line per seed) plus a density slice per run at SLICE_MYR."""
    fig = plt.figure(figsize=(15, 9.5))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.25), hspace=0.22)
    top = outer[0].subgridspec(1, 3, wspace=0.25)
    bottom = outer[1].subgridspec(1, 8)
    ax_out, ax_x, ax_pin = (fig.add_subplot(top[0, i]) for i in range(3))
    ref = "data/studies/emission_A1024.npz"
    if os.path.exists(ref):
        emis = np.load(ref)
        ax_x.semilogy(emis["per_snapshot__time_myr"], emis["per_snapshot__L_xray_cgols_xz"] / 1e39,
                      color="0.3", lw=0.9, ls=":", label="FD 1024 (reference)")
    for scheme, members in results.items():
        for seed, r in enumerate(members):
            style = {"color": color(scheme, seed), "lw": 1.0, "ls": "--" if seed == 0 else "-"}
            label = f"{LABELS[scheme]} seed {seed}" + (" (ckpt-segmented)" if seed == 0 else "")
            ax_out.plot(r["t"], r["out5"], label=label, **style)
            ax_x.semilogy(r["te"], r["lx"], **style)
            ax_pin.plot(r["t"], r["pinch"], **style)
    ax_pin.axhline(PINCH_THRESHOLD, color="0.5", lw=0.7, ls=":")
    ax_pin.text(40.5, PINCH_THRESHOLD + 0.03, "pinched above", fontsize=8, color="0.35")
    ax_out.set(xlabel="t [Myr]", ylabel="net outflow at |z| = 5 kpc [Msun/yr]", xlim=(0, 75))
    ax_x.set(xlabel="t [Myr]", ylabel="L_X1, CGOLS Eq. 17 [1e39 erg/s]", xlim=(0, 75), ylim=(1, 2e3))
    ax_pin.set(xlabel="t [Myr]", ylabel="pinch: axis <log n> at 4 < |z| < 9 kpc", xlim=(40, 75))
    for ax in (ax_out, ax_x, ax_pin):
        ax.grid(alpha=0.25, lw=0.5)
        for T in (15, 40, 50, 60):
            ax.axvline(T, color="0.85", lw=0.6, zorder=0)

    image = None
    for col, (scheme, seed, r) in enumerate((s, k, r) for s in results for k, r in enumerate(results[s])):
        ax = fig.add_subplot(bottom[0, col])
        image = ax.imshow(np.log10(r["slice"]["n_xz"]).T, origin="lower", vmin=-4.5, vmax=0.5,
                          cmap="viridis", extent=(-5, 5, -10, 10))
        ax.set_title(f"{scheme} seed {seed}\n{float(r['slice']['time_myr']):.1f} Myr, pinch {pinch(r['slice']):.2f}",
                     fontsize=8, color=color(scheme, 0))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=fig.axes[-8:], fraction=0.015, pad=0.01, label="log n [cm^-3] (x-z slice)")
    handles, labels = ax_out.get_legend_handles_labels()
    extra_handles, extra_labels = ax_x.get_legend_handles_labels()
    fig.legend(handles + extra_handles, labels + extra_labels, loc="upper center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, 0.955), frameon=False)
    fig.suptitle(f"A{dim}: FD vs FV-HLLC, IC-noise seeds 0-3 (dashed = seed 0, checkpoint-segmented baseline)",
                 y=0.99)
    path = f"figures/cgols/comparisons/seed_scatter_A{dim}.png"
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for dim in sys.argv[1:] or ("128", "256"):
        results = {
            scheme: [measure(frame_dir, tag) for _, frame_dir, tag in members]
            for scheme, members in runs(dim).items()
        }
        summary_table(dim, results)
        nock_check(dim)
        figure(dim, results)


if __name__ == "__main__":
    main()
