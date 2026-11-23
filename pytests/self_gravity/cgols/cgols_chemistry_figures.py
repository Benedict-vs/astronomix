"""Figures of the two chemistry runs from t = 0 to 75 Myr (fixed code, 256 x 256 x 512), with model B (and A) for reference.

usage: python cgols_chemistry_figures.py [out_dir]   (from pytests/self_gravity/cgols)

Runs (frames of the edge-on plane y = 0, every 1.25 Myr):
  model B     cooling curve (KI02) with the FUV field of every cell   cgols_snapshots_B256
  field       chemistry, FUV field of every cell, corrected H2 + H+ rate   cgols_snapshots_chem256fuv_v2
  Draine      chemistry, one Draine field everywhere, network as shipped   cgols_snapshots_chem256_v2
  model A     adiabatic (outflow rates only)                          cgols_snapshots_A256

Model B has no composition: where runs are compared, the thermal state is P / rho written as T / mu,
and the density is rho in m_p cm^-3. Disk statistics use the edge-on plane weighted with |x|
(axisymmetric estimate), disk = |z| < 0.5 kpc, rho > 0.135 m_p cm^-3, R < 4.5 kpc.
CPU only, a minute or two.
"""
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures/cgols/chem256_v2")
os.makedirs(OUT, exist_ok=True)

SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": TEXT, "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE})
HE_PER_H, C_PER_H = 0.085, 1.4e-4
MASS_PER_HYDROGEN = 1.0 + 4 * HE_PER_H + 16 * 3.2e-4 + 12 * C_PER_H + 28 * 1.8e-6  # [m_p]

RUNS = {
    "B": dict(label="model B: cooling curve, FUV field of every cell", colour="#2f6fb3", path=f"{HERE}/cgols_snapshots_B256"),
    "field": dict(label="chemistry, FUV field of every cell (corrected H$_2$ + H$^+$ rate)", colour="#d9772b",
                  path=f"{HERE}/cgols_snapshots_chem256fuv_v2"),
    "draine": dict(label="chemistry, one Draine field everywhere (network as shipped)", colour="#55524c",
                   path=f"{HERE}/cgols_snapshots_chem256_v2"),
    "A": dict(label="model A: adiabatic", colour="#9a9893", path=f"{HERE}/cgols_snapshots_A256"),
}
CHEM = ("field", "draine")
SHORT = {"B": "model B (cooling curve)", "field": "chemistry, field of every cell", "draine": "chemistry, one Draine field",
         "A": "model A (adiabatic)"}


def frame_files(run):
    return sorted(glob.glob(os.path.join(RUNS[run]["path"], "frame_*.npz")))


def load(run, index):
    """rho [m_p cm^-3], T/mu [K], T [K] and the frame itself."""
    data = np.load(frame_files(run)[index])
    if run in CHEM:
        particles = sum(np.asarray(data[k], dtype=float) for k in data.files if k.startswith("x_") and k.endswith("_xz"))
        rho = np.asarray(data["n_H_xz"], dtype=float) * MASS_PER_HYDROGEN
        temperature = np.asarray(data["T_xz"], dtype=float)
        return rho, temperature * particles / MASS_PER_HYDROGEN, temperature, data
    rho = np.asarray(data["n_xz"], dtype=float) * 0.6
    temperature = np.asarray(data["T_xz"], dtype=float)
    return rho, temperature / 0.6, temperature, data


def index_at(run, time_myr):
    times = [float(np.load(f)["time_myr"]) for f in frame_files(run)]
    return int(np.argmin(np.abs(np.array(times) - time_myr)))


NX, NZ = 256, 512
x = (np.arange(NX) + 0.5 - NX / 2) * 10.0 / NX
z = (np.arange(NZ) + 0.5 - NZ / 2) * 20.0 / NZ
RADIUS = np.abs(x)[:, None] * np.ones((1, NZ))
HEIGHT = np.abs(z)[None, :] * np.ones((NX, 1))
EXTENT = (-5, 5, -10, 10)


def disk_mask(rho):
    return (HEIGHT < 0.5) & (rho > 0.135) & (RADIUS < 4.5)


def ramp(*colours):
    return LinearSegmentedColormap.from_list("ramp", list(colours))


DENSITY_MAP = ramp("#fcfcfb", "#a9c6e4", "#2f6fb3", "#0d2a4d", "#050f1d")
TEMPERATURE_MAP = ramp("#fcfcfb", "#e8b98c", "#7a2c08", "#2b0f02")
FRACTION_MAP = ramp("#fcfcfb", "#b9d9cd", "#2a8a68", "#0a3a2a")
ION_MAP = ramp("#fcfcfb", "#d9c6e8", "#7b4fa3", "#2c1340")
FIELD_MAP = ramp("#fcfcfb", "#f1d38a", "#b07a12", "#3d2804")


def save(figure, name):
    path = os.path.join(OUT, name)
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    print("written", path)


# --------------------------------------------------------------------------------------------- 1. maps over time
MAP_TIMES = (10.0, 25.0, 50.0, 75.0)
for quantity, colour_map, norm, label in (
    ("density", DENSITY_MAP, LogNorm(1e-4, 1e2), r"density [m$_p$ cm$^{-3}$]"),
    ("temperature", TEMPERATURE_MAP, LogNorm(3e2, 3e7), r"P / $\rho$ as T / $\mu$ [K]"),
):
    figure, axes = plt.subplots(3, len(MAP_TIMES), figsize=(2.25 * len(MAP_TIMES) + 1.0, 3 * 4.3), sharex=True, sharey=True,
                                gridspec_kw=dict(wspace=0.04, hspace=0.12))
    for row, run in enumerate(("B", "field", "draine")):
        for column, time in enumerate(MAP_TIMES):
            rho, t_over_mu, _, data = load(run, index_at(run, time))
            image = axes[row, column].imshow((rho if quantity == "density" else t_over_mu).T, origin="lower", extent=EXTENT,
                                             cmap=colour_map, norm=norm, interpolation="nearest")
            if row == 0:
                axes[row, column].set_title(f"{float(data['time_myr']):.0f} Myr", fontsize=10)
            if column == 0:
                axes[row, column].set_ylabel(f"{SHORT[run]}\nz [kpc]", fontsize=9)
            if row == 2:
                axes[row, column].set_xlabel("x [kpc]")
    figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02).set_label(label)
    figure.suptitle(f"Edge-on {quantity}, plane y = 0, 256 x 256 x 512 cells", x=0.45, y=0.915, fontsize=11)
    save(figure, f"maps_{quantity}.png")

# --------------------------------------------------------------------------------------------- 2. chemistry of the wind (full domain)
for time in (25.0, 75.0):
    columns = (
        ("temperature [K]", lambda d, T: T, TEMPERATURE_MAP, LogNorm(1e2, 3e7)),
        ("neutral H: n(H) / n$_H$", lambda d, T: d["x_H_xz"], ION_MAP, LogNorm(1e-7, 1)),
        ("He not fully ionised: 1 - n(He$^{++}$) / n$_{He}$", lambda d, T: np.clip(1 - d["x_Hepp_xz"] / HE_PER_H, 1e-7, 1), ION_MAP, LogNorm(1e-6, 1)),
        ("neutral O: n(O) / n$_O$", lambda d, T: d["x_O_xz"] / 3.2e-4, ION_MAP, LogNorm(1e-6, 1)),
    )
    figure, axes = plt.subplots(2, len(columns), figsize=(2.3 * len(columns), 2 * 4.6), sharex=True, sharey=True,
                                gridspec_kw=dict(wspace=0.05, hspace=0.25))
    for row, run in enumerate(CHEM):
        _, _, temperature, data = load(run, index_at(run, time))
        for column, (label, value, colour_map, norm) in enumerate(columns):
            image = axes[row, column].imshow(np.asarray(value(data, temperature), dtype=float).T, origin="lower", extent=EXTENT,
                                             cmap=colour_map, norm=norm, interpolation="nearest")
            if row == 1:
                cbar = figure.colorbar(image, ax=axes[:, column], orientation="horizontal", fraction=0.035, pad=0.06)
                cbar.set_label(label, fontsize=8.5)
                axes[row, column].set_xlabel("x [kpc]")
            if column == 0:
                axes[row, column].set_ylabel(f"{SHORT[run]}\nz [kpc]", fontsize=9)
    figure.suptitle(f"Chemical state of the wind at {time:.0f} Myr (edge-on plane)", y=0.93, fontsize=11)
    save(figure, f"chemistry_wind_maps_{time:.0f}myr.png")

# --------------------------------------------------------------------------------------------- 3. chemistry of the disk (zoom)
for time in (25.0, 75.0):
    rows = (
        (r"n$_H$ [cm$^{-3}$]", lambda d, T: d["n_H_xz"], DENSITY_MAP, LogNorm(1e-2, 1e3)),
        ("temperature [K]", lambda d, T: T, TEMPERATURE_MAP, LogNorm(3e1, 3e6)),
        ("2 n(H$_2$) / n$_H$", lambda d, T: 2 * d["x_H2_xz"], FRACTION_MAP, LogNorm(1e-5, 1)),
        ("n(H$^+$) / n$_H$", lambda d, T: d["x_Hp_xz"], ION_MAP, LogNorm(1e-5, 1)),
        ("electrons per H nucleus", lambda d, T: d["x_E_xz"], ION_MAP, LogNorm(1e-4, 1.3)),
        ("FUV field J / J$_0$ (Draine)", lambda d, T: d["fuv_xz"] if "fuv_xz" in d.files else np.ones_like(T), FIELD_MAP, LogNorm(1e-2, 3e3)),
    )
    figure, axes = plt.subplots(len(rows), 2, figsize=(13, 1.55 * len(rows) + 0.6), sharex=True, sharey=True,
                                gridspec_kw=dict(wspace=0.04, hspace=0.12))
    for column, run in enumerate(CHEM):
        _, _, temperature, data = load(run, index_at(run, time))
        axes[0, column].set_title(SHORT[run], fontsize=10)
        for row, (label, value, colour_map, norm) in enumerate(rows):
            image = axes[row, column].imshow(np.asarray(value(data, temperature), dtype=float).T, origin="lower", extent=EXTENT,
                                             cmap=colour_map, norm=norm, interpolation="nearest", aspect="auto")
            axes[row, column].set_ylim(-1, 1)
            if column == 1:
                figure.colorbar(image, ax=axes[row, :], fraction=0.02, pad=0.01).set_label(label, fontsize=8.5)
            if column == 0:
                axes[row, column].set_ylabel("z [kpc]")
        axes[-1, column].set_xlabel("x [kpc]")
    figure.suptitle(f"The disk at {time:.0f} Myr (|z| < 1 kpc; vertical scale stretched 2.5x)", y=0.95, fontsize=11)
    save(figure, f"chemistry_disk_maps_{time:.0f}myr.png")


# --------------------------------------------------------------------------------------------- 4. outflow rates
def outflow_history(run):
    times, rates = [], {1.0: [], 2.0: [], 5.0: []}
    for path in frame_files(run):
        data = np.load(path)
        mdot = np.asarray(data["mdot_z"], dtype=float)
        times.append(float(data["time_myr"]))
        for height in rates:
            top, bottom = np.argmin(np.abs(z - height)), np.argmin(np.abs(z + height))
            rates[height].append(mdot[top] - mdot[bottom])
    return np.array(times), {h: np.array(r) for h, r in rates.items()}


histories = {run: outflow_history(run) for run in ("A", "B", "field", "draine")}
figure, axes = plt.subplots(1, 4, figsize=(16, 3.8), gridspec_kw=dict(wspace=0.28))
for run in ("A", "B", "field", "draine"):
    times, rates = histories[run]
    style = dict(color=RUNS[run]["colour"], lw=1.2 if run == "A" else 1.8, ls="--" if run == "A" else "-", label=SHORT[run])
    for axis, height in zip(axes[:3], (1.0, 2.0, 5.0)):
        axis.plot(times, rates[height], **style)
    steps = np.diff(times, prepend=0.0) * 1e6
    axes[3].plot(times, np.cumsum(rates[5.0] * steps) / 1e6, **style)
for axis, height in zip(axes[:3], (1.0, 2.0, 5.0)):
    axis.set_title(f"through |z| = {height:g} kpc (both sides)", fontsize=10, loc="left")
    axis.set_ylabel(r"net outflow [M$_\odot$ yr$^{-1}$]")
axes[3].set_title("mass through |z| = 5 kpc, cumulative", fontsize=10, loc="left")
axes[3].set_ylabel(r"[10$^6$ M$_\odot$]")
for axis in axes:
    axis.set_xlabel("time [Myr]")
    axis.grid(color=GRID, lw=0.6)
    axis.axhline(0, color=MUTED, lw=0.6)
    axis.set_xlim(0, 75)
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
save(figure, "outflow_rates.png")
print("mean outflow 40-75 Myr [Msun/yr], |z| = 1 / 2 / 5 kpc:")
for run in ("A", "B", "field", "draine"):
    times, rates = histories[run]
    late = times > 40.0
    print(f"   {SHORT[run]:32s}" + " ".join(f"{rates[h][late].mean():7.2f}" for h in (1.0, 2.0, 5.0)))


# --------------------------------------------------------------------------------------------- 5. disk history
def weighted(value, weight):
    return (value * weight).sum() / weight.sum()


def disk_history(run):
    out = {k: [] for k in ("time", "t_over_mu", "cold", "mass", "h2", "hplus", "field", "cold_true")}
    for index in range(len(frame_files(run))):
        rho, t_over_mu, temperature, data = load(run, index)
        mask = disk_mask(rho)
        weight = (rho * RADIUS)[mask]
        out["time"].append(float(data["time_myr"]))
        out["t_over_mu"].append(np.exp(weighted(np.log(t_over_mu[mask]), weight)))
        out["cold"].append(100 * weight[t_over_mu[mask] < 1e3].sum() / weight.sum())
        out["mass"].append(weight.sum())
        if run in CHEM:
            out["h2"].append(100 * weighted(2 * np.asarray(data["x_H2_xz"], float)[mask], weight))
            out["hplus"].append(100 * weighted(np.asarray(data["x_Hp_xz"], float)[mask], weight))
            out["field"].append(weighted(np.asarray(data["fuv_xz"], float)[mask], weight) if "fuv_xz" in data.files else 1.0)
            out["cold_true"].append(100 * weight[temperature[mask] < 1e3].sum() / weight.sum())
    return {k: np.array(v) for k, v in out.items()}


disk = {run: disk_history(run) for run in ("B", "field", "draine")}
panels = (
    ("t_over_mu", r"mass-weighted T / $\mu$ [K]", ("B", "field", "draine"), "log"),
    ("cold", r"mass below T / $\mu$ = 1000 K [%]", ("B", "field", "draine"), "linear"),
    ("mass", "disk gas mass in the plane / initial\n(one plane: noisy)", ("B", "field", "draine"), "linear"),
    ("h2", r"hydrogen in H$_2$ [% of the disk H]", CHEM, "log"),
    ("hplus", r"hydrogen ionised [% of the disk H]", CHEM, "log"),
    ("field", r"mass-weighted J / J$_0$ in the disk", CHEM, "log"),
)
figure, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True, gridspec_kw=dict(wspace=0.3, hspace=0.12))
for axis, (key, label, runs, scale) in zip(axes.flat, panels):
    for run in runs:
        value = disk[run][key] / disk[run]["mass"][0] if key == "mass" else disk[run][key]
        axis.plot(disk[run]["time"], value, color=RUNS[run]["colour"], lw=1.8, label=SHORT[run])
    axis.set_ylabel(label)
    axis.set_yscale(scale)
    axis.grid(color=GRID, lw=0.6)
    axis.set_xlim(0, 75)
for axis in axes[1]:
    axis.set_xlabel("time [Myr]")
axes[0, 0].legend(fontsize=8, frameon=False)
axes[0, 1].set_ylim(0, 100)
figure.suptitle("Disk gas (|z| < 0.5 kpc, R < 4.5 kpc), edge-on plane weighted with |x|", y=0.93, fontsize=11)
save(figure, "disk_history.png")
print("disk, mean over 40-75 Myr: T/mu [K], below 1000 K [%], mass/initial at 75 Myr, H in H2 [%], H ionised [%], J/J0")
for run in ("B", "field", "draine"):
    late = disk[run]["time"] > 40.0
    extra = "" if run == "B" else (f"  H2 {disk[run]['h2'][late].mean():6.2f} %  H+ {disk[run]['hplus'][late].mean():6.2f} %"
                                   f"  J/J0 {disk[run]['field'][late].mean():6.1f}  (true T < 1000 K: {disk[run]['cold_true'][late].mean():4.1f} %)")
    print(f"   {SHORT[run]:32s} T/mu {np.exp(np.log(disk[run]['t_over_mu'][late]).mean()):6.0f}  cold {disk[run]['cold'][late].mean():5.1f} %"
          f"  mass {disk[run]['mass'][-1] / disk[run]['mass'][0]:5.2f}" + extra)

# --------------------------------------------------------------------------------------------- 6. phase diagrams
PHASE_TIMES = (25.0, 75.0)
figure, axes = plt.subplots(len(PHASE_TIMES), 3, figsize=(14, 4.3 * len(PHASE_TIMES)), sharex=True, sharey=True,
                            gridspec_kw=dict(wspace=0.05, hspace=0.18))
density_bins, temperature_bins = np.logspace(-5, 3.5, 120), np.logspace(1, 9, 120)
for row, time in enumerate(PHASE_TIMES):
    for column, run in enumerate(("B", "field", "draine")):
        rho, t_over_mu, _, data = load(run, index_at(run, time))
        weight = rho * RADIUS
        histogram, _, _ = np.histogram2d(rho.ravel(), t_over_mu.ravel(), bins=(density_bins, temperature_bins), weights=weight.ravel())
        histogram /= histogram.sum()
        image = axes[row, column].pcolormesh(density_bins, temperature_bins, np.where(histogram > 0, histogram, np.nan).T,
                                             cmap=ramp("#d9e6f2", "#2f6fb3", "#0d2a4d"), norm=LogNorm(1e-6, 1e-1), shading="flat")
        axes[row, column].set_xscale("log")
        axes[row, column].set_yscale("log")
        axes[row, column].grid(color=GRID, lw=0.5)
        axes[row, column].set_title(f"{SHORT[run]}, {float(data['time_myr']):.0f} Myr", fontsize=10, loc="left")
        if row == len(PHASE_TIMES) - 1:
            axes[row, column].set_xlabel(r"density [m$_p$ cm$^{-3}$]")
    axes[row, 0].set_ylabel(r"P / $\rho$ as T / $\mu$ [K]")
figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02).set_label("mass fraction per bin (edge-on plane, weighted with |x|)")
save(figure, "phase_diagrams.png")


# --------------------------------------------------------------------------------------------- 7. ionisation against temperature
def cie_hydrogen(temperature):
    """n(H+) / n_H in collisional ionisation equilibrium (rates of Cen 1992 as tabulated by Katz, Weinberg & Hernquist 1996)."""
    t = temperature
    ionisation = 5.85e-11 * np.sqrt(t) * np.exp(-157809.1 / t) / (1 + np.sqrt(t / 1e5))
    recombination = 8.4e-11 / np.sqrt(t) * (t / 1e3) ** -0.2 / (1 + (t / 1e6) ** 0.7)
    return ionisation / (ionisation + recombination)


def cie_helium(temperature):
    """n(He), n(He+), n(He++) / n_He in collisional ionisation equilibrium (same rate set)."""
    t = temperature
    ion0 = 2.38e-11 * np.sqrt(t) * np.exp(-285335.4 / t) / (1 + np.sqrt(t / 1e5))
    ion1 = 5.68e-12 * np.sqrt(t) * np.exp(-631515.0 / t) / (1 + np.sqrt(t / 1e5))
    rec1 = 1.5e-10 * t**-0.6353 + 1.9e-3 * t**-1.5 * np.exp(-470000.0 / t) * (1 + 0.3 * np.exp(-94000.0 / t))
    rec2 = 3.36e-10 / np.sqrt(t) * (t / 1e3) ** -0.2 / (1 + (t / 1e6) ** 0.7)
    ratio1, ratio2 = ion0 / rec1, ion1 / rec2
    total = 1 + ratio1 + ratio1 * ratio2
    return 1 / total, ratio1 / total, ratio1 * ratio2 / total


late_indices = range(index_at("field", 50.0), index_at("field", 75.0) + 1)
figure, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True, gridspec_kw=dict(wspace=0.08, hspace=0.12), sharey="row")
temperature_bins = np.logspace(2, 8.5, 130)
fraction_bins = np.logspace(-6, 0.05, 120)
grid = np.logspace(2, 8.5, 400)
for column, run in enumerate(CHEM):
    stacks = {"H": [], "He": [], "T": [], "w": []}
    for index in late_indices:
        rho, _, temperature, data = load(run, index)
        outside = HEIGHT > 0.5  # the wind and the halo, not the disk
        stacks["T"].append(temperature[outside])
        stacks["H"].append(np.asarray(data["x_Hp_xz"], float)[outside])
        stacks["He"].append(np.asarray(data["x_Hepp_xz"], float)[outside] / HE_PER_H)
        stacks["w"].append((rho * RADIUS)[outside])
    temperature, weight = np.concatenate(stacks["T"]), np.concatenate(stacks["w"])
    for row, (key, label, cie) in enumerate((("H", r"n(H$^+$) / n$_H$", cie_hydrogen(grid)),
                                             ("He", r"n(He$^{++}$) / n$_{He}$", cie_helium(grid)[2]))):
        histogram, _, _ = np.histogram2d(temperature, np.clip(np.concatenate(stacks[key]), 1.1e-6, 1.0), bins=(temperature_bins, fraction_bins), weights=weight)
        histogram /= histogram.sum()
        axis = axes[row, column]
        image = axis.pcolormesh(temperature_bins, fraction_bins, np.where(histogram > 0, histogram, np.nan).T,
                                cmap=ramp("#e8dcf1", "#7b4fa3", "#2c1340"), norm=LogNorm(1e-6, 1e-1), shading="flat")
        axis.plot(grid, cie, color="#d9772b", lw=2, label="collisional ionisation equilibrium")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(color=GRID, lw=0.5)
        if row == 0:
            axis.set_title(f"{SHORT[run]}", fontsize=10, loc="left")
        else:
            axis.set_xlabel("temperature [K]")
        if column == 0:
            axis.set_ylabel(label)
axes[0, 0].legend(fontsize=8.5, frameon=False, loc="lower right")
figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02).set_label("mass fraction per bin")
figure.suptitle("Ionisation of the gas outside the disk (|z| > 0.5 kpc), all frames 50-75 Myr\n"
                "equilibrium curve: collisional only, rates of Cen (1992)", y=0.96, fontsize=10.5)
save(figure, "ionisation_vs_temperature.png")

# --------------------------------------------------------------------------------------------- 8. vertical structure of the wind
height_bins = np.linspace(0.5, 10.0, 39)
centres = 0.5 * (height_bins[1:] + height_bins[:-1])
profiles = {}
for run in ("B", "field", "draine"):
    sums = {k: np.zeros(len(centres)) for k in ("w", "rho", "logT", "hot", "warm", "cool", "hplus", "h2")}
    for index in range(index_at(run, 50.0), index_at(run, 75.0) + 1):
        rho, t_over_mu, temperature, data = load(run, index)
        weight = rho * RADIUS
        which = np.digitize(HEIGHT, height_bins) - 1
        inside = (which >= 0) & (which < len(centres))
        def add(key, value):
            sums[key] += np.bincount(which[inside], weights=(weight * value)[inside], minlength=len(centres))
        add("w", 1.0)
        sums["rho"] += np.bincount(which[inside], weights=(rho * RADIUS)[inside], minlength=len(centres)) / np.bincount(
            which[inside], weights=RADIUS[inside], minlength=len(centres))
        add("logT", np.log10(temperature))
        add("cool", temperature < 2e4)
        add("warm", (temperature >= 2e4) & (temperature < 5e5))
        add("hot", temperature >= 5e5)
        if run in CHEM:
            add("hplus", np.asarray(data["x_H_xz"], float))
            add("h2", 2 * np.asarray(data["x_H2_xz"], float))
    frames = index_at(run, 75.0) - index_at(run, 50.0) + 1
    profiles[run] = {k: (v / frames if k == "rho" else v / sums["w"]) for k, v in sums.items() if k != "w"}
figure, axes = plt.subplots(1, 5, figsize=(19, 3.9), gridspec_kw=dict(wspace=0.32))
for run in ("B", "field", "draine"):
    style = dict(color=RUNS[run]["colour"], lw=1.8, label=SHORT[run])
    axes[0].plot(centres, profiles[run]["rho"], **style)
    axes[1].plot(centres, 10 ** profiles[run]["logT"], **style)
    axes[2].plot(centres, 100 * profiles[run]["cool"], **style)
    axes[2].plot(centres, 100 * profiles[run]["hot"], color=RUNS[run]["colour"], lw=1.8, ls=":")
    if run in CHEM:
        axes[3].plot(centres, profiles[run]["hplus"], **style)
        axes[4].plot(centres, profiles[run]["h2"], **style)
labels = (r"mean density [m$_p$ cm$^{-3}$]", "mass-weighted T [K]", "mass in cool (T < 2e4 K, solid)\nand hot (T > 5e5 K, dotted) gas [%]",
          r"neutral H: n(H) / n$_H$, mass-weighted", r"2 n(H$_2$) / n$_H$, mass-weighted")
for axis, label, scale in zip(axes, labels, ("log", "log", "linear", "log", "log")):
    axis.set_ylabel(label, fontsize=9)
    axis.set_yscale(scale)
    axis.set_xlabel("height |z| [kpc]")
    axis.grid(color=GRID, lw=0.6)
axes[2].set_ylim(0, 100)
axes[0].legend(fontsize=8, frameon=False)
figure.suptitle("Vertical structure of the outflow, averaged over 50-75 Myr (edge-on plane, weighted with |x|; model B: T for mu = 0.6)",
                y=1.0, fontsize=10.5)
save(figure, "wind_vertical_profiles.png")


# --------------------------------------------------------------------------------------------- 9. health and cost
def diag_times(log):
    times = []
    with open(log) as handle:
        for line in handle:
            if line.startswith("[diag] t="):
                times.append(float(line.split()[1][2:]))
    return np.array(times) * 9.7779  # code time unit -> Myr


figure, axes = plt.subplots(1, 2, figsize=(13, 4.0), gridspec_kw=dict(wspace=0.25))
elements = ("C", "H", "He", "O", "Si")
markers = dict(zip(elements, ("o", "s", "^", "D", "v")))
for run in CHEM:
    times, errors = [], []
    for path in frame_files(run):
        data = np.load(path)
        times.append(float(data["time_myr"]))
        errors.append(np.asarray(data["element_error"], float))
    errors = np.array(errors)
    for k, element in enumerate(elements):
        axes[0].plot(times, np.maximum(errors[:, k], 1e-17), color=RUNS[run]["colour"], lw=0.8, marker=markers[element], ms=3,
                     label=f"{element}" if run == "field" else None)
axes[0].set_yscale("log")
axes[0].set_ylim(1e-17, 1e-10)
axes[0].axhline(2.2e-16, color=MUTED, lw=0.8, ls="--")
axes[0].text(1, 3e-16, "double-precision rounding", fontsize=8, color=MUTED)
axes[0].set_title("largest error of the atoms per unit mass over the whole grid\n(orange: field of every cell, grey: one Draine field)",
                  fontsize=9.5, loc="left")
axes[0].set_xlabel("time [Myr]")
axes[0].legend(fontsize=8, frameon=False, ncol=5, loc="upper left")
for run, log in (("field", f"{HERE}/cgols_logs/run_chem256fuv_v2.out"), ("draine", f"{HERE}/cgols_logs/run_chem256_v2.out")):
    times = diag_times(log)
    edges = np.arange(0, 76, 2.5)
    counts, _ = np.histogram(times, bins=edges)
    axes[1].step(edges[:-1], counts / 2.5, where="post", color=RUNS[run]["colour"], lw=1.8, label=f"{SHORT[run]}: {len(times)} steps")
axes[1].set_ylabel("hydro steps per Myr")
axes[1].set_xlabel("time [Myr]")
axes[1].set_title("time steps (wall time 13.2 h and 13.4 h on one H200)", fontsize=9.5, loc="left")
axes[1].legend(fontsize=8, frameon=False)
for axis in axes:
    axis.grid(color=GRID, lw=0.6)
    axis.set_xlim(0, 75)
save(figure, "health_and_cost.png")
