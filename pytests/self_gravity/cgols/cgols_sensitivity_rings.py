"""What are the rings in the sensitivity maps? (asgard-papers #16)

The forward-mode sensitivity maps of model A (cgols_sensitivity.py) show concentric rings around the starburst.
This script measures their radial wavelength and tests where they come from, from the saved maps only (no run):

* wavelength at 128^3 against 256^3, same time (30 Myr, float64): a wavelength fixed in kpc points to a physical
  wave, one proportional to the cell size to a numerical pattern;
* the full central finite-difference maps of the 128^3 check against the tangent: if they show the same rings,
  the rings are in the response of the discrete solution, not an error of the tangent;
* the wavelength at 8, 30 and 45 Myr (128^3, float64, Sigma only);
* a beam: d(B * S)/dp = B * dS/dp, so the observable log-derivative at a resolution B is
  (B * dS/dp) / (B * S); how much ring power survives and whether the halo statistics change.

Radial spectra: rays from the centre at 20-75 deg from the minor axis (both sides, both hemispheres), r = 1.5-5 kpc,
only pixels brighter than the median emitting pixel (as in the maps), linear interpolation at half-cell steps, a
0.3 kpc running mean removed, Hann window, mean power over the rays.

CPU only, a few seconds:

    python cgols_sensitivity_rings.py   # figures/cgols/studies/sensitivity_rings_A.png + data/studies/sensitivity_rings_A.npz
                                        # + the 256^3 X-ray sensitivity map in a 312 pc beam (..._beam312.png)
"""

# ruff: noqa: C408 (keyword dicts for plot styles)

# general
import os

# plotting
import matplotlib

# numerics
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIES = os.path.join(SCRIPT_DIR, "data", "studies")
FIG_PATH = os.path.join(SCRIPT_DIR, "figures", "cgols", "studies", "sensitivity_rings_A.png")
OUT_PATH = os.path.join(STUDIES, "sensitivity_rings_A.npz")
BEAM_FIG_PATH = os.path.join(SCRIPT_DIR, "figures", "cgols", "studies",
                             "sensitivity_xray_atomdb_d256_t30myr_x64_xray_beam312.png")  # fmt: skip
MAP_BEAM_PC = 312.0
TITLES = os.environ.get("CGOLS_FIGURE_TITLES", "1") != "0"

R_RANGE_KPC = (1.5, 5.0)
THETA_DEG = (20.0, 75.0)
HIGH_PASS_KPC = 0.3
RING_BAND_CELLS = (5.0, 10.0)
BEAMS_PC = (0.0, 156.0, 312.0)  # FWHM; 156 pc is the observable resolution named in #16, 312 pc twice that
FD_EPS = 0.001


def load(name):
    return np.load(os.path.join(STUDIES, name))


def emitting(s0):
    """Pixels brighter than the median emitting pixel (the blanking of the maps)."""
    return s0 > np.nanpercentile(s0[s0 > 0], 50)


def log_derivative(tangent, s0, mask):
    with np.errstate(all="ignore"):
        return np.where(mask, tangent / s0, 0.0)


def radial_spectrum(field, x, z, mask):
    """Wavelengths [kpc] and mean radial power spectrum over the rays, high-passed (see module docstring)."""
    dx = x[1] - x[0]
    dr = dx / 2
    r = np.arange(*R_RANGE_KPC, dr)
    spectra = []
    for theta in np.radians(np.arange(*THETA_DEG, 1.0)):
        for sx in (1, -1):
            for sz in (1, -1):
                coords = [(sx * r * np.sin(theta) - x[0]) / dx, (sz * r * np.cos(theta) - z[0]) / (z[1] - z[0])]
                if map_coordinates(mask.astype(float), coords, order=0).mean() < 0.9:
                    continue
                profile = map_coordinates(field, coords, order=1, mode="nearest")
                high = (profile - gaussian_filter1d(profile, HIGH_PASS_KPC / dr)) * np.hanning(len(profile))
                spectra.append(np.abs(np.fft.rfft(high)) ** 2)
    k = np.fft.rfftfreq(len(r), dr)[1:]
    return 1.0 / k, np.mean(spectra, axis=0)[1:], len(spectra)


def peak_wavelength(lam, power, lo=0.1, hi=1.0):
    sel = (lam > lo) & (lam < hi)
    return float(lam[sel][np.argmax((power / lam)[sel])])


def band_share(lam, power, dx):
    band = (lam > RING_BAND_CELLS[0] * dx) & (lam < RING_BAND_CELLS[1] * dx)
    return float(power[band].sum() / power.sum())


def main():
    out = {}
    maps = {res: load(f"cgols_sensitivity_d{res}_t30myr_x64_xray.npz") for res in (128, 256)}
    spectra = {}
    for res, d in maps.items():
        x, z = d["x"], d["z"]
        dx = x[1] - x[0]
        mask = emitting(d["sx_atomdb0"])
        for k in ("E", "M"):
            f = log_derivative(d[f"dsx_atomdb_d{k}"], d["sx_atomdb0"], mask)
            lam, power, rays = radial_spectrum(f, x, z, mask)
            spectra[(res, k)] = (lam, power, dx)
            out[f"peak_kpc__sx_d{k}__{res}"] = peak_wavelength(lam, power)
            out[f"peak_cells__sx_d{k}__{res}"] = out[f"peak_kpc__sx_d{k}__{res}"] / dx
            out[f"band_share__sx_d{k}__{res}"] = band_share(lam, power, dx)
            print(f"{res}^3 dlnS_X/dln{k}: peak {1e3 * out[f'peak_kpc__sx_d{k}__{res}']:.0f} pc = "
                  f"{out[f'peak_cells__sx_d{k}__{res}']:.1f} cells, ring-band power share "
                  f"{out[f'band_share__sx_d{k}__{res}']:.2f} ({rays} rays)")  # fmt: skip

    # finite differences against the tangent, 128^3, d / dln Mdot
    d128 = maps[128]
    x, z = d128["x"], d128["z"]
    dx = x[1] - x[0]
    fd = load("cgols_sensitivity_d128_t30myr_x64_xray_fdcheck_eps0.002-0.001.npz")
    fd_maps = {}
    for name, base in (("sigma", "sigma0"), ("sx_atomdb", "sx_atomdb0")):
        s0 = d128[base]
        mask = emitting(s0)
        ad = log_derivative(fd[f"fdmap__{name}__ad"], s0, mask)
        diff = (fd[f"fdmap__{name}__plus_{FD_EPS}"] - fd[f"fdmap__{name}__minus_{FD_EPS}"]) / (2 * FD_EPS)
        fdm = log_derivative(diff, s0, mask)
        fd_maps[name] = (ad, fdm, mask)
        lam, p_ad, _ = radial_spectrum(ad, x, z, mask)
        _, p_fd, _ = radial_spectrum(fdm, x, z, mask)
        band = (lam > RING_BAND_CELLS[0] * dx) & (lam < RING_BAND_CELLS[1] * dx)
        out[f"fd_over_ad_band_power__{name}"] = float(p_fd[band].sum() / p_ad[band].sum())
        out[f"fd_peak_kpc__{name}"] = peak_wavelength(lam, p_fd)
        out[f"ad_peak_kpc__{name}"] = peak_wavelength(lam, p_ad)
        print(f"128^3 {name} d/dln Mdot: FD (eps {FD_EPS}) / tangent ring-band power {out[f'fd_over_ad_band_power__{name}']:.2f}, "
              f"peaks FD {1e3 * out[f'fd_peak_kpc__{name}']:.0f} / tangent {1e3 * out[f'ad_peak_kpc__{name}']:.0f} pc")  # fmt: skip

    # time dependence, 128^3, Sigma
    times = (8, 30, 45)
    for t in times:
        d = load(f"cgols_sensitivity_d128_t{t}myr_x64.npz")
        mask = emitting(d["sigma0"])
        lam, power, _ = radial_spectrum(log_derivative(d["dsig_dE"], d["sigma0"], mask), d["x"], d["z"], mask)
        out[f"peak_cells__sigma_dE__128__t{t}"] = peak_wavelength(lam, power) / (d["x"][1] - d["x"][0])
        print(f"128^3 Sigma dln/dlnE at {t} Myr: peak {out[f'peak_cells__sigma_dE__128__t{t}']:.1f} cells")

    # beams, 256^3
    d = maps[256]
    x, z = d["x"], d["z"]
    dx = x[1] - x[0]
    zz = np.broadcast_to(np.abs(z)[None, :], d["sx_atomdb0"].shape)
    floor = np.nanpercentile(d["sx_atomdb0"][d["sx_atomdb0"] > 0], 50)
    beamed = {}
    for fwhm in BEAMS_PC:
        sigma = fwhm / 2354.8 / dx
        smooth = (lambda a, s=sigma: gaussian_filter(a, s)) if sigma else (lambda a: a)
        s = smooth(d["sx_atomdb0"])
        mask = s > floor
        f = log_derivative(smooth(d["dsx_atomdb_dE"]), s, mask)
        beamed[fwhm] = f
        lam, power, _ = radial_spectrum(f, x, z, mask)
        halo = mask & (zz > 1.0)
        out[f"beam{fwhm:.0f}__band_share"] = band_share(lam, power, dx)
        out[f"beam{fwhm:.0f}__halo_positive_fraction"] = float((f[halo] > 0).mean())
        out[f"beam{fwhm:.0f}__halo_median"] = float(np.median(f[halo]))
        print(f"256^3 beam {fwhm:.0f} pc: ring-band share {out[f'beam{fwhm:.0f}__band_share']:.2f}, halo "
              f"dlnS/dlnE > 0 in {out[f'beam{fwhm:.0f}__halo_positive_fraction']:.0%}, median "
              f"{out[f'beam{fwhm:.0f}__halo_median']:+.2f}")  # fmt: skip
    np.savez(OUT_PATH, **out)
    print(f"wrote {OUT_PATH}")

    # figure
    fig = plt.figure(figsize=(17, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 5)
    zoom = dict(xlim=(0, 5), ylim=(0, 5))
    kw = dict(origin="lower", cmap="RdBu_r", vmin=-15, vmax=15, aspect="equal")
    panels = [
        (log_derivative(maps[128]["dsx_atomdb_dE"], maps[128]["sx_atomdb0"], emitting(maps[128]["sx_atomdb0"])),
         maps[128], r"$\partial\ln S_X/\partial\ln\dot E$, 128$^3$ (78 pc)"),
        (beamed[0.0], maps[256], r"$\partial\ln S_X/\partial\ln\dot E$, 256$^3$ (39 pc)"),
        (beamed[312.0], maps[256], r"256$^3$, beam FWHM 312 pc"),
        (fd_maps["sx_atomdb"][0], d128, r"$\partial\ln S_X/\partial\ln\dot M$, 128$^3$, tangent"),
        (fd_maps["sx_atomdb"][1], d128, rf"128$^3$, central difference, $\epsilon$ = {FD_EPS:g}"),
    ]
    for col, (field, d, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, col])
        image = ax.imshow(np.where(field == 0, np.nan, field).T, extent=tuple(d["extent_xz"]), **kw)
        ax.set(xlabel="x [kpc]", ylabel="z [kpc]" if col == 0 else None, **zoom)
        ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=fig.axes[:5], location="bottom", shrink=0.4, pad=0.02, label="dimensionless")

    ax_kpc, ax_cells, ax_time = (fig.add_subplot(grid[1, c]) for c in (slice(0, 2), slice(2, 4), 4))
    for (res, k), (lam, power, dx) in spectra.items():
        style = dict(color="tab:blue" if res == 128 else "tab:red", ls="-" if k == "E" else "--",
                     label=rf"{res}$^3$, $\partial/\partial\ln\dot{{{'E' if k == 'E' else 'M'}}}$")  # fmt: skip
        ax_kpc.loglog(1e3 * lam, power / lam, **style)
        ax_cells.loglog(lam / dx, power / lam, **style)
    ax_kpc.set(xlabel="radial wavelength [pc]", ylabel="power / wavelength (arbitrary)")
    ax_cells.set(xlabel="radial wavelength [cells]")
    for ax in (ax_kpc, ax_cells):
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    ax_cells.axvspan(*RING_BAND_CELLS, color="0.85", zorder=0)
    ax_time.plot(times, [out[f"peak_cells__sigma_dE__128__t{t}"] for t in times], "o-", color="tab:blue")
    ax_time.set(xlabel="t [Myr]", ylabel="peak wavelength [cells]", ylim=(0, 15),
                title=r"128$^3$, $\partial\ln\Sigma/\partial\ln\dot E$")  # fmt: skip
    ax_time.grid(alpha=0.3)
    if TITLES:
        fig.suptitle("Rings in the sensitivity maps, model A, 30 Myr, float64: wavelength in kpc and in cells", fontsize=13)
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=130)
    print(f"wrote {FIG_PATH}")

    plot_beamed_map(maps[256], MAP_BEAM_PC)


def plot_beamed_map(d, fwhm_pc):
    """The 1x3 X-ray map of cgols_sensitivity.plot_xray_maps (S_X, dln S_X/dln Mdot, dln S_X/dln Edot) in a beam."""
    from matplotlib.colors import LogNorm

    dx = d["x"][1] - d["x"][0]
    sigma = fwhm_pc / 2354.8 / dx
    s0 = gaussian_filter(d["sx_atomdb0"], sigma)
    floor = np.nanpercentile(d["sx_atomdb0"][d["sx_atomdb0"] > 0], 50)  # same blanking level as the native map
    safe = np.where(s0 > floor, s0, np.nan)
    rel = {k: gaussian_filter(d[f"dsx_atomdb_d{k}"], sigma) / safe for k in ("M", "E")}
    lim = max(float(np.nanpercentile(np.abs(r), 99)) for r in rel.values())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.4), constrained_layout=True)
    panels = (
        (np.clip(s0, 1e-30, None), dict(cmap="magma", norm=LogNorm(vmin=floor * 1e-2, vmax=np.nanmax(s0))),
         r"soft X-ray surface brightness $S_X$", r"$S_X$, absorbed 0.3-1 keV [photons s$^{-1}$ cm$^{-2}$ sr$^{-1}$]"),
        (rel["M"], dict(cmap="RdBu_r", vmin=-lim, vmax=lim), r"$\partial\ln S_X/\partial\ln\dot M$", "dimensionless"),
        (rel["E"], dict(cmap="RdBu_r", vmin=-lim, vmax=lim), r"$\partial\ln S_X/\partial\ln\dot E$", "dimensionless"),
    )  # fmt: skip
    for ax, (data, kw, title, label) in zip(axes, panels):
        image = ax.imshow(data.T, origin="lower", extent=tuple(d["extent_xz"]), aspect="equal", **kw)
        ax.set_title(f"{title}, beam {fwhm_pc:g} pc", fontsize=10)
        ax.set(xlabel="x [kpc]", ylabel="z [kpc]")
        fig.colorbar(image, ax=ax, shrink=0.8, label=label)
    if TITLES:
        fig.suptitle(f"X-ray sensitivity to the feedback loading, model A, 256$^3$, 30 Myr, Gaussian beam FWHM "
                     f"{fwhm_pc:g} pc", fontsize=12)  # fmt: skip
    fig.savefig(BEAM_FIG_PATH, dpi=180)
    plt.close(fig)
    print(f"wrote {BEAM_FIG_PATH}")


if __name__ == "__main__":
    main()
