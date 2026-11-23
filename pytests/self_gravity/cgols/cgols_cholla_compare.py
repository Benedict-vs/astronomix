"""
Our emission pipeline on a public CGOLS (Cholla) snapshot, side by side with our own checkpoint.

TobiBu/asgard-papers issue #1: our A-series L_X lies below CGOLS I's values (1.9e40 erg/s for A-2048 at 60 Myr)
under every reading of their recipe. Running the same pipeline on their state tells whether the offset is in the
simulated gas or in how the published number was computed. This script runs the unchanged cgols_emission
projector on one state of either code and adds what locates the difference:

* the cgols_emission cache (box L_X, edge-on maps, minor-axis profiles) in two variants per pass: the default
  0.3-2 keV energy products (CGOLS recipe X1, AtomDB X2) and the 0.3-1 keV photons after TBabs absorption by
  N_H = 4e20 cm^-2 (the band and units of the Strickland et al. 2004 M82 fits). Same file names and keys as
  cgols_emission, so paper_exports reads them unchanged:
  data/studies/emission<tag>.npz and emission<tag>_sx0p3_1keV_ph_nh4e20.npz;
* data/studies/cholla_compare<tag>.npz: (log n, log T) histograms weighted by the CGOLS-recipe X-ray
  emissivity and by mass, and per z cell the emissivity-weighted sums of log n and log T of the 2-15 MK gas
  (their means against height).

The plot mode also scans the lower edge of the CGOLS-recipe temperature window (Eq. 17 with (3/2) k_B T in
0.3-2 keV puts it at 2.32 MK): from the mass histogram, the box L_X is rescaled by
sum(M n Lambda_A4(T)) over the bins inside each window, normalised to the full calculation at the default window.

Sources:
  --h5 FILE           one Cholla snapshot (README of the CGOLS data release, doi:10.13139/OLCF/3366667):
                      datasets density [Msun/kpc^3], GasEnergy (thermal energy density) [Msun/(kpc kyr^2)],
                      attributes t [kyr] and gamma; read in x slabs.
  --checkpoint DIR    one of our Orbax checkpoint step directories, the same reader as cgols_emission.

Usage (CPU only; every option it does not know goes to cgols_emission, --tag is required there):
    python cgols_cholla_compare.py measure --h5 ~/cgols_public/CGOLS_I/60.h5 --tag _CGOLSI --threads 16
    python cgols_cholla_compare.py measure --checkpoint ~/asgard/model_A/2048/A2048/checkpoints/48 --tag _A2048c
    python cgols_cholla_compare.py plot --tag _CGOLSI --against _A2048c
"""

# general
import argparse
import os
import sys
import time

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("mode", choices=("measure", "plot"))
_parser.add_argument("--h5", default=None, help="Cholla HDF5 snapshot")
_parser.add_argument("--checkpoint", default=None, help="our Orbax checkpoint step directory")
_parser.add_argument("--slab", type=int, default=32, help="x planes per block")
_parser.add_argument("--against", default=None, help="plot: tag of the second state")
_parser.add_argument("--figure", default=None, help="plot: output path")
OWN, FORWARDED = _parser.parse_known_args()
# cgols_emission parses its options at import: hand it the rest.
sys.argv = [sys.argv[0], *FORWARDED]

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# numerics, units, plotting
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from astropy import units as u

matplotlib.use("Agg")
import cgols_emission as ce
import matplotlib.pyplot as plt

from astronomix._modules._cooling._cooling import cooling_rate_cie_parabolic
from astronomix._modules._cooling.cooling_options import CIEParabolicParams
from astronomix.analysis_helpers.emission import (
    CGOLS_XRAY_TEMPERATURE_MAX_K,
    CGOLS_XRAY_TEMPERATURE_MIN_K,
    XRAY_EMISSIVITY_UNIT,
    XRAY_TABLE_PATHS,
    EmissionMapSettings,
    cgols_xray_emissivity,
    emission_conversions,
    load_xray_emissivity_table,
)

jax.config.update("jax_enable_x64", False)

#: (band, absorbing column) variants measured in one pass; the suffix is cgols_emission.xray_suffix()'s.
VARIANTS = (("0.3-2", None, ""), ("0.3-1", 4e20, "_sx0p3_1keV_ph_nh4e20"))
#: Histogram grid: log10 n [cm^-3] and log10 T [K].
LOG_N_EDGES = np.linspace(-6.0, 1.0, 141)
LOG_T_EDGES = np.linspace(4.0, 9.0, 201)
#: The gas Tobias asked about: 2-15 MK.
HOT_RANGE_K = (2e6, 1.5e7)
#: Legend names of the tags used for issue #1.
READABLE = {"_CGOLSI": "CGOLS I (public, 2048$^3$)", "_A2048c": "our A-2048"}
#: Lower window edges [K] for the scan of the CGOLS-recipe temperature window.
LOWER_EDGES_K = np.geomspace(1.0e6, 4.0e6, 31)

CONVERSIONS = emission_conversions(ce.CODE_UNITS)
#: sr -> arcmin^2, for the profiles in photons per arcmin^2.
ARCMIN2_PER_STERADIAN = float((1.0 * u.sr).to(u.arcmin**2).value)
#: Cholla CGOLS units -> our code units.
CHOLLA_DENSITY_TO_CODE = float((1 * u.M_sun / u.kpc**3).to(ce.CODE_UNITS.code_density).value)
CHOLLA_ENERGY_DENSITY_TO_CODE = float((1 * u.M_sun / (u.kpc * u.kyr**2)).to(ce.CODE_UNITS.code_pressure).value)


# -------------------------------------------------------------
# =============== ↓ Readers ↓ =================================
# -------------------------------------------------------------


def cholla_blocks(path, slab):
    """(grid, time [Myr], iterator of (x slice, density, pressure) in code units) of a Cholla snapshot."""
    import h5py

    handle = h5py.File(path, "r")
    names = {name.lower(): name for name in handle}
    density_set = handle[names["density"]]
    nx, ny, nz = density_set.shape
    gamma = float(handle.attrs["gamma"][0] if np.ndim(handle.attrs["gamma"]) else handle.attrs["gamma"])
    time_kyr = float(np.atleast_1d(handle.attrs["t"])[0])
    thermal = names.get("gasenergy")

    def blocks():
        for start in range(0, nx, slab):
            rows = slice(start, min(start + slab, nx))
            density = density_set[rows].astype(np.float32)
            if thermal is not None:
                internal = handle[thermal][rows].astype(np.float32)
            else:  # total minus kinetic energy density
                momentum2 = sum(handle[names[f"momentum_{axis}"]][rows].astype(np.float32) ** 2 for axis in "xyz")
                internal = handle[names["energy"]][rows].astype(np.float32) - 0.5 * momentum2 / density
            pressure = (gamma - 1.0) * internal * CHOLLA_ENERGY_DENSITY_TO_CODE
            yield rows, density * CHOLLA_DENSITY_TO_CODE, pressure
        handle.close()

    return (nx, ny, nz), time_kyr * 1e-3, blocks()


def checkpoint_blocks(step_directory, slab, threads):
    """The same for one of our Orbax checkpoints (the reader of cgols_emission.checkpoint_snapshot)."""
    import tensorstore

    context = tensorstore.Context(
        {
            "cache_pool": {"total_bytes_limit": 0},
            "data_copy_concurrency": {"limit": threads},
            "file_io_concurrency": {"limit": threads},
        }
    )
    state = ce._open_checkpoint_array(step_directory, "primitive_state", context)
    time_code = float(ce._open_checkpoint_array(step_directory, "time", context).read().result())
    num_vars, nx, ny, nz = state.shape
    registered = ce.get_registered_variables(
        ce.SimulationConfig(dimensionality=3, solver_mode=ce.FINITE_DIFFERENCE, dual_energy=(num_vars == 6))
    )

    def blocks():
        for start in range(0, nx, slab):
            rows = slice(start, min(start + slab, nx))
            block = state[:, rows, :, :].read().result()
            yield rows, block[registered.density_index], block[registered.pressure_index]

    return (nx, ny, nz), time_code * ce.CODE_TIME_MYR, blocks()


# -------------------------------------------------------------
# =============== ↓ Measurement ↓ =============================
# -------------------------------------------------------------


@jax.jit
def phase_statistics(density, pressure):
    """Histograms and per-z sums of one block (density, pressure in code units, shape (bx, ny, nz))."""
    number_density = density * CONVERSIONS.total_number_density_per_code_density
    temperature = pressure / density * CONVERSIONS.kelvin_per_code_specific_energy
    emissivity = cgols_xray_emissivity(number_density, temperature)  # XRAY_EMISSIVITY_UNIT
    log_n = jnp.log10(number_density).ravel()
    log_t = jnp.log10(temperature).ravel()
    hist_emission, _, _ = jnp.histogram2d(log_n, log_t, bins=(LOG_N_EDGES, LOG_T_EDGES), weights=emissivity.ravel())
    hist_mass, _, _ = jnp.histogram2d(log_n, log_t, bins=(LOG_N_EDGES, LOG_T_EDGES), weights=density.ravel())
    hot = (temperature >= HOT_RANGE_K[0]) & (temperature <= HOT_RANGE_K[1])
    weight = jnp.where(hot, emissivity, 0.0)
    return {
        "hist_emission": hist_emission,
        "hist_mass": hist_mass,
        "z_weight_hot": weight.sum(axis=(0, 1)),
        "z_log_n_hot": (weight * jnp.log10(number_density)).sum(axis=(0, 1)),
        "z_log_t_hot": (weight * jnp.log10(temperature)).sum(axis=(0, 1)),
        "z_mass_hot": jnp.where(hot, density, 0.0).sum(axis=(0, 1)),
    }


def measure():
    """Read one state slab by slab, project it (both variants) and accumulate the phase statistics."""
    args = ce.ARGS
    if (OWN.h5 is None) == (OWN.checkpoint is None):
        raise SystemExit("give exactly one of --h5 / --checkpoint")
    if OWN.h5:
        grid, time_myr, blocks = cholla_blocks(os.path.expanduser(OWN.h5), OWN.slab)
        source = f"cholla_hdf5:{os.path.abspath(os.path.expanduser(OWN.h5))}"
    else:
        grid, time_myr, blocks = checkpoint_blocks(os.path.expanduser(OWN.checkpoint), OWN.slab, args.threads)
        source = f"checkpoint:{os.path.abspath(os.path.expanduser(OWN.checkpoint))}"
    nx, ny, nz = grid
    if nz != 2 * nx or nx != ny:
        raise SystemExit(f"unexpected CGOLS grid {grid}")
    cell_size_kpc = ce.BOX_SIZE_KPC[0] / nx
    cell_size_code = float((cell_size_kpc * u.kpc).to(ce.CODE_UNITS.code_length).value)
    print(f"{args.tag}: {source}, grid {grid}, t = {time_myr:.3f} Myr", flush=True)

    settings = EmissionMapSettings(projection_axes=(1, 0), solar_abundance_set=args.solar_abundances)
    projectors = {}
    for band, column, suffix in VARIANTS:
        table = load_xray_emissivity_table(
            XRAY_TABLE_PATHS[band], photons=column is not None, absorbing_column_cm2=column or 0.0
        )
        ce._BLOCK_PROJECTORS.clear()  # memoised per cell size only: one projector per table
        projectors[suffix] = ce.make_block_projector(table, settings, cell_size_code)
    maps = {suffix: {} for suffix in projectors}
    stats = None
    start = time.perf_counter()
    for count, (rows, density, pressure) in enumerate(blocks):
        density, pressure = jnp.asarray(density), jnp.asarray(pressure)
        for suffix, project in projectors.items():
            for name, block_map in project(density, pressure).items():
                plane = name[-2:]
                if name not in maps[suffix]:
                    maps[suffix][name] = np.zeros((nx if plane == "xz" else ny, nz))
                if plane == "xz":
                    maps[suffix][name][rows] += np.asarray(block_map, dtype=np.float64)
                else:
                    maps[suffix][name] += np.asarray(block_map, dtype=np.float64)
        block_stats = {key: np.asarray(value, dtype=np.float64) for key, value in phase_statistics(density, pressure).items()}
        stats = block_stats if stats is None else {key: stats[key] + block_stats[key] for key in stats}
        print(f"  slab {count + 1}/{-(-nx // OWN.slab)} ({time.perf_counter() - start:.0f} s)", flush=True)

    index = round(time_myr * 1000)  # the "snapshot index" of a single-state cache: t in kyr
    for band, column, suffix in VARIANTS:
        snapshot = ce.Snapshot(index, time_myr, maps[suffix], cell_size_kpc, grid)
        summary, products = ce.summarise(snapshot, args.metallicity, args.halpha_disk_mask_kpc)
        summary["time_myr"] = time_myr
        metadata = {
            "source": source, "metallicity": args.metallicity, "solar_abundances": args.solar_abundances,
            "xray_band": band, "xray_photons_nh": np.nan if column is None else column,
            "halpha_disk_mask_kpc": np.nan, "map_time_myr": time_myr, "map_snapshot_index": index,
        }  # fmt: skip
        path = os.path.join(ce.SCRIPT_DIR, "data", "studies", f"emission{args.tag}{suffix}.npz")
        ce.save_cache(path, {index: summary}, products, metadata)
        print(f"wrote {path}: L_X1 = {summary['L_xray_cgols_xz']:.4e}, L_X2 = {summary['L_xray_atomdb_xz']:.4e}")
    z_centers = (np.arange(nz) + 0.5) * cell_size_kpc - ce.BOX_SIZE_KPC[2] / 2
    path = os.path.join(ce.SCRIPT_DIR, "data", "studies", f"cholla_compare{args.tag}.npz")
    np.savez(
        path, **stats, log_n_edges=LOG_N_EDGES, log_t_edges=LOG_T_EDGES, hot_range_k=np.array(HOT_RANGE_K),
        z_kpc=z_centers, time_myr=time_myr, cell_size_kpc=cell_size_kpc, source=source,
        emissivity_unit=XRAY_EMISSIVITY_UNIT,
    )  # fmt: skip
    print(f"wrote {path}")


# -------------------------------------------------------------
# =============== ↓ Figure ↓ ==================================
# -------------------------------------------------------------


def _load(tag):
    studies = os.path.join(ce.SCRIPT_DIR, "data", "studies")
    phase = dict(np.load(os.path.join(studies, f"cholla_compare{tag}.npz"), allow_pickle=False))
    records, _, _ = ce.load_cache(os.path.join(studies, f"emission{tag}{VARIANTS[1][2]}.npz"))
    return phase, next(iter(records.values()))


def _height_means(phase, bin_kpc=0.25):
    """|z| bins: emissivity-weighted mean log n and log T of the 2-15 MK gas."""
    z = np.abs(phase["z_kpc"])
    edges = np.arange(0.0, z.max() + bin_kpc, bin_kpc)
    which = np.digitize(z, edges) - 1
    centers, log_n, log_t = [], [], []
    for b in range(len(edges) - 1):
        sel = which == b
        weight = phase["z_weight_hot"][sel].sum()
        if weight > 0:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            log_n.append(phase["z_log_n_hot"][sel].sum() / weight)
            log_t.append(phase["z_log_t_hot"][sel].sum() / weight)
    return np.array(centers), np.array(log_n), np.array(log_t)


def lower_edge_scan(phase, box_luminosity, upper_k=CGOLS_XRAY_TEMPERATURE_MAX_K, lower_edges_k=LOWER_EDGES_K):
    """Box L_X of the CGOLS recipe for each lower window edge (upper edge fixed), from the mass histogram.

    Each bin contributes M n Lambda_A4(T) at its centre (n^2 Lambda V = (rho V) n Lambda / (mu m_p)); the sum is
    normalised so the default window (2.32-15.5 MK) gives ``box_luminosity``, the full cell-by-cell result.
    """
    density = 10.0 ** (0.5 * (LOG_N_EDGES[1:] + LOG_N_EDGES[:-1]))
    temperature = 10.0 ** (0.5 * (LOG_T_EDGES[1:] + LOG_T_EDGES[:-1]))
    curve = np.asarray(cooling_rate_cie_parabolic(jnp.asarray(temperature), CIEParabolicParams(log10_lambda_cgs_to_code=0.0)))
    per_t = (phase["hist_mass"] * density[:, None]).sum(axis=0) * curve

    def window_sum(low, high):
        return per_t[(temperature >= low) & (temperature <= high)].sum()

    scale = box_luminosity / window_sum(CGOLS_XRAY_TEMPERATURE_MIN_K, CGOLS_XRAY_TEMPERATURE_MAX_K)
    return np.array([scale * window_sum(low, upper_k) for low in lower_edges_k])


def plot():
    """xray_cgols_vs_A: S_X(z) against Strickland, the emission-weighted n-T distributions, n and T against |z|."""
    labels = {tag: READABLE.get(tag, tag.lstrip("_")) for tag in (ce.ARGS.tag, OWN.against)}
    states = {tag: _load(tag) for tag in labels}
    fig, axes = plt.subplots(1, 5, figsize=(23.5, 4.4))
    published = ce.CGOLS_LUMINOSITIES["A-2048"]
    for (tag, (phase, record)), color in zip(states.items(), ("C3", "C0"), strict=True):
        z = record["profile_z_kpc"]
        for side, style in (("north", "-"), ("south", "--")):
            axes[0].plot(z, record[f"profile_xray_atomdb_xz_{side}"] / ARCMIN2_PER_STERADIAN,
                         color=color, ls=style, label=f"{labels[tag]} ({side})")  # fmt: skip
        centers, log_n, log_t = _height_means(phase)
        axes[3].plot(centers, log_n, color=color, label=f"{labels[tag]}: log n")
        axes[3].plot(centers, log_t - 7.0, color=color, ls="--", label=f"{labels[tag]}: log T - 7")
        box = float(record["L_xray_cgols_xz"])
        for upper, style, text in ((CGOLS_XRAY_TEMPERATURE_MAX_K, "-", "15.5 MK"), (np.inf, ":", "none")):
            scan = lower_edge_scan(phase, box, upper_k=upper)
            axes[4].plot(LOWER_EDGES_K / 1e6, scan, color=color, ls=style, label=f"{labels[tag]}, upper edge {text}")
            crossing = np.interp(np.log(published), np.log(scan[::-1]), LOWER_EDGES_K[::-1])
            print(f"{tag:10s} upper edge {text:8s}: L_X = {box:.3e} erg/s at 2.32 MK; "
                  f"{published:.2g} erg/s needs a lower edge of {crossing / 1e6:.2f} MK")
    for side, _instrument, (z_min, z_max), scale_height, central in ce.STRICKLAND04_M82_FITS:
        fit_z = np.linspace(z_min, z_max, 100)
        axes[0].plot(fit_z, central * 1e-9 * np.exp(-fit_z / scale_height) * 3600.0, color="k", lw=2, alpha=0.5,
                     ls="-" if side == "north" else "--", label=f"Strickland+04 {side}")  # fmt: skip
    axes[0].set(yscale="log", xlim=(0, 10), xlabel="|z| [kpc]",
                ylabel="S_X, 0.3-1 keV absorbed [photons s$^{-1}$ cm$^{-2}$ arcmin$^{-2}$]",
                title="minor-axis profile, |x| < 5 kpc")  # fmt: skip
    axes[0].legend(fontsize=7)
    for ax, (tag, (phase, _record)) in zip(axes[1:3], states.items(), strict=True):
        hist = phase["hist_emission"] / phase["hist_emission"].sum()
        extent = (LOG_N_EDGES[0], LOG_N_EDGES[-1], LOG_T_EDGES[0], LOG_T_EDGES[-1])
        image = ax.imshow(np.log10(np.maximum(hist.T, 1e-8)), origin="lower", extent=extent, aspect="auto",
                          vmin=-6, vmax=-1, cmap="magma")  # fmt: skip
        for limit in HOT_RANGE_K:
            ax.axhline(np.log10(limit), color="w", lw=0.8, ls=":")
        ax.set(xlim=(-5, 0), ylim=(5.5, 8.5), xlabel="log n [cm$^{-3}$]", ylabel="log T [K]",
               title=f"{labels[tag]}, L_X-weighted (CGOLS recipe)")  # fmt: skip
    fig.colorbar(image, ax=axes[2], label="log fraction of L_X per bin")
    axes[3].set(xlabel="|z| [kpc]", ylabel="L_X-weighted mean, 2-15 MK gas", title="n and T against height")
    axes[3].legend(fontsize=7)
    axes[4].axhline(published, color="k", lw=1, alpha=0.6, label="CGOLS I, A-2048, 60 Myr")
    axes[4].axvline(CGOLS_XRAY_TEMPERATURE_MIN_K / 1e6, color="0.5", lw=0.8, ls="--", label="(3/2) kT = 0.3 keV")
    axes[4].set(xscale="log", yscale="log", xlabel="lower edge of the temperature window [MK]",
                ylabel="box L_X, CGOLS recipe [erg s$^{-1}$]", title="L_X against the window's lower edge")  # fmt: skip
    axes[4].set_xticks([1, 1.5, 2, 3, 4], labels=["1", "1.5", "2", "3", "4"])
    axes[4].legend(fontsize=7)
    fig.tight_layout()
    out = OWN.figure or os.path.join(ce.SCRIPT_DIR, "figures", "cgols", "comparisons", "xray_cgols_vs_A.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    if OWN.mode == "measure":
        measure()
    else:
        if OWN.against is None:
            raise SystemExit("plot needs --against <tag>")
        plot()
