"""
Synthetic soft X-ray and H-alpha emission of a CGOLS run (CGOLS I Figs. 9, 10).

Post-processes the full-state Orbax checkpoints of one run (or, when those are
absent, the emission maps cgols.py stores in every frame, see
CGOLS_EMISSION_FRAMES) with the kernels of astronomix/analysis_helpers/
emission.py, and writes

  data/studies/emission<tag>.npz        cache: per-snapshot luminosities and
                                        minor-axis profiles, maps at --map-steps
  figures/cgols/emission/<tag>_*.png    Fig. 9-like maps, Fig. 10-like S_X(z)
                                        profiles, L_X(t) and L_Halpha(t)

Emission models (see emission.py for the physics):
  xray_cgols        the CGOLS I Eq. 17 recipe, n^2 Lambda_A4(T) in the
                    2.32e6-1.55e7 K window
  xray_atomdb       n_e n_H epsilon_{0.3-2 keV}(T) from AtomDB, no T window,
                    solar metals (Asplund+09, or AG89 with
                    --solar-abundances AG89) scaled by --metallicity
  xray_atomdb_lopez the same with the Lopez+20 M82 wind abundance pattern
  halpha            Eq. 3 of arXiv:2606.10762 in 5e3-5e4 K
  emission_measure  int n_e n_H dl of the same gas [pc cm^-6]

Memory: a checkpoint is never loaded whole. The state is read chunk by chunk
(the Orbax chunks are (x, y)-tiles spanning all of z) straight from the OCDBT
store with tensorstore; every map is a line-of-sight sum, so the tiles'
contributions are simply added. Peak RAM is about two chunks (2 x 1.3 GB at
1024, the next one is prefetched) plus the maps. CPU only, with a bounded
thread count (--threads, default 16).

Usage (from this directory, any python env with astronomix + tensorstore):
    python cgols_emission.py --tag _A512new
    python cgols_emission.py --tag _A1024 --checkpoint-dir data/legacy/cgols_checkpoints_A1024
    python cgols_emission.py --tag _A2048 --source frames
    python cgols_emission.py --tag _A512new --figures-only
The cache is resumable: snapshots already in it are skipped.
"""

# general
import argparse
import glob
import os
import time

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--tag", required=True, help="run tag, e.g. _A512new (names the cache and figures)")
_parser.add_argument(
    "--checkpoint-dir",
    default=None,
    help="Orbax checkpoint root (default data/cgols_checkpoints<tag>)",
)
_parser.add_argument("--frames-dir", default=None, help="frame dir (default cgols_snapshots<tag>)")
_parser.add_argument(
    "--source",
    choices=("auto", "checkpoints", "frames"),
    default="auto",
    help="auto: checkpoints if the directory has any, else the frames' in-situ maps",
)
_parser.add_argument("--steps", default="all", help="checkpoint steps / frame indices: 'all', '48' or '40-60'")
_parser.add_argument(
    "--map-time-myr",
    type=float,
    default=60.0,
    help="store the full maps of the snapshot closest to this time (Figs. 9 / 10)",
)
_parser.add_argument(
    "--metallicity",
    type=float,
    default=1.0,
    help="Z / Z_sun of the AtomDB model, relative to --solar-abundances",
)
_parser.add_argument(
    "--solar-abundances",
    choices=("A09", "AG89"),
    default="A09",
    help="solar set Z_sun refers to: Asplund+09 (default) or Anders & Grevesse 89; "
    "in-situ frames carry A09-normalised metals only",
)
_parser.add_argument(
    "--halpha-disk-mask-kpc",
    type=float,
    default=None,
    help="drop H-alpha / EM at |z| below this (kpc); default off",
)
_parser.add_argument("--threads", type=int, default=16, help="CPU threads (XLA + tensorstore)")
_parser.add_argument("--figures-only", action="store_true", help="re-render the figures from the cache")
_parser.add_argument("--no-figures", action="store_true", help="only update the cache")
_parser.add_argument("--cache", default=None, help="override the cache path")
_parser.add_argument("--figure-dir", default=None, help="override the figure directory")
ARGS = _parser.parse_args()

# ==== CPU only, bounded threads (before jax is imported) ====
# The machine is shared: pin the process to the --threads least busy CPUs.
# XLA's CPU thread pool and the BLAS pools size themselves from the affinity
# mask; tensorstore gets an explicit concurrency limit below.
os.environ["JAX_PLATFORMS"] = "cpu"
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = str(ARGS.threads)
try:
    import psutil

    _allowed = sorted(os.sched_getaffinity(0))
    _load = psutil.cpu_percent(interval=0.5, percpu=True)
    _quietest = sorted(_allowed, key=lambda cpu: _load[cpu])[: ARGS.threads]
    os.sched_setaffinity(0, _quietest)
except (ImportError, AttributeError, OSError):
    pass
# ruff: noqa: E402

# jax
import jax

# numerics
import numpy as np

# checkpoint I/O
import tensorstore

# plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# units and constants
from astropy import units as u

# astronomix constants
from astronomix.analysis_helpers.emission import (
    ARCMIN2_PER_STERADIAN,
    MAP_UNITS,
    SOLAR_ABUNDANCE_SET,
)
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE

# astronomix containers
from astronomix import CodeUnits, SimulationConfig
from astronomix.analysis_helpers.emission import EmissionMapSettings

# astronomix functions
from astronomix import get_registered_variables
from astronomix.analysis_helpers.emission import (
    emission_conversions,
    emission_maps,
    load_xray_emissivity_table,
    luminosity_from_map,
    minor_axis_profile,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The CGOLS code units and box. Must match cgols.py, which this script does
# not import: its import-time GPU selection (autocvd) would claim a GPU for a
# CPU-only job.
CODE_UNITS = CodeUnits(1 * u.kiloparsec, 1e6 * u.M_sun, 100 * u.km / u.s)
BOX_SIZE_KPC = (10.0, 10.0, 20.0)
CODE_TIME_MYR = float((1 * CODE_UNITS.code_time).to(u.Myr).value)

#: Fig. 10 of CGOLS I: |x| < 5 kpc (the full box width), 0.15625 kpc |z| bins.
PROFILE_HALF_WIDTH_KPC = 5.0
PROFILE_BIN_KPC = 0.15625

#: The products derived per snapshot; "xray_atomdb" is H+He + Z x metals.
PRODUCTS = ("xray_cgols", "xray_atomdb", "xray_atomdb_lopez", "halpha", "emission_measure")
PLANES = ("xz", "yz")

#: CGOLS I, Sec. V: L_X (0.3-2 keV window recipe) of the A-series at 60 Myr,
#: and M82 (Strickland+04 via CGOLS I).
CGOLS_LUMINOSITIES = {"A-512": 2.9e40, "A-1024": 1.4e40, "A-2048": 1.9e40}
M82_XRAY_LUMINOSITY = 4.3e40

#: Strickland et al. 2004 (ApJS 151, 193) Table 6, M82 exponential minor-axis
#: fits Sigma_0 exp(-|z| / H), Sigma_0 in 1e-9 photons s^-1 cm^-2 arcsec^-2.
#: NOTE: these fits are in the 0.3-1.0 keV band, not 0.3-2.0 keV. CGOLS I
#: plotted "the best-fit exponential profile" against its 0.3-2.0 keV window
#: converted at 1 photon ~ 2e-9 erg (Strickland's 0.3-1.0 keV factor, which he
#: says holds for the energy flux in either band; ~2x higher for the ACIS-S
#: data of M82). We follow CGOLS and flag the band mismatch on the figure.
STRICKLAND04_M82_FITS = (
    # side, instrument, z range [kpc], H [kpc], Sigma_0
    ("north", "ACIS-I", (0.5, 5.0), 1.42, 29.16),
    ("south", "ACIS-I", (0.2, 10.0), 0.73, 103.4),
)
STRICKLAND04_ERG_PER_PHOTON = 2e-9


def _in_dir(subdir, name):
    """``SCRIPT_DIR/subdir/name``, creating the directory."""
    directory = os.path.join(SCRIPT_DIR, subdir)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)


def _resolve(path):
    """A user path relative to the script directory."""
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def parse_steps(spec, available):
    """The subset of ``available`` selected by 'all', 'k' or 'a-b'."""
    if spec == "all":
        return list(available)
    if "-" in spec:
        first, last = (int(part) for part in spec.split("-"))
        return [step for step in available if first <= step <= last]
    return [step for step in available if step == int(spec)]


# -------------------------------------------------------------
# =============== ↓ Map sources ↓ =============================
# -------------------------------------------------------------


class Snapshot:
    """The edge-on maps of one snapshot plus its grid."""

    def __init__(self, index, time_myr, maps, cell_size_kpc, num_cells):
        self.index = index
        self.time_myr = time_myr
        #: ``{f"{product}_{plane}": float64 map}`` in physical units.
        self.maps = maps
        self.cell_size_kpc = cell_size_kpc
        self.num_cells = num_cells


def _open_checkpoint_array(step_directory, name, context):
    """Open one array of an Orbax (OCDBT) checkpoint for chunked reading."""
    spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "ocdbt",
            "base": "file://" + os.path.abspath(step_directory) + "/",
            "path": name,
        },
    }
    return tensorstore.open(spec, context=context, read=True).result()


def checkpoint_steps(checkpoint_directory):
    """The numbered step directories of a checkpoint root, sorted."""
    if not os.path.isdir(checkpoint_directory):
        return []
    return sorted(
        int(entry) for entry in os.listdir(checkpoint_directory) if entry.isdigit()
    )


_BLOCK_PROJECTORS = {}


def make_block_projector(table, settings, cell_size_code):
    """The jitted (density, pressure) block -> maps function.

    Memoised per cell size, so the snapshots of one run share one compiled
    function instead of re-tracing it for every checkpoint.
    """
    if cell_size_code in _BLOCK_PROJECTORS:
        return _BLOCK_PROJECTORS[cell_size_code]
    conversions = emission_conversions(CODE_UNITS)

    @jax.jit
    def project_block(density, pressure):
        return emission_maps(
            density,
            pressure,
            conversions,
            table,
            cell_size_code,
            settings,
        )

    _BLOCK_PROJECTORS[cell_size_code] = project_block
    return project_block


def checkpoint_snapshot(step_directory, step, table, settings, threads):
    """Project one checkpoint, chunk by chunk."""
    context = tensorstore.Context(
        {
            "cache_pool": {"total_bytes_limit": 0},
            "data_copy_concurrency": {"limit": threads},
            "file_io_concurrency": {"limit": threads},
        }
    )
    state = _open_checkpoint_array(step_directory, "primitive_state", context)
    time_code = float(_open_checkpoint_array(step_directory, "time", context).read().result())

    num_vars, nx, ny, nz = state.shape
    if nz != 2 * nx or nx != ny:
        raise ValueError(f"{step_directory}: unexpected CGOLS grid {state.shape}")
    # Index the state through the registry of the matching configuration:
    # new-numerics runs carry the dual-energy internal energy as a 6th variable.
    registered_variables = get_registered_variables(
        SimulationConfig(
            dimensionality=3,
            solver_mode=FINITE_DIFFERENCE,
            dual_energy=(num_vars == 6),
        )
    )
    if registered_variables.num_vars != num_vars:
        raise ValueError(f"{step_directory}: {num_vars} variables, expected 5 or 6")

    cell_size_kpc = BOX_SIZE_KPC[0] / nx
    cell_size_code = float((cell_size_kpc * u.kpc).to(CODE_UNITS.code_length).value)
    project_block = make_block_projector(table, settings, cell_size_code)

    chunk_x, chunk_y = state.chunk_layout.read_chunk.shape[1:3]
    tiles = [
        (slice(x_start, min(x_start + chunk_x, nx)), slice(y_start, min(y_start + chunk_y, ny)))
        for x_start in range(0, nx, chunk_x)
        for y_start in range(0, ny, chunk_y)
    ]
    # One read per chunk: every chunk spans all variables, so reading the two
    # needed ones separately would decode it twice. The next tile's read is
    # issued before the current one is projected, overlapping I/O and compute
    # at the price of a second chunk in memory.
    pending_read = state[:, tiles[0][0], tiles[0][1], :].read()
    maps = {}
    for tile_index, (x_slice, y_slice) in enumerate(tiles):
        block = pending_read.result()
        if tile_index + 1 < len(tiles):
            next_x, next_y = tiles[tile_index + 1]
            pending_read = state[:, next_x, next_y, :].read()
        block_maps = project_block(
            block[registered_variables.density_index],
            block[registered_variables.pressure_index],
        )
        del block
        for name, block_map in block_maps.items():
            plane = name[-2:]
            if name not in maps:
                maps[name] = np.zeros((nx if plane == "xz" else ny, nz))
            rows = x_slice if plane == "xz" else y_slice
            maps[name][rows] += np.asarray(block_map, dtype=np.float64)
    return Snapshot(step, time_code * CODE_TIME_MYR, maps, cell_size_kpc, (nx, ny, nz))


def frame_files(frames_directory):
    """Frame files carrying in-situ emission maps, sorted by time."""
    files = sorted(glob.glob(os.path.join(frames_directory, "frame_*.npz")))
    with_maps = []
    for path in files:
        with np.load(path) as frame:
            if "emission_xray_cgols_xz" in frame.files:
                with_maps.append((float(frame["time_myr"]), path))
    return [path for _, path in sorted(with_maps)]


def frame_snapshot(path, index):
    """The in-situ maps of one frame (cgols.py, CGOLS_EMISSION_FRAMES)."""
    with np.load(path) as frame:
        maps = {
            key[len("emission_"):]: np.asarray(frame[key], dtype=np.float64)
            for key in frame.files
            if key.startswith("emission_")
        }
        time_myr = float(frame["time_myr"])
    nx, nz = maps["xray_cgols_xz"].shape
    return Snapshot(index, time_myr, maps, BOX_SIZE_KPC[0] / nx, (nx, nx, nz))


# -------------------------------------------------------------
# =============== ↑ Map sources ↑ =============================
# -------------------------------------------------------------


def product_maps(snapshot, metallicity):
    """``{f"{product}_{plane}": map}`` of the derived products of a snapshot."""
    products = {}
    for plane in PLANES:
        if f"xray_cgols_{plane}" not in snapshot.maps:
            continue
        maps = snapshot.maps
        products[f"xray_cgols_{plane}"] = maps[f"xray_cgols_{plane}"]
        products[f"xray_atomdb_{plane}"] = (
            maps[f"xray_atomdb_hhe_{plane}"] + metallicity * maps[f"xray_atomdb_metals_{plane}"]
        )
        # Frames written before the Lopez+20 map existed do not carry it.
        if f"xray_atomdb_lopez_{plane}" in maps:
            products[f"xray_atomdb_lopez_{plane}"] = maps[f"xray_atomdb_lopez_{plane}"]
        products[f"halpha_{plane}"] = maps[f"halpha_{plane}"]
        products[f"emission_measure_{plane}"] = maps[f"emission_measure_{plane}"]
    return products


def map_unit_name(product):
    """The emission.MAP_UNITS key of a derived product."""
    return "xray_cgols" if product.startswith("xray") else product


def apply_disk_mask(products, snapshot, mask_half_height_kpc):
    """Zero H-alpha / EM map rows with |z| < mask (edge-on: rows are z)."""
    if mask_half_height_kpc is None:
        return products
    nz = snapshot.num_cells[2]
    z_centers = (np.arange(nz) + 0.5) * snapshot.cell_size_kpc - BOX_SIZE_KPC[2] / 2
    inside_disk = np.abs(z_centers) < mask_half_height_kpc
    for name in products:
        if name.startswith(("halpha", "emission_measure")):
            products[name] = np.where(inside_disk[None, :], 0.0, products[name])
    return products


def summarise(snapshot, metallicity, mask_half_height_kpc):
    """Luminosities and north/south minor-axis profiles of one snapshot."""
    products = apply_disk_mask(product_maps(snapshot, metallicity), snapshot, mask_half_height_kpc)
    nx, ny, nz = snapshot.num_cells
    cell_cm = float((snapshot.cell_size_kpc * u.kpc).to(u.cm).value)
    horizontal = {
        "xz": (np.arange(nx) + 0.5) * snapshot.cell_size_kpc - BOX_SIZE_KPC[0] / 2,
        "yz": (np.arange(ny) + 0.5) * snapshot.cell_size_kpc - BOX_SIZE_KPC[1] / 2,
    }
    z_centers = (np.arange(nz) + 0.5) * snapshot.cell_size_kpc - BOX_SIZE_KPC[2] / 2

    summary = {}
    for name, product_map in products.items():
        product, plane = name[:-3], name[-2:]
        summary[f"L_{name}"] = luminosity_from_map(product_map, cell_cm**2, map_unit_name(product))
        bin_centers, north, south = minor_axis_profile(
            product_map,
            horizontal[plane],
            z_centers,
            PROFILE_HALF_WIDTH_KPC,
            PROFILE_BIN_KPC,
        )
        summary["profile_z_kpc"] = bin_centers
        summary[f"profile_{name}_north"] = north
        summary[f"profile_{name}_south"] = south
    return summary, products


# -------------------------------------------------------------
# =============== ↓ Cache ↓ ===================================
# -------------------------------------------------------------


def load_cache(path):
    """``(per-snapshot records keyed by index, stored maps, metadata)`` or empties."""
    if not os.path.exists(path):
        return {}, {}, {}
    with np.load(path, allow_pickle=False) as cache:
        indices = cache["snapshot_index"]
        records = {}
        for row, index in enumerate(indices):
            records[int(index)] = {
                key[len("per_snapshot__"):]: cache[key][row]
                for key in cache.files
                if key.startswith("per_snapshot__")
            }
        maps = {
            key[len("map__"):]: cache[key] for key in cache.files if key.startswith("map__")
        }
        metadata = {
            key[len("meta__"):]: cache[key] for key in cache.files if key.startswith("meta__")
        }
    return records, maps, metadata


def save_cache(path, records, maps, metadata):
    """Write the cache atomically (a crash never leaves a truncated file)."""
    indices = sorted(records)
    keys = sorted({key for record in records.values() for key in record})
    arrays = {"snapshot_index": np.array(indices, dtype=np.int32)}
    for key in keys:
        # One row per snapshot; a product missing from a snapshot (e.g. the
        # Lopez+20 maps of older frames) is NaN-filled.
        rows = [np.atleast_1d(np.asarray(records[index].get(key, np.nan), dtype=np.float64)) for index in indices]
        width = max(row.size for row in rows)
        table = np.stack([row if row.size == width else np.full(width, np.nan) for row in rows])
        arrays[f"per_snapshot__{key}"] = table[:, 0] if width == 1 else table
    for key, value in maps.items():
        arrays[f"map__{key}"] = np.asarray(value, dtype=np.float32)
    for key, value in metadata.items():
        arrays[f"meta__{key}"] = np.asarray(value)
    temporary = path + ".tmp.npz"
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


# -------------------------------------------------------------
# =============== ↑ Cache ↑ ===================================
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Figures ↓ =================================
# -------------------------------------------------------------


def _solar_label(metadata):
    """Legend text of the AtomDB model's metallicity, e.g. 'Z = 1 Z_sun (A09)'."""
    metallicity = float(metadata.get("metallicity", 1.0))
    solar_set = str(metadata.get("solar_abundances", "AG89"))
    return rf"Z = {metallicity:g} Z$_\odot$ ({solar_set})"


def _map_extent(map_shape):
    """imshow extent (kpc) of an edge-on map, horizontal x vertical."""
    half_width = BOX_SIZE_KPC[0] / 2
    half_height = BOX_SIZE_KPC[2] / 2
    return (-half_width, half_width, -half_height, half_height)


def plot_maps(maps, metadata, tag, figure_directory):
    """Fig. 9-like edge-on maps (x-z plane, LOS along y) at the map time."""
    time_myr = float(metadata.get("map_time_myr", np.nan))
    panels_xray = [
        ("xray_cgols_xz", "X1: CGOLS I recipe\n" + r"$n^2\Lambda_{A4}$, $2.3$-$15.5$ MK"),
        ("xray_atomdb_xz", "X2: AtomDB 0.3-2 keV\n" + r"$n_e n_H\epsilon(T)$, " + _solar_label(metadata)),
    ]
    panels_halpha = [
        ("halpha_xz", r"H$\alpha$ (Eq. 3), $5\times10^3$-$5\times10^4$ K"),
        ("emission_measure_xz", r"EM $\int n_e n_H\,dl$, same window"),
    ]
    # Pixels without emitting gas (outside a temperature window) are zero;
    # draw them as the colormap's floor rather than as holes.
    emission_colormap = plt.get_cmap("magma").with_extremes(bad="black")
    for suffix, panels, label in (
        ("maps_xray", panels_xray, r"$S_X$ [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$]"),
        ("maps_halpha", panels_halpha, None),
    ):
        available = [(key, title) for key, title in panels if key in maps]
        if not available:
            continue
        figure, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 7.0), squeeze=False)
        for axis, (key, title) in zip(axes[0], available):
            image = np.asarray(maps[key], dtype=np.float64)
            positive = image[image > 0]
            if positive.size == 0:
                axis.set_title(title + "\n(empty)")
                continue
            vmax = np.percentile(positive, 99.9)
            norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
            shown = axis.imshow(
                np.where(image > 0, image, np.nan).T,
                origin="lower",
                extent=_map_extent(image.shape),
                norm=norm,
                cmap=emission_colormap,
            )
            unit = MAP_UNITS[map_unit_name(key[:-3])][0]
            figure.colorbar(shown, ax=axis, shrink=0.8, label=label or unit)
            axis.set_title(title, fontsize=9)
            axis.set_xlabel("x [kpc]")
            axis.set_ylabel("z [kpc]")
        figure.suptitle(f"{tag.lstrip('_')}: edge-on (LOS along y), t = {time_myr:.2f} Myr")
        figure.tight_layout()
        out = os.path.join(figure_directory, f"{tag.lstrip('_')}_{suffix}.png")
        figure.savefig(out, dpi=150)
        plt.close(figure)
        print(f"wrote {out}")


def plot_profiles(records, metadata, tag, figure_directory):
    """Fig. 10-like minor-axis S_X(z) at the map time, with Strickland+04 fits."""
    map_index = int(metadata["map_snapshot_index"])
    record = records[map_index]
    z = np.asarray(record["profile_z_kpc"])
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    styles = {
        "xray_cgols": ("C0", "X1 CGOLS recipe"),
        "xray_atomdb": ("C1", "X2 AtomDB, " + _solar_label(metadata)),
        "xray_atomdb_lopez": ("C2", "X2 AtomDB, Lopez+20 M82 wind"),
    }
    for product, (color, label) in styles.items():
        key = f"profile_{product}_xz_north"
        if key not in record or np.all(~np.isfinite(record[key])):
            continue
        for side, line_style in (("north", "-"), ("south", "--")):
            profile = np.asarray(record[f"profile_{product}_xz_{side}"]) / ARCMIN2_PER_STERADIAN
            axis.plot(z, profile, color=color, linestyle=line_style, label=f"{label} ({side})")
    for side, instrument, (z_min, z_max), scale_height, central in STRICKLAND04_M82_FITS:
        fit_z = np.linspace(z_min, z_max, 100)
        photons = central * 1e-9 * np.exp(-fit_z / scale_height)
        energy_per_arcmin2 = photons * STRICKLAND04_ERG_PER_PHOTON * 3600.0
        axis.plot(
            fit_z,
            energy_per_arcmin2,
            color="red",
            linestyle="-" if side == "north" else "--",
            linewidth=2.0,
            alpha=0.7,
            label=f"Strickland+04 {side} {instrument}, H = {scale_height} kpc",
        )
    axis.set_yscale("log")
    axis.set_xlim(0, BOX_SIZE_KPC[2] / 2)
    axis.set_xlabel("|z| [kpc]")
    axis.set_ylabel(r"$S_X$ [erg s$^{-1}$ cm$^{-2}$ arcmin$^{-2}$]")
    axis.set_title(
        f"{tag.lstrip('_')}: minor-axis profile, |x| < 5 kpc, t = "
        f"{float(record['time_myr']):.2f} Myr",
        fontsize=10,
    )
    axis.text(
        0.02,
        0.02,
        "Simulated: 0.3-2.0 keV. Strickland+04 Table 6 fits: 0.3-1.0 keV band,\n"
        "converted at 2e-9 erg/photon as in CGOLS I (band mismatch).",
        transform=axis.transAxes,
        fontsize=7,
        color="red",
    )
    finite = [
        np.asarray(record[key]) / ARCMIN2_PER_STERADIAN
        for key in record
        if key.startswith("profile_xray") and key.endswith(("north", "south"))
    ]
    finite = np.concatenate([values[np.isfinite(values) & (values > 0)] for values in finite] or [np.ones(1)])
    axis.set_ylim(max(finite.min(), finite.max() * 1e-6), finite.max() * 3)
    axis.legend(fontsize=7, loc="upper right")
    figure.tight_layout()
    out = os.path.join(figure_directory, f"{tag.lstrip('_')}_sx_profile.png")
    figure.savefig(out, dpi=150)
    plt.close(figure)
    print(f"wrote {out}")


def plot_luminosities(records, metadata, tag, figure_directory):
    """L_X(t) and L_Halpha(t) of every processed snapshot."""
    indices = sorted(records)
    times = np.array([float(records[index]["time_myr"]) for index in indices])
    figure, (axis_xray, axis_halpha) = plt.subplots(2, 1, figsize=(6.4, 7.0), sharex=True)
    for product, color, label in (
        ("xray_cgols", "C0", "X1 CGOLS recipe"),
        ("xray_atomdb", "C1", "X2 AtomDB, " + _solar_label(metadata)),
        ("xray_atomdb_lopez", "C2", "X2 AtomDB, Lopez+20"),
    ):
        key = f"L_{product}_xz"
        values = np.array([float(records[index].get(key, np.nan)) for index in indices])
        if np.any(np.isfinite(values)):
            axis_xray.plot(times, values, "o-", color=color, markersize=3, label=label)
    for (name, value), marker in zip(CGOLS_LUMINOSITIES.items(), ("s", "^", "D")):
        axis_xray.plot(60.0, value, marker, color="k", markerfacecolor="none", label=f"CGOLS I {name}")
    axis_xray.axhline(M82_XRAY_LUMINOSITY, color="red", linestyle=":", label="M82 (Strickland+04)")
    axis_xray.set_yscale("log")
    axis_xray.set_ylabel(r"$L_X$ [erg s$^{-1}$]")
    axis_xray.legend(fontsize=7)
    values = np.array([float(records[index].get("L_halpha_xz", np.nan)) for index in indices])
    axis_halpha.plot(times, values, "o-", color="C3", markersize=3, label=r"H$\alpha$, Eq. 3")
    axis_halpha.set_yscale("log")
    axis_halpha.set_ylabel(r"$L_{H\alpha}$ [erg s$^{-1}$]")
    axis_halpha.set_xlabel("t [Myr]")
    axis_halpha.legend(fontsize=7)
    figure.suptitle(f"{tag.lstrip('_')}: total luminosities (whole box)")
    figure.tight_layout()
    out = os.path.join(figure_directory, f"{tag.lstrip('_')}_luminosities.png")
    figure.savefig(out, dpi=150)
    plt.close(figure)
    print(f"wrote {out}")


# -------------------------------------------------------------
# =============== ↑ Figures ↑ =================================
# -------------------------------------------------------------


def main():
    """Process the requested snapshots into the cache, then render the figures."""
    tag = ARGS.tag
    cache_path = _resolve(ARGS.cache) if ARGS.cache else _in_dir(os.path.join("data", "studies"), f"emission{tag}.npz")
    figure_directory = (
        _resolve(ARGS.figure_dir) if ARGS.figure_dir else os.path.join(SCRIPT_DIR, "figures", "cgols", "emission")
    )
    os.makedirs(figure_directory, exist_ok=True)
    records, stored_maps, metadata = load_cache(cache_path)

    if not ARGS.figures_only:
        checkpoint_directory = _resolve(ARGS.checkpoint_dir or os.path.join("data", f"cgols_checkpoints{tag}"))
        frames_directory = _resolve(ARGS.frames_dir or f"cgols_snapshots{tag}")
        steps = checkpoint_steps(checkpoint_directory)
        source = ARGS.source
        if source == "auto":
            source = "checkpoints" if steps else "frames"
        if source == "checkpoints":
            available = steps
            located = {step: os.path.join(checkpoint_directory, str(step)) for step in steps}
        else:
            files = frame_files(frames_directory)
            available = list(range(len(files)))
            located = dict(enumerate(files))
        selected = parse_steps(ARGS.steps, available)
        if not selected:
            raise SystemExit(f"no {source} found for --steps {ARGS.steps} ({checkpoint_directory} / {frames_directory})")
        print(f"{tag}: {len(selected)} snapshot(s) from {source} ({len(records)} already cached)")

        if source == "frames" and ARGS.solar_abundances != SOLAR_ABUNDANCE_SET:
            raise SystemExit(
                f"--source frames: the in-situ metal maps are normalised to "
                f"{SOLAR_ABUNDANCE_SET} solar; --solar-abundances {ARGS.solar_abundances} "
                "needs the checkpoints"
            )
        table = load_xray_emissivity_table()
        settings = EmissionMapSettings(
            projection_axes=(1, 0),
            solar_abundance_set=ARGS.solar_abundances,
        )
        metadata.update(
            source=source,
            metallicity=ARGS.metallicity,
            solar_abundances=ARGS.solar_abundances,
            halpha_disk_mask_kpc=np.nan if ARGS.halpha_disk_mask_kpc is None else ARGS.halpha_disk_mask_kpc,
        )
        for index in selected:
            if index in records:
                continue
            start = time.perf_counter()
            if source == "checkpoints":
                snapshot = checkpoint_snapshot(located[index], index, table, settings, ARGS.threads)
            else:
                snapshot = frame_snapshot(located[index], index)
            summary, products = summarise(snapshot, ARGS.metallicity, ARGS.halpha_disk_mask_kpc)
            summary["time_myr"] = snapshot.time_myr
            records[index] = summary

            # Keep the full maps of the snapshot closest to the map time.
            current = float(metadata.get("map_time_myr", np.inf))
            if abs(snapshot.time_myr - ARGS.map_time_myr) < abs(current - ARGS.map_time_myr):
                stored_maps = dict(products)
                metadata.update(map_time_myr=snapshot.time_myr, map_snapshot_index=index)
            save_cache(cache_path, records, stored_maps, metadata)
            print(
                f"  [{index:3d}] t = {snapshot.time_myr:6.2f} Myr  "
                f"L_X1 = {summary['L_xray_cgols_xz']:.3e}  "
                f"L_X2 = {summary['L_xray_atomdb_xz']:.3e}  "
                f"L_Ha = {summary['L_halpha_xz']:.3e} erg/s  "
                f"({time.perf_counter() - start:.1f} s)",
                flush=True,
            )
        print(f"wrote {cache_path}")

    if ARGS.no_figures:
        return
    if not records:
        raise SystemExit(f"nothing to plot: {cache_path} is empty or missing")
    plot_maps(stored_maps, metadata, tag, figure_directory)
    plot_profiles(records, metadata, tag, figure_directory)
    plot_luminosities(records, metadata, tag, figure_directory)


if __name__ == "__main__":
    main()
