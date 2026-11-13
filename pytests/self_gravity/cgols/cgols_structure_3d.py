"""
3D small-scale structure of the CGOLS A-series outflow, per Orbax checkpoint: the statistics that should converge
even where the individual morphology of a chaotic flow does not (cgols_cost_accuracy plan, section 2/3).

Per checkpoint and hemisphere (north / south), in the outflow slab 1 < |z| < 4.5 kpc (full x, y):
  clumps        connected regions (6-connectivity) of entrained gas, log10 K < 5.5 (the entrained / mixed edge of
                cgols_convergence_3d.ENTROPY_EDGES): number, mass function (fixed log-mass bins in Msun) and size
                function (equivalent radius (3V / 4 pi)^(1/3) in pc), total and largest clump mass. Clumps cut by
                the slab faces count as clumps (same cut at every resolution).
  mixing layer  area of the log K = 5.5 and log K = 6.8 isosurfaces (entrained | mixed and mixed | wind) from face
                crossings between neighbouring cells, A = (2/3) N_faces dx^2 (exact on average for isotropically
                oriented surfaces; the same staircase estimator at every resolution), and the mixed-gas volume
                (5.5 <= log K < 6.8), so V_mixed / A is a mean mixing-layer thickness.
  spectra       3D power spectra of log10 n, log10 T and v (sum of the three component spectra) in the box
                |x|, |y| < 2.5 kpc, 1 < |z| < 3.5 kpc: mean removed, separable Hann window, continuous normalisation
                P(k) = |FFT|^2 dx^6 / V (units [f]^2 kpc^3, comparable between resolutions), averaged in shells of
                k = |k| [cycles / kpc] with the bins of cgols_cost_accuracy_extra.K_EDGES.
  profile       mass-weighted mean log T and log K per z plane of the slab (sanity check).

Reads each checkpoint once, (x, y) tile by tile (the chunks are full-height columns), keeping only the slab
(2048: ~45 GB peak per worker; 1024: ~6 GB). CPU only. One worker per checkpoint, single-threaded numerics and a
capped tensorstore thread pool, so --workers is the number of cores used (plus I/O threads).

    python cgols_structure_3d.py --tag _A1024 --checkpoint-dir /path/to/checkpoints [--steps 2-60/2] [--workers 8]
                                 [--out-dir DIR]

Writes OUT_DIR/structure3d_v1<TAG>.npz (default data/studies/), a dict {step: record}, saved after every checkpoint;
a rerun measures only the missing steps and skips step directories without an Orbax commit marker; with --ready-dir
only steps whose marker file <TAG>_<step> exists there (for checkpoints still being transferred: the marker is
written once the transfer has verified the step).
"""

# ruff: noqa: E402

# general
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# single-threaded numerics in every worker (set before numpy is imported)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS"):
    os.environ.setdefault(_var, "1" if _var != "XLA_FLAGS" else "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# numerics
import numpy as np
import scipy.fft
import tensorstore
from scipy import ndimage

# astronomix
from astronomix import SimulationConfig, get_registered_variables
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE

import cgols_convergence_3d as c3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SLAB_KPC = (1.0, 4.5)  # |z| range of the clump / mixing-layer slab
BOX_HALF_XY_KPC = 2.5  # spectra box: |x|, |y| < 2.5 kpc
BOX_Z_KPC = (1.0, 3.5)  # spectra box: 1 < |z| < 3.5 kpc
LOGK_ENTRAINED, LOGK_WIND = c3.ENTROPY_EDGES[0], c3.ENTROPY_EDGES[1]  # 5.5, 6.8
CLUMP_MASS_EDGES = np.logspace(-1.0, 8.0, 37)  # Msun
CLUMP_RADIUS_EDGES = np.geomspace(1.0, 4000.0, 37)  # pc, equivalent radius
K_EDGES = np.geomspace(0.1, 128.0, 49)  # cycles / kpc (= cgols_cost_accuracy_extra.K_EDGES)
FIELDS = ("logn", "logT", "v")


def _open(step_directory, name):
    # capped thread pools: the worker count is the parallelism, not tensorstore's
    context = tensorstore.Context({
        "cache_pool": {"total_bytes_limit": 0},
        "data_copy_concurrency": {"limit": 2},
        "file_io_concurrency": {"limit": 4},
    })
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "ocdbt", "base": "file://" + os.path.abspath(step_directory) + "/", "path": name},
    }
    return tensorstore.open(spec, context=context, read=True).result()


def shell_spectrum(field, dx, shape_kpc):
    """Shell-averaged 3D power spectrum (continuous normalisation) of a non-periodic box: mean removed, Hann window."""
    f = field.astype(np.float32) - np.float32(field.mean(dtype=np.float64))
    for axis, n in enumerate(f.shape):
        w = np.hanning(n).astype(np.float32)
        f *= w.reshape([-1 if a == axis else 1 for a in range(3)])
    # window power correction: <w^2> per axis
    norm = np.prod([np.mean(np.hanning(n) ** 2) for n in f.shape])
    spec = scipy.fft.rfftn(f, workers=1)
    del f
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    del spec
    volume = np.prod(shape_kpc)
    power *= np.float32(dx**6 / volume / norm)
    kx = np.fft.fftfreq(power.shape[0], d=dx).astype(np.float32)
    ky = np.fft.fftfreq(power.shape[1], d=dx).astype(np.float32)
    kz = np.fft.rfftfreq(field.shape[2], d=dx).astype(np.float32)
    sums = np.zeros(len(K_EDGES) - 1)
    counts = np.zeros(len(K_EDGES) - 1)
    # rfft: modes with 0 < kz < Nyquist stand for two (k, -k) modes
    weight_z = np.full(power.shape[2], 2.0, dtype=np.float32)
    weight_z[0] = 1.0
    if field.shape[2] % 2 == 0:
        weight_z[-1] = 1.0
    for i in range(power.shape[0]):  # plane by plane: no 3D |k| array
        kk = np.sqrt(kx[i] ** 2 + ky[:, None] ** 2 + kz[None, :] ** 2)
        idx = np.digitize(kk.ravel(), K_EDGES) - 1
        ok = (idx >= 0) & (idx < len(sums))
        w = np.broadcast_to(weight_z[None, :], kk.shape).ravel()[ok]
        sums += np.bincount(idx[ok], weights=(power[i].ravel()[ok] * w).astype(np.float64), minlength=len(sums))
        counts += np.bincount(idx[ok], weights=w.astype(np.float64), minlength=len(sums))
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan), counts


def face_crossings(mask):
    """Number of cell faces between mask and ~mask inside the array (all three axes)."""
    n = 0
    for axis in range(3):
        a = np.moveaxis(mask, axis, 0)
        for i in range(a.shape[0] - 1):  # plane by plane keeps the temporaries small
            n += int(np.count_nonzero(a[i] != a[i + 1]))
    return n


def measure_checkpoint(step_directory):
    """All structure measurements of one checkpoint (see the module docstring)."""
    t_start = time.time()
    state = _open(step_directory, "primitive_state")
    time_myr = float(_open(step_directory, "time").read().result()) * c3.CODE_TIME_MYR
    num_vars, nx, ny, nz = state.shape
    variables = get_registered_variables(
        SimulationConfig(dimensionality=3, solver_mode=FINITE_DIFFERENCE, dual_energy=(num_vars == 6))
    )
    dx = c3.BOX_SIZE_KPC[0] / nx
    x = (np.arange(nx) + 0.5) * dx - c3.BOX_SIZE_KPC[0] / 2
    z = (np.arange(nz) + 0.5) * dx - c3.BOX_SIZE_KPC[2] / 2
    n_conv = np.float32(c3.CONVERSIONS.total_number_density_per_code_density)
    t_conv = np.float32(c3.CONVERSIONS.kelvin_per_code_specific_energy)

    # slab (both hemispheres) and spectra box index ranges; south is stored mirrored (index 0 = nearest the disk)
    north = np.where((z > SLAB_KPC[0]) & (z < SLAB_KPC[1]))[0]
    south = np.where((z < -SLAB_KPC[0]) & (z > -SLAB_KPC[1]))[0][::-1]
    nzs = len(north)
    assert len(south) == nzs
    box_xy = np.where(np.abs(x) < BOX_HALF_XY_KPC)[0]
    bx0, bx1 = box_xy[0], box_xy[-1] + 1
    zs_slab = z[north]  # |z| of the slab planes
    box_z = np.where((zs_slab > BOX_Z_KPC[0]) & (zs_slab < BOX_Z_KPC[1]))[0]
    bz0, bz1 = box_z[0], box_z[-1] + 1
    nb, nbz = bx1 - bx0, bz1 - bz0

    hemis = {}
    for name in ("north", "south"):
        hemis[name] = {
            "kbin": np.empty((nx, ny, nzs), dtype=np.int8),
            "rho": np.empty((nx, ny, nzs), dtype=np.float32),
            "box": {f: np.empty((nb, nb, nbz), dtype=np.float32) for f in ("logn", "logT", "vx", "vy", "vz")},
            "prof_m": np.zeros(nzs), "prof_mlogT": np.zeros(nzs), "prof_mlogK": np.zeros(nzs),
        }

    chunk_x, chunk_y = state.chunk_layout.read_chunk.shape[1:3]
    tiles = [(slice(i, min(i + chunk_x, nx)), slice(j, min(j + chunk_y, ny)))
             for i in range(0, nx, chunk_x) for j in range(0, ny, chunk_y)]
    zmin, zmax = min(north.min(), south.min()), max(north.max(), south.max()) + 1
    pending = state[:, tiles[0][0], tiles[0][1], zmin:zmax].read()
    for index, (xs, ys) in enumerate(tiles):
        block = pending.result()
        if index + 1 < len(tiles):
            pending = state[:, tiles[index + 1][0], tiles[index + 1][1], zmin:zmax].read()
        for name, kidx in (("north", north - zmin), ("south", south - zmin)):
            h = hemis[name]
            rho = block[variables.density_index][:, :, kidx]
            pressure = block[variables.pressure_index][:, :, kidx]
            temperature = pressure / rho * t_conv
            log_k = np.log10(temperature / (rho * n_conv) ** np.float32(2 / 3))
            h["kbin"][xs, ys] = np.digitize(log_k, c3.ENTROPY_EDGES).astype(np.int8)
            h["rho"][xs, ys] = rho
            m = rho.astype(np.float64)
            h["prof_m"] += m.sum(axis=(0, 1))
            h["prof_mlogT"] += (m * np.log10(temperature)).sum(axis=(0, 1))
            h["prof_mlogK"] += (m * log_k).sum(axis=(0, 1))
            # spectra box part of this tile
            ix = np.arange(xs.start, xs.stop)
            iy = np.arange(ys.start, ys.stop)
            sx = (ix >= bx0) & (ix < bx1)
            sy = (iy >= bx0) & (iy < bx1)
            if sx.any() and sy.any():
                tx = slice(ix[sx][0] - bx0, ix[sx][-1] + 1 - bx0)
                ty = slice(iy[sy][0] - bx0, iy[sy][-1] + 1 - bx0)
                sub = np.ix_(np.where(sx)[0], np.where(sy)[0])
                h["box"]["logn"][tx, ty] = np.log10(rho[sub][:, :, bz0:bz1] * n_conv)
                h["box"]["logT"][tx, ty] = np.log10(temperature[sub][:, :, bz0:bz1])
                for comp, vi in (("vx", variables.velocity_index.x), ("vy", variables.velocity_index.y),
                                 ("vz", variables.velocity_index.z)):
                    h["box"][comp][tx, ty] = block[vi][:, :, kidx][sub][:, :, bz0:bz1]
            del rho, pressure, temperature, log_k, m
        del block
    t_read = time.time() - t_start

    cell_mass_msun = float(dx**3 * c3.CODE_MASS_MSUN)
    out = {"time_myr": time_myr, "dim": nx, "dx_pc": dx * 1e3, "z_slab_kpc": zs_slab, "t_read_s": t_read}
    box_kpc = (nb * dx, nb * dx, nbz * dx)
    for name, h in hemis.items():
        kbin, rho = h["kbin"], h["rho"]
        r = {}
        # clumps of entrained gas
        entrained = kbin == 0
        labels, num = ndimage.label(entrained)
        r["clump_n"] = num
        if num:
            index = np.arange(1, num + 1)
            mass = ndimage.sum_labels(rho, labels, index) * cell_mass_msun
            ncell = np.bincount(labels.ravel(), minlength=num + 1)[1:]
            radius_pc = (3 * ncell * (dx * 1e3) ** 3 / (4 * np.pi)) ** (1 / 3)
            r["clump_mass_hist"] = np.histogram(mass, CLUMP_MASS_EDGES)[0]
            r["clump_radius_hist"] = np.histogram(radius_pc, CLUMP_RADIUS_EDGES)[0]
            r["clump_mass_by_radius"] = np.histogram(radius_pc, CLUMP_RADIUS_EDGES, weights=mass)[0]
            r["clump_mass_total"] = float(mass.sum())
            r["clump_mass_max"] = float(mass.max())
            r["clump_n_1cell"] = int((ncell == 1).sum())
            del mass, ncell, radius_pc, index
        else:
            r["clump_mass_hist"] = np.zeros(len(CLUMP_MASS_EDGES) - 1, dtype=int)
            r["clump_radius_hist"] = np.zeros(len(CLUMP_RADIUS_EDGES) - 1, dtype=int)
            r["clump_mass_by_radius"] = np.zeros(len(CLUMP_RADIUS_EDGES) - 1)
            r["clump_mass_total"] = r["clump_mass_max"] = 0.0
            r["clump_n_1cell"] = 0
        del labels
        # mixing layer: isosurface areas and mixed volume
        r["area_ent_kpc2"] = 2.0 / 3.0 * face_crossings(entrained) * dx**2
        del entrained
        r["area_wind_kpc2"] = 2.0 / 3.0 * face_crossings(kbin <= 1) * dx**2
        mixed = kbin == 1
        r["vol_mixed_kpc3"] = float(np.count_nonzero(mixed)) * dx**3
        r["mass_mixed_msun"] = float(rho[mixed].sum(dtype=np.float64)) * cell_mass_msun
        r["vol_ent_kpc3"] = float(np.count_nonzero(kbin == 0)) * dx**3
        r["mass_ent_msun"] = float(rho[kbin == 0].sum(dtype=np.float64)) * cell_mass_msun
        del mixed
        # spectra
        for f in ("logn", "logT"):
            r[f"pk_{f}"], counts = shell_spectrum(h["box"][f], dx, box_kpc)
            h["box"][f] = None
        pv = None
        for comp in ("vx", "vy", "vz"):
            p, counts = shell_spectrum(h["box"][comp], dx, box_kpc)
            pv = p if pv is None else pv + p
            h["box"][comp] = None
        r["pk_v"] = pv
        r["pk_modes"] = counts
        with np.errstate(invalid="ignore", divide="ignore"):
            r["prof_logT"] = h["prof_mlogT"] / h["prof_m"]
            r["prof_logK"] = h["prof_mlogK"] / h["prof_m"]
        r["prof_mass_msun"] = h["prof_m"] * cell_mass_msun
        out[name] = r
        h["kbin"] = h["rho"] = None
    out["k_nyquist"] = 0.5 / dx
    out["box_kpc"] = box_kpc
    out["t_total_s"] = time.time() - t_start
    return out


def _measure(args):
    step, step_directory = args
    return step, measure_checkpoint(step_directory)


def cache_path(tag, out_dir):
    return os.path.join(out_dir, f"structure3d_v1{tag}.npz")


def load_cache(tag, out_dir):
    path = cache_path(tag, out_dir)
    return np.load(path, allow_pickle=True)["records"].item() if os.path.exists(path) else {}


def save_cache(tag, out_dir, records):
    path = cache_path(tag, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(path + ".tmp.npz", records=np.array(records, dtype=object))
    os.replace(path + ".tmp.npz", path)


def run(tag, checkpoint_dir, steps_spec, workers, out_dir, ready_dir=None):
    available = sorted(int(e) for e in os.listdir(checkpoint_dir) if e.isdigit())
    records = load_cache(tag, out_dir)
    todo = [s for s in c3.parse_steps(steps_spec, available) if s not in records]
    todo = [s for s in todo if os.path.exists(os.path.join(checkpoint_dir, str(s), "_CHECKPOINT_METADATA"))]
    if ready_dir:  # checkpoints still arriving: only those the transfer has verified (marker file <tag>_<step>)
        todo = [s for s in todo if os.path.exists(os.path.join(ready_dir, f"{tag}_{s}"))]
    print(f"{tag}: {len(records)} cached, {len(todo)} to measure from {checkpoint_dir} with {workers} workers",
          flush=True)
    if not todo:
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_measure, (s, os.path.join(checkpoint_dir, str(s)))) for s in todo]
        for future in as_completed(futures):
            step, result = future.result()
            records[step] = result
            save_cache(tag, out_dir, records)
            n = result["north"]
            print(f"  {tag} [{step:3d}] t = {result['time_myr']:6.2f} Myr  clumps N {n['clump_n']}  "
                  f"A(5.5) N {n['area_ent_kpc2']:.1f} kpc^2  read {result['t_read_s']:.0f} s, "
                  f"total {result['t_total_s']:.0f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--steps", default="2-60/2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "data", "studies"))
    parser.add_argument("--ready-dir", default=None, help="only steps with a marker file <tag>_<step> in this dir")
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir or os.path.join(SCRIPT_DIR, "data", f"cgols_checkpoints{args.tag}")
    run(args.tag, checkpoint_dir, args.steps, args.workers, args.out_dir, args.ready_dir)


if __name__ == "__main__":
    main()
