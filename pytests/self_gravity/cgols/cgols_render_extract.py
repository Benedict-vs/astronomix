"""Reduce CGOLS Orbax checkpoints to compact 3D volumes for volume rendering.

Reads the full primitive state of each checkpoint tile by tile (like
cgols_convergence_3d.py), block-averages it by an integer FACTOR and writes
float16 volumes that cgols_render.py ray-marches:

  logn   log10 total number density [cm^-3] of the block-mean density
  logT   log10 mass-weighted temperature [K] = mean(P) / mean(rho) (T ~ P / rho)
  vr     mass-weighted spherical radial velocity [km/s] about the box centre

    data/renders/<tag>_f<factor>/<step>/{logn,logT,vr}.npy + meta.json

A 2048x2048x4096 checkpoint at factor 2 gives 1024x1024x2048 cells, 4.3 GB per
field. CPU only; never imports cgols.py.

    python cgols_render_extract.py --tag _C512 --steps 28 --factor 1
    python cgols_render_extract.py --tag _A2048 --steps 16,28,40,60 --factor 2 --workers 24
"""

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import tensorstore
from astropy import units as u

from astronomix import CodeUnits, SimulationConfig, get_registered_variables
from astronomix.analysis_helpers.emission import emission_conversions
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_UNITS = CodeUnits(1 * u.kiloparsec, 1e6 * u.M_sun, 100 * u.km / u.s)
CONVERSIONS = emission_conversions(CODE_UNITS)
CODE_TIME_MYR = float((1 * CODE_UNITS.code_time).to(u.Myr).value)
KMS_PER_CODE_VELOCITY = 100.0
BOX_SIZE_KPC = (10.0, 10.0, 20.0)
FIELDS = ("logn", "logT", "vr")


def _open(step_directory, name):
    context = tensorstore.Context({"cache_pool": {"total_bytes_limit": 0}})
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "ocdbt", "base": "file://" + os.path.abspath(step_directory) + "/", "path": name},
    }
    return tensorstore.open(spec, context=context, read=True).result()


def _block_sum(a, f):
    """Sum over f x f x f blocks of a 3D array (shape divisible by f)."""
    if f == 1:
        return a
    nx, ny, nz = a.shape
    return a.reshape(nx // f, f, ny // f, f, nz // f, f).sum(axis=(1, 3, 5), dtype=np.float64)


def reduce_tile(step_directory, xs, ys, factor, fields=FIELDS):
    """Block-averaged fields of one (x, y) tile column; returns (xs, ys, dict)."""
    state = _open(step_directory, "primitive_state")
    num_vars, nx, ny, nz = state.shape
    variables = get_registered_variables(
        SimulationConfig(dimensionality=3, solver_mode=FINITE_DIFFERENCE, dual_energy=(num_vars == 6))
    )
    block = state[:, xs, ys, :].read().result()
    rho = block[variables.density_index]
    pressure = block[variables.pressure_index]
    dx = BOX_SIZE_KPC[0] / nx
    x = ((np.arange(xs.start, xs.stop) + 0.5) * dx - BOX_SIZE_KPC[0] / 2).astype(np.float32)
    y = ((np.arange(ys.start, ys.stop) + 0.5) * dx - BOX_SIZE_KPC[1] / 2).astype(np.float32)
    z = ((np.arange(nz) + 0.5) * dx - BOX_SIZE_KPC[2] / 2).astype(np.float32)
    mass = _block_sum(rho, factor)
    out = {}
    if "vr" in fields:
        radius = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2 + z[None, None, :] ** 2)
        radius = np.maximum(radius, np.float32(0.5 * dx))
        momentum_r = rho * (
            block[variables.velocity_index.x] * x[:, None, None]
            + block[variables.velocity_index.y] * y[None, :, None]
            + block[variables.velocity_index.z] * z[None, None, :]
        ) / radius
        del radius
        out["vr"] = _block_sum(momentum_r, factor) / mass * KMS_PER_CODE_VELOCITY
        del momentum_r
    del block
    if "logn" in fields:
        out["logn"] = np.log10(mass / factor**3 * CONVERSIONS.total_number_density_per_code_density)
    if "logT" in fields:
        out["logT"] = np.log10(_block_sum(pressure, factor) / mass * CONVERSIONS.kelvin_per_code_specific_energy)
    return xs, ys, {k: v.astype(np.float16) for k, v in out.items()}


def parse_steps(spec, available):
    if spec in (None, "all"):
        return available
    steps = []
    for part in spec.split(","):
        if "-" in part:
            rng, _, stride = part.partition("/")
            a, b = (int(v) for v in rng.split("-"))
            steps += list(range(a, b + 1, int(stride or 1)))
        else:
            steps.append(int(part))
    return [s for s in steps if s in available]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--steps", default="all")
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--fields", default=",".join(FIELDS), help="subset of logn,logT,vr (movies: logn,logT)")
    args = parser.parse_args()
    fields = tuple(f for f in FIELDS if f in args.fields.split(","))

    checkpoint_dir = args.checkpoint_dir or os.path.join(SCRIPT_DIR, "data", f"cgols_checkpoints{args.tag}")
    out_root = args.out_dir or os.path.join(SCRIPT_DIR, "data", "renders", f"{args.tag.lstrip('_')}_f{args.factor}")
    available = sorted(int(e) for e in os.listdir(checkpoint_dir) if e.isdigit())
    steps = [s for s in parse_steps(args.steps, available)
             if not os.path.exists(os.path.join(out_root, str(s), "meta.json"))]
    print(f"{args.tag}: {len(steps)} checkpoints to extract at factor {args.factor} -> {out_root}", flush=True)

    # spawn, not fork: tensorstore aborts in a forked child once the parent opened a store
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for step in steps:
            step_dir = os.path.join(checkpoint_dir, str(step))
            state = _open(step_dir, "primitive_state")
            _, nx, ny, nz = state.shape
            f = args.factor
            chunk_x, chunk_y = state.chunk_layout.read_chunk.shape[1:3]
            chunk_x, chunk_y = max(chunk_x, f), max(chunk_y, f)
            assert nx % f == 0 and nz % f == 0 and chunk_x % f == 0 and chunk_y % f == 0
            time_myr = float(_open(step_dir, "time").read().result()) * CODE_TIME_MYR
            out_dir = os.path.join(out_root, str(step))
            os.makedirs(out_dir, exist_ok=True)
            shape = (nx // f, ny // f, nz // f)
            volumes = {
                k: np.lib.format.open_memmap(os.path.join(out_dir, f"{k}.npy.tmp"), mode="w+",
                                             dtype=np.float16, shape=shape)
                for k in fields
            }
            tiles = [(slice(i, min(i + chunk_x, nx)), slice(j, min(j + chunk_y, ny)))
                     for i in range(0, nx, chunk_x) for j in range(0, ny, chunk_y)]
            start = time.time()
            futures = [pool.submit(reduce_tile, step_dir, xs, ys, f, fields) for xs, ys in tiles]
            for done, future in enumerate(as_completed(futures), 1):
                xs, ys, reduced = future.result()
                for k, v in reduced.items():
                    volumes[k][xs.start // f: xs.stop // f, ys.start // f: ys.stop // f, :] = v
                if done % max(1, len(tiles) // 8) == 0:
                    print(f"  [{step}] {done}/{len(tiles)} tiles, {time.time() - start:.0f} s", flush=True)
            for k, vol in volumes.items():
                vol.flush()
                os.replace(os.path.join(out_dir, f"{k}.npy.tmp"), os.path.join(out_dir, f"{k}.npy"))
            volumes.clear()
            meta = {"tag": args.tag, "step": step, "time_myr": time_myr, "factor": f, "shape": shape,
                    "source_shape": [nx, ny, nz], "box_kpc": BOX_SIZE_KPC, "dx_kpc": BOX_SIZE_KPC[0] / shape[0],
                    "fields": {k: v for k, v in {"logn": "log10 n_tot [cm^-3]", "logT": "log10 T [K], mass-weighted",
                                                 "vr": "radial velocity [km/s], mass-weighted"}.items() if k in fields}}
            with open(os.path.join(out_dir, "meta.json"), "w") as fh:
                json.dump(meta, fh, indent=1)
            print(f"  [{step}] t = {time_myr:.2f} Myr, {shape}, {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
