"""
Density / pressure floor census of the CGOLS A-series, FV (VL2 + HLLC) vs FD (WENO5).

The FV runs use AthenaPK's hard floor (POSITIVITY_HARD_FLOOR: rho <- max(rho,
1e-4) with momentum and total energy kept, P <- max(P, 1e-5), in every VL2
stage and once more at the end of the step), which adds mass. The FD runs have
no clip: the positivity-preserving WENO limits fluxes against min(floor, rho/2),
so their density can sink below 1e-4 conservatively. This script measures how
often, where and with what effect the floor binds:

  diag   per-step min_rho / min_P statistics of the cgols_diag<tag>.log files
  state  a chunked scan of the final 75 Myr states (data/final/*.npy, primitive
         variables, float32, mmap): cell counts at / near / below the floor and
         their (R, z) distribution, the log rho histogram, the mass in and the
         mass deficit below the floor, the |z| = 5 kpc mass flux carried by
         near-floor cells, and the CGOLS Eq. 17 L_X1 with the near-floor cells
         excluded / at half density
  frames per-frame floor fraction of the x-z density slice and the axis
         "pinch" metric (cgols_fv_riemann.pinch) with the floored cells excluded
  budget the mass budget per run: diag mass vs injection, the z-face outflow of
         the frames and the residual = outflow through the x/y side faces
         (checked against direct side-face fluxes of the checkpoints to < 1%)

Writes data/studies/density_floor_<tag>.npz (state-scan cache) and
figures/cgols/comparisons/fv_density_floor.png.

Usage (CPU only; the emission import needs the autocvd shim):
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=gpuq/autocvd_shim \\
        python cgols_density_floor.py [--workers 64] [--rescan]
"""

# general
import argparse
import os
import re
import multiprocessing

# numerics
import numpy as np

# plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Floors of cgols.build_simulation_params (code units: 1e6 Msun / kpc^3, (100 km/s)^2 * that).
RHO_FLOOR = 1e-4
P_FLOOR = 1e-5
# Conversions as in astronomix.analysis_helpers.emission.emission_conversions (mu = 0.6).
N_PER_RHO = 0.06743808654933546  # n [cm^-3] per code density
T_PER_P_OVER_RHO = 726885.0766661186  # K per code P / rho
CM_PER_KPC = 3.0856775814913673e21
MSUN_YR_PER_CODE_MDOT = 0.1022712165045695
MYR_PER_CODE_TIME = 9.777922216807893
BOX = (10.0, 10.0, 20.0)  # kpc

RUNS = {
    # tag: (label, final state, frame dir)
    "_A1024fv": ("FV-1024", "data/final/cgols_final_state_A1024fv.npy", "cgols_snapshots_A1024fv"),
    "_A1024": ("FD-1024", "data/final/cgols_final_state_A1024.npy", "cgols_snapshots_A1024"),
    "_A512fvf32": ("FV-512", "data/final/cgols_final_state_A512fvf32.npy", "cgols_snapshots_A512fvf32"),
    "_A512": ("FD-512", "data/final/cgols_final_state_A512.npy", "cgols_snapshots_A512"),
}
DIAG_TAGS = ("_A1024fv", "_A1024", "_A512fvf32", "_A512fvf64", "_A512", "_A512fvhlle", "_A256fv", "_A256")

R_EDGES = np.linspace(0.0, 7.1, 72)
Z_EDGES = np.linspace(-10.0, 10.0, 201)
LOGRHO_EDGES = np.linspace(-7.0, 3.5, 526)
LOGT_EDGES = np.linspace(3.0, 10.0, 141)
XRAY_TMIN, XRAY_TMAX = 2.32e6, 1.55e7  # CGOLS 0.3-2 keV window [K]


# -------------------------------------------------------------
# =============== ↓ diag logs ↓ ===============================
# -------------------------------------------------------------

_DIAG = re.compile(r"t=(\S+)\s+min_rho=(\S+)\s+min_P=(\S+)\s+max\|v\|=(\S+).*?mass=(\S+) dM=(\S+)")


def load_diag(tag):
    """(t [Myr], min_rho, min_P, mass) of a run's per-step diag log."""
    for path in (f"cgols_logs/cgols_diag{tag}.log", f"cgols_logs/diagnostics/cgols_diag{tag}.log"):
        if os.path.exists(path):
            break
    else:
        return None
    rows = []
    with open(path) as fh:
        for line in fh:
            m = _DIAG.search(line)
            if m:
                rows.append([float(m.group(i)) for i in (1, 2, 3, 5)])
    a = np.array(rows)
    a[:, 0] *= MYR_PER_CODE_TIME
    return a


def diag_table():
    """Print the floor-hit statistics of every diag log."""
    print("per-step diag: fraction of steps with min_rho <= floor (FV) / < floor (FD), by window")
    print(f"  {'run':12s} {'40-60':>6s} {'60-75':>6s} {'first':>6s} {'min_rho':>9s} {'med60-75':>9s} {'P 60-75':>7s} {'min_P':>9s}")
    for tag in DIAG_TAGS:
        a = load_diag(tag)
        if a is None:
            continue
        t, rho, p = a[:, 0], a[:, 1], a[:, 2]
        hit, phit = rho <= RHO_FLOOR * 1.0001, p <= P_FLOOR * 1.0001
        late, mid = t >= 60, (t >= 40) & (t < 60)
        first = f"{t[hit][0]:.1f}" if hit.any() else "-"
        print(
            f"  {tag:12s} {hit[mid].mean():6.3f} {hit[late].mean():6.3f} {first:>6s} {rho.min():9.2e}"
            f" {np.median(rho[late]):9.2e} {phit[late].mean():7.3f} {p.min():9.2e}"
        )


# -------------------------------------------------------------
# =============== ↓ final-state scan ↓ ========================
# -------------------------------------------------------------

_XRAY_LOGT = None
_XRAY_LAMBDA = None


def xray_table():
    """Lambda_A4(T) of the CGOLS Eq. 17 recipe [1e-30 erg cm^3 / s] on a fine log T grid."""
    from astronomix.analysis_helpers.emission import cgols_xray_emissivity

    log_t = np.linspace(np.log10(XRAY_TMIN), np.log10(XRAY_TMAX), 20001)
    lam = np.asarray(cgols_xray_emissivity(np.ones_like(log_t), 10.0**log_t), dtype=np.float64)
    return log_t, lam


def _open_state(path):
    """The (num_vars, nx, ny, nz) primitive state: a .npy (mmap) or an Orbax checkpoint step directory."""
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    import tensorstore

    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "ocdbt", "base": "file://" + os.path.abspath(path) + "/", "path": "primitive_state"},
    }
    context = tensorstore.Context({"cache_pool": {"total_bytes_limit": 0}})
    return tensorstore.open(spec, context=context, read=True).result()


def checkpoint_time_myr(path):
    """Simulation time [Myr] of an Orbax checkpoint step directory."""
    import tensorstore

    spec = {"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": "file://" + os.path.abspath(path) + "/", "path": "time"}}
    return float(tensorstore.open(spec, read=True).result().read().result()) * MYR_PER_CODE_TIME


def _init_worker(path, log_t, lam):
    global _STATE, _XRAY_LOGT, _XRAY_LAMBDA
    _STATE = _open_state(path)
    _XRAY_LOGT, _XRAY_LAMBDA = log_t, lam


def _scan_slab(bounds):
    """Accumulators of the tile [x0, x1) x [y0, y1) x all z (see scan_state)."""
    x0, x1, y0, y1 = bounds
    _, nx, ny, nz = _STATE.shape
    dx = BOX[0] / nx
    block = _STATE[:, x0:x1, y0:y1]
    block = np.asarray(block if isinstance(block, np.ndarray) else block.read().result(), dtype=np.float32)
    rho, vz, p = block[0], block[3], block[4]
    x = (np.arange(x0, x1) + 0.5) * dx - BOX[0] / 2
    y = (np.arange(y0, y1) + 0.5) * dx - BOX[1] / 2
    z = (np.arange(nz) + 0.5) * dx - BOX[2] / 2
    R = np.broadcast_to(np.hypot(x[:, None], y[None, :])[..., None], rho.shape)
    Z = np.broadcast_to(z[None, None, :], rho.shape)
    T = p / rho * T_PER_P_OVER_RHO
    at_floor = rho < 1.01 * RHO_FLOOR
    near_floor = rho < 2.0 * RHO_FLOOR
    below_floor = rho < 0.9999 * RHO_FLOOR  # FD only (FV clips)
    p_floor = p < 1.01 * P_FLOOR
    out = {}
    out["n_cells"] = rho.size
    out["n_at"] = at_floor.sum()
    out["n_near"] = near_floor.sum()
    out["n_below"] = below_floor.sum()
    out["n_at_pfloor"] = (at_floor & p_floor).sum()
    out["n_pfloor"] = p_floor.sum()
    rz = (R.ravel(), Z.ravel())
    out["h_all"] = np.histogram2d(*rz, bins=(R_EDGES, Z_EDGES))[0]
    out["h_at"] = np.histogram2d(R[at_floor], Z[at_floor], bins=(R_EDGES, Z_EDGES))[0]
    out["h_near"] = np.histogram2d(R[near_floor], Z[near_floor], bins=(R_EDGES, Z_EDGES))[0]
    out["h_logrho"] = np.histogram(np.log10(np.maximum(rho, 1e-30)), bins=LOGRHO_EDGES)[0]
    out["h_logT_at"] = np.histogram(np.log10(T[at_floor]), bins=LOGT_EDGES)[0]
    out["h_logT_near"] = np.histogram(np.log10(T[near_floor]), bins=LOGT_EDGES)[0]
    out["xz_at"] = at_floor.sum(axis=1)  # column counts along y, (x1 - x0, nz)
    out["xz_near"] = near_floor.sum(axis=1)
    # Mass [code], with dV; the deficit is what a floor would add to fill these cells once.
    dv = dx**3
    out["mass"] = rho.sum(dtype=np.float64) * dv
    out["mass_at"] = rho[at_floor].sum(dtype=np.float64) * dv
    out["mass_near"] = rho[near_floor].sum(dtype=np.float64) * dv
    out["deficit"] = np.maximum(RHO_FLOOR - rho, 0.0).sum(dtype=np.float64) * dv
    # Outflow through z = +-5 kpc (signed outward), total and from near / at floor cells.
    kn, ks = np.argmin(np.abs(z - 5.0)), np.argmin(np.abs(z + 5.0))
    flux_n, flux_s = rho[:, :, kn] * vz[:, :, kn], -rho[:, :, ks] * vz[:, :, ks]
    for name, sel_n, sel_s in (
        ("", None, None),
        ("_at", at_floor[:, :, kn], at_floor[:, :, ks]),
        ("_near", near_floor[:, :, kn], near_floor[:, :, ks]),
    ):
        fn = flux_n if sel_n is None else flux_n[sel_n]
        fs = flux_s if sel_s is None else flux_s[sel_s]
        out["mdot5" + name] = (fn.sum(dtype=np.float64) + fs.sum(dtype=np.float64)) * dx * dx
        out["area5" + name] = fn.size + fs.size
    # CGOLS Eq. 17 L_X1 [erg/s]: n^2 Lambda(T) in the 0.3-2 keV window.
    in_window = (T >= XRAY_TMIN) & (T <= XRAY_TMAX)
    lam = np.zeros_like(rho, dtype=np.float64)
    lam[in_window] = np.interp(np.log10(T[in_window]), _XRAY_LOGT, _XRAY_LAMBDA)
    emis = (rho.astype(np.float64) * N_PER_RHO) ** 2 * lam * 1e-30 * (dx * CM_PER_KPC) ** 3
    out["lx"] = emis.sum()
    out["lx_at"] = emis[at_floor].sum()
    out["lx_near"] = emis[near_floor].sum()
    # Axis pinch band (R < 0.5 kpc, 4 < |z| < 9 kpc).
    band = (R < 0.5) & (np.abs(Z) > 4) & (np.abs(Z) < 9)
    out["n_band"] = band.sum()
    out["n_band_at"] = (band & at_floor).sum()
    out["n_band_near"] = (band & near_floor).sum()
    # Near the outer faces (within 4 cells).
    edge = np.zeros_like(at_floor)
    edge[:, :, :4] = edge[:, :, -4:] = True
    if y0 < 4:
        edge[:, : 4 - y0] = True
    if y1 > ny - 4:
        edge[:, ny - 4 - y0 :] = True
    if x0 < 4:
        edge[: 4 - x0] = True
    if x1 > nx - 4:
        edge[nx - 4 - x0 :] = True
    out["n_at_edge"] = (at_floor & edge).sum()
    out["n_near_edge"] = (near_floor & edge).sum()
    # Thermo-kinematics of the at-floor cells.
    speed = np.sqrt(block[1] ** 2 + block[2] ** 2 + vz**2)
    out["sum_speed_at"] = speed[at_floor].sum(dtype=np.float64)
    out["sum_vr_out_at"] = (np.sign(Z[at_floor]) * vz[at_floor] > 0).sum()
    return (x0, x1), out


def scan_state(path, workers, slab):
    """Scan one state (.npy or checkpoint) in tiles; returns the summed accumulators + the xz maps."""
    state = _open_state(path)
    shape = tuple(int(n) for n in state.shape)
    nx, ny, nz = shape[1:]
    if path.endswith(".npy"):
        tile_x, tile_y = slab, ny
    else:  # one task per Orbax chunk, so every chunk is decoded once
        tile_x, tile_y = state.chunk_layout.read_chunk.shape[1:3]
    bounds = [
        (x0, min(x0 + tile_x, nx), y0, min(y0 + tile_y, ny))
        for x0 in range(0, nx, tile_x)
        for y0 in range(0, ny, tile_y)
    ]
    del state
    log_t, lam = xray_table()
    total = {}
    xz_at = np.zeros((nx, nz), dtype=np.int32)
    xz_near = np.zeros((nx, nz), dtype=np.int32)
    context = multiprocessing.get_context("spawn")
    with context.Pool(min(workers, len(bounds)), initializer=_init_worker, initargs=(path, log_t, lam)) as pool:
        for (x0, x1), out in pool.imap_unordered(_scan_slab, bounds):
            xz_at[x0:x1] += out.pop("xz_at")
            xz_near[x0:x1] += out.pop("xz_near")
            for k, v in out.items():
                total[k] = total.get(k, 0) + v
    total = {k: np.asarray(v) for k, v in total.items()}
    total["xz_at"], total["xz_near"], total["shape"] = xz_at, xz_near, np.array(shape)
    return total


def load_or_scan(tag, workers, slab, rescan):
    cache = f"data/studies/density_floor{tag}.npz"
    if os.path.exists(cache) and not rescan:
        return dict(np.load(cache))
    _, path, _ = RUNS[tag]
    if not os.path.exists(path):
        return None
    print(f"scanning {path} ...", flush=True)
    res = scan_state(path, workers, slab)
    np.savez(cache, **res)
    return res


#: Orbax checkpoints scanned for the floor census over time (step = frame index, 1.25 Myr apart).
CHECKPOINTS = {
    "_A1024fv": ("data/cgols_checkpoints_A1024fv", (24, 48, 59, 60)),
    "_A1024": ("data/cgols_checkpoints_A1024", (40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60)),
    "_A512fvf32": ("data/cgols_checkpoints_A512fvf32", (40, 44, 48, 52, 56, 60)),
}


def checkpoint_series(tag, workers, rescan):
    """{time [Myr]: accumulators} of the checkpoints of a run (cached per step)."""
    root, steps = CHECKPOINTS[tag]
    series = {}
    for step in steps:
        path = os.path.join(root, str(step))
        cache = f"data/studies/density_floor{tag}_ck{step:03d}.npz"
        if os.path.exists(cache) and not rescan:
            res = dict(np.load(cache))
        elif os.path.exists(os.path.join(path, "_CHECKPOINT_METADATA")):
            print(f"scanning {path} ...", flush=True)
            res = scan_state(path, workers, None)
            res["time_myr"] = np.asarray(checkpoint_time_myr(path))
            np.savez(cache, **res)
        else:
            continue
        series[float(res["time_myr"])] = res
    return series


def print_state(tag, s):
    label = RUNS[tag][0]
    n = float(s["n_cells"])
    nx = int(s["shape"][1])
    dv = (BOX[0] / nx) ** 3
    print(f"\n== {label} ({tag}) final state, 75 Myr, {nx}x{nx}x{2 * nx}")
    print(
        f"  cells rho<1.01e-4 {int(s['n_at']):>11d} ({s['n_at'] / n:.2e} of volume)"
        f"  rho<2e-4 {int(s['n_near']):>11d} ({s['n_near'] / n:.2e})  rho<1e-4 {int(s['n_below']):>11d} ({s['n_below'] / n:.2e})"
    )
    print(
        f"  at-floor cells also at P floor: {s['n_at_pfloor'] / max(s['n_at'], 1):.2f}; P<1.01e-5 cells {int(s['n_pfloor'])}"
        f"; within 4 cells of a face: at {int(s['n_at_edge'])} near {int(s['n_near_edge'])}"
    )
    if s["n_at"] > 0:
        corner = (R_EDGES[1:, None] > 5.0) & (np.abs(0.5 * (Z_EDGES[1:] + Z_EDGES[:-1]))[None, :] > 6.0)
        print(f"  share in the box corners (R > 5 kpc, |z| > 6 kpc): at floor {s['h_at'][corner].sum() / s['n_at']:.2f},"
              f" rho<2e-4 {s['h_near'][corner].sum() / s['n_near']:.2f}")
        print(
            f"  at-floor cells: mean |v| {100 * s['sum_speed_at'] / s['n_at']:.0f} km/s,"
            f" moving away from midplane {s['sum_vr_out_at'] / s['n_at']:.2f}"
        )
    print(
        f"  mass [1e6 Msun]: total {float(s['mass']):.2f}  in at-floor cells {float(s['mass_at']):.3e}"
        f"  in rho<2e-4 {float(s['mass_near']):.3e}  deficit below floor {float(s['deficit']):.3e}"
        f"  (one floor fill of all at-floor cells: {s['n_at'] * RHO_FLOOR * dv:.3e})"
    )
    m5 = float(s["mdot5"]) * MSUN_YR_PER_CODE_MDOT
    print(
        f"  outflow |z|=5 kpc: {m5:.3f} Msun/yr; carried by rho<1.01e-4 {float(s['mdot5_at']) * MSUN_YR_PER_CODE_MDOT:.2e}"
        f" (area {s['area5_at'] / s['area5']:.2e}), by rho<2e-4 {float(s['mdot5_near']) * MSUN_YR_PER_CODE_MDOT:.2e}"
        f" (area {s['area5_near'] / s['area5']:.2e})"
    )
    lx = float(s["lx"])
    print(
        f"  L_X1 {lx / 1e39:.3f} e39 erg/s; from rho<1.01e-4 {float(s['lx_at']) / lx:.2e}, rho<2e-4 {float(s['lx_near']) / lx:.2e}"
        f"; with rho<2e-4 cells at half density {(lx - 0.75 * float(s['lx_near'])) / 1e39:.3f} e39"
    )
    print(
        f"  axis band R<0.5, 4<|z|<9: {int(s['n_band'])} cells, at floor {s['n_band_at'] / s['n_band']:.3f},"
        f" rho<2e-4 {s['n_band_near'] / s['n_band']:.3f}"
    )


# -------------------------------------------------------------
# =============== ↓ frames ↓ ==================================
# -------------------------------------------------------------


def frame_series(frame_dir):
    """Per frame: time, slice floor fractions (all, axis band), pinch and pinch without floored cells."""
    names = sorted(f for f in os.listdir(frame_dir) if f.startswith("frame_"))
    n_floor = RHO_FLOOR * N_PER_RHO
    rows = []
    for name in names:
        with np.load(os.path.join(frame_dir, name)) as f:
            t, n = float(f["time_myr"]), f["n_xz"].astype(np.float64)
        nx, nz = n.shape
        z = (np.arange(nz) + 0.5) * BOX[2] / nz - BOX[2] / 2
        band = (np.abs(z) > 4) & (np.abs(z) < 9)
        w = max(1, nx // 128)
        axis = n[nx // 2 - w : nx // 2 + w][:, band]
        at = n < 1.01 * n_floor
        pinch = np.log10(axis.mean(0)).mean()
        # Excluding floored cells from the axis average (NaN-safe: a column with only floor cells is dropped).
        masked = np.where(axis < 1.01 * n_floor, np.nan, axis)
        with np.errstate(all="ignore"):
            pinch_ex = np.nanmean(np.log10(np.nanmean(masked, 0)))
        # Floored axis cells halved (a bound on how much lower the unfloored solution could sit).
        pinch_half = np.log10(np.where(axis < 1.01 * n_floor, 0.5 * axis, axis).mean(0)).mean()
        rows.append([t, at.mean(), (axis < 1.01 * n_floor).mean(), (n < 0.9999 * n_floor).mean(), pinch, pinch_ex, pinch_half])
    return np.array(rows)


# -------------------------------------------------------------
# =============== ↓ mass budget ↓ =============================
# -------------------------------------------------------------

#: cgols.build_cgols_wind_params: injection rate knots [Myr, Msun/yr] (step to the low state at 5 Myr).
_INJ_T = (0.0, 5.0, 5.0 + 1e-7, 10.0, 40.0, 45.0, 1e3)
_INJ_R = (0.0, 0.0, 1.5, 12.0, 12.0, 1.5, 1.5)


def mass_budget(tag):
    """(t, M, injected, out through the z faces, residual = out through the x/y side faces) [1e6 Msun] per frame.

    M from the diag log; the z-face outflow integrates the frames' mdot_z of the outermost planes; the side faces
    (also diode) carry the remainder M0 + injected - M - z-out (no floor term: the floor bound is < 0.1).
    """
    rows = []
    frame_dir = RUNS[tag][2]
    for name in sorted(f for f in os.listdir(frame_dir) if f.startswith("frame_")):
        with np.load(os.path.join(frame_dir, name)) as f:
            rows.append((float(f["time_myr"]), float(f["mdot_z"][-1] - f["mdot_z"][0])))
    t, zface = np.array(rows).T
    diag = load_diag(tag)
    mass = np.interp(t, diag[:, 0], diag[:, 3])
    tt = np.linspace(0, 75, 150001)
    rate = np.interp(tt, _INJ_T, _INJ_R)
    injected = np.interp(t, tt, np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(tt))]))
    zout = np.concatenate([[0], np.cumsum(0.5 * (zface[1:] + zface[:-1]) * np.diff(t))])
    return t, mass, injected, zout, diag[0, 3] + injected - mass - zout


# -------------------------------------------------------------
# =============== ↓ figure ↓ ==================================
# -------------------------------------------------------------


def figure(states, frames, series):
    fig = plt.figure(figsize=(17, 14.5))
    gs = fig.add_gridspec(3, 4, height_ratios=(1, 1.15, 0.9))
    colors = {"_A1024fv": "C3", "_A1024": "C0", "_A512fvf32": "C1", "_A512": "C9"}

    # (a) per-step min_rho
    ax = fig.add_subplot(gs[0, 0])
    for tag, ls in (("_A1024fv", "-"), ("_A1024", "-"), ("_A512fvf32", "--"), ("_A512", "--")):
        a = load_diag(tag)
        if a is None:
            continue
        step = max(1, len(a) // 4000)
        ax.semilogy(a[::step, 0], a[::step, 1], ls, color=colors[tag], lw=0.7, label=RUNS[tag][0])
    ax.axhline(RHO_FLOOR, color="k", lw=0.8, ls=":")
    ax.set(xlabel="t [Myr]", ylabel="min rho per step [code]", xlim=(30, 75), ylim=(3e-7, 3e-3),
           title="(a) per-step min rho (diag logs)")
    ax.legend(fontsize=7)

    # (b) log rho histogram of the final states
    ax = fig.add_subplot(gs[0, 1])
    centers = 0.5 * (LOGRHO_EDGES[1:] + LOGRHO_EDGES[:-1])
    for tag, s in states.items():
        ls = "-" if "1024" in tag else "--"
        ax.semilogy(centers + np.log10(N_PER_RHO), s["h_logrho"] / s["n_cells"], ls, color=colors[tag], lw=0.9, label=RUNS[tag][0])
    ax.axvline(np.log10(RHO_FLOOR * N_PER_RHO), color="k", lw=0.8, ls=":")
    ax.set(xlabel="log n [cm$^{-3}$]", ylabel="volume fraction per 0.02 dex", xlim=(-8, -2.5), ylim=(1e-9, 3e-2),
           title="(b) 75 Myr: volume pdf of log n (floor dotted)")
    ax.legend(fontsize=7)

    # (c) pinch with / without floored cells
    ax = fig.add_subplot(gs[0, 2])
    for tag, rows in frames.items():
        ls = "-" if "1024" in tag else "--"
        ax.plot(rows[:, 0], rows[:, 4], ls, color=colors[tag], lw=1, label=f"{RUNS[tag][0]}")
        if "fv" in tag:
            ax.plot(rows[:, 0], rows[:, 6], ":", color=colors[tag], lw=1.2, label=f"{RUNS[tag][0]}, floored cells x0.5")
    ax.axhline(np.log10(RHO_FLOOR * N_PER_RHO), color="k", lw=0.8, ls=":")
    ax.text(41, np.log10(RHO_FLOOR * N_PER_RHO) + 0.05, "log n_floor", fontsize=7)
    ax.set(xlabel="t [Myr]", ylabel="axis <log n>, 4<|z|<9 kpc", xlim=(40, 75), ylim=(-5.4, -2.8),
           title="(c) cavity pinch metric")
    ax.legend(fontsize=6)

    # (d) checkpoint census over time: cells at (FV) / below (FD) the floor, and near it
    ax = fig.add_subplot(gs[0, 3])
    for tag, ser in series.items():
        if not ser:
            continue
        t = np.array(sorted(ser))
        ls = "-" if "1024" in tag else "--"
        key, lab = ("n_at", "rho<1.01e-4") if "fv" in tag else ("n_below", "rho<1e-4")
        n = np.array([float(ser[x][key]) / float(ser[x]["n_cells"]) for x in t])
        near = np.array([float(ser[x]["n_near"]) / float(ser[x]["n_cells"]) for x in t])
        ax.semilogy(t, np.maximum(n, 1e-9), ls, marker="o", ms=3, color=colors[tag], lw=1, label=f"{RUNS[tag][0]} {lab}")
        ax.semilogy(t, np.maximum(near, 1e-9), ls, marker="x", ms=3, color=colors[tag], lw=0.6, alpha=0.6, label=f"{RUNS[tag][0]} rho<2e-4")
    ax.set(xlabel="t [Myr]", ylabel="volume fraction", xlim=(28, 76), ylim=(1e-8, 1e-2),
           title="(d) ckpts: cells at/below/near floor")
    ax.legend(fontsize=6)

    # (e, f) column counts of at-floor (FV) / below-floor-equivalent (FD: rho < 1.01e-4) cells
    for i, tag in enumerate(("_A1024fv", "_A1024")):
        if tag not in states:
            continue
        s = states[tag]
        ax = fig.add_subplot(gs[1, i])
        img = s["xz_at"].T.astype(float)
        img[img == 0] = np.nan
        im = ax.imshow(img, origin="lower", extent=(-5, 5, -10, 10), cmap="viridis", norm=matplotlib.colors.LogNorm(1, max(np.nanmax(img), 10)))
        fig.colorbar(im, ax=ax, label="cells along y with rho < 1.01e-4")
        ax.set(xlabel="x [kpc]", ylabel="z [kpc]", title=f"({'ef'[i]}) {RUNS[tag][0]} 75 Myr: rho<1.01e-4 columns")
        ax.axhline(5, color="w", lw=0.5, ls="--")
        ax.axhline(-5, color="w", lw=0.5, ls="--")

    # (g) (R, z) fraction of cells near the floor, FV-1024
    ax = fig.add_subplot(gs[1, 2])
    s = states["_A1024fv"]
    with np.errstate(all="ignore"):
        frac = s["h_near"] / s["h_all"]
    frac[frac == 0] = np.nan
    im = ax.pcolormesh(R_EDGES, Z_EDGES, frac.T, cmap="magma", norm=matplotlib.colors.LogNorm(1e-5, 1))
    fig.colorbar(im, ax=ax, label="fraction of cells with rho < 2e-4")
    ax.set(xlabel="R [kpc]", ylabel="z [kpc]", title="(g) FV-1024 75 Myr: near-floor fraction in (R, z)")

    # (h) T of the at-floor (FV) / sub-floor (FD) cells: the FV floor heats them
    ax = fig.add_subplot(gs[1, 3])
    tc = 0.5 * (LOGT_EDGES[1:] + LOGT_EDGES[:-1])
    for tag in ("_A1024fv", "_A1024"):
        s = states.get(tag)
        if s is None:
            continue
        ax.step(tc, s["h_logT_at"], where="mid", color=colors[tag], lw=1.2, label=f"{RUNS[tag][0]} rho<1.01e-4")
        ax.step(tc, s["h_logT_near"], where="mid", ls="--", color=colors[tag], lw=0.7, label=f"{RUNS[tag][0]} rho<2e-4")
    ax.axvspan(np.log10(XRAY_TMIN), np.log10(XRAY_TMAX), color="0.9", zorder=0, label="L_X1 window")
    ax.axvline(np.log10(P_FLOOR / RHO_FLOOR * T_PER_P_OVER_RHO), color="k", lw=0.8, ls=":")
    ax.set(xlabel="log T [K]", ylabel="cells", yscale="log", xlim=(4, 8.5), ylim=(0.8, 1e5),
           title="(h) 75 Myr: T of floor cells (dotted: P, rho floored)")
    ax.legend(fontsize=7)

    # (i) cumulative mass budget, (j) FV - FD differences
    budgets = {tag: mass_budget(tag) for tag in states}
    ax = fig.add_subplot(gs[2, 0:2])
    for tag, (t, m, inj, zout, side) in budgets.items():
        ls = "-" if "1024" in tag else "--"
        ax.plot(t, zout, ls, color=colors[tag], lw=0.8)
        ax.plot(t, side, ls, color=colors[tag], lw=1.6, label=RUNS[tag][0])
        ax.plot(t, m - m[0], ls, color=colors[tag], lw=0.8, alpha=0.6)
    ax.plot(t, inj, "k:", lw=1, label="injected")
    ax.text(62, 640, "x/y side faces (residual)", fontsize=8)
    ax.text(62, 125, "z faces", fontsize=8)
    ax.text(62, -230, "M(t) - M(0)", fontsize=8)
    ax.set(xlabel="t [Myr]", ylabel="cumulative [1e6 Msun]", xlim=(0, 75),
           title="(i) mass budget: M0 + injected = M + out(z faces) + out(side faces); floor < 0.1")
    ax.legend(fontsize=7, loc="upper left")
    ax = fig.add_subplot(gs[2, 2:4])
    for fv, fd, ls in (("_A1024fv", "_A1024", "-"), ("_A512fvf32", "_A512", "--")):
        if fv not in budgets or fd not in budgets:
            continue
        (t, m, _, z, sd), (t2, m2, _, z2, sd2) = budgets[fv], budgets[fd]
        res = RUNS[fv][0][3:]
        ax.plot(t, m - np.interp(t, t2, m2), ls, color="k", label=f"{res}: M(FV) - M(FD)")
        ax.plot(t, z - np.interp(t, t2, z2), ls, color="C2", label=f"{res}: z-face outflow FV - FD")
        ax.plot(t, sd - np.interp(t, t2, sd2), ls, color="C4", label=f"{res}: side-face outflow FV - FD")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set(xlabel="t [Myr]", ylabel="[1e6 Msun]", xlim=(0, 75), title="(j) where FV's missing mass goes")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("CGOLS A-series density floor (rho = 1e-4 code, log n = %.2f): FV VL2+HLLC hard floor vs FD WENO5 (no clip)" % np.log10(RHO_FLOOR * N_PER_RHO))
    fig.tight_layout()
    path = "figures/cgols/comparisons/fv_density_floor.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--slab", type=int, default=8, help="x-planes per worker task")
    parser.add_argument("--rescan", action="store_true", help="ignore the state-scan caches")
    parser.add_argument("--no-checkpoints", action="store_true", help="skip the checkpoint census")
    parser.add_argument("--tags", default=",".join(RUNS), help="comma-separated run tags to scan")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    diag_table()
    print(f"\nfloor: rho = {RHO_FLOOR:g} code = n {RHO_FLOOR * N_PER_RHO:.3e} cm^-3 (log n = {np.log10(RHO_FLOOR * N_PER_RHO):.2f});"
          f" rho and P both floored: T = {P_FLOOR / RHO_FLOOR * T_PER_P_OVER_RHO:.3g} K")

    states = {}
    for tag in args.tags.split(","):
        s = load_or_scan(tag, args.workers, args.slab, args.rescan)
        if s is not None:
            states[tag] = s
            print_state(tag, s)

    series = {}
    if not args.no_checkpoints:
        print("\ncheckpoints: t [Myr], cells rho<1.01e-4 / <2e-4 / <1e-4, R>5 kpc share of rho<2e-4, mass in rho<1.01e-4 cells"
              " and deficit below the floor [Msun], L_X1 share of rho<2e-4 cells, |z|=5 kpc outflow share of rho<2e-4 cells")
        for tag in CHECKPOINTS:
            if tag not in states:
                continue
            series[tag] = checkpoint_series(tag, args.workers, args.rescan)
            for t, c in sorted(series[tag].items()):
                n_r5 = c["h_near"][R_EDGES[1:] > 5.0].sum()
                print(
                    f"  {RUNS[tag][0]:8s} {t:5.1f}  {int(c['n_at']):8d} {int(c['n_near']):8d} {int(c['n_below']):8d}"
                    f"  {n_r5 / max(c['n_near'], 1):.2f}  {1e6 * float(c['mass_at']):9.3g} {1e6 * float(c['deficit']):9.3g}"
                    f"  {float(c['lx_near']) / float(c['lx']):.1e}  {float(c['mdot5_near']) / float(c['mdot5']):.1e}"
                    f"  L_X1 {float(c['lx']) / 1e39:.2f}e39"
                )

    frames = {}
    print("\nframes: late (60-75 Myr) means of the slice floor fraction and the pinch metric")
    for tag in states:
        rows = frame_series(RUNS[tag][2])
        frames[tag] = rows
        late = rows[:, 0] >= 60
        print(
            f"  {RUNS[tag][0]:8s} slice at floor {rows[late, 1].mean():.2e}  axis-band at floor {rows[late, 2].mean():.3f}"
            f"  below {rows[late, 3].mean():.2e}  pinch {rows[late, 4].mean():.3f}  excl. floored {np.nanmean(rows[late, 5]):.3f}"
            f"  floored x0.5 {rows[late, 6].mean():.3f}"
        )
    print("\nmass budget [1e6 Msun] at 75 Myr: M, injected, out z faces, out side faces (residual)")
    for tag in states:
        t, m, inj, zout, side = mass_budget(tag)
        print(f"  {RUNS[tag][0]:8s} M {m[-1]:8.2f}  inj {inj[-1]:6.2f}  z-out {zout[-1]:6.2f}  side-out {side[-1]:6.2f}"
              f"   (40 Myr: M {np.interp(40, t, m):8.2f} z-out {np.interp(40, t, zout):6.2f} side-out {np.interp(40, t, side):6.2f})")
    figure(states, frames, series)


if __name__ == "__main__":
    main()
