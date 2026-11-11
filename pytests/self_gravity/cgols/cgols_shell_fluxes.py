"""
Outflow rates through spherical shells, from the full 3D CGOLS checkpoints.

The plane fluxes of cgols_convergence_3d.py integrate rho v_z over a flat
x-y plane at |z| = H. This script measures the radial fluxes the way CGOLS IV
(Schneider et al. 2020, Sec. 5.2) does: for every radial bin [r, r + DR) it
sums over the cells in the bin and divides by the bin width,

    Mdot(r) = sum(m v_r) / DR,   v_r = (x v_x + y v_y + z v_z) / r,

and likewise the radial momentum flux sum((m v_r^2 + P dV)) / DR of the
outflowing gas and the energy flux sum((m v^2 / 2 + gamma P dV / (gamma - 1)) v_r) / DR.
Each sum is binned in

  hemisphere  z > 0 | z < 0,
  radius      DR = 0.125 kpc (CGOLS IV's bin) from 0 to the box corner,
  angle       theta from the nearer pole (0 = along the minor axis,
              90 = in the disk plane) in 10 degree bins, so that CGOLS IV's
              30 degree bicone and the full sphere are both sums of bins,
  phase       cool (T < 2e4 K) | intermediate | hot (T > 5e5 K), CGOLS IV's
              cuts (not the 3e4 K of cgols_convergence_3d.py),
  and again   by entropy (the log K bins of cgols_convergence_3d.py),

together with the gas mass and the cell volume in every bin. The volume tells
how much of each shell bin lies inside the 10 x 10 x 20 kpc box: the full
sphere is complete only to r = 5 kpc, the 30 degree bicone to r = 10 kpc.

It also measures what leaves the box: the outward mass flux through the four
side walls (outermost cell layer, per z cell) and through the top and bottom
faces, net and the unbound part (outflowing with Bernoulli parameter
v^2 / 2 + gamma P / ((gamma - 1) rho) + Phi > 0 in the static disk + NFW
potential of cgols.py), by entropy bin. The diode boundaries never let gas
back in, so this is the box's true mass loss.

The cache is data/studies/shells_v2<tag>.npz (resumable, per checkpoint).

    python cgols_shell_fluxes.py --tag _A512 [--steps 2-60/2] [--workers 16] [--parts 1]
    python cgols_shell_fluxes.py --plot A128=_A128 A256=_A256 A512=_A512

CPU only; like cgols_convergence_3d.py it never imports cgols.py.
"""

# ruff: noqa: E402

# general
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# numerics
import numpy as np

# shared checkpoint reading, units and entropy bins
from cgols_convergence_3d import (
    BOX_SIZE_KPC,
    CODE_MASS_MSUN,
    CONVERSIONS,
    ENTROPY_BINS,
    ENTROPY_EDGES,
    ERG_PER_S,
    GAMMA,
    MOMENTUM_PER_YR,
    MSUN_PER_YR,
    SCRIPT_DIR,
    _open,
    parse_steps,
)

from astronomix import SimulationConfig, get_registered_variables
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE

DR_KPC = 0.125
R_MAX_KPC = float(np.sqrt(np.sum(np.square(np.array(BOX_SIZE_KPC) / 2))))  # box corner
NUM_R = int(np.ceil(R_MAX_KPC / DR_KPC))
R_EDGES_KPC = np.arange(NUM_R + 1) * DR_KPC
THETA_STEP_DEG = 10.0
NUM_THETA = int(90 / THETA_STEP_DEG)
THETA_EDGES_DEG = np.arange(NUM_THETA + 1) * THETA_STEP_DEG
PHASE_EDGES_K = (2e4, 5e5)  # CGOLS IV
PHASES = ("cool", "intermediate", "hot")
NUM_P, NUM_K = len(PHASES), len(ENTROPY_BINS)
SHAPE = (2, NUM_R, NUM_THETA)  # hemisphere (0 = north), radius, angle
SLAB_CELLS = 2**20  # ~1e6 cells per slab: small temporaries are reused, not page-faulted in (13x faster)

PHASE_KEYS = ("mdot_net", "mdot_out", "pdot_out", "edot_net", "edot_out", "mass", "volume")
ENTROPY_KEYS = ("mdot_net_K", "mdot_out_K", "edot_net_K", "mass_K")

# Static potential of cgols.py (Miyamoto-Nagai stellar disk + NFW halo); must match it.
G_KPC_KMS2_PER_MSUN = 4.30091e-6
M_DISK_MSUN, R_DISK_KPC, Z_DISK_KPC = 1e10, 0.8, 0.15
M_HALO_MSUN, R_HALO_KPC, C_HALO = 5e10, 5.3, 10.0
CODE_VELOCITY_KMS = 100.0


def potential_code(R, z):
    """Static potential [code velocity^2] at cylindrical R, height z [kpc]."""
    r = np.sqrt(R**2 + z**2)
    phi_disk = -G_KPC_KMS2_PER_MSUN * M_DISK_MSUN / np.sqrt(R**2 + (R_DISK_KPC + np.sqrt(z**2 + Z_DISK_KPC**2)) ** 2)
    phi_halo = (-G_KPC_KMS2_PER_MSUN * M_HALO_MSUN / (r * (np.log(1 + C_HALO) - C_HALO / (1 + C_HALO)))
                * np.log(1 + r / R_HALO_KPC))
    return ((phi_disk + phi_halo) / CODE_VELOCITY_KMS**2).astype(np.float32)


def _face_fluxes(out, face, normal, along, coord_normal, z, variables, dx):
    """Add the outward fluxes through one box face, given its outermost cell layer.

    face: (num_vars, n_along, nz) for a side wall, normal: index of the normal
    velocity component (0, 1) and its sign; along: the coordinate along the wall.
    """
    axis, sign = normal
    rho, pressure = face[variables.density_index], face[variables.pressure_index]
    v = [face[variables.velocity_index.x], face[variables.velocity_index.y], face[variables.velocity_index.z]]
    v_n = sign * v[axis]
    R = np.sqrt(along[:, None] ** 2 + coord_normal**2)
    bernoulli = 0.5 * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + GAMMA / (GAMMA - 1) * pressure / rho + potential_code(R, z[None, :])
    temperature = pressure / rho * np.float32(CONVERSIONS.kelvin_per_code_specific_energy)
    kbin = np.digitize(np.log10(temperature / (rho * np.float32(CONVERSIONS.total_number_density_per_code_density))
                                ** np.float32(2 / 3)), ENTROPY_EDGES)
    flux = (rho * v_n).astype(np.float64) * dx**2
    unbound = np.where((v_n > 0) & (bernoulli > 0), flux, 0.0)
    energy = ((0.5 * rho * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + GAMMA / (GAMMA - 1) * pressure) * v_n).astype(np.float64) * dx**2
    for b in range(NUM_K):
        sel = kbin == b
        out["side_mdot_net_z"][b] += np.where(sel, flux, 0).sum(axis=0)
        out["side_mdot_unbound_z"][b] += np.where(sel, unbound, 0).sum(axis=0)
    out["side_edot_net_z"] += energy.sum(axis=0)


def _cap_fluxes(out, layer, sign, x, y, z_face, variables, dx):
    """Add the outward fluxes through the top (sign +1) or bottom (-1) face from its cell layer (num_vars, nx, ny)."""
    rho, pressure = layer[variables.density_index], layer[variables.pressure_index]
    v = [layer[variables.velocity_index.x], layer[variables.velocity_index.y], layer[variables.velocity_index.z]]
    v_n = sign * v[2]
    R = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    bernoulli = 0.5 * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + GAMMA / (GAMMA - 1) * pressure / rho + potential_code(R, z_face)
    temperature = pressure / rho * np.float32(CONVERSIONS.kelvin_per_code_specific_energy)
    kbin = np.digitize(np.log10(temperature / (rho * np.float32(CONVERSIONS.total_number_density_per_code_density))
                                ** np.float32(2 / 3)), ENTROPY_EDGES)
    flux = (rho * v_n).astype(np.float64) * dx**2
    unbound = np.where((v_n > 0) & (bernoulli > 0), flux, 0.0)
    energy = ((0.5 * rho * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + GAMMA / (GAMMA - 1) * pressure) * v_n).astype(np.float64) * dx**2
    for b in range(NUM_K):
        sel = kbin == b
        out["cap_mdot_net"][b] += flux[sel].sum()
        out["cap_mdot_unbound"][b] += unbound[sel].sum()
    out["cap_edot_net"] += energy.sum()


# -------------------------------------------------------------
# ===================== ↓ Measuring ↓ =========================
# -------------------------------------------------------------


def _accumulate(out, block, x, y, z, variables, cell_volume, size):
    """Add the shell sums of one (x, y, z) block of the primitive state to `out`."""
    rho = block[variables.density_index]
    vx = block[variables.velocity_index.x]
    vy = block[variables.velocity_index.y]
    vz = block[variables.velocity_index.z]
    pressure = block[variables.pressure_index]
    del block

    X, Y, Z = x[:, None, None], y[None, :, None], z[None, None, :]
    r = np.sqrt(X**2 + Y**2 + Z**2)
    v_r = (X * vx + Y * vy + Z * vz) / r
    cos_theta = np.abs(Z) / r
    theta_bin = np.minimum(
        (np.degrees(np.arccos(np.clip(cos_theta, 0, 1))) / THETA_STEP_DEG).astype(np.int32), NUM_THETA - 1
    )
    r_bin = np.minimum((r / DR_KPC).astype(np.int32), NUM_R - 1)
    hemi = np.broadcast_to((Z < 0).astype(np.int32), r.shape)
    geo = ((hemi * NUM_R + r_bin) * NUM_THETA + theta_bin).ravel()
    del r, cos_theta, theta_bin, r_bin, hemi

    temperature = pressure / rho * np.float32(CONVERSIONS.kelvin_per_code_specific_energy)
    phase = np.digitize(temperature, PHASE_EDGES_K).astype(np.int32).ravel()
    log_entropy = np.log10(
        temperature / (rho * np.float32(CONVERSIONS.total_number_density_per_code_density)) ** np.float32(2 / 3)
    )
    kbin = np.digitize(log_entropy, ENTROPY_EDGES).astype(np.int32).ravel()
    del temperature, log_entropy

    mass = (rho * np.float32(cell_volume)).ravel()
    v_r = v_r.ravel()
    v2 = (vx**2 + vy**2 + vz**2).ravel()
    p_vol = (pressure * np.float32(cell_volume)).ravel()
    del rho, vx, vy, vz, pressure
    outflowing = v_r > 0
    mass_flux = mass * v_r
    energy_flux = (0.5 * mass * v2 + GAMMA / (GAMMA - 1) * p_vol) * v_r
    weights = {
        "mdot_net": mass_flux,
        "mdot_out": np.where(outflowing, mass_flux, 0),
        "pdot_out": np.where(outflowing, mass * v_r**2 + p_vol, 0),
        "edot_net": energy_flux,
        "edot_out": np.where(outflowing, energy_flux, 0),
        "mass": mass,
        "volume": None,
    }
    index_p = geo * NUM_P + phase
    index_k = geo * NUM_K + kbin
    for key, w in weights.items():
        out[key] += np.bincount(index_p, weights=None if w is None else w.astype(np.float64), minlength=size * NUM_P)
    for key, w in (("mdot_net_K", mass_flux), ("mdot_out_K", weights["mdot_out"]),
                   ("edot_net_K", energy_flux), ("mass_K", mass)):
        out[key] += np.bincount(index_k, weights=w.astype(np.float64), minlength=size * NUM_K)
    del weights, mass_flux, energy_flux, mass, v_r, v2, p_vol, geo, phase, kbin, index_p, index_k


def measure_checkpoint(step_directory, part=0, parts=1):
    """Shell sums of one checkpoint (see the module docstring); `part` of `parts` tile groups."""
    state = _open(step_directory, "primitive_state")
    time_myr = float(_open(step_directory, "time").read().result())
    num_vars, nx, ny, nz = state.shape
    variables = get_registered_variables(
        SimulationConfig(dimensionality=3, solver_mode=FINITE_DIFFERENCE, dual_energy=(num_vars == 6))
    )
    dx = BOX_SIZE_KPC[0] / nx
    x = ((np.arange(nx) + 0.5) * dx - BOX_SIZE_KPC[0] / 2).astype(np.float32)
    y = ((np.arange(ny) + 0.5) * dx - BOX_SIZE_KPC[1] / 2).astype(np.float32)
    z = ((np.arange(nz) + 0.5) * dx - BOX_SIZE_KPC[2] / 2).astype(np.float32)
    cell_volume = dx**3
    size = int(np.prod(SHAPE))
    out = {key: np.zeros(size * NUM_P) for key in PHASE_KEYS}
    out.update({key: np.zeros(size * NUM_K) for key in ENTROPY_KEYS})
    out.update({"side_mdot_net_z": np.zeros((NUM_K, nz)), "side_mdot_unbound_z": np.zeros((NUM_K, nz)),
                "side_edot_net_z": np.zeros(nz), "cap_mdot_net": np.zeros(NUM_K),
                "cap_mdot_unbound": np.zeros(NUM_K), "cap_edot_net": np.zeros(())})

    chunk_x, chunk_y = state.chunk_layout.read_chunk.shape[1:3]
    tiles = [
        (slice(i, min(i + chunk_x, nx)), slice(j, min(j + chunk_y, ny)))
        for i in range(0, nx, chunk_x)
        for j in range(0, ny, chunk_y)
    ][part::parts]
    start = time.time()
    pending = state[:, tiles[0][0], tiles[0][1], :].read() if tiles else None
    for index, (xs, ys) in enumerate(tiles):
        block = pending.result()
        if index + 1 < len(tiles):
            pending = state[:, tiles[index + 1][0], tiles[index + 1][1], :].read()
        # Box faces touched by this tile (outermost interior cell layer, outward normal).
        half_x, half_y = abs(float(x[0])), abs(float(y[0]))  # outermost cell centres
        if xs.start == 0:
            _face_fluxes(out, block[:, 0], (0, -1), y[ys], half_x, z, variables, dx)
        if xs.stop == nx:
            _face_fluxes(out, block[:, -1], (0, 1), y[ys], half_x, z, variables, dx)
        if ys.start == 0:
            _face_fluxes(out, block[:, :, 0], (1, -1), x[xs], half_y, z, variables, dx)
        if ys.stop == ny:
            _face_fluxes(out, block[:, :, -1], (1, 1), x[xs], half_y, z, variables, dx)
        _cap_fluxes(out, block[..., -1], 1, x[xs], y[ys], z[-1], variables, dx)
        _cap_fluxes(out, block[..., 0], -1, x[xs], y[ys], z[0], variables, dx)
        # Thin z-slabs keep the float temporaries small (a 512^3 read chunk is 256 x 512 x 1024 cells).
        slab = max(1, SLAB_CELLS // (block.shape[1] * block.shape[2]))
        for z0 in range(0, nz, slab):
            _accumulate(out, block[..., z0:z0 + slab], x[xs], y[ys], z[z0:z0 + slab], variables, cell_volume, size)
        del block

    print(f"    {os.path.basename(step_directory)} part {part + 1}/{parts}: {len(tiles)} tiles "
          f"in {time.time() - start:.0f} s", flush=True)
    out["volume"] *= cell_volume
    for key in ("mdot_net", "mdot_out", "mdot_net_K", "mdot_out_K"):
        out[key] *= MSUN_PER_YR / DR_KPC
    out["pdot_out"] *= MOMENTUM_PER_YR / DR_KPC
    for key in ("edot_net", "edot_out", "edot_net_K"):
        out[key] *= ERG_PER_S / DR_KPC
    for key in ("mass", "mass_K"):
        out[key] *= CODE_MASS_MSUN
    for key in ("side_mdot_net_z", "side_mdot_unbound_z", "cap_mdot_net", "cap_mdot_unbound"):
        out[key] *= MSUN_PER_YR
    for key in ("side_edot_net_z", "cap_edot_net"):
        out[key] *= ERG_PER_S
    for key in PHASE_KEYS:
        out[key] = out[key].reshape(*SHAPE, NUM_P)
    for key in ENTROPY_KEYS:
        out[key] = out[key].reshape(*SHAPE, NUM_K)
    from cgols_convergence_3d import CODE_TIME_MYR

    out["time_myr"] = time_myr * CODE_TIME_MYR
    out["dim"] = nx
    return out


def _measure(args):
    step, step_directory, part, parts = args
    start = time.time()
    return step, part, measure_checkpoint(step_directory, part, parts), time.time() - start


_PER_SNAPSHOT = ("time_myr", "dim")


def merge_parts(results):
    merged = dict(results[0])
    for result in results[1:]:
        for key, value in result.items():
            if key not in _PER_SNAPSHOT:
                merged[key] = merged[key] + value
    return merged


def cache_path(tag, suffix=""):
    return os.path.join(SCRIPT_DIR, "data", "studies", f"shells_v2{tag}{suffix}.npz")


def load_cache(tag, suffix=""):
    path = cache_path(tag, suffix)
    return np.load(path, allow_pickle=True)["records"].item() if os.path.exists(path) else {}


def save_cache(tag, records, suffix=""):
    path = cache_path(tag, suffix)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", records=np.array(records, dtype=object))
    os.replace(path + ".tmp.npz", path)


def run(tag, checkpoint_dir, steps_spec, workers, parts=1):
    checkpoint_dir = checkpoint_dir or os.path.join(SCRIPT_DIR, "data", f"cgols_checkpoints{tag}")
    available = sorted(int(e) for e in os.listdir(checkpoint_dir) if e.isdigit())
    records = load_cache(tag)
    todo = [s for s in parse_steps(steps_spec, available) if s not in records]
    todo = [s for s in todo if os.path.exists(os.path.join(checkpoint_dir, str(s), "_CHECKPOINT_METADATA"))]
    done = {key: value for key, value in load_cache(tag, "_parts").items() if value["parts"] == parts}
    print(f"{tag}: {len(records)} cached, {len(todo)} to measure from {checkpoint_dir}", flush=True)
    for step in todo:
        if all((step, p) in done for p in range(parts)):
            records[step] = merge_parts([done[(step, p)]["result"] for p in range(parts)])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(s, os.path.join(checkpoint_dir, str(s)), p, parts)
                for s in todo if s not in records for p in range(parts) if (s, p) not in done]
        futures = [pool.submit(_measure, job) for job in jobs]
        for future in as_completed(futures):
            step, part, result, seconds = future.result()
            if parts > 1:
                done[(step, part)] = {"result": result, "parts": parts}
                save_cache(tag, done, "_parts")
                if not all((step, p) in done for p in range(parts)):
                    continue
                result = merge_parts([done[(step, p)]["result"] for p in range(parts)])
            records[step] = result
            save_cache(tag, records)
            i5 = int(5.0 / DR_KPC)
            print(f"  [{step:3d}] t = {result['time_myr']:6.2f} Myr  Mdot(r = 5 kpc, 4pi) = "
                  f"{result['mdot_net'][:, i5].sum():6.2f} Msun/yr  ({seconds:.0f} s)", flush=True)
    save_cache(tag, records)


# -------------------------------------------------------------
# ===================== ↓ Reductions ↓ ========================
# -------------------------------------------------------------


def shell_volume(theta_max_deg=90.0):
    """Analytic volume per (hemisphere, radius bin) of the polar cap theta < theta_max, kpc^3."""
    cap = 2 * np.pi * (1 - np.cos(np.radians(theta_max_deg)))
    return cap * np.diff(R_EDGES_KPC**3) / 3


def reduce(record, key, theta_max_deg=90.0, hemispheres=(0, 1)):
    """Sum of `key` over theta < theta_max and the given hemispheres: (radius, phase-or-entropy)."""
    n_theta = int(round(theta_max_deg / THETA_STEP_DEG))
    return record[key][list(hemispheres), :, :n_theta].sum(axis=(0, 2))


def coverage(record, theta_max_deg=90.0):
    """Fraction of the analytic shell-bin volume (both hemispheres) that lies inside the box."""
    return reduce(record, "volume", theta_max_deg).sum(axis=-1) / (2 * shell_volume(theta_max_deg))


def r_centers():
    return 0.5 * (R_EDGES_KPC[1:] + R_EDGES_KPC[:-1])


# -------------------------------------------------------------
# ===================== ↓ Plotting ↓ ==========================
# -------------------------------------------------------------

HIGH, LOW = (15.0, 40.0), (50.0, 75.0)  # Myr: high and low state of the feedback schedule
CONES = ((30, r"30$\degree$ bicone"), (60, r"60$\degree$ bicone"), (90, r"$4\pi$ shell"))
ENTROPY_COLORS = {"entrained": "tab:brown", "mixed": "tab:purple", "wind": "tab:orange", "halo": "0.5"}


def complete(theta_max_deg=90.0):
    """Radius bins whose cap theta < theta_max lies wholly inside the box (|x|, |y| < 5, |z| < 10 kpc)."""
    r_outer = R_EDGES_KPC[1:]
    sin_max = np.sin(np.radians(theta_max_deg))
    return (r_outer * sin_max <= BOX_SIZE_KPC[0] / 2 + 1e-9) & (r_outer <= BOX_SIZE_KPC[2] / 2 + 1e-9)


def solid_angle_bins():
    """Solid angle of every theta bin, both hemispheres together."""
    return 2 * 2 * np.pi * -np.diff(np.cos(np.radians(THETA_EDGES_DEG)))


def window_mean(rec, window, fn):
    steps = [s for s in sorted(rec) if window[0] <= rec[s]["time_myr"] <= window[1]]
    return np.mean([fn(rec[s]) for s in steps], axis=0)


def series(rec, fn):
    steps = sorted(rec)
    return np.array([rec[s]["time_myr"] for s in steps]), np.array([fn(rec[s]) for s in steps])


def _plane_cache(tag):
    """The cgols_convergence_3d plane fluxes of the same run, if cached."""
    from cgols_convergence_3d import HEIGHTS_KPC, load_cache as load_planes

    return HEIGHTS_KPC, load_planes(tag)


def plot(runs, prefix="a_series"):
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(SCRIPT_DIR, "figures", "cgols", "comparisons")
    os.makedirs(fig_dir, exist_ok=True)
    data = {label: (tag, load_cache(tag)) for label, tag in runs}
    data = {label: v for label, v in data.items() if v[1]}
    labels = list(data)
    colors = dict(zip(labels, plt.get_cmap("Blues")(np.linspace(0.4, 1.0, len(labels)))))
    best = labels[-1]  # the highest resolution, last on the command line
    rc = r_centers()
    i_of = lambda r: int(round(r / DR_KPC)) - 1  # the bin [r - DR, r): the full sphere at 5 kpc stays in the box

    # 1) Radial profiles of Mdot, pdot, Edot: bicones and sphere, every resolution, both states.
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    quantities = (("mdot_net", 1, r"net $\dot M(r)$ [$M_\odot$ yr$^{-1}$]"),
                  ("pdot_out", 1e3, r"outward $\dot p(r)$ [$10^3\,M_\odot$ km s$^{-1}$ yr$^{-1}$]"),
                  ("edot_net", 1e42, r"net $\dot E(r)$ [$10^{42}$ erg s$^{-1}$]"))
    for col, (theta, name) in enumerate(CONES):
        ok = complete(theta)
        for label in labels:
            rec = data[label][1]
            for window, ls in ((HIGH, "-"), (LOW, "--")):
                for row, (key, unit, _) in enumerate(quantities):
                    prof = window_mean(rec, window, lambda r_: reduce(r_, key, theta).sum(-1)) / unit
                    ax = axes[row, col]
                    ax.plot(rc[ok], prof[ok], ls, color=colors[label], lw=1.3,
                            label=f"{label}, {window[0]:.0f}-{window[1]:.0f} Myr" if row == 0 else None)
                    ax.plot(rc[~ok], prof[~ok], ls, color=colors[label], lw=0.8, alpha=0.3)
        axes[0, col].set_title(name)
        axes[-1, col].set_xlabel("r [kpc]")
    for row, (_, _, ylabel) in enumerate(quantities):
        axes[row, 0].set_ylabel(ylabel)
        for ax in axes[row]:
            ax.axhline(0, color="0.6", lw=0.8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Outflow rates through spherical shells (time means; faint: the shell leaves the box)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{prefix}_shells_profiles.png"), dpi=130)
    plt.close(fig)

    # 2) Angular distribution dMdot/dOmega at r = 1, 2, 3, 5 kpc by entropy, high state.
    radii = (1.0, 2.0, 3.0, 5.0)
    omega = solid_angle_bins()
    theta_mid = 0.5 * (THETA_EDGES_DEG[1:] + THETA_EDGES_DEG[:-1])
    fig, axes = plt.subplots(2, len(radii), figsize=(16, 7.5), sharex=True)
    for col, radius in enumerate(radii):
        i = i_of(radius)
        for label in labels:
            rec = data[label][1]
            total = window_mean(rec, HIGH, lambda r_: r_["mdot_net"][:, i].sum(axis=(0, 2)))
            axes[0, col].plot(theta_mid, total / omega, "o-", ms=3, color=colors[label], label=label)
            if label == best:
                by_k = window_mean(rec, HIGH, lambda r_: r_["mdot_net_K"][:, i].sum(axis=0))  # (theta, K)
                for b, name in enumerate(ENTROPY_BINS):
                    if np.abs(by_k[:, b]).max() > 0.02 * np.abs(total).max():
                        axes[1, col].plot(theta_mid, by_k[:, b] / omega, "o-", ms=3,
                                          color=ENTROPY_COLORS[name], label=name)
                axes[1, col].plot(theta_mid, total / omega, "k-", lw=0.8, label="total")
        axes[0, col].set_title(f"r = {radius:g} kpc")
        axes[1, col].set_xlabel(r"$\theta$ from the minor axis [deg]")
        for ax in axes[:, col]:
            ax.axhline(0, color="0.6", lw=0.8)
            ax.axvline(30, color="0.8", lw=0.8, ls=":")
    axes[0, 0].set_ylabel(r"$d\dot M / d\Omega$ [$M_\odot$ yr$^{-1}$ sr$^{-1}$]")
    axes[1, 0].set_ylabel(f"{best}, by entropy " + r"[$M_\odot$ yr$^{-1}$ sr$^{-1}$]")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f"Angular distribution of the net mass flux through spherical shells, "
                 f"{HIGH[0]:.0f}-{HIGH[1]:.0f} Myr mean (isotropic 12 M$_\\odot$/yr = {12 / (4 * np.pi):.2f} per sr)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{prefix}_shells_angular.png"), dpi=130)
    plt.close(fig)

    # 3) Shells vs planes at 5 kpc (and 2 kpc) against time.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for col, radius in enumerate((2.0, 5.0)):
        i = i_of(radius)
        for label in labels:
            tag, rec = data[label]
            c = colors[label]
            t, sphere = series(rec, lambda r_: reduce(r_, "mdot_net", 90)[i].sum())
            _, cone60 = series(rec, lambda r_: reduce(r_, "mdot_net", 60)[i].sum())
            _, wind = series(rec, lambda r_: r_["mdot_net_K"][:, i, :, ENTROPY_BINS.index("wind")].sum())
            _, e_sphere = series(rec, lambda r_: reduce(r_, "edot_net", 90)[i].sum())
            axes[0, col].plot(t, sphere, "-", color=c, lw=1.2, label=f"{label} $4\\pi$ shell" if col == 0 else None)
            axes[0, col].plot(t, cone60, ":", color=c, lw=1.2, label=f"{label} 60$\\degree$ bicone" if col == 0 else None)
            axes[1, col].plot(t, e_sphere / 1e42, "-", color=c, lw=1.2)
            heights, planes = _plane_cache(tag)
            if planes and radius in heights:
                h = int(np.argmin(np.abs(heights - radius)))
                tp, plane = series(planes, lambda r_: r_["mdot_net"][h].sum())
                _, e_plane = series(planes, lambda r_: r_["edot_net"][h].sum())
                axes[0, col].plot(tp, plane, "--", color=c, lw=1.0, label=f"{label} planes |z| = r" if col == 0 else None)
                axes[1, col].plot(tp, e_plane / 1e42, "--", color=c, lw=1.0)
            if label == best:
                axes[0, col].plot(t, wind, "-", color="tab:orange", lw=1.0,
                                  label=f"{label} $4\\pi$ shell, wind entropy only" if col == 0 else None)
        axes[0, col].set_title(f"r = |z| = {radius:g} kpc")
        axes[1, col].set_xlabel("t [Myr]")
        for ax in axes[:, col]:
            ax.axhline(0, color="0.6", lw=0.8)
    axes[0, 0].set_ylabel(r"net $\dot M$ [$M_\odot$ yr$^{-1}$]")
    axes[1, 0].set_ylabel(r"net $\dot E$ [$10^{42}$ erg s$^{-1}$]  (solid $4\pi$ shell, dashed planes)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Spherical shells vs the two x-y planes")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{prefix}_shells_vs_planes.png"), dpi=130)
    plt.close(fig)


    # 4) What leaves the box, against time: side walls vs top/bottom faces vs the |z| = 5 kpc planes.
    sched_t = np.array([0.0, 5.0, 5.0, 10.0, 40.0, 45.0, 75.0])
    sched_mdot = np.array([0.0, 0.0, 1.5, 12.0, 12.0, 1.5, 1.5])
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(5.2 * (len(labels) + 1), 4.8))
    for ax in axes[1:-1]:
        ax.sharey(axes[0])
    i5 = i_of(5.0)
    for ax, label in zip(axes, labels):
        tag, rec = data[label]
        nz = next(iter(rec.values()))["side_mdot_net_z"].shape[1]
        z = (np.arange(nz) + 0.5) * BOX_SIZE_KPC[2] / nz - BOX_SIZE_KPC[2] / 2
        low = np.abs(z) < 5.0
        t, side_low = series(rec, lambda r_: r_["side_mdot_net_z"][:, low].sum())
        _, side_high = series(rec, lambda r_: r_["side_mdot_net_z"][:, ~low].sum())
        _, caps = series(rec, lambda r_: r_["cap_mdot_net"].sum())
        _, unbound = series(rec, lambda r_: r_["side_mdot_unbound_z"].sum() + r_["cap_mdot_unbound"].sum())
        _, sphere = series(rec, lambda r_: reduce(r_, "mdot_net", 90)[i5].sum())
        ax.fill_between(t, 0, side_low, color="tab:purple", alpha=0.35, lw=0, label="side walls, |z| < 5 kpc")
        ax.fill_between(t, side_low, side_low + side_high, color="tab:orange", alpha=0.35, lw=0,
                        label="side walls, |z| > 5 kpc")
        ax.fill_between(t, side_low + side_high, side_low + side_high + caps, color="tab:red", alpha=0.35, lw=0,
                        label="top + bottom faces")
        ax.plot(t, unbound, "k:", lw=1.0, label="of which unbound (Bernoulli > 0)")
        ax.plot(t, sphere, "-", color="tab:green", lw=1.3, label=r"$4\pi$ shell r = 5 kpc (net)")
        heights, planes = _plane_cache(tag)
        if planes:
            h = int(np.argmin(np.abs(heights - 5.0)))
            tp, plane = series(planes, lambda r_: r_["mdot_net"][h].sum())
            ax.plot(tp, plane, "-", color="tab:blue", lw=1.3, label="planes |z| = 5 kpc (net)")
        ax.plot(sched_t, sched_mdot, color="0.4", lw=1.0, ls="--", label="injected")
        total = np.trapezoid(side_low + side_high + caps, t) * 1e6
        ax.set_title(f"{label}: box loses {total / 1e8:.1f}e8 Msun in 75 Myr")
        ax.set_xlabel("t [Myr]")
    ax = axes[-1]
    for label in labels:
        rec = data[label][1]
        nz = next(iter(rec.values()))["side_mdot_net_z"].shape[1]
        z = (np.arange(nz) + 0.5) * BOX_SIZE_KPC[2] / nz - BOX_SIZE_KPC[2] / 2
        prof = window_mean(rec, HIGH, lambda r_: r_["side_mdot_net_z"].sum(0)) / (BOX_SIZE_KPC[2] / nz)
        half = nz // 2
        ax.plot(0.5 * (prof[half:] + prof[:half][::-1]), z[half:], color=colors[label], label=label)
    ax.set_title(f"side-wall loss per kpc of height, {HIGH[0]:.0f}-{HIGH[1]:.0f} Myr")
    ax.set_xlabel(r"$d\dot M_{\rm side}/d|z|$ per hemisphere [$M_\odot$ yr$^{-1}$ kpc$^{-1}$]")
    ax.set_ylabel("|z| [kpc]")
    ax.axvline(0, color="0.6", lw=0.8)
    ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\dot M$ [$M_\odot$ yr$^{-1}$]")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{prefix}_shells_box_budget.png"), dpi=130)
    plt.close(fig)

    # Tables.
    print(f"{HIGH[0]:.0f}-{HIGH[1]:.0f} Myr | {LOW[0]:.0f}-{LOW[1]:.0f} Myr means of net Mdot [Msun/yr] (Edot [1e42])")
    for radius in (1.0, 2.0, 3.0, 5.0, 8.0, 10.0):
        i = i_of(radius)
        for label in labels:
            rec = data[label][1]
            cells = []
            for window in (HIGH, LOW):
                for theta, _ in CONES:
                    if complete(theta)[i]:
                        m = window_mean(rec, window, lambda r_: reduce(r_, "mdot_net", theta)[i].sum())
                        e = window_mean(rec, window, lambda r_: reduce(r_, "edot_net", theta)[i].sum())
                        cells.append(f"{theta:2d}: {m:6.2f} ({e / 1e42:4.2f})")
                    else:
                        cells.append(f"{theta:2d}:    -   (  - )")
            print(f"  r = {radius:3g}  {label:6s} " + "  ".join(cells))
    print(f"wrote {prefix}_shells_{{profiles,angular,vs_planes,box_budget}}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--steps", default="all", help="'all', 'a-b', 'a-b/k' or a comma list")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--plot", nargs="*", help="LABEL=TAG pairs of cached runs, lowest resolution first")
    parser.add_argument("--prefix", default="a_series", help="figure-name prefix")
    args = parser.parse_args()
    if args.plot:
        plot([tuple(spec.split("=", 1)) for spec in args.plot], args.prefix)
    else:
        run(args.tag, args.checkpoint_dir, args.steps, args.workers, args.parts)


if __name__ == "__main__":
    main()
