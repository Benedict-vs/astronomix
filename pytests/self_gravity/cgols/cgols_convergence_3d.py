"""
Resolution convergence of the CGOLS A-series from the full 3D checkpoints.

The frames only carry the total vertical mass flux and a few slices. This
script reads the Orbax checkpoints (the full primitive state, one per frame)
tile by tile, like cgols_emission.py, and measures per snapshot

  fluxes   through the planes |z| = HEIGHTS (both sides, outward positive):
           net and outflow-only mass flux [Msun/yr], outward momentum flux
           rho v_out^2 + P of the outflowing gas [Msun km/s / yr] and energy
           flux (rho v^2 / 2 + gamma P / (gamma - 1)) v_out [erg/s], each in
           the phases cold (T < 3e4 K), warm and hot (T > 5e5 K); the mass
           flux also per cylindrical radius bin (wind core vs cone walls);
  masses   in the disk (|z| < 0.5 kpc) and the halo (|z| > 0.5 kpc) by phase;
  phases   the mass-weighted (log n, log T) histogram of |z| > 1 kpc;
  axis     mass-weighted n, T, v_out against z inside R < 0.3 kpc, for the
           comparison with the Chevalier & Clegg (1985) free wind;

and caches them in data/studies/convergence3d<tag>.npz (resumable). With
--plot it compares cached runs (figures/cgols/comparisons/
a_series_convergence3d_*.png).

    python cgols_convergence_3d.py --tag _A512 [--steps 2-60/2] [--workers 8]
    python cgols_convergence_3d.py --plot A128=_A128 A256=_A256 A512=_A512 A1024=_A1024

CPU only; it never imports cgols.py (whose autocvd would wait for a GPU).
"""

# ruff: noqa: E402

# general
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

# CPU only: astronomix imports JAX, which would otherwise grab a (shared) GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# numerics
import numpy as np
import tensorstore

# units
from astropy import units as u

# astronomix
from astronomix import CodeUnits, SimulationConfig, get_registered_variables
from astronomix.analysis_helpers.emission import emission_conversions
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Must match cgols.py (not imported, see above).
CODE_UNITS = CodeUnits(1 * u.kiloparsec, 1e6 * u.M_sun, 100 * u.km / u.s)
BOX_SIZE_KPC = (10.0, 10.0, 20.0)
GAMMA = 5.0 / 3.0
CODE_TIME_MYR = float((1 * CODE_UNITS.code_time).to(u.Myr).value)
MSUN_PER_YR = float((1 * CODE_UNITS.code_mass / CODE_UNITS.code_time).to(u.M_sun / u.yr).value)
MOMENTUM_PER_YR = float(
    (1 * CODE_UNITS.code_mass * CODE_UNITS.code_velocity / CODE_UNITS.code_time).to(u.M_sun * u.km / u.s / u.yr).value
)
ERG_PER_S = float((1 * CODE_UNITS.code_energy / CODE_UNITS.code_time).to(u.erg / u.s).value)
CODE_MASS_MSUN = float((1 * CODE_UNITS.code_mass).to(u.M_sun).value)
CONVERSIONS = emission_conversions(CODE_UNITS)

HEIGHTS_KPC = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
PHASE_EDGES_K = (3e4, 5e5)  # cold | warm | hot
PHASES = ("cold", "warm", "hot")
RADIUS_EDGES_KPC = (0.5, 1.0, 2.0)  # R < 0.5 | 0.5-1 | 1-2 | > 2
DISK_HALF_HEIGHT_KPC = 0.5
HIST_MIN_HEIGHT_KPC = 1.0
LOG_N_EDGES = np.linspace(-6.0, 3.0, 91)
LOG_T_EDGES = np.linspace(3.0, 9.0, 121)
AXIS_RADIUS_KPC = 0.3

# High state of the feedback schedule (cgols.build_cgols_wind_params).
CC85_MDOT_MSUN_YR, CC85_EDOT_ERG_S, CC85_RADIUS_KPC = 12.0, 5.4e42, 0.3


# -------------------------------------------------------------
# ===================== ↓ Measuring ↓ =========================
# -------------------------------------------------------------


def _open(step_directory, name):
    context = tensorstore.Context({"cache_pool": {"total_bytes_limit": 0}})
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "ocdbt", "base": "file://" + os.path.abspath(step_directory) + "/", "path": name},
    }
    return tensorstore.open(spec, context=context, read=True).result()


def measure_checkpoint(step_directory):
    """All per-snapshot measurements of one checkpoint (see the module docstring)."""
    state = _open(step_directory, "primitive_state")
    time_myr = float(_open(step_directory, "time").read().result()) * CODE_TIME_MYR
    num_vars, nx, ny, nz = state.shape
    variables = get_registered_variables(
        SimulationConfig(dimensionality=3, solver_mode=FINITE_DIFFERENCE, dual_energy=(num_vars == 6))
    )
    dx = BOX_SIZE_KPC[0] / nx  # kpc = code length
    x = (np.arange(nx) + 0.5) * dx - BOX_SIZE_KPC[0] / 2
    y = (np.arange(ny) + 0.5) * dx - BOX_SIZE_KPC[1] / 2
    z = (np.arange(nz) + 0.5) * dx - BOX_SIZE_KPC[2] / 2
    planes = [(np.argmin(np.abs(z - h)), np.argmin(np.abs(z + h))) for h in HEIGHTS_KPC]
    num_h, num_p, num_r = len(HEIGHTS_KPC), len(PHASES), len(RADIUS_EDGES_KPC) + 1

    out = {
        "mdot_net": np.zeros((num_h, num_p)),
        "mdot_out": np.zeros((num_h, num_p)),
        "pdot_out": np.zeros((num_h, num_p)),
        "edot_net": np.zeros((num_h, num_p)),
        "mdot_net_radius": np.zeros((num_h, num_r)),
        "mass_disk": np.zeros(num_p),
        "mass_halo": np.zeros(num_p),
        "hist_nT": np.zeros((len(LOG_N_EDGES) - 1, len(LOG_T_EDGES) - 1)),
        "axis_mass": np.zeros(nz),
        "axis_mass_v": np.zeros(nz),
        "axis_mass_logT": np.zeros(nz),
        "axis_mass_logn": np.zeros(nz),
    }
    sign_z = np.sign(z).astype(np.float32)
    in_disk = np.abs(z) < DISK_HALF_HEIGHT_KPC
    in_hist = np.abs(z) > HIST_MIN_HEIGHT_KPC

    chunk_x, chunk_y = state.chunk_layout.read_chunk.shape[1:3]
    tiles = [
        (slice(i, min(i + chunk_x, nx)), slice(j, min(j + chunk_y, ny)))
        for i in range(0, nx, chunk_x)
        for j in range(0, ny, chunk_y)
    ]
    pending = state[:, tiles[0][0], tiles[0][1], :].read()
    for index, (xs, ys) in enumerate(tiles):
        block = pending.result()
        if index + 1 < len(tiles):
            pending = state[:, tiles[index + 1][0], tiles[index + 1][1], :].read()
        rho = block[variables.density_index]
        vx = block[variables.velocity_index.x]
        vy = block[variables.velocity_index.y]
        vz = block[variables.velocity_index.z]
        pressure = block[variables.pressure_index]
        del block
        temperature = pressure / rho * np.float32(CONVERSIONS.kelvin_per_code_specific_energy)
        phase = np.digitize(temperature, PHASE_EDGES_K).astype(np.int8)
        radius = np.hypot(x[xs][:, None], y[ys][None, :]).astype(np.float32)
        radius_bin = np.digitize(radius, RADIUS_EDGES_KPC)

        # Plane fluxes (code units, per cell area dx^2).
        for h, (k_north, k_south) in enumerate(planes):
            for k, side in ((k_north, 1.0), (k_south, -1.0)):
                r, v_out, p = rho[:, :, k], side * vz[:, :, k], pressure[:, :, k]
                v2 = vx[:, :, k] ** 2 + vy[:, :, k] ** 2 + vz[:, :, k] ** 2
                ph = phase[:, :, k]
                mass_flux = r * v_out * dx**2
                outflowing = v_out > 0
                energy_flux = (0.5 * r * v2 + GAMMA / (GAMMA - 1) * p) * v_out * dx**2
                momentum_flux = np.where(outflowing, r * v_out**2 + p, 0.0) * dx**2
                for q in range(num_p):
                    sel = ph == q
                    out["mdot_net"][h, q] += mass_flux[sel].sum(dtype=np.float64)
                    out["mdot_out"][h, q] += mass_flux[sel & outflowing].sum(dtype=np.float64)
                    out["pdot_out"][h, q] += momentum_flux[sel].sum(dtype=np.float64)
                    out["edot_net"][h, q] += energy_flux[sel].sum(dtype=np.float64)
                out["mdot_net_radius"][h] += np.bincount(
                    radius_bin.ravel(), weights=mass_flux.ravel().astype(np.float64), minlength=num_r
                )

        # Masses by region and phase.
        cell_mass = rho * dx**3
        for q in range(num_p):
            sel = phase == q
            out["mass_disk"][q] += cell_mass[:, :, in_disk][sel[:, :, in_disk]].sum(dtype=np.float64)
            out["mass_halo"][q] += cell_mass[:, :, ~in_disk][sel[:, :, ~in_disk]].sum(dtype=np.float64)

        # Mass-weighted (log n, log T) histogram of the outflow region.
        log_n = np.log10(rho[:, :, in_hist] * np.float32(CONVERSIONS.total_number_density_per_code_density))
        log_t = np.log10(temperature[:, :, in_hist])
        i_n = np.clip(np.searchsorted(LOG_N_EDGES, log_n.ravel()) - 1, 0, len(LOG_N_EDGES) - 2)
        i_t = np.clip(np.searchsorted(LOG_T_EDGES, log_t.ravel()) - 1, 0, len(LOG_T_EDGES) - 2)
        out["hist_nT"] += np.bincount(
            i_n * (len(LOG_T_EDGES) - 1) + i_t,
            weights=cell_mass[:, :, in_hist].ravel().astype(np.float64),
            minlength=out["hist_nT"].size,
        ).reshape(out["hist_nT"].shape)
        del log_n, log_t, i_n, i_t

        # Free-wind axis profile, R < AXIS_RADIUS_KPC.
        axis = radius < AXIS_RADIUS_KPC
        if axis.any():
            m = cell_mass[axis]  # (cells, nz)
            out["axis_mass"] += m.sum(axis=0, dtype=np.float64)
            out["axis_mass_v"] += (m * vz[axis] * sign_z).sum(axis=0, dtype=np.float64)
            out["axis_mass_logT"] += (m * np.log10(temperature[axis])).sum(axis=0, dtype=np.float64)
            out["axis_mass_logn"] += (
                m * np.log10(rho[axis] * np.float32(CONVERSIONS.total_number_density_per_code_density))
            ).sum(axis=0, dtype=np.float64)
        del rho, vx, vy, vz, pressure, temperature, phase, cell_mass

    # Physical units.
    for key in ("mdot_net", "mdot_out", "mdot_net_radius"):
        out[key] *= MSUN_PER_YR
    out["pdot_out"] *= MOMENTUM_PER_YR
    out["edot_net"] *= ERG_PER_S
    for key in ("mass_disk", "mass_halo", "hist_nT"):
        out[key] *= CODE_MASS_MSUN
    out["time_myr"] = time_myr
    out["z_kpc"] = z
    out["dim"] = nx
    return out


def _measure(args):
    step, step_directory = args
    start = time.time()
    result = measure_checkpoint(step_directory)
    return step, result, time.time() - start


def parse_steps(spec, available):
    """'all', 'a-b', 'a-b/k' (every k-th) or a comma list, intersected with available."""
    if spec == "all":
        return list(available)
    if "," in spec:
        wanted = {int(s) for s in spec.split(",")}
        return [s for s in available if s in wanted]
    stride = 1
    if "/" in spec:
        spec, stride = spec.split("/")
        stride = int(stride)
    first, last = (int(p) for p in spec.split("-")) if "-" in spec else (int(spec), int(spec))
    return [s for s in available if first <= s <= last and (s - first) % stride == 0]


def cache_path(tag):
    return os.path.join(SCRIPT_DIR, "data", "studies", f"convergence3d{tag}.npz")


def load_cache(tag):
    path = cache_path(tag)
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=True)
    return data["records"].item()


def save_cache(tag, records):
    path = cache_path(tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", records=np.array(records, dtype=object))
    os.replace(path + ".tmp.npz", path)


def run(tag, checkpoint_dir, steps_spec, workers):
    checkpoint_dir = checkpoint_dir or os.path.join(SCRIPT_DIR, "data", f"cgols_checkpoints{tag}")
    available = sorted(int(e) for e in os.listdir(checkpoint_dir) if e.isdigit())
    records = load_cache(tag)
    todo = [s for s in parse_steps(steps_spec, available) if s not in records]
    # Skip step directories still being transferred (no Orbax commit marker yet).
    todo = [s for s in todo if os.path.exists(os.path.join(checkpoint_dir, str(s), "_CHECKPOINT_METADATA"))]
    print(f"{tag}: {len(records)} cached, {len(todo)} to measure from {checkpoint_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(s, os.path.join(checkpoint_dir, str(s))) for s in todo]
        for step, result, seconds in pool.map(_measure, jobs):
            records[step] = result
            save_cache(tag, records)
            print(
                f"  [{step:3d}] t = {result['time_myr']:6.2f} Myr  "
                f"Mdot(5 kpc) = {result['mdot_net'][5].sum():6.2f} Msun/yr  ({seconds:.0f} s)",
                flush=True,
            )


# -------------------------------------------------------------
# ===================== ↓ CC85 reference ↓ ====================
# -------------------------------------------------------------


def cc85_profile(radius_kpc, mdot=CC85_MDOT_MSUN_YR, edot=CC85_EDOT_ERG_S, r_inj=CC85_RADIUS_KPC, gamma=GAMMA):
    """Chevalier & Clegg (1985) n [cm^-3], T [K], v [km/s] at the given radii."""
    from scipy.optimize import brentq

    def inside(mach, x):
        return (
            ((3 * gamma + 1 / mach**2) / (1 + 3 * gamma)) ** (-(3 * gamma + 1) / (5 * gamma + 1))
            * (((gamma - 1) + 2 / mach**2) / (1 + gamma)) ** ((gamma + 1) / (2 * (5 * gamma + 1)))
            - x
        )

    def outside(mach, x):
        return mach ** (2 / (gamma - 1)) * (((gamma - 1) + 2 / mach**2) / (1 + gamma)) ** (
            (gamma + 1) / (2 * (gamma - 1))
        ) - x**2

    mdot_cgs = (mdot * u.M_sun / u.yr).to(u.g / u.s).value
    r_cgs = (r_inj * u.kpc).to(u.cm).value
    v_inf = np.sqrt(2 * edot / mdot_cgs)
    mu_mp = 0.6 * 1.6726e-24
    n, temperature, velocity = [], [], []
    for r in np.atleast_1d(radius_kpc):
        x = max(r / r_inj, 1e-4)
        if x < 1:
            mach = brentq(inside, 1e-6, 1.0, args=(x,))
        else:
            mach = brentq(outside, 1.0, 1e4, args=(x,))
        # v from Bernoulli: v^2/2 + c^2/(gamma-1) = v_inf^2/2, c = v/M.
        v = v_inf / np.sqrt(1 + 2 / ((gamma - 1) * mach**2))
        c = v / mach
        if x < 1:
            # Uniform mass source q = Mdot / V: rho v = q r / 3.
            rho = (mdot_cgs / (4 / 3 * np.pi * r_cgs**3)) * (x * r_cgs) / (3 * v)
        else:
            rho = mdot_cgs / (4 * np.pi * (x * r_cgs) ** 2 * v)
        n.append(rho / mu_mp)
        temperature.append(c**2 * mu_mp / (gamma * 1.380649e-16))
        velocity.append(v / 1e5)
    return np.array(n), np.array(temperature), np.array(velocity)


# -------------------------------------------------------------
# ===================== ↓ Plotting ↓ ==========================
# -------------------------------------------------------------


RIM_DENSITY_CM3 = 3.0  # between the cavity (~0.1) and the disk (~25-75 cm^-3)
RIM_SEARCH_KPC = (0.4, 3.0)


def rim_metrics(frame_dir, num_angles=720):
    """Cavity-rim roughness in the midplane slice n_xy of every frame.

    Along num_angles rays from the centre, the rim radius is the first radius
    in RIM_SEARCH_KPC where n exceeds RIM_DENSITY_CM3. Returns times [Myr],
    the mean rim radius [kpc], its relative scatter sigma_R / <R>, and the
    dominant azimuthal mode number m of R(phi) - mean.
    """
    files = sorted(f for f in os.listdir(frame_dir) if f.startswith("frame_") and f.endswith(".npz"))
    radii = np.linspace(*RIM_SEARCH_KPC, 600)
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    times, mean_r, scatter, mode = [], [], [], []
    for name in files:
        frame = np.load(os.path.join(frame_dir, name))
        density = frame["n_xy"]
        cells = density.shape[0]
        dx = BOX_SIZE_KPC[0] / cells
        ix = np.clip(((radii[None, :] * np.cos(angles)[:, None] + BOX_SIZE_KPC[0] / 2) / dx).astype(int), 0, cells - 1)
        iy = np.clip(((radii[None, :] * np.sin(angles)[:, None] + BOX_SIZE_KPC[1] / 2) / dx).astype(int), 0, cells - 1)
        above = density[ix, iy] > RIM_DENSITY_CM3
        first = np.where(above.any(axis=1), above.argmax(axis=1), len(radii) - 1)
        rim = radii[first]
        times.append(float(frame["time_myr"]))
        mean_r.append(rim.mean())
        scatter.append(rim.std() / rim.mean())
        power = np.abs(np.fft.rfft(rim - rim.mean())) ** 2
        mode.append(int(np.argmax(power[1:]) + 1))
    order = np.argsort(times)
    return tuple(np.array(a)[order] for a in (times, mean_r, scatter, mode))


def _series(records, key, reducer):
    steps = sorted(records)
    t = np.array([records[s]["time_myr"] for s in steps])
    return t, np.array([reducer(records[s][key]) for s in steps])


def plot(runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data = {label: load_cache(tag) for label, tag in runs}
    data = {label: rec for label, rec in data.items() if rec}
    colors = dict(zip(data, plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(data)))))
    fig_dir = os.path.join(SCRIPT_DIR, "figures", "cgols", "comparisons")
    os.makedirs(fig_dir, exist_ok=True)
    h1, h5 = int(np.argmin(np.abs(HEIGHTS_KPC - 1))), int(np.argmin(np.abs(HEIGHTS_KPC - 5)))

    # 1) Phase-resolved fluxes against time at 1 and 5 kpc.
    fig, axes = plt.subplots(3, 4, figsize=(22, 12), sharex=True)
    for label, rec in data.items():
        for col, (h, name) in enumerate(((h1, "1 kpc"), (h5, "5 kpc"))):
            for row, q in enumerate(range(3)):
                t, v = _series(rec, "mdot_out", lambda a, h=h, q=q: a[h, q])
                axes[row, col].plot(t, v, color=colors[label], label=label)
                axes[row, col].set_title(f"outflowing mass flux, {PHASES[q]}, |z| = {name}")
            t, v = _series(rec, "edot_net", lambda a, h=h: a[h].sum())
            axes[0, 2 + col].plot(t, v / 1e42, color=colors[label], label=label)
            axes[0, 2 + col].set_title(f"net energy flux |z| = {name} [1e42 erg/s]")
            t, v = _series(rec, "pdot_out", lambda a, h=h: a[h].sum())
            axes[1, 2 + col].plot(t, v / 1e3, color=colors[label], label=label)
            axes[1, 2 + col].set_title(f"outward momentum flux |z| = {name} [1e3 Msun km/s/yr]")
            t, v = _series(rec, "mass_disk", lambda a: a.sum())
            axes[2, 2].plot(t, v / 1e9, color=colors[label], label=label)
            t, v = _series(rec, "mass_halo", lambda a: a[0] + a[1])
            axes[2, 3].plot(t, v / 1e8, color=colors[label], label=label)
    axes[2, 2].set_title("disk mass |z| < 0.5 kpc [1e9 Msun]")
    axes[2, 3].set_title("cold+warm halo mass |z| > 0.5 kpc [1e8 Msun]")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.axvspan(15, 40, color="0.93", zorder=0)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$M_\odot$/yr")
    for ax in axes[-1]:
        ax.set_xlabel("t [Myr]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("A-series, full 3D states: phase-resolved fluxes (cold < 3e4 K < warm < 5e5 K < hot)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "a_series_convergence3d_fluxes.png"), dpi=130)

    # 2) Time-averaged flux profiles against height, two windows, by phase.
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for row, window in enumerate(((15, 30), (30, 45))):
        for label, rec in data.items():
            steps = [s for s in sorted(rec) if window[0] <= rec[s]["time_myr"] <= window[1]]
            if not steps:
                continue
            mean = lambda key: np.mean([rec[s][key] for s in steps], axis=0)
            for q in range(3):
                axes[row, q].plot(HEIGHTS_KPC, mean("mdot_out")[:, q], "o-", color=colors[label], label=label)
                axes[row, q].set_title(f"{PHASES[q]} outflow, {window[0]}-{window[1]} Myr")
            axes[row, 3].plot(HEIGHTS_KPC, mean("mdot_net").sum(axis=1), "o-", color=colors[label], label=label)
            axes[row, 3].set_title(f"net total, {window[0]}-{window[1]} Myr")
    for ax in axes.flat:
        ax.set(xlabel="|z| [kpc]", ylabel=r"$\dot M$ [$M_\odot$/yr]")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "a_series_convergence3d_profiles.png"), dpi=130)

    # 3) Phase diagrams at ~35 Myr, one per resolution, and the free-wind axis vs CC85.
    fig, axes = plt.subplots(2, max(len(data), 3), figsize=(5.5 * max(len(data), 3), 10), squeeze=False)
    for col, (label, rec) in enumerate(data.items()):
        step = min(rec, key=lambda s: abs(rec[s]["time_myr"] - 35))
        r = rec[step]
        axes[0, col].pcolormesh(LOG_N_EDGES, LOG_T_EDGES, r["hist_nT"].T, norm=LogNorm(1e3, None), cmap="viridis")
        axes[0, col].set(title=f"{label}: mass in |z| > 1 kpc, {r['time_myr']:.1f} Myr", xlabel="log n", ylabel="log T")
    zs = np.linspace(0.05, 9.5, 200)
    n_cc, t_cc, v_cc = cc85_profile(zs)
    for label, rec in data.items():
        steps = [s for s in sorted(rec) if 20 <= rec[s]["time_myr"] <= 40]
        if not steps:
            continue
        z = rec[steps[0]]["z_kpc"]
        m = np.mean([rec[s]["axis_mass"] for s in steps], axis=0)
        north = z > 0
        for ax, key, conv in (
            (axes[1, 0], "axis_mass_logn", lambda a: 10**a),
            (axes[1, 1], "axis_mass_logT", lambda a: 10**a),
            (axes[1, 2], "axis_mass_v", lambda a: a * 100),
        ):
            value = np.mean([rec[s][key] for s in steps], axis=0) / np.maximum(m, 1e-300)
            ax.plot(z[north], conv(value[north]), color=colors[label], label=label)
    for ax, ref, name in ((axes[1, 0], n_cc, "n [cm^-3]"), (axes[1, 1], t_cc, "T [K]"), (axes[1, 2], v_cc, "v_z [km/s]")):
        ax.plot(zs, ref, "k--", lw=1, label="CC85 (high state)")
        ax.set(xlabel="z [kpc]", ylabel=name, title=f"free-wind axis R < {AXIS_RADIUS_KPC} kpc, 20-40 Myr")
        if name != "v_z [km/s]":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    for ax in axes[1, 3:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "a_series_convergence3d_phases_axis.png"), dpi=130)

    # 4) Instability onset at the cavity rim, from the frames' midplane slices.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label, tag in runs:
        frame_dir = os.path.join(SCRIPT_DIR, f"cgols_snapshots{tag}")
        if not os.path.isdir(frame_dir):
            continue
        t, mean_r, scatter, _ = rim_metrics(frame_dir)
        color = colors.get(label, "k")
        axes[0].plot(t, scatter, color=color, label=label)
        axes[1].plot(t, mean_r, color=color, label=label)
    axes[0].set(xlabel="t [Myr]", ylabel=r"$\sigma_R / \langle R \rangle$",
                title=f"midplane cavity rim roughness (n = {RIM_DENSITY_CM3:g} cm$^{{-3}}$)", yscale="log")
    axes[1].set(xlabel="t [Myr]", ylabel=r"$\langle R_\mathrm{rim} \rangle$ [kpc]", title="mean cavity rim radius")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "a_series_convergence_rim.png"), dpi=130)

    # Table: window means at 1/5 kpc.
    for window in ((15, 30), (30, 45), (45, 75)):
        print(f"\n{window[0]}-{window[1]} Myr  outflowing Mdot [Msun/yr] cold/warm/hot at 1 kpc | 5 kpc ; Edot(5) [1e42]")
        for label, rec in data.items():
            steps = [s for s in sorted(rec) if window[0] <= rec[s]["time_myr"] <= window[1]]
            if not steps:
                continue
            mo = np.mean([rec[s]["mdot_out"] for s in steps], axis=0)
            ed = np.mean([rec[s]["edot_net"] for s in steps], axis=0)
            print(
                f"  {label:7s} {mo[h1, 0]:6.2f} {mo[h1, 1]:6.2f} {mo[h1, 2]:6.2f} | "
                f"{mo[h5, 0]:6.2f} {mo[h5, 1]:6.2f} {mo[h5, 2]:6.2f} ; {ed[h5].sum() / 1e42:5.2f}  ({len(steps)} snaps)"
            )
    print("wrote a_series_convergence3d_{fluxes,profiles,phases_axis}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--steps", default="all", help="'all', 'a-b', 'a-b/k' or a comma list")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plot", nargs="*", help="LABEL=TAG pairs of cached runs to compare")
    args = parser.parse_args()
    if args.plot:
        plot([tuple(spec.split("=", 1)) for spec in args.plot])
    else:
        run(args.tag, args.checkpoint_dir, args.steps, args.workers)


if __name__ == "__main__":
    main()
