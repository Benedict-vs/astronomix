"""
FV vs FD synthesis for the CGOLS model-A runs: one metric table with noise, significance, convergence and cost.

FD = WENO5-Z finite difference (PP-WENO + dual energy, RK4-LSRK, CFL 0.9 summed over axes), FV = VL2 (donor-cell
predictor + van Leer PLM corrector) + HLLC (CFL 0.4, FOFC, hard floor); both float32, same IC, injection and box.
Everything is read from existing outputs -- frames (cgols_snapshots_*), emission caches (data/studies/emission_*),
3D caches (data/studies/convergence3d_v3_*), per-step diag logs -- nothing is re-simulated.

Per run (cached in data/studies/fvfd_synthesis_metrics.json, --recompute to rebuild) the script measures

  frames    net outflow through |z| = 5 kpc (15-40, 40-50, 50-75 Myr means, peak, cumulative 0-75 Myr), the
            axis pinch <log n> (4 < |z| < 9 kpc) and pinched fraction at t >= 60 Myr, the up-down mirror asymmetry
            (50-75 Myr mean), the small-scale x-variance fraction of log n in 1 < |z| < 6 kpc (256 grid, 80-300 pc
            as in cgols_fv_riemann.py spectrum; and a 128 grid, 160-600 pc, so 128^3 runs have a value), the free-
            wind axis against Chevalier & Clegg (1985) in the high state (20-38 Myr, 1.5 < |z| < 4 kpc: median
            log n / n_CC85 and v / v_CC85 - 1 -- the one absolute reference the data offer), the mass budget
            (side-face loss = M0 + injected - M - z-face outflow, cgols_density_floor.py convention);
  emission  L_X1 (CGOLS Eq. 17), AtomDB 0.3-2 keV and H-alpha luminosities in the high (15-40) and late (60-75)
            states;
  3D        entropy-split fluxes (wind-entropy outflow at 1 kpc, mixed outflow at 1 kpc, Edot at 5 kpc), the
            cumulative energy through |z| = 5 kpc and the cone half-opening at z = 1 kpc -- on the common 5 Myr
            snapshot cadence (FV-512 has every 4th checkpoint) and, for the 1024 pair, at the matched times 30 and
            60 Myr (FV-1024 has only 4 checkpoints).

Noise. sigma_traj = sample std over the realisations of one scheme at one N: IC-noise seeds 0-3, the checkpoint-
segmented vs unsegmented baselines and reruns (fp32 only). The IC noise (1e-3) perturbs less than checkpoint
segmentation does, so this is a LOWER bound. Where a scheme/N has no ensemble (FD-512, FD-1024, FV-1024) it is
extrapolated as "noise does not shrink with N, and FD is as noisy as FV": the largest sigma measured for either
scheme at any N' <= N (relative for positive fluxes/luminosities, absolute for log or shape metrics); the
pessimistic variant also applies the FV 256 -> 512 growth once more at 1024. sigma_t is the standard error of a
window mean from 5 Myr block means (a stationarity-based proxy that exists for every run). The conservative
sigma = max(sigma_traj, sigma_t); a difference is called significant if |FD - FV| > 3 sigma_diff with
sigma_diff = hypot(sigma_FD, sigma_FV). 3D metrics have no ensembles: they borrow the relative sigma_traj of the
matching frames metric (out5 window) and take the max with their own sigma_t.

Statistics printed per metric: FD, FV, FD-FV and z = (FD-FV)/sigma_diff at 128/256/512/1024; the power-law slope b
of |FD-FV| ~ N^b (weighted fit, 256-1024, only if the sign is constant); per scheme the successive differences
256->512->1024, the observed order p = log2(|d1|/|d2|) and a Richardson value M_inf = M_1024 + d2/(2^p - 1) when
monotone (Monte Carlo over sigma: the fraction of draws that are monotone, the 16-84% range); the equivalent-
resolution factor k with FV(N) = FD(kN) (interpolated in log2 N on the FD ladder); and the GPU-hours per run. The
ablations (HLLE, fp64, CFL 0.3, PPM) are listed as offsets from their FV baseline in units of sigma.

    python cgols_fvfd_synthesis.py [--recompute] [--json OUT] [--no-figures] \
        [--ppm LABEL=ROLE:FRAMES_DIR:DIAG_LOG[:EMISSION_TAG] ...]

    # 2026-10-29 incl. the CPU FV-128 runs from the fv-ppm worktree (frames on scratch, emission caches built
    # with cgols_emission.py --source frames --frames-dir ...); adds figures/cgols/comparisons/fv_ppm_A128.png
    python cgols_fvfd_synthesis.py \
        --ppm FV128ppm=PPM:/export/scratch/bschuber/ppmrun_cpu/frames/A128fvppm:<diag log>:_A128fvppm \
        --ppm "FV128plmcpu=PLM CPU:/export/scratch/bschuber/ppmrun_cpu/frames/A128fvplmcpu:<diag log>:_A128fvplmcpu"

    # 2026-10-31: + the GPU FV-PPM 256/512 runs (PPM_GPU_RUNS; emission caches from the frames with
    # cgols_emission.py --tag _A{256,512}fvppm --source frames); prints the PPM direction / cost tables and adds
    # figures/cgols/comparisons/fv_ppm_A256_A512.png (the other figures are unchanged)
    python cgols_fvfd_synthesis.py --ppm-gpu [--ppm ... as above]

Writes figures/cgols/comparisons/fvfd_summary_significance.png, fvfd_convergence.png and fvfd_cost.png and prints
the markdown tables. CPU only, numpy + scipy + matplotlib (no jax).
"""

# general
import argparse
import itertools
import json
import os
import re

# plotting
import matplotlib

# numerics
import numpy as np
from scipy.optimize import brentq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# shared CGOLS frame helpers (numpy only)
from cgols_fv_riemann import net_outflow, pinch, small_scale_fraction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIAG = "cgols_snapshots_diagnostics"
LOGS = "cgols_logs"
CACHE = "data/studies/fvfd_synthesis_metrics.json"
FIG_DIR = "figures/cgols/comparisons"
BOX_Z = 20.0
MYR_PER_CODE_TIME = 9.777922216807893
RES = (128, 256, 512, 1024)
HEIGHTS_KPC = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])  # = cgols_convergence_3d.HEIGHTS_KPC
RNG = np.random.default_rng(1)

# -------------------------------------------------------------
# =============== ↓ Runs ↓ ====================================
# -------------------------------------------------------------

#: label -> (scheme, N, role, frame dir, emission tag, 3D tag, diag log). role: base = the ladder run of the
#: comparison figures; ens = another fp32 realisation (seed / segmentation / rerun) of the same scheme and N;
#: anything else = an ablation of that FV (or FD) baseline.
RUNS = {
    "FD128": ("FD", 128, "base", "cgols_snapshots_A128", "_A128", "_A128", f"{LOGS}/cgols_diag_A128.log"),
    **{f"FD128s{s}": ("FD", 128, "ens", f"{DIAG}/A128s{s}", f"_A128s{s}", None,
                      f"{LOGS}/diagnostics/cgols_diag_A128s{s}.log") for s in (1, 2, 3)},
    "FV128": ("FV", 128, "base", f"{DIAG}/A128fv", "_A128fv", "_A128fv", f"{LOGS}/diagnostics/cgols_diag_A128fv.log"),
    # FV128j = FV128 to 1e-7 (same config, also checkpointed; JUPITER rerun with in-situ maps) and FV128r2 = FV128r1
    # bit for bit: kept for reference, excluded from the ensemble (role "duplicate") so they do not shrink sigma.
    "FV128j": ("FV", 128, "duplicate", "cgols_snapshots_A128fvf32", "_A128fvf32", None, f"{LOGS}/cgols_diag_A128fvf32.log"),
    "FV128r1": ("FV", 128, "ens", f"{DIAG}/A128r1fv", None, None, f"{LOGS}/diagnostics/cgols_diag_A128r1fv.log"),
    "FV128r2": ("FV", 128, "duplicate", f"{DIAG}/A128r2fv", None, None, f"{LOGS}/diagnostics/cgols_diag_A128r2fv.log"),
    **{f"FV128s{s}": ("FV", 128, "ens", f"{DIAG}/A128fvs{s}", f"_A128fvs{s}", None,
                      f"{LOGS}/diagnostics/cgols_diag_A128fvs{s}.log") for s in (1, 2, 3)},
    "FV128f64": ("FV", 128, "fp64", "cgols_snapshots_A128fvf64", None, None, f"{LOGS}/cgols_diag_A128fvf64.log"),
    "FV128c03": ("FV", 128, "CFL 0.3", f"{DIAG}/A128c03fv", None, None, f"{LOGS}/diagnostics/cgols_diag_A128c03fv.log"),
    "FD128f64": ("FD", 128, "fp64", "cgols_snapshots_A128fdf64", None, None, f"{LOGS}/cgols_diag_A128fdf64.log"),
    "FD128node": ("FD", 128, "no dual energy", "cgols_snapshots_A128fdnode", None, None,
                  f"{LOGS}/cgols_diag_A128fdnode.log"),
    "FD256": ("FD", 256, "base", "cgols_snapshots_A256", "_A256", "_A256", f"{LOGS}/cgols_diag_A256.log"),
    **{f"FD256s{s}": ("FD", 256, "ens", f"{DIAG}/A256s{s}", f"_A256s{s}", None,
                      f"{LOGS}/diagnostics/cgols_diag_A256s{s}.log") for s in (1, 2, 3)},
    "FV256": ("FV", 256, "base", f"{DIAG}/A256fv", "_A256fv", "_A256fv", f"{LOGS}/diagnostics/cgols_diag_A256fv.log"),
    **{f"FV256s{s}": ("FV", 256, "ens", f"{DIAG}/A256fvs{s}", f"_A256fvs{s}", None,
                      f"{LOGS}/diagnostics/cgols_diag_A256fvs{s}.log") for s in (1, 2, 3)},
    "FV256nock": ("FV", 256, "ens", f"{DIAG}/A256nockfv", None, None, f"{LOGS}/diagnostics/cgols_diag_A256nockfv.log"),
    "FV256hlle": ("FV", 256, "HLLE", f"{DIAG}/A256fvhlle", "_A256fvhlle", None,
                  f"{LOGS}/diagnostics/cgols_diag_A256fvhlle.log"),
    "FV256c03": ("FV", 256, "CFL 0.3", f"{DIAG}/A256c03fv", None, None, f"{LOGS}/diagnostics/cgols_diag_A256c03fv.log"),
    "FD512": ("FD", 512, "base", "cgols_snapshots_A512", "_A512", "_A512", f"{LOGS}/cgols_diag_A512.log"),
    "FV512": ("FV", 512, "base", "cgols_snapshots_A512fvf32", "_A512fvf32", "_A512fvf32",
              f"{LOGS}/cgols_diag_A512fvf32.log"),
    "FV512b": ("FV", 512, "ens", f"{DIAG}/A512fv", None, None, f"{LOGS}/diagnostics/cgols_diag_A512fv.log"),
    "FV512f64": ("FV", 512, "fp64", "cgols_snapshots_A512fvf64", "_A512fvf64", None, f"{LOGS}/cgols_diag_A512fvf64.log"),
    "FV512hlle": ("FV", 512, "HLLE", f"{DIAG}/A512fvhlle", "_A512fvhlle", None,
                  f"{LOGS}/diagnostics/cgols_diag_A512fvhlle.log"),
    "FD1024": ("FD", 1024, "base", "cgols_snapshots_A1024", "_A1024", "_A1024", f"{LOGS}/cgols_diag_A1024.log"),
    "FV1024": ("FV", 1024, "base", "cgols_snapshots_A1024fv", "_A1024fv", "_A1024fv", f"{LOGS}/cgols_diag_A1024fv.log"),
}

#: Solver GPU-hours (H200) to 75 Myr from the console logs (time_integration wall x GPUs); 128/256 from the
#: un-checkpointed seed runs, 512 incl. checkpoint I/O (FD every frame, FV every 4th), 1024 on 4 GPUs (FD: 3.13
#: s/step x 42341 steps; its 61.5 h wall over 3 legs incl. restarts = 246 GPU-h).
GPU_HOURS = {
    ("FD", 128): 98.5 / 3600, ("FV", 128): 35.9 / 3600,
    ("FD", 256): 1154.8 / 3600, ("FV", 256): 198.6 / 3600,
    ("FD", 512): 20898.4 / 3600, ("FV", 512): 3571.9 / 3600,
    ("FD", 1024): 3.13 * 42341 * 4 / 3600, ("FV", 1024): 17449.4 * 4 / 3600,
}
GPU_HOURS_FD1024_TOTAL = 246.0

#: FV-PPM on GPU (HoreKa jobs 112240 / 112241, 1 H200 each, code b834d0f of branch fv-ppm): FV production (VL2 +
#: HLLC, CFL 0.4, FOFC, hard floor, fp32) with the AthenaPK ppm_simple corrector instead of PLM. Added only with
#: --ppm-gpu, so the default outputs do not change.
PPM_GPU_RUNS = {
    "FV256ppm": ("FV", 256, "PPM", "cgols_snapshots_A256fvppm", "_A256fvppm", None, f"{LOGS}/cgols_diag_A256fvppm.log"),
    "FV512ppm": ("FV", 512, "PPM", "cgols_snapshots_A512fvppm", "_A512fvppm", None, f"{LOGS}/cgols_diag_A512fvppm.log"),
}
#: label -> (time_integration wall [s], GPUs, diag log for the step count, note). PPM: HoreKa job logs
#: (cgols_logs/horeka/cgols_11224{0,1}.out); PLM / FD: local console logs as in GPU_HOURS (256: un-checkpointed seed
#: run, 512: the base runs incl. checkpoint I/O).
PPM_COST = {
    "FD256": (1154.8, 1, f"{LOGS}/diagnostics/cgols_diag_A256s1.log", "local, seed run, no ckpts"),
    "FV256": (198.6, 1, f"{LOGS}/diagnostics/cgols_diag_A256fvs1.log", "local, seed run, no ckpts"),
    "FV256ppm": (516.7, 1, f"{LOGS}/cgols_diag_A256fvppm.log", "HoreKa job 112240"),
    "FD512": (20898.4, 1, f"{LOGS}/cgols_diag_A512.log", "local, ckpt every frame"),
    "FV512": (3571.9, 1, f"{LOGS}/cgols_diag_A512fvf32.log", "local, ckpt every 4th frame"),
    "FV512ppm": (7458.5, 1, f"{LOGS}/cgols_diag_A512fvppm.log", "HoreKa job 112241"),
}

# -------------------------------------------------------------
# =============== ↓ Metrics ↓ =================================
# -------------------------------------------------------------

#: key -> (label, noise scaling 'rel' | 'abs', source, 3D twin for the borrowed sigma or None)
METRICS = {
    "out5_15_40": ("outflow |z|=5 kpc 15-40 Myr [Msun/yr]", "rel", "frames"),
    "out5_40_50": ("outflow |z|=5 kpc 40-50 Myr [Msun/yr]", "rel", "frames"),
    "out5_50_75": ("outflow |z|=5 kpc 50-75 Myr [Msun/yr]", "rel", "frames"),
    "out5_peak": ("outflow |z|=5 kpc peak [Msun/yr]", "rel", "frames"),
    "mout5_cum": ("mass out |z|=5 kpc 0-75 Myr [1e6 Msun]", "rel", "frames"),
    "lx1_15_40": ("L_X1 15-40 Myr [1e39 erg/s]", "rel", "emission"),
    "lx1_60_75": ("L_X1 60-75 Myr [1e39 erg/s]", "rel", "emission"),
    "lx2_15_40": ("AtomDB 0.3-2 keV 15-40 Myr [1e39]", "rel", "emission"),
    "lx2_60_75": ("AtomDB 0.3-2 keV 60-75 Myr [1e39]", "rel", "emission"),
    "ha_15_40": ("H-alpha 15-40 Myr [1e41 erg/s]", "rel", "emission"),
    "ha_60_75": ("H-alpha 60-75 Myr [1e41 erg/s]", "rel", "emission"),
    "pinch_60_75": ("axis pinch <log n> 60-75 Myr", "abs", "frames"),
    "pinched_frac": ("pinched fraction t>=60 Myr", "abs", "frames"),
    "asym_50_75": ("up-down asymmetry rms dlog n 50-75", "abs", "frames"),
    "ssv256": ("small-scale var. 80-300 pc (256 grid)", "rel", "frames"),
    "ssv128": ("small-scale var. 160-600 pc (128 grid)", "rel", "frames"),
    "cc85_dlogn": ("axis log n/n_CC85, 20-38 Myr", "abs", "frames"),
    "cc85_dv": ("axis v/v_CC85 - 1, 20-38 Myr", "abs", "frames"),
    "side_loss": ("side-face mass loss 0-75 Myr [1e6 Msun]", "rel", "frames"),
    "mass_75": ("box mass at 75 Myr [1e6 Msun]", "rel", "frames"),
    "wind1_15_40": ("3D: wind-entropy outflow 1 kpc 15-40 [Msun/yr]", "rel", "3d"),
    "mixed1_15_40": ("3D: mixed outflow 1 kpc 15-40 [Msun/yr]", "rel", "3d"),
    "mixed1_50_75": ("3D: entr.+mixed outflow 1 kpc 50-75 [Msun/yr]", "rel", "3d"),
    "edot5_15_40": ("3D: Edot |z|=5 kpc 15-40 [1e42 erg/s]", "rel", "3d"),
    "ecum5": ("3D: energy out |z|=5 kpc 5-75 Myr [1e57 erg]", "rel", "3d"),
    "cone1_20_40": ("3D: cone half-angle z=1 kpc 20-40 [deg]", "abs", "3d"),
    "cone1_60_75": ("3D: cone half-angle z=1 kpc 60-75 [deg]", "abs", "3d"),
    "wind1_t30": ("3D@30 Myr: wind-entropy outflow 1 kpc", "rel", "3d"),
    "edot1_t30": ("3D@30 Myr: Edot 1 kpc [1e42]", "rel", "3d"),
    "edot5_t30": ("3D@30 Myr: Edot 5 kpc [1e42]", "rel", "3d"),
    "cone1_t30": ("3D@30 Myr: cone half-angle z=1 [deg]", "abs", "3d"),
    "cone1_t60": ("3D@60 Myr: cone half-angle z=1 [deg]", "abs", "3d"),
    "mixedmass_t60": ("3D@60 Myr: mixed-entropy mass |z|>0.5 [1e8 Msun]", "rel", "3d"),
}
#: 3D metric -> frames metric whose relative sigma_traj it borrows
BORROW = {
    "wind1_15_40": "out5_15_40", "mixed1_15_40": "out5_15_40", "edot5_15_40": "out5_15_40",
    "mixed1_50_75": "out5_50_75", "ecum5": "mout5_cum", "cone1_20_40": "out5_15_40", "cone1_60_75": "out5_50_75",
    "wind1_t30": "out5_15_40", "edot1_t30": "out5_15_40", "edot5_t30": "out5_15_40", "cone1_t30": "out5_15_40",
    "cone1_t60": "out5_50_75", "mixedmass_t60": "out5_50_75",
}
WINDOW_MARGIN = 0.2  # Myr, as cgols_seed_scatter.window


def window(t, start, end):
    return (t >= start - WINDOW_MARGIN) & (t <= end + WINDOW_MARGIN)


def block_se(t, y, start, end, block=5.0):
    """Standard error of the window mean of y from block means (>= 2 blocks) of the linearly detrended series.

    A proxy for the realisation noise of a window mean in a quasi-stationary chaotic phase (late metrics only;
    for the laminar high state it would measure the deterministic transient, not noise).
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    inside = window(t, start, end) & np.isfinite(y)
    if inside.sum() < 4:
        return np.nan
    resid = y - np.polyval(np.polyfit(t[inside], y[inside], 1), t)
    edges = np.arange(start, end + 1e-6, block)
    means = []
    for a, b in itertools.pairwise(edges):
        sel = inside & (t >= a - WINDOW_MARGIN) & ((t < b - WINDOW_MARGIN) if b < end - 1e-6 else (t <= b + WINDOW_MARGIN))
        if sel.any():
            means.append(np.mean(resid[sel]))
    return float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) >= 2 else np.nan


#: Metrics of the laminar / deterministic phase (t < ~40 Myr, or time integrals dominated by it): their noise is
#: sigma_traj alone. All others are "late" (chaotic): sigma = max(sigma_traj, sigma_t).
EARLY = {"out5_15_40", "out5_peak", "mout5_cum", "lx1_15_40", "lx2_15_40", "ha_15_40", "cc85_dlogn", "cc85_dv",
         "wind1_15_40", "mixed1_15_40", "edot5_15_40", "ecum5", "cone1_20_40", "wind1_t30", "edot1_t30",
         "edot5_t30", "cone1_t30"}


# ---- CC85 free wind (copy of cgols_convergence_3d.cc85_profile, without astropy) ----
CC85_MDOT, CC85_EDOT, CC85_R = 12.0, 5.4e42, 0.3  # Msun/yr, erg/s, kpc (high state)


def cc85(radius_kpc, gamma=5.0 / 3.0):
    """(n [cm^-3], v [km/s]) of the CC85 wind at radius_kpc > R."""
    mdot = CC85_MDOT * 1.98847e33 / 3.15576e7
    v_inf = np.sqrt(2 * CC85_EDOT / mdot)
    kpc = 3.085677581e21

    def outside(mach, x):
        return mach ** (2 / (gamma - 1)) * (((gamma - 1) + 2 / mach**2) / (1 + gamma)) ** (
            (gamma + 1) / (2 * (gamma - 1))) - x**2

    n, v = [], []
    for r in np.atleast_1d(radius_kpc):
        mach = brentq(outside, 1.0, 1e4, args=(r / CC85_R,))
        vel = v_inf / np.sqrt(1 + 2 / ((gamma - 1) * mach**2))
        n.append(mdot / (4 * np.pi * (r * kpc) ** 2 * vel) / (0.6 * 1.6726e-24))
        v.append(vel / 1e5)
    return np.array(n), np.array(v)


_DIAG_RE = re.compile(r"t=(\S+)\s+min_rho=.*?mass=(\S+) dM=")
_INJ_T = (0.0, 5.0, 5.0 + 1e-7, 10.0, 40.0, 45.0, 1e3)
_INJ_R = (0.0, 0.0, 1.5, 12.0, 12.0, 1.5, 1.5)


def diag_mass(path):
    """(t [Myr], box mass [1e6 Msun]) from a per-step diag log, or None."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = [(float(m.group(1)), float(m.group(2))) for m in map(_DIAG_RE.search, fh) if m]
    a = np.array(rows)
    return a[:, 0] * MYR_PER_CODE_TIME, a[:, 1]


def frame_metrics(frame_dir, diag_path):
    """All frames-based metrics of one run (+ the series needed for sigma_t)."""
    names = sorted(f for f in os.listdir(frame_dir) if f.startswith("frame_") and f.endswith(".npz"))
    t, mdot, pin, asym, ssv256, ssv128, dlogn, dv = [], [], [], [], [], [], [], []
    zc = np.linspace(1.5, 4.0, 11)
    n_cc, v_cc = cc85(zc)
    for name in names:
        with np.load(os.path.join(frame_dir, name)) as f:
            frame = {k: f[k] for k in ("time_myr", "n_xz", "mdot_z", "vr_xz")}
        time = float(frame["time_myr"])
        t.append(time)
        mdot.append(frame["mdot_z"])
        pin.append(pinch(frame))
        log_n = np.log10(frame["n_xz"].astype(np.float64))
        asym.append(np.sqrt(np.mean((log_n - log_n[:, ::-1]) ** 2)))
        s = small_scale_fraction(frame, 256, (0.08, 0.3))
        ssv256.append(np.nan if s is None else s)
        ssv128.append(small_scale_fraction(frame, 128, (0.16, 0.6)))
        nx, nz = frame["n_xz"].shape
        z = (np.arange(nz) + 0.5) * BOX_Z / nz - BOX_Z / 2
        w = max(1, nx // 128)
        n_axis = frame["n_xz"][nx // 2 - w: nx // 2 + w].mean(0)
        v_axis = frame["vr_xz"][nx // 2 - w: nx // 2 + w].mean(0)
        # both sides folded: |z| = zc
        n_side = 0.5 * (np.interp(zc, z, n_axis) + np.interp(-zc, z, n_axis))
        v_side = 0.5 * (np.interp(zc, z, v_axis) + np.interp(-zc, z, v_axis))
        dlogn.append(np.median(np.log10(n_side / n_cc)))
        dv.append(np.median(v_side / v_cc - 1))
    order = np.argsort(t)
    t = np.array(t)[order]
    mdot = np.array(mdot)[order]
    arr = {k: np.array(v)[order] for k, v in
           (("pinch", pin), ("asym", asym), ("ssv256", ssv256), ("ssv128", ssv128), ("dlogn", dlogn), ("dv", dv))}
    out5 = net_outflow([{"mdot_z": m} for m in mdot], 5.0)
    zface = mdot[:, -1] - mdot[:, 0]
    cum = lambda y: np.concatenate([[0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])  # Myr * Msun/yr = 1e6 Msun
    late = window(t, 60, 75)
    m = {
        "out5_15_40": out5[window(t, 15, 40)].mean(),
        "out5_40_50": out5[window(t, 40, 50)].mean(),
        "out5_50_75": out5[window(t, 50, 75)].mean(),
        "out5_peak": out5.max(),
        "mout5_cum": cum(out5)[-1],
        "pinch_60_75": arr["pinch"][late].mean(),
        "pinched_frac": np.mean(arr["pinch"][late] > -3.5),
        "asym_50_75": arr["asym"][window(t, 50, 75)].mean(),
        "ssv256": np.nanmean(arr["ssv256"][window(t, 50, 75)]) if np.isfinite(arr["ssv256"]).any() else np.nan,
        "ssv128": arr["ssv128"][window(t, 50, 75)].mean(),
        "cc85_dlogn": arr["dlogn"][window(t, 20, 38)].mean(),
        "cc85_dv": arr["dv"][window(t, 20, 38)].mean(),
    }
    se = {
        "out5_15_40": block_se(t, out5, 15, 40), "out5_50_75": block_se(t, out5, 50, 75),
        "out5_40_50": block_se(t, out5, 40, 50), "pinch_60_75": block_se(t, arr["pinch"], 60, 75),
        "asym_50_75": block_se(t, arr["asym"], 50, 75), "ssv128": block_se(t, arr["ssv128"], 50, 75),
        "cc85_dlogn": block_se(t, arr["dlogn"], 20, 38), "cc85_dv": block_se(t, arr["dv"], 20, 38),
    }
    if np.isfinite(m["ssv256"]):
        se["ssv256"] = block_se(t, arr["ssv256"], 50, 75)
    diag = diag_mass(diag_path)
    if diag is not None:
        mass = np.interp(t, *diag)
        tt = np.linspace(0, 75, 150001)
        rate = np.interp(tt, _INJ_T, _INJ_R)
        inj = np.interp(t, tt, np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(tt))]))
        m["side_loss"] = float(diag[1][0] + inj[-1] - mass[-1] - cum(zface)[-1])
        m["mass_75"] = float(mass[-1])
    series = {"t": t.tolist(), "out5": out5.tolist(), "pinch": arr["pinch"].tolist(), "ssv256": arr["ssv256"].tolist(),
              "ssv128": arr["ssv128"].tolist()}
    return m, se, series


def emission_metrics(tag):
    e = np.load(f"data/studies/emission{tag}.npz", allow_pickle=True)
    t = e["per_snapshot__time_myr"]
    out, se = {}, {}
    for key, name, scale in (("lx1", "L_xray_cgols_xz", 1e39), ("lx2", "L_xray_atomdb_xz", 1e39),
                             ("ha", "L_halpha_xz", 1e41)):
        y = e[f"per_snapshot__{name}"] / scale
        for a, b in ((15, 40), (60, 75)):
            out[f"{key}_{a}_{b}"] = y[window(t, a, b)].mean()
            se[f"{key}_{a}_{b}"] = block_se(t, y, a, b)
    return out, se, {"te": t.tolist(), "lx1": (e["per_snapshot__L_xray_cgols_xz"] / 1e39).tolist()}


def metrics_3d(tag):
    """3D metrics on the common 5 Myr cadence (and at matched 30/60 Myr)."""
    rec = np.load(f"data/studies/convergence3d_v3{tag}.npz", allow_pickle=True)["records"].item()
    steps = sorted(rec, key=lambda s: rec[s]["time_myr"])
    t = np.array([rec[s]["time_myr"] for s in steps])
    on5 = np.abs(t / 5.0 - np.round(t / 5.0)) < 0.02
    steps, t = [s for s, k in zip(steps, on5) if k], t[on5]
    h1, h5 = int(np.argmin(np.abs(HEIGHTS_KPC - 1))), int(np.argmin(np.abs(HEIGHTS_KPC - 5)))

    def cone(r):
        z = r["z_kpc"]
        north = z > 0
        area = 0.5 * (r["zarea_fast"][north] + r["zarea_fast"][~north][::-1])
        k = int(np.argmin(np.abs(z[north] - 1.0)))
        return np.degrees(np.arctan(np.sqrt(area[k] / np.pi) / z[north][k]))

    ser = {
        "wind1": np.array([rec[s]["mdot_out_K"][h1, 2] for s in steps]),
        "mixed1": np.array([rec[s]["mdot_out_K"][h1, 1] for s in steps]),
        "entmixed1": np.array([rec[s]["mdot_out_K"][h1, 0] + rec[s]["mdot_out_K"][h1, 1] for s in steps]),
        "edot5": np.array([rec[s]["edot_net"][h5].sum() / 1e42 for s in steps]),
        "edot1": np.array([rec[s]["edot_net"][h1].sum() / 1e42 for s in steps]),
        "cone": np.array([cone(rec[s]) for s in steps]),
        "mixedmass": np.array([rec[s]["mass_halo_K"][1] / 1e8 for s in steps]),
    }
    out, se = {}, {}

    def win(key, name, a, b):
        sel = window(t, a, b)
        if sel.sum() >= 3:
            out[name] = ser[key][sel].mean()
            se[name] = block_se(t, ser[key], a, b, block=10.0) if b - a >= 20 else np.nan

    win("wind1", "wind1_15_40", 15, 40)
    win("mixed1", "mixed1_15_40", 15, 40)
    win("entmixed1", "mixed1_50_75", 50, 75)
    win("edot5", "edot5_15_40", 15, 40)
    win("cone", "cone1_20_40", 20, 40)
    win("cone", "cone1_60_75", 60, 75)
    if window(t, 5, 75).sum() >= 10:  # full 5 Myr cadence only
        sel = window(t, 5, 75)
        tt, yy = t[sel], ser["edot5"][sel] * 1e42 * 3.15576e13  # erg/s * s/Myr
        out["ecum5"] = float(np.sum(0.5 * (yy[1:] + yy[:-1]) * np.diff(tt)) / 1e57)
    for key, name, T in (("wind1", "wind1_t30", 30), ("edot1", "edot1_t30", 30), ("edot5", "edot5_t30", 30),
                         ("cone", "cone1_t30", 30), ("cone", "cone1_t60", 60), ("mixedmass", "mixedmass_t60", 60)):
        k = int(np.argmin(np.abs(t - T)))
        if abs(t[k] - T) < 0.1:
            out[name] = ser[key][k]
            nb = [ser[key][j] for j in (k - 1, k + 1) if 0 <= j < len(t) and abs(t[j] - T) <= 5.1]
            # snapshot-to-snapshot variability (neighbours 5 Myr away) as a crude sigma_t
            se[name] = float(np.std(np.r_[ser[key][k], nb], ddof=1)) if nb else np.nan
    return out, se


def measure(label, spec):
    _scheme, _n, _role, frame_dir, emis, tag3d, diag = spec
    m, se, series = frame_metrics(frame_dir, diag)
    if emis and os.path.exists(f"data/studies/emission{emis}.npz"):
        a, b, s = emission_metrics(emis)
        m.update(a)
        se.update(b)
        series.update(s)
    if tag3d and os.path.exists(f"data/studies/convergence3d_v3{tag3d}.npz"):
        a, b = metrics_3d(tag3d)
        m.update(a)
        se.update(b)
    clean = lambda d: {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in d.items()}
    return {"m": clean(m), "se": clean(se), "series": series, "nframes": len(series["t"])}


def load_all(recompute, extra):
    runs = dict(RUNS)
    runs.update(extra)
    cache = {}
    if os.path.exists(CACHE) and not recompute:
        with open(CACHE) as fh:
            cache = json.load(fh)
    for label, spec in runs.items():
        if not os.path.isdir(spec[3]):
            print(f"  [skip] {label}: no {spec[3]}")
            continue
        nframes = len([f for f in os.listdir(spec[3]) if f.startswith("frame_")])
        if label in cache and cache[label]["nframes"] == nframes:
            continue
        print(f"  measuring {label} ({spec[3]}, {nframes} frames)", flush=True)
        cache[label] = measure(label, spec)
    with open(CACHE, "w") as fh:
        json.dump(cache, fh)
    return runs, {k: v for k, v in cache.items() if k in runs}


# -------------------------------------------------------------
# =============== ↓ Statistics ↓ ==============================
# -------------------------------------------------------------


def stats(runs, data):
    """Per metric and (scheme, N): value, sigma_traj (measured or extrapolated), sigma_t, sigma."""
    out = {}
    for key, (_, scaling, _) in METRICS.items():
        cell = {}
        for scheme in ("FD", "FV"):
            for n in RES:
                base = [lab for lab, s in runs.items() if s[:3] == (scheme, n, "base") and lab in data]
                if not base or data[base[0]]["m"].get(key) is None:
                    continue
                members = [data[lab]["m"].get(key) for lab, s in runs.items()
                           if s[0] == scheme and s[1] == n and s[2] in ("base", "ens") and lab in data]
                members = np.array([v for v in members if v is not None])
                value = data[base[0]]["m"][key]
                cell[(scheme, n)] = {
                    "value": value, "members": members, "n_members": len(members),
                    "sig_meas": float(np.std(members, ddof=1)) if len(members) >= 2 else None,
                    "sig_t": data[base[0]]["se"].get(key),
                }
        out[key] = cell
    # extrapolate sigma_traj
    for key, (_, scaling, src) in METRICS.items():
        cell = out[key]

        def norm(c, scaling=scaling):
            return abs(c["value"]) if scaling == "rel" and c["value"] else 1.0

        for (scheme, n), c in cell.items():
            if src == "3d":
                twin = out[BORROW[key]].get((scheme, n))
                rel = twin["sig_traj"] / abs(twin["value"]) if twin and twin.get("sig_traj") is not None else 0.0
                c["sig_traj"], c["sig_traj_pess"], c["extrap"] = rel * abs(c["value"]), rel * abs(c["value"]), True
                continue
            measured = [(m, cc["sig_meas"] / norm(cc)) for (s, m), cc in cell.items() if cc["sig_meas"] is not None]
            below = [r for m, r in measured if m <= n]
            if c["sig_meas"] is not None:
                c["sig_traj"], c["extrap"] = c["sig_meas"], False
            else:
                c["sig_traj"], c["extrap"] = (max(below) * norm(c) if below else None), True
            c["sig_traj_pess"] = c["sig_traj"]
            fv = {m: cc["sig_meas"] / norm(cc) for (s, m), cc in cell.items() if s == "FV" and cc["sig_meas"]}
            if n == 1024 and c["extrap"] and 256 in fv and 512 in fv and c["sig_traj"] is not None:
                c["sig_traj_pess"] = max(c["sig_traj"], fv[512] * max(1.0, fv[512] / fv[256]) * norm(c))
        for c in cell.values():
            if key in EARLY:
                c["sig_t_used"] = None
            else:
                c["sig_t_used"] = c.get("sig_t")
            parts = [v for v in (c.get("sig_traj"), c["sig_t_used"]) if v is not None and np.isfinite(v)]
            c["sigma"] = max(parts) if parts else np.nan
            parts_p = [v for v in (c.get("sig_traj_pess"), c["sig_t_used"]) if v is not None and np.isfinite(v)]
            c["sigma_pess"] = max(parts_p) if parts_p else np.nan
    return out


def differences(cell):
    rows = {}
    for n in RES:
        if ("FD", n) in cell and ("FV", n) in cell:
            fd, fv = cell[("FD", n)], cell[("FV", n)]
            delta = fd["value"] - fv["value"]
            sig = np.hypot(fd["sigma"], fv["sigma"])
            sig_lo = np.hypot(fd.get("sig_traj") or 0.0, fv.get("sig_traj") or 0.0)
            sig_p = np.hypot(fd["sigma_pess"], fv["sigma_pess"])
            # the schemes' own resolution step into N (128: the step 128 -> 256)
            steps = []
            for scheme in ("FD", "FV"):
                a, b = (n // 2, n) if n > 128 else (128, 256)
                if (scheme, a) in cell and (scheme, b) in cell:
                    steps.append(abs(cell[(scheme, b)]["value"] - cell[(scheme, a)]["value"]))
            res_step = max(steps) if steps else np.nan
            rows[n] = {"delta": delta, "sigma": sig, "z": delta / sig if sig > 0 else np.inf,
                       "res_step": res_step, "R": abs(delta) / res_step if res_step > 0 else np.inf,
                       "z_lo": delta / sig_lo if sig_lo > 0 else np.inf,
                       "z_pess": delta / sig_p if sig_p > 0 else np.inf,
                       "rel": delta / (0.5 * (abs(fd["value"]) + abs(fv["value"])) or 1.0)}
    return rows


def verdict(row):
    """'=' within noise (|z| < 3); '<' differs, but by less than one resolution doubling (R < 1); '>' by more."""
    if abs(row["z"]) < 3:
        return "="
    return "<" if row["R"] < 1 else ">"


def trend(rows, ns=(256, 512, 1024)):
    """Weighted fit |Delta| ~ N^b over ns if Delta keeps its sign; (b, err, note)."""
    pts = [(n, rows[n]["delta"], rows[n]["sigma"]) for n in ns if n in rows]
    if len(pts) < 3:
        return None, None, "<3 points"
    signs = {np.sign(d) for _, d, _ in pts}
    if len(signs) > 1:
        return None, None, "sign flips"
    x = np.log2([p[0] for p in pts])
    y = np.log2([abs(p[1]) for p in pts])
    w = np.array([abs(p[1]) / max(p[2], 1e-30) for p in pts])  # 1 / sigma of log2|Delta| ~ |Delta|/sigma
    w = np.clip(w, 1e-3, 1e3)
    A = np.vstack([x, np.ones_like(x)]).T * w[:, None]
    coef, *_ = np.linalg.lstsq(A, y * w, rcond=None)
    resid = y - (coef[0] * x + coef[1])
    cov = np.linalg.inv(A.T @ A) * max(1.0, np.sum((resid * w) ** 2) / (len(x) - 2))
    return coef[0], float(np.sqrt(cov[0, 0])) / np.log2(2), "ok"


def richardson(cell, scheme, ns=(256, 512, 1024), draws=4000):
    """Observed order and Richardson value of one scheme's ladder, with an MC over sigma."""
    if not all((scheme, n) in cell for n in ns):
        return None
    v = np.array([cell[(scheme, n)]["value"] for n in ns])
    s = np.array([cell[(scheme, n)]["sigma"] for n in ns])
    s = np.where(np.isfinite(s), s, 0.0)
    d1, d2 = v[1] - v[0], v[2] - v[1]
    out = {"d1": d1, "d2": d2, "values": v.tolist()}
    mono = np.sign(d1) == np.sign(d2) and abs(d2) < abs(d1)
    out["monotone"] = bool(mono)
    if mono:
        p = np.log2(abs(d1) / abs(d2))
        out["p"], out["minf"] = p, v[2] + d2 / (2**p - 1)
    sample = v[None, :] + RNG.standard_normal((draws, 3)) * s[None, :]
    a, b = sample[:, 1] - sample[:, 0], sample[:, 2] - sample[:, 1]
    ok = (np.sign(a) == np.sign(b)) & (np.abs(b) < np.abs(a))
    out["frac_mono"] = float(ok.mean())
    if ok.sum() > 20:
        p = np.log2(np.abs(a[ok]) / np.abs(b[ok]))
        minf = sample[ok, 2] + b[ok] / (2**p - 1)
        out["p_q"] = np.percentile(p, [16, 50, 84]).tolist()
        out["minf_q"] = np.percentile(minf, [16, 50, 84]).tolist()
    return out


#: bimodal / shape metrics for which an equivalent resolution is not meaningful
NO_K = {"pinch_60_75", "pinched_frac", "asym_50_75"}


def k_factor(cell, n, draws=2000, key=None):
    """FV(n) = FD(k n): interpolate the FD ladder in log2 N; (k, 16-84% or None, note)."""
    if key in NO_K:
        return None, None, "n/a (bimodal)"
    if ("FV", n) not in cell:
        return None, None, "no FV"
    ns = [m for m in RES if ("FD", m) in cell]

    def monotone(points):
        d = np.diff([cell[("FD", m)]["value"] for m in points])
        return len(points) >= 2 and (np.all(d > 0) or np.all(d < 0))

    if not monotone(ns):  # the full FD ladder must be monotone; else try 256-1024 only
        ns = [m for m in ns if m >= 256]
        if not monotone(ns):
            return None, None, "FD non-monotone"

    def solve(fd, target):
        x = np.log2(ns)
        hits = []
        for i in range(len(ns) - 1):
            a, b = fd[i], fd[i + 1]
            if (a - target) * (b - target) <= 0 and a != b:
                hits.append(x[i] + (target - a) / (b - a) * (x[i + 1] - x[i]))
        if len(hits) == 1:
            return 2 ** hits[0] / n
        if not hits:
            lo = min(fd)
            return -np.inf if (fd[-1] > fd[0]) == (target < lo) else np.inf  # below / above the ladder
        return np.nan  # ambiguous (non-monotone FD)

    fd = np.array([cell[("FD", m)]["value"] for m in ns])
    target = cell[("FV", n)]["value"]
    k = solve(fd, target)
    sfd = np.nan_to_num(np.array([cell[("FD", m)]["sigma"] for m in ns]))
    sfv = np.nan_to_num(cell[("FV", n)]["sigma"])
    ks = np.array([solve(fd + RNG.standard_normal(len(ns)) * sfd, target + RNG.standard_normal() * sfv)
                   for _ in range(draws)])
    finite = ks[np.isfinite(ks)]
    q = np.percentile(finite, [16, 84]).tolist() if finite.size > draws * 0.5 else None
    if np.isfinite(k):
        note = "ok"
    elif np.isnan(k):
        note = "ambiguous"
    else:  # FV outside the FD ladder: k beyond its finest / below its coarsest member
        note = f"> {ns[-1] / n:g}" if k > 0 else f"< {ns[0] / n:g}"
    return k, q, note


# -------------------------------------------------------------
# =============== ↓ Output ↓ ==================================
# -------------------------------------------------------------


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 0.1 or a == 0:
        return f"{v:.{digits - 1}f}" if digits > 1 else f"{v:.2f}"
    return f"{v:.2g}"


def tables(st, runs, data):
    summary = {}
    print("\n## Metric table: FD | FV (+- sigma) and z = (FD-FV)/sigma_diff per N  [* = sigma_traj extrapolated; "
          "late metrics: z with sigma = max(traj, t) / z with sigma_traj only; (p ...) = pessimistic 1024]\n")
    print("| metric | " + " | ".join(f"{n}: FD / FV | z" for n in RES)
          + " | slope b of abs(FD-FV)~N^b (256-1024) | R = abs(FD-FV) / resolution step, per N; verdict |")
    print("|---|" + "---|---|" * len(RES) + "---|---|")
    for key, (label, _, _) in METRICS.items():
        cell = st[key]
        rows = differences(cell)
        b, berr, note = trend(rows)
        line = f"| {label} |"
        for n in RES:
            fd, fv = cell.get(("FD", n)), cell.get(("FV", n))
            if fd and fv:
                star = lambda c: "*" if c.get("extrap") else ""
                r = rows[n]
                zs = fmt(r["z"], 2) + ("" if key in EARLY else f" / {fmt(r['z_lo'], 2)}")
                zs += f" (p {fmt(r['z_pess'], 2)})" if n == 1024 and r["z_pess"] != r["z"] else ""
                line += (f" {fmt(fd['value'])}±{fmt(fd['sigma'], 2)}{star(fd)} / "
                         f"{fmt(fv['value'])}±{fmt(fv['sigma'], 2)}{star(fv)} | {zs} |")
            else:
                vals = " / ".join(fmt(c["value"]) if c else "-" for c in (fd, fv))
                line += f" {vals} | - |"
        line += f" {fmt(b, 2)}±{fmt(berr, 2)} |" if b is not None else f" {note} |"
        line += " " + " ".join(f"{fmt(rows[n]['R'], 2)}{verdict(rows[n])}" for n in RES if n in rows) + " |"
        print(line)
        summary[key] = {"rows": {str(n): r for n, r in rows.items()}, "trend": [b, berr, note]}

    print("\n## Self-convergence per scheme (256 -> 512 -> 1024): d1, d2, order p, Richardson M_inf; "
          "MC: fraction monotone, p and M_inf 16/50/84%\n")
    print("| metric | scheme | 256 / 512 / 1024 | d1 | d2 | p | M_inf | MC mono | MC p | MC M_inf |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, (label, _, _) in METRICS.items():
        for scheme in ("FD", "FV"):
            r = richardson(st[key], scheme)
            if r is None:
                continue
            q = lambda name, r=r: "/".join(fmt(x) for x in r[name]) if name in r else "-"
            print(f"| {label} | {scheme} | {' / '.join(fmt(x) for x in r['values'])} | {fmt(r['d1'])} | "
                  f"{fmt(r['d2'])} | {fmt(r.get('p'), 2)} | {fmt(r.get('minf'))} | {r['frac_mono']:.2f} | "
                  f"{q('p_q')} | {q('minf_q')} |")
            summary[key][f"rich_{scheme}"] = r

    print("\n## Equivalent resolution k: FV(N) = FD(k N)  (16-84%)\n")
    print("| metric | " + " | ".join(f"k @ FV-{n}" for n in RES) + " |")
    print("|---|" + "---|" * len(RES))
    for key, (label, _, _) in METRICS.items():
        line = f"| {label} |"
        ks = {}
        for n in RES:
            k, q, note = k_factor(st[key], n, key=key)
            ks[n] = (k, q, note)
            if k is None:
                line += f" {note} |"
            elif np.isfinite(k):
                line += f" {k:.2f}" + (f" [{q[0]:.2f}, {q[1]:.2f}]" if q else " [unbounded]") + " |"
            else:
                line += f" {note} |"
        print(line)
        summary[key]["k"] = {str(n): [v[0], v[1], v[2]] for n, v in ks.items()}

    print("\n## Ablations: offset from their FV/FD baseline at the same N, in units of the baseline's sigma\n")
    print("| run | role | " + " | ".join(METRICS[k][0] for k in ABLATION_KEYS) + " |")
    print("|---|---|" + "---|" * len(ABLATION_KEYS))
    abl = {}
    for lab, (scheme, n, role, *_rest) in runs.items():
        if role in ("base", "ens", "duplicate") or lab not in data:
            continue
        base = next(lb for lb, s in runs.items() if s[:3] == (scheme, n, "base"))
        cells = []
        for key in ABLATION_KEYS:
            v, b0 = data[lab]["m"].get(key), data[base]["m"].get(key)
            sig = st[key].get((scheme, n), {}).get("sigma")
            if v is None or b0 is None:
                cells.append("-")
                continue
            z = (v - b0) / sig if sig else np.inf
            cells.append(f"{fmt(v)} ({z:+.1f}σ)")
            abl.setdefault(lab, {})[key] = [v, b0, z]
        print(f"| {lab} | {role} | " + " | ".join(cells) + " |")
    summary["_ablations"] = abl

    print("\n## Attribution: fraction f of the FD - FV gap at the same N that an ablation closes, "
          "f = (ablation - FV) / (FD - FV) (1 = the ablation reproduces FD, 0 = no effect, < 0 widens); "
          "FV ablations also g = (ablation - FV(N)) / (FV(2N) - FV(N)) (1 = like doubling N). "
          "+- from sigma of the ablation and baseline; † = the gap itself is < 2 sigma_diff\n")
    print("| run | role | " + " | ".join(METRICS[k][0] for k in ATTR_KEYS) + " |")
    print("|---|---|" + "---|" * len(ATTR_KEYS))
    attr = {}
    for lab, (scheme, n, role, *_rest) in runs.items():
        if role in ("base", "ens", "duplicate") or lab not in data or scheme != "FV":
            continue
        base = next(lb for lb, s in runs.items() if s[:3] == (scheme, n, "base"))
        cells = []
        for key in ATTR_KEYS:
            v, b0 = data[lab]["m"].get(key), data[base]["m"].get(key)
            c_fd, c_fv = st[key].get(("FD", n)), st[key].get(("FV", n))
            c_2n = st[key].get(("FV", 2 * n))
            if v is None or b0 is None or c_fd is None:
                cells.append("-")
                continue
            sig = c_fv["sigma"] if np.isfinite(c_fv["sigma"]) else 0.0
            gap = c_fd["value"] - b0
            txt = "-"
            if gap != 0:
                f = (v - b0) / gap
                weak = abs(gap) <= 2 * np.hypot(sig, c_fd["sigma"])
                txt = f"f {f:+.2f}±{np.sqrt(2) * sig / abs(gap):.2f}" + ("†" if weak else "")
                attr.setdefault(lab, {})[key] = {"f": f, "weak_gap": bool(weak)}
            if c_2n is not None and abs(c_2n["value"] - b0) > 2 * sig:
                g = (v - b0) / (c_2n["value"] - b0)
                txt += f"; g {g:+.2f}"
                attr.setdefault(lab, {}).setdefault(key, {})["g"] = g
            cells.append(txt)
        print(f"| {lab} | {role} | " + " | ".join(cells) + " |")
    summary["_attribution"] = attr
    return summary


ATTR_KEYS = ("out5_15_40", "out5_40_50", "out5_50_75", "out5_peak", "lx1_15_40", "lx1_60_75", "ha_60_75",
             "pinch_60_75", "asym_50_75", "ssv128", "ssv256", "cc85_dv", "side_loss", "mass_75")


ABLATION_KEYS = ("out5_15_40", "out5_40_50", "out5_50_75", "out5_peak", "lx1_15_40", "lx1_60_75", "pinch_60_75",
                 "asym_50_75", "ssv128", "ssv256", "cc85_dv", "side_loss")


def noise_table(st):
    print("\n## Noise: measured relative sigma_traj (n members) and sigma_t of the base run\n")
    print("| metric | " + " | ".join(f"{s}-{n}" for n in RES for s in ("FD", "FV")) + " |")
    print("|---|" + "---|" * (2 * len(RES)))
    for key in ("out5_15_40", "out5_40_50", "out5_50_75", "out5_peak", "mout5_cum", "lx1_15_40", "lx1_60_75",
                "lx2_60_75", "ha_60_75", "pinch_60_75", "asym_50_75", "ssv128", "ssv256", "cc85_dv", "side_loss"):
        label, scaling, _ = METRICS[key]
        line = f"| {label} |"
        for n in RES:
            for s in ("FD", "FV"):
                c = st[key].get((s, n))
                if not c:
                    line += " - |"
                    continue
                nrm = abs(c["value"]) if scaling == "rel" else 1.0
                tr = c["sig_meas"]
                txt = (f"{tr / nrm:.2g} ({c['n_members']})" if tr is not None else "-")
                txt += f"; t {c['sig_t'] / nrm:.2g}" if c.get("sig_t") is not None else ""
                line += f" {txt} |"
        print(line)


# -------------------------------------------------------------
# =============== ↓ Figures ↓ =================================
# -------------------------------------------------------------

FIG_KEYS = ("out5_15_40", "out5_40_50", "out5_50_75", "out5_peak", "mout5_cum", "lx1_15_40", "lx1_60_75",
            "lx2_60_75", "ha_60_75", "pinch_60_75", "asym_50_75", "ssv128", "ssv256", "cc85_dv", "side_loss",
            "wind1_15_40", "edot5_15_40", "cone1_20_40")


HEAT_KEYS = ("out5_15_40", "out5_40_50", "out5_50_75", "out5_peak", "mout5_cum", "lx1_15_40", "lx1_60_75",
             "lx2_15_40", "lx2_60_75", "ha_15_40", "ha_60_75", "pinch_60_75", "pinched_frac", "asym_50_75",
             "ssv256", "ssv128", "cc85_dlogn", "cc85_dv", "side_loss", "mass_75", "wind1_t30", "edot1_t30",
             "edot5_t30", "cone1_t30", "cone1_t60", "mixedmass_t60", "mixed1_15_40", "edot5_15_40", "ecum5",
             "cone1_60_75")


def fig_significance(st):
    """Heat maps metric x N: (a) z = (FD - FV) / sigma with the relative difference and R, (b) the k factor."""
    from matplotlib.colors import SymLogNorm, TwoSlopeNorm

    keys = [k for k in HEAT_KEYS if differences(st[k])]
    z = np.full((len(keys), len(RES)), np.nan)
    kk = np.full((len(keys), len(RES)), np.nan)
    fig, (ax, ax_k) = plt.subplots(1, 2, figsize=(17, 0.42 * len(keys) + 2.6), gridspec_kw={"width_ratios": (1.5, 1)})
    for i, key in enumerate(keys):
        rows = differences(st[key])
        for j, n in enumerate(RES):
            if n in rows:
                r = rows[n]
                z[i, j] = np.clip(r["z"], -1e3, 1e3)
                rel = f"{100 * r['rel']:+.0f}%" if METRICS[key][1] == "rel" else f"{r['delta']:+.2g}"
                ax.text(j, i, f"{rel}  R {fmt(r['R'], 2)} {verdict(r)}", ha="center", va="center", fontsize=7.5,
                        color="k" if abs(r["z"]) < 30 else "w")
            k, q, note = k_factor(st[key], n, key=key)
            txt = ""
            if k is not None and np.isfinite(k) and k > 0:
                kk[i, j] = np.log2(k)
                txt = f"{k:.2f}" + (f"\n[{q[0]:.2f}-{q[1]:.2f}]" if q else "\n[unbounded]")
            elif note == "FD non-monotone":
                txt = "FD non-\nmonotone"
            elif note in ("ambiguous", "n/a (bimodal)"):
                txt = note
            elif k is not None:
                kk[i, j] = 3.5 if k > 0 else -3.5
                txt = note
            dark = np.isfinite(kk[i, j]) and abs(kk[i, j]) > 2.3
            ax_k.text(j, i, txt, ha="center", va="center", fontsize=6.8, color="w" if dark else "k")
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", norm=SymLogNorm(linthresh=3, vmin=-100, vmax=100))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("z = (FD - FV) / sigma_diff (symlog, |z| < 3 pale)")
    imk = ax_k.imshow(kk, aspect="auto", cmap="PuOr", norm=TwoSlopeNorm(0, -3.5, 3.5))
    cbk = fig.colorbar(imk, ax=ax_k, fraction=0.05, pad=0.01)
    cbk.set_label("log2 k   (FV(N) = FD(k N); k > 1: FV acts like a finer FD grid)")
    labels = [METRICS[k][0] + ("" if k in EARLY else "  [late]") for k in keys]
    for a_ in (ax, ax_k):
        a_.set_xticks(range(len(RES)), [str(n) for n in RES])
        a_.set_xlabel("N")
    ax.set_yticks(range(len(keys)), labels, fontsize=8)
    ax_k.set_yticks(range(len(keys)), [""] * len(keys))
    ax.set_title("(a) FD - FV: relative difference, R = |FD-FV| / resolution step, verdict\n"
                 "(= within noise |z| < 3; < differs by less than one resolution doubling; > by more)",
                 fontsize=10)
    ax_k.set_title("(b) equivalent resolution k (16-84% over sigma);\nblank: no FV run or FD ladder not monotone there",
                   fontsize=10)
    fig.suptitle("CGOLS model A, FD (WENO5-Z) vs FV (VL2 + PLM + HLLC): every metric, 128-1024", fontsize=12)
    fig.tight_layout()
    path = f"{FIG_DIR}/fvfd_summary_significance.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(path)


CONV_KEYS = ("out5_15_40", "out5_40_50", "out5_50_75", "out5_peak", "lx1_15_40", "lx1_60_75", "pinch_60_75",
             "asym_50_75", "ssv256", "ssv128", "cc85_dv", "side_loss")


def fig_convergence(st, runs, data):
    fig, axes = plt.subplots(3, 4, figsize=(17, 11.5))
    abl_style = {"HLLE": ("C2", "s"), "fp64": ("C4", "D"), "CFL 0.3": ("C8", "v"), "PPM": ("C1", "*"),
                 "PLM CPU": ("C5", "x"), "no dual energy": ("C6", "^")}
    for ax, key in zip(axes.ravel(), CONV_KEYS):
        cell = st[key]
        for scheme, color, off in (("FD", "C0", 0.97), ("FV", "C3", 1.03)):
            ns = [n for n in RES if (scheme, n) in cell]
            v = np.array([cell[(scheme, n)]["value"] for n in ns])
            s = np.array([cell[(scheme, n)]["sigma"] for n in ns])
            ax.errorbar(np.array(ns) * off, v, yerr=np.nan_to_num(s), color=color, marker="o", capsize=3,
                        label=f"{scheme} {'WENO5' if scheme == 'FD' else 'VL2+HLLC'}")
            for n in ns:
                mem = cell[(scheme, n)]["members"]
                if len(mem) > 1:
                    ax.plot(np.full(len(mem), n * off * (0.95 if scheme == "FD" else 1.05)), mem, ls="none",
                            marker="_", color=color, alpha=0.6, ms=6)
            r = richardson(cell, scheme)
            if r and "minf_q" in r and r["frac_mono"] > 0.5:
                lo, _mid, hi = r["minf_q"]
                ax.axhspan(lo, hi, xmin=0.9, xmax=1.0, color=color, alpha=0.25)
        for lab, (scheme, n, role, *_r) in runs.items():
            if role in ("base", "ens", "duplicate") or lab not in data or data[lab]["m"].get(key) is None:
                continue
            color, mk = abl_style.get(role, ("0.4", "x")) if scheme == "FV" else ("0.25", "^" if "dual" in role else "D")
            ax.plot([n * 1.09], [data[lab]["m"][key]], marker=mk, color=color, ls="none", ms=8 if mk == "*" else 6,
                    label=f"FV {role}" if scheme == "FV" else f"FD {role}")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(RES, [str(n) for n in RES])
        ax.set_xlim(100, 1500)
        if key in ("lx1_60_75", "lx1_15_40", "ssv256", "ssv128"):
            ax.set_yscale("log")
        ax.set_title(METRICS[key][0], fontsize=9.5)
        ax.grid(alpha=0.3)
    handles = {}
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=8, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Self-convergence of FD and FV (error bars: conservative sigma; dashes: realisations; "
                 "right-edge band: Richardson M_inf 16-84% if monotone; offset markers: ablations)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path = f"{FIG_DIR}/fvfd_convergence.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(path)


COST_KEYS = ("out5_peak", "lx1_60_75", "ssv256", "cc85_dv")


def fig_cost(st):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    ax = axes[0]
    for i, key in enumerate(COST_KEYS):
        cell = st[key]
        for scheme, ls in (("FD", "-"), ("FV", "--")):
            ns = [n for n in RES if (scheme, n) in cell]
            pts = []
            for a, b in itertools.pairwise(ns):
                va, vb = cell[(scheme, a)]["value"], cell[(scheme, b)]["value"]
                pts.append((GPU_HOURS[(scheme, b)], abs(vb - va) / abs(vb)))
            if pts:
                x, y = zip(*pts)
                ax.loglog(x, np.maximum(y, 1e-4), ls=ls, marker="o" if scheme == "FD" else "s", color=f"C{i}",
                          label=f"{METRICS[key][0]} {scheme}" if True else None)
    ax.set(xlabel="GPU-hours (H200) of the finer run", ylabel="|M(N) - M(N/2)| / |M(N)|  (self-convergence error)",
           title="Cost per accuracy: solid FD, dashed FV")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    ax = axes[1]
    for scheme, color in (("FD", "C0"), ("FV", "C3")):
        ax.loglog(RES, [GPU_HOURS[(scheme, n)] for n in RES], marker="o", color=color, label=scheme)
    ax.loglog([1024], [GPU_HOURS_FD1024_TOTAL], marker="o", mfc="none", color="C0", ls="none",
              label="FD-1024 incl. 3-leg restarts/I/O")
    for n in RES:
        ratio = GPU_HOURS[("FD", n)] / GPU_HOURS[("FV", n)]
        ax.text(n, GPU_HOURS[("FV", n)] * 0.55, f"x{ratio:.1f}", ha="center", fontsize=9, color="0.3")
    ax.set(xlabel="N", ylabel="GPU-hours to 75 Myr", title="Run cost (FD / FV ratio under the FV points)",
           ylim=(3e-3, 600))
    ax.minorticks_off()
    ax.set_xticks(RES, [str(n) for n in RES])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = f"{FIG_DIR}/fvfd_cost.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(path)


def fig_ppm(runs, data, st, labels):
    """FV-PPM (and the CPU PLM reference) at 128 against FD-128, FV-128 (+ realisations) and FV-256."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax_out, ax_late, ax_x = axes[0]
    styles = {"FD128": ("C0", "-", "FD-128"), "FV128": ("C3", "-", "FV-128 PLM (GPU)"),
              "FV256": ("C3", ":", "FV-256 PLM"), "FD256": ("C0", ":", "FD-256")}
    for k, lab in enumerate(labels):
        styles[lab] = (("C1", "C5", "C6")[k % 3], "-", f"FV-128 {runs[lab][2]}")
    for lab, (color, ls, name) in styles.items():
        if lab not in data:
            continue
        ser = data[lab]["series"]
        t, out5 = np.array(ser["t"]), np.array(ser["out5"])
        lw = 2.0 if lab in labels else 1.2
        ax_out.plot(t, out5, color=color, ls=ls, lw=lw, label=name)
        ax_late.plot(t, out5, color=color, ls=ls, lw=lw)
        if "te" in ser:
            ax_x.semilogy(ser["te"], ser["lx1"], color=color, ls=ls, lw=lw)
    for lab, s in runs.items():  # FV-128 realisations as thin lines
        if s[:2] == ("FV", 128) and s[2] == "ens" and lab in data:
            ser = data[lab]["series"]
            ax_late.plot(ser["t"], ser["out5"], color="C3", lw=0.5, alpha=0.5)
    ax_out.set(xlabel="t [Myr]", ylabel="net outflow |z| = 5 kpc [Msun/yr]", xlim=(0, 75))
    ax_late.set(xlabel="t [Myr]", ylabel="net outflow |z| = 5 kpc [Msun/yr]", xlim=(40, 75), ylim=(0, 8),
                title="late phase (thin: FV-128 realisations)")
    ax_x.set_title("soft X-ray")
    ax_x.set(xlabel="t [Myr]", ylabel="L_X1 [1e39 erg/s]", xlim=(0, 75), ylim=(1, 3e3))
    ax_out.legend(fontsize=8)
    for ax in axes[0]:
        ax.grid(alpha=0.3)
    # bar panels: position of each 128 run between FV-128 (0) and FD-128 / FV-256
    keys = ("out5_15_40", "out5_peak", "out5_50_75", "lx1_15_40", "lx1_60_75", "ssv128", "cc85_dv", "side_loss")
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1]:
        ax.remove()
    ax_b = fig.add_subplot(gs[1, :])
    width = 0.8 / (len(labels) + 2)
    refs = [("FD128", "C0", "FD-128"), ("FV256", "C3", "FV-256")] + [(lab, styles[lab][0], styles[lab][2])
                                                                     for lab in labels]
    for j, (lab, color, name) in enumerate(refs):
        vals = []
        for key in keys:
            base = data["FV128"]["m"].get(key)
            v = data.get(lab, {}).get("m", {}).get(key)
            vals.append(100 * (v - base) / abs(base) if v is not None and base else np.nan)
        ax_b.bar(np.arange(len(keys)) + (j - (len(refs) - 1) / 2) * width, vals, width, color=color,
                 alpha=0.85 if lab in labels else 0.45, label=name, edgecolor="k" if lab in labels else "none")
    for i, key in enumerate(keys):
        c = st[key].get(("FV", 128))
        if c and np.isfinite(c["sigma"]) and c["value"]:
            sig = 100 * 3 * c["sigma"] / abs(c["value"])
            ax_b.add_patch(plt.Rectangle((i - 0.45, -sig), 0.9, 2 * sig, color="0.85", zorder=0))
    ax_b.axhline(0, color="k", lw=0.7)
    ax_b.set_xticks(range(len(keys)), [METRICS[k][0] for k in keys], fontsize=8, rotation=12)
    ax_b.set_ylabel("change vs FV-128 PLM (GPU) [%]  (grey: 3 sigma of FV-128)")
    ax_b.set_title("Does PPM at 128 move FV toward FD-128 or toward FV-256?")
    ax_b.legend(fontsize=8, ncol=4)
    ax_b.grid(alpha=0.3, axis="y")
    fig.suptitle("CGOLS A128, FV VL2 + HLLC: PPM (AthenaPK ppm_simple) vs PLM corrector (CPU, native JAX)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = f"{FIG_DIR}/fv_ppm_A128.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(path)


# -------------------------------------------------------------
# =============== ↓ FV-PPM on GPU (256, 512) ↓ ================
# -------------------------------------------------------------

#: the metrics of the PPM comparison; PPM_INTEGRATED = the 7 integrated ones of fv_ppm_A128.png (its bar panel
#: without the small-scale variance), for the "toward / away from FD" count
PPM_KEYS = ("out5_15_40", "out5_40_50", "out5_50_75", "out5_peak", "mout5_cum", "lx1_15_40", "lx1_60_75",
            "lx2_60_75", "ha_60_75", "pinch_60_75", "pinched_frac", "asym_50_75", "ssv256", "cc85_dlogn", "cc85_dv",
            "side_loss", "mass_75")
PPM_INTEGRATED = ("out5_15_40", "out5_peak", "out5_50_75", "lx1_15_40", "lx1_60_75", "cc85_dv", "side_loss")


def ppm_shift(st, data, key, n, label):
    """PPM(N) against PLM(N) and FD(N), FD(2N).

    z = (PPM - PLM) / sigma_diff with sigma_diff = sqrt(2) sigma_FV(N) (PPM is one realisation, given the PLM noise);
    the direction says whether PPM lies closer to FD(N) [and to FD(2N)] than PLM does: 'toward' / 'away' if |z| >= 3,
    with '?' for 2 <= |z| < 3, '=' (neither) if |z| < 2. f = (PPM - PLM) / (FD(N) - PLM) as in the attribution table.
    """
    c_fv, c_fd, c_fd2 = (st[key].get((s, m)) for s, m in (("FV", n), ("FD", n), ("FD", 2 * n)))
    v = data.get(label, {}).get("m", {}).get(key)
    if v is None or c_fv is None or c_fd is None:
        return None
    plm, fd = c_fv["value"], c_fd["value"]
    sig = np.sqrt(2) * c_fv["sigma"] if np.isfinite(c_fv["sigma"]) else np.nan
    z = (v - plm) / sig if sig > 0 else np.inf

    def direction(target):
        if abs(z) < 2:
            return "="
        word = "toward" if abs(v - target) < abs(plm - target) else "away"
        return word + ("?" if abs(z) < 3 else "")

    return {"ppm": v, "plm": plm, "fd": fd, "fd2": c_fd2["value"] if c_fd2 else None, "sigma": sig, "z": z,
            "dir": direction(fd), "dir2": direction(c_fd2["value"]) if c_fd2 else "-",
            "f": (v - plm) / (fd - plm) if fd != plm else np.nan}


def k_of(st, key, n, value, sigma):
    """Equivalent FD resolution k of an arbitrary FV-like value at N (k_factor with that value in place of FV(N))."""
    cell = dict(st[key])
    cell[("FV", n)] = {"value": value, "sigma": sigma}
    return k_factor(cell, n, key=key)


def steps_of(path):
    with open(path) as fh:
        return sum(1 for line in fh if "[diag] t=" in line)


def asym_series(frame_dir):
    """(t [Myr], up-down mirror asymmetry rms dlog n of the xz slice) per frame, as frame_metrics."""
    t, asym = [], []
    for name in sorted(f for f in os.listdir(frame_dir) if f.startswith("frame_") and f.endswith(".npz")):
        with np.load(os.path.join(frame_dir, name)) as f:
            t.append(float(f["time_myr"]))
            log_n = np.log10(f["n_xz"].astype(np.float64))
        asym.append(np.sqrt(np.mean((log_n - log_n[:, ::-1]) ** 2)))
    order = np.argsort(t)
    return np.array(t)[order], np.array(asym)[order]


def ppm_tables(st, runs, data):
    """Markdown tables of the GPU FV-PPM runs: per metric FD / PLM / PPM / FD(2N) with z and direction; the
    equivalent FD resolution of the small-scale variance; mirror asymmetry; cost. Returns a summary dict."""
    rng_state = RNG.bit_generator.state  # k_factor draws from RNG: restored below so the later figures do not change
    summary = {}
    labels = [lab for lab in PPM_GPU_RUNS if lab in data]
    print("\n## FV-PPM (GPU) vs FV-PLM and FD at the same N: z = (PPM - PLM) / (sqrt(2) sigma_FV(N)); direction "
          "relative to FD(N) / FD(2N): toward | away (|z| >= 3; ? for 2 <= |z| < 3) | = (|z| < 2); "
          "f = (PPM - PLM) / (FD - PLM)\n")
    print("| metric | N | FD(N) | PLM ± sigma | PPM | FD(2N) | z | vs FD(N) | vs FD(2N) | f |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key in PPM_KEYS:
        for lab in labels:
            n = runs[lab][1]
            r = ppm_shift(st, data, key, n, lab)
            if r is None:
                continue
            summary.setdefault(key, {})[str(n)] = r
            print(f"| {METRICS[key][0]} | {n} | {fmt(r['fd'])} | {fmt(r['plm'])} ± {fmt(r['sigma'] / np.sqrt(2), 2)} | "
                  f"{fmt(r['ppm'])} | {fmt(r['fd2'])} | {r['z']:+.1f} | {r['dir']} | {r['dir2']} | {r['f']:+.2f} |")
    print("\nIntegrated metrics (" + ", ".join(PPM_INTEGRATED) + "): direction vs FD(N)")
    for lab in labels:
        n = runs[lab][1]
        dirs = [summary.get(k, {}).get(str(n), {}).get("dir", "-") for k in PPM_INTEGRATED]
        print(f"  FV-PPM-{n}: toward {sum(d.startswith('toward') for d in dirs)}, away "
              f"{sum(d.startswith('away') for d in dirs)}, neither {dirs.count('=')} of {len(dirs)}  ({' '.join(dirs)})")

    print("\n## Small-scale variance fraction 80-300 pc (50-75 Myr) and its equivalent FD resolution k (16-84%)\n")
    print("| N | FD(N) | FD(2N) | PLM | PPM | PPM / PLM | k PLM | k PPM |")
    print("|---|---|---|---|---|---|---|---|")
    for lab in labels:
        n = runs[lab][1]
        c = st["ssv256"]
        ppm = data[lab]["m"].get("ssv256")
        if ppm is None or ("FV", n) not in c:
            continue
        kp = k_factor(c, n, key="ssv256")
        kq = k_of(st, "ssv256", n, ppm, c[("FV", n)]["sigma"])
        txt = lambda k: (f"{k[0]:.2f}" + (f" [{k[1][0]:.2f}, {k[1][1]:.2f}]" if k[1] else "")
                         if k[0] is not None and np.isfinite(k[0]) else k[2])
        fd2 = c.get(("FD", 2 * n), {}).get("value")
        print(f"| {n} | {fmt(c[('FD', n)]['value'])} | {fmt(fd2)} | {fmt(c[('FV', n)]['value'])} | {fmt(ppm)} | "
              f"{ppm / c[('FV', n)]['value']:.2f} | {txt(kp)} | {txt(kq)} |")
        summary.setdefault("_ssv_k", {})[str(n)] = {"plm": kp[:2], "ppm": kq[:2]}

    print("\n## Up-down mirror asymmetry rms dlog n (xz slice): 50-75 Myr mean, max over 0-75 Myr, first t > 0.05\n")
    print("| run | 50-75 mean | max | first t [Myr] with asym > 0.05 |")
    print("|---|---|---|---|")
    for n in (256, 512):
        for lab in (f"FD{n}", f"FV{n}", f"FV{n}ppm"):
            if lab not in data:
                continue
            t, a = asym_series(runs[lab][3])
            first = t[np.argmax(a > 0.05)] if (a > 0.05).any() else np.nan
            print(f"| {lab} | {fmt(data[lab]['m']['asym_50_75'])} | {fmt(a.max())} | {fmt(first)} |")
            summary.setdefault("_asym", {})[lab] = [data[lab]["m"]["asym_50_75"], float(a.max()), float(first)]

    print("\n## Cost: solver wall (time_integration) to 75 Myr, steps, s/step, GPU-hours\n")
    print("| run | steps | wall [s] | s/step | GPU-h | source |")
    print("|---|---|---|---|---|---|")
    for lab, (wall, gpus, diag, note) in PPM_COST.items():
        steps = steps_of(diag)
        summary.setdefault("_cost", {})[lab] = {"steps": steps, "wall": wall, "s_step": wall / steps,
                                                "gpu_h": wall * gpus / 3600}
        print(f"| {lab} | {steps} | {wall:.1f} | {wall / steps:.4f} | {wall * gpus / 3600:.3f} | {note} |")
    cost = summary["_cost"]
    for n in (256, 512):
        a, b = cost[f"FV{n}ppm"], cost[f"FV{n}"]
        print(f"  {n}: PPM/PLM per step {a['s_step'] / b['s_step']:.2f}x, steps {a['steps'] / b['steps']:.3f}x, "
              f"GPU-h {a['gpu_h'] / b['gpu_h']:.2f}x; FD/PPM GPU-h {cost[f'FD{n}']['gpu_h'] / a['gpu_h']:.2f}x")
    RNG.bit_generator.state = rng_state
    return summary


def fig_ppm_ladder(runs, data, st, summary):
    """FV-PPM 256 / 512 (GPU) against FV-PLM, FD at the same N and FD at 2N: outflow, X-ray, asymmetry, small-scale
    variance, the per-metric shift and the cost (the 256 / 512 counterpart of fv_ppm_A128.png)."""
    fig = plt.figure(figsize=(21, 13))
    gs = fig.add_gridspec(3, 4)
    styles = lambda n: {f"FD{n}": ("C0", "-", f"FD-{n}"), f"FV{n}": ("C3", "-", f"FV-{n} PLM"),
                        f"FV{n}ppm": ("C1", "-", f"FV-{n} PPM (GPU)"), f"FD{2 * n}": ("C0", ":", f"FD-{2 * n}")}
    ax_as, ax_ss = fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3])
    for row, n in enumerate((256, 512)):
        ax_out, ax_late, ax_x = (fig.add_subplot(gs[row, j]) for j in range(3))
        for lab, (color, ls, name) in styles(n).items():
            if lab not in data:
                continue
            ser = data[lab]["series"]
            t, out5 = np.array(ser["t"]), np.array(ser["out5"])
            lw = 2.0 if lab.endswith("ppm") else 1.2
            ax_out.plot(t, out5, color=color, ls=ls, lw=lw, label=name)
            ax_late.plot(t, out5, color=color, ls=ls, lw=lw)
            if "te" in ser:
                ax_x.semilogy(ser["te"], ser["lx1"], color=color, ls=ls, lw=lw)
            if lab.startswith("FV") or lab == f"FD{n}":  # asymmetry / small scales: same-N runs only
                lw2 = 1.8 if n == 512 else 1.0
                ls2 = "-" if n == 512 else "--"
                t_a, a = asym_series(runs[lab][3])
                ax_as.semilogy(t_a, np.maximum(a, 1e-7), color=color, ls=ls2, lw=lw2, label=name)
                ax_ss.semilogy(t, ser["ssv256"], color=color, ls=ls2, lw=lw2, label=name)
        for lab, s in runs.items():  # FV realisations (seeds, segmentation, reruns) as thin lines
            if s[:2] == ("FV", n) and s[2] == "ens" and lab in data:
                ser = data[lab]["series"]
                ax_late.plot(ser["t"], ser["out5"], color="C3", lw=0.5, alpha=0.5)
        ax_out.set(xlabel="t [Myr]", ylabel="net outflow |z| = 5 kpc [Msun/yr]", xlim=(0, 75), title=f"N = {n}")
        ax_late.set(xlabel="t [Myr]", ylabel="net outflow |z| = 5 kpc [Msun/yr]", xlim=(40, 75), ylim=(0, 8),
                    title=f"late phase, N = {n} (thin: FV-PLM realisations)")
        ax_x.set(xlabel="t [Myr]", ylabel="L_X1 [1e39 erg/s]", xlim=(0, 75), ylim=(1, 3e3), title=f"soft X-ray, N = {n}")
        ax_out.legend(fontsize=8)
        for ax in (ax_out, ax_late, ax_x):
            ax.grid(alpha=0.3)
    ax_as.set(xlabel="t [Myr]", ylabel="up-down asymmetry rms dlog n (xz)", xlim=(0, 75),
              title="mirror asymmetry (dashed 256, solid 512)")
    ax_ss.set(xlabel="t [Myr]", ylabel="small-scale variance fraction 80-300 pc", xlim=(0, 75),
              title="small scales, 1 < |z| < 6 kpc (dashed 256, solid 512)")
    ax_as.legend(fontsize=7.5, ncol=2)
    for ax in (ax_as, ax_ss):
        ax.grid(alpha=0.3, which="both")
    # bar panel: change vs FV-N PLM of FD(N), FD(2N) and PPM(N), grey = 3 sigma_diff
    keys = ("out5_15_40", "out5_40_50", "out5_50_75", "out5_peak", "lx1_15_40", "lx1_60_75", "ha_60_75", "ssv256",
            "side_loss", "mass_75")
    ax_b = fig.add_subplot(gs[2, :3])
    refs = [(256, "FD", "C0", 0.45), (256, "FD2", "C0", 0.2), (256, "PPM", "C1", 0.5),
            (512, "FD", "C0", 0.85), (512, "FD2", "C0", 0.45), (512, "PPM", "C1", 0.95)]
    width = 0.8 / len(refs)
    for j, (n, which, color, alpha) in enumerate(refs):
        vals = []
        for key in keys:
            r = summary.get(key, {}).get(str(n))
            v = None if r is None else {"FD": r["fd"], "FD2": r["fd2"], "PPM": r["ppm"]}[which]
            vals.append(100 * (v - r["plm"]) / abs(r["plm"]) if v is not None and r["plm"] else np.nan)
        name = {"FD": f"FD-{n}", "FD2": f"FD-{2 * n}", "PPM": f"FV-{n} PPM"}[which]
        ax_b.bar(np.arange(len(keys)) + (j - (len(refs) - 1) / 2) * width, vals, width, color=color, alpha=alpha,
                 label=f"{name} vs FV-{n} PLM", edgecolor="k" if which == "PPM" else "none",
                 hatch="//" if which == "FD2" else None)
    for i, key in enumerate(keys):
        for half, n in ((-1, 256), (1, 512)):
            r = summary.get(key, {}).get(str(n))
            if r and np.isfinite(r["sigma"]) and r["plm"]:
                sig = 100 * 3 * r["sigma"] / abs(r["plm"])
                x0 = i - 0.42 if half < 0 else i
                ax_b.add_patch(plt.Rectangle((x0, -sig), 0.42, 2 * sig, color="0.85", zorder=0))
                if key in PPM_INTEGRATED or key == "ssv256":
                    ax_b.text(i + half * 0.2, 1.0, r["dir"], transform=ax_b.get_xaxis_transform(), ha="center",
                              va="top", fontsize=7, color="C1")
    ax_b.axhline(0, color="k", lw=0.7)
    ax_b.set_yscale("symlog", linthresh=10)
    ax_b.set_xticks(range(len(keys)), [METRICS[k][0] for k in keys], fontsize=8, rotation=12)
    ax_b.set_ylabel("change vs FV-N PLM [%] (symlog; grey: 3 sigma_diff,\nleft half 256, right half 512)")
    ax_b.set_title("Does PPM move FV-N toward FD-N / FD-2N?  (orange text: PPM direction vs FD-N, 256 | 512)")
    ax_b.legend(fontsize=7.5, ncol=3, loc="lower left")
    ax_b.grid(alpha=0.3, axis="y")
    # cost panel
    ax_c = fig.add_subplot(gs[2, 3])
    cost = summary["_cost"]
    for j, (lab_fmt, color, name) in enumerate((("FD{}", "C0", "FD"), ("FV{}", "C3", "FV PLM"),
                                               ("FV{}ppm", "C1", "FV PPM"))):
        xs = np.arange(2) + (j - 1) * 0.27
        ys = [cost[lab_fmt.format(n)]["gpu_h"] for n in (256, 512)]
        ax_c.bar(xs, ys, 0.27, color=color, label=name)
        for x, n, y in zip(xs, (256, 512), ys):
            c = cost[lab_fmt.format(n)]
            ax_c.text(x, y * 1.08, f"{c['s_step'] * 1e3:.0f} ms\n{c['steps']}", ha="center", fontsize=7)
    ax_c.set_yscale("log")
    ax_c.set_xticks(range(2), ["256", "512"])
    ax_c.set(ylabel="GPU-hours (H200) to 75 Myr", ylim=(0.02, 30),
             title="cost (labels: ms/step, steps)\nPPM on HoreKa, PLM / FD local")
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("CGOLS A256 / A512, FV VL2 + HLLC: PPM (AthenaPK ppm_simple, GPU) vs PLM corrector, against FD at N "
                 "and 2N")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = f"{FIG_DIR}/fv_ppm_A256_A512.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(path)


def main():
    os.chdir(SCRIPT_DIR)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--ppm", action="append", default=[],
                        help="LABEL=ROLE:FRAMES_DIR:DIAG_LOG[:EMISSION_TAG] extra FV-128 ablation (e.g. the CPU PPM run)")
    parser.add_argument("--ppm-gpu", action="store_true",
                        help="add the GPU FV-PPM 256/512 runs (PPM_GPU_RUNS): tables + fv_ppm_A256_A512.png")
    parser.add_argument("--json", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    extra = {}
    for item in args.ppm:
        label, rest = item.split("=", 1)
        parts = rest.split(":")
        role, frames, diag = parts[:3]
        emis = parts[3] if len(parts) > 3 else None
        extra[label] = ("FV", 128, role, frames, emis, None, diag)
    ppm128 = list(extra)
    if args.ppm_gpu:
        extra.update(PPM_GPU_RUNS)
    runs, data = load_all(args.recompute, extra)
    st = stats(runs, data)
    noise_table(st)
    summary = tables(st, runs, data)
    if args.ppm_gpu:
        summary["_ppm_gpu"] = ppm_tables(st, runs, data)
    if args.json:
        conv = lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o) if isinstance(o, np.floating) else str(o)
        stj = {k: {f"{s}{n}": {kk: vv for kk, vv in c.items()} for (s, n), c in cell.items()} for k, cell in st.items()}
        with open(args.json, "w") as fh:
            json.dump({"stats": stj, "summary": summary, "ablation_values": {k: data[k]["m"] for k in data}},
                      fh, default=conv, indent=1)
    if not args.no_figures:
        fig_significance(st)
        # the GPU PPM runs only feed fv_ppm_A256_A512.png and the tables: the standard figures stay as they are
        fig_convergence(st, {k: v for k, v in runs.items() if k not in PPM_GPU_RUNS}, data)
        fig_cost(st)
        if ppm128:
            fig_ppm(runs, data, st, ppm128)
        if args.ppm_gpu:
            fig_ppm_ladder(runs, data, st, summary["_ppm_gpu"])


if __name__ == "__main__":
    main()
