"""Diagnostic analysis of the cgols.py blow-up at ~48 Myr.

Reads the surviving host-offloaded snapshot frames (cgols_snapshots/frame_*.npz)
and builds a figure characterising the crash. Pure numpy/matplotlib, no GPU.
"""
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import astropy.units as u
from astropy import constants as const

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _here(name):
    return os.path.join(SCRIPT_DIR, name)


# ---- Feedback schedule (from cgols.py build_cgols_wind_params) ----
code_length = 1 * u.kpc
code_mass = 1e6 * u.M_sun
code_velocity = 100 * u.km / u.s
code_time = code_length / code_velocity
code_energy = code_mass * code_velocity ** 2
mu, gamma = 0.6, 5 / 3
mp, kB = const.m_p, const.k_B
T_factor = (mu * mp / kB * code_velocity ** 2).to(u.K).value

mdot_to_code = lambda x: (x * u.M_sun / u.yr).to(code_mass / code_time).value
edot_to_code = lambda x: (x * u.erg / u.s).to(code_energy / code_time).value
mdot_low, mdot_high = mdot_to_code(1.5), mdot_to_code(12.0)
edot_low, edot_high = edot_to_code(1.5e42), edot_to_code(5.4e42)
knot_t = np.array([0.0, 5.0, 6.0, 10.0, 40.0, 45.0])
knot_md = np.array([0.0, 0.0, mdot_low, mdot_high, mdot_high, mdot_low])
knot_ed = np.array([0.0, 0.0, edot_low, edot_high, edot_high, edot_low])

tt = np.linspace(0, 50, 1000)
md_t = np.interp(tt, knot_t, knot_md)
ed_t = np.interp(tt, knot_t, knot_ed)
# injection-region temperature T = (gamma-1) * (Edot/Mdot) * T_factor
with np.errstate(divide="ignore", invalid="ignore"):
    Tinj_t = (gamma - 1) * np.where(md_t > 0, ed_t / np.maximum(md_t, 1e-30), 0.0) * T_factor

# ---- Load snapshots ----
files = sorted(glob.glob(_here("cgols_snapshots/frame_*.npz")))
frames = [np.load(f) for f in files]
order = np.argsort([float(fr["time_myr"]) for fr in frames])
frames = [frames[i] for i in order]
t = np.array([float(fr["time_myr"]) for fr in frames])
Tmax = np.array([np.nanmax(fr["T_xz"]) for fr in frames])
nmax = np.array([np.nanmax(fr["n_xz"]) for fr in frames])
mdmax = np.array([np.nanmax(np.abs(fr["mdot_z"])) for fr in frames])
mdtop = np.array([fr["mdot_z"][-1] for fr in frames])

nz = frames[0]["T_xz"].shape[1]
zc = (np.arange(nz) + 0.5) * (20.0 / nz) - 10.0
nx = frames[0]["T_xz"].shape[0]
xc = (np.arange(nx) + 0.5) * (10.0 / nx) - 5.0

# ---- Figure ----
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1])

# (A) feedback schedule + injection temperature
axA = fig.add_subplot(gs[0, 0])
axA.plot(tt, ed_t, "C0-", label=r"$\dot E$ [code]")
axA.plot(tt, md_t * 30, "C1-", label=r"$\dot M \times 30$ [code]")
axA.axvspan(40, 45, color="grey", alpha=0.2)
axA.axvline(47.5, color="k", ls=":", lw=1)
axA.text(47.6, axA.get_ylim()[1] * 0.5, "last clean\nframe", fontsize=8)
axA.set_xlabel("t [Myr]"); axA.set_ylabel("injection rate (code)")
axA.set_title("Feedback schedule")
axA.legend(fontsize=8)

axA2 = fig.add_subplot(gs[0, 1])
axA2.plot(tt, Tinj_t, "C3-")
axA2.axvspan(40, 45, color="grey", alpha=0.2, label="high→low ramp")
axA2.axvline(47.5, color="k", ls=":", lw=1)
axA2.set_xlabel("t [Myr]"); axA2.set_ylabel("analytic inj. T [K]")
axA2.set_title(r"Wind specific energy $\dot E/\dot M$  ($T\propto$)")
axA2.legend(fontsize=8)

# (B) observed Tmax vs time
axB = fig.add_subplot(gs[0, 2])
axB.plot(t, Tmax, "C3o-", ms=4)
axB.axvspan(40, 45, color="grey", alpha=0.2)
axB.set_xlabel("t [Myr]"); axB.set_ylabel(r"max $T$ in x-z slice [K]")
axB.set_title("Observed peak T (central bubble)")
axB.annotate("doubles AFTER\nfeedback drops", xy=(47, 4.4e7), xytext=(30, 3.5e7),
             fontsize=8, arrowprops=dict(arrowstyle="->"))

# (C) nmax + mdot
axC = fig.add_subplot(gs[0, 3])
axC.plot(t, nmax, "C0o-", ms=4, label=r"max $n$ (disk)")
axC.set_xlabel("t [Myr]"); axC.set_ylabel(r"max $n$ [cm$^{-3}$]", color="C0")
axC.tick_params(axis="y", labelcolor="C0")
axC2 = axC.twinx()
axC2.plot(t, mdmax, "C2s-", ms=3, label=r"max $|\dot M(z)|$")
axC2.plot(t, np.abs(mdtop), "C4^-", ms=3, label=r"$|\dot M|$ at z-boundary")
axC2.set_ylabel(r"$\dot M$ [M$_\odot$/yr]", color="C2")
axC2.tick_params(axis="y", labelcolor="C2")
axC.set_title("Disk density & vertical mass flux")
lines = axC.get_lines() + axC2.get_lines()
axC.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="center right")

# (D-G) edge-on temperature for 4 late frames
for k, idx in enumerate([30, 34, 36, 38]):
    ax = fig.add_subplot(gs[1, k])
    T = frames[idx]["T_xz"].T  # (z, x)
    im = ax.imshow(T, origin="lower", extent=[xc[0], xc[-1], zc[0], zc[-1]],
                   aspect="auto", cmap="inferno",
                   norm=LogNorm(vmin=1e4, vmax=8e7))
    ax.set_title(f"T  t={t[idx]:.1f} Myr\n(max={Tmax[idx]:.1e} K)", fontsize=9)
    ax.set_xlabel("x [kpc]")
    if k == 0:
        ax.set_ylabel("z [kpc]")
    plt.colorbar(im, ax=ax, fraction=0.046, label="T [K]" if k == 3 else "")

fig.suptitle(
    "cgols.py blow-up diagnosis: hotter LOW-state wind (Edot/Mdot up 2.2x after t=40-45 Myr "
    "ramp-down) over-heats the central bubble -> crash at ~48 Myr (last clean frame t=47.5)",
    fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = _here("cgols_crash_analysis.png")
plt.savefig(out, dpi=140)
print("wrote", out)
