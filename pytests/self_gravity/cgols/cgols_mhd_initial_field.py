"""Render the CGOLS MHD initial magnetic field (CGOLS_MHD=1).

Builds the IC on CPU with cgols.build_initial_conditions() and plots: face-on
midplane |B| with in-plane field lines, edge-on x-z |B| with beta contours, the
midplane B(R) profile against M82 estimates, a vertical profile and a 3D
field-line render (the default field is purely toroidal, so lines are circles).

    JAX_PLATFORMS=cpu CGOLS_MHD=1 CGOLS_DIM=256 CGOLS_IC_MODE=insitu \
        CGOLS_RUN_TAG=_Bfieldic python cgols_mhd_initial_field.py out.png
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import cgols

OUT = sys.argv[1]
state, config, params, reg = cgols.build_initial_conditions()
state = np.asarray(state, dtype=np.float64)
to_uG = float(cgols._code_field_to_microgauss())
mi = reg.magnetic_index
Bx, By, Bz = (state[mi.x] * to_uG, state[mi.y] * to_uG, state[mi.z] * to_uG)
B = np.sqrt(Bx**2 + By**2 + Bz**2)
rho = state[reg.density_index]
P = state[reg.pressure_index]
P_B = 0.5 * (state[mi.x] ** 2 + state[mi.y] ** 2 + state[mi.z] ** 2)
beta = np.where(P_B > 0, P / np.maximum(P_B, 1e-300), np.inf)

Lx, Ly, Lz = config.box_size.x, config.box_size.y, config.box_size.z
nx, ny, nz = B.shape
x = (np.arange(nx) + 0.5) * Lx / nx - Lx / 2
y = (np.arange(ny) + 0.5) * Ly / ny - Ly / 2
z = (np.arange(nz) + 0.5) * Lz / nz - Lz / 2
kz, jy = nz // 2, ny // 2
cmap = plt.get_cmap("inferno").with_extremes(bad="black")
bnorm = LogNorm(1, 200)

fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.25])

# (a) face-on midplane |B| + in-plane field lines
ax = fig.add_subplot(gs[0, 0])
Bm = np.where(B[:, :, kz] > 0, B[:, :, kz], np.nan)
im = ax.imshow(Bm.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap, norm=bnorm)
ax.streamplot(x, y, Bx[:, :, kz].T, By[:, :, kz].T, color="white", linewidth=0.6, density=1.3, arrowsize=0.7)
ax.set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), xlabel="x [kpc]", ylabel="y [kpc]", title="face-on, midplane z = 0: |B| and field lines")
fig.colorbar(im, ax=ax, label=r"|B| [$\mu$G]", shrink=0.85)

# (b) edge-on x-z |B| with beta contours (zoom on the disk)
ax = fig.add_subplot(gs[0, 1])
Bs = np.where(B[:, jy, :] > 0, B[:, jy, :], np.nan)
im = ax.imshow(Bs.T, origin="lower", extent=[x[0], x[-1], z[0], z[-1]], cmap=cmap, norm=bnorm, aspect="auto")
cs = ax.contour(x, z, np.log10(np.clip(beta[:, jy, :], 1e-3, 1e6)).T, levels=[0.0, 1.0, 2.0], colors=["cyan", "deepskyblue", "steelblue"], linewidths=0.8)
ax.clabel(cs, fmt={0.0: r"$\beta$=1", 1.0: r"$\beta$=10", 2.0: r"$\beta$=100"}, fontsize=7)
ax.set(xlim=(-4.5, 4.5), ylim=(-1.0, 1.0), xlabel="x [kpc]", ylabel="z [kpc]", title="edge-on slice y = 0 (|z| < 1 kpc): |B|, β contours")
fig.colorbar(im, ax=ax, label=r"|B| [$\mu$G]", shrink=0.85)

# (c) 3D field-line render: toroidal circles coloured by |B|
ax = fig.add_subplot(gs[:, 2], projection="3d")
interp = lambda R, zz: B[np.clip(np.searchsorted(x, R), 0, nx - 1), jy, np.clip(np.searchsorted(z, zz), 0, nz - 1)]
phi = np.linspace(0, 2 * np.pi, 200)
for zz in (-0.15, -0.075, 0.0, 0.075, 0.15):
    for R in np.arange(0.25, 3.75, 0.25):
        b = interp(R, zz)
        if b < 1:
            continue
        ax.plot(R * np.cos(phi), R * np.sin(phi), np.full_like(phi, zz), color=cmap(bnorm(b)), lw=1.2, alpha=0.9)
ax.set(xlim=(-4, 4), ylim=(-4, 4), zlim=(-1, 1), xlabel="x [kpc]", ylabel="y [kpc]", zlabel="z [kpc]")
ax.set_box_aspect((1, 1, 0.35)); ax.view_init(elev=22, azim=-60)
ax.set_title("field lines (purely toroidal), coloured by |B|")
sm = plt.cm.ScalarMappable(norm=bnorm, cmap=cmap); fig.colorbar(sm, ax=ax, label=r"|B| [$\mu$G]", shrink=0.5, pad=0.08)

# (d) midplane B(R) vs M82 estimates
ax = fig.add_subplot(gs[1, 0])
R = x[x > 0]; prof = B[x > 0, jy, kz]
ax.semilogy(R, prof, "k-", lw=2, label=r"IC midplane |B|, $\beta$ = 1")
ax.axhline(98, color="C3", ls="--", label=r"M82 core equipartition 98 $\mu$G (Adebahr+13)")
ax.axhline(75, color="C1", ls=":", label=r"$B_{eq}$ for CGOLS $\Sigma$ ~75 $\mu$G (Thompson+06)")
ax.axhspan(17, 25, color="C0", alpha=0.2, label=r"M82 halo 17–25 $\mu$G (radio)")
ax.set(xlim=(0, 4.5), ylim=(1, 300), xlabel="R [kpc]", ylabel=r"|B| [$\mu$G]", title="midplane radial profile")
ax.legend(fontsize=7)

# (e) vertical profile at R = 1 kpc
ax = fig.add_subplot(gs[1, 1])
i1 = np.argmin(abs(x - 1.0))
ax.semilogy(z, B[i1, jy, :], "k-", lw=2, label="|B| at R = 1 kpc")
ax2 = ax.twinx(); ax2.semilogy(z, rho[i1, jy, :], "C2-", lw=1, label="gas density (code)")
ax.set(xlim=(-1, 1), ylim=(0.1, 300), xlabel="z [kpc]", ylabel=r"|B| [$\mu$G]", title="vertical profile at R = 1 kpc (field tied to disk gas)")
ax2.set_ylabel("ρ [code]", color="C2")
fig.suptitle(f"CGOLS MHD initial magnetic field: toroidal, midplane β = {cgols.B_BETA:g}, {nx}²×{nz} (halo field-free)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT, "max|B| %.1f uG" % np.nanmax(B))
