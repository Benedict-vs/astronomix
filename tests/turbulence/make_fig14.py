"""
Exact reproduction of the WENO panels of HOW-MHD (Seo & Ryu 2023) Fig. 14:
2D slices of the magnetic energy log E_B for

  WENO ISM  (M_turb ~ 10,  beta_p = 0.1)   -- top
  WENO ICM  (M_turb ~ 0.5, beta_p = 10^6)  -- bottom

from 256^3 isothermal-MHD runs. The simulations are run in the cheap v_rms~1
normalisation; E_B is converted to the paper's a=1 normalisation by the exact
similarity scaling E_B^paper = M_turb^2 * E_B (velocity/field unit ratio
lambda = M_turb), so the colorbars match the paper (ISM 0..3, ICM -6..0).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

DATA = "data_fig14"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
})


def load(tag):
    f = os.path.join(DATA, f"paper_{tag}.npz")
    return dict(np.load(f, allow_pickle=True)) if os.path.exists(f) else None


def log_EB_paper(d):
    """log10 E_B in the paper's a=1 normalisation (E_B^paper = M_turb^2 E_B)."""
    EB = np.asarray(d["EB_slice"], dtype=np.float64)
    lam2 = float(d["mturb_aim"]) ** 2
    EB = np.where(EB > 0, EB, np.nan) * lam2
    return np.log10(EB)


ism = load(os.environ.get("ISM_TAG", "ISM_N256"))
icm = load(os.environ.get("ICM_TAG", "ICM_N256"))

fig, axes = plt.subplots(2, 1, figsize=(5.2, 9.6))
panels = [
    (axes[0], "WENO ISM", ism, 0.0, 3.0),
    (axes[1], "WENO ICM", icm, -6.0, 0.0),
]
extent = [-0.5, 0.5, -0.5, 0.5]
for ax, title, d, vmin, vmax in panels:
    if d is None:
        ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=16)
        continue
    img = log_EB_paper(d)
    im = ax.imshow(img.T, origin="lower", extent=extent, cmap="Blues",
                   vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=17)
    ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\log\,E_B$", fontsize=15)
    cb.ax.tick_params(labelsize=11)

fig.tight_layout()
out = os.path.join(FIG, "fig14_WENO_reproduction")
fig.savefig(out + ".png", dpi=300)
fig.savefig(out + ".pdf")
print("wrote", out + ".png/.pdf")

# also report the measured M_turb / E_B range for the record
for nm, d in [("ISM", ism), ("ICM", icm)]:
    if d is None:
        continue
    img = log_EB_paper(d)
    print(f"{nm}: N={int(d['N'])}, M_turb(meas, last)={float(d['Ms_t'][-1]):.2f}, "
          f"log E_B range [{np.nanmin(img):.2f}, {np.nanmax(img):.2f}]")
