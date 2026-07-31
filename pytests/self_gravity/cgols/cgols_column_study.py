"""1D central-column study of the CGOLS initial conditions, with cooling.

Regenerates the ``CGOLS_SMOOTHING_SIGMA`` table quoted in ``cgols.py`` (n_c, T_c,
disk-halo transition) and extends it with the radiative-cooling diagnostics that
decide whether sigma = 0 is affordable and stable once cooling is on.

The table in ``cgols.py`` was produced ad hoc and never committed; this script is
the reproducible version of it.

WHY A SLAB AND NOT THE FULL 3D IC
---------------------------------
``build_initial_conditions()`` materialises 512^2x1024 fields. We only need the
central column, and ``_gaussian_smooth_3d`` is separable with kernel radius
``int(4*sigma + 0.5)`` (6 cells at sigma = 1.5), so a small (2W+1)^2 x Nz slab
centred on the axis reproduces the centre column EXACTLY as long as W exceeds
that radius. Everything else - the analytic profiles, the smoothing, the HSE
pressure rebuild - is imported from ``cgols`` rather than reimplemented, so the
numbers are directly comparable to the production IC.

CPU-ONLY. Importing ``cgols`` normally runs ``autocvd``, which blocks waiting for
a free GPU; we stub it out and pin JAX to CPU so this never contends with a run
in flight.

    python cgols_column_study.py                 # sigma = 0, 1.5, 3
    python cgols_column_study.py --sigma 0 1.5   # explicit list
"""

import argparse
import os
import sys
import types

# ---- CPU-only shim: must happen BEFORE importing cgols ----------------------
# cgols.py calls autocvd(num_gpus=...) at import time, which waits (indefinitely)
# for fully free GPUs. This study is pure CPU, so stub the module out and pin the
# JAX backend to CPU.
_stub = types.ModuleType("autocvd")
_stub.autocvd = lambda *a, **k: None
sys.modules.setdefault("autocvd", _stub)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CGOLS_SHARD_SPLIT", "(1, 1, 1, 1)")

import astropy.units as u  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cgols import (  # noqa: E402
    M_gas,
    Phi_disk_function,
    Phi_halo_function,
    R_gas,
    R_trunc,
    T_disk,
    T_factor,
    T_halo,
    _fig,
    _gaussian_smooth_3d,
    build_config,
    code_length,
    code_mass,
    code_units,
    cooling_lambda_cgs,
    gamma,
    k_B,
    m_p,
    mu,
    rho_0h,
    vertical_hse_pressure,
)

PC_PER_CODE = code_length.to(u.pc).value
CODE_DENSITY_CGS = (code_mass / code_length**3).to(u.g / u.cm**3).value
M_P_CGS = m_p.to(u.g).value
K_B_CGS = k_B.to(u.erg / u.K).value
SEC_PER_YR = (1 * u.yr).to(u.s).value


def build_central_column(sigma_cells, window=None):
    """Build the on-axis column of the CGOLS IC for a given smoothing sigma.

    Replicates ``build_initial_conditions`` (cgols.py:831-940) on a
    (2*window+1)^2 x dim_z slab centred on the axis. Returns the centre column.
    """
    config, _ = build_config()
    L_x, L_y, L_z = config.box_size.x, config.box_size.y, config.box_size.z
    dim_x, dim_y, dim_z = config.num_cells.x, config.num_cells.y, config.num_cells.z

    dx, dy, dz = L_x / dim_x, L_y / dim_y, L_z / dim_z
    mid_x, mid_y, mid_z = dim_x // 2, dim_y // 2, dim_z // 2

    # Kernel radius used by _gaussian_smooth_3d; the slab must be wider than it
    # or edge-replicate padding leaks into the centre column.
    radius = int(4 * sigma_cells + 0.5)
    if window is None:
        window = radius + 4
    if sigma_cells > 0 and window <= radius:
        raise ValueError(
            f"window {window} must exceed the smoothing kernel radius {radius} "
            f"for sigma = {sigma_cells}, else the centre column is contaminated "
            "by edge-replicate padding"
        )

    # Cell centres, origin at box centre - matches helper_data.geometric_centers
    # ((i + 0.5) * d) shifted by -L/2, as used at cgols.py:814-819.
    ix = jnp.arange(mid_x - window, mid_x + window + 1)
    iy = jnp.arange(mid_y - window, mid_y + window + 1)
    iz = jnp.arange(dim_z)
    X_c = ((ix + 0.5) * dx - L_x / 2)[:, None, None]
    Y_c = ((iy + 0.5) * dy - L_y / 2)[None, :, None]
    Z_c = ((iz + 0.5) * dz - L_z / 2)[None, None, :]
    X_c, Y_c, Z_c = jnp.broadcast_arrays(X_c, Y_c, Z_c)

    R_cyl = jnp.maximum(jnp.sqrt(X_c**2 + Y_c**2), 0.25 * dx)
    r_sph = jnp.maximum(jnp.sqrt(X_c**2 + Y_c**2 + Z_c**2), 0.25 * dx)

    Phi_total = Phi_disk_function(R_cyl, Z_c) + Phi_halo_function(r_sph)

    k_B_code = k_B.to(code_units.code_energy / u.K).value
    m_p_code = m_p.to(code_units.code_mass).value
    c_s_d = jnp.sqrt(k_B_code * T_disk.to(u.K).value / (mu * m_p_code))
    c_s_h = jnp.sqrt(k_B_code * T_halo.to(u.K).value / (mu * m_p_code))
    rho_0h_code = rho_0h.to(code_units.code_density).value
    K_const = c_s_h**2 * rho_0h_code ** (1 - gamma) / gamma

    # ---- Disk: surface density then isothermal vertical structure (Eq. 4) ----
    M_gas_code = M_gas.to(code_units.code_mass).value
    R_gas_code = R_gas.to(code_units.code_length).value
    sigma_0 = M_gas_code / (2 * jnp.pi * R_gas_code**2)
    R_trunc_code = R_trunc.to(code_units.code_length).value
    delta = (0.15 * u.kiloparsec).to(code_units.code_length).value
    cutoff = jnp.where(
        R_cyl < R_trunc_code, 1.0, jnp.exp(-((R_cyl - R_trunc_code) / delta) ** 2)
    )
    Sigma = sigma_0 * jnp.exp(-R_cyl / R_gas_code) * cutoff

    Phi_0d = Phi_total[:, :, mid_z]
    exp_factor = jnp.exp(-(Phi_total - Phi_0d[:, :, None]) / c_s_d**2)
    integral = jnp.sum(exp_factor, axis=2) * dz
    rho_disk = (Sigma[:, :, mid_z] / integral)[:, :, None] * exp_factor

    # ---- Halo: adiabatic hydrostatic atmosphere ----
    Phi_sph = Phi_disk_function(0, r_sph) + Phi_halo_function(r_sph)
    r_ref_code = (100 * u.kiloparsec).to(code_units.code_length).value
    Phi_0h = Phi_disk_function(0, r_ref_code) + Phi_halo_function(r_ref_code)
    bracket = 1 - (gamma - 1) * (Phi_sph - Phi_0h) / c_s_h**2
    rho_halo = rho_0h_code * jnp.maximum(bracket, 0.0) ** (1 / (gamma - 1))

    # ---- Branch on smoothing, exactly as cgols.py:908-938 ----
    if sigma_cells > 0:
        P_top_2d = K_const * rho_halo[:, :, -1:] ** gamma
        rho_total = _gaussian_smooth_3d(rho_disk + rho_halo, sigma_cells)
        P_total = vertical_hse_pressure(rho_total, Phi_total, dz, P_top_2d)
    else:
        rho_total = rho_disk + rho_halo
        P_total = rho_disk * c_s_d**2 + K_const * rho_halo**gamma

    return {
        "z_code": Z_c[window, window, :],
        "rho": rho_total[window, window, :],
        "P": P_total[window, window, :],
        "dz": dz,
        "mid_z": mid_z,
        "sigma": sigma_cells,
    }


def analyse_column(col):
    """Derive the table entries plus the cooling diagnostics for one column."""
    rho, P, dz, mid_z = col["rho"], col["P"], col["dz"], col["mid_z"]
    z_pc = col["z_code"] * PC_PER_CODE

    n = rho * CODE_DENSITY_CGS / (mu * M_P_CGS)  # cm^-3
    T = P / rho * T_factor  # K

    n_c = float(n[mid_z])
    T_c = float(T[mid_z])

    # Disk-halo transition: half-width at half maximum of the density column,
    # i.e. the |z| at which n falls to n_c / 2 (linear interp on the upper half).
    #
    # NB this does NOT reproduce the "transition" column of the sigma table in
    # cgols.py (88 / 147 / 225 pc); this HWHM gives 24 / 43 / 74 pc. The original
    # table was produced by a throwaway script that was never committed, so its
    # definition is unrecoverable - and none of the obvious candidates reproduce
    # it either (n/10: 45/78/135; n/100: 65/108/190; T=1e5 K: 76/133/228;
    # T=1e6 K: 96/162/271; disk=halo density crossing at sigma=0: 106). The ratio
    # to the old numbers is not even constant (3.7/3.4/3.0), so it is a different
    # measure, not a units slip. n_c, T_c and t_cool all reproduce exactly, so
    # treat this column as newly defined rather than as a regression check.
    upper_n = n[mid_z:]
    upper_z = jnp.abs(z_pc[mid_z:])
    idx = int(jnp.argmax(upper_n < n_c / 2))
    if idx == 0:
        transition_pc = float("nan")
    else:
        n0, n1 = float(upper_n[idx - 1]), float(upper_n[idx])
        z0, z1 = float(upper_z[idx - 1]), float(upper_z[idx])
        transition_pc = z0 + (n_c / 2 - n0) * (z1 - z0) / (n1 - n0)

    # ---- Cooling diagnostics ----
    # t_cool = (3/2) n_tot k T / (n^2 Lambda), with the n^2 convention cgols uses.
    lam = cooling_lambda_cgs(T)  # erg cm^3 s^-1, zero strictly below 1e4 K
    e_thermal = 1.5 * n * K_B_CGS * T  # erg cm^-3 (n_tot k T = P)
    rate = n**2 * lam  # erg cm^-3 s^-1
    t_cool_yr = jnp.where(rate > 0, e_thermal / jnp.maximum(rate, 1e-300), jnp.inf)
    t_cool_yr = t_cool_yr / SEC_PER_YR

    return {
        "sigma": col["sigma"],
        "z_pc": z_pc,
        "n": n,
        "T": T,
        "t_cool_yr": t_cool_yr,
        "n_c": n_c,
        "T_c": T_c,
        "transition_pc": transition_pc,
        "t_cool_c_yr": float(t_cool_yr[mid_z]),
        "t_cool_min_yr": float(jnp.min(t_cool_yr)),
        "dz_pc": dz * PC_PER_CODE,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.0, 1.5, 3.0])
    ap.add_argument(
        "--window",
        type=int,
        default=None,
        help="slab half-width in cells (default: kernel radius + 4)",
    )
    args = ap.parse_args()

    results = []
    for s in args.sigma:
        col = build_central_column(s, window=args.window)
        results.append(analyse_column(col))

    r0 = results[0]
    print(f"\n1D central column, dz = {r0['dz_pc']:.1f} pc")
    print("Paper Fig. 4 targets: n_c ~ 200 cm^-3, T_disk = 1e4 K, transition ~80 pc\n")
    print(
        f"{'sigma':>6}  {'n_c [cm^-3]':>12}  {'T_c [K]':>10}  "
        f"{'HWHM [pc]':>11}  {'t_cool(c) [yr]':>15}  {'min t_cool [yr]':>16}"
    )
    for r in results:
        print(
            f"{r['sigma']:>6.1f}  {r['n_c']:>12.1f}  {r['T_c']:>10.3g}  "
            f"{r['transition_pc']:>11.1f}  {r['t_cool_c_yr']:>15.1f}  "
            f"{r['t_cool_min_yr']:>16.1f}"
        )

    fig, axes = plt.subplots(3, 1, figsize=(7, 12), sharex=True)
    for r in results:
        lbl = f"sigma = {r['sigma']}"
        axes[0].plot(r["z_pc"], r["n"], label=lbl)
        axes[1].plot(r["z_pc"], r["T"], label=lbl)
        axes[2].plot(r["z_pc"], r["t_cool_yr"], label=lbl)
    axes[0].set_ylabel(r"n [cm$^{-3}$]")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("T [K]")
    axes[1].set_yscale("log")
    axes[2].set_ylabel(r"$t_{\rm cool}$ [yr]")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("z [pc]")
    for ax in axes:
        ax.set_xlim(-600, 600)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_title("CGOLS central column vs IC smoothing")
    plt.tight_layout()
    out = _fig("cgols_column_study.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
