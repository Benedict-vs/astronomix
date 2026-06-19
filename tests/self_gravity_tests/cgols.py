"""
CGOLS replication.

Sets up the initial conditions for a Cholla-Galactic-OutfLow-Simulations-style
disk + hot-halo galaxy (Schneider & Robertson 2018, arXiv:1803.01008) and runs
a short adiabatic hydro integration with astronomix.

This reproduces the *adiabatic* A-series setup of that paper, which by design
uses NO radiative cooling. See the cooling section at the end of the file for
the paper's cooling curve and notes on the radiative B/C series.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# At 512x512x1024 the compiled solver peaks at ~28.5 GB/device, which fits on a
# 40 GB A100 but exceeds JAX's default 0.75 preallocation (~30 GB) once BFC
# fragmentation is accounted for. Raise the fraction so a ~20 GB intermediate
# buffer can be placed. setdefault lets a command-line override still win.
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
import numpy as np

import astropy.constants as const
from astropy import units as u

import matplotlib.pyplot as plt

from astronomix import (
    CARTESIAN,
    CodeUnits,
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    OPEN_BOUNDARY,
    PALLAS,
    PERIODIC_BOUNDARY,
    RK4_LSRK,
    SIMPLE_SOURCE_TERM,
    BoundarySettings,
    BoundarySettings1D,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
)
from astronomix._modules._cgols_wind.cgols_wind_options import (
    CGOLSWindConfig,
    CGOLSWindParams,
)

jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# Tunable knobs
# ---------------------------------------------------------------------------
# Optional HSE-preserving smoothing of the initial conditions, in units of cells.
# 0.0 disables it (the run then matches the reference setup exactly). A value of
# ~1-2 smooths the sharp disk-halo contact so the grid can hold it: the density
# is Gaussian-smoothed and the pressure is then REBUILT from vertical hydrostatic
# equilibrium (so dP/dz = -rho dPhi/dz is satisfied and the disk does not "ring").
# This is a mitigation for the under-resolved vertical disk, not a substitute for
# resolving it.
SMOOTHING_SIGMA_CELLS = 3.0

# Total integration time. The paper runs 75 Myr total; bump this up for a real
# stability test (1 Myr is far shorter than an orbital or vertical-crossing time
# and will look static even if the configuration is not).
TOTAL_TIME = 75 * u.Myr


# ---------------------------------------------------------------------------
# Code units
# ---------------------------------------------------------------------------
# code_time is derived automatically by CodeUnits as code_length / code_velocity.
code_length = 1 * u.kiloparsec
code_mass = 1e6 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G = const.G
k_B = const.k_B
m_p = const.m_p

# --- Stellar disk + disk gas ---
M_disk = 1e10 * u.M_sun
R_disk = 0.8 * u.kiloparsec
z_disk = 0.15 * u.kiloparsec
R_trunc = 4.5 * u.kiloparsec

R_gas = 1.6 * u.kiloparsec  # = 2 * R_disk
M_gas = 2.5e9 * u.M_sun
T_disk = 1e4 * u.Kelvin

# --- Dark-matter halo + halo gas ---
M_halo = 5e10 * u.M_sun
R_vir = 53 * u.kiloparsec
c = 10  # NFW concentration
R_halo = R_vir / c

rho_0h = 3e3 * u.M_sun / u.kiloparsec**3
T_halo = 1e6 * u.Kelvin  # at r = 100 kpc

mu = 0.6
gamma = 5 / 3

# Code-units-to-Kelvin factor for temperature: T = (P / rho) * T_factor, where
# P / rho is in code_velocity^2. Computing it as a single well-scaled constant
# (~7e5 K) avoids single precision: the individual code-unit constants k_B_code
# (~7e-70) and m_p_code (~8e-64) both underflow to 0.0 in float32, which turns
# T = P * mu * m_p_code / (rho * k_B_code) into 0/0 = NaN. Their ratio does not.
T_factor = (mu * m_p / k_B * code_units.code_velocity**2).to(u.K).value


# ---------------------------------------------------------------------------
# CGOLS central starburst wind (Chevalier & Clegg 1985 feedback)
# ---------------------------------------------------------------------------
# Schneider & Robertson 2018, Section 2: mass and thermal energy are injected
# at uniform volumetric rates into a 300 pc sphere at the galaxy center, with
#   Mdot = beta * SFR              and   Edot = alpha * 3e41 erg/s * SFR.
# Low state (SFR = 5 Msun/yr, beta = 0.3, alpha = 1.0):
#   Mdot = 1.5 Msun/yr, Edot = 1.5e42 erg/s.
# High state (SFR = 20 Msun/yr, beta = 0.6, alpha = 0.9):
#   Mdot = 12 Msun/yr,  Edot = 5.4e42 erg/s.
# Schedule: no feedback for the first 5 Myr (equilibration), then the low
# state, a 5 Myr linear ramp up, 30 Myr in the high state, a 5 Myr ramp back
# down, and the low state for the remaining 30 Myr (75 Myr total).
def build_cgols_wind_params():
    mdot_to_code = lambda x: (x * u.M_sun / u.yr).to(
        code_units.code_mass / code_units.code_time
    ).value
    edot_to_code = lambda x: (x * u.erg / u.s).to(
        code_units.code_energy / code_units.code_time
    ).value
    myr_to_code = (1 * u.Myr).to(code_units.code_time).value

    mdot_low, mdot_high = mdot_to_code(1.5), mdot_to_code(12.0)
    edot_low, edot_high = edot_to_code(1.5e42), edot_to_code(5.4e42)

    # Feedback turns on with a smooth 1 Myr ramp (5 -> 6 Myr) rather than a
    # near-instantaneous step: the abrupt switch-on is a strong transient for
    # the explicit scheme. All other transitions are the paper's 5 Myr linear
    # ramps via interpolation.
    knot_times_myr = jnp.array([0.0, 5.0, 6.0, 10.0, 40.0, 45.0])
    return CGOLSWindParams(
        # 500 pc rather than the paper's 300 pc: spreading the same Mdot/Edot
        # over a larger gain region lowers the injected energy density (and thus
        # the peak temperature / sound speed), which relaxes the stiffness of
        # the source and the CFL hit once feedback is on.
        injection_radius=(500 * u.pc).to(code_units.code_length).value,
        schedule_times=knot_times_myr * myr_to_code,
        schedule_mass_rates=jnp.array(
            [0.0, 0.0, mdot_low, mdot_high, mdot_high, mdot_low]
        ),
        schedule_energy_rates=jnp.array(
            [0.0, 0.0, edot_low, edot_high, edot_high, edot_low]
        ),
    )


# ---------------------------------------------------------------------------
# Gravitational potentials
# ---------------------------------------------------------------------------
M_disk_code = M_disk.to(code_units.code_mass).value
R_disk_code = R_disk.to(code_units.code_length).value
z_disk_code = z_disk.to(code_units.code_length).value
G_code = G.to(
    code_units.code_length**3 / (code_units.code_mass * code_units.code_time**2)
).value

M_halo_code = M_halo.to(code_units.code_mass).value
R_halo_code = R_halo.to(code_units.code_length).value


def Phi_disk_function(r, z):
    """Miyamoto-Nagai disk gravitational potential.

    Parameters
    ----------
    r : Cylindrical radius in code units.
    z : Vertical coordinate in code units.
    """
    return -(G_code * M_disk_code) / jnp.sqrt(
        r**2 + (R_disk_code + jnp.sqrt(z**2 + z_disk_code**2)) ** 2
    )


def Phi_halo_function(r):
    """NFW dark-matter halo gravitational potential.

    Parameters
    ----------
    r : Spherical radius in code units.
    """
    return (
        -(G_code * M_halo_code)
        / (r * (jnp.log(1 + c) - c / (1 + c)))
        * jnp.log(1 + r / R_halo_code)
    )


def vertical_hse_pressure(rho, Phi, dz_, P_top):
    """Pressure in discrete vertical HSE: integrate dP/dz = -rho dPhi/dz downward.

    Trapezoidal integration along each (x, y) column from the top boundary,
    where P = P_top. Returns a pressure consistent with the vertical force
    balance for the supplied density, so a (re)constructed IC does not ring
    vertically. (Matches the analytic isothermal solution to the discretisation
    error of the integrator.)
    """
    dPhi_dz = jnp.gradient(Phi, dz_, axis=2)
    q = rho * dPhi_dz  # equals -dP/dz
    incr = 0.5 * (q[:, :, :-1] + q[:, :, 1:]) * dz_  # P[k] - P[k+1]
    rev = jnp.cumsum(incr[:, :, ::-1], axis=2)[:, :, ::-1]
    return jnp.concatenate([P_top + rev, P_top], axis=2)


def _gaussian_smooth_3d(field, sigma):
    """Separable 3D Gaussian filter with edge-replicate padding, native JAX.

    Three sequential 1D convolutions on the GPU (one per axis); no GPU-host
    round-trip. Numerically equivalent (within float rounding) to
    scipy.ndimage.gaussian_filter(..., mode="nearest", truncate=4.0) and uses
    the same 4*sigma kernel radius. Returns a field of the same shape as the
    input.
    """
    if sigma <= 0:
        return field
    radius = int(4.0 * sigma + 0.5)
    xs = jnp.arange(-radius, radius + 1, dtype=field.dtype)
    k1d = jnp.exp(-0.5 * (xs / sigma) ** 2)
    k1d = k1d / k1d.sum()

    for axis in range(field.ndim):
        pad = [(0, 0)] * field.ndim
        pad[axis] = (radius, radius)
        padded = jnp.pad(field, pad, mode="edge")[None, None]
        kshape = [1, 1, 1]
        kshape[axis] = 2 * radius + 1
        kernel = k1d.reshape(kshape)[None, None]
        field = jax.lax.conv_general_dilated(
            padded, kernel,
            window_strides=(1, 1, 1),
            padding="VALID",
            dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
        )[0, 0]
    return field


# ---------------------------------------------------------------------------
# Initial-condition build
# ---------------------------------------------------------------------------
def build_config():
    """Build the SimulationConfig + registered_variables.

    Cheap: no 3D IC fields are materialised here. Factored out of
    `build_initial_conditions` so post-processing scripts (e.g. cgols_analyse.py)
    can reconstruct the grid/config without re-running the full IC build.
    """
    # ---- Grid / box ----
    bx_size_x = 10 * u.kiloparsec
    bx_size_y = 10 * u.kiloparsec
    bx_size_z = 20 * u.kiloparsec

    L_x = bx_size_x.to(code_units.code_length).value
    L_y = bx_size_y.to(code_units.code_length).value
    L_z = bx_size_z.to(code_units.code_length).value

    dim_x = dim_y = 512
    dim_z = dim_x * 2

    print(f"Rendering in {dim_x} x {dim_y} x {dim_z} dimensions")

    config = SimulationConfig(
        memory_analysis=True,
        geometry=CARTESIAN,
        solver_mode=FINITE_DIFFERENCE,
        # SSPRK4 (RK4_SSP) would be more robust to the strong wind-driven shocks,
        # but it carries one more full-state register and OOMs at 512x512x1024.
        # Staying on the memory-lean 2N-storage RK4_LSRK and relying on the
        # tighter CFL / larger injection radius / smoother onset for stability.
        time_integrator=RK4_LSRK,
        backend=PALLAS,
        pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True,
        pallas_interpret=False,
        dimensionality=3,
        box_size=StaticFloatVector(L_x, L_y, L_z),
        num_cells=StaticIntVector(dim_x, dim_y, dim_z),
        self_gravity=False,
        self_gravity_version=SIMPLE_SOURCE_TERM,
        progress_bar=True,
        monitor_diagnostics=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY),
            BoundarySettings1D(left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY),
            BoundarySettings1D(left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY),
        ),
        external_potential=True,
        donate_state=True,
        cgols_wind_config=CGOLSWindConfig(cgols_wind=True),
    )
    registered_variables = get_registered_variables(config)
    return config, registered_variables


def build_initial_conditions():
    """Build the simulation IC and return the four objects time_integration needs.

    Memory layout: at any moment only a handful of 3D fields are alive on the GPU.
    Intermediates (cutoff, Sigma, exp_factor, bracket, Phi_disk_sph, Phi_sph, the
    pressure components, the pressure gradients, ...) are deleted as soon as
    their last consumer is done. The 3D Phi_disk / Phi_halo arrays are never
    materialised - the rotation-curve diagnostic is computed on a 1D midplane
    line instead.
    """
    config, registered_variables = build_config()
    L_x = config.box_size.x
    L_y = config.box_size.y
    L_z = config.box_size.z
    dim_x = config.num_cells.x
    dim_y = config.num_cells.y
    dim_z = config.num_cells.z

    helper_data = get_helper_data(config)
    centers = helper_data.geometric_centers  # shape (dim_x, dim_y, dim_z, 3)

    # Shift origin to box center; derive scalar cell sizes; drop the unshifted views.
    X_c = centers[..., 0] - L_x / 2
    Y_c = centers[..., 1] - L_y / 2
    Z_c = centers[..., 2] - L_z / 2
    dx = float(X_c[1, 0, 0] - X_c[0, 0, 0])
    dy = float(Y_c[0, 1, 0] - Y_c[0, 0, 0])
    dz = float(Z_c[0, 0, 1] - Z_c[0, 0, 0])
    del centers, helper_data

    # Cylindrical and spherical radii (code units), floored to avoid divide-by-zero.
    R_cyl = jnp.maximum(jnp.sqrt(X_c**2 + Y_c**2), 0.25 * dx)
    r_sph = jnp.maximum(jnp.sqrt(X_c**2 + Y_c**2 + Z_c**2), 0.25 * dx)

    mid_x = dim_x // 2
    mid_y = dim_y // 2
    mid_z = dim_z // 2

    # ---- Total potential (full 3D Phi_disk / Phi_halo never materialised) ----
    Phi_total = Phi_disk_function(R_cyl, Z_c) + Phi_halo_function(r_sph)

    # ---- Constants needed for both gas components ----
    k_B_code = k_B.to(code_units.code_energy / u.K).value
    T_disk_code = T_disk.to(u.K).value
    m_p_code = m_p.to(code_units.code_mass).value
    c_s_d = jnp.sqrt(k_B_code * T_disk_code / (mu * m_p_code))  # isothermal disk

    T_halo_code = T_halo.to(u.K).value
    rho_0h_code = rho_0h.to(code_units.code_density).value
    c_s_h = jnp.sqrt(k_B_code * T_halo_code / (mu * m_p_code))
    K_const = c_s_h**2 * rho_0h_code ** (1 - gamma) / gamma  # adiabatic P = K rho^gamma

    # ---- Disk gas: surface density (exponential with smooth truncation) ----
    M_gas_code = M_gas.to(code_units.code_mass).value
    R_gas_code = R_gas.to(code_units.code_length).value
    sigma_0 = M_gas_code / (2 * jnp.pi * R_gas_code**2)
    R_trunc_code = R_trunc.to(code_units.code_length).value
    delta = (0.15 * u.kiloparsec).to(code_units.code_length).value  # ramp width

    cutoff = jnp.where(
        R_cyl < R_trunc_code,
        1.0,
        jnp.exp(-((R_cyl - R_trunc_code) / delta) ** 2),
    )
    Sigma = sigma_0 * jnp.exp(-R_cyl / R_gas_code) * cutoff
    del cutoff

    # ---- Disk gas: vertical density structure (isothermal, Eq. 4) ----
    #   rho(z) = rho_0d * exp[-(Phi(z) - Phi_0d) / c_s_d^2]
    # Note: c_s_d is the *isothermal* sound speed, so P_disk = rho * c_s_d**2 is the
    # true thermal pressure rho * k_B * T / (mu * m_p). The paper writes
    # P = rho c_s^2 / gamma in Eq. 6, but there "c_s" is the adiabatic sound speed;
    # with the isothermal c_s_d defined in Eq. 4 the 1/gamma must be dropped.
    Phi_0d = Phi_total[:, :, mid_z]
    exp_factor = jnp.exp(-(Phi_total - Phi_0d[:, :, None]) / c_s_d**2)
    integral = jnp.sum(exp_factor, axis=2) * dz  # shape (dim_x, dim_y)
    rho_0d = Sigma[:, :, mid_z] / integral
    rho_disk = rho_0d[:, :, None] * exp_factor
    del exp_factor, Sigma, integral, rho_0d, Phi_0d

    # ---- Halo gas: hydrostatic, adiabatic atmosphere ----
    #   rho(r) = rho_0h * [1 - (gamma-1) (Phi - Phi_0h) / c_s_h^2]^(1/(gamma-1))
    # Spherically symmetric potential (disk evaluated on the sphere). We recompute
    # Phi_halo on r_sph here rather than caching the 3D field.
    Phi_sph = Phi_disk_function(0, r_sph) + Phi_halo_function(r_sph)
    r_ref_code = (100 * u.kiloparsec).to(code_units.code_length).value
    Phi_0h = Phi_disk_function(0, r_ref_code) + Phi_halo_function(r_ref_code)

    # Correct hydrostatic-equilibrium solution. Integrating dP/dr = -rho dPhi/dr for
    # an adiabatic gas (P = K rho^gamma) gives rho ~ [1 + (gamma-1)(Phi_0h - Phi)/c_s_h^2],
    # i.e. the bracket below. Inside r = 100 kpc, Phi < Phi_0h so the bracket is > 1
    # and density rises inward (matching the paper's Figure 4). The printed Eq. 7 reads
    # with the opposite sign on (Phi - Phi_0h), but this physical form is the one that
    # reproduces their profiles, so we keep it. Floored at 0 for safety outside the box.
    bracket = 1 - (gamma - 1) * (Phi_sph - Phi_0h) / c_s_h**2
    del Phi_sph
    rho_halo = rho_0h_code * jnp.maximum(bracket, 0.0) ** (1 / (gamma - 1))
    del bracket, r_sph

    # ---- Vertical resolution diagnostic (only needs Phi_total + scalars) ----
    phi_col = Phi_total[mid_x, mid_y, :]
    d2Phi_dz2_mid = jnp.gradient(jnp.gradient(phi_col, dz), dz)[mid_z]
    H_gas = c_s_d / jnp.sqrt(jnp.maximum(d2Phi_dz2_mid, 1e-30))
    pc_per_code = code_length.to(u.pc).value
    print(f"Cell size dz               = {dz * pc_per_code:8.1f} pc")
    print(f"Central gas scale height H = {float(H_gas) * pc_per_code:8.1f} pc")
    print(f"Cells per gas scale height = {float(H_gas) / dz:8.2f}")
    del phi_col

    # ---- Branch on smoothing: build (rho_total, P_total, v_phi, ux, uy, uz) ----
    # Smoothed path: linear smoothing of rho_total (= smooth(rho_disk)+smooth(rho_halo)),
    # then REBUILD the pressure in vertical HSE with the smoothed density. Using the
    # *total* P and rho means the pressure-supported halo gets a_phi ~ 0 automatically,
    # so no disk-fraction weighting is needed.
    # Unsmoothed path: keep P_disk for the pressure-gradient term and weight v_phi by
    # rho_disk / rho_total so the static hot halo is not spun up.
    if SMOOTHING_SIGMA_CELLS > 0:
        # Top-boundary halo pressure for HSE: a (Nx, Ny, 1) slice, not full 3D.
        P_top_2d = K_const * rho_halo[:, :, -1:] ** gamma
        rho_total = rho_disk + rho_halo
        del rho_disk, rho_halo

        rho_total = _gaussian_smooth_3d(rho_total, SMOOTHING_SIGMA_CELLS)
        P_total = vertical_hse_pressure(rho_total, Phi_total, dz, P_top_2d)
        del P_top_2d

        # Rotation from total fields.
        dPhi_dx = jnp.gradient(Phi_total, dx, axis=0)
        dPhi_dy = jnp.gradient(Phi_total, dy, axis=1)
        dPhi_dR = (X_c * dPhi_dx + Y_c * dPhi_dy) / R_cyl
        del dPhi_dx, dPhi_dy

        dP_dx = jnp.gradient(P_total, dx, axis=0)
        dP_dy = jnp.gradient(P_total, dy, axis=1)
        dP_dR = (X_c * dP_dx + Y_c * dP_dy) / R_cyl
        del dP_dx, dP_dy

        a_phi = dPhi_dR + dP_dR / rho_total
        del dPhi_dR, dP_dR
        v_phi = jnp.sqrt(jnp.maximum(a_phi * R_cyl, 0.0))
        del a_phi

        print(f"Applied HSE-preserving smoothing: sigma = {SMOOTHING_SIGMA_CELLS} cells")
    else:
        rho_total = rho_disk + rho_halo
        # Build P_total without ever holding both 3D pressure components.
        P_total = rho_disk * c_s_d**2 + K_const * rho_halo**gamma
        del rho_halo

        dPhi_dx = jnp.gradient(Phi_total, dx, axis=0)
        dPhi_dy = jnp.gradient(Phi_total, dy, axis=1)
        dPhi_dR = (X_c * dPhi_dx + Y_c * dPhi_dy) / R_cyl
        del dPhi_dx, dPhi_dy

        # Pressure gradient uses the *disk* pressure only (P_disk = rho_disk * c_s_d**2).
        # We recompute it inside the gradient call rather than caching a 3D P_disk.
        P_disk_for_grad = rho_disk * c_s_d**2
        dP_dx = jnp.gradient(P_disk_for_grad, dx, axis=0)
        dP_dy = jnp.gradient(P_disk_for_grad, dy, axis=1)
        del P_disk_for_grad
        dP_dR = (X_c * dP_dx + Y_c * dP_dy) / R_cyl
        del dP_dx, dP_dy

        # Centripetal acceleration. Gravity (dPhi/dR > 0) drives the rotation; the
        # outward pressure gradient (dP/dR < 0) reduces it. The disk density in the
        # pressure term is floored so the 1/rho term does not blow up where the disk
        # gas vanishes far from the plane.
        safe_rho_disk = jnp.maximum(rho_disk, 1e-12)
        a_phi = dPhi_dR + dP_dR / safe_rho_disk
        del safe_rho_disk, dPhi_dR, dP_dR
        v_phi = jnp.sqrt(jnp.maximum(a_phi * R_cyl, 0.0))
        del a_phi

        # Mass-weight the rotation by the disk fraction so the static hot halo is not
        # spun up: halo-dominated cells smoothly go to v -> 0, while the disk midplane
        # (rho_disk >> rho_halo) keeps the full circular speed.
        v_phi = v_phi * (rho_disk / rho_total)
        del rho_disk

    # ---- Velocity components from v_phi (Eq. 6 chain rule):
    # dPhi/dR = (x/R) dPhi/dx + (y/R) dPhi/dy, v_x = -v_phi y/R, v_y = v_phi x/R.
    ux = -v_phi * Y_c / R_cyl
    uy = v_phi * X_c / R_cyl
    uz = jnp.zeros_like(ux)
    del v_phi, R_cyl

    print(f"P_total min: {P_total.min():.3e} code units")
    print(f"P_total max: {P_total.max():.3e} code units")

    # ---- Diagnostic plot: rotation curve (midplane), from 1D potential slices ----
    x_full = X_c[:, mid_y, mid_z]
    r_line = jnp.maximum(jnp.abs(x_full), 0.25 * dx)
    z_zero = jnp.zeros_like(r_line)
    Phi_disk_line = Phi_disk_function(r_line, z_zero)
    Phi_halo_line = Phi_halo_function(r_line)

    dPhi_disk_dx = jnp.gradient(Phi_disk_line, dx)
    dPhi_halo_dx = jnp.gradient(Phi_halo_line, dx)
    dPhi_total_dx = dPhi_disk_dx + dPhi_halo_dx

    # v_circ = sign(x) * sqrt(|x * dPhi/dx|)
    v_disk_rot = jnp.sign(x_full) * jnp.sqrt(jnp.maximum(jnp.abs(x_full * dPhi_disk_dx), 0))
    v_halo_rot = jnp.sign(x_full) * jnp.sqrt(jnp.maximum(jnp.abs(x_full * dPhi_halo_dx), 0))
    v_total_rot = jnp.sign(x_full) * jnp.sqrt(jnp.maximum(jnp.abs(x_full * dPhi_total_dx), 0))

    v_to_kms = code_velocity.to(u.km / u.s).value

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x_full, v_total_rot * v_to_kms, "b-", linewidth=1.5, label="combined")
    ax.plot(x_full, v_disk_rot * v_to_kms, color="orange", linestyle="-.", linewidth=1.5, label="disk")
    ax.plot(x_full, v_halo_rot * v_to_kms, "g--", linewidth=1.5, label="halo")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-200, 200)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel(r"$v_{\rm circ}$ [km s$^{-1}$]")
    ax.legend()
    ax.set_title("Rotation curve (midplane)")
    plt.tight_layout()
    plt.savefig("cgols_rotation_curve.png", dpi=300)
    plt.close(fig)
    del r_line, z_zero, Phi_disk_line, Phi_halo_line
    del dPhi_disk_dx, dPhi_halo_dx, dPhi_total_dx
    del v_disk_rot, v_halo_rot, v_total_rot

    # ---- Diagnostic plot: density and temperature profiles ----
    code_density_cgs = (code_mass / code_length**3).to(u.g / u.cm**3).value
    m_p_cgs = m_p.to(u.g).value

    R_kpc = jnp.sqrt(X_c[mid_x:, mid_y, mid_z] ** 2 + Y_c[mid_x:, mid_y, mid_z] ** 2)
    z_kpc = Z_c[mid_x, mid_y, mid_z:]

    n_midplane = rho_total[mid_x:, mid_y, mid_z] * code_density_cgs / (mu * m_p_cgs)
    n_zaxis = rho_total[mid_x, mid_y, mid_z:] * code_density_cgs / (mu * m_p_cgs)
    T_midplane = P_total[mid_x:, mid_y, mid_z] / rho_total[mid_x:, mid_y, mid_z] * T_factor
    T_zaxis = P_total[mid_x, mid_y, mid_z:] / rho_total[mid_x, mid_y, mid_z:] * T_factor

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    ax1.plot(R_kpc, n_midplane, "b--", linewidth=2, label="xy-plane")
    ax1.plot(z_kpc, n_zaxis, "r:", linewidth=2, label="z-axis")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(1e-2, 1e1)
    ax1.set_ylim(1e-4, 1e3)
    ax1.set_xlabel("r [kpc]")
    ax1.set_ylabel(r"n [cm$^{-3}$]")
    ax1.legend()
    ax1.set_title("Density profiles")

    ax2.plot(R_kpc, T_midplane, "b--", linewidth=2, label="xy-plane")
    ax2.plot(z_kpc, T_zaxis, "r:", linewidth=2, label="z-axis")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1e-2, 1e1)
    ax2.set_ylim(1e3, 1e7)
    ax2.set_xlabel("r [kpc]")
    ax2.set_ylabel("T [K]")
    ax2.legend()
    ax2.set_title("Temperature profiles")

    plt.tight_layout()
    plt.savefig("cgols_initial_profiles.png", dpi=300)
    plt.close(fig)
    del R_kpc, z_kpc, n_midplane, n_zaxis, T_midplane, T_zaxis
    del X_c, Y_c, Z_c

    # ---- Simulation setup ----
    t_end = TOTAL_TIME.to(code_units.code_time).value

    params = SimulationParams(
        t_end=t_end,
        # Keep the default 0.4: the non-SSP RK4_LSRK + the strong wind-driven
        # shocks need the CFL margin once feedback is on. Running at 0.8 (2x the
        # default) blew up ~47 Myr in - a flux overshoot into a near-vacuum cell
        # ran |v| away faster than the rho/P floors could contain it.
        C_cfl=0.8,
        gamma=gamma,
        # See load_initial_conditions() for the rationale: floors ~2 orders below
        # the ambient minimums, not the dynamically-zero 1e-14 default.
        minimum_density=1e-4,
        minimum_pressure=1e-5,
        gravitational_potential=Phi_total,
        cgols_wind_params=build_cgols_wind_params(),
    )
    
    jnp.save("cgols_initial_potential.npy", Phi_total)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho_total,
        velocity_x=ux,
        velocity_y=uy,
        velocity_z=uz,
        gas_pressure=P_total,
    )
    del rho_total, ux, uy, uz, P_total, Phi_total

    print(f"Initial density min:    {initial_state[registered_variables.density_index].min():.3e}")
    print(f"Initial density max:    {initial_state[registered_variables.density_index].max():.3e}")
    print(f"Initial velocity_x min: {initial_state[registered_variables.velocity_index.x].min():.3e}")
    print(f"Initial velocity_x max: {initial_state[registered_variables.velocity_index.x].max():.3e}")
    print(f"Initial velocity_y min: {initial_state[registered_variables.velocity_index.y].min():.3e}")
    print(f"Initial velocity_y max: {initial_state[registered_variables.velocity_index.y].max():.3e}")
    print(f"Initial velocity_z min: {initial_state[registered_variables.velocity_index.z].min():.3e}")
    print(f"Initial velocity_z max: {initial_state[registered_variables.velocity_index.z].max():.3e}")
    print(f"Initial pressure min:   {initial_state[registered_variables.pressure_index].min():.3e}")
    print(f"Initial pressure max:   {initial_state[registered_variables.pressure_index].max():.3e}")

    config = finalize_config(config, initial_state.shape)

    return initial_state, config, params, registered_variables

def load_initial_conditions():
    """Load the initial state and config from disk, for post-processing without re-running the IC build."""
    config, registered_variables = build_config()
    initial_state = jnp.load("cgols_initial_state.npy")
    config = finalize_config(config, initial_state.shape)
    Phi_total = jnp.load("cgols_initial_potential.npy")
    params = SimulationParams(
        t_end=TOTAL_TIME.to(code_units.code_time).value,
        # Default 0.4; see build_initial_conditions() - 0.8 blew up ~47 Myr in.
        C_cfl=0.8,
        gamma=gamma,
        # Floors ~2 orders below the ambient box minimums (min_rho~8e-3,
        # min_P~3e-3). The 1e-14 default is dynamically zero: a wind-cavity cell
        # floored to 1e-14 next to a 3e-3 neighbour is a ~1e11 pressure ratio,
        # whose flux evacuates the cell in one step -> NaN. These floors cap that
        # gradient while leaving the rarefied cavity room to form.
        minimum_density=1e-4,
        minimum_pressure=1e-5,
        gravitational_potential=Phi_total,
        cgols_wind_params=build_cgols_wind_params(),
    )
    return initial_state, config, params, registered_variables
    

# ---------------------------------------------------------------------------
# Post-simulation analysis
# ---------------------------------------------------------------------------
def analyse_results(final_state, config, registered_variables, initial_state=None):
    """Static-equilibrium check + vertical-velocity diagnostic.

    Wrapped in a function so the recomputed grid and the plot scratch arrays
    are released once the figures are written.

    If `initial_state` is None (e.g. when `time_integration` was called with
    `donate_state=True` and the buffer was donated), the function skips the
    initial-vs-final comparisons and shows final-state-only diagnostics.
    """
    have_initial = initial_state is not None
    
    print(jax.devices()[0].memory_stats())
    print({k: v.shape for k, v in zip(['rho','P'], (final_state[registered_variables.density_index], 
                                                    final_state[registered_variables.pressure_index]))})

    # ---- Recompute the grid we need for the plots ----
    L_x = config.box_size.x
    L_y = config.box_size.y
    L_z = config.box_size.z

    helper_data = get_helper_data(config)
    centers = helper_data.geometric_centers
    X_c = centers[..., 0] - L_x / 2
    Y_c = centers[..., 1] - L_y / 2
    Z_c = centers[..., 2] - L_z / 2
    R_cyl = jnp.sqrt(X_c**2 + Y_c**2)
    del centers, helper_data

    dim_x, dim_y, dim_z = final_state.shape[1:4]
    mid_x = dim_x // 2
    mid_y = dim_y // 2
    mid_z = dim_z // 2

    code_density_cgs = (code_mass / code_length**3).to(u.g / u.cm**3).value
    m_p_cgs = m_p.to(u.g).value
    v_to_kms = code_velocity.to(u.km / u.s).value

    R_kpc = R_cyl[mid_x:, mid_y, mid_z]
    z_kpc = Z_c[mid_x, mid_y, mid_z:]
    del R_cyl

    # ---- Static-equilibrium check: compare the final state to the initial state ----
    def density_temperature_profiles(state):
        """Return (n_midplane, n_zaxis, T_midplane, T_zaxis) for a primitive state.

        Slice the 1-D midplane / z-axis lines out of rho and P *first*, then derive
        n and T on those lines, so the full-3D T and n arrays are never materialised.
        """
        rho = state[registered_variables.density_index]
        P = state[registered_variables.pressure_index]
        rho_mid, rho_z = rho[mid_x:, mid_y, mid_z], rho[mid_x, mid_y, mid_z:]
        P_mid, P_z = P[mid_x:, mid_y, mid_z], P[mid_x, mid_y, mid_z:]
        n_factor = code_density_cgs / (mu * m_p_cgs)
        return (
            rho_mid * n_factor,
            rho_z * n_factor,
            P_mid / rho_mid * T_factor,
            P_z / rho_z * T_factor,
        )

    n_mid_f, n_z_f, T_mid_f, T_z_f = density_temperature_profiles(final_state)
    jax.block_until_ready((n_mid_f, n_z_f, T_mid_f, T_z_f))

    if have_initial:
        n_mid_i, n_z_i, T_mid_i, T_z_i = density_temperature_profiles(initial_state)
        jax.block_until_ready((n_mid_i, n_z_i, T_mid_i, T_z_i))
        rho_i = initial_state[registered_variables.density_index]
        rho_f = final_state[registered_variables.density_index]
        rel_drift = jnp.abs(rho_f - rho_i) / jnp.maximum(jnp.abs(rho_i), 1e-30)
        print(f"Density drift over {TOTAL_TIME}: max = {rel_drift.max():.3e}, mean = {rel_drift.mean():.3e}")
        del rho_i, rho_f, rel_drift

    def _style_loglog(ax, xlabel, ylabel, ylim, title):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-2, 1e1)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    if have_initial:
        ax1.plot(R_kpc, n_mid_i, "b-", linewidth=2, label="initial")
    ax1.plot(R_kpc, n_mid_f, "co", markersize=3, markevery=2, label="final")
    _style_loglog(ax1, "R [kpc]", r"n [cm$^{-3}$]", (1e-4, 1e3), "Density - midplane")

    if have_initial:
        ax2.plot(z_kpc, n_z_i, "r-", linewidth=2, label="initial")
    ax2.plot(z_kpc, n_z_f, "mo", markersize=3, markevery=3, label="final")
    _style_loglog(ax2, "z [kpc]", r"n [cm$^{-3}$]", (1e-4, 1e3), "Density - z-axis")

    if have_initial:
        ax3.plot(R_kpc, T_mid_i, "b-", linewidth=2, label="initial")
    ax3.plot(R_kpc, T_mid_f, "co", markersize=3, markevery=2, label="final")
    _style_loglog(ax3, "R [kpc]", "T [K]", (1e3, 1e7), "Temperature - midplane")

    if have_initial:
        ax4.plot(z_kpc, T_z_i, "r-", linewidth=2, label="initial")
    ax4.plot(z_kpc, T_z_f, "mo", markersize=3, markevery=3, label="final")
    _style_loglog(ax4, "z [kpc]", "T [K]", (1e3, 1e7), "Temperature - z-axis")

    suptitle = (
        f"Static check: initial vs final after t = {TOTAL_TIME}"
        if have_initial
        else f"Final state after t = {TOTAL_TIME}"
    )
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig("cgols_static_check.png", dpi=300)
    plt.close(fig)
    del n_mid_f, n_z_f, T_mid_f, T_z_f
    if have_initial:
        del n_mid_i, n_z_i, T_mid_i, T_z_i

    # ---- Diagnostic: vertical velocity (v_z) ----
    #
    # v_z starts at 0 everywhere. If the disk is out of (discrete) vertical
    # equilibrium it bounces/puffs, which shows up directly as growing |v_z|. The
    # left panel is the xy-plane rms of v_z vs height; the right panel is an x-z
    # slice of the final v_z (blue/red = down/up flows) so vertical outflow off the
    # disk is visible.
    vz_f = final_state[registered_variables.velocity_index.z]
    print(f"Final   |v_z| max: {float(jnp.abs(vz_f).max()) * v_to_kms:.3e} km/s")
    print(f"Final   |v_z| mean: {float(jnp.abs(vz_f).mean()) * v_to_kms:.3e} km/s")

    vz_rms_f = jnp.sqrt(jnp.mean(vz_f**2, axis=(0, 1))) * v_to_kms
    z_line = Z_c[mid_x, mid_y, :]

    if have_initial:
        vz_i = initial_state[registered_variables.velocity_index.z]
        print(f"Initial |v_z| max: {float(jnp.abs(vz_i).max()) * v_to_kms:.3e} km/s")
        vz_rms_i = jnp.sqrt(jnp.mean(vz_i**2, axis=(0, 1))) * v_to_kms

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))

    if have_initial:
        axa.plot(z_line, vz_rms_i, "k-", linewidth=1.5, label="initial")
    axa.plot(z_line, vz_rms_f, "r-", linewidth=1.5, label="final")
    axa.set_xlabel("z [kpc]")
    axa.set_ylabel(r"rms $v_z$ over xy-plane [km s$^{-1}$]")
    axa.set_title("Vertical velocity growth")
    axa.legend()

    vz_slice = np.asarray(vz_f[:, mid_y, :].T) * v_to_kms  # (dim_z, dim_x)
    extent = [
        float(X_c[0, mid_y, mid_z]),
        float(X_c[-1, mid_y, mid_z]),
        float(Z_c[mid_x, mid_y, 0]),
        float(Z_c[mid_x, mid_y, -1]),
    ]
    vmax = float(np.abs(vz_slice).max()) or 1.0
    im = axb.imshow(
        vz_slice, origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    axb.set_xlabel("x [kpc]")
    axb.set_ylabel("z [kpc]")
    axb.set_title("Final $v_z$ (x-z slice, y=0)")
    plt.colorbar(im, ax=axb, label=r"$v_z$ [km s$^{-1}$]")

    plt.tight_layout()
    plt.savefig("cgols_vz_diagnostic.png", dpi=300)
    plt.close(fig)

    # ---- Final-state morphology, phase diagram, vertical mass flux ----
    #
    # Four panels:
    #   (A) Edge-on (x-z) log density - shows disk puffing, fountains, halo asymmetries.
    #   (B) Face-on (x-y) log density - checks the disk stays axisymmetric (grid-aligned
    #       m=4 patterns would show here).
    #   (C) Mass-weighted log n - log T phase diagram - shows whether the disk and
    #       halo still occupy their two intended loci or have collapsed onto a single
    #       phase.
    #   (D) Net vertical mass flux Mdot(z) = sum_xy(rho * v_z) * dx * dy, in M_sun/yr -
    #       positive = upward mass transport. Quantifies any outflow / fountain.
    from matplotlib.colors import LogNorm

    rho_f = final_state[registered_variables.density_index]
    P_f = final_state[registered_variables.pressure_index]
    T_f = P_f / rho_f * T_factor
    n_f = rho_f * code_density_cgs / (mu * m_p_cgs)

    dx = L_x / dim_x
    dy = L_y / dim_y
    dz_cell = L_z / dim_z
    cell_volume = dx * dy * dz_cell  # code length^3
    code_mass_to_msun = code_mass.to(u.M_sun).value
    code_mdot_to_msun_per_yr = (code_mass / code_units.code_time).to(u.M_sun / u.yr).value

    rho_xz = np.asarray(rho_f[:, mid_y, :].T)  # (dim_z, dim_x)
    rho_xy = np.asarray(rho_f[:, :, mid_z].T)  # (dim_y, dim_x)
    extent_xz = [
        float(X_c[0, mid_y, mid_z]),
        float(X_c[-1, mid_y, mid_z]),
        float(Z_c[mid_x, mid_y, 0]),
        float(Z_c[mid_x, mid_y, -1]),
    ]
    extent_xy = [
        float(X_c[0, mid_y, mid_z]),
        float(X_c[-1, mid_y, mid_z]),
        float(Y_c[mid_x, 0, mid_z]),
        float(Y_c[mid_x, -1, mid_z]),
    ]

    # Mass-weighted phase histogram in solar masses per bin. Flattening 512*512*1024
    # cells is ~1 GB per array in float32; bump `stride` to 2 if RAM is tight.
    stride = 1
    sl = (slice(None, None, stride),) * 3
    cell_mass_msun = np.asarray(rho_f[sl]).ravel() * cell_volume * code_mass_to_msun * stride**3
    log_n_arr = np.log10(np.maximum(np.asarray(n_f[sl]).ravel(), 1e-30))
    log_T_arr = np.log10(np.maximum(np.asarray(T_f[sl]).ravel(), 1e-30))
    h, x_edges, y_edges = np.histogram2d(
        log_n_arr, log_T_arr,
        bins=(120, 120),
        range=[[-6, 4], [2, 8]],
        weights=cell_mass_msun,
    )
    del log_n_arr, log_T_arr, cell_mass_msun
    h_plot = np.where(h > 0, h, np.nan)

    mdot_z_f = np.asarray(jnp.sum(rho_f * vz_f, axis=(0, 1))) * float(dx * dy) * code_mdot_to_msun_per_yr
    mdot_z_i = None
    if have_initial:
        rho_i = initial_state[registered_variables.density_index]
        mdot_z_i = np.asarray(jnp.sum(rho_i * vz_i, axis=(0, 1))) * float(dx * dy) * code_mdot_to_msun_per_yr
        del rho_i
    z_full = np.asarray(Z_c[mid_x, mid_y, :])

    rho_max = float(rho_f.max())
    rho_floor = max(float(rho_f.min()), rho_max * 1e-6)
    del rho_f, P_f, T_f, n_f

    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(13, 11))

    im_a = axA.imshow(
        rho_xz, origin="lower", extent=extent_xz, aspect="auto",
        cmap="magma", norm=LogNorm(vmin=rho_floor, vmax=rho_max),
    )
    axA.set_xlabel("x [kpc]")
    axA.set_ylabel("z [kpc]")
    axA.set_title(r"Final $\rho$ - edge-on (x-z, y=0)")
    plt.colorbar(im_a, ax=axA, label=r"$\rho$ [code units]")

    im_b = axB.imshow(
        rho_xy, origin="lower", extent=extent_xy, aspect="equal",
        cmap="magma", norm=LogNorm(vmin=rho_floor, vmax=rho_max),
    )
    axB.set_xlabel("x [kpc]")
    axB.set_ylabel("y [kpc]")
    axB.set_title(r"Final $\rho$ - face-on (x-y, z=0)")
    plt.colorbar(im_b, ax=axB, label=r"$\rho$ [code units]")

    im_c = axC.imshow(
        h_plot.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(),
    )
    axC.set_xlabel(r"$\log_{10} n$ [cm$^{-3}$]")
    axC.set_ylabel(r"$\log_{10} T$ [K]")
    axC.set_title("Mass-weighted phase diagram (final)")
    plt.colorbar(im_c, ax=axC, label=r"M [$M_\odot$ / bin]")

    axD.axhline(0, color="0.6", linewidth=0.8)
    if mdot_z_i is not None:
        axD.plot(z_full, mdot_z_i, "k-", linewidth=1.0, label="initial")
    axD.plot(z_full, mdot_z_f, "r-", linewidth=1.5, label="final")
    axD.set_xlabel("z [kpc]")
    axD.set_ylabel(r"net $\dot M(z)$ [$M_\odot$ yr$^{-1}$]")
    axD.set_title("Net vertical mass flux through z-planes")
    axD.legend()

    plt.tight_layout()
    plt.savefig("cgols_extras.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reference: radiative cooling (NOT applied here)
#
# The run above reproduces the paper's ADIABATIC A-series, which uses no cooling;
# the paper reports those initial conditions are stable for >1 Gyr AT THEIR
# RESOLUTION (dx ~ 5 pc, i.e. ~30 cells per 0.15 kpc disk scale height). At the
# 128^3 / 256 resolution here (dx ~ 78 pc) the scale height spans ~2 cells, which
# is why the disk puffs - that is a resolution issue, not a missing-cooling one.
#
# Cooling is used only in the radiative B/C series (companion paper). For those,
# the paper applies an operator-split CIE cooling source term with the analytic
# solar-metallicity fit below (Appendix A.3, Eq. A4) and a 10^4 K temperature
# floor. The volumetric cooling rate is n^2 * Lambda(T). This function is
# provided for reference; it would attach as a post-hydro-step source term if you
# move to a radiative run (check astronomix for a cooling / source-term hook).
# ---------------------------------------------------------------------------
def cooling_lambda_cgs(T):
    """CIE cooling function Lambda(T) [erg s^-1 cm^3], Schneider & Robertson 2018 Eq. A4.

    T in Kelvin. Returns 0 below the 10^4 K temperature floor.
    """
    logT = jnp.log10(T)
    lam = jnp.where(
        logT < 5.9,
        10.0 ** (-1.3 * (logT - 5.25) ** 2 - 21.25),
        jnp.where(
            logT < 7.4,
            10.0 ** (0.7 * (logT - 7.1) ** 2 - 22.8),
            10.0 ** (0.45 * logT - 26.065),
        ),
    )
    return jnp.where(T < 1e4, 0.0, lam)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    # initial_state, config, params, registered_variables = build_initial_conditions()

    # Save BEFORE integration: donate_state=True consumes the initial buffer in-place,
    # so this device->host copy must happen while the buffer is still valid.
    # jnp.save("cgols_initial_state.npy", initial_state)
    
    initial_state, config, params, registered_variables = load_initial_conditions()

    final_state = time_integration(initial_state, config, params, registered_variables)
    jnp.save("cgols_final_state.npy", final_state)

    # Analysis runs in a separate process (cgols_analyse.py) on a fresh, empty GPU.
    # Doing it here would OOM: XLA still holds the sim's ~32 GB pool.
    # print(
    #     "Saved cgols_initial_state.npy / cgols_final_state.npy. "
    #     "Run `python cgols_analyse.py` to produce the figures."
    # )