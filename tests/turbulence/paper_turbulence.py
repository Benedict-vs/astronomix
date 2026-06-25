"""
HOW-MHD (arXiv:2304.04360) turbulence tests, sections 3.9 / 3.10, reproduced in
astronomix with OU (temporally-correlated) forcing instead of the paper's
white-in-time driving, and the Pallas backend throughout.

Paper cases (one per invocation):
  ISM : isothermal, M_turb ~ 10,  beta_p = 0.1,    run to  5 t_cross
  ICM : isothermal, M_turb ~ 0.5, beta_p = 1e6,    run to 30 t_cross
  CMP : M_turb ~ 1, beta_p = 1, isothermal AND adiabatic, run to 4 t_cross

M_turb = v_rms / a, beta_p = P_thermal / (B0^2/2). Normalisation here keeps the
driven v_rms ~ 1 and sets a = 1/M_turb, B0 = sqrt(2 * P_thermal / beta) with
P_thermal = a^2 * rho0 (isothermal) or P0 (adiabatic, a = sqrt(gamma P0/rho0)).
This is physically identical to the paper's a=1 / v_rms=M_turb normalisation for
all dimensionless diagnostics (M_turb, beta, spectral slopes, density contrast).

Forcing spectrum matches the paper, |dv_k|^2 ~ k^6 exp(-8k/k_exp), k_exp = 4 pi/L0
=> OU forcing_wavenumber k_f = 0.75 * k_exp = 3 pi.

Per-snapshot diagnostics (density / kinetic / magnetic power spectra + the Fig-16
time series M_turb, (rho-rho0)_rms, E_K, E_B) are accumulated and written to npz;
make_fig_paper.py builds the figures.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import argparse
import os
import time as walltime

import numpy as np
import jax.numpy as jnp

from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    ISOTHERMAL,
    PALLAS,
    PERIODIC_BOUNDARY,
    POSITIVITY_NONE,
    POSITIVITY_HARD_FLOOR,
    POSITIVITY_REDISTRIBUTE,
    POSITIVITY_FOLLOW_LEGACY,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SnapshotSettings,
    finalize_config,
)

_POS = {"legacy": POSITIVITY_FOLLOW_LEGACY, "none": POSITIVITY_NONE,
        "floor": POSITIVITY_HARD_FLOOR, "redist": POSITIVITY_REDISTRIBUTE}
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.analysis_helpers.energy_spectrum import (
    vector_field_energy_spectrum,
    get_kinetic_energy_spectrum,
    get_magnetic_energy_spectrum,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mturb", type=float, required=True, help="target turbulent Mach number")
    p.add_argument("--beta", type=float, required=True, help="initial plasma beta")
    p.add_argument("--eos", choices=["iso", "adiabatic"], default="iso")
    p.add_argument("--gamma", type=float, default=5.0 / 3.0)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--tcross", type=float, default=5.0, help="t_end in crossing times")
    p.add_argument("--F0", type=float, default=3.5, help="OU forcing amplitude")
    p.add_argument("--tau", type=float, default=0.5, help="OU correlation time")
    p.add_argument("--kf", type=float, default=3.0 * np.pi, help="OU peak wavenumber (= 0.75 k_exp)")
    p.add_argument("--nsnap", type=int, default=60)
    p.add_argument("--cfl", type=float, default=1.5)
    p.add_argument("--protect", type=int, default=-1, help="vacuum protection (prot): -1 auto, 0 off, 1 on")
    p.add_argument("--hardfloor", type=int, default=-1,
                   help="enforce_positivity (per-RK-substage + floor): -1 auto (on for supersonic), 0 off, 1 on")
    p.add_argument("--stage_mode", choices=list(_POS), default="legacy",
                   help="per-RK-substage positivity mode (legacy=follow --hardfloor)")
    p.add_argument("--step_mode", choices=list(_POS), default="legacy",
                   help="per-step positivity mode (legacy=follow --hardfloor)")
    p.add_argument("--rhomin", type=float, default=0.02, help="density floor / protection threshold")
    p.add_argument("--vmax", type=float, default=50.0, help="vacuum-protection velocity ceiling")
    p.add_argument("--vmaxcap", type=float, default=float("inf"),
                   help="per-stage positivity velocity cap (REDISTRIBUTE only); inf = off")
    p.add_argument("--vacuum_rest", type=int, default=0,
                   help="zero momentum (v=0) in floored cells so a low rhomin stays stable: 0 off, 1 on")
    p.add_argument("--outdir", type=str, default="data_paper")
    p.add_argument("--norm", choices=["vrms1", "paper"], default="vrms1",
                   help="vrms1: a=1/M_turb, drive v_rms~1 (cheap, dimensionless). "
                        "paper: a=1, drive v_rms~M_turb, B0=sqrt(2/beta) — the "
                        "normalisation in which HOW-MHD Fig.14 log E_B is defined.")
    p.add_argument("--tag", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rho0 = 1.0
    if args.norm == "paper":
        a = 1.0                  # sound speed fixed; drive v_rms to M_turb
        vrms_target = args.mturb
    else:
        a = 1.0 / args.mturb     # v_rms ~ 1 normalisation
        vrms_target = 1.0
    adiabatic = args.eos == "adiabatic"
    if adiabatic:
        P0 = rho0 * a ** 2 / args.gamma            # so sqrt(gamma P0/rho0) = a
        P_thermal = P0
    else:
        P0 = None
        P_thermal = a ** 2 * rho0                  # isothermal gas pressure c_s^2 rho
    B0 = float(np.sqrt(2.0 * P_thermal / args.beta))

    # vacuum protection = HOW-MHD `prot` conservative neighbour redistribution
    # (on for supersonic by default). Hard floor (enforce_positivity) is OFF by
    # default to match the paper, which never hard-floors the evolved state.
    vacuum = (args.mturb >= 2.0) if args.protect < 0 else bool(args.protect)
    # per-RK-substage enforce_positivity: the CFL lever. With prot (vacuum) now
    # wired into the OU path, prot + per-substage positivity is stable at the
    # paper's CFL=1.5 and floor=0.02 for M_turb~10 (no hard tuning needed).
    protect = (args.mturb >= 2.0) if args.hardfloor < 0 else bool(args.hardfloor)

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS if adiabatic else ISOTHERMAL,
        backend=PALLAS,
        pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True,
        pallas_interpret=False,
        memory_analysis=False,
        progress_bar=False,  # our own diagnostics; avoids int(NaN) callback crash on blow-up
        dimensionality=3,
        num_cells=args.N,
        enforce_positivity=protect,
        box_size=1.0,
        mhd=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
        ),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True, ou_forcing=True, vacuum_protection=vacuum,
        ),
        positivity_per_stage_mode=_POS[args.stage_mode],
        positivity_per_step_mode=_POS[args.step_mode],
        positivity_vacuum_rest=bool(args.vacuum_rest),
        return_snapshots=True,
        num_snapshots=args.nsnap,
        snapshot_settings=SnapshotSettings(return_states=True),
    )

    L_inj = config.box_size / 2.0
    t_cross = L_inj / vrms_target

    params = SimulationParams(
        C_cfl=args.cfl,
        gamma=args.gamma,
        isothermal_sound_speed=a,
        t_end=args.tcross * t_cross,
        turbulent_forcing_params=TurbulentForcingParams(
            forcing_amplitude=args.F0, correlation_time=args.tau, forcing_wavenumber=args.kf,
            protection_density_threshold=args.rhomin, protection_max_velocity=args.vmax,
        ),
        positivity_max_velocity=args.vmaxcap,
        minimum_density=args.rhomin,
        minimum_pressure=1e-6,
    )

    rv = get_registered_variables(config)
    density = jnp.ones((args.N, args.N, args.N), dtype=jnp.float32) * rho0
    zero = jnp.zeros_like(density)
    Bz = jnp.ones_like(density) * B0
    bxb, byb, bzb = initialize_interface_fields(zero, zero, Bz)
    ic = dict(
        config=config, registered_variables=rv, density=density,
        velocity_x=zero, velocity_y=zero, velocity_z=zero,
        magnetic_field_x=zero, magnetic_field_y=zero, magnetic_field_z=Bz,
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb, interface_magnetic_field_z=bzb,
    )
    if adiabatic:
        ic["gas_pressure"] = jnp.ones_like(density) * P0
    initial_state = construct_primitive_state(**ic)
    config = finalize_config(config, initial_state.shape)

    print(f"[{args.tag}] eos={args.eos} M_turb_aim={args.mturb} beta={args.beta:g} "
          f"a={a:.3f} P0={P0} B0={B0:.4g} F0={args.F0} kf={args.kf:.3f} "
          f"cfl={args.cfl} rhomin={args.rhomin} prot={vacuum} hardfloor={protect}")

    t0 = walltime.time()
    result = time_integration(initial_state, config, params, rv)
    states = result.states
    time_points = np.asarray(result.time_points)
    states.block_until_ready()
    print(f"[{args.tag}] wall {walltime.time()-t0:.1f}s, snaps={states.shape[0]}")

    di = rv.density_index
    vx_i, vy_i, vz_i = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z
    bx_i, by_i, bz_i = rv.magnetic_index.x, rv.magnetic_index.y, rv.magnetic_index.z

    # per-snapshot diagnostics
    nsny = states.shape[0]
    Ms_t = np.zeros(nsny); drho_t = np.zeros(nsny)
    EK_t = np.zeros(nsny); EB_t = np.zeros(nsny); vrms_t = np.zeros(nsny)
    spec_rho = []; spec_EK = []; spec_EB = []; kvec = None
    first_bad = -1
    for s in range(nsny):
        st = states[s]
        rho = st[di]; vx, vy, vz = st[vx_i], st[vy_i], st[vz_i]
        Bx, By, Bz_ = st[bx_i], st[by_i], st[bz_i]
        if first_bad < 0 and not bool(jnp.all(jnp.isfinite(st))):
            first_bad = s
        rho_pos = jnp.maximum(rho, 1e-12)
        v2 = vx**2 + vy**2 + vz**2
        vrms = float(jnp.sqrt(jnp.mean(jnp.nan_to_num(v2))))
        if adiabatic:
            P = jnp.maximum(st[rv.pressure_index], 1e-12)
            sound = float(jnp.mean(jnp.sqrt(args.gamma * P / rho_pos)))
        else:
            sound = a
        vrms_t[s] = vrms
        Ms_t[s] = vrms / sound
        drho_t[s] = float(jnp.sqrt(jnp.mean(jnp.nan_to_num((rho - rho0) ** 2))))
        EK_t[s] = float(jnp.mean(jnp.nan_to_num(0.5 * rho * v2)))
        EB_t[s] = float(jnp.mean(jnp.nan_to_num(0.5 * (Bx**2 + By**2 + Bz_**2))))
        rho_c = jnp.nan_to_num(rho); vx = jnp.nan_to_num(vx); vy = jnp.nan_to_num(vy); vz = jnp.nan_to_num(vz)
        Bx = jnp.nan_to_num(Bx); By = jnp.nan_to_num(By); Bz_ = jnp.nan_to_num(Bz_)
        k, Pr = vector_field_energy_spectrum(rho_c - jnp.mean(rho_c), jnp.zeros_like(rho_c), jnp.zeros_like(rho_c))
        _, Pk = get_kinetic_energy_spectrum(vx, vy, vz, jnp.maximum(rho_c, 0.0))
        _, Pb = get_magnetic_energy_spectrum(Bx, By, Bz_)
        kvec = np.asarray(k)
        spec_rho.append(np.asarray(Pr)); spec_EK.append(np.asarray(Pk)); spec_EB.append(np.asarray(Pb))
    spec_rho = np.array(spec_rho); spec_EK = np.array(spec_EK); spec_EB = np.array(spec_EB)

    t_over_tc = time_points / t_cross
    n3 = max(1, nsny // 3)
    print(f"[{args.tag}] stationary(last 1/3): M_turb={np.mean(Ms_t[-n3:]):.2f} "
          f"v_rms={np.mean(vrms_t[-n3:]):.3f} (rho-rho0)rms={np.mean(drho_t[-n3:]):.3f} "
          f"E_K={np.mean(EK_t[-n3:]):.3f} E_B={np.mean(EB_t[-n3:]):.3f} "
          f"first_bad_snap={first_bad}"
          + (f" @t/tc={t_over_tc[first_bad]:.2f}" if first_bad >= 0 else " (all finite)"))
    print(f"[{args.tag}] M_turb(t/tc): " +
          " ".join(f"{tc:.2f}:{m:.1f}" for tc, m in zip(t_over_tc, Ms_t)))

    # E_B / E_K mid-plane slices from the LAST ALIVE snapshot (so a late
    # floor-cascade collapse doesn't leave a vacuum slice)
    z = args.N // 2
    alive = np.where(vrms_t > 0.1)[0]
    sidx = int(alive[-1]) if len(alive) else nsny - 1
    fin = states[sidx]
    print(f"[{args.tag}] slice from snapshot {sidx} (t/tc={t_over_tc[sidx]:.2f})")
    rho = fin[di]; v2 = fin[vx_i]**2 + fin[vy_i]**2 + fin[vz_i]**2
    EK_slice = np.asarray((0.5 * rho * v2)[:, :, z])
    EB_slice = np.asarray((0.5 * (fin[bx_i]**2 + fin[by_i]**2 + fin[bz_i]**2))[:, :, z])
    rho_slice = np.asarray(rho[:, :, z])

    out = os.path.join(args.outdir, f"paper_{args.tag}.npz")
    np.savez(
        out, tag=args.tag, eos=args.eos, mturb_aim=args.mturb, beta=args.beta, gamma=args.gamma,
        N=args.N, a=a, B0=B0, F0=args.F0, kf=args.kf, tcross=args.tcross, protect=protect,
        time_points=time_points, t_over_tc=t_over_tc,
        Ms_t=Ms_t, vrms_t=vrms_t, drho_t=drho_t, EK_t=EK_t, EB_t=EB_t,
        k=kvec, spec_rho=spec_rho, spec_EK=spec_EK, spec_EB=spec_EB,
        EK_slice=EK_slice, EB_slice=EB_slice, rho_slice=rho_slice,
        EB0=float(0.5 * B0**2), first_bad=first_bad,
    )
    print(f"[{args.tag}] wrote {out}")


if __name__ == "__main__":
    main()
