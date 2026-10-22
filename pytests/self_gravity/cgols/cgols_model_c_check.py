"""Smoke test of the CGOLS model-C cluster feedback, every option (CPU, seconds).

For each CGOLS_CLUSTER_* option set below, a child process builds cgols.py's
config and model-C params at CGOLS_DIM=64 and evaluates the wind source on a
random padded state at several times across the schedule, checking that

  - the injected mass and energy are exactly Mdot dt and Edot dt (summed over
    the cells; overlapping clusters included), to float32 rounding;
  - the kinetic energy of every cell is unchanged (momentum rescaling), or,
    with CGOLS_CLUSTER_MOMENTUM=rest, the momentum is unchanged, or, with
    CGOLS_CLUSTER_MOMENTUM=cluster (ejecta moving with their cluster), the
    injected momentum is Mdot dt <v_c> and the injected energy
    (Edot + Mdot <|v_c|^2> / 2) dt, from the host-side cluster positions and
    velocities, and no z momentum is injected;
  - no other variable receives a source;
  - the wind-onset dt limit is finite while the source is on (+inf before).

Usage:
    JAX_PLATFORMS=cpu python cgols_model_c_check.py           # all option sets
    JAX_PLATFORMS=cpu python cgols_model_c_check.py --child   # current env only
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

OPTION_SETS = {
    "default (physical)": {},
    "preset cholla": {"CGOLS_CLUSTER_PRESET": "cholla"},
    "random": {"CGOLS_CLUSTER_PLACEMENT": "random"},
    "uniform_area": {"CGOLS_CLUSTER_PLACEMENT": "random", "CGOLS_CLUSTER_RADIAL": "uniform_area"},
    "exponential R": {
        "CGOLS_CLUSTER_PLACEMENT": "random",
        "CGOLS_CLUSTER_RADIAL": "exponential",
        "CGOLS_CLUSTER_RMAX_PC": "4500",
    },
    "gaussian z": {
        "CGOLS_CLUSTER_PLACEMENT": "random",
        "CGOLS_CLUSTER_VERTICAL": "gaussian",
        "CGOLS_CLUSTER_ZMAX_PC": "150",
    },
    "exponential z": {
        "CGOLS_CLUSTER_PLACEMENT": "random",
        "CGOLS_CLUSTER_VERTICAL": "exponential",
        "CGOLS_CLUSTER_ZMAX_PC": "150",
    },
    "midplane": {"CGOLS_CLUSTER_PLACEMENT": "random", "CGOLS_CLUSTER_VERTICAL": "midplane"},
    "staggered 40 Myr": {
        "CGOLS_CLUSTER_PLACEMENT": "random",
        "CGOLS_CLUSTER_REDRAW": "staggered",
        "CGOLS_CLUSTER_LIFETIME_MYR": "40",
    },
    "union": {"CGOLS_CLUSTER_OVERLAP": "union", "CGOLS_CLUSTER_MOMENTUM": "scale"},
    "min separation": {"CGOLS_CLUSTER_PLACEMENT": "random", "CGOLS_CLUSTER_MIN_SEP_PC": "300"},
    "corotate": {"CGOLS_CLUSTER_PLACEMENT": "random", "CGOLS_CLUSTER_COROTATE": "1"},
    "power-law masses": {
        "CGOLS_CLUSTER_PLACEMENT": "random",
        "CGOLS_CLUSTER_MASS_FUNCTION": "power_law",
    },
    "27 clusters": {"CGOLS_CLUSTER_PLACEMENT": "random", "CGOLS_NUM_CLUSTERS": "27"},
    "preset paper_text": {"CGOLS_CLUSTER_PRESET": "paper_text"},
    "preset physical": {"CGOLS_CLUSTER_PRESET": "physical"},
    "preset m82": {"CGOLS_CLUSTER_PRESET": "m82"},
    "momentum rest": {"CGOLS_CLUSTER_MOMENTUM": "rest"},
    "momentum scale": {"CGOLS_CLUSTER_MOMENTUM": "scale"},
    "cluster, static": {"CGOLS_CLUSTER_COROTATE": "0"},
    "cluster + union": {"CGOLS_CLUSTER_OVERLAP": "union"},
    "cholla + cluster": {"CGOLS_CLUSTER_PRESET": "cholla", "CGOLS_CLUSTER_MOMENTUM": "cluster", "CGOLS_CLUSTER_COROTATE": "1"},
    "cluster + power law": {"CGOLS_CLUSTER_MASS_FUNCTION": "power_law"},
}


def child():
    """Check the source for the option set in the environment."""
    import numpy as np
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, HERE)
    import cgols
    from astronomix.option_classes.simulation_config import finalize_config
    from astronomix.data_classes.simulation_helper_data import (
        _helper_data_requirements,
        get_helper_data,
    )
    from astronomix._modules._cgols_wind._cgols_wind import (
        _cgols_wind_source,
        cgols_wind_time_step_limit,
    )
    from astronomix._modules._cgols_wind.cluster_placement import cluster_positions

    def _cluster_velocities(clusters, time):
        # Orbital velocities of the living clusters, in the order of
        # cluster_positions.
        births = np.asarray(clusters.birth_times)
        num_born = np.sum(births <= time, axis=0)
        alive = num_born > 0
        occupant = np.maximum(num_born - 1, 0)
        slots = np.arange(births.shape[1])
        pick = lambda a: np.asarray(a)[occupant, slots]
        phase = pick(clusters.azimuths) + pick(clusters.angular_velocities) * (
            time - pick(clusters.birth_times)
        )
        speed = pick(clusters.radii) * pick(clusters.angular_velocities)
        return np.stack([-speed * np.sin(phase), speed * np.cos(phase), 0 * speed], axis=-1)[alive]

    config, rv = cgols.build_config()
    shape = (rv.num_vars, cgols.RESOLUTION, cgols.RESOLUTION, 2 * cgols.RESOLUTION)
    config = finalize_config(config, shape)
    helper_data = get_helper_data(
        config, padded=True, requirements=_helper_data_requirements(config)
    )
    wind_params = cgols.build_cgols_wind_params()
    dx = config.grid_spacing
    g = config.num_ghost_cells
    padded = (rv.num_vars,) + tuple(n + 2 * g for n in shape[1:])

    momenta = slice(rv.momentum_index.x, rv.momentum_index.z + 1)
    key_state, key_momentum = jax.random.split(jax.random.PRNGKey(0))
    state = jnp.abs(jax.random.normal(key_state, padded)) + 0.5
    state = state.at[momenta].set(jax.random.normal(key_momentum, (3,) + padded[1:]))
    sourced = {rv.density_index, rv.energy_index, *range(momenta.start, momenta.stop)}

    def kinetic(s):
        return 0.5 * jnp.sum(s[momenta] ** 2, axis=0) / s[rv.density_index]

    dt = 1e-4
    myr = (1 * cgols.u.Myr).to(cgols.code_units.code_time).value
    worst = 0.0
    for t_myr in (4.0, 5.0, 7.3, 21.0, 37.0, 52.0, 70.0):
        t = t_myr * myr
        source = _cgols_wind_source(wind_params, state, dt, t, config, helper_data, rv)
        mdot = float(jnp.interp(t, wind_params.schedule_times, wind_params.schedule_mass_rates))
        edot = float(jnp.interp(t, wind_params.schedule_times, wind_params.schedule_energy_rates))
        dm = np.sum(np.asarray(source[rv.density_index], np.float64)) * dx**3
        de = np.sum(np.asarray(source[rv.energy_index], np.float64)) * dx**3
        rel_m = abs(dm / (mdot * dt) - 1) if mdot > 0 else abs(dm)
        if cgols.CLUSTER_MOMENTUM == "cluster" and cgols.CLUSTER_OVERLAP == "union":
            # One union region with the weight-averaged velocity per cell: the
            # energy is Edot dt plus the kinetic energy of the injected
            # momentum, cell by cell, and the momentum direction is that of
            # the clusters (bounded by the fastest one).
            src = {k: np.asarray(source[k], np.float64) for k in range(rv.num_vars)}
            drho = src[rv.density_index]
            px, py = src[rv.momentum_index.x], src[rv.momentum_index.y]
            ke = np.where(drho > 0, 0.5 * (px**2 + py**2) / np.where(drho > 0, drho, 1.0), 0.0)
            expected_e = edot * dt + np.sum(ke) * dx**3
            rel_e = abs(de / expected_e - 1) if expected_e > 0 else abs(de)
            v_c = _cluster_velocities(wind_params.clusters, t)
            v_max = np.max(np.linalg.norm(v_c, axis=1)) if len(v_c) else 0.0
            v_cell = np.sqrt(2 * ke / np.where(drho > 0, drho, 1.0))
            dke = max(float(np.max(v_cell)) / max(v_max, 1e-30) - 1.0, 0.0)
            dke = max(dke, float(np.max(np.abs(src[rv.momentum_index.z]))))
        elif cgols.CLUSTER_MOMENTUM == "cluster":
            # Share-weighted cluster velocity and |v|^2 / 2 (host side): the
            # orbital velocity Omega R (-sin phi, cos phi, 0) of every living
            # cluster. Every cluster is entirely on the grid here.
            xyz, shares = cluster_positions(wind_params.clusters, t)
            v_c = _cluster_velocities(wind_params.clusters, t)
            mean_v = np.sum(shares[:, None] * v_c, axis=0)
            mean_ke = np.sum(shares * 0.5 * np.sum(v_c**2, axis=1))
            expected_e = (edot + mdot * mean_ke) * dt
            rel_e = abs(de / expected_e - 1) if expected_e > 0 else abs(de)
            dp = np.array(
                [np.sum(np.asarray(source[i], np.float64)) * dx**3 for i in range(momenta.start, momenta.stop)]
            )
            # Relative to the scale of the expected momentum (the mean can
            # nearly cancel among clusters on opposite sides).
            p_scale = mdot * dt * max(np.sum(shares * np.linalg.norm(v_c, axis=1)), 1e-30)
            dke = float(np.max(np.abs(dp - mdot * dt * mean_v))) / p_scale if mdot > 0 else float(np.max(np.abs(dp)))
            print(
                f"    momentum budget: sum d(rho v) = {dp}, Mdot dt <v_c> = {mdot * dt * mean_v}, "
                f"<|v_c|^2>/2 / (Edot/Mdot) = {mean_ke * mdot / max(edot, 1e-30):.2e}"
            )
        else:
            rel_e = abs(de / (edot * dt) - 1) if edot > 0 else abs(de)
        if cgols.CLUSTER_MOMENTUM == "cluster":
            pass
        elif cgols.CLUSTER_MOMENTUM == "rest":
            # Mass added at rest: no momentum source at all.
            dke = float(jnp.max(jnp.abs(source[momenta])))
        else:
            dke = float(jnp.max(jnp.abs(kinetic(state + source) / kinetic(state) - 1)))
        other = max(
            [float(jnp.max(jnp.abs(source[i]))) for i in range(rv.num_vars) if i not in sourced]
            or [0.0]
        )
        dt_limit = float(
            cgols_wind_time_step_limit(
                wind_params, state, t, 0.9, 5 / 3, config, helper_data, rv
            )
        )
        ok_limit = np.isinf(dt_limit) if edot == 0 else np.isfinite(dt_limit)
        worst = max(worst, rel_m, rel_e, dke if dke > 1e-5 else 0.0, other, 0.0 if ok_limit else 1.0)
        print(
            f"  t = {t_myr:5.1f} Myr: |dM/(Mdot dt) - 1| = {rel_m:.1e}, "
            f"|dE/(Edot dt) - 1| = {rel_e:.1e}, max |dKE/KE| (rest: |dm|) = {dke:.1e}, "
            f"other vars {other:.0e}, dt limit {dt_limit:.2e}"
        )
    print(f"RESULT worst {worst:.2e} {'PASS' if worst < 1e-5 else 'FAIL'}")


def main():
    failures = []
    for label, env in OPTION_SETS.items():
        child_env = dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            CGOLS_FEEDBACK="clusters",
            CGOLS_DIM="64",
            CGOLS_SHARD_SPLIT="(1, 1, 1, 1)",
            CGOLS_RUN_TAG="_model_c_check",
            **env,
        )
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child"],
            env=child_env, capture_output=True, text=True,
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.startswith(("  t =", "RESULT"))]
        status = lines[-1] if lines and lines[-1].startswith("RESULT") else "RESULT crashed FAIL"
        print(f"{label:20s} {status}")
        if "PASS" not in status:
            failures.append(label)
            print(result.stdout[-3000:], result.stderr[-3000:])
    print("ALL PASS" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
