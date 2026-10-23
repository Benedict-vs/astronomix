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

The "smooth ..." sets (CGOLS_FEEDBACK=smooth, model C's N -> infinity limit)
instead compare every cell's mass, momentum and energy source with an
independent numpy evaluation of Sigma(R) h(z) and v_c(R), and check by
sampling that the tables are the distributions the cluster centres are drawn
from (Kolmogorov-Smirnov distance of 2e5 draws).

Usage:
    JAX_PLATFORMS=cpu python cgols_model_c_check.py           # all option sets
    JAX_PLATFORMS=cpu python cgols_model_c_check.py --child   # current env only
    MODEL_C_CHECK_ONLY=smooth python cgols_model_c_check.py   # sets whose label matches

cgols.py calls autocvd, which waits for a free GPU even on the CPU: put a no-op
autocvd module first on PYTHONPATH when the GPUs are busy.
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
    "smooth (default)": {"CGOLS_FEEDBACK": "smooth"},
    "smooth, static": {"CGOLS_FEEDBACK": "smooth", "CGOLS_CLUSTER_COROTATE": "0"},
    "smooth, scale": {"CGOLS_FEEDBACK": "smooth", "CGOLS_CLUSTER_MOMENTUM": "scale"},
    "smooth, rest": {"CGOLS_FEEDBACK": "smooth", "CGOLS_CLUSTER_MOMENTUM": "rest"},
    "smooth m82": {"CGOLS_FEEDBACK": "smooth", "CGOLS_CLUSTER_PRESET": "m82"},
    "smooth area, exp z": {
        "CGOLS_FEEDBACK": "smooth",
        "CGOLS_CLUSTER_RADIAL": "uniform_area",
        "CGOLS_CLUSTER_VERTICAL": "exponential",
    },
    "smooth uniform z": {"CGOLS_FEEDBACK": "smooth", "CGOLS_CLUSTER_VERTICAL": "uniform"},
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
    if cgols.SMOOTH:
        worst = _check_smooth_tables(cgols, wind_params.smooth_disk)
        disk = wind_params.smooth_disk
        x = np.asarray(helper_data.cell_centers_x, np.float64) - config.box_size.x / 2
        y = np.asarray(helper_data.cell_centers_y, np.float64) - config.box_size.y / 2
        z = np.asarray(helper_data.cell_centers_z, np.float64) - config.box_size.z / 2
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        R = np.hypot(X, Y)
        knots = np.asarray(disk.radii, np.float64)
        hk = np.asarray(disk.heights, np.float64)
        # Cell averages over a 4 x 4 x 8 sub-grid, written out point by point.
        weight = np.zeros_like(R)
        for ox in (np.arange(4) + 0.5) / 4 - 0.5:
            for oy in (np.arange(4) + 0.5) / 4 - 0.5:
                Rs = np.hypot(X[:, :, :1] + ox * dx, Y[:, :, :1] + oy * dx)
                sigma = np.where(Rs <= knots[-1], np.interp(Rs, knots, np.asarray(disk.surface_density)), 0.0)
                for oz in (np.arange(8) + 0.5) / 8 - 0.5:
                    Zs = Z[:1, :1, :] + oz * dx
                    h = np.where(np.abs(Zs) <= hk[-1], np.interp(Zs, hk, np.asarray(disk.vertical_profile)), 0.0)
                    weight += sigma * h / 128
        smooth_profile = weight / (weight.sum() * dx**3)
        v_c = np.interp(R, knots, np.asarray(disk.circular_velocity))
        smooth_vx = -v_c * Y / np.maximum(R, 1e-30)
        smooth_vy = v_c * X / np.maximum(R, 1e-30)
    for t_myr in (4.0, 5.0, 7.3, 21.0, 37.0, 52.0, 70.0):
        t = t_myr * myr
        source = _cgols_wind_source(wind_params, state, dt, t, config, helper_data, rv)
        mdot = float(jnp.interp(t, wind_params.schedule_times, wind_params.schedule_mass_rates))
        edot = float(jnp.interp(t, wind_params.schedule_times, wind_params.schedule_energy_rates))
        dm = np.sum(np.asarray(source[rv.density_index], np.float64)) * dx**3
        de = np.sum(np.asarray(source[rv.energy_index], np.float64)) * dx**3
        rel_m = abs(dm / (mdot * dt) - 1) if mdot > 0 else abs(dm)
        if cgols.SMOOTH:
            # Cell by cell against the numpy profile, relative to its peak.
            src = {k: np.asarray(source[k], np.float64) for k in range(rv.num_vars)}
            peak = max(mdot * dt * smooth_profile.max(), 1e-300)
            rel_m = max(rel_m, float(np.max(np.abs(src[rv.density_index] - mdot * dt * smooth_profile))) / peak)
            if cgols.CLUSTER_MOMENTUM == "cluster":
                v_scale = max(np.max(np.abs(v_c)), 1.0)  # code velocity, 100 km/s
                dke = max(
                    float(np.max(np.abs(src[rv.momentum_index.x] - mdot * dt * smooth_profile * smooth_vx))),
                    float(np.max(np.abs(src[rv.momentum_index.y] - mdot * dt * smooth_profile * smooth_vy))),
                    float(np.max(np.abs(src[rv.momentum_index.z]))),
                ) / (peak * v_scale)
                expected = edot * dt * smooth_profile + mdot * dt * smooth_profile * 0.5 * v_c**2
                rel_e = float(np.max(np.abs(src[rv.energy_index] - expected))) / max(expected.max(), 1e-300)
                print(f"    <v_c^2/2>_S / (Edot/Mdot) = {np.sum(smooth_profile * 0.5 * v_c**2) * dx**3 * mdot / max(edot, 1e-30):.2e}")
            else:
                expected = edot * dt * smooth_profile
                rel_e = abs(de / (edot * dt) - 1) if edot > 0 else abs(de)
                rel_e = max(rel_e, float(np.max(np.abs(src[rv.energy_index] - expected))) / max(expected.max(), 1e-300))
        elif cgols.CLUSTER_MOMENTUM == "cluster" and cgols.CLUSTER_OVERLAP == "union":
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


def _check_smooth_tables(cgols, disk, num_draws=200_000):
    """KS distance between the smooth tables and the cluster-centre draws."""
    import numpy as np
    from astronomix._modules._cgols_wind.cluster_placement import (
        ClusterPlacementSettings,
        _draw_height,
        _draw_radius,
    )

    pc = (1 * cgols.u.pc).to(cgols.code_units.code_length).value
    settings = ClusterPlacementSettings(
        radial_profile=cgols.CLUSTER_RADIAL,
        radial_scale_length=cgols.cluster_radial_scale_pc() * pc,
        max_radius=cgols.CLUSTER_RMAX_PC * pc,
        vertical_profile=cgols.CLUSTER_VERTICAL,
        vertical_scale_height=cgols.CLUSTER_ZSCALE_PC * pc,
        max_height=cgols.CLUSTER_ZMAX_PC * pc,
    )
    rng = np.random.default_rng(1)
    radii = np.sort([_draw_radius(rng, settings) for _ in range(num_draws)])
    heights = np.sort([_draw_height(rng, settings) for _ in range(num_draws)])

    def ks(samples, knots, density):
        # density per unit of the sampled coordinate -> CDF on the knots
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(knots))])
        cdf /= cdf[-1]
        empirical = np.arange(1, len(samples) + 1) / len(samples)
        return float(np.max(np.abs(np.interp(samples, knots, cdf) - empirical)))

    knots = np.asarray(disk.radii, np.float64)
    d_r = ks(radii, knots, knots * np.asarray(disk.surface_density, np.float64))  # dN/dR = 2 pi R Sigma
    d_z = ks(heights, np.asarray(disk.heights, np.float64), np.asarray(disk.vertical_profile, np.float64))
    # 99.9% KS critical value for n = 2e5 is ~1.95 / sqrt(n) = 4.4e-3.
    print(f"    KS distance tables vs cluster draws: R {d_r:.1e}, z {d_z:.1e}")
    return 0.0 if max(d_r, d_z) < 1.95 / np.sqrt(num_draws) else 1.0


def main():
    failures = []
    only = os.environ.get("MODEL_C_CHECK_ONLY", "")
    for label, env in OPTION_SETS.items():
        if only and only not in label:
            continue
        child_env = dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            CGOLS_FEEDBACK="clusters",
            CGOLS_DIM="64",
            CGOLS_SHARD_SPLIT="(1, 1, 1, 1)",
            CGOLS_RUN_TAG="_model_c_check",
        )
        child_env.update(env)
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
