"""Chevalier & Clegg (1985) central starburst wind, CGOLS-style.

Implements the supernova feedback model of the CGOLS suite
(Schneider & Robertson 2018, arXiv:1803.01008): mass and thermal energy
are continuously injected at uniform volumetric rates Mdot / V and
Edot / V into a spherical "gain" region of radius R (300 pc in the
paper) at the box center. Cells along the edge of the sphere are
weighted by their approximate overlap with the sphere so the correct
total injection rate is recovered at low resolution and grid effects
are alleviated. No net momentum is injected.

The rates follow the piecewise-linear schedule in
``params.cgols_wind_params`` (see ``cgols_wind_options.py``), which is
how the paper's low / ramp / high / ramp / low feedback history is
expressed. The schedule is evaluated at the start-of-step time; its
Myr-scale features are vastly longer than a hydro timestep, so the
first-order time sampling inside the RK stages is irrelevant.

With ``CGOLSWindConfig.injection_geometry = CLUSTER_SPHERES`` the same
total rates are instead shared among a set of star-cluster spheres (model C
of the suite, Schneider, Robertson & Thompson 2018, arXiv:1803.01005: eight
R = 150 pc clusters re-drawn every 15 Myr), see
:func:`_cluster_injection_profile`. The central-sphere path is unchanged by
that option.

Structured after ``_stellar_wind.stellar_wind._wind_ei3D_source``:
operates on the conserved state and returns a dt-scaled source term for
the finite-difference RK integrators.
"""

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# astronomix constants
from astronomix._modules._cgols_wind.cgols_wind_options import (
    CLUSTER_OVERLAP_SUM,
    CLUSTER_OVERLAP_UNION,
    CLUSTER_SPHERES,
)

# astronomix containers
from astronomix._modules._cgols_wind.cgols_wind_options import CGOLSWindParams
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cgols_wind_source(
    cgols_wind_params: CGOLSWindParams,
    conserved_state: STATE_TYPE,
    dt: Float[Array, ""],
    current_time: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """dt-scaled conserved-state source term for the CGOLS central wind.

    Args:
        cgols_wind_params: The CGOLS wind parameters (rates schedule, radius).
        conserved_state: The conserved state array.
        dt: The (stage-effective) time step.
        current_time: The current simulation time, used to evaluate the
            injection-rate schedule.
        config: The simulation configuration.
        helper_data: The helper data (uses ``r``, distance to box center).
        registered_variables: The registered variables.

    Returns:
        The source term to add to the conserved state.
    """

    if config.cgols_wind_config.injection_geometry == CLUSTER_SPHERES:
        return _cgols_cluster_source(
            cgols_wind_params,
            conserved_state,
            dt,
            current_time,
            config,
            helper_data,
            registered_variables,
        )

    source_term = jnp.zeros_like(conserved_state)

    mdot = jnp.interp(
        current_time,
        cgols_wind_params.schedule_times,
        cgols_wind_params.schedule_mass_rates,
    )
    edot = jnp.interp(
        current_time,
        cgols_wind_params.schedule_times,
        cgols_wind_params.schedule_energy_rates,
    )

    r_inj = cgols_wind_params.injection_radius
    dx = config.grid_spacing

    # Edge-cell weighting: 1 deep inside the sphere, 0 outside, with a
    # linear ramp over one cell width approximating the fractional volume
    # of the cell inside the sphere. Normalising V by sum(w) * dx^3 then
    # recovers the requested total rates exactly, independent of how the
    # sphere is resolved.
    weights = jnp.clip((r_inj + dx / 2 - helper_data.r) / dx, 0.0, 1.0)
    V = jnp.sum(weights) * dx**3

    # mass injection at uniform volumetric rate Mdot / V
    drho = mdot / V * dt * weights
    source_term = source_term.at[registered_variables.density_index].set(drho)

    # The injected gas carries no net momentum. Keeping the cell momentum
    # fixed while the density rises would drain kinetic energy
    # (E_kin = m^2 / (2 rho)), silently converting it to thermal energy on
    # top of the requested Edot. As in _wind_ei3D_source, scale the
    # momentum by sqrt(rho_new / rho_old) so the pre-existing kinetic
    # energy is untouched and exactly Edot enters as thermal energy.
    momentum_source_factor = (
        jnp.sqrt(1 + drho / conserved_state[registered_variables.density_index]) - 1.0
    )
    momentum_source_factor = jnp.where(weights > 0, momentum_source_factor, 0.0)

    source_term = source_term.at[registered_variables.momentum_index.x].set(
        conserved_state[registered_variables.momentum_index.x] * momentum_source_factor
    )
    source_term = source_term.at[registered_variables.momentum_index.y].set(
        conserved_state[registered_variables.momentum_index.y] * momentum_source_factor
    )
    source_term = source_term.at[registered_variables.momentum_index.z].set(
        conserved_state[registered_variables.momentum_index.z] * momentum_source_factor
    )

    # thermal energy injection at uniform volumetric rate Edot / V
    source_term = source_term.at[registered_variables.energy_index].set(
        edot / V * dt * weights
    )

    return source_term


def _cluster_injection_profile(
    cgols_wind_params: CGOLSWindParams,
    current_time: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
) -> Float[Array, "num_cells_x num_cells_y num_cells_z"]:
    """Spatial profile of the cluster feedback, normalised to unit integral.

    Returns ``S`` with ``sum(S) * dx^3 = 1`` whenever a cluster is alive, so a
    total rate ``Mdot`` enters as the volumetric rate ``Mdot * S``.

    Each cluster contributes the edge-weighted sphere of the central path,
    ``w_k = clip((R_c + dx / 2 - |x - x_k|) / dx, 0, 1)``, centred on its
    current position. With ``CLUSTER_OVERLAP_SUM`` every cluster is
    normalised on its own, ``S = sum_k f_k w_k / (sum(w_k) dx^3)`` with the
    living clusters' rate fractions ``f_k`` summing to one, so each cluster
    injects exactly its share wherever it sits and overlapping spheres add
    up. With ``CLUSTER_OVERLAP_UNION`` the union ``w = max_k w_k`` is one
    uniform region, ``S = w / (sum(w) dx^3)``, and the rate fractions only
    decide which slots are alive.

    The distances are built from the replicated 1D cell-center axes by
    broadcasting, so the profile is elementwise in the grid plus one global
    reduction per cluster: it shards without any gather or scatter.

    Args:
        cgols_wind_params: The CGOLS wind parameters (uses ``clusters``).
        current_time: The current simulation time; selects the living cluster
            of every slot and its orbital phase.
        config: The simulation configuration.
        helper_data: The helper data (uses ``cell_centers_x/y/z``).

    Returns:
        The injection profile on the (padded) grid of ``helper_data``.
    """

    if config.dimensionality != 3:
        raise ValueError("CGOLS cluster feedback requires a 3D Cartesian grid.")

    clusters = cgols_wind_params.clusters
    dx = config.grid_spacing
    cluster_radius = clusters.cluster_radius

    # ---- the living occupant of every slot ----
    # Occupant e of slot s lives from birth_times[e, s] until the birth of
    # occupant e + 1; before its first birth a slot is empty.
    num_born = jnp.sum(clusters.birth_times <= current_time, axis=0)
    alive = num_born > 0
    occupant = jnp.maximum(num_born - 1, 0)[None, :]

    def _current(per_occupant):
        return jnp.take_along_axis(per_occupant, occupant, axis=0)[0]

    birth_time = _current(clusters.birth_times)
    radius = _current(clusters.radii)
    azimuth = _current(clusters.azimuths) + _current(
        clusters.angular_velocities
    ) * (current_time - birth_time)
    height = _current(clusters.heights)
    rate_fraction = jnp.where(alive, _current(clusters.rate_fractions), 0.0)
    rate_fraction = rate_fraction / jnp.maximum(jnp.sum(rate_fraction), 1e-30)

    # Cluster positions in box coordinates. The box center is the origin of
    # the cylindrical cluster coordinates, as it is the center of the disk.
    center_x = config.box_size.x / 2 + radius * jnp.cos(azimuth)
    center_y = config.box_size.y / 2 + radius * jnp.sin(azimuth)
    center_z = config.box_size.z / 2 + height

    # ---- per-cluster edge-weighted spheres ----
    # Squared offsets per axis, (num_slots, N_axis) each; the 3D distance is
    # their broadcast sum, never a stored coordinate field.
    offset_x2 = (helper_data.cell_centers_x[None, :] - center_x[:, None]) ** 2
    offset_y2 = (helper_data.cell_centers_y[None, :] - center_y[:, None]) ** 2
    offset_z2 = (helper_data.cell_centers_z[None, :] - center_z[:, None]) ** 2

    def _sphere_weights(slot):
        distance = jnp.sqrt(
            offset_x2[slot][:, None, None]
            + offset_y2[slot][None, :, None]
            + offset_z2[slot][None, None, :]
        )
        return jnp.clip((cluster_radius + dx / 2 - distance) / dx, 0.0, 1.0)

    # The slot count is an array shape, hence static: the loops below unroll
    # into one fused elementwise pass and one reduction per cluster.
    num_slots = clusters.birth_times.shape[1]
    overlap = config.cgols_wind_config.cluster_overlap

    if overlap == CLUSTER_OVERLAP_SUM:
        profile = 0.0
        for slot in range(num_slots):
            weights = _sphere_weights(slot)
            volume = jnp.sum(weights) * dx**3
            # A cluster entirely off the grid (volume 0) injects nothing.
            profile = profile + rate_fraction[slot] / jnp.maximum(volume, 1e-30) * weights
        return profile

    if overlap == CLUSTER_OVERLAP_UNION:
        union = 0.0
        for slot in range(num_slots):
            union = jnp.maximum(
                union, jnp.where(alive[slot], _sphere_weights(slot), 0.0)
            )
        volume = jnp.sum(union) * dx**3
        return union / jnp.maximum(volume, 1e-30)

    raise ValueError(f"Unknown CGOLS cluster_overlap mode {overlap}.")


def _cgols_cluster_source(
    cgols_wind_params: CGOLSWindParams,
    conserved_state: STATE_TYPE,
    dt: Float[Array, ""],
    current_time: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """dt-scaled source term of the cluster feedback (CGOLS model C).

    The physics is that of the central sphere -- uniform volumetric mass and
    thermal-energy rates, no net momentum, the momentum rescaled to keep the
    kinetic energy -- with the uniform sphere replaced by the normalised
    cluster profile of :func:`_cluster_injection_profile`.

    Args:
        cgols_wind_params: The CGOLS wind parameters (rates schedule, clusters).
        conserved_state: The conserved state array.
        dt: The (stage-effective) time step.
        current_time: The current simulation time.
        config: The simulation configuration.
        helper_data: The helper data (uses ``cell_centers_x/y/z``).
        registered_variables: The registered variables.

    Returns:
        The source term to add to the conserved state.
    """

    mdot = jnp.interp(
        current_time,
        cgols_wind_params.schedule_times,
        cgols_wind_params.schedule_mass_rates,
    )
    edot = jnp.interp(
        current_time,
        cgols_wind_params.schedule_times,
        cgols_wind_params.schedule_energy_rates,
    )

    profile = _cluster_injection_profile(
        cgols_wind_params, current_time, config, helper_data
    )

    drho = mdot * dt * profile

    # Keep the pre-existing kinetic energy, see _cgols_wind_source.
    momentum_source_factor = jnp.where(
        profile > 0,
        jnp.sqrt(1 + drho / conserved_state[registered_variables.density_index])
        - 1.0,
        0.0,
    )

    # Assembled by stacking rather than by one .at[].set per variable; the
    # variables without a source (e.g. the dual-energy internal energy) stay
    # zero.
    sources = {
        registered_variables.density_index: drho,
        registered_variables.momentum_index.x: (
            conserved_state[registered_variables.momentum_index.x]
            * momentum_source_factor
        ),
        registered_variables.momentum_index.y: (
            conserved_state[registered_variables.momentum_index.y]
            * momentum_source_factor
        ),
        registered_variables.momentum_index.z: (
            conserved_state[registered_variables.momentum_index.z]
            * momentum_source_factor
        ),
        registered_variables.energy_index: edot * dt * profile,
    }
    zeros = jnp.zeros_like(drho)
    return jnp.stack(
        [sources.get(index, zeros) for index in range(conserved_state.shape[0])]
    )


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def cgols_wind_time_step_limit(
    cgols_wind_params: CGOLSWindParams,
    primitive_state: STATE_TYPE,
    current_time: Union[float, Float[Array, ""]],
    C_cfl: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> Float[Array, ""]:
    """Upper bound on dt from the heating of the injection region.

    Within one step a cell gains specific internal energy q dt with
    q = (Edot / V) w / rho, so its sound speed reaches at least
    sqrt(gamma (gamma - 1) q dt). Requiring that alone to satisfy the CFL
    condition, dt sqrt(gamma (gamma - 1) q dt) <= C_cfl dx, gives
    dt <= (C_cfl dx)^(2/3) / (gamma (gamma - 1) q_max)^(1/3).
    This only binds while the injection region is still cold (the step that
    switches the wind on, and the next few); once it is hot the ordinary CFL
    estimate is smaller. Returns +inf while the source is off.
    """
    edot = jnp.interp(
        current_time,
        cgols_wind_params.schedule_times,
        cgols_wind_params.schedule_energy_rates,
    )
    dx = config.grid_spacing
    if config.cgols_wind_config.injection_geometry == CLUSTER_SPHERES:
        # The same bound with the cluster profile in place of w / V: q peaks
        # in the densest gas of the most concentrated living cluster.
        profile = _cluster_injection_profile(
            cgols_wind_params, current_time, config, helper_data
        )
        rho = primitive_state[registered_variables.density_index]
        q_max = jnp.max(edot * profile / rho)
    else:
        weights = jnp.clip(
            (cgols_wind_params.injection_radius + dx / 2 - helper_data.r) / dx,
            0.0,
            1.0,
        )
        V = jnp.sum(weights) * dx**3
        rho = primitive_state[registered_variables.density_index]
        q_max = jnp.max(edot / V * weights / rho)
    dt = (C_cfl * dx) ** (2.0 / 3.0) / jnp.cbrt(gamma * (gamma - 1.0) * q_max)
    return jnp.where(q_max > 0, dt, jnp.inf)


"""Claude notes:
The CGOLS Chevalier & Clegg wind is implemented, wired into the time integrator, enabled in 
cgols.py, and verified with conservation tests. Here's a rundown.

The model (from Schneider & Robertson 2018, §2)

Mass and thermal energy are injected at uniform volumetric rates Ṁ/V and Ė/V into a 300 pc 
sphere at the box center, with edge cells weighted to recover the exact total rates. No net 
momentum is injected. The feedback history is: off for 0–5 Myr, low state (Ṁ=1.5 M⊙/yr, Ė=1.5×10⁴² erg/s) 
at 5 Myr, a 5 Myr linear ramp to the high state (Ṁ=12 M⊙/yr, Ė=5.4×10⁴² erg/s), high until 40 Myr, 
ramp down 40–45 Myr, low until 75 Myr.

What I built

New module astronomix/_modules/_cgols_wind/:
- cgols_wind_options.py — CGOLSWindConfig (just the on/off flag, static) and CGOLSWindParams 
(injection radius + a piecewise-linear rate schedule as schedule_times / schedule_mass_rates 
/ schedule_energy_rates knot arrays, evaluated with jnp.interp). The schedule representation 
handles the paper's whole off/low/ramp/high history and lives in params, so changing it never 
triggers recompilation.
- _cgols_wind.py — _cgols_wind_source, structured after _wind_ei3D_source: a dt-scaled 
conserved-state source using helper_data.r (distance to box center) with a one-cell linear edge
taper normalized so V = Σw·dx³ recovers the requested rates exactly. Like the stellar wind, momentum
is rescaled by √(ρ_new/ρ_old) so the added mass doesn't silently drain kinetic energy and 
exactly Ė enters as thermal.

Time plumbing — the schedule needs the simulation time, which didn't reach the source terms. 
I passed the loop's start-of-step time through _evolve_state_fd → all four FD integrators in 
_ssprk.py → _time_integrator_sources (new trailing current_time arg, default 0.0, so the FV 
call sites are untouched). Start-of-step sampling is first-order, but dt (~kyr) vs. the 5 Myr
ramps makes that error ~10⁻⁴.

Wiring — cgols_wind_config / cgols_wind_params fields on SimulationConfig / SimulationParams; 
the source hooked in at your # if cgols wind insort wind to source here comment (FD-gated); 
helper-data requirements now materialize r when the wind is on.

cgols.py — build_cgols_wind_params() converts the paper's rates to code units (Ṁ_low = 14.67, 
Ṁ_high = 117.3 code mass/time; Ė_low = 2328,
Ė_high = 8380 code energy/time — well-scaled for floatedule; the config enables the wind and 
both paramsconstructions (build + load paths) carry it.

Verification

- Uniform periodic box on CPU, both RK4_LSRK and SSPRK4: injected mass and energy match rate × 
t_end to ~10⁻⁵ relative; a ramped schedule
integrates correctly within the expected sampling erro
- shock_tube1D.py (FV+FD) and a small CP Alfvén wave run through both MHD-CT integrators still 
pass, so the plumbing didn't break existing
paths.
- Sanity check: Ė/Ṁ gives a central injection-region temperature ~2×10⁷ K, matching the paper's hot wind.

Two side notes: at your 512²×1024 resolution the sphere is ~15 cells in radius, so it's well resolved;
and running the tests left a harmless figures/shock_tube1D_test.svg artifact you mato watch on the real 
run: the hot injection regionraises the sound speed substantially, so expect the CFL timestep to drop 
once feedback switches on at 5 Myr.
"""
