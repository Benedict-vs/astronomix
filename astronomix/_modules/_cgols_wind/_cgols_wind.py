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

Structured after ``_stellar_wind.stellar_wind._wind_ei3D_source``:
operates on the conserved state and returns a dt-scaled source term for
the finite-difference RK integrators.
"""

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

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
