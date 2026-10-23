"""
Configuration and parameter containers for the CGOLS starburst feedback.

Defines the integer tags that select the injection geometry (one central
sphere, CGOLS models A/B, a set of star-cluster spheres, CGOLS model C, or
the smooth disk profile that model C tends to for infinitely many clusters)
and the overlap treatment of the cluster spheres, together with the static
:class:`CGOLSWindConfig` and the dynamic :class:`CGOLSWindParams` /
:class:`CGOLSClusterParams` / :class:`CGOLSSmoothDiskParams`.
"""

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp

# Injection-geometry tags.
CENTRAL_SPHERE = 0  # one sphere at the box center (CGOLS models A and B)
CLUSTER_SPHERES = 1  # several star-cluster spheres (CGOLS model C)
#: A static axisymmetric profile Sigma(R) h(z) (the N -> infinity limit of the
#: cluster spheres at fixed total volume, see CGOLSSmoothDiskParams).
SMOOTH_DISK = 2

# How overlapping cluster spheres share a cell.
#: Every cluster injects exactly its own share of the total rate into its own
#: sphere; where two spheres overlap the two contributions add up, as two
#: physically distinct sources would.
CLUSTER_OVERLAP_SUM = 0
#: The union of all spheres is one injection region with a uniform volumetric
#: rate; overlapping clusters merge into a single, larger source.
CLUSTER_OVERLAP_UNION = 1

# Momentum of the mass injected by the clusters.
#: Rescale the cell momentum by sqrt(rho_new / rho_old), keeping the kinetic
#: energy, so exactly Edot enters as heat (the central-sphere convention).
CLUSTER_MOMENTUM_SCALE = 0
#: The injected mass carries no momentum (Cholla): the cell momentum is kept,
#: the gas slows down and its lost kinetic energy is thermalised on top of Edot.
CLUSTER_MOMENTUM_REST = 1
#: The injected mass moves with its cluster (the ejecta leave the stars with the
#: stars' velocity): momentum += drho * v_cluster and energy += Edot dt +
#: drho |v_cluster|^2 / 2, i.e. mass added at rest plus Edot as heat in the
#: cluster's own frame, Galilean-boosted to the box frame. The mixing of the
#: ejecta with gas moving at a different velocity thermalises
#: rho drho / (rho + drho) |v_gas - v_cluster|^2 / 2, as an inelastic collision
#: does. For clusters at rest this is CLUSTER_MOMENTUM_REST.
CLUSTER_MOMENTUM_CLUSTER = 2


class CGOLSWindConfig(NamedTuple):
    """Static configuration of the CGOLS-style starburst feedback.

    Mass and thermal energy are injected at (time-dependent) constant
    volumetric rates into a sphere at the box center, following the
    Chevalier & Clegg (1985) model as implemented in the CGOLS suite
    (Schneider & Robertson 2018, arXiv:1803.01008), or into a set of
    star-cluster spheres (model C of the suite, Schneider, Robertson &
    Thompson 2018, arXiv:1803.01005). Finite-difference solver mode only.
    """

    cgols_wind: bool = False

    #: Also bound the hydro time step by the heating of the injection region
    #: (see ``_cgols_wind.cgols_wind_time_step_limit``). The CFL estimate only
    #: sees the start-of-step state, so on the step where the source switches
    #: on it is set by the cold pre-onset gas while the step itself heats the
    #: region by orders of magnitude.
    timestep_limit: bool = False

    #: ``CENTRAL_SPHERE`` (one sphere of ``injection_radius`` at the box center),
    #: ``CLUSTER_SPHERES`` (the spheres of ``cgols_wind_params.clusters``) or
    #: ``SMOOTH_DISK`` (the profile of ``cgols_wind_params.smooth_disk``); the
    #: last two are 3D Cartesian only.
    injection_geometry: int = CENTRAL_SPHERE

    #: Overlap treatment of the cluster spheres, ``CLUSTER_OVERLAP_SUM`` or
    #: ``CLUSTER_OVERLAP_UNION``. Ignored for the central sphere.
    cluster_overlap: int = CLUSTER_OVERLAP_SUM

    #: Momentum of the injected mass, ``CLUSTER_MOMENTUM_SCALE``,
    #: ``CLUSTER_MOMENTUM_REST`` or ``CLUSTER_MOMENTUM_CLUSTER`` (for the smooth
    #: disk: the ejecta move with ``smooth_disk.circular_velocity``). Ignored
    #: for the central sphere.
    cluster_momentum: int = CLUSTER_MOMENTUM_SCALE


class CGOLSClusterParams(NamedTuple):
    """Positions and lifetimes of the feedback clusters, all in code units.

    The clusters are organised in ``num_slots`` slots, each of which hosts one
    cluster at a time: occupant ``e`` of slot ``s`` is born at
    ``birth_times[e, s]`` and is replaced by occupant ``e + 1`` at
    ``birth_times[e + 1, s]`` (the last occupant lives forever). Before the
    first birth time a slot is empty. Simultaneous re-draws (CGOLS model C:
    all eight clusters every 15 Myr) give every slot the same birth times;
    staggered re-draws offset them per slot.

    A cluster sits at cylindrical radius ``radii``, azimuth ``azimuths`` and
    height ``heights`` relative to the box center. It orbits the z axis at
    the angular velocity ``angular_velocities`` (zero for the static clusters
    of the paper), so its azimuth at time ``t`` is
    ``azimuths + angular_velocities * (t - birth_times)``.

    The total feedback rate of the schedule in :class:`CGOLSWindParams` is
    shared among the clusters alive at time ``t`` in proportion to their
    ``rate_fractions`` (renormalised over the living clusters, so the total
    is always exactly the schedule's rate).

    All per-cluster arrays have the shape ``(num_occupants, num_slots)``.
    """

    #: Radius of each cluster's injection sphere, code length. CGOLS model C
    #: uses 150 pc, so that the eight spheres add up to the volume of the
    #: 300 pc sphere of models A and B.
    cluster_radius: float = 0.15

    #: Time at which each occupant of each slot becomes active, code time.
    #: Non-decreasing along the first axis.
    birth_times: jnp.ndarray = jnp.zeros((1, 1))

    #: Cylindrical radius of each cluster relative to the box center.
    radii: jnp.ndarray = jnp.zeros((1, 1))

    #: Azimuth of each cluster at its birth time, radians.
    azimuths: jnp.ndarray = jnp.zeros((1, 1))

    #: Height of each cluster above the box mid-plane.
    heights: jnp.ndarray = jnp.zeros((1, 1))

    #: Angular velocity of each cluster about the z axis, radians / code time.
    angular_velocities: jnp.ndarray = jnp.zeros((1, 1))

    #: Relative share of the total feedback rate, e.g. proportional to the
    #: cluster mass. Equal for CGOLS model C.
    rate_fractions: jnp.ndarray = jnp.ones((1, 1))


class CGOLSSmoothDiskParams(NamedTuple):
    """Smooth axisymmetric feedback profile, all in code units.

    The injection is distributed as ``Sigma(R) h(z)`` about the box center
    (``R`` cylindrical radius, ``z`` height), both given as tables that are
    linearly interpolated and zero outside their range; the profile is
    renormalised on the grid, so its overall scale is irrelevant. The rates
    of the schedule in :class:`CGOLSWindParams` enter as volumetric rates
    ``Mdot S`` with ``sum(S) dx^3 = 1``.

    This is the limit of the cluster spheres for N -> infinity at fixed total
    volume (R_c = R_0 / N^(1/3) -> 0): the time-averaged cluster injection is
    the distribution of the cluster centres smoothed by one sphere, and its
    fluctuations about that mean fall as 1 / sqrt(N). Drawing the centres
    from a surface density ``Sigma`` and a height distribution ``h`` thus
    tends to exactly this profile. It is static: rotation leaves an
    axisymmetric profile unchanged and only enters through the momentum of
    the ejecta, ``v = v_c(R) phi_hat`` from the ``circular_velocity`` table.
    """

    #: Radius knots of ``surface_density`` and ``circular_velocity``.
    radii: jnp.ndarray = jnp.array([0.0, 1.0])

    #: Relative surface density of the injection at ``radii``.
    surface_density: jnp.ndarray = jnp.array([0.0, 0.0])

    #: Azimuthal velocity of the ejecta at ``radii`` (momentum mode
    #: ``CLUSTER_MOMENTUM_CLUSTER``), code velocity.
    circular_velocity: jnp.ndarray = jnp.array([0.0, 0.0])

    #: Height knots of ``vertical_profile`` (signed, relative to the mid-plane).
    heights: jnp.ndarray = jnp.array([-1.0, 1.0])

    #: Relative vertical distribution of the injection at ``heights``.
    vertical_profile: jnp.ndarray = jnp.array([0.0, 0.0])


class CGOLSWindParams(NamedTuple):
    """Dynamic parameters of the CGOLS wind, all in code units.

    The injection rates follow a piecewise-linear schedule: at simulation
    time t the rates are ``jnp.interp(t, schedule_times, schedule_*_rates)``
    (clamped to the first/last value outside the knots). The CGOLS
    low -> ramp -> high -> ramp -> low history is expressed by choosing the
    knots accordingly; duplicate-adjacent knots give step changes.
    """

    #: Radius of the spherical injection ("gain") region, code length.
    #: CGOLS uses 300 pc. Central-sphere geometry only.
    injection_radius: float = 0.3 # since code units in cgols.py is 1 * u.kiloparsec

    #: Schedule knots, code time, must be non-decreasing.
    schedule_times: jnp.ndarray = jnp.array([0.0])

    #: Mass injection rate at each knot, code mass / code time.
    schedule_mass_rates: jnp.ndarray = jnp.array([0.0])

    #: Thermal energy injection rate at each knot, code energy / code time.
    schedule_energy_rates: jnp.ndarray = jnp.array([0.0])

    #: The feedback clusters, cluster-sphere geometry only.
    clusters: CGOLSClusterParams = CGOLSClusterParams()

    #: The smooth injection profile, smooth-disk geometry only.
    smooth_disk: CGOLSSmoothDiskParams = CGOLSSmoothDiskParams()
