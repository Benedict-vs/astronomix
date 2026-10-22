"""
Host-side placement of the CGOLS feedback clusters.

Draws the positions, birth times and rate shares of the star clusters of the
CGOLS cluster feedback (model C of Schneider, Robertson & Thompson 2018,
arXiv:1803.01005) with a seeded numpy generator and packs them into a
:class:`CGOLSClusterParams`. Everything here runs once, before the
simulation, so the jitted solver only ever sees fixed arrays and a run is
reproducible from its seed. :func:`cluster_positions` evaluates the same
schedule on the host (for plots and diagnostics), mirroring the selection in
``_cgols_wind._cluster_injection_profile``.
"""

# general
import ctypes

# typing
from typing import Callable, NamedTuple, Optional

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix containers
from astronomix._modules._cgols_wind.cgols_wind_options import CGOLSClusterParams

# Radial distribution of the cluster positions (surface density of clusters).
RADIAL_UNIFORM_AREA = "uniform_area"  # constant surface density inside max_radius
RADIAL_UNIFORM_RADIUS = "uniform_radius"  # constant dN/dR, i.e. centrally concentrated
RADIAL_EXPONENTIAL = "exponential"  # exp(-R / scale_length), truncated at max_radius

# Vertical distribution of the cluster positions.
VERTICAL_UNIFORM = "uniform"  # uniform in [-max_height, max_height]
VERTICAL_GAUSSIAN = "gaussian"  # exp(-z^2 / (2 h^2)), truncated at max_height
VERTICAL_EXPONENTIAL = "exponential"  # exp(-|z| / h), truncated at max_height
VERTICAL_MIDPLANE = "midplane"  # z = 0

# How the clusters are replaced.
REDRAW_SIMULTANEOUS = "simultaneous"  # all clusters every lifetime (CGOLS model C)
REDRAW_STAGGERED = "staggered"  # one cluster every lifetime / num_clusters

# Distribution of the rate shares among the clusters.
MASS_EQUAL = "equal"  # every cluster the same share (CGOLS model C)
MASS_POWER_LAW = "power_law"  # dN/dM ~ M^slope, share ~ M (CGOLS V uses slope -2)


class ClusterPlacementSettings(NamedTuple):
    """How the feedback clusters are drawn. Lengths and times in code units.

    The distribution defaults are CGOLS model C as it was run (Cholla
    ``Grid3D::Add_Supernovae``, evaneschneider/cholla 645a076, in a kpc code
    unit): eight equal, static clusters, the radius drawn uniformly in
    ``R < 0.75`` kpc (uniform in R, not in area, so centrally concentrated),
    the height uniformly in ``|z| <= 105`` pc, all re-drawn together. The
    paper text reads "within the central 1.5 kpc" (a diameter) and "up to
    100 pc". The caller sets the times (model C: first draw at the 5 Myr
    feedback onset, ``lifetime`` = 15 Myr).
    """

    #: Number of clusters alive at any time.
    num_clusters: int = 8

    #: Radius of each cluster's injection sphere.
    cluster_radius: float = 0.15

    #: Time of the first draw, and the cluster lifetime (= re-draw interval
    #: of a slot). The schedule extends to ``end_time``.
    start_time: float = 0.0
    lifetime: float = 1.0
    end_time: float = 1.0

    #: ``REDRAW_SIMULTANEOUS`` or ``REDRAW_STAGGERED``. Staggered slots are
    #: pre-aged evenly at ``start_time``, so one cluster is replaced every
    #: ``lifetime / num_clusters`` from the start on.
    redraw: str = REDRAW_SIMULTANEOUS

    #: Radial distribution, its scale length (exponential only) and cut-off.
    radial_profile: str = RADIAL_UNIFORM_RADIUS
    radial_scale_length: float = 1.0
    max_radius: float = 0.75

    #: Vertical distribution, its scale height (gaussian: the dispersion,
    #: exponential: the e-folding height) and cut-off.
    vertical_profile: str = VERTICAL_UNIFORM
    vertical_scale_height: float = 0.05
    max_height: float = 0.105

    #: Minimum distance between the centres of clusters alive at the same
    #: time (rejection sampling); 0 allows overlaps. ``2 * cluster_radius``
    #: keeps the spheres disjoint.
    min_separation: float = 0.0

    #: Rate shares: ``MASS_EQUAL`` or ``MASS_POWER_LAW`` with the slope of
    #: dN/dM and the mass range (any mass unit, only ratios matter).
    mass_function: str = MASS_EQUAL
    mass_slope: float = -2.0
    min_mass: float = 1e4
    max_mass: float = 5e6

    #: Seed of the numpy generator; the whole schedule is a function of it.
    seed: int = 0


def _draw_radius(rng, settings: ClusterPlacementSettings) -> float:
    """One cylindrical radius from the configured surface-density profile."""
    max_radius = settings.max_radius
    if settings.radial_profile == RADIAL_UNIFORM_AREA:
        # dN/dR ~ R for a constant surface density.
        return max_radius * np.sqrt(rng.uniform())
    if settings.radial_profile == RADIAL_UNIFORM_RADIUS:
        return max_radius * rng.uniform()
    if settings.radial_profile == RADIAL_EXPONENTIAL:
        # dN/dR ~ R exp(-R / R_s) is a Gamma(2, R_s) distribution; truncate
        # by rejection.
        while True:
            radius = rng.gamma(2.0, settings.radial_scale_length)
            if radius <= max_radius:
                return radius
    raise ValueError(f"Unknown radial_profile {settings.radial_profile!r}.")


def _draw_height(rng, settings: ClusterPlacementSettings) -> float:
    """One height above the mid-plane from the configured vertical profile."""
    max_height = settings.max_height
    if settings.vertical_profile == VERTICAL_UNIFORM:
        return rng.uniform(-max_height, max_height)
    if settings.vertical_profile == VERTICAL_MIDPLANE:
        return 0.0
    if settings.vertical_profile in (VERTICAL_GAUSSIAN, VERTICAL_EXPONENTIAL):
        while True:
            if settings.vertical_profile == VERTICAL_GAUSSIAN:
                height = rng.normal(0.0, settings.vertical_scale_height)
            else:
                height = rng.laplace(0.0, settings.vertical_scale_height)
            if abs(height) <= max_height:
                return height
    raise ValueError(f"Unknown vertical_profile {settings.vertical_profile!r}.")


def _draw_mass(rng, settings: ClusterPlacementSettings) -> float:
    """One cluster mass (relative rate share)."""
    if settings.mass_function == MASS_EQUAL:
        return 1.0
    if settings.mass_function == MASS_POWER_LAW:
        # Inverse-CDF sampling of dN/dM ~ M^slope on [min_mass, max_mass].
        a = settings.mass_slope + 1.0
        lo, hi = settings.min_mass, settings.max_mass
        u = rng.uniform()
        if abs(a) < 1e-12:
            return lo * (hi / lo) ** u
        return (lo**a + u * (hi**a - lo**a)) ** (1.0 / a)
    raise ValueError(f"Unknown mass_function {settings.mass_function!r}.")


def _birth_times(settings: ClusterPlacementSettings) -> np.ndarray:
    """Birth time of every occupant of every slot, ``(num_occupants, num_slots)``."""
    n = settings.num_clusters
    lifetime = settings.lifetime
    if settings.redraw == REDRAW_SIMULTANEOUS:
        offsets = np.zeros(n)
    elif settings.redraw == REDRAW_STAGGERED:
        # Slot s's first replacement is due (s + 1) / n lifetimes after the
        # start, i.e. its first occupant was born that much before the end of
        # a full lifetime.
        offsets = (np.arange(n) + 1.0) / n * lifetime - lifetime
    else:
        raise ValueError(f"Unknown redraw scheme {settings.redraw!r}.")
    duration = settings.end_time - settings.start_time
    num_occupants = int(np.ceil(duration / lifetime)) + 1
    births = settings.start_time + offsets[None, :] + lifetime * np.arange(
        num_occupants
    )[:, None]
    # The first occupant of every slot is alive from the start.
    births[0, :] = settings.start_time
    return births


def draw_cluster_schedule(
    settings: ClusterPlacementSettings,
    angular_velocity: Optional[Callable[[float], float]] = None,
) -> CGOLSClusterParams:
    """Draw every cluster of the run.

    Args:
        settings: How the clusters are drawn.
        angular_velocity: Optional ``Omega(R)``; if given, every cluster
            orbits the z axis with the angular velocity at its radius (e.g.
            the circular velocity of the potential over R), as the stars of a
            cluster share the rotation of the gas they formed from. ``None``
            keeps the clusters at rest, as in CGOLS model C.

    Returns:
        The cluster parameters for ``CGOLSWindParams.clusters``.
    """

    rng = np.random.default_rng(settings.seed)
    births = _birth_times(settings)
    num_occupants, num_slots = births.shape

    radii = np.zeros_like(births)
    azimuths = np.zeros_like(births)
    heights = np.zeros_like(births)
    masses = np.zeros_like(births)
    omegas = np.zeros_like(births)

    # Draw in birth order, so that the separation test sees every cluster
    # alive at a birth: those of the other slots born before it (at their
    # orbital phase at that time). Simultaneous births are drawn slot by slot,
    # each tested against the slots already drawn.
    drawn = np.zeros_like(births, dtype=bool)

    def _alive_positions(time, skip_slot):
        positions = []
        for other in range(num_slots):
            if other == skip_slot:
                continue
            e_other = int(np.sum(births[:, other] <= time)) - 1
            if e_other < 0 or not drawn[e_other, other]:
                continue
            phase = azimuths[e_other, other] + omegas[e_other, other] * (
                time - births[e_other, other]
            )
            positions.append(
                [
                    radii[e_other, other] * np.cos(phase),
                    radii[e_other, other] * np.sin(phase),
                    heights[e_other, other],
                ]
            )
        return np.array(positions).reshape(-1, 3)

    order = sorted(
        (births[e, s], s, e) for e in range(num_occupants) for s in range(num_slots)
    )
    for birth, slot, occupant in order:
        others = (
            _alive_positions(birth, slot) if settings.min_separation > 0.0 else None
        )
        for _attempt in range(10000):
            radius = _draw_radius(rng, settings)
            azimuth = rng.uniform(0.0, 2.0 * np.pi)
            height = _draw_height(rng, settings)
            if others is None or len(others) == 0:
                break
            position = np.array(
                [radius * np.cos(azimuth), radius * np.sin(azimuth), height]
            )
            if np.min(np.linalg.norm(others - position, axis=1)) >= settings.min_separation:
                break
        else:
            raise RuntimeError(
                "Could not place a cluster with min_separation "
                f"{settings.min_separation}; lower it or the cluster count."
            )
        radii[occupant, slot] = radius
        azimuths[occupant, slot] = azimuth
        heights[occupant, slot] = height
        masses[occupant, slot] = _draw_mass(rng, settings)
        if angular_velocity is not None:
            omegas[occupant, slot] = angular_velocity(radius)
        drawn[occupant, slot] = True

    return CGOLSClusterParams(
        cluster_radius=settings.cluster_radius,
        birth_times=jnp.asarray(births),
        radii=jnp.asarray(radii),
        azimuths=jnp.asarray(azimuths),
        heights=jnp.asarray(heights),
        angular_velocities=jnp.asarray(omegas),
        rate_fractions=jnp.asarray(masses),
    )


def _glibc_rand():
    """``(srand, rand)`` of the C library, or ``None`` if it is not glibc."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        gnu_get_libc_version = libc.gnu_get_libc_version
    except (OSError, AttributeError):
        return None
    gnu_get_libc_version.restype = ctypes.c_char_p
    libc.srand.argtypes = [ctypes.c_uint]
    libc.rand.restype = ctypes.c_int
    return libc.srand, libc.rand


def cholla_cluster_schedule(
    settings: ClusterPlacementSettings,
    angular_velocity: Optional[Callable[[float], float]] = None,
) -> Optional[CGOLSClusterParams]:
    """The cluster positions of Cholla's own model-C runs, replayed exactly.

    Cholla's ``Grid3D::Add_Supernovae`` (evaneschneider/cholla 645a076, the
    code of arXiv:1803.01005) calls ``srand(int(t) / 15 + 1)`` every step, with
    ``t`` the time in Myr since the feedback onset, and then draws each of the
    eight clusters as::

        r   = R_s * float(rand() % 10000) / 10000.0
        phi = 2 pi * float(rand() % 10000) / 10000.0
        z   = 2 z_s * float(rand() % 10000) / 10000.0 - z_s

    so epoch ``k`` (``k * 15 <= t < (k + 1) * 15``) always holds the same eight
    clusters. Replaying that with the C library's ``rand()`` reproduces the
    paper's realisation, assuming Titan's C library was glibc (its generator
    is the only one this replays). Only the simultaneous, equal-share, static
    scheme exists in Cholla; ``settings`` supplies the cluster count, the
    radius, ``max_radius`` (R_s), ``max_height`` (z_s) and the times. With
    ``angular_velocity`` the same clusters are born at Cholla's positions and
    then orbit the z axis (a deliberate deviation from the static paper model).

    Args:
        settings: The placement settings (distribution fields are ignored).
        angular_velocity: Optional ``Omega(R)``, see :func:`draw_cluster_schedule`.

    Returns:
        The cluster parameters, or ``None`` when the C library is not glibc
        (the caller then falls back to :func:`draw_cluster_schedule`).
    """
    rand_functions = _glibc_rand()
    if rand_functions is None:
        return None
    srand, rand = rand_functions

    births = _birth_times(settings._replace(redraw=REDRAW_SIMULTANEOUS))
    num_occupants, num_slots = births.shape
    radii = np.zeros_like(births)
    azimuths = np.zeros_like(births)
    heights = np.zeros_like(births)
    for epoch in range(num_occupants):
        srand(epoch + 1)
        for slot in range(num_slots):
            radii[epoch, slot] = settings.max_radius * float(rand() % 10000) / 10000.0
            azimuths[epoch, slot] = 2.0 * np.pi * float(rand() % 10000) / 10000.0
            heights[epoch, slot] = (
                2.0 * settings.max_height * float(rand() % 10000) / 10000.0
                - settings.max_height
            )

    return CGOLSClusterParams(
        cluster_radius=settings.cluster_radius,
        birth_times=jnp.asarray(births),
        radii=jnp.asarray(radii),
        azimuths=jnp.asarray(azimuths),
        heights=jnp.asarray(heights),
        angular_velocities=jnp.asarray(
            np.vectorize(angular_velocity)(radii)
            if angular_velocity is not None
            else np.zeros(births.shape)
        ),
        rate_fractions=jnp.ones(births.shape),
    )


def cluster_table(clusters: CGOLSClusterParams) -> str:
    """Every cluster of the schedule as a text table (for the run log)."""
    births = np.asarray(clusters.birth_times)
    lines = ["slot occupant  birth      R          phi        z          omega      share"]
    for occupant in range(births.shape[0]):
        for slot in range(births.shape[1]):
            lines.append(
                f"{slot:4d} {occupant:8d}  "
                f"{births[occupant, slot]:9.5f}  "
                f"{float(clusters.radii[occupant, slot]):9.5f}  "
                f"{float(clusters.azimuths[occupant, slot]):9.5f}  "
                f"{float(clusters.heights[occupant, slot]):+9.5f}  "
                f"{float(clusters.angular_velocities[occupant, slot]):9.5f}  "
                f"{float(clusters.rate_fractions[occupant, slot]):9.4g}"
            )
    return "\n".join(lines)


def cluster_positions(clusters: CGOLSClusterParams, time: float):
    """Positions and rate shares of the living clusters at ``time`` (host side).

    Mirrors the occupant selection of ``_cluster_injection_profile``.

    Args:
        clusters: The cluster parameters.
        time: The simulation time, code units.

    Returns:
        ``(xyz, shares)``: the ``(num_alive, 3)`` positions relative to the box
        center and the matching normalised rate shares.
    """
    births = np.asarray(clusters.birth_times)
    num_born = np.sum(births <= time, axis=0)
    alive = num_born > 0
    occupant = np.maximum(num_born - 1, 0)
    slots = np.arange(births.shape[1])

    def _current(per_occupant):
        return np.asarray(per_occupant)[occupant, slots]

    azimuth = _current(clusters.azimuths) + _current(clusters.angular_velocities) * (
        time - _current(clusters.birth_times)
    )
    radius = _current(clusters.radii)
    xyz = np.stack(
        [radius * np.cos(azimuth), radius * np.sin(azimuth), _current(clusters.heights)],
        axis=-1,
    )[alive]
    shares = _current(clusters.rate_fractions)[alive]
    return xyz, shares / max(shares.sum(), 1e-30)
