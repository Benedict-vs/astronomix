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
:func:`_cluster_injection_profile`. ``SMOOTH_DISK`` replaces the spheres by
their N -> infinity limit, a static axisymmetric profile Sigma(R) h(z) (see
:func:`_smooth_disk_profile`), and otherwise shares the cluster path. The
central-sphere path is unchanged by either option.

The cluster spheres are evaluated either on the whole grid, one cluster after
the other (``CLUSTER_KERNEL_FULL_GRID``, cost proportional to the number of
clusters), or in a small cube around every cluster
(``CLUSTER_KERNEL_LOCAL_BOX``, :func:`_cluster_profile_local_boxes`, cost
independent of the grid), which is what makes ~1000 clusters affordable.

Structured after ``_stellar_wind.stellar_wind._wind_ei3D_source``:
operates on the conserved state and returns a dt-scaled source term for
the finite-difference RK integrators.
"""

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float

# astronomix constants
from astronomix._modules._cgols_wind.cgols_wind_options import (
    CLUSTER_KERNEL_LOCAL_BOX,
    CLUSTER_MOMENTUM_CLUSTER,
    CLUSTER_MOMENTUM_REST,
    CLUSTER_OVERLAP_SUM,
    CLUSTER_OVERLAP_UNION,
    CLUSTER_SPHERES,
    SMOOTH_DISK,
)

# astronomix sharding helpers
from astronomix._pallas_helpers import (
    _current_pallas_mesh,
    _current_pallas_spec,
    _default_state_pspec,
    _spatial_sharded_axes,
    _with_string_axis_names,
    shard_map,
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

    if config.cgols_wind_config.injection_geometry in (CLUSTER_SPHERES, SMOOTH_DISK):
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


def _smooth_disk_profile(
    cgols_wind_params: CGOLSWindParams,
    config: SimulationConfig,
    helper_data: HelperData,
    with_velocity: bool = False,
):
    """Smooth-disk feedback profile, normalised to unit integral.

    ``S = <Sigma(R)> <h(z)> / (sum(<Sigma> <h>) dx^3)`` about the box center,
    with ``Sigma`` and ``h`` interpolated from the ``smooth_disk`` tables (zero
    beyond their last knot, which makes the tables' cut-offs sharp) and
    averaged over each cell on a ``4 x 4`` (x, y) and ``8`` (z) sub-grid: a
    disk thinner than a cell (sigma_z = 50 pc is 1.3 dx at 256^3) then still
    deposits its share in the right cells instead of hinging on where the
    cell centres fall. The
    limit of :func:`_cluster_injection_profile` for infinitely many clusters
    drawn from ``Sigma`` and ``h``, and returned in the same form, so the
    cluster source terms apply unchanged. As there, the cylindrical radius
    is built from the replicated 1D axes by broadcasting.

    With ``with_velocity`` the ejecta move on circular orbits,
    ``v = v_c(R) (-y, x) / R`` from the ``circular_velocity`` table, and the
    velocity moments ``(S, S v_x, S v_y, S |v|^2 / 2)`` are returned.
    """

    if config.dimensionality != 3:
        raise ValueError("CGOLS smooth-disk feedback requires a 3D Cartesian grid.")

    disk = cgols_wind_params.smooth_disk
    dx = config.grid_spacing
    x = helper_data.cell_centers_x - config.box_size.x / 2
    y = helper_data.cell_centers_y - config.box_size.y / 2
    z = helper_data.cell_centers_z - config.box_size.z / 2
    radius = jnp.sqrt(x[:, None] ** 2 + y[None, :] ** 2)

    # Cell averages on the sub-grid (offsets in units of dx about the centre).
    sub_xy = (jnp.arange(4) + 0.5) / 4 - 0.5
    sub_z = (jnp.arange(8) + 0.5) / 8 - 0.5
    sub_x = x[:, None] + sub_xy[None, :] * dx
    sub_y = y[:, None] + sub_xy[None, :] * dx
    sub_radius = jnp.sqrt(
        sub_x[:, None, :, None] ** 2 + sub_y[None, :, None, :] ** 2
    )
    surface = jnp.mean(
        jnp.interp(sub_radius, disk.radii, disk.surface_density, right=0.0),
        axis=(2, 3),
    )
    vertical = jnp.mean(
        jnp.interp(
            z[:, None] + sub_z[None, :] * dx,
            disk.heights,
            disk.vertical_profile,
            left=0.0,
            right=0.0,
        ),
        axis=1,
    )
    weights = surface[:, :, None] * vertical[None, None, :]
    profile = weights / jnp.maximum(jnp.sum(weights) * dx**3, 1e-30)
    if not with_velocity:
        return profile

    speed = jnp.interp(radius, disk.radii, disk.circular_velocity)
    inverse_radius = 1.0 / jnp.maximum(radius, 1e-30)
    velocity_x = -speed * y[None, :] * inverse_radius
    velocity_y = speed * x[:, None] * inverse_radius
    return (
        profile,
        profile * velocity_x[:, :, None],
        profile * velocity_y[:, :, None],
        profile * (0.5 * speed**2)[:, :, None],
    )


def _cluster_injection_profile(
    cgols_wind_params: CGOLSWindParams,
    current_time: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    with_velocity: bool = False,
):
    """Spatial profile of the cluster feedback, normalised to unit integral.

    Returns ``S`` with ``sum(S) * dx^3 = 1`` whenever a cluster is alive, so a
    total rate ``Mdot`` enters as the volumetric rate ``Mdot * S``. For the
    ``SMOOTH_DISK`` geometry this is :func:`_smooth_disk_profile` instead.

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

    With ``with_velocity`` the profile comes with the velocity moments of the
    clusters that feed each cell: ``(S, S v_x, S v_y, S |v|^2 / 2)`` with
    ``S v = sum_k S_k v_k`` (``CLUSTER_OVERLAP_SUM``, every cluster's ejecta
    carry its own velocity) or ``S v_w`` with the ``w_k``-weighted mean
    velocity ``v_w`` of the overlapping spheres (``CLUSTER_OVERLAP_UNION``).
    A cluster at cylindrical radius ``R`` and azimuth ``phi`` moves with
    ``v = Omega R (-sin phi, cos phi, 0)``.

    Args:
        cgols_wind_params: The CGOLS wind parameters (uses ``clusters``).
        current_time: The current simulation time; selects the living cluster
            of every slot and its orbital phase.
        config: The simulation configuration.
        helper_data: The helper data (uses ``cell_centers_x/y/z``).
        with_velocity: Also return the velocity moments (static).

    Returns:
        The injection profile on the (padded) grid of ``helper_data``, or the
        tuple ``(S, S v_x, S v_y, S |v|^2 / 2)`` with ``with_velocity``.
    """

    if config.cgols_wind_config.injection_geometry == SMOOTH_DISK:
        return _smooth_disk_profile(
            cgols_wind_params, config, helper_data, with_velocity
        )

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
    # into one fused elementwise pass. The per-cluster volumes are reduced
    # beforehand, one slot after the other (lax.map): a reduction per slot
    # inside the unrolled sum would keep every slot's 3D weights alive at
    # once, ~0.15 GB per slot at 512^3 on the GPU, so 128 clusters no longer
    # fit an H200 (C512n128 asked for 154 GiB, 2026-10-23).
    num_slots = clusters.birth_times.shape[1]
    overlap = config.cgols_wind_config.cluster_overlap

    if with_velocity:
        # Orbital velocity of every cluster centre, d/dt of the position above.
        orbital_speed = radius * _current(clusters.angular_velocities)
        velocity_x = -orbital_speed * jnp.sin(azimuth)
        velocity_y = orbital_speed * jnp.cos(azimuth)
        half_speed2 = 0.5 * (velocity_x**2 + velocity_y**2)

    if config.cgols_wind_config.cluster_kernel == CLUSTER_KERNEL_LOCAL_BOX:
        if overlap != CLUSTER_OVERLAP_SUM:
            raise NotImplementedError(
                "CLUSTER_KERNEL_LOCAL_BOX supports CLUSTER_OVERLAP_SUM only."
            )
        if with_velocity:
            slot_moments = jnp.stack(
                [jnp.ones_like(velocity_x), velocity_x, velocity_y, half_speed2],
                axis=-1,
            )
        else:
            slot_moments = jnp.ones_like(rate_fraction)[:, None]
        fields = _cluster_profile_local_boxes(
            (center_x, center_y, center_z),
            rate_fraction,
            slot_moments,
            cluster_radius,
            config,
            helper_data,
        )
        return fields if with_velocity else fields[0]

    if overlap == CLUSTER_OVERLAP_SUM:
        volumes = jax.lax.map(
            lambda slot: jnp.sum(_sphere_weights(slot)), jnp.arange(num_slots)
        ) * dx**3
        profile = 0.0
        if with_velocity:
            momentum_x = momentum_y = kinetic = 0.0
        for slot in range(num_slots):
            weights = _sphere_weights(slot)
            # A cluster entirely off the grid (volume 0) injects nothing.
            slot_profile = rate_fraction[slot] / jnp.maximum(volumes[slot], 1e-30) * weights
            profile = profile + slot_profile
            if with_velocity:
                momentum_x = momentum_x + slot_profile * velocity_x[slot]
                momentum_y = momentum_y + slot_profile * velocity_y[slot]
                kinetic = kinetic + slot_profile * half_speed2[slot]
        if with_velocity:
            return profile, momentum_x, momentum_y, kinetic
        return profile

    if overlap == CLUSTER_OVERLAP_UNION:
        union = 0.0
        if with_velocity:
            weight_sum = weighted_vx = weighted_vy = 0.0
        for slot in range(num_slots):
            slot_weights = jnp.where(alive[slot], _sphere_weights(slot), 0.0)
            union = jnp.maximum(union, slot_weights)
            if with_velocity:
                weight_sum = weight_sum + slot_weights
                weighted_vx = weighted_vx + slot_weights * velocity_x[slot]
                weighted_vy = weighted_vy + slot_weights * velocity_y[slot]
        volume = jnp.sum(union) * dx**3
        profile = union / jnp.maximum(volume, 1e-30)
        if with_velocity:
            # The union region is one source; a cell in several spheres gets
            # their weight-averaged velocity.
            mean_vx = weighted_vx / jnp.maximum(weight_sum, 1e-30)
            mean_vy = weighted_vy / jnp.maximum(weight_sum, 1e-30)
            return (
                profile,
                profile * mean_vx,
                profile * mean_vy,
                profile * 0.5 * (mean_vx**2 + mean_vy**2),
            )
        return profile

    raise ValueError(f"Unknown CGOLS cluster_overlap mode {overlap}.")


#: Sort key of a cube cell that injects nothing: behind every cell index.
_NO_CELL = 0xFFFFFFFF


def _local_box_fields(
    centers,
    rate_fraction,
    slot_moments,
    cluster_radius,
    axis_centers,
    origin,
    local_shape,
    dx,
    box,
):
    """The cluster profile on one block of the grid, from one cube per cluster.

    Every cluster only has non-zero weights within ``cluster_radius + dx / 2``
    of its centre, i.e. inside a cube of ``box`` cells per axis. The weights
    are evaluated on those cubes alone, ``(num_slots, box, box, box)`` values
    instead of ``num_slots`` passes over the grid, and written into the block
    in three steps that keep the result independent of the order in which the
    device applies the updates:

    1. the cube cells of all clusters are sorted by their cell index in the
       block (a stable sort, so the clusters of one cell stay in slot order);
    2. a segmented prefix sum adds up the contributions that share a cell, so
       the last entry of every run holds the cell's total;
    3. those last entries, one per cell, are scattered into the block.

    A scatter-add of the unsorted cubes would be the same sum, but where three
    or more spheres overlap the GPU would add them in a different order from
    run to run, and the result would no longer be reproducible bit for bit.

    The per-cluster volume is summed over the cluster's whole cube on the
    global grid, read from the replicated cell-centre axes, so every block
    normalises a cluster by the bit-identical volume wherever the cluster's
    cells lie.

    Args:
        centers: The cluster centres in box coordinates, three ``(num_slots,)``
            arrays.
        rate_fraction: Each slot's share of the total rate, ``(num_slots,)``.
        slot_moments: Per-slot factors of the returned fields,
            ``(num_slots, num_fields)``: ones for the profile, the cluster
            velocity for its momentum moments.
        cluster_radius: The radius of the injection spheres.
        axis_centers: The cell centres of the global grid along x, y and z.
        origin: The global index of the block's first cell along each axis, or
            ``None`` if the block is the whole grid.
        local_shape: The shape of the block (static).
        dx: The grid spacing.
        box: Cells per axis of one cluster's cube (static).

    Returns:
        One array of ``local_shape`` per column of ``slot_moments``: the
        normalised profile times that column, summed over the clusters.
    """
    num_cells_x, num_cells_y, num_cells_z = local_shape
    box_offsets = jnp.arange(box, dtype=jnp.int32)

    def cube_weights(first_cells):
        """Edge weights of every cluster on its cube starting at ``first_cells``."""
        offsets2 = []
        for axis in range(3):
            cells = first_cells[axis][:, None] + box_offsets[None, :]
            offsets2.append((axis_centers[axis][cells] - centers[axis][:, None]) ** 2)
        distance = jnp.sqrt(
            offsets2[0][:, :, None, None]
            + offsets2[1][:, None, :, None]
            + offsets2[2][:, None, None, :]
        )
        return jnp.clip((cluster_radius + dx / 2 - distance) / dx, 0.0, 1.0)

    # The cube starts at the last cell centre at or below the lower edge of
    # the sphere's reach and is kept on the grid (a cluster near the boundary
    # only loses the cells that do not exist).
    global_first = [
        jnp.clip(
            jnp.floor(
                (centers[axis] - (cluster_radius + dx / 2) - axis_centers[axis][0]) / dx
            ).astype(jnp.int32),
            0,
            axis_centers[axis].shape[0] - box,
        )
        for axis in range(3)
    ]
    global_weights = cube_weights(global_first)
    volumes = jnp.sum(global_weights, axis=(1, 2, 3)) * dx**3
    # A cluster entirely off the grid (volume 0) injects nothing.
    share = rate_fraction / jnp.maximum(volumes, 1e-30)
    # The cube must hold the sphere (cgols_wind_options.cluster_box_cells);
    # poison the source rather than truncate the spheres silently.
    share = jnp.where(2.0 * cluster_radius / dx + 2.5 > box, jnp.nan, share)

    if origin is None:
        local_first = global_first
        weights = global_weights
    else:
        # The same cube shifted into the block: the weights come from the
        # true cell centres, so the cells that the shift adds carry weight 0
        # and the cluster's cells in the block are all inside the cube.
        local_first = [
            jnp.clip(global_first[axis] - origin[axis], 0, local_shape[axis] - box)
            for axis in range(3)
        ]
        weights = cube_weights([local_first[axis] + origin[axis] for axis in range(3)])
    values = share[:, None, None, None] * weights

    # Cell index in the block of every cube cell.
    cell_x, cell_y, cell_z = (
        (local_first[axis][:, None] + box_offsets[None, :]).astype(jnp.uint32)
        for axis in range(3)
    )
    keys = (
        cell_x[:, :, None, None] * num_cells_y + cell_y[:, None, :, None]
    ) * num_cells_z + cell_z[:, None, None, :]
    keys = jnp.where(values != 0, keys, jnp.uint32(_NO_CELL)).reshape(-1)

    order = jnp.argsort(keys, stable=True)
    sorted_keys = keys[order]
    sorted_values = (
        values[..., None] * slot_moments[:, None, None, None, :]
    ).reshape(-1, slot_moments.shape[-1])[order]

    def add_same_cell(left, right):
        left_key, left_sum = left
        right_key, right_sum = right
        same_cell = (left_key == right_key)[..., None]
        return right_key, jnp.where(same_cell, left_sum + right_sum, right_sum)

    _, cell_sums = jax.lax.associative_scan(add_same_cell, (sorted_keys, sorted_values))

    next_keys = jnp.concatenate(
        [sorted_keys[1:], jnp.full((1,), _NO_CELL, dtype=jnp.uint32)]
    )
    is_cell_total = (sorted_keys != next_keys) & (sorted_keys != jnp.uint32(_NO_CELL))
    # Every other entry gets an index beyond the block and is dropped.
    index_x = jnp.where(
        is_cell_total,
        (sorted_keys // (num_cells_y * num_cells_z)).astype(jnp.int32),
        num_cells_x,
    )
    index_y = ((sorted_keys // num_cells_z) % num_cells_y).astype(jnp.int32)
    index_z = (sorted_keys % num_cells_z).astype(jnp.int32)
    return tuple(
        jnp.zeros(local_shape, dtype=cell_sums.dtype)
        .at[index_x, index_y, index_z]
        .add(cell_sums[:, field], mode="drop")
        for field in range(cell_sums.shape[-1])
    )


def _cluster_profile_local_boxes(
    centers,
    rate_fraction,
    slot_moments,
    cluster_radius,
    config: SimulationConfig,
    helper_data: HelperData,
):
    """``CLUSTER_KERNEL_LOCAL_BOX``: the cluster profile from one cube per cluster.

    The arithmetic is :func:`_local_box_fields`. On one device it runs on the
    whole grid. On a device mesh it runs inside a ``shard_map``, once per
    block of the grid: every device builds only its own block, from the
    replicated cluster table and cell-centre axes and its position in the
    mesh. Nothing is exchanged between devices; in particular no array of the
    size of the grid is gathered, which a scatter into the sharded global
    array would cause.

    Args:
        centers: The cluster centres in box coordinates, three ``(num_slots,)``
            arrays.
        rate_fraction: Each slot's share of the total rate, ``(num_slots,)``.
        slot_moments: Per-slot factors of the returned fields,
            ``(num_slots, num_fields)``.
        cluster_radius: The radius of the injection spheres.
        config: The simulation configuration.
        helper_data: The helper data (uses ``cell_centers_x/y/z``).

    Returns:
        One array of the (padded) grid shape per column of ``slot_moments``.
    """
    dx = config.grid_spacing
    box = config.cgols_wind_config.cluster_box_cells
    axis_centers = (
        helper_data.cell_centers_x,
        helper_data.cell_centers_y,
        helper_data.cell_centers_z,
    )
    global_shape = tuple(int(axis.shape[0]) for axis in axis_centers)
    if box < 3:
        raise ValueError(
            "CLUSTER_KERNEL_LOCAL_BOX needs CGOLSWindConfig.cluster_box_cells "
            "(see cgols_wind_options.cluster_box_cells)."
        )

    # The spatial axes that the active device mesh splits, as for the Pallas
    # kernels: (array axis of the (var, x, y, z) state, mesh axis, devices).
    mesh = _current_pallas_mesh()
    split_axes = []
    if mesh is not None and mesh.size > 1:
        spec = _current_pallas_spec()
        if spec is None:
            spec = _default_state_pspec(mesh, 3)
        spec = PartitionSpec(*spec, *((None,) * (4 - len(spec))))
        # String axis names, which jax.lax.axis_index needs.
        mesh, spec = _with_string_axis_names(mesh, spec)
        split_axes = _spatial_sharded_axes(mesh, spec, 3)

    local_shape = list(global_shape)
    for array_axis, _, num_devices in split_axes:
        if local_shape[array_axis - 1] != global_shape[array_axis - 1]:
            raise NotImplementedError(
                "CLUSTER_KERNEL_LOCAL_BOX supports one mesh axis per array axis only."
            )
        if global_shape[array_axis - 1] % num_devices:
            raise ValueError(
                f"{global_shape[array_axis - 1]} cells do not split evenly over "
                f"{num_devices} devices."
            )
        local_shape[array_axis - 1] //= num_devices
    local_shape = tuple(local_shape)
    if min(local_shape) < box:
        raise ValueError(
            f"A {local_shape} block is smaller than the {box}-cell cluster cube."
        )
    if local_shape[0] * local_shape[1] * local_shape[2] >= _NO_CELL:
        raise ValueError(f"A {local_shape} block has too many cells for uint32 keys.")

    cluster_radius = jnp.asarray(cluster_radius)
    if not split_axes:
        return _local_box_fields(
            centers,
            rate_fraction,
            slot_moments,
            cluster_radius,
            axis_centers,
            None,
            local_shape,
            dx,
            box,
        )

    def block_fields(
        center_x, center_y, center_z, shares, moments, radius, axis_x, axis_y, axis_z
    ):
        """The profile of this device's block."""
        origin = [jnp.int32(0)] * 3
        for array_axis, mesh_axis_name, _ in split_axes:
            origin[array_axis - 1] = (
                jax.lax.axis_index(mesh_axis_name).astype(jnp.int32)
                * local_shape[array_axis - 1]
            )
        return _local_box_fields(
            (center_x, center_y, center_z),
            shares,
            moments,
            radius,
            (axis_x, axis_y, axis_z),
            origin,
            local_shape,
            dx,
            box,
        )

    block_spec = PartitionSpec(*spec[1:4])
    return shard_map(
        block_fields,
        mesh=mesh,
        in_specs=(PartitionSpec(),) * 9,
        out_specs=(block_spec,) * int(slot_moments.shape[-1]),
    )(*centers, rate_fraction, slot_moments, cluster_radius, *axis_centers)


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
    thermal-energy rates -- with the uniform sphere replaced by the normalised
    cluster profile of :func:`_cluster_injection_profile`. The momentum of the
    injected mass follows ``config.cgols_wind_config.cluster_momentum``:
    ``CLUSTER_MOMENTUM_SCALE`` rescales the cell momentum to keep its kinetic
    energy (the central-sphere convention, no net momentum, exactly Edot as
    heat); ``CLUSTER_MOMENTUM_REST`` adds the mass at rest (Cholla);
    ``CLUSTER_MOMENTUM_CLUSTER`` adds it with its cluster's orbital velocity,
    ``d(rho v) = drho v_c`` and ``dE = Edot dt S + drho |v_c|^2 / 2`` (at rest
    plus Edot in the cluster frame, Galilean-boosted to the box frame).

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

    momentum_mode = config.cgols_wind_config.cluster_momentum
    if momentum_mode == CLUSTER_MOMENTUM_CLUSTER:
        return _cgols_cluster_source_comoving(
            cgols_wind_params,
            conserved_state,
            dt,
            current_time,
            mdot,
            edot,
            config,
            helper_data,
            registered_variables,
        )

    profile = _cluster_injection_profile(
        cgols_wind_params, current_time, config, helper_data
    )

    drho = mdot * dt * profile

    # Keep the pre-existing kinetic energy, see _cgols_wind_source; or, with
    # CLUSTER_MOMENTUM_REST, add the mass at rest (no momentum source), as
    # Cholla does.
    if config.cgols_wind_config.cluster_momentum == CLUSTER_MOMENTUM_REST:
        momentum_source_factor = jnp.zeros_like(drho)
    else:
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


def _cgols_cluster_source_comoving(
    cgols_wind_params: CGOLSWindParams,
    conserved_state: STATE_TYPE,
    dt: Float[Array, ""],
    current_time: Union[float, Float[Array, ""]],
    mdot: Float[Array, ""],
    edot: Float[Array, ""],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Cluster source with the ejecta moving with their cluster.

    ``CLUSTER_MOMENTUM_CLUSTER``: in its own rest frame a cluster adds mass at
    rest and Edot as heat; boosted to the box frame by the cluster velocity
    ``v_c`` that is ``d(rho v) = drho v_c`` and ``dE = edot dt S +
    drho |v_c|^2 / 2``. The totals are exact: ``sum d(rho v) dx^3 = Mdot dt
    <v_c>`` and ``sum dE dx^3 = (Edot + Mdot <|v_c|^2> / 2) dt``, the share-
    weighted cluster means. No momentum along z: the clusters orbit in planes
    parallel to the mid-plane.
    """
    profile, momentum_x, momentum_y, kinetic = _cluster_injection_profile(
        cgols_wind_params, current_time, config, helper_data, with_velocity=True
    )
    mass_scale = mdot * dt
    sources = {
        registered_variables.density_index: mass_scale * profile,
        registered_variables.momentum_index.x: mass_scale * momentum_x,
        registered_variables.momentum_index.y: mass_scale * momentum_y,
        registered_variables.energy_index: edot * dt * profile
        + mass_scale * kinetic,
    }
    zeros = jnp.zeros_like(profile)
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
    if config.cgols_wind_config.injection_geometry in (CLUSTER_SPHERES, SMOOTH_DISK):
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
