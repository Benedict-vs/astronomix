"""
Geometric helper data for the simulation.

Bundles the precomputed per-cell geometry (cell centers, radii, volumes, cell
boundaries) used throughout the solver and the snapshot diagnostics, together
with the machinery to build only the fields the active subsystems actually need
(see :class:`HelperDataRequirements`), optionally on the host, and to pad /
shard / unpad them for the time integrator.
"""

# general
import math

# typing
from types import NoneType
from typing import NamedTuple, Union

# jax
import jax
from jax import NamedSharding
from jax.sharding import PartitionSpec
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    CYLINDRICAL,
    FINITE_VOLUME,
    SPHERICAL,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
)

# astronomix functions
from astronomix._geometry.geometry import _center_of_volume, _r_hat_alpha


class HelperData(NamedTuple):
    """Helper data used throughout the simulation."""

    #: The geometric centers of the cells.
    #: For Cartesian ``dimensionality > 1`` this is the full meshgrid of
    #: cell coordinates with shape ``(Nx, Ny[, Nz], dim)``. Subsystems
    #: that only need a single axis should prefer
    #: :attr:`cell_centers_x` / :attr:`cell_centers_y` / :attr:`cell_centers_z`
    #: so the full meshgrid does not need to be materialised.
    geometric_centers: jnp.ndarray = None

    #: The volumetric centers of the cells.
    #: Same as the geometric centers for Cartesian geometry.
    volumetric_centers: jnp.ndarray = None

    #: 1D arrays of cell centers along each Cartesian axis.
    #: Shape ``(N_axis + 2 * ngc,)``. These broadcast against the state
    #: array so consumers that only need one axis (e.g. the frame
    #: tracker) avoid materialising the full meshgrid.
    cell_centers_x: jnp.ndarray = None
    cell_centers_y: jnp.ndarray = None
    cell_centers_z: jnp.ndarray = None

    #: cell center to box center distances
    #: only for config.dimensionality > 1
    r: jnp.ndarray = None

    #: A helper variable, defined as
    #: \hat{r}^\alpha = V_j / (2 * \alpha * \pi * \Delta r)
    #: with V_j the volume of cell j, \alpha the geometry factor
    #: and \Delta r the cell width.
    r_hat_alpha: jnp.ndarray = None

    #: The cell volumes.
    cell_volumes: jnp.ndarray = None

    #: Coordinates of the inner cell boundaries.
    inner_cell_boundaries: jnp.ndarray = None

    #: Coordinates of the outer cell boundaries.
    outer_cell_boundaries: jnp.ndarray = None


class HelperDataRequirements(NamedTuple):
    """Per-field requirements for :class:`HelperData`.

    Each flag indicates whether the matching :class:`HelperData` field
    is consumed by any active subsystem (solver, physics module,
    snapshot calculator). Fields whose flag is ``False`` are left
    unmaterialised and stay at the ``HelperData`` default of ``None``.
    """

    needs_geometric_centers: bool = False
    needs_volumetric_centers: bool = False
    needs_cell_centers_x: bool = False
    needs_cell_centers_y: bool = False
    needs_cell_centers_z: bool = False
    needs_r: bool = False
    needs_r_hat_alpha: bool = False
    needs_cell_volumes: bool = False
    needs_inner_cell_boundaries: bool = False
    needs_outer_cell_boundaries: bool = False


def _helper_data_requirements(config: SimulationConfig) -> HelperDataRequirements:
    """Derive which :class:`HelperData` fields the configuration needs.

    Args:
        config: The simulation configuration.

    Returns:
        A :class:`HelperDataRequirements` whose flags are ``True`` only for the
        helper-data fields consumed by some currently active subsystem, so the
        builder can leave the rest unmaterialised.
    """

    dimensionality = config.dimensionality
    curvilinear = config.geometry != CARTESIAN
    finite_volume = config.solver_mode == FINITE_VOLUME

    needs_geometric_centers = False
    needs_volumetric_centers = False
    needs_r_hat_alpha = False
    needs_cell_volumes = False
    needs_inner_cell_boundaries = False
    needs_outer_cell_boundaries = False
    needs_r = False

    # Finite-volume geometric source / reconstruction terms in
    # curvilinear geometry (1D spherical / cylindrical).
    if finite_volume and curvilinear:
        needs_geometric_centers = True
        needs_volumetric_centers = True
        needs_r_hat_alpha = True
        # 1D curvilinear get_helper_data also produces cell_volumes
        # and inner/outer boundaries; downstream code reads them in
        # the wind / cosmic-ray paths, gated by their own flags.

    # Stellar wind injection.
    if config.wind_config.stellar_wind:
        needs_cell_volumes = True
        needs_inner_cell_boundaries = True
        needs_outer_cell_boundaries = True
        if dimensionality > 1:
            needs_geometric_centers = True
            needs_r = True

    # CGOLS central starburst wind: needs the distance to the box center.
    if config.cgols_wind_config.cgols_wind:
        needs_r = True

    # Cosmic-ray diffusive shock acceleration.
    if config.cosmic_ray_config.diffusive_shock_acceleration:
        needs_geometric_centers = True
        needs_cell_volumes = True

    # Neural-network body force / cnn corrector positions.
    if config.neural_net_force_config.neural_net_force:
        needs_geometric_centers = True

    # Frame tracking (3D Cartesian) only needs the z-axis array.
    needs_cell_centers_x = False
    needs_cell_centers_y = False
    needs_cell_centers_z = False
    if config.frame_tracking:
        needs_cell_centers_z = True

    # Snapshot diagnostics.
    if config.return_snapshots:
        snapshot_settings = config.snapshot_settings
        volume_consumers = (
            snapshot_settings.return_total_mass
            or snapshot_settings.return_total_energy
            or snapshot_settings.return_internal_energy
            or snapshot_settings.return_kinetic_energy
            or snapshot_settings.return_gravitational_energy
        )
        if dimensionality == 1 and volume_consumers:
            needs_cell_volumes = True
        if snapshot_settings.return_radial_momentum:
            if dimensionality == 1:
                needs_cell_volumes = True
            else:
                needs_geometric_centers = True

    return HelperDataRequirements(
        needs_geometric_centers=needs_geometric_centers,
        needs_volumetric_centers=needs_volumetric_centers,
        needs_cell_centers_x=needs_cell_centers_x,
        needs_cell_centers_y=needs_cell_centers_y,
        needs_cell_centers_z=needs_cell_centers_z,
        needs_r=needs_r,
        needs_r_hat_alpha=needs_r_hat_alpha,
        needs_cell_volumes=needs_cell_volumes,
        needs_inner_cell_boundaries=needs_inner_cell_boundaries,
        needs_outer_cell_boundaries=needs_outer_cell_boundaries,
    )


def _normalize_config_vectors(config: SimulationConfig) -> SimulationConfig:
    """Promote scalar box_size / num_cells to StaticVectors and pick grid_spacing."""

    if isinstance(config.box_size, float):
        config = config._replace(
            box_size=StaticFloatVector(
                config.box_size, config.box_size, config.box_size
            )
        )

    if isinstance(config.num_cells, int):
        config = config._replace(
            num_cells=StaticIntVector(
                config.num_cells, config.num_cells, config.num_cells
            )
        )

    grid_spacing_vec = config.box_size / config.num_cells

    if config.dimensionality == 1:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
    elif config.dimensionality == 2:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
        if not math.isclose(grid_spacing_vec.x, grid_spacing_vec.y):
            raise ValueError(
                "For now, we assume the grid spacing is the same in all dimensions. "
                f"Got grid spacing {grid_spacing_vec}."
            )
    elif config.dimensionality == 3:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
        if not (
            math.isclose(grid_spacing_vec.x, grid_spacing_vec.y)
            and math.isclose(grid_spacing_vec.x, grid_spacing_vec.z)
        ):
            raise ValueError(
                "For now, we assume the grid spacing is the same in all dimensions. "
                f"Got grid spacing {grid_spacing_vec}."
            )
    return config


_ALL_FIELDS_REQUIRED = HelperDataRequirements(
    needs_geometric_centers=True,
    needs_volumetric_centers=True,
    needs_cell_centers_x=True,
    needs_cell_centers_y=True,
    needs_cell_centers_z=True,
    needs_r=True,
    needs_r_hat_alpha=True,
    needs_cell_volumes=True,
    needs_inner_cell_boundaries=True,
    needs_outer_cell_boundaries=True,
)


def get_helper_data(
    config: SimulationConfig,
    sharding: Union[NoneType, NamedSharding] = None,
    padded: bool = False,
    requirements: Union[NoneType, HelperDataRequirements] = None,
) -> HelperData:
    """Generate the helper data for the simulation.

    Only the fields requested by ``requirements`` are materialised; the
    rest stay at the :class:`HelperData` default of ``None``. When
    ``requirements`` is ``None`` (the default), all fields are built
    so that scripts using ``get_helper_data`` to construct initial
    conditions keep working. The time integrator passes a
    config-derived :class:`HelperDataRequirements` explicitly so the
    in-simulation helper data is minimal.

    With ``config.host_helper_data=True`` the arrays are built inside a
    CPU device context and only the requested fields are moved onto the
    target device (respecting ``sharding`` for the 3D Cartesian spatial
    fields).

    Otherwise, when a ``sharding`` is supplied the build runs inside a
    ``jax.jit`` with ``out_shardings``, so each device — and, in a
    multi-process run, each process — materialises only its own shards
    and the global field never has to fit anywhere. This is what makes
    the 2048^2 x 4096 grid buildable: ``r`` alone is ~69 GB globally.
    Note that a jitted build is not bit-identical to the eager one (XLA
    fuses the coordinate arithmetic; measured ~1e-7 relative, 5 ulp on
    ``r``), so a run that switches ``host_helper_data`` off changes the
    last bits of the geometry — it is a float32 reassociation, not a
    formula change.
    """

    config = _normalize_config_vectors(config)

    if requirements is None:
        requirements = _ALL_FIELDS_REQUIRED

    if padded:
        ngc = config.num_ghost_cells
    else:
        ngc = 0

    grid_spacing = config.grid_spacing

    host_build = config.host_helper_data
    if host_build:
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError(
                "host_helper_data=True requires a CPU device to be available."
            )
        build_ctx = jax.default_device(cpu_devices[0])

        with build_ctx:
            helper_data = _build_helper_data(config, requirements, ngc, grid_spacing)
        helper_data = _move_helper_data_to_device(helper_data, sharding, config)
    elif sharding is not None:
        # Build straight into the target shards. Without the ``out_shardings``
        # jit the builder would materialise every field globally on one device
        # first and only then be constrained — at 2048^2 x 4096 the ``r`` field
        # alone is ~69 GB, which no single device holds. Under a multi-process
        # mesh this also means each process only ever computes its own shard.
        def _build():
            return _build_helper_data(config, requirements, ngc, grid_spacing)

        out_shardings = _helper_data_shardings(
            jax.eval_shape(_build), sharding, config
        )
        helper_data = jax.jit(_build, out_shardings=out_shardings)()
        # The eager builder hands the SAME array to both center fields in
        # Cartesian geometry; a jit returns them as two tuple elements, which
        # XLA is not obliged to alias. Re-alias so the sharded path does not
        # silently hold a second copy of a (Nx, Ny, Nz, 3) field.
        if (
            config.geometry == CARTESIAN
            and helper_data.geometric_centers is not None
            and helper_data.volumetric_centers is not None
        ):
            helper_data = helper_data._replace(
                volumetric_centers=helper_data.geometric_centers
            )
    else:
        helper_data = _build_helper_data(config, requirements, ngc, grid_spacing)

    return helper_data


def _build_helper_data(
    config: SimulationConfig,
    requirements: HelperDataRequirements,
    ngc: int,
    grid_spacing: float,
) -> HelperData:
    """Compute only the fields the requirements ask for.

    Intermediates (e.g. the X, Y, Z meshgrid used to derive ``r``) are
    held as locals; they are not stored in the returned struct unless
    explicitly requested.
    """

    fields = {}

    if config.geometry == SPHERICAL or config.geometry == CYLINDRICAL:
        # 1D curvilinear geometry.
        r_axis = jnp.linspace(
            grid_spacing / 2 - ngc * grid_spacing,
            config.box_size.x + grid_spacing / 2 + ngc * grid_spacing,
            config.num_cells.x + 2 * ngc,
            endpoint=False,
        )

        if requirements.needs_geometric_centers:
            fields["geometric_centers"] = r_axis
        if requirements.needs_cell_centers_x:
            fields["cell_centers_x"] = r_axis
        if requirements.needs_inner_cell_boundaries:
            fields["inner_cell_boundaries"] = r_axis - grid_spacing / 2
        if requirements.needs_outer_cell_boundaries:
            fields["outer_cell_boundaries"] = r_axis + grid_spacing / 2

        # r_hat_alpha is a building block for both volumetric_centers and
        # cell_volumes; compute it lazily.
        r_hat_alpha = None
        if (
            requirements.needs_r_hat_alpha
            or requirements.needs_volumetric_centers
            or requirements.needs_cell_volumes
        ):
            r_hat_alpha = _r_hat_alpha(r_axis, grid_spacing, config.geometry)

        if requirements.needs_volumetric_centers:
            fields["volumetric_centers"] = _center_of_volume(
                r_axis, grid_spacing, config.geometry
            )
        if requirements.needs_r_hat_alpha:
            fields["r_hat_alpha"] = r_hat_alpha
        if requirements.needs_cell_volumes:
            fields["cell_volumes"] = (
                2 * config.geometry * jnp.pi * grid_spacing * r_hat_alpha
            )

        return HelperData(**fields)

    # Cartesian geometry.
    if config.dimensionality == 1:
        r_axis = jnp.linspace(
            grid_spacing / 2 - ngc * grid_spacing,
            config.box_size.x - grid_spacing / 2 + ngc * grid_spacing,
            config.num_cells.x + 2 * ngc,
        )

        if requirements.needs_geometric_centers:
            fields["geometric_centers"] = r_axis
        if requirements.needs_volumetric_centers:
            fields["volumetric_centers"] = r_axis
        if requirements.needs_cell_centers_x:
            fields["cell_centers_x"] = r_axis
        if requirements.needs_r_hat_alpha:
            fields["r_hat_alpha"] = grid_spacing * jnp.ones_like(r_axis)
        if requirements.needs_cell_volumes:
            fields["cell_volumes"] = grid_spacing * jnp.ones_like(r_axis)
        if requirements.needs_inner_cell_boundaries:
            fields["inner_cell_boundaries"] = r_axis - grid_spacing / 2
        if requirements.needs_outer_cell_boundaries:
            fields["outer_cell_boundaries"] = r_axis + grid_spacing / 2

        return HelperData(**fields)

    # Cartesian dimensionality > 1.
    # Build the 1D axis arrays lazily and only the ones we need.
    box_sizes = (config.box_size.x, config.box_size.y, config.box_size.z)
    cell_counts = (config.num_cells.x, config.num_cells.y, config.num_cells.z)

    need_centers = requirements.needs_geometric_centers
    need_vol_centers = requirements.needs_volumetric_centers
    need_r = requirements.needs_r
    # ``r`` is deliberately NOT a meshgrid consumer: when it is the only field
    # wanted (the CGOLS-wind case) it is assembled from broadcast 1D axes
    # below, which avoids materialising the two ``(Nx, Ny, Nz, dim)`` coordinate
    # arrays. At 2048^2 x 4096 those are ~208 GB each — building them just to
    # take a norm is what made the field unbuildable at scale.
    need_meshgrid = need_centers or need_vol_centers

    per_axis_requested = (
        requirements.needs_cell_centers_x,
        requirements.needs_cell_centers_y,
        requirements.needs_cell_centers_z,
    )[: config.dimensionality]

    # The 1D per-axis arrays are needed either when explicitly requested
    # or as building blocks for the meshgrid / the analytic ``r``.
    axis_needed = [need_meshgrid or need_r or pa for pa in per_axis_requested]

    axis_arrays = []
    for size, ncells, needed in zip(
        box_sizes[: config.dimensionality],
        cell_counts[: config.dimensionality],
        axis_needed,
    ):
        if needed:
            axis_arrays.append(
                jnp.linspace(
                    grid_spacing / 2 - ngc * grid_spacing,
                    size + grid_spacing / 2 + ngc * grid_spacing,
                    ncells + 2 * ngc,
                    endpoint=False,
                )
            )
        else:
            axis_arrays.append(None)

    if requirements.needs_cell_centers_x and axis_arrays[0] is not None:
        fields["cell_centers_x"] = axis_arrays[0]
    if config.dimensionality >= 2 and requirements.needs_cell_centers_y and axis_arrays[1] is not None:
        fields["cell_centers_y"] = axis_arrays[1]
    if config.dimensionality >= 3 and requirements.needs_cell_centers_z and axis_arrays[2] is not None:
        fields["cell_centers_z"] = axis_arrays[2]

    if not need_meshgrid:
        if need_r:
            fields["r"] = _radius_from_axes(axis_arrays, box_sizes, config)
        return HelperData(**fields)

    geometric_centers = jnp.array(jnp.meshgrid(*axis_arrays, indexing="ij"))
    geometric_centers = jnp.moveaxis(geometric_centers, 0, -1)

    if need_centers:
        fields["geometric_centers"] = geometric_centers
    if need_vol_centers:
        # In Cartesian, volumetric centers coincide with geometric centers.
        fields["volumetric_centers"] = geometric_centers
    if need_r:
        # The meshgrid already exists here, so reuse it rather than rebuilding
        # the coordinates; bit-identical to the _radius_from_axes branch.
        box_center = jnp.array(
            [size / 2 for size in box_sizes[: config.dimensionality]]
        )
        fields["r"] = jnp.linalg.norm(
            geometric_centers - box_center, axis=-1
        )

    return HelperData(**fields)


def _radius_from_axes(axis_arrays, box_sizes, config: SimulationConfig) -> jnp.ndarray:
    """Cell-center-to-box-center distance, built from the 1D axis arrays.

    Equivalent to ``norm(meshgrid(...) - box_center)`` but without ever forming
    the ``(N_x, N_y, N_z, dim)`` coordinate array (two of them, counting the
    ``moveaxis`` copy). Each axis is offset by its own half-box first and then
    broadcast against the others, so under a jit with ``out_shardings`` XLA can
    produce the result directly in its sharded layout.

    ``jnp.broadcast_arrays`` is required: ``jnp.stack`` alone rejects the
    unequal per-axis shapes.
    """
    dimensionality = config.dimensionality
    offset_axes = []
    for axis_index in range(dimensionality):
        shape = [1] * dimensionality
        shape[axis_index] = axis_arrays[axis_index].shape[0]
        offset_axes.append(
            axis_arrays[axis_index].reshape(shape) - box_sizes[axis_index] / 2
        )
    return jnp.linalg.norm(jnp.stack(jnp.broadcast_arrays(*offset_axes)), axis=0)


def _field_sharding(
    name: str,
    ndim: int,
    sharding: NamedSharding,
    config: SimulationConfig,
) -> NamedSharding:
    """The target sharding for one :class:`HelperData` field.

    The spatial fields co-locate with the primitive state; everything else
    (the 1D per-axis arrays, the 1D curvilinear fields) is replicated.

    primitive_state is (vars, X, Y, Z); geometric_centers is (X, Y, Z, vec) and
    ``r`` is (X, Y, Z). Reusing the primitive-state PartitionSpec positionally
    would put the vars-axis mesh assignment on the X-axis of the field and
    shift the spatial assignments by one — i.e. Y would inherit XAXIS (split)
    while X would inherit VARAXIS (replicated), breaking co-location with
    primitive_state. Drop the leading vars entry, and pad with ``None`` for the
    trailing vector index of the centers (``r`` has no such axis).
    """
    replicated = NamedSharding(sharding.mesh, PartitionSpec())
    if config.geometry != CARTESIAN or config.dimensionality != 3:
        return replicated
    if name in ("geometric_centers", "volumetric_centers") and ndim == 4:
        return NamedSharding(
            sharding.mesh, PartitionSpec(*sharding.spec[1:4], None)
        )
    if name == "r" and ndim == 3:
        return NamedSharding(sharding.mesh, PartitionSpec(*sharding.spec[1:4]))
    return replicated


def _helper_data_shardings(
    struct: HelperData,
    sharding: NamedSharding,
    config: SimulationConfig,
) -> HelperData:
    """``out_shardings`` pytree for a :func:`_build_helper_data` jit.

    ``struct`` is the ``jax.eval_shape`` result, i.e. a :class:`HelperData`
    whose materialised fields are ``ShapeDtypeStruct``s and whose unused fields
    are ``None`` — so the returned struct has a leaf exactly where the built
    helper data has one.
    """
    return HelperData(
        **{
            name: _field_sharding(name, value.ndim, sharding, config)
            for name, value in struct._asdict().items()
            if value is not None
        }
    )


def _move_helper_data_to_device(
    helper_data: HelperData,
    sharding: Union[NoneType, NamedSharding],
    config: SimulationConfig,
) -> HelperData:
    """Move only materialised fields onto the target device.

    With ``sharding`` provided, the spatial fields (``geometric_centers`` /
    ``volumetric_centers`` / ``r``) are placed with the matching spatial
    sharding — see :func:`_field_sharding` for the axis correction — and the
    remaining 1D arrays are replicated. ``r`` in particular MUST be placed
    explicitly: a bare ``jax.device_put`` commits it to a single device, which
    a multi-process pjit cannot consume at all.
    """

    moved = {}
    transferred = {}  # id(host array) -> device array, so aliased fields
    # (geometric_centers / volumetric_centers point at the same array in
    # Cartesian) are transferred once and stay aliased on device.
    for name, value in helper_data._asdict().items():
        if value is None:
            continue
        if id(value) in transferred:
            moved[name] = transferred[id(value)]
            continue
        if sharding is not None:
            moved[name] = jax.device_put(
                value, _field_sharding(name, value.ndim, sharding, config)
            )
        else:
            moved[name] = jax.device_put(value)
        transferred[id(value)] = moved[name]
    return HelperData(**moved)


def _unpad_helper_data(
    helper_data: HelperData,
    config: SimulationConfig,
) -> HelperData:
    """Return a view of ``helper_data`` with ghost-cell slices removed.

    Used inside the time integrator to compute snapshot diagnostics on
    the physical domain without keeping a separate unpadded helper-data
    struct around. Slicing is a free operation under jit.
    """

    ngc = config.num_ghost_cells
    if ngc == 0:
        return helper_data

    fields = {}
    for name, value in helper_data._asdict().items():
        if value is None:
            continue
        fields[name] = _slice_ghost(value, ngc, config.dimensionality)
    return HelperData(**fields)


def _slice_ghost(value: jnp.ndarray, ngc: int, dimensionality: int) -> jnp.ndarray:
    """Strip ngc ghost cells from each spatial axis.

    Spatial axes are the leading ``dimensionality`` axes for the
    Cartesian fields (e.g. shape ``(Nx+2g, Ny+2g, Nz+2g, 3)``); 1D
    curvilinear fields are simple ``(N+2g,)`` arrays.
    """

    slicer = [slice(None)] * value.ndim
    for axis in range(min(dimensionality, value.ndim)):
        slicer[axis] = slice(ngc, value.shape[axis] - ngc)
    return value[tuple(slicer)]
