"""
Pallas backend of the sub-cycled cooling update with heating.

Generated with the ``pallasify`` skill from ``update_temperature_subcycled``
(``with_heating`` branch) in ``_cooling.py``; do not edit by hand, regenerate
from the native function instead.

The native update carries four full fields through ``max_subcycles`` passes of
a ``fori_loop`` plus a 12-pass bisection, although only a small fraction of
the cells is stiff enough to need more than one sub-cycle. The kernel does the
whole update of a block of cells in registers: one read of the inputs, one
write of the result, and the sub-cycle and bisection loops of a block stop as
soon as none of its cells has anything left to do. The skipped iterations are
exact no-ops of the native loop, so the result is unchanged.

Supported: 3D, the piecewise power-law cooling curve with evenly log-spaced
tables (``CoolingCurveConfig.uniform_log_tables``) and the temperature
dependent heating factor table. Everything else stays on the native path.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix._modules._cooling.cooling_options import PIECEWISE_POWER_LAW
from astronomix.option_classes.simulation_config import PALLAS

# astronomix functions
from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _pallas_call_sharded,
    _pallas_compiler_params,
    diffable_pallas_call_n,
    pl,
)

#: Fixed trip count of the residual bisection, as in the native function.
_NUM_BISECTIONS = 12


def _subcycled_cooling_pallas_supported(
    density,
    cooling_config,
    backend_config,
    with_heating,
    heating_factor_table,
) -> bool:
    """
    Whether the Pallas kernel of the sub-cycled cooling update can be used.

    Args:
        density: The density field.
        cooling_config: The cooling configuration.
        backend_config: The ``BackendConfig`` of the run, or ``None``.
        with_heating: Whether the net rate includes the heating.
        heating_factor_table: The heating factor table.

    Returns:
        True if the kernel can be used.
    """
    if pl is None or backend_config is None:
        return False
    if backend_config.backend != PALLAS:
        return False
    # Opt-in (CoolingConfig.pallas_subcycling), like pallas_ct.
    if not cooling_config.pallas_subcycling:
        return False
    curve_config = cooling_config.cooling_curve_config
    if curve_config.cooling_curve_type != PIECEWISE_POWER_LAW:
        return False
    if not curve_config.uniform_log_tables:
        return False
    if not with_heating:
        return False
    if heating_factor_table is None or heating_factor_table.shape[0] < 2:
        return False
    if density.ndim != 3:
        return False
    # Triton-x64 caveat (see the implementation guide): float64 runs stay
    # native unless the kernel is interpreted.
    if jax.config.jax_enable_x64 and not backend_config.pallas_interpret:
        return False
    block_shape = _as_3tuple_block_shape(
        backend_config.pallas_block_shape, 3, spatial_shape=density.shape
    )
    return all(
        int(extent) % int(block) == 0
        for extent, block in zip(density.shape, block_shape, strict=True)
    )


def update_temperature_subcycled_pallas(
    density,
    temperature,
    time_step,
    hydrogen_mass_fraction,
    metal_mass_fraction,
    gamma,
    cooling_curve_config,
    cooling_curve_params,
    floor_temperature,
    max_subcycles,
    max_fractional_temperature_change,
    mean_molecular_weight,
    heating_rate,
    heating_log10_T_table,
    heating_factor_table,
    backend_config,
):
    """
    Sub-cycled cooling with heating, Pallas kernel (public, shard-aware).

    Same arguments and result as ``update_temperature_subcycled`` with
    ``with_heating=True``, plus the ``backend_config`` of the run. Automatic
    differentiation goes through the native function.

    Returns:
        The temperature after one full ``time_step``.
    """
    from astronomix._modules._cooling._cooling import (  # lazy: circular import
        _volumetric_cooling_prefactor,
        get_effective_molecular_weights,
        update_temperature_subcycled,
    )

    dtype = temperature.dtype
    block_shape = _as_3tuple_block_shape(
        backend_config.pallas_block_shape, 3, spatial_shape=density.shape
    )

    # Every traced value enters the two branches as an argument (a custom-JVP
    # function must not close over tracers); only static Python values
    # (the curve configuration, the sub-cycle count) are closed over.
    def pallas_branch(density_in, temperature_in, time_step_in, heating_rate_in, traced):
        (
            hydrogen_mass_fraction_in,
            metal_mass_fraction_in,
            gamma_in,
            curve_params_in,
            floor_temperature_in,
            max_fraction_in,
            mean_molecular_weight_in,
            heating_log10_T_in,
            heating_factor_in,
        ) = traced
        mu, mu_e, mu_H = get_effective_molecular_weights(
            hydrogen_mass_fraction_in, metal_mass_fraction_in, mean_molecular_weight_in
        )
        prefactor = _volumetric_cooling_prefactor(cooling_curve_config, mu, mu_e, mu_H)
        log10_T_table = curve_params_in.log10_T_table
        tables = (
            jnp.asarray(10**log10_T_table, dtype=dtype),
            jnp.asarray(10**curve_params_in.log10_Lambda_table, dtype=dtype),
            jnp.asarray(curve_params_in.alpha_table, dtype=dtype),
            jnp.asarray(heating_log10_T_in, dtype=dtype),
            jnp.asarray(heating_factor_in, dtype=dtype),
        )
        scalars = (
            jnp.asarray(log10_T_table[0], dtype=dtype),
            jnp.asarray(log10_T_table[1] - log10_T_table[0], dtype=dtype),
            jnp.asarray(heating_log10_T_in[0], dtype=dtype),
            jnp.asarray(heating_log10_T_in[1] - heating_log10_T_in[0], dtype=dtype),
            jnp.asarray(heating_log10_T_in[-1], dtype=dtype),
            jnp.asarray(gamma_in - 1, dtype=dtype),
            jnp.asarray(prefactor, dtype=dtype),
            jnp.asarray(floor_temperature_in, dtype=dtype),
            jnp.asarray(max_fraction_in, dtype=dtype),
        )
        # The sharded-call machinery works on vars-first state-shaped arrays,
        # so the four per-cell inputs ride in one stacked array.
        fields = jnp.stack(
            [
                density_in,
                temperature_in,
                jnp.broadcast_to(time_step_in, temperature_in.shape).astype(dtype),
                jnp.broadcast_to(heating_rate_in, temperature_in.shape).astype(dtype),
            ]
        )
        new_temperature = _pallas_call_sharded(
            lambda fields_local, *other: _update_temperature_subcycled_pallas_local(
                fields_local,
                other[: len(tables)],
                other[len(tables):],
                max_subcycles,
                backend_config,
            ),
            state_inputs=(fields,),
            other_args=tables + scalars,
            halo=(0, 0, 0),
            block_shape=block_shape,
        )
        return new_temperature[0]

    def native_branch(density_in, temperature_in, time_step_in, heating_rate_in, traced):
        (
            hydrogen_mass_fraction_in,
            metal_mass_fraction_in,
            gamma_in,
            curve_params_in,
            floor_temperature_in,
            max_fraction_in,
            mean_molecular_weight_in,
            heating_log10_T_in,
            heating_factor_in,
        ) = traced
        return update_temperature_subcycled(
            density_in,
            temperature_in,
            time_step_in,
            hydrogen_mass_fraction_in,
            metal_mass_fraction_in,
            gamma_in,
            cooling_curve_config,
            curve_params_in,
            floor_temperature_in,
            max_subcycles,
            max_fraction_in,
            mean_molecular_weight_in,
            with_heating=True,
            heating_rate=heating_rate_in,
            heating_log10_T_table=heating_log10_T_in,
            heating_factor_table=heating_factor_in,
        )

    def as_array(value):
        return jnp.asarray(value, dtype=dtype)

    traced = (
        as_array(hydrogen_mass_fraction),
        as_array(metal_mass_fraction),
        as_array(gamma),
        jax.tree.map(as_array, cooling_curve_params),
        as_array(floor_temperature),
        as_array(max_fractional_temperature_change),
        as_array(mean_molecular_weight),
        as_array(heating_log10_T_table),
        as_array(heating_factor_table),
    )
    return diffable_pallas_call_n(
        (density, temperature, as_array(time_step), as_array(heating_rate), traced),
        pallas_branch=pallas_branch,
        native_branch=native_branch,
    )


def _update_temperature_subcycled_pallas_local(
    fields,
    tables,
    scalars,
    max_subcycles,
    backend_config,
):
    """
    Single-shard ``pl.pallas_call`` build of the sub-cycled cooling kernel.

    Args:
        fields: Stacked ``(density, temperature, time_step, heating_rate)``,
            shape ``(4, nx, ny, nz)`` of this shard.
        tables: ``(T, Lambda, alpha, heating log10 T, heating factor)`` tables.
        scalars: The scalar parameters, in the order built by the wrapper.
        max_subcycles: The (static) sub-cycle count.
        backend_config: The ``BackendConfig`` of the run.

    Returns:
        The new temperature, shape ``(1, nx, ny, nz)``.
    """
    dtype = fields.dtype
    nx, ny, nz = (int(extent) for extent in fields.shape[1:])
    bx, by, bz = _as_3tuple_block_shape(
        backend_config.pallas_block_shape, 3, spatial_shape=(nx, ny, nz)
    )
    grid = (nx // bx, ny // by, nz // bz)

    num_cooling_bins = int(tables[0].shape[0])
    num_heating_nodes = int(tables[3].shape[0])
    last_cooling_bin = num_cooling_bins - 2
    last_heating_bin = num_heating_nodes - 2
    tiny = float(jnp.finfo(dtype).tiny)

    fields_spec = pl.BlockSpec((4, bx, by, bz), lambda bi, bj, bk: (0, bi, bj, bk))
    out_spec = pl.BlockSpec((1, bx, by, bz), lambda bi, bj, bk: (0, bi, bj, bk))
    table_specs = [
        pl.BlockSpec(table.shape, lambda bi, bj, bk: (0,)) for table in tables
    ]
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(
        fields_ref,
        T_table_ref,
        Lambda_table_ref,
        alpha_table_ref,
        heating_log10_T_ref,
        heating_factor_ref,
        log10_T_first_ref,
        log10_T_spacing_ref,
        heating_log10_T_first_ref,
        heating_log10_T_spacing_ref,
        heating_log10_T_last_ref,
        gamma_minus_one_ref,
        prefactor_ref,
        floor_ref,
        max_fraction_ref,
        out_ref,
    ):
        density = fields_ref[0, ...]
        temperature = fields_ref[1, ...]
        time_step = fields_ref[2, ...]
        heating_rate = fields_ref[3, ...]

        log10_T_first = log10_T_first_ref[()]
        log10_T_spacing = log10_T_spacing_ref[()]
        heating_log10_T_first = heating_log10_T_first_ref[()]
        heating_log10_T_spacing = heating_log10_T_spacing_ref[()]
        heating_log10_T_last = heating_log10_T_last_ref[()]
        gamma_minus_one = gamma_minus_one_ref[()]
        prefactor = prefactor_ref[()]
        floor_temperature = floor_ref[()]
        max_fraction = max_fraction_ref[()]
        table_minimum_temperature = T_table_ref[0]
        table_maximum_temperature = T_table_ref[num_cooling_bins - 1]

        zero = floor_temperature - floor_temperature
        one = zero + 1.0
        minus_one = zero - 1.0

        def first_bin(log_x, first, spacing, last):
            """floor((log_x - first) / spacing), clamped to [0, last]."""
            position = jnp.floor((log_x - first) / spacing)
            # A NaN must not become an out-of-range table index.
            position = jnp.where(position == position, position, zero)
            return jnp.clip(position, 0.0, float(last)).astype(jnp.int32)

        def net_rate(T):
            """dT/dt of ``dtemperature_dt``: cooling plus heating."""
            log_T = jnp.log10(jnp.maximum(T, tiny))

            # Lambda(T) on the piecewise power-law table
            # (_evaluate_piecewise_power_law with _uniform_log_bin).
            k = first_bin(log_T, log10_T_first, log10_T_spacing, last_cooling_bin)
            k = k - (T <= T_table_ref[k]).astype(jnp.int32)
            k = k + (
                T > T_table_ref[jnp.clip(k + 1, 0, last_cooling_bin + 1)]
            ).astype(jnp.int32)
            k = jnp.clip(k, 0, last_cooling_bin)
            alpha_k = alpha_table_ref[k]
            T_k = T_table_ref[k]
            Lambda_k = Lambda_table_ref[k]
            T_safe = jnp.clip(T, table_minimum_temperature, table_maximum_temperature)
            cooling_rate = jnp.where(
                (T >= table_minimum_temperature) & (T <= table_maximum_temperature),
                Lambda_k * (T_safe / T_k) ** alpha_k,
                zero,
            )

            # Heating factor f(T~), linear in log10 T~ (clamped at the ends).
            m = first_bin(
                log_T, heating_log10_T_first, heating_log10_T_spacing, last_heating_bin
            )
            m = m - (log_T <= heating_log10_T_ref[m]).astype(jnp.int32)
            m = m + (
                log_T > heating_log10_T_ref[jnp.clip(m + 1, 0, last_heating_bin + 1)]
            ).astype(jnp.int32)
            m = jnp.clip(m, 0, last_heating_bin)
            x0 = heating_log10_T_ref[m]
            x1 = heating_log10_T_ref[m + 1]
            f0 = heating_factor_ref[m]
            f1 = heating_factor_ref[m + 1]
            x = jnp.clip(log_T, heating_log10_T_first, heating_log10_T_last)
            heating = heating_rate * (f0 + (x - x0) / (x1 - x0) * (f1 - f0))

            return (
                -cooling_rate * gamma_minus_one * density * prefactor
                + gamma_minus_one * heating
            )

        def equilibrium_root(T_old, rate_old, T_new, rate_new):
            safe_denominator = jnp.where(rate_new != rate_old, rate_new - rate_old, one)
            return T_old - rate_old * (T_new - T_old) / safe_denominator

        def count(mask):
            return jnp.sum(mask.astype(jnp.int32))

        # ---- sub-cycles: stop once no cell of the block has time left ----
        def subcycle_pending(carry):
            iteration, _, dt_left, _, _ = carry
            return (iteration < max_subcycles) & (count(dt_left > 0.0) > 0)

        def subcycle(carry):
            iteration, T, dt_left, T_prev, rate_prev = carry
            rate = net_rate(T)

            crossed = (dt_left > 0.0) & (rate * rate_prev < 0.0)
            T = jnp.where(crossed, equilibrium_root(T_prev, rate_prev, T, rate), T)
            dt_left = jnp.where(crossed, zero, dt_left)

            active = (
                (dt_left > 0.0)
                & (rate != 0.0)
                & ((T > floor_temperature) | (rate > 0.0))
            )
            safe_rate = jnp.where(active, rate, minus_one)
            dt_limit = max_fraction * T / jnp.abs(safe_rate)
            dt_sub = jnp.where(active, jnp.minimum(dt_left, dt_limit), zero)
            T_new = jnp.where(
                active, jnp.maximum(T + rate * dt_sub, floor_temperature), T
            )
            return (
                iteration + 1,
                T_new,
                jnp.where(active, dt_left - dt_sub, zero),
                T,
                jnp.where(active, rate, zero),
            )

        _, temperature, dt_left, temperature_prev, rate_prev = jax.lax.while_loop(
            subcycle_pending,
            subcycle,
            (
                jnp.int32(0),
                temperature,
                time_step,
                temperature,
                temperature - temperature,
            ),
        )

        # ---- a crossing in the last sub-cycle, then the residual time ----
        rate = net_rate(temperature)
        crossed = (dt_left > 0.0) & (rate * rate_prev < 0.0)
        temperature = jnp.where(
            crossed,
            equilibrium_root(temperature_prev, rate_prev, temperature, rate),
            temperature,
        )
        residual_active = (
            (dt_left > 0.0)
            & ~crossed
            & ((temperature > floor_temperature) | (rate > 0.0))
        )
        residual_temperature = jnp.maximum(
            temperature + rate * jnp.where(residual_active, dt_left, zero),
            floor_temperature,
        )
        residual_rate = net_rate(residual_temperature)

        # The bisection only matters in the cells with residual time.
        num_residual = count(residual_active)

        def bisection_pending(carry):
            iteration = carry[0]
            return (iteration < _NUM_BISECTIONS) & (num_residual > 0)

        def bisect(carry):
            iteration, T_a, rate_a, T_b, rate_b = carry
            T_m = jnp.sqrt(T_a * T_b)
            rate_m = net_rate(T_m)
            same_as_a = rate_m * rate_a > 0.0
            return (
                iteration + 1,
                jnp.where(same_as_a, T_m, T_a),
                jnp.where(same_as_a, rate_m, rate_a),
                jnp.where(same_as_a, T_b, T_m),
                jnp.where(same_as_a, rate_b, rate_m),
            )

        _, T_a, rate_a, T_b, rate_b = jax.lax.while_loop(
            bisection_pending,
            bisect,
            (jnp.int32(0), temperature, rate, residual_temperature, residual_rate),
        )
        residual_temperature = jnp.where(
            rate * residual_rate < 0.0,
            equilibrium_root(T_a, rate_a, T_b, rate_b),
            residual_temperature,
        )
        out_ref[0, ...] = jnp.where(residual_active, residual_temperature, temperature)

    kwargs = {}
    compiler_params = _pallas_compiler_params_from_backend(backend_config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, nx, ny, nz), dtype),
        grid=grid,
        in_specs=[fields_spec, *table_specs, *([scalar_spec] * len(scalars))],
        out_specs=out_spec,
        interpret=backend_config.pallas_interpret,
        name="subcycled_cooling_with_heating",
        **kwargs,
    )(fields, *tables, *scalars)


class _BackendOnly:
    """Minimal stand-in exposing ``backend_config`` like a SimulationConfig."""

    def __init__(self, backend_config):
        self.backend_config = backend_config


def _pallas_compiler_params_from_backend(backend_config):
    """The Triton compiler parameters of ``_pallas_compiler_params``."""
    return _pallas_compiler_params(_BackendOnly(backend_config))

