# general
from contextlib import nullcontext
from types import NoneType
import jax
from jax.sharding import PartitionSpec
import jax.numpy as jnp


from typing import Union

# runtime debugging
from jax.experimental import checkify

# astronomix constants
from astronomix._finite_difference._state_evolution._evolve_state import _evolve_state_fd
from astronomix._finite_difference._timestep_estimation._timestep_estimator import _cfl_time_step_fd, _cfl_time_step_fd_hydro
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._pallas_helpers import pallas_mesh_context
from astronomix.data_classes.simulation_state_struct import StateStruct
from astronomix.option_classes.simulation_config import BACKWARDS, FINITE_DIFFERENCE, FINITE_VOLUME, FORWARDS, GHOST_CELLS, PERIODIC_ROLL, STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.data_classes.simulation_helper_data import (
    HelperData,
    _helper_data_requirements,
    _unpad_helper_data,
    get_helper_data,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.data_classes.simulation_snapshot_data import SnapshotData

# astronomix functions
from astronomix._finite_volume._state_evolution.evolve_state import _evolve_state_fv
from astronomix._modules._iteration_level_updates import _iteration_level_updates
from astronomix._finite_volume._timestep_estimation._timestep_estimator import (
    _cfl_time_step,
    _source_term_aware_time_step,
)
from astronomix._snapshotting._snapshot_diagnostics import (
    build_snapshot_store,
    record_snapshot,
)
from astronomix.time_stepping._utils import _pad, _unpad

# progress bar
from astronomix.time_stepping._progress_bar import _show_progress

# generic time-integration loop driver
from astronomix.time_stepping._time_loop import (
    ADAPTIVE_CHECKPOINTED,
    ADAPTIVE_WHILE,
    FIXED_STEP,
    SnapshotSpec,
    integrate,
    times_close,
)

# timing
from timeit import default_timer as timer


# @jaxtyped(typechecker=typechecker)
def time_integration(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    snapshot_callable = None,
    sharding: Union[NoneType, jax.NamedSharding] = None,
) -> Union[STATE_TYPE, SnapshotData]:
    """
    Integrate the fluid equations in time. For the options of
    the time integration see the simulation configuration and
    the simulation parameters.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        snapshot_callable: A callable which is called at certain time points
            if config.activate_snapshot_callback is True. The callable must
            have the signature
                callable(time: float, state: STATE_TYPE, registered_variables: RegisteredVariables) -> None
            and can be used to e.g. output the current state to disk or
            directly produce intermediate plots. Note that inside the callable,
            to pass data to memory, one must use
                jax.debug.callback(
                    function, args...
                )
            To avoid moving large amounts of data to the host, only pass
            the necessary data to the function in the jax.debug.callback call,
            e.g. only the slice or summary statistics you need.
        sharding: The sharding to use for the padded helper data. If None,
                  no sharding is applied.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots)
        either the final state of the fluid after the time
        integration of snapshots of the time evolution.

    """

    # Here we prepare everything for the actual time integration function,
    # _time_integration, which is jitted below. This includes setting up
    # runtime debugging via checkify if requested, printing the elapsed
    # time if requested, compiling the function for memory analysis if
    # requested, etc.

    # depending on the boundary handling, we might need to pad the state
    #  - for periodic boundaries implicitly enforced by only rolling arrays
    #    this is not necessary
    # Only build the helper-data fields actually consumed by the
    # active subsystems; the unpadded variant needed for snapshot
    # diagnostics is recovered by slicing the padded one inside the
    # update step (see _unpad_helper_data).
    requirements = _helper_data_requirements(config)
    helper_data_pad = get_helper_data(
        config,
        sharding,
        padded = config.boundary_handling != PERIODIC_ROLL,
        requirements = requirements,
    )

    # When the user supplies a multi-device sharding, pjit dispatch needs
    # every JIT input leaf to carry a sharding compatible with the target
    # mesh. SimulationParams has both Python-scalar fields (gamma, t_end,
    # C_cfl, ...) and size-(0,) placeholder arrays (the default
    # ``fixed_boundary_state``); JAX converts those into numpy 0-d /
    # empty arrays for the JIT call and pjit cannot infer a sharding for
    # them on a multi-device mesh, surfacing as
    # ``AttributeError: 'UnspecifiedValue' object has no attribute
    # '_addressable_device_assignment'`` at dispatch time. Promote every
    # leaf of ``params`` onto a fully-replicated NamedSharding on the
    # supplied mesh so pjit always sees a concrete sharding.
    if sharding is not None:
        replicated = jax.NamedSharding(sharding.mesh, PartitionSpec())
        params = jax.tree.map(
            lambda leaf: jax.device_put(leaf, replicated),
            params,
        )

    if config.donate_state:
        time_integration_jit = jax.jit(
            _time_integration,
            static_argnames=[
                "config",
                "registered_variables",
                "snapshot_callable"
            ],
            donate_argnames=["state"],
        )
    else:
        time_integration_jit = jax.jit(
            _time_integration,
            static_argnames=[
                "config",
                "registered_variables",
                "snapshot_callable"
            ],
        )

    if config.runtime_debugging:
        errors = (
            checkify.user_checks
            | checkify.index_checks
            | checkify.float_checks
            | checkify.nan_checks
            | checkify.div_checks
        )
        checked_integration = checkify.checkify(_time_integration, errors)

        err, final_state = checked_integration(
            primitive_state,
            config,
            params,
            registered_variables,
            helper_data_pad,
            snapshot_callable,
        )
        err.throw()

    else:
        memory_stats = None
        # Activate the user-provided mesh for every trace/compile of
        # ``_time_integration`` so any inner ``with_sharding_constraint``
        # calls (used to pin auxiliary scalar outputs to replicated
        # sharding) have a mesh to bind to.
        mesh_ctx = sharding.mesh if sharding is not None else nullcontext()
        # Multi-GPU Pallas: the Pallas kernels (WENO, divergence, positivity)
        # are opaque to GSPMD, so on a sharded input XLA would otherwise
        # all-gather the full state on every device before each
        # ``pallas_call``. ``pallas_mesh_context`` flips them into a
        # ``shard_map`` + ppermute halo-exchange shape instead, which is
        # the difference between ~0.95x and ~2x strong-scaling on FD
        # Pallas. The context only needs to be live while the JIT body is
        # traced; it is read by ``_pallas_call_sharded`` at trace time.
        pallas_mesh = sharding.mesh if sharding is not None else None
        if config.memory_analysis:
          with mesh_ctx, pallas_mesh_context(pallas_mesh):
            compiled_step = time_integration_jit.lower(
                primitive_state,
                config,
                params,
                registered_variables,
                helper_data_pad,
                snapshot_callable,
            ).compile()
            compiled_stats = compiled_step.memory_analysis()
            if compiled_stats is not None:
                # Calculate total memory usage including temporary storage,
                # arguments, and outputs (but excluding aliases)
                total = (
                    compiled_stats.temp_size_in_bytes
                    + compiled_stats.argument_size_in_bytes
                    + compiled_stats.output_size_in_bytes
                    - compiled_stats.alias_size_in_bytes
                )
                memory_stats = (
                    int(compiled_stats.temp_size_in_bytes),
                    int(compiled_stats.argument_size_in_bytes),
                    int(total),
                )
                print("=== Compiled memory usage PER DEVICE ===")
                print(
                    f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**2):.2f} MB"
                )
                print(
                    f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**2):.2f} MB"
                )
                print(f"Total size: {total / (1024**2):.2f} MB")
                print("========================================")

        if config.print_elapsed_time:
            if not config.memory_analysis:
                # compile the time integration function
                with mesh_ctx, pallas_mesh_context(pallas_mesh):
                    time_integration_jit.lower(
                        primitive_state,
                        config,
                        params,
                        registered_variables,
                        helper_data_pad,
                        snapshot_callable,
                    ).compile()

            start_time = timer()
            print("🚀 Starting simulation...")

        with mesh_ctx, pallas_mesh_context(pallas_mesh):
            final_state = time_integration_jit(
                primitive_state,
                config,
                params,
                registered_variables,
                helper_data_pad,
                snapshot_callable,
            )

        # For certain backend/size combinations (notably FD JAX at large
        # N with a multi-device mesh) pjit returns some scalar/auxiliary
        # output leaves with an ``UnspecifiedValue`` sharding. Their
        # device buffers are valid; the wrapper just never bound a
        # public Sharding, and every host-side accessor
        # (``is_fully_replicated``, ``is_fully_addressable``,
        # ``_value``) then crashes. Rebuild each such leaf as a regular
        # single-device array by going through its underlying per-device
        # buffer.
        if sharding is not None:
            from jax._src.sharding_impls import UnspecifiedValue as _Unspec

            def _force_concrete(leaf):
                if isinstance(leaf, jax.Array) and isinstance(leaf.sharding, _Unspec):
                    return jnp.asarray(leaf._arrays[0])
                return leaf

            final_state = jax.tree.map(_force_concrete, final_state)

        if config.print_elapsed_time:
            if config.return_snapshots and config.snapshot_settings.return_final_state:
                final_state.final_state.block_until_ready()
            else:
                final_state.block_until_ready()
            end_time = timer()
            print("🏁 Simulation finished!")
            print(f"⏱️ Time elapsed: {end_time - start_time:.2f} seconds")
            if config.return_snapshots:
                num_iterations = final_state.num_iterations
                print(f"🔄 Number of iterations: {num_iterations}")
                # print the time per iteration
                print(
                    f"⏱️ / 🔄 time per iteration: {(end_time - start_time) / num_iterations} seconds"
                )
                final_state = final_state._replace(runtime=end_time - start_time)

        if memory_stats is not None and config.return_snapshots:
            temp_b, arg_b, total_b = memory_stats
            final_state = final_state._replace(
                temporary_memory_bytes=temp_b,
                argument_memory_bytes=arg_b,
                total_memory_bytes=total_b,
            )

    return final_state


def _time_integration(
    state: Union[STATE_TYPE, StateStruct],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data_pad: Union[HelperData, NoneType],
    snapshot_callable = None,
) -> Union[STATE_TYPE, StateStruct, SnapshotData]:
    """
    Time integration.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots)
        either the final state of the fluid after the time integration
        of snapshots of the time evolution.
    """

    # in simulations, where we also follow e.g. star particles,
    # the state may be a struct containing the primitive state
    # and the star particle data
    if config.state_struct:
        primitive_state = state.primitive_state
    else:
        primitive_state = state

    # we must pad the state with ghost cells
    # pad the primitive state with two ghost cells on each side
    # to account for the periodic boundary conditions
    original_shape = primitive_state.shape

    if config.boundary_handling != PERIODIC_ROLL:
        primitive_state = _pad(primitive_state, config)

    if config.boundary_handling == GHOST_CELLS:
        # important for active boundaries influencing
        # the time step criterion for now only gas state
        if config.mhd:
            primitive_state = primitive_state.at[:-3, ...].set(
                _boundary_handler(primitive_state[:-3, ...], config, registered_variables, params)
            )
        else:
            primitive_state = _boundary_handler(primitive_state, config, registered_variables, params)

    # -------------------------------------------------------------
    # =============== ↓ Setup of the snapshot array ↓ =============
    # -------------------------------------------------------------

    # In case the user requests the fluid state (or given
    # statistics) at certain time points (and not only a
    # final state at the end), we have to set up the arrays
    # to store this data.

    # The maximum timestep is also limited by the number of
    # snapshots we want to take.
    if config.return_snapshots:
        params = params._replace(
            dt_max=jnp.minimum(params.dt_max, params.t_end / config.num_snapshots)
        )

    if config.return_snapshots:
        snapshot_data = build_snapshot_store(
            config, config.num_snapshots, original_shape
        )
    elif config.activate_snapshot_callback:
        snapshot_data = SnapshotData(current_checkpoint=0)

    # -------------------------------------------------------------
    # =============== ↑ Setup of the snapshot array ↑ =============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================ ↓ step / record closures ↓ =================
    # -------------------------------------------------------------

    # The physics-specific pieces handed to the generic loop driver
    # (``astronomix.time_stepping._time_loop.integrate``): a ``step`` that
    # advances the state by one (adaptive) timestep, and — when snapshots are
    # requested — a recorder plus the predicate that decides when to record.

    def _step(time, state, snapshot_index):
        """Advance the state by one timestep.

        Estimates ``dt``, clamps it to land on the next snapshot time / the
        end time, runs the per-step modules and evolves the state.  Returns
        ``(dt, new_state)``; the driver advances the time.
        """
        key, primitive_state = state

        # determine the time step size
        if not config.fixed_timestep:
            if config.solver_mode == FINITE_VOLUME:
                if config.source_term_aware_timestep:
                    dt = jax.lax.stop_gradient(
                        _source_term_aware_time_step(
                            primitive_state, config, params, helper_data_pad,
                            registered_variables, time,
                        )
                    )
                else:
                    dt = jax.lax.stop_gradient(
                        _cfl_time_step(
                            primitive_state, config, params, registered_variables,
                        )
                    )
            elif config.solver_mode == FINITE_DIFFERENCE:
                if config.mhd:
                    dt = jax.lax.stop_gradient(
                        _cfl_time_step_fd(
                            primitive_state, config.grid_spacing, params.dt_max,
                            params.gamma, config, params, registered_variables,
                            params.C_cfl,
                        )
                    )
                else:
                    dt = jax.lax.stop_gradient(
                        _cfl_time_step_fd_hydro(
                            primitive_state, config.grid_spacing, params.dt_max,
                            params.gamma, config, params, registered_variables,
                            params.C_cfl,
                        )
                    )
        else:
            dt = params.t_end / config.num_timesteps

        # make sure we exactly hit the snapshot time points
        if config.use_specific_snapshot_timepoints and (
            config.return_snapshots or config.activate_snapshot_callback
        ):
            dt = jnp.minimum(
                dt, params.snapshot_timepoints[snapshot_index] - time
            )

        # make sure we exactly hit the end time
        if config.exact_end_time and not config.use_specific_snapshot_timepoints:
            dt = jnp.minimum(dt, params.t_end - time)

        # modules that run every time step
        key, primitive_state = _iteration_level_updates(
            primitive_state, key, dt, config, params, helper_data_pad,
            registered_variables, time + dt,
        )

        # evolve the state
        if config.solver_mode == FINITE_VOLUME:
            primitive_state = _evolve_state_fv(
                primitive_state, dt, params.gamma, config, params,
                helper_data_pad, registered_variables,
            )
        elif config.solver_mode == FINITE_DIFFERENCE:
            primitive_state = _evolve_state_fd(
                primitive_state, dt, params.gamma, config, params,
                helper_data_pad, registered_variables, time,
            )

        return dt, (key, primitive_state)

    def _record_snapshot(time, state, store, idx):
        """Record snapshot ``idx`` (the requested diagnostics)."""
        _, primitive_state = state

        if config.boundary_handling != PERIODIC_ROLL:
            unpad_primitive_state = _unpad(primitive_state, config)
        else:
            unpad_primitive_state = primitive_state

        # Recover the unpadded helper data by slicing — free under jit.
        helper_data_unpad = _unpad_helper_data(helper_data_pad, config)

        return record_snapshot(
            store,
            idx,
            time,
            unpad_primitive_state,
            helper_data_unpad,
            params,
            config,
            registered_variables,
        )

    def _should_record_snapshot(time, idx):
        """Whether snapshot ``idx`` is due at the start of a step at ``time``."""
        if config.use_specific_snapshot_timepoints:
            return times_close(time, params.snapshot_timepoints[idx])
        return time >= idx * params.t_end / config.num_snapshots

    def _record_callback(time, state, store, _idx):
        """Snapshot recorder for ``activate_snapshot_callback``: invoke the
        user callable; no preallocated buffers are written.

        NOTE: to pass data to the host, the callable must use
        ``jax.debug.callback`` internally, and should only pass the slice /
        summary statistics actually needed to avoid moving large arrays.
        """
        _, primitive_state = state
        snapshot_callable(time, primitive_state, registered_variables)
        return store

    def _should_record_callback(time, idx):
        return time >= idx * params.t_end / config.num_snapshots

    # -------------------------------------------------------------
    # ================ ↑ step / record closures ↑ =================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =================== ↓ loop-level logic ↓ ====================
    # -------------------------------------------------------------

    # Assemble the snapshot collection (when requested) and pick the loop
    # backend, then hand it all to the generic time-loop driver.

    if config.return_snapshots:
        snapshot_spec = SnapshotSpec(
            store=snapshot_data,
            record=_record_snapshot,
            should_record=_should_record_snapshot,
            record_final=True,
        )
    elif config.activate_snapshot_callback:
        snapshot_spec = SnapshotSpec(
            store=snapshot_data,
            record=_record_callback,
            should_record=_should_record_callback,
            record_final=True,
        )
    else:
        snapshot_spec = None

    # Fixed-step runs use a plain fori_loop; adaptive runs use a while loop,
    # checkpointed for reverse-mode differentiability.
    if config.fixed_timestep:
        backend = FIXED_STEP
        num_steps = config.num_timesteps
        num_checkpoints = None
    elif config.differentiation_mode == BACKWARDS:
        backend = ADAPTIVE_CHECKPOINTED
        num_steps = None
        num_checkpoints = config.num_checkpoints
    elif config.differentiation_mode == FORWARDS:
        backend = ADAPTIVE_WHILE
        num_steps = None
        num_checkpoints = None
    else:
        raise ValueError("Unknown differentiation mode.")

    initial_loop_state = (jax.random.key(config.random_seed), primitive_state)

    _, loop_state, snapshot_store, num_iterations = integrate(
        initial_loop_state,
        _step,
        params.t_end,
        backend=backend,
        num_steps=num_steps,
        num_checkpoints=num_checkpoints,
        snapshots=snapshot_spec,
        progress=_show_progress if config.progress_bar else None,
    )

    _, primitive_state = loop_state

    # -------------------------------------------------------------
    # =================== ↑ loop-level logic ↑ ====================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ===================== ↓ return logic ↓ ======================
    # -------------------------------------------------------------

    # Finally, we need to unpack the results from the loops and
    # return them in the appropriate format.

    if config.return_snapshots:
        snapshot_data = snapshot_store._replace(num_iterations=num_iterations)
        if config.snapshot_settings.return_final_state:
            if config.boundary_handling != PERIODIC_ROLL:
                unpad_primitive_state = _unpad(primitive_state, config)
            else:
                unpad_primitive_state = primitive_state
            snapshot_data = snapshot_data._replace(final_state=unpad_primitive_state)
        return snapshot_data

    # No-snapshot path (also the snapshot-callback case): return the state.
    if config.boundary_handling != PERIODIC_ROLL:
        primitive_state = _unpad(primitive_state, config)

    if config.state_struct:
        return StateStruct(primitive_state=primitive_state)

    return primitive_state

    # -------------------------------------------------------------
    # ===================== ↑ return logic ↑ ======================
    # -------------------------------------------------------------