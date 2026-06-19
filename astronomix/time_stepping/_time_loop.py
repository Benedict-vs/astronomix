"""Generic JAX time-integration loop driver.

Repo-agnostic: depends only on ``jax`` (and equinox's checkpointed while loop
for reverse-mode-friendly adaptive integration).  It owns the parts of a
time-stepping loop that are independent of the physics being integrated:

  * the loop backend — fixed-step ``fori_loop`` / adaptive ``while_loop`` /
    reverse-mode-friendly ``checkpointed_while_loop``;
  * collecting snapshots of the evolving state into preallocated buffers at
    chosen times (plus an optional final snapshot);
  * a host-side progress callback;
  * counting the number of steps taken.

The physics is supplied entirely through caller closures, so this module
carries no domain knowledge and can be reused by any project:

    step(t, state, snapshot_index) -> (dt, new_state)
        Advance ``state`` by one step.  Returns the timestep actually taken
        (the driver advances ``t``) and the new state.  ``snapshot_index`` is
        the current snapshot counter (0 when snapshots are disabled), passed
        through so the step can clamp ``dt`` to land on snapshot times.

    SnapshotSpec.record(t, state, store, index) -> store
        Write whatever diagnostics are wanted for snapshot ``index`` into the
        preallocated ``store`` pytree (typically ``store.field.at[index].set``).

    SnapshotSpec.should_record(t, index) -> bool
        Whether snapshot ``index`` is due at the start of a step at time ``t``
        (e.g. evenly spaced, or matching explicit timepoints).

``state`` and ``store`` are opaque pytrees to the driver.
"""

from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from equinox.internal._loop.checkpointed import checkpointed_while_loop

# ---- loop backends ----
FIXED_STEP = 0            # jax.lax.fori_loop over a fixed number of steps
ADAPTIVE_WHILE = 1        # jax.lax.while_loop until t >= t_end (forward-mode AD)
ADAPTIVE_CHECKPOINTED = 2  # checkpointed_while_loop until t >= t_end (reverse-mode AD)


def times_close(t, target):
    """Float-precision-aware test that ``t`` has reached ``target``.

    The tolerance is scaled by the working float epsilon and the magnitude of
    ``target``, so the test behaves correctly in both float32 and float64.  A
    fixed absolute tolerance (e.g. ``1e-12``) silently fails in float32, where
    a step landing on ``target`` is typically only accurate to ~1e-7·|target|.
    """
    dtype = jnp.result_type(jnp.asarray(t), jnp.asarray(target))
    atol = 8.0 * jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(target), 1.0)
    return jnp.abs(t - target) <= atol


class SnapshotSpec(NamedTuple):
    """Bundles everything the driver needs to collect snapshots.

    Attributes:
        store: Preallocated output pytree the ``record`` callback writes into.
        record: ``record(t, state, store, index) -> store``.
        should_record: ``should_record(t, index) -> bool`` crossing test.
        record_final: Also record once when a step lands on ``t_end``.
    """

    store: Any
    record: Callable
    should_record: Callable
    record_final: bool = True


def integrate(
    state: Any,
    step: Callable,
    t_end,
    *,
    backend: int,
    num_steps: Optional[int] = None,
    num_checkpoints: Optional[int] = None,
    snapshots: Optional[SnapshotSpec] = None,
    progress: Optional[Callable] = None,
    monitor: Optional[Callable] = None,
):
    """Run a time-integration loop.

    Args:
        state: Initial evolving-state pytree (opaque to the driver).
        step: ``step(t, state, snapshot_index) -> (dt, new_state)``.
        t_end: Integration end time.
        backend: One of ``FIXED_STEP`` / ``ADAPTIVE_WHILE`` /
            ``ADAPTIVE_CHECKPOINTED``.
        num_steps: Number of steps for ``FIXED_STEP``.
        num_checkpoints: Checkpoint count for ``ADAPTIVE_CHECKPOINTED``.
        snapshots: A :class:`SnapshotSpec`, or ``None`` to disable collection.
        progress: ``progress(t, t_end)`` host callback, or ``None``.
        monitor: ``monitor(t, state)`` traced per-step hook (computes its own
            on-device reductions and offloads scalars via ``jax.debug.callback``
            internally), or ``None``.

    Returns:
        ``(t, state, store, num_iterations)``.  ``store`` is the (possibly
        updated) snapshot store, or ``None`` when snapshots are disabled.
    """
    has_snap = snapshots is not None

    def body(carry):
        if has_snap:
            t, state, idx, n_iter, store = carry

            # Record the snapshot that is due at the start of this step
            # (captures the state *before* this step's update).
            def _record(operand):
                store_, idx_ = operand
                return snapshots.record(t, state, store_, idx_), idx_ + 1

            store, idx = jax.lax.cond(
                snapshots.should_record(t, idx),
                _record,
                lambda operand: operand,
                (store, idx),
            )
        else:
            t, state, idx, n_iter = carry  # idx stays 0

        dt, state = step(t, state, idx)
        t = t + dt
        n_iter = n_iter + 1

        if has_snap and snapshots.record_final:
            def _record_final(operand):
                store_, idx_ = operand
                return snapshots.record(t, state, store_, idx_), idx_ + 1

            store, idx = jax.lax.cond(
                times_close(t, t_end),
                _record_final,
                lambda operand: operand,
                (store, idx),
            )

        if progress is not None:
            jax.debug.callback(progress, t, t_end)

        if monitor is not None:
            monitor(t, state)

        if has_snap:
            return (t, state, idx, n_iter, store)
        return (t, state, idx, n_iter)

    if has_snap:
        carry = (0.0, state, 0, 0, snapshots.store)
    else:
        carry = (0.0, state, 0, 0)

    if backend == FIXED_STEP:
        carry = jax.lax.fori_loop(0, num_steps, lambda _i, c: body(c), carry)
    elif backend == ADAPTIVE_WHILE:
        carry = jax.lax.while_loop(lambda c: c[0] < t_end, body, carry)
    elif backend == ADAPTIVE_CHECKPOINTED:
        carry = checkpointed_while_loop(
            lambda c: c[0] < t_end, body, carry, checkpoints=num_checkpoints
        )
    else:
        raise ValueError(f"Unknown loop backend: {backend}")

    if has_snap:
        t, state, _idx, n_iter, store = carry
        return t, state, store, n_iter
    t, state, _idx, n_iter = carry
    return t, state, None, n_iter
