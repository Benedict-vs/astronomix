"""Rank helpers for multi-process (multi-node) runs.

Under the one-process-per-GPU launch model every rank executes the same Python
program, so any host-side side effect — a print, a log append, a ``mkdir``, an
``rmtree`` — happens ``process_count`` times unless it is guarded. On a shared
filesystem with 32 ranks that turns a diagnostics log into interleaved garbage
and a startup cleanup into a race against the rank that is already writing.

Two helpers, so those guards read the same way everywhere::

    from astronomix.parallel import barrier, is_primary

    if is_primary():
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
    barrier("cgols:cleanup")

Both are no-ops in a single-process run (``is_primary()`` is ``True``,
``barrier`` returns immediately), so guarded code paths behave identically on
one GPU.

IMPORTANT: :func:`barrier` is a **collective** — every process must reach it,
in the same order, the same number of times. Never call it inside a
``is_primary()`` block, and never guard a collective (an Orbax
``save``/``restore``, a ``jax.lax`` reduction) with :func:`is_primary`; that
deadlocks every other rank.

``jax`` is imported lazily so that importing this module does not create the
JAX backend before ``jax.distributed.initialize()`` had its chance.
"""

from __future__ import annotations


def process_index() -> int:
    """This process's rank, ``0`` in a single-process run."""
    import jax

    return jax.process_index()


def process_count() -> int:
    """The number of JAX processes, ``1`` in a single-process run."""
    import jax

    return jax.process_count()


def is_primary() -> bool:
    """True on rank 0 — the one process that should perform host-side I/O.

    Always ``True`` in a single-process run, so a guarded block still executes
    on one GPU.
    """
    return process_index() == 0


def barrier(name: str) -> None:
    """Wait until every process has reached the barrier called ``name``.

    A no-op when running single-process. ``name`` is mandatory and must be
    unique per wait site (JAX matches barriers by key, so two different waits
    sharing a name can pair up the wrong ranks); include the step or iteration
    when a barrier fires more than once, e.g. ``f"cgols:prune:{step}"``.
    """
    import jax
    from jax.experimental.multihost_utils import sync_global_devices

    if jax.process_count() == 1:
        return
    sync_global_devices(name)


def broadcast_from_primary(value):
    """Return rank 0's ``value`` on every process (identity single-process).

    Used where every rank must agree on a host-side decision that only rank 0
    can make reliably — e.g. which checkpoint step to restore after a prune.
    This is a collective: call it on every process.
    """
    import jax
    from jax.experimental.multihost_utils import broadcast_one_to_all

    if jax.process_count() == 1:
        return value
    return broadcast_one_to_all(value)
