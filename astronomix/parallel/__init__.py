"""Parallel / multi-process helpers for astronomix.

This subpackage is intentionally lightweight at import time: importing
``astronomix.parallel.distributed`` must not trigger JAX backend
initialisation, because ``jax.distributed.initialize()`` has to run before
the backend is created.  ``jax`` is therefore imported lazily inside the
functions that need it.
"""

from astronomix.parallel.distributed import DistInfo, init_distributed
from astronomix.parallel.rank import (
    barrier,
    broadcast_from_primary,
    is_primary,
    process_count,
    process_index,
)

__all__ = [
    "DistInfo",
    "barrier",
    "broadcast_from_primary",
    "init_distributed",
    "is_primary",
    "process_count",
    "process_index",
]
