"""
Convenience functions for operations that combine multiple elements
of an array based on some stencil, e.g. b_i <- a_{i + 1} + a_{i - 1}.
Allows for code "closer to the math".
"""

# general
from functools import partial

# typing
from typing import Tuple, Union
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

# jax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

# astronomix
from astronomix._pallas_helpers import _current_pallas_mesh

try:
    from jax.shard_map import shard_map  # jax >= 0.8 (promoted out of experimental)
except ImportError:  # jax < 0.8
    from jax.experimental.shard_map import shard_map


def _sharded_roll_layout(input_array: jnp.ndarray, shift: int, axis: int):
    """``(mesh, pspec, mesh_axis_name, num_dev)`` when ``axis`` is split over the
    active device mesh and a neighbour exchange can do the roll, else ``None``.

    The layout is the solver's standard one (see ``_default_state_pspec``):
    a state is sharded ``(VARAXIS, XAXIS, YAXIS, ZAXIS)``, a single field
    ``(XAXIS, YAXIS, ZAXIS)``. Arrays of any other rank take the plain path.
    """
    mesh = _current_pallas_mesh()
    if mesh is None or mesh.size <= 1 or shift == 0:
        return None
    names = tuple(mesh.axis_names)
    if input_array.ndim == len(names):
        spec = names
    elif input_array.ndim == len(names) - 1:
        spec = names[1:]
    else:
        return None
    name = spec[axis % input_array.ndim]
    num_dev = mesh.shape[name]
    size = input_array.shape[axis]
    if num_dev <= 1 or size % num_dev or abs(shift) > size // num_dev:
        return None
    return mesh, PartitionSpec(*spec), name, num_dev


def _sharded_roll(input_array, shift, axis, mesh, pspec, name, num_dev):
    """The periodic roll as a per-shard roll plus a ``|shift|``-cell ppermute.

    GSPMD partitions the slice + concatenate of :func:`custom_roll` on a sharded
    axis by resharding the whole array (an all-to-all of the full local field on
    GPU), on every call. Shard ``k``'s rolled block is its own cells shifted by
    ``shift`` plus the ``|shift|`` edge cells of its ring neighbour, so this
    moves only that thin slab. Pure data movement: bit-identical to the
    unsharded roll.
    """

    def body(x):
        n = x.shape[axis]
        if shift > 0:
            edge = jax.lax.slice_in_dim(x, n - shift, n, axis=axis)
            perm = [(j, (j + 1) % num_dev) for j in range(num_dev)]
            halo = jax.lax.ppermute(edge, name, perm=perm)
            return jax.lax.concatenate(
                [halo, jax.lax.slice_in_dim(x, 0, n - shift, axis=axis)], axis
            )
        edge = jax.lax.slice_in_dim(x, 0, -shift, axis=axis)
        perm = [(j, (j - 1) % num_dev) for j in range(num_dev)]
        halo = jax.lax.ppermute(edge, name, perm=perm)
        return jax.lax.concatenate(
            [jax.lax.slice_in_dim(x, -shift, n, axis=axis), halo], axis
        )

    return shard_map(
        body, mesh=mesh, in_specs=(pspec,), out_specs=pspec, check_rep=False
    )(input_array)


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["shift", "axis"])
def custom_roll(input_array: jnp.ndarray, shift: int, axis: int) -> jnp.ndarray:
    """Periodic roll of ``input_array`` by ``shift`` along ``axis``.

    Equivalent to ``jnp.roll`` but expressed via two static slices and a
    concatenate, which keeps ``shift`` / ``axis`` compile-time constants so the
    stencil helpers built on top of it fuse cleanly. Under a multi-device mesh
    (``pallas_mesh_context``) a roll along a sharded axis is done as a
    neighbour exchange instead, see :func:`_sharded_roll`.

    Args:
        input_array: The array to roll.
        shift: The (signed) number of positions to roll by.
        axis: The axis along which to roll.

    Returns:
        The rolled array.
    """
    layout = _sharded_roll_layout(input_array, shift, axis)
    if layout is not None:
        return _sharded_roll(input_array, shift, axis, *layout)
    i = (-shift) % input_array.shape[axis]
    return jax.lax.concatenate(
        [
            jax.lax.slice_in_dim(input_array, i, input_array.shape[axis], axis=axis),
            jax.lax.slice_in_dim(input_array, 0, i, axis=axis),
        ],
        dimension=axis,
    )


def _shift(input_array: jnp.ndarray, shift: int, axis: int) -> jnp.ndarray:
    """Shift ``input_array`` by ``shift`` along ``axis``.

    A thin indirection over :func:`custom_roll`: the shift is currently periodic,
    but routing every stencil through this single entry point leaves room to
    support other boundary conditions later without touching call sites.
    """
    return custom_roll(input_array, shift, axis)

# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["indices", "axis"])
def _stencil_add(
    input_array: jnp.ndarray,
    indices: Tuple[int, ...],
    factors: Tuple[Union[float, Float[Array, ""]], ...],
    axis: int,
) -> jnp.ndarray:
    """
    Combines elements of an array additively
        output_i <- sum_j factors_j * input_array_{i + indices_j}

    Args:
        input_array: The array to operate on.
        indices: output_i <- sum_j factors_j * input_array_{i + indices_j}
        factors: output_i <- sum_j factors_j * input_array_{i + indices_j}
        axis: The axis along which to operate.

    Returns:
        output_i <- sum_j factors_j * input_array_{i + indices_j}
    """

    output = sum(
        factor * custom_roll(input_array, -index, axis=axis)
        for factor, index in zip(factors, indices)
    )

    return output
