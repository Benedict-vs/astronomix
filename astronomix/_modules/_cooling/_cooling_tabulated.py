r"""
Tabulated exact integration of cooling with heating
(``CoolingConfig.cooling_method = TABULATED_EXACT_COOLING``).

At fixed density and heating field the temperature of a cell obeys

    dT/dt = (gamma - 1) c rho [psi f(T) - Lambda(T)],    psi = h / (c rho),

with ``Lambda`` the cooling curve, ``h f(T)`` the heating rate, ``c`` the
density-convention prefactor and ``psi`` the only per-cell parameter besides
the scaled time ``tau = (gamma - 1) c rho dt``. Townsend (2009) integrates the
cooling-only case exactly through the temporal evolution function
``Y(T) = int dT / Lambda``; with heating the net rate ``g(T; psi) = psi f -
Lambda`` changes sign at the thermal equilibria, where ``Y`` diverges, and it
depends on ``psi``.

Here ``g`` is tabulated on a grid of temperatures (the nodes of the cooling
table, optionally refined) and of ``psi`` (log-spaced), and taken as linear in
T between two temperature nodes. For that model the solution is known in
closed form: inside a bin ``g`` grows or decays exponentially in time, a bin
without a sign change is crossed in a finite time, and a bin with a sign
change (an equilibrium) is either never left (stable) or left in a finite time
(unstable). The crossing times are accumulated per ``psi`` row into ``Y``
(float64 on the host, stored as a pair of float32), so a cell is advanced over
any ``tau`` by one in-bin evaluation, one binary search over the nodes it
passes, and one final in-bin evaluation - a fixed cost, no sub-cycles, no
overshoot of an equilibrium. The result of the two neighbouring ``psi`` rows
is interpolated in ``log psi`` (the solution is smooth in ``psi``; ``Y`` is
not).
"""

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp

# numerics
import numpy as np


class TabulatedCoolingParams(NamedTuple):
    """Tables of the tabulated exact cooling integrator (see the module docstring).

    Built by :func:`build_tabulated_cooling`; the defaults are placeholders.
    """

    #: Temperature nodes (rescaled temperature, ascending), shape (N_T,).
    temperature_nodes: jnp.ndarray = jnp.ones((2,))
    #: log10 of the first node and the (uniform) log10 spacing of the nodes.
    log10_T_first: float = 0.0
    log10_T_spacing: float = 1.0
    #: log10 of the first psi row and the log10 spacing of the rows.
    log10_psi_first: float = 0.0
    log10_psi_spacing: float = 1.0
    #: Net rate g = psi f - Lambda at the nodes, shape (N_psi, N_T).
    net_rate: jnp.ndarray = jnp.zeros((2, 2))
    #: Cumulative crossing time Y of the bins without a sign change, as the
    #: leading and the trailing float32 of the float64 value, (N_psi, N_T).
    crossing_time_high: jnp.ndarray = jnp.zeros((2, 2))
    crossing_time_low: jnp.ndarray = jnp.zeros((2, 2))
    #: For node c: the lowest bin with a sign change at or above c (N_T - 1 if
    #: none) and the highest one below c (-1 if none), shape (N_psi, N_T).
    next_root_bin_above: jnp.ndarray = jnp.zeros((2, 2), dtype=jnp.int32)
    previous_root_bin_below: jnp.ndarray = jnp.zeros((2, 2), dtype=jnp.int32)


def build_tabulated_cooling(
    temperature_nodes,
    cooling_rate_nodes,
    heating_factor_nodes,
    psi_minimum,
    psi_maximum,
    num_psi=512,
    dtype=np.float32,
) -> TabulatedCoolingParams:
    """
    Build the tables of the tabulated exact cooling integrator (host side).

    Args:
        temperature_nodes: The temperature nodes, evenly spaced in log10,
            ascending. Cells outside their range are left unchanged.
        cooling_rate_nodes: Lambda at the nodes, in the units the cooling
            module uses (``_cooling_rate``).
        heating_factor_nodes: The heating factor f at the nodes (ones for a
            temperature-independent heating).
        psi_minimum: Lowest tabulated ``psi = heating_rate / (c rho)``; lower
            values use this row.
        psi_maximum: Highest tabulated psi; higher values use this row.
        num_psi: The number of (log-spaced) psi rows.
        dtype: The floating-point type of the simulation.

    Returns:
        The tables.
    """
    nodes = np.asarray(temperature_nodes, dtype=np.float64)
    num_nodes = nodes.shape[0]
    log10_psi = np.linspace(np.log10(psi_minimum), np.log10(psi_maximum), num_psi)
    # The rate the solver sees is the rounded one; classify the bins with it.
    net_rate = (
        (10.0**log10_psi)[:, None] * np.asarray(heating_factor_nodes, np.float64)[None, :]
        - np.asarray(cooling_rate_nodes, np.float64)[None, :]
    ).astype(dtype)
    nodes_rounded = nodes.astype(dtype).astype(np.float64)
    g = net_rate.astype(np.float64)

    g_low, g_high = g[:, :-1], g[:, 1:]
    regular = g_low * g_high > 0.0  # no sign change, no zero at a node
    width = np.diff(nodes_rounded)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(regular, (g_high - g_low) / g_low, 0.0)
        log_term = np.where(np.abs(x) < 1e-8, 1.0 - 0.5 * x, np.log1p(x) / np.where(x == 0.0, 1.0, x))
        crossing = np.where(regular, width / np.where(regular, g_low, 1.0) * log_term, 0.0)
    crossing_time = np.concatenate(
        [np.zeros((num_psi, 1)), np.cumsum(crossing, axis=1)], axis=1
    )
    high = crossing_time.astype(dtype)
    low = (crossing_time - high.astype(np.float64)).astype(dtype)

    # next_root_bin_above[c]: smallest root bin b >= c (else N_T - 1);
    # previous_root_bin_below[c]: largest root bin b <= c - 1 (else -1).
    bins = np.arange(num_nodes - 1)[None, :]
    root_or_big = np.where(~regular, bins, num_nodes - 1)
    above = np.minimum.accumulate(root_or_big[:, ::-1], axis=1)[:, ::-1]
    next_root_bin_above = np.concatenate(
        [above, np.full((num_psi, 1), num_nodes - 1)], axis=1
    )
    root_or_small = np.where(~regular, bins, -1)
    below = np.maximum.accumulate(root_or_small, axis=1)
    previous_root_bin_below = np.concatenate([np.full((num_psi, 1), -1), below], axis=1)

    log10_nodes = np.log10(nodes)
    return TabulatedCoolingParams(
        temperature_nodes=jnp.asarray(nodes, dtype=dtype),
        log10_T_first=float(log10_nodes[0]),
        log10_T_spacing=float((log10_nodes[-1] - log10_nodes[0]) / (num_nodes - 1)),
        log10_psi_first=float(log10_psi[0]),
        log10_psi_spacing=float(log10_psi[1] - log10_psi[0]),
        net_rate=jnp.asarray(net_rate),
        crossing_time_high=jnp.asarray(high),
        crossing_time_low=jnp.asarray(low),
        next_root_bin_above=jnp.asarray(next_root_bin_above, dtype=jnp.int32),
        previous_root_bin_below=jnp.asarray(previous_root_bin_below, dtype=jnp.int32),
    )


def _exponential_factor(x):
    """(exp(x) - 1) / x, with the limit 1 at x = 0."""
    x = jnp.minimum(x, 60.0)
    safe = jnp.where(jnp.abs(x) < 1e-4, 1.0, x)
    return jnp.where(jnp.abs(x) < 1e-4, 1.0 + 0.5 * x, jnp.expm1(safe) / safe)


def _logarithm_factor(x):
    """log(1 + x) / x, with the limit 1 at x = 0 (x > -1)."""
    safe = jnp.where(jnp.abs(x) < 1e-4, 1.0, x)
    return jnp.where(jnp.abs(x) < 1e-4, 1.0 - 0.5 * x, jnp.log1p(safe) / safe)


def _advance_on_row(row, temperature, scaled_time, tables: TabulatedCoolingParams):
    """Advance ``temperature`` over ``scaled_time`` on the psi rows ``row``."""
    nodes = tables.temperature_nodes
    num_nodes = nodes.shape[0]
    num_search_steps = int(np.ceil(np.log2(num_nodes))) + 1
    offset = row * num_nodes

    def node_value(table, index):
        return jnp.take(table.reshape(-1), offset + index)

    def node_temperature(index):
        return jnp.take(nodes, index)

    T = jnp.clip(temperature, nodes[0], nodes[-1])

    # The bin of T (as _uniform_log_bin: the estimate, then one comparison
    # each way against the nodes).
    j = jnp.floor(
        (jnp.log10(T) - tables.log10_T_first) / tables.log10_T_spacing
    ).astype(jnp.int32)
    j = jnp.clip(j, 0, num_nodes - 2)
    j = j - (T < node_temperature(j)).astype(jnp.int32)
    j = j + (T >= node_temperature(jnp.clip(j + 1, 0, num_nodes - 1))).astype(jnp.int32)
    j = jnp.clip(j, 0, num_nodes - 2)

    T_low, T_high = node_temperature(j), node_temperature(j + 1)
    g_low, g_high = node_value(tables.net_rate, j), node_value(tables.net_rate, j + 1)
    slope = (g_high - g_low) / (T_high - T_low)
    g = g_low + (g_high - g_low) * ((T - T_low) / (T_high - T_low))

    upward = g > 0.0
    moving = g != 0.0
    g_safe = jnp.where(moving, g, 1.0)

    # ---- stay inside the starting bin? ----
    edge = j + upward.astype(jnp.int32)  # the node ahead
    T_edge = jnp.where(upward, T_high, T_low)
    g_edge = jnp.where(upward, g_high, g_low)
    reachable = g_edge * g > 0.0  # no equilibrium between T and the node ahead
    time_to_edge = jnp.where(
        reachable,
        (T_edge - T) / g_safe * _logarithm_factor(jnp.where(reachable, (g_edge - g) / g_safe, 0.0)),
        jnp.inf,
    )
    leaves_bin = scaled_time >= time_to_edge
    T_within = jnp.clip(
        T + g * scaled_time * _exponential_factor(slope * scaled_time), T_low, T_high
    )

    # ---- otherwise: the furthest node passed in the remaining time ----
    remaining = jnp.where(leaves_bin, scaled_time - time_to_edge, 0.0)
    limit_above = node_value(tables.next_root_bin_above, edge)
    limit_below = node_value(tables.previous_root_bin_below, edge)
    max_passed = jnp.where(upward, limit_above - edge, edge - (limit_below + 1))
    max_passed = jnp.maximum(max_passed, 0)
    step = jnp.where(upward, 1, -1)
    time_high_edge = node_value(tables.crossing_time_high, edge)
    time_low_edge = node_value(tables.crossing_time_low, edge)

    def elapsed(count):
        """Time from the node ahead to the node ``count`` nodes further on."""
        node = jnp.clip(edge + step * count, 0, num_nodes - 1)
        return jnp.maximum(
            (node_value(tables.crossing_time_high, node) - time_high_edge)
            + (node_value(tables.crossing_time_low, node) - time_low_edge),
            0.0,
        )

    low = jnp.zeros_like(max_passed)
    high = max_passed
    for _ in range(num_search_steps):
        middle = (low + high + 1) // 2
        passed = elapsed(middle) <= remaining
        low = jnp.where(passed, middle, low)
        high = jnp.where(passed, high, middle - 1)

    last_node = jnp.clip(edge + step * low, 0, num_nodes - 1)
    time_left = jnp.maximum(remaining - elapsed(low), 0.0)
    at_table_end = jnp.where(upward, last_node == num_nodes - 1, last_node == 0)
    final_bin = jnp.clip(jnp.where(upward, last_node, last_node - 1), 0, num_nodes - 2)
    T_final_low, T_final_high = node_temperature(final_bin), node_temperature(final_bin + 1)
    final_slope = (
        node_value(tables.net_rate, final_bin + 1) - node_value(tables.net_rate, final_bin)
    ) / (T_final_high - T_final_low)
    T_node = node_temperature(last_node)
    T_beyond = jnp.clip(
        T_node
        + node_value(tables.net_rate, last_node)
        * time_left
        * _exponential_factor(final_slope * time_left),
        T_final_low,
        T_final_high,
    )
    T_beyond = jnp.where(at_table_end, T_node, T_beyond)

    return jnp.where(leaves_bin, T_beyond, T_within)


def update_temperature_tabulated(
    density,
    temperature,
    time_step,
    gamma,
    prefactor,
    heating_rate,
    tables: TabulatedCoolingParams,
):
    """
    Advance the temperature over ``time_step`` with the tabulated exact
    integrator (see the module docstring).

    Args:
        density: The density field.
        temperature: The (rescaled) temperature field.
        time_step: The time step, a scalar or a per-cell field.
        gamma: The adiabatic index.
        prefactor: The density-convention prefactor of the cooling term
            (``_volumetric_cooling_prefactor``).
        heating_rate: The heating rate, scalar or per-cell field (the factor
            multiplying the tabulated ``f(T)``).
        tables: The tables of :func:`build_tabulated_cooling`.

    Returns:
        The temperature after ``time_step``. Cells outside the tabulated
        temperature range are returned unchanged.
    """
    nodes = tables.temperature_nodes
    num_psi = tables.net_rate.shape[0]
    scaled_time = (gamma - 1) * prefactor * density * time_step
    psi = heating_rate / (prefactor * density)

    position = (
        jnp.log10(jnp.maximum(psi, jnp.finfo(temperature.dtype).tiny))
        - tables.log10_psi_first
    ) / tables.log10_psi_spacing
    position = jnp.clip(position, 0.0, num_psi - 1.0)
    row = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, num_psi - 2)
    weight = position - row

    lower = _advance_on_row(row, temperature, scaled_time, tables)
    upper = _advance_on_row(row + 1, temperature, scaled_time, tables)
    advanced = jnp.exp((1.0 - weight) * jnp.log(lower) + weight * jnp.log(upper))

    inside = (temperature >= nodes[0]) & (temperature <= nodes[-1])
    return jnp.where(inside, advanced, temperature)
