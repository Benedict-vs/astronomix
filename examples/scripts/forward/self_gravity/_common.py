"""Shared helpers for the self-gravity methods-paper figures.

Every script in ``examples/scripts/forward/self_gravity`` imports the scheme definitions,
consistent labels/styles, the Pallas backend configuration and a couple of
plotting utilities from here, so the four figures stay visually and
terminologically consistent.

``astronomix`` is installed non-editably in site-packages, so run the scripts
with the repo on ``PYTHONPATH`` to pick up this worktree, e.g.

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/self_gravity/jeans_convergence.py

Each script caches its simulation output under ``data/`` (``jnp.savez``) and
regenerates the figure from that cache unless ``--rerun`` is passed, so plots
can be re-styled without re-running any simulation.
"""

# general
from pathlib import Path

# typing
from typing import NamedTuple, Optional

# numerics
import numpy as np

# astronomix constants
from astronomix import FINITE_DIFFERENCE
from astronomix.option_classes.simulation_config import (
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# --- consistent scheme naming across all gravity figures -------------------
# These are the three finite-difference self-gravity source-term schemes
# compared throughout the self-gravity section of the methods paper.
SCHEME_LABELS = {
    SIMPLE_SOURCE: "FD, simple source",
    SECOND_ORDER_CONSERVATIVE: "FD, flux-based source",
    FOURTH_ORDER_CONSERVATIVE: "FD, corrected flux-based source",
}


class Scheme(NamedTuple):
    """A self-gravity source-term scheme with its consistent plot styling.

    ``solver_mode`` defaults to finite difference (the three FD source-term
    schemes shared by every figure). ``label_override`` lets a figure add a
    differently-named scheme (e.g. the finite-volume comparison in the Evrard
    energy plot) without disturbing the shared FD ``SCHEME_LABELS``.
    """

    self_gravity_version: int
    marker: str
    linestyle: str
    color: str
    linewidth: float = 2.0
    solver_mode: int = FINITE_DIFFERENCE
    label_override: Optional[str] = None

    @property
    def label(self) -> str:
        """The legend label, honouring ``label_override`` when set."""
        return self.label_override or SCHEME_LABELS[self.self_gravity_version]


# Order and styling shared by all figures. Colours/markers are fixed per
# scheme so a reader can match curves across the four panels.
SCHEMES = [
    Scheme(SIMPLE_SOURCE, marker="o", linestyle=":", color="C0", linewidth=2.5),
    Scheme(SECOND_ORDER_CONSERVATIVE, marker="s", linestyle="-.", color="C1", linewidth=2.0),
    Scheme(FOURTH_ORDER_CONSERVATIVE, marker="^", linestyle="--", color="C2", linewidth=2.0),
]

# The scheme the convergence-order fit is reported for.
FIT_SCHEME_VERSION = FOURTH_ORDER_CONSERVATIVE


# --- Pallas backend --------------------------------------------------------
def block_shape_for(dims, cap=8) -> tuple:
    """Largest power-of-two block (per axis, capped) that divides ``dims``.

    A robust fallback when resolutions are not all multiples of 16 — keeps the
    Pallas WENO kernel engaged for any grid.

    Args:
        dims: The per-axis cell counts of the grid.
        cap: The maximum block size per axis.

    Returns:
        A tuple of per-axis block sizes, each the largest capped power of two
        that evenly divides the corresponding entry of ``dims``.
    """

    def largest_power_of_two_divisor(cells):
        block = 1
        while block * 2 <= cap and cells % (block * 2) == 0:
            block *= 2
        return block

    return tuple(largest_power_of_two_divisor(int(cells)) for cells in dims)


# --- analysis helpers ------------------------------------------------------
def fit_power_law(n, error, floor_factor=3.0):
    """Least-squares fit of ``error ~ A * N**p`` in log-log space.

    The fit is restricted to the pre-saturation range: points whose error sits
    above ``floor_factor * min(error)`` (so points that have flattened at the
    machine-precision / scheme floor are excluded from the order estimate).

    Args:
        n: The resolutions (cell counts) at which the error was measured.
        error: The measured error for each resolution.
        floor_factor: Multiple of the minimum error below which points are
            treated as saturated and excluded from the fit.

    Returns:
        ``(p, A, n_fit, error_fit)`` where ``p`` is the fitted exponent, ``A``
        the prefactor, and ``n_fit`` / ``error_fit`` are the points that
        entered the fit (sorted by ``n``).
    """
    n = np.asarray(n, dtype=float)
    error = np.asarray(error, dtype=float)
    order = np.argsort(n)
    n, error = n[order], error[order]

    floor = floor_factor * float(np.min(error))
    mask = error > floor
    if mask.sum() < 2:  # not enough unsaturated points; fall back to all
        mask = np.ones_like(error, dtype=bool)

    n_fit, error_fit = n[mask], error[mask]
    p, log_a = np.polyfit(np.log(n_fit), np.log(error_fit), 1)
    return float(p), float(np.exp(log_a)), n_fit, error_fit


def apply_paper_style():
    """A light, consistent matplotlib style for the gravity figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.markersize": 6,
        }
    )


def integer_log_ticks(ax, resolutions, axis="x"):
    """Label a log axis directly with the integer cell counts used.

    Args:
        ax: The matplotlib axes whose ticks are being set.
        resolutions: The cell counts to place as tick labels.
        axis: Which axis to relabel, either ``"x"`` or ``"y"``.
    """
    # Local import keeps matplotlib off the import path for callers that only
    # use the (matplotlib-free) analysis helpers.
    from matplotlib.ticker import NullLocator

    resolutions = sorted(int(r) for r in np.unique(np.asarray(resolutions)))
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(NullLocator())
    target.set_minor_locator(NullLocator())
    if axis == "x":
        ax.set_xticks(resolutions)
        ax.set_xticklabels([str(r) for r in resolutions])
    else:
        ax.set_yticks(resolutions)
        ax.set_yticklabels([str(r) for r in resolutions])
