"""
Host-side progress bar for the time-integration loop.

Renders a single-line, terminal-width-aware progress bar that is driven from
inside the jitted loop via ``jax.debug.callback``. The "iteration" it is fed is
the simulation time, so the bar tracks progress towards ``t_end``.
"""

# general
import math
import os
import shutil
import sys

# Percent-bucket throttle for non-interactive stdout (slurm/batch log files).
# The in-place "\r" status line only works on a terminal; in a redirected log
# nothing is overwritten, so every step would append a full line and a long
# run floods the log with thousands of lines. When stdout is not a TTY, a
# plain newline-terminated line is emitted only when the completion fraction
# advances by ASTRONOMIX_STATUS_EVERY_PCT percent (default 1.0, i.e. at most
# ~100 status lines per run; <= 0 restores a line every step). Keyed per
# stream so the diagnostics line and the plain progress bar throttle
# independently.
_last_bucket = {}


def _advanced_a_bucket(fraction, key):
    """True when ``fraction`` entered a new percent bucket (or throttling is off)."""
    step = float(os.environ.get("ASTRONOMIX_STATUS_EVERY_PCT", "1"))
    if step <= 0:
        return True
    frac = float(fraction)
    frac = 1.0 if not math.isfinite(frac) else min(max(frac, 0.0), 1.0)
    bucket = int(100.0 * frac / step)
    if _last_bucket.get(key) == bucket:
        return False
    _last_bucket[key] = bucket
    return True


def _show_diagnostics(
    t, min_density, min_pressure, max_speed, max_temperature, has_nan,
    max_density=None, max_density_index=None,
    fraction=None,
) -> None:
    """Host-side per-step diagnostic line.

    Prints reduction scalars so a diverging run can be localised in time and by
    variable: which of density / pressure / speed / temperature degrades first,
    and when ``has_nan`` first trips.

    ``max_density`` (with its interior grid location ``max_density_index``)
    watches the one direction no positivity clip bounds: a density runaway is
    invisible in the min/max floors and ceilings above (they all saturate at
    their clip values), so it is reported explicitly.

    When the environment variable ``ASTRONOMIX_DIAG_LOG`` names a file, every
    diagnostics line is also appended there. The in-place status line keeps no
    history (each step overwrites the last), so the log is the only record of
    *when* a quantity first degraded after the run has moved on.

    When ``fraction`` (completion fraction in ``[0, 1]``) is given, a percentage
    is prepended to the line. This is used when ``progress_bar`` and
    ``monitor_diagnostics`` are both on: the diagnostics line and the separate
    progress bar would otherwise fight over the same in-place status line (the
    diagnostics, printed last each step, win and hide the bar), so instead of an
    animated bar the completion percentage rides along on the diagnostics line.

    On a terminal the line is rewritten in place (carriage return, padded to the
    terminal width) so successive steps update one status line instead of
    scrolling. The one exception is divergence: when ``has_nan`` trips the line
    is committed with a trailing newline so the crash point is preserved in the
    scrollback rather than overwritten by the next step.

    When stdout is NOT a terminal (e.g. an sbatch log file) the in-place
    rewrite is pointless and would append one line per step, so instead a
    normal line is printed only every ``ASTRONOMIX_STATUS_EVERY_PCT`` percent
    of completion (see ``_advanced_a_bucket``; needs ``fraction``, i.e. the
    ``progress_bar`` config flag — without it every step still prints). NaN
    lines and the per-step ``ASTRONOMIX_DIAG_LOG`` file are never throttled.
    """
    nan = bool(has_nan)
    flag = "  <-- NaN/inf!" if nan else ""

    # Optional completion percentage, shown in place of the progress bar. A
    # diverged run produces a non-finite time; clamp so the readout stays sane
    # and the real failure surfaces via the diagnostics / NaN flag instead.
    pct = ""
    if fraction is not None:
        frac = float(fraction)
        frac = 1.0 if not math.isfinite(frac) else min(max(frac, 0.0), 1.0)
        pct = f"[{100 * frac:5.1f}%] "

    max_rho = ""
    if max_density is not None:
        at = ""
        if max_density_index is not None:
            at = "@(" + ",".join(str(int(i)) for i in max_density_index) + ")"
        max_rho = f"  max_rho={float(max_density):.3e}{at}"

    msg = (
        f"{pct}[diag] t={float(t):.6e}  min_rho={float(min_density):.3e}  "
        f"min_P={float(min_pressure):.3e}  max|v|={float(max_speed):.3e}  "
        f"max_T(code)={float(max_temperature):.3e}{max_rho}{flag}"
    )

    log_path = os.environ.get("ASTRONOMIX_DIAG_LOG")
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    interactive = sys.stdout.isatty()
    if nan:
        # Commit the divergence line permanently (may wrap; that is fine).
        print(f"\r{msg}" if interactive else msg, flush=True)
    elif interactive:
        # Rewrite one status line in place, clipped/padded to the terminal width so
        # a previous, longer line is fully cleared and the line does not wrap.
        width = shutil.get_terminal_size((80, 20)).columns
        print(f"\r{msg[:width].ljust(width)}", end="", flush=True)
    elif fraction is None or _advanced_a_bucket(fraction, "diag"):
        # Batch log: plain scrolling lines, at most one per percent bucket.
        print(msg, flush=True)


def _show_progress(
    iteration, total, prefix="", suffix="", decimals=1, fill="█", printEnd="\r"
) -> None:
    """
    Render one frame of the progress bar, sized to the current terminal width.

    Args:
        iteration: The current progress value (the simulation time).
        total: The value of ``iteration`` at which the bar is full (``t_end``).
        prefix: Text printed before the bar.
        suffix: Text printed after the percentage.
        decimals: Number of decimal places shown in the percentage.
        fill: Character used for the filled portion of the bar.
        printEnd: Line terminator; ``"\\r"`` keeps overwriting the same line.
    """
    # On a blow-up the simulation time goes non-finite, and ``int(NaN)`` would
    # raise and abort the whole run. Clamp to ``total`` so the bar finishes
    # cleanly instead of crashing; the diagnostics elsewhere report the NaN.
    try:
        if not math.isfinite(float(iteration)):
            iteration = total
    except (TypeError, ValueError):
        iteration = total

    # Recompute the terminal width every frame so the bar keeps filling the
    # line correctly even if the terminal is resized mid-run.
    terminal_width = shutil.get_terminal_size((80, 20)).columns

    # A diverged simulation produces NaN/inf time. Don't crash the run inside the
    # host callback; flag it and clamp the fraction so the real failure surfaces
    # via the (NaN-filled) output rather than an opaque callback traceback.
    fraction = iteration / float(total)
    if not math.isfinite(fraction):
        suffix = (suffix + " [NaN/inf time -- simulation diverged]").strip()
        fraction = 1.0
    else:
        fraction = min(max(fraction, 0.0), 1.0)

    # Format percentage string
    percent = ("{0:." + str(decimals) + "f}").format(100 * fraction)

    # Batch log (stdout not a terminal): the animated in-place bar would append
    # one full line per step. Emit a plain percentage line per percent bucket
    # instead (see _advanced_a_bucket / ASTRONOMIX_STATUS_EVERY_PCT).
    if not sys.stdout.isatty():
        if _advanced_a_bucket(fraction, "bar"):
            print(f"{prefix} {percent}% {suffix}".strip(), flush=True)
        return

    # Size the bar so the whole line fits the terminal: subtract the fixed
    # decorations (prefix, suffix, percentage, separators) from the width, and
    # never shrink below a readable minimum.
    fixed_part = f"{prefix} | | {percent}% {suffix}"
    fixed_length = len(fixed_part)
    bar_length = max(10, terminal_width - fixed_length)

    # Compute filled length of the bar
    filledLength = int(bar_length * fraction)
    bar = fill * filledLength + "-" * (bar_length - filledLength)

    progress_line = f"{prefix} |{bar}| {percent}% {suffix}"

    # Pad the line out to the full terminal width so a shorter line never leaves
    # leftover characters from the previous, longer frame.
    padded_line = progress_line.ljust(terminal_width)

    print(f"\r{padded_line}", end=printEnd, flush=True)

    # Drop to a fresh line once the bar is full so subsequent output is clean.
    if iteration == total:
        print()
