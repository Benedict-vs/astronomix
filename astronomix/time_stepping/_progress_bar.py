"""
Host-side progress bar for the time-integration loop.

Renders a single-line, terminal-width-aware progress bar that is driven from
inside the jitted loop via ``jax.debug.callback``. The "iteration" it is fed is
the simulation time, so the bar tracks progress towards ``t_end``. When stdout
is not a terminal (queued or redirected runs) a throttled plain-text log is
written instead.
"""

# general
import math
import os
import shutil
import sys

# Rank guard for the host-side output below. Both the status line and the
# ASTRONOMIX_DIAG_LOG append are driven from inside the jitted loop via
# jax.debug.callback, which fires on EVERY process: at 32 ranks the log file
# becomes 32 interleaved copies of the same lines on a shared filesystem, and
# stdout is 32x duplicated. The values reported are global reductions and are
# therefore identical on every rank, so this is purely duplicated I/O and rank
# 0's copy is the complete record.
#
# Resolved on first use and then cached: the rank never changes during a run,
# and a per-callback jax.process_index() would sit in the per-step host path.
# Deliberately NOT resolved at import time — that would create the JAX backend
# as a side effect of importing astronomix, before a driver had the chance to
# call jax.distributed.initialize().
_is_primary_cache = []


def _is_primary() -> bool:
    """True on rank 0 (and always, single-process)."""
    if not _is_primary_cache:
        from astronomix.parallel import is_primary

        _is_primary_cache.append(is_primary())
    return _is_primary_cache[0]

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


# Host-side previous total mass, for the per-step mass change on the diag line.
_last_mass: dict = {}


def _show_diagnostics(
    t, min_density, min_pressure, max_speed, max_temperature, has_nan,
    max_density=None, max_density_index=None,
    fraction=None, total_mass=None,
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
    if not _is_primary():
        # Every rank runs this callback with the same (globally reduced)
        # values; only rank 0 writes them out. See the _is_primary note above.
        return

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

    # Total gas mass and its change since the previous step (code units). The
    # density floor adds mass when the update drives a cell negative; that shows
    # up as a jump in dM far above the steady wind loading / boundary outflow.
    mass = ""
    if total_mass is not None:
        m = float(total_mass)
        prev = _last_mass.get("m")
        _last_mass["m"] = m
        dm = f"{m - prev:+.3e}" if prev is not None else "n/a"
        mass = f"  mass={m:.7e} dM={dm}"

    msg = (
        f"{pct}[diag] t={float(t):.6e}  min_rho={float(min_density):.3e}  "
        f"min_P={float(min_pressure):.3e}  max|v|={float(max_speed):.3e}  "
        f"max_T(code)={float(max_temperature):.3e}{max_rho}{mass}{flag}"
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

import time

#: Longest wall-time gap, in seconds, between two lines of the plain-text log,
#: even when the progress percentage has not moved.
_HEARTBEAT_SECONDS = 60.0


class _PlainTextProgressLog:
    """
    The throttling state of the plain-text progress log of one run.

    The progress callback fires once per solver step, so the difference between
    consecutive simulation times IS the current ``dt``: a collapsing ``dt`` (the
    typical symptom of an unstable run) is visible directly in the log, and the
    wall-time heartbeat keeps emitting lines even when the percentage stalls, so
    a stalled run can never look identical to a slow one.

    Attributes:
        last_logged_bucket: The progress bucket (in 0.5 % increments) of the
            last emitted line, or ``None`` before the first line.
        last_logged_wall_time: The wall time of the last emitted line.
        last_simulation_time: The simulation time of the previous callback.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Forget the previous run, so the next callback starts a fresh log."""
        self.last_logged_bucket = None
        self.last_logged_wall_time = None
        self.last_simulation_time = None


#: The plain-text log state of the current run. ``time_integration`` resets it
#: at the start of every run; a run is also detected as new when the simulation
#: time goes backwards (e.g. a jitted function wrapping ``time_integration`` is
#: executed again).
_progress_log = _PlainTextProgressLog()


def _reset_progress_log() -> None:
    """Start a fresh plain-text progress log for a new run."""
    _progress_log.reset()


def _show_progress(
    iteration, total, prefix="", suffix="", decimals=1, fill="█", printEnd="\r"
) -> None:
    """
    Render one frame of the progress bar, sized to the current terminal width.

    When stdout is not a terminal, a carriage-return bar would flood the log
    with full-width frames, so a plain progress line is printed instead: one
    line per 0.5 % of progress, plus a heartbeat line at least every
    ``_HEARTBEAT_SECONDS``, each with the simulation time and the current
    per-step ``dt``. ``decimals``, ``fill`` and ``printEnd`` only apply to the
    terminal bar.

    Args:
        iteration: The current progress value (the simulation time).
        total: The value of ``iteration`` at which the bar is full (``t_end``).
        prefix: Text printed before the bar.
        suffix: Text printed after the percentage.
        decimals: Number of decimal places shown in the percentage.
        fill: Character used for the filled portion of the bar.
        printEnd: Line terminator; ``"\\r"`` keeps overwriting the same line.
    """
    if not _is_primary():
        # One bar, not process_count of them interleaved. See _is_primary above.
        return

    # On a blow-up the simulation time goes non-finite, and ``int(NaN)`` would
    # raise and abort the whole run. Clamp to ``total`` so the bar finishes
    # cleanly instead of crashing; the diagnostics elsewhere report the NaN.
    try:
        time_is_finite = math.isfinite(float(iteration))
    except (TypeError, ValueError):
        time_is_finite = False
    if not time_is_finite:
        iteration = total

    if not sys.stdout.isatty():
        current_time = float(iteration)
        previous_time = _progress_log.last_simulation_time
        if previous_time is not None and current_time < previous_time:
            # The clock went backwards: this callback belongs to a new run.
            _progress_log.reset()
            previous_time = None
        # The step size is inferred from consecutive callbacks; it is unknown
        # for the first callback of a run and meaningless for a clamped NaN time.
        if previous_time is None or not time_is_finite:
            dt = None
        else:
            dt = current_time - previous_time
        _progress_log.last_simulation_time = current_time

        percent = 100.0 * current_time / float(total)
        half_percent_bucket = int(percent * 2)
        wall_time = time.monotonic()
        heartbeat_due = (
            _progress_log.last_logged_wall_time is None
            or wall_time - _progress_log.last_logged_wall_time >= _HEARTBEAT_SECONDS
        )
        if half_percent_bucket == _progress_log.last_logged_bucket and not heartbeat_due:
            return
        _progress_log.last_logged_bucket = half_percent_bucket
        _progress_log.last_logged_wall_time = wall_time
        dt_note = "" if dt is None else f"  dt = {dt:.3e}"
        print(
            f"{prefix}progress {percent:5.1f}%  "
            f"t = {current_time:.6g} / {float(total):.6g}{dt_note} {suffix}".rstrip(),
            flush=True,
        )
        return

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
