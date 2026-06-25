import math
import shutil


def _show_diagnostics(
    t, min_density, min_pressure, max_speed, max_temperature, has_nan,
    fraction=None,
) -> None:
    """Host-side per-step diagnostic line.

    Prints reduction scalars so a diverging run can be localised in time and by
    variable: which of density / pressure / speed / temperature degrades first,
    and when ``has_nan`` first trips.

    When ``fraction`` (completion fraction in ``[0, 1]``) is given, a percentage
    is prepended to the line. This is used when ``progress_bar`` and
    ``monitor_diagnostics`` are both on: the diagnostics line and the separate
    progress bar would otherwise fight over the same in-place status line (the
    diagnostics, printed last each step, win and hide the bar), so instead of an
    animated bar the completion percentage rides along on the diagnostics line.

    The line is rewritten in place (carriage return, padded to the terminal width)
    so successive steps update one status line instead of scrolling. The one
    exception is divergence: when ``has_nan`` trips the line is committed with a
    trailing newline so the crash point is preserved in the scrollback rather than
    overwritten by the next step.
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

    msg = (
        f"{pct}[diag] t={float(t):.6e}  min_rho={float(min_density):.3e}  "
        f"min_P={float(min_pressure):.3e}  max|v|={float(max_speed):.3e}  "
        f"max_T(code)={float(max_temperature):.3e}{flag}"
    )
    width = shutil.get_terminal_size((80, 20)).columns
    if nan:
        # Commit the divergence line permanently (may wrap; that is fine).
        print(f"\r{msg}", flush=True)
    else:
        # Rewrite one status line in place, clipped/padded to the terminal width so
        # a previous, longer line is fully cleared and the line does not wrap.
        print(f"\r{msg[:width].ljust(width)}", end="", flush=True)


def _show_progress(
    iteration, total, prefix="", suffix="", decimals=1, fill="█", printEnd="\r"
) -> None:
    """
    Progress bar that adapts to terminal width and handles resizing.
    """
    # Get terminal width
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

    # Fixed parts (prefix + suffix + percent + " |" + "| " + spaces)
    fixed_part = f"{prefix} | | {percent}% {suffix}"
    fixed_length = len(fixed_part)

    # Compute bar length dynamically
    bar_length = max(10, terminal_width - fixed_length)

    # Compute filled length of the bar
    filledLength = int(bar_length * fraction)
    bar = fill * filledLength + "-" * (bar_length - filledLength)

    # Assemble full line
    progress_line = f"{prefix} |{bar}| {percent}% {suffix}"

    # Pad with spaces to ensure full overwrite (avoids leftovers)
    padded_line = progress_line.ljust(terminal_width)

    # Print progress line with carriage return
    print(f"\r{padded_line}", end=printEnd, flush=True)

    # Print newline when complete
    if iteration == total:
        print()
