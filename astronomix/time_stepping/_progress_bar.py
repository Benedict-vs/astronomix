import math
import shutil


def _show_diagnostics(t, min_density, min_pressure, max_speed, max_temperature, has_nan) -> None:
    """Host-side per-step diagnostic line.

    Prints reduction scalars so a diverging run can be localised in time and by
    variable: which of density / pressure / speed / temperature degrades first,
    and when ``has_nan`` first trips. Printed on its own line (newline, not a
    carriage return) so it does not fight the progress bar for the terminal.
    """
    flag = "  <-- NaN/inf!" if bool(has_nan) else ""
    print(
        f"\n[diag] t={float(t):.6e}  min_rho={float(min_density):.3e}  "
        f"min_P={float(min_pressure):.3e}  max|v|={float(max_speed):.3e}  "
        f"max_T(code)={float(max_temperature):.3e}{flag}",
        flush=True,
    )


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
