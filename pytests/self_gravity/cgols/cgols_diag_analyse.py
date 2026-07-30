"""Analyse an ASTRONOMIX_DIAG_LOG history: find and localise a density runaway.

Usage:
    python cgols_diag_analyse.py cgols_logs/cgols_diag_forensic512.log [--window 40 47]

Prints the max_rho growth curve (per ~0.1 Myr bin), flags where growth turns
exponential, and reports the argmax grid location (interior indices) with its
physical (x, y, z) position, assuming the cgols box (10x10x20 kpc, z = 2x).
No GPU / astronomix import needed — pure log parsing.
"""

import argparse
import re

import numpy as np

MYR_PER_CODE = 9.784  # 1 kpc / (100 km/s)

PAT = re.compile(
    r"t=([0-9.e+-]+)\s+min_rho=([0-9.e+-]+)\s+min_P=([0-9.e+-]+)\s+"
    r"max\|v\|=([0-9.e+-]+)\s+max_T\(code\)=([0-9.e+-]+)\s+"
    r"max_rho=([0-9.e+-]+|nan|inf)@\(([-0-9,]+)\)"
)


def parse(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                rows.append(
                    (
                        float(m.group(1)) * MYR_PER_CODE,  # t [Myr]
                        float(m.group(6)),                  # max_rho [code]
                        float(m.group(4)),                  # max |v|
                        float(m.group(5)),                  # max T (code)
                        tuple(int(i) for i in m.group(7).split(",")),
                    )
                )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--window", nargs=2, type=float, default=None,
                    metavar=("T0", "T1"), help="Myr window to detail")
    ap.add_argument("--dim", type=int, default=512, help="x/y cells (z is 2x)")
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        raise SystemExit("no parseable diag lines (is this a max_rho-enabled log?)")
    t = np.array([r[0] for r in rows])
    rho = np.array([r[1] for r in rows])
    print(f"{len(rows)} diag lines, t = {t[0]:.2f} .. {t[-1]:.2f} Myr, "
          f"max_rho {rho.min():.3e} .. {rho.max():.3e} code")

    # growth-onset scan: first time max_rho exceeds successive decade thresholds
    base = np.median(rho[: max(len(rho) // 10, 1)])
    print(f"\nearly-run baseline max_rho ~ {base:.3e} code; decade crossings:")
    for mult in (3, 10, 30, 100, 1e3, 1e4, 1e5):
        idx = np.argmax(rho > base * mult)
        if rho[idx] > base * mult:
            r = rows[idx]
            print(f"  > {mult:>7g} x baseline at t = {r[0]:7.3f} Myr, "
                  f"max_rho = {r[1]:.3e} @ {r[4]}")

    lo, hi_t = args.window if args.window else (t[-1] - 5, t[-1])
    sel = [r for r in rows if lo <= r[0] <= hi_t]
    if sel:
        print(f"\ndetail window {lo:.1f}-{hi_t:.1f} Myr "
              f"(one line per ~{max((hi_t-lo)/40, 1e-9):.3f} Myr):")
        dx = 10.0 / args.dim
        step = max(len(sel) // 40, 1)
        for r in sel[::step]:
            i, j, k = r[4]
            x, y = (i + 0.5) * dx - 5, (j + 0.5) * dx - 5
            z = (k + 0.5) * dx - 10
            print(f"  t={r[0]:8.3f}  max_rho={r[1]:.3e}  max|v|={r[2]:8.2f}  "
                  f"max_T={r[3]:7.1f}  @ ({i},{j},{k}) = ({x:+.2f},{y:+.2f},{z:+.2f}) kpc")


if __name__ == "__main__":
    main()
