"""
Self-convergence of each scheme and the cheapest run against an unbiased reference (asgard-papers #11, items 1, 3, 5).

cgols_cost_accuracy.py judges every run against R = the mean of FD-2048 and FV-2048. Where those two disagree, a
scheme that converges early towards its own 2048 run is close to R by construction, so "converges earlier" is not
yet "more accurate". This script separates the two questions, on the same quantities, values and noise estimates
(cgols_cost_accuracy.collect: synthesis cache + extra cache + structure3d, sigma from the seed runs):

1. Self-convergence per scheme. Error of a run against its own scheme's 2048 run, e_self = (x - x_2048) / |x_2048|
   (percentage points for the CC85 offsets, as in cost_accuracy). A scheme is self-converged at tolerance tol from
   resolution N on if every run from N up to 1024 has |e_self| <= tol ("nominal"), or |e_self| + sigma_pair <= tol
   with sigma_pair = hypot(sigma_N, sigma_2048) ("confident"). Observed order p = log2(d1 / d2) of the last three
   rungs that change monotonically with the first step above 2 sigma (512-1024-2048, else 256-512-1024); the
   scheme's Richardson limit x_2048 + d2 / (2^p - 1) where the 512-1024-2048 ladder qualifies.
3. Seed noise at the top. z_top = (FD-2048 - FV-2048) / hypot(sigma_FD2048, sigma_FV2048), sigma from the seed
   runs (FV-2048 measured with seed 1; FD-2048 extrapolated from FD-1024 and its seed 1). |z_top| < 2: the two
   finest runs agree within their noise, the difference is not a scheme effect.
5. Cheapest run against an unbiased reference, per quantity:
   - |z_top| < 2: R = the mean of FD-2048 and FV-2048, u = |FD - FV| / 2 (as cost_accuracy);
   - else, if both schemes have a Richardson limit and the two agree within 5 % of R: R = their mean, u = half
     their difference;
   - else the finest runs differ by a scheme effect: R = their mean is biased by up to u = |FD - FV| / 2, so the
     quantity only counts at tolerances above u (as in cost_accuracy), and it is labelled "2048 differ". Where u
     is above every tolerance, no unbiased reference exists yet (the quantities a higher-resolution reference run
     would decide; asgard-papers #11 item 2); these are listed, not judged.
   Per family and tolerance the cheapest run (H200 GPU-hours to 75 Myr) whose worst error over the decided
   quantities is within tol, with every finer run of its scheme below 2048 passing too (as cost_accuracy).

The time-resolved and histogram-distance quantities (ts:, tv:) are distances from the mean 2048 run and have no
per-scheme form; they are left out here (cost_accuracy has them).

    python cgols_scheme_convergence.py [--tols 5,10,20] [--json OUT]

CPU only, reads only caches (about a minute). Prints the tables; with --json also writes them.
"""


# general
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.argv, _ARGV = sys.argv[:1], sys.argv  # the imported scripts parse no arguments; keep ours

# numerics
import cgols_cost_accuracy as ca
import cgols_fvfd_synthesis as syn
import numpy as np

sys.argv = _ARGV
SCHEMES, NS, TOP = ("FD", "FV"), (128, 256, 512, 1024, 2048), 2048
Z_AGREE = 2.0
RICHARDSON_AGREE = 5.0  # percent of R


def richardson_limit(cells, scheme, ns=(512, 1024, 2048)):
    """(limit, order p) of a scheme's ladder over three rungs, or (None, None) if it does not qualify."""
    if not all((scheme, n) in cells for n in ns):
        return None, None
    v = [cells[(scheme, n)]["value"] for n in ns]
    d1, d2 = v[1] - v[0], v[2] - v[1]
    sig = cells[(scheme, ns[1])].get("sigma", np.nan)
    real = not np.isfinite(sig) or abs(d1) > 2 * sig
    if not (real and d2 != 0 and np.sign(d1) == np.sign(d2) and abs(d2) <= 0.5 * abs(d1)):
        return None, None
    p = float(np.log2(abs(d1) / abs(d2)))
    return v[2] + d2 / (2**p - 1), p


def order(cells, scheme):
    """Observed order from the finest qualifying three rungs: (p, rungs) or (None, None)."""
    for ns in ((512, 1024, 2048), (256, 512, 1024)):
        _limit, p = richardson_limit(cells, scheme, ns)
        if p is not None:
            return p, ns
    return None, None


def self_convergence(key, cells, scheme, tol):
    """(nominal N, confident N) from which the scheme stays within tol of its own 2048 run; None = only 2048."""
    if (scheme, TOP) not in cells:
        return None, None
    top = cells[(scheme, TOP)]
    k = ca.scale(key, top["value"])
    res = {}
    for n in NS[:-1]:
        if (scheme, n) not in cells:
            continue
        c = cells[(scheme, n)]
        err = abs(ca.error(key, c["value"], top["value"]))
        sig = np.hypot(c["sigma"], top["sigma"]) * k if np.isfinite(c["sigma"]) and np.isfinite(top["sigma"]) else 0.0
        res[n] = (err <= tol, err + sig <= tol)
    out = []
    for i in (0, 1):
        ok = [n for n in res if all(res[m][i] for m in res if m >= n)]
        out.append(min(ok) if ok else None)
    return tuple(out)


def reference(key, cells):
    """(R, u [%], kind, z_top) for item 5, or (None, None, reason, z_top)."""
    fd, fv = cells.get(("FD", TOP)), cells.get(("FV", TOP))
    if fd is None or fv is None:
        return None, None, "no 2048 pair", None
    sig = np.hypot(fd["sigma"], fv["sigma"])
    z = (fd["value"] - fv["value"]) / sig if np.isfinite(sig) and sig > 0 else np.inf
    mean = 0.5 * (fd["value"] + fv["value"])
    if abs(z) < Z_AGREE:
        return mean, 0.5 * abs(fd["value"] - fv["value"]) * ca.scale(key, mean), "2048 agree", z
    lim = {s: richardson_limit(cells, s)[0] for s in SCHEMES}
    if all(v is not None for v in lim.values()):
        r = 0.5 * (lim["FD"] + lim["FV"])
        u = 0.5 * abs(lim["FD"] - lim["FV"]) * ca.scale(key, r)
        if u * 2 <= RICHARDSON_AGREE:
            return r, u, "Richardson limits agree", z
    u = 0.5 * abs(fd["value"] - fv["value"]) * ca.scale(key, mean)
    return mean, u, "2048 differ", z


def analyse(scal):
    rows = {}
    for family, items in ca.FAMILIES.items():
        for key, label in items:
            if ca.is_distance(key) or key not in scal:
                continue
            cells = {c: v for c, v in scal[key]["cells"].items() if c[1] in NS}
            if ("FD", TOP) not in cells or ("FV", TOP) not in cells:
                continue
            r, u, kind, z = reference(key, cells)
            rows[key] = {"key": key, "family": family, "label": label, "cells": cells, "R": r, "u": u, "ref_kind": kind,
                         "z_top": z, "order": {s: order(cells, s) for s in SCHEMES},
                         "limit": {s: richardson_limit(cells, s)[0] for s in SCHEMES}}
    return rows


def cheapest(rows, tol):
    """Per family: (nominal cell, confident cell, decided quantities, undecided labels) against the unbiased R."""
    out = {}
    cells_by_cost = sorted(syn.GPU_HOURS, key=syn.GPU_HOURS.get)
    for family in ca.FAMILIES:
        fam = [r for r in rows.values() if r["family"] == family]
        if not fam:
            continue
        decided = [r for r in fam if r["R"] is not None and r["u"] < tol]
        passes = {}
        for cell in cells_by_cost:
            have = [r for r in decided if cell in r["cells"]]
            if not decided or len(have) < len(decided):
                continue
            errs = [abs(ca.error(r["key"], r["cells"][cell]["value"], r["R"])) for r in decided]
            us = [r["u"] for r in decided]
            passes[cell] = (max(errs) <= tol, max(e + u for e, u in zip(errs, us)) <= tol)

        def ok(cell, i, passes=passes):
            finer = [c for c in passes if c[0] == cell[0] and cell[1] < c[1] < TOP]
            return passes[cell][i] and all(passes[c][i] for c in finer)

        nominal = next((c for c in cells_by_cost if c in passes and ok(c, 0)), None)
        confident = next((c for c in cells_by_cost if c in passes and ok(c, 1)), None)
        out[family] = {"nominal": nominal, "confident": confident, "decided": len(decided), "total": len(fam),
                       "undecided": [f"{r['label']} ({r['ref_kind']}, u = {r['u']:.0f} %)" if r["u"] is not None
                                     else f"{r['label']} ({r['ref_kind']})"
                                     for r in fam if r["key"] not in {d["key"] for d in decided}]}
    return out


def name(cell):
    return "-" if cell is None else f"{cell[0]}-{cell[1]} ({syn.GPU_HOURS[cell]:.3g} GPU-h)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--tols", default="5,10,20")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    tols = [float(t) for t in args.tols.split(",")]
    _runs, _data, _ex, scal = ca.collect()
    rows = analyse(scal)

    print("\n## 1. Self-convergence per scheme: lowest N from which every run up to 1024 is within tol of the same "
          "scheme at 2048 (nominal / confident incl. seed noise); order p (rungs)")
    print("| family: quantity | " + " | ".join(f"FD {t:g}% | FV {t:g}%" for t in tols) + " | p FD | p FV |")
    self_table = {}
    for key, r in rows.items():
        cols, entry = [], {}
        for tol in tols:
            for s in SCHEMES:
                nom, conf = self_convergence(key, r["cells"], s, tol)
                entry[f"{s}_{tol:g}"] = (nom, conf)
                cols.append(f"{nom or '2048'} / {conf or '2048'}")
        ps = []
        for s in SCHEMES:
            p, ns = r["order"][s]
            ps.append("-" if p is None else f"{p:.1f} ({ns[0]}-{ns[2]})")
            entry[f"order_{s}"] = p
        self_table[key] = entry
        print(f"| {r['family']}: {r['label']} | " + " | ".join(cols) + " | " + " | ".join(ps) + " |")

    print("\n## 1b. Per family: resolution from which the whole family is self-converged (worst quantity, nominal)")
    print("| family | " + " | ".join(f"FD {t:g}% | FV {t:g}%" for t in tols) + " |")
    family_self = {}
    for family in ca.FAMILIES:
        keys = [k for k, r in rows.items() if r["family"] == family]
        if not keys:
            continue
        cols, entry = [], {}
        for tol in tols:
            for s in SCHEMES:
                ns = [self_table[k][f"{s}_{tol:g}"][0] or TOP for k in keys]
                entry[f"{s}_{tol:g}"] = max(ns)
                cols.append(str(max(ns)))
        family_self[family] = entry
        print(f"| {family} | " + " | ".join(cols) + " |")

    print(f"\n## 3. Seed noise at the top: z_top = (FD-2048 - FV-2048) / sigma; |z| < {Z_AGREE:g}: no scheme effect")
    print("| family: quantity | FD-2048 | FV-2048 | sigma FD / FV | z_top | reference for item 5 |")
    for key, r in rows.items():
        fd, fv = r["cells"][("FD", TOP)], r["cells"][("FV", TOP)]
        print(f"| {r['family']}: {r['label']} | {fd['value']:.4g} | {fv['value']:.4g} | "
              f"{fd['sigma']:.2g} / {fv['sigma']:.2g} | {r['z_top']:+.1f} | {r['ref_kind']} |")

    print("\n## 5. Cheapest run per family within tol of the unbiased reference (nominal / confident incl. u); "
          "quantities without an unbiased reference listed")
    summary = {tol: cheapest(rows, tol) for tol in tols}
    print("| family | decided | " + " | ".join(f"{t:g}% nominal | {t:g}% confident" for t in tols) + " |")
    for family in summary[tols[0]]:
        cols = []
        for tol in tols:
            c = summary[tol][family]
            cols += [name(c["nominal"]), name(c["confident"])]
        s0 = summary[tols[0]][family]
        print(f"| {family} | {s0['decided']}/{s0['total']} | " + " | ".join(cols) + " |")
    print("\nWithout an unbiased reference (5 % tolerance):")
    for family, c in summary[tols[0]].items():
        if c["undecided"]:
            print(f"  {family}: " + "; ".join(c["undecided"]))

    if args.json:
        cell = lambda c: None if c is None else f"{c[0]}-{c[1]}"
        out = {
            "self": {k: {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in e.items()}
                     for k, e in self_table.items()},
            "family_self": family_self,
            "quantities": {k: {"family": r["family"], "label": r["label"], "R": r["R"], "u": r["u"],
                               "ref_kind": r["ref_kind"], "z_top": r["z_top"],
                               "limit": r["limit"], "order": {s: r["order"][s][0] for s in SCHEMES},
                               "values": {cell(c): v["value"] for c, v in r["cells"].items()},
                               "sigma": {cell(c): v["sigma"] for c, v in r["cells"].items()}}
                           for k, r in rows.items()},
            "cheapest": {str(t): {f: {**d, "nominal": cell(d["nominal"]), "confident": cell(d["confident"])}
                                  for f, d in s.items()} for t, s in summary.items()},
            "gpu_hours": {cell(c): h for c, h in syn.GPU_HOURS.items()},
        }
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
