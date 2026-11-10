"""
Cost against accuracy of the CGOLS A-series runs: which scheme (FD WENO5 / FV VL2) and resolution reaches a given
accuracy in which quantity for the fewest GPU-hours.

Reference: per quantity the mean R of FD-2048 and FV-2048; its uncertainty u = |FD-2048 - FV-2048| / 2 (the two
best runs disagree by that much, so no run can be judged more precisely). Error of a run: e = (x - R) / |R|
(percentage points for the CC85 velocity offset, percent density for the CC85 log-density offset). A run reaches a
tolerance tol "confidently" if |e| + u / |R| <= tol, "nominally" if only |e| <= tol. The noise sigma of a run (seed
scatter, measured or extrapolated, and the time-sampling error; cgols_fvfd_synthesis.stats) marks the errors that
are within 2 sigma of the reference, i.e. not distinguishable from it by one run. The 2048 runs are part of the
reference: their error is u by construction.

Quantities (families):
  mass outflow 5 kpc     the cgols_fvfd_synthesis frame metrics (outflow windows, peak, cumulative, side loss)
  outflow by phase 1 kpc wind / mixed / entrained+mixed outflow at 1 kpc, mixed mass (3D caches)
  energy flux            Edot at 1 and 5 kpc in 15-40 and 50-75 Myr, cumulative energy at 5 kpc
  momentum flux          outward momentum flux at 1 and 5 kpc in 15-40 and 50-75 Myr (3D caches)
  emission               L_X1, L_X2 (AtomDB 0.3-2 keV), L_Halpha in 15-40 and 60-75 Myr
  wind structure         cone half-angle at 1 kpc, free-wind axis against CC85
  small-scale morphology up-down asymmetry, small-scale density variance, cavity-rim roughness 20-35 Myr

The metrics come from the cgols_fvfd_synthesis cache (data/studies/fvfd_synthesis_metrics.json; run that script
first); the momentum / 1 kpc energy / late energy metrics are computed here from the 3D caches on the same 5 Myr
cadence and borrow the relative seed scatter of the 5 kpc outflow (as the synthesis does for its 3D metrics); the
rim roughness is computed from the frames of every base and seed run (cgols_convergence_3d.rim_metrics).

Cost: solver GPU-hours (H200) to 75 Myr, cgols_fvfd_synthesis.GPU_HOURS (time integration only; FD-1024 = job 93951,
8 H200, 124 GPU-h).

    python cgols_cost_accuracy.py [--tols 5,10,20] [--json OUT]

Writes figures/cgols/comparisons/cost_accuracy_grid.png (error of every quantity for every run, runs ordered by
cost) and cost_accuracy_families.png (worst error per family against GPU-hours) and prints the "cheapest run per
family and tolerance" table. CPU only.
"""

# ruff: noqa: E402

# general
import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# numerics
import numpy as np

# plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

# CGOLS analysis
import cgols_fvfd_synthesis as syn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures", "cgols", "comparisons")
REF_N = 2048
SCHEMES = ("FD", "FV")
NS = (128, 256, 512, 1024, 2048)

#: family -> [(metric key, short label)]; keys without a synthesis METRICS entry are computed here (EXTRA_3D, RIM)
FAMILIES = {
    "mass outflow 5 kpc": [
        ("out5_15_40", "outflow 15-40 Myr"), ("out5_40_50", "outflow 40-50 Myr"),
        ("out5_50_75", "outflow 50-75 Myr"), ("out5_peak", "peak outflow"), ("mout5_cum", "mass out 0-75 Myr"),
        ("side_loss", "side-face mass loss")],
    "outflow by phase 1 kpc": [
        ("wind1_15_40", "wind 15-40 Myr"), ("mixed1_15_30", "mixed 15-30 Myr"), ("mixed1_30_45", "mixed 30-45 Myr"),
        ("mixed1_50_75", "entr.+mixed 50-75 Myr"), ("mixedmass_t60", "mixed mass in halo, 60 Myr")],
    "energy flux": [
        ("edot5_15_40", "5 kpc, 15-40 Myr"), ("edot5_50_75", "5 kpc, 50-75 Myr"), ("ecum5", "cumulative 5 kpc"),
        ("edot1_15_40", "1 kpc, 15-40 Myr"), ("edot1_50_75", "1 kpc, 50-75 Myr")],
    "momentum flux": [
        ("pdot5_15_40", "5 kpc, 15-40 Myr"), ("pdot5_50_75", "5 kpc, 50-75 Myr"),
        ("pdot1_15_40", "1 kpc, 15-40 Myr"), ("pdot1_50_75", "1 kpc, 50-75 Myr")],
    "emission": [
        ("lx1_15_40", "L_X1 15-40 Myr"), ("lx1_60_75", "L_X1 60-75 Myr"), ("lx2_15_40", "L_X2 15-40 Myr"),
        ("lx2_60_75", "L_X2 60-75 Myr"), ("ha_15_40", "H-alpha 15-40 Myr"), ("ha_60_75", "H-alpha 60-75 Myr")],
    "wind structure": [
        ("cone1_20_40", "cone angle 20-40 Myr"), ("cone1_60_75", "cone angle 60-75 Myr"),
        ("cc85_dv", "axis v vs CC85 [pp]"), ("cc85_dlogn", "axis n vs CC85")],
    "small-scale morphology": [
        ("asym_50_75", "up-down asymmetry 50-75"), ("ssv256", "small-scale var. 80-300 pc"),
        ("ssv128", "small-scale var. 160-600 pc"), ("rim_20_35", "cavity-rim roughness 20-35")],
}

#: computed here from the 3D caches: key -> (record key, entropy/phase reducer, height [kpc], window, twin for sigma)
EXTRA_3D = {
    "edot1_15_40": ("edot_net", 1.0, (15, 40), "out5_15_40", 1e42),
    "edot1_50_75": ("edot_net", 1.0, (50, 75), "out5_50_75", 1e42),
    "edot5_50_75": ("edot_net", 5.0, (50, 75), "out5_50_75", 1e42),
    "pdot1_15_40": ("pdot_out", 1.0, (15, 40), "out5_15_40", 1e3),
    "pdot1_50_75": ("pdot_out", 1.0, (50, 75), "out5_50_75", 1e3),
    "pdot5_15_40": ("pdot_out", 5.0, (15, 40), "out5_15_40", 1e3),
    "pdot5_50_75": ("pdot_out", 5.0, (50, 75), "out5_50_75", 1e3),
}
RIM_WINDOW = (20, 35)


def extra_3d(tag):
    """The EXTRA_3D window means and their time-sampling errors from one 3D cache (5 Myr cadence)."""
    rec = np.load(os.path.join(SCRIPT_DIR, f"data/studies/convergence3d_v3{tag}.npz"), allow_pickle=True)
    rec = rec["records"].item()
    steps = sorted(rec, key=lambda s: rec[s]["time_myr"])
    t = np.array([rec[s]["time_myr"] for s in steps])
    on5 = np.abs(t / 5.0 - np.round(t / 5.0)) < 0.02
    steps, t = [s for s, k in zip(steps, on5) if k], t[on5]
    out = {}
    for key, (field, height, (a, b), _twin, scale) in EXTRA_3D.items():
        h = int(np.argmin(np.abs(syn.HEIGHTS_KPC - height)))
        y = np.array([rec[s][field][h].sum() / scale for s in steps])
        sel = syn.window(t, a, b)
        if sel.sum() >= 3:
            out[key] = (float(y[sel].mean()), syn.block_se(t, y, a, b, block=5.0))
    return out


def rim_value(frame_dir):
    from cgols_convergence_3d import rim_metrics

    t, _mean_r, scatter, _mode = rim_metrics(os.path.join(SCRIPT_DIR, frame_dir))
    sel = syn.window(t, *RIM_WINDOW)
    return float(np.mean(scatter[sel])) if sel.sum() >= 3 else None


def collect():
    """{key: {(scheme, N): {"value", "sigma"}}} for every metric of FAMILIES."""
    os.chdir(SCRIPT_DIR)
    runs, data = syn.load_all(False, {})
    st = syn.stats(runs, data)
    table = {}
    for family in FAMILIES.values():
        for key, _ in family:
            if key in st:
                table[key] = {cell: {"value": c["value"], "sigma": c["sigma"]} for cell, c in st[key].items()}
    # 3D extras: value, sigma = max(borrowed relative seed scatter x value, time-sampling error)
    for scheme in SCHEMES:
        for n in NS:
            base = [lab for lab, s in runs.items() if s[:3] == (scheme, n, "base")]
            if not base or not runs[base[0]][5]:
                continue
            tag3d = runs[base[0]][5]
            if not os.path.exists(f"data/studies/convergence3d_v3{tag3d}.npz"):
                continue
            for key, (value, se) in extra_3d(tag3d).items():
                twin = st[EXTRA_3D[key][3]].get((scheme, n))
                rel = (twin["sig_traj"] / abs(twin["value"])
                       if twin and twin.get("sig_traj") is not None and twin["value"] else 0.0)
                parts = [v for v in (rel * abs(value), se) if v is not None and np.isfinite(v)]
                table.setdefault(key, {})[(scheme, n)] = {"value": value, "sigma": max(parts) if parts else np.nan}
    # rim roughness from the frames of the base and seed runs: measured seed scatter where there are >= 2 members,
    # else the largest measured relative scatter at <= N (as the synthesis extrapolates sigma_traj)
    members = {}
    for label, spec in runs.items():
        scheme, n, role, frames = spec[:4]
        if role not in ("base", "ens") or not os.path.isdir(frames):
            continue
        value = rim_value(frames)
        if value is not None:
            members.setdefault((scheme, n), {})[role if role == "base" else label] = value
    rim = {}
    for cell, vals in members.items():
        if "base" not in vals:
            continue
        arr = np.array(list(vals.values()))
        rim[cell] = {"value": vals["base"], "sig_meas": float(np.std(arr, ddof=1)) if len(arr) >= 2 else None}
    for (scheme, n), c in rim.items():
        if c["sig_meas"] is not None:
            c["sigma"] = c["sig_meas"]
        else:
            below = [cc["sig_meas"] / cc["value"] for (s, m), cc in rim.items() if cc["sig_meas"] and m <= n]
            c["sigma"] = max(below) * c["value"] if below else np.nan
    table["rim_20_35"] = {cell: {"value": c["value"], "sigma": c["sigma"]} for cell, c in rim.items()}
    return table


def error(key, x, ref):
    """Relative error in percent (CC85 offsets: percentage points / percent density)."""
    if key == "cc85_dv":
        return 100.0 * (x - ref)
    if key == "cc85_dlogn":
        return 100.0 * (10 ** (x - ref) - 1.0)
    return 100.0 * (x - ref) / abs(ref)


def scale(key, ref):
    """Percent per unit of the metric (for sigma and u)."""
    if key == "cc85_dv":
        return 100.0
    if key == "cc85_dlogn":
        return 100.0 * np.log(10.0)
    return 100.0 / abs(ref)


def analyse(table):
    """Per metric: reference, u [%], and per (scheme, N) the error, sigma [%] and cost."""
    result = {}
    for family, items in FAMILIES.items():
        for key, label in items:
            cells = table.get(key, {})
            if ("FD", REF_N) not in cells or ("FV", REF_N) not in cells:
                print(f"  [skip] {key}: no 2048 pair")
                continue
            fd, fv = cells[("FD", REF_N)]["value"], cells[("FV", REF_N)]["value"]
            ref = 0.5 * (fd + fv)
            k = scale(key, ref)
            row = {"family": family, "label": label, "ref": ref, "u": 0.5 * abs(fd - fv) * k, "runs": {}}
            for cell, c in cells.items():
                if cell[1] not in NS or cell not in syn.GPU_HOURS:
                    continue
                row["runs"][cell] = {"value": c["value"], "err": error(key, c["value"], ref),
                                     "sigma": c["sigma"] * k if np.isfinite(c["sigma"]) else np.nan,
                                     "cost": syn.GPU_HOURS[cell]}
            result[key] = row
    return result


def cheapest(result, tol):
    """Per family: the cheapest run whose worst error in the family is <= tol, confidently and nominally."""
    out = {}
    cells = sorted(syn.GPU_HOURS, key=syn.GPU_HOURS.get)
    for family in FAMILIES:
        rows = [r for r in result.values() if r["family"] == family]
        conf = nom = None
        for cell in cells:
            if not all(cell in r["runs"] for r in rows):
                continue
            worst = max(abs(r["runs"][cell]["err"]) for r in rows)
            worst_conf = max(abs(r["runs"][cell]["err"]) + r["u"] for r in rows)
            if nom is None and worst <= tol:
                nom = cell
            if conf is None and worst_conf <= tol:
                conf = cell
        limit = max(r["u"] for r in rows)
        out[family] = {"confident": conf, "nominal": nom, "ref_u_max": limit}
    return out


def name(cell):
    return f"{cell[0]}-{cell[1]}"


def fig_grid(result):
    cells = sorted({c for r in result.values() for c in r["runs"]}, key=syn.GPU_HOURS.get)
    keys = list(result)
    err = np.full((len(keys), len(cells)), np.nan)
    within = np.zeros_like(err, dtype=bool)
    for i, key in enumerate(keys):
        for j, cell in enumerate(cells):
            run = result[key]["runs"].get(cell)
            if run:
                err[i, j] = run["err"]
                within[i, j] = np.isfinite(run["sigma"]) and abs(run["err"]) <= 2 * run["sigma"]
    bounds = [0, 5, 10, 20, 50, 1e9]
    cmap = ListedColormap(["#1a9850", "#a6d96a", "#fee08b", "#fc8d59", "#d73027"])
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(2.0 + 1.05 * (len(cells) + 1), 0.36 * len(keys) + 2.6))
    ax.imshow(np.abs(err), cmap=cmap, norm=norm, aspect="auto")
    ax.set_facecolor("0.85")
    for i in range(len(keys)):
        for j in range(len(cells)):
            if np.isfinite(err[i, j]):
                text = f"{err[i, j]:+.0f}" if abs(err[i, j]) >= 9.5 else f"{err[i, j]:+.1f}"
                ax.text(j, i, text + ("*" if within[i, j] else ""), ha="center", va="center", fontsize=7.5)
    # the reference uncertainty u as an extra column
    ax.set_xlim(-0.5, len(cells) + 0.5)
    for i, key in enumerate(keys):
        ax.text(len(cells), i, f"{result[key]['u']:.1f}", ha="center", va="center", fontsize=7.5, color="0.2")
    ax.set_xticks(range(len(cells) + 1))
    ax.set_xticklabels([f"{name(c)}\n{syn.GPU_HOURS[c]:.3g} GPU-h" for c in cells] + ["ref. unc.\nu [%]"],
                       fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([result[k]["label"] for k in keys], fontsize=8)
    # family separators and labels
    start = 0
    families = [result[k]["family"] for k in keys]
    for i in range(1, len(keys) + 1):
        if i == len(keys) or families[i] != families[i - 1]:
            ax.axhline(i - 0.5, color="k", lw=1.2)
            ax.text(-0.02, 1 - (start + i) / 2 / len(keys), families[i - 1], transform=ax.transAxes, ha="right",
                    va="center", fontsize=9, fontweight="bold", rotation=0,
                    bbox={"boxstyle": "round", "fc": "white", "ec": "none"})
            start = i
    ax.axvline(len(cells) - 0.5, color="k", lw=1.2)
    ax.tick_params(axis="y", pad=150)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(5)]
    labels = ["|e| < 5%", "5-10%", "10-20%", "20-50%", "> 50%"]
    ax.legend(handles + [Line2D([], [], ls="none")], labels + ["* within 2 sigma (run noise) of the reference"],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=6, fontsize=8, frameon=False)
    ax.set_title("CGOLS A: error e = (x - R) / |R| [%] against R = mean(FD-2048, FV-2048), runs ordered by "
                 "GPU-hours (2048 runs: |e| = u by construction)", fontsize=10, pad=34)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "cost_accuracy_grid.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def fig_families(result, tols):
    fams = list(FAMILIES)
    ncol = 4
    nrow = int(np.ceil(len(fams) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.2 * nrow), sharex=True, sharey=True)
    styles = {"FD": ("C0", "o", "-"), "FV": ("C3", "s", "--")}
    for ax, family in zip(axes.flat, fams):
        rows = [r for r in result.values() if r["family"] == family]
        for scheme in SCHEMES:
            pts = []
            for n in NS:
                cell = (scheme, n)
                if n == REF_N or not all(cell in r["runs"] for r in rows):
                    continue
                errs = [abs(r["runs"][cell]["err"]) for r in rows]
                pts.append((syn.GPU_HOURS[cell], max(errs), float(np.median(errs)), n))
            if not pts:
                continue
            x, worst, med, ns = map(np.array, zip(*pts))
            color, marker, ls = styles[scheme]
            ax.plot(x, worst, ls=ls, marker=marker, color=color, label=f"{scheme}: worst quantity")
            ax.plot(x, med, ls=ls, marker=marker, color=color, alpha=0.35, ms=4, label=f"{scheme}: median")
            for xi, yi, n in zip(x, worst, ns):
                ax.annotate(f"{int(n)}", (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7, color=color)
        u = max(r["u"] for r in rows)
        ax.axhspan(0.1, u, color="0.85", zorder=0, label="reference uncertainty (max u)")
        for cell in (("FD", REF_N), ("FV", REF_N)):
            ax.axvline(syn.GPU_HOURS[cell], color=styles[cell[0]][0], lw=0.8, ls=":")
        for tol in tols:
            ax.axhline(tol, color="0.4", lw=0.7, ls="-.")
        ax.set(xscale="log", yscale="log", title=family, ylim=(0.3, 800))
        ax.grid(alpha=0.3, which="both")
    for ax in axes.flat[len(fams):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("GPU-hours (H200) to 75 Myr")
    for ax in axes[:, 0]:
        ax.set_ylabel("|error| vs 2048 reference [%]")
    axes.flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("CGOLS A: worst and median error per quantity family against cost; dotted verticals: cost of "
                 "FD-2048 (blue) / FV-2048 (red); dash-dot: tolerances " + ", ".join(f"{t:g}%" for t in tols))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "cost_accuracy_families.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--tols", default="5,10,20", help="tolerances in percent")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    tols = [float(t) for t in args.tols.split(",")]
    table = collect()
    result = analyse(table)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_grid(result)
    fig_families(result, tols)

    print("\n| quantity | " + " | ".join(name(c) for c in sorted(syn.GPU_HOURS, key=syn.GPU_HOURS.get)) + " | u |")
    for key, r in result.items():
        cells = sorted(syn.GPU_HOURS, key=syn.GPU_HOURS.get)
        vals = [f"{r['runs'][c]['err']:+.1f}" if c in r["runs"] else "" for c in cells]
        print(f"| {r['family']}: {r['label']} | " + " | ".join(vals) + f" | {r['u']:.1f} |")

    print("\nCheapest run per family (worst quantity of the family within tol; confident = incl. reference "
          "uncertainty u):")
    print("| family | max u [%] | " + " | ".join(f"{t:g}% confident | {t:g}% nominal" for t in tols) + " |")
    summary = {}
    for tol in tols:
        summary[tol] = cheapest(result, tol)
    for family in FAMILIES:
        cols = []
        for tol in tols:
            c = summary[tol][family]
            fmt = lambda cell: f"{name(cell)} ({syn.GPU_HOURS[cell]:.3g})" if cell else "none"
            cols += [fmt(c["confident"]), fmt(c["nominal"])]
        print(f"| {family} | {summary[tols[0]][family]['ref_u_max']:.1f} | " + " | ".join(cols) + " |")
    if args.json:
        out = {k: {**{kk: vv for kk, vv in r.items() if kk != "runs"},
                   "runs": {name(c): v for c, v in r["runs"].items()}} for k, r in result.items()}
        summ = {str(t): {f: {kk: (name(vv) if isinstance(vv, tuple) else vv) for kk, vv in d.items()}
                         for f, d in s.items()} for t, s in summary.items()}
        with open(args.json, "w") as fh:
            json.dump({"metrics": out, "cheapest": summ}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
