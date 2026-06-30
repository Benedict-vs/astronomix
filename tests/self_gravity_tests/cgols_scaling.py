"""CGOLS strong-scaling benchmark: sec/step + per-device memory vs resolution and GPU count.

Rather than running the full ~11 h CGOLS simulation 9 times, this drives cgols.py
in its lightweight *benchmark mode* (CGOLS_BENCH_STEPS): a handful of equal-sized
fixed timesteps with the snapshot/monitor machinery off. The per-step *compute* is
identical to the production run (same WENO5/Pallas kernels, wind source, external
potential, positivity), so seconds-per-step and peak per-device memory measured this
way are representative — and each config finishes in minutes, not hours.

For every (resolution, SHARD_SPLIT) it launches cgols.py as a fresh subprocess
(JAX/GPU device selection is process-global, so each config must be its own process)
with environment overrides, then parses the two numbers cgols already prints:

    🏁 ... ⏱️ Time elapsed: <X> seconds        (post-compile execution of N steps)
    === Compiled memory usage PER DEVICE ===
    Total size: <Y> MB                          (peak memory on ONE device)

sec_per_step = X / N ;  mem_per_device = Y.

Resolutions need their own initial conditions (cgols saves a resolution-specific
.npy). The driver builds tagged ICs (CGOLS_IC_TAG) for the smaller grids so it never
touches your production cgols_initial_state.npy / cgols_initial_potential.npy; for
512 it reuses those existing files if present.

Usage
-----
    python cgols_scaling.py run            # run the sweep, write cgols_scaling_results.json
    python cgols_scaling.py plot           # (re)plot + extrapolate from the JSON
    python cgols_scaling.py all            # run then plot (default)
    python cgols_scaling.py run --steps 40 --resolutions 128,256 --splits "(1,1,1,1),(1,2,1,1)"

Nothing here selects GPUs itself — cgols.py / autocvd does, per subprocess. Configs
that need 2 or 4 free GPUs will fail gracefully (recorded, sweep continues) if the
GPUs are busy.
"""
# ruff: noqa: E402
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CGOLS = os.path.join(HERE, "cgols.py")
RESULTS_JSON = os.path.join(HERE, "cgols_scaling_results.json")
PLOT_PNG = os.path.join(HERE, "cgols_scaling.png")
TABLE_CSV = os.path.join(HERE, "cgols_scaling_estimates.csv")

# Default sweep ------------------------------------------------------------
DEFAULT_RESOLUTIONS = [128, 256, 512]          # x=y=dim, z=2*dim
DEFAULT_SPLITS = [(1, 1, 1, 1), (1, 2, 1, 1), (1, 2, 2, 1)]  # 1, 2, 4 GPUs
DEFAULT_STEPS = 90                              # fixed timesteps per timing run

# Anchor: a known full-resolution full-duration single-GPU wall time, used ONLY to
# convert the measured sec/step into an absolute full-run estimate (number of steps
# to reach t_end is not measured here). From the project log: the 512x512x1024 run
# completes 75 Myr in ~11 h on one A100. Override with --anchor-hours.
DEFAULT_ANCHOR_DIM = 512
DEFAULT_ANCHOR_HOURS = 11.0

# Memory budget per GPU (A100-40GB) used for the "min GPUs to fit" extrapolation.
GPU_MEM_GB = 40.0
GPU_MEM_USABLE_GB = 38.0   # leave headroom for fragmentation

PER_RUN_TIMEOUT_S = 3600   # generous: compile (minutes at 512^3) + N cheap steps


# ------------------------------------------------------------------------- #
# Running                                                                    #
# ------------------------------------------------------------------------- #
def ncells(dim):
    return dim * dim * (2 * dim)


def ngpus_of(split):
    n = 1
    for s in split:
        n *= s
    return n


def _run_cgols(env_overrides, timeout=PER_RUN_TIMEOUT_S):
    """Launch cgols.py as a subprocess with CGOLS_* overrides; return (rc, stdout, stderr)."""
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_overrides.items()})
    try:
        p = subprocess.run(
            [sys.executable, CGOLS],
            cwd=HERE, env=env, capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), (e.stderr or "") + f"\n[timeout after {timeout}s]"


def _ic_files(dim, tag):
    s = os.path.join(HERE, f"cgols_initial_state{tag}.npy")
    p = os.path.join(HERE, f"cgols_initial_potential{tag}.npy")
    return s, p


def _npy_shape(path):
    """Read a .npy array's shape WITHOUT loading its data (memory-map the header)."""
    import numpy as np
    return tuple(np.load(path, mmap_mode="r").shape)


def _ic_matches(dim, tag):
    """True iff both IC files for `tag` exist AND their grid is exactly (dim, dim, 2dim).

    The shape check is the safety net: a filename does NOT prove its resolution.
    Running cgols.py by hand at another dim overwrites the untagged
    cgols_initial_state.npy, so the benchmark must never trust a file it did not
    just build — it verifies the trailing 3 spatial axes (state is
    (num_vars, x, y, z); potential is (x, y, z)).
    """
    s, p = _ic_files(dim, tag)
    if not (os.path.exists(s) and os.path.exists(p)):
        return False
    want = (dim, dim, 2 * dim)
    try:
        return _npy_shape(s)[-3:] == want and _npy_shape(p)[-3:] == want
    except Exception:
        return False


def ensure_ic(dim, force=False):
    """Make sure a correctly-shaped IC exists for `dim`; return the IC_TAG to use.

    Reuse policy (skipped under --force-ic): reuse any existing file whose *actual
    grid shape* matches `dim` — the untagged production file first (so a genuine
    512³ run reuses cgols_initial_state.npy), then the per-resolution _d<dim> file.
    A shape mismatch (e.g. you ran cgols.py by hand at another dim and overwrote the
    untagged file) is ignored, not trusted. When a build is needed it goes into the
    tagged _d<dim> file so the production IC is never clobbered — only --force-ic at
    dim=512 deliberately rebuilds the untagged production file.
    """
    want = (dim, dim, 2 * dim)
    if not force:
        for tag in ("", f"_d{dim}"):
            if _ic_matches(dim, tag):
                s, _ = _ic_files(dim, tag)
                print(f"  [IC] dim={dim}: reusing {os.path.basename(s)} "
                      f"(verified grid {want})")
                return tag

    tag = "" if (force and dim == 512) else f"_d{dim}"
    s, p = _ic_files(dim, tag)
    print(f"  [IC] dim={dim}: building initial conditions (tag={tag!r}) ...")
    t0 = time.time()
    rc, out, err = _run_cgols({
        "CGOLS_CREATE_IC": "1",
        "CGOLS_DIM": dim,
        "CGOLS_IC_TAG": tag,
        "CGOLS_SHARD_SPLIT": "(1, 1, 1, 1)",
        "CGOLS_BENCH_STEPS": "0",
    })
    if rc != 0 or not _ic_matches(dim, tag):
        raise RuntimeError(
            f"IC build failed or wrong shape for dim={dim} (rc={rc}, want grid {want}).\n"
            "--- stderr tail ---\n" + "\n".join(err.splitlines()[-25:])
        )
    print(f"  [IC] dim={dim}: built in {time.time() - t0:.0f}s")
    return tag


_RE_ELAPSED = re.compile(r"Time elapsed:\s*([0-9.]+)\s*seconds")
_RE_TOTAL_MB = re.compile(r"Total size:\s*([0-9.]+)\s*MB")
_RE_SHARD = re.compile(r"shard shape:\s*(\([^)]*\))")


def parse_run(out):
    """Pull elapsed seconds, per-device total MB and (if sharded) shard shape from stdout."""
    el = _RE_ELAPSED.search(out)
    mb = _RE_TOTAL_MB.search(out)
    sh = _RE_SHARD.search(out)
    return (
        float(el.group(1)) if el else None,
        float(mb.group(1)) if mb else None,
        sh.group(1) if sh else None,
    )


def bench_one(dim, split, tag, steps):
    ng = ngpus_of(split)
    print(f"  [bench] dim={dim} split={split} ({ng} GPU) ...", flush=True)
    t0 = time.time()
    rc, out, err = _run_cgols({
        "CGOLS_DIM": dim,
        "CGOLS_SHARD_SPLIT": str(split),
        "CGOLS_IC_TAG": tag,
        "CGOLS_CREATE_IC": "0",
        "CGOLS_BENCH_STEPS": steps,
    })
    elapsed, mb, shard = parse_run(out)
    rec = {
        "dim": dim, "split": list(split), "ngpus": ng, "ncells": ncells(dim),
        "cells_per_gpu": ncells(dim) / ng, "steps": steps,
        "elapsed_s": elapsed, "mem_total_mb": mb, "shard_shape": shard,
        "wall_s": round(time.time() - t0, 1), "rc": rc,
    }
    if rc == 0 and elapsed is not None:
        rec["status"] = "ok"
        rec["sec_per_step"] = elapsed / steps
        print(f"    -> {rec['sec_per_step']:.3f} s/step, "
              f"{(mb or 0) / 1024:.2f} GB/device  (wall {rec['wall_s']:.0f}s)")
    else:
        rec["status"] = "failed"
        rec["stderr_tail"] = "\n".join(err.splitlines()[-25:])
        rec["stdout_tail"] = "\n".join(out.splitlines()[-15:])
        print(f"    -> FAILED (rc={rc}); see stderr_tail in JSON")
    return rec


def run_sweep(resolutions, splits, steps, anchor_dim, anchor_hours, force_ic):
    runs = []
    for dim in resolutions:
        print(f"[dim {dim}x{dim}x{2 * dim}]")
        try:
            tag = ensure_ic(dim, force=force_ic)
        except RuntimeError as e:
            print(f"  !! {e}")
            for split in splits:
                runs.append({"dim": dim, "split": list(split), "ngpus": ngpus_of(split),
                             "status": "ic_failed", "error": str(e)})
            continue
        for split in splits:
            runs.append(bench_one(dim, split, tag, steps))
    data = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "steps": steps, "anchor_dim": anchor_dim, "anchor_hours": anchor_hours,
            "gpu_mem_gb": GPU_MEM_GB, "gpu_mem_usable_gb": GPU_MEM_USABLE_GB,
        },
        "runs": runs,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nWrote {RESULTS_JSON} ({sum(r.get('status') == 'ok' for r in runs)}/"
          f"{len(runs)} configs ok)")
    return data


# ------------------------------------------------------------------------- #
# Plotting / extrapolation                                                   #
# ------------------------------------------------------------------------- #
def _powerfit(x, y):
    """Least-squares fit y = c * x**a in log-log space. Returns (a, c)."""
    import numpy as np
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    a, b = np.polyfit(lx, ly, 1)
    return a, float(np.exp(b))


def plot(data):
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    # The 3-wide layout with multi-line titles + legends triggers a harmless
    # matplotlib layout-engine UserWarning (emitted via warn_external, so it is
    # attributed to this module); the figure still renders fine. Filter by message.
    warnings.filterwarnings("ignore", message=r".*(constrained_layout|Tight layout).*")

    runs = [r for r in data["runs"] if r.get("status") == "ok"]
    if not runs:
        print("No successful runs to plot.")
        return
    meta = data["meta"]
    anchor_dim, anchor_hours = meta["anchor_dim"], meta["anchor_hours"]

    by_gpu = {}
    for r in runs:
        by_gpu.setdefault(r["ngpus"], []).append(r)
    for v in by_gpu.values():
        v.sort(key=lambda r: r["ncells"])
    gpu_counts = sorted(by_gpu)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    _cmap = matplotlib.colormaps["viridis"]
    colors = {g: c for g, c in zip(gpu_counts, _cmap(np.linspace(0.1, 0.85, len(gpu_counts))))}

    # --- Panel A: sec/step vs cells, per GPU count, with power-law extrapolation ---
    axA = axes[0]
    extrap_dims = [1024, 2048]
    fits = {}
    for g in gpu_counts:
        xs = [r["ncells"] for r in by_gpu[g]]
        ys = [r["sec_per_step"] for r in by_gpu[g]]
        axA.plot(xs, ys, "o-", color=colors[g], label=f"{g} GPU")
        if len(xs) >= 2:
            a, c = _powerfit(xs, ys)
            fits[g] = (a, c)
            xx = np.array(xs + [ncells(d) for d in extrap_dims], float)
            xx.sort()
            axA.plot(xx, c * xx**a, "--", color=colors[g], alpha=0.5)
    axA.set_xscale("log"); axA.set_yscale("log")
    # Read ylim AFTER switching to log: on the still-linear axis the bottom margin
    # can be <=0, which is an invalid log coordinate for the annotation below and
    # makes savefig's tight bbox blow the canvas up to ~350 Mpx (the "white line").
    _y0 = axA.get_ylim()[0]
    for d in extrap_dims:
        axA.axvline(ncells(d), color="grey", ls=":", alpha=0.4)
        axA.text(ncells(d), _y0, f" {d}³*", rotation=90,
                 va="bottom", ha="right", fontsize=8, color="grey")
    axA.set_xlabel("number of cells  (dim·dim·2dim)")
    axA.set_ylabel("seconds / step")
    axA.set_title("Per-step cost vs resolution\n(dashed = power-law fit, dotted = extrapolation)")
    axA.legend(); axA.grid(True, which="both", alpha=0.3)

    # --- Panel B: strong scaling (speedup vs GPUs), per resolution ---
    axB = axes[1]
    by_dim = {}
    for r in runs:
        by_dim.setdefault(r["dim"], {})[r["ngpus"]] = r["sec_per_step"]
    maxg = max(gpu_counts)
    for dim in sorted(by_dim):
        d = by_dim[dim]
        if 1 not in d:
            continue
        gs = sorted(d)
        sp = [d[1] / d[g] for g in gs]
        axB.plot(gs, sp, "o-", label=f"{dim}³ grid")
    axB.plot([1, maxg], [1, maxg], "k--", alpha=0.5, label="ideal (linear)")
    axB.set_xlabel("number of GPUs")
    axB.set_ylabel("strong-scaling speedup  (t₁ / tₙ)")
    axB.set_title("Strong scaling (same grid, more GPUs)")
    axB.set_xticks(gpu_counts)
    axB.legend(); axB.grid(True, alpha=0.3)

    # --- Panel C: per-device memory vs cells, with GPU budget + min-GPU guidance ---
    axC = axes[2]
    all_cpg, all_mem = [], []
    for g in gpu_counts:
        xs = [r["cells_per_gpu"] for r in by_gpu[g]]
        ys = [r["mem_total_mb"] / 1024 for r in by_gpu[g] if r["mem_total_mb"]]
        xs = xs[:len(ys)]
        axC.plot(xs, ys, "o", color=colors[g], label=f"{g} GPU")
        all_cpg += xs; all_mem += ys
    mem_fit = None
    if len(all_cpg) >= 2:
        a, c = _powerfit(all_cpg, all_mem)
        mem_fit = (a, c)
        xx = np.array(sorted(all_cpg + [ncells(d) for d in extrap_dims]), float)
        axC.plot(xx, c * xx**a, "k--", alpha=0.5, label=f"fit  GB≈{c:.2e}·cpg^{a:.2f}")
    axC.axhline(GPU_MEM_GB, color="red", ls="-", alpha=0.6, label=f"{GPU_MEM_GB:.0f} GB GPU")
    axC.axhline(GPU_MEM_USABLE_GB, color="orange", ls=":", alpha=0.6, label=f"{GPU_MEM_USABLE_GB:.0f} GB usable")
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_xlabel("cells per GPU")
    axC.set_ylabel("peak memory / device  [GB]")
    axC.set_title("Memory per device\n(sets the minimum GPU count for a resolution)")
    axC.legend(fontsize=8); axC.grid(True, which="both", alpha=0.3)

    fig.suptitle("CGOLS strong-scaling benchmark", fontsize=14)
    fig.savefig(PLOT_PNG, dpi=150)
    print(f"Wrote {PLOT_PNG}")

    _write_estimates(runs, fits, mem_fit, anchor_dim, anchor_hours, extrap_dims)


def _write_estimates(runs, fits, mem_fit, anchor_dim, anchor_hours, extrap_dims):
    """Convert measured sec/step into full-run wall-time estimates + extrapolate.

    steps-to-t_end scales linearly with `dim` (CFL: dt ∝ dx ∝ 1/dim, t_end fixed),
    calibrated to the anchor full run. Per-step cost at unmeasured (dim, gpus) comes
    from the per-GPU-count power-law fit of sec/step vs cells.
    """
    # Calibrate steps(dim) from the anchor single-GPU full run.
    anchor = next((r for r in runs if r["dim"] == anchor_dim and r["ngpus"] == 1), None)
    steps_per_dim = None
    if anchor:
        steps_anchor = anchor_hours * 3600.0 / anchor["sec_per_step"]
        steps_per_dim = steps_anchor / anchor_dim   # steps ∝ dim
        note = (f"calibrated: {anchor_dim}³ @1GPU = {anchor_hours:.1f} h "
                f"=> ~{steps_anchor:,.0f} steps to t_end")
    else:
        note = (f"no {anchor_dim}³/1-GPU datapoint; full-run hours omitted "
                f"(run that config or pass --anchor-* )")

    def sec_step(dim, g):
        # measured if present, else fit
        for r in runs:
            if r["dim"] == dim and r["ngpus"] == g:
                return r["sec_per_step"], "measured"
        if g in fits:
            a, c = fits[g]
            return c * ncells(dim)**a, "extrapolated"
        return None, None

    measured_dims = sorted({r["dim"] for r in runs})
    all_dims = sorted(set(measured_dims + extrap_dims))
    all_gpus = sorted({r["ngpus"] for r in runs})

    lines = [f"# CGOLS scaling estimates  ({note})",
             "# full_run_hours = sec_per_step * steps_to_t_end(dim);  steps ∝ dim",
             "dim,grid,gpus,sec_per_step,source,mem_GB_per_device,full_run_hours"]
    print("\n=== Estimated full-run wall time (hours) ===")
    print(f"  {note}")
    header = "  dim | " + " | ".join(f"{g}GPU" for g in all_gpus)
    print(header); print("  " + "-" * (len(header) - 2))
    for dim in all_dims:
        cells = ncells(dim)
        row = f"  {dim:>4}"
        for g in all_gpus:
            ss, src = sec_step(dim, g)
            mem_gb = ""
            if mem_fit is not None:
                a, c = mem_fit
                mem_gb = f"{c * (cells / g)**a:.1f}"
            if ss is None:
                row += " |   -- "
                continue
            hrs = ss * steps_per_dim * dim / 3600.0 if steps_per_dim else float("nan")
            tag = "*" if src == "extrapolated" else " "
            row += f" | {hrs:6.1f}{tag}" if steps_per_dim else f" | {ss:.3f}s{tag}"
            lines.append(f"{dim},{dim}x{dim}x{2*dim},{g},{ss:.5f},{src},{mem_gb},"
                         f"{hrs if steps_per_dim else ''}")
        print(row)
    print("  (* = extrapolated from the power-law fit, not measured;"
          " dims marked ³* in Panel A)")

    # Min GPUs to fit each (extrapolated) resolution under the memory budget.
    if mem_fit is not None:
        a, c = mem_fit
        print("\n=== Minimum GPUs to fit under "
              f"{GPU_MEM_USABLE_GB:.0f} GB/device (memory fit) ===")
        for dim in all_dims:
            cells = ncells(dim)
            # smallest g (power of 2) with c*(cells/g)^a <= budget
            g = 1
            while c * (cells / g)**a > GPU_MEM_USABLE_GB and g < 4096:
                g *= 2
            print(f"  {dim:>4}³-ish ({dim}x{dim}x{2*dim}): >= {g} GPU "
                  f"(~{c * (cells / g)**a:.1f} GB/device)")

    with open(TABLE_CSV, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {TABLE_CSV}")


# ------------------------------------------------------------------------- #
# CLI                                                                        #
# ------------------------------------------------------------------------- #
def _parse_splits(s):
    val = ast.literal_eval(s if s.strip().startswith("[") else f"[{s}]")
    return [tuple(t) for t in val]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="all", choices=["run", "plot", "all"])
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"fixed timesteps per timing run (default {DEFAULT_STEPS})")
    ap.add_argument("--resolutions", type=str, default=None,
                    help="comma list of x/y dims, e.g. 128,256,512")
    ap.add_argument("--splits", type=str, default=None,
                    help='comma list of SHARD_SPLIT tuples, e.g. "(1,1,1,1),(1,2,2,1)"')
    ap.add_argument("--anchor-dim", type=int, default=DEFAULT_ANCHOR_DIM)
    ap.add_argument("--anchor-hours", type=float, default=DEFAULT_ANCHOR_HOURS,
                    help="known full-run wall time at anchor-dim on 1 GPU (default 11 h)")
    ap.add_argument("--force-ic", action="store_true",
                    help="rebuild ICs even if the .npy files already exist")
    args = ap.parse_args()

    resolutions = ([int(x) for x in args.resolutions.split(",")]
                   if args.resolutions else DEFAULT_RESOLUTIONS)
    splits = _parse_splits(args.splits) if args.splits else DEFAULT_SPLITS

    data = None
    if args.mode in ("run", "all"):
        data = run_sweep(resolutions, splits, args.steps,
                         args.anchor_dim, args.anchor_hours, args.force_ic)
    if args.mode in ("plot", "all"):
        if args.mode == "plot":
            with open(RESULTS_JSON) as f:
                data = json.load(f)
            # CLI overrides for re-plotting without re-running
            data["meta"]["anchor_dim"] = args.anchor_dim
            data["meta"]["anchor_hours"] = args.anchor_hours
        plot(data)


if __name__ == "__main__":
    main()
