"""
Write the ASGARD Paper I numbers and figures into paper_exports/paper1/manifest.yaml.

One exporter for the whole paper instead of one line per analysis script, so
every number uses the same conventions:

* Time means use every cached snapshot with t0 < t <= t1 (the snapshot at the
  window start is left out; agreed with Tobias on 18 Nov 2026). Snapshot times
  are rounded to 1e-3 Myr before the comparison, because the 3D caches store
  times like 14.9999998 and 45.00000056.
* "Burst" = (15, 40] Myr, "low state" = (50, 75] Myr, "late" = (60, 75] Myr.
* Planar outflow through |z| = h: net mass flux through both planes, outward
  positive, from the in-situ frames (1.25 Myr cadence; A-2048 has no 51.25 Myr
  frame). Cumulative masses integrate that rate (trapezoid) from 0 to 75 Myr.
* Convergence verdict of a resolution ladder: "converged" if the last doubling
  changes the value by less than CONVERGED_TOLERANCE, else "not_converged"; the
  change is written into the description.

Numbers are computed here from the measurement caches the analysis scripts
wrote (data/studies/*.npz, the frames); the analysis scripts themselves are
not run. Figures are listed with the script that drew them. Keys whose
inputs are missing (e.g. the 0.3-1.0 keV reprocessing before it has finished)
are skipped with a message, not written as placeholders.

    python paper_exports/paper1_numbers.py            # write the manifest
    python paper_exports/paper1_numbers.py --dry-run  # print only

CPU only, about a minute (it reads ~600 frames).
"""

# ruff: noqa: C408 (entries are keyword dicts)

# general
import argparse
import glob
import json
import os
import sys
from pathlib import Path

# numerics
import numpy as np

HERE = Path(__file__).resolve().parent
CGOLS = HERE.parent
sys.path.insert(0, str(CGOLS))

from paper_exports.export import export_figure, export_number

PAPER = "paper1"
STUDIES = CGOLS / "data" / "studies"
FIGURES = CGOLS / "figures" / "cgols"
BOX_HEIGHT_KPC = 20.0  # z extent of the box, -10 ... 10 kpc
CONVERGED_TOLERANCE = 0.05

BURST = (15.0, 40.0)
RIM = (30.0, 45.0)
SX_WINDOW = (50.0, 70.0)
LATE = (60.0, 75.0)

#: convergence3d_v3 heights and shells_v2 radial bins (cgols_convergence_3d.py, cgols_shell_fluxes.py).
HEIGHTS_KPC = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
SHELL_DR_KPC = 0.125
MIXED_ENTROPY_BIN = 1  # entrained / mixed / wind / halo

#: Strickland et al. (2004) M82 north fit, 0.3-1.0 keV photons at the telescope (ACIS-I).
STRICKLAND_NORTH_SIGMA0 = 29.16e-9  # photons s^-1 cm^-2 arcsec^-2
STRICKLAND_NORTH_H_KPC = 1.42
ARCMIN2_PER_STERADIAN = (180.0 * 60.0 / np.pi) ** 2
#: Photon-mode reprocessing: 0.3-1.0 keV, TBabs with the Galactic column toward M82.
SX_SUFFIX = "_sx0p3_1keV_ph_nh4e20"

LABEL = {
    "A512": "A-512", "A1024": "A-1024", "A2048": "A-2048", "A2048fv": "FV-2048",
    "C256": "C-256", "C512": "C-512", "C1024": "C-1024", "C1024n1000": "C-1024 (N = 1000)",
}  # fmt: skip


# --------------------------------------------------------------------------- inputs


def window(times, bounds):
    """Boolean mask t0 < t <= t1 on times rounded to 1e-3 Myr."""
    t = np.round(np.asarray(times, dtype=float), 3)
    return (t > bounds[0]) & (t <= bounds[1])


def frames_dir(tag):
    """In-situ frame directory of a run (resolved; the cgols_snapshots_* entries are symlinks)."""
    for candidate in (CGOLS / f"cgols_snapshots_{tag}", CGOLS / "cgols_snapshots_Nsweep" / tag):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"no frames for {tag}")


_PLANE_CACHE = {}


def plane_outflow(tag, height_kpc=5.0):
    """(times [Myr], net outflow through both |z| = h planes [Msun/yr]) from the frames."""
    if (tag, height_kpc) not in _PLANE_CACHE:
        times, rates = [], []
        for path in sorted(glob.glob(str(frames_dir(tag) / "frame_*.npz"))):
            with np.load(path) as frame:
                mdot_z = np.asarray(frame["mdot_z"], dtype=float)
                times.append(float(frame["time_myr"]))
            z = (np.arange(mdot_z.size) + 0.5) * BOX_HEIGHT_KPC / mdot_z.size - BOX_HEIGHT_KPC / 2
            rates.append(mdot_z[np.argmin(np.abs(z - height_kpc))] - mdot_z[np.argmin(np.abs(z + height_kpc))])
        order = np.argsort(times)
        _PLANE_CACHE[(tag, height_kpc)] = (np.asarray(times)[order], np.asarray(rates)[order])
    return _PLANE_CACHE[(tag, height_kpc)]


def plane_mean(tag, bounds=BURST, height_kpc=5.0):
    """Window mean of the planar outflow [Msun/yr]."""
    times, rates = plane_outflow(tag, height_kpc)
    return float(rates[window(times, bounds)].mean())


def cumulative_mass(tag, height_kpc=5.0):
    """Mass through both planes from 0 to the last frame (75 Myr) [Msun]."""
    times, rates = plane_outflow(tag, height_kpc)
    return float(np.sum(0.5 * (rates[1:] + rates[:-1]) * np.diff(times)) * 1e6)


def records(prefix, tag):
    """(times, records) of a convergence3d_v3 / shells_v2 cache, sorted by time."""
    path = STUDIES / f"{prefix}{tag}.npz"
    recs = np.load(path, allow_pickle=True)["records"].item()
    recs = sorted(recs.values(), key=lambda record: float(record["time_myr"]))
    return np.array([float(record["time_myr"]) for record in recs]), recs, path


def record_mean(prefix, tag, bounds, extract):
    """Window mean of extract(record) over a 3D cache."""
    times, recs, _ = records(prefix, tag)
    mask = window(times, bounds)
    return float(np.mean([extract(record) for record, keep in zip(recs, mask, strict=True) if keep]))


def h_index(height_kpc):
    """Index of a height in the convergence3d_v3 HEIGHTS_KPC."""
    return int(np.flatnonzero(np.isclose(HEIGHTS_KPC, height_kpc))[0])


def shell_index(radius_kpc):
    """shells_v2 radial bin [r - DR, r) (as cgols_shell_fluxes.py)."""
    return round(radius_kpc / SHELL_DR_KPC) - 1


def emission(tag, suffix=""):
    """An emission cache as a dict of arrays, or None if it does not exist."""
    path = STUDIES / f"emission_{tag}{suffix}.npz"
    if not path.exists():
        return None, path
    with np.load(path, allow_pickle=True) as cache:
        return {key: cache[key] for key in cache.files if key.startswith("per_snapshot__")}, path


def at_time(cache, time_myr):
    """Snapshot index closest to a time."""
    return int(np.argmin(np.abs(cache["per_snapshot__time_myr"] - time_myr)))


def strickland_north_photons_per_arcmin2(z_kpc):
    """Strickland+04 north fit [photons s^-1 cm^-2 arcmin^-2]."""
    return STRICKLAND_NORTH_SIGMA0 * np.exp(-np.asarray(z_kpc) / STRICKLAND_NORTH_H_KPC) * 3600.0


def sx_north(cache, index):
    """(z [kpc], absorbed 0.3-1.0 keV photon surface brightness, north [photons s^-1 cm^-2 arcmin^-2])."""
    z = np.asarray(cache["per_snapshot__profile_z_kpc"][index])
    profile = np.asarray(cache["per_snapshot__profile_xray_atomdb_xz_north"][index]) / ARCMIN2_PER_STERADIAN
    return z, profile


def sx_ratio(cache, z0_kpc):
    """Geometric mean over SX_WINDOW of S_X(z0) / Strickland north fit(z0)."""
    ratios = []
    for index in np.flatnonzero(window(cache["per_snapshot__time_myr"], SX_WINDOW)):
        z, profile = sx_north(cache, index)
        ratios.append(np.interp(z0_kpc, z, profile) / strickland_north_photons_per_arcmin2(z0_kpc))
    return float(np.exp(np.mean(np.log(ratios)))), len(ratios)


def scale_height(cache, time_myr=60.0, z_range=(0.5, 5.0)):
    """Exponential scale height [kpc] of the north profile, least squares in log S_X over z_range."""
    z, profile = sx_north(cache, at_time(cache, time_myr))
    select = (z >= z_range[0]) & (z <= z_range[1])
    return float(-1.0 / np.polyfit(z[select], np.log(profile[select]), 1)[0])


def ladder(values):
    """Convergence verdict and text from {resolution label: value}, ordered low to high."""
    labels = list(values)
    if len(labels) < 2:
        return "unknown", ""
    change = values[labels[-1]] / values[labels[-2]] - 1.0
    verdict = "converged" if abs(change) < CONVERGED_TOLERANCE else "not_converged"
    text = "; ".join(f"{label} {value:.3g}" for label, value in values.items())
    return verdict, f" Ladder: {text} ({change:+.1%} at the last doubling)."


def value_range(values):
    """uncertainty {kind: range} over a ladder."""
    return {"kind": "range", "lo": min(values.values()), "hi": max(values.values())}


# --------------------------------------------------------------------------- numbers


def numbers():
    """Yield (key, kwargs for export_number)."""
    burst = "15-40 Myr (every snapshot after 15 Myr)"

    # Model A planar outflow through |z| = 5 kpc, burst.
    ladder_a = {LABEL[tag]: plane_mean(tag) for tag in ("A512", "A1024", "A2048")}
    verdict, text = ladder(ladder_a)
    for key, tag in (("MdotAfiveKpc", "A2048"), ("MdotAfiveKpcTenTwentyFour", "A1024"),
                     ("MdotAfiveKpcFiveTwelve", "A512")):  # fmt: skip
        yield key, dict(
            value=ladder_a[LABEL[tag]], unit="Msun/yr", status=verdict, convergence=verdict,
            uncertainty=value_range(ladder_a),
            description=f"Mean net outflow through both |z| = 5 kpc planes, {burst}, {LABEL[tag]}, "
                        f"from the frames (1.25 Myr).{text}",
            runs=[LABEL[tag]], data=frames_dir(tag),
        )  # fmt: skip

    masses = {LABEL[tag]: cumulative_mass(tag) for tag in ("A512", "A1024", "A2048")}
    verdict, text = ladder(masses)
    yield "MassAfiveKpc", dict(
        value=masses["A-2048"], unit="Msun", status=verdict, convergence=verdict, uncertainty=value_range(masses),
        description=f"Mass through both |z| = 5 kpc planes from 0 to 75 Myr, A-2048, trapezoid over the frames.{text}",
        runs=["A-2048"], data=frames_dir("A2048"),
    )

    edot = {
        LABEL[tag]: record_mean("convergence3d_v3_", tag, BURST, lambda r: np.sum(r["edot_net"][h_index(5.0)]))
        for tag in ("A512", "A1024", "A2048")
    }
    verdict, text = ladder(edot)
    yield "EdotAfiveKpc", dict(
        value=edot["A-2048"], unit="erg/s", status=verdict, convergence=verdict, uncertainty=value_range(edot),
        description=f"Net energy flux through both |z| = 5 kpc planes, {burst}, A-2048, every checkpoint "
                    f"(1.25 Myr), all phases.{text}",
        runs=["A-2048"], data=STUDIES / "convergence3d_v3_A2048.npz",
    )  # fmt: skip

    synthesis_path = STUDIES / "fvfd_synthesis_metrics.json"
    synthesis = json.loads(synthesis_path.read_text())
    cc85 = {f"FD-{n}": 100.0 * synthesis[f"FD{n}"]["m"]["cc85_dv"] for n in (512, 1024, 2048)}
    verdict, text = ladder(cc85)
    yield "AxisVelocityOffsetCCEightyFive", dict(
        value=cc85["FD-2048"], unit="percent", status=verdict, convergence=verdict, uncertainty=value_range(cc85),
        description="Median offset of the free-wind velocity on the axis (|z| = 1.5-4 kpc) from the CC85 "
                    "solution (Mdot 12 Msun/yr, Edot 5.4e42 erg/s, R = 0.3 kpc), frames 20-38 Myr, A-2048 "
                    f"(cgols_fvfd_synthesis.py, metric cc85_dv); FV-2048 "
                    f"{100.0 * synthesis['FV2048']['m']['cc85_dv']:.2f} %.{text}",
        runs=["A-2048", "FV-2048"], data=synthesis_path, script=CGOLS / "cgols_fvfd_synthesis.py",
    )  # fmt: skip

    mixed = {
        LABEL[tag]: record_mean(
            "convergence3d_v3_", tag, RIM, lambda r: r["mdot_out_K"][h_index(1.0), MIXED_ENTROPY_BIN]
        )
        for tag in ("A512", "A1024", "A2048")
    }
    verdict, text = ladder(mixed)
    yield "MixedMdotAoneKpcRim", dict(
        value=mixed["A-2048"], unit="Msun/yr", status=verdict, convergence=verdict,
        description="Outflow of mixed-entropy gas through both |z| = 1 kpc planes, 30-45 Myr (every checkpoint "
                    f"after 30 Myr, 45 Myr included), A-2048 (A-512 and A-1024 every 2.5 Myr).{text}",
        runs=["A-2048"], data=STUDIES / "convergence3d_v3_A2048.npz",
    )  # fmt: skip

    for key, radius, word in (("ShellMdotAfiveKpc", 5.0, "5"), ("ShellMdotAoneKpc", 1.0, "1")):
        index = shell_index(radius)
        shell = {
            LABEL[tag]: record_mean("shells_v2_", tag, BURST, lambda r, i=index: np.sum(r["mdot_net"][:, i]))
            for tag in ("A512", "A1024", "A2048")
        }
        verdict, text = ladder(shell)
        # The 5 kpc sphere passes the 5 % test only narrowly (+4.6 %, from a 30-45 Myr episode that grows with
        # resolution), and the 1 kpc sphere moves without a trend: both stay provisional.
        yield key, dict(
            value=shell["A-2048"], unit="Msun/yr", status="provisional", convergence=verdict,
            uncertainty=value_range(shell),
            description=f"Net outflow through the full sphere r = {word} kpc (bin [r - 0.125, r) kpc), {burst}, "
                        f"A-2048, every checkpoint.{text}",
            runs=["A-2048"], data=STUDIES / "shells_v2_A2048.npz",
        )  # fmt: skip

    # Model C (N = 16) against A.
    ladder_c = {LABEL[tag]: plane_mean(tag) for tag in ("C256", "C512", "C1024")}
    verdict, text = ladder(ladder_c)
    yield "MdotCfiveKpc", dict(
        value=ladder_c["C-1024"], unit="Msun/yr", status="provisional", convergence=verdict,
        uncertainty=value_range(ladder_c),
        description=f"Mean net outflow through both |z| = 5 kpc planes, {burst}, C-1024 (N = 16), from the frames. "
                    f"Provisional until C-2048.{text}",
        runs=["C-1024"], data=frames_dir("C1024"),
    )  # fmt: skip
    yield "RatioMdotCtoA", dict(
        value=ladder_c["C-1024"] / ladder_a["A-1024"], unit="", status="provisional", convergence=None,
        description=f"Burst outflow through |z| = 5 kpc of C over A at 1024^3, {burst}, frames "
                    f"(C-512 / A-512: {ladder_c['C-512'] / ladder_a['A-512']:.2f}).",
        runs=["C-1024", "A-1024"], data=[frames_dir("C1024"), frames_dir("A1024")],
    )  # fmt: skip
    shell_c = record_mean("shells_v2_", "C1024", BURST, lambda r: np.sum(r["mdot_net"][:, shell_index(5.0)]))
    plane_c = record_mean("convergence3d_v3_", "C1024", BURST, lambda r: np.sum(r["mdot_net"][h_index(5.0)]))
    yield "ShellToPlaneC", dict(
        value=shell_c / plane_c, unit="", status="provisional", convergence=None,
        description=f"Net outflow through the sphere r = 5 kpc over that through both |z| = 5 kpc planes, {burst}, "
                    f"C-1024 (N = 16), the same 2.5 Myr checkpoints ({shell_c:.2f} / {plane_c:.2f} Msun/yr).",
        runs=["C-1024"], data=[STUDIES / "shells_v2_C1024.npz", STUDIES / "convergence3d_v3_C1024.npz"],
    )  # fmt: skip

    # Soft X-rays, CGOLS I recipe (0.3-2 keV window, Eq. A4 cooling curve).
    lx = {}
    for tag in ("A512", "A1024", "A2048", "C256", "C512", "C1024", "C1024n1000"):
        cache, _ = emission(tag)
        lx[tag] = float(cache["per_snapshot__L_xray_cgols_xz"][at_time(cache, 60.0)])
    for key, tags, run in (("LxAsixtyMyr", ("A512", "A1024", "A2048"), "A-2048"),
                           ("LxCsixtyMyr", ("C256", "C512", "C1024"), "C-1024")):  # fmt: skip
        verdict, text = ladder({LABEL[t]: lx[t] for t in tags})
        yield key, dict(
            value=lx[tags[-1]], unit="erg/s", status="single_snapshot", convergence=verdict,
            description=f"Soft X-ray luminosity at 60 Myr, CGOLS I recipe (n^2 Lambda_A4, (3/2) kT in 0.3-2 keV, "
                        f"mu = 0.6), whole box, {run}.{text}",
            runs=[run], data=STUDIES / f"emission_{tags[-1]}.npz",
        )  # fmt: skip

    # Minor-axis soft X-ray profile against Strickland et al. (2004), 0.3-1.0 keV absorbed photons.
    photon = {tag: emission(tag, SX_SUFFIX) for tag in ("A512", "A1024", "A2048", "C1024")}
    missing = [tag for tag, (cache, _) in photon.items() if cache is None]
    if missing:
        print(f"skipped ScaleHeight*/SXratio*: no 0.3-1.0 keV photon caches for {missing} ({SX_SUFFIX})")
    else:
        sx_text = ("AtomDB 3.1.3 0.3-1.0 keV photons after TBabs absorption by N_H = 4e20 cm^-2 (Galactic, toward "
                   "M82), edge-on x-z projection, north")  # fmt: skip
        for key, tags, run in (("ScaleHeightA", ("A512", "A1024", "A2048"), "A-2048"),
                               ("ScaleHeightC", ("C1024",), "C-1024")):  # fmt: skip
            heights = {LABEL[t]: scale_height(photon[t][0]) for t in tags}
            verdict, text = ladder(heights)
            yield key, dict(
                value=heights[LABEL[tags[-1]]], unit="kpc", status="single_snapshot", convergence=verdict,
                description=f"Exponential scale height of the minor-axis soft X-ray profile, least squares in "
                            f"log S_X over 0.5-5 kpc, 60 Myr, {run}; {sx_text}. Strickland+04 north: 1.42 kpc.{text}",
                runs=[run], data=photon[tags[-1]][1],
            )  # fmt: skip
        for model, tags in (("A", ("A512", "A1024", "A2048")), ("C", ("C1024",))):
            for z0, word in ((2.0, "two"), (5.0, "five")):
                ratios = {LABEL[t]: sx_ratio(photon[t][0], z0)[0] for t in tags}
                count = sx_ratio(photon[tags[-1]][0], z0)[1]
                verdict, text = ladder(ratios)
                yield f"SXratio{model}{word}Kpc", dict(
                    value=ratios[LABEL[tags[-1]]], unit="", status="provisional", convergence=verdict,
                    description=f"S_X / Strickland+04 north fit (Sigma_0 = 29.16e-9 photons s^-1 cm^-2 arcsec^-2, "
                                f"H = 1.42 kpc, no conversion factor) at |z| = {z0:g} kpc, geometric mean over "
                                f"50-70 Myr ({count} snapshots after 50 Myr), {LABEL[tags[-1]]}; {sx_text}.{text}",
                    runs=[LABEL[tags[-1]]], data=photon[tags[-1]][1],
                )  # fmt: skip

    # Cluster number N at 512^3 (one run per N).
    sweep = {1: "C512n1", 4: "C512n4", 16: "C512", 64: "C512n64", 128: "C512n128"}
    sweep_means = {n: plane_mean(tag) for n, tag in sweep.items()}
    slope, intercept = np.polyfit(np.log2(list(sweep_means)), list(sweep_means.values()), 1)
    yield "NsweepSlope", dict(
        value=float(slope), unit="Msun/yr", status="provisional", convergence=None,
        description=f"Slope of the mean outflow through |z| = 5 kpc ({burst}, frames) against log2 N, 512^3, "
                    f"N = 1, 4, 16, 64, 128, one run each; means "
                    + ", ".join(f"{value:.2f}" for value in sweep_means.values())
                    + f" Msun/yr; intercept {intercept:.2f}.",
        runs=[f"C-512 N = {n}" for n in sweep], data=[frames_dir(tag) for tag in sweep.values()],
    )  # fmt: skip
    for key, n in (("CumMassNsixteen", 16), ("CumMassNmax", 128)):
        yield key, dict(
            value=cumulative_mass(sweep[n]), unit="Msun", status="provisional", convergence=None,
            description=f"Mass through both |z| = 5 kpc planes from 0 to 75 Myr, C-512 with N = {n}, frames.",
            runs=[f"C-512 N = {n}"], data=frames_dir(sweep[n]),
        )  # fmt: skip

    yield "MdotCthousand", dict(
        value=plane_mean("C1024n1000"), unit="Msun/yr", status="provisional", convergence=None,
        description=f"Mean net outflow through both |z| = 5 kpc planes, {burst}, C-1024 with N = 1000 (one run), "
                    "from the frames.",
        runs=["C-1024 (N = 1000)"], data=frames_dir("C1024n1000"),
    )  # fmt: skip
    yield "LxDropCthousand", dict(
        value=lx["C1024"] / lx["C1024n1000"], unit="", status="single_snapshot", convergence=None,
        description="L_X(N = 16) / L_X(N = 1000) at 60 Myr, C-1024, CGOLS I recipe "
                    f"({lx['C1024']:.3g} / {lx['C1024n1000']:.3g} erg/s).",
        runs=["C-1024", "C-1024 (N = 1000)"],
        data=[STUDIES / "emission_C1024.npz", STUDIES / "emission_C1024n1000.npz"],
    )  # fmt: skip

    # Finite volume against finite difference at 2048^3.
    sys.argv = sys.argv[:1]  # the synthesis module parses no arguments at import, but be safe
    from cgols_fvfd_synthesis import GPU_HOURS

    fd, fv = GPU_HOURS[("FD", 2048)], GPU_HOURS[("FV", 2048)]
    yield "CostRatioFDFV", dict(
        value=fd / fv, unit="", status="converged", convergence=None,
        description=f"GPU-hours to 75 Myr of FD-2048 over FV-2048 ({fd:.0f} / {fv:.0f} H200-h, time integration "
                    "wall time x GPUs; FD without the 780 GPU-h of a leg lost to an InfiniBand hang). "
                    "A measured cost, not a convergence statement.",
        runs=["A-2048", "FV-2048"], data=None, script=CGOLS / "cgols_fvfd_synthesis.py",
    )  # fmt: skip
    for key, tag in (("LxLateFD", "A2048"), ("LxLateFV", "A2048fv")):
        cache, path = emission(tag)
        mask = window(cache["per_snapshot__time_myr"], LATE)
        yield key, dict(
            value=float(np.mean(cache["per_snapshot__L_xray_cgols_xz"][mask])), unit="erg/s",
            status="not_converged", convergence="not_converged",
            description=f"Soft X-ray luminosity, CGOLS I recipe, mean over 60-75 Myr ({mask.sum()} snapshots after "
                        f"60 Myr), {LABEL[tag]}. Neither scheme has converged in L_X.",
            runs=[LABEL[tag]], data=path,
        )  # fmt: skip


# --------------------------------------------------------------------------- figures


def figures():
    """Yield (key, path, kwargs for export_figure)."""
    sx_a = FIGURES / "emission" / f"A2048{SX_SUFFIX}_sx_profile.png"
    sx_c = FIGURES / "emission" / f"C1024{SX_SUFFIX}_sx_profile.png"
    sx_caption = ("Minor-axis soft X-ray surface brightness of {run} at 60 Myr, 0.3-1.0 keV photons after "
                  "Galactic absorption (N_H = 4e20 cm^-2), against the Strickland et al. (2004) M82 fits")  # fmt: skip
    entries = [
        ("slices_A2048", FIGURES / "paper" / "paper_slices_A2048.png", "cgols_paper_slices.py",
         "A-2048 density and temperature x-z slices at 10, 25, 50, 60 Myr (layout of CGOLS I Fig. 6), 300 dpi",
         ["A-2048"]),
        ("outflow_A_ladder", FIGURES / "comparisons" / "a_series_outflow_128_2048.png", "cgols_model_c_compare.py",
         "Net outflow of model A through |z| = 1, 5, 8 kpc and cumulative mass through 5 kpc, 128^3-2048^3",
         ["A-128", "A-256", "A-512", "A-1024", "A-2048"]),
        ("outflow_C_vs_A", FIGURES / "comparisons" / "model_c_outflow_c1024_a2048.png", "cgols_model_c_compare.py",
         "Outflow of C-1024 (N = 16) against A-1024 and A-2048", ["C-1024", "A-1024", "A-2048"]),
        ("sx_profile_A2048", sx_a, "cgols_emission.py", sx_caption.format(run="A-2048"), ["A-2048"]),
        ("sx_profile_C1024", sx_c, "cgols_emission.py", sx_caption.format(run="C-1024"), ["C-1024"]),
        ("shells_A", FIGURES / "comparisons" / "a_series_shells_profiles.png", "cgols_shell_fluxes.py",
         "Mass, momentum and energy flux through spherical shells and bicones, model A 128^3-2048^3",
         ["A-128", "A-256", "A-512", "A-1024", "A-2048"]),
        ("outflow_per_R_A_C", FIGURES / "comparisons" / "outflow_per_R_A_C.png", "cgols_outflow_per_radius.py",
         ("Net mass flux through |z| = 1 and 5 kpc per cylindrical radius bin (R < 0.5, 0.5-1, 1-2, > 2 kpc), "
          "(15, 40] and (50, 75] Myr, models A (256-2048) and C (256-1024, N = 16 and 1000)"),
         ["A-256", "A-512", "A-1024", "A-2048", "C-256", "C-512", "C-1024", "C-1024 (N = 1000)"]),
        ("cluster_number_512", FIGURES / "comparisons" / "model_c_cluster_number_scaling_512.png",
         "cgols_cluster_number_scaling.py", "Outflow against cluster number N at 512^3 (N = 1-128, smooth limit)",
         ["C-512 N = 1-128"]),
        ("fvfd_summary", FIGURES / "comparisons" / "fvfd_summary_significance.png", "cgols_fvfd_synthesis.py",
         "FD - FV differences for all metrics, 128^3-2048^3, in units of their combined scatter",
         ["FD/FV 128-2048"]),
        ("cost_accuracy", FIGURES / "comparisons" / "cost_accuracy_families.png", "cgols_cost_accuracy.py",
         "Error against cost per family of quantities, FD and FV", ["FD/FV 128-2048"]),
    ]  # fmt: skip
    for key, path, script, caption, runs in entries:
        yield key, path, dict(caption_hint=caption, runs=runs, script=CGOLS / script)


# --------------------------------------------------------------------------- main


def main():
    """Compute and export (or print) every number and figure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print, do not write the manifest")
    args = parser.parse_args()
    os.chdir(CGOLS)

    for key, entry in numbers():
        print(f"{key:32s} {entry['value']:11.4g} {entry['unit']:8s} {entry['status']:15s} {entry['convergence']}")
        if not args.dry_run:
            export_number(PAPER, key, **entry)
    for key, path, entry in figures():
        if not path.is_file():
            print(f"{key:32s} MISSING {path}")
            continue
        print(f"{key:32s} {path.relative_to(CGOLS)}")
        if not args.dry_run:
            export_figure(PAPER, key, path, **entry)


if __name__ == "__main__":
    main()
