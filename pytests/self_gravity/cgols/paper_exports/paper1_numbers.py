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

from paper_exports.export import export_figure, export_number, prune

PAPER = "paper1"
STUDIES = CGOLS / "data" / "studies"
FIGURES = CGOLS / "figures" / "cgols"
BOX_HEIGHT_KPC = 20.0  # z extent of the box, -10 ... 10 kpc
CONVERGED_TOLERANCE = 0.05

BURST = (15.0, 40.0)
EARLY = (15.0, 30.0)  # before the 30-45 Myr episode
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
        # The 5 kpc sphere passes the 5 % test only narrowly (+4.6 %), from a 30-45 Myr episode that grows with
        # resolution, and the 1 kpc sphere moves without a trend: both not converged (decision 2026-11-19,
        # asgard-papers #3); the converged 5 kpc loading is ShellMdotAfiveKpcEarly.
        yield key, dict(
            value=shell["A-2048"], unit="Msun/yr", status="provisional", convergence="not_converged",
            uncertainty=value_range(shell),
            description=f"Net outflow through the full sphere r = {word} kpc (bin [r - 0.125, r) kpc), {burst}, "
                        f"A-2048, every checkpoint.{text}",
            runs=["A-2048"], data=STUDIES / "shells_v2_A2048.npz",
        )  # fmt: skip
    early = {
        LABEL[tag]: record_mean("shells_v2_", tag, EARLY, lambda r: np.sum(r["mdot_net"][:, shell_index(5.0)]))
        for tag in ("A512", "A1024", "A2048")
    }
    verdict, text = ladder(early)
    yield "ShellMdotAfiveKpcEarly", dict(
        value=early["A-2048"], unit="Msun/yr", status=verdict, convergence=verdict, uncertainty=value_range(early),
        description="Net outflow through the full sphere r = 5 kpc (bin [r - 0.125, r) kpc), 15-30 Myr (every "
                    "snapshot after 15 Myr), before the 30-45 Myr episode, A-2048, every checkpoint; the "
                    f"burst-inclusive 15-40 Myr value is ShellMdotAfiveKpc.{text}",
        runs=["A-2048"], data=STUDIES / "shells_v2_A2048.npz",
    )  # fmt: skip

    # Model C (N = 16) against A. Keys carry the resolution (TenTwentyFour) until C-2048 is measured; the
    # unsuffixed C keys are reserved for C-2048, as for A (decision 2026-11-19, asgard-papers #2).
    ladder_c = {LABEL[tag]: plane_mean(tag) for tag in ("C256", "C512", "C1024")}
    verdict, text = ladder(ladder_c)
    # -4.8 % passes the 5 % test only just, and the ladder still moves one way (-15 %, then -4.8 %): not
    # converged until C-2048 closes it (decision 2026-11-19, asgard-papers #2).
    yield "MdotCfiveKpcTenTwentyFour", dict(
        value=ladder_c["C-1024"], unit="Msun/yr", status="provisional", convergence="not_converged",
        uncertainty=value_range(ladder_c),
        description=f"Mean net outflow through both |z| = 5 kpc planes, {burst}, C-1024 (N = 16), from the frames. "
                    f"Not converged until C-2048 closes the ladder (one-way trend).{text}",
        runs=["C-1024"], data=frames_dir("C1024"),
    )  # fmt: skip
    yield "RatioMdotCtoATenTwentyFour", dict(
        value=ladder_c["C-1024"] / ladder_a["A-1024"], unit="", status="provisional", convergence=None,
        description=f"Burst outflow through |z| = 5 kpc of C over A at 1024^3, {burst}, frames "
                    f"(C-512 / A-512: {ladder_c['C-512'] / ladder_a['A-512']:.2f}).",
        runs=["C-1024", "A-1024"], data=[frames_dir("C1024"), frames_dir("A1024")],
    )  # fmt: skip
    shell_c = record_mean("shells_v2_", "C1024", BURST, lambda r: np.sum(r["mdot_net"][:, shell_index(5.0)]))
    plane_c = record_mean("convergence3d_v3_", "C1024", BURST, lambda r: np.sum(r["mdot_net"][h_index(5.0)]))
    yield "ShellToPlaneCTenTwentyFour", dict(
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
                           ("LxCsixtyMyrTenTwentyFour", ("C256", "C512", "C1024"), "C-1024")):  # fmt: skip
        verdict, text = ladder({LABEL[t]: lx[t] for t in tags})
        yield key, dict(
            value=lx[tags[-1]], unit="erg/s", status="single_snapshot", convergence=verdict,
            description=f"Soft X-ray luminosity at 60 Myr, CGOLS I recipe (n^2 Lambda_A4, (3/2) kT in 0.3-2 keV, "
                        f"mu = 0.6), whole box, {run}.{text}",
            runs=[run], data=STUDIES / f"emission_{tags[-1]}.npz",
        )  # fmt: skip

    # Our pipeline on the public CGOLS I snapshot (issue #1); the release has only the 2048^3 run.
    cgols_cache, cgols_path = emission("CGOLSI")
    if cgols_cache is None:
        print(f"skipped LxCGOLSsnapshot: no {cgols_path.name} (cgols_cholla_compare.py measure --h5 ... --tag _CGOLSI)")
    else:
        index = at_time(cgols_cache, 60.0)
        lx_cgols = float(cgols_cache["per_snapshot__L_xray_cgols_xz"][index])
        published = 1.9e40
        yield "LxCGOLSsnapshot", dict(
            value=lx_cgols, unit="erg/s", status="single_snapshot", convergence="single_snapshot",
            description=f"Soft X-ray luminosity, CGOLS I recipe as for LxAsixtyMyr, from our pipeline (unchanged) "
                        f"on the public CGOLS I 2048^3 state at "
                        f"{float(cgols_cache['per_snapshot__time_myr'][index]):.3f} Myr (Schneider 2026, "
                        f"doi:10.13139/OLCF/3366667, file 'CGOLS I/60.h5', grid 2048 x 2048 x 4096), whole box. "
                        f"CGOLS I publish {published:.2g} erg/s for this state ({published / lx_cgols:.2f}x); our "
                        f"A-2048 at 60 Myr gives {lx['A2048']:.3g} erg/s. cgols_cholla_compare.py plot: "
                        f"{published:.2g} erg/s needs a window lower edge of about 1.6 MK instead of 2.32 MK "
                        f"((3/2) kT = 0.3 keV), on both states.",
            runs=["CGOLS I 2048^3 (public)"], data=cgols_path, script=CGOLS / "cgols_cholla_compare.py",
        )  # fmt: skip

    # Minor-axis soft X-ray profile against Strickland et al. (2004), 0.3-1.0 keV absorbed photons.
    photon = {tag: emission(tag, SX_SUFFIX) for tag in ("A512", "A1024", "A2048", "C256", "C512", "C1024")}
    missing = [tag for tag, (cache, _) in photon.items() if cache is None]
    if missing:
        print(f"skipped ScaleHeight*/SXratio*: no 0.3-1.0 keV photon caches for {missing} ({SX_SUFFIX})")
    else:
        sx_text = ("AtomDB 3.1.3 0.3-1.0 keV photons after TBabs absorption by N_H = 4e20 cm^-2 (Galactic, toward "
                   "M82), edge-on x-z projection, north")  # fmt: skip
        for key, tags, run in (("ScaleHeightA", ("A512", "A1024", "A2048"), "A-2048"),
                               ("ScaleHeightCTenTwentyFour", ("C256", "C512", "C1024"), "C-1024")):  # fmt: skip
            heights = {LABEL[t]: scale_height(photon[t][0]) for t in tags}
            verdict, text = ladder(heights)
            yield key, dict(
                value=heights[LABEL[tags[-1]]], unit="kpc", status="single_snapshot", convergence=verdict,
                description=f"Exponential scale height of the minor-axis soft X-ray profile, least squares in "
                            f"log S_X over 0.5-5 kpc, 60 Myr, {run}; {sx_text}. Strickland+04 north: 1.42 kpc.{text}",
                runs=[run], data=photon[tags[-1]][1],
            )  # fmt: skip
        for model, tags, suffix in (("A", ("A512", "A1024", "A2048"), ""),
                                    ("C", ("C256", "C512", "C1024"), "TenTwentyFour")):  # fmt: skip
            for z0, word in ((2.0, "two"), (5.0, "five")):
                ratios = {LABEL[t]: sx_ratio(photon[t][0], z0)[0] for t in tags}
                count = sx_ratio(photon[tags[-1]][0], z0)[1]
                verdict, text = ladder(ratios)
                yield f"SXratio{model}{word}Kpc{suffix}", dict(
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

    yield from numbers_v5()
    yield from numbers_v6()
    yield from numbers_v7()
    yield from numbers_v7_sensitivity()
    yield from numbers_v8()
    yield from numbers_v10()


# --------------------------------------------------------------------------- numbers, v5 (asgard-papers #9, #3)

#: Feedback schedule of model A (cgols.build_cgols_wind_params, step onset): knots [Myr] and rates.
SCHEDULE_TIMES_MYR = np.array([0.0, 5.0, 5.0, 10.0, 40.0, 45.0])
SCHEDULE_MDOT = np.array([0.0, 0.0, 1.5, 12.0, 12.0, 1.5])  # Msun/yr; constant after 45 Myr
SCHEDULE_EDOT = np.array([0.0, 0.0, 1.5e42, 5.4e42, 5.4e42, 1.5e42])  # erg/s
WIND_ENTROPY_BIN = 2  # entrained / mixed / wind / halo
R_OUTSIDE_TWO_KPC_BIN = 3  # convergence3d mdot_net_radius bins: R < 0.5, 0.5-1, 1-2, > 2 kpc
FAST_WIND_KMS = 600.0


def injected_mean(rates, bounds):
    """Mean of an injected rate over (t0, t1], from the schedule."""
    t = np.linspace(bounds[0], bounds[1], 200001)[1:]
    return float(np.mean(np.interp(t, SCHEDULE_TIMES_MYR, rates)))


def injected_total_mass(t_end=75.0):
    """Mass injected from 0 to t_end [Msun]."""
    t = np.linspace(0.0, t_end, 750001)
    return float(np.trapezoid(np.interp(t, SCHEDULE_TIMES_MYR, SCHEDULE_MDOT), t) * 1e6)


def cone_half_angle(record):
    """Half-opening angle [deg] of the fast wind at |z| = 1 kpc: tan = sqrt(A_fast / pi) / z, hemispheres averaged.

    A_fast is the area of gas with v_out > 600 km/s in the z cell nearest 1 kpc (convergence3d zarea_fast); the
    same definition as cgols_fvfd_synthesis.metrics_3d (cone1_*), here on every checkpoint.
    """
    z = record["z_kpc"]
    north = z > 0
    area = 0.5 * (record["zarea_fast"][north] + record["zarea_fast"][~north][::-1])
    k = int(np.argmin(np.abs(z[north] - 1.0)))
    return float(np.degrees(np.arctan(np.sqrt(area[k] / np.pi) / z[north][k])))


def structure_mean(tag, key, bounds):
    """(window mean of north + south ``key``, snapshot count) of a structure3d_v2 cache."""
    path = STUDIES / f"structure3d_v2_{tag}.npz"
    recs = np.load(path, allow_pickle=True)["records"].item()
    recs = sorted(recs.values(), key=lambda record: float(record["time_myr"]))
    mask = window([float(record["time_myr"]) for record in recs], bounds)
    values = [record["north"][key] + record["south"][key] for record, keep in zip(recs, mask, strict=True) if keep]
    return float(np.mean(values)), len(values), path


def box_unbound_fraction(tag):
    """(unbound / net mass leaving the box through all six faces, 0-75 Myr; side-face share; mass out [Msun])."""
    times, recs, _ = records("shells_v2_", tag)
    t = np.r_[0.0, times]

    def integral(extract):
        return float(np.trapezoid(np.r_[0.0, [extract(r) for r in recs]], t))

    net = integral(lambda r: r["side_mdot_net_z"].sum() + r["cap_mdot_net"].sum())
    unbound = integral(lambda r: r["side_mdot_unbound_z"].sum() + r["cap_mdot_unbound"].sum())
    side = integral(lambda r: r["side_mdot_net_z"].sum())
    return unbound / net, side / net, net * 1e6


def common_window_mean(prefix, tags, bounds, extract):
    """Window means of extract(record) over the checkpoints all ``tags`` share (times rounded to 1e-3 Myr)."""
    series = {}
    for tag in tags:
        times, recs, _ = records(prefix, tag)
        series[tag] = dict(zip(np.round(times, 3), recs, strict=True))
    common = sorted(set.intersection(*(set(s) for s in series.values())))
    keep = [t for t in common if bounds[0] < t <= bounds[1]]
    return {tag: float(np.mean([extract(series[tag][t]) for t in keep])) for tag in tags}, len(keep)


def seed_range(members):
    """(max - min) / mean of a set of seed realisations."""
    members = np.asarray(members, dtype=float)
    return float((members.max() - members.min()) / members.mean())


def numbers_v5():
    """Yield (key, kwargs for export_number) for the keys requested in asgard-papers #9 and the rest of #3."""
    burst = "15-40 Myr (every snapshot after 15 Myr)"
    early = "15-30 Myr (every snapshot after 15 Myr)"
    ladder_tags = ("A512", "A1024", "A2048")
    h1, h5, i5 = h_index(1.0), h_index(5.0), shell_index(5.0)
    conv = lambda tag: STUDIES / f"convergence3d_v3_{tag}.npz"
    shells = lambda tag: STUDIES / f"shells_v2_{tag}.npz"
    power_burst = injected_mean(SCHEDULE_EDOT, BURST)
    power_early = injected_mean(SCHEDULE_EDOT, EARLY)
    mdot_early = injected_mean(SCHEDULE_MDOT, EARLY)

    # ---- Section 3, model A
    edot1 = {LABEL[t]: record_mean("convergence3d_v3_", t, BURST, lambda r: np.sum(r["edot_net"][h1])) / power_burst
             for t in ladder_tags}  # fmt: skip
    verdict, text = ladder(edot1)
    yield "EdotAoneKpcFraction", dict(
        value=edot1["A-2048"], unit="", status=verdict, convergence=verdict, uncertainty=value_range(edot1),
        description=f"Net energy flux through both |z| = 1 kpc planes, {burst}, A-2048, every checkpoint, all phases, "
                    f"over the mean injected power in the same window ({power_burst:.3g} erg/s).{text}",
        runs=["A-2048"], data=conv("A2048"),
    )  # fmt: skip

    shell_plane = {}
    for t in ladder_tags:
        shell = record_mean("shells_v2_", t, EARLY, lambda r: np.sum(r["mdot_net"][:, i5]))
        plane = record_mean("convergence3d_v3_", t, EARLY, lambda r: np.sum(r["mdot_net"][h5]))
        shell_plane[LABEL[t]] = (shell / plane, shell, plane)
    ratios = {k: v[0] for k, v in shell_plane.items()}
    verdict, text = ladder(ratios)
    _, shell, plane = shell_plane["A-2048"]
    yield "ShellToPlaneA", dict(
        value=ratios["A-2048"], unit="", status=verdict, convergence=verdict, uncertainty=value_range(ratios),
        description=f"Net outflow through the full sphere r = 5 kpc (bin [r - 0.125, r) kpc) over that through both "
                    f"|z| = 5 kpc planes, {early}, A-2048, the same checkpoints ({shell:.2f} / {plane:.2f} Msun/yr; "
                    f"window of ShellMdotAfiveKpcEarly).{text}",
        runs=["A-2048"], data=[shells("A2048"), conv("A2048")],
    )  # fmt: skip

    wind = {LABEL[t]: record_mean("shells_v2_", t, EARLY, lambda r: np.sum(r["mdot_net_K"][:, i5, :, WIND_ENTROPY_BIN]))
            for t in ladder_tags}  # fmt: skip
    verdict, text = ladder(wind)
    yield "ShellWindMdotA", dict(
        value=wind["A-2048"], unit="Msun/yr", status=verdict, convergence=verdict, uncertainty=value_range(wind),
        description=f"Wind-entropy part (6.8 < log K < 7.9) of the net outflow through the full sphere r = 5 kpc, "
                    f"{early}, A-2048, every checkpoint; injected mass rate in the window {mdot_early:.3g} Msun/yr "
                    f"({wind['A-2048'] / mdot_early:.3f}x).{text}",
        runs=["A-2048"], data=shells("A2048"),
    )  # fmt: skip

    edot_shell = {LABEL[t]: record_mean("shells_v2_", t, EARLY, lambda r: np.sum(r["edot_net"][:, i5])) / power_early
                  for t in ladder_tags}  # fmt: skip
    verdict, text = ladder(edot_shell)
    yield "ShellEdotFractionA", dict(
        value=edot_shell["A-2048"], unit="", status=verdict, convergence=verdict, uncertainty=value_range(edot_shell),
        description=f"Net energy flux through the full sphere r = 5 kpc, {early}, A-2048, every checkpoint, all "
                    f"phases, over the mean injected power in the same window ({power_early:.3g} erg/s). Definition "
                    f"proposed here (asgard-papers #3 named the key only).{text}",
        runs=["A-2048"], data=shells("A2048"),
    )  # fmt: skip

    unbound = {LABEL[t]: box_unbound_fraction(t) for t in ladder_tags}
    fractions = {k: v[0] for k, v in unbound.items()}
    verdict, text = ladder(fractions)
    fraction, side_share, mass_out = unbound["A-2048"]
    yield "UnboundFractionA", dict(
        value=fraction, unit="", status=verdict, convergence=verdict, uncertainty=value_range(fractions),
        description=f"Fraction of the mass leaving the box through all six faces, 0-75 Myr, that leaves with positive "
                    f"Bernoulli parameter (v^2/2 + gamma/(gamma-1) P/rho + Phi > 0, outflowing cells), A-2048, "
                    f"trapezoid over the checkpoints. {mass_out:.3g} Msun leave the box, {side_share:.0%} of it "
                    f"through the side faces.{text}",
        runs=["A-2048"], data=shells("A2048"),
    )  # fmt: skip

    def r_outside(tag):
        num = record_mean("convergence3d_v3_", tag, BURST, lambda r: r["mdot_net_radius"][h5, R_OUTSIDE_TWO_KPC_BIN])
        return num / record_mean("convergence3d_v3_", tag, BURST, lambda r: np.sum(r["mdot_net_radius"][h5]))

    for key, tags, run in (("OutflowFracROutsideTwoKpcA", ("A256", "A512", "A1024", "A2048"), "A-2048"),
                           ("OutflowFracROutsideTwoKpcCTenTwentyFour", ("C256", "C512", "C1024"), "C-1024")):  # fmt: skip
        fracs = {LABEL.get(t, t.replace("A", "A-")): r_outside(t) for t in tags}
        verdict, text = ladder(fracs)
        extra = f" C-1024 with N = 1000: {r_outside('C1024n1000'):.3f}." if run == "C-1024" else ""
        yield key, dict(
            value=fracs[run], unit="", status=verdict, convergence=verdict, uncertainty=value_range(fracs),
            description=f"Fraction of the net outflow through both |z| = 5 kpc planes that crosses at cylindrical "
                        f"R > 2 kpc (ratio of window means), {burst}, {run}, every checkpoint.{extra}{text}",
            runs=[run], data=conv(tags[-1]),
        )  # fmt: skip

    cone = {LABEL[t]: record_mean("convergence3d_v3_", t, BURST, cone_half_angle) for t in ladder_tags}
    verdict, text = ladder(cone)
    yield "ConeHalfAngleA", dict(
        value=cone["A-2048"], unit="deg", status=verdict, convergence=verdict, uncertainty=value_range(cone),
        description=f"Half-opening angle of the fast wind (v_out > {FAST_WIND_KMS:.0f} km/s) at |z| = 1 kpc, "
                    f"tan(theta) = sqrt(A_fast / pi) / z, hemispheres averaged, {burst}, A-2048, every checkpoint.{text}",
        runs=["A-2048"], data=conv("A2048"),
    )  # fmt: skip

    mixed = {LABEL[t]: record_mean("convergence3d_v3_", t, RIM, lambda r: r["mdot_out_K"][h1, MIXED_ENTROPY_BIN])
             for t in ladder_tags}  # fmt: skip
    yield "MixedMdotAoneKpcRimChange", dict(
        value=100.0 * (mixed["A-2048"] / mixed["A-1024"] - 1.0), unit="percent", status="not_converged",
        convergence=None,
        description="Relative change of MixedMdotAoneKpcRim (mixed-entropy outflow through both |z| = 1 kpc planes, "
                    f"30-45 Myr) from A-1024 to A-2048: {mixed['A-1024']:.3g} -> {mixed['A-2048']:.3g} Msun/yr "
                    f"(A-512 {mixed['A-512']:.3g}).",
        runs=["A-1024", "A-2048"], data=[conv("A1024"), conv("A2048")],
    )  # fmt: skip

    fd = {n: structure_mean(tag, "area_wind_kpc2", (45.0, 75.0)) for n, tag in ((1024, "A1024"), (2048, "A2048"))}
    fv = {n: structure_mean(tag, "area_wind_kpc2", (45.0, 75.0)) for n, tag in ((1024, "A1024fvr"), (2048, "A2048fv"))}
    fv_seeds = {1024: [structure_mean(t, "area_wind_kpc2", (45.0, 75.0))[0] for t in ("A1024fvr", "A1024fvs2", "A1024fvs3")],
                2048: [structure_mean(t, "area_wind_kpc2", (45.0, 75.0))[0] for t in ("A2048fv", "A2048fvs1")]}  # fmt: skip
    fd_seed_1024 = structure_mean("A1024s1", "area_wind_kpc2", (45.0, 75.0))[0]
    yield "MixingLayerAreaGrowth", dict(
        value=fd[2048][0] / fd[1024][0], unit="", status="not_converged", convergence="not_converged",
        description="Growth factor of the mixing-layer area (log K = 6.8 isosurface, both hemispheres) from 1024^3 to "
                    f"2048^3, mean over 45-75 Myr (every snapshot after 45 Myr, {fd[2048][1]} snapshots), FD: "
                    f"{fd[1024][0]:.0f} -> {fd[2048][0]:.0f} kpc^2 (FD-1024 seed 1: {fd_seed_1024:.0f}). FV: "
                    f"{fv[1024][0]:.0f} -> {fv[2048][0]:.0f} kpc^2 ({fv[2048][0] / fv[1024][0]:.2f}x); seed means "
                    f"{np.mean(fv_seeds[1024]):.0f} -> {np.mean(fv_seeds[2048]):.0f} kpc^2 "
                    f"({np.mean(fv_seeds[2048]) / np.mean(fv_seeds[1024]):.2f}x). FD-512 "
                    f"{structure_mean('A512', 'area_wind_kpc2', (45.0, 75.0))[0]:.0f} kpc^2.",
        runs=["A-1024", "A-2048", "FV-1024", "FV-2048"], data=[fd[1024][2], fd[2048][2], fv[1024][2], fv[2048][2]],
    )  # fmt: skip

    # ---- Section 4, model C
    ladder_c = ("C256", "C512", "C1024")
    wind_c = {LABEL[t]: record_mean("convergence3d_v3_", t, EARLY, lambda r: r["mdot_out_K"][h5, WIND_ENTROPY_BIN])
              for t in ladder_c}  # fmt: skip
    verdict, text = ladder(wind_c)
    yield "WindMdotCfiveKpcTenTwentyFour", dict(
        value=wind_c["C-1024"], unit="Msun/yr", status=verdict, convergence=verdict, uncertainty=value_range(wind_c),
        description=f"Outflow of wind-entropy gas (6.8 < log K < 7.9) through both |z| = 5 kpc planes, {early}, "
                    f"C-1024 (N = 16), every checkpoint (2.5 Myr).{text}",
        runs=["C-1024"], data=conv("C1024"),
    )  # fmt: skip
    mixed_c = {LABEL[t]: record_mean("convergence3d_v3_", t, EARLY, lambda r: r["mdot_out_K"][h5, MIXED_ENTROPY_BIN])
               for t in ladder_c}  # fmt: skip
    drops = (mixed_c["C-256"] / mixed_c["C-512"], mixed_c["C-512"] / mixed_c["C-1024"])
    yield "MixedMdotCfiveKpcDrop", dict(
        value=float(np.sqrt(drops[0] * drops[1])), unit="", status="not_converged", convergence="not_converged",
        uncertainty={"kind": "range", "lo": min(drops), "hi": max(drops)},
        description=f"Factor by which the outflow of mixed-entropy gas through both |z| = 5 kpc planes falls per "
                    f"doubling, {early}, C-256 -> C-1024 (geometric mean of {drops[0]:.2f} and {drops[1]:.2f}; "
                    + ", ".join(f"{k} {v:.3g}" for k, v in mixed_c.items()) + " Msun/yr). The drop steepens.",
        runs=list(mixed_c), data=[conv(t) for t in ladder_c],
    )  # fmt: skip
    disc = {}
    for t in ladder_c:
        times, recs, _ = records("convergence3d_v3_", t)
        disc[LABEL[t]] = (float(np.sum(recs[-1]["mass_disk"])), float(times[-1]), float(np.sum(recs[0]["mass_disk"])),
                          float(times[0]))  # fmt: skip
    masses = {k: v[0] for k, v in disc.items()}
    verdict, text = ladder(masses)
    yield "DiscMassCTenTwentyFour", dict(
        value=masses["C-1024"], unit="Msun", status=verdict, convergence=verdict, uncertainty=value_range(masses),
        description=f"Gas mass at |z| < 0.5 kpc at {disc['C-1024'][1]:.0f} Myr, C-1024 (N = 16); "
                    f"{disc['C-1024'][2]:.3g} Msun at {disc['C-1024'][3]:.1f} Myr, before the 5 Myr feedback onset.{text}",
        runs=["C-1024"], data=conv("C1024"),
    )  # fmt: skip

    # ---- Section 5, soft X-rays
    scatter = {}
    for t in ("A2048", "C1024"):
        cache, path = emission(t)
        sel = window(cache["per_snapshot__time_myr"], SX_WINDOW)
        lx = cache["per_snapshot__L_xray_cgols_xz"][sel]
        scatter[t] = (float(lx.max() / lx.min()), int(sel.sum()), float(np.exp(np.median(np.abs(np.log(lx[1:] / lx[:-1]))))),
                      path)  # fmt: skip
    yield "LxSnapshotScatter", dict(
        value=scatter["A2048"][0], unit="", status="provisional", convergence=None,
        description=f"Max / min of the soft X-ray luminosity (CGOLS I recipe) over 50-70 Myr ({scatter['A2048'][1]} "
                    f"snapshots after 50 Myr, 1.25 Myr apart), A-2048; median ratio of neighbouring snapshots "
                    f"{scatter['A2048'][2]:.2f}. C-1024: max / min {scatter['C1024'][0]:.2f}, neighbours "
                    f"{scatter['C1024'][2]:.2f} ({scatter['C1024'][1]} snapshots).",
        runs=["A-2048", "C-1024"], data=[scatter["A2048"][3], scatter["C1024"][3]],
    )  # fmt: skip

    photon = {t: emission(t, SX_SUFFIX) for t in ("A1024", "A2048", "C1024n1000")}
    changes = {z0: sx_ratio(photon["A2048"][0], z0)[0] / sx_ratio(photon["A1024"][0], z0)[0] - 1.0 for z0 in (2.0, 5.0)}
    largest = max(changes.values(), key=abs)
    yield "SXratioAchange", dict(
        value=100.0 * abs(largest), unit="percent", status="not_converged", convergence=None,
        description="Largest relative change of SXratioAtwoKpc and SXratioAfiveKpc from A-1024 to A-2048: "
                    f"{100 * changes[2.0]:+.1f} % at 2 kpc, {100 * changes[5.0]:+.1f} % at 5 kpc.",
        runs=["A-1024", "A-2048"], data=[photon["A1024"][1], photon["A2048"][1]],
    )  # fmt: skip
    sx_text = ("AtomDB 3.1.3 0.3-1.0 keV photons after TBabs absorption by N_H = 4e20 cm^-2, edge-on x-z projection, "
               "north")  # fmt: skip
    cache_1k, path_1k = photon["C1024n1000"]
    for z0, word in ((2.0, "two"), (5.0, "five")):
        value, count = sx_ratio(cache_1k, z0)
        yield f"SXratioCthousand{word}Kpc", dict(
            value=value, unit="", status="provisional", convergence="unknown",
            description=f"S_X / Strickland+04 north fit at |z| = {z0:g} kpc, geometric mean over 50-70 Myr ({count} "
                        f"snapshots after 50 Myr), C-1024 with N = 1000 (one resolution); {sx_text}.",
            runs=["C-1024 (N = 1000)"], data=path_1k,
        )  # fmt: skip
    yield "ScaleHeightCthousand", dict(
        value=scale_height(cache_1k), unit="kpc", status="single_snapshot", convergence="unknown",
        description=f"Exponential scale height of the minor-axis soft X-ray profile, least squares in log S_X over "
                    f"0.5-5 kpc, 60 Myr, C-1024 with N = 1000 (one resolution); {sx_text}. Strickland+04 north: 1.42 kpc.",
        runs=["C-1024 (N = 1000)"], data=path_1k,
    )  # fmt: skip
    c16, path16 = emission("C1024")
    c1k, path1k = emission("C1024n1000")
    i16, i1k = at_time(c16, 15.0), at_time(c1k, 15.0)
    lx_burst = {key: cache["per_snapshot__L_xray_cgols_xz"][window(cache["per_snapshot__time_myr"], BURST)].mean()
                for key, cache in (("16", c16), ("1000", c1k))}  # fmt: skip
    yield "LxBurstRatioCthousand", dict(
        value=float(c1k["per_snapshot__L_xray_cgols_xz"][i1k] / c16["per_snapshot__L_xray_cgols_xz"][i16]),
        unit="", status="single_snapshot", convergence="unknown",
        description="L_X(N = 1000) / L_X(N = 16) at 15 Myr, C-1024, CGOLS I recipe (one snapshot each); the ratio of "
                    f"the {burst} means is {lx_burst['1000'] / lx_burst['16']:.2f}.",
        runs=["C-1024", "C-1024 (N = 1000)"], data=[path16, path1k],
    )  # fmt: skip

    # ---- Appendix A, FV against FD (the synthesis's own statistics: sigma from seeds and time sampling)
    import cgols_fvfd_synthesis as synthesis

    synthesis_path = STUDIES / "fvfd_synthesis_metrics.json"
    runs_fvfd, data_fvfd = synthesis.load_all(False, {})
    st = synthesis.stats(runs_fvfd, data_fvfd)
    late_lx = synthesis.differences(st["lx1_60_75"])[2048]
    yield "LxLateZ", dict(
        value=late_lx["z"], unit="", status="provisional", convergence=None,
        description="(FD - FV) / sigma_diff of the soft X-ray luminosity (CGOLS I recipe) over 60-75 Myr at 2048^3 "
                    f"(cgols_fvfd_synthesis.py: sigma from seed scatter and time sampling); FD - FV = "
                    f"{late_lx['delta']:.3g}e39 erg/s ({100 * late_lx['rel']:+.1f} %).",
        runs=["A-2048", "FV-2048"], data=synthesis_path,
    )  # fmt: skip
    rows = synthesis.differences(st["out5_15_40"])
    slope, error, _ = synthesis.trend(rows, ns=(128, 256, 512, 1024))
    slope_256, error_256, _ = synthesis.trend(rows, ns=(256, 512, 1024))
    yield "FVFDGapExponent", dict(
        value=-slope, unit="", status="provisional", convergence=None,
        uncertainty={"kind": "noise", "lo": -slope - error, "hi": -slope + error},
        description="Exponent p in |FD - FV| ~ N^-p of the mean outflow through |z| = 5 kpc, 15-40 Myr, weighted fit "
                    f"over 128^3-1024^3 (cgols_fvfd_synthesis.trend); over 256^3-1024^3 p = {-slope_256:.2f} +- "
                    f"{error_256:.2f}. FD - FV = " + ", ".join(f"{rows[n]['delta']:+.3f}" for n in sorted(rows))
                    + f" Msun/yr at 128^3-2048^3; at 2048^3 the sign flips within the noise (z = {rows[2048]['z']:.1f}).",
        runs=["FD/FV 128-2048"], data=synthesis_path,
    )  # fmt: skip
    ha = synthesis.differences(st["ha_60_75"])[2048]
    fd_ha, fv_ha = st["ha_60_75"][("FD", 2048)]["value"], st["ha_60_75"][("FV", 2048)]["value"]
    yield "HalphaLateDiffFVFD", dict(
        value=100.0 * (fv_ha / fd_ha - 1.0), unit="percent", status="provisional", convergence=None,
        description=f"(FV - FD) / FD of the H-alpha luminosity over 60-75 Myr at 2048^3 ({fv_ha:.3g} vs {fd_ha:.3g} "
                    f"e41 erg/s); z = (FD - FV) / sigma_diff = {ha['z']:.1f} (cgols_fvfd_synthesis.py).",
        runs=["A-2048", "FV-2048"], data=synthesis_path,
    )  # fmt: skip
    mix = synthesis.differences(st["mixed1_30_45"])[2048]
    fd_mix, fv_mix = st["mixed1_30_45"][("FD", 2048)]["value"], st["mixed1_30_45"][("FV", 2048)]["value"]
    native, count = common_window_mean("convergence3d_v3_", ("A2048", "A2048fv", "A2048fvs1"), RIM,
                                       lambda r: r["mdot_out_K"][h1, MIXED_ENTROPY_BIN])  # fmt: skip
    yield "MixedMdotFVFDDiff", dict(
        value=100.0 * (fv_mix / fd_mix - 1.0), unit="percent", status="provisional", convergence=None,
        description=f"(FV - FD) / FD of the mixed-entropy outflow through both |z| = 1 kpc planes, 30-45 Myr, 2048^3, "
                    f"on the synthesis's 5 Myr cadence ({fv_mix:.3g} vs {fd_mix:.3g} Msun/yr, z = (FD - FV) / "
                    f"sigma_diff = {mix['z']:.1f}). On the {count} checkpoints all three runs share (2.5 Myr): FD "
                    f"{native['A2048']:.3g}, FV {native['A2048fv']:.3g} ({100 * (native['A2048fv'] / native['A2048'] - 1):+.1f} %), "
                    f"FV seed 1 {native['A2048fvs1']:.3g} ({100 * (native['A2048fvs1'] / native['A2048'] - 1):+.1f} %).",
        runs=["A-2048", "FV-2048"], data=[synthesis_path, conv("A2048"), conv("A2048fv"), conv("A2048fvs1")],
    )  # fmt: skip
    shell_fvfd, count = common_window_mean("shells_v2_", ("A2048", "A2048fv"), BURST,
                                           lambda r: np.sum(r["mdot_net"][:, i5]))  # fmt: skip
    shell_fvfd_1, _ = common_window_mean("shells_v2_", ("A2048", "A2048fv"), BURST,
                                         lambda r: np.sum(r["mdot_net"][:, shell_index(1.0)]))  # fmt: skip
    yield "ShellFVFDRatioFiveKpc", dict(
        value=shell_fvfd["A2048fv"] / shell_fvfd["A2048"], unit="", status="provisional", convergence=None,
        description=f"FV-2048 / FD-2048 net outflow through the full sphere r = 5 kpc, {burst}, on the {count} "
                    f"checkpoints both share ({shell_fvfd['A2048fv']:.2f} / {shell_fvfd['A2048']:.2f} Msun/yr); r = 1 kpc: "
                    f"{shell_fvfd_1['A2048fv'] / shell_fvfd_1['A2048']:.3f}.",
        runs=["A-2048", "FV-2048"], data=[shells("A2048"), shells("A2048fv")],
    )  # fmt: skip

    # ---- Appendix B, cluster number
    yield "InjectedMassTotal", dict(
        value=injected_total_mass(), unit="Msun", status="converged", convergence=None,
        description="Mass injected from 0 to 75 Myr by the feedback schedule (cgols.py build_cgols_wind_params: "
                    "0 before 5 Myr, 1.5 Msun/yr from 5 Myr, linear to 12 Msun/yr at 10 Myr, 12 to 40 Myr, linear "
                    "to 1.5 at 45 Myr, then 1.5); the same in every model.",
        runs=["all"], data=None, script=CGOLS / "cgols.py",
    )  # fmt: skip
    groups = {16: ("C256", "C256s2", "C256s3"), 64: ("C256n64", "C256n64s2", "C256n64s3")}
    spread = {n: {t: plane_mean(t) for t in tags} for n, tags in groups.items()}
    size = {n: plane_mean(f"C256R119n{n}") / plane_mean(f"C256n{n}") - 1.0 for n in (1, 4, 8, 32, 64)}
    yield "NsweepRealisationScatter", dict(
        value=100.0 * seed_range(list(spread[16].values())), unit="percent", status="provisional", convergence=None,
        description=f"(max - min) / mean of the mean outflow through |z| = 5 kpc, {burst}, over three random cluster "
                    "realisations at 256^3 with N = 16 (" + ", ".join(f"{v:.2f}" for v in spread[16].values())
                    + f" Msun/yr; std/mean {100 * np.std(list(spread[16].values()), ddof=1) / np.mean(list(spread[16].values())):.1f} %). "
                    f"N = 64: {100 * seed_range(list(spread[64].values())):.1f} %. Cluster radius fixed at 119 pc instead of "
                    "300 pc / N^(1/3): " + ", ".join(f"N = {n} {100 * v:+.1f} %" for n, v in size.items()) + ".",
        runs=["C-256 N = 16 (3 realisations)", "C-256 N = 64 (3 realisations)"],
        data=[frames_dir(t) for tags in groups.values() for t in tags],
    )  # fmt: skip

    # ---- Appendix C, noise and seeds (seed sets of the synthesis: base + ensemble members)
    for key, metric, what in (("SeedScatterEarly", "out5_15_40", f"mean outflow through |z| = 5 kpc, {burst}"),
                              ("SeedScatterLateLx", "lx1_60_75", "soft X-ray luminosity (CGOLS I recipe), 60-75 Myr")):  # fmt: skip
        sets = {f"{scheme}-{n}": st[metric][(scheme, n)]["members"]
                for scheme, n in (("FD", 512), ("FD", 1024), ("FV", 1024), ("FV", 2048))
                if (scheme, n) in st[metric] and st[metric][(scheme, n)]["n_members"] >= 2}  # fmt: skip
        ranges = {label: 100.0 * seed_range(m) for label, m in sets.items()}
        worst = max(ranges, key=ranges.get)
        yield key, dict(
            value=ranges[worst], unit="percent", status="provisional", convergence=None,
            description=f"Largest (max - min) / mean between initial-condition seeds of the {what}, over the seed sets "
                        "at 512^3 and above: " + "; ".join(f"{label} {ranges[label]:.2f} % ({len(sets[label])} seeds)"
                                                           for label in sets) + f". Largest: {worst}.",
            runs=list(sets), data=synthesis_path,
        )  # fmt: skip
    asym = st["asym_50_75"]
    yield "UpDownAsymmetryFD", dict(
        value=asym[("FD", 2048)]["value"], unit="dex", status="provisional", convergence=None,
        uncertainty={"kind": "noise", "lo": asym[("FD", 2048)]["value"] - asym[("FD", 2048)]["sigma"],
                     "hi": asym[("FD", 2048)]["value"] + asym[("FD", 2048)]["sigma"]},
        description="rms of log10 n(x, z) - log10 n(x, -z) over the x-z density slice, mean over 50-75 Myr (frames), "
                    f"FD-2048; FV-2048 {asym[('FV', 2048)]['value']:.3f} (z = "
                    f"{synthesis.differences(asym)[2048]['z']:.1f}); uncertainty = the synthesis's sigma.",
        runs=["A-2048", "FV-2048"], data=synthesis_path,
    )  # fmt: skip


# --------------------------------------------------------------------------- numbers, v6 (asgard-papers #11)


def numbers_v6():
    """Yield (key, kwargs for export_number) for the Appendix A keys of asgard-papers #11."""
    import cgols_cost_accuracy as cost_accuracy
    import cgols_fvfd_synthesis as synthesis

    synthesis_path = STUDIES / "fvfd_synthesis_metrics.json"
    runs_fvfd, data_fvfd = synthesis.load_all(False, {})
    st = synthesis.stats(runs_fvfd, data_fvfd)

    def k_text(key, ns=(256, 512, 1024)):
        parts = []
        for n in ns:
            k, q, note = synthesis.k_factor(st[key], n, key=key)
            parts.append(f"{n}^3: {k:.2f} [{q[0]:.2f}, {q[1]:.2f}]" if q and np.isfinite(k) else f"{n}^3: {note}")
        return "; ".join(parts)

    for key, metric, name, n in (
        ("FVEquivalentResolutionSideLoss", "side_loss", "mass lost through the side faces, 0-75 Myr", 512),
        ("FVEquivalentResolutionBoxMass", "mass_75", "gas mass in the box at 75 Myr", 512),
        ("FVEquivalentResolutionLateOutflow", "out5_50_75", "mean outflow through |z| = 5 kpc, 50-75 Myr", 1024),
    ):
        k, q, _note = synthesis.k_factor(st[metric], n, key=metric)
        yield key, dict(
            value=float(k), unit="", status="provisional", convergence=None,
            uncertainty={"kind": "noise", "lo": q[0], "hi": q[1]} if q else None,
            description=f"FV at N matches FD at k N in the {name} (FD ladder interpolated in log2 N; 16-84 % from the "
                        f"seed noise; cgols_fvfd_synthesis.k_factor, Fig. A.1b); value at N = {n}. Per N: "
                        f"{k_text(metric)}." + (" Below 1024 FV lies above every FD rung up to 2048, so no k exists."
                                                 if metric == "out5_50_75" else ""),
            runs=["FD/FV 128-2048"], data=synthesis_path,
        )  # fmt: skip

    h1 = h_index(1.0)
    tags = ("A2048", "A2048fv", "A2048fvs1")
    early, count = common_window_mean("convergence3d_v3_", tags, EARLY, lambda r: r["mdot_out_K"][h1, MIXED_ENTROPY_BIN])
    yield "MixedMdotFVFDDiffEarly", dict(
        value=100.0 * (early["A2048fv"] / early["A2048"] - 1.0), unit="percent", status="provisional",
        convergence=None,
        description=f"(FV - FD) / FD of the mixed-entropy outflow through both |z| = 1 kpc planes, 15-30 Myr, 2048^3, "
                    f"on the {count} checkpoints FD-2048 and both FV-2048 seeds share (2.5 Myr): FD {early['A2048']:.3g}, "
                    f"FV {early['A2048fv']:.3g}, FV seed 1 {early['A2048fvs1']:.3g} Msun/yr. The FV seeds agree to "
                    f"{100 * abs(early['A2048fvs1'] / early['A2048fv'] - 1):.0f} %: a scheme effect. Its cause is timing: the "
                    "mixing episode at the cavity rim starts at 25 Myr in FV-2048 and at 27.5 Myr in FD-2048 (FV-1024: "
                    "27.5, FD-1024: 30 Myr), so the window mean depends on where the onset falls and its time-sampling "
                    "error is large (z = -0.9 against it).",
        runs=["A-2048", "FV-2048"], data=[STUDIES / f"convergence3d_v3_{t}.npz" for t in tags],
        script=CGOLS / "paper_exports" / "paper1_numbers.py",
    )  # fmt: skip

    rim = cost_accuracy.rim_values(runs_fvfd)
    value = {lab: v[0] for lab, v in rim.items()}
    fv_2048 = [value[lab] for lab in ("FV2048", "FV2048s1") if lab in value]
    fv_1024 = [value[lab] for lab in ("FV1024", "FV1024s1", "FV1024s2", "FV1024s3") if lab in value]
    common = ("roughness sigma_R / <R> of the midplane cavity rim (n = 0.1 cm^-3 contour), mean over 20-35 Myr "
              "(cgols_convergence_3d.rim_metrics, the cost-accuracy quantity)")  # fmt: skip
    ladder_text = (f"FD 512 / 1024 / 2048: {value['FD512']:.3f} / {value['FD1024']:.3f} / {value['FD2048']:.3f} "
                   f"(FD-1024 seed 1 {value.get('FD1024s1', np.nan):.3f}); FV 512 / 1024 (4 seeds) / 2048 (2 seeds): "
                   f"{value['FV512']:.3f} / {np.mean(fv_1024):.3f} / {np.mean(fv_2048):.3f}.")  # fmt: skip
    yield "RimRoughnessFD", dict(
        value=value["FD2048"], unit="", status="not_converged", convergence="not_converged",
        description=f"Cavity-rim {common}, FD-2048; doubles per resolution doubling. {ladder_text}",
        runs=["A-2048", "A-1024", "A-512"], data=None, script=CGOLS / "cgols_cost_accuracy.py",
    )  # fmt: skip
    yield "RimRoughnessFV", dict(
        value=float(np.mean(fv_2048)), unit="", status="not_converged", convergence="not_converged",
        uncertainty={"kind": "range", "lo": min(fv_2048), "hi": max(fv_2048)},
        description=f"Cavity-rim {common}, FV-2048, mean of the two seeds ({', '.join(f'{v:.3f}' for v in fv_2048)}); "
                    f"FV-512 is already as rough as FD-2048. {ladder_text}",
        runs=["FV-2048", "FV-1024", "FV-512"], data=None, script=CGOLS / "cgols_cost_accuracy.py",
    )  # fmt: skip


# --------------------------------------------------------------------------- numbers, v7 (asgard-papers #11, 20 Nov)
RIM_ONSET_MDOT = 1.0  # Msun/yr: mixed-entropy outflow through |z| = 1 kpc that marks the onset of rim mixing


def rim_onset(tag):
    """First checkpoint after the 15-25 Myr minimum where the mixed outflow at |z| = 1 kpc exceeds RIM_ONSET_MDOT,
    and the checkpoint before it (the onset lies in between)."""
    times, recs, _ = records("convergence3d_v3_", tag)
    times = np.round(times, 3)
    y = np.array([r["mdot_out_K"][h_index(1.0), MIXED_ENTROPY_BIN] for r in recs])
    early = (times > 15.0) & (times <= 25.0)
    i_min = int(np.flatnonzero(early)[np.argmin(y[early])])
    above = np.flatnonzero((np.arange(len(y)) > i_min) & (y > RIM_ONSET_MDOT))
    return float(times[above[0]]), float(times[above[0] - 1])


def numbers_v7():
    """Yield (key, kwargs for export_number) for the Section 4 numbers of asgard-papers #11 (comment of 20 Nov)."""
    import cgols_cost_accuracy as cost_accuracy
    import cgols_fvfd_statistics as statistics
    import cgols_scheme_convergence as scheme_convergence

    sc_script = CGOLS / "cgols_scheme_convergence.py"
    st_script = CGOLS / "cgols_fvfd_statistics.py"
    runs_ca, data_ca, ex_ca, scal = cost_accuracy.collect()
    rows = scheme_convergence.analyse(scal)

    # self-convergence per scheme at 10 %
    tol = 10.0
    count = {"FD": 0, "FV": 0, "same": 0, "neither": 0}
    for key, r in rows.items():
        n_fd = scheme_convergence.self_convergence(key, r["cells"], "FD", tol)[0]
        n_fv = scheme_convergence.self_convergence(key, r["cells"], "FV", tol)[0]
        if n_fd is None and n_fv is None:
            count["neither"] += 1
        elif n_fd == n_fv:
            count["same"] += 1
        elif n_fv is None or (n_fd is not None and n_fd < n_fv):
            count["FD"] += 1
        else:
            count["FV"] += 1
    sc_common = (f"of the {len(rows)} cost-accuracy quantities (scalars; cgols_scheme_convergence.py item 1): lowest N "
                 f"from which every run of a scheme up to 1024^3 stays within {tol:g} % of the same scheme at 2048^3 "
                 f"(nominal, without the seed noise)")  # fmt: skip
    for key, which, text in (
        ("SelfConvFDEarlier", "FD", "Number for which FD self-converges at a lower N than FV"),
        ("SelfConvFVEarlier", "FV", "Number for which FV self-converges at a lower N than FD"),
        ("SelfConvSame", "same", "Number for which FD and FV self-converge at the same N below 2048^3"),
        ("SelfConvNeither", "neither", "Number for which neither scheme self-converges below 2048^3"),
    ):
        yield key, dict(
            value=count[which], unit="", status="provisional", convergence=None, digits=2,
            description=f"{text}, {sc_common}. All four: FD earlier {count['FD']}, FV earlier {count['FV']}, same "
                        f"{count['same']}, neither {count['neither']}.",
            runs=["FD/FV 128-2048"], data=None, script=sc_script,
        )  # fmt: skip
    yield "NQuantitiesFDFV", dict(
        value=len(rows), unit="", status="provisional", convergence=None, digits=2,
        description="Number of scalar quantities in the FD/FV comparison of cgols_scheme_convergence.py (the "
                    "cost-accuracy quantities without the time-series and n-T distribution distances).",
        runs=["FD/FV 128-2048"], data=None, script=sc_script,
    )  # fmt: skip
    n_sig = sum(abs(r["z_top"]) > scheme_convergence.Z_AGREE for r in rows.values())
    yield "NSigDiffFDFV", dict(
        value=n_sig, unit="", status="provisional", convergence=None, digits=2,
        description=f"Number of the {len(rows)} quantities for which FD-2048 and FV-2048 differ by more than "
                    f"{scheme_convergence.Z_AGREE:g} sigma of the seed noise (sigma from FV-2048 seed 1 and FD-1024 "
                    "seed 1; cgols_scheme_convergence.py item 3).",
        runs=["A-2048", "FV-2048", "FV-2048 seed 1", "A-1024 seed 1"], data=None, script=sc_script,
    )  # fmt: skip

    # cheapest run per family (the cost-accuracy table)
    result = cost_accuracy.analyse(runs_ca, data_ca, ex_ca, scal)
    cases = {"nominal": [], "confident": []}
    for t in (5.0, 10.0, 20.0):
        for d in cost_accuracy.cheapest(result, t).values():
            for kind, winners in cases.items():
                if d[kind] is not None and d[kind][1] < cost_accuracy.REF_N:
                    winners.append(d[kind][0])
    n_fv, n_all = cases["nominal"].count("FV"), len(cases["nominal"])
    c_fv, c_all = cases["confident"].count("FV"), len(cases["confident"])
    for key, value, text in (
        ("CheapestRunFVCount", n_fv, "Number of them in which the cheapest run is FV"),
        ("CheapestRunCasesCount", n_all, "Number of family x tolerance cases"),
    ):
        yield key, dict(
            value=value, unit="", status="provisional", convergence=None, digits=2,
            description=f"{text}: of the 12 families x 3 tolerances (5, 10, 20 %) of the cost-accuracy table, the "
                        f"cases whose cheapest run (nominal, worst quantity within the tolerance) lies below 2048^3: "
                        f"{n_all}, of which FV {n_fv}. With the reference uncertainty counted (confident column): "
                        f"{c_all}, of which FV {c_fv}.",
            runs=["FD/FV 128-2048"], data=None, script=CGOLS / "cgols_cost_accuracy.py",
        )  # fmt: skip

    # higher-order statistics at 2048^3 (base runs FD-2048 and FV-2048 seed 0; seed noise from the seeds)
    stats = statistics.collect()
    table = statistics.scalar_table(stats)
    comp = {name: statistics.compare(values) for name, values in table.items()}

    def at2048(name):
        return comp[name][2048]

    def seed1(name):
        return table[name][("FV", 2048)][1]

    st_json = STUDIES / "fvfd_statistics.json"
    widths = {f"{w} {phase} {var}": at2048(f"PDF {w}: {phase} width log {var}")
              for w in statistics.WINDOWS for phase in ("mixed", "wind") for var in ("n", "T")}  # fmt: skip
    dev = {k: 100.0 * abs(v["FV"] / v["FD"] - 1.0) for k, v in widths.items()}
    worst = max(dev, key=dev.get)
    yield "PDFWidthAgreement", dict(
        value=dev[worst], unit="percent", status="provisional", convergence=None, digits=2,
        description=f"Largest |FV / FD - 1| of the widths (mass-weighted std) of the log n and log T PDFs of the mixed- "
                    f"and wind-entropy gas at |z| > 1 kpc, FD-2048 vs FV-2048, windows (30, 45] and (45, 75] Myr; the "
                    f"largest is {worst} ({dev[worst]:.1f} %), the others " + ", ".join(
                        f"{k} {v:.1f} %" for k, v in dev.items() if k != worst) + ".",
        runs=["A-2048", "FV-2048"], data=st_json, script=st_script,
    )  # fmt: skip
    det = at2048("clumps 45-75: detached clumps > 1e+02 Msun")
    big = at2048("clumps 45-75: all clumps > 1e+03 Msun")
    clump_note = (f" Detached clumps above 1e2 Msun: FD {det['FD']:.3g}, FV {det['FV']:.3g} (z = {det['z']:.1f}); above "
                  f"1e3 Msun FD and FV agree within the noise ({big['FD']:.3g} vs {big['FV']:.3g}, z = {big['z']:.1f}).")
    for key, name, text, note in (
        ("MixedGasExcessFD", "PDF 45-75: mixed mass [Msun]", "mixed-entropy gas mass at |z| > 1 kpc", ""),
        ("ClumpExcessFD", "clumps 45-75: all clumps > 1e+02 Msun", "number of clumps above 1e2 Msun per snapshot",
         clump_note),
    ):
        e = at2048(name)
        yield key, dict(
            value=100.0 * (e["FD"] / e["FV"] - 1.0), unit="percent", status="provisional", convergence=None, digits=2,
            description=f"FD-2048 / FV-2048 - 1 of the {text}, mean over (45, 75] Myr: FD {e['FD']:.3g}, FV "
                        f"{e['FV']:.3g} (FV seed 1 {seed1(name):.3g}); z = {e['z']:.1f} against the seed noise.{note}",
            runs=["A-2048", "FV-2048"], data=st_json, script=st_script,
        )  # fmt: skip
    sx = {res: at2048(f"map 60 Myr: xray width log S ({res})") for res in ("native", "20 pc", "80 pc", "160 pc")}
    yield "SXwidthRatioFDFV", dict(
        value=sx["80 pc"]["FD"] / sx["80 pc"]["FV"], unit="", status="single_snapshot", convergence=None,
        description="FD-2048 / FV-2048 of the width (std) of the log S_X distribution of the 60 Myr x-z map at "
                    "1 < |z| < 5 kpc (0.3-2 keV AtomDB), map smoothed to an 80 pc beam: "
                    f"{sx['80 pc']['FD']:.3f} vs {sx['80 pc']['FV']:.3f} dex (z = {sx['80 pc']['z']:.1f}). Other beams: "
                    + ", ".join(f"{r} {e['FD'] / e['FV']:.3f} (z = {e['z']:.1f})" for r, e in sx.items() if r != "80 pc")
                    + ". One snapshot per run.",
        runs=["A-2048", "FV-2048"], data=st_json, script=st_script,
    )  # fmt: skip
    ha = at2048("map 60 Myr: halpha log S power 100-30 pc")
    yield "HalphaPowerRatioFDFV", dict(
        value=ha["FD"] / ha["FV"], unit="", status="single_snapshot", convergence=None,
        description=f"FD-2048 / FV-2048 of the power of log S_Halpha of the 60 Myr x-z map at 1 < |z| < 5 kpc in the "
                    f"30-100 pc band: {ha['FD']:.3g} vs {ha['FV']:.3g} (FV seed 1 {seed1('map 60 Myr: halpha log S power 100-30 pc'):.3g}); "
                    f"z = {ha['z']:.1f}. One snapshot per run.",
        runs=["A-2048", "FV-2048"], data=st_json, script=st_script,
    )  # fmt: skip

    # spectra: agreement down to which band, and FV's velocity-power deficit at 30-60 pc
    bands = list(statistics.BANDS_PC)
    fields = ("log n", "log T", "|v|")
    agree_pc, worst_rel, failing = None, 0.0, []
    for band in bands:
        es = [at2048(f"spectrum {w}: {f} power {band}") for w in statistics.WINDOWS for f in fields]
        if all(e["verdict"] == "noise" for e in es):
            agree_pc = statistics.BANDS_PC[band][1]
            worst_rel = max(worst_rel, max(abs(e["FV"] / e["FD"] - 1.0) for e in es))
        else:
            failing = [f"{w} {f} (FV/FD = {e['FV'] / e['FD']:.2f}, z = {e['z']:.1f})"
                       for w in statistics.WINDOWS for f in fields
                       for e in [at2048(f"spectrum {w}: {f} power {band}")] if e["verdict"] != "noise"]  # fmt: skip
            break
    yield "SpectraAgreeScaleFDFV", dict(
        value=agree_pc, unit="pc", status="provisional", convergence=None, digits=2,
        description=f"Smallest scale down to which the 3D power spectra of log n, log T and |v| (1 < |z| < 3.5 kpc) of "
                    f"FD-2048 and FV-2048 agree within the seed noise, in the bands 1000-300 / 300-100 / 100-30 pc and "
                    f"both windows (30, 45] and (45, 75] Myr; down to there the band powers differ by at most "
                    f"{100 * worst_rel:.0f} %. The next band fails in: {'; '.join(failing) or 'none'}.",
        runs=["A-2048", "FV-2048"], data=st_json, script=st_script,
    )  # fmt: skip
    k = np.sqrt(statistics.K_EDGES[1:] * statistics.K_EDGES[:-1])
    spec = {tag: stats[tag]["structure"]["45-75"] for tag in ("A2048", "A2048fv", "A2048fvs1")}
    kmax = min(s["k_nyquist"] for s in spec.values())
    sel = (1e3 / k >= 30.0) & (1e3 / k <= 60.0) & (k <= kmax)
    deficit = {tag: 100.0 * (1.0 - float(np.mean(spec[tag]["pk_v"][sel] / spec["A2048"]["pk_v"][sel])))
               for tag in ("A2048fv", "A2048fvs1")}  # fmt: skip
    yield "VelPowerDeficitFV", dict(
        value=deficit["A2048fv"], unit="percent", status="provisional", convergence=None, digits=2,
        uncertainty={"kind": "range", "lo": min(deficit.values()), "hi": max(deficit.values())},
        description=f"1 - FV-2048 / FD-2048 of the 3D velocity power spectrum P_|v|(k) at 1 < |z| < 3.5 kpc, mean of the "
                    f"ratio over the {int(sel.sum())} k bins between 30 and 60 pc (6-12 cells), (45, 75] Myr: "
                    f"{deficit['A2048fv']:.0f} % (FV seed 1: {deficit['A2048fvs1']:.0f} %). In (30, 45] Myr the "
                    f"deficit is smaller.",
        runs=["A-2048", "FV-2048", "FV-2048 seed 1"], data=[STUDIES / f"structure3d_v2_{t}.npz" for t in spec],
        script=st_script,
    )  # fmt: skip

    # onset of rim mixing
    names = {"A2048": "FD-2048", "A2048fv": "FV-2048", "A2048fvs1": "FV-2048 seed 1", "A1024": "FD-1024",
             "A1024fv": "FV-1024"}
    onsets = {tag: rim_onset(tag) for tag in names}
    others = ", ".join(f"{names[t]} {v[0]:g}" for t, v in onsets.items())
    for key, tag in (("RimOnsetFDTwentyFortyEight", "A2048"), ("RimOnsetFVTwentyFortyEight", "A2048fv")):
        t_on, t_before = onsets[tag]
        yield key, dict(
            value=t_on, unit="Myr", status="provisional", convergence="not_converged", digits=3,
            uncertainty={"kind": "range", "lo": t_before, "hi": t_on},
            description=f"Onset of the mixing episode at the cavity rim, {names[tag]}: first checkpoint after "
                        f"the 15-25 Myr minimum at which the mixed-entropy outflow through |z| = 1 kpc exceeds "
                        f"{RIM_ONSET_MDOT:g} Msun/yr; the onset lies between {t_before:g} and {t_on:g} Myr (checkpoint "
                        f"cadence). All runs: {others} Myr. The onset moves earlier with resolution in both schemes, "
                        f"so it is not converged.",
            runs=[names[tag]], data=STUDIES / f"convergence3d_v3_{tag}.npz",
            script=CGOLS / "paper_exports" / "paper1_numbers.py",
        )  # fmt: skip


# --------------------------------------------------------------------------- numbers, v7 (asgard-papers #4)
SENS_LOGS = Path("/export/scratch/bschuber/diagnostics/asgard_issue4")


def wall_hours(memory_log):
    """Wall time of a sensitivity run from its per-minute GPU-memory log (first to last line, across midnight)."""
    lines = [ln.split()[0] for ln in memory_log.read_text().splitlines() if ln.strip()]
    seconds = [int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in (t.split(":") for t in (lines[0], lines[-1]))]
    return ((seconds[1] - seconds[0]) % 86400) / 3600.0


def tangent_error(tag32, tag64, name):
    """Relative L2 difference of a float32 tangent map from the float64 one, and the finite fraction of float32."""
    a = np.load(STUDIES / f"cgols_sensitivity{tag32}.npz")[name].astype(np.float64)
    b = np.load(STUDIES / f"cgols_sensitivity{tag64}.npz")[name]
    return float(np.linalg.norm(np.nan_to_num(a - b)) / np.linalg.norm(b)), float(np.isfinite(a).mean())


def numbers_v7_sensitivity():
    """Yield (key, kwargs for export_number) for the float32/float64 sensitivity keys of asgard-papers #4."""
    h64 = wall_hours(SENS_LOGS / "gpu0_mem_d.log")
    h32 = wall_hours(SENS_LOGS / "gpu4_mem_e.log")
    yield "JVPCostRatio", dict(
        value=h64 / h32, unit="", status="provisional", convergence=None,
        description=f"Wall time of the forward-mode sensitivity run (both directions, model A, 256^3, 0-30 Myr, one "
                    f"fully free H200 each) in float64 over float32: {h64:.1f} h / {h32:.1f} h. Peak device memory "
                    "141 GB (float64, needs the whole 143 GB card with the async allocator) vs 83 GB (float32).",
        runs=["A-256 sensitivity"], data=[SENS_LOGS / "gpu0_mem_d.log", SENS_LOGS / "gpu4_mem_e.log"],
        script=CGOLS / "cgols_sensitivity.py",
    )  # fmt: skip
    err = {(n, t, k): tangent_error(f"_d{n}_t{t}myr", f"_d{n}_t{t}myr_x64", k)
           for n, t in ((256, 30), (128, 30), (128, 45)) for k in ("dsig_dM", "dsig_dE")}  # fmt: skip
    finite = min(v[1] for v in err.values())
    ladder = "; ".join(f"{n}^3 at {t} Myr: {100 * err[(n, t, 'dsig_dM')][0]:.0f} % / {100 * err[(n, t, 'dsig_dE')][0]:.0f} %"
                       for n, t in ((128, 30), (128, 45), (256, 30)))  # fmt: skip
    yield "TangentErrorFloatThirtyTwo", dict(
        value=err[(256, 30, "dsig_dM")][0], unit="", status="provisional", convergence=None,
        description="Relative L2 difference ||t32 - t64|| / ||t64|| of the float32 tangent dSigma/dln Mdot of the "
                    "edge-on column density from the float64 one, model A 256^3 at 30 Myr (same IC and code). The "
                    f"float32 tangent stays finite (finite fraction {finite:.3f}) but is dominated by amplified rounding "
                    "noise; the primal Sigma agrees to 2e-4 (median). Mass / energy loading: " + ladder + ". The "
                    "float64 tangent passes the finite-difference check at 128^3 (45 Myr); float32 is not usable at "
                    "256^3 by 30 Myr.",
        runs=["A-256 sensitivity", "A-128 sensitivity"],
        data=[STUDIES / f"cgols_sensitivity_d256_t30myr{s}.npz" for s in ("", "_x64")],
        script=CGOLS / "cgols_sensitivity.py",
    )  # fmt: skip


# --------------------------------------------------------------------------- numbers, v8 (asgard-papers #4, PR #14)
DISC_PHASES = ("cold", "warm", "hot")  # cgols_convergence_3d PHASE_EDGES_K = (3e4, 5e5)


def numbers_v8():
    """Yield (key, kwargs for export_number) for the X-ray tangent check (#4) and the N = 1000 keys of PR #14."""
    fd_path = STUDIES / "cgols_sensitivity_d128_t30myr_x64_xray_fdcheck_eps0.002-0.001.npz"
    fd_coarse = STUDIES / "cgols_sensitivity_d128_t30myr_x64_xray_fdcheck_eps0.05-0.005.npz"
    if fd_path.is_file() and fd_coarse.is_file():
        fd = {**dict(np.load(fd_coarse)), **dict(np.load(fd_path))}

        def m(name, region, eps, key):
            return float(fd[f"fdcheck__{name}__{region}__eps{eps:g}__{key}"])

        ladder_text = ", ".join(f"{m('sx_atomdb', 'all', e, 'median_rel_err'):.0%} at {e:g}"
                                for e in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001))  # fmt: skip
        yield "SXTangentFDCheck", dict(
            value=100.0 * m("sx_atomdb", "all", 0.001, "median_rel_err"), unit="percent", status="provisional",
            convergence=None, digits=2,
            description="Median relative error of the forward-mode tangent dS_X/dln Mdot of the edge-on soft X-ray map "
                        "(AtomDB 0.3-1.0 keV photons after N_H = 4e20, Z = Z_sun A09) against the central finite "
                        "difference [S(1 + eps) - S(1 - eps)] / 2 eps with eps = 1e-3, over the 300 pixels with the "
                        "largest tangent; model A, 128^3, float64, 30 Myr. Correlation "
                        f"{m('sx_atomdb', 'all', 0.001, 'corr'):.3f}, slope FD/AD {m('sx_atomdb', 'all', 0.001, 'slope'):.3f}; "
                        f"halo |z| > 1 kpc: {m('sx_atomdb', 'halo', 0.001, 'median_rel_err'):.1%}. The error falls with "
                        f"eps ({ladder_text}): S_X responds linearly only to changes of the loading below ~1 %. Sigma "
                        f"for comparison: {m('sigma', 'all', 0.001, 'median_rel_err'):.1%}. The CGOLS recipe's hard "
                        "temperature window gives a tangent uncorrelated with the finite differences (correlation "
                        f"{m('sx_cgols', 'all', 0.001, 'corr'):.2f} at eps = 1e-3).",
            runs=["A-128 sensitivity"], data=[fd_coarse, fd_path], script=CGOLS / "cgols_sensitivity.py",
        )  # fmt: skip

    disc = {}
    for tag in ("C1024", "C1024n1000"):
        times, recs, _ = records("convergence3d_v3_", tag)
        disc[tag] = (np.asarray(recs[-1]["mass_disk"], dtype=float), float(times[-1]),
                     float(np.sum(recs[0]["mass_disk"])), float(times[0]))  # fmt: skip
    n1k, n16 = disc["C1024n1000"], disc["C1024"]
    phases = lambda a: ", ".join(f"{ph} {v:.3g}" for ph, v in zip(DISC_PHASES, a))
    yield "DiscMassCthousand", dict(
        value=float(n1k[0].sum()), unit="Msun", status="provisional", convergence="unknown",
        description=f"Gas mass at |z| < 0.5 kpc at {n1k[1]:.0f} Myr, C-1024 with N = 1000 (one resolution); "
                    f"{n1k[2]:.3g} Msun at {n1k[3]:.1f} Myr, before the feedback onset. By temperature (< 3e4 K / "
                    f"3e4-5e5 K / > 5e5 K): {phases(n1k[0])} Msun. N = 16 at the same time: {n16[0].sum():.3g} Msun "
                    f"({phases(n16[0])}), so N = 1000 leaves {100 * (1 - n1k[0].sum() / n16[0].sum()):.0f} % less disc "
                    "gas but more cold gas. No radial split in this cache.",
        runs=["C-1024 (N = 1000)", "C-1024"], data=[STUDIES / f"convergence3d_v3_{t}.npz" for t in disc],
    )  # fmt: skip

    c16, path16 = emission("C1024")
    c1k, path1k = emission("C1024n1000")
    means = {k: (float(c["per_snapshot__L_xray_cgols_xz"][window(c["per_snapshot__time_myr"], BURST)].mean()),
                 int(window(c["per_snapshot__time_myr"], BURST).sum())) for k, c in (("16", c16), ("1000", c1k))}  # fmt: skip
    yield "LxBurstMeanRatioCthousand", dict(
        value=means["1000"][0] / means["16"][0], unit="", status="provisional", convergence="unknown",
        description=f"L_X(N = 1000) / L_X(N = 16), means over (15, 40] Myr ({means['1000'][1]} and {means['16'][1]} "
                    f"snapshots), C-1024, CGOLS I recipe: {means['1000'][0]:.3g} / {means['16'][0]:.3g} erg/s. "
                    "One resolution. The single 15 Myr snapshot ratio is LxBurstRatioCthousand.",
        runs=["C-1024", "C-1024 (N = 1000)"], data=[path16, path1k],
    )  # fmt: skip


# --------------------------------------------------------------------------- figures


# --------------------------------------------------------------------------- numbers, v10 (asgard-papers #9, #3, #16)
SFR_MSUN_YR = 20.0  # star-formation rate for eta = Mdot / SFR (Tobias, asgard-papers #3, 21 Nov)
MIDPLANE_BIN_DEG = 80.0  # shells_v2 has 10 deg polar-angle bins: the last one, 80-90 deg, holds the mid-plane


def sensitivity_xray_stats(path):
    """Halo and box statistics of the X-ray sensitivity maps (cgols_sensitivity.py with CGOLS_SENS_XRAY=1)."""
    d = np.load(path)
    s0 = d["sx_atomdb0"]
    zz = np.broadcast_to(np.abs(d["z"])[None, :], s0.shape)
    bright = s0 > np.nanpercentile(s0[s0 > 0], 50)  # the blanking of the maps
    stats = {"inner": float(s0[zz < 0.5].sum() / s0.sum())}
    for k in ("M", "E"):
        tangent = d[f"dsx_atomdb_d{k}"]
        stats[f"box_{k}"] = float(tangent.sum() / s0.sum())
        stats[f"halo_{k}"] = float(tangent[zz > 1.0].sum() / s0[zz > 1.0].sum())
        for name, sel in (("halo", bright & (zz > 1.0)), ("low", bright & (zz > 1.0) & (zz < 3.0)),
                          ("high", bright & (zz > 3.0))):  # fmt: skip
            rel = tangent[sel] / s0[sel]
            stats[f"positive_{name}_{k}"] = float((rel > 0).mean())
            stats[f"median_{name}_{k}"] = float(np.median(rel))
    return stats


def shell_midplane_share(tag, bounds, radius_kpc=5.0):
    """Share of the net outflow through the sphere r that crosses it within 10 deg of the mid-plane."""
    i = shell_index(radius_kpc)
    times, recs, _ = records("shells_v2_", tag)
    sel = window(np.asarray(times), bounds)
    full = np.mean([np.sum(r["mdot_net"][:, i]) for r, s in zip(recs, sel) if s])
    n_theta = int(MIDPLANE_BIN_DEG / 10.0)
    cut = np.mean([np.sum(r["mdot_net"][:, i, :n_theta]) for r, s in zip(recs, sel) if s])
    return float(1.0 - cut / full)


def numbers_v10():
    """Yield (key, kwargs for export_number) for the sensitivity keys of #9, eta of #3 and the rings of #16."""
    paths = {res: STUDIES / f"cgols_sensitivity_d{res}_t30myr_x64_xray.npz" for res in (128, 256)}
    if all(path.is_file() for path in paths.values()):
        st = {res: sensitivity_xray_stats(path) for res, path in paths.items()}
        a, b = st[256], st[128]
        model = ("model A, 256^3, float64, 30 Myr, edge-on x-z map in AtomDB 0.3-1.0 keV photons after N_H = 4e20 "
                 "cm^-2 (Z = Z_sun A09), forward-mode tangents (cgols_sensitivity.py)")  # fmt: skip
        rings = STUDIES / "sensitivity_rings_A.npz"
        beam = ""
        if rings.is_file():
            r = np.load(rings)
            beam = (f" In a 312 pc beam, which removes the numerical rings (SensRingWavelengthCells): "
                    f"{100 * float(r['beam312__halo_positive_fraction']):.0f} %, median "
                    f"{float(r['beam312__halo_median']):+.2f}.")  # fmt: skip
        yield "SXsensEdotPositiveFraction", dict(
            value=100.0 * a["positive_halo_E"], unit="percent", status="provisional", convergence=None,
            description=f"Fraction of the halo pixels (|z| > 1 kpc, brighter than the median emitting pixel, as "
                        f"plotted) with dln S_X/dln Edot > 0; {model}. Median {a['median_halo_E']:+.2f}; at 1-3 kpc "
                        f"{100 * a['positive_low_E']:.0f} % (median {a['median_low_E']:+.2f}), at |z| > 3 kpc "
                        f"{100 * a['positive_high_E']:.0f} % (median {a['median_high_E']:+.2f}). 128^3: "
                        f"{100 * b['positive_halo_E']:.0f} %.{beam} Mass loading for comparison: dln S_X/dln Mdot > 0 "
                        f"in {100 * a['positive_low_M']:.0f} % at 1-3 kpc (median {a['median_low_M']:+.2f}) and "
                        f"{100 * a['positive_high_M']:.0f} % at |z| > 3 kpc (median {a['median_high_M']:+.2f}).",
            runs=["A-256 sensitivity"], data=paths[256], script=CGOLS / "cgols_sensitivity.py",
        )  # fmt: skip
        for k, word, name in (("M", "Mdot", "mass"), ("E", "Edot", "energy")):
            yield f"LxSens{word}", dict(
                value=a[f"box_{k}"], unit="", status="provisional", convergence=None, digits=2,
                description=f"d ln L_X / d ln {word}: response of the soft X-ray luminosity of the whole box to the "
                            f"{name} loading, sum of the tangent map over the sum of S_X; {model}. Halo |z| > 1 kpc: "
                            f"{a[f'halo_{k}']:+.2f}. 128^3: box {b[f'box_{k}']:+.2f}, halo {b[f'halo_{k}']:+.2f}: the sign is the "
                            f"same at both resolutions, the size is not converged. Dominated by |z| < 0.5 kpc "
                            f"(SXinnerFraction).",
                runs=["A-256 sensitivity"], data=paths[256], script=CGOLS / "cgols_sensitivity.py",
            )  # fmt: skip
        yield "SXinnerFraction", dict(
            value=100.0 * a["inner"], unit="percent", status="single_snapshot", convergence=None,
            description=f"Share of the soft X-ray luminosity of the box from |z| < 0.5 kpc; {model} (the unperturbed "
                        f"run of the sensitivity maps). 128^3: {100 * b['inner']:.0f} %.",
            runs=["A-256 sensitivity"], data=paths[256], script=CGOLS / "cgols_sensitivity.py",
        )  # fmt: skip
    else:
        print(f"skipped SXsens*/LxSens*/SXinnerFraction: missing {[str(p) for p in paths.values() if not p.is_file()]}")

    photon = {t: emission(t, SX_SUFFIX) for t in ("C512", "C1024")}
    if all(cache is not None for cache, _ in photon.values()):
        changes = {z0: sx_ratio(photon["C1024"][0], z0)[0] / sx_ratio(photon["C512"][0], z0)[0] - 1.0
                   for z0 in (2.0, 5.0)}  # fmt: skip
        largest = max(changes.values(), key=abs)
        yield "SXratioCchange", dict(
            value=100.0 * abs(largest), unit="percent", status="not_converged", convergence=None,
            description="Largest relative change of SXratioCtwoKpcTenTwentyFour and SXratioCfiveKpcTenTwentyFour from "
                        f"C-512 to C-1024: {100 * changes[2.0]:+.1f} % at 2 kpc, {100 * changes[5.0]:+.1f} % at 5 kpc "
                        "(both rising, towards M82). Compare SXratioAchange.",
            runs=["C-512", "C-1024"], data=[photon["C512"][1], photon["C1024"][1]],
        )  # fmt: skip

    five = shell_index(5.0)
    early = {LABEL[t]: record_mean("shells_v2_", t, EARLY, lambda r: np.sum(r["mdot_net"][:, five]))
             for t in ("A512", "A1024", "A2048")}  # fmt: skip
    eta = {label: value / SFR_MSUN_YR for label, value in early.items()}
    verdict, text = ladder(eta)
    mid = {t: shell_midplane_share(t, EARLY) for t in ("A512", "A1024", "A2048")}
    burst_a = record_mean("shells_v2_", "A2048", BURST, lambda r: np.sum(r["mdot_net"][:, five]))
    yield "EtaMAfiveKpc", dict(
        value=eta["A-2048"], unit="", status=verdict, convergence=verdict, uncertainty=value_range(eta),
        description=f"Mass loading eta = Mdot / SFR with SFR = {SFR_MSUN_YR:g} Msun/yr: ShellMdotAfiveKpcEarly (net "
                    f"outflow through the full sphere r = 5 kpc, 15-30 Myr) / SFR, A-2048. Over 15-40 Myr: "
                    f"{burst_a / SFR_MSUN_YR:.2f}. Mid-plane: the net flow within 10 deg of the mid-plane (80-90 deg "
                    f"from the pole) carries {', '.join(f'{100 * v:.1f} %' for v in mid.values())} of it at "
                    f"512/1024/2048^3, so leaving out 5 deg on each side, as CGOLS V do, lowers eta by less than "
                    f"{100 * mid['A2048']:.1f} % (about half of that if the flow is uniform in solid angle).{text}",
        runs=["A-2048"], data=STUDIES / "shells_v2_A2048.npz",
    )  # fmt: skip
    c = {t: (record_mean("shells_v2_", t, EARLY, lambda r: np.sum(r["mdot_net"][:, five])),
             record_mean("shells_v2_", t, BURST, lambda r: np.sum(r["mdot_net"][:, five])),
             shell_midplane_share(t, EARLY)) for t in ("C1024", "C1024n1000")}  # fmt: skip
    yield "EtaMCfiveKpcTenTwentyFour", dict(
        value=c["C1024"][0] / SFR_MSUN_YR, unit="", status="provisional", convergence="unknown",
        description=f"Mass loading eta = Mdot / SFR with SFR = {SFR_MSUN_YR:g} Msun/yr, net outflow through the full "
                    f"sphere r = 5 kpc over 15-30 Myr (the window of EtaMAfiveKpc; checkpoints every 2.5 Myr), "
                    f"C-1024 (N = 16), one resolution. Over 15-40 Myr: {c['C1024'][1] / SFR_MSUN_YR:.2f}. N = 1000: "
                    f"{c['C1024n1000'][0] / SFR_MSUN_YR:.2f} (15-30 Myr), {c['C1024n1000'][1] / SFR_MSUN_YR:.2f} "
                    f"(15-40 Myr). Within 10 deg of the mid-plane: {100 * c['C1024'][2]:.1f} % (N = 16), "
                    f"{100 * c['C1024n1000'][2]:.1f} % (N = 1000).",
        runs=["C-1024", "C-1024 (N = 1000)"], data=[STUDIES / "shells_v2_C1024.npz", STUDIES / "shells_v2_C1024n1000.npz"],
    )  # fmt: skip

    rings = STUDIES / "sensitivity_rings_A.npz"
    if rings.is_file():
        r = {k: float(v) for k, v in np.load(rings).items()}
        yield "SensRingWavelengthCells", dict(
            value=r["peak_cells__sx_dE__256"], unit="", status="provisional", convergence=None, digits=2,
            description="Radial wavelength of the rings in the sensitivity maps, in cells: peak of the radial power "
                        "spectrum of dln S_X/dln Edot along rays 20-75 deg from the minor axis, r = 1.5-5 kpc, model "
                        f"A, 30 Myr, float64. 256^3: {1e3 * r['peak_kpc__sx_dE__256']:.0f} pc = "
                        f"{r['peak_cells__sx_dE__256']:.1f} cells; 128^3: {1e3 * r['peak_kpc__sx_dE__128']:.0f} pc = "
                        f"{r['peak_cells__sx_dE__128']:.1f} cells. The wavelength follows the cell size, so the rings "
                        "are numerical, not waves of fixed physical size. The central finite-difference maps "
                        "(eps = 1e-3, 128^3) carry the same rings "
                        f"({r['fd_over_ad_band_power__sx_atomdb']:.2f} of the tangent's power at 5-10 cells), so they "
                        "are in the response of the discrete solution, not an error of the tangent. A 312 pc beam "
                        f"removes them (ring-band power share {r['beam0__band_share']:.2f} -> "
                        f"{r['beam312__band_share']:.2f}) without changing the halo statistics.",
            runs=["A-128 sensitivity", "A-256 sensitivity"], data=rings, script=CGOLS / "cgols_sensitivity_rings.py",
        )  # fmt: skip
    else:
        print(f"skipped SensRingWavelengthCells: no {rings} (run cgols_sensitivity_rings.py)")


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
         ("Mass, momentum and energy flux through spherical shells and bicones, model A 128^3-2048^3 (time means; faint "
          "where the shell leaves the box)"),
         ["A-128", "A-256", "A-512", "A-1024", "A-2048"]),
        ("outflow_per_R_A_C", FIGURES / "comparisons" / "outflow_per_R_A_C.png", "cgols_outflow_per_radius.py",
         ("Net mass flux through |z| = 1 and 5 kpc per cylindrical radius bin (R < 0.5, 0.5-1, 1-2, > 2 kpc), "
          "(15, 40] and (50, 75] Myr, models A (256-2048, blue) and C (256-1024, N = 16 and 1000, orange)"),
         ["A-256", "A-512", "A-1024", "A-2048", "C-256", "C-512", "C-1024", "C-1024 (N = 1000)"]),
        ("cluster_number_512", FIGURES / "comparisons" / "model_c_cluster_number_scaling_512.png",
         "cgols_cluster_number_scaling.py",
         ("Outflow against cluster number N at 512^3 (N = 1-128, smooth limit), model C, adiabatic: the same total "
          "injected mass, energy and volume, split into N regions"),
         ["C-512 N = 1-128"]),
        ("sensitivity_A256", FIGURES / "studies" / "sensitivity_maps_d256_t30myr_x64.png", "cgols_sensitivity.py",
         ("Forward-mode sensitivity of the edge-on column density of model A (256^3, float64) at 30 Myr to the mass "
          "and energy loading: top, Sigma, dSigma/dln Mdot, dSigma/dln Edot; bottom, the mid-plane density slice and "
          "the relative sensitivities dln Sigma/dln Mdot, dln Sigma/dln Edot (pixels below the 5th percentile of "
          "Sigma blanked)"),
         ["A-256 sensitivity"]),
        ("sensitivity_Lx_A256", FIGURES / "studies" / "sensitivity_xray_atomdb_d256_t30myr_x64_xray.png",
         "cgols_sensitivity.py",
         ("Forward-mode sensitivity of the edge-on soft X-ray surface brightness of model A (256^3, float64) at 30 Myr: "
          "S_X in absorbed 0.3-1.0 keV photons (AtomDB, N_H = 4e20 cm^-2, Z = Z_sun A09), dln S_X/dln Mdot and "
          "dln S_X/dln Edot; pixels fainter than the median of the emitting pixels blanked. Local (linear) response: "
          "valid for changes of the loading below ~1 % (SXTangentFDCheck)"),
         ["A-256 sensitivity"]),
        ("sensitivity_Lx_A256_beam", FIGURES / "studies" / "sensitivity_xray_atomdb_d256_t30myr_x64_xray_beam312.png",
         "cgols_sensitivity_rings.py",
         ("sensitivity_Lx_A256 in a Gaussian beam of FWHM 312 pc: S_X, dln S_X/dln Mdot and dln S_X/dln Edot, each "
          "map and tangent smoothed before the ratio (the log-derivative of the smoothed map); the beam removes the "
          "numerical rings (SensRingWavelengthCells) and keeps the cone walls; blanking as in sensitivity_Lx_A256"),
         ["A-256 sensitivity"]),
        ("sensitivity_rings_A", FIGURES / "studies" / "sensitivity_rings_A.png", "cgols_sensitivity_rings.py",
         ("The rings in the sensitivity maps, model A, 30 Myr, float64: top, dln S_X/dln Edot at 128^3 and 256^3, "
          "256^3 in a 312 pc beam, and dln S_X/dln Mdot at 128^3 from the tangent and from central differences "
          "(eps = 1e-3); bottom, radial power spectra against wavelength in pc and in cells, and the peak wavelength "
          "of dln Sigma/dln Edot at 8, 30 and 45 Myr (128^3)"),
         ["A-128 sensitivity", "A-256 sensitivity"]),
        ("xray_cgols_vs_A", FIGURES / "comparisons" / "xray_cgols_vs_A.png", "cgols_cholla_compare.py",
         ("Public CGOLS I 2048^3 state against our A-2048, both at 60 Myr, same emission pipeline: minor-axis "
          "0.3-1 keV absorbed photon profile against Strickland+04, L_X-weighted n-T distributions (CGOLS recipe), "
          "L_X-weighted n and T of the 2-15 MK gas against |z|, and box L_X against the lower edge of the "
          "recipe's temperature window with CGOLS I's published 1.9e40 erg/s"),
         ["CGOLS I 2048^3 (public)", "A-2048"]),
        ("fvfd_summary", FIGURES / "comparisons" / "fvfd_summary_significance.png", "cgols_fvfd_synthesis.py",
         ("FD (WENO5-Z) - FV (VL2 + PLM + HLLC) differences for every metric, model A 128^3-2048^3, in units of "
          "their combined scatter"),
         ["FD/FV 128-2048"]),
        ("cost_accuracy", FIGURES / "comparisons" / "cost_accuracy_families.png", "cgols_cost_accuracy.py",
         ("Worst and median error per family of quantities against cost, FD and FV, model A; dotted verticals: "
          "cost of FD-2048 (blue) and FV-2048 (red); dash-dot: tolerances 5, 10, 20 %; up-arrows: error > 800 %"),
         ["FD/FV 128-2048"]),
        ("fvfd_stats_pdfs", FIGURES / "comparisons" / "fvfd_stats_pdfs.png", "cgols_fvfd_statistics.py",
         ("Mass-weighted PDFs of log n and log T of the mixed, wind and halo gas (entropy phases) at |z| > 1 kpc, "
          "45-75 Myr, FD and FV at 1024^3 and 2048^3; thin lines: seed runs"),
         ["A-1024", "A-2048", "FV-1024", "FV-2048"]),
        ("fvfd_stats_spectra", FIGURES / "comparisons" / "fvfd_stats_spectra.png", "cgols_fvfd_statistics.py",
         ("3D power spectra k P(k) of log n, log T and |v| at 1 < |z| < 3.5 kpc, 45-75 Myr, FD and FV at 1024^3 and "
          "2048^3; lower row: FV / FD; dotted: 1000, 300, 100, 30 pc"),
         ["A-1024", "A-2048", "FV-1024", "FV-2048"]),
        ("fvfd_stats_clumps", FIGURES / "comparisons" / "fvfd_stats_clumps.png", "cgols_fvfd_statistics.py",
         ("Cumulative mass functions N(> M) of all entrained clumps and of the clumps detached from the disc, "
          "1 < |z| < 3.5 kpc, mean per snapshot over 45-75 Myr, FD and FV at 1024^3 and 2048^3; thin lines: seeds"),
         ["A-1024", "A-2048", "FV-1024", "FV-2048"]),
        ("fvfd_stats_maps", FIGURES / "comparisons" / "fvfd_stats_maps.png", "cgols_fvfd_statistics.py",
         ("Morphology of the edge-on soft X-ray (0.3-2 keV) and H-alpha maps at 60 Myr, 1 < |z| < 5 kpc: PDF of "
          "log S at the native resolution and in an 80 pc beam, and the power spectrum of log S; FD and FV at 1024^3 "
          "and 2048^3, thin lines: seeds"),
         ["A-1024", "A-2048", "FV-1024", "FV-2048"]),
        ("fv_riemann_A512", FIGURES / "comparisons" / "fv_riemann_A512.png", "cgols_fv_riemann.py",
         "Riemann-solver ablation at 512^3: FV with HLLC, FV with HLLE and FD (outflow, late X-rays, slices)",
         ["A-512", "FV-512", "FV-512 HLLE"]),
        ("fv_ppm_A256_A512", FIGURES / "comparisons" / "fv_ppm_A256_A512.png", "cgols_fvfd_synthesis.py",
         "FV with PPM against FV with PLM at 256^3 and 512^3, against FD at N and 2N",
         ["FV-256", "FV-512", "FV-256 PPM", "FV-512 PPM", "A-256", "A-512", "A-1024"]),
        ("fv_precision_A512", FIGURES / "comparisons" / "fv_precision_A512.png", "cgols_fv_precision.py",
         "Precision test at 512^3: FV in float32 (two runs) and float64 against FD",
         ["A-512", "FV-512", "FV-512 fp64"]),
        ("fv_density_floor", FIGURES / "comparisons" / "fv_density_floor.png", "cgols_density_floor.py",
         ("Density floor census: FV hard floor (rho = 1e-4 code units) against FD without a floor, and the FV "
          "mass budget"),
         ["A-512", "FV-512"]),
        ("rim_fdfv_hires", FIGURES / "comparisons" / "a_series_convergence_rim_fdfv_hires.png",
         "cgols_convergence_3d.py",
         "Cavity-rim roughness and mean rim radius against time, FD and FV at 1024^3 and 2048^3",
         ["A-1024", "A-2048", "FV-1024", "FV-2048"]),
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

    keys = {"numbers": set(), "figures": set()}
    for key, entry in numbers():
        keys["numbers"].add(key)
        print(f"{key:32s} {entry['value']:11.4g} {entry['unit']:8s} {entry['status']:15s} {entry['convergence']}")
        if not args.dry_run:
            export_number(PAPER, key, **entry)
    for key, path, entry in figures():
        keys["figures"].add(key)
        if not path.is_file():
            print(f"{key:32s} MISSING {path}")
            continue
        print(f"{key:32s} {path.relative_to(CGOLS)}")
        if not args.dry_run:
            export_figure(PAPER, key, path, **entry)
    if not args.dry_run:
        for section, keep in keys.items():
            for key in prune(PAPER, section, keep, script=__file__):
                print(f"removed {section[:-1]} {key} (no longer exported)")


if __name__ == "__main__":
    main()
