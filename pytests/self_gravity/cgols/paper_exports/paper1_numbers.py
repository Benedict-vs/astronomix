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
    photon = {tag: emission(tag, SX_SUFFIX) for tag in ("A512", "A1024", "A2048", "C1024")}
    missing = [tag for tag, (cache, _) in photon.items() if cache is None]
    if missing:
        print(f"skipped ScaleHeight*/SXratio*: no 0.3-1.0 keV photon caches for {missing} ({SX_SUFFIX})")
    else:
        sx_text = ("AtomDB 3.1.3 0.3-1.0 keV photons after TBabs absorption by N_H = 4e20 cm^-2 (Galactic, toward "
                   "M82), edge-on x-z projection, north")  # fmt: skip
        for key, tags, run in (("ScaleHeightA", ("A512", "A1024", "A2048"), "A-2048"),
                               ("ScaleHeightCTenTwentyFour", ("C1024",), "C-1024")):  # fmt: skip
            heights = {LABEL[t]: scale_height(photon[t][0]) for t in tags}
            verdict, text = ladder(heights)
            yield key, dict(
                value=heights[LABEL[tags[-1]]], unit="kpc", status="single_snapshot", convergence=verdict,
                description=f"Exponential scale height of the minor-axis soft X-ray profile, least squares in "
                            f"log S_X over 0.5-5 kpc, 60 Myr, {run}; {sx_text}. Strickland+04 north: 1.42 kpc.{text}",
                runs=[run], data=photon[tags[-1]][1],
            )  # fmt: skip
        for model, tags, suffix in (("A", ("A512", "A1024", "A2048"), ""), ("C", ("C1024",), "TenTwentyFour")):
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
