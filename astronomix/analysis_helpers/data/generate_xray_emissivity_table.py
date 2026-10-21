"""
Generate the soft X-ray band-emissivity table used by
``astronomix.analysis_helpers.emission``.

A one-off build script, NOT a runtime dependency: it needs ``pyatomdb`` and the
AtomDB emissivity files, which astronomix itself never imports. Run it in a
separate environment, e.g.

    python -m venv --system-site-packages /path/to/venv
    /path/to/venv/bin/pip install pyatomdb
    # one-off AtomDB data download (~1.4 GB), answer the prompts or call
    # pyatomdb.util.download_atomdb_emissivity_files(<dir>, "00000000", <version>)
    ATOMDB=/path/to/atomdb /path/to/venv/bin/python generate_xray_emissivity_table.py

It writes ``xray_emissivity_0p3_2keV_atomdb.npz`` next to this file.

What is tabulated
-----------------
The APEC collisional-ionisation-equilibrium (CIE) emissivity integrated over
the 0.3-2.0 keV band (energy, not photons), per unit ``n_e n_H``:

    epsilon(T) = sum_bins E_bin * spectrum(T)_bin     [erg cm^3 s^-1]

split into a hydrogen + helium part and one part per metal at its solar
abundance (Anders & Grevesse 1989, the AtomDB / XSPEC default). Emission is
linear in each element's abundance at fixed ``n_e n_H`` (APEC normalises per
``n_e n_H`` and the metals' electrons are not fed back), so any abundance
pattern is a weighted sum of the per-element rows; the runtime code builds
``epsilon_HHe + sum_el A_el epsilon_el``. The Grevesse & Asplund (2010)
abundances are stored next to the AG89 ones so a pattern quoted relative to
Asplund et al. (2009) can be rescaled.

The grid is log10 T = 5.0 ... 9.0 K in 0.02 dex (201 points); AtomDB's CIE
grid ends at 1e9 K. No absorption, no instrument response.
"""

# general
import os

# numerics
import numpy as np

# atomic data (build-time only)
import pyatomdb

LOG10_TEMPERATURE_GRID = np.round(np.arange(5.0, 9.0 + 1e-9, 0.02), 4)
BAND_KEV = (0.3, 2.0)
BIN_WIDTH_KEV = 1e-3
KEV_TO_ERG = 1.602176634e-9

# Floor for log10 of an identically-zero emissivity (e.g. an element with no
# line in the band at some temperature); 10^-60 is zero for every purpose.
LOG10_EMISSIVITY_FLOOR = -60.0

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "xray_emissivity_0p3_2keV_atomdb.npz",
)


def band_emissivity(session, energy_centers_erg):
    """Band-integrated energy emissivity [erg cm^3 s^-1] on the temperature grid."""
    emissivity = np.empty(LOG10_TEMPERATURE_GRID.size)
    for i, log10_temperature in enumerate(LOG10_TEMPERATURE_GRID):
        # Raw spectrum (no response): photons cm^3 s^-1 per energy bin.
        photons_per_bin = session.return_spectrum(
            10.0**log10_temperature,
            teunit="K",
        )
        emissivity[i] = np.sum(photons_per_bin * energy_centers_erg)
    return emissivity


def main():
    """Build and save the table."""
    energy_edges_kev = np.arange(
        BAND_KEV[0],
        BAND_KEV[1] + 0.5 * BIN_WIDTH_KEV,
        BIN_WIDTH_KEV,
    )
    energy_centers_erg = 0.5 * (energy_edges_kev[1:] + energy_edges_kev[:-1]) * KEV_TO_ERG

    session = pyatomdb.spectrum.CIESession(abundset="AG89")
    session.set_response(energy_edges_kev, raw=True)
    metals = [int(element) for element in session.elements if element > 2]

    # Hydrogen + helium only.
    session.set_abund(metals, 0.0)
    emissivity_hydrogen_helium = band_emissivity(session, energy_centers_erg)
    print(f"H+He done; eps(1e7 K) = {emissivity_hydrogen_helium[100]:.3e}", flush=True)

    # One metal at a time on top of H+He; the difference is that metal's part.
    emissivity_metals = np.empty((len(metals), LOG10_TEMPERATURE_GRID.size))
    for row, element in enumerate(metals):
        session.set_abund(metals, 0.0)
        session.set_abund(element, 1.0)
        with_element = band_emissivity(session, energy_centers_erg)
        emissivity_metals[row] = np.maximum(with_element - emissivity_hydrogen_helium, 0.0)
        print(f"Z = {element:2d} done; eps(1e7 K) = {emissivity_metals[row, 100]:.3e}", flush=True)

    # Cross-check: the full solar mixture must equal the sum of the parts.
    session.set_abund(metals, 1.0)
    emissivity_full = band_emissivity(session, energy_centers_erg)
    reconstructed = emissivity_hydrogen_helium + emissivity_metals.sum(axis=0)
    relative_error = np.max(np.abs(reconstructed - emissivity_full) / emissivity_full)
    print(f"max relative error of the per-element decomposition: {relative_error:.2e}")

    atomdb_root = os.environ.get("ATOMDB", "")
    version_file = os.path.join(atomdb_root, "VERSION")
    atomdb_data_version = "unknown"
    if os.path.exists(version_file):
        with open(version_file) as version_handle:
            atomdb_data_version = version_handle.read().strip()

    abundance_ag89 = pyatomdb.atomdb.get_abundance(abundset="AG89")
    abundance_ga10 = pyatomdb.atomdb.get_abundance(abundset="GA10")

    def to_log10(values):
        return np.log10(np.maximum(values, 10.0**LOG10_EMISSIVITY_FLOOR))

    np.savez(
        OUTPUT_PATH,
        log10_temperature=LOG10_TEMPERATURE_GRID,
        log10_emissivity_hydrogen_helium=to_log10(emissivity_hydrogen_helium),
        metal_atomic_numbers=np.array(metals, dtype=np.int32),
        log10_emissivity_metals=to_log10(emissivity_metals),
        solar_abundance_ag89=np.array([abundance_ag89[z] for z in metals]),
        solar_abundance_ga10=np.array([abundance_ga10[z] for z in metals]),
        band_kev=np.array(BAND_KEV),
        decomposition_relative_error=np.float64(relative_error),
        description=np.array(
            "APEC CIE band emissivity 0.3-2.0 keV [erg cm^3 s^-1 per n_e n_H], "
            f"AtomDB data {atomdb_data_version}, pyatomdb {pyatomdb.__version__}; "
            "metals at AG89 solar abundance"
        ),
    )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
