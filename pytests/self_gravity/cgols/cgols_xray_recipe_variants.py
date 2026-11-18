"""
How much do the ambiguities of the CGOLS I X-ray recipe change L_X? (asgard-papers issue #1)

CGOLS I (Schneider & Robertson 2018, Sec. IV, Eq. 17) integrates n^2 Lambda(T)
over "cells with temperatures between 0.3 and 2.0 keV", with Lambda the
analytic Eq. A4 cooling curve. Two readings are not fixed by the text:

  window   T from (3/2) k_B T in 0.3-2 keV (2.32e6-1.55e7 K, our default) or
           from k_B T in 0.3-2 keV (3.48e6-2.32e7 K);
  density  n^2 with n = rho / (mu m_p), mu = 0.6 (our default), or n_e n_H.

This script computes the box-integrated L_X of one checkpoint for every
combination (and without a window), reading the Orbax state tile by tile.
CGOLS I quotes 2.9e40 (A-512), 1.4e40 (A-1024) and 1.9e40 (A-2048) erg/s at
60 Myr.

    python cgols_xray_recipe_variants.py <checkpoint step dir> [--threads 8]

CPU only (numpy + tensorstore), no GPU.
"""

# general
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("step_dir")
parser.add_argument("--threads", type=int, default=8)
ARGS = parser.parse_args()
os.environ["JAX_PLATFORMS"] = "cpu"
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = str(ARGS.threads)

# numerics
import numpy as np
import tensorstore

# units
import astropy.constants as const
from astropy import units as u

# astronomix
from astronomix import CodeUnits, SimulationConfig, get_registered_variables
from astronomix.analysis_helpers.emission import emission_conversions
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE

CODE_UNITS = CodeUnits(1 * u.kiloparsec, 1e6 * u.M_sun, 100 * u.km / u.s)
BOX_KPC = 10.0
KEV_K = float((1 * u.keV / const.k_B).to(u.K).value)
WINDOWS = {
    "(3/2)kT in 0.3-2 keV": (0.3 * KEV_K / 1.5, 2.0 * KEV_K / 1.5),
    "kT in 0.3-2 keV": (0.3 * KEV_K, 2.0 * KEV_K),
    "no window": (0.0, np.inf),
}


def lambda_a4(temperature):
    """CGOLS I Eq. A4 analytic CIE cooling curve [erg cm^3 s^-1], numpy copy of the cooling module's."""
    from astronomix._modules._cooling._cooling import cooling_rate_cie_parabolic
    from astronomix._modules._cooling.cooling_options import CIEParabolicParams

    return np.asarray(cooling_rate_cie_parabolic(temperature, CIEParabolicParams(log10_lambda_cgs_to_code=0.0)))


def open_array(step_dir, name, context):
    """One array of an Orbax OCDBT checkpoint."""
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "ocdbt", "base": "file://" + os.path.abspath(step_dir) + "/", "path": name},
    }
    return tensorstore.open(spec, context=context, read=True).result()


def main():
    """Print L_X for every recipe variant."""
    context = tensorstore.Context({"data_copy_concurrency": {"limit": ARGS.threads}})
    state = open_array(ARGS.step_dir, "primitive_state", context)
    time_myr = float(open_array(ARGS.step_dir, "time", context).read().result()) * float(
        (1 * CODE_UNITS.code_time).to(u.Myr).value
    )
    num_vars, nx, ny, nz = state.shape
    variables = get_registered_variables(
        SimulationConfig(dimensionality=3, solver_mode=FINITE_DIFFERENCE, dual_energy=(num_vars == 6))
    )
    conversions = emission_conversions(CODE_UNITS)
    cell_volume_cm3 = (BOX_KPC / nx * u.kpc).to(u.cm).value ** 3
    chunk_x, chunk_y = state.chunk_layout.read_chunk.shape[1:3]

    totals = {(window, density): 0.0 for window in WINDOWS for density in ("n^2", "n_e n_H")}
    for x0 in range(0, nx, chunk_x):
        for y0 in range(0, ny, chunk_y):
            block = state[:, x0 : x0 + chunk_x, y0 : y0 + chunk_y, :].read().result()
            rho = block[variables.density_index].astype(np.float64)
            temperature = block[variables.pressure_index] / rho * conversions.kelvin_per_code_specific_energy
            del block
            lam = lambda_a4(temperature)
            n_tot = rho * conversions.total_number_density_per_code_density
            ne_nh = (rho * conversions.electron_density_per_code_density) * (
                rho * conversions.hydrogen_density_per_code_density
            )
            for window, (t_low, t_high) in WINDOWS.items():
                inside = (temperature >= t_low) & (temperature <= t_high)
                totals[(window, "n^2")] += float(np.sum(np.where(inside, n_tot**2 * lam, 0.0)))
                totals[(window, "n_e n_H")] += float(np.sum(np.where(inside, ne_nh * lam, 0.0)))

    print(f"{ARGS.step_dir}: {nx}x{ny}x{nz}, t = {time_myr:.2f} Myr; box-integrated L_X [erg/s], Eq. A4 curve")
    for (window, density), total in totals.items():
        print(f"  {window:22s} {density:8s} {total * cell_volume_cm3:.3e}")


if __name__ == "__main__":
    main()
