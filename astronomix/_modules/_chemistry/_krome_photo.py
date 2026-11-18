"""
Photo-reactions of a KROME network from a binned radiation spectrum.

KROME (Grassi et al. 2014) describes the radiation field as a specific intensity
``J`` [eV s^-1 cm^-2 sr^-1 Hz^-1] in a set of energy bins (its ``-photoBins``
option). From it and the cross sections of the photo-reactions of the network it
computes, once per spectrum,

* the rate of every photo-reaction [1/s] and its heating per reaction [erg/s],
  ``4 pi / h  sum_j sigma_ij J_j dE_j / E_j`` over the bins whose central
  energy lies above the threshold (times ``E_j - E_th`` for the heating);
* the Habing flux, the energy flux between 6 and 13.6 eV in units of
  1.6e-3 erg cm^-2 s^-1, which enters the recombination and the photoelectric
  heating on dust.

This module does the same with NumPy at set-up time. The cross section of a
reaction in a bin is, as in KROME,

* for ``auto``: the fit of Verner et al. (1996) from KROME's database
  (``data/database/photoioniziation.dat``), at the central energy of the bin;
* for ``@xsecFile=SWRI`` / ``@xsecFile=LEIDEN``: the average over the bin of the
  cross section table that KROME derives from the SwRI and Leiden databases
  (``data/database/swri_xsecs``, ``data/database/leiden_xsecs``), or the table
  of a KROME build folder when one is given;
* otherwise the Fortran expression of ``energy_eV`` written in the network file,
  at the central energy of the bin.

Nothing of KROME is shipped: the database is read from the ``data`` folder of a
KROME installation.
"""

# general
import os

# numerics
import numpy as np

# KROME's constants (krome_constants.f90)
PLANCK_EV = 4.135667516e-15  # eV s
BOLTZMANN_EV = 8.617332478e-5  # eV / K
SPEED_OF_LIGHT = 2.99792458e10  # cm / s
ELECTRON_VOLT_TO_ERG = 1.60217646e-12
PI = 3.14159265359
HABING_FLUX = 1.6e-3  # erg cm^-2 s^-1

# energy range of a photo-reaction without limits in the network file
DEFAULT_THRESHOLD = 2.73
DEFAULT_UPPER_ENERGY = 1.0e8
TABLE_POINTS = 5000  # points of a cross section table derived from a database
TABLE_MAXIMUM_ENERGY = 30.0  # eV


def _lines(path):
    with open(path, errors="replace") as file:
        return file.readlines()


# -------------------------------------------------------------
# ==================== ↓ Bins and spectra ↓ ===================
# -------------------------------------------------------------


def linear_bins(lower, upper, number):
    """Edges ``(left, right)`` [eV] of ``number`` bins of equal width (``krome_set_photobinE_lin``)."""
    width = abs(upper - lower) / number
    index = np.arange(number)
    return width * index + lower, width * (index + 1) + lower


def logarithmic_bins(lower, upper, number):
    """Edges ``(left, right)`` [eV] of ``number`` bins of equal logarithmic width (``krome_set_photobinE_log``)."""
    if lower >= upper:
        raise ValueError("logarithmic bins: the lower limit must be below the upper one")
    low, high = np.log10(lower), np.log10(upper)
    index = np.arange(number)
    return 10.0 ** (index * (high - low) / number + low), 10.0 ** ((index + 1) * (high - low) / number + low)


def draine_intensity(energy):
    """Interstellar radiation field of Draine (1978) [eV s^-1 cm^-2 sr^-1 Hz^-1], between 5 and 13.6 eV."""
    energy = np.asarray(energy, dtype=float)
    value = (1.658e6 * energy - 2.152e5 * energy**2 + 6.919e3 * energy**3) * energy * PLANCK_EV
    return np.where((energy < 13.6) & (energy > 5.0), value, 0.0)


def black_body_intensity(energy, temperature):
    """Planck function [eV s^-1 cm^-2 sr^-1 Hz^-1] at the photon energies ``energy`` [eV]."""
    energy = np.asarray(energy, dtype=float)
    exponent = energy / BOLTZMANN_EV / temperature
    valid = (exponent < 3e2) & (energy > 1e-10)
    safe = np.where(valid, exponent, 1.0)
    return np.where(valid, 2.0 * energy**3 / PLANCK_EV**2 / SPEED_OF_LIGHT**2 / (np.exp(safe) - 1.0), 0.0)


def j21_intensity(energy):
    """Power law of unit ``J21`` at 13.6 eV, falling as 1/E [eV s^-1 cm^-2 sr^-1 Hz^-1]."""
    return 6.2415e-10 * (13.6 / np.asarray(energy, dtype=float))


def habing_field(left, right, intensity, single_precision_literals=False):
    """Habing flux of a binned spectrum: the bins whose central energy lies in [6, 13.6] eV."""
    left, right, intensity = (np.asarray(array, dtype=float) for array in (left, right, intensity))
    middle = 0.5 * (left + right)
    upper = float(np.float32(13.6)) if single_precision_literals else 13.6
    inside = (middle >= 6.0) & (middle <= upper)
    return float(np.sum((intensity * (right - left))[inside]) * 4.0 * PI / HABING_FLUX / PLANCK_EV * ELECTRON_VOLT_TO_ERG)


def photo_spectrum(shape, lower, upper, number, spacing="linear", temperature=None, scale=1.0, habing=None):
    """A binned spectrum as KROME's ``krome_set_photoBin_*`` routines build it.

    Args:
        shape: ``"draine"``, ``"black_body"`` (needs ``temperature`` [K]) or ``"j21"``.
        lower, upper: Energy range of the bins [eV].
        number: Number of bins (KROME's ``-photoBins``).
        spacing: ``"linear"`` or ``"logarithmic"``.
        scale: Factor on the intensity (``krome_photoBin_scale``).
        habing: Rescale the intensity so that the Habing flux has this value.

    Returns:
        ``{"left": ..., "right": ..., "intensity": ...}``, the value of
        ``krome_photo_bins``; the intensity is evaluated at the bin centres.
    """
    left, right = (linear_bins if spacing == "linear" else logarithmic_bins)(lower, upper, number)
    middle = 0.5 * (left + right)
    if shape == "draine":
        intensity = draine_intensity(middle)
    elif shape == "black_body":
        if temperature is None:
            raise ValueError("a black-body spectrum needs its temperature")
        intensity = black_body_intensity(middle, temperature)
    elif shape == "j21":
        intensity = j21_intensity(middle)
    else:
        raise ValueError(f"spectrum '{shape}' unknown (draine, black_body, j21)")
    intensity = intensity * scale
    if habing is not None:
        current = habing_field(left, right, intensity)
        if current <= 0.0:
            raise ValueError("the spectrum has no flux between 6 and 13.6 eV: it cannot be scaled to a Habing flux")
        intensity = intensity * (habing / current)
    return {"left": left, "right": right, "intensity": intensity}


# -------------------------------------------------------------
# ==================== ↑ Bins and spectra ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ==================== ↓ Cross sections ↓ =====================
# -------------------------------------------------------------


def verner_cross_section(energy, e0, sigma0, ya, p, yw, y0, y1):
    """Photoionisation cross section [cm^2] of Verner et al. (1996) (KROME's ``sigma_v96``)."""
    x = energy / e0 - y0
    y = np.sqrt(x**2 + y1**2)
    return 1e-18 * sigma0 * ((x - 1.0) ** 2 + yw**2) * y ** (0.5 * p - 5.5) * (1.0 + np.sqrt(y / ya)) ** (-p)


def h2_lyman_werner_cross_section(energy, ortho_para_ratio=3.0, single_precision_literals=False):
    """Direct photodissociation cross section of H2 [cm^2] (Abel et al. 1997; KROME's ``H2_sigmaLW``).

    KROME writes part of the constants in single precision; with
    ``single_precision_literals`` they are rounded in the same way.
    """
    s = (lambda value: float(np.float32(value))) if single_precision_literals else float
    energy = np.asarray(energy, dtype=float)
    lyman_para = np.where(
        (energy > s(14.675)) & (energy < s(16.820)),
        1e-18 * 10.0 ** (s(15.1289) - s(1.05139) * energy),
        np.where(
            (energy > s(16.820)) & (energy < 17.6),
            1e-18 * 10.0 ** (-31.41 + 1.8042e-2 * energy**3 - 4.2339e-5 * energy**5),
            0.0,
        ),
    )
    werner_para = np.where((energy > 14.675) & (energy < 17.7), 1e-18 * 10.0 ** (13.5311 - s(0.9182618) * energy), 0.0)
    lyman_ortho = np.where(
        (energy > 14.159) & (energy < 15.302),
        1e-18 * 10.0 ** (12.0218406 - s(0.819429) * energy),
        np.where((energy > 15.302) & (energy < 17.2), 1e-18 * 10.0 ** (16.04644 - s(1.082438) * energy), 0.0),
    )
    werner_ortho = np.where((energy > 14.159) & (energy < 17.2), 1e-18 * 10.0 ** (s(12.87367) - s(0.85088597) * energy), 0.0)
    para = 1.0 / (ortho_para_ratio + 1.0)
    return para * (lyman_para + werner_para) + (1.0 - para) * (lyman_ortho + werner_ortho)


def _twelve_digits(value):
    # KROME writes its tables with twelve significant digits
    return float("%.12g" % value)


def _regular_table(energy, cross_section, lowest, highest, lines=()):
    """Cross section on ``TABLE_POINTS`` equally spaced energies, as KROME tabulates a database entry."""
    grid = np.array([_twelve_digits(i * (highest - lowest) / (TABLE_POINTS - 1) + lowest) for i in range(TABLE_POINTS)])
    exact = np.array([i * (highest - lowest) / (TABLE_POINTS - 1) + lowest for i in range(TABLE_POINTS)])
    if lines:
        values = np.zeros(TABLE_POINTS)
        for i, point in enumerate(exact):
            for line_left, line_right, line_value in lines:
                if line_left <= point <= line_right:
                    values[i] = line_value
                    break
    else:
        values = np.interp(exact, energy, cross_section)
    return grid, np.array([_twelve_digits(value) for value in values])


def leiden_table(path, ionisation):
    """Cross section table from a file of the Leiden database (wavelength [nm], absorption, dissociation, ionisation)."""
    rows = [[float(x) for x in line.split()] for line in _lines(path) if line.strip() and not line.strip().startswith("#")]
    data = np.array(rows)[::-1]
    energy = SPEED_OF_LIGHT * PLANCK_EV / (data[:, 0] * 1e-7)
    cross_section = data[:, 3 if ionisation else 2]
    return _regular_table(energy, cross_section, energy.min(), min(energy.max(), TABLE_MAXIMUM_ENERGY))


def swri_table(path, products):
    """Cross section table of one branch of a file of the SwRI database (wavelength [A], one column per branch).

    ``products`` are the names of the products with the electron as ``E``. As in
    KROME, the excited states of the products are merged into the ground state
    and a line given as ``1e-35, value, 1e-35`` is a rectangle of that value.
    """
    header, rows = None, []
    for line in _lines(path):
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "Lambda":
            header, rows = fields, []
            continue
        rows.append(fields)
    if header is None:
        raise ValueError(f"{path}: no header line starting with 'Lambda'")
    for state in ("1D", "1S", "3P"):
        header = [name.replace(state, "") for name in header]
    values = [[float(x) for x in row] for row in rows]
    # columns with the same products are summed into the first of them
    merged = set()
    for first in range(len(header) - 1):
        for second in range(first + 1, len(header)):
            if header[first] == header[second] and second not in merged:
                merged.add(second)
                for row in values:
                    row[first] += row[second]
    branches = [sorted("E" if x == "e" else x for x in name.replace("+", "+/E/").split("/") if x) for name in header]
    wanted = sorted(products)
    if wanted not in branches:
        raise ValueError(f"{path}: no branch with the products {wanted} (branches: {branches[2:]})")
    column = branches.index(wanted)

    energy, cross_section, lines = [], [], []
    lowest = highest = None
    in_line, previous, line = False, 0.0, {"left": 0.0, "value": 0.0}
    for row in values[::-1]:
        if row[0] <= 0.0:
            continue
        point = SPEED_OF_LIGHT * PLANCK_EV / (row[0] * 1e-8)
        value = row[column]
        if value == 1e-35:
            in_line = not in_line
            if previous == 1e-35:
                in_line = True
            if in_line:
                line = {"left": point, "value": line["value"]}
            else:
                lines.append((line["left"], point, line["value"]))
        if value != 1e-35 and in_line:
            line["value"] = value
        energy.append(point)
        cross_section.append(value)
        if value > 1e-40 and lowest is None:
            lowest = point
        if value > 1e-40:
            highest = point
        previous = value
    if lowest is None or highest is None or highest <= lowest:
        raise ValueError(f"{path}: no cross section for the products {wanted}")
    return _regular_table(np.array(energy), np.array(cross_section), lowest, min(highest, TABLE_MAXIMUM_ENERGY), tuple(lines))


def read_table(path):
    """A cross section table of a KROME build folder: energy [eV] and cross section [cm^2], equally spaced."""
    rows = [line.split() for line in _lines(path) if line.strip() and not line.strip().startswith("#")]
    data = np.array([[float(x) for x in row[:2]] for row in rows])
    steps = np.diff(data[:, 0])
    if np.any(np.abs(steps / steps[0] - 1.0) > 1e-6):
        raise ValueError(f"{path}: the energies of a cross section table must be equally spaced")
    return data[:, 0], data[:, 1]


def bin_average(left, right, table_energy, table_value):
    """Average of a tabulated cross section over the bin ``[left, right]`` (KROME's ``xsec_interp``).

    Every interval of the table that starts inside the bin contributes its value
    times the part of it that lies in the bin.
    """
    step = 1.0 / (1.0 / (table_energy[1] - table_energy[0]))
    interval_left = np.arange(len(table_value)) * step + table_energy[0]
    interval_right = interval_left + step
    counted = ~(((right < interval_left) & (left < interval_left)) | ((right > interval_left) & (left > interval_left)))
    overlap = np.minimum(interval_right, right) - np.maximum(interval_left, left)
    return float(np.sum((table_value * overlap * (1.0 / (right - left)))[counted]))


def _database_entries(folder):
    """The automatic reactions of KROME's database folder, in KROME's reading order."""
    entries = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or "~" in name or not name.endswith(".dat"):
            continue
        entry = {}
        for raw in _lines(path):
            line = raw.strip()
            if not line:
                continue
            if line == "#BREAK DATABASE":
                break
            if line.startswith("#"):
                continue
            lowered = line.lower()
            if "@photoxsec:" in lowered or "@cr:" in lowered or "@photoav:" in lowered:
                break
            if "@var" in line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.replace("@", "").strip().lower()
            if key == "type":
                entry = {"limits": ""}
            entry[key] = value.strip()
            if key == "rate" and "reacts" in entry and "prods" in entry:
                entries.append(entry)
                entry = {}
    return entries


def _number(text):
    return float(text.strip().lower().replace("d", "e"))


def _strip_operator(text):
    for symbol in (".GE.", ".GT.", ".LE.", ".LT.", ">=", "<=", ">", "<"):
        text = text.upper().replace(symbol, "")
    return text


def cross_sections(
    photo_reactions, species, left, right, data_folder=None, xsec_folder=None, ortho_para_ratio=3.0,
    single_precision_literals=False, evaluate=None, syntax_tree=None,
):
    """Cross section of every photo-reaction in every bin, and the energy thresholds.

    Args:
        photo_reactions: One ``(reactant names, product names, lower limit,
            upper limit, rate)`` per photo-reaction, as written in the network
            file (the limits are energies in eV, or None).
        species: Function from an astronomix species name to KROME's name.
        left, right: Edges of the bins [eV].
        data_folder: The ``data`` folder of a KROME installation (database of
            the automatic reactions and of the cross section files).
        xsec_folder: A KROME build folder; a table found there
            (``swri_*.dat``, ``leiden_*.dat``) is used as it is.
        evaluate, syntax_tree: The Fortran-expression evaluator of the reader.

    Returns:
        ``(table, thresholds)``: cross sections [cm^2] of shape
        ``(reactions, bins)`` and the threshold energy [eV] of every reaction.
    """
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    middle = 0.5 * (left + right)
    table = np.zeros((len(photo_reactions), len(middle)))
    thresholds = np.zeros(len(photo_reactions))
    database = None

    functions = {
        "exp": np.exp, "dexp": np.exp, "log": np.log, "dlog": np.log, "log10": np.log10, "dlog10": np.log10,
        "sqrt": np.sqrt, "dsqrt": np.sqrt, "abs": np.abs, "max": np.maximum, "min": np.minimum,
        "sigma_v96": verner_cross_section,
        "h2_sigmalw": lambda energy: h2_lyman_werner_cross_section(energy, ortho_para_ratio, single_precision_literals),
    }

    for slot, (reactants, products, lower, upper, rate) in enumerate(photo_reactions):
        label = " + ".join(reactants) + " -> " + " + ".join(products)
        threshold = DEFAULT_THRESHOLD if lower is None else _number(_strip_operator(lower))
        highest = DEFAULT_UPPER_ENERGY if upper is None else _number(_strip_operator(upper))
        text = rate.strip()
        tabulated = None

        if text.lower() == "auto":
            if data_folder is None:
                raise ValueError(f"photo-reaction {label}: its cross section is in KROME's database; give krome_data_folder")
            if database is None:
                database = _database_entries(os.path.join(data_folder, "database"))
            wanted = (sorted(species(name).upper() for name in reactants), sorted(species(name).upper() for name in products))
            found = [
                entry for entry in database
                if (sorted(x.strip().upper() for x in entry["reacts"].split(",")),
                    sorted(x.strip().upper() for x in entry["prods"].split(","))) == wanted
            ]
            if not found:
                raise ValueError(f"photo-reaction {label}: not in KROME's database of automatic reactions")
            if len(found) > 1:
                raise NotImplementedError(f"photo-reaction {label}: several entries in KROME's database")
            text = found[0]["rate"]
            if found[0]["limits"].strip():
                threshold, highest = (_number(x) for x in found[0]["limits"].split(","))
            else:
                threshold, highest = DEFAULT_THRESHOLD, DEFAULT_UPPER_ENERGY
        elif "@xsecfile=" in text.lower():
            kind = text.lower().split("@xsecfile=")[1].strip()
            if kind not in ("swri", "leiden"):
                raise NotImplementedError(f"photo-reaction {label}: cross section file '{kind}' not supported")
            if lower is None:
                threshold = 0.0
            reactant = species(reactants[0])
            names = [species(name) for name in products]
            candidates = [f"{kind}_{reactant}__{'_'.join(order)}.dat" for order in (names, sorted(names), sorted(names)[::-1])]
            for candidate in candidates if xsec_folder is not None else ():
                if os.path.isfile(os.path.join(xsec_folder, candidate)):
                    tabulated = read_table(os.path.join(xsec_folder, candidate))
                    break
            if tabulated is None:
                if data_folder is None:
                    raise ValueError(
                        f"photo-reaction {label}: its cross section comes from the {kind.upper()} files of KROME; "
                        "give krome_data_folder (or krome_xsec_folder, a KROME build folder with the table)"
                    )
                if kind == "leiden":
                    folder = os.path.join(data_folder, "database", "leiden_xsecs")
                    paths = [os.path.join(folder, f"{reactant}__{'_'.join(order)}.dat") for order in (sorted(names), sorted(names)[::-1])]
                    existing = [path for path in paths if os.path.isfile(path)]
                    if not existing:
                        raise FileNotFoundError(f"photo-reaction {label}: no file {paths[0]}")
                    tabulated = leiden_table(existing[0], "E" in names)
                else:
                    path = os.path.join(data_folder, "database", "swri_xsecs", f"{reactant}.dat")
                    if not os.path.isfile(path):
                        raise FileNotFoundError(f"photo-reaction {label}: no file {path}")
                    tabulated = swri_table(path, names)

        thresholds[slot] = threshold
        inside = (middle > threshold) & (middle < highest)
        if tabulated is not None:
            for j in np.nonzero(inside)[0]:
                table[slot, j] = bin_average(left[j], right[j], *tabulated)
        else:
            tree = syntax_tree(text, single_precision_literals)
            for j in np.nonzero(inside)[0]:
                table[slot, j] = float(evaluate(tree, {"energy_ev": middle[j]}, functions))
    return table, thresholds


def rates_and_heats(table, thresholds, left, right, intensity):
    """Rate [1/s] and heating per reaction [erg/s] of every photo-reaction (KROME's ``calc_photoBins``)."""
    left, right, intensity = (np.asarray(array, dtype=float) for array in (left, right, intensity))
    middle, width = 0.5 * (left + right), right - left
    rates, heats = np.zeros(len(thresholds)), np.zeros(len(thresholds))
    for j in range(len(middle)):
        for i in range(len(thresholds)):
            if middle[j] > thresholds[i]:
                term = table[i, j] * intensity[j] / middle[j] * width[j]
                rates[i] += term
                heats[i] += term * (middle[j] - thresholds[i])
    return 4.0 * PI * rates / PLANCK_EV, 4.0 * PI * heats / PLANCK_EV * ELECTRON_VOLT_TO_ERG


# -------------------------------------------------------------
# ==================== ↑ Cross sections ↑ =====================
# -------------------------------------------------------------
