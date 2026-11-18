"""
Heating and cooling of KROME, for networks in the KROME format.

The terms follow KROME's ``krome_cooling.f90`` and ``krome_heating.f90`` (Grassi et
al. 2014) as built with ``-cooling=H2,ATOMIC,CHEM,Z_CIENOUV,COMPTON`` and
``-heating=CHEM,PHOTO,PHOTODUSTNET,CR``:

* cooling: the collisional excitation, ionisation, recombination and
  bremsstrahlung of H and He (Cen 1992); the fine-structure and metastable
  lines of the metals below 1e4 K, from the statistical equilibrium of two- and
  three-level systems; the tabulated metal cooling of Shen et al. (2013) above
  1e4 K; Compton cooling on the CMB; and the H2 lines (Glover & Abel 2008,
  Glover 2015), with the low-density limit summed over the colliders;
* heating: the photoelectric effect on dust net of the recombination cooling
  (Bakes & Tielens 1994), with the dust abundance relative to solar and the
  density of hydrogen nuclei as in KROME; the chemical term (formation and
  collisional dissociation of H2); the cosmic rays, through the reactions of
  the cosmic-ray block of the network; and the photo-reactions, through the
  energy of the photons above their thresholds (with photo bins).

The atomic data of the metal lines and the cooling table are not part of
astronomix: they are read from the ``data`` folder of a KROME installation
(``coolZ.dat`` and ``coolZ_CIE2012NOUV.dat``).
"""

# general
import re

# jax
import jax.numpy as jnp
import numpy as np

# KROME's constants (krome_constants.f90)
BOLTZMANN_ERG = 1.380648e-16
ELECTRON_VOLT_TO_ERG = 1.60217646e-12
CMB_TEMPERATURE_KELVIN = 2.73

METAL_LINES_FILE = "coolZ.dat"
METAL_TABLE_FILE = "coolZ_CIE2012NOUV.dat"
# name of a collider in coolZ.dat -> species of the network and the share of its density
COLLIDERS = {"h": ("H", "all"), "h+": ("H+", "all"), "e": ("E", "all"), "h2or": ("H2", "ortho"), "h2pa": ("H2", "para")}

# Reactions of KROME's chemical heating (kromeobj.py): reactants, products,
# energy per reaction [eV], and whether the fraction that thermalises applies.
CHEMICAL_HEATING = (
    (("H", "H", "H"), ("H", "H2"), 4.48, True),
    (("H", "H", "H2"), ("H2", "H2"), 4.48, True),
    (("H", "H-"), ("E", "H2"), 3.53, True),
    (("H", "H2+"), ("H+", "H2"), 1.83, True),
    (("H", "H2"), ("H", "H", "H"), -4.48, False),
    (("E", "H2"), ("E", "H", "H"), -4.48, False),
    (("H2", "H2"), ("H", "H", "H2"), -4.48, False),
)


# -------------------------------------------------------------
# ================= ↓ Reading KROME's data ↓ ==================
# -------------------------------------------------------------


def read_metal_lines(path, species_name):
    """Read the metal coolants of KROME's ``coolZ.dat``.

    Args:
        path: The file.
        species_name: Function from a KROME species name to the name used in
            the network.

    Returns:
        ``(variables, upper_limit, metals)``: the ``@var`` definitions, the
        temperature above which the collisional rates vanish, and for every
        metal its name, levels ``(energy [K], weight)``, radiative transitions
        ``(upper, lower, A [1/s])`` and collisional de-excitations ``(collider,
        upper, lower, expression, ((threshold, expression), ...))``; an
        expression after ``if(Tgas > threshold):`` replaces the first above the
        threshold.
    """
    variables, upper_limit, metals, current = [], None, [], None
    for number, raw in enumerate(open(path), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith("@var:"):
            name, expression = line[5:].split("=", 1)
            variables.append((name.strip().lower(), expression.strip()))
        elif lowered.startswith("@tgasupperlimit:"):
            upper_limit = float(line.split(":", 1)[1].lower().replace("d", "e"))
        elif lowered.startswith("metal:"):
            current = {"name": species_name(line[6:]), "levels": [], "radiative": [], "collisional": []}
        elif lowered.startswith("endmetal"):
            metals.append(current)
            current = None
        elif lowered.startswith("para/ortho"):
            continue  # the ortho-to-para ratio of H2 is a parameter of the run
        elif current is None:
            raise ValueError(f"{path}:{number}: line outside a metal block: {line[:50]}")
        elif lowered.startswith("level"):
            energy, weight = line.split(":", 1)[1].split(",")
            current["levels"].append((float(energy.lower().replace("d", "e")), float(weight)))
        elif "->" in line.split(",")[0]:
            transition, coefficient = line.split(",", 1)
            upper, lower = (int(level) for level in transition.split("->"))
            current["radiative"].append((upper, lower, float(coefficient.lower().replace("d", "e"))))
        elif lowered.startswith("if"):
            match = re.match(r"if\s*\(\s*tgas\s*>\s*([0-9.de+-]+)\s*\)\s*:(.*)", lowered)
            if match is None or not current["collisional"]:
                raise NotImplementedError(f"{path}:{number}: condition not supported: {line[:50]}")
            collider, upper, lower, expression, overrides = current["collisional"][-1]
            overrides = overrides + ((float(match.group(1).replace("d", "e")), match.group(2).strip()),)
            current["collisional"][-1] = (collider, upper, lower, expression, overrides)
        else:
            collider, upper, lower, expression = (field.strip() for field in line.split(",", 3))
            if collider.lower() not in COLLIDERS:
                raise NotImplementedError(f"{path}:{number}: collider '{collider}' not supported")
            current["collisional"].append((collider.lower(), int(upper), int(lower), expression, ()))
    return tuple(variables), upper_limit, metals


def read_cooling_table(path):
    """Read a two-dimensional cooling table of KROME (``log n_H, log T, log Lambda``)."""
    rows, shape = [], None
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if shape is None:
            shape = tuple(int(value) for value in line.split(","))
            continue
        rows.append([float(value) for value in line.split()])
    table = np.asarray(rows).reshape(shape[0], shape[1], 3)
    return table[:, 0, 0], table[0, :, 1], table[:, :, 2]


# -------------------------------------------------------------
# ================= ↑ Reading KROME's data ↑ ==================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ Cooling ↓ =========================
# -------------------------------------------------------------


def atomic_cooling(temperature, electrons, hydrogen, helium, ionized_hydrogen, ionized_helium, doubly_ionized_helium):
    """Cooling by H and He [erg cm^-3 s^-1] (Cen 1992, as in KROME's ``cooling_Atomic``)."""
    temp = jnp.maximum(temperature, 10.0)
    t5 = temp / 1e5
    damping = 1.0 / (1.0 + jnp.sqrt(t5))
    recombination = jnp.sqrt(temp) * (temp / 1e3) ** (-0.2) / (1.0 + (temp / 1e6) ** 0.7)
    cool = (
        # collisional ionisation of H, He, He+ and of He(2^3S)
        1.27e-21 * jnp.sqrt(temp) * damping * jnp.exp(-1.578091e5 / temp) * electrons * hydrogen
        + 9.38e-22 * jnp.sqrt(temp) * damping * jnp.exp(-2.853354e5 / temp) * electrons * helium
        + 4.95e-22 * jnp.sqrt(temp) * damping * jnp.exp(-6.31515e5 / temp) * electrons * ionized_helium
        + 5.01e-27 * temp ** (-0.1687) * damping * jnp.exp(-5.5338e4 / temp) * electrons**2 * ionized_helium
        # recombination of H+, He+, He++ and dielectronic recombination of He+
        + 8.7e-27 * recombination * electrons * ionized_hydrogen
        + 1.55e-26 * temp**0.3647 * electrons * ionized_helium
        + 3.48e-26 * recombination * electrons * doubly_ionized_helium
        + 1.24e-13 * temp ** (-1.5) * jnp.exp(-4.7e5 / temp) * (1.0 + 0.3 * jnp.exp(-9.4e4 / temp)) * electrons * ionized_helium
        # collisional excitation of H, He(2^3S) and He+
        + 7.5e-19 * damping * jnp.exp(-1.18348e5 / temp) * electrons * hydrogen
        + 9.1e-27 * temp ** (-0.1687) * damping * jnp.exp(-1.3179e4 / temp) * electrons**2 * ionized_helium
        + 5.54e-17 * temp ** (-0.397) * damping * jnp.exp(-4.73638e5 / temp) * electrons * ionized_helium
    )
    return jnp.maximum(cool, 0.0)


def compton_cooling(temperature, electrons):
    """Compton cooling on the CMB at redshift zero [erg cm^-3 s^-1]."""
    return 5.65e-36 * (temperature - CMB_TEMPERATURE_KELVIN) * electrons


def metal_line_cooling(temperature, density, metals, variables, upper_limit, evaluate, syntax_tree, ortho_para_ratio):
    """Line cooling of the metals below ``upper_limit`` [erg cm^-3 s^-1] (KROME's ``cooling_Z``).

    For every metal the level populations follow from the balance of collisional
    excitation and de-excitation and radiative decay; the cooling is the energy
    radiated. ``density`` maps a species name to its number density; ``metals``
    holds only two- and three-level systems.
    """
    # the collisional rates are zero from the upper limit on; below, the fits hold
    active = temperature < upper_limit
    safe_temperature = jnp.minimum(temperature, upper_limit)
    names = {"tgas": safe_temperature}
    functions = {"exp": jnp.exp, "log": jnp.log, "log10": jnp.log10, "sqrt": jnp.sqrt}
    for name, expression in variables:
        names[name] = evaluate(syntax_tree(expression), names, functions)
    molecular_hydrogen = jnp.maximum(density("H2"), 0.0)
    share = {"all": 1.0, "ortho": ortho_para_ratio / (ortho_para_ratio + 1.0), "para": 1.0 / (ortho_para_ratio + 1.0)}

    cooling = 0.0
    for metal in metals:
        levels, total = metal["levels"], density(metal["name"])
        size = len(levels)
        # down[u][l]: de-excitation u -> l, collisional plus radiative; up[l][u]: excitation
        down = [[0.0] * size for _ in range(size)]
        up = [[0.0] * size for _ in range(size)]
        for collider, upper, lower, expression, overrides in metal["collisional"]:
            species, part = COLLIDERS[collider]
            collider_density = molecular_hydrogen * share[part] if species == "H2" else jnp.maximum(density(species), 0.0)
            rate = evaluate(syntax_tree(expression), names, functions)
            for threshold, replacement in overrides:
                rate = jnp.where(safe_temperature > threshold, evaluate(syntax_tree(replacement), names, functions), rate)
            rate = jnp.where(active, rate, 0.0) * collider_density
            weights = levels[upper][1] / levels[lower][1]
            down[upper][lower] = down[upper][lower] + rate
            up[lower][upper] = up[lower][upper] + rate * weights * jnp.exp(-(levels[upper][0] - levels[lower][0]) / temperature)
        radiated = {}
        for upper, lower, coefficient in metal["radiative"]:
            down[upper][lower] = down[upper][lower] + coefficient
            radiated[(upper, lower)] = coefficient * (levels[upper][0] - levels[lower][0])
        if size == 2:
            a21, a22 = up[0][1], -down[1][0]
            scale = total / (a22 - a21)
            population = (a22 * scale, -a21 * scale)
        else:
            a21, a31 = up[0][1], up[0][2]
            a22, a33 = -(down[1][0] + up[1][2]), -(down[2][0] + down[2][1])
            a23, a32 = down[2][1], up[1][2]
            scale = total / (a21 * (a33 - a32) + a22 * (a31 - a33) + a23 * (a32 - a31))
            population = ((a23 * a32 - a22 * a33) * scale, -(a23 * a31 - a21 * a33) * scale, (a31 * a22 - a21 * a32) * scale)
        metal_cooling = sum(population[upper] * energy for (upper, lower), energy in radiated.items())
        cooling = cooling + jnp.where(total >= 1e-15, metal_cooling, 0.0)
    return cooling * BOLTZMANN_ERG


def tabulated_metal_cooling(temperature, hydrogen_nuclei, metallicity, log_density, log_temperature, log_lambda):
    """Metal cooling of a table in ``(log n_H, log T)`` [erg cm^-3 s^-1] (KROME's ``cooling_Z_CIENOUV``).

    Linear in both directions, continued linearly beyond the edges, as in KROME.
    """
    safe_nuclei = jnp.maximum(hydrogen_nuclei, 1e-20)
    x, y = jnp.log10(safe_nuclei), jnp.log10(temperature)
    x_scale = 1.0 / (log_density[1] - log_density[0])
    y_scale = 1.0 / (log_temperature[1] - log_temperature[0])
    i = jnp.clip(jnp.floor((x - log_density[0]) * x_scale).astype(int), 0, log_density.shape[0] - 2)
    j = jnp.clip(jnp.floor((y - log_temperature[0]) * y_scale).astype(int), 0, log_temperature.shape[0] - 2)
    p = (x - log_density[i]) * x_scale
    left = p * (log_lambda[i + 1, j] - log_lambda[i, j]) + log_lambda[i, j]
    right = p * (log_lambda[i + 1, j + 1] - log_lambda[i, j + 1]) + log_lambda[i, j + 1]
    value = (y - log_temperature[j]) * y_scale * (right - left) + left
    return jnp.where(hydrogen_nuclei >= 1e-20, 10.0**value * safe_nuclei * safe_nuclei * metallicity, 0.0)


def metal_blend(temperature):
    """Weight of the tabulated metal cooling; the lines carry one minus it (KROME's ``sigmoid`` at 1e4 K)."""
    return 10.0 / (10.0 + jnp.exp(-10.0 * (jnp.log10(temperature) - 4.0)))


# -------------------------------------------------------------
# ======================= ↑ Cooling ↑ =========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ Heating ↓ =========================
# -------------------------------------------------------------


def photoelectric_heating(temperature, electrons, hydrogen, hydrogen_nuclei, habing_field, dust_to_gas):
    """Photoelectric heating on dust net of recombination cooling [erg cm^-3 s^-1].

    Bakes & Tielens (1994), Eqs. 42 and 44, as in KROME's ``heat_netPhotoDust``:
    ``dust_to_gas`` is the dust abundance relative to solar and the heating
    scales with the density of hydrogen nuclei.
    """
    charged = (electrons > 0.0) & (habing_field > 0.0)
    psi = jnp.where(charged, habing_field * jnp.sqrt(temperature) / jnp.where(charged, electrons, 1.0), 0.0)
    safe_psi = jnp.where(charged, psi, 1.0)
    recombination_cooling = jnp.where(
        charged,
        4.65e-30 * temperature**0.94 * safe_psi ** (0.735 * temperature ** (-0.068)) * electrons * hydrogen,
        0.0,
    )
    efficiency = 4.9e-2 / (1.0 + 4e-3 * jnp.where(charged, safe_psi**0.73, 0.0)) + 3.7e-2 * (temperature * 1e-4) ** 0.7 / (
        1.0 + 2e-4 * psi
    )
    return (1.3e-24 * efficiency * habing_field * hydrogen_nuclei - recombination_cooling) * dust_to_gas


def cosmic_ray_heat_per_h2_ionisation(molecular_hydrogen, polynomial, linear):
    """Heat deposited per cosmic-ray ionisation of H2 [eV] as a function of n(H2) (KROME's ``heat_CR``).

    A polynomial in ``log10 n(H2)`` between 1e5 and 1e10 cm^-3, its value at
    1e10 above, a straight line below 1e5 down to where it reaches zero.
    """
    logarithm = jnp.log10(jnp.maximum(molecular_hydrogen, 1e-40))
    slope, intercept = linear
    lowest = 10.0 ** (-intercept / slope)
    highest = sum(coefficient * 10.0**power for power, coefficient in enumerate(polynomial))
    fit = sum(coefficient * logarithm**power for power, coefficient in enumerate(polynomial))
    return jnp.where(
        molecular_hydrogen > 1e10,
        highest,
        jnp.where(
            (molecular_hydrogen >= lowest) & (molecular_hydrogen < 1e5),
            slope * logarithm + intercept,
            jnp.where(molecular_hydrogen < lowest, 0.0, fit),
        ),
    )


def thermalised_fraction(temperature, hydrogen, molecular_hydrogen, hydrogen_nuclei):
    """Fraction of the H2 formation energy that heats the gas (KROME's ``h2heatfac``)."""
    safe_nuclei = jnp.maximum(hydrogen_nuclei, 1e-30)
    critical_density = 1.0e6 * temperature ** (-0.5) / (
        1.6 * jnp.exp(-((4.0e2 / temperature) ** 2)) * hydrogen / safe_nuclei
        + 1.4 * jnp.exp(-1.2e4 / (temperature + 1.2e3)) * molecular_hydrogen / safe_nuclei
        + 1e-300
    )
    return 1.0 / (1.0 + critical_density / safe_nuclei)


# -------------------------------------------------------------
# ======================= ↑ Heating ↑ =========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ H2 line cooling ↓ =====================
# -------------------------------------------------------------


def _window(log_temperature, low, high):
    """KROME's ``wCool``: one inside ``[low, high]`` in ``log10 T``, falling smoothly to zero outside."""
    x = (log_temperature - low) / (high - low)
    sigmoid = lambda argument, midpoint: 10.0 / (10.0 + jnp.exp(-50.0 * (argument - midpoint)))
    value = 10.0 ** (200.0 * (sigmoid(x, -0.2) * sigmoid(-x, -1.2) - 1.0))
    return jnp.where(value < 1e-199, 0.0, value)


def krome_h2_line_cooling(
    temperature, molecular_hydrogen, hydrogen, ionized_hydrogen, electrons, helium, single_precision_literals=False
):
    """H2 line cooling [erg cm^-3 s^-1], KROME's ``cooling_H2`` term by term.

    The low-density limit is the sum over the colliders H, H+, H2, e and He
    (Glover & Abel 2008; Glover 2015), the high-density limit that of
    Hollenbach & McKee (1979) below 2000 K and of Glover (2015) above; the two
    combine as ``n(H2) / (1/HDL + 1/LDL)``. Every fit is evaluated at the
    temperature clipped into its own range, so that no branch overflows where
    another one applies. KROME writes part of the constants in single
    precision; with ``single_precision_literals`` they are rounded in the same way.
    """
    s = (lambda value: float(np.float32(value))) if single_precision_literals else float
    log_temperature = jnp.log10(temperature)

    def fit(coefficients, low, high):
        x = jnp.log10(jnp.clip(temperature, low, high) * 1e-3)
        return sum(coefficient * x**power for power, coefficient in enumerate(coefficients))

    window_1_4 = _window(log_temperature, 1.0, 4.0)
    window_2_4 = _window(log_temperature, 2.0, 4.0)
    lowest = 1e-30  # lower limit of the clip of a fit without one, never reached by the gas temperature

    with_hydrogen = hydrogen * jnp.where(
        temperature <= 1e2,
        10.0 ** fit((-16.818342, 37.383713, 58.145166, 48.656103, 20.159831, 3.8479610), lowest, 1e2),
        jnp.where(
            temperature <= 1e3,
            10.0 ** fit((-24.311209, 3.5692468, -11.332860, -27.850082, -21.328264, -4.2519023), 1e2, 1e3),
            jnp.where(
                temperature <= 6e3,
                10.0 ** fit((-24.311209, 4.6450521, -3.7209846, 5.9369081, -5.5108049, 1.5538288), 1e3, 6e3),
                s(1.862314467912518e-22) * _window(log_temperature, 1.0, float(np.log10(6e3))),
            ),
        ),
    )
    with_protons = ionized_hydrogen * jnp.where(
        (temperature > 1e1) & (temperature <= 1e4),
        10.0 ** fit((-22.089523, 1.5714711, 0.015391166, -0.23619985, -0.51002221, 0.32168730), 1e1, 1e4),
        s(1.182509139382060e-21) * window_1_4,
    )
    with_molecules = molecular_hydrogen * window_2_4 * 10.0 ** fit(
        (-23.962112, 2.09433740, -0.77151436, 0.43693353, -0.14913216, -0.033638326), 2.0, 1e9
    )
    with_electrons = electrons * window_2_4 * jnp.where(
        temperature <= 5e2,
        10.0 ** jnp.minimum(
            fit((-21.928796, 16.815730, 96.743155, 343.19180, 734.71651, 983.67576, 801.81247, 364.14446, 70.609154), 2.0, 5e2),
            30.0,
        ),
        10.0 ** fit(
            (-22.921189, 1.6802758, 0.93310622, 4.0406627, -4.7274036, -8.8077017, s(8.9167183), s(6.4380698), s(-6.3701156)),
            5e2, 1e9,
        ),
    )
    with_helium = helium * jnp.where(
        (temperature > 1e1) & (temperature <= 1e4),
        10.0 ** fit((-23.689237, 2.1892372, -0.81520438, 0.29036281, -0.16596184, 0.19191375), 1e1, 1e4),
        s(1.002560385050777e-22) * window_1_4,
    )
    low_density = with_hydrogen + with_protons + with_molecules + with_electrons + with_helium

    t3 = jnp.clip(temperature, 2.0, 2e3) * 1e-3
    rotational = s(9.5e-22) * t3 ** s(3.76) / (1.0 + s(0.12) * t3 ** s(2.1)) * jnp.exp(-((s(0.13) / t3) ** 3)) + s(3.0e-24) * jnp.exp(
        -s(0.51) / t3
    )
    vibrational = s(6.7e-19) * jnp.exp(-s(5.86) / t3) + s(1.6e-18) * jnp.exp(-s(11.7) / t3)
    high_density = jnp.where(
        temperature < 2e3,
        rotational + vibrational,
        jnp.where(
            temperature <= 1e4,
            10.0 ** fit(
                (-20.584225, s(5.0194035), s(-1.5738805), s(-4.7155769), s(2.4714161), s(5.4710750), s(-3.9467356),
                 s(-2.2148338), s(1.8161874)),
                2e3, 1e4,
            ),
            s(5.531333679406485e-19) / (1.0 + jnp.exp(jnp.minimum((temperature - 3e4) * 2e-4, 3e2))),
        ),
    )
    # n(H2) / (1/HDL + 1/LDL), written with the smaller of the two limits over the
    # larger so that neither the value nor its derivative overflows when one of
    # them is vanishingly small
    active = (low_density > 0.0) & (high_density > 0.0)
    smaller = jnp.where(active, jnp.minimum(low_density, high_density), 0.0)
    larger = jnp.where(active, jnp.maximum(low_density, high_density), 1.0)
    return molecular_hydrogen * smaller / (1.0 + smaller / larger)


# -------------------------------------------------------------
# =================== ↑ H2 line cooling ↑ =====================
# -------------------------------------------------------------
