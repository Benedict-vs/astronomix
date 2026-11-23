"""
Reaction networks written in the KROME format.

KROME (Grassi et al. 2014) describes a network as a text file of reactions whose
rate coefficients are Fortran expressions of the gas temperature. This module
reads such a file and builds its right-hand side as a JAX function with the call
signature of a carbox network, so that the chemistry module integrates it with
the same implicit solvers (the vmapped Diffrax Kvaerno5 and the per-cell Warp
kernel), doing with it what KROME does.

What is read (the subset needed by ``react_galaxy_ism`` and its variants):

* reaction lines under the default layout or under a ``@format:`` line, with
  temperature limits (``NONE``, ``.LE.5.5e3``, ``>5.5e3``, a bare number, ...);
* ``@var:`` definitions and ``@common:`` user variables;
* rate coefficients as Fortran expressions of ``Tgas`` and KROME's temperature
  shortcuts (``Te``, ``lnTe``, ``invT``, ``T32``, ...);
* the rate functions KROME evaluates internally: recombination on dust of
  H+, He+, C+, O+ and Si+ (Weingartner & Draine 2001) and the Lyman-Werner
  photodissociation of H2 from a flux (``H2_solomonLW``);
* photo-reactions (``@photo_start`` ... ``@photo_stop``). KROME obtains their
  rates by integrating cross sections over a radiation spectrum given in energy
  bins; so does this module when ``photo_bins`` is given (``_krome_photo.py``),
  and the Habing flux of the dust terms is then that of the spectrum. Without
  bins every photo-reaction takes its rate [1/s] from ``photo_rates`` (zero by
  default, i.e. no ionising radiation);
* the cosmic-ray block (``@CR_start`` ... ``@CR_stop``), whose reactions heat
  the gas as in KROME;
* ``@reactionModifier_start`` ... ``@reactionModifier_stop``: statements
  ``k(i) = ...`` applied to the coefficients after they are all computed, and
  the shielding functions ``shield_dust(n,Tgas,gamma)`` and
  ``krome_fshield(n,Tgas)`` in a rate expression or a ``@var``.

A rate coefficient is zero outside its temperature limits, as in KROME. A line
whose number of fields does not match the layout in force is skipped with a
warning, as KROME does, and the reactions are numbered in the order in which
they are read (KROME's reaction index); any other directive of the format raises
an error instead of being skipped.

What KROME takes from its build options is not in the network file; here these
are arguments of :func:`build_krome_network`:

* ``-useDustH2const``: the formation of H2 on dust grains with the rate of Jura
  (1975), ``3.5e-17 n(H) n_H D/D_sun`` H2 molecules per cm^3 per second;
* ``-shieldHabingDust``: the Habing flux attenuated by dust,
  ``exp(-1.33e-21 D/D_sun N_H)``;
* ``-shielding=DB96 | WG11 | R14``: the H2 self-shielding that
  ``krome_fshield`` returns;
* ``-columnDensityMethod=DEFAULT | JEANS``: how a number density becomes the
  column density of the two shielding terms;
* ``-photoBins``: the binned spectrum;
* ``-cooling=FF``: the bremsstrahlung of the atomic ions (off by default).

The heating and cooling of a KROME network are KROME's too (see
``_krome_thermochemistry.py``): the network carries its own temperature
derivative, which the chemistry module uses in place of its default one.
"""

# general
import ast
import re
import warnings
from functools import lru_cache
from typing import Tuple

# jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np

# KROME's constants (krome_constants.f90), kept to the digit so that the rate
# coefficients agree with KROME's.
KELVIN_TO_ELECTRON_VOLT = 8.617343e-5
ELECTRON_VOLT_TO_ERG = 1.60217646e-12
MINIMUM_TEMPERATURE_KELVIN = 2.73
MAXIMUM_TEMPERATURE_KELVIN = 1.0e9

# layout of a reaction line without a @format line: three reactants, four products
DEFAULT_FORMAT = ("idx", "r", "r", "r", "p", "p", "p", "p", "tmin", "tmax", "rate")
# two-letter element symbols restored from KROME's upper-case species names
TWO_LETTER_ELEMENTS = ("He", "Si", "Mg", "Fe", "Na", "Ne", "Cl", "Ar", "Ca", "Al", "Li")
NO_LIMIT = ("", "N", "NONE", "N/A", "NO")
IGNORED_DIRECTIVES = ("@notab_start", "@notab_stop", "@notabnext")

# KROME's constants (krome_constants.f90, kromeobj.py) for the column densities,
# the self-shielding and the cosmic-ray heating
BOLTZMANN_ERG = 1.380648e-16
GRAVITY = 6.674e-8
PI = 3.14159265359
ELECTRON_MASS = 9.10938188e-28
PROTON_MASS = 1.67262158e-24
NEUTRON_MASS = 1.6725e-24
# protons and neutrons of the elements in KROME's species masses
NUCLEONS = {
    "H": (1, 0), "D": (1, 1), "He": (2, 2), "Li": (3, 4), "Be": (4, 5), "B": (5, 6), "C": (6, 6), "N": (7, 7), "O": (8, 8),
    "F": (9, 10), "Ne": (10, 10), "Na": (11, 12), "Mg": (12, 12), "Al": (13, 14), "Si": (14, 14), "P": (15, 16),
    "S": (16, 16), "Cl": (17, 18), "Ar": (18, 22), "Ti": (22, 26), "Fe": (26, 29),
}
SHIELDINGS = ("DB96", "WG11", "R14")
COLUMN_DENSITY_METHODS = ("DEFAULT", "JEANS")
# Cosmic-ray heating (KROME's heat_CR): energy per ionisation of H and He [eV],
# of any other cosmic-ray reaction, and the fit of the energy per ionisation of
# H2 as a function of its number density (Glassgold et al. 2012 and
# arXiv:1502.03380), linear in log10 n(H2) below 1e5 cm^-3 and constant above 1e10.
COSMIC_RAY_ELECTRON_VOLT_TO_ERG = 1.60217662e-12
COSMIC_RAY_HEAT_ATOMIC = 4.3
COSMIC_RAY_HEAT_OTHER = 20.0
COSMIC_RAY_HEAT_H2_POLYNOMIAL = (9.1462198642562, -0.443120145068494, 0.601271617560071, -0.0710284101055109, 0.00242079494592405)
COSMIC_RAY_HEAT_H2_LINEAR = (1.4510849135088, 7.23277032061136)

# Weingartner & Draine (2001), Table 2: recombination of an ion on dust grains,
# alpha = 1e-14 C0 / (1 + C1 psi^C2 (1 + C3 T^C4 psi^(-C5 - C6 ln T))) cm^3/s,
# with psi = G0 sqrt(T) / n_e.
DUST_RECOMBINATION = {
    "h": (12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5),
    "he": (5.572, 3.185e-7, 1.512, 5.115e3, 3.903e-7, 0.4956, 5.494e-7),
    "c": (45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4),
    "si": (2.166, 5.678e-8, 1.874, 4.375e4, 1.635e-6, 0.8964, 7.538e-5),
}


# -------------------------------------------------------------
# ======================= ↓ Reading the file ↓ ================
# -------------------------------------------------------------


def _species_name(token):
    """KROME species name -> the name used in astronomix (``HE+`` -> ``He+``)."""
    name = token.strip().upper()
    for element in TWO_LETTER_ELEMENTS:
        name = name.replace(element.upper(), element)
    return name


def _elements_and_charge(name):
    """Element counts and charge of a species from its name."""
    if name == "E":
        return {}, -1
    charge = name.count("+") - name.count("-")
    counts = {}
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", name.replace("+", "").replace("-", "")):
        counts[element] = counts.get(element, 0) + (int(number) if number else 1)
    return counts, charge


def _free_free_charge_squared(name):
    """Weight of a species in the bremsstrahlung, as KROME's generator sets it.

    KROME sums the charge squared times the density over the species that are
    single atoms with a positive charge (``bms_ions``); molecular ions, anions
    and the electrons do not radiate in it.

    Args:
        name: The species name.

    Returns:
        The charge squared of an atomic ion, zero for every other species.
    """
    counts, charge = _elements_and_charge(name)
    is_atom = sum(counts.values()) == 1
    return float(charge * charge) if is_atom and charge > 0 else 0.0


def _species_mass(name):
    """Mass of a species [g] as KROME computes it: protons, neutrons and electrons of its atoms."""
    counts, charge = _elements_and_charge(name)
    mass = -charge * ELECTRON_MASS
    for element, count in counts.items():
        if element not in NUCLEONS:
            raise NotImplementedError(f"species {name}: the mass of the element {element} is not tabulated")
        protons, neutrons = NUCLEONS[element]
        mass += count * (protons * (ELECTRON_MASS + PROTON_MASS) + neutrons * NEUTRON_MASS)
    return mass


def _fortran_number(text):
    return float(text.strip().lower().replace("d", "e"))


def _temperature_limit(text, default_operator):
    """``.LE.5.5e3`` / ``>5.5e3`` / ``1d6`` / ``NONE`` -> (operator, value) or None."""
    text = text.strip()
    if text.upper() in NO_LIMIT:
        return None
    for symbol, operator in ((".GE.", "GE"), (".GT.", "GT"), (".LE.", "LE"), (".LT.", "LT"), (">=", "GE"), ("<=", "LE"), (">", "GT"), ("<", "LT")):
        if text.upper().startswith(symbol):
            return operator, _fortran_number(text[len(symbol):])
    return default_operator, _fortran_number(text)


def parse_krome_network(path):
    """Read a KROME network file.

    Returns:
        A dictionary with ``species`` (names in order of first appearance, the
        electron as ``E``), ``reactions`` (one tuple per reaction: KROME index,
        reactant indices, product indices, lower limit, upper limit, rate
        expression, photo slot or -1), ``variables`` (the ``@var`` definitions
        in file order), ``commons`` (the ``@common`` user variables),
        ``photo`` (for every photo-reaction its position, the two limits as
        written, i.e. energies in eV or None, and the cross section as
        written), ``cosmic_ray`` (the positions of the reactions of the
        cosmic-ray block), ``modifiers`` (the statements of the
        ``@reactionModifier`` blocks) and ``skipped`` (the lines left out).

    The KROME index of a reaction is its position among the reactions read,
    starting from one, as in KROME (the first column of the file is not used).
    """
    species, reactions, variables, commons = [], [], [], []
    photo_reactions, cosmic_ray, modifiers, skipped = [], [], [], []
    layout, photo, photo_count, cosmic_rays, modifier_block, unordered = DEFAULT_FORMAT, False, 0, False, False, False

    def index(token):
        name = _species_name(token)
        if name not in species:
            species.append(name)
        return species.index(name)

    with open(path) as file:
        lines = file.readlines()
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith(("@reactionmodifier_stop", "@reactionmodifier_end")):
            modifier_block = False
        elif modifier_block:
            if "@" in line:
                raise ValueError(f"{path}:{number}: a directive inside a @reactionModifier block")
            modifiers.append(line)
        elif lowered.startswith(("@reactionmodifier_start", "@reactionmodifier_begin")):
            modifier_block = True
        elif lowered.startswith("@var:"):
            name, expression = line[5:].split("=", 1)
            variables.append((name.strip().lower(), expression.strip()))
        elif lowered.startswith("@common:"):
            commons += [name.strip().lower() for name in line[8:].split(",") if name.strip()]
        elif lowered.startswith("@format:"):
            layout = tuple(token.strip().lower() for token in line[8:].split(","))
        elif lowered.startswith("@photo_start"):
            photo = True
        elif lowered.startswith("@photo_stop"):
            photo = False
        elif lowered.startswith(("@cr_start", "@cr_begin")):
            cosmic_rays = True
        elif lowered.startswith(("@cr_stop", "@cr_end")):
            cosmic_rays = False
        elif lowered.startswith(IGNORED_DIRECTIVES):
            continue  # tabulation hints: no effect on the rates
        elif line.startswith("@"):
            raise NotImplementedError(f"{path}:{number}: KROME directive not supported: {line.split(':')[0]}")
        else:
            fields = [field.strip() for field in line.split(",", len(layout) - 1)]
            fields[-1] = fields[-1].rsplit("!", 1)[0].rsplit("#", 1)[0].strip()  # a comment after the rate
            if len(fields) != len(layout):
                # KROME prints "WARNING: wrong format for reaction" and goes on without the line
                warnings.warn(
                    f"{path}:{number}: wrong format for reaction {len(reactions) + 1}, line skipped as in KROME: "
                    f"'{line[:60]}' has {len(fields)} fields, the format in force ({','.join(layout)}) has "
                    f"{len(layout)}. A @format line is probably missing or commented out above it.",
                    stacklevel=2,
                )
                skipped.append((number, line))
                continue
            entry = {"r": [], "p": [], "tmin": "NONE", "tmax": "NONE"}
            for key, value in zip(layout, fields):
                if key in ("r", "p"):
                    if value and value.lower() != "g":  # photons are not followed
                        entry[key].append(index(value))
                elif key in ("idx", "tmin", "tmax", "rate"):
                    entry[key] = value
                else:
                    raise NotImplementedError(f"{path}:{number}: column '{key}' of @format not supported")
            slot = -1
            if photo:
                slot, photo_count = photo_count, photo_count + 1
                photo_reactions.append((
                    len(reactions),
                    None if entry["tmin"].upper() in NO_LIMIT else entry["tmin"],
                    None if entry["tmax"].upper() in NO_LIMIT else entry["tmax"],
                    entry["rate"],
                ))
            if cosmic_rays:
                cosmic_ray.append(len(reactions))
            if "idx" in entry and entry["idx"].lstrip("+-").isdigit() and int(entry["idx"]) != len(reactions) + 1:
                unordered = True
            reactions.append((
                len(reactions) + 1, tuple(entry["r"]), tuple(entry["p"]),
                None if photo else _temperature_limit(entry["tmin"], "GE"),
                None if photo else _temperature_limit(entry["tmax"], "LT"),
                entry["rate"], slot,
            ))
    if unordered:
        # KROME: "WARNING: index in <file> are not sequential!"
        warnings.warn(
            f"{path}: the indices in the first column are not sequential; as in KROME the reactions are numbered "
            "in the order in which they are read.",
            stacklevel=2,
        )

    # every reaction must conserve the nuclei and the charge (photons carry neither)
    for krome_index, reactants, products, _, _, _, _ in reactions:
        totals = {}
        for sign, group in ((1, reactants), (-1, products)):
            for i in group:
                counts, charge = _elements_and_charge(species[i])
                for element, count in counts.items():
                    totals[element] = totals.get(element, 0) + sign * count
                totals["charge"] = totals.get("charge", 0) + sign * charge
        if any(totals.values()):
            names = " + ".join(species[i] for i in reactants) + " -> " + " + ".join(species[i] for i in products)
            raise ValueError(f"{path}: reaction {krome_index} ({names}) does not conserve {[k for k, v in totals.items() if v]}")

    return {
        "species": tuple(species), "reactions": tuple(reactions), "variables": tuple(variables), "commons": tuple(commons),
        "photo": tuple(photo_reactions), "cosmic_ray": tuple(cosmic_ray), "modifiers": tuple(modifiers),
        "skipped": tuple(skipped),
    }


# -------------------------------------------------------------
# ======================= ↑ Reading the file ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ============ ↓ Fortran rate expressions in JAX ↓ ============
# -------------------------------------------------------------

_FORTRAN_NUMBER = re.compile(r"(?<![\w.])(\d+\.?\d*|\.\d+)(?:([de])([+-]?\d+))?")


@lru_cache(maxsize=None)
def _syntax_tree(expression, single_precision_literals=False):
    """Fortran expression -> Python syntax tree (names in lower case).

    In Fortran a real constant written without a ``d`` exponent (``0.928``,
    ``2.1e-20``) is a single-precision number. By default the constants are
    read as the decimal numbers they stand for; with
    ``single_precision_literals`` they are rounded to single precision first,
    which reproduces the coefficients of a compiled KROME to round-off (the two
    readings differ by up to 1e-5 in a coefficient).
    """

    def number(match):
        mantissa, letter, exponent = match.groups()
        if letter is None and "." not in mantissa:
            return mantissa  # an integer stays an integer
        value = float(mantissa + ("e" + exponent if letter else ""))
        if single_precision_literals and letter != "d":
            value = float(np.float32(value))
        return repr(value)

    return ast.parse(_FORTRAN_NUMBER.sub(number, expression.strip().lower()), mode="eval").body


def _evaluate(node, names, functions):
    """Evaluate a syntax tree with Fortran's arithmetic on JAX values."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise NameError(f"KROME rate expression: unknown name '{node.id}'")
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate(node.operand, names, functions)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, names, functions)
        right = _evaluate(node.right, names, functions)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if isinstance(left, int) and isinstance(right, int):
                return int(left / right)  # Fortran integer division truncates
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right  # an integer exponent stays an integer power
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        if node.func.id not in functions:
            raise NameError(f"KROME rate expression: function '{node.func.id}' not supported")
        return functions[node.func.id](*[_evaluate(argument, names, functions) for argument in node.args])
    raise SyntaxError(f"KROME rate expression: construct not supported: {ast.dump(node)[:80]}")


def _temperature_names(temperature, single_precision_literals=False):
    """Tgas and KROME's temperature shortcuts (kromelib.py, get_Tshortcut)."""
    te = temperature * KELVIN_TO_ELECTRON_VOLT
    # KROME writes this factor without a ``d`` exponent, i.e. in single precision
    t32 = temperature * (float(np.float32(0.0033333333333333335)) if single_precision_literals else 0.0033333333333333335)
    sqrtgas = jnp.sqrt(temperature)
    tgas2 = temperature * temperature
    t0 = 288.0
    return {
        "tgas": temperature, "logt": jnp.log10(temperature), "lnt": jnp.log(temperature),
        "te": te, "lnte": jnp.log(te), "t32": t32, "t3": t32, "t4": temperature * 1e-4,
        "invt": 1.0 / temperature, "invt32": 1.0 / t32, "invtgas": 1.0 / temperature, "invte": 1.0 / te,
        "sqrtgas": sqrtgas, "invsqrt32": 1.0 / jnp.sqrt(t32), "invsqrt": 1.0 / sqrtgas, "sqrt32": jnp.sqrt(t32),
        "tgas2": tgas2, "tgas3": tgas2 * temperature, "tgas4": tgas2 * tgas2,
        "t0": t0, "t02": t0 * t0, "t03": t0 ** 3, "t04": t0 ** 4, "t0inv": 1.0 / t0,
    }


def _dust_recombination(ion, electron_density, temperature, habing_field, dust_to_gas, single_precision_literals=False):
    """Recombination coefficient of an ion on dust grains [cm^3/s] (Weingartner & Draine 2001).

    Zero without electrons or without a radiation field, as in KROME.
    """
    c0, c1, c2, c3, c4, c5, c6 = DUST_RECOMBINATION[ion]
    if single_precision_literals:
        # the exponents that KROME's source writes without a ``d`` exponent
        c2, c5 = float(np.float32(c2)), float(np.float32(c5))
        if ion in ("h", "c"):
            c4 = float(np.float32(c4))
    active = (electron_density >= 1e-20) & (habing_field > 0.0)
    psi = jnp.where(active, habing_field, 1.0) * jnp.sqrt(temperature) / jnp.where(active, electron_density, 1.0)
    value = 1e-14 * c0 * dust_to_gas / (
        1.0 + c1 * psi ** c2 * (1.0 + c3 * temperature ** c4 * psi ** (-c5 - c6 * jnp.log(temperature)))
    )
    return jnp.where(active, value, 0.0)


# -------------------------------------------------------------
# ============ ↑ Fortran rate expressions in JAX ↑ ============
# -------------------------------------------------------------


class KromeNetwork(eqx.Module):
    """Right-hand side of a KROME network, callable like a carbox ``JNetwork``.

    ``network(time, abundances, temperature, cosmic_ray_rate, fuv_field,
    visual_extinction, rate_modifier_a, rate_modifier_b)`` returns the time
    derivative of the number densities [cm^-3 s^-1]. ``cosmic_ray_rate`` is
    KROME's ``user_crate`` and ``fuv_field`` its unattenuated Habing flux
    ``GHabing_thin`` (it enters the recombination and the photoelectric heating
    on dust); with photo bins the Habing flux is that of the binned spectrum and
    ``fuv_field`` is not used. The visual extinction is not used.
    """

    # The rates are products of the number densities and guarded fits, finite for
    # the slightly negative values an implicit solver probes, so the chemistry
    # module does not clamp the abundances before the call. With the clamp, the
    # solve stalls when an ion decays to zero, as it does here without cosmic
    # rays and photo-rates.
    accepts_negative_abundances = True
    # The implicit stages of the Kvaerno5 solve use Newton instead of the chord
    # iteration, which stalls on the ionisation equilibrium of hot, dense gas.
    newton_root_finder = True
    # The temperature derivative below replaces the default one of the chemistry
    # module (heating and cooling as in KROME).
    own_thermochemistry = True

    stoichiometry: jnp.ndarray  # [species, reactions], products minus reactants
    photo_rates: jnp.ndarray  # [photo-reactions], 1/s
    photo_heats: jnp.ndarray  # [photo-reactions], heating per reaction [erg/s]
    bins_habing_field: jnp.ndarray  # Habing flux of the binned spectrum
    masses: jnp.ndarray  # [species], g
    user_values: jnp.ndarray  # values of the @common variables other than user_crate
    dust_to_gas: jnp.ndarray  # dust-to-gas ratio relative to solar (KROME's dust2gas_ratio)
    h2_on_dust_coefficient: jnp.ndarray  # cm^3/s; 0 switches the H2 formation on dust off
    hydrogen_nuclei: jnp.ndarray  # [species], hydrogen atoms in every species
    h2_on_dust_change: jnp.ndarray  # [species], -2 for H and +1 for H2
    metallicity: jnp.ndarray  # metal abundance relative to solar, the factor of the tabulated metal cooling
    metal_table: tuple  # (log n_H, log T, log Lambda) of the metal cooling above 1e4 K, or ()
    species: Tuple[str, ...] = eqx.field(static=True)
    metal_lines: tuple = eqx.field(static=True)  # (variables, upper limit, metals) of KROME's coolZ.dat, or ()
    chemical_terms: tuple = eqx.field(static=True)  # (position, reactants, energy [eV], thermalised) of the chemical heating
    ortho_para_ratio: float = eqx.field(static=True)
    single_precision_literals: bool = eqx.field(static=True)
    reactions: tuple = eqx.field(static=True)
    variables: tuple = eqx.field(static=True)
    user_names: Tuple[str, ...] = eqx.field(static=True)
    reactions_number: int = eqx.field(static=True)
    modifiers: tuple = eqx.field(static=True)  # (position, expression) of the @reactionModifier statements
    cosmic_ray_terms: tuple = eqx.field(static=True)  # (position, reactant, kind) of the cosmic-ray reactions
    photo_reactants: tuple = eqx.field(static=True)  # reactant of every photo-reaction
    has_photo_bins: bool = eqx.field(static=True)
    shielding: str = eqx.field(static=True)  # "", "DB96", "WG11" or "R14": what krome_fshield returns
    shield_habing_dust: bool = eqx.field(static=True)
    column_density_method: str = eqx.field(static=True)
    cooling_floor_temperature: float = eqx.field(static=True)
    # charge squared of every species in the bremsstrahlung (KROME's bms_ions), or () without that term
    free_free_charges: Tuple[float, ...] = eqx.field(static=True)
    # the metal table without a look-up (for the per-cell Warp kernel), see ``build_krome_network``
    metal_table_without_lookup: bool = eqx.field(static=True, default=False)
    # user variables that are radiation fluxes: they scale with ``radiation_scale`` like the photo-reactions
    radiation_user_names: Tuple[str, ...] = eqx.field(static=True, default=())
    # the photoelectric heating on dust (and its recombination cooling) only in the neutral hydrogen: times
    # 1 - n(H+)/n_H, see ``build_krome_network``
    photoelectric_neutral_only: bool = eqx.field(static=True, default=False)

    #: The radiation of this network can be scaled per call (``radiation_scale`` of ``__call__`` and of the
    #: heating): the chemistry module passes the radiation field of a cell through it.
    scales_its_radiation = True

    def _single(self, value):
        # a real constant that KROME's Fortran writes without a ``d`` exponent
        return float(np.float32(value)) if self.single_precision_literals else float(value)

    def _density(self, abundances, name):
        return abundances[self.species.index(name)] if name in self.species else 0.0

    def jeans_length(self, abundances, temperature):
        """Jeans length [cm] (KROME's ``get_jeans_length``)."""
        mass_density = jnp.maximum(jnp.dot(self.masses, abundances), 1e-40)
        mean_weight = mass_density / jnp.maximum(jnp.sum(abundances), 1e-40) / PROTON_MASS
        return jnp.sqrt(PI * BOLTZMANN_ERG * temperature / mass_density / PROTON_MASS / GRAVITY / mean_weight)

    def column_density(self, density, abundances, temperature):
        """Column density [cm^-2] of a species of number density ``density`` (KROME's ``num2col``).

        ``DEFAULT``: ``1.87e21 (n / 1e3 cm^-3)^(2/3)``; ``JEANS``: half a Jeans length.
        """
        density = jnp.maximum(density, 1e-40)
        if self.column_density_method == "JEANS":
            return 0.5 * density * self.jeans_length(abundances, temperature)
        exponent = float(np.float32(2.0) / np.float32(3.0)) if self.single_precision_literals else 2.0 / 3.0
        return 1.87e21 * (density * 1e-3) ** exponent

    def dust_shielding(self, abundances, temperature, gamma):
        """Attenuation by dust, ``exp(-2e-21 gamma D/D_sun N_H)`` (KROME's ``shield_dust``)."""
        column = 0.0
        for name, atoms in (("H", 1.0), ("H+", 1.0), ("H2", 2.0)):
            if name in self.species:
                column = column + atoms * self.column_density(self._density(abundances, name), abundances, temperature)
        return jnp.exp(-2e-21 * self.dust_to_gas * gamma * column)

    def h2_self_shielding(self, abundances, temperature):
        """Self-shielding factor of H2 (KROME's ``krome_fshield``), one without the ``shielding`` option."""
        if not self.shielding:
            return 1.0
        column = 2.0 * self.column_density(jnp.maximum(1e-40, self._density(abundances, "H2")), abundances, temperature)
        hydrogen_mass = PROTON_MASS + ELECTRON_MASS
        if self.shielding == "DB96":  # Draine & Bertoldi (1996)
            return jnp.minimum(1.0, (column * 1e-14) ** (-0.75))
        if self.shielding == "WG11":  # Wolcott-Green et al. (2011)
            x = column * 2e-15
            b5 = (BOLTZMANN_ERG * temperature / hydrogen_mass) ** 0.5 * 1e-5
            return 0.965 / (1.0 + x / b5) ** 1.1 + 0.035 / (1.0 + x) ** 0.5 * jnp.exp(-8.5e-4 * (1.0 + x) ** 0.5)
        # Richings et al. (2014), temperature dependent; as in KROME every
        # temperature from 3000 K on takes the fit of the 3000-4000 K range
        one = self._single
        b5 = (BOLTZMANN_ERG * temperature / hydrogen_mass + (7.1 * 1e5) ** 2) ** 0.5 * 1e-5
        if self.single_precision_literals:
            inverse = float(np.float32(1.0) / np.float32(1.3))
        else:
            inverse = 1.0 / 1.3
        omega = 0.013 * (1.0 + (temperature / 2.7e3) ** one(1.3)) ** inverse * jnp.exp(-((temperature / 3.9e3) ** one(14.6)))
        cold = temperature < 3e3
        warm_temperature = jnp.maximum(temperature, 3e3)  # the warm fit is evaluated where it applies
        alpha = jnp.where(cold, one(1.4), (warm_temperature / 4.5e3) ** one(-0.8))
        critical = jnp.where(cold, 1.3 * (1.0 + (temperature / 6e2) ** one(0.8)), (warm_temperature / 4.76e3) ** one(-3.8))
        x = column * 1e-14 / critical
        return (1.0 - omega) / (1.0 + x / b5) ** alpha * jnp.exp(-5e-7 * (1.0 + x)) + omega / jnp.sqrt(1.0 + x) * jnp.exp(
            -8.5e-4 * jnp.sqrt(1.0 + x)
        )

    def habing_field(self, abundances, temperature, fuv_field, radiation_scale=None):
        """The Habing flux of the dust terms: that of the photo bins or ``fuv_field``, times ``radiation_scale`` when
        that is given, attenuated by dust on request."""
        field = self.bins_habing_field if self.has_photo_bins else fuv_field
        if radiation_scale is not None:
            field = field * radiation_scale
        if self.shield_habing_dust:
            field = field * self.dust_shielding(abundances, temperature, 0.665)
        return field

    def get_rates(self, temperature, cosmic_ray_rate, fuv_field, visual_extinction, abundances, radiation_scale=None):
        """Rate coefficients of every reaction, in the order of the file (cm^3/s or 1/s).

        ``radiation_scale`` multiplies the radiation the network was built with: the rates of the photo-reactions,
        the Habing flux of the dust terms and the user variables that are radiation fluxes
        (``radiation_user_variables`` of :func:`build_krome_network`). The spectrum keeps its shape. ``None`` (the
        default) is the network as it was built.
        """
        temperature = jnp.clip(temperature, MINIMUM_TEMPERATURE_KELVIN, MAXIMUM_TEMPERATURE_KELVIN)
        electron_density = abundances[self.species.index("E")] if "E" in self.species else 0.0
        habing_field = self.habing_field(abundances, temperature, fuv_field, radiation_scale)
        functions = {
            "exp": jnp.exp, "dexp": jnp.exp, "log": jnp.log, "dlog": jnp.log, "log10": jnp.log10, "dlog10": jnp.log10,
            "sqrt": jnp.sqrt, "dsqrt": jnp.sqrt, "abs": jnp.abs, "max": jnp.maximum, "min": jnp.minimum,
            "h2_solomonlw": lambda flux: 1.38e9 * flux * ELECTRON_VOLT_TO_ERG,
        }
        # KROME's shielding functions. The cell is the one of this call whatever the
        # arguments, so every factor is computed once and shared by the expressions
        # that use it.
        shielding_factors = {}

        def shield_dust(n, t, gamma):
            if not isinstance(gamma, (int, float)):  # an exponent that is itself computed
                return self.dust_shielding(abundances, temperature, gamma)
            if float(gamma) not in shielding_factors:
                shielding_factors[float(gamma)] = self.dust_shielding(abundances, temperature, gamma)
            return shielding_factors[float(gamma)]

        def krome_fshield(n, t):
            if "self" not in shielding_factors:
                shielding_factors["self"] = self.h2_self_shielding(abundances, temperature)
            return shielding_factors["self"]

        functions["shield_dust"], functions["krome_fshield"] = shield_dust, krome_fshield
        for ion in DUST_RECOMBINATION:
            functions[f"{ion}_recombination_on_dust"] = (
                lambda n, t, ion=ion: _dust_recombination(
                    ion, electron_density, t, habing_field, self.dust_to_gas, self.single_precision_literals
                )
            )
        functions["o_recombination_on_dust"] = lambda n, t: 0.25 * functions["h_recombination_on_dust"](n, t)
        users = {"user_crate": cosmic_ray_rate, "n": abundances}
        users.update({name: self.user_values[i] for i, name in enumerate(self.user_names)})
        if radiation_scale is not None:
            users.update({name: users[name] * radiation_scale for name in self.radiation_user_names})

        environments = {}

        def names_at(lower, upper):
            # A coefficient with temperature limits is evaluated at the temperature
            # clipped into them and masked outside, so that neither its value nor
            # its derivative is ever taken where the fit does not hold.
            key = (None if lower is None else lower[1], None if upper is None else upper[1])
            if key not in environments:
                names = _temperature_names(jnp.clip(temperature, key[0], key[1]), self.single_precision_literals)
                names.update(users)
                for name, expression in self.variables:
                    names[name] = _evaluate(_syntax_tree(expression, self.single_precision_literals), names, functions)
                environments[key] = names
            return environments[key]

        compare = {"GE": jnp.greater_equal, "GT": jnp.greater, "LE": jnp.less_equal, "LT": jnp.less}
        rates = []
        for _, _, _, lower, upper, expression, photo_slot in self.reactions:
            if photo_slot >= 0:
                photo_rate = self.photo_rates[photo_slot]
                rates.append(photo_rate if radiation_scale is None else photo_rate * radiation_scale)
                continue
            value = _evaluate(_syntax_tree(expression, self.single_precision_literals), names_at(lower, upper), functions)
            for limit in (lower, upper):
                if limit is not None:
                    value = jnp.where(compare[limit[0]](temperature, limit[1]), value, 0.0)
            rates.append(value * jnp.ones_like(temperature))
        if self.modifiers:
            # statements ``k(i) = ...`` of the @reactionModifier blocks, in file order
            functions["k"] = lambda index: rates[int(index) - 1]
            names = names_at(None, None)
            for position, expression in self.modifiers:
                value = _evaluate(_syntax_tree(expression, self.single_precision_literals), names, functions)
                rates[position] = value * jnp.ones_like(temperature)
        return jnp.stack(rates)

    def __call__(self, time, abundances, temperature, cosmic_ray_rate, fuv_field, visual_extinction, rate_modifier_a, rate_modifier_b,
                 radiation_scale=None):
        rates = self.get_rates(temperature, cosmic_ray_rate, fuv_field, visual_extinction, abundances, radiation_scale)
        rates = rate_modifier_a * rates + rate_modifier_b
        fluxes = []
        for position, (_, reactants, _, _, _, _, _) in enumerate(self.reactions):
            flux = rates[position]
            for i in reactants:
                flux = flux * abundances[i]
            fluxes.append(flux)
        derivative = self.stoichiometry @ jnp.stack(fluxes)
        # H2 formation on dust (Jura): n(H) n_H k D/D_sun molecules per cm^3 per second
        atomic_hydrogen = jnp.dot(jnp.maximum(-self.h2_on_dust_change, 0.0), abundances) * 0.5
        formation = self.h2_on_dust_coefficient * self.dust_to_gas * atomic_hydrogen * jnp.dot(self.hydrogen_nuclei, abundances)
        return derivative + self.h2_on_dust_change * formation

    def cooling_terms(self, abundances, temperature):
        """The cooling terms [erg cm^-3 s^-1], by name (see ``_krome_thermochemistry.py``)."""
        from astronomix._modules._chemistry import _krome_thermochemistry as thermo
        from astronomix._modules._chemistry._thermochemistry import _atomic_oxygen_cooling, _ionized_carbon_cooling

        temperature = jnp.clip(temperature, MINIMUM_TEMPERATURE_KELVIN, MAXIMUM_TEMPERATURE_KELVIN)
        n = jnp.maximum(abundances, 0.0)
        index = lambda name: self.species.index(name) if name in self.species else -1
        density = lambda name: n[self.species.index(name)] if name in self.species else 0.0
        electrons, hydrogen = density("E"), density("H")
        nuclei = jnp.dot(self.hydrogen_nuclei, n)

        terms = {
            "cooling_h2": thermo.krome_h2_line_cooling(
                temperature, density("H2"), hydrogen, density("H+"), electrons, density("He"), self.single_precision_literals
            ),
            "cooling_atomic": thermo.atomic_cooling(
                temperature, electrons, hydrogen, density("He"), density("H+"), density("He+"), density("He++")
            ),
            "cooling_compton": thermo.compton_cooling(temperature, electrons),
        }
        if self.free_free_charges:
            # The term exists only in a network built with it, so that a network
            # built without it keeps the right-hand side (and the kernel) it had.
            terms["cooling_free_free"] = thermo.free_free_cooling(
                temperature,
                electrons,
                jnp.dot(jnp.asarray(self.free_free_charges), n),
            )
        if self.metal_lines:
            variables, upper_limit, metals = self.metal_lines
            lines = thermo.metal_line_cooling(
                temperature, density, [dict(metal) for metal in metals], variables, upper_limit,
                _evaluate, _syntax_tree, self.ortho_para_ratio,
            )
            blend = thermo.metal_blend(temperature)
            terms["cooling_metal_lines"] = (1.0 - blend) * lines
            if self.metal_table:
                tabulated = (
                    thermo.tabulated_metal_cooling_without_lookup
                    if self.metal_table_without_lookup
                    else thermo.tabulated_metal_cooling
                )
                terms["cooling_metal_table"] = blend * tabulated(temperature, nuclei, self.metallicity, *self.metal_table)
        else:
            # without KROME's data: the two fine-structure lines of the chemistry module
            terms["cooling_metal_lines"] = _ionized_carbon_cooling(
                n, temperature, index("C+"), index("E"), index("H"), index("H2")
            ) + _atomic_oxygen_cooling(n, temperature, index("O"), index("H"), index("H2"))
        return terms

    def heating_terms(self, abundances, temperature, cosmic_ray_rate, fuv_field, radiation_scale=None):
        """The heating terms [erg cm^-3 s^-1], by name (see ``_krome_thermochemistry.py``)."""
        from astronomix._modules._chemistry import _krome_thermochemistry as thermo

        temperature = jnp.clip(temperature, MINIMUM_TEMPERATURE_KELVIN, MAXIMUM_TEMPERATURE_KELVIN)
        n = jnp.maximum(abundances, 0.0)
        density = lambda name: n[self.species.index(name)] if name in self.species else 0.0
        electrons, hydrogen, molecular = density("E"), density("H"), density("H2")
        nuclei = jnp.dot(self.hydrogen_nuclei, n)
        terms = {}

        fraction = thermo.thermalised_fraction(temperature, hydrogen, molecular, nuclei)
        rates = self.get_rates(temperature, cosmic_ray_rate, fuv_field, 0.0, n, radiation_scale)
        chemical = self.h2_on_dust_coefficient * self.dust_to_gas * hydrogen * nuclei * (4.2 * fraction + 0.2)
        for position, reactants, energy, thermalised in self.chemical_terms:
            flux = rates[position] * energy * (fraction if thermalised else 1.0)
            for i in reactants:
                flux = flux * n[i]
            chemical = chemical + flux
        terms["heating_chemical"] = chemical * thermo.ELECTRON_VOLT_TO_ERG
        terms["heating_photoelectric"] = thermo.photoelectric_heating(
            temperature, electrons, hydrogen, nuclei, self.habing_field(n, temperature, fuv_field, radiation_scale), self.dust_to_gas
        )
        if self.photoelectric_neutral_only:
            # only in the neutral gas, as the photoelectric heating of the KI02 cooling: Gamma (1 - x_HII)
            ionised = density("H+") / jnp.where(nuclei > 0.0, nuclei, 1.0)
            terms["heating_photoelectric"] = terms["heating_photoelectric"] * jnp.clip(1.0 - ionised, 0.0, 1.0)
        # cosmic rays: every reaction of the cosmic-ray block deposits an energy per event
        cosmic_rays = 0.0
        for position, reactant, kind in self.cosmic_ray_terms:
            if kind == "H2":
                energy = thermo.cosmic_ray_heat_per_h2_ionisation(
                    molecular, COSMIC_RAY_HEAT_H2_POLYNOMIAL, COSMIC_RAY_HEAT_H2_LINEAR
                )
            else:
                energy = COSMIC_RAY_HEAT_ATOMIC if kind == "atomic" else COSMIC_RAY_HEAT_OTHER
            cosmic_rays = cosmic_rays + rates[position] * n[reactant] * energy
        terms["heating_cosmic_rays"] = cosmic_rays * COSMIC_RAY_ELECTRON_VOLT_TO_ERG
        # photo-reactions: the energy of the photons above the threshold
        photo = 0.0
        for slot, reactant in enumerate(self.photo_reactants):
            photo = photo + self.photo_heats[slot] * n[reactant]
        terms["heating_photo"] = photo if radiation_scale is None else photo * radiation_scale
        return terms

    def heating_and_cooling_terms(self, abundances, temperature, cosmic_ray_rate, fuv_field, radiation_scale=None):
        """The heating and cooling terms [erg cm^-3 s^-1], by name (see ``_krome_thermochemistry.py``)."""
        terms = self.cooling_terms(abundances, temperature)
        terms.update(self.heating_terms(abundances, temperature, cosmic_ray_rate, fuv_field, radiation_scale))
        return terms

    def temperature_derivative(self, abundances, temperature, adiabatic_index, cosmic_ray_rate, fuv_field, radiation_scale=None):
        """dT/dt [K/s] from the heating and cooling of KROME, at constant density.

        As in KROME the cooling at the floor temperature is subtracted from the
        cooling, so that the gas does not cool below the floor.
        """
        heating = sum(self.heating_terms(abundances, temperature, cosmic_ray_rate, fuv_field, radiation_scale).values())
        cooling = sum(self.cooling_terms(abundances, temperature).values())
        floor = sum(self.cooling_terms(abundances, self.cooling_floor_temperature * jnp.ones_like(temperature)).values())
        particles = jnp.maximum(jnp.sum(jnp.maximum(abundances, 0.0)), 1e-30)
        return (adiabatic_index - 1.0) * (heating - cooling + floor) / (BOLTZMANN_ERG * particles)


def build_krome_network(
    path, photo_rates=None, user_variables=None, dust_to_gas_relative=1.0, h2_formation_on_dust=False,
    data_folder=None, metal_table=True, metallicity=None, ortho_para_ratio=3.0, single_precision_literals=False,
    photo_bins=None, xsec_folder=None, shielding=None, shield_habing_dust=False, column_density_method="DEFAULT",
    floor_temperature=MINIMUM_TEMPERATURE_KELVIN,
    free_free_cooling=False,
    radiation_user_variables=(),
    photoelectric_neutral_only=False,
):
    """Build the right-hand side of a KROME network file.

    Args:
        path: The KROME network file.
        photo_rates: Rates [1/s] of the photo-reactions, as a dictionary from the
            KROME reaction index (the position of the reaction among those
            read, from one) to the rate; the others are zero. Not with
            ``photo_bins``.
        user_variables: Values of the ``@common`` user variables other than
            ``user_crate`` (which is the cosmic-ray rate of the chemistry
            module); a variable that the rates use must be given.
        dust_to_gas_relative: Dust-to-gas ratio relative to solar, the factor of
            the recombination on dust and of the H2 formation on dust.
        h2_formation_on_dust: Add the formation of H2 on dust grains with the
            rate of Jura (KROME's ``-useH2dust=JURA``); ``True`` for the
            coefficient 3.5e-17 cm^3/s, or the coefficient itself (it absorbs a
            clumping factor). It is not part of the network file.
        data_folder: The ``data`` folder of a KROME installation. With it the
            metals cool as in KROME: the lines of ``coolZ.dat`` below 1e4 K and
            the table ``coolZ_CIE2012NOUV.dat`` above. Without it they cool
            through the [C II] and [O I] lines of the chemistry module only.
        metal_table: Read the table (default). The per-cell Warp kernel cannot
            look a table up. With ``"kernel"`` the table is interpolated without
            a look-up, every node entering with its weight
            (``tabulated_metal_cooling_without_lookup``): the same cooling to
            round-off in a form the kernel can take, at one product per table
            value in every call. With ``False`` the table is left out: the
            metals then cool through their lines alone, i.e. not above 1e4 K.
        metallicity: Metal abundance relative to solar, the factor of the
            tabulated metal cooling (default: ``dust_to_gas_relative``).
        ortho_para_ratio: Ortho-to-para ratio of H2 as a collider of the metal lines.
        single_precision_literals: Round the real constants written without a
            ``d`` exponent to single precision, as a Fortran compiler does (for
            comparisons with KROME to round-off; see ``_syntax_tree``).
        photo_bins: The radiation spectrum in energy bins (KROME's
            ``-photoBins``): a dictionary with the edges ``left`` and ``right``
            [eV] and the ``intensity`` [eV s^-1 cm^-2 sr^-1 Hz^-1] of every bin
            (``_krome_photo.photo_spectrum`` builds the spectra of KROME). The
            rates and the heating of the photo-reactions and the Habing flux
            follow from it as in KROME; it needs ``data_folder`` for the cross
            sections of KROME's database.
        xsec_folder: A KROME build folder; a cross section table found there
            (``swri_*.dat``, ``leiden_*.dat``) is used instead of the one
            derived from the database.
        shielding: The H2 self-shielding that ``krome_fshield(n,Tgas)`` returns
            in a rate expression: ``"DB96"``, ``"WG11"`` or ``"R14"`` (KROME's
            ``-shielding``); without it the function returns one.
        shield_habing_dust: Attenuate the Habing flux by dust (KROME's
            ``-shieldHabingDust``).
        column_density_method: ``"DEFAULT"`` or ``"JEANS"`` (KROME's
            ``-columnDensityMethod``), for the two shielding terms.
        floor_temperature: Temperature [K] at which the cooling is subtracted
            from itself (KROME's ``phys_Tfloor``).
        free_free_cooling: Add the bremsstrahlung of the atomic ions of the
            network (KROME's ``-cooling=FF``), which dominates the cooling of
            ionised gas above about 1e7 K. Off by default, as in a KROME build
            without that option.
        radiation_user_variables: Names of ``@common`` user variables that are
            radiation fluxes, e.g. the Lyman-Werner flux of an
            ``H2_solomonLW(user_...)`` rate. Their values in ``user_variables``
            are those of the spectrum the network is built with; a
            ``radiation_scale`` given at a call (the radiation field of a cell)
            multiplies them together with the photo-reactions and the Habing
            flux, so that every radiative process follows the same field.
        photoelectric_neutral_only: Multiply the photoelectric heating on dust
            (net of the recombination cooling on the grains, Bakes & Tielens
            1994) by the neutral fraction of hydrogen, 1 - n(H+)/n_H. KROME
            applies it at every temperature, but the fit holds below about
            1e4 K and its efficiency grows as T^0.7 above; in hot, ionised gas
            the grains are also eroded by sputtering. The factor is the one of
            the photoelectric heating of the KI02 cooling (Gamma (1 - x_HII)).
            Off by default, as in KROME.

    Returns:
        The network (a :class:`KromeNetwork`) and the species names.
    """
    parsed = parse_krome_network(path)
    species, reactions = parsed["species"], parsed["reactions"]

    stoichiometry = np.zeros((len(species), len(reactions)))
    for position, (_, reactants, products, _, _, _, _) in enumerate(reactions):
        for i in reactants:
            stoichiometry[i, position] -= 1.0
        for i in products:
            stoichiometry[i, position] += 1.0

    photo_rates = dict(photo_rates or {})
    photo_indices = {krome_index: slot for krome_index, _, _, _, _, _, slot in reactions if slot >= 0}
    unknown = sorted(set(photo_rates) - set(photo_indices))
    if unknown:
        raise ValueError(f"{path}: reactions {unknown} are not photo-reactions (photo-reactions: {sorted(photo_indices)})")
    photo = np.zeros(len(photo_indices))
    for krome_index, rate in photo_rates.items():
        photo[photo_indices[krome_index]] = rate
    photo_heats, bins_habing_field = np.zeros(len(photo_indices)), 0.0
    if photo_bins is not None:
        from astronomix._modules._chemistry import _krome_photo

        if photo_rates:
            raise ValueError(f"{path}: give the photo-rates or the photo bins, not both")
        if isinstance(photo_bins, dict):
            left, right, intensity = photo_bins["left"], photo_bins["right"], photo_bins["intensity"]
        else:
            left, right, intensity = photo_bins
        left, right, intensity = (np.atleast_1d(np.asarray(array, dtype=float)) for array in (left, right, intensity))
        if not (left.shape == right.shape == intensity.shape) or np.any(right <= left):
            raise ValueError(f"{path}: the photo bins need as many left edges, right edges and intensities, with right > left")
        table, thresholds = _krome_photo.cross_sections(
            [
                (tuple(species[i] for i in reactions[position][1]), tuple(species[i] for i in reactions[position][2]), lower, upper, rate)
                for position, lower, upper, rate in parsed["photo"]
            ],
            lambda name: name.upper(), left, right, data_folder=data_folder, xsec_folder=xsec_folder,
            ortho_para_ratio=ortho_para_ratio, single_precision_literals=single_precision_literals,
            evaluate=_evaluate, syntax_tree=_syntax_tree,
        )
        photo, photo_heats = _krome_photo.rates_and_heats(table, thresholds, left, right, intensity)
        bins_habing_field = _krome_photo.habing_field(left, right, intensity, single_precision_literals)
    for position, _, _, _ in parsed["photo"]:
        if len(reactions[position][1]) != 1:
            raise ValueError(f"{path}: photo-reaction {position + 1} must have one reactant")

    if shielding is not None and shielding not in SHIELDINGS:
        raise ValueError(f"shielding '{shielding}' unknown: one of {SHIELDINGS}")
    if column_density_method not in COLUMN_DENSITY_METHODS:
        raise ValueError(f"column density method '{column_density_method}' unknown: one of {COLUMN_DENSITY_METHODS}")
    uses_shielding = any(
        "krome_fshield" in text.lower()
        for text in [r[5] for r in reactions] + [expression for _, expression in parsed["variables"]] + list(parsed["modifiers"])
    )
    if shielding is not None and not uses_shielding:
        # KROME: "WARNING: no krome_fshield(n(:),Tgas) variable found in rate coefficient"
        warnings.warn(
            f"{path}: the shielding {shielding} has no effect, since no rate coefficient of the network uses "
            "krome_fshield(n,Tgas), as in KROME.",
            stacklevel=2,
        )
    if (shielding is not None and uses_shielding) and "H2" not in species:
        raise ValueError(f"{path}: the H2 self-shielding needs the species H2")

    # statements ``k(i) = ...`` of the @reactionModifier blocks
    modifiers = []
    for statement in parsed["modifiers"]:
        match = re.fullmatch(r"\s*k\(\s*(\d+)\s*\)\s*=(.*)", statement, flags=re.IGNORECASE)
        if match is None or not 1 <= int(match.group(1)) <= len(reactions):
            raise NotImplementedError(f"{path}: reaction modifier not supported: {statement}")
        modifiers.append((int(match.group(1)) - 1, match.group(2).strip()))

    # cosmic-ray reactions and the energy each deposits (KROME's heat_CR)
    cosmic_ray_terms = []
    for position in parsed["cosmic_ray"]:
        _, reactants, products, _, _, _, _ = reactions[position]
        if len(reactants) != 1:
            raise ValueError(f"{path}: cosmic-ray reaction {position + 1} must have one reactant")
        reactant, made = species[reactants[0]], sorted(species[i] for i in products)
        if (reactant, made) in (("H", ["E", "H+"]), ("He", ["E", "He+"])):
            kind = "atomic"
        elif (reactant, made) == ("H2", ["E", "H2+"]):
            kind = "H2"
        else:
            kind = "other"
        cosmic_ray_terms.append((position, reactants[0], kind))

    user_variables = {name.lower(): value for name, value in (user_variables or {}).items()}
    user_names = tuple(name for name in parsed["commons"] if name != "user_crate")
    unknown = sorted(set(user_variables) - set(user_names))
    if unknown:
        raise ValueError(f"{path}: {unknown} are not @common variables of the network ({list(user_names)})")
    used = " ".join(expression.lower() for _, expression in parsed["variables"]) + " " + " ".join(r[5].lower() for r in reactions)
    missing = [name for name in user_names if name not in user_variables and re.search(rf"\b{name}\b", used)]
    if missing:
        raise ValueError(f"{path}: the rates use the user variables {missing}; give them in user_variables")
    radiation_user_names = tuple(name.lower() for name in radiation_user_variables)
    unknown = sorted(set(radiation_user_names) - set(user_names))
    if unknown:
        raise ValueError(f"{path}: radiation_user_variables {unknown} are not @common variables of the network ({list(user_names)})")

    coefficient = 3.5e-17 if h2_formation_on_dust is True else float(h2_formation_on_dust or 0.0)
    change = np.zeros(len(species))
    if coefficient:
        if "H" not in species or "H2" not in species:
            raise ValueError(f"{path}: H2 formation on dust needs the species H and H2")
        change[species.index("H")], change[species.index("H2")] = -2.0, 1.0

    # heating and cooling: metal coolants of KROME's data folder, reactions of the chemical heating
    if metal_table not in (True, False, "kernel"):
        raise ValueError(f"metal_table must be True, False or 'kernel', got {metal_table!r}")
    with_table, without_lookup, metal_lines, metal_table = bool(metal_table), metal_table == "kernel", (), ()
    if data_folder is not None:
        import os

        from astronomix._modules._chemistry import _krome_thermochemistry as thermo

        variables, upper_limit, metals = thermo.read_metal_lines(os.path.join(data_folder, thermo.METAL_LINES_FILE), _species_name)
        kept = []
        for metal in metals:
            if metal["name"] not in species:
                continue
            if len(metal["levels"]) not in (2, 3):
                raise NotImplementedError(f"{metal['name']}: {len(metal['levels'])}-level coolant (two and three levels are solved)")
            kept.append(tuple((key, tuple(value) if isinstance(value, list) else value) for key, value in metal.items()))
        metal_lines = (variables, upper_limit, tuple(kept))
        if with_table:
            metal_table = tuple(jnp.asarray(array) for array in thermo.read_cooling_table(os.path.join(data_folder, thermo.METAL_TABLE_FILE)))
    from astronomix._modules._chemistry._krome_thermochemistry import CHEMICAL_HEATING

    chemical_terms = []
    for position, (_, reactants, products, _, _, _, _) in enumerate(reactions):
        key = (tuple(sorted(species[i] for i in reactants)), tuple(sorted(species[i] for i in products)))
        for wanted_reactants, wanted_products, energy, thermalised in CHEMICAL_HEATING:
            if key == (tuple(sorted(wanted_reactants)), tuple(sorted(wanted_products))):
                chemical_terms.append((position, reactants, energy, thermalised))

    network = KromeNetwork(
        stoichiometry=jnp.asarray(stoichiometry),
        photo_rates=jnp.asarray(photo),
        photo_heats=jnp.asarray(photo_heats),
        bins_habing_field=jnp.asarray(float(bins_habing_field)),
        masses=jnp.asarray([_species_mass(name) for name in species]),
        user_values=jnp.asarray([float(user_variables.get(name, 0.0)) for name in user_names]),
        dust_to_gas=jnp.asarray(float(dust_to_gas_relative)),
        h2_on_dust_coefficient=jnp.asarray(coefficient),
        hydrogen_nuclei=jnp.asarray([float(_elements_and_charge(name)[0].get("H", 0)) for name in species]),
        h2_on_dust_change=jnp.asarray(change),
        metallicity=jnp.asarray(float(dust_to_gas_relative if metallicity is None else metallicity)),
        metal_table=metal_table,
        species=species,
        metal_lines=metal_lines,
        chemical_terms=tuple(chemical_terms),
        ortho_para_ratio=float(ortho_para_ratio),
        single_precision_literals=bool(single_precision_literals),
        reactions=reactions,
        variables=parsed["variables"],
        user_names=user_names,
        reactions_number=len(reactions),
        modifiers=tuple(modifiers),
        cosmic_ray_terms=tuple(cosmic_ray_terms),
        photo_reactants=tuple(reactions[position][1][0] for position, _, _, _ in parsed["photo"]),
        has_photo_bins=photo_bins is not None,
        shielding=shielding or "",
        shield_habing_dust=bool(shield_habing_dust),
        column_density_method=column_density_method,
        cooling_floor_temperature=float(floor_temperature),
        metal_table_without_lookup=without_lookup,
        radiation_user_names=radiation_user_names,
        photoelectric_neutral_only=bool(photoelectric_neutral_only),
        free_free_charges=(
            tuple(_free_free_charge_squared(name) for name in species) if free_free_cooling else ()
        ),
    )
    return network, species
