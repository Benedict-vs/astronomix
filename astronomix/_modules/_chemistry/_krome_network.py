"""
Reaction networks written in the KROME format.

KROME (Grassi et al. 2014) describes a network as a text file of reactions whose
rate coefficients are Fortran expressions of the gas temperature. This module
reads such a file and builds its right-hand side as a JAX function with the call
signature of a carbox network, so that the chemistry module integrates it with
the same implicit solvers (the vmapped Diffrax Kvaerno5 and the per-cell Warp
kernel) and the same thermochemistry.

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
  rates by integrating cross sections over a radiation spectrum set at run
  time; here every photo-reaction takes its rate [1/s] from ``photo_rates``
  (zero by default, i.e. no ionising radiation), and the cross sections are not
  read.

A rate coefficient is zero outside its temperature limits, as in KROME. Any other
directive of the format raises an error instead of being skipped.

Processes that KROME adds at build time are not in the network file. One of them
is available here as an option, the formation of H2 on dust grains with the rate
of Jura (1975) (KROME's ``-useH2dust=JURA``): ``3.5e-17 n(H) n_H D/D_sun`` H2
molecules per cm^3 per second.

The heating and cooling of a KROME network are KROME's too (see
``_krome_thermochemistry.py``): the network carries its own temperature
derivative, which the chemistry module uses in place of its default one.
"""

# general
import ast
import re
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
IGNORED_DIRECTIVES = ("@notab_start", "@notab_stop", "@notabnext", "@cr_start", "@cr_stop")

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
        in file order) and ``commons`` (the ``@common`` user variables).
    """
    species, reactions, variables, commons = [], [], [], []
    layout, photo, photo_count = DEFAULT_FORMAT, False, 0

    def index(token):
        name = _species_name(token)
        if name not in species:
            species.append(name)
        return species.index(name)

    for number, raw in enumerate(open(path), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith("@var:"):
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
        elif lowered.startswith(IGNORED_DIRECTIVES):
            continue  # tabulation hints and cosmic-ray block markers: no effect on the rates
        elif line.startswith("@"):
            raise NotImplementedError(f"{path}:{number}: KROME directive not supported: {line.split(':')[0]}")
        else:
            fields = [field.strip() for field in line.split(",", len(layout) - 1)]
            if len(fields) != len(layout):
                raise ValueError(
                    f"{path}:{number}: the line '{line[:60]}' has {len(fields)} fields, the format in force "
                    f"({','.join(layout)}) has {len(layout)}. KROME skips such a reaction with a warning; "
                    "a @format line is probably missing or commented out above it."
                )
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
            reactions.append((
                int(entry["idx"]), tuple(entry["r"]), tuple(entry["p"]),
                None if photo else _temperature_limit(entry["tmin"], "GE"),
                None if photo else _temperature_limit(entry["tmax"], "LT"),
                entry["rate"], slot,
            ))

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

    return {"species": tuple(species), "reactions": tuple(reactions), "variables": tuple(variables), "commons": tuple(commons)}


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
    KROME's ``user_crate`` and ``fuv_field`` its Habing field ``GHabing`` (it
    enters the recombination on dust); the visual extinction is not used.
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

    def get_rates(self, temperature, cosmic_ray_rate, fuv_field, visual_extinction, abundances):
        """Rate coefficients of every reaction, in the order of the file (cm^3/s or 1/s)."""
        temperature = jnp.clip(temperature, MINIMUM_TEMPERATURE_KELVIN, MAXIMUM_TEMPERATURE_KELVIN)
        electron_density = abundances[self.species.index("E")] if "E" in self.species else 0.0
        functions = {
            "exp": jnp.exp, "dexp": jnp.exp, "log": jnp.log, "dlog": jnp.log, "log10": jnp.log10, "dlog10": jnp.log10,
            "sqrt": jnp.sqrt, "dsqrt": jnp.sqrt, "abs": jnp.abs, "max": jnp.maximum, "min": jnp.minimum,
            "h2_solomonlw": lambda flux: 1.38e9 * flux * ELECTRON_VOLT_TO_ERG,
        }
        for ion in DUST_RECOMBINATION:
            functions[f"{ion}_recombination_on_dust"] = (
                lambda n, t, ion=ion: _dust_recombination(
                    ion, electron_density, t, fuv_field, self.dust_to_gas, self.single_precision_literals
                )
            )
        functions["o_recombination_on_dust"] = lambda n, t: 0.25 * functions["h_recombination_on_dust"](n, t)
        users = {"user_crate": cosmic_ray_rate, "n": abundances}
        users.update({name: self.user_values[i] for i, name in enumerate(self.user_names)})

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
                rates.append(self.photo_rates[photo_slot])
                continue
            value = _evaluate(_syntax_tree(expression, self.single_precision_literals), names_at(lower, upper), functions)
            for limit in (lower, upper):
                if limit is not None:
                    value = jnp.where(compare[limit[0]](temperature, limit[1]), value, 0.0)
            rates.append(value * jnp.ones_like(temperature))
        return jnp.stack(rates)

    def __call__(self, time, abundances, temperature, cosmic_ray_rate, fuv_field, visual_extinction, rate_modifier_a, rate_modifier_b):
        rates = self.get_rates(temperature, cosmic_ray_rate, fuv_field, visual_extinction, abundances)
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

    def heating_and_cooling_terms(self, abundances, temperature, cosmic_ray_rate, fuv_field):
        """The heating and cooling terms [erg cm^-3 s^-1], by name (see ``_krome_thermochemistry.py``)."""
        from astronomix._modules._chemistry import _krome_thermochemistry as thermo
        from astronomix._modules._chemistry._thermochemistry import _atomic_oxygen_cooling, _ionized_carbon_cooling

        temperature = jnp.clip(temperature, 3.0, 1.0e9)
        n = jnp.maximum(abundances, 0.0)
        index = lambda name: self.species.index(name) if name in self.species else -1
        density = lambda name: n[self.species.index(name)] if name in self.species else 0.0
        electrons, hydrogen, molecular = density("E"), density("H"), density("H2")
        nuclei = jnp.dot(self.hydrogen_nuclei, n)

        terms = {
            "cooling_h2": thermo.h2_line_cooling(n, temperature, index),
            "cooling_atomic": thermo.atomic_cooling(
                temperature, electrons, hydrogen, density("He"), density("H+"), density("He+"), density("He++")
            ),
            "cooling_compton": thermo.compton_cooling(temperature, electrons),
        }
        if self.metal_lines:
            variables, upper_limit, metals = self.metal_lines
            lines = thermo.metal_line_cooling(
                temperature, density, [dict(metal) for metal in metals], variables, upper_limit,
                _evaluate, _syntax_tree, self.ortho_para_ratio,
            )
            blend = thermo.metal_blend(temperature)
            terms["cooling_metal_lines"] = (1.0 - blend) * lines
            if self.metal_table:
                terms["cooling_metal_table"] = blend * thermo.tabulated_metal_cooling(
                    temperature, nuclei, self.metallicity, *self.metal_table
                )
        else:
            # without KROME's data: the two fine-structure lines of the chemistry module
            terms["cooling_metal_lines"] = _ionized_carbon_cooling(
                n, temperature, index("C+"), index("E"), index("H"), index("H2")
            ) + _atomic_oxygen_cooling(n, temperature, index("O"), index("H"), index("H2"))

        fraction = thermo.thermalised_fraction(temperature, hydrogen, molecular, nuclei)
        rates = self.get_rates(temperature, cosmic_ray_rate, fuv_field, 0.0, n)
        chemical = self.h2_on_dust_coefficient * self.dust_to_gas * hydrogen * nuclei * (4.2 * fraction + 0.2)
        for position, reactants, energy, thermalised in self.chemical_terms:
            flux = rates[position] * energy * (fraction if thermalised else 1.0)
            for i in reactants:
                flux = flux * n[i]
            chemical = chemical + flux
        terms["heating_chemical"] = chemical * thermo.ELECTRON_VOLT_TO_ERG
        terms["heating_photoelectric"] = thermo.photoelectric_heating(
            temperature, electrons, hydrogen, nuclei, fuv_field, self.dust_to_gas
        )
        terms["heating_cosmic_rays"] = cosmic_ray_rate * (5.5e-12 * hydrogen + 2.5e-11 * molecular)
        return terms

    def temperature_derivative(self, abundances, temperature, adiabatic_index, cosmic_ray_rate, fuv_field):
        """dT/dt [K/s] from the heating and cooling of KROME, at constant density."""
        from astronomix._modules._chemistry._krome_thermochemistry import BOLTZMANN_ERG

        terms = self.heating_and_cooling_terms(abundances, temperature, cosmic_ray_rate, fuv_field)
        net = sum(value for name, value in terms.items() if name.startswith("heating")) - sum(
            value for name, value in terms.items() if name.startswith("cooling")
        )
        particles = jnp.maximum(jnp.sum(jnp.maximum(abundances, 0.0)), 1e-30)
        return (adiabatic_index - 1.0) * net / (BOLTZMANN_ERG * particles)


def build_krome_network(
    path, photo_rates=None, user_variables=None, dust_to_gas_relative=1.0, h2_formation_on_dust=False,
    data_folder=None, metal_table=True, metallicity=None, ortho_para_ratio=3.0, single_precision_literals=False,
):
    """Build the right-hand side of a KROME network file.

    Args:
        path: The KROME network file.
        photo_rates: Rates [1/s] of the photo-reactions, as a dictionary from the
            KROME reaction index to the rate; the others are zero.
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
            look a table up; with ``False`` the metals cool through their lines
            alone, i.e. not above 1e4 K, and the kernel can be used.
        metallicity: Metal abundance relative to solar, the factor of the
            tabulated metal cooling (default: ``dust_to_gas_relative``).
        ortho_para_ratio: Ortho-to-para ratio of H2 as a collider of the metal lines.
        single_precision_literals: Round the real constants written without a
            ``d`` exponent to single precision, as a Fortran compiler does (for
            comparisons with KROME to round-off; see ``_syntax_tree``).

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

    user_variables = {name.lower(): value for name, value in (user_variables or {}).items()}
    user_names = tuple(name for name in parsed["commons"] if name != "user_crate")
    unknown = sorted(set(user_variables) - set(user_names))
    if unknown:
        raise ValueError(f"{path}: {unknown} are not @common variables of the network ({list(user_names)})")
    used = " ".join(expression.lower() for _, expression in parsed["variables"]) + " " + " ".join(r[5].lower() for r in reactions)
    missing = [name for name in user_names if name not in user_variables and re.search(rf"\b{name}\b", used)]
    if missing:
        raise ValueError(f"{path}: the rates use the user variables {missing}; give them in user_variables")

    coefficient = 3.5e-17 if h2_formation_on_dust is True else float(h2_formation_on_dust or 0.0)
    change = np.zeros(len(species))
    if coefficient:
        if "H" not in species or "H2" not in species:
            raise ValueError(f"{path}: H2 formation on dust needs the species H and H2")
        change[species.index("H")], change[species.index("H2")] = -2.0, 1.0

    # heating and cooling: metal coolants of KROME's data folder, reactions of the chemical heating
    with_table, metal_lines, metal_table = bool(metal_table), (), ()
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
    )
    return network, species
