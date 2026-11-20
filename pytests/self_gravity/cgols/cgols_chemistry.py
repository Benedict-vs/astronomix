"""
Non-equilibrium chemistry of a CGOLS run (``CGOLS_CHEMISTRY=1`` in ``cgols.py``).

The gas carries the species of a reaction network in the KROME format
(``astronomix/_modules/_chemistry``; developed by Lorenzo Branca). Every hydro
step the species are moved with the gas by the finite-difference solver
(``_chemistry/_species_transport.py``) and reacted in every cell together with
the temperature, so the network supplies the heating and cooling: the run takes
no cooling curve. This file holds what is specific to CGOLS:

* :func:`build_chemistry` - the chemistry containers in the code units of
  ``cgols.py``, with the per-cell implicit solver (the gradsolve kernel);
* :func:`equilibrium_table` and :func:`equilibrium_species` - the initial
  composition: every cell starts in ionisation equilibrium at its own density
  and temperature;
* :func:`frame_products` - what a snapshot frame stores of the chemistry.

``cgols.py`` imports this file only when the chemistry is on, so a run without
it needs none of the chemistry packages (diffrax, gradsolve, warp-lang).

The network file and the ``data`` folder of a KROME installation are not part of
this repository (KROME is GPL): ``CGOLS_CHEM_NETWORK`` and
``CGOLS_CHEM_KROME_DATA`` point at them.
"""

import hashlib
import os
import re
import warnings
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from astronomix._modules._chemistry._krome_photo import draine_intensity, photo_spectrum
from astronomix._modules._chemistry._species_transport import restore_conserved_totals
from astronomix._modules._chemistry.chemistry_options import KVAERNO5, SOURCE_COMPOSITION_FIXED
from astronomix.setup_helpers import (
    attach_per_cell_kernel,
    build_chemistry_from_network_file,
    equilibrium_abundances,
)

#: Nuclei per hydrogen nucleus of the gas: helium, and carbon, oxygen and
#: silicon in the gas phase of the solar neighbourhood (the values of KROME's
#: tests and of ``examples/chemistry_krome_network.py``).
ABUNDANCES = {"H": 1.0, "He": 8.5e-2, "C": 1.4e-4, "O": 3.2e-4, "Si": 1.8e-6}

#: The species each element starts in for the equilibrium integration (the
#: equilibrium does not depend on it; half-ionised hydrogen lets ionisation and
#: recombination both run from the first step). Per nucleus of the element.
_START = {
    "H": {"H": 0.5, "H+": 0.5},
    "He": {"He": 1.0},
    "C": {"C+": 1.0},
    "O": {"O": 1.0},
    "Si": {"Si+": 1.0},
}

#: The grid of the equilibrium table: log10 of the temperature [K] and of the
#: hydrogen nuclei [cm^-3]. A CGOLS box spans 10 K to a few 1e8 K and 1e-6 to a
#: few 1e2 cm^-3; outside the grid the edge value is used. The temperatures are
#: offset by 0.004 dex from the round values: reactions of a KROME network
#: switch on and off at temperature limits such as 100 K, 1e4 K and 1e6 K, and
#: at a node that sits exactly on a limit rounding would decide which side the
#: equilibrium is computed on (seen for Si+ / Si++ at 1e6 K).
TABLE_LOG_TEMPERATURE = np.linspace(1.004, 9.004, 81)
TABLE_LOG_HYDROGEN = np.linspace(-7.0, 4.0, 23)

#: Bump when the set-up below changes what the equilibrium table holds, so that
#: a cached table is not reused.
_TABLE_VERSION = 2

PROTON_MASS_CGS = 1.67262192e-24
#: The energy at which the Lyman-Werner flux of the Solomon process is taken (Glover & Jappsen 2007; Bovino et al.
#: 2016, eq. 4).
LYMAN_WERNER_ENERGY_EV = 12.87


class ChemistrySetup(NamedTuple):
    """The chemistry of a CGOLS run.

    Attributes:
        config: The chemistry configuration (static part).
        params: The chemistry parameters.
        species: The names of the species, in the order of the state.
        start_abundances: The abundances per hydrogen nucleus the equilibrium
            integration starts from, shape ``[species]``.
        mass_per_hydrogen: The mass of the gas per hydrogen nucleus, in proton
            masses (about 1.35).
        key: A hash of everything that defines the network and its parameters
            (the per-cell kernel is cached under it).
        equilibrium_key: A hash of what the ionisation equilibrium depends on:
            the network, the abundances, the radiation field and the cosmic
            rays, not the cooling.
    """

    config: object
    params: object
    species: tuple
    start_abundances: np.ndarray
    mass_per_hydrogen: float
    key: str
    equilibrium_key: str


def build_chemistry(
    network_path,
    krome_data_folder,
    *,
    number_density_unit_cgs,
    temperature_unit_kelvin,
    time_unit_seconds,
    adiabatic_index,
    solver="kernel",
    relative_tolerance=1e-4,
    absolute_tolerance=1e-12,
    max_steps=8192,
    cosmic_ray_rate=3e-17,
    draine_field=1.0,
    step_target=0.0,
    cooling_courant=0.0,
    floor_temperature=10.0,
    reaction_chunk_size=0,
    free_free=True,
    first_step_fraction=1.0,
    radiation_field_per_cell=False,
    radiation_attenuation=0,
    radiation_disk_source=False,
    source_composition_fixed=True,
):
    """The chemistry of a CGOLS run in the code units of ``cgols.py``.

    Args:
        network_path: The network file in the KROME format.
        krome_data_folder: The ``data`` folder of a KROME installation (metal
            lines, the metal cooling table, the photo cross sections).
        number_density_unit_cgs: cm^-3 per code number-density unit; the species
            of the state are number densities in this unit (code density over
            the proton mass).
        temperature_unit_kelvin: Kelvin per unit of ``P / n`` in code units.
        time_unit_seconds: Seconds per code time unit.
        adiabatic_index: The adiabatic index of the gas.
        solver: ``"kernel"`` (the per-cell gradsolve kernel) or ``"kvaerno5"``
            (the vmapped diffrax solver; many times slower on a grid).
        relative_tolerance, absolute_tolerance: The tolerances of the solver
            (the absolute one in cm^-3 and K).
        max_steps: The most solver steps a cell may take in one reaction.
        cosmic_ray_rate: The cosmic-ray ionisation rate [s^-1] (KROME's
            ``user_crate``); it acts only if the file has a cosmic-ray block.
        draine_field: The radiation field between 5 and 13.6 eV in units of the
            Draine field, the same in every cell and attenuated by dust.
        step_target: React once the hydro time since the last reaction has
            reached this many code time units; 0 reacts every hydro step.
        cooling_courant: The largest fractional change of the temperature per
            reaction; 0 for no limit.
        floor_temperature: The temperature floor of the reaction [K].
        reaction_chunk_size: Cells per chunk of the vmapped solver (memory).
        free_free: Bremsstrahlung of the ions (KROME's ``-cooling=FF``). The
            wind is 1e7 to 1e8 K, where it is the main cooling.
        first_step_fraction: The first trial step of the kernel in every cell,
            as a fraction of the hydro step. 1 (default): the whole step. Nine
            cells in ten of a CGOLS box are halo gas that hardly changes in a
            step and then take one solver step, not the four that the kernel's
            own default of 0.01 costs every cell (1.6 against 4.4 steps per cell
            on frames of the 256 run, at the same accuracy against a tight
            tolerance); a cell that needs short steps is cut back by a factor
            of up to five per rejected step.
        radiation_field_per_cell: The radiation of every cell is
            ``draine_field`` times the FUV field J / J_0 at that cell, the
            field that scales the photoelectric heating of the cooling-curve
            runs (model B): the starburst as a point source diluted with
            distance and attenuated by dust, plus the disk's own star
            formation. The caller puts the sources into
            ``params.fuv_source_params``. The network then does not attenuate
            the Habing flux by a dust column of its own (the dust between the
            sources and the cell is in the field already), and the network file
            must take the Lyman-Werner flux that dissociates H2 from the user
            variable ``user_myfluxLW`` (see below).
        radiation_attenuation: The attenuation of the point source, one of the
            ``FUV_ATTENUATION_*`` modes of the cooling module.
        radiation_disk_source: Add the field of the disk's own star formation.

    **The Lyman-Werner flux.** KROME's ``react_galaxy_ism`` files dissociate H2
    with ``H2_solomonLW(myflux_eV)``, where ``myflux_eV`` is a constant of the
    file: the metagalactic background of Haardt & Madau (2012) at 12.87 eV, for
    which the rate is 4.4e-13 s^-1. The file says that this must be updated for
    another radiation field (Bovino et al. 2016, eq. 4: the rate is 1.38e9 s^-1
    times the intensity of the field in use at 12.87 eV). A file whose rate is
    ``H2_solomonLW(user_myfluxLW)`` gets that flux from here: the intensity of
    the Draine field at 12.87 eV times ``draine_field`` (5.2e-11 s^-1 for one
    Draine field, without shielding), and it follows the field of the cell with
    ``radiation_field_per_cell``. If the rate also calls ``krome_fshield``, the
    self-shielding of H2 is the function of Wolcott-Green et al. (2011), the
    one Bovino et al. (2016) use with this network, with KROME's default column
    density. A file with the constant keeps the constant.

    Returns:
        A :class:`ChemistrySetup`.
    """
    if solver not in ("kernel", "kvaerno5"):
        raise ValueError(f"solver must be 'kernel' or 'kvaerno5', got {solver!r}")
    for path, what in ((network_path, "network file"), (krome_data_folder, "KROME data folder")):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"chemistry: the {what} {path!r} does not exist (CGOLS_CHEM_NETWORK / "
                "CGOLS_CHEM_KROME_DATA; neither is part of the repository)"
            )

    with open(network_path) as file:
        network_text = "\n".join(line.split("#")[0] for line in file.read().lower().splitlines())
    # H2 dissociated by the Lyman-Werner flux of the radiation field in use, and with self-shielding?
    lyman_werner_from_field = re.search(r"h2_solomonlw\(\s*user_myfluxlw\s*\)", network_text) is not None
    self_shielding = "WG11" if "krome_fshield" in network_text else None
    if radiation_field_per_cell and not lyman_werner_from_field:
        raise ValueError(
            f"chemistry: {network_path} dissociates H2 with a Lyman-Werner flux that is a constant of the file, "
            "whatever the radiation field of the cell. With radiation_field_per_cell the rate of that reaction must "
            "be H2_solomonLW(user_myfluxLW) (times krome_fshield(n,Tgas) for the self-shielding): see the docstring "
            "of cgols_chemistry.build_chemistry"
        )
    lyman_werner = {}
    if lyman_werner_from_field:
        lyman_werner = dict(
            krome_user_variables={"user_myfluxLW": float(draine_field) * float(draine_intensity(LYMAN_WERNER_ENERGY_EV))},
            krome_radiation_user_variables=("user_myfluxLW",),
        )

    with warnings.catch_warnings():
        # lines that KROME skips too (react_galaxy_ism_plus has three)
        warnings.simplefilter("ignore", UserWarning)
        chemistry_config, chemistry_params, species = build_chemistry_from_network_file(
            network_path,
            "krome",
            number_density_unit_cgs=number_density_unit_cgs,
            temperature_unit_kelvin=temperature_unit_kelvin,
            time_unit_seconds=time_unit_seconds,
            cosmic_ray_rate=cosmic_ray_rate,
            thermochemistry=True,
            # a limit on the temperature change per reaction slows the cooling
            # of a radiative shock down by the number of steps it then takes
            cooling_courant=cooling_courant if cooling_courant > 0.0 else 1.0e3,
            floor_temperature=floor_temperature,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            max_steps=max_steps,
            reaction_chunk_size=reaction_chunk_size,
            solver=KVAERNO5,
            krome_h2_formation_on_dust=True,
            krome_data_folder=krome_data_folder,
            krome_metal_table=True,
            krome_photo_bins=photo_spectrum("draine", 5.0, 13.6, 10, scale=draine_field),
            # the local dust column of KROME stands in for the dust between the
            # sources and the cell; with the field of the cell that dust is in
            # the field, and counting it here too would count it twice
            krome_shield_habing_dust=not radiation_field_per_cell,
            krome_shielding=self_shielding,
            krome_free_free_cooling=free_free,
            **lyman_werner,
            # all particles of the gas are species of the network
            temperature_from_species=True,
        )
    species = tuple(species)
    chemistry_config = chemistry_config._replace(
        chemistry_step_target=float(step_target),
        radiation_field_per_cell=bool(radiation_field_per_cell),
        radiation_attenuation=int(radiation_attenuation),
        radiation_disk_source=bool(radiation_disk_source),
    )

    missing = [
        name for element in _START.values() for name in element if name not in species
    ]
    if missing or "E" not in species:
        raise ValueError(
            f"chemistry: the network {network_path} lacks the species {missing or ['E']}; the CGOLS "
            "set-up needs H, H+, He, C+, O, Si+ and E (KROME's react_galaxy_ism_plus)"
        )
    start = {name: 1.0e-20 for name in species}
    for element, shares in _START.items():
        for name, share in shares.items():
            start[name] = ABUNDANCES[element] * share
    charge = np.asarray(chemistry_config.species_charges, dtype=np.float64)
    start["E"] = 0.0
    start["E"] = float(sum(q * start[name] for q, name in zip(charge, species) if name != "E"))
    start_abundances = np.asarray([start[name] for name in species])

    masses = np.asarray(chemistry_params.network.masses, dtype=np.float64) / PROTON_MASS_CGS
    mass_per_hydrogen = float(np.dot(masses, start_abundances))

    # The gas the wind injection adds has the element abundances of the run, whatever the cell it is added to
    # holds. (Per unit mass in the units of the state: a species is a number density in units of the code density
    # over the proton mass.) With the composition of the cell instead, an error of the element abundances grows in
    # the injection sphere by 0.65 % per step and never leaves: see ``ChemistryConfig.source_composition``.
    # ``source_composition_fixed=False`` is the behaviour up to 2026-11-20, kept to reproduce it. The composition
    # is set either way: it is also the reference of :func:`element_errors`.
    if source_composition_fixed:
        chemistry_config = chemistry_config._replace(source_composition=SOURCE_COMPOSITION_FIXED)
    chemistry_params = chemistry_params._replace(
        source_abundances=jnp.asarray(start_abundances / mass_per_hydrogen)
    )

    with open(network_path, "rb") as file:
        network_hash = hashlib.sha1(file.read()).hexdigest()
    equilibrium = (
        network_hash, os.path.abspath(krome_data_folder), species, tuple(ABUNDANCES.items()),
        float(cosmic_ray_rate), float(draine_field),
    )  # fmt: skip
    if lyman_werner_from_field or self_shielding or radiation_field_per_cell:
        # only then: the keys of a run without any of these stay what they were
        equilibrium += (bool(lyman_werner_from_field), self_shielding or "", not radiation_field_per_cell)
    equilibrium_key = hashlib.sha1(repr(equilibrium).encode()).hexdigest()
    key = hashlib.sha1(
        repr(
            equilibrium
            + (
                float(number_density_unit_cgs), float(temperature_unit_kelvin), float(time_unit_seconds),
                float(adiabatic_index), float(floor_temperature), float(cooling_courant), bool(free_free),
                float(first_step_fraction),
            )
            + ((True,) if radiation_field_per_cell else ())
        ).encode()
    ).hexdigest()  # fmt: skip

    if solver == "kernel":
        chemistry_config = attach_per_cell_kernel(
            chemistry_config,
            chemistry_params,
            adiabatic_index,
            cache_key=("cgols", key),
            first_step_fraction=first_step_fraction,
        )

    return ChemistrySetup(
        config=chemistry_config,
        params=chemistry_params,
        species=species,
        start_abundances=start_abundances,
        mass_per_hydrogen=mass_per_hydrogen,
        key=key,
        equilibrium_key=equilibrium_key,
    )


# -----------------------------------------------------------------------------
# ======================== ↓ Initial composition ↓ =============================
# -----------------------------------------------------------------------------

def equilibrium_table(setup, cache_path=None, verbose=print):
    """Ionisation equilibrium of the network on a grid of temperature and density.

    Every point is held at its temperature and density of hydrogen nuclei for
    10 Gyr (``equilibrium_abundances``). The table depends on the network, the
    abundances and the radiation and cosmic-ray parameters only, so it is
    computed once and kept in ``cache_path`` (an ``.npz`` file).

    Args:
        setup: The chemistry of the run.
        cache_path: Where the table is kept; ``None`` computes it every time.
        verbose: Called with one line of progress (``None``: silent).

    Returns:
        ``log10`` of the abundances per hydrogen nucleus, a numpy array of
        shape ``[species, density, temperature]`` on ``TABLE_LOG_HYDROGEN`` and
        ``TABLE_LOG_TEMPERATURE``.
    """
    say = verbose if verbose is not None else (lambda *args, **kwargs: None)
    table_key = hashlib.sha1(
        repr(
            (setup.equilibrium_key, _TABLE_VERSION, TABLE_LOG_TEMPERATURE.tolist(), TABLE_LOG_HYDROGEN.tolist())
        ).encode()
    ).hexdigest()
    if cache_path is not None and os.path.exists(cache_path):
        with np.load(cache_path) as cached:
            if str(cached["key"]) == table_key:
                say(f"[chemistry] equilibrium table read from {cache_path}")
                return cached["log_abundances"]
        say(f"[chemistry] {cache_path} belongs to another set-up: recomputing")

    say(
        f"[chemistry] computing the equilibrium table ({len(TABLE_LOG_HYDROGEN)} densities x "
        f"{len(TABLE_LOG_TEMPERATURE)} temperatures, held for 10 Gyr each)"
    )
    grid_hydrogen, grid_temperature = (
        grid.ravel()
        for grid in np.meshgrid(10.0**TABLE_LOG_HYDROGEN, 10.0**TABLE_LOG_TEMPERATURE, indexing="ij")
    )
    abundances, last_change = equilibrium_abundances(
        setup.config, setup.params, grid_hydrogen, grid_temperature, setup.start_abundances,
        return_last_change=True,
    )
    if not np.isfinite(abundances).all():
        raise RuntimeError(
            "chemistry: the equilibrium integration left non-finite abundances in "
            f"{int((~np.isfinite(abundances).all(axis=0)).sum())} of {abundances.shape[1]} table points"
        )
    # Has every point settled? Thin, cold gas recombines more slowly than the
    # hold lasts and keeps a memory of the start; a CGOLS box has none of it
    # (its thin gas is hot), so this is reported, not refused.
    unsettled = last_change > 0.01
    if unsettled.any():
        say(
            f"[chemistry] {int(unsettled.sum())} of {last_change.size} table points still changed by more than 0.01 dex "
            f"over the last 5.9 Gyr (largest {last_change.max():.2f} dex): all with n_H <= "
            f"{grid_hydrogen[unsettled].max():.1e} cm^-3 and T <= {grid_temperature[unsettled].max():.1e} K"
        )
    else:
        say(f"[chemistry] every table point has settled (largest change over the last 5.9 Gyr: {last_change.max():.1e} dex)")
    log_abundances = np.log10(np.maximum(abundances, 1.0e-30)).reshape(
        len(setup.species), len(TABLE_LOG_HYDROGEN), len(TABLE_LOG_TEMPERATURE)
    )
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(
            cache_path,
            key=table_key,
            log_abundances=log_abundances,
            last_change_dex=last_change.reshape(len(TABLE_LOG_HYDROGEN), len(TABLE_LOG_TEMPERATURE)),
            log_hydrogen=TABLE_LOG_HYDROGEN,
            log_temperature=TABLE_LOG_TEMPERATURE,
            species=np.asarray(setup.species),
        )
        say(f"[chemistry] equilibrium table written to {cache_path}")
    return log_abundances


def _bilinear(table, log_hydrogen, log_temperature):
    """Bilinear interpolation of ``table[density, temperature]`` on the grid of
    the equilibrium table; outside the grid the edge value."""
    def position(values, grid):
        spacing = grid[1] - grid[0]
        scaled = jnp.clip((values - grid[0]) / spacing, 0.0, len(grid) - 1.0)
        lower = jnp.clip(jnp.floor(scaled).astype(jnp.int32), 0, len(grid) - 2)
        return lower, scaled - lower

    i, u = position(log_hydrogen, TABLE_LOG_HYDROGEN)
    j, v = position(log_temperature, TABLE_LOG_TEMPERATURE)
    return (
        (1.0 - u) * (1.0 - v) * table[i, j]
        + u * (1.0 - v) * table[i + 1, j]
        + (1.0 - u) * v * table[i, j + 1]
        + u * v * table[i + 1, j + 1]
    )


def equilibrium_species(density, pressure, log_abundances, setup):
    """
    The species of gas in ionisation equilibrium at its own density and temperature.

    The temperature of a cell is ``T = P / (k n)`` with ``n`` the particles of
    the network, and those depend on the ionisation, hence on the temperature:
    neutral disk gas has half the particles of ionised gas, so at a given
    pressure it is twice as hot. ``T`` is therefore found by bisection of
    ``T S(T, n_H) = P / (k n_H)``, where ``S`` is the number of particles per
    hydrogen nucleus of the equilibrium table (``T S`` rises with ``T``). The
    interpolated abundances are then scaled so that every element has exactly
    its abundance and the gas is neutral.

    Pure and jittable (it runs inside the sharded build of the initial state).

    Args:
        density: The density in code units.
        pressure: The pressure in code units.
        log_abundances: The equilibrium table (:func:`equilibrium_table`).
        setup: The chemistry of the run.

    Returns:
        ``(species, temperature)``: the species block in code number densities,
        shape ``(num_species,) + grid``, and the temperature [K].
    """
    params = setup.params
    log_abundances = jnp.asarray(log_abundances, dtype=density.dtype)
    hydrogen = density / setup.mass_per_hydrogen  # hydrogen nuclei, code units
    log_hydrogen = jnp.log10(hydrogen * params.number_density_unit_cgs)
    # T S = P / n_H in Kelvin
    target = jnp.log10(pressure / hydrogen * params.temperature_unit_kelvin)
    log_particles = jnp.log10(jnp.sum(10.0**log_abundances, axis=0))

    def bisect(_, bounds):
        lower, upper = bounds
        middle = 0.5 * (lower + upper)
        too_cold = middle + _bilinear(log_particles, log_hydrogen, middle) < target
        return jnp.where(too_cold, middle, lower), jnp.where(too_cold, upper, middle)

    lower, upper = jax.lax.fori_loop(
        0,
        48,
        bisect,
        (
            jnp.full_like(density, TABLE_LOG_TEMPERATURE[0]),
            jnp.full_like(density, TABLE_LOG_TEMPERATURE[-1]),
        ),
    )
    log_temperature = 0.5 * (lower + upper)

    abundances = jnp.stack(
        [
            10.0 ** _bilinear(log_abundances[index], log_hydrogen, log_temperature)
            for index in range(len(setup.species))
        ]
    )
    # exactly the abundance of every element, and neutral
    grid_axes = (1,) * density.ndim
    element_totals = jnp.asarray(
        [ABUNDANCES[element] for element in setup.config.species_element_names], dtype=density.dtype
    ).reshape((-1,) + grid_axes) * jnp.ones_like(density)[None, ...]
    abundances = restore_conserved_totals(
        abundances, element_totals, jnp.zeros_like(density), setup.config
    )
    species = abundances * hydrogen[None, ...]
    temperature = pressure / jnp.sum(species, axis=0) * params.temperature_unit_kelvin
    return species, temperature


# -----------------------------------------------------------------------------
# ======================== ↑ Initial composition ↑ =============================
# -----------------------------------------------------------------------------


def frame_name(species_name):
    """The key of a species in a frame file: ``H+`` -> ``x_Hp_xz``, ``H-`` -> ``x_Hm_xz``."""
    return "x_" + species_name.replace("+", "p").replace("-", "m") + "_xz"


def element_errors(species, density, setup):
    """
    How far the element content of the worst cell is from the composition of the run.

    The atoms of an element per unit mass are the same in every cell of a run
    that starts with one composition and whose sources add gas of that
    composition: no reaction and no transport changes them. The largest
    relative deviation over the cells is therefore a check of the whole
    chemistry coupling that costs one reduction: round-off (1e-13) in a sound
    run. It was 1e-2 after 40 Myr, and 1 after 45 Myr, in the run of 2026-11-19
    whose injected gas copied the composition of its cell
    (``ChemistryConfig.source_composition``).

    Args:
        species: The species, shape ``(num_species,) + grid``, code number densities.
        density: The density, shape ``grid``, code units.
        setup: The chemistry of the run.

    Returns:
        The largest ``|atoms per unit mass / reference - 1|`` of every element,
        in the order of ``setup.config.species_element_names``.
    """
    elements = jnp.asarray(setup.config.species_element_matrix, dtype=species.dtype)
    reference = elements @ jnp.asarray(setup.params.source_abundances, dtype=species.dtype)
    atoms = jnp.tensordot(elements, species, axes=1) / density[None, ...]
    grid_axes = tuple(range(1, atoms.ndim))
    return jnp.max(jnp.abs(atoms / reference.reshape((-1,) + (1,) * len(grid_axes)) - 1.0), axis=grid_axes)


def frame_products(species_plane, pressure_plane, setup):
    """
    What a frame stores of the chemistry on the edge-on plane.

    Args:
        species_plane: The species on the plane, shape ``(num_species, x, z)``,
            code number densities.
        pressure_plane: The pressure on the plane, code units.
        setup: The chemistry of the run.

    Returns:
        ``(temperature, products)``: the temperature [K] from the particles of
        the network, and a dictionary with the abundance per hydrogen nucleus of
        every species (:func:`frame_name`) and ``n_H_xz``, the hydrogen nuclei
        per cm^3.
    """
    particles = jnp.maximum(jnp.sum(jnp.maximum(species_plane, 0.0), axis=0), 1.0e-300)
    temperature = pressure_plane / particles * setup.params.temperature_unit_kelvin
    hydrogen_row = setup.config.species_element_names.index("H")
    hydrogen_atoms = jnp.asarray(
        setup.config.species_element_matrix[hydrogen_row], dtype=species_plane.dtype
    )
    hydrogen = jnp.maximum(jnp.tensordot(hydrogen_atoms, species_plane, axes=1), 1.0e-300)
    products = {"n_H_xz": hydrogen * setup.params.number_density_unit_cgs}
    for index, name in enumerate(setup.species):
        # float32 on disk: keep the trace species above its smallest number
        products[frame_name(name)] = jnp.maximum(species_plane[index] / hydrogen, 1.0e-30)
    return temperature, products
