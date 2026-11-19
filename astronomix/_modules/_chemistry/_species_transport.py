"""
Transport of the chemical species in the finite-difference solver.

The chemistry module carries the species as number densities ``n_i`` in a block
of the state (``registered_variables.chemistry_species_index``). The
finite-volume solver moves them with a flux of their own. The finite-difference
solver has no such flux: its WENO flux is characteristic-wise, and a species in
it would mean extending the eigenstructure in the native kernel and in the
Pallas kernels. The species are moved here instead, operator-split, with the
machinery that already moves the passive scalars of that solver
(``_fluid_equations/_passive_scalars.py``).

**What is moved.** Without reactions the abundance per unit mass of a species,

    X_i = n_i / rho,        dX_i/dt + v . grad X_i = 0,

is a label of the fluid parcel, which is what a passive scalar is. One step is:

1. ``X_i = n_i / rho`` with the density at the start of the step;
2. the ``X_i`` are advected with the flow at the start of the step, fifth-order
   WENO in space and SSP-RK3 in time, sub-cycled as the flow requires
   (:func:`advect_passive_scalars`). Every ``X_i`` obeys a maximum principle;
3. the hydro update of the step runs on the state without the species;
4. ``n_i = X_i rho`` with the density at the end of the step.

Gas that a source term adds within the step (a wind, a feedback region) takes
the composition of the cell it is added to, because the density of step 4 is the
one after the sources. The transport is not in flux form: the total number of a
species in the box is conserved to the accuracy of the scheme, not to round-off.

**The conserved totals.** The atoms of an element per unit mass,
``A_e = sum_i a_ei X_i``, and the charge per unit mass, ``Q = sum_i q_i X_i``,
are labels of the parcel too, and no reaction changes them. A linear scheme
would keep them: the sum of the advected ``X_i`` would be the advected sum. WENO
is not linear. Each ``X_i`` gets its own nonlinear weights, so where three or
more species of an element have a front in the same cells (He, He+ and He++ at
the edge of hot gas) the atoms no longer add up, by a little every step and
always at the same place. (For two species the weights agree and the sum is
kept; that is why the simplest test does not show the problem.) Therefore
``A_e`` and ``Q`` are advected as further scalars, and afterwards

* the species of every element are scaled so that its atoms are the advected
  ``A_e`` again. This is exact when every species holds one element (H, He,
  C+, ...). A species of several elements (CO) couples the factors of its
  elements; the elements are then swept ``ELEMENT_SWEEPS`` times, and what is
  left of the mismatch is about 1/300 of what the advection made;
* the electrons take up the charge: ``X_E = sum_(i != E) q_i X_i - Q``.

The totals of gas of one composition stay exactly what they were, and a
gradient of the metallicity is moved like any other label. Nothing here assumes
that the species add up to the density. ``ChemistryConfig.conserve_elements_in_transport``
switches the restoration off; it needs the composition of the species
(``species_element_matrix``, ``species_charges``), which
``build_chemistry_from_network_file`` reads from their names.
"""

# general
import numpy as np

# jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._passive_scalars import advect_passive_scalars

#: Sweeps over the elements when a species holds several of them. One sweep
#: makes the atoms of the element swept last exact and reduces the mismatch of
#: the others by about a factor of four (measured on species of comparable
#: abundance, the worst case), so four sweeps leave about 1/300 of it.
ELEMENT_SWEEPS = 4


def _composition(chemistry_config):
    """The static composition of the species: ``(element_matrix, charges)`` as
    numpy arrays, each ``None`` where the configuration does not give it or the
    restoration is switched off."""
    if not chemistry_config.conserve_elements_in_transport:
        return None, None
    number_of_species = chemistry_config.number_of_chemical_species
    element_matrix = np.asarray(chemistry_config.species_element_matrix, dtype=np.int64)
    if element_matrix.size == 0:
        element_matrix = None
    elif element_matrix.shape[1] != number_of_species:
        raise ValueError(
            "ChemistryConfig.species_element_matrix has "
            f"{element_matrix.shape[1]} columns for {number_of_species} species."
        )
    charges = np.asarray(chemistry_config.species_charges, dtype=np.int64)
    electron = chemistry_config.electron_index
    if charges.size == 0 or electron < 0:
        # without an electron species there is nothing to take the charge up
        charges = None
    elif charges.shape[0] != number_of_species:
        raise ValueError(
            "ChemistryConfig.species_charges has "
            f"{charges.shape[0]} entries for {number_of_species} species."
        )
    return element_matrix, charges


def conserved_totals(abundances, chemistry_config):
    """
    The atoms of every element and the charge carried by the species.

    Args:
        abundances: The species, shape ``(num_species,) + grid``; number
            densities or abundances per unit mass.
        chemistry_config: The chemistry configuration (composition of the species).

    Returns:
        ``(element_totals, charge_total)``: the atoms of every element, shape
        ``(num_elements,) + grid``, and the charge, shape ``grid``, in the units
        of ``abundances``; each ``None`` where the composition is not known.
    """
    element_matrix, charges = _composition(chemistry_config)
    element_totals = None
    charge_total = None
    if element_matrix is not None:
        element_totals = jnp.tensordot(
            jnp.asarray(element_matrix, dtype=abundances.dtype), abundances, axes=1
        )
    if charges is not None:
        charge_total = jnp.tensordot(
            jnp.asarray(charges, dtype=abundances.dtype), abundances, axes=1
        )
    return element_totals, charge_total


def restore_conserved_totals(abundances, element_totals, charge_total, chemistry_config):
    """
    Scale the species so that they carry the given atoms and charge.

    Elements: the species of an element are multiplied by the factor that makes
    its atoms the target (iterative proportional fitting; it keeps every species
    non-negative and a species that is zero stays zero). For species of one
    element each, one sweep over the elements is exact. A species of several
    elements couples the factors; the elements are then swept
    ``ELEMENT_SWEEPS`` times, after which the element swept last is exact and
    the others are close. An element with no atoms in a cell is left alone.
    Charge: the electrons are set to what the other species and the target
    leave, not below zero.

    Args:
        abundances: The species, shape ``(num_species,) + grid``.
        element_totals: The target atoms of every element, shape
            ``(num_elements,) + grid``, or ``None``.
        charge_total: The target charge, shape ``grid``, or ``None``.
        chemistry_config: The chemistry configuration (composition of the species).

    Returns:
        The species with the totals restored, same shape as ``abundances``.
    """
    element_matrix, charges = _composition(chemistry_config)
    grid_axes = (1,) * (abundances.ndim - 1)

    if element_matrix is not None and element_totals is not None:
        several_elements = bool(np.any(np.count_nonzero(element_matrix, axis=0) > 1))
        for _ in range(ELEMENT_SWEEPS if several_elements else 1):
            for element, atoms in enumerate(element_matrix):
                if not np.any(atoms > 0):
                    continue
                carried = jnp.tensordot(
                    jnp.asarray(atoms, dtype=abundances.dtype), abundances, axes=1
                )
                factor = jnp.where(
                    carried > 0.0,
                    element_totals[element] / jnp.where(carried > 0.0, carried, 1.0),
                    1.0,
                )
                has_element = jnp.asarray(atoms > 0).reshape((-1,) + grid_axes)
                abundances = jnp.where(has_element, abundances * factor[None, ...], abundances)

    if charges is not None and charge_total is not None:
        electron = chemistry_config.electron_index
        other_charges = charges.copy()
        other_charges[electron] = 0
        ions = jnp.tensordot(
            jnp.asarray(other_charges, dtype=abundances.dtype), abundances, axes=1
        )
        # Q = ions + q_E X_E with q_E = -1 for the electron
        electrons = (charge_total - ions) / float(charges[electron])
        abundances = abundances.at[electron].set(jnp.maximum(electrons, 0.0))

    return abundances


def advect_chemical_species(
    species,
    primitive_state,
    dt,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Advect the chemical species over one hydro step (steps 1 and 2 of the module
    docstring, and the restoration of the conserved totals).

    Args:
        species: The species block at the start of the step, number densities
            of shape ``(num_species,) + grid``.
        primitive_state: The primitive state at the start of the step without
            the species block; its density and velocity are the advecting flow.
        dt: The hydro time step.
        config: The simulation configuration.
        registered_variables: The registered variables of ``primitive_state``.

    Returns:
        The advected abundances per unit mass ``X_i``, same shape as
        ``species``. The caller multiplies them by the density at the end of the
        step (:func:`species_from_abundances`).
    """
    chemistry_config = config.chemistry_config
    number_of_species = species.shape[0]

    density = primitive_state[registered_variables.density_index]
    # The hydro keeps the density positive; the floor only keeps the quotient
    # finite in a cell that broke that.
    abundances = species / jnp.maximum(density, jnp.finfo(density.dtype).tiny)[None, ...]

    # The conserved totals ride along as further scalars.
    element_totals, charge_total = conserved_totals(abundances, chemistry_config)
    scalars = [abundances]
    bounds = [(0.0, float("inf"))] * number_of_species
    number_of_elements = 0
    if element_totals is not None:
        number_of_elements = element_totals.shape[0]
        scalars.append(element_totals)
        bounds += [(0.0, float("inf"))] * number_of_elements
    if charge_total is not None:
        scalars.append(charge_total[None, ...])
        bounds.append(None)  # a charge has either sign
    scalars = jnp.concatenate(scalars, axis=0) if len(scalars) > 1 else abundances

    # The advection of the passive scalars reads the bounds of its stack and the
    # shock-history layout from the configuration; here the stack is the species
    # and their totals, each non-negative, and no shock history.
    advection_config = config._replace(
        track_shock_history=False,
        passive_scalar_bounds=tuple(bounds),
    )
    advected = advect_passive_scalars(
        scalars,
        primitive_state,
        dt,
        config.grid_spacing,
        advection_config,
        registered_variables,
    )

    abundances = advected[:number_of_species]
    if element_totals is not None:
        element_totals = advected[number_of_species : number_of_species + number_of_elements]
    if charge_total is not None:
        charge_total = advected[number_of_species + number_of_elements]
    return restore_conserved_totals(abundances, element_totals, charge_total, chemistry_config)


def species_from_abundances(abundances, density):
    """The number densities ``n_i = X_i rho`` (step 4 of the module docstring)."""
    return abundances * density[None, ...]
