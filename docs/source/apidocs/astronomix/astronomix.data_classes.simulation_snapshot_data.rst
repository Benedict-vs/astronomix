:py:mod:`astronomix.data_classes.simulation_snapshot_data`
==========================================================

.. py:module:: astronomix.data_classes.simulation_snapshot_data

.. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SnapshotData <astronomix.data_classes.simulation_snapshot_data.SnapshotData>`
     - .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData
          :summary:

API
~~~

.. py:class:: SnapshotData
   :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData

   .. py:attribute:: time_points
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.time_points
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.time_points

   .. py:attribute:: states
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.states
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.states

   .. py:attribute:: final_state
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.final_state
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.final_state

   .. py:attribute:: total_mass
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.total_mass
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.total_mass

   .. py:attribute:: total_energy
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.total_energy
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.total_energy

   .. py:attribute:: internal_energy
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.internal_energy
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.internal_energy

   .. py:attribute:: kinetic_energy
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.kinetic_energy
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.kinetic_energy

   .. py:attribute:: gravitational_energy
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.gravitational_energy
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.gravitational_energy

   .. py:attribute:: radial_momentum
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.radial_momentum
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.radial_momentum

   .. py:attribute:: magnetic_divergence
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.magnetic_divergence
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.magnetic_divergence

   .. py:attribute:: kinetic_energy_spectrum
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.kinetic_energy_spectrum
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.kinetic_energy_spectrum

   .. py:attribute:: magnetic_energy_spectrum
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.magnetic_energy_spectrum
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.magnetic_energy_spectrum

   .. py:attribute:: helicity_spectrum
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.helicity_spectrum
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.helicity_spectrum

   .. py:attribute:: k_spectra
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.k_spectra
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.k_spectra

   .. py:attribute:: temperature_pdf
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.temperature_pdf
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.temperature_pdf

   .. py:attribute:: runtime
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.runtime
      :type: float
      :value: 0.0

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.runtime

   .. py:attribute:: num_iterations
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.num_iterations
      :type: int
      :value: 0

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.num_iterations

   .. py:attribute:: temporary_memory_bytes
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.temporary_memory_bytes
      :type: int
      :value: 0

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.temporary_memory_bytes

   .. py:attribute:: argument_memory_bytes
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.argument_memory_bytes
      :type: int
      :value: 0

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.argument_memory_bytes

   .. py:attribute:: total_memory_bytes
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.total_memory_bytes
      :type: int
      :value: 0

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.total_memory_bytes

   .. py:attribute:: current_checkpoint
      :canonical: astronomix.data_classes.simulation_snapshot_data.SnapshotData.current_checkpoint
      :type: int
      :value: 0

      .. autodoc2-docstring:: astronomix.data_classes.simulation_snapshot_data.SnapshotData.current_checkpoint
