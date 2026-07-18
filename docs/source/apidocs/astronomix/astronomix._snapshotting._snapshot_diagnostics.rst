:orphan:

:py:mod:`astronomix._snapshotting._snapshot_diagnostics`
========================================================

.. py:module:: astronomix._snapshotting._snapshot_diagnostics

.. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SnapshotQuantity <astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity>`
     - .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`spectra_requested <astronomix._snapshotting._snapshot_diagnostics.spectra_requested>`
     - .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.spectra_requested
          :summary:
   * - :py:obj:`spectral_wavenumbers <astronomix._snapshotting._snapshot_diagnostics.spectral_wavenumbers>`
     - .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.spectral_wavenumbers
          :summary:
   * - :py:obj:`enabled_snapshot_quantities <astronomix._snapshotting._snapshot_diagnostics.enabled_snapshot_quantities>`
     - .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.enabled_snapshot_quantities
          :summary:
   * - :py:obj:`build_snapshot_store <astronomix._snapshotting._snapshot_diagnostics.build_snapshot_store>`
     - .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.build_snapshot_store
          :summary:
   * - :py:obj:`record_snapshot <astronomix._snapshotting._snapshot_diagnostics.record_snapshot>`
     - .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.record_snapshot
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SNAPSHOT_QUANTITIES <astronomix._snapshotting._snapshot_diagnostics.SNAPSHOT_QUANTITIES>`
     - .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.SNAPSHOT_QUANTITIES
          :summary:

API
~~~

.. py:class:: SnapshotQuantity
   :canonical: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity

   .. py:attribute:: field
      :canonical: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity.field
      :type: str
      :value: None

      .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity.field

   .. py:attribute:: is_enabled
      :canonical: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity.is_enabled
      :type: typing.Callable
      :value: None

      .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity.is_enabled

   .. py:attribute:: per_snapshot_shape
      :canonical: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity.per_snapshot_shape
      :type: typing.Callable
      :value: None

      .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity.per_snapshot_shape

   .. py:attribute:: compute
      :canonical: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity.compute
      :type: typing.Callable
      :value: None

      .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.SnapshotQuantity.compute

.. py:function:: spectra_requested(config) -> bool
   :canonical: astronomix._snapshotting._snapshot_diagnostics.spectra_requested

   .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.spectra_requested

.. py:function:: spectral_wavenumbers(config)
   :canonical: astronomix._snapshotting._snapshot_diagnostics.spectral_wavenumbers

   .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.spectral_wavenumbers

.. py:data:: SNAPSHOT_QUANTITIES
   :canonical: astronomix._snapshotting._snapshot_diagnostics.SNAPSHOT_QUANTITIES
   :value: ()

   .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.SNAPSHOT_QUANTITIES

.. py:function:: enabled_snapshot_quantities(config)
   :canonical: astronomix._snapshotting._snapshot_diagnostics.enabled_snapshot_quantities

   .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.enabled_snapshot_quantities

.. py:function:: build_snapshot_store(config, number_of_snapshots, state_shape) -> astronomix.data_classes.simulation_snapshot_data.SnapshotData
   :canonical: astronomix._snapshotting._snapshot_diagnostics.build_snapshot_store

   .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.build_snapshot_store

.. py:function:: record_snapshot(store, snapshot_index, time, primitive_state, helper_data, params, config, registered_variables) -> astronomix.data_classes.simulation_snapshot_data.SnapshotData
   :canonical: astronomix._snapshotting._snapshot_diagnostics.record_snapshot

   .. autodoc2-docstring:: astronomix._snapshotting._snapshot_diagnostics.record_snapshot
