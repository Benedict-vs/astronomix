:orphan:

:py:mod:`astronomix.time_stepping._time_loop`
=============================================

.. py:module:: astronomix.time_stepping._time_loop

.. autodoc2-docstring:: astronomix.time_stepping._time_loop
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SnapshotSpec <astronomix.time_stepping._time_loop.SnapshotSpec>`
     - .. autodoc2-docstring:: astronomix.time_stepping._time_loop.SnapshotSpec
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`times_close <astronomix.time_stepping._time_loop.times_close>`
     - .. autodoc2-docstring:: astronomix.time_stepping._time_loop.times_close
          :summary:
   * - :py:obj:`integrate <astronomix.time_stepping._time_loop.integrate>`
     - .. autodoc2-docstring:: astronomix.time_stepping._time_loop.integrate
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`FIXED_STEP <astronomix.time_stepping._time_loop.FIXED_STEP>`
     - .. autodoc2-docstring:: astronomix.time_stepping._time_loop.FIXED_STEP
          :summary:
   * - :py:obj:`ADAPTIVE_WHILE <astronomix.time_stepping._time_loop.ADAPTIVE_WHILE>`
     - .. autodoc2-docstring:: astronomix.time_stepping._time_loop.ADAPTIVE_WHILE
          :summary:
   * - :py:obj:`ADAPTIVE_CHECKPOINTED <astronomix.time_stepping._time_loop.ADAPTIVE_CHECKPOINTED>`
     - .. autodoc2-docstring:: astronomix.time_stepping._time_loop.ADAPTIVE_CHECKPOINTED
          :summary:

API
~~~

.. py:data:: FIXED_STEP
   :canonical: astronomix.time_stepping._time_loop.FIXED_STEP
   :value: 0

   .. autodoc2-docstring:: astronomix.time_stepping._time_loop.FIXED_STEP

.. py:data:: ADAPTIVE_WHILE
   :canonical: astronomix.time_stepping._time_loop.ADAPTIVE_WHILE
   :value: 1

   .. autodoc2-docstring:: astronomix.time_stepping._time_loop.ADAPTIVE_WHILE

.. py:data:: ADAPTIVE_CHECKPOINTED
   :canonical: astronomix.time_stepping._time_loop.ADAPTIVE_CHECKPOINTED
   :value: 2

   .. autodoc2-docstring:: astronomix.time_stepping._time_loop.ADAPTIVE_CHECKPOINTED

.. py:function:: times_close(t, target)
   :canonical: astronomix.time_stepping._time_loop.times_close

   .. autodoc2-docstring:: astronomix.time_stepping._time_loop.times_close

.. py:class:: SnapshotSpec
   :canonical: astronomix.time_stepping._time_loop.SnapshotSpec

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.time_stepping._time_loop.SnapshotSpec

   .. py:attribute:: store
      :canonical: astronomix.time_stepping._time_loop.SnapshotSpec.store
      :type: typing.Any
      :value: None

      .. autodoc2-docstring:: astronomix.time_stepping._time_loop.SnapshotSpec.store

   .. py:attribute:: record
      :canonical: astronomix.time_stepping._time_loop.SnapshotSpec.record
      :type: typing.Callable
      :value: None

      .. autodoc2-docstring:: astronomix.time_stepping._time_loop.SnapshotSpec.record

   .. py:attribute:: should_record
      :canonical: astronomix.time_stepping._time_loop.SnapshotSpec.should_record
      :type: typing.Callable
      :value: None

      .. autodoc2-docstring:: astronomix.time_stepping._time_loop.SnapshotSpec.should_record

   .. py:attribute:: record_final
      :canonical: astronomix.time_stepping._time_loop.SnapshotSpec.record_final
      :type: bool
      :value: True

      .. autodoc2-docstring:: astronomix.time_stepping._time_loop.SnapshotSpec.record_final

   .. py:attribute:: final_index
      :canonical: astronomix.time_stepping._time_loop.SnapshotSpec.final_index
      :type: typing.Optional[int]
      :value: None

      .. autodoc2-docstring:: astronomix.time_stepping._time_loop.SnapshotSpec.final_index

.. py:function:: integrate(state: typing.Any, step: typing.Callable, t_end, *, backend: int, t_start=0.0, num_steps: typing.Optional[int] = None, num_checkpoints: typing.Optional[int] = None, snapshots: typing.Optional[astronomix.time_stepping._time_loop.SnapshotSpec] = None, progress: typing.Optional[typing.Callable] = None)
   :canonical: astronomix.time_stepping._time_loop.integrate

   .. autodoc2-docstring:: astronomix.time_stepping._time_loop.integrate
