:py:mod:`astronomix.setup_helpers.restart`
==========================================

.. py:module:: astronomix.setup_helpers.restart

.. autodoc2-docstring:: astronomix.setup_helpers.restart
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`restart_from_latest_checkpoint <astronomix.setup_helpers.restart.restart_from_latest_checkpoint>`
     - .. autodoc2-docstring:: astronomix.setup_helpers.restart.restart_from_latest_checkpoint
          :summary:
   * - :py:obj:`latest_checkpoint_step <astronomix.setup_helpers.restart.latest_checkpoint_step>`
     - .. autodoc2-docstring:: astronomix.setup_helpers.restart.latest_checkpoint_step
          :summary:

API
~~~

.. py:function:: restart_from_latest_checkpoint(directory, params: astronomix.option_classes.simulation_params.SimulationParams, *, step: typing.Optional[int] = None, sharding: typing.Optional[jax.sharding.Sharding] = None) -> typing.Tuple[jax.Array, astronomix.option_classes.simulation_params.SimulationParams, astronomix.time_stepping.time_integration.LoopState]
   :canonical: astronomix.setup_helpers.restart.restart_from_latest_checkpoint

   .. autodoc2-docstring:: astronomix.setup_helpers.restart.restart_from_latest_checkpoint

.. py:function:: latest_checkpoint_step(directory) -> typing.Optional[int]
   :canonical: astronomix.setup_helpers.restart.latest_checkpoint_step

   .. autodoc2-docstring:: astronomix.setup_helpers.restart.latest_checkpoint_step
