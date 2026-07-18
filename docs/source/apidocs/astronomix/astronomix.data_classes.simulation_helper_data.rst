:py:mod:`astronomix.data_classes.simulation_helper_data`
========================================================

.. py:module:: astronomix.data_classes.simulation_helper_data

.. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HelperData <astronomix.data_classes.simulation_helper_data.HelperData>`
     - .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData
          :summary:
   * - :py:obj:`HelperDataRequirements <astronomix.data_classes.simulation_helper_data.HelperDataRequirements>`
     - .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_helper_data <astronomix.data_classes.simulation_helper_data.get_helper_data>`
     - .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.get_helper_data
          :summary:

API
~~~

.. py:class:: HelperData
   :canonical: astronomix.data_classes.simulation_helper_data.HelperData

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData

   .. py:attribute:: geometric_centers
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.geometric_centers
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.geometric_centers

   .. py:attribute:: volumetric_centers
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.volumetric_centers
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.volumetric_centers

   .. py:attribute:: cell_centers_x
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.cell_centers_x
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.cell_centers_x

   .. py:attribute:: cell_centers_y
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.cell_centers_y
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.cell_centers_y

   .. py:attribute:: cell_centers_z
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.cell_centers_z
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.cell_centers_z

   .. py:attribute:: r
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.r
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.r

   .. py:attribute:: r_hat_alpha
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.r_hat_alpha
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.r_hat_alpha

   .. py:attribute:: cell_volumes
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.cell_volumes
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.cell_volumes

   .. py:attribute:: inner_cell_boundaries
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.inner_cell_boundaries
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.inner_cell_boundaries

   .. py:attribute:: outer_cell_boundaries
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.outer_cell_boundaries
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.outer_cell_boundaries

.. py:class:: HelperDataRequirements
   :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements

   .. py:attribute:: needs_geometric_centers
      :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_geometric_centers
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_geometric_centers

   .. py:attribute:: needs_volumetric_centers
      :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_volumetric_centers
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_volumetric_centers

   .. py:attribute:: needs_cell_centers_x
      :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_cell_centers_x
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_cell_centers_x

   .. py:attribute:: needs_cell_centers_y
      :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_cell_centers_y
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_cell_centers_y

   .. py:attribute:: needs_cell_centers_z
      :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_cell_centers_z
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_cell_centers_z

   .. py:attribute:: needs_r
      :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_r
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_r

   .. py:attribute:: needs_r_hat_alpha
      :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_r_hat_alpha
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_r_hat_alpha

   .. py:attribute:: needs_cell_volumes
      :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_cell_volumes
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_cell_volumes

   .. py:attribute:: needs_inner_cell_boundaries
      :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_inner_cell_boundaries
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_inner_cell_boundaries

   .. py:attribute:: needs_outer_cell_boundaries
      :canonical: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_outer_cell_boundaries
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperDataRequirements.needs_outer_cell_boundaries

.. py:function:: get_helper_data(config: astronomix.option_classes.simulation_config.SimulationConfig, sharding: typing.Union[types.NoneType, jax.NamedSharding] = None, padded: bool = False, requirements: typing.Union[types.NoneType, astronomix.data_classes.simulation_helper_data.HelperDataRequirements] = None) -> astronomix.data_classes.simulation_helper_data.HelperData
   :canonical: astronomix.data_classes.simulation_helper_data.get_helper_data

   .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.get_helper_data
