:py:mod:`astronomix.analysis_helpers.jacobians`
===============================================

.. py:module:: astronomix.analysis_helpers.jacobians

.. autodoc2-docstring:: astronomix.analysis_helpers.jacobians
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`single_xmode_rhs_jacobian2D <astronomix.analysis_helpers.jacobians.single_xmode_rhs_jacobian2D>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.jacobians.single_xmode_rhs_jacobian2D
          :summary:
   * - :py:obj:`single_xmode_jacobian2Dt <astronomix.analysis_helpers.jacobians.single_xmode_jacobian2Dt>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.jacobians.single_xmode_jacobian2Dt
          :summary:

API
~~~

.. py:function:: single_xmode_rhs_jacobian2D(primitive_state_unperturbed, config, params, registered_variables, helper_data, wavelength, assembly_batch_size=4)
   :canonical: astronomix.analysis_helpers.jacobians.single_xmode_rhs_jacobian2D

   .. autodoc2-docstring:: astronomix.analysis_helpers.jacobians.single_xmode_rhs_jacobian2D

.. py:function:: single_xmode_jacobian2Dt(primitive_state_unperturbed, config, params, registered_variables, helper_data, wavelength, assembly_batch_size=4)
   :canonical: astronomix.analysis_helpers.jacobians.single_xmode_jacobian2Dt

   .. autodoc2-docstring:: astronomix.analysis_helpers.jacobians.single_xmode_jacobian2Dt
