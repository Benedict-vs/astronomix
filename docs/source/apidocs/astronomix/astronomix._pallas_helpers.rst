:orphan:

:py:mod:`astronomix._pallas_helpers`
====================================

.. py:module:: astronomix._pallas_helpers

.. autodoc2-docstring:: astronomix._pallas_helpers
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`pallas_mesh_context <astronomix._pallas_helpers.pallas_mesh_context>`
     - .. autodoc2-docstring:: astronomix._pallas_helpers.pallas_mesh_context
          :summary:
   * - :py:obj:`diffable_pallas_call <astronomix._pallas_helpers.diffable_pallas_call>`
     - .. autodoc2-docstring:: astronomix._pallas_helpers.diffable_pallas_call
          :summary:
   * - :py:obj:`diffable_pallas_call_n <astronomix._pallas_helpers.diffable_pallas_call_n>`
     - .. autodoc2-docstring:: astronomix._pallas_helpers.diffable_pallas_call_n
          :summary:
   * - :py:obj:`pallas_vjp_call <astronomix._pallas_helpers.pallas_vjp_call>`
     - .. autodoc2-docstring:: astronomix._pallas_helpers.pallas_vjp_call
          :summary:

API
~~~

.. py:function:: pallas_mesh_context(mesh)
   :canonical: astronomix._pallas_helpers.pallas_mesh_context

   .. autodoc2-docstring:: astronomix._pallas_helpers.pallas_mesh_context

.. py:function:: diffable_pallas_call(state, params, *, pallas_branch, native_branch)
   :canonical: astronomix._pallas_helpers.diffable_pallas_call

   .. autodoc2-docstring:: astronomix._pallas_helpers.diffable_pallas_call

.. py:function:: diffable_pallas_call_n(primals, *, pallas_branch, native_branch)
   :canonical: astronomix._pallas_helpers.diffable_pallas_call_n

   .. autodoc2-docstring:: astronomix._pallas_helpers.diffable_pallas_call_n

.. py:function:: pallas_vjp_call(state, aux, *, pallas_forward, pallas_backward)
   :canonical: astronomix._pallas_helpers.pallas_vjp_call

   .. autodoc2-docstring:: astronomix._pallas_helpers.pallas_vjp_call
