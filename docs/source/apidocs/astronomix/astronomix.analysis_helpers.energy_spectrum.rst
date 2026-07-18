:py:mod:`astronomix.analysis_helpers.energy_spectrum`
=====================================================

.. py:module:: astronomix.analysis_helpers.energy_spectrum

.. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`vector_field_energy_spectrum <astronomix.analysis_helpers.energy_spectrum.vector_field_energy_spectrum>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.vector_field_energy_spectrum
          :summary:
   * - :py:obj:`vector_field_cross_spectrum <astronomix.analysis_helpers.energy_spectrum.vector_field_cross_spectrum>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.vector_field_cross_spectrum
          :summary:
   * - :py:obj:`get_kinetic_energy_spectrum <astronomix.analysis_helpers.energy_spectrum.get_kinetic_energy_spectrum>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.get_kinetic_energy_spectrum
          :summary:
   * - :py:obj:`get_magnetic_energy_spectrum <astronomix.analysis_helpers.energy_spectrum.get_magnetic_energy_spectrum>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.get_magnetic_energy_spectrum
          :summary:
   * - :py:obj:`get_cross_helicity_spectrum <astronomix.analysis_helpers.energy_spectrum.get_cross_helicity_spectrum>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.get_cross_helicity_spectrum
          :summary:
   * - :py:obj:`get_magnetic_helicity_spectrum <astronomix.analysis_helpers.energy_spectrum.get_magnetic_helicity_spectrum>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.get_magnetic_helicity_spectrum
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOG_BINNING <astronomix.analysis_helpers.energy_spectrum.LOG_BINNING>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.LOG_BINNING
          :summary:
   * - :py:obj:`INTEGER_BINNING <astronomix.analysis_helpers.energy_spectrum.INTEGER_BINNING>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.INTEGER_BINNING
          :summary:
   * - :py:obj:`PHYSICAL_BINNING <astronomix.analysis_helpers.energy_spectrum.PHYSICAL_BINNING>`
     - .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.PHYSICAL_BINNING
          :summary:

API
~~~

.. py:data:: LOG_BINNING
   :canonical: astronomix.analysis_helpers.energy_spectrum.LOG_BINNING
   :value: 0

   .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.LOG_BINNING

.. py:data:: INTEGER_BINNING
   :canonical: astronomix.analysis_helpers.energy_spectrum.INTEGER_BINNING
   :value: 1

   .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.INTEGER_BINNING

.. py:data:: PHYSICAL_BINNING
   :canonical: astronomix.analysis_helpers.energy_spectrum.PHYSICAL_BINNING
   :value: 2

   .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.PHYSICAL_BINNING

.. py:function:: vector_field_energy_spectrum(fx, fy, fz, energy_coeff=1.0, binning=PHYSICAL_BINNING)
   :canonical: astronomix.analysis_helpers.energy_spectrum.vector_field_energy_spectrum

   .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.vector_field_energy_spectrum

.. py:function:: vector_field_cross_spectrum(f1x, f1y, f1z, f2x, f2y, f2z, coeff=1.0, binning=PHYSICAL_BINNING)
   :canonical: astronomix.analysis_helpers.energy_spectrum.vector_field_cross_spectrum

   .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.vector_field_cross_spectrum

.. py:function:: get_kinetic_energy_spectrum(vx, vy, vz, rho, binning=PHYSICAL_BINNING)
   :canonical: astronomix.analysis_helpers.energy_spectrum.get_kinetic_energy_spectrum

   .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.get_kinetic_energy_spectrum

.. py:function:: get_magnetic_energy_spectrum(Bx, By, Bz, mu0=1.0, binning=PHYSICAL_BINNING)
   :canonical: astronomix.analysis_helpers.energy_spectrum.get_magnetic_energy_spectrum

   .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.get_magnetic_energy_spectrum

.. py:function:: get_cross_helicity_spectrum(vx, vy, vz, Bx, By, Bz, binning=PHYSICAL_BINNING)
   :canonical: astronomix.analysis_helpers.energy_spectrum.get_cross_helicity_spectrum

   .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.get_cross_helicity_spectrum

.. py:function:: get_magnetic_helicity_spectrum(Bx, By, Bz, binning=PHYSICAL_BINNING)
   :canonical: astronomix.analysis_helpers.energy_spectrum.get_magnetic_helicity_spectrum

   .. autodoc2-docstring:: astronomix.analysis_helpers.energy_spectrum.get_magnetic_helicity_spectrum
