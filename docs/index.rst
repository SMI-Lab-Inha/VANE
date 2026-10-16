.. SPDX-License-Identifier: Apache-2.0

VANE
====

**VANE** (Visualization and Automated Natural-frequency Extraction) is an open-source
(`Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_) Python tool for the
linearization analysis of wind turbines in OpenFAST. It takes an OpenFAST model and the
turbine's operating curve to labelled Campbell diagrams and modal results:

#. automatic set-up of the linearization case of every operating point, with
   pre-flight checks of the deck (:doc:`user_guide/linearization_sweep`);
#. the sweep, run in parallel, with automatic trim retries and settle runs for
   floating turbines (:doc:`user_guide/trim_verification`);
#. the MBC3 transform and the azimuth average on the delivered azimuths
   (:doc:`user_guide/mode_identification`);
#. automatic mode identification, labelling and tracking across operating points;
#. Campbell diagrams with excitation lines, resonance crossings, and tabular and Excel
   summaries (:doc:`user_guide/campbell_diagrams`);
#. mode-shape visualization, static, 3-D and VTK (:doc:`user_guide/visualization`);
#. Floquet analysis and per-branch averaging-error estimates
   (:doc:`user_guide/azimuth_averaging`); and
#. state-space and LPV export for estimator and controller design
   (:doc:`user_guide/state_space_export`).

VANE handles fixed-bottom and floating turbines. It is developed at the Marine
Structural Mechanics and Integrity Lab (SMI Lab), Inha University.

The current stable release is 0.1.0. The public API is documented in the
:doc:`reference/api_reference` and is stable. VANE follows semantic versioning: a
breaking change to the documented public API is made only with a new major version and
is recorded in the :doc:`user_guide/changelog`.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/installation
   user_guide/quickstart
   user_guide/linearization_sweep
   user_guide/trim_verification
   user_guide/mode_identification
   user_guide/campbell_diagrams
   user_guide/visualization
   user_guide/plot_gallery
   user_guide/azimuth_averaging
   user_guide/state_space_export
   user_guide/data_sources
   user_guide/coordinate_systems
   user_guide/units_conventions
   user_guide/limitations

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/api_reference
   reference/api_contract
   reference/linearization_restrictions

.. toctree::
   :maxdepth: 1
   :caption: Project

   user_guide/validation
   user_guide/citing
   user_guide/changelog
   developer_guide/contributing
   developer_guide/architecture
   developer_guide/release

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
