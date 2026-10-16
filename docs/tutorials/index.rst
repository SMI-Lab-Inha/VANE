.. SPDX-License-Identifier: Apache-2.0

Tutorials
=========

The tutorials follow VANE's main path: from an OpenFAST linearization sweep, through
the multi-blade transform and the eigenanalysis, to labelled modes, a Campbell
diagram, mode-shape views, the error of the azimuth average, and finally the
state-space and parameter-varying models used for estimation and control. Each
tutorial is one script in the repository's ``examples/`` folder. Each page shows the
code, the output of an actual run and the figure it draws, and explains what the
numbers mean. Work through them in order the first time.

.. list-table::
   :header-rows: 1
   :widths: 5 30 50 15

   * - #
     - Tutorial
     - You learn
     - Run time
   * - 1
     - :doc:`linearization_sweep`
     - the operating-curve CSV, the settings VANE infers from the deck and the curve,
       the outcome of each operating point
     - minutes per point (needs OpenFAST)
   * - 2
     - :doc:`mbc_eigenanalysis`
     - reading ``.lin`` files, the MBC3 transform and azimuth average, frequencies,
       damping and participation
     - seconds
   * - 3
     - :doc:`mode_identification_campbell`
     - physical labels, tracking across operating points, the Campbell diagram,
       resonance crossings, tables and the sweep axis
     - seconds
   * - 4
     - :doc:`mode_shapes`
     - participation charts, 3-D mode-shape views and a VTK animation
     - seconds
   * - 5
     - :doc:`azimuth_averaging`
     - the blade-number rule for the azimuth count, pinning the azimuth direction, the
       averaging-error estimate and its check against Floquet exponents
     - seconds
   * - 6
     - :doc:`state_space_export`
     - state-space export, modal reduction, a Kalman filter, mode-tracked and
       physical-state families
     - seconds

.. toctree::
   :maxdepth: 1
   :hidden:

   linearization_sweep
   mbc_eigenanalysis
   mode_identification_campbell
   mode_shapes
   azimuth_averaging
   state_space_export

Before you start
----------------

* Install VANE (see :doc:`../user_guide/installation`) and work from the root of the
  repository, which holds the ``examples/`` folder.
* Run a script with ``python examples/<script>.py``. Without options, the generated
  data go to a temporary directory that is deleted at the end, and the figures to
  ``docs/tutorials/figures/``. ``--output DIR`` sends every file to ``DIR`` instead.
  ``--help`` lists the options of each script.
* Only tutorial 1 needs OpenFAST. Tutorials 2 to 6 run on small synthetic ``.lin``
  files that the helper module ``examples/_synthetic.py`` writes, because OpenFAST
  linearization files are not distributed with VANE.

.. _tutorial-synthetic-turbine:

The synthetic turbine
---------------------

The synthetic turbine has six degrees of freedom: the first tower fore-aft and
side-side bending modes, the generator azimuth, and the first flap mode of each of
three blades. Each blade is a damped oscillator in its own rotating frame whose
frequency rises with rotor speed (centrifugal stiffening). The tower fore-aft mode
couples to the three blades equally, the tower side-side mode to each blade through
the sine of its azimuth, and a small 3P term modulates the tower fore-aft stiffness.
The files are written as OpenFAST writes them: tower states in the fixed frame, blade
states in the rotating frame (flag ``T``), with ElastoDyn state descriptions, two
inputs (wind speed and collective pitch) and eight outputs. Twelve azimuths make one
operating point. Below rated the rotor speed follows a constant tip-speed ratio; above
rated (10.6 m/s) it stays at 12.1 rpm.

The numbers resemble a 5 MW-class turbine but describe no real machine, and the
model leaves out much of a real linearization (edgewise and second modes,
drivetrain, aerodynamic states). It is small enough to follow by hand and has the
features the tutorials need: the MBC3 transform splits the flap triplet into
regressive, collective and progressive modes, and the tower and the regressive flap
mode cross the 3P line in the operating range.

.. admonition:: A clean run is not a validation
   :class: tip

   Every tutorial checks its result against something independent: the frozen-azimuth
   eigenvalues, the excitation-line arithmetic, an exact eigen-solution, the Floquet
   exponents, or a simulated truth. Keep that habit for your own models, and see
   :doc:`../user_guide/validation` for the evidence behind VANE itself.
