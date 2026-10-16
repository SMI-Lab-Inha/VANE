.. SPDX-License-Identifier: Apache-2.0

Tutorial 1: An OpenFAST linearization sweep
===========================================

**Goal:** linearize the NREL 5 MW land-based turbine at several wind speeds with
:func:`~vane.run.run_operating_sweep`, starting from nothing but the deck and a
steady operating curve, and read the result.

**Script:** ``examples/01_linearization_sweep.py`` · **Needs:** an OpenFAST
executable and the OpenFAST ``r-test`` repository · **Run time:** minutes per
operating point

.. important::

   This is the only tutorial that runs OpenFAST. Without the two inputs below the
   script prints what is missing and exits with status 0; the other tutorials do not
   depend on it.

Set up OpenFAST and the deck
----------------------------

1. Install an OpenFAST executable. Note the release: ``openfast -v`` prints it.
2. Clone the public regression-test repository, which holds the reference decks:

   .. code-block:: console

      $ git clone https://github.com/OpenFAST/r-test.git

   The input format of OpenFAST decks changes between releases, and the ``r-test``
   decks follow the OpenFAST development line. Use an ``r-test`` commit that matches
   your executable: an OpenFAST source tree includes ``r-test`` as the submodule
   ``reg_tests/r-test``, and ``git submodule status reg_tests/r-test`` in that tree
   shows the commit of its release.
3. Point the script at both:

   .. code-block:: console

      $ export VANE_RTEST_DIR=/path/to/r-test       # Windows: set VANE_RTEST_DIR=...
      $ export VANE_OPENFAST=/path/to/openfast      # or put openfast on PATH

The script uses the deck ``glue-codes/openfast/5MW_Land_DLL_WTurb/5MW_Land_DLL_WTurb.fst``
of the clone. It reads files from the shared ``5MW_Baseline`` folder next to it; VANE
finds and copies them for each case.

The operating curve
-------------------

VANE does not compute rotor performance: the rotor speed and blade pitch at each wind
come from the user. The script reads them from ``examples/data/nrel5mw_operating_curve.csv``:

.. literalinclude:: ../../examples/data/nrel5mw_operating_curve.csv
   :language: text

The three required columns are ``wind_speed`` (m/s), ``rotor_speed_rpm`` (rpm) and
``blade_pitch_deg`` (deg). The optional ``cp`` column lets VANE find the rated point
and the rated torque. This curve is the NREL 5 MW-shaped peak-shaving schedule used in
VANE's own tests (``tests/test_run/test_auto_config.py``): the pitch starts to rise at
11.1 m/s, and the rotor speed and the power level off at 12.121 m/s. It is close to,
but not identical with, the turbine's published schedule. For your own turbine, use its
own steady operating schedule. :doc:`../user_guide/linearization_sweep` lists the
accepted column names and units.

The code
--------

The script first checks its inputs, then parses the curve and asks VANE what it will
configure, without launching anything:

.. literalinclude:: ../../examples/01_linearization_sweep.py
   :language: python
   :start-after: # 1. The operating curve
   :end-before: if args.configure_only:
   :dedent: 4

:func:`~vane.run.detect_deck_generation` reads the deck's keywords to tell which
OpenFAST releases can read it, and :func:`~vane.run.auto_configure_sweep` infers the
turbine constants from the deck and the curve, whose ``cp`` column
:func:`~vane.run.auto_config.curve_power_coefficient` reads as fractions (a column
declared ``cp [%]`` is converted). All three only read files. The sweep itself is one
call:

.. literalinclude:: ../../examples/01_linearization_sweep.py
   :language: python
   :start-after: results in {sweep_dir}")
   :end-before: # One row per operating point.
   :dedent: 4

:func:`~vane.run.run_operating_sweep` copies the deck once per wind speed into
``sweep/ws<wind>``, writes the curve's rotor speed and pitch as initial conditions,
switches on a ``CalcSteady`` linearization with 36 azimuths (``NLinTimes = 36``), runs
the cases in parallel (by default on all cores but one), retries a trim that does not
converge, and collects the ``.lin`` files.

Run it
------

Check the configuration first. ``--configure-only`` reads the deck and the curve and
stops before OpenFAST is launched:

.. code-block:: console

   $ python examples/01_linearization_sweep.py --configure-only

On the ``r-test`` decks in the OpenFAST 5 format it prints:

.. code-block:: text

   Operating curve nrel5mw_operating_curve.csv: 8 points
     wind  3.000 m/s   rotor  6.9000 rpm   pitch  1.60 deg
     wind  6.000 m/s   rotor  6.9000 rpm   pitch  0.00 deg
     wind  9.770 m/s   rotor 10.4000 rpm   pitch  0.00 deg
     wind 11.116 m/s   rotor 11.8350 rpm   pitch  2.47 deg
     wind 12.121 m/s   rotor 12.1376 rpm   pitch  4.12 deg
     wind 14.430 m/s   rotor 12.1376 rpm   pitch  9.55 deg
     wind 18.390 m/s   rotor 12.1376 rpm   pitch 15.45 deg
     wind 25.000 m/s   rotor 12.1376 rpm   pitch 23.12 deg

   Deck 5MW_Land_DLL_WTurb.fst: input format v5, needs OpenFAST >=5.0.0, <6.0.0
   Inferred settings:
     module_preset = 'offshore'  (LinearizationSettings.offshore())
     unsteady_aero_mode = 0  (NRELOffshrBsline5MW_Onshore_AeroDyn.dat UA_Mod)
     rated_wind_speed = 12.121  (operating curve)
     rated_generator_torque = 43235.438684341585  (performance curve cp at rated)
     trim_gain_pitch = 0.0002  (ElastoDyn TipRad)
     trim_tol = 0.001  (ElastoDyn TipRad)
     NLinTimes = 36, CalcSteady = True

Each inferred value is printed with its source:

``module_preset``
   The module settings that make the deck linearizable, adapted to it.
``unsteady_aero_mode``
   The deck's unsteady-aerodynamics model is not linearizable, so it is switched off
   (``UA_Mod = 0``) in the case copies.
``rated_wind_speed``
   The first curve wind at which the rotor speed and ``cp * V**3`` reach their
   plateau. It decides torque trim below rated and pitch trim at and above it.
``rated_generator_torque``
   This deck's controller is a compiled library with no ``DISCON.IN`` file to read,
   so the rated torque comes from the curve's ``cp`` at rated. The above-rated trim
   holds it constant.
``trim_gain_pitch``, ``trim_tol``
   Scaled to the rotor radius read from ElastoDyn ``TipRad``; for this 63 m rotor they
   keep their reference values.

The OpenFAST release the deck needs is checked as well: an executable that cannot read
the deck is rejected before any case is copied. Then run the sweep:

.. code-block:: console

   $ python examples/01_linearization_sweep.py --winds 6 10 14 18 --output sweep_results

The script prints a table of the cases, one row per wind speed: the rotor speed and
pitch it was linearized at, ``converged``, the number of OpenFAST ``attempts`` and the
number of ``.lin`` files (36 for a converged point). It then runs
:class:`~vane.ModalPipeline` over ``sweep_results/sweep`` on the wind-speed axis, since
the rotor speed is constant above rated, and prints each labelled track with its
frequency at the four wind speeds.

Reading the result
------------------

``result.cases`` holds one :class:`~vane.run.SweepCase` per wind speed. Check three
fields before using a point:

* ``converged`` is ``True`` only when OpenFAST found the periodic steady state and
  wrote every ``.lin`` file with finite numbers. A point whose trim never converged is
  recorded ``False`` rather than accepted with a forced single ``.lin`` file.
* ``attempts`` counts the OpenFAST runs the retry ladder needed. Each attempt's full
  output is in the case directory as ``<deck>.attempt<k>.log``.
* ``n_lin`` is the number of full-system ``.lin`` files.

``result.auto_configuration`` holds every inferred value with its source, including the
executable and the copy root chosen at run time, and the same record is written to
``sweep/auto_configuration.json``. The ``.lin`` files in ``sweep/ws<wind>`` are the
input of tutorials 2 to 6: replace the synthetic directory of any later tutorial with
``sweep_results/sweep``.

Where next
----------

* :doc:`../user_guide/linearization_sweep`: every option of the sweep, floating
  turbines, the retry ladder and the controller library.
* :doc:`../user_guide/trim_verification`: check each trimmed point against the curve.
* :doc:`../theory/linearization_mbc`: what the periodic linearization is.
* Next tutorial: :doc:`mbc_eigenanalysis`.
