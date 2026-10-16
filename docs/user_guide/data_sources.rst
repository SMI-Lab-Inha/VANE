.. SPDX-License-Identifier: Apache-2.0

Data sources
============

VANE reads the text outputs of an OpenFAST linearization analysis and, for context, the
OpenFAST input files of the model.

Linearization files (``.lin``)
------------------------------

The primary input is the modern OpenFAST (ModGlue) ``.lin`` format, written when
``Linearize = True``. Each file corresponds to one linearization time (one rotor
azimuth) and contains:

* a header with the simulation time, rotor speed, azimuth, and wind speed;
* operating-point tables for the continuous states ``x``, state derivatives
  ``xdot``, inputs ``u``, and outputs ``y``, each with a description, a
  rotating-frame flag, and a derivative order per channel;
* the state-space matrices ``A``, ``B``, ``C``, and ``D``. OpenFAST does not write an
  empty block, such as ``B`` and ``D`` for a model with no inputs.

:func:`~vane.io.lin_reader.read_lin_file` parses a file into a
:class:`~vane.io.lin_reader.LinFile` and validates it on reading. A non-finite header
scalar or operating-point value, a state count that disagrees with the header, a missing
``A`` matrix, or a matrix whose shape disagrees with the channel tables is rejected with
:class:`~vane.io.lin_reader.LinFileFormatError`. Fortran overflow markers (runs of
``*``) inside matrix data are parsed as ``nan`` with a logged warning rather than failing
the parse.

.. note::

   The legacy ``.lin`` format (without the per-channel derivative-order column) is
   detected and rejected with an explicit error. Regenerate the linearization with a
   current OpenFAST release.

Model files (``.fst`` and ElastoDyn)
------------------------------------

:func:`~vane.io.fst_reader.read_fst_file` and
:func:`~vane.io.fst_reader.read_elastodyn_geometry` extract the linearization
configuration, active-module switches, referenced sub-files, and rotor and tower
geometry from a primary ``.fst`` deck and its ElastoDyn input. These give context
(geometry, module set) for interpreting the modes, for example the blade and tower
lengths used by :func:`~vane.modal.participation.participation_scale_factors`.

Values are read the way OpenFAST's Fortran list-directed input reads them: a value
may be followed by a comma (``1,   0   LinInputs`` reads ``1``), and a logical is
decided by its first character after an optional leading ``.`` (``True``, ``T``,
``.true.`` are true; ``False``, ``F``, ``.false.`` are false). ``TowerBsHt`` is
optional and defaults to ``0`` when the ElastoDyn file omits it. Quoted strings are
tokenized before the ``" - "`` description separator is located, so a quoted path
that itself contains ``" - "`` is read (and edited by
:class:`~vane.io.fst_writer.FastInputFile`) intact.

Azimuth sweeps and operating points
-----------------------------------

One operating point is a set of ``.lin`` files at the same rotor speed sampled at
several azimuths over a revolution. OpenFAST names them ``<case>.<n>.lin``, and
:func:`~vane.study.discover_operating_points` groups a directory of files by their
``<case>`` root, leaving out the per-module files written under ``LinOutMod = True``.

:func:`~vane.io.mbc_transform.mbc3_transform` averages the files of one operating point
into the non-rotating frame and checks that they form one coherent sweep. It rejects
duplicate azimuths (including the ``0``/``360`` deg wrap-around) and a rotor speed or
wind speed that drifts across the files by more than 1 % (adjustable with
``rotor_speed_drift_rtol`` and ``wind_speed_drift_rtol``), since an averaged model is
only meaningful for a single periodic operating point. For a spinning rotor it logs a
warning when there are fewer than three azimuths or the spacing is far from uniform.

A sweep is a series of operating points at different rotor or wind speeds; this is
what produces a Campbell diagram (see :doc:`campbell_diagrams`).

Rotor-performance schedule (driving the sweep)
----------------------------------------------

When VANE produces its own linearizations with
:func:`~vane.run.auto_sweep.run_operating_sweep`, it needs the turbine's steady operating
schedule (rotor speed and blade pitch versus wind) to set each operating point. This is
a user-supplied input: VANE computes no rotor performance itself and has no
aerodynamic-performance (blade-element) dependency.
:func:`~vane.run.operating_points.operating_curve_from_csv` reads it from a header-row
CSV with three required columns, ``wind_speed`` (m/s), ``rotor_speed_rpm`` (rpm) and
``blade_pitch_deg`` (deg). Header matching ignores case, whitespace and separators, and
accepts common variants such as ``RotSpeed`` or ``BldPitch``. Extra columns (``cp``,
``ct``, ...) are ignored. A unit in square brackets or parentheses on a required column
is honoured: ``km/h``, ``rad/s`` and ``rad`` are converted to m/s, rpm and deg, and any
other unit is rejected with an error. Alternatively the schedule is passed as the
``curve_*`` arrays directly, or derived from controller ``settle`` runs. See
:doc:`linearization_sweep` for a worked example.

Test data
---------

VANE does not need, and the repository does not contain, real turbine ``.lin``,
``.outb`` or proprietary model files. The tests use synthetic data generated in code.
Optional integration tests run against a local copy of the OpenFAST ``r-test`` suite
(located through the ``VANE_RTEST_DIR`` environment variable) and are skipped when it is
not available.
