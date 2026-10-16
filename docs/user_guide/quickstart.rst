.. SPDX-License-Identifier: Apache-2.0

Quick start
===========

VANE works on OpenFAST linearization (``.lin``) files. This page shows the shortest
routes from those files to identified modes, a Campbell diagram and a state-space model.
To produce the ``.lin`` files with VANE, see :doc:`linearization_sweep`.

The modal pipeline
------------------

:class:`~vane.ModalPipeline` runs the whole analysis over a set of operating points:
the MBC3 transform, the eigensolution, physical labelling, cross-operating-point
tracking, Campbell assembly and resonance detection.

.. code-block:: python

   from vane import ModalPipeline, discover_operating_points

   # One entry per operating point; each holds the .lin files of its azimuth sweep.
   operating_points = discover_operating_points("path/to/lin_files")

   result = ModalPipeline().run(operating_points, parameter_name="wind_speed")

   for track in result.campbell.tracks:
       print(track.label.label, track.natural_frequencies_hz)

   for crossing in result.resonances:
       print(f"{crossing.harmonic}P at {crossing.rotor_speed_rpm:.1f} rpm")

   system = result.state_space(0)  # A, B, C, D at the first operating point

:func:`~vane.discover_operating_points` groups the files of a directory into operating
points. OpenFAST writes one ``.lin`` file per linearization time, named
``<case>.<n>.lin``; files that share a ``<case>`` root form one operating point's azimuth
sweep, and a directory per operating point is read recursively. A list of lists of
:class:`~vane.io.LinFile` objects (from :func:`~vane.io.read_lin_file`) works equally
well. The pipeline sorts the operating points by the operating parameter, so the input
order does not matter.

The operating parameter is rotor speed by default (``parameter_name="rotor_speed_rpm"``).
Above rated wind the trimmed rotor speed is constant, so a sweep that extends past rated
must use ``parameter_name="wind_speed"``; the pipeline rejects a rotor-speed axis on
which two points coincide rather than switching silently. Resonance (``nP`` crossing)
detection needs the rotor-speed axis, so a wind-speed run reports none. See
:doc:`campbell_diagrams` for the details.

A reproducible study
--------------------

:func:`~vane.run_study` runs the same pipeline and records the provenance of the
result: the source files with their content hashes, the azimuth coverage of each
operating point, the tracking settings, the environment and the VANE version.

.. code-block:: python

   from vane import discover_operating_points, run_study

   study = run_study(discover_operating_points("path/to/lin_files"))
   print(study.provenance.vane_version, study.provenance.n_tracks)

   # provenance.json, campbell.csv and a self-contained report.html
   study.write_bundle("study_output")

The HTML report holds the validation checks, the warnings, the Campbell diagram,
the tracking diagnostics and the file hashes, so every figure can be traced to its
inputs.

The command line
----------------

The ``vane`` command offers the same three entry points:

.. code-block:: console

   $ vane modes case.1.lin case.2.lin case.3.lin -o modes.csv
   $ vane campbell path/to/lin_files --parameter wind -o campbell.csv
   $ vane study path/to/lin_files --parameter wind -o study_output

``vane modes`` tabulates the modes of one operating point, ``vane campbell`` writes the
tracked Campbell table (``.csv``, ``.json``, ``.xlsx`` or ``.mat`` by suffix, or the
terminal without ``-o``), and ``vane study`` writes the study bundle. ``vane study``
also accepts ``--tracking simultaneous``, ``--branch-review``, ``--resolve`` and
``--no-report``; ``vane --help`` and ``vane <command> --help`` list every option, and
``vane --version`` prints the version.

Where to go next
----------------

* :doc:`linearization_sweep`: run OpenFAST over an operating curve to produce the
  ``.lin`` files, including floating turbines.
* :doc:`mode_identification`: the individual steps (MBC3, eigensolution, participation,
  labels, tracking) and their options.
* :doc:`campbell_diagrams`: Campbell diagrams, excitation lines and resonance crossings.
* :doc:`state_space_export`: state-space, reduced-order, Kalman-filter and LPV models.
* :doc:`azimuth_averaging`: how large the error of the azimuth-averaged model is, and how
  many azimuths a linearization needs.
