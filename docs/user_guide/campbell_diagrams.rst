.. SPDX-License-Identifier: Apache-2.0

Campbell diagrams
=================

A Campbell diagram plots the natural frequency of each tracked mode against an
operating parameter, rotor speed or wind speed, together with the per-revolution
(``nP``) excitation lines. VANE builds it from the mode tracks of
:doc:`mode_identification`, finds where the tracks cross the excitation lines, plots
it, and exports it as a table. The theory is on :doc:`../theory/campbell_uncertainty`.

.. contents:: On this page
   :local:
   :depth: 1

From the pipeline
-----------------

:meth:`ModalPipeline.run <vane.pipeline.ModalPipeline.run>` returns the diagram as
``result.campbell`` and the resonance crossings as ``result.resonances``:

.. code-block:: python

   from vane import ModalPipeline, discover_operating_points

   operating_points = discover_operating_points("lin_files")
   result = ModalPipeline().run(operating_points)

   diagram = result.campbell
   for track in diagram.tracks:
       print(track.label.label, track.natural_frequencies_hz)

   for crossing in result.resonances:
       print(
           f"track {crossing.track_index}: {crossing.harmonic}P at "
           f"{crossing.rotor_speed_rpm:.2f} rpm, {crossing.frequency_hz:.3f} Hz, "
           f"{crossing.severity.value}"
       )

The pipeline sorts the operating points by the operating parameter, so the input order
does not matter. ``ModalPipeline(harmonics=...)`` sets the excitation orders searched for
crossings (default ``(1, 3, 6, 9)``).

Choosing the sweep axis
-----------------------

The operating parameter is the axis along which the sweep is sorted, tracked and
plotted: rotor speed (``parameter_name="rotor_speed_rpm"``, the default; CLI
``--parameter rpm``) or wind speed (``parameter_name="wind_speed"``; CLI
``--parameter wind``).

Above rated, the trimmed rotor speed is constant, so the rotor-speed values of those
points are equal up to round-off and ordering by them would scramble the sweep. The
pipeline therefore rejects a sweep axis on which two operating points lie within
``parameter_rtol`` (default ``1e-4``, relative to the largest-magnitude value on the
axis) of each other. The ``ValueError`` names the two values and recommends the
wind-speed axis. The axis is never switched automatically; for a sweep that extends past
rated, run on wind speed:

.. code-block:: python

   result = ModalPipeline().run(operating_points, parameter_name="wind_speed")

.. code-block:: console

   $ vane campbell lin_files --parameter wind

``parameter_rtol`` is accepted by :meth:`~vane.pipeline.ModalPipeline.run` and
:func:`~vane.study.run_study`; ``0`` keeps only the check for exact duplicates.
Resonance detection needs the rotor-speed axis, so a wind-speed run returns an empty
``result.resonances``.

Building a diagram directly
---------------------------

:func:`~vane.campbell.builder.build_campbell` pairs an
:class:`~vane.modal.identifier.IdentificationResult` with the parameter value of each
operating point. The values must be finite and strictly increasing, one per operating
point, in the order of the solutions that were tracked:

.. code-block:: python

   from vane.campbell import build_campbell
   from vane.io import mbc3_transform
   from vane.modal import identify_modes, modes_from_mbc

   mbc_results = sorted(
       (mbc3_transform(point) for point in operating_points),
       key=lambda result: result.rotor_speed_rpm,
   )
   solutions = [modes_from_mbc(result) for result in mbc_results]
   rpm = [result.rotor_speed_rpm for result in mbc_results]

   diagram = build_campbell(identify_modes(solutions), rpm)

``parameter_name`` (default ``"rotor_speed_rpm"``) names the axis, and
``include_heavily_damped=False`` leaves the heavily damped tracks out of the diagram.
:func:`~vane.campbell.builder.campbell_from_solutions` runs the tracking and the
assembly in one call.

A :class:`~vane.campbell.builder.CampbellDiagram` holds ``parameter_values``,
``parameter_name``, ``n_operating_points``, the ``tracks``, and the
``heavily_damped_tracks``. :meth:`~vane.campbell.builder.CampbellDiagram.track_curve`
returns the parameter values, natural frequencies and damping ratios along one track:

.. code-block:: python

   for track in diagram.tracks:
       x, frequency_hz, damping = diagram.track_curve(track)

Heavily damped tracks (damping ratio at or above 0.99, see
:doc:`mode_identification`) are kept apart from ``tracks``. They are drawn distinctly
and never enter the resonance search, since a mode damped near critically cannot
resonate.

Excitation lines and resonance crossings
----------------------------------------

A rotor turning at ``rpm`` excites the structure at ``n * rpm / 60`` Hz, the ``nP``
lines. :func:`~vane.campbell.excitation.excitation_frequencies` evaluates them, and
:func:`~vane.campbell.excitation.find_resonances` finds where a track's natural
frequency crosses one, interpolating linearly between operating points:

.. code-block:: python

   from vane.campbell import find_resonances

   crossings = find_resonances(diagram, harmonics=(1, 3, 6), operating_range=(6.9, 12.1))

``harmonics`` defaults to ``DEFAULT_HARMONICS``, ``(1, 3, 6, 9)``: the 1P rotor
imbalance and the blade-passing lines of a three-bladed rotor. ``operating_range`` keeps
only the crossings whose rotor speed lies in the given ``(min_rpm, max_rpm)`` window.
The function raises ``ValueError`` for a diagram whose parameter is not
``"rotor_speed_rpm"``.

Each :class:`~vane.campbell.excitation.ResonanceCrossing` carries the ``track_index``,
the ``harmonic``, the interpolated ``rotor_speed_rpm``, ``frequency_hz`` and
``damping_ratio``, and a :class:`~vane.campbell.excitation.ResonanceSeverity` from the
damping ratio at the crossing:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Severity
     - Damping ratio at the crossing
   * - ``HIGH``
     - below 0.01
   * - ``MEDIUM``
     - from 0.01 to below 0.05
   * - ``LOW``
     - 0.05 or more

Plotting
--------

:func:`~vane.viz.campbell_plot.plot_campbell_static` draws the tracks, the ``nP`` lines
and the crossings as a Matplotlib figure for publication, and
:func:`~vane.viz.damping_plot.plot_damping_static` plots the damping ratio of every track
against the operating parameter with stability bands.
:func:`~vane.viz.campbell_plot.plot_campbell_panels` puts the two side by side, and
:func:`~vane.viz.campbell_plot.plot_campbell` and
:func:`~vane.viz.damping_plot.plot_damping` draw them as interactive Plotly figures:

.. code-block:: python

   from vane.viz import (
       plot_campbell,
       plot_campbell_panels,
       plot_campbell_static,
       save_figure,
   )

   fig = plot_campbell_static(diagram, operating_range=(6.9, 12.1), size="single")
   save_figure(fig, "campbell", size="single")      # campbell.pdf, .svg and .png
   save_figure(plot_campbell_panels(diagram, size="double"), "campbell_panels.pdf")
   plot_campbell(diagram, operating_range=(6.9, 12.1)).write_html("campbell.html")

Each track takes the colour of its mode family (blades green, tower black, platform red,
drivetrain magenta, other modes cyan), the line style cycles within a family, and the
marker shows the multi-blade type (circle collective, downward triangle regressive,
upward triangle progressive). The ``nP`` lines are thin, blue ("Blade Passing") and
tagged inline. The static figure names each mode at its line end and keeps the key to
the families and the overlays; the operating window, when given, is shaded and marked.
Resonance crossings are filled crosses: red for high severity, orange for medium and grey
for low. The excitation lines and crossings are drawn only for a rotor-speed diagram.
See :doc:`visualization` for the plot options, the encoding, the figure sizes and how to
change the style, and :doc:`plot_gallery` for examples.

Export and the command line
---------------------------

:func:`~vane.export.campbell_table` flattens a diagram into a long-format
:class:`pandas.DataFrame`, one row per track and operating point, with the columns
``track``, ``label``, ``multiblade``, the operating parameter (named after
``parameter_name``), ``natural_frequency_hz``, ``damping_ratio``, ``confidence`` and
``is_ambiguous``. :func:`~vane.export.resonance_table` does the same for the crossings,
and :func:`~vane.export.write_table` writes either table to ``.csv``, ``.json``,
``.xlsx`` (with the ``excel`` extra) or ``.mat``, chosen by the file suffix:

.. code-block:: python

   from vane.export import campbell_table, resonance_table, write_table

   write_table(campbell_table(result.campbell), "campbell.csv")
   write_table(resonance_table(result.resonances), "resonances.csv")

The ``vane campbell`` command runs the pipeline over a directory of ``.lin`` files and
prints the Campbell table, or writes it with ``--output`` (``-o``):

.. code-block:: console

   $ vane campbell lin_files --output campbell.csv
   $ vane campbell lin_files --parameter wind -o campbell.xlsx

``--parameter`` (``-p``) is ``rpm`` (the default) or ``wind``. The ``vane study``
command writes ``campbell.csv`` together with the provenance record and an HTML report
that contains the plotted diagram.

Azimuth-spread uncertainty
--------------------------

The averaged model leaves out how the rotor's state matrix varies with azimuth.
:func:`~vane.modal.uncertainty.azimuth_spread` estimates, per mode, the standard
deviation of natural frequency and damping ratio across the sampled azimuths. It needs
the per-azimuth matrices, which the pipeline does not keep, so the MBC transform is run
again with ``retain_per_azimuth=True``. The plots have no built-in uncertainty band; the
spread can be added as error bars:

.. code-block:: python

   import numpy as np
   import plotly.graph_objects as go

   from vane.io import mbc3_transform
   from vane.modal import azimuth_spread

   # The same order as the pipeline: ascending rotor speed.
   ordered = sorted(operating_points, key=lambda p: np.mean([f.rotor_speed for f in p]))
   spreads = [azimuth_spread(mbc3_transform(p, retain_per_azimuth=True)) for p in ordered]

   fig = plot_campbell(result.campbell)
   for track in result.campbell.tracks:
       x, frequency_hz, _ = result.campbell.track_curve(track)
       sigma = [
           spreads[op].natural_frequency_std[mode]
           for op, mode in zip(track.operating_points, track.mode_indices)
       ]
       fig.add_trace(
           go.Scatter(
               x=x, y=frequency_hz, mode="markers", showlegend=False,
               error_y={"type": "data", "array": sigma},
           )
       )

The spread is zero for an isotropic rotor and grows with rotor anisotropy. For the exact
treatment of the periodic system, see :doc:`azimuth_averaging` and
:doc:`../theory/floquet`.
