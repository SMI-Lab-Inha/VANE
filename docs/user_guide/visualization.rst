.. SPDX-License-Identifier: Apache-2.0

Visualization
=============

Each figure in :mod:`vane.viz` is drawn from the identified modes, the tracking results
and the operating points behind them, so what it shows can be traced back to the
analysis. Every plot function returns a figure object that can be shown, saved or
embedded.

Every plot exists in two forms. The interactive form returns a Plotly figure
(``plotly.graph_objects.Figure``) for exploring results on screen, with hover labels on
every point. The static form returns a Matplotlib figure for reports and papers; its name
ends in ``_static`` (for example :func:`~vane.viz.campbell_plot.plot_campbell_static`),
except :func:`~vane.viz.modeshape_static.plot_mode_participation`, which is static
only. Static figures are drawn at a printed size (``size="single"`` for one journal
column, ``"double"`` for the full page width) and written with
:func:`~vane.viz.save_figure`. All figures share one style (see
:ref:`figure-style`), and :doc:`plot_gallery` shows each plot at both sizes.

The examples below assume a completed pipeline run (see :doc:`quickstart`):

.. code-block:: python

   from vane import ModalPipeline

   result = ModalPipeline().run(operating_points)
   diagram = result.campbell        # a CampbellDiagram
   solution = result.solutions[0]   # the ModalSolution at the first operating point

Campbell diagram
----------------

:func:`~vane.viz.campbell_plot.plot_campbell_static` draws the Campbell diagram with the
labelled mode tracks, the per-rev (``nP``) excitation lines, and the resonance crossings
where a track meets a line; :func:`~vane.viz.campbell_plot.plot_campbell` draws the same
diagram as an interactive figure.

.. code-block:: python

   from vane.viz import plot_campbell, plot_campbell_static, save_figure

   fig = plot_campbell_static(
       diagram,
       harmonics=(1, 3, 6, 9),          # per-rev excitation orders to draw
       operating_range=(6.9, 12.1),     # shade this window; search resonances in it
       size="double",                   # 17 cm wide, 9 pt labels
   )
   save_figure(fig, "campbell", size="double")   # campbell.pdf, .svg and .png

   plot_campbell(diagram, harmonics=(1, 3, 6, 9)).show()

The figure follows the lab's Campbell conventions:

* each track is coloured by its mode family (blades green, tower black, platform red,
  drivetrain magenta, other modes cyan), and the line style cycles within a family so
  that neighbouring lines of one family can be told apart;
* the marker shows the multi-blade type (circle collective, downward triangle
  regressive, upward triangle progressive);
* the ``nP`` lines ("Blade Passing") are thin and blue and are tagged inline (``3P``)
  where they are clear of the tracks, rather than listed in the legend;
* the frequency axis ends 30 % above the highest mode, so the steep ``nP`` lines run off
  the top and the modes fill the figure;
* in the static figure each mode is named at its line end, in a column to the right of
  the axes, moved up or down only as far as needed to keep the names apart, with a thin
  leader back to the line. In figures narrower than 12 cm the names are shortened
  (``"1st flap reg."``);
* a compact key above the plot explains the families, Blade Passing and any overlay
  used: resonance crossings (filled crosses coloured by severity), low identification
  confidence (reduced opacity), ambiguity (open diamonds) and heavily damped tracks
  (dotted grey).

``operating_range`` restricts the resonance search to the given window of the operating
parameter and shades it grey, with a labelled double arrow ("Operating Speed Range", or
"Operating Wind Range" on a wind-speed axis); ``show_operating_range=False`` keeps the
search window but drops the shading. The excitation lines and resonance markers are
drawn only when the diagram's parameter is rotor speed.

On a rotor-speed diagram the per-rev lines are the rays ``f = n · rpm / 60``. Structural
modes appear as near-horizontal tracks, and the regressive and progressive blade modes
as the branches that slope with rotor speed. Resonance markers are coloured by severity
(see :doc:`campbell_diagrams`). The ``show_excitation``, ``show_resonances``,
``show_confidence`` (line opacity scaled by track confidence) and ``show_ambiguity``
(open-diamond markers on ambiguous tracks and points) flags turn each overlay on or off.
No title is drawn unless ``title=`` is given, so that the caption stays in the document.

The interactive figure keeps a legend grouped by family, so that a family or a single
track can be switched off by clicking it, and its hover label gives the mode, the
operating point, the frequency, the damping ratio and the identification confidence.
``figure=`` draws into an existing Plotly figure and ``template=`` replaces the VANE
Plotly template. In the static figure, ``ax=`` draws into existing axes (the mode names
then stand just inside the right edge) and ``direct_labels=False`` replaces the names
and the key with a legend of one entry per track, grouped by family.

Heavily damped (near-critical) modes, for example a first blade flap whose aerodynamic
damping has passed critical in the rotating frame, are tracked apart from the lightly
damped lines (see :doc:`../theory/modal_identification`) and stored in
``diagram.heavily_damped_tracks``. Both Campbell plots draw them as dotted grey lines
with open markers; the interactive hover adds the damped frequency. They are never
marked as resonances. Pass ``show_heavily_damped=False`` to hide them,
``build_campbell(..., include_heavily_damped=False)`` to leave them out of the diagram,
or ``ModalPipeline(heavily_damped_threshold=None)`` to track every mode together.

Damping and stability
---------------------

:func:`~vane.viz.damping_plot.plot_damping_static` and
:func:`~vane.viz.damping_plot.plot_damping` plot the damping ratio of every track against
the operating parameter, with stability bands that show where a mode approaches or
crosses zero damping, and red crosses at unstable points. The tracks use the encoding of
the Campbell diagram, and the static figure names them at their line ends. The
interactive figure has a title only as a warning, naming the modes that become unstable.

.. code-block:: python

   from vane.viz import plot_damping_static

   fig = plot_damping_static(
       diagram,
       as_percent=True,            # damping ratio in %
       show_stability_bands=True,  # shade the unstable and caution zones
       caution_damping=0.01,       # upper edge of the caution band, as a fraction
       size="single",
   )

Frequency and damping together
------------------------------

:func:`~vane.viz.campbell_plot.plot_campbell_panels` draws the Campbell diagram (a) and
the damping ratios (b) as two panels on a shared operating-parameter axis, labelled
``(a)`` and ``(b)`` outside their top-left corners, with one key for both. At
double-column width the panels stand side by side; at single-column width they are
stacked (``layout="row"`` or ``"column"`` chooses explicitly).

.. code-block:: python

   from vane.viz import plot_campbell_panels, save_figure

   fig = plot_campbell_panels(diagram, harmonics=(1, 3, 6), size="double")
   save_figure(fig, "campbell_panels.pdf")

:func:`~vane.viz.add_panel_labels` adds the same panel labels to a
multi-panel figure of your own.

Mode shapes
-----------

VANE shows a mode in two ways: the deflected shape on the physical structure, and
per-DOF views of which degrees of freedom make up the mode and their relative phase.
Following the lab's mode-shape conventions, the mode is named with its natural
frequency (``Mode 2 (0.5564 Hz)``) in the legend, the degrees of freedom carry short
names (``Tower FA1``, ``Flap 1 cos.``; for the participation chart with
``short_names=True``) and each series takes the next standard line colour.

Deflected shape on the structure:

* :func:`~vane.viz.modeshape_fowt.plot_mode_fowt` and
  :func:`~vane.viz.modeshape_fowt.plot_mode_fowt_animation` draw the mode on a
  reference geometry of the whole turbine (tower, rotor, platform, mooring), at one
  phase or animated. :func:`~vane.viz.modeshape_fowt.plot_mode_fowt_static` draws the
  single-phase view with Matplotlib, coloured by member as the mode families (tower
  black, blades green, platform red), on equal axes sized to the turbine.

Per-DOF composition and phase:

* :func:`~vane.viz.modeshape_static.plot_mode_participation` is a signed
  participation bar chart of the most active degrees of freedom (static only).
  :func:`~vane.viz.modeshape_static.plot_mode_shape` is an alias for it; it plots
  participation, not geometry.
* :func:`~vane.viz.modeshape_phasor.plot_mode_phasor` (static:
  :func:`~vane.viz.modeshape_phasor.plot_mode_phasor_static`) and
  :func:`~vane.viz.modeshape_phasor.plot_mode_phasor_animation` draw the dominant
  degrees of freedom as phasors on a polar wheel.
* :func:`~vane.viz.modeshape_phasor.plot_blade_components` (static:
  :func:`~vane.viz.modeshape_phasor.plot_blade_components_static`) shows how a mode's
  participation divides among the collective, cosine and sine blade coordinates.
* :func:`~vane.viz.modeshape_3d.plot_mode_3d` (static:
  :func:`~vane.viz.modeshape_3d.plot_mode_3d_static`) draws the most active degrees of
  freedom as 3-D complex phasors (a per-DOF phase view, not a structural deflection).

Animated whole-turbine mode shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~vane.viz.modeshape_fowt.plot_mode_fowt_animation` animates a mode on the
structure, with a play/pause control and a frame slider.

.. code-block:: python

   from vane.viz import plot_mode_fowt_animation

   fig = plot_mode_fowt_animation(
       solution,
       mode_index=0,
       motion="swing",         # "cycle" (default) = complex orbit; "swing" = standing wave
       scale=2.0,              # magnification factor for the deflection
       deflection_ratio=0.12,  # peak deflection as a fraction of the structure size
       n_frames=36,
   )
   fig.show()

The ``motion`` option selects the kind of motion drawn:

* ``"cycle"`` sweeps the complex phase to show the orbit of each node (a travelling
  motion for complex modes).
* ``"swing"`` scales the real peak-deflection field by a cosine, giving a standing-wave
  swing about the undeformed shape.

``scale`` and ``deflection_ratio`` set the visual magnification. A mode shape has
arbitrary amplitude, so the deflection is scaled to a visible fraction of the structure
size. An unrecognised ``motion`` raises ``ValueError``.

If no ``geometry`` is given, VANE infers a reference geometry from the mode's degrees of
freedom (floating or fixed-bottom) with :func:`~vane.viz.modeshape_fowt.infer_geometry`.
Pass an explicit geometry to draw the correct support structure (see below).

Animation in a VTK viewer
~~~~~~~~~~~~~~~~~~~~~~~~~

For a VTK viewer, :func:`~vane.viz.vtk_export.write_mode_animation_vtk`
writes the mode's oscillation as a time series: one legacy-VTK ``POLYDATA`` frame per
phase step plus a ``.pvd`` collection that ties them together. Opening the
``.pvd`` plays the motion.

.. code-block:: python

   from vane.viz import write_mode_animation_vtk

   # points: (n_nodes, 3) undeformed node coordinates; displacement: the (real or complex)
   # mode-shape vector mapped onto those nodes; lines: optional polyline connectivity.
   frames = write_mode_animation_vtk(
       points, displacement, "mode.pvd",
       n_frames=24, scale=2.0, lines=[[0, 1, 2]],
   )
   # writes mode.0000.vtk ... mode.0023.vtk next to mode.pvd; open mode.pvd in a VTK viewer

The motion at phase :math:`\theta` is
:math:`\operatorname{Re}(\mathbf{d})\cos\theta - \operatorname{Im}(\mathbf{d})\sin\theta`:
a real mode swings as a standing wave, a complex one traces its orbit. The writer needs no
``vtk`` package and shares its serializer with
:func:`~vane.viz.vtk_export.write_mode_shape_vtk`, which writes a single static deformed
frame. A static frame needs a real displacement:
:func:`~vane.viz.vtk_export.write_mode_shape_vtk` raises ``ValueError`` for a complex
field with a non-negligible imaginary part instead of keeping only its real part. To
freeze a complex mode at phase :math:`\theta`, pass ``np.real(d * np.exp(1j * theta))``.

Reference geometries for support structures
-------------------------------------------

The whole-turbine views draw the mode on a reference geometry: a parametric schematic of
the turbine and its support structure. :func:`~vane.config.system_config.default_geometry`
returns a preset for each support type:

.. code-block:: python

   from vane.config import SystemType, default_geometry
   from vane.viz import plot_mode_fowt_animation

   geometry = default_geometry(SystemType.JACKET)
   fig = plot_mode_fowt_animation(solution, 0, geometry=geometry)

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - ``SystemType``
     - Builder
     - Structure
   * - ``LAND``
     - :func:`~vane.config.system_config.onshore_geometry`
     - Onshore tower on the ground plane
   * - ``MONOPILE``
     - ``default_geometry(SystemType.MONOPILE)`` [#monopile]_
     - Offshore monopile through the water column
   * - ``FIXED``
     - :func:`~vane.config.system_config.monopile_geometry`
     - Bottom-fixed pile with the seabed at ``z = 0`` (no water column)
   * - ``JACKET``
     - :func:`~vane.config.system_config.jacket_geometry`
     - Battered-leg lattice with X-braces
   * - ``SPAR``
     - :func:`~vane.config.system_config.spar_geometry`
     - Deep spar buoy with catenary mooring
   * - ``SEMI``
     - :func:`~vane.config.system_config.semisub_geometry`
     - Semi-submersible with spread mooring
   * - ``TLP``
     - :func:`~vane.config.system_config.tlp_geometry`
     - Tension-leg platform with vertical taut tendons

Each builder takes the turbine's principal dimensions (hub height, blade length, draft,
water depth, mooring layout and so on) as keyword arguments, so the schematic can be
matched to a specific turbine. Offshore presets draw a still-water plane at ``z = 0``
with the seabed at ``-water_depth``; the onshore preset places the tower base on the
ground plane with no water or mooring. Call
:func:`~vane.config.system_config.default_geometry` for the preset of a support type, or
a builder directly to change the dimensions.

.. [#monopile] ``default_geometry(SystemType.MONOPILE)`` returns the offshore monopile
   through the water column. Calling :func:`~vane.config.system_config.monopile_geometry`
   directly with its defaults returns the bottom-fixed geometry (``SystemType.FIXED``,
   seabed at the ``z = 0`` reference plane, no water column); pass
   ``mudline_z=-water_depth`` to place it through the water column.

Rotor sign conventions
~~~~~~~~~~~~~~~~~~~~~~

The schematic rotor follows OpenFAST's conventions, so blade-resolved and multi-blade
(MBC3) modes are drawn on the blade they belong to and deflect the right way:

* **Azimuth.** Blade *b* sits at azimuth :math:`\psi_b = 2\pi (b-1)/N`, a right-hand
  rotation about the downwind shaft axis from vertical. Blade 1 points up and, for a
  three-bladed rotor, blade 2 points to ``-y`` (right, looking downwind). This is the
  azimuth the MBC3 transform uses, so a sine-cyclic coordinate deflects the ``-y`` side
  of the rotor in the positive direction.
* **Blade bending.** Positive flap points nominally downwind (``+x``) and positive edge
  points against the direction of rotation (toward the trailing edge), as for the
  ElastoDyn blade frame. Tower fore-aft and side-side bending are positive along ``+x``
  and ``+y``.
* **Shaft tilt and precone** take the ElastoDyn ``ShftTilt`` and ``PreCone`` signs:
  positive tilt raises the downwind shaft end and positive precone cones the blades
  downwind. The defaults (``shaft_tilt_deg=-5``, ``precone_deg=-2.5``) describe an
  upwind rotor tilted nose-up with blades coned upwind, away from the tower. Pass the
  values from an ElastoDyn input file unchanged.


Diagnostics
-----------

These figures support a review of the identification itself; each has a static form
(``*_static``) as well:

* :func:`~vane.viz.diagnostics.plot_mac_matrix` draws the MAC matrix between two
  solutions, for example adjacent operating points, as a heatmap with the modes labelled
  by frequency. A near-diagonal pattern means the modes track cleanly; off-diagonal
  values flag crossings or veering.
* :func:`~vane.viz.diagnostics.plot_tracking_confidence` is a bar chart of each track's
  identification confidence, with ambiguous tracks in orange and the review threshold
  dashed.
* :func:`~vane.viz.diagnostics.plot_branch_review` draws one zoomed panel per crossing or
  veering group (see :doc:`mode_identification`).
* :func:`~vane.viz.diagnostics.plot_azimuth_coverage` shows the azimuths sampled at each
  operating point of a study, from ``study.provenance``.
* :func:`~vane.viz.qoi_plot.plot_load_mode_heatmap` shows how much each mode contributes
  to each load of a :class:`~vane.sysid.qoi.QoiMap` (see :doc:`state_space_export`).

.. code-block:: python

   from vane.viz import (
       plot_branch_review_static,
       plot_mac_matrix_static,
       plot_tracking_confidence,
   )

   plot_mac_matrix_static(result.solutions[0], result.solutions[1], size="single")
   plot_tracking_confidence(diagram).show()
   plot_branch_review_static(diagram, result.identification.branch_groups)

.. _figure-style:

Figure style
------------

All figures follow the plotting conventions of the Marine Structural Mechanics and
Integrity Lab (SMI Lab), Inha University, so that the lab's tools share one look. The
style is defined once in :mod:`vane.viz`:

* the standard engineering-paper line colours, in order: black, red ``(0.85, 0, 0)``,
  blue ``(0, 0, 0.85)``, green ``(0, 0.6, 0)``, magenta ``(0.8, 0, 0.8)``, orange
  ``(1, 0.5, 0)`` and cyan ``(0, 0.7, 0.7)`` (:data:`vane.viz.STANDARD_LINES`). Black
  comes first, so that a single curve reads as line art, and the colours stay apart when
  printed in grey;
* Helvetica or Arial (Liberation Sans or DejaVu Sans where neither is installed), 10 pt
  text with 9 pt tick labels and legends, and normal-weight titles;
* boxed axes with a soft dark-grey frame, ticks pointing inward on all four sides,
  1.0 pt lines and 5 pt markers, and a white background;
* no grid unless switched on with ``ax.grid(True)``, which then draws dashed light grey
  lines;
* legends with a thin grey border and an opaque white fill;
* saved figures at 300 dpi, trimmed to their content.

The interactive figures apply the Plotly template ``"vane"``, which mirrors this style
(the same colours and fonts, boxed axes with mirrored inward ticks, a white background
and the same legend frame) and is registered in ``plotly.io.templates`` when
:mod:`vane.viz` is imported. The static figures are drawn inside the context manager
:func:`~vane.viz.plot_style`, which sets the ``rcParams`` of :func:`~vane.viz.style_rc`
for the figure being created and restores the previous values afterwards, so the
global Matplotlib configuration is not changed.

The same style can be used for other figures:

.. code-block:: python

   import matplotlib.pyplot as plt
   import plotly.graph_objects as go
   import plotly.io as pio

   from vane.viz import apply_style, plot_style

   with plot_style():                    # Matplotlib, inside this block only
       fig, ax = plt.subplots()
       ax.plot(time, tension)
       ax.set_xlabel("Time (s)")
       ax.set_ylabel("Tension (kN)")

   apply_style(fig)                      # restyle an existing figure or axes in place
   apply_style()                         # Matplotlib, for the rest of the session

   go.Figure(layout={"template": "vane"})   # Plotly, one figure
   pio.templates.default = "vane"           # Plotly, every new figure

Overriding the style
~~~~~~~~~~~~~~~~~~~~

* Plotly: every interactive plot function takes ``template=``. ``None`` (the default)
  applies ``"vane"``. Another template name, such as ``"plotly_white"``, or a
  ``plotly.graph_objects.layout.Template`` object replaces it.
  :func:`~vane.viz.plotly_template` returns a copy of the VANE template to modify.
* Matplotlib: the static plot functions take ``style=``. ``"vane"`` (the default) is
  the VANE style. ``None`` draws with the current ``rcParams``, so a
  ``plt.style.context`` opened around the call takes effect (and no size preset is
  applied unless ``size=`` is given). Any other value (a style name, a file path or a
  dictionary of ``rcParams``) is passed to ``matplotlib.style.context``. Axes passed with
  ``ax=`` keep their own box and ticks; call :func:`~vane.viz.apply_style` on them to
  restyle them.

.. code-block:: python

   import matplotlib.pyplot as plt

   from vane.viz import plot_campbell, plot_campbell_static

   fig = plot_campbell(diagram, template="plotly_white")

   with plt.style.context("ggplot"):
       fig = plot_campbell_static(diagram, style=None)

   fig = plot_campbell_static(diagram, style={"lines.linewidth": 2.0})

.. _figure-sizes:

Figure sizes and export
~~~~~~~~~~~~~~~~~~~~~~~

A figure for a paper is drawn at the width it is printed, so that its text is readable
without scaling. Every static plot function takes ``size=``:

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - ``size``
     - Width
     - Fonts
   * - ``"single"``
     - 8.5 cm (one column)
     - axis labels 9 pt, ticks and legends 8 pt
   * - ``"double"``
     - 17 cm (page width)
     - axis labels 9 pt, ticks and legends 8 pt
   * - a number
     - that width in cm
     - as above; 1 pt smaller below 8 cm

The height follows from the plot: three quarters of the width below 12 cm, about half
of it above, adjusted for plots with many rows. A Campbell or damping figure grows
taller when its direct labels need the room. :func:`~vane.viz.figure_size` resolves a
size request into a :class:`~vane.viz.FigureSize`.

:func:`~vane.viz.save_figure` writes a figure at a size preset. Without a suffix it
writes vector PDF and SVG and a PNG at 600 dpi; with a suffix, that file only. The file
has exactly the requested size. A Plotly figure is laid out at the preset size and fonts
and written through Plotly's static image export, which needs the separately installed
``kaleido`` package; without it :func:`~vane.viz.save_figure` raises ``ImportError``
with the install command.

.. code-block:: python

   from vane.viz import plot_campbell, plot_damping_static, save_figure

   fig = plot_damping_static(diagram, size="single")
   save_figure(fig, "figures/damping", size="single")     # .pdf, .svg and .png
   save_figure(fig, "figures/damping.png", dpi=300)       # one file

   save_figure(plot_campbell(diagram), "campbell.pdf", size="double")  # needs kaleido

Pass the same ``size`` to the plot function and to :func:`~vane.viz.save_figure`: the
plot function sets the fonts and lays out the labels for that size.

.. _campbell-family-styles:

Mode families in the Campbell and damping plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Campbell and damping plots, interactive and static, draw each track in the colour of
its mode family. The family follows from the DOF category of the track label
(:class:`~vane.config.dof_map.DofCategory`):

.. list-table::
   :header-rows: 1
   :widths: 16 12 72

   * - Family
     - Colour
     - DOF categories
   * - Blades
     - green
     - ``BLADE_FLAP_1``, ``BLADE_FLAP_2``, ``BLADE_FLAP``, ``BLADE_EDGE_1``,
       ``BLADE_EDGE``, ``BLADE_PITCH``, ``BLADE_TORSION``, ``BLADE_AXIAL``
   * - Tower
     - black
     - ``TOWER_FORE_AFT_1``, ``TOWER_SIDE_SIDE_1``, ``TOWER_FORE_AFT_2``,
       ``TOWER_SIDE_SIDE_2``
   * - Platform
     - red
     - ``PLATFORM_SURGE``, ``PLATFORM_SWAY``, ``PLATFORM_HEAVE``, ``PLATFORM_ROLL``,
       ``PLATFORM_PITCH``, ``PLATFORM_YAW``
   * - Drivetrain
     - magenta
     - ``GENERATOR_AZIMUTH`` (the rotor/generator mode), ``DRIVETRAIN_TORSION``
   * - Other
     - cyan
     - ``NACELLE_YAW``, ``ROTOR_FURL``, ``TAIL_FURL``, ``TEETER``, ``UNKNOWN``
   * - Blade Passing
     - blue
     - the ``nP`` excitation lines (not a mode family)

Within a family the line style cycles in the order of the tracks: solid, dashed,
dash-dotted, dotted, then three finer patterns. The marker shows the multi-blade type: a
circle for a collective mode and for modes without a multi-blade type, a downward
triangle for a regressive mode, an upward triangle for a progressive mode and a square
for a standing cyclic mode. The triangles point the way the whirling branches slope on a
rotor-speed diagram. In addition:

* line opacity scales with the identification confidence, from 0.35 at zero confidence
  to opaque at full confidence;
* ambiguous tracks, and the ambiguous points of clear tracks, have open-diamond markers;
* heavily damped tracks are dotted grey lines with open circles;
* resonance crossings are filled crosses, red for high severity, orange for medium and
  grey for low; unstable damping points are red crosses.

:func:`~vane.viz.mode_line_style` returns the style of a label, for use in other
figures; ``index`` is the position of the track within its family.

Colour-vision deficiency
~~~~~~~~~~~~~~~~~~~~~~~~

For readers with a colour-vision deficiency,
:func:`~vane.viz.palette.apply_colorblind_palette` recolours a finished Plotly figure
with the Okabe-Ito palette. Each standard colour is replaced by the Okabe-Ito colour of
the nearest hue (black stays black, red becomes vermillion, green becomes bluish green),
so the family distinctions are kept; greys and colour scales are left unchanged. The
line styles, markers and direct labels also tell the tracks apart without colour.

.. code-block:: python

   from vane.viz import apply_colorblind_palette, plot_campbell

   fig = apply_colorblind_palette(plot_campbell(diagram))

Style reference
~~~~~~~~~~~~~~~

.. py:data:: vane.viz.STANDARD_LINES
   :type: tuple[tuple[float, float, float], ...]

   The standard line colours as RGB triples in ``[0, 1]``: black, red, blue, green,
   magenta, orange, cyan.

.. py:data:: vane.viz.STANDARD_COLORS
   :type: tuple[str, ...]

   The same colours as hex strings.

.. py:data:: vane.viz.PLOTLY_TEMPLATE
   :type: str
   :value: "vane"

   Name of the registered Plotly template that every interactive figure applies.

.. autofunction:: vane.viz.plot_style

.. autofunction:: vane.viz.apply_style

.. autofunction:: vane.viz.style_rc

.. autofunction:: vane.viz.plotly_template

.. autofunction:: vane.viz.figure_size

.. autoclass:: vane.viz.FigureSize
   :members: rc

.. autofunction:: vane.viz.save_figure

.. autofunction:: vane.viz.add_panel_labels

.. autofunction:: vane.viz.mode_line_style

.. autofunction:: vane.viz.mode_family

.. autoclass:: vane.viz.ModeLineStyle
   :no-members:

.. autoclass:: vane.viz.ModeFamily
   :no-members:

Exporting figures and reports
-----------------------------

* :func:`~vane.viz.report.write_report` writes a self-contained HTML study report with
  the audit findings, the interactive Campbell diagram, the tracked-mode and resonance
  tables (with per-mode and per-crossing damping), the branch review, the tracking and
  azimuth diagnostics, and the provenance. Resonance crossings are only searched for when
  the study is swept along rotor speed (``parameter_name="rotor_speed_rpm"``). For a
  wind-speed sweep the report marks the resonance check *Not evaluated* rather than
  passing it. ``study.write_bundle(...)`` writes this report as ``report.html``.
* :func:`~vane.viz.report.write_report_pdf` writes the same findings, provenance and
  tables as a PDF. It needs the ``pdf`` extra (WeasyPrint). The PDF has no interactive
  Campbell figure; that is in the HTML report.
* :func:`~vane.viz.vtk_export.write_mode_shape_vtk` writes a deformed-mesh mode shape as
  legacy VTK for VTK viewers.
* :func:`~vane.viz.vtk_export.write_mode_animation_vtk` writes the same deformed shape as
  a ``.pvd`` time series (one VTK frame per phase step).

Plotly figures can be saved to standalone HTML with ``fig.write_html("figure.html")``;
static images of them, through :func:`~vane.viz.save_figure` or ``fig.write_image``,
need the ``kaleido`` package, which is installed separately.
