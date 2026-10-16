.. SPDX-License-Identifier: Apache-2.0

Mode identification
===================

:class:`~vane.pipeline.ModalPipeline` runs every stage on this page in one call (see
:doc:`quickstart`). This page takes the stages one at a time: reading the ``.lin``
files, the MBC3 transform, the eigensolution, participation, physical labels, tracking
across operating points, the branch review, and the confidence measures. The equations
behind each stage are on :doc:`../theory/modal_identification`.

.. contents:: On this page
   :local:
   :depth: 1

Reading linearization files
---------------------------

:func:`~vane.io.lin_reader.read_lin_file` parses one modern (ModGlue) OpenFAST ``.lin``
file into a :class:`~vane.io.lin_reader.LinFile`: the header scalars (``sim_time``,
``rotor_speed`` in rad/s, ``azimuth`` in rad, ``wind_speed`` in m/s), the
operating-point tables ``x``, ``xdot``, ``u`` and ``y``, and the matrices ``a``, ``b``,
``c`` and ``d`` (``None`` when OpenFAST did not write the block). A malformed, truncated
or legacy-format file raises :class:`~vane.io.lin_reader.LinFileFormatError`. See
:doc:`data_sources` for what the reader accepts.

One operating point is the set of files written over one rotor revolution
(``<case>.1.lin``, ``<case>.2.lin``, ...). :func:`~vane.study.discover_operating_points`
groups a directory of files by their ``<case>`` root and leaves out the per-module files
written under ``LinOutMod = True``:

.. code-block:: python

   from vane import discover_operating_points

   operating_points = discover_operating_points("lin_files")
   for point in operating_points:
       print(point.name, len(point))     # case name and number of azimuths

Each returned :class:`~vane.study.DiscoveredOperatingPoint` is a sequence of
:class:`~vane.io.lin_reader.LinFile` objects, so it can be passed to every function
below that takes the files of one operating point.

The MBC3 transform
------------------

:func:`~vane.io.mbc_transform.mbc3_transform` takes the parsed files of one operating
point, transforms the rotating blade states into multi-blade (collective, cosine, sine)
coordinates, and averages the result over the sampled azimuths:

.. code-block:: python

   from vane.io import mbc3_transform

   mbc = mbc3_transform(operating_points[0])
   print(mbc.rotor_speed_rpm, mbc.wind_speed, mbc.azimuths_deg)
   print(mbc.ndof2, mbc.ndof1, mbc.performed_transformation)

The returned :class:`~vane.io.mbc_transform.MBCResult` holds the averaged matrices
``avg_a``, ``avg_b``, ``avg_c`` and ``avg_d`` in the ``[q2, q2dot, q1]`` state layout
(see :doc:`coordinate_systems`), the reordered ``state_descriptions``, and the
multi-blade coordinate of each state in ``mbc_coordinates``. The keyword options are:

``retain_per_azimuth`` (default ``False``)
   Keep the transformed matrices at each azimuth in ``per_azimuth_a`` to
   ``per_azimuth_d``. The azimuth spread, the Floquet analysis and the averaging-error
   estimate need them (see :doc:`azimuth_averaging`).
``rotor_speed_drift_rtol``, ``wind_speed_drift_rtol`` (default ``0.01`` each)
   The relative spread of rotor speed and wind speed allowed across the files before
   they are rejected as more than one operating point.
``omega_dot`` (default ``0.0``)
   Rotor acceleration in rad/s², zero for a steady periodic operating point.

The transform raises ``ValueError`` for duplicate azimuths (including a ``0``/``360``
deg pair), for inconsistent channel layouts, and for rotor- or wind-speed drift. It logs
a warning, and continues, when a rotating operating point has fewer than three azimuths
or a non-uniform azimuth spacing. With no complete blade triplet (for example a one- or
two-bladed rotor) it returns a plain azimuth average in the original state order, with
``performed_transformation=False``.

Selecting states
----------------

:func:`~vane.io.state_selection.select_states` keeps or removes states by a
case-insensitive regular expression on the state description or by degree-of-freedom
class (:class:`~vane.config.dof_map.DofCategory`, as assigned by
:func:`~vane.config.dof_map.classify_dof`). It works on a :class:`~vane.io.lin_reader.LinFile`
before the MBC3 transform and on an :class:`~vane.io.mbc_transform.MBCResult` after it,
and returns a copy in which ``A`` keeps the selected rows and columns, ``B`` the
selected rows and ``C`` the selected columns; ``D``, the inputs and the outputs are unchanged.

.. code-block:: python

   from vane.io import mbc3_transform, read_lin_file, select_states

   # Before the transform: remove the HydroDyn radiation states at every azimuth.
   sweep = [select_states(read_lin_file(p), drop=r"^HD ") for p in lin_paths]
   result = mbc3_transform(sweep)

   # After it: keep only the first tower modes.
   tower = select_states(
       result, keep_categories=["tower_fore_aft_1", "tower_side_side_1"]
   )

The displacement and velocity of a second-order degree of freedom are always selected
together, so the ``[q2, q2dot, q1]`` layout stays consistent; for an ``MBCResult``,
``ndof2``, ``ndof1``, the MBC coordinate labels and the per-azimuth matrices follow the
selection. A pattern or class that matches no state raises ``ValueError``, as does a
selection that would remove every state.

The eigensolution
-----------------

:func:`~vane.modal.eigensolver.modes_from_mbc` solves the eigenvalue problem of
``avg_a`` and returns a :class:`~vane.modal.eigensolver.ModalSolution`. For a state
matrix from another source, :func:`~vane.modal.eigensolver.compute_modes` takes the
matrix and the ``ndof2``/``ndof1`` block sizes directly.

.. code-block:: python

   from vane.modal import modes_from_mbc

   solution = modes_from_mbc(mbc)
   for k in range(solution.n_modes):
       print(
           f"{solution.natural_frequencies_hz[k]:.3f} Hz, "
           f"zeta = {solution.damping_ratios[k]:.4f}"
       )

Only oscillatory modes are kept, one per complex-conjugate pair, sorted by natural
frequency. The main attributes are:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Attribute
     - Content
   * - ``eigenvalues``
     - Retained eigenvalues (positive imaginary part), shape ``(n_modes,)``.
   * - ``natural_frequencies_hz``, ``damped_frequencies_hz``, ``damping_ratios``
     - Per-mode modal parameters.
   * - ``mode_shapes``
     - Complex shapes over the displacement and first-order states, shape
       ``(ndof2 + ndof1, n_modes)``; the rows are named by ``dof_descriptions`` and
       ``dof_mbc_coordinates``.
   * - ``full_eigenvectors``, ``full_left_eigenvectors``
     - Right and left eigenvectors over all states, named by ``state_descriptions``.
   * - ``condition_numbers``, ``is_degenerate``
     - Eigenvalue condition number and a flag for near-repeated eigenvalues.
   * - ``n_unstable``, ``n_overdamped``, ``n_rigid_body_modes``
     - Counts of eigenvalues with positive real part, of real stable eigenvalues, and
       of degrees of freedom with no oscillatory mode.
   * - ``n_split_rigid_body_pairs``
     - Complex pairs below ``rigid_body_frequency_hz`` (default 3 mHz) whose imaginary
       part the data does not resolve. They are counted as rigid-body eigenvalues, not
       as modes.
   * - ``rotor_speed_rpm``
     - Rotor speed copied from the MBC result.

Participation
-------------

:func:`~vane.modal.participation.compute_participation` measures how strongly each
degree of freedom takes part in each mode from the magnitudes of the mode shape. The
:class:`~vane.modal.participation.ParticipationResult` holds the normalized
``magnitude`` (each column's maximum is 1), a ``signed_magnitude`` (negative for
components more than 90 degrees out of phase with the dominant one), ``phase_deg``, and
the ``dominant_state`` of each mode.

Blade and tower deflections are in metres while the other degrees of freedom are
rotations or generalized coordinates, so raw magnitudes are not comparable.
:func:`~vane.modal.participation.participation_scale_factors` divides blade deflections
by the blade length and tower deflections by the tower length:

.. code-block:: python

   from pathlib import Path

   from vane.io import read_elastodyn_geometry
   from vane.modal import participation_from_modes, participation_scale_factors

   geometry = read_elastodyn_geometry(Path("model/ElastoDyn.dat"))
   factors = participation_scale_factors(
       solution.dof_descriptions,
       blade_length=geometry.blade_length,
       tower_length=geometry.tower_length,
   )
   participation = participation_from_modes(solution, factors)

The labels below do not use these magnitudes. They use the biorthogonal (left-right)
participation of :func:`~vane.modal.biorthogonal.biorthogonal_participation`, which does
not depend on the units of the states.

Physical labels
---------------

:func:`~vane.modal.labeler.label_solution` returns one
:class:`~vane.modal.labeler.ModeLabel` per mode:

.. code-block:: python

   from vane.modal import label_solution

   for label in label_solution(solution):
       print(label.label, label.category.name, f"{label.confidence:.2f}")
       if label.is_ambiguous:
           print("   could also be", label.alternative)

The fields are:

``label``, ``category``
   The name (for example ``"1st Tower Fore-Aft"`` or ``"1st Blade Flap (Regressive)"``)
   and the dominant :class:`~vane.config.dof_map.DofCategory`.
``confidence``
   The share of the mode's participation carried by the winning category, in
   ``[0, 1]``. A low value marks a mixed mode.
``dominant_dofs``
   Descriptions of the (up to three) most active degrees of freedom.
``multiblade``
   ``"collective"``, ``"regressive"``, ``"progressive"`` or ``"cyclic"`` (a standing
   cyclic pattern with no whirl direction) for a blade mode, otherwise ``None``.
``is_ambiguous``, ``alternative``, ``alternative_category``, ``alternative_confidence``
   The runner-up label and its share. ``is_ambiguous`` is ``True`` when the runner-up
   carries at least ``LABEL_AMBIGUITY_RATIO`` (0.75) of the winner's weight, when the
   multi-blade type is undecided, or when the per-point multi-blade assignment changed
   the mode's own type.

A mode receives a physical label only if classified degrees of freedom carry at least
``min_known_participation`` (default ``0.5``) of its participation; otherwise it is
``"Unidentified"``. The tower modes of each plane are numbered in frequency order, and
each blade mode family has one collective, one regressive and one progressive member at
every operating point. :func:`~vane.modal.labeler.label_mode` and
:func:`~vane.modal.labeler.label_modes` apply the category vote to a single weight vector
or to a :class:`~vane.modal.participation.ParticipationResult`.

Heavily damped modes
--------------------

A mode whose damping ratio is at or above ``DEFAULT_HEAVILY_DAMPED_THRESHOLD`` (0.99)
barely oscillates: its natural frequency mostly measures its decay rate. A typical case
is a first blade flap whose aerodynamic damping exceeds critical in the rotating frame.
For blade whirl modes the damping is read in the rotating frame, where the blade mode is
defined. :func:`~vane.modal.identifier.heavily_damped_modes` returns the mask for one
operating point, and :func:`~vane.modal.identifier.rotor_frame_damping_ratios` returns
the damping ratios it compares with the threshold:

.. code-block:: python

   from vane.modal import heavily_damped_modes

   mask = heavily_damped_modes(solution)

The tracker keeps these modes out of the main tracks and links them among themselves
into ``heavily_damped_tracks``. Pass ``heavily_damped_threshold=None`` to
:func:`~vane.modal.identifier.identify_modes` or to
:class:`~vane.pipeline.ModalPipeline` to track every mode together.

Tracking across operating points
--------------------------------

:func:`~vane.modal.identifier.identify_modes` links the modes of successive operating
points into tracks, the lines of the Campbell diagram. The solutions must be in sweep
order:

.. code-block:: python

   from vane.io import mbc3_transform
   from vane.modal import identify_modes, modes_from_mbc

   mbc_results = sorted(
       (mbc3_transform(point) for point in operating_points),
       key=lambda result: result.rotor_speed_rpm,
   )
   solutions = [modes_from_mbc(result) for result in mbc_results]
   rpm = [result.rotor_speed_rpm for result in mbc_results]

   identification = identify_modes(solutions, parameter_values=rpm)
   for track in identification.tracks:
       print(track.label.label, track.natural_frequencies_hz, track.confidence)

A link between two modes is scored by their MAC, reduced by a frequency-continuity
penalty. The main options are:

``mac_threshold`` (default ``0.5``)
   Links with a lower MAC are not allowed.
``frequency_weight`` (default ``0.5``)
   Weight of the frequency-continuity penalty, in ``[0, 1]``.
``ambiguity_margin`` (default ``0.2``)
   A link is ambiguous when the best and second-best continuations differ in MAC by
   less than this.
``strategy`` (default ``"single_path"``)
   ``"single_path"`` extracts one optimal line at a time; ``"simultaneous"`` matches
   all modes of each transition at once and suits dense spectra with many crossings,
   such as floating-platform sweeps.
``continuity_metric`` (default ``"natural_frequency"``)
   ``"eigenvalue"`` measures the distance between modes in the complex plane, so a
   change of decay rate also counts.
``rigid_body_frequency_hz`` (default ``1e-3``)
   Modes below this frequency are not tracked.
``rotor_speeds_rpm``
   The rotor speed of each point, for the rotating-frame damping of whirl modes. By
   default each solution's own ``rotor_speed_rpm`` is used.

Each :class:`~vane.modal.identifier.ModeTrack` lists its ``operating_points``,
``mode_indices``, ``natural_frequencies_hz`` and ``damping_ratios``, a representative
``label``, the ``point_labels`` at each point, ``label_agreement``, a ``confidence``
(the weakest link MAC along the track), and ``is_ambiguous`` (a near-tie link, a
disagreeing point label, or a single-point track). The
:class:`~vane.modal.identifier.IdentificationResult` holds the ``tracks``,
``heavily_damped_tracks``, the per-point ``heavily_damped_modes``, and
``branch_groups``. ``parameter_values`` is used only by the branch resolution described
next.

Crossings and veerings
----------------------

Where two tracks cross or veer, the tracker is least certain which mode continues which
line. With ``branch_review=True``, :func:`~vane.modal.identifier.identify_modes` runs
:func:`~vane.modal.branch.detect_branch_groups` and stores the contested clusters as
:class:`~vane.modal.branch.BranchGroup` records in ``branch_groups``. A pair of tracks
is flagged only where their frequencies approach and their mode shapes compete. With
``resolve=True`` (which implies the review),
:func:`~vane.modal.branch.resolve_branch_groups` also attaches to each eligible group of
two to four tracks the minimum-curvature alternative, ``smooth_branches``, computed
against ``parameter_values``. Neither option changes the tracks.

.. code-block:: python

   identification = identify_modes(solutions, resolve=True, parameter_values=rpm)
   for group in identification.branch_groups:
       print(
           group.kind.value, group.track_indices, group.op_lo, group.op_hi,
           f"{group.min_separation_hz:.4f} Hz", group.resolved,
       )

:class:`~vane.pipeline.ModalPipeline` takes the same ``branch_review`` and ``resolve``
options, and the ``vane study`` command takes ``--branch-review`` and ``--resolve``; the
study report then lists the groups. :func:`~vane.viz.diagnostics.plot_branch_review`
draws one panel per group.

MAC and MACXP
-------------

:func:`~vane.modal.mac.compute_mac` returns the MAC matrix between two sets of mode
shapes, and :func:`~vane.modal.mac.compute_macxp` the pole-weighted MACXP, which suits
complex state-space modes. MACXP rows and columns of undamped poles are ``nan``.
:func:`~vane.modal.mac.match_modes` pairs two mode sets by an optimal assignment on a
correlation matrix:

.. code-block:: python

   from vane.modal import compute_mac, compute_macxp, match_modes

   a, b = solutions[0], solutions[1]
   mac = compute_mac(a.mode_shapes, b.mode_shapes)
   macxp = compute_macxp(a.mode_shapes, a.eigenvalues, b.mode_shapes, b.eigenvalues)
   pairs = match_modes(mac)                 # [(mode in a, mode in b), ...]

:func:`~vane.viz.diagnostics.plot_mac_matrix` draws the MAC of two solutions as a
heatmap.

Confidence and its calibration
------------------------------

:func:`~vane.modal.uncertainty.unified_mode_confidence` combines, per mode, the
eigenvalue conditioning, a degeneracy penalty, an optional azimuth-stability factor and
an optional tracking confidence into one value in ``[0, 1]``. The azimuth factor uses
the frequency spread from :func:`~vane.modal.uncertainty.azimuth_spread`, which needs an
MBC result computed with ``retain_per_azimuth=True``:

.. code-block:: python

   from vane.modal import azimuth_spread, unified_mode_confidence

   mbc = mbc3_transform(operating_points[0], retain_per_azimuth=True)
   solution = modes_from_mbc(mbc)
   spread = azimuth_spread(mbc)
   confidence = unified_mode_confidence(
       solution, frequency_spread=spread.natural_frequency_std
   )

This value differs from ``ModeLabel.confidence``, which is a participation share.

:func:`~vane.modal.confidence_calibration.calibrate_confidence` checks a confidence score
against known correct and incorrect outcomes and fits a monotone (isotonic) map from
confidence to the probability of being correct.
:func:`~vane.modal.confidence_calibration.recalibrate` applies that map, and
:func:`~vane.modal.confidence_calibration.confidence_threshold` returns the smallest
confidence that reaches a target accuracy:

.. code-block:: python

   from vane.modal import calibrate_confidence, confidence_threshold, recalibrate

   # correct[i] is True when mode i's label matches a known reference label.
   calibration = calibrate_confidence(confidence, correct)
   print(calibration.expected_calibration_error, calibration.brier_score)

   probability = recalibrate(calibration, confidence)
   threshold = confidence_threshold(calibration, 0.9)   # confidence for >= 90 % correct

``confidence_threshold`` raises ``ValueError`` when the calibrated probability never
reaches the target. The method is described on :doc:`../theory/campbell_uncertainty`.

Custom trackers and labellers
-----------------------------

:mod:`vane.plugins` defines two :class:`typing.Protocol` contracts for static type
checking:

* :class:`~vane.plugins.Tracker`: a callable taking the per-point solutions and a
  keyword ``parameter_values`` and returning an
  :class:`~vane.modal.identifier.IdentificationResult`.
  :func:`~vane.modal.identifier.identify_modes` conforms.
* :class:`~vane.plugins.Labeler`: a callable taking a
  :class:`~vane.modal.eigensolver.ModalSolution` and returning one
  :class:`~vane.modal.labeler.ModeLabel` per mode.
  :func:`~vane.modal.labeler.label_solution` conforms.

:class:`~vane.pipeline.ModalPipeline` accepts a custom tracker, which replaces the
built-in one. The pipeline's tracking options (``frequency_weight``, ``mac_threshold``,
``tracking_strategy``, ``branch_review``, ``resolve`` and ``heavily_damped_threshold``)
then do not apply:

.. code-block:: python

   from functools import partial

   from vane import ModalPipeline
   from vane.modal import identify_modes

   tracker = partial(identify_modes, strategy="simultaneous", mac_threshold=0.6)
   result = ModalPipeline(tracker=tracker).run(operating_points)

The pipeline has no labeller argument. The ``Labeler`` protocol is the contract for
code of your own that takes a labelling function as input.
