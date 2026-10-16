.. SPDX-License-Identifier: Apache-2.0

Limitations
===========

VANE states its assumptions explicitly so results are not over-interpreted.

Input format and rotor
----------------------

* **Modern ``.lin`` only.** The legacy OpenFAST linearization format is rejected.
* **Three-bladed rotors.** The MBC3 transform and the blade
  collective/regressive/progressive taxonomy assume a three-bladed rotor. Two- or
  many-bladed rotors are out of scope for the multi-blade coordinate path; purely
  non-rotating models (no blade triplets) are averaged without the transform.

Averaged-model assumptions
--------------------------

* The averaged non-rotating model approximates a linear time-periodic system by its
  azimuth average. It is exact for an isotropic rotor and degrades with anisotropy;
  the **azimuth-spread** uncertainty quantifies how much. A coherent sweep (distinct
  azimuths, constant rotor speed) is required and enforced.
* Adequate azimuth coverage is the user's responsibility; VANE rejects degenerate
  sweeps but does not synthesize missing azimuths. :func:`~vane.modal.averaging_error_estimate.estimate_averaging_error`
  estimates the error of the averaged model on a delivered azimuth grid (see
  :doc:`azimuth_averaging`).
* **Hill's method needs near-uniform azimuths.** :func:`~vane.modal.floquet_modes` and
  :func:`~vane.modal.averaging_error` require every azimuth interval to equal the mean
  interval within a relative :math:`10^{-3}` and raise a ``ValueError`` otherwise.
  Delivered OpenFAST grids are often uneven; analyse them with
  :func:`~vane.modal.monodromy_exponents`, which accepts any distinct azimuths (see
  :doc:`../theory/floquet`).

Excluding the linearization azimuths near tower passage is not a remedy for averaging
error. The true average over the revolution includes tower passage, so dropping those
samples biases the average rather than correcting it. The recommended practice is an
azimuth count that is not a multiple of the blade number, the interval-weighted mean on
the delivered grid, and pinning the azimuth direction in a dense average (see
:doc:`azimuth_averaging`).

Mode tracking
-------------

The cross-operating-point tracker offers two strategies. The default,
``strategy="single_path"``, extracts each Campbell line as the **globally optimal
single-path** continuity through the sweep, then removes it and repeats. This is
deterministic and avoids greedy adjacent-point errors, but it is a *sequential*
extraction: in dense spectra or at near-degenerate crossings, an earlier high-scoring
path can still claim a node a later path would have preferred.
``strategy="simultaneous"`` instead resolves each operating-point transition with one
maximum-weight assignment (the Hungarian algorithm), so no early link can strand a
mode a different pairing would have used. It is better suited to dense, crossing-rich
spectra such as floating-platform sweeps.

Passing ``branch_review=True`` adds a deterministic **branch-review** pass that flags
the crossing and veering hotspots where tracking is least certain — the clusters of
tracks whose frequencies approach *and* whose mode shapes genuinely compete — and
surfaces them (as ``branch_groups``, a diagnostic figure, and a report section with
per-hotspot warnings) for manual review. Adding ``resolve=True`` also computes, for an
eligible hotspot of up to four tracks, the **smooth (minimum-curvature) alternative** —
the non-crossing branch convention — and attaches it as ``smooth_branches`` for
comparison. Both passes only **report**: they never re-draw the tracks, and VANE does
not perform a simultaneous globally optimal assignment over the *whole*
frequency-vs-parameter grid at once, nor resolve clusters of more than four tracks.
Treat flagged hotspots, ambiguous links (``is_ambiguous``), and low-``confidence``
tracks as candidates for manual review.

Resonance detection
-------------------

Resonance detection reports **line crossings** between a mode curve and an ``nP``
excitation line, with the crossing rotor speed, frequency, and damping interpolated
linearly, a **damping-aware severity** (a lightly damped crossing is more dangerous
than a well-damped one), and optional **operating-window filtering**. The excitation
family defaults to 1P/3P/6P/9P and can be set to any positive-integer harmonics. It
does not provide closest-approach margin bands or uncertainty-propagated crossings.

Physical labels for BeamDyn blades
----------------------------------

BeamDyn blade modes are labelled flap, edge, or torsion from the blade-frame direction
of their nodal states, and numbered from the spanwise sign pattern of the section
rotation and their frequency order (see :doc:`../theory/modal_identification`). The
remaining limits are:

* **Flap/edge is the pitch-axis split.** ``X``/``Y`` of the BeamDyn frame follow the
  blade pitch but not the local structural twist, so a strongly twisted, bend--twist
  coupled blade can split a mode between the flap and edge families; such a mode is
  named after the larger one with a correspondingly lower confidence, and flagged
  ambiguous when the two shares are within
  :data:`~vane.modal.labeler.LABEL_AMBIGUITY_RATIO` of each other. For the reference
  blades (root twist up to about 15°) the leakage is below 7 % of a mode's
  participation. The linearization file carries no structural twist, so the split is
  not twist-resolved.
* **Nodal states carry rigid root motion.** The linearized nodal states include the
  motion the blade root carries. Participation keeps the tower and platform modes on
  their ElastoDyn DOFs, and the shape order is taken from rotation increments, which
  are blind to a rigid root rotation; a mode whose blades only turn rigidly is not
  numbered. A rigid *collective* edgewise mode is reported as *Rotor/Generator*
  (BeamDyn turbines with a free generator show the rotor-azimuth mode near 1P).
* **Drivetrain coupling.** With ``DrTrDOF`` enabled the drivetrain-torsion mode's
  participation sits in the BeamDyn nodal states rather than the ElastoDyn drivetrain
  DOF. It is recognized from the blade-root rotation (in phase with the tip for a
  locked generator, an elastic collective edgewise shape for a free generator; see
  :doc:`../theory/modal_identification`) and labelled *Drivetrain Torsion*. The rule
  is calibrated on the NREL 5 MW BeamDyn linearizations of the OpenFAST regression
  tests; a blade whose first
  collective edgewise mode itself rotates the hub in phase by more than a quarter of
  the tip rotation would be misread as drivetrain torsion.
* **Higher modes on coarse meshes.** The shape order needs enough nodes to resolve the
  sign changes; when it under-counts, the frequency order within the family and
  multi-blade type still numbers the mode correctly, but a missing lower mode can
  then only be detected up to the resolution of the mesh.

Labels on coupled models
------------------------

The physical label names a mode after a DOF category only when the classified
(ElastoDyn or BeamDyn) DOFs carry at least ``min_known_participation`` (default 0.5) of its
left--right participation. Modes dominated by aerodynamic, hydrodynamic radiation,
mooring, or controller states are reported as *Unidentified* by design. A genuinely
mixed mode is named after its largest classified category, with a low confidence and
``is_ambiguous=True`` when its runner-up carries at least
:data:`~vane.modal.labeler.LABEL_AMBIGUITY_RATIO` (0.75) of the winner's weight; the
runner-up is given as ``alternative``. For example, the NREL 5 MW
collective-edge/drivetrain mode near 1.7 Hz splits its participation roughly 48/43
between drivetrain torsion and blade edge and is labelled *Drivetrain Torsion*,
ambiguous, with *1st Blade Edge* as the alternative. Read ``is_ambiguous`` and the
confidence together with the label; a track's ``is_ambiguous``, ``label_agreement``
and ``point_labels`` show where along a Campbell line the identification is in
doubt.

Driving OpenFAST linearizations
-------------------------------

VANE can drive OpenFAST to produce its own ``.lin`` data
(:func:`~vane.run.auto_sweep.run_operating_sweep` and the lower-level :mod:`vane.run`
functions), but the linearization itself is bounded by OpenFAST physics and
configuration, not by VANE:

* **The rotor-performance schedule must be supplied.** VANE never guesses the rotor speed
  and blade pitch at a wind speed — the sweep requires an operating curve, a
  rotor-performance CSV, or a controller ``settle`` run, and raises an error before any
  OpenFAST run without one. VANE computes no rotor performance itself (it has no
  aerodynamic-performance or blade-element dependency); the schedule is the user's
  input.
* **Floating ``settle`` runs are slow.** Deriving the operating point by settling the
  fully-coupled controller is physics-bound: the lightly-damped platform modes take tens
  of minutes of simulated time per point to come to rest. A precomputed rotor-performance
  CSV or curve avoids the settle runs. The opt-in two-stage settle
  (``settle_platform_damping=``) damps the platform temporarily and then confirms the
  equilibrium without the damping. It shortens the wait for a decaying transient, but
  it cannot settle a point whose undamped model oscillates without end. Above rated, a
  pitch-regulated rotor can remove damping from the slow surge mode. The confirmation
  then reports that the oscillation returns, and ``settle_criterion="mean_drift"`` is
  the explicit way to linearize about its mean.
* **The default disables the drivetrain-torsion DOF.** The ``structural_dofs``
  default enables the blade and tower flexibility DOFs (a deck shipped with rigid
  blades, like the published IEA 3.4-MW one, would otherwise lose every blade mode)
  and turns off ``YawDOF`` (a spurious free nacelle-yaw rigid-body mode, since a
  linearization has no yaw controller) and ``DrTrDOF``. Disabling ``DrTrDOF`` drops the
  drivetrain-torsion mode (about 0.62 Hz on the NREL 5 MW turbine) from the extracted
  spectrum but markedly improves ``CalcSteady`` trim convergence, a deliberate
  trade-off. Pass
  ``structural_dofs={"DrTrDOF": True}`` to recover the drivetrain-torsion mode (at some
  convergence cost) while keeping the free-yaw mode suppressed.
* **A non-converged trim is not averaged.** A periodic (``CalcSteady``) operating point
  is sampled at ``NLinTimes`` azimuths and MBC-averaged. When the trim does not
  converge, OpenFAST writes a single forced linearization instead. VANE treats this as a
  failed trim: it retries the point on the retry ladder and, if the point never
  converges, records it as ``converged=False`` rather than averaging the single file
  into a model.
* **The default inflow is steady and uniform.** Both the settle run and the
  linearization write ``PLExp = 0`` (``wind_shear_exponent=0.0``), so a settled point is
  linearized in the wind it settled in, but the model then carries no shear-induced
  once-per-revolution loading. Set ``wind_shear_exponent`` (used by both) for a sheared
  study, or ``None`` to keep the template deck's ``PLExp``.
* **Explicit times do not establish equilibrium.** Disabling ``CalcSteady`` and
  supplying a full-revolution ``linearization_times`` sequence only controls where
  OpenFAST evaluates the Jacobians. The user must first settle the rotating system.
  The driver accepts the run when every requested ``.lin`` file exists; to check the
  equilibrium as well, set ``LinearizationDriver(acceptance_policy=...)``
  (:class:`~vane.run.acceptance.LinearizationAcceptancePolicy`), which rejects a
  secular residual in the platform-rate states.

Scope
-----

VANE drives OpenFAST linearizations — configuring a deck for an operating point,
running the executable, and collecting the ``.lin`` output (:mod:`vane.run`) — and
post-processes them; it is not a model-building or design-assessment tool.

Full-system mode-shape visualization deforms the *entire* turbine — floating platform,
mooring lines, tower, nacelle, and blades — with the computed complex mode shape. It is
a communication and debugging aid, **not** a geometry-resolved structural-validation
artifact: the hull and mooring outlines are schematic and the tower/blade deflections
use Euler-Bernoulli (cantilever) shape functions, so the *relative* motion across the
structure is faithful to the computed mode while the absolute geometry is approximate.

The quantity-of-interest (load) residue map (:func:`~vane.sysid.qoi.build_qoi_map`) reports
each load's modal residues ``C·Φ`` — *which modes drive which loads*, an observability
and screening aid. It is **not** a fatigue or design-load ranking: that requires input
(modal-coordinate) scaling, unit normalization, environmental spectra or time-domain
loading, and design-load-case/fatigue context, none of which the linear residue alone
provides.
