.. SPDX-License-Identifier: Apache-2.0

State-space export
==================

Each operating point of a linearization is a linear model
:math:`\dot{\mathbf x} = \mathbf A\mathbf x + \mathbf B\mathbf u`,
:math:`\mathbf y = \mathbf C\mathbf x + \mathbf D\mathbf u` about its trimmed state.
:mod:`vane.sysid` exports that model, reduces it, attaches noise covariances, and builds
Kalman filters and operating-point (LPV) families from it. This page shows the main
steps; the equations are in :doc:`../theory/reduction_observability` and
:doc:`../theory/lpv_estimation`. The examples start from the ``result`` of
:class:`~vane.pipeline.ModalPipeline` (see :doc:`quickstart`).

.. contents:: On this page
   :local:
   :depth: 1


The full-order model
--------------------

:meth:`~vane.pipeline.PipelineResult.state_space` returns the azimuth-averaged (MBC)
model of one operating point as a :class:`~vane.sysid.state_space.StateSpace`:

.. code-block:: python

   system = result.state_space(0)          # index into result.mbc_results
   print(system.n_states, system.n_inputs, system.n_outputs, system.is_stable())
   a, b, c, d = system.a, system.b, system.c, system.d
   print(system.state_names[:3], system.input_names[:3], system.output_names[:3])

The names are the ``.lin`` descriptions, in the MBC frame: a blade-triplet channel
appears as collective, cosine and sine coordinates, recorded per output in
``system.output_mbc_coordinates`` (``""`` for a channel that is not transformed).
``system.dt`` is ``None`` for a continuous-time model. The matrices act on perturbations
about the operating point, in the units of the ``.lin`` file (see
:doc:`units_conventions`). The absolute operating-point values are
``result.mbc_results[0].output_operating_point`` and ``input_operating_point``; subtract
them from measured channels before they reach a model.
:func:`~vane.sysid.state_space.state_space_from_mbc` does the same export from any
:class:`~vane.io.mbc_transform.MBCResult`, and
:func:`~vane.sysid.state_space.static_gain` returns the zero-frequency gain
:math:`\mathbf D - \mathbf C\mathbf A^{-1}\mathbf B`.

``system.discretized(dt)`` returns the zero-order-hold discretization at sample time
``dt`` (seconds), with :math:`\mathbf C` and :math:`\mathbf D` unchanged. A full
linearization usually contains unstable or marginal aerodynamic and azimuth states, so
it is rarely the model a filter should run: reduce it first.


Modal reduction
---------------

:func:`~vane.sysid.rom.modal_reduction` keeps the underdamped modes with the largest
residue in the outputs you care about, as a real block-diagonal model with two states per
mode and the original inputs and outputs:

.. code-block:: python

   from vane.export import write_rom
   from vane.sysid import modal_reduction

   rom = modal_reduction(
       system,
       n_modes=8,
       frequency_band_hz=(0.01, 3.0),   # the structural band
       rank_output="TwrBsMyt",          # keep the modes that drive this load
   )
   reduced = rom.system                  # states modal_0_re, modal_0_im, ...
   print(rom.frequencies_hz, rom.damping_ratios, rom.energy_retained, rom.n_stabilized)
   write_rom(rom, "rom_bundle")          # state_space.mat, modes.csv, manifest.json

The main choices:

* ``rank_output`` ranks by one output (an index, or a name matched exactly, else as a
  unique substring). ``rank_outputs`` ranks jointly over several outputs, each normalized
  to unit weight, so channels in different units count equally. With neither, modes are
  ranked by their residue norm over all outputs.
* ``n_modes=None`` keeps every underdamped mode in the band.
* ``stabilize=True`` (default) reflects a retained unstable pole into the stable
  half-plane, so the model is a usable estimator basis; ``rom.stabilized`` marks such
  modes, and ``rom.damping_ratios`` then reports the reflected value.
* ``match_dc_gain=True`` corrects the reduced :math:`\mathbf D` so that the static gain
  equals that of the full model (default ``False``).
* ``required_transfer_pairs`` with ``n_required_transfer_modes`` reserve modes for chosen
  input-to-output paths, for an observer that must keep a measured response to an
  estimated input.

``rom.state_projection`` and ``rom.state_reconstruction`` map between the full state and
the modal coordinates. :func:`~vane.export.write_rom` writes the matrices
(``matrix_format="mat"`` or ``"npz"``), a per-mode table and a manifest that records the
VANE version.

For a stable, minimal model with both :math:`\mathbf B` and :math:`\mathbf C`,
:func:`~vane.sysid.balanced.balanced_truncation` gives an input-output reduction with an
a-priori :math:`H_\infty` error bound. Apply it to the modal model rather than to the raw
linearization: ``balanced_truncation(reduced, n_states=10)`` (or ``tol=1e-3``) returns
the model with its ``hankel_singular_values`` and ``error_bound``.


Load maps
---------

:func:`~vane.sysid.qoi.qoi_map_from_mbc` turns the output equation into a map of each
load's state and input sensitivity and, with a modal solution, its modal residues:

.. code-block:: python

   from vane.sysid import qoi_map_from_mbc

   qmap = qoi_map_from_mbc(result.mbc_results[0], result.solutions[0])
   for load in qmap.loads():
       print(load.name, load.dominant_modes(top_n=3))
   table = qmap.modal_participation_table()   # one row per (load, mode)

The residues show which modes drive a load, which helps in choosing ``rank_output``.
They are in the linearization's units and are not fatigue, design or extreme loads.
``loads_only=False`` includes every output channel.


Noise covariances and the Kalman model
--------------------------------------

A filter's measurement equation contains only the sensors. Select the sensor rows, then
build the covariances from physical specifications:

.. code-block:: python

   import numpy as np
   from vane.sysid import CovarianceSet, StateSpace, build_kalman_model
   from vane.sysid import discretize_process_noise, measurement_noise_from_specs
   from vane.sysid import modal_process_noise

   tags = ("GenSpeed", "NacAccFA", "PtfmPitch")
   rows = [i for i, n in enumerate(reduced.output_names) if any(t in n for t in tags)]
   plant = StateSpace(
       a=reduced.a, b=reduced.b, c=reduced.c[rows], d=reduced.d[rows],
       state_names=reduced.state_names, input_names=reduced.input_names,
       output_names=[reduced.output_names[i] for i in rows],
   )

   dt = 0.05
   q_c = modal_process_noise([1.0e-4] * len(rom.frequencies_hz))  # one per mode
   q_d = discretize_process_noise(plant.a, q_c, dt)              # exact over one step
   r = measurement_noise_from_specs([0.1, 0.02, 0.01])          # sensor std, own units
   covariances = CovarianceSet(q=q_d, r=r, p0=np.eye(plant.n_states))
   model = build_kalman_model(plant, covariances, dt=dt)

:func:`~vane.sysid.kalman_interface.build_kalman_model` discretizes :math:`\mathbf A` and
:math:`\mathbf B` when ``dt`` is given but uses ``q`` as given, so pass the discrete
:math:`\mathbf Q_d`. It checks that :math:`\mathbf Q` and :math:`\mathbf P_0` are
positive semidefinite and :math:`\mathbf R` is positive definite.
:func:`~vane.sysid.covariance_init.covariances_from_confidence` builds a
:math:`\mathbf P_0` that is larger for less confidently identified modes, and
:func:`~vane.sysid.observability.reduced_estimator` combines sensor selection,
covariances and discretization in one call, with a hand-set diagonal :math:`\mathbf Q`.


Running the filter
------------------

:func:`~vane.sysid.kalman_filter.run_filter` runs the filter over a record. Row ``k`` of
``controls`` is the input logged with ``measurements[k]``; both are perturbations about
the operating point.

.. code-block:: python

   from vane.sysid import filter_consistency, reconstruct_load, rts_smoother, run_filter

   # measurements: (n_steps, 3) sensor record; controls: (n_steps, n_inputs)
   run = run_filter(model, measurements, controls=controls, gate_probability=0.999)
   print(filter_consistency(run).consistent)      # NIS chi-square test
   smoothed = rts_smoother(model, run)            # off-line refinement

   row = next(i for i, n in enumerate(reduced.output_names) if "TwrBsMyt" in n)
   tower = reduced.c[[row]]                       # the load as a functional of the state
   load, load_std = reconstruct_load(run, tower, smoothed=smoothed)

``available``, a boolean ``(n_steps, n_outputs)`` mask, encodes sensor dropout, and
``gate_probability`` (or a fixed ``gate``) rejects outliers by their normalized innovation
squared. :func:`~vane.sysid.kalman_filter.reconstruct_load` returns
:math:`\mathbf L\hat{\mathbf x}` and its standard deviation; add any feedthrough term
yourself. For step-by-step use, :class:`~vane.sysid.kalman_filter.KalmanFilter` has
``predict``, ``update`` and ``step`` methods, and
:class:`~vane.sysid.kalman_filter.SteadyStateKalmanFilter` runs with a fixed gain.


Estimator assessments
---------------------

:mod:`vane.sysid.assessment` gives the steady-state accuracy of a filter before it runs,
from the discrete model and its covariances alone:

.. code-block:: python

   from vane.sysid import SteadyStateKalmanFilter, assess_load, steady_state_estimator

   discrete = plant.discretized(dt)
   estimator = steady_state_estimator(discrete, process_noise=q_d, measurement_noise=r)
   [bound] = assess_load(estimator, tower, tolerance=500.0, names=["TwrBsMyt"])
   print(bound.error_std, bound.within_tolerance)

   fixed_gain = SteadyStateKalmanFilter(
       discrete.a, discrete.c, estimator.gain, b=discrete.b, d=discrete.d
   )

``error_std`` is :math:`\sqrt{\mathbf L\mathbf P^+\mathbf L^{\mathsf T}}` from the
steady-state Riccati solution. ``within_tolerance`` is ``True`` only when a
``tolerance`` was given and met. The value is conditional on the model and on the supplied
:math:`\mathbf Q_d` and :math:`\mathbf R`; it is not a measured field accuracy. If the
sensors cannot detect an unstable or marginal mode, or such a mode has no process noise,
:func:`~vane.sysid.assessment.steady_state_estimator` raises ``ValueError``.

When choosing sensors, :func:`~vane.sysid.observability.load_reconstructability` tells
whether a load is observable from a sensor set and gives its noise amplification, and
:func:`~vane.sysid.observability.recommend_sensors` and
:func:`~vane.sysid.observability.constrained_sensor_selection` pick sensor sets by the
D-optimal criterion. These need a stable model or a finite ``horizon``.


Families across operating points
--------------------------------

A modal family tracks the same modes, in the same state order and phase, across a sweep,
so that its members can be interpolated:

.. code-block:: python

   from vane.sysid import build_rom_family, schedule_model

   systems = [result.state_space(i) for i in range(len(result.mbc_results))]
   winds = [mbc.wind_speed for mbc in result.mbc_results]
   family = build_rom_family(
       systems, winds, n_modes=8, frequency_band_hz=(0.01, 3.0), rank_output="TwrBsMyt"
   )
   print(family.tracking_mac.min(axis=0))     # low values mark uncertain tracking

   at_9 = schedule_model(family, 9.5)         # linear blend of the two neighbours
   print(at_9.min_tracking_mac, at_9.extrapolated)

The operating points must be strictly monotonic, and each point needs at least
``n_modes`` underdamped modes. ``family.damping_ratios`` is the physical damping, negative
for an unstable mode even where the stored model was stabilized (``family.stabilized``).
:func:`~vane.sysid.lpv.run_scheduled_filter` schedules and discretizes the family at every
step from a per-step ``operating_points`` array and carries the state and covariance
across steps. Its measurements are the family's outputs, so build the family from
sensor-row systems or mask the other channels with ``available``.
:func:`~vane.sysid.rom_family.combine_rom_families` joins two families with disjoint
modes, for example a drivetrain branch that the load ranking left out.

A controller usually needs named physical states rather than modal ones.
:func:`vane.export.build_physical_lpv_family` reads a directory of ``.lin`` files,
averages each operating point's azimuth files in the rotating frame (no MBC transform),
and selects the named channels:

.. code-block:: python

   from vane.export import build_physical_lpv_family

   plant_family = build_physical_lpv_family(
       "sweep_out",
       states=["ED PtfmPitch", "ED RotSpeed"],
       inputs=["GenTq"],
       scheduling_var="WindSpeed",       # or "RotorSpeed", in rad/s
   )
   print(plant_family.state_names)         # the final list, rate twins added
   print(plant_family.auto_added_states)   # what pairing added
   print(plant_family.dropped_coupling)    # per-point truncation measure

A second-order DOF requested without its ``"First time derivative of"`` twin gets the
twin added automatically (``pair_derivatives=True``). A subset that still discards more
than 5 % relative coupling to the unselected states at any operating point is rejected
with a ``ValueError`` naming the worst point and the dropped states responsible
(``max_dropped_coupling=0.05``). Add those states, select the full plant and condense it
in the controller design, or pass ``max_dropped_coupling=None`` to accept the truncation
knowingly. The measure is taken in the linearization's own units, so it screens for gross
truncation rather than bounding the model error.


Saving and loading
------------------

:mod:`vane.sysid.serialize` stores results as ``.npz`` files that load with
``allow_pickle=False`` and record the VANE version:

.. code-block:: python

   from vane.sysid import load_result, load_rom_family, save_result, save_rom_family

   save_result(run, "run.npz")      # FilterResult or ScheduledFilterResult
   run = load_result("run.npz")
   save_rom_family(family, "family.npz")
   family = load_rom_family("family.npz")

The physical-state family from :func:`~vane.export.build_physical_lpv_family` has its
own pair, also importable from :mod:`vane.export`:

.. code-block:: python

   from vane.export import load_lpv_family, save_lpv_family

   save_lpv_family(plant_family, "plant_family.npz")
   plant_family = load_lpv_family("plant_family.npz")

The archive holds the scheduling points, every member's ``A``, ``B``, ``C`` and ``D``
matrices and the dropped-coupling norms as plain arrays, and the channel names, the
scheduling variable and its unit, the auto-added states and each member's names and
sample time as metadata. The reloaded family is identical to the saved one: the same
arrays with the same data types, and the same names. Each archive records a schema
version (:data:`~vane.sysid.serialize.LPV_FAMILY_SCHEMA_VERSION`, currently 1) and the
VANE version that wrote it. :func:`~vane.sysid.serialize.load_lpv_family` reads only
the current schema; an archive of another schema, one without a schema version, an
archive of another kind, and an incomplete or altered archive are rejected with a
``ValueError`` that names the problem (for a schema mismatch, both schema versions and
the writing VANE version). Both functions check that the family is complete: at least
one member, strictly ascending finite scheduling points, and the same matrix shapes,
names and sample time at every member.

A single reduced model is written with :func:`~vane.export.write_rom` (see above).
