.. SPDX-License-Identifier: Apache-2.0

The OpenFAST linearization sweep
================================

:func:`~vane.run.run_operating_sweep` produces the ``.lin`` files that the rest of VANE
analyses. It drives OpenFAST over a list of wind speeds: for each operating point it
copies and configures the deck, runs a ``CalcSteady`` linearization, retries a trim that
does not converge, and collects the files. The points run in parallel; the worker count
defaults to the CPU count less one reserved core (``max_workers=``, ``reserve_cpus=``).
The defaults are ``NLinTimes = 36`` (10 degree azimuth spacing) with ``CalcSteady``,
which gives the periodic linearization that the MBC3 transform needs.

.. contents:: On this page
   :local:
   :depth: 1

The operating schedule
----------------------

VANE does not guess the rotor speed and blade pitch at each wind, and it computes no
rotor performance. Supply the turbine's steady rotor-performance schedule through one
of these sources:

* a **rotor-performance CSV** (``performance_csv=``),
* the operating-curve arrays directly (``curve_wind`` / ``curve_rotor_speed`` /
  ``curve_pitch``), or
* ``settle=True``, which runs the fully-coupled controller at each wind and reads the
  settled rotor speed, pitch, and (for a floating deck) platform equilibrium from the
  time series. The settle runs keep the deck's controller library (ServoDyn
  ``DLL_FileName``) active. Published decks often name a library of another machine
  (the IEA decks point at a Linux ``libdiscon.so``), so VANE looks for a host build of
  the same library (see :ref:`controller-library` below) and writes it into the settle
  copies only; ``controller_library=`` names one explicitly.

With no source, or with both a CSV and curve arrays, the sweep **fails fast**, before
any OpenFAST process is launched, rather than silently linearizing at a guessed
operating point. VANE has no aerodynamic-performance (for example blade-element)
dependency; the schedule is the user's input.

.. _preparing-performance-csv:

Preparing a rotor-performance CSV
---------------------------------

The CSV needs a header row and three required columns — ``wind_speed`` (m/s),
``rotor_speed_rpm`` (rpm), and ``blade_pitch_deg`` (deg). Header matching is tolerant of
case, whitespace, and separators, so ``WindSpeed``, ``V [m/s]``, ``Omega [rpm]``, and
``pitch [deg]`` all resolve. A unit in square brackets or parentheses is read, not
ignored: a column in ``km/h``, ``rad/s`` (e.g. ``Rotor Speed [rad/s]``), or ``rad`` is
converted to m/s, rpm, or deg, and any other unit on these three columns is rejected
with an error naming the column and the accepted units. A column with no unit is taken
to be in m/s, rpm, or deg. Units are read from the header only: a separate units row
below the header is not supported and is rejected as a non-numeric value. At least two
data rows are required (the schedule is interpolated to the requested winds); negative
wind or rotor speed is rejected, but a negative collective blade pitch (a fine-pitch
schedule) is allowed.

Two optional columns are read when present: the power coefficient (``cp``, also
``power_coefficient``, ``RtAeroCp`` or ``RtFldCp``) and the thrust coefficient (``ct``,
also ``thrust_coefficient``, ``RtAeroCt`` or ``RtFldCt``). The sweep uses ``cp`` to find
the rated point and the rated generator torque, and
:func:`~vane.run.verify_sweep_operating_points` compares both with the trimmed points.
Each may be given as a fraction or as a percentage, declared in its header bracket:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Header
     - Read as
     - Example
   * - ``cp``, ``cp [-]``, ``cp [1]``, ``cp [fraction]``
     - fraction
     - ``0.48``
   * - ``cp [%]``, ``cp (%)``, ``cp [percent]``, ``cp [pct]``
     - percentage, divided by 100
     - ``48``

The same forms apply to ``ct``. Any other unit, such as ``cp [kW]``, is rejected with an
error that lists the accepted forms. The scale is also checked against the declared
unit, and a contradiction stops the sweep before OpenFAST is launched rather than being
guessed at:

* a fraction above the physical bound, the Betz limit 16/27 for ``cp`` or 2 for ``ct``
  (the high-induction limit of blade-element momentum theory), is rejected as a
  percentage written without its unit (``cp`` holding ``48``);
* a column declared in percent whose non-zero values are all at most 1 is rejected as
  fractions labelled as percent (``cp [%]`` holding ``0.48``).

Blank or non-numeric coefficient cells are read as unknown. Other columns, such as
``cq`` and ``gen_power``, are ignored, units included, so a standard
performance export can be used unchanged.

The coefficient columns are optional; this CSV gives ``cp`` as a fraction:

.. code-block:: text

   wind_speed,rotor_speed_rpm,blade_pitch_deg,cp
   3.0,5.0,0.0,0.41
   8.0,7.5,0.0,0.48
   11.0,9.6,0.0,0.49
   16.0,9.6,11.3,0.30
   25.0,9.6,22.8,0.06

Running the sweep
-----------------

.. code-block:: python

   from vane.run import run_operating_sweep

   result = run_operating_sweep(
       "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile.fst",
       output_dir="sweep_out",
       wind_speeds=[4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
       performance_csv="rotor_performance.csv",
   )

   for case in result.cases:
       print(case.operating_point.wind_speed, case.converged, case.n_lin)
   for item in result.auto_configuration.inferred:
       print(item.name, item.value, item.source)

No further settings are needed. Everything the linearization needs beyond the curve is
read from the deck and the curve you supplied (``auto_configure=True``, the default),
and each value is logged, returned in
``result.auto_configuration`` and written with its source to
``sweep_out/auto_configuration.json`` and the run provenance:

* **Copy root.** When the deck references files outside its folder, the cases are copied
  from the deepest directory that holds the deck and every file it references by a
  relative path (:func:`~vane.run.deck_files.infer_template_root`).
* **OpenFAST executable.** OpenFAST's input format changes between releases (a v4.0 deck
  fails on v4.2, a v4 deck fails on v5). VANE recognises the deck's format from its
  keywords (:func:`~vane.run.detect_deck_generation`) and runs an executable whose
  release reads it, chosen from ``executable=``, the ``VANE_OPENFAST_EXECUTABLES``
  environment variable (paths separated by ``;`` on Windows, ``:`` elsewhere), the file
  named by ``VANE_OPENFAST_CONFIG`` (default ``~/.vane/openfast_executables.txt``, one
  path per line) and ``PATH``. Each candidate's release is read once from
  ``openfast -v``. When none fits, the sweep stops before launch and names the release
  the deck needs.
* **Rated wind speed** -- the first curve wind where the rotor speed has reached its
  rated plateau and, with a ``cp`` column, ``cp * V**3`` has too (a peak-shaving curve
  that pitches before rated is handled). It is cross-checked against the controller's
  rated speed and power when the deck has a ``DISCON.IN`` controller input file.
* **Rated generator torque** -- ``VS_RtTq`` of the controller input file (ServoDyn
  ``DLL_InFile``), else ServoDyn ``VS_RtTq`` when ``VSContrl = 1``, else the curve's
  aerodynamic torque at rated from its ``cp`` column.
* **Pitch trim gain and trim tolerance**, scaled for the rotor size (see
  :func:`~vane.run.scaled_trim_gain_pitch` and :func:`~vane.run.scaled_trim_tol`).
* **Module settings** (when ``settings`` is not given): the
  :meth:`~vane.run.LinearizationSettings.offshore` preset adapted to the deck. A missing
  state-space hydrodynamics file (``.ss``/``.ssexctn``) falls back to
  ``RdtnMod``/``ExctnMod`` 0 with a warning, and a deck's own linearizable unsteady
  aerodynamics (``UA_Mod`` 4-6) is kept, with ``UAEndRad`` 0.99 where it would reach the
  blade tip under tip loss.

Explicit arguments always win: ``template_root=``, ``executable=`` and ``settings=`` are
used as given (an explicit executable known not to read the deck is rejected before
launch), and only the constants your settings leave unset are inferred.
``auto_configure=False`` turns the inference off. Whatever still cannot be made
linearizable (nacelle drag, a free-vortex wake, a lidar, a curve that stops below rated)
fails before launch with the keyword and the remedy.

To check that each trimmed point reproduces your rotor-performance data (rotor speed,
pitch, and the power and thrust coefficients from the steady torque balance), pass the
result to :func:`~vane.run.verify_sweep_operating_points` -- see
:doc:`trim_verification`.

Each operating point gets its own copy of the deck. When the deck references files
outside its own folder (``../5MW_Baseline/...`` in the OpenFAST regression tests, or
``../IEA-15-240-RWT/...`` in the IEA reference-turbine repositories), the copy root
(``template_root``) is the directory that contains all of them, e.g.
``r-test/glue-codes/openfast`` or ``IEA-15-240-RWT/OpenFAST``; VANE infers it, and the
``output_dir`` must lie outside it. A broad root costs nothing extra: by default
(``template_copy="referenced"``) VANE copies only the files the deck references -- found
recursively from the ``.fst`` (module inputs, blade and tower files, airfoil polars and
coordinate files, the controller library and its input and performance files, wind files,
and every ``<root>.<ext>`` file of a HydroDyn ``PotFile`` or SeaState ``WvKinFile`` root
name) -- at their paths relative to ``template_root``, and hard-links them where the
filesystem allows. A case of the regression-test NREL 5-MW land deck is about 9 MB instead
of the 306 MB of the whole suite, and other cases' outputs (including their ``.lin`` files)
never enter it. If the walk meets a relative reference that leaves ``template_root``, or an
input it cannot read, it logs a warning and copies the whole tree instead;
``template_copy="tree"`` asks for that whole-tree copy outright.
:func:`~vane.run.deck_file_closure` lists the files a deck would bring.

The collected ``.lin`` files feed straight into the modal pipeline and the physical-state
LPV export:

.. code-block:: python

   from vane import discover_operating_points, run_study
   from vane.export import build_physical_lpv_family

   study = run_study(discover_operating_points("sweep_out"))
   study.write_bundle("sweep_report")        # Campbell diagram + provenance + report.html

   # Or the named-DOF plant family a gain-scheduled controller schedules on:
   family = build_physical_lpv_family(
       "sweep_out",
       states=["ED PtfmPitch", "ED RotSpeed"],
       inputs=["GenTq"],
       scheduling_var="WindSpeed",
   )
   print(family.state_names)         # the final list, rate twins added
   print(family.auto_added_states)   # what pairing added
   print(family.dropped_coupling)    # per-point truncation norm

A second-order DOF requested without its ``"First time derivative of"`` twin gets the twin
added automatically (``pair_derivatives=True``), and a subset that still discards more
than 5 % relative coupling to the unselected states at any operating point is rejected
with a ``ValueError`` naming the worst point and the dropped states responsible. Add those
states, select the full plant and condense it in the controller design, or pass
``max_dropped_coupling=None`` to accept the truncation knowingly.

``rated_wind_speed`` is the turbine's own rated wind speed (10.66 m/s on the IEA 15-MW
published curve, 12.12 m/s on a published NREL 5-MW curve). It chooses torque trim below rated
and pitch trim at and above it, so there is no turbine-independent default. It is
inferred from the operating curve (see above); a rotating sweep where it is neither given
nor inferable (a ``settle=True`` sweep without a curve, or a curve that stops below rated)
fails before any OpenFAST run starts. It is not needed when no point rotates or when neither
``calc_steady`` nor ``trim_controller`` is enabled. A ``settle=True`` sweep at positive
wind always needs it (with trim or controller handling), because whether a settled point
turns is only known after its settle run, and the sweep must fail before launching one.

For the below-rated torque trim VANE derives the trim gain as
``rated_generator_torque / 10`` (a too-low torque gain is the most common cause of
non-convergence); the rated torque is inferred as described above, or pass it in
``settings``. With ``trim_controller`` enabled (the ``offshore()``
default), VANE also frees the generator DOF (``GenDOF``) for every trimmed rotating point
and sets the generator torque per region — keeping the generator off below rated so the
torque trim owns the torque, and flattening the variable-speed law to a constant
``VS_RtTq`` (from ``rated_generator_torque``) at and above rated so the pitch trim
converges. On a DLL-controlled template (original ``VSContrl`` other than ``1``) the
deck's ``VS_RtTq`` is a placeholder, so an above-rated point then needs the rated torque
from the controller input file, the curve's ``cp`` or ``rated_generator_torque``; without
any of them the sweep fails before launch.

The sweep auto-retries an operating point whose ``CalcSteady`` trim does not converge, up
to ``LinearizationRetryPolicy.max_attempts`` runs (default 4), choosing each retry from
the classified failure (``vane.run.classify_openfast_output``):

* **trim not converged** (OpenFAST forced a single ``.lin``): ``TMax`` doubles
  (``tmax_factor``, capped at ``max_sim_time``, 3600 s) and ``TrimTol`` loosens threefold
  (``trim_tol_factor``); gains and damping are unchanged;
* **motoring** (``u%HSS_Spd is negative``): the pitch gain falls
  (``pitch_trim_gain_factor``) and the torque gain is divided by ``trim_gain_factor``;
  a failure within ``early_failure_time`` (1 s) of the start also removes the trim
  damping;
* **rotor overspeed** or **divergence**: the gains fall as for motoring and the trim
  damping is multiplied by ``damping_backoff_factor`` (removed for an early failure);
* **initialisation error**: not retried; the OpenFAST message is logged;
* **non-finite linearization**: OpenFAST converged and wrote every ``.lin`` file, but a
  file holds a NaN or infinity. Not retried; the first offending variable and a hint
  are logged. AeroDyn unsteady aerodynamics left on at the blade-tip node under tip loss
  is the known cause; set ``ua_end_radius=0.99``;
* **timeout** or an unrecognised failure: the generic step, which loosens ``TrimTol``,
  raises the damping above both the previous value and the ``min_retry_damping`` floor,
  raises the torque gain and lowers the pitch gain.

``adaptive=False`` applies the same escalation to every failure regardless of its
class, and ``final_fallback`` supplies a last-resort settings change for the final
attempt after an unstable trim. ``TrimTol`` is never loosened beyond ``0.1``
(``vane.run.lin_config.MAX_TRIM_TOL``): a looser tolerance would accept revolutions
that differ by the size of the response itself, so a "converged" point would mean
nothing. The as-run tolerance and ``TMax`` are recorded per case. Each attempt's full OpenFAST output is saved in the case directory as
``<deck stem>.attempt<k>.log``, and ``SweepCase.attempt_history`` (also in the
provenance) records each attempt's failure mode, settings and log file. A point that
never converges is recorded ``converged=False`` with its attempt count rather than
silently accepting a forced single-``.lin`` result. Passing ``trim_tuning=TrimTuningPolicy()`` additionally tunes each
point's *starting* trim tolerance and damping for its wind-speed region **before** that
ladder runs — at/above-rated points (where ``CalcSteady`` is hardest) start from the
looser-tolerance, higher-damping floors those trims need, so they tend to converge on the
first attempt rather than after a reactive retry.

The ElastoDyn structural DOFs are set through ``structural_dofs=``, merged on top of the
defaults: blade and tower flexibility on (``FlapDOF1``, ``FlapDOF2``, ``EdgeDOF``,
``TwFADOF1``, ``TwFADOF2``, ``TwSSDOF1``, ``TwSSDOF2``), nacelle yaw (``YawDOF``) and
drivetrain torsion (``DrTrDOF``) off. Yaw is off because a linearization has no yaw
controller, so a free yaw DOF adds a spurious free mode; drivetrain torsion is off
because it makes the trim harder to converge, at the cost of the drivetrain-torsion
mode. Pass ``structural_dofs={"DrTrDOF": True}`` to keep it (see :doc:`limitations`).

Operating-point acceptance
~~~~~~~~~~~~~~~~~~~~~~~~~~

For a floating turbine, an independent physical check of the operating point can be
enabled on the :class:`~vane.run.LinearizationDriver`:

.. code-block:: python

   from vane.run import LinearizationAcceptancePolicy, LinearizationDriver

   driver = LinearizationDriver(
       fst_path="Main.fst",
       output_dir="linearizations",
       acceptance_policy=LinearizationAcceptancePolicy(
           max_mean_translational_speed=1.0e-3,
           max_mean_angular_speed=1.0e-4,
       ),
   )

The check estimates the azimuth mean of the platform-rate states with quadrature weights
that exactly cancel the 1P, 2P and 3P harmonics at the sampled azimuths (the mean of a
least-squares mean-plus-harmonics fit), so a periodic platform motion cancels whereas
secular drift remains -- also on the slightly non-uniform azimuths ``CalcSteady``
actually writes. A rotating point needs 12 unique azimuths by default; an explicit
override cannot use fewer than four. A layout from which the mean cannot be separated
from 1P--3P motion without amplifying noise more than twofold (for example a partial
revolution) is reported as unassessed. The check is optional; its complete record is
stored on ``SweepCase.operating_point_residual`` and in the run provenance.

If the automatic periodic trim does not remove a slow floating-platform transient, a
separately demonstrated periodic operating point may be sampled at explicit absolute
times. Set ``calc_steady=False``, keep ``n_lin_times`` equal to the number of supplied
``linearization_times``, and span one complete final rotor revolution. VANE validates
the time ordering and simulation bounds, requires every requested ``.lin`` member, and
applies the same physical residual check. This is a controlled fallback for an already
settled rotating system, not a substitute for establishing equilibrium.

Floating turbines: settling first
---------------------------------

The CSV and the curve arrays are two forms of one curve, so give at most one of them.
Either may be **combined with** ``settle=True``: the curve then only seeds the settle
runs. Each settle copy starts from the curve's rotor speed and pitch at its wind
(ElastoDyn ``RotSpeed`` and ``BlPitch(1..3)``) instead of from 0 rpm and 0 deg, so the
rotor need not spin up from rest and the start-up transient is smaller. The controller
still decides where the turbine settles, and the **settled** point is what is
linearized. Afterwards the settled rotor speed and pitch are compared with the curve
(by default within 1 % and 0.5 deg, the :class:`~vane.run.TrimTolerances` defaults;
override with ``settle_curve_tolerances=``). A larger deviation means the controller and
the curve disagree. It is logged as a warning and recorded as the point's
``settle_curve_check`` (and as ``settle_curve_*`` in the run provenance), but it does
not fail the sweep:

.. code-block:: python

   from vane.run import LinearizationSettings, PlatformPosition, run_operating_sweep

   result = run_operating_sweep(
       "IEA-15-240-RWT-UMaineSemi/IEA-15-240-RWT-UMaineSemi.fst",
       "runs/iea15_semi",
       [8.0, 10.0, 12.0],
       settle=True,
       performance_csv="iea15_performance.csv",  # initial conditions only
       settle_initial_platform={12.0: PlatformPosition(surge=16.0, pitch=3.5)},
       settings=LinearizationSettings.offshore(
           rated_wind_speed=10.66, rated_generator_torque=19786800.0
       ),
   )
   for point in result.operating_points:
       print(point.wind_speed, point.settle_curve_check.status)

VANE does not estimate the platform offset from the curve (that would need the rotor
thrust, which VANE does not compute): the platform starts at the deck's position unless
you pass ``settle_initial_platform=``, either one :class:`~vane.run.PlatformPosition` for
every wind or a mapping from wind speed to position (for example the mean offsets of an
earlier settle). Without a curve, a settle starts from 0 rpm and 0 deg. The optional
initial position is the only floating input: the controller library,
the settle damping and the settle criterion are all chosen automatically, as the
following sections describe.

.. _controller-library:

Finding the controller library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without ``controller_library=``, :func:`~vane.run.find_controller_library` looks for a
library the host can load, in this order:

1. the deck's own ``DLL_FileName``, when it exists with this host's suffix (``.dll`` on
   Windows, ``.so`` on Linux, ``.dylib`` on macOS);
2. a build of the same library next to it, next to the ServoDyn file, or next to the
   primary deck -- matched by name with the ``lib`` prefix, a bitness tag (``_x64``,
   ``_win32``, ...) and the suffix ignored, so ``libdiscon.so`` matches
   ``libdiscon.dll``, ``libdiscon_x64.dll`` or ``DISCON.dll`` (a 64-bit build is
   preferred to a 32-bit one);
3. the files or directories in ``controller_search_paths=``;
4. the ``VANE_CONTROLLER_LIBRARIES`` environment variable (entries separated by
   ``os.pathsep``);
5. the configuration file ``controller_libraries.txt`` (one file or directory per line,
   ``#`` comments) in VANE's configuration directory: ``VANE_CONFIG_DIR``, else
   ``%APPDATA%\vane`` on Windows and ``$XDG_CONFIG_HOME/vane`` or ``~/.config/vane``
   elsewhere.

Setting the environment variable once is enough for every deck that uses the same
controller:

.. code-block:: bash

   set VANE_CONTROLLER_LIBRARIES=C:\controllers\discon      # Windows
   export VANE_CONTROLLER_LIBRARIES=$HOME/controllers/lib    # Linux, macOS

A found library must still match the deck's controller input file (``DLL_InFile``);
VANE cannot check the controller version.

**No library at all.** When no host build exists the sweep does not fail. Each settle
runs **without a controller** instead: ServoDyn is switched off in the settle copy and
ElastoDyn's generator and drivetrain DOFs are locked, so the rotor turns at the
operating curve's speed with the blades at its pitch while the platform finds its
equilibrium under the resulting aerodynamic load (``prescribed_rotor`` of
:func:`~vane.run.solve_steady_state`). CalcSteady then trims the rotor at that curve
point, starting from the platform equilibrium the settle found. This fallback needs an
operating curve or ``performance_csv`` (without one the sweep fails before launch).
The settled point is linearized even when the settle did not converge, because it only
seeds the trim, which finds the equilibrium itself. Each point records
``settle_convergence.prescribed_rotor``, and ``result.floating_plan`` records why
(``result.floating_plan.controller.status == "missing"``) and where VANE looked.

The settle copies keep the template deck's own ElastoDyn DOF flags, whereas the
linearization cases apply ``structural_dofs`` (blade and tower flexibility on,
``YawDOF`` and ``DrTrDOF`` off by default). The settle runs the deck's controller on
the model it was tuned for; the settled means carried into the linearization (rotor
speed, pitch, platform position) do not depend on the drivetrain or yaw flexibility,
and only slightly on blade flexibility when a deck ships with rigid blades. The flags
the settles ran with are returned as ``result.settle_structural_dofs``, and any
difference is logged.

A wind that does not settle never aborts the sweep. Every wind is settled. A point
whose settle did not reach a confirmed equilibrium, or whose settle run failed, is
listed in ``result.settle_failures`` with its diagnostics: reason, message, last mean,
and every attempt's RMS and horizon. ``failure.to_dict()`` gives JSON. The other winds
are linearized as usual. ``require_settled`` (default ``True``) decides whether an
unsettled point is skipped (``True``) or linearized at its last mean (``False``). Pass
``strict_settle=True`` to raise :class:`~vane.run.SweepSettleError` at the end instead.
The completed result stays available as ``error.result``.

The same holds for the linearization cases. A case that raises an unexpected exception
(say, a file the operating system kept locked) is recorded in ``result.cases`` with
``converged=False`` and ``failure_mode == "case_error"``, its traceback in the
attempt log, and the other points run to completion. Check
``[case for case in result.cases if not case.converged]`` after a sweep, or pass
``strict_cases=True`` to raise :class:`~vane.run.SweepCaseError` at the end, with the
completed result as ``error.result``.

Settling a floating turbine faster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A semi-submersible's surge, sway and yaw modes have periods of 100 s or more and very
little damping, so a plain settle spends most of its time waiting for the platform to
stop ringing long after the rotor has reached its operating point. For a floating deck
(free ElastoDyn platform DOFs, no SubDyn substructure, mooring or HydroDyn active) the
sweep therefore turns each settle into two stages automatically; pass
``settle_auto_damping=False`` to opt out, or ``settle_platform_damping=`` to choose the
damping yourself:

1. **Damped stage.** A :class:`~vane.run.SettlePlatformDamping` is added to the diagonal
   of HydroDyn's ``AddBLin`` matrix in the settle copy, and the settle runs (adaptively,
   as configured) until the platform is still. A load proportional to velocity is zero
   when the platform is at rest, so it cannot move the equilibrium; it only makes the
   transient die out in a cycle or two.
2. **Undamped confirmation.** The deck's own ``AddBLin`` is restored byte for byte, and
   an undamped run starts from the stage-1 means (rotor speed, pitch, platform
   position, tower-top and, when output, blade-tip deflections). Its length sizes
   itself: the analysed last three quarters hold eight periods of the slowest platform
   mode (the probe's longest period, 125.5 s on the IEA 15-MW semi, gives 1339 s),
   never less than 1200 s and never more than ``max_settle_time`` when given (else
   3600 s); ``settle_confirm_time=`` fixes it instead. Its means are averaged over a
   whole number of the residual oscillation's period, aligned to the end of the run,
   so a partial cycle does not bias them. Two conditions must hold for the point to be
   accepted. First, the means of this undamped run match stage 1 within
   ``settle_equilibrium_tolerances`` (an :class:`~vane.run.EquilibriumTolerances`; by
   default 0.1 m, 0.05 deg, 0.5 % rotor speed, 0.1 deg pitch). Second, the undamped
   run is stable: either it settles by itself, or its residual oscillation dies out
   with a damping ratio of at least ``residual_damping_ratio`` (0.5 % by default). A
   restart rings the slow modes even at the equilibrium, because the controller has
   its own start-up transient, and a 1 %-damped surge mode cannot ring down within a
   short check. When the residual is decaying but its means have not yet come back
   (the restart transient still biases them), the confirmation is rerun once, twice
   as long. The accepted point is the stage-1 equilibrium. Otherwise the settle is
   reported as not converged. In a sweep the point is then listed in
   ``result.settle_failures`` and, with ``require_settled=True`` (the default), it is
   not linearized. It is never accepted silently.

The damping is written to the settle copies only. Each linearization case is copied
afresh from the template and keeps the deck's own ``AddBLin``.

**How the damping is sized.** It is half of critical damping on every platform DOF,
measured from the model by :func:`~vane.run.probe_platform` before the first settle.
The probe runs two single-snapshot standstill linearizations of a copy of the deck in
``output_dir/probe`` (still air and water, rotor parked, AeroDyn and ServoDyn off,
radiation memory off, so no state-space hydrodynamic fit or controller library is
needed), the second with a known damping added to ``AddBLin``. The difference of the two
state matrices is the exact inverse mass matrix (structure plus added mass) on the
platform DOFs; eigenvalue perturbation of the first then gives, for each DOF, the
platform mode its damper damps the most, that mode's natural period, and the mass of a
single-DOF oscillator damped at the same rate. The probe takes seconds. On the IEA 15-MW
VolturnUS-S it measures surge/sway 3.1e7 kg at 124--126 s, heave 4.6e7 kg at 20.0 s,
roll/pitch 5.7e10 kg m^2 at 28.6--28.8 s and yaw 4.5e10 kg m^2 at 85 s; the damping
it sizes is within 12 % of a damping sized by hand from independently identified values
(3.3e7 kg, 4.5e7 kg, 5e10 kg m^2; 142, 20.6, 28, 90 s), which cut the 10 m/s settle
from 2373 s to 505 s of wall time at the same mean. If the probe fails, the failure is
noted in ``result.floating_plan.notes`` and the settle runs undamped. To size the
damping yourself, give the effective mass and natural period of each DOF:

.. code-block:: python

   from vane.run import SettlePlatformDamping

   damping = SettlePlatformDamping.from_critical_fraction(
       0.5,
       mass={"surge": 3.3e7, "sway": 3.3e7, "yaw": 5.0e10},  # kg, kg m^2
       natural_period={"surge": 110.0, "sway": 110.0, "yaw": 80.0},  # s
   )
   result = run_operating_sweep(
       ...,
       settle=True,
       max_settle_time=3600.0,
       settle_platform_damping=damping,
   )

Precision is not needed. Any positive damping leaves the equilibrium where it is, and
about half of critical damping removes the transient in one or two cycles without the
slow creep of an overdamped DOF. Absolute values in N/(m/s) and N m/(rad/s) can be given
directly, for example ``SettlePlatformDamping(surge=2e6)``.

**When the platform never stops moving.** Above rated, a pitch-regulated rotor can
remove damping from the slow surge mode. The undamped turbine then keeps oscillating
about its equilibrium (period about 106 s on the IEA 15-MW semi-submersible at
12 m/s), and the velocity criterion can never pass, however long the settle. The
default ``settle_criterion="auto"`` handles this: it judges every run by the velocity
criterion first, and only when that fails *and* the platform carries a sustained
oscillation -- one whose cycle-by-cycle RMS neither decays nor grows, its damping
ratio within ``residual_damping_ratio`` (0.5 %) of zero -- it switches to the
mean-drift criterion: the run is settled when the means of the platform position, rotor
speed and pitch over the two halves of the analysis window (each averaged over whole
oscillation periods) agree within the equilibrium tolerances. A transient that is
still decaying keeps the velocity criterion (an adaptive settle runs longer), and a
single settle run never accepts a growing oscillation. The oscillation is still
reported: its velocity RMS, dominant period
(``settle_convergence.oscillation_period``) and damping ratio
(``settle_convergence.residual_damping_ratio``). Make each half of the analysis window
span a couple of oscillation periods. ``"velocity"`` and ``"mean_drift"`` force one
criterion.

In a two-stage settle the same oscillation shows up as an undamped confirmation that
does not decay, about the equilibrium the damped stage found. With ``"auto"`` such a
point is accepted when the equilibrium is *confirmed* and the oscillation is
*measurable*: the confirmation's whole-period means match the damped stage's within
the equilibrium tolerances, its residual damping ratio is known (two or more
analysed cycles), and its platform-velocity RMS stays within
``oscillation_velocity_rms`` (0.2 m/s and deg/s), so its mean is a well-defined
measurement. The oscillation may be sustained *or growing*. The added damping is
linear in the platform velocity, which is zero at an equilibrium, so it cannot move
the equilibrium: the damped stage's point is the undamped model's, stable or not. A
growing oscillation is the closed loop with the deck's controller (near rated,
for example), and the linearization never uses that controller -- CalcSteady trims
the open-loop plant. The damped stage's equilibrium is then linearized, a warning is
logged, and ``settle_convergence.sustained_oscillation`` (and the confirmation's
``sustained_oscillation``) records it. When the residual damping ratio is below
``-growing_damping_ratio`` (-0.02 by default: an amplitude growing 13 % per cycle,
2.7-fold over eight cycles) the point is also flagged
``settle_convergence.growing_oscillation`` with a distinct warning: the equilibrium
is unstable under the deck's controller, and that controller-in-the-loop instability
is not part of the linear model. The IEA 22-MW semi at 12 m/s grows at -0.026 about a
reproduced equilibrium (surge shift -0.017 m) and is linearized this way; the IEA
15-MW semi's 12--14 m/s oscillation (-0.007 to -0.009) is sustained, not growing. A
too large oscillation, or a mean that moved, still fails, with its reason. The
``"velocity"`` and ``"mean_drift"`` criteria accept neither.

The criterion that judged each point is recorded as ``settle_convergence.criterion``
(``"velocity"`` or ``"mean_drift"``, also per attempt), the requested one as
``settle_convergence.requested_criterion``, a switch as
``settle_convergence.criterion_switched``, and the criterion as ``settle_criterion`` in
the run provenance. The two-stage evidence is recorded as
``settle_convergence.confirmation`` and ``settle_confirmation``, and each attempt
carries its ``stage`` (``"damped"`` or ``"confirm"``).

Missing state-space radiation fits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A floating linearization needs HydroDyn's state-space radiation model
(``RdtnMod = 2``), read from ``<PotFile>.ss``; the convolution model is not
linearizable. When a deck ships the WAMIT ``.1`` file but no ``.ss`` fit,
:func:`~vane.run.ensure_radiation_state_space` generates one in a working copy: every
significant entry of the radiation kernel ``K(jw) = B(w) + jw [A(w) - A(inf)]`` is
vector-fitted with stable poles and ``K(0) = 0``, raising the order until the relative
error over the WAMIT frequencies is below 5 %. On the IEA 15-MW VolturnUS-S the
generated fit is closer to the WAMIT data than the shipped one (4--5 % against
10--21 % per entry) and a standstill linearization with it reproduces the platform
frequencies of one with the shipped file to within 1 % and its damping ratios to
within 0.003. A fit worse than 20 % is not written; then linearize with
``hydro_radiation_mode=0``. Wave-excitation fits
(``.ssexctn``) are not generated: in still water they only add an unexcited wave
input, so use ``hydro_excitation_mode=0``.
