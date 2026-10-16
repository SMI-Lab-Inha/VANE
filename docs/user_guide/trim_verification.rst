.. SPDX-License-Identifier: Apache-2.0

Verifying the trimmed operating point
=====================================

VANE does not compute rotor performance: you supply the turbine's steady operating
schedule (see :doc:`quickstart`), and OpenFAST's ``CalcSteady`` trims the turbine about
it. Below rated it trims the **generator torque** at the prescribed pitch and rotor
speed; above rated it trims the **blade pitch** at constant generator torque. The point
OpenFAST converges to is therefore a check on your data. If it does not reproduce your
curve, something disagrees: the curve itself, the rotor radius or air density it was
normalized with, the gearbox ratio or loss, or the deck.

:func:`~vane.run.trim_verification.verify_operating_points` reads the trimmed point
back out of the ``.lin`` files and compares it with the curve:

.. code-block:: python

   from vane.run import find_lin_files, verify_operating_points

   lin_sets = [find_lin_files(f"sweep_out/ws{ws}/Main.fst") for ws in (6, 10, 14, 20)]
   report = verify_operating_points(
       lin_sets,
       "rotor_performance.csv",  # cp / ct columns are used when present
       deck="sweep_out/ws6/Main.fst",
   )
   print(report.to_table())
   assert report.passed, [p.label for p in report.flagged]

Either ``deck=`` (an ``.fst`` file the turbine constants are read from) or
``constants=`` (see `Turbine constants`_) is required. Each entry of ``lin_sets`` holds
the ``.lin`` files of one linearization, as paths or parsed
:class:`~vane.io.lin_reader.LinFile` objects. The CSV is read by
:func:`~vane.run.rotor_performance_from_csv`; its ``cp`` and ``ct`` columns may be
fractions or declared percentages (``cp [%]``), as described in
:ref:`the linearization-sweep guide <preparing-performance-csv>`.

After :func:`~vane.run.auto_sweep.run_operating_sweep`, pass the result to
:func:`~vane.run.trim_verification.verify_sweep_operating_points`. It reads the
constants from the first verified case's deck, labels each point with its case name,
and lists the cases that did not converge or wrote no ``.lin`` files in
``report.skipped`` rather than dropping them:

.. code-block:: python

   from vane.run import run_operating_sweep, verify_sweep_operating_points

   sweep = run_operating_sweep(..., performance_csv="rotor_performance.csv")
   report = verify_sweep_operating_points(sweep, "rotor_performance.csv")

``report.passed`` is true only when every verified point passes and no case was
skipped. ``report.to_records()`` gives one flat record per point (for a
``pandas.DataFrame``).

What is compared
----------------

For every linearization the azimuth files are averaged, and for each operating point
the report carries:

* the trimmed **wind speed** and **rotor speed** (from the ``.lin`` headers);
* the trimmed **blade pitch**: the mean over the blades and the azimuth files of the
  ``ED Blade <i> pitch command`` inputs (or the collective pitch-command input), plus
  its spread (zero for a collective trim);
* the **generator torque** (the ``ED Generator torque`` input);
* the **aerodynamic power** and **power coefficient** :math:`C_p` from the torque
  balance below;
* the **thrust coefficient** :math:`C_t`, when the outputs carry AeroDyn's rotor thrust
  (``RtFldFxh``, or ``RtAeroFxh`` in older AeroDyn versions, in the AeroDyn
  ``OutList``);
* the curve values at the trimmed wind speed (the same monotone PCHIP interpolation the
  sweep uses, without extrapolation), the deviations, and a ``pass`` / ``flag`` /
  ``incomplete`` status.

The optional ``cp`` and ``ct`` columns of the CSV (also accepted as
``power_coefficient`` and ``thrust_coefficient``) are read as fractions; a blank or
non-numeric cell counts as unknown.

Below rated the rotor speed and pitch are prescribed from your curve, so their
agreement is by construction and :math:`C_p` is the informative check; above rated the
trimmed pitch is. The default tolerances (:class:`~vane.run.trim_verification.TrimTolerances`)
are 1 % on rotor speed, 0.5 deg on pitch, and 5 % on :math:`C_p` and :math:`C_t`. Pass
``tolerances=TrimTolerances(...)`` to change them.

Torque balance
--------------

At a periodic steady state the azimuth-mean rotor acceleration is zero, so the mean
aerodynamic torque on the low-speed shaft equals the torque the shaft transmits. With
gearbox ratio :math:`N` and efficiency :math:`\eta` (ElastoDyn ``GBRatio`` and
``GBoxEff``), the low-speed-shaft torque that balances the high-speed-shaft generator
torque :math:`Q_g` is

.. math::

   Q_{LSS} = N\,Q_g\,\eta^{-\operatorname{sgn}(Q_g\Omega)},

that is, :math:`N Q_g / \eta` when power flows from the rotor to the generator (the loss
is added back) and :math:`N Q_g \eta` when the generator motors the rotor. Then

.. math::

   P = Q_{LSS}\,\Omega, \qquad
   C_p = \frac{P}{\tfrac12\rho\pi R^2 U^3}, \qquad
   C_t = \frac{T}{\tfrac12\rho\pi R^2 U^2}.

Assumptions:

* **Steady state.** The point is a converged periodic trim. A point the retry ladder
  could not converge should not be verified.
* **Gearbox losses as modelled by** ``GBoxEff`` **and nothing else.** Bearing friction
  and other drivetrain losses are not in ElastoDyn and are not added.
* **No generator efficiency.** :math:`Q_g` is the mechanical torque on the high-speed
  shaft, so the electrical efficiency (``GenEff``) does not enter. A curve's
  ``gen_power`` column is electrical and is not compared.
* **No other torque on the rotor.** ``CalcSteady``'s artificial trim damping
  (``Twr_Kdmp``/``Bld_Kdmp``) violates this; see `Trim damping`_ below. It is off by
  default, and the retry ladder adds it only on its generic escalation step (a
  timeout or an unrecognised failure).
* **The thrust is AeroDyn's** rotor aerodynamic thrust along the shaft. ElastoDyn's
  ``RotThrust`` is not used: it is the low-speed-shaft thrust, which also carries the
  axial component of the rotor weight on a tilted shaft.

Trim damping
------------

During ``CalcSteady`` OpenFAST can add artificial damping forces on the tower and blade
nodes (``Twr_Kdmp``, ``Bld_Kdmp``) to settle the trim faster. Those forces also act on
the rotor, so at the trimmed point the generator torque balances the aerodynamic
torque *minus* what the damping absorbs, and the torque balance understates the
aerodynamic power.

In tests on the IEA 10 MW monopile deck (OpenFAST 4.0.5) trimmed with
``Twr_Kdmp = Bld_Kdmp = 5000`` N/(m/s), AeroDyn's ``RtFldPwr`` was 436.6 kW at 4 m/s
against 396.7 kW from the torque balance, and 10.95 MW against 10.60 MW at 25 m/s.
The other possible explanations were ruled out on the data:

* the ``ED Generator torque`` input already contains ServoDyn's torque-trim offset
  (``y%GenTrq = GenTrq + CtrlOffset``), and ElastoDyn's ``RotTorq`` output equals it;
* the turbine is direct drive with ``GBRatio = 1`` and ``GBoxEff = 100 %``, so there is
  no high- versus low-speed-shaft ambiguity;
* AeroDyn's ``RtSpeed`` matches the rotor speed;
* the azimuth-mean rotor acceleration is :math:`10^{-6}` rad/s² or less below rated, so
  the rotor is not still accelerating;
* ``RtAeroPwr`` is :math:`\Omega M_{x,h}` of the blade loads alone (no tower or nacelle
  term).

Rerunning the 4 m/s point with ``Twr_Kdmp = Bld_Kdmp = 0`` raised the trimmed generator
torque from 631 to 685 kN m, left AeroDyn's power unchanged (436.0 kW), and reduced the
difference from 10.1 % to 1.3 %. In tests on three above-rated NREL 5 MW points trimmed
with 10000 N/(m/s) under OpenFAST 5.0.0, the torque balance still matched the curve
within 0.7 %. How much torque the damping absorbs therefore depends on the model and on
the OpenFAST version.

VANE therefore reads the trim damping of the deck beside each set's ``.lin`` files
(``<root>.fst`` beside ``<root>.<k>.lin``) into
``TrimmedOperatingPoint.trim_damping``: the larger of ``Twr_Kdmp`` and ``Bld_Kdmp``,
``0.0`` when ``CalcSteady`` is off, and ``None`` when the deck cannot be read. When it
is non-zero and the outputs carry AeroDyn's rotor power (``RtFldPwr``, else
``RtAeroPwr``), that power gives :math:`C_p`. ``point.power_source`` names the source
used (``"torque balance"`` or the output channel, for example
``"output: AD RtFldPwr"``). Without the AeroDyn output the balance is kept and a
warning is logged: add ``RtFldPwr`` to the AeroDyn ``OutList`` for damped sweeps.

The damping also moves the trimmed operating point itself (the generator torque is
trimmed lower than the undamped balance would give). Verification cannot undo that.
Where the operating point matters, trim without damping, or with as little as
convergence allows.

Turbine constants
-----------------

With ``deck=`` the constants are read by
:meth:`~vane.run.trim_verification.TurbineConstants.from_deck`: ``GBRatio``
(``GBoxRatio`` for Simplified-ElastoDyn), ``GBoxEff`` (percent; a lossless gearbox when
absent), ``TipRad`` and ``PreCone`` from the structural (``EDFile``) input, and
AeroDyn's ``AirDens``, or the ``.fst`` ``AirDens`` when AeroDyn's is ``"default"`` or
absent. BeamDyn decks keep the rotor geometry in ElastoDyn, so they are read the same
way.

The rotor radius defaults to the rotor-plane (projected) radius
:math:`R = \text{TipRad}\cos(\text{PreCone})`, the swept-area radius blade-element
tools normally normalize with. ``radius_convention="tip"`` uses ``TipRad`` itself. A
curve normalized with some other radius or density needs explicit constants:

.. code-block:: python

   from vane.run import TurbineConstants, verify_operating_points

   constants = TurbineConstants(
       gearbox_ratio=97.0,
       gearbox_efficiency=0.944,  # a fraction, not percent
       rotor_radius=63.0,
       air_density=1.225,
   )
   report = verify_operating_points(lin_sets, "curve.csv", constants=constants)

Missing data
------------

When the linearization was written without inputs (``LinInputs = 0``) the pitch and
generator torque are read from the ``ED BldPitch<i>`` and ``GenTq`` outputs instead. A
quantity that cannot be found at all is ``NaN``, its check is ``unavailable``, the
point is ``incomplete``, and ``point.issues`` says what to add (``LinInputs`` or an
``OutList`` channel). A curve without ``cp``/``ct`` columns, or the
``(wind, rpm, pitch)`` tuple that :func:`~vane.run.operating_points.operating_curve_from_csv`
returns, skips those comparisons without making a point incomplete. A wind speed
outside the curve's range is reported, not extrapolated.

Reference turbines
------------------

In tests with ``CalcSteady`` sweeps of the OpenFAST reference decks against their
published rotor-performance curves, the trimmed :math:`C_p` agreed within about 2 % and
the above-rated pitch within 0.4 deg for the NREL 5 MW turbine (published curve), and
within about 2 % and 0.15 deg for the IEA 3.4 MW turbine (blade-element curve, with its
95.5 % gearbox efficiency). The IEA 10 MW turbine, trimmed with damping and so verified
on AeroDyn's power, was 4 to 6 % above its blade-element curve below rated and within
2.5 % above rated. A larger deviation is worth investigating before the linearization
is used.
