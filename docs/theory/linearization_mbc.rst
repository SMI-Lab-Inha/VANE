.. SPDX-License-Identifier: Apache-2.0

Linearized dynamics and the multi-blade transform
==================================================

.. contents:: On this page
   :local:
   :depth: 1


Linearized turbine dynamics
---------------------------

OpenFAST linearizes the coupled aero-hydro-servo-elastic equations of motion about a
periodic steady operating point, which gives a first-order state-space model

.. math::
   :label: lti

   \dot{\mathbf{x}}(t) = \mathbf{A}(\psi)\,\mathbf{x}(t) + \mathbf{B}(\psi)\,\mathbf{u}(t),
   \qquad
   \mathbf{y}(t) = \mathbf{C}(\psi)\,\mathbf{x}(t) + \mathbf{D}(\psi)\,\mathbf{u}(t),

where the state :math:`\mathbf{x}` collects the structural and actuator degrees of
freedom (DOFs), their rates, and the first-order states of the other modules. For a
spinning rotor the system matrices depend on the azimuth :math:`\psi=\Omega t`, so
:eq:`lti` is **linear time-periodic** (LTP) with period :math:`T = 2\pi/\Omega`, not
time-invariant. The structural block has the second-order form

.. math::

   \mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}_d\dot{\mathbf{q}} + \mathbf{K}\mathbf{q}
   = \mathbf{f}.

OpenFAST does not export :math:`\mathbf M`, :math:`\mathbf C_d` or :math:`\mathbf K`,
so every modal quantity in VANE is computed from :math:`\mathbf A`. VANE reorders the
states into the layout
:math:`\mathbf{x} = [\,\mathbf{q}_2;\ \dot{\mathbf{q}}_2;\ \mathbf{q}_1\,]` (second-order
displacements, their rates, then first-order states); see
:doc:`../user_guide/coordinate_systems`.

The periodic steady state
~~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`vane.run` subpackage can also produce this model. For a rotating point it
uses OpenFAST's periodic steady-state trim (``CalcSteady``) to find the operating point
before linearizing at ``NLinTimes`` azimuths (36 by default, one every 10°). Below rated
wind speed the trim adjusts the generator torque at the scheduled blade pitch; at and
above rated it adjusts the blade pitch at constant generator torque
(``above_rated_trim="pitch"``, the default; ``"torque"`` trims the torque at the
scheduled pitch instead). The steady-state schedule of rotor speed and pitch against
wind speed is supplied by the user; see :doc:`../user_guide/linearization_sweep`.

The trim is a PI law on the rotor-speed error. Its proportional gain is
``1000 * TrimGain`` and its integral gain is ``TrimGain / DT``. A torque trim gain that
is too low is the most common cause of a below-rated trim that does not converge. A
gain of about one tenth of the rated generator torque works, so
:class:`~vane.run.LinearizationSettings` derives ``rated_generator_torque / 10`` when the
rated torque is given (the automatic sweep configuration infers it from the deck's
controller or ServoDyn input, or from the operating curve, when it is not). The pitch trim gain defaults to :math:`2\times10^{-4}` rad/(rad/s) for a 63 m
rotor and is scaled as :math:`2\times10^{-4}\,(63\ \mathrm{m}/R)^3` for a rotor of
radius :math:`R`, since the aerodynamic torque per unit pitch grows with about
:math:`R^3`. ``TrimTol`` bounds the normalized change of the trimmed response between
successive revolutions. It defaults to :math:`10^{-3}\,(R/63\ \mathrm{m})^3`, kept
within :math:`[10^{-3}, 10^{-2}]`; an explicit value may not exceed 0.1, because a value
near one would accept a drifting transient as periodic.

When a run fails, :class:`~vane.run.LinearizationRetryPolicy` chooses the next attempt
from the failure that OpenFAST reports:

* a trim that ran out of time gets a longer ``TMax``, a tolerance loosened by a factor
  of three but not past :math:`10^{-2}`, and a lower pitch gain. The torque gain and the
  trim damping are left unchanged, because raising them makes a slow trim diverge;
* a sweep whose rotor speed still varies across the azimuth files gets its tolerance
  tightened back to the starting value and a longer ``TMax``;
* motoring (negative generator speed), rotor overspeed or divergence lowers the trim
  gains. A failure within the first second of simulation is attributed to the trim
  damping, which is then removed. After a later failure the damping is kept for
  motoring and halved for overspeed or divergence;
* a failure during module initialization is an input error and is not retried;
* a timeout or an unrecognized failure takes a generic step: looser tolerance, more
  trim damping, a higher torque gain and a lower pitch gain.

If an at/above-rated pitch trim keeps failing by motoring, overspeed or divergence, the
last attempt trims the generator torque at the scheduled pitch instead. An optional
:class:`~vane.run.TrimTuningPolicy` sets a looser starting tolerance and a trim-damping
floor for at/above-rated points, so these points tend to converge at the first attempt.

**Physical acceptance of a floating point.** A completed trim is not by itself a
physical equilibrium. A loose tolerance can stop while the platform still drifts and
still produce finite matrices. The optional
:class:`~vane.run.LinearizationAcceptancePolicy` estimates the azimuth mean of the
ElastoDyn platform-rate states over the sweep. The mean removes periodic motion but
keeps a secular translational or angular velocity, and an operating point whose mean
exceeds either limit is rejected before the MBC transform. A rotating point needs at
least 12 distinct azimuths by default. The mean is a weighted quadrature
:math:`\bar x = \sum_i w_i x(\psi_i)` whose weights are the minimum-norm solution of

.. math::

   \sum_i w_i = 1, \qquad \sum_i w_i \cos k\psi_i = \sum_i w_i \sin k\psi_i = 0,
   \quad k = 1, 2, 3.

For seven or more samples this is the constant term of a least-squares fit of a mean
plus 1P to 3P harmonics at the azimuths actually sampled. On a uniform sweep every
:math:`w_i = 1/n`. ``CalcSteady`` takes each snapshot at the first time step at or past
its target azimuth, so the delivered azimuths deviate by up to :math:`\Omega\,\Delta t`;
the 1P to 3P harmonics still cancel exactly. The layout is rejected when the conditions
cannot be met (a harmonic aliases onto the mean) or when the weights increase the
variance of the mean, :math:`n\sum_i w_i^2`, by more than a factor of two over equal
weights. This happens on a partial revolution, where the mean would be extrapolated
from large alternating weights. The result is recorded with the run.

**Generator settings for the trim.** The torque trim regulates rotor speed through the
generator, so VANE keeps the generator DOF (``GenDOF``) enabled at a trimmed rotating
point unless the user overrides it. The Bladed-style controller library is not
linearizable, and ``trim_controller=True`` (part of
:meth:`~vane.run.LinearizationSettings.offshore`) replaces it with ServoDyn's simple
variable-speed law and the CalcSteady trim. ServoDyn's linearization checks require
timed generator control, so ``GenTiStr`` and ``GenTiStp`` are set and the timed start
or stop is placed beyond ``TMax``. Below rated, the generator stays off for the whole
run and the torque trim alone sets the generator torque; the simple law is neutralized
(``VS_Rgn2K = 0`` with small ``VS_RtGnSp`` and ``VS_SlPc``) so that ServoDyn's input
checks pass even when the deck's simple-law fields are placeholders. At and above rated,
the pitch trim needs a constant generator torque, so the law is flattened to the
constant ``VS_RtTq``. That torque is taken from ``rated_generator_torque`` when given,
otherwise from the deck's ``VS_RtTq``, but only when the deck's original ``VSContrl`` is
the simple law (``1``). A deck controlled by a library (``VSContrl = 5``) carries a
placeholder there (``9999.9`` in the NREL 5-MW decks), so an at/above-rated trimmed
point on such a deck requires ``rated_generator_torque`` and is rejected before any file
is written.


The multi-blade coordinate (Coleman) transform
-----------------------------------------------

A direct eigenanalysis of :math:`\mathbf{A}(\psi)` depends on azimuth, and the blade
DOFs are coupled by the rotation. The **multi-blade coordinate** (MBC), or Coleman,
transform [Coleman1958]_ [Bir2008]_ projects the :math:`N_b` identical blade DOFs into
the fixed (non-rotating) frame, where an isotropic rotor becomes time-invariant.
:func:`vane.io.mbc3_transform` implements the three-blade transform; a rotating state
that forms no blade triplet (for example on a two-bladed rotor) stays in the rotating
frame, and a warning is logged.

Let :math:`q^{(b)}` be a blade DOF on blade :math:`b` at azimuth
:math:`\psi_b = \psi + 2\pi(b-1)/N_b`. The blade coordinates are written as

.. math::

   q^{(b)} = q_0 + q_c\cos\psi_b + q_s\sin\psi_b ,

and the inverse gives the collective, cosine-cyclic and sine-cyclic coordinates

.. math::
   :label: mbc

   q_0 = \frac{1}{N_b}\sum_{b=1}^{N_b} q^{(b)}, \qquad
   q_{c} = \frac{2}{N_b}\sum_{b=1}^{N_b} q^{(b)}\cos\psi_b, \qquad
   q_{s} = \frac{2}{N_b}\sum_{b=1}^{N_b} q^{(b)}\sin\psi_b .

Let :math:`\mathbf t(\psi)` be the block-diagonal matrix that applies the
:math:`3\times3` map :math:`[1,\ \cos\psi_b,\ \sin\psi_b]` to each blade triplet and the
identity to the fixed-frame DOFs (tower, drivetrain, platform). Because
:math:`\dot{\mathbf q} = \mathbf t\,\dot{\mathbf q}_{\mathrm{NR}} +
\Omega\,\partial_\psi\mathbf t\,\mathbf q_{\mathrm{NR}}`, the full rotating state is
:math:`\mathbf x = \mathbf L(\psi)\,\mathbf x_{\mathrm{NR}}` with

.. math::

   \mathbf L =
   \begin{bmatrix}
   \mathbf t & \mathbf 0 & \mathbf 0\\
   \Omega\,\partial_\psi\mathbf t & \mathbf t & \mathbf 0\\
   \mathbf 0 & \mathbf 0 & \mathbf t_1
   \end{bmatrix},

where :math:`\mathbf t_1` is the same map for the first-order states. Differentiating
:math:`\mathbf x = \mathbf L\mathbf x_{\mathrm{NR}}` gives the exact fixed-frame state
matrix at each azimuth,

.. math::
   :label: mbc-A

   \mathbf{A}_{\mathrm{NR}}(\psi) =
   \mathbf L^{-1}\big(\mathbf A\,\mathbf L - \dot{\mathbf L}\big),
   \qquad \dot{\mathbf L} = \Omega\,\partial_\psi\mathbf L + \dot\Omega\,\partial_\Omega\mathbf L .

The term :math:`\dot{\mathbf L}` contains the first and second azimuth derivatives of
:math:`\mathbf t`. The first contributes the **Coriolis** coupling
(:math:`\propto\Omega`) and the second the **centrifugal** coupling
(:math:`\propto\Omega^2`); together they produce the split between the regressive and
progressive frequencies. The implementation forms these blocks explicitly, with
:math:`\dot\Omega` set by ``omega_dot`` (zero by default, for a steady periodic point).
At zero rotor speed :eq:`mbc-A` is a similarity transform and preserves the
eigenvalues of :math:`\mathbf A`.

VANE detects the blade triplets from the channel descriptions and transforms all four
system matrices: :math:`\mathbf A` as in :eq:`mbc-A`,
:math:`\mathbf B_{\mathrm{NR}} = \mathbf L^{-1}\mathbf B\,\mathbf t_u`,
:math:`\mathbf C_{\mathrm{NR}} = \mathbf t_y^{-1}\mathbf C\,\mathbf L` and
:math:`\mathbf D_{\mathrm{NR}} = \mathbf t_y^{-1}\mathbf D\,\mathbf t_u`, where
:math:`\mathbf t_u` and :math:`\mathbf t_y` are the multi-blade maps of the inputs and
outputs. The :math:`\dot{\mathbf L}` term comes only from differentiating the state and
therefore appears only in the state equation.

The input and output operating-point vectors are transformed by the same input and
output maps and averaged with the matrices. They are kept on
:class:`vane.io.MBCResult` as ``input_operating_point`` and ``output_operating_point``.
They are needed whenever the time-invariant perturbation model is driven by absolute
measured or simulated channels: the baseline must be subtracted in the **same
transformed coordinates** before applying
:math:`\delta\mathbf y=\mathbf C\,\delta\mathbf x+\mathbf D\,\delta\mathbf u`.

**Azimuth averaging.** The averaged fixed-frame model is

.. math::
   :label: avg

   \bar{\mathbf{A}} = \frac{1}{2\pi}\int_0^{2\pi}\mathbf{A}_{\mathrm{NR}}(\psi)\,
   \mathrm{d}\psi \approx \frac{1}{N_\psi}\sum_{k=1}^{N_\psi}\mathbf{A}_{\mathrm{NR}}(\psi_k),

the arithmetic mean over the :math:`N_\psi` azimuth samples (and likewise for
:math:`\mathbf B`, :math:`\mathbf C`, :math:`\mathbf D`). For an **isotropic** rotor
every :math:`\mathbf{A}_{\mathrm{NR}}(\psi)` is identical, so :eq:`avg` is exact.
Anisotropy introduces an azimuth dependence that the :doc:`floquet` page treats exactly
and the :doc:`campbell_uncertainty` page measures as a spread. Before averaging,
:func:`~vane.io.mbc3_transform` requires distinct azimuths and a rotor speed and wind
speed that are constant across the files to within 1 % (``rotor_speed_drift_rtol``,
``wind_speed_drift_rtol``). For a rotating point it logs a warning when the azimuths
are too few or far from uniform. The number of azimuths the average needs is measured
rather than assumed; see the convergence study on the :doc:`modal_identification` page
and the error estimates in :doc:`../user_guide/azimuth_averaging`.

Fixed-basis periodic reduction
------------------------------

With ``retain_per_azimuth=True``, :func:`~vane.io.mbc3_transform` keeps the transformed
matrices of every azimuth (``per_azimuth_a`` and the corresponding ``b``, ``c``, ``d``
lists). :func:`vane.sysid.periodic_state_space_from_mbc` uses them to build a periodic
reduced model in one fixed modal basis. Let :math:`\mathbf R` reconstruct the retained
real modal coordinates into the full MBC state and :math:`\mathbf L_r` project the full
state onto them, both taken from one modal reduction of :math:`\bar{\mathbf A}`. The
reduced member at azimuth :math:`\psi_k` is

.. math::

   \mathbf A_r(\psi_k) = \bar{\mathbf A}_r
   + \mathbf L_r\left[\mathbf A(\psi_k)-\bar{\mathbf A}\right]\mathbf R,

with :math:`\mathbf B_r(\psi_k)=\bar{\mathbf B}_r+\mathbf L_r[\mathbf B(\psi_k)-\bar{\mathbf B}]`,
:math:`\mathbf C_r(\psi_k)=\bar{\mathbf C}_r+[\mathbf C(\psi_k)-\bar{\mathbf C}]\mathbf R`
and :math:`\mathbf D_r(\psi_k)=\bar{\mathbf D}_r+\mathbf D(\psi_k)-\bar{\mathbf D}`.
One basis is needed because independent modal reductions at each azimuth have arbitrary
sign, phase and mode order, so interpolating their state matrices does not define a
physical scheduled state.

This construction keeps the periodic forcing and sensing that averaging removes, but it
does not add observability. A periodic observer can be accurate in well-observed
kinematics and still miss, or wrongly amplify, a harmonic load that is local to one
mooring line or member. The transfer behaviour, the innovations and the loads at
withheld sensors should therefore still be checked against simulation or measurement.
