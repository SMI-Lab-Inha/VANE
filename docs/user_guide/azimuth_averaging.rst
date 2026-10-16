.. SPDX-License-Identifier: Apache-2.0

Azimuth averaging
=================

The azimuth-averaged (MBC) model of a rotating turbine carries three errors: the
**sampling** error of the azimuth grid, the **quadrature** error of the averaging rule
on the azimuths the linearization actually delivered, and the **periodicity** error of
replacing the periodic system by its mean. The rotor-azimuth position state can make
all three behave badly.
:mod:`vane.modal.averaging` and :mod:`vane.modal.averaging_error_estimate` predict each
term per branch from the averaged model and the harmonic content of the samples, with no
periodic (Floquet) solution, and report them in the same units: the relative change of
the damped frequency :math:`|\operatorname{Im}\lambda|` and of the damping ratio
:math:`\zeta=-\operatorname{Re}\lambda/|\lambda|`, in percent of the reference value.

.. contents:: On this page
   :local:
   :depth: 1


1. Keep the per-azimuth matrices
--------------------------------

Every step works on the fixed-frame state matrices at the delivered azimuths, so keep
them when transforming:

.. code-block:: python

   import numpy as np
   from vane.io import read_lin_file
   from vane.io.mbc_transform import mbc3_transform
   from vane.modal.averaging import locate_azimuth_state

   result = mbc3_transform(
       [read_lin_file(p) for p in lin_paths], retain_per_azimuth=True
   )
   azimuths = np.deg2rad(result.azimuths_deg)
   samples = np.asarray(result.per_azimuth_a)
   omega = result.rotor_speed_rpm * 2.0 * np.pi / 60.0
   azimuth = locate_azimuth_state(result.state_descriptions, ndof2=result.ndof2)

:func:`~vane.modal.averaging.locate_azimuth_state` finds the ElastoDyn generator-azimuth
position (``DOF_GeAz``) and rotor-speed states from their descriptions; nothing is
assumed about their index.


2. Form the reference mean on the delivered grid
------------------------------------------------

A linearization snapshot is taken at the first time step at or past its target
azimuth, so the delivered grid is uneven by up to :math:`\Omega\Delta t`. The reference
mean uses the interval weights of :func:`~vane.modal.averaging.interval_weights`,

.. math::

   \bar{\mathbf A}=\sum_\ell w_\ell\,\mathbf A(\psi_\ell),\qquad
   w_\ell=\frac{g_{\ell-1}+g_\ell}{4\pi},

the exact mean of the periodic piecewise-linear interpolant (:math:`g_\ell` are the
periodic gaps). The MBC transform's own average is the arithmetic mean; the difference
between the two is the quadrature term of the error estimate.
:func:`~vane.modal.averaging.weight_kernel` gives the discrete Fourier transform of any
weights, :math:`W(h)=\sum_\ell w_\ell e^{jh\psi_\ell}`. With the interpolating harmonic
model of :func:`~vane.modal.averaging.fit_harmonics`, the error of *any* weighted mean of
the samples, a coarse sub-grid or a hypothetical design, is exactly

.. math::

   \mathbf E=\sum_h\bigl[W_S(h)-W_{\mathrm{ref}}(h)\bigr]\mathbf A_h ,

(:func:`~vane.modal.averaging.grid_error_matrix`). On a uniform :math:`N`-grid
:math:`W(h)` vanishes unless :math:`N\mid h`; on the delivered grid it leaks at every
order, which the kernel form keeps.


3. Test the azimuth direction
-----------------------------

For an autonomous system linearized about a periodic orbit, the time derivative of the
orbit (the orbit tangent, azimuth component one) solves the linearized equations, so
one Floquet exponent is exactly zero. The averaged model then has an azimuth eigenvalue
of second order in the harmonics that its sideband correction cancels:

.. math::

   \gamma_{\mathrm{az}}=\lambda_{\mathrm{az}}+\sum_{h\neq0}\sum_k
   \frac{(\mathbf u_{\mathrm{az}}^H\mathbf A_{-h}\mathbf v_k)
   (\mathbf u_k^H\mathbf A_h\mathbf v_{\mathrm{az}})}
   {\lambda_{\mathrm{az}}+jh\Omega-\lambda_k}=0 .

The azimuth position and rotor-speed states form a pair, condensed
:math:`s^2-bs-\kappa=0`, and both of its modes are corrected this way, giving
:math:`\gamma_1` (the growing one, largest real part) and :math:`\gamma_2`. In a saddle
(:math:`\kappa>0`) the growing root is the one to quote; the root with the larger
azimuth share can be the stable one. The defect is the **residual stiffness**
:math:`\kappa_r=-\gamma_1\gamma_2`, zero when one root vanishes, and the **residual
exponent** :math:`r=\min|\gamma_{1,2}|`. A positive :math:`b` with :math:`\kappa_r=0` is
the open-loop rotor-speed divergence of a rotor at constant generator torque: it grows,
but it is physical and consistent.

.. code-block:: python

   from vane.modal.averaging import azimuth_consistency, fit_harmonics

   model = fit_harmonics(azimuths, samples, method="piecewise_linear")
   test = azimuth_consistency(model, omega, azimuth, branches=[surge, pitch])
   print(test.assessment, test.pair_exponents, test.residual_stiffness)
   for risk in test.branch_risks:
       print(risk.index, risk.decay_ratio, risk.coalescence_ratio, risk.risky)

A residual matters only for the branches it can reach. For a branch
:math:`\lambda_j=-\sigma_j+j\omega_j` the test forms

.. math::

   \nu_j=\frac{r}{\sigma_j},\qquad
   \mu_j=\frac{|\kappa_r|}{\omega_j^2+b_\gamma^2/4},\qquad
   b_\gamma=\operatorname{Re}(\gamma_1+\gamma_2).

:math:`\nu_j` compares the residual with the branch decay rate: below the scale
tolerance :math:`\tau` the residual cannot move the decay rate by more than
:math:`\tau\sigma_j` even if fully mixed into the branch. :math:`\mu_j` is the
coalescence condition of the minimal azimuth model: the azimuth pendulum, which decays
at :math:`|b|/2`, can be tuned to the branch frequency only when the azimuth stiffness
reaches :math:`\omega_j^2+b^2/4`, so :math:`\mu_j` is the share of that threshold the
residual already consumes and :math:`\sqrt{\mu_j}` the frequency of the residual pair
relative to it. The graded assessment is

* ``"consistent"``: :math:`\nu_j\le\tau` for every branch;
* ``"inconsistent-and-risky"``: some branch has :math:`\nu_j>\tau` and
  :math:`\mu_j>\tau`, so the residual is neither slow against its decay nor (for
  :math:`\tau=0.01`) a decade below it in frequency;
* ``"inconsistent-but-harmless"`` otherwise.

:func:`~vane.modal.averaging_error_estimate.estimate_averaging_error` uses its damping
tolerance as :math:`\tau` and does not let a harmless residual void the branch budget.
A fixed absolute tolerance on :math:`|\gamma|` (``1e-4`` 1/s, the ``tolerance``
argument) is reported separately as ``absolute_consistent``. In tests on 66 operating
points of five reference turbines, that absolute test rejected 64 points whatever
their branch errors. With the scaled test every fixed-bottom point is consistent or
harmless, and the floating points near rated wind, where the slow platform branches
sit within a decade of the residual pair, are risky, as are stored OC3 spar
linearizations at 8 and 12 m/s.

The test is a property of the sampled Jacobians, not of averaging, and it detects but
does not locate the defect (a trim controller still acting during the linearization
revolution is one candidate). ``stiffness`` and ``speed_damping`` are :math:`\kappa` and
:math:`b` of the condensation of the reference mean: when :math:`b` is small the pair is
nearly defective and a small error in :math:`\kappa` moves the azimuth eigenvalue by
:math:`\sqrt{\kappa}`. A negative :math:`\kappa` on a coarse grid creates an azimuth
pendulum (``pendulum_frequency``) that can merge with a platform branch.


4. Choose the azimuth treatment
-------------------------------

The near-Jordan azimuth/speed pair is what makes coarse grids fail non-perturbatively.
Three treatments remove it:

* **Tangent pinning** (recommended), :func:`~vane.modal.averaging.pin_azimuth` with the
  tangent of the dense reference mean, :func:`~vane.modal.averaging.azimuth_tangent`:
  :math:`\mathbf A'(\psi)=\mathbf A(\psi)(\mathbf I-\boldsymbol\xi\mathbf e_{\mathrm{az}}^T)`.
  Every sample then has :math:`\boldsymbol\xi` as an exact null vector, so every
  sub-grid mean has an exact zero eigenvalue, the periodic model an exact unit
  multiplier, and every other reference eigenvalue is unchanged (Brauer's theorem).
* **Zeroing the azimuth column**, :func:`~vane.modal.averaging.zero_azimuth_column`, or
  equivalently **deleting the azimuth state**,
  :func:`~vane.modal.averaging.delete_azimuth_state` (the usual Campbell practice).
  They remove the sampling sensitivity too, but **they bias the reference**: the mean of
  the azimuth column is physical (the rotation of the gravity-sagged rotor, cancelled
  by the cyclic blade coupling), and on stored floating-turbine linearizations its
  removal moved platform damping by up to 7 % of its value.

.. code-block:: python

   from vane.modal.averaging import azimuth_mean, azimuth_tangent, pin_azimuth

   reference = azimuth_mean(azimuths, samples)
   tangent = azimuth_tangent(reference, azimuth, n_samples=len(azimuths))
   pinned = pin_azimuth(samples, tangent, azimuth)

The tangent must come from a dense grid at the same operating point: the neutral
direction of a coarse mean is itself contaminated by aliasing, and a warning is issued
below 18 azimuths.


5. Estimate the error of each branch
------------------------------------

.. code-block:: python

   from vane.modal.averaging_error_estimate import estimate_averaging_error

   budget = estimate_averaging_error(
       azimuths,
       samples,
       omega,
       branches={"surge": surge_eigenvalue, "pitch": pitch_eigenvalue},
       azimuth=azimuth,
       grid=azimuths[::2],  # an interleaved half grid, measured exactly
       damping_tolerance_pct=1.0,
       frequency_tolerance_pct=0.1,
   )
   for label, branch in budget.branches.items():
       print(label, branch.sampling, branch.sampling_exact, branch.quadrature,
             branch.periodicity, branch.condition, branch.mixed,
             branch.within_tolerance)
   print(budget.sampling_status, budget.azimuth_assessment, budget.within_tolerance)

The result is an :class:`~vane.modal.averaging_error_estimate.AveragingErrorEstimate`
with one :class:`~vane.modal.averaging_error_estimate.BranchErrorEstimate` per branch.

Branches are given as approximate eigenvalues (for example from
:func:`~vane.modal.eigensolver.compute_modes` and the labeller) or as indices; by
default every oscillatory mode is assessed. With the azimuth state given, the sampling
and quadrature terms use the **structured** predictor
(:func:`~vane.modal.averaging_error_estimate.predict_branch_shift`): the azimuth column of
the error is kept as an exact rank-one term and the modes with azimuth or rotor-speed
participation are coupled exactly, every other mode at first order. Its predicted modal
assurance with the reference branch is the **hybridisation flag**. Without the azimuth
state (or after pinning) ordinary second-order perturbation is accurate. The
periodicity term is the sideband estimate above, applied to each branch.

A branch is ``within_tolerance`` when every term is within the damping and frequency
tolerances and none of the four checks below rejects it.

* **Exact term.** With ``exact=True`` (the default) the trial eigenproblem of the grid
  mean is also solved (:func:`~vane.modal.averaging_error_estimate.trial_branch_shift`) and
  its MAC-tracked branch is the sampling term used for the assessment
  (``sampling_exact``; ``sampling_term`` returns whichever is used).
* **Mode mixing.** Every estimate carries a general mode-mixing MAC, independent of the
  azimuth subspace: the exact tracking MAC of the trial eigenproblem, or, for a
  prediction, the MAC from an exact solve over the modes linked to the branch by a
  chain of couplings above 0.5 % of their detuning. It catches two branches that alias
  into each other,
  for example two heavily damped flap modes on a three-azimuth grid, which the
  azimuth-structured flag cannot see.
* **Conditioning.** ``condition`` is the eigenvalue condition number
  :math:`\kappa_j=1/|\hat{\mathbf u}_j^H\hat{\mathbf v}_j|`. To first order
  :math:`|\delta\lambda_j|\le\kappa_j\|\mathbf E\|_2` (the pseudospectral radius,
  reported as ``shift_bound``); that bound is rigorous but far from tight on
  multi-scale models. Above ``max_condition`` (default :math:`10^3`) the branch is
  ill-conditioned (``well_conditioned`` is ``False``): the perturbative terms cannot be
  trusted. On the IEA 22 MW semi-submersible at 8 m/s the platform sway branch
  (:math:`\kappa\approx1178`) moves by 4.1 % in damping on one six-azimuth sub-grid,
  where every predictor gives 0.5 to 0.7 %; the check flags it and the exact term
  measures it. Conditioning is not the
  only cause there: on the same sub-grid the surge branch (:math:`\kappa=552`) is
  under-predicted by 1.3 points, which only the exact term reveals.
* **Alias resolution.** A grid that is a sub-grid of the delivered samples is
  *measured*. Any other grid is predicted from the interpolating harmonic model, which
  holds the orders up to the Nyquist order of the reference and nothing above. An
  :math:`N`-azimuth grid of a :math:`B`-bladed rotor is first aliased at
  :math:`\operatorname{lcm}(N,B)`; when that order exceeds the resolved orders of the
  reference, the sampling term is only the resolved part of the error, the branch is
  not accepted, and ``sampling_status`` reads ``"unresolved: needs denser
  reference ..."``. A 36-azimuth reference resolves orders up to 17: it confirms
  5, 6, 9, 12 or 15 azimuths but not 7, 8, 10, 13 or 18.

The alias-resolution check matters in practice. In tests on 90 OpenFAST
linearizations of two floating turbines (8, 10, 12, 13 and 18 azimuths at three start
azimuths each), the estimate built from a 36-azimuth model predicted the 12-azimuth
errors exactly but recovered only 7 to 133 % of the others, whose first aliases 36
samples cannot observe. Without the check, 9 of 180 delivered branch cases would be
accepted although they exceed 1 % in damping or 0.1 % in frequency; with it, none is.
The same estimate built from a 72-azimuth run predicts all of them, so the method is
exact for the content its reference resolves, and a denser reference is the remedy.

The sampling error of the analyst's *own* full grid cannot be measured from that grid.
Assess its interleaved sub-grids, which bound it when the harmonic sensitivity falls
with order, or extrapolate the harmonic model as in the next step.


6. Design the azimuth count
---------------------------

After MBC an isotropic :math:`B`-bladed rotor leaves content only at multiples of
:math:`B`, so a uniform :math:`N`-sample mean is aliased only by common multiples of
:math:`N` and :math:`B`, the first being :math:`\operatorname{lcm}(N,B)`. A count that
is not a multiple of the blade number is worth :math:`B` times as many samples; the
customary 12, 18 and 36 are the least efficient choices for a three-bladed rotor.

.. code-block:: python

   from vane.modal.averaging import fit_harmonics, tower_passage_decay
   from vane.modal.averaging_error_estimate import (
       alias_free_azimuth_count,
       azimuth_count_table,
       commensurate_time_step,
   )

   alias_free_azimuth_count(3, 18)  # -> 7: lcm(7, 3) = 21 lies above order 18
   tower = tower_passage_decay(fit_harmonics(azimuths, samples), azimuth)
   table = azimuth_count_table(
       fit_harmonics(azimuths, pinned), branches, blades=3,
       damping_tolerance_pct=0.5, tail_decay=tower,
   )
   print(table.recommended_resolved, table.recommended_extrapolated,
         table.extrapolation_status)
   step = commensurate_time_step(omega, table.recommended, requested_time_step=0.025)

:func:`~vane.modal.averaging_error_estimate.azimuth_count_table` evaluates the exact
eigenvalues of each uniform :math:`N`-mean of the fitted model over a range of start
azimuths (the **resolved** error) and gives two clearly separated recommendations.

* ``recommended_resolved`` (also ``recommended``) is the count the samples support:
  the smallest count whose first alias the samples resolve (``alias_resolved``) and
  whose resolved error meets the tolerance.
* ``recommended_extrapolated`` adds a first-order, worst-phase **tail estimate** of the
  aliases above the highest resolved order. The tail envelope of each branch's
  sensitivity :math:`s_j(h)=|\mathbf u_j^H\mathbf A_h\mathbf v_j|+
  |\mathbf u_j^H\mathbf A_{-h}\mathbf v_j|` is its own geometric decay when
  :math:`s_j` peaks inside the resolved orders and falls after; otherwise (flap modes
  whose sensitivity still grows at order 15) it is the observed maximum times the
  tower-passage law :math:`(h/H)\rho^{h-H}`, with the decay :math:`\rho` fitted to the
  azimuth column of the **delivered** samples
  (:func:`~vane.modal.averaging.tower_passage_decay`); with neither, the branch is
  reported ``"unresolved: needs denser linearization"`` and no extrapolated count is
  given. The envelopes are summable, so the estimate does not depend on
  ``max_alias_order``. It is a model-based estimate of orders the samples cannot
  observe, **not a bound**: on the 90 linearizations above it was exceeded at 13 and
  18 azimuths on the semi-submersible and was up to 30 times conservative on the spar.

Evaluate the table on a pinned or neutralised model, since first-order extrapolation
is not valid with the delivered azimuth column.
:func:`~vane.modal.averaging_error_estimate.commensurate_time_step` suggests a time step,
never larger than the requested one, that puts each target azimuth on a whole number of
steps, so the delivered grid is uniform when the rotor runs at its trimmed speed.

Excluding the linearization azimuths near tower passage is not a remedy for averaging
error. The true average over the revolution includes tower passage, and dropping those
samples biases the mean. The recommended practice is an azimuth count that is not a
multiple of the blade number, the interval-weighted mean on the delivered grid, and
pinning the azimuth direction in a dense average.


Limits
------

* The structured and perturbative predictors are local: they assume a well-conditioned
  averaged eigensolution apart from the azimuth/speed pair. The error estimate checks
  this through ``max_condition`` and reports the exact term next to the prediction.
* Predictions are exact only for the content the reference resolves. Counts whose
  first alias lies above it are not accepted, and the tail estimate is not a bound.
* The sideband estimate retains every mode of the averaged model; truncating it to the
  modes nearest in detuning is not admissible.
* The consistency test needs the rotor-speed state for :math:`\kappa` and :math:`b`
  of the condensation; the pair exponents and the assessment do not.
* The framework assumes an isotropic rotor for the lcm rule. Imbalanced or two-bladed
  rotors carry content at every order; use ``blades=1``.
