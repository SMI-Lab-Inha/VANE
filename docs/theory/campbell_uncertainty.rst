.. SPDX-License-Identifier: Apache-2.0

Campbell diagram, resonance, and uncertainty
============================================

With the modes identified and tracked, VANE assembles the Campbell diagram, locates the
excitation resonances on it, and attaches a confidence to every mode. The confidence is
built from the numerical conditioning, the azimuth sensitivity and the tracking quality
of the mode. It can set the initial covariance of an estimator; see
:doc:`reduction_observability` and :doc:`lpv_estimation`.

.. contents:: On this page
   :local:
   :depth: 1


The sweep axis
--------------

Tracking links modes between *adjacent* operating points, so the sweep must be ordered
along a strictly monotonic operating parameter. Rotor speed increases below rated but
is constant above it, where the trimmed values differ only by round-off. Sorting by
those values would order the points arbitrarily and link points that are not
neighbours. :meth:`vane.pipeline.ModalPipeline.run` therefore rejects an axis on which
two points lie within ``parameter_rtol`` (default :math:`10^{-4}`) times the
largest-magnitude value on the axis, far above round-off and far below a physical sweep
step, and recommends the wind-speed axis, which stays monotonic across rated.
:func:`vane.campbell.build_campbell` requires a finite, strictly increasing axis.


Excitation lines and resonance
------------------------------

A rotor turning at :math:`\Omega` excites the structure at integer multiples of the
rotational frequency, the :math:`nP` lines at :math:`f = n\,\mathrm{rpm}/60`.
:func:`vane.campbell.find_resonances` tests 1P, 3P, 6P and 9P by default. A potential
resonance occurs where a mode line crosses an :math:`nP` line. Each crossing is located
by a sign change of :math:`f_{\text{mode}} - n\,\mathrm{rpm}/60` between adjacent
operating points, and the rotor speed, frequency and damping ratio at the crossing are
interpolated linearly. Each crossing carries a **damping-based severity**: high below a
damping ratio of 0.01, medium below 0.05, and low otherwise, since a lightly damped mode
at a crossing can build a large response. An optional ``operating_range`` keeps only the
crossings inside a rotor-speed window. Resonance detection needs a diagram whose axis is
the rotor speed in rev/min; the pipeline skips it on a wind-speed axis.


Azimuth-spread uncertainty
--------------------------

The averaged model :eq:`avg` discards the azimuth dependence; the **azimuth spread**
(:func:`vane.modal.azimuth_spread`) measures how much. It uses the per-azimuth matrices
kept with ``retain_per_azimuth=True`` and measures how each frozen-time matrix
:math:`\mathbf{A}_{\mathrm{NR}}(\psi)` perturbs the *averaged* model. For an isotropic
rotor every :math:`\mathbf{A}_{\mathrm{NR}}(\psi)=\bar{\mathbf{A}}` and the spread is
zero. Anisotropy makes the per-azimuth matrices differ from the average by an amount the
spread captures. Applying :eq:`cond` with
:math:`\delta\mathbf{A} = \mathbf{A}_{\mathrm{NR}}(\psi)-\bar{\mathbf{A}}` gives

.. math::

   \lambda_i(\psi) \approx \lambda_i +
   \frac{\mathbf{y}_i^{\mathsf H}\big(\mathbf{A}_{\mathrm{NR}}(\psi)-\bar{\mathbf{A}}\big)
   \mathbf{x}_i}{\mathbf{y}_i^{\mathsf H}\mathbf{x}_i},

and the standard deviations over the azimuth samples of the natural frequency
:math:`|\lambda_i(\psi)|/2\pi` and of the damping ratio
:math:`-\operatorname{Re}\lambda_i(\psi)/|\lambda_i(\psi)|` form the spread. The modes
are those of :func:`vane.modal.modes_from_mbc` for the same result, in the same order.
This first-order spread is the inexpensive companion to the exact :doc:`floquet`
treatment: it flags azimuth sensitivity from the averaged eigenpairs already computed,
while the Floquet error :eq:`avg-error` measures the averaging error exactly.

The sources are combined into one **unified confidence** in :math:`[0,1]`
(:func:`vane.modal.unified_mode_confidence`) as a product of independent factors,

.. math::
   :label: confidence

   c_i = \underbrace{\kappa_i^{-1}}_{\text{conditioning}}\ \cdot\
   \underbrace{d_i}_{\text{degeneracy}}\ \cdot\
   \underbrace{\exp\!\big(-\sigma_{f,i}/(f_i\,s)\big)}_{\text{azimuth stability}}\ \cdot\
   \underbrace{t_i}_{\text{tracking}},

where :math:`\kappa_i` is the eigenvalue condition number, :math:`d_i` is 0.5 for a
degenerate mode and 1 otherwise, :math:`\sigma_{f,i}` is the frequency spread,
:math:`s` is the relative-spread scale (``spread_scale``, default 0.05), and
:math:`t_i` is the optional tracking confidence. The spread and tracking factors are
omitted when they are not supplied. A low :math:`c_i` can be mapped to a larger initial
covariance for the estimator (:func:`vane.sysid.covariances_from_confidence`).


Confidence calibration
----------------------

The unified confidence :math:`c_i` is useful only if it is *calibrated*: among the modes
assigned confidence :math:`c`, a fraction of about :math:`c` should be correct.
:func:`vane.modal.calibrate_confidence` measures this against a correctness label
:math:`y_i\in\{0,1\}` supplied by the user (for instance, whether a predicted physical
label matches a known one) with a **reliability diagram**, confidence binned into equal
bins (``n_bins``, default 10) against the observed accuracy. It is summarized by the
**expected** and **maximum calibration errors**

.. math::

   \mathrm{ECE} = \sum_b \frac{n_b}{N}\,\big|\,\mathrm{acc}_b - \mathrm{conf}_b\,\big|,
   \qquad
   \mathrm{MCE} = \max_b \big|\,\mathrm{acc}_b - \mathrm{conf}_b\,\big|,

over the non-empty bins :math:`b` (:math:`n_b` samples each, :math:`N` in total),
together with the **Brier score** :math:`\tfrac1N\sum_i (c_i-y_i)^2` [Brier1950]_. A
monotone **recalibration map** :math:`g:[0,1]\to[0,1]` is fitted by isotonic regression
(pool-adjacent-violators [Barlow1972]_): the non-decreasing function minimizing
:math:`\sum_i (g(c_i)-y_i)^2`. The value :math:`g(c)` then reads as a probability of
being correct (:func:`vane.modal.recalibrate`), and
:func:`vane.modal.confidence_threshold` inverts it to find the smallest confidence that
meets a target accuracy.

Two properties follow from the construction. If the labels are drawn as
:math:`y_i\sim\mathrm{Bernoulli}\big(h(c_i)\big)` for a known function :math:`h`, a large
sample recovers :math:`g\approx h`. For :math:`h(c)=c` with :math:`c` uniform on
:math:`[0,1]` the ECE is close to zero and the Brier score approaches
:math:`\mathbb{E}[c-c^2]=\tfrac16`, while an over-confident :math:`h(c)=c^2` shows a
positive calibration gap that the map removes. Because the isotonic fit minimizes the
squared error over all non-decreasing maps, which include the identity, recalibration
never increases the Brier score on the data it was fitted to.
