.. SPDX-License-Identifier: Apache-2.0

Floquet analysis of the periodic system
=======================================

The multi-blade transform of the :doc:`linearization_mbc` page removes the azimuth
dependence of :eq:`lti` **only for an isotropic system**: identical blades on an
axisymmetric support in uniform inflow. Wind shear, tower shadow, gravity, a two-bladed
rotor, or a floating support whose mooring and hydrodynamics break the rotational
symmetry leave the transformed model time-periodic, and the azimuth-averaged model
:eq:`avg` becomes an approximation. Floquet theory is the exact treatment of the
periodic system. VANE computes it (:mod:`vane.modal.floquet`) and uses it as the
reference for measuring the error of the averaged model.

.. contents:: On this page
   :local:
   :depth: 1


Floquet decomposition
---------------------

For the periodic system :math:`\dot{\mathbf x} = \mathbf A(\psi)\,\mathbf x` with
:math:`\mathbf A(\psi+2\pi) = \mathbf A(\psi)` and :math:`\psi = \Omega t`, Floquet's
theorem [Floquet1883]_ writes every solution as

.. math::
   :label: floquet-form

   \mathbf x(t) = e^{\lambda t}\,\mathbf p(t), \qquad \mathbf p(t+T) = \mathbf p(t),

with :math:`T = 2\pi/\Omega`. The **Floquet exponents** :math:`\lambda` are the modal
eigenvalues of the periodic system. The natural frequency :math:`|\lambda|` and the
damping ratio :math:`-\operatorname{Re}\lambda/|\lambda|` are read from them as in the
time-invariant case. The periodic factor carries the **harmonic mode-shape components**

.. math::

   \mathbf p(t) = \sum_{m} \mathbf p_m\, e^{j m \Omega t}.

VANE provides two independent routes to the exponents, following the wind-turbine
literature on Floquet analysis [Skjoldan2009]_ [Bir2008]_: Hill's harmonic-balance
method [Hill1886]_, which recovers the frequencies **unfolded**
(:func:`vane.modal.floquet_modes`), and the monodromy matrix, an independent check
(:func:`vane.modal.monodromy_exponents`). Both take the stack of frozen-time matrices
:math:`\mathbf A(\psi_k)` over one revolution (for example ``MBCResult.per_azimuth_a``,
kept with ``retain_per_azimuth=True``), the azimuths in radians, and the rotor speed
:math:`\Omega` in rad/s.


Hill's harmonic-balance method
------------------------------

Expand the periodic matrix in a Fourier series and substitute :eq:`floquet-form`. With
:math:`\mathbf A(\psi) = \sum_n \mathbf A_n e^{j n\psi}` and
:math:`\mathbf p(t) = \sum_m \mathbf p_m e^{j m\Omega t}`, matching harmonics turns the
periodic eigenproblem into an ordinary (infinite) eigenproblem for the **Hill matrix**.
Truncated at :math:`|m|\le M` (``n_harmonics``, default 4), its blocks are

.. math::
   :label: hill

   \mathbf H_{kl} = \mathbf A_{k-l} - j\,k\,\Omega\,\delta_{kl}\,\mathbf I,
   \qquad -M \le k,l \le M .

The Fourier coefficients are estimated from the azimuth samples by the discrete transform

.. math::
   :label: fourier

   \mathbf A_n = \frac{1}{N_\psi}\sum_{k=1}^{N_\psi} e^{-j n \psi_k}\,\mathbf A(\psi_k).

The Hill blocks need coefficients up to order :math:`2M`, so
:func:`~vane.modal.floquet_modes` requires at least :math:`4M+1` samples. With the
default :math:`M = 4` this is 17 azimuths; a 36-azimuth sweep allows :math:`M \le 8`.

.. admonition:: Limitation: Hill's method needs near-uniform azimuths

   :eq:`fourier` gives every sample the same weight, which is exact only on a uniform
   grid. :func:`~vane.modal.floquet_modes` therefore requires every azimuth interval,
   including the one that closes the revolution, to equal the mean interval within a
   relative :math:`10^{-3}` (:data:`~vane.modal.floquet.HILL_UNIFORM_RTOL`). OpenFAST
   linearizes at the time steps nearest the requested azimuths, so the delivered grid
   is uneven unless the time step is commensurate with the azimuth spacing
   (:func:`~vane.modal.averaging_error_estimate.commensurate_time_step`) and the rotor
   runs at its trimmed speed. On an uneven grid :func:`~vane.modal.floquet_modes`, and
   :func:`~vane.modal.averaging_error`, which uses it, raise a ``ValueError`` that
   names the worst interval and its relative deviation. Analyse a delivered, uneven
   grid with the monodromy route below, which weights each matrix by its own interval
   and accepts any distinct azimuths.

The eigenvalues of :eq:`hill` come in families shifted by :math:`j m\Omega`; the
**principal** exponents are the physical ones. VANE selects them as the eigenvalues whose
eigenvectors concentrate their energy in the fundamental :math:`m=0` harmonic block.
Candidates are taken in descending order of :math:`\lVert\mathbf v^{(0)}\rVert /
\lVert\mathbf v\rVert`, and one is **rejected as a shifted copy** of an exponent
:math:`\lambda_a` already kept when it equals :math:`\lambda_a + jk\Omega` (to a small
tolerance) *and* its eigenvector is :math:`\lambda_a`'s eigenvector shifted by :math:`k`
harmonic blocks. The selection stops at :math:`n` distinct solutions for :math:`n`
states. Keeping the :math:`n` highest scores is not enough: inside a
parametric-resonance region (a mode near :math:`\Omega/2`, real negative multipliers)
the exponents are :math:`\sigma_{1,2} + j\Omega/2`. The conjugate
:math:`\sigma_1 - j\Omega/2` of one is its own shifted copy with an equal score, so the
top-:math:`n` rule would return that pair twice and drop the other family, which may be
the unstable one. The eigenvector test keeps distinct solutions that share a multiplier
(a repeated eigenvalue, or an isotropic mode at exactly :math:`\Omega/2` whose
:math:`\pm j\Omega/2` pair coincide in the Hill spectrum); a degenerate Hill eigenspace
is first rotated so that its basis is ordered by fundamental energy. Ties between a
conjugate-symmetric pair prefer the non-negative imaginary part.

The selected set is then compared with the monodromy multipliers below
(:math:`e^{\lambda T}`, compared modulo :math:`j\Omega`), and a ``RuntimeWarning`` is
issued if they differ by more than :math:`10^{-2}\Omega`. The retained eigenvector
blocks are the harmonic mode shapes :math:`\mathbf p_m`
(``FloquetSolution.harmonic_mode_shapes``). Because the frequency is read from the
fundamental block rather than from a wrapped logarithm, Hill's method returns the
frequencies **unfolded**, with no ambiguity of a multiple of :math:`\Omega`. In the
isotropic limit every :math:`\mathbf A(\psi)` is equal, so :math:`\mathbf A_n = 0` for
:math:`n\neq 0`, the Hill matrix is block-diagonal, and the :math:`m=0` block is
:math:`\mathbf A_0`. Hill's method then reduces exactly to the eigenvalues of
:math:`\mathbf A_0`, the same answer as the averaged model.


The monodromy matrix
--------------------

The second route integrates the state-transition matrix over one revolution. Holding
each sample constant over the azimuth interval that follows it, the **monodromy matrix**
is the ordered product of matrix exponentials

.. math::
   :label: monodromy

   \boldsymbol\Phi = \prod_{k} \exp\!\Big(\mathbf A(\psi_k)\,\frac{\Delta\psi_k}{\Omega}\Big),
   \qquad \Delta\psi_k = \psi_{k+1}-\psi_k,\quad \sum_k \Delta\psi_k = 2\pi,

whose eigenvalues are the **Floquet multipliers** :math:`\mu = e^{\lambda T}`. The
exponents follow from

.. math::
   :label: multiplier

   \lambda = \frac{\ln \mu}{T}.

This route needs no uniform sampling: each matrix is held over its own interval
:math:`\Delta\psi_k`, so it works on any sweep of distinct azimuths, including the
uneven grids OpenFAST delivers. It is an independent check on Hill's method: different arithmetic, no Fourier fit, and no
selection rule. Its limitation is intrinsic. The multiplier :math:`\mu` is invariant
under :math:`\lambda \mapsto \lambda + j k\Omega`, so :eq:`multiplier` fixes the **real
part (damping or growth rate) exactly but the imaginary part (frequency) only modulo**
:math:`\Omega`. The principal branch of the logarithm folds the frequency into the
strip :math:`(-\Omega/2,\ \Omega/2]`, so a mode whose damped frequency exceeds
:math:`\Omega/2` is returned aliased: a 0.8 Hz mode under a 1 Hz rotor comes back at
0.2 Hz. The monodromy exponents are therefore an exact check on damping and on
frequencies inside the strip, and Hill's method gives the unfolded spectrum.


The azimuth-averaging error
---------------------------

The practical use of the Floquet analysis is to measure the error of the averaged
Campbell diagram. :func:`vane.modal.averaging_error` compares the Floquet natural
frequencies (from Hill's method) with those of the averaged model, the eigenvalues of
the mean :math:`\tfrac{1}{N_\psi}\sum_k \mathbf A(\psi_k)` that the MBC path uses, and
returns the largest gap

.. math::
   :label: avg-error

   e_{\text{avg}} = \max_i\ \big| f_i^{\text{Floquet}} - f_{\sigma(i)}^{\text{avg}} \big|
   \quad \text{(Hz)},

where the matching :math:`\sigma` is the optimal one-to-one assignment (Hungarian)
between the two spectra, not a pairing by sorted position. Under strong anisotropy the
two spectra reorder, and a positional difference would compare different physical modes
and could either inflate or hide the gap. For an isotropic rotor
:math:`e_{\text{avg}} = 0`; it grows with anisotropy and indicates when the averaged
diagram should be replaced by the Floquet spectrum.

.. list-table:: The two routes at a glance
   :header-rows: 1
   :widths: 22 20 20 38

   * - Route
     - Sampling
     - Frequency
     - Role
   * - Hill (:eq:`hill`)
     - uniform, :math:`\ge 4M+1`
     - unfolded
     - principal spectrum and harmonic mode shapes
   * - Monodromy (:eq:`monodromy`)
     - any distinct azimuths
     - folded into :math:`(-\Omega/2,\Omega/2]`
     - independent check of damping and in-strip frequencies

Both reduce to the averaged answer as the system becomes isotropic. The Floquet analysis
therefore reproduces the averaged eigenanalysis when averaging is valid and gives the
exact periodic spectrum, with a measured difference, when it is not.


Error estimates without a periodic solution
-------------------------------------------

The same errors can be predicted for each branch from the averaged eigensolution and the
harmonic content of the samples, without solving the periodic problem.
:func:`vane.modal.estimate_averaging_error` assembles these estimates; usage is described
in :doc:`../user_guide/azimuth_averaging`. The quantities below use the conventions of
:mod:`vane.modal.averaging`, with azimuths in radians.

**Quadrature on the delivered grid.** ``CalcSteady`` takes each snapshot at the first
time step at or past its target azimuth, so the delivered azimuths are slightly
non-uniform. :func:`vane.modal.interval_weights` gives the weights of the periodic
piecewise-linear interpolant of the samples. With the azimuths sorted and
:math:`g_\ell = \psi_{\ell+1}-\psi_\ell` the periodic gaps,

.. math::

   w_\ell = \frac{g_{\ell-1}+g_\ell}{4\pi},

which reduces to :math:`1/N` on a uniform grid. The difference between the arithmetic
mean that the MBC transform applies and this interval-weighted mean is the
**quadrature** error term.

**Sampling and aliasing.** Represent the samples by a harmonic model
:math:`\mathbf A(\psi)=\sum_h \mathbf A_h e^{jh\psi}` (:func:`vane.modal.fit_harmonics`).
A weighted mean :math:`\sum_\ell w_\ell\mathbf A(\psi_\ell)` then equals
:math:`\sum_h W(h)\,\mathbf A_h`, with the weight kernel
(:func:`vane.modal.weight_kernel`)

.. math::

   W(h)=\sum_\ell w_\ell\, e^{jh\psi_\ell}.

The matrix error of a grid :math:`S` relative to a reference grid is therefore

.. math::

   \mathbf E=\sum_h\big[W_S(h)-W_{\mathrm{ref}}(h)\big]\mathbf A_h .

On a uniform :math:`N`-point grid with equal weights, :math:`W(h)=e^{jh\psi_0}` when
:math:`N` divides :math:`h` and zero otherwise, so only harmonics at multiples of
:math:`N` alias onto the mean. For a rotor of :math:`B` identical blades the
fixed-frame content after MBC lies only at multiples of :math:`B`, so the first
harmonic that aliases is

.. math::
   :label: alias-order

   h_{\text{alias}} = \operatorname{lcm}(N, B)

(:func:`~vane.modal.averaging_error_estimate.first_alias_order`). A count :math:`N` that
is not a multiple of :math:`B` moves the first alias from :math:`N` to :math:`BN`.
:func:`vane.modal.alias_free_azimuth_count` returns the smallest :math:`N` with
:math:`\operatorname{lcm}(N, B)` above a given harmonic order, and
:func:`vane.modal.azimuth_count_table` predicts the error of each uniform count from
the fitted harmonics. The effect of :math:`\mathbf E` on each branch follows from
eigenvalue perturbation of the averaged eigensolution, to first or second order, or
with a structured predictor that treats the rotor-azimuth state exactly.

**Periodicity.** With biorthonormal left and right eigenvectors
:math:`\mathbf u_k^{\mathsf H}\mathbf v_k = 1` of the averaged matrix
:math:`\mathbf A_0`, second-order perturbation of the Hill matrix :eq:`hill` gives the
**sideband correction** of mode :math:`j` (:func:`~vane.modal.averaging.sideband_correction`),

.. math::
   :label: sideband

   \widehat{\Delta\lambda}_j=\sum_{h\neq0}\sum_k
   \frac{(\mathbf u_j^{\mathsf H}\mathbf A_{-h}\mathbf v_k)
   (\mathbf u_k^{\mathsf H}\mathbf A_h\mathbf v_j)}{\lambda_j+jh\Omega-\lambda_k},

the estimate of the Floquet exponent minus the averaged eigenvalue. Every mode
:math:`k` is kept in the sum; truncating it to the modes nearest in detuning is not
admissible. The denominator shows where averaging fails: when a mode lies close to a
harmonic shift :math:`\lambda_k - jh\Omega` of another, the correction grows large.

**Azimuth consistency.** For a linearization about a periodic orbit, the time derivative
of that orbit solves the linearized equations, so one Floquet exponent is exactly zero.
:func:`vane.modal.azimuth_consistency` locates the generator-azimuth and rotor-speed
states, condenses the averaged model onto that pair, and checks that one of the two
sideband-corrected exponents vanishes. The result's ``assessment`` is
``"consistent"``, ``"inconsistent-but-harmless"`` or ``"inconsistent-and-risky"``.
