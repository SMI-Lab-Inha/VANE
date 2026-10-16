.. SPDX-License-Identifier: Apache-2.0

Realization, reduction, and observability
=========================================

The identified modes describe the turbine's dynamics; a digital twin needs them as a
compact, well-conditioned state-space model it can run, together with an account of
which states its sensors can actually see. This page covers that middle layer: turning the
modes into a real state-space realization, reducing it to the modes that matter, measuring
observability, choosing sensors, and building the noise covariances a filter needs. The
user-level workflow is on :doc:`../user_guide/state_space_export`.

.. contents:: On this page
   :local:
   :depth: 1


Real modal realization and discretization
-----------------------------------------

An identified mode with eigenvalue :math:`\lambda = \sigma + j\omega` and complex mode
shape :math:`\boldsymbol\phi` is placed, through its real and imaginary modal coordinates,
in a :math:`2\times 2` block

.. math::
   :label: modal-block

   \mathbf A_i = \begin{bmatrix} \sigma & -\omega \\ \omega & \sigma \end{bmatrix},
   \qquad \text{eigenvalues } \sigma \pm j\omega,

so a block-diagonal assembly of these reproduces the identified frequencies and damping to
machine precision. When mode shapes and input/output matrices are supplied, the output
block maps the two modal states to physical responses through
:math:`[\,2\operatorname{Re}(\mathbf C\boldsymbol\phi),\ -2\operatorname{Im}(\mathbf
C\boldsymbol\phi)\,]`, and the input block is formed from the **left** eigenvector
:math:`\mathbf w_i` (normalized bi-orthonormally, :math:`\mathbf w_i^{\mathsf H}\mathbf
v_i = 1`) as :math:`[\operatorname{Re}(\mathbf w_i^{\mathsf H}\mathbf B);\
\operatorname{Im}(\mathbf w_i^{\mathsf H}\mathbf B)]`. The feedthrough :math:`\mathbf D`
is kept unchanged. A defective repeated mode, whose eigenvector cluster is
ill-conditioned (:math:`\operatorname{cond}(\mathbf V_{\text{cluster}}) > 10^{8}`),
cannot be realized as independent blocks and is rejected rather than reduced to a model
with a different transfer function.

The continuous model is discretized at sample time :math:`\Delta t` by a zero-order hold
(:meth:`~vane.sysid.state_space.StateSpace.discretized`), computed from the augmented
matrix exponential

.. math::
   :label: zoh

   \begin{bmatrix} \mathbf A_d & \mathbf B_d \\ \mathbf 0 & \mathbf I \end{bmatrix}
   = \exp\!\left(\begin{bmatrix} \mathbf A & \mathbf B \\ \mathbf 0 & \mathbf 0 \end{bmatrix}\Delta t\right),
   \qquad \mathbf A_d = e^{\mathbf A\Delta t},

so the discrete eigenvalues are :math:`e^{\lambda\Delta t}` and stability is preserved
(:math:`\operatorname{Re}\lambda<0 \Leftrightarrow |e^{\lambda\Delta t}|<1`). The output
matrices :math:`\mathbf C` and :math:`\mathbf D` are unchanged. A non-positive or
non-finite :math:`\Delta t` is rejected, since it cannot define a hold.


Modal reduction by load observability
-------------------------------------

A full linearization carries hundreds of states, many of them fast aerodynamic or
marginal modes irrelevant to a structural digital twin. Modal reduction
(:func:`~vane.sysid.rom.modal_reduction`) keeps only the underdamped modes that actually
drive a chosen output. For each mode it forms the residue

.. math::
   :label: residue-rank

   r_i = \big| \mathbf c_{\text{out}}\,\mathbf v_i \big|
   \quad\text{(ranked output)}, \qquad
   r_i = \lVert \mathbf C\,\mathbf v_i \rVert_2 \quad\text{(all outputs)},

ranks the modes by :math:`r_i`, and realizes the top :math:`n` as real blocks
(:eq:`modal-block`). Several outputs in different units (``rank_outputs``) are ranked
jointly by the dimensionless score :eq:`joint-qoi-residue`, in which each output row is
first normalized to unit residue norm. Selection runs over the underdamped modes only
(over-damped and real modes are excluded up front, and ``frequency_band_hz`` can narrow
the candidates further) and is by residue regardless of stability. Unlike balanced
truncation below, the source model therefore need not be stable: a fast unstable
aerodynamic mode is dropped only when its residue is low, like any other weak mode, and
an underdamped unstable mode with a high residue is retained. The retained fraction of
modal content is reported as :math:`\sum_{\text{kept}} r_i^2 / \sum_{\text{all}} r_i^2`
(``energy_retained``), and the retained frequencies are reproduced exactly.

Because a retained mode can be spuriously unstable, a stabilization step
(``stabilize=True``, the default) reflects it into the stable half-plane
(:math:`\sigma \mapsto -|\sigma|`) and counts it in ``n_stabilized``. The reported damping
of such a mode is then the reflected (positive) value, whereas with stabilization disabled
a retained unstable mode keeps its true growing (negative) damping.

Truncation changes the static gain :math:`\mathbf G(0) = \mathbf D - \mathbf C\mathbf
A^{-1}\mathbf B` of the model. With ``match_dc_gain=True`` the reduced feedthrough is
corrected so that the reduced and full models have the same zero-frequency gain; see
:eq:`static-gain-correction` for the case where this matters most.


Balanced truncation
-------------------

When the source model is stable and minimal, balanced truncation
(:func:`~vane.sysid.balanced.balanced_truncation`) [Moore1981]_ gives an
input–output-optimal reduction with an a-priori error bound. The controllability and
observability Gramians solve the Lyapunov equations

.. math::
   :label: lyap

   \mathbf A\mathbf W_c + \mathbf W_c\mathbf A^{\mathsf T} + \mathbf B\mathbf B^{\mathsf T}
   = \mathbf 0, \qquad
   \mathbf A^{\mathsf T}\mathbf W_o + \mathbf W_o\mathbf A + \mathbf C^{\mathsf T}\mathbf C
   = \mathbf 0

(with the discrete analogues :math:`\mathbf A\mathbf W\mathbf A^{\mathsf T} - \mathbf W +
\cdots = \mathbf 0`). With square-root factors :math:`\mathbf W_c = \mathbf L_c\mathbf
L_c^{\mathsf T}`, :math:`\mathbf W_o = \mathbf L_o\mathbf L_o^{\mathsf T}`, the Hankel
singular values are the singular values of :math:`\mathbf L_o^{\mathsf T}\mathbf L_c`.
The Gramians of a lightly damped structure are positive semi-definite in exact arithmetic
but numerically indefinite, so a Cholesky factorization can fail even for a minimal
system. VANE therefore symmetrizes each Gramian and factors it from its symmetric
eigendecomposition, :math:`\mathbf L = \mathbf E\operatorname{diag}(\sqrt{\max(\mu_i,
0)})` with eigenpairs :math:`(\mu_i, \mathbf e_i)`, which leaves the rounding-level
directions as a harmless tail of near-zero Hankel values,

.. math::

   \mathbf L_o^{\mathsf T}\mathbf L_c = \mathbf U\,\boldsymbol\Sigma\,\mathbf V^{\mathsf T},
   \qquad \boldsymbol\Sigma = \operatorname{diag}(\sigma_1 \ge \cdots \ge \sigma_n),

and the balancing transformation :math:`\mathbf T = \mathbf L_c\mathbf V\boldsymbol
\Sigma^{-1/2}` makes both Gramians equal to :math:`\boldsymbol\Sigma`. Truncating to the
:math:`k` largest :math:`\sigma_i` (``n_states``, or the values above ``tol`` times the
largest) gives the guaranteed infinity-norm bound [Glover1984]_

.. math::
   :label: hinf-bound

   \lVert \mathbf G - \mathbf G_r \rVert_\infty \ \le\ 2\sum_{i=k+1}^{n}\sigma_i ,

reported as ``error_bound``. A retained Hankel value at the level of machine precision is
rejected: it signals that the model's controllable-and-observable order is below the
requested :math:`k`, so the :math:`\sigma^{-1/2}` scaling would be numerically
meaningless. Only the retained values are checked, because the balancing transform is
formed from the retained singular vectors alone and a truncated rounding-level tail never
enters it.


Tracking a family across operating points
-----------------------------------------

A turbine is reduced at many operating points, but residue ranking (:eq:`residue-rank`)
reorders modes as they veer and cross, so mode index :math:`k` is not automatically the
same physical mode everywhere. ROM-family tracking
(:func:`~vane.sysid.rom_family.build_rom_family`) fixes a consistent state order. It
reduces every point to all candidate modes, then matches the tracked modes between
adjacent points by the MAC :eq:`mac` of their complex output residues :math:`\mathbf r =
\mathbf C\mathbf v` under an optimal (Hungarian) assignment with a frequency tie-breaker
(``frequency_weight``), as in cross-operating-point tracking. Because LAPACK returns
eigenvectors with an arbitrary phase, each matched mode is phase-aligned by the rotation

.. math::
   :label: phase-align

   \mathbf R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix},
   \qquad \theta = -\arg\!\big(\mathbf r_{\text{prev}}^{\mathsf H}\,\mathbf r_{\text{curr}}\big),

applied to the mode's block. :math:`\mathbf A` is unchanged since the rotation commutes
with :math:`\sigma\mathbf I + \omega\mathbf J`, while :math:`\mathbf B \mapsto \mathbf
R^{\mathsf T}\mathbf B` and :math:`\mathbf C \mapsto \mathbf C\mathbf R`; the rotation
multiplies the mode's complex residue by :math:`e^{j\theta}`, which aligns it with the
previous point. The phase is not the only arbitrary choice: the eigenvector normalization
also fixes each mode's real *scale* through the full model's basis, so two members that
differ only by a state-space similarity would store differently scaled
:math:`\mathbf B`/:math:`\mathbf C` blocks, and an entrywise blend of them would carry a
gain error. Before matching, each mode's two states are therefore scaled by

.. math::
   :label: modal-scale

   s_k = \sqrt{\lVert \mathbf c_k\rVert / \lVert \mathbf b_k\rVert},
   \qquad \mathbf b_k \mapsto s_k\mathbf b_k,\quad \mathbf c_k \mapsto \mathbf c_k/s_k,

with the complex block coordinates :math:`\mathbf b_k = \mathbf B_{2k} + j\mathbf
B_{2k+1}` and :math:`\mathbf c_k = \mathbf C_{:,2k} - j\mathbf C_{:,2k+1}`. This leaves
the :math:`\mathbf A` block and the transfer residue unchanged and makes
:math:`\lVert\mathbf b_k\rVert = \lVert\mathbf c_k\rVert`, a value set by the residue
alone (a mode with no input or output coordinates keeps its scale). The result is a
family of realizations in a common, phase-continuous, scale-normalized basis, which is
the prerequisite for the LPV interpolation on the :doc:`lpv_estimation` page. Each member
carries a per-mode tracking MAC (``RomFamily.tracking_mac``) that flags the
veering and crossing regions where the correspondence is uncertain.

All previously tracked modes are assigned jointly against the full candidate set, so
identity is decided by residue shape and frequency alone. Transfer-reserved observer modes
(``required_transfer_pairs`` with ``n_required_transfer_modes``) are fixed at the
reference point and then follow their physical modes by identity rather than being
re-selected by each point's transfer ranking, which can reshuffle. The one extra
constraint is that a reserved mode may not be matched to a candidate with a numerically
negligible transfer residue in those pairs, which would otherwise let output-only MAC
substitute a co-shaped structural mode the unknown input does not drive.

The family reduces every point with the same stabilization option as
:func:`~vane.sysid.rom.modal_reduction` (reflection on by default), but it reports the
physical damping of each tracked mode, :math:`\zeta = -\sigma/|\lambda|` of the
linearization's own eigenvalue, so an unstable operating point is visible as a negative
damping ratio in ``RomFamily.damping_ratios``. The boolean ``RomFamily.stabilized`` mask
records where the stored realization differs from that physics (its :math:`\mathbf A`
block carries :math:`-|\sigma|`), and a warning is logged whenever any mode was
reflected.


Composing disjoint tracked families
------------------------------------

One residue ranking need not retain every dynamic path needed by an observer. For example,
platform and load outputs may select the best structural modes while omitting a weak but
essential generator-torque-to-rotor-speed branch.
:func:`~vane.sysid.rom_family.combine_rom_families` composes two independently tracked
:class:`~vane.sysid.rom_family.RomFamily` objects without re-reducing either one. At
each common operating point the construction is

.. math::
   :label: rom-family-compose

   \mathbf A_c = \begin{bmatrix}\mathbf A_b&\mathbf0\\\mathbf0&\mathbf A_e\end{bmatrix},
   \qquad
   \mathbf B_c = \begin{bmatrix}\mathbf B_b\\\mathbf B_e\end{bmatrix},
   \qquad
   \mathbf C_c = \begin{bmatrix}\mathbf C_b&\widetilde{\mathbf C}_e\end{bmatrix},
   \qquad
   \mathbf D_c = \mathbf D_b = \mathbf D_e.

Here ``b`` denotes the base family and ``e`` the enrichment. The operating-point schedule,
input/output names, MBC labels, time domain, and feedthrough must agree exactly. A
per-operating-point frequency-separation check (``duplicate_frequency_tolerance_hz``,
default :math:`10^{-8}` Hz) rejects a modal branch present in both families, because
retaining it twice would double-count its transfer residue. Base modes remain first in
the state vector and their matrices are copied unchanged, so the base subsystem can be
compared bit for bit.

The optional ``enrichment_outputs`` argument makes the composition suitable for separated
observers. It replaces all undeclared rows of :math:`\mathbf C_e` by exact zeros, producing
:math:`\widetilde{\mathbf C}_e`, while leaving :math:`\mathbf A_e`, :math:`\mathbf B_e`,
and the complete base subsystem unchanged. A torque-channel enrichment can therefore
repair only the rotor-speed measurement path without contaminating platform, IMU, or load
reconstruction rows. Output selectors accept unambiguous names or integer row indices;
empty, duplicate, ambiguous, and out-of-range selections are rejected.

Use :func:`~vane.sysid.rom_family.combine_rom_families` only for modal families known to
represent disjoint physical branches. It is a structure-preserving union, not a substitute
for balanced truncation or a guarantee that the resulting estimator is observable. The
combined family still requires the transfer-function, stability, observability,
innovation, and held-out validation checks appropriate to its application.


Unknown-input augmentation
--------------------------

A digital twin ultimately estimates loads that are never measured: wave, wind, or mooring
forces. :func:`~vane.sysid.augmentation.augment_with_disturbance` appends a disturbance
shaping model to the state, a prior on the unknown input's dynamics, so a filter can
estimate it jointly with the structural states. With a physical model
:math:`\dot{\mathbf x} = \mathbf A\mathbf x + \mathbf B\mathbf u + \mathbf E\mathbf d`,
:math:`\mathbf y = \mathbf C\mathbf x + \mathbf D\mathbf u + \mathbf F\mathbf d` and a
shaping model :math:`\dot{\mathbf z} = \mathbf A_d\mathbf z + \mathbf w`, :math:`\mathbf d
= \mathbf C_d\mathbf z`, the augmented system on :math:`[\mathbf x;\ \mathbf z]` is

.. math::
   :label: augment

   \begin{bmatrix}\dot{\mathbf x}\\ \dot{\mathbf z}\end{bmatrix}
   = \begin{bmatrix}\mathbf A & \mathbf E\mathbf C_d\\ \mathbf 0 & \mathbf A_d\end{bmatrix}
   \begin{bmatrix}\mathbf x\\ \mathbf z\end{bmatrix}
   + \begin{bmatrix}\mathbf B\\ \mathbf 0\end{bmatrix}\mathbf u + \cdots,
   \qquad
   \mathbf C_{\text{aug}} = \begin{bmatrix}\mathbf C & \mathbf F\mathbf C_d\end{bmatrix},
   \qquad
   \hat{\mathbf d} = \begin{bmatrix}\mathbf 0 & \mathbf C_d\end{bmatrix}
   \begin{bmatrix}\mathbf x\\ \mathbf z\end{bmatrix},

with a block-diagonal process-noise density (physical block plus disturbance intensity).
The entry map :math:`\mathbf E` (``entry``) must be supplied, since VANE does not guess
where a disturbance enters; the output map :math:`\mathbf F` (``output_entry``) is
optional and zero by default. The priors provided are:

* a random walk for a quasi-constant load
  (:func:`~vane.sysid.augmentation.random_walk_disturbance`; :math:`\mathbf A_d = \mathbf
  0`, :math:`\mathbf C_d = \mathbf I`, one state per channel);
* a first-order Gauss–Markov process for a bounded perturbation
  (:func:`~vane.sysid.augmentation.gauss_markov_disturbance`; :math:`\dot z = -z/\tau +
  w`);
* a harmonic oscillator for a narrowband load at a known frequency :math:`\omega`
  (:func:`~vane.sysid.augmentation.harmonic_disturbance`; block
  :math:`[[0,\omega],[-\omega,0]]`, two states per frequency, the disturbance read from
  the first state), for example the wave peak;
* a single load made of a random walk plus harmonics, all summed into one signal
  (:func:`~vane.sysid.augmentation.random_walk_harmonic_disturbance`), for a rotor load
  with a slow component and known 1P/3P orders.

Independent priors are stacked with
:func:`~vane.sysid.augmentation.combine_disturbance_models`. The construction is
deterministic and can be audited.


Observability and the load Gramian
----------------------------------

Whether the estimator can recover a state or a load from a given sensor set is decided by
the observability Gramian (:func:`~vane.sysid.observability.observability_gramian`). For a
measurement matrix :math:`\mathbf C` with noise covariance :math:`\mathbf R`, the
noise-weighted (Fisher-information) form uses :math:`\mathbf M = \mathbf C^{\mathsf
T}\mathbf R^{-1}\mathbf C` (computed by a Cholesky solve, never forming :math:`\mathbf
R^{-1}`). The infinite-horizon Gramian solves the Lyapunov equation

.. math::

   \mathbf A^{\mathsf T}\mathbf W_o + \mathbf W_o\mathbf A = -\mathbf M
   \quad\text{(continuous)}, \qquad
   \mathbf A^{\mathsf T}\mathbf W_o\mathbf A - \mathbf W_o = -\mathbf M
   \quad\text{(discrete)},

and requires a stable system. The finite-horizon Gramian over :math:`[0,T]`
(:func:`~vane.sysid.observability.finite_horizon_observability_gramian`) needs no
stability and is evaluated exactly by the Van Loan block-exponential identity
[VanLoan1978]_

.. math::
   :label: vanloan-obs

   \exp\!\left(\begin{bmatrix} -\mathbf A^{\mathsf T} & \mathbf M \\ \mathbf 0 & \mathbf A \end{bmatrix} T\right)
   = \begin{bmatrix} \mathbf G_{11} & \mathbf G_{12} \\ \mathbf 0 & \mathbf G_{22}\end{bmatrix},
   \qquad \mathbf W_o(T) = \mathbf G_{22}^{\mathsf T}\mathbf G_{12}.

The :math:`-\mathbf A^{\mathsf T}` block grows for every stable mode, so over a whole
window it is the fastest mode that overflows the exponential. VANE therefore applies
:eq:`vanloan-obs` over :math:`2^k` sub-intervals :math:`\tau = T/2^k` with
:math:`\max|\operatorname{Re}\lambda|\,\tau \le 1` and recombines by doubling,
:math:`\mathbf W_o(2\tau) = \mathbf W_o(\tau) + \boldsymbol\Phi(\tau)^{\mathsf T}\mathbf
W_o(\tau)\boldsymbol\Phi(\tau)` with :math:`\boldsymbol\Phi = e^{\mathbf A\tau}`. A
Gramian that is itself non-finite (an unstable mode over a long window) is rejected. The
discrete finite horizon of :math:`N = \operatorname{round}(T/\Delta t)` steps is the direct
sum :math:`\sum_{k=0}^{N-1}(\mathbf A^{\mathsf T})^k\mathbf M\mathbf A^k`. This
finite-horizon form is the measure that remains valid for the marginal and aerodynamic
modes a turbine linearization carries.

From the Gramian, :func:`~vane.sysid.observability.observability_scores` reports a score
per output (the trace of the Gramian built from that output row alone) and per mode (the
trace of the Gramian's diagonal block over that mode's states).
:func:`~vane.sysid.observability.load_reconstructability` analyses a load output
:math:`\mathbf c_{\text{load}}` against a sensor set that excludes it. Projecting the
load direction onto the observable subspace (the non-null eigenvectors :math:`\mathbf
p_i`, with eigenvalues :math:`\lambda_i`, of :math:`\mathbf W_o`) gives the
reconstructable fraction. If any load component is blind the noise amplification is
infinite; otherwise it is the Cramér–Rao lower bound

.. math::
   :label: crb

   \text{noise-amp} = \sum_i \frac{(\mathbf p_i\cdot\mathbf c_{\text{load}})^2}{\lambda_i},

the a-priori variance with which the load can be recovered from that sensor set.

Whether a load is reconstructable or blind, and its noise amplification, are independent
of the state basis: under :math:`\mathbf x = \mathbf T\mathbf x'` the Gramian becomes
:math:`\mathbf T^{\mathsf T}\mathbf W_o\mathbf T` and the load row
:math:`\mathbf c_{\text{load}}\mathbf T`, leaving :math:`\mathbf c_{\text{load}}\mathbf
W_o^{+}\mathbf c_{\text{load}}^{\mathsf T}` unchanged for a reconstructable load. The
*size* of a partial projection is not: splitting a load between the observable and blind
subspaces needs a state inner product, and the reported fraction uses the Euclidean one of
the model's coordinates, so it changes with a similarity transform or state rescaling.
Rank and compare sensor sets by the noise amplification, not by a partial fraction.


Sensor selection
----------------

Given candidate sensors, :func:`~vane.sysid.observability.recommend_sensors` chooses a
subset that maximizes observability by the D-optimal criterion, the log-determinant of
the Gramian [Joshi2009]_. Because the combinatorial optimum is intractable, it grows the
set greedily,

.. math::

   \text{next} = \arg\max_{c}\ \log\det \mathbf W_o(\text{selected}\cup\{c\}),

with a small eigenvalue floor so that a singular intermediate Gramian is handled. The
constrained variant :func:`~vane.sysid.observability.constrained_sensor_selection`
honours deployment limits (``required`` and ``forbidden`` channels, per-sensor ``costs``
against a ``budget``, ``max_sensors``, and per-group caps). At each step it adds the most
informative candidate that keeps every constraint feasible, and it stops when none
remains, which may leave budget unspent; this is reported with the result. The greedy
result is a heuristic, not a guaranteed global optimum.


Process noise and covariance initialization
-------------------------------------------

A discrete filter needs the continuous process-noise spectral density :math:`\mathbf Q_c`
integrated over one step. :func:`~vane.sysid.process_noise.discretize_process_noise` does
this exactly, again by a Van Loan identity [VanLoan1978]_,

.. math::
   :label: vanloan-q

   \mathbf Q_d = \int_0^{\Delta t} e^{\mathbf A s}\,\mathbf Q_c\,e^{\mathbf A^{\mathsf T} s}\,\mathrm d s,
   \qquad
   \exp\!\left(\begin{bmatrix}-\mathbf A & \mathbf Q_c\\ \mathbf 0 & \mathbf A^{\mathsf T}\end{bmatrix}\Delta t\right)
   = \begin{bmatrix}\mathbf G_{11} & \mathbf G_{12}\\ \mathbf 0 & \mathbf G_{22}\end{bmatrix},

from which :math:`\boldsymbol\Phi = \mathbf G_{22}^{\mathsf T} = e^{\mathbf A\Delta t}` and
:math:`\mathbf Q_d = \boldsymbol\Phi\,\mathbf G_{12}`. For a stiff system the step is split
into :math:`2^k` sub-steps (chosen so the per-step decay stays bounded) and recombined by
the doubling recursion :math:`\mathbf Q_d(2\tau) = \mathbf Q_d(\tau) + \boldsymbol
\Phi(\tau)\,\mathbf Q_d(\tau)\,\boldsymbol\Phi(\tau)^{\mathsf T}`, avoiding overflow. This
:math:`\mathbf Q_d` is consistent with the same zero-order hold that produces
:math:`\mathbf A_d` in :eq:`zoh`. A modal :math:`\mathbf Q_c` with one intensity per mode
(on both of its states) is built by :func:`~vane.sysid.process_noise.modal_process_noise`,
and a measurement covariance :math:`\mathbf R = \operatorname{diag}(\sigma_i^2)` from
per-channel sensor noise standard deviations by
:func:`~vane.sysid.process_noise.measurement_noise_from_specs`.

The initial state covariance can carry the modal confidence :eq:`confidence` forward
(:func:`~vane.sysid.covariance_init.covariances_from_confidence`). Each mode's two states
start with variance

.. math::
   :label: p0-inflate

   p_k = p_{\text{base}}\,\big(1 + \alpha\,(1 - c_k)\big),

with :math:`p_{\text{base}}` = ``base_variance`` (default 1) and :math:`\alpha` =
``uncertainty_scale`` (default 10). A fully trusted mode (:math:`c_k = 1`) starts at the
base variance and a fully distrusted one (:math:`c_k = 0`) starts inflated by
:math:`(1+\alpha)`; the process and measurement covariances are scaled identities
(``process_noise`` and ``measurement_noise``). This is the mechanism by which a
less certain identification produces a more cautious estimator, linking mode confidence
to the filter on the :doc:`lpv_estimation` page.
