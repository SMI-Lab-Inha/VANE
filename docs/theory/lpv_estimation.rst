.. SPDX-License-Identifier: Apache-2.0

LPV export, load maps, and estimation
=====================================

The final layer packages the reduced, observable model for a downstream digital twin or
controller and provides the runtime estimator VANE ships: a parameter-scheduled Kalman
filter, a consistency check on its innovations, and an assessment that bounds its
steady-state accuracy before it runs. Kalman-filter virtual sensing from OpenFAST
linearizations has been demonstrated for the tower loads of a floating turbine by Branlard
et al. [Branlard2024]_. The user-level workflow is on
:doc:`../user_guide/state_space_export`.

.. contents:: On this page
   :local:
   :depth: 1


The physical-LPV family
-----------------------

A gain-scheduled controller needs a plant whose states, inputs, and outputs are the named
physical DOFs it acts on (``ED PtfmPitch``, ``GenTq``, ``BldPitch1``), not modal
coordinates. :func:`vane.export.build_physical_lpv_family` reads a directory of
linearizations and, for each operating point, assembles such a model in the physical
(rotating) frame with no MBC transform. For each operating point's azimuth files it
averages the matrices by index,

.. math::
   :label: phys-avg

   \mathbf A = \frac{1}{N_\psi}\sum_{k} \mathbf A(\psi_k), \quad
   \text{likewise } \mathbf B, \mathbf C, \mathbf D,

after checking that the files share one operating condition (wind speed matched to a
relative tolerance of :math:`10^{-3}`, rotor speed to a looser :math:`5\times10^{-2}` that
admits steady-state ripple, azimuths distinct) and an identical channel layout. The
requested channel names are resolved against the (unit-stripped) descriptions, preferring
an exact match, then a unique substring, and the position DOF over its time-derivative
twin. The corresponding submatrices are extracted by index,

.. math::

   \mathbf A_{\text{sel}} = \mathbf A[\,\mathbf i_x,\ \mathbf i_x\,], \quad
   \mathbf B_{\text{sel}} = \mathbf B[\,\mathbf i_x,\ \mathbf i_u\,], \quad
   \mathbf C_{\text{sel}} = \mathbf C[\,\mathbf i_y,\ \mathbf i_x\,], \quad
   \mathbf D_{\text{sel}} = \mathbf D[\,\mathbf i_y,\ \mathbf i_u\,].

Extracting a state subset is a truncation: the rows and columns of the unselected states
:math:`\bar{\mathbf i}_x` are removed together with their coupling into the selected
ones. The most damaging case is a second-order DOF without its rate. The position row of
:math:`\mathbf A` is exactly the unit entry in the velocity column, so selecting a
position alone yields :math:`\mathbf A_{\text{sel}}=[0]`. Before extraction VANE
therefore completes each selected DOF with its ``"First time derivative of"`` twin
(position immediately followed by velocity; opt out with ``pair_derivatives=False``). It
then measures, at every operating point, the relative coupling the selection still
drops,

.. math::
   :label: dropped-coupling

   \rho_A = \frac{\lVert \mathbf A[\,\mathbf i_x,\ \bar{\mathbf i}_x\,]\rVert_F}
                 {\lVert \mathbf A[\,\mathbf i_x,\ :\,]\rVert_F}, \qquad
   \rho_C = \frac{\lVert \mathbf C[\,\mathbf i_y,\ \bar{\mathbf i}_x\,]\rVert_F}
                 {\lVert \mathbf C[\,\mathbf i_y,\ :\,]\rVert_F},

(:math:`\rho_C=0` when no outputs are selected, and both vanish for a full-state
selection) and rejects the family if :math:`\max(\rho_A,\rho_C)` at its worst point
exceeds ``max_dropped_coupling`` (default 0.05). Both norms are recorded per point on the
returned family (``dropped_coupling_a``, ``dropped_coupling_c``). Because the Frobenius
norms are taken in the linearization's own units, :math:`\rho` screens for gross
truncations, such as a DOF cut from a stiff structural mode it is strongly coupled to,
rather than bounding the model error. To keep coupled dynamics without carrying the
states, select the full plant and condense it downstream.

The family is sorted by its scheduling variable, wind speed (m/s) or rotor speed (rad/s,
the OpenFAST convention, carried explicitly in the family's ``scheduling_unit`` so that a
controller cannot mistake it for rpm), which must be strictly increasing. Every operating
point must resolve the requested names to the *same* full descriptions, so state index
:math:`k` means the same physical quantity across the whole family. This is the VANE
entry point for gain-scheduled controller design.


Load maps (quantities of interest)
----------------------------------

To monitor a structural load, one needs its sensitivity to the state. For an output row
:math:`\mathbf c_i` of :math:`\mathbf C`, the load is :math:`y_i = \mathbf c_i\mathbf x +
\mathbf d_i\mathbf u`, giving its state and input sensitivities directly. Written in modal
coordinates :math:`\mathbf x = \boldsymbol\Phi\,\mathbf q`, the load's dependence on the
identified modes is the residue matrix

.. math::
   :label: qoi-residue

   \mathbf R = \mathbf C\,\boldsymbol\Phi \in \mathbb C^{\,n_y \times n_{\text{modes}}},
   \qquad y_i = (\mathbf c_i\boldsymbol\Phi)\,\mathbf q = \mathbf R_i\,\mathbf q,

so :math:`|\mathbf R_{ik}|` and :math:`\arg \mathbf R_{ik}` are the gain and phase with
which mode :math:`k` appears in load :math:`i`. This links mode identification to load
monitoring by answering *which modes drive this load*.
:func:`~vane.sysid.qoi.build_qoi_map` tabulates these per (load, mode)
(:meth:`~vane.sysid.qoi.QoiMap.modal_participation_table`), normalized by the strongest
mode for each load. The residues are a screening quantity in the linearization's own
units over the retained underdamped modes. They are not fatigue, design, or extreme
loads, and must not substitute for a load-analysis or assessment calculation. The
reduced state-space model, its per-mode metadata, and a provenance manifest are written
as a self-contained ROM bundle by :func:`vane.export.write_rom` (MATLAB or NumPy
matrices, including the full-state projection and reconstruction maps when the reduction
carries them, plus a JSON manifest recording the VANE version, dimensions, retained and
stabilized mode counts, and time domain) ready for the estimator below.


Joint ranking of loads in different units
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One reduced model often has to preserve several loads at once. For heterogeneous loads,
raw residue magnitude is not a valid joint ranking, because a load in newtons would
numerically dominate one in metres or radians. For selected output rows
:math:`i\in\mathcal Q`, VANE uses the dimensionless score

.. math::
   :label: joint-qoi-residue

   r_k = \left[\sum_{i\in\mathcal Q_+}
   \frac{|R_{ik}|^2}{\sum_j |R_{ij}|^2}\right]^{1/2},

where the inner sum runs over the candidate modes and :math:`\mathcal Q_+` contains the
rows with numerically resolved candidate-mode content, so each participating row has unit
total weight. This joint ranking is available through
``modal_reduction(..., rank_outputs=...)`` and ``build_rom_family(..., rank_outputs=...)``.

Static gain and feedthrough correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a continuous state space, the equilibrium input-output gain
(:func:`~vane.sysid.state_space.static_gain`) is

.. math::
   :label: static-gain-correction

   \mathbf G(0) = \mathbf D - \mathbf C\mathbf A^{-1}\mathbf B.

The distinction between :math:`\mathbf G(0)` and :math:`\mathbf D` matters most for a
subsystem with many internal states driven by kinematic inputs, such as a dynamic
mooring-line model in an OpenFAST linearization. Dynamic and quasi-static line models
differ in the regimes dominated by line inertia and hydrodynamic drag [Hall2014]_, and a
dynamic model can expose a very large instantaneous endpoint-stiffness term in
:math:`\mathbf D` that is cancelled at equilibrium by the response of its internal
nodes. A modal truncation that keeps the original :math:`\mathbf D` destroys that
cancellation. With ``modal_reduction(..., match_dc_gain=True)``, VANE instead sets

.. math::

   \mathbf D_r = \mathbf G(0) + \mathbf C_r\mathbf A_r^{-1}\mathbf B_r,

so the reduced and full models have exactly the same zero-frequency gain. Matching the
DC gain is a necessary reduction invariant, not evidence that the reduced dynamics are
accurate.


The scheduled (LPV) estimator
-----------------------------

At runtime the operating point drifts, so the estimator schedules its model to the current
parameter each step, in the gain-scheduling sense [Rugh2000]_. Given a family tracked into
a common basis (see :doc:`reduction_observability`) and a target parameter value,
:func:`~vane.sysid.lpv.schedule_model` brackets it between the two adjacent family
members and blends the realizations entry by entry,

.. math::
   :label: lpv-blend

   t = \frac{v - p_{\text{lo}}}{p_{\text{hi}} - p_{\text{lo}}} \in [0,1], \qquad
   \mathbf M(t) = (1-t)\,\mathbf M_{\text{lo}} + t\,\mathbf M_{\text{hi}}
   \quad (\mathbf M \in \{\mathbf A, \mathbf B, \mathbf C, \mathbf D\}),

which is legitimate because the family shares a phase-continuous, scale-normalized,
mode-consistent basis (:eq:`phase-align`, :eq:`modal-scale`). A value outside the family
range is clamped to the nearest endpoint and flagged as extrapolated. In
:func:`~vane.sysid.lpv.run_scheduled_filter`, each step's scheduled continuous model is
discretized (:eq:`zoh`), the filter's plant is replaced by it, and the predict/update
runs in the common basis, so state and covariance carry across the sweep without
re-basing. The covariances :math:`(\mathbf Q, \mathbf R, \mathbf P_0)` are held constant;
only the plant is scheduled. Every step reports its minimum tracking MAC and the
extrapolation flag, so low-confidence regions are visible in the output; a MAC below
``mac_warn_threshold`` (default 0.8) is logged.


The Kalman filter
-----------------

The core recursion (:class:`~vane.sysid.kalman_filter.KalmanFilter`,
:func:`~vane.sysid.kalman_filter.run_filter`) is the discrete-time Kalman filter
[Kalman1960]_ in a numerically stable form. Its control timing is that of the
zero-order-hold discretization, which holds each input over the sample interval that
*follows* it: the sampled plant is :math:`\mathbf x_k = \mathbf A_d\mathbf x_{k-1} +
\mathbf B_d\mathbf u_{k-1} + \mathbf w_{k-1}`, :math:`\mathbf y_k = \mathbf C\mathbf x_k
+ \mathbf D\mathbf u_k + \mathbf v_k`, with :math:`\mathbf u_k` the input logged
alongside the measurement :math:`\mathbf z_k`. The prediction into step :math:`k` is
therefore

.. math::
   :label: kf-predict

   \mathbf x_k^- = \mathbf A_d\mathbf x_{k-1}^+ + \mathbf B_d\mathbf u_{k-1}, \qquad
   \mathbf P_k^- = \mathbf A_d\mathbf P_{k-1}^+\mathbf A_d^{\mathsf T} + \mathbf Q_d,

with the prior :math:`(\mathbf x_{-1}^+, \mathbf P_{-1}^+) = (\mathbf x_0, \mathbf P_0)`
one step *before* the first measurement and the first prediction driven by
:math:`\mathbf u_{-1}` (``initial_control``, defaulting to :math:`\mathbf u_0`, i.e. the
input held at its first logged value). The update uses the control logged with the
measurement only in the feedthrough. With innovation :math:`\boldsymbol\nu` and
innovation covariance :math:`\mathbf S`,

.. math::
   :label: kf-update

   \boldsymbol\nu = \mathbf z_k - (\mathbf C\mathbf x_k^- + \mathbf D\mathbf u_k), \quad
   \mathbf S = \mathbf C\mathbf P^-\mathbf C^{\mathsf T} + \mathbf R, \quad
   \mathbf K = \mathbf P^-\mathbf C^{\mathsf T}\mathbf S^{-1},

the gain is applied and the covariance updated in Joseph form, which stays symmetric
positive-definite under round-off,

.. math::
   :label: joseph

   \mathbf x^+ = \mathbf x^- + \mathbf K\boldsymbol\nu, \qquad
   \mathbf P^+ = (\mathbf I - \mathbf K\mathbf C)\,\mathbf P^-\,(\mathbf I - \mathbf K\mathbf C)^{\mathsf T}
   + \mathbf K\mathbf R\mathbf K^{\mathsf T}.

The gain and solves use the Cholesky factor of :math:`\mathbf S` rather than an explicit
inverse. Each step computes the normalized innovation squared
:math:`\text{NIS} = \boldsymbol\nu^{\mathsf T}\mathbf S^{-1}\boldsymbol\nu`; a measurement
with :math:`\text{NIS}` above a :math:`\chi^2` outlier gate is rejected and the step left
at its prediction. The gate is either a fixed threshold (``gate``) or, with
``gate_probability`` :math:`p`, the per-step quantile :math:`\chi^2_{p,\,m_k}` for the
:math:`m_k` channels that reported, so the false-rejection rate stays :math:`1-p` under
dropout. Partial-measurement masks (``available``) let channels drop out. The fixed-gain
:class:`~vane.sysid.kalman_filter.SteadyStateKalmanFilter` variant uses the steady-state
gain of the next section and skips the per-step covariance recursion.

**Consistency.** The filter is consistent when the mean NIS sits inside its
:math:`\chi^2` confidence band [BarShalom2001]_
(:func:`~vane.sysid.kalman_filter.filter_consistency`),

.. math::
   :label: nis-consistency

   \frac{\chi^2_{\alpha/2,\,\nu}}{n} \ \le\ \frac{1}{n}\sum_{k} \text{NIS}_k \ \le\
   \frac{\chi^2_{1-\alpha/2,\,\nu}}{n}, \qquad
   \nu = \sum_k n_{\text{active},k},

with :math:`n` the number of tested steps and :math:`\nu` the total number of measurement
channels in the tested innovations. The sum runs over every step with a finite NIS,
including gated-out steps (their NIS is computed before the gate): the accepted NIS alone
follow a :math:`\chi^2` truncated at the gate, whose lower mean would make a correctly
tuned, gated filter look under-confident. A backward RTS smoother [Rauch1965]_
(:func:`~vane.sysid.kalman_filter.rts_smoother`) refines the estimate off-line, and any
linear functional of the state, a load :math:`\mathbf z = \mathbf L\mathbf x`, is
reconstructed with its own error bar :math:`\boldsymbol\sigma =
\sqrt{\operatorname{diag}(\mathbf L\mathbf P\mathbf L^{\mathsf T})}`
(:func:`~vane.sysid.kalman_filter.reconstruct_load`).


Steady-state assessment
-----------------------

Before a filter is run, its steady-state accuracy is bounded from the model alone
(:func:`~vane.sysid.assessment.steady_state_estimator`). Solving the discrete algebraic
Riccati equation gives the steady predicted covariance :math:`\mathbf P^-`, hence the
steady gain :math:`\mathbf K = \mathbf P^-\mathbf C^{\mathsf T}\mathbf S^{-1}` and the
filtered covariance :math:`\mathbf P^+` (Joseph form), from which each load's error
standard deviation (:func:`~vane.sysid.assessment.assess_load`) is

.. math::
   :label: assessment

   \sigma_i = \sqrt{\mathbf L_i\,\mathbf P^+\,\mathbf L_i^{\mathsf T}}.

The load is marked ``within_tolerance`` only when a tolerance is supplied and
:math:`\sigma_i` is within it; without a tolerance the value is reported but not marked.
If the Riccati equation has no stabilizing solution, because
:math:`(\mathbf A_d, \mathbf C)` is not detectable from the sensors or a marginal or
unstable mode receives no process noise, the function raises an error rather than
returning a bound. The assessment is an a-priori
bound that needs no data, and it is conditional on the model: it assumes that the
linearization describes the plant and that :math:`\mathbf Q_d` and :math:`\mathbf R` are
the true, white, Gaussian noise covariances. It is not a field-validated accuracy claim:
validate the estimator against nonlinear time-domain simulation or measured data, and use
the NIS consistency check of :eq:`nis-consistency` on every run, before relying on it in
operation.
