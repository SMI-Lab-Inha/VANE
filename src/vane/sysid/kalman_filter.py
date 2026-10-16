# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Runnable, auditable Kalman estimator over a :class:`KalmanModel`.

The rest of :mod:`vane.sysid` *builds and confirms* an estimator-ready model; this
module *runs* it. Given the discrete model and its covariances, :class:`KalmanFilter`
performs the recursive predict/update, with the properties a load-estimation digital
twin needs to be trusted in the field:

* **Numerically stable** -- the gain is formed by a Cholesky solve (never an explicit
  inverse) and the covariance by the **Joseph form**, so the error covariance stays
  symmetric positive-semidefinite even for ill-conditioned innovations.
* **Tolerant of real measurement streams** -- each step accepts an ``available`` mask,
  so a dropped or missing sensor updates on the channels that reported; an optional
  NIS gate rejects outliers (a measurement whose normalized innovation squared exceeds a
  chi-square threshold) and the step is flagged rather than corrupting the estimate.
* **Zero-order-hold control timing** -- consistent with
  :meth:`~vane.sysid.StateSpace.discretized`, the prediction into step ``k`` uses the
  previous sample's control and the control logged with the measurement enters only
  the feedthrough: :math:`x_k = A x_{k-1} + B u_{k-1}`, :math:`y_k = C x_k + D u_k`.
* **Self-diagnosing** -- every step records the innovation and its normalized innovation
  squared (NIS); :func:`filter_consistency` checks the NIS sequence against its
  chi-square expectation so a mis-tuned ``Q``/``R`` is visible, not silent.
* **Auditable offline** -- :func:`rts_smoother` refines a recorded run with the
  Rauch-Tung-Striebel backward pass, and :func:`reconstruct_load` reads a linear load
  functional off the filtered state with its propagated standard deviation.

:class:`SteadyStateKalmanFilter` runs the same recursion with the constant steady-state
gain (from :func:`~vane.sysid.steady_state_estimator`), trading per-step Riccati updates
for a fixed-gain filter once the covariance has converged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import cho_factor, cho_solve

if TYPE_CHECKING:
    import numpy.typing as npt

    from vane.sysid.kalman_interface import KalmanModel

__all__ = [
    "ConsistencyReport",
    "FilterResult",
    "KalmanFilter",
    "SteadyStateKalmanFilter",
    "filter_consistency",
    "reconstruct_load",
    "rts_smoother",
    "run_filter",
]


@dataclass
class FilterResult:
    """The recorded trajectory of a Kalman run, for diagnostics and smoothing.

    Parameters
    ----------
    states : npt.NDArray[np.float64]
        Filtered (a-posteriori) state means, shape ``(n_steps, n_states)``.
    covariances : npt.NDArray[np.float64]
        Filtered error covariances, shape ``(n_steps, n_states, n_states)``.
    predicted_states : npt.NDArray[np.float64]
        A-priori (predicted) state means before each update, same shape as ``states``;
        retained so the RTS smoother can run without re-filtering.
    predicted_covariances : npt.NDArray[np.float64]
        A-priori error covariances, shape ``(n_steps, n_states, n_states)``.
    nis : npt.NDArray[np.float64]
        Normalized innovation squared at each step (``nan`` where no measurement was
        available; a step rejected by the NIS gate keeps its finite value).
    accepted : npt.NDArray[np.bool_]
        Whether each step's measurement was applied (``False`` when fully masked or
        rejected by the NIS gate).
    n_active : npt.NDArray[np.int_]
        Number of measurement channels in each step's innovation (its NIS degrees of
        freedom), recorded for a gated-out step too; ``0`` only where no channel was
        available. Equals ``n_outputs`` unless a partial ``available`` mask reduced it.
    state_names : list[str]
        Names of the states, aligned with the last axis of ``states``.
    """

    states: npt.NDArray[np.float64]
    covariances: npt.NDArray[np.float64]
    predicted_states: npt.NDArray[np.float64]
    predicted_covariances: npt.NDArray[np.float64]
    nis: npt.NDArray[np.float64]
    accepted: npt.NDArray[np.bool_]
    n_active: npt.NDArray[np.int_]
    state_names: list[str] = field(default_factory=list)


def _as_available(
    available: npt.NDArray[np.bool_] | None, n_outputs: int
) -> npt.NDArray[np.bool_]:
    """Return a boolean output mask of length ``n_outputs`` (all-True if ``None``)."""
    if available is None:
        return np.ones(n_outputs, dtype=bool)
    mask = np.asarray(available, dtype=bool).ravel()
    if mask.shape != (n_outputs,):
        msg = f"available must be a length-{n_outputs} mask, got shape {mask.shape}"
        raise ValueError(msg)
    return mask


def _control_width(
    b: npt.NDArray[np.float64] | None, d: npt.NDArray[np.float64] | None
) -> int | None:
    """Return the input dimension used by ``B`` (or else ``D``); ``None`` if neither."""
    if b is not None:
        return int(b.shape[1])
    if d is not None:
        return int(d.shape[1])
    return None


def _validated_control(
    control: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64] | None,
    d: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64]:
    """Return ``control`` as a finite 1-D input vector sized for ``B``/``D``.

    The control may drive the state (``B``) and/or the measurement feedthrough (``D``);
    a feedthrough-only model (``B`` absent, ``D`` present) still needs it passed so the
    update can subtract ``D u``. Reject it only when neither ``B`` nor ``D`` uses it.
    """
    n_inputs = _control_width(b, d)
    if n_inputs is None:
        msg = "model has neither B nor D; a control input has no effect"
        raise ValueError(msg)
    u = np.asarray(control, dtype=np.float64).ravel()
    if u.shape != (n_inputs,):
        msg = f"control must have length {n_inputs}, got {u.shape}"
        raise ValueError(msg)
    if not np.isfinite(u).all():
        msg = "control has non-finite values"
        raise ValueError(msg)
    return u


def _validate_gate(gate: float | None, gate_probability: float | None) -> None:
    """Validate the (mutually exclusive) fixed and probability NIS gates."""
    if gate is not None and gate_probability is not None:
        msg = "give either gate (a fixed NIS threshold) or gate_probability, not both"
        raise ValueError(msg)
    if gate is not None and (not np.isfinite(gate) or gate <= 0.0):
        msg = f"gate must be a positive finite NIS threshold, got {gate}"
        raise ValueError(msg)
    if gate_probability is not None and not 0.0 < gate_probability < 1.0:
        msg = f"gate_probability must be in (0, 1), got {gate_probability}"
        raise ValueError(msg)


def _gate_threshold(
    gate: float | None, gate_probability: float | None, n_used: int
) -> float | None:
    """Return the NIS threshold for a step with ``n_used`` channels (``None``: no gate).

    A fixed ``gate`` is used as given; a ``gate_probability`` ``p`` becomes the
    chi-square quantile ``chi2.ppf(p, n_used)``, so the per-step false-rejection rate
    stays ``1 - p`` however many channels reported.
    """
    if gate is not None:
        return float(gate)
    if gate_probability is not None:
        from scipy.stats import chi2

        return float(chi2.ppf(gate_probability, n_used))
    return None


class KalmanFilter:
    r"""A recursive discrete Kalman filter over a :class:`KalmanModel`.

    Holds the current state estimate ``x`` and error covariance ``P`` and advances them
    with :meth:`predict` and :meth:`update` (or :meth:`step` for both). The model must
    be **discrete** (``dt`` set) and carry a ``C`` matrix.

    **Control timing (zero-order hold).** The discrete model produced by
    :meth:`~vane.sysid.StateSpace.discretized` holds each input constant over the
    sample interval that *follows* it, so the sampled plant is

    .. math::

        \mathbf x_k = \mathbf A\mathbf x_{k-1} + \mathbf B\mathbf u_{k-1}
        + \mathbf w_{k-1}, \qquad
        \mathbf y_k = \mathbf C\mathbf x_k + \mathbf D\mathbf u_k + \mathbf v_k,

    with ``u_k`` the input logged alongside the measurement ``y_k``. :meth:`step`
    therefore predicts with the **previous** sample's control and uses the **current**
    one only in the feedthrough ``D u_k``. The initial ``(x0, P0)`` is the prior one
    step *before* the first measurement; the control driving that first prediction is
    ``initial_control`` when given, and otherwise the first control passed to
    :meth:`step` (the input is taken as held at its first logged value before the
    record, so a constant input is filtered exactly).

    Parameters
    ----------
    model : KalmanModel
        The discrete model and its ``Q``, ``R``, ``P0`` covariances.
    initial_state : npt.NDArray[np.float64] or None, optional
        Initial state mean ``x0`` (the prior one step before the first measurement);
        defaults to zeros.
    initial_control : npt.NDArray[np.float64] or None, optional
        The input ``u_{-1}`` held over the interval before the first measurement, which
        drives the first :meth:`step`'s prediction. ``None`` (default) uses the first
        control passed to :meth:`step` (pass zeros for a plant at rest before the
        record).

    Raises
    ------
    ValueError
        If the model is continuous (no ``dt``), has no ``C`` matrix,
        ``initial_state`` does not match the state dimension, or ``initial_control`` is
        mis-sized, non-finite, or given for a model with neither ``B`` nor ``D``.
    """

    def __init__(
        self,
        model: KalmanModel,
        *,
        initial_state: npt.NDArray[np.float64] | None = None,
        initial_control: npt.NDArray[np.float64] | None = None,
    ) -> None:
        if model.dt is None:
            msg = "KalmanFilter needs a discrete model (dt set); discretize it first"
            raise ValueError(msg)
        if model.c is None:
            msg = "KalmanFilter needs a model with a C matrix"
            raise ValueError(msg)
        self.a = np.asarray(model.a, dtype=np.float64)
        self.b = None if model.b is None else np.asarray(model.b, dtype=np.float64)
        self.c = np.asarray(model.c, dtype=np.float64)
        self.d = None if model.d is None else np.asarray(model.d, dtype=np.float64)
        self.q = np.asarray(model.q, dtype=np.float64)
        self.r = np.asarray(model.r, dtype=np.float64)
        self.state_names = list(model.state_names)
        n = self.a.shape[0]
        if initial_state is None:
            self.x: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        else:
            self.x = np.asarray(initial_state, dtype=np.float64).ravel()
            if self.x.shape != (n,):
                msg = f"initial_state must have length {n}, got {self.x.shape}"
                raise ValueError(msg)
        self.p = np.asarray(model.p0, dtype=np.float64).copy()
        # The control held over the next prediction interval (ZOH), and whether it has
        # been set (by initial_control or a previous step) -- until then, the first
        # step's own control is held backward over the first interval.
        self._held_control: npt.NDArray[np.float64] | None = (
            None
            if initial_control is None
            else _validated_control(initial_control, self.b, self.d)
        )
        self._control_primed = initial_control is not None

    def predict(self, control: npt.NDArray[np.float64] | None = None) -> None:
        r"""Advance one step with the control held over this interval.

        :math:`x \leftarrow A x (+ B u)`, :math:`P \leftarrow A P A^{\mathsf T} + Q`.

        This is the raw transition: ``control`` is the input held (zero-order hold)
        over the interval being predicted -- for the interval ending at measurement
        ``k`` that is the **previous** sample's input ``u_{k-1}``, not ``u_k``.
        :meth:`step` and :func:`run_filter` apply this timing automatically.

        Parameters
        ----------
        control : npt.NDArray[np.float64] or None, optional
            The held input; ``None`` for no input over the interval.

        Raises
        ------
        ValueError
            If ``control`` is mis-sized, non-finite, or given for a model with neither
            ``B`` nor ``D``.
        """
        x = self.a @ self.x
        if control is not None:
            u = _validated_control(control, self.b, self.d)
            if self.b is not None:
                x = x + self.b @ u
        self.x = x
        self.p = self.a @ self.p @ self.a.T + self.q
        self.p = 0.5 * (self.p + self.p.T)

    def update(
        self,
        measurement: npt.NDArray[np.float64],
        *,
        control: npt.NDArray[np.float64] | None = None,
        available: npt.NDArray[np.bool_] | None = None,
        gate: float | None = None,
        gate_probability: float | None = None,
    ) -> tuple[float, bool, int]:
        r"""Correct the state with a measurement; return its NIS, acceptance, and dof.

        The predicted measurement is :math:`C x (+ D u_k)`, where ``u_k`` is the control
        logged **with this measurement** (not the one that drove the prediction), so a
        model with a nonzero feedthrough ``D`` requires it. Uses only the **available**
        output channels (a dropped sensor is simply omitted), a Cholesky solve for the
        gain, and the Joseph-form covariance update. With a gate set, a measurement
        whose normalized innovation squared exceeds the threshold is rejected (the state
        is left at its prediction) and reported as not accepted.

        Parameters
        ----------
        measurement : npt.NDArray[np.float64]
            The measured outputs, length ``n_outputs``; entries at masked-out channels
            are ignored.
        control : npt.NDArray[np.float64] or None, optional
            The control input logged with this measurement, used to subtract the
            feedthrough ``D u`` from the predicted measurement. Required (with a
            nonzero ``D`` matrix) for a correct innovation; ignored if the model has no
            ``D``.
        available : npt.NDArray[np.bool_] or None, optional
            Boolean mask of reporting channels (length ``n_outputs``); ``None`` uses
            every channel.
        gate : float or None, optional
            A **fixed** chi-square NIS threshold for outlier rejection, applied as is
            whatever the number of reporting channels. Prefer ``gate_probability`` when
            channels can drop out.
        gate_probability : float or None, optional
            Outlier-gate acceptance probability ``p`` in ``(0, 1)``: the threshold is
            ``chi2.ppf(p, n_used)`` for the ``n_used`` channels that reported, so a
            correctly tuned filter rejects a fraction ``1 - p`` of steps regardless of
            dropout. Mutually exclusive with ``gate``; neither applies every update.

        Returns
        -------
        tuple[float, bool, int]
            The normalized innovation squared (``nan`` if no channel was available),
            whether the update was applied, and the number of channels in the
            innovation -- the NIS degrees of freedom, reported for a gated-out step
            too; ``0`` only when no channel was available.

        Raises
        ------
        ValueError
            If both gates are given, ``gate`` is not a positive finite threshold,
            ``gate_probability`` is not in ``(0, 1)``, ``measurement`` is not length
            ``n_outputs`` or has non-finite available channels, or (with a nonzero
            ``D``) ``control`` is missing, mis-sized, or non-finite.
        """
        _validate_gate(gate, gate_probability)
        n_outputs = self.c.shape[0]
        z_full = np.asarray(measurement, dtype=np.float64).ravel()
        if z_full.shape != (n_outputs,):
            msg = (
                f"measurement must have length {n_outputs} (n_outputs), got "
                f"{z_full.shape}"
            )
            raise ValueError(msg)
        mask = _as_available(available, n_outputs)
        if not mask.any():
            return float("nan"), False, 0
        n_used = int(mask.sum())
        c = self.c[mask]
        r = self.r[np.ix_(mask, mask)]
        z = z_full[mask]
        if not np.isfinite(z).all():
            msg = "measurement has non-finite values on an available channel"
            raise ValueError(msg)
        predicted = c @ self.x
        if self.d is not None and np.any(self.d):
            # A nonzero feedthrough makes the measurement depend on the input; omitting
            # the control would silently treat it as zero and bias every innovation, so
            # require it explicitly (pass zeros for a genuinely autonomous run).
            if control is None:
                msg = (
                    "model has a nonzero feedthrough D; provide the control input so "
                    "D u can be subtracted from the innovation (pass zeros if "
                    "autonomous)"
                )
                raise ValueError(msg)
            u = np.asarray(control, dtype=np.float64).ravel()
            n_inputs = self.d.shape[1]
            if u.shape != (n_inputs,):
                msg = f"control must have length {n_inputs}, got {u.shape}"
                raise ValueError(msg)
            if not np.isfinite(u).all():
                msg = "control has non-finite values"
                raise ValueError(msg)
            predicted = predicted + (self.d @ u)[mask]
        innovation = z - predicted
        s = c @ self.p @ c.T + r
        s = 0.5 * (s + s.T)
        chol = cho_factor(s)
        nis = float(innovation @ cho_solve(chol, innovation))
        threshold = _gate_threshold(gate, gate_probability, n_used)
        if threshold is not None and nis > threshold:
            return nis, False, n_used
        gain = cho_solve(chol, c @ self.p).T  # K = P C^T S^-1
        self.x = self.x + gain @ innovation
        i_kc = np.eye(self.a.shape[0]) - gain @ c
        self.p = i_kc @ self.p @ i_kc.T + gain @ r @ gain.T  # Joseph form
        self.p = 0.5 * (self.p + self.p.T)
        return nis, True, n_used

    def step(
        self,
        measurement: npt.NDArray[np.float64],
        control: npt.NDArray[np.float64] | None = None,
        *,
        available: npt.NDArray[np.bool_] | None = None,
        gate: float | None = None,
        gate_probability: float | None = None,
    ) -> tuple[float, bool, int]:
        """Predict then update; return the update's NIS, acceptance, and dof.

        ``control`` is the input ``u_k`` logged with ``measurement``. Under the
        zero-order-hold convention (see the class docstring) the prediction is driven by
        the **previous** step's control ``u_{k-1}`` (on the first step,
        ``initial_control`` or else ``u_k`` itself), and ``u_k`` enters only the
        feedthrough ``D u_k``; it is then held for the next step's prediction.

        Parameters
        ----------
        measurement : npt.NDArray[np.float64]
            The measured outputs at this step.
        control : npt.NDArray[np.float64] or None, optional
            The input logged with this measurement; ``None`` for no input.
        available, gate, gate_probability : optional
            As for :meth:`update`.

        Returns
        -------
        tuple[float, bool, int]
            As for :meth:`update`.
        """
        held = self._held_control if self._control_primed else control
        self.predict(held)
        out = self.update(
            measurement,
            control=control,
            available=available,
            gate=gate,
            gate_probability=gate_probability,
        )
        self._held_control = (
            None if control is None else np.asarray(control, dtype=np.float64).copy()
        )
        self._control_primed = True
        return out


def run_filter(
    model: KalmanModel,
    measurements: npt.NDArray[np.float64],
    *,
    controls: npt.NDArray[np.float64] | None = None,
    available: npt.NDArray[np.bool_] | None = None,
    gate: float | None = None,
    gate_probability: float | None = None,
    initial_state: npt.NDArray[np.float64] | None = None,
    initial_control: npt.NDArray[np.float64] | None = None,
) -> FilterResult:
    r"""Run a Kalman filter over a measurement sequence and record the trajectory.

    **Control timing (zero-order hold).** Row ``k`` of ``controls`` is the input
    ``u_k`` logged alongside ``measurements[k]``. Consistent with the ZOH
    discretization, the recursion is

    .. math::

        \mathbf x_k^- = \mathbf A\mathbf x_{k-1}^+ + \mathbf B\mathbf u_{k-1}, \qquad
        \nu_k = \mathbf z_k - (\mathbf C\mathbf x_k^- + \mathbf D\mathbf u_k),

    i.e. the prediction into step ``k`` uses the previous sample's control and the
    current control enters only the feedthrough. ``initial_state`` (with the model's
    ``P0``) is the prior at step ``-1``, one step *before* ``measurements[0]``; the
    first prediction is driven by ``initial_control`` (``u_{-1}``), which defaults to
    ``controls[0]`` -- the input held at its first logged value before the record, so a
    constant input gives exactly the same result as holding it throughout.

    Parameters
    ----------
    model : KalmanModel
        The discrete model and covariances.
    measurements : npt.NDArray[np.float64]
        Measured outputs, shape ``(n_steps, n_outputs)``.
    controls : npt.NDArray[np.float64] or None, optional
        Control inputs logged with the measurements, shape ``(n_steps, n_inputs)``;
        ``None`` for an autonomous model.
    available : npt.NDArray[np.bool_] or None, optional
        Per-step channel mask, shape ``(n_steps, n_outputs)``; ``None`` uses all
        channels at every step. Use this to encode sensor dropout.
    gate : float or None, optional
        A fixed chi-square NIS threshold for per-step outlier rejection (not scaled by
        the number of reporting channels).
    gate_probability : float or None, optional
        Outlier-gate acceptance probability in ``(0, 1)``; each step's threshold is the
        chi-square quantile for its number of reporting channels. Mutually exclusive
        with ``gate``.
    initial_state : npt.NDArray[np.float64] or None, optional
        Prior state mean one step before ``measurements[0]``; defaults to zeros.
    initial_control : npt.NDArray[np.float64] or None, optional
        The input ``u_{-1}`` driving the first prediction; defaults to ``controls[0]``.
        Pass zeros for a plant at rest (no input) before the record. Requires
        ``controls``.

    Returns
    -------
    FilterResult
        The filtered and predicted state means/covariances, per-step NIS, and acceptance
        flags.

    Raises
    ------
    ValueError
        If ``measurements`` is not 2-D with ``n_outputs`` columns, ``controls`` /
        ``available`` lengths disagree with it, ``initial_control`` is given without
        ``controls``, or the gate arguments are invalid.
    """
    if initial_control is not None and controls is None:
        msg = "initial_control requires controls (it is the input before controls[0])"
        raise ValueError(msg)
    _validate_gate(gate, gate_probability)
    kf = KalmanFilter(model, initial_state=initial_state)
    z = np.asarray(measurements, dtype=np.float64)
    n_outputs = kf.c.shape[0]
    if z.ndim != 2 or z.shape[1] != n_outputs:
        msg = f"measurements must be (n_steps, {n_outputs}), got {z.shape}"
        raise ValueError(msg)
    n_steps, n = z.shape[0], kf.a.shape[0]
    if controls is not None and len(controls) != n_steps:
        msg = f"controls must have {n_steps} rows, got {len(controls)}"
        raise ValueError(msg)
    if available is not None and len(available) != n_steps:
        msg = f"available must have {n_steps} rows, got {len(available)}"
        raise ValueError(msg)

    states = np.empty((n_steps, n), dtype=np.float64)
    covariances = np.empty((n_steps, n, n), dtype=np.float64)
    pred_states = np.empty((n_steps, n), dtype=np.float64)
    pred_covariances = np.empty((n_steps, n, n), dtype=np.float64)
    nis = np.empty(n_steps, dtype=np.float64)
    accepted = np.empty(n_steps, dtype=bool)
    n_active = np.empty(n_steps, dtype=int)
    # ZOH: the prediction into step k is driven by u_{k-1}; the first by u_{-1}.
    held: npt.NDArray[np.float64] | None
    if initial_control is not None:
        held = initial_control
    elif controls is not None and n_steps > 0:
        held = controls[0]
    else:
        held = None
    for k in range(n_steps):
        control_k = None if controls is None else controls[k]
        kf.predict(held)
        pred_states[k] = kf.x
        pred_covariances[k] = kf.p
        step_nis, step_ok, step_dof = kf.update(
            z[k],
            control=control_k,
            available=None if available is None else available[k],
            gate=gate,
            gate_probability=gate_probability,
        )
        held = control_k
        states[k] = kf.x
        covariances[k] = kf.p
        nis[k] = step_nis
        accepted[k] = step_ok
        n_active[k] = step_dof
    return FilterResult(
        states=states,
        covariances=covariances,
        predicted_states=pred_states,
        predicted_covariances=pred_covariances,
        nis=nis,
        accepted=accepted,
        n_active=n_active,
        state_names=list(kf.state_names),
    )


class SteadyStateKalmanFilter:
    """A constant-gain Kalman filter using the precomputed steady-state gain.

    Once the error covariance has converged (the steady-state DARE solution from
    :func:`~vane.sysid.steady_state_estimator`), the gain is constant, so each step is a
    matrix-vector predict/correct with no per-step Riccati update -- cheaper and ideal
    for an embedded twin. The covariance is fixed at the steady-state value.

    It follows the same zero-order-hold control timing as :class:`KalmanFilter`: the
    prediction into step ``k`` uses the previous step's control ``u_{k-1}`` (on the
    first step, ``initial_control`` or else that step's own control), and the control
    logged with the measurement enters only the feedthrough ``D u_k``.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        Discrete state matrix.
    c : npt.NDArray[np.float64]
        Output matrix.
    gain : npt.NDArray[np.float64]
        The steady-state Kalman gain ``K`` (``EstimatorAssessment.gain``).
    b : npt.NDArray[np.float64] or None, optional
        Input matrix, for control inputs into the state.
    d : npt.NDArray[np.float64] or None, optional
        Feedthrough matrix; when given, ``D u`` is subtracted from the measurement so a
        controlled model with ``y = C x + D u`` is filtered correctly.
    initial_state : npt.NDArray[np.float64] or None, optional
        Initial state mean (the prior one step before the first measurement); defaults
        to zeros.
    state_names : list[str] or None, optional
        Names of the states.
    initial_control : npt.NDArray[np.float64] or None, optional
        The input held over the interval before the first measurement (drives the first
        prediction); ``None`` (default) uses the first step's own control.

    Raises
    ------
    ValueError
        If the gain shape is inconsistent with ``a``/``c``, ``initial_state`` does
        not match the state dimension, or ``initial_control`` is mis-sized,
        non-finite, or given for a model with neither ``b`` nor ``d``.
    """

    def __init__(
        self,
        a: npt.NDArray[np.float64],
        c: npt.NDArray[np.float64],
        gain: npt.NDArray[np.float64],
        *,
        b: npt.NDArray[np.float64] | None = None,
        d: npt.NDArray[np.float64] | None = None,
        initial_state: npt.NDArray[np.float64] | None = None,
        state_names: list[str] | None = None,
        initial_control: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.a = np.asarray(a, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.gain = np.asarray(gain, dtype=np.float64)
        self.b = None if b is None else np.asarray(b, dtype=np.float64)
        self.d = None if d is None else np.asarray(d, dtype=np.float64)
        n = self.a.shape[0]
        if self.gain.shape != (n, self.c.shape[0]):
            msg = (
                f"gain must be (n_states, n_outputs) = {(n, self.c.shape[0])}, "
                f"got {self.gain.shape}"
            )
            raise ValueError(msg)
        if initial_state is None:
            self.x: npt.NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        else:
            self.x = np.asarray(initial_state, dtype=np.float64).ravel()
            if self.x.shape != (n,):
                msg = f"initial_state must have length {n}, got {self.x.shape}"
                raise ValueError(msg)
        self.state_names = list(state_names) if state_names is not None else []
        # ZOH held control for the next prediction (see KalmanFilter).
        self._held_control: npt.NDArray[np.float64] | None = (
            None
            if initial_control is None
            else _validated_control(initial_control, self.b, self.d)
        )
        self._control_primed = initial_control is not None

    def step(
        self,
        measurement: npt.NDArray[np.float64],
        control: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Predict then correct with the constant gain; return the new state estimate.

        :math:`x^- = A x (+ B u_{k-1})`, then
        :math:`x = x^- + K (z - C x^- (- D u_k))`, with ``u_k = control`` the input
        logged with ``measurement`` and ``u_{k-1}`` the previous step's control (the
        zero-order-hold timing of :class:`KalmanFilter`).
        """
        if control is None and self.d is not None and np.any(self.d):
            # A nonzero feedthrough makes the measurement depend on the input; omitting
            # the control would silently drop D u and bias the innovation.
            msg = (
                "model has a nonzero feedthrough D; provide the control input so D u "
                "can be subtracted from the innovation (pass zeros if autonomous)"
            )
            raise ValueError(msg)
        # The control feeds the state (B) and/or the measurement (D); validate it
        # against whichever uses it. A control with neither has no effect -- reject it
        # rather than ignore the caller's intent.
        u = None if control is None else _validated_control(control, self.b, self.d)
        held = self._held_control if self._control_primed else u
        x = self.a @ self.x
        if held is not None and self.b is not None:
            x = x + self.b @ held  # ZOH: the previous step's control drives x_k
        predicted = self.c @ x
        if u is not None and self.d is not None:
            predicted = predicted + self.d @ u  # measurement feedthrough D u
        z = np.asarray(measurement, dtype=np.float64).ravel()
        if z.shape != (self.c.shape[0],):
            msg = (
                f"measurement must have length {self.c.shape[0]} (n_outputs), got "
                f"{z.shape}"
            )
            raise ValueError(msg)
        if not np.isfinite(z).all():
            msg = "measurement has non-finite values"
            raise ValueError(msg)
        innovation = z - predicted
        self.x = x + self.gain @ innovation
        self._held_control = u
        self._control_primed = True
        return self.x

    def run(
        self,
        measurements: npt.NDArray[np.float64],
        *,
        controls: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Filter a measurement sequence; return state estimates ``(n_steps, n)``.

        Row ``k`` of ``controls`` is the input logged with ``measurements[k]``; the
        zero-order-hold timing of :meth:`step` applies (continuing from any previous
        :meth:`step` calls on this filter).

        Raises
        ------
        ValueError
            If ``measurements`` is not 2-D with ``n_outputs`` columns, or ``controls``
            has a different number of rows.
        """
        z = np.asarray(measurements, dtype=np.float64)
        n_outputs = self.c.shape[0]
        if z.ndim != 2 or z.shape[1] != n_outputs:
            msg = f"measurements must be (n_steps, {n_outputs}), got {z.shape}"
            raise ValueError(msg)
        if controls is not None and len(controls) != z.shape[0]:
            msg = f"controls must have {z.shape[0]} rows, got {len(controls)}"
            raise ValueError(msg)
        out = np.empty((z.shape[0], self.a.shape[0]), dtype=np.float64)
        for k in range(z.shape[0]):
            out[k] = self.step(z[k], None if controls is None else controls[k])
        return out


def rts_smoother(
    model: KalmanModel, result: FilterResult
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Refine a filtered run with the Rauch-Tung-Striebel backward pass.

    The smoother conditions each state on the **whole** record (not just the past),
    which an offline audit can afford: it lowers the variance and removes the filter's
    lag. The control inputs enter only through the run's recorded predictions
    (``result.predicted_states``, formed with the zero-order-hold timing of
    :func:`run_filter`), so the backward pass needs no controls of its own.

    Parameters
    ----------
    model : KalmanModel
        The same discrete model the run used (for ``A``).
    result : FilterResult
        A trajectory from :func:`run_filter`.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        The smoothed state means ``(n_steps, n_states)`` and covariances
        ``(n_steps, n_states, n_states)``.
    """
    a = np.asarray(model.a, dtype=np.float64)
    states = result.states
    covariances = result.covariances
    pred_states = result.predicted_states
    pred_covariances = result.predicted_covariances
    n_steps = states.shape[0]
    sm_states = states.copy()
    sm_cov = covariances.copy()
    for k in range(n_steps - 2, -1, -1):
        # Smoother gain G = P_k A^T (P^-_{k+1})^-1. The prior covariance can be singular
        # for a valid run (e.g. zero process noise and zero initial variance), so use a
        # least-squares solve (pseudo-inverse) instead of an exact solve, which would
        # raise on a singular prior even though the smoothed estimate is well-defined.
        prior_next = pred_covariances[k + 1]
        gain = np.linalg.lstsq(prior_next, a @ covariances[k], rcond=None)[0].T
        sm_states[k] = states[k] + gain @ (sm_states[k + 1] - pred_states[k + 1])
        sm_cov[k] = covariances[k] + gain @ (sm_cov[k + 1] - prior_next) @ gain.T
        sm_cov[k] = 0.5 * (sm_cov[k] + sm_cov[k].T)
    return sm_states, sm_cov


@dataclass
class ConsistencyReport:
    """Whether a run's innovations are statistically consistent with the model.

    Parameters
    ----------
    mean_nis : float
        Average normalized innovation squared over the tested steps -- every step with
        a finite NIS, **including** steps the outlier gate rejected (their NIS is
        computed before the gate decision).
    expected_nis : float
        The expected mean NIS under a correctly tuned filter -- the average number of
        measurement channels per tested step (``dof / (n_accepted + n_gated)``).
    lower, upper : float
        The two-sided chi-square acceptance bounds for ``mean_nis`` at the chosen level.
    consistent : bool
        Whether ``mean_nis`` lies within ``[lower, upper]``; ``False`` flags a mis-tuned
        ``Q``/``R`` (over- or under-confident filter) or a record with more outliers
        than the model's noise explains.
    n_accepted : int
        Number of accepted (applied) steps included in the statistic.
    n_gated : int, optional
        Number of gated-out (rejected) steps included in the statistic (default ``0``).
    """

    mean_nis: float
    expected_nis: float
    lower: float
    upper: float
    consistent: bool
    n_accepted: int
    n_gated: int = 0


def filter_consistency(
    result: FilterResult, *, confidence: float = 0.95
) -> ConsistencyReport:
    """Check a run's NIS sequence against its chi-square expectation (NIS test).

    For a correctly tuned filter each step's NIS, computed from the predicted
    covariance **before** any gate decision, is chi-square distributed with as many
    degrees of freedom as channels reported, so the sum over the tested steps is
    chi-square with ``dof`` = the **total number of channels** in those innovations
    (``result.n_active`` summed) -- a partial ``available`` mask, which lowers a step's
    dof, is accounted for. Steps rejected by an outlier gate are **included**: keeping
    only the accepted ones would test a chi-square truncated at the gate, whose mean is
    lower, so a correctly tuned, gated filter would be reported as under-confident. A
    mean NIS far outside the acceptance band means the filter is over-confident
    (``R``/``Q`` too small) or under-confident (too large) -- a tuning fault that would
    otherwise be invisible -- or that the record holds more outliers than the model's
    noise explains.

    Parameters
    ----------
    result : FilterResult
        A trajectory from :func:`run_filter`.
    confidence : float, optional
        Two-sided confidence level for the acceptance band (default ``0.95``).

    Returns
    -------
    ConsistencyReport
        The mean NIS, its expectation, the acceptance band, and the outcome.

    Raises
    ------
    ValueError
        If no step has a finite NIS to test, or ``confidence`` is not in ``(0, 1)``.

    Notes
    -----
    A step with a finite NIS but ``n_active == 0`` (a gated step recorded without
    the degrees of freedom of its rejected innovation) carries no degrees of freedom
    and is left out.
    """
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must be in (0, 1), got {confidence}"
        raise ValueError(msg)
    tested = np.isfinite(result.nis) & (np.asarray(result.n_active) > 0)
    tested_nis = result.nis[tested]
    n = int(tested_nis.size)
    if n == 0:
        msg = "no accepted steps (and no gated steps) with a finite NIS to test"
        raise ValueError(msg)
    from scipy.stats import chi2

    # Degrees of freedom = total channels in the tested innovations (a partial mask
    # lowers a step's dof), not n_tested * n_outputs.
    dof = int(np.asarray(result.n_active)[tested].sum())
    alpha = 1.0 - confidence
    lower = float(chi2.ppf(alpha / 2.0, dof) / n)
    upper = float(chi2.ppf(1.0 - alpha / 2.0, dof) / n)
    mean_nis = float(tested_nis.mean())
    n_accepted = int(np.count_nonzero(tested & result.accepted))
    return ConsistencyReport(
        mean_nis=mean_nis,
        expected_nis=dof / n,
        lower=lower,
        upper=upper,
        consistent=bool(lower <= mean_nis <= upper),
        n_accepted=n_accepted,
        n_gated=n - n_accepted,
    )


def reconstruct_load(
    result: FilterResult,
    load_map: npt.NDArray[np.float64],
    *,
    smoothed: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Read a linear load off the estimated state with its propagated uncertainty.

    For a load functional :math:`z = L x`, returns the reconstructed load
    :math:`\hat z_k = L \hat x_k` and its standard deviation
    :math:`\sqrt{\operatorname{diag}(L P_k L^{\mathsf T})}` at each step -- the virtual
    sensor and its error bar.

    Parameters
    ----------
    result : FilterResult
        A trajectory from :func:`run_filter`.
    load_map : npt.NDArray[np.float64]
        The load functionals ``L``, shape ``(n_loads, n_states)``.
    smoothed : tuple or None, optional
        A ``(states, covariances)`` pair from :func:`rts_smoother` to read the load from
        the smoothed estimate instead of the filtered one.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        The reconstructed loads ``(n_steps, n_loads)`` and their standard deviations
        ``(n_steps, n_loads)``.

    Raises
    ------
    ValueError
        If ``load_map`` does not match the state dimension or is non-finite.
    """
    states, covariances = (
        (result.states, result.covariances) if smoothed is None else smoothed
    )
    load = np.asarray(load_map, dtype=np.float64)
    n_states = states.shape[1]
    if load.ndim != 2 or load.shape[1] != n_states:
        msg = f"load_map must be (n_loads, {n_states}), got {load.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(load)):
        msg = "load_map must be finite (no NaN or inf)"
        raise ValueError(msg)
    loads = states @ load.T
    variances = np.einsum("ij,tjk,ik->ti", load, covariances, load)
    stds = np.sqrt(np.clip(variances, 0.0, None))
    return loads, stds
