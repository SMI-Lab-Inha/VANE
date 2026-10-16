# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Steady-state estimator assessment — model-conditional reconstruction-error bounds.

Before a single time step runs, this module states *how well* a load can be
reconstructed from a sensor set. Given the exported
discrete model (:math:`\mathbf A_d,\mathbf C`), the discrete process-noise covariance
:math:`\mathbf Q_d` (:func:`~vane.sysid.discretize_process_noise`) and the
measurement-noise covariance :math:`\mathbf R`
(:func:`~vane.sysid.measurement_noise_from_specs`), it solves the discrete algebraic
Riccati equation for the steady-state Kalman filter and returns:

* the steady Kalman gain :math:`\mathbf K`,
* the a-priori (predicted) and a-posteriori (filtered) steady-state estimation-error
  covariances, and
* per-state and per-load error standard deviations, with an optional pass/flag
  assessment "this load is reconstructable to :math:`\pm\sigma` from this sensor set".

The per-load :math:`\sigma=\sqrt{\mathbf L\mathbf P^+\mathbf L^{\mathsf T}}` is the
dynamic, filter-exact counterpart of the static reconstructability bound in
:func:`~vane.sysid.load_reconstructability`: where that projects a load onto the
observable subspace over one window, this accounts for the full closed-loop estimator,
the process and measurement noise, and the disturbance augmentation
(:func:`~vane.sysid.augment_with_disturbance`). It is the model-conditional accuracy
bound that can be checked before committing to a sensor layout.

What the bound is conditional on
--------------------------------
The :math:`\sigma` is the *exact* steady-state Kalman error covariance **of the supplied
model** — not a guarantee about the physical turbine. It holds only to the extent that
the assumptions behind that model hold:

* the linearized :math:`(\mathbf A_d,\mathbf C)` is an accurate model of the plant at
  the operating point (no un-modelled nonlinearity, no model error);
* the supplied :math:`\mathbf Q_d` and :math:`\mathbf R` are the true process- and
  measurement-noise covariances (the bound scales with whatever you pass in);
* the noises are zero-mean, white, Gaussian, and mutually independent, and the system is
  stationary at the operating point;
* the sensors are calibrated and unbiased (no systematic error), and the pair is
  detectable from the chosen sensors.

A reported :math:`\sigma` is therefore a model-conditional bound, not a field-validated
accuracy claim; validate against nonlinear time-domain simulation or measured data
before relying on it operationally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_discrete_are

from vane.sysid._covariance import symmetric_part, validate_covariance

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.sysid.state_space import StateSpace

__all__ = [
    "EstimatorAssessment",
    "LoadAssessment",
    "assess_load",
    "steady_state_estimator",
]


@dataclass
class EstimatorAssessment:
    """The steady-state Kalman filter and its predicted and filtered error covariances.

    Parameters
    ----------
    gain : npt.NDArray[np.float64]
        The steady-state Kalman gain ``K``, shape ``(n_states, n_outputs)``.
    error_covariance : npt.NDArray[np.float64]
        The a-posteriori (filtered) steady-state estimation-error covariance ``P+``,
        shape ``(n_states, n_states)`` — the accuracy after each measurement update.
    predicted_covariance : npt.NDArray[np.float64]
        The a-priori (predicted) steady-state error covariance ``P-`` (the DARE
        solution), shape ``(n_states, n_states)``.
    state_std : npt.NDArray[np.float64]
        Per-state a-posteriori error standard deviation, ``sqrt(diag(P+))``.
    """

    gain: npt.NDArray[np.float64]
    error_covariance: npt.NDArray[np.float64]
    predicted_covariance: npt.NDArray[np.float64]
    state_std: npt.NDArray[np.float64]


@dataclass
class LoadAssessment:
    """The reconstruction-error bound for one target load.

    Parameters
    ----------
    load_name : str
        The load's name (or ``"load[i]"``).
    error_std : float
        The steady-state filtered (a-posteriori) reconstruction-error standard
        deviation, ``sqrt(L P+ L^T)`` — the load is reconstructable to ``±error_std``.
    tolerance : float or None
        The acceptance tolerance the load was checked against (``None`` if only
        reported, not checked).
    within_tolerance : bool
        Whether the load **met an explicit tolerance**: ``True`` only when a
        ``tolerance`` was given *and* ``error_std <= tolerance``. ``False`` when the
        error exceeds the tolerance, and also ``False`` when no tolerance was given
        (the error is reported but **not** accepted). Use ``tolerance is None`` to
        distinguish "reported, unchecked" from "checked and failed".
    """

    load_name: str
    error_std: float
    tolerance: float | None
    within_tolerance: bool


def steady_state_estimator(
    system: StateSpace,
    *,
    process_noise: npt.NDArray[np.float64],
    measurement_noise: npt.NDArray[np.float64],
) -> EstimatorAssessment:
    r"""Solve the steady-state Kalman filter and its predicted/filtered covariances.

    For the discrete model :math:`\mathbf x_{k+1}=\mathbf A_d\mathbf x_k+\mathbf w_k`,
    :math:`\mathbf y_k=\mathbf C\mathbf x_k+\mathbf v_k` with
    :math:`\mathbf w\sim\mathcal N(0,\mathbf Q_d)`,
    :math:`\mathbf v\sim\mathcal N(0,\mathbf R)`, solves the discrete algebraic Riccati
    equation for the predicted error covariance
    :math:`\mathbf P^-=\mathbf A_d\mathbf P^+\mathbf A_d^{\mathsf T}+\mathbf Q_d`,
    then forms the gain
    :math:`\mathbf K=\mathbf P^-\mathbf C^{\mathsf T}
    (\mathbf C\mathbf P^-\mathbf C^{\mathsf T}+\mathbf R)^{-1}` and the filtered
    covariance :math:`\mathbf P^+=(\mathbf I-\mathbf K\mathbf C)\mathbf P^-`.

    Parameters
    ----------
    system : StateSpace
        A **discrete-time** system (``dt`` set) with a ``C`` matrix.
    process_noise : npt.NDArray[np.float64]
        The discrete process-noise covariance ``Q_d``, shape ``(n_states, n_states)``,
        symmetric positive-semidefinite.
    measurement_noise : npt.NDArray[np.float64]
        The measurement-noise covariance ``R``, shape ``(n_outputs, n_outputs)``,
        symmetric positive-definite.

    Returns
    -------
    EstimatorAssessment
        The steady Kalman gain and the steady-state error covariances.

    Raises
    ------
    ValueError
        If the system is continuous, has no ``C`` matrix, the covariances have the wrong
        shape or are not valid (``Q_d`` PSD, ``R`` PD), or the filter has no
        steady-state solution (``(A, C)`` not detectable from the sensors, or
        ``(A, Q^1/2)`` not stabilisable -- a marginal mode with no process noise).
    """
    if system.dt is None:
        msg = "steady_state_estimator requires a discrete-time system (dt is set)"
        raise ValueError(msg)
    if system.c is None:
        msg = "system has no C matrix; cannot assess an estimator"
        raise ValueError(msg)
    ad = np.asarray(system.a, dtype=np.float64)
    c = np.asarray(system.c, dtype=np.float64)
    n, p = ad.shape[0], c.shape[0]
    qd = _validate_covariance(process_noise, (n, n), "process_noise", definite=False)
    r = _validate_covariance(
        measurement_noise, (p, p), "measurement_noise", definite=True
    )

    try:
        predicted = solve_discrete_are(ad.T, c.T, qd, r)
    except (np.linalg.LinAlgError, ValueError) as exc:
        # solve_discrete_are raises when no stabilising solution exists; the exception
        # type varies by SciPy version, so normalise both to a clear domain error.
        msg = (
            "the steady-state filter has no stabilising solution: either (A, C) is not "
            "detectable (an unstable or marginal mode is unobservable from these "
            "sensors), or (A, Q^1/2) is not stabilisable (a marginal or unstable mode "
            "receives no process noise, e.g. an unexcited integrator or undamped "
            "mode -- give it a small process-noise variance)"
        )
        raise ValueError(msg) from exc
    predicted = np.asarray(0.5 * (predicted + predicted.T), dtype=np.float64)

    # Innovation covariance S = C P- C^T + R (symmetric positive-definite, since R is).
    innovation = c @ predicted @ c.T + r
    innovation = np.asarray(0.5 * (innovation + innovation.T), dtype=np.float64)
    # Gain K = P- C^T S^-1 via a Cholesky solve rather than an explicit inverse, and the
    # Joseph-form update P+ = (I-KC) P- (I-KC)^T + K R K^T, which stays symmetric
    # positive-semidefinite even for ill-conditioned S (the (I-KC)P- form does not).
    chol = cho_factor(innovation)
    gain = cho_solve(chol, c @ predicted).T
    i_kc = np.eye(n) - gain @ c
    filtered = i_kc @ predicted @ i_kc.T + gain @ r @ gain.T
    filtered = np.asarray(0.5 * (filtered + filtered.T), dtype=np.float64)
    state_std = np.sqrt(np.clip(np.diag(filtered), 0.0, None))

    return EstimatorAssessment(
        gain=gain,
        error_covariance=filtered,
        predicted_covariance=predicted,
        state_std=state_std,
    )


def assess_load(
    assessment: EstimatorAssessment,
    load_map: npt.NDArray[np.float64],
    *,
    tolerance: float | Sequence[float] | None = None,
    names: Sequence[str] | None = None,
) -> list[LoadAssessment]:
    r"""Bound the reconstruction error of each target load from the assessment.

    For each load row :math:`\mathbf L_i` (a linear functional of the state, e.g. a row
    of ``C`` for a named output or the recovery map of
    :func:`~vane.sysid.augment_with_disturbance`), reports the steady-state filtered
    (a-posteriori) error standard deviation
    :math:`\sigma_i=\sqrt{\mathbf L_i\mathbf P^+\mathbf L_i^{\mathsf T}}` and, if a
    tolerance is given, whether the load is reconstructable to within it.

    Parameters
    ----------
    assessment : EstimatorAssessment
        An assessment from :func:`steady_state_estimator`.
    load_map : npt.NDArray[np.float64]
        The load functionals ``L``, shape ``(n_loads, n_states)``; ``z = L x``.
    tolerance : float or Sequence[float] or None, optional
        Acceptance tolerance on the error standard deviation; a scalar applies to every
        load, or one value per load. ``None`` reports the error without checking.
    names : Sequence[str] or None, optional
        Per-load names (length ``n_loads``). ``None`` uses ``"load[i]"``.

    Returns
    -------
    list[LoadAssessment]
        One assessment per load row.

    Raises
    ------
    ValueError
        If ``load_map`` does not match the state dimension or is non-finite, or
        ``tolerance``/``names`` have the wrong length.
    """
    load = np.asarray(load_map, dtype=np.float64)
    n = assessment.error_covariance.shape[0]
    if load.ndim != 2 or load.shape[1] != n:
        msg = f"load_map must be (n_loads, {n}) to match the states, got {load.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(load)):
        msg = "load_map must be finite (no NaN or inf)"
        raise ValueError(msg)
    n_loads = load.shape[0]

    tolerances = _broadcast_tolerance(tolerance, n_loads)
    if names is not None and len(names) != n_loads:
        msg = f"names must have one entry per load ({n_loads}), got {len(names)}"
        raise ValueError(msg)

    variances = np.einsum("ij,jk,ik->i", load, assessment.error_covariance, load)
    stds = np.sqrt(np.clip(variances, 0.0, None))

    results = []
    for i in range(n_loads):
        name = names[i] if names is not None else f"load[{i}]"
        tol = tolerances[i]
        # "Within tolerance" means it met an explicit tolerance. Without a tolerance the
        # error is reported but not accepted; calling an unchecked value
        # "within_tolerance" would overstate what was checked.
        within_tolerance = tol is not None and bool(stds[i] <= tol)
        results.append(
            LoadAssessment(
                load_name=name,
                error_std=float(stds[i]),
                tolerance=tol,
                within_tolerance=within_tolerance,
            )
        )
    return results


def _validate_covariance(
    matrix: npt.NDArray[np.float64],
    shape: tuple[int, int],
    name: str,
    *,
    definite: bool,
) -> npt.NDArray[np.float64]:
    """Validate and return a symmetric PSD (or PD) covariance of the given shape."""
    cov = np.asarray(matrix, dtype=np.float64)
    if cov.shape != shape:
        msg = f"{name} must be {shape}, got {cov.shape}"
        raise ValueError(msg)
    validate_covariance(cov, name, definite=definite)
    return symmetric_part(cov)


def _broadcast_tolerance(
    tolerance: float | Sequence[float] | None, n: int
) -> list[float | None]:
    """Broadcast a scalar/sequence tolerance to a length-``n`` list (or all None)."""
    if tolerance is None:
        return [None] * n
    values = np.atleast_1d(np.asarray(tolerance, dtype=np.float64))
    if values.size == 1:
        values = np.repeat(values, n)
    if values.shape != (n,):
        msg = f"tolerance must be a scalar or have {n} values, got {values.shape}"
        raise ValueError(msg)
    if np.any(values <= 0.0):
        msg = "tolerance values must be positive"
        raise ValueError(msg)
    return [float(v) for v in values]
