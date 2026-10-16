# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Process- and measurement-noise covariances for the discrete estimator.

A Kalman filter on a discretized model needs a **discrete** process-noise covariance
:math:`\mathbf Q_d` consistent with the continuous model it came from, not a hand-picked
diagonal. This module builds the covariances the reduced estimator consumes:

* :func:`discretize_process_noise` converts a continuous process-noise spectral density
  :math:`\mathbf Q_c` to the exact discrete covariance
  :math:`\mathbf Q_d=\int_0^{\Delta t}e^{\mathbf A s}\mathbf Q_c
  e^{\mathbf A^{\mathsf T} s}\,\mathrm ds` over one step, by the Van Loan
  block-exponential identity — the discretization that matches the zero-order-hold
  ``A_d`` the state space already uses.
* :func:`modal_process_noise` assembles a block-diagonal continuous :math:`\mathbf Q_c`
  for the real modal realization (two states per mode), one intensity per mode, so the
  per-mode process noise can be set from each mode's damping and identification
  uncertainty rather than a single global value.
* :func:`measurement_noise_from_specs` turns per-channel sensor noise standard
  deviations into the diagonal measurement-noise covariance :math:`\mathbf R`.

Together they replace the placeholder diagonal ``Q``/``R`` with covariances traceable to
the continuous model and the sensor specifications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

from vane.sysid._covariance import symmetric_part, validate_covariance

# The Van Loan block exponentiates the (positive-real) modes of -A and A^T, so it is
# accurate only while the largest |Re(eigenvalue)| * sub-step stays modest. We split the
# step into 2**k sub-steps to keep it below this bound, then recombine by doubling.
_DECAY_PER_SUBSTEP = 1.0

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "discretize_process_noise",
    "measurement_noise_from_specs",
    "modal_process_noise",
]


def discretize_process_noise(
    a: npt.NDArray[np.float64],
    spectral_density: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    r"""Discretize a continuous process-noise spectral density over one step.

    Computes the discrete process-noise covariance

    .. math::

        \mathbf Q_d = \int_0^{\Delta t} e^{\mathbf A s}\,\mathbf Q_c\,
        e^{\mathbf A^{\mathsf T} s}\,\mathrm d s

    exactly, by the Van Loan block-exponential identity: for
    :math:`\mathbf F = \begin{bmatrix}-\mathbf A & \mathbf Q_c\\ \mathbf 0 &
    \mathbf A^{\mathsf T}\end{bmatrix}`, with
    :math:`e^{\mathbf F\Delta t}=\begin{bmatrix}\mathbf G_{11}&\mathbf G_{12}\\
    \mathbf 0&\mathbf G_{22}\end{bmatrix}`, the discrete-time transition is
    :math:`\mathbf A_d=\mathbf G_{22}^{\mathsf T}` (equal to the zero-order-hold
    :math:`e^{\mathbf A\Delta t}`) and :math:`\mathbf Q_d=\mathbf A_d\mathbf G_{12}`.
    This is the process-noise companion to the ZOH discretization of ``(A, B)``.

    The block exponentiates :math:`-\mathbf A`, which grows for stable modes; for a
    stiff system that intermediate overflows before the cancellation. To stay finite and
    well conditioned for stiff, near-marginal, and mixed systems alike, the step is
    split into :math:`2^k` equal sub-steps small enough that the block is well behaved,
    and the sub-step covariance is recombined by doubling,
    :math:`\mathbf Q_d(2\tau)=\mathbf Q_d(\tau)+\boldsymbol\Phi(\tau)\mathbf Q_d(\tau)
    \boldsymbol\Phi(\tau)^{\mathsf T}` with :math:`\boldsymbol\Phi=e^{\mathbf A\tau}`.
    No stability is required.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        Continuous-time system matrix, shape ``(n_states, n_states)``.
    spectral_density : npt.NDArray[np.float64]
        Continuous process-noise spectral density ``Q_c``, a symmetric positive
        semidefinite matrix of shape ``(n_states, n_states)``.
    dt : float
        Sample step ``Δt`` (must be positive and finite).

    Returns
    -------
    npt.NDArray[np.float64]
        The discrete process-noise covariance ``Q_d``, a symmetric ``(n_states,
        n_states)`` matrix.

    Raises
    ------
    ValueError
        If ``a`` is not square, ``spectral_density`` does not match ``a``, is not a
        valid (symmetric positive-semidefinite) covariance, or ``dt`` is not positive
        and finite.
    """
    a = np.asarray(a, dtype=np.float64)
    qc = np.asarray(spectral_density, dtype=np.float64)
    n = a.shape[0]
    if a.ndim != 2 or a.shape[1] != n:
        msg = f"a must be a square matrix, got shape {a.shape}"
        raise ValueError(msg)
    if qc.shape != (n, n):
        msg = f"spectral_density must be ({n}, {n}) to match a, got {qc.shape}"
        raise ValueError(msg)
    if not (np.isfinite(dt) and dt > 0.0):
        msg = f"dt must be a positive, finite sample step, got {dt}"
        raise ValueError(msg)
    _validate_covariance(qc, "spectral_density")
    qc = symmetric_part(qc)

    # Split into 2**k sub-steps so the largest |Re(eigenvalue)| * sub-step is bounded,
    # keeping the Van Loan block exponential finite and well conditioned even for stiff
    # systems; recombine by doubling.
    k = 0
    if n > 0:
        decay = float(np.max(np.abs(np.linalg.eigvals(a).real)))
        if decay * dt > _DECAY_PER_SUBSTEP:
            k = int(np.ceil(np.log2(decay * dt / _DECAY_PER_SUBSTEP)))
    sub_dt = dt / 2**k

    block = np.block([[-a, qc], [np.zeros((n, n)), a.T]])
    exp_block = expm(block * sub_dt)
    phi = np.asarray(exp_block[n:, n:].T, dtype=np.float64)  # G22^T = e^{A sub_dt}
    qd = phi @ exp_block[:n, n:]  # Q_d over one sub-step
    for _ in range(k):
        qd = qd + phi @ qd @ phi.T
        phi = phi @ phi
    return np.asarray(0.5 * (qd + qd.T), dtype=np.float64)


def _validate_covariance(matrix: npt.NDArray[np.float64], name: str) -> None:
    """Validate that ``matrix`` is a symmetric positive-semidefinite covariance."""
    validate_covariance(matrix, name, definite=False)


def modal_process_noise(
    intensities: Sequence[float],
) -> npt.NDArray[np.float64]:
    """Build a block-diagonal continuous process-noise density over modal states.

    The real modal realization carries two states per mode; this places one intensity on
    each mode's two states, giving the continuous spectral density ``Q_c`` for
    :func:`discretize_process_noise`. Setting a larger intensity on a less-trusted or
    more lightly damped mode lets the filter admit more drift there.

    Parameters
    ----------
    intensities : Sequence[float]
        Per-mode process-noise intensity (one value per mode), all non-negative.

    Returns
    -------
    npt.NDArray[np.float64]
        The block-diagonal spectral density ``Q_c`` of shape
        ``(2 * n_modes, 2 * n_modes)``, with ``intensities[k]`` on the two states of
        mode ``k``.

    Raises
    ------
    ValueError
        If any intensity is negative or non-finite.
    """
    values = np.asarray(intensities, dtype=np.float64)
    if values.ndim != 1:
        msg = f"intensities must be a 1-D sequence, got shape {values.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        msg = "process-noise intensities must be finite and non-negative"
        raise ValueError(msg)
    return np.diag(np.repeat(values, 2))


def measurement_noise_from_specs(
    noise_std: Sequence[float],
) -> npt.NDArray[np.float64]:
    """Build the measurement-noise covariance from per-channel noise specs.

    Each sensor's noise standard deviation maps to a variance on the diagonal of ``R``,
    so the measurement-noise covariance is traceable to the instrument specifications
    rather than a single global guess.

    Parameters
    ----------
    noise_std : Sequence[float]
        Per-channel noise standard deviation (one per sensor), all non-negative, in the
        sensor's measurement units.

    Returns
    -------
    npt.NDArray[np.float64]
        The diagonal measurement-noise covariance ``R = diag(noise_std**2)`` of shape
        ``(n_sensors, n_sensors)``.

    Raises
    ------
    ValueError
        If any standard deviation is negative or non-finite.
    """
    std = np.asarray(noise_std, dtype=np.float64)
    if std.ndim != 1:
        msg = f"noise_std must be a 1-D sequence, got shape {std.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(std)) or np.any(std < 0.0):
        msg = "noise standard deviations must be finite and non-negative"
        raise ValueError(msg)
    return np.diag(std**2)
