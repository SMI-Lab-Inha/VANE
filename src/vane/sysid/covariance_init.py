# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Kalman covariance initialization, informed by mode-identification uncertainty.

A Kalman filter needs a process-noise covariance ``Q``, a measurement-noise
covariance ``R``, and an initial state covariance ``P0``. When the modes were
classified with uncertainty quantification, that uncertainty is propagated into
``P0``: a mode identified with low confidence is given a larger initial variance,
so the filter trusts well-identified modes more than ambiguous ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "CovarianceSet",
    "covariances_from_confidence",
    "initialize_covariances",
]


@dataclass
class CovarianceSet:
    """Kalman-filter covariance matrices.

    Parameters
    ----------
    q : npt.NDArray[np.float64]
        Process-noise covariance, shape ``(n_states, n_states)``.
    r : npt.NDArray[np.float64]
        Measurement-noise covariance, shape ``(n_outputs, n_outputs)``.
    p0 : npt.NDArray[np.float64]
        Initial state covariance, shape ``(n_states, n_states)``.
    """

    q: npt.NDArray[np.float64]
    r: npt.NDArray[np.float64]
    p0: npt.NDArray[np.float64]


def initialize_covariances(
    n_states: int,
    n_outputs: int,
    *,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-2,
    initial_variance: float = 1.0,
) -> CovarianceSet:
    """Build uniform diagonal covariance matrices.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_outputs : int
        Number of measured outputs.
    process_noise, measurement_noise, initial_variance : float, optional
        Diagonal values for ``Q``, ``R``, and ``P0`` respectively; all must be
        non-negative.

    Returns
    -------
    CovarianceSet
        The diagonal covariance matrices.

    Raises
    ------
    ValueError
        If a dimension is negative, or a noise value or ``initial_variance`` is
        negative or non-finite.
    """
    _validate_nonnegative(n_states, n_outputs, process_noise, measurement_noise)
    if not np.isfinite(initial_variance) or initial_variance < 0.0:
        msg = "initial_variance must be finite and non-negative"
        raise ValueError(msg)
    return CovarianceSet(
        q=process_noise * np.eye(n_states, dtype=np.float64),
        r=measurement_noise * np.eye(n_outputs, dtype=np.float64),
        p0=initial_variance * np.eye(n_states, dtype=np.float64),
    )


def covariances_from_confidence(
    confidences: Sequence[float],
    n_outputs: int,
    *,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-2,
    base_variance: float = 1.0,
    uncertainty_scale: float = 10.0,
) -> CovarianceSet:
    """Build covariances whose ``P0`` grows with mode-identification uncertainty.

    Each mode maps to two modal states (real and imaginary parts); the initial
    variance of those states is ``base_variance * (1 + uncertainty_scale * (1 -
    confidence))``, so a low-confidence mode starts with a larger covariance.

    Parameters
    ----------
    confidences : Sequence[float]
        Per-mode identification confidence in ``[0, 1]``; values are clipped to
        that range.
    n_outputs : int
        Number of measured outputs.
    process_noise, measurement_noise : float, optional
        Diagonal values for ``Q`` and ``R``; must be non-negative.
    base_variance : float, optional
        Initial variance of a fully confident mode; must be non-negative.
    uncertainty_scale : float, optional
        How strongly uncertainty inflates the initial variance; must be
        non-negative.

    Returns
    -------
    CovarianceSet
        The covariance matrices, with ``2 * len(confidences)`` states.

    Raises
    ------
    ValueError
        If a noise value, ``base_variance``, ``uncertainty_scale``, or
        ``n_outputs`` is negative or non-finite, or a confidence is non-finite.
    """
    confidence_array = np.asarray(confidences, dtype=np.float64)
    if not np.all(np.isfinite(confidence_array)):
        # np.clip does not sanitise NaN (it propagates), so a NaN confidence would
        # otherwise reach P0 as a NaN variance; reject it before clipping.
        msg = "confidences must be finite"
        raise ValueError(msg)
    clipped = np.clip(confidence_array, 0.0, 1.0)
    n_states = 2 * clipped.shape[0]
    _validate_nonnegative(n_states, n_outputs, process_noise, measurement_noise)
    if (
        not np.isfinite(base_variance)
        or base_variance < 0.0
        or not np.isfinite(uncertainty_scale)
        or uncertainty_scale < 0.0
    ):
        msg = "base_variance and uncertainty_scale must be finite and non-negative"
        raise ValueError(msg)

    per_mode = base_variance * (1.0 + uncertainty_scale * (1.0 - clipped))
    p0_diagonal = np.repeat(per_mode, 2)
    return CovarianceSet(
        q=process_noise * np.eye(n_states, dtype=np.float64),
        r=measurement_noise * np.eye(n_outputs, dtype=np.float64),
        p0=np.diag(p0_diagonal),
    )


def _validate_nonnegative(
    n_states: int, n_outputs: int, process_noise: float, measurement_noise: float
) -> None:
    """Validate dimension and noise non-negativity."""
    if n_states < 0 or n_outputs < 0:
        msg = "State and output counts must be non-negative"
        raise ValueError(msg)
    if (
        not np.isfinite(process_noise)
        or not np.isfinite(measurement_noise)
        or process_noise < 0.0
        or measurement_noise < 0.0
    ):
        msg = "Noise values must be finite and non-negative"
        raise ValueError(msg)
