# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Calibration of the per-mode confidence into a reliability metric.

:func:`vane.modal.unified_mode_confidence` returns a per-mode confidence in ``[0, 1]``
built from physically grounded factors (numerical conditioning, degeneracy, azimuth
spread, tracking ambiguity). That confidence is only *useful* if it is **calibrated** —
among the modes assigned confidence ``~c``, about a fraction ``c`` should be correct.
A raw confidence that is systematically over- or under-confident misleads any downstream
decision (which modes to trust, how to weight the Kalman covariance).

This module measures that calibration against a ground-truth correctness label and, when
the raw confidence is miscalibrated, learns a **monotone recalibration map** so the
reported number can be read as a probability of being correct:

- :func:`calibrate_confidence` bins ``(confidence, correct)`` pairs into a reliability
  diagram and reports the expected/maximum calibration error and the Brier score, plus a
  pool-adjacent-violators (isotonic) map ``confidence -> P(correct)``.
- :func:`recalibrate` applies that map to new confidences.
- :func:`confidence_threshold` reads off the smallest confidence at which the calibrated
  probability of being correct reaches a target — the evidence behind a claim like
  "confidence >= 0.85 implies >= 95 % correct".

The ``correct`` label is supplied by the caller and is domain-specific: for mode
identification it is ``predicted_label == true_label`` against a synthetic ground truth;
for cross-operating-point tracking it is whether a track was assigned to the right mode.
The calibration itself is agnostic to how correctness was decided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "ConfidenceCalibration",
    "calibrate_confidence",
    "confidence_threshold",
    "recalibrate",
]


@dataclass
class ConfidenceCalibration:
    """The calibration of a confidence score against a correctness label.

    Parameters
    ----------
    bin_edges : npt.NDArray[np.float64]
        The ``n_bins + 1`` equal-width bin edges spanning ``[0, 1]``.
    bin_confidence : npt.NDArray[np.float64]
        Mean confidence within each bin, shape ``(n_bins,)``; ``nan`` for an empty bin.
    bin_accuracy : npt.NDArray[np.float64]
        Empirical fraction correct within each bin, shape ``(n_bins,)``; ``nan`` for an
        empty bin. The reliability diagram plots ``bin_confidence`` vs ``bin_accuracy``.
    bin_count : npt.NDArray[np.int_]
        Number of samples in each bin, shape ``(n_bins,)``.
    expected_calibration_error : float
        Sample-weighted mean absolute gap between confidence and accuracy across
        non-empty bins (ECE); ``0`` is perfectly calibrated.
    max_calibration_error : float
        Largest absolute confidence-accuracy gap over non-empty bins (MCE).
    brier_score : float
        Mean squared error of confidence against the correctness label; lower is better.
    map_confidence : npt.NDArray[np.float64]
        Strictly increasing confidence values of the isotonic recalibration map.
    map_accuracy : npt.NDArray[np.float64]
        Non-decreasing calibrated ``P(correct)`` at each ``map_confidence`` point.
    n_samples : int
        Number of ``(confidence, correct)`` pairs the calibration was built from.
    """

    bin_edges: npt.NDArray[np.float64]
    bin_confidence: npt.NDArray[np.float64]
    bin_accuracy: npt.NDArray[np.float64]
    bin_count: npt.NDArray[np.int_]
    expected_calibration_error: float
    max_calibration_error: float
    brier_score: float
    map_confidence: npt.NDArray[np.float64]
    map_accuracy: npt.NDArray[np.float64]
    n_samples: int


def _pool_adjacent_violators(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Return the non-decreasing weighted isotonic fit of ``values``.

    Standard pool-adjacent-violators algorithm: it returns the monotone non-decreasing
    sequence closest to ``values`` in weighted least squares, by repeatedly averaging
    adjacent blocks that violate monotonicity. ``values`` must already be ordered by the
    independent variable (here, ascending confidence); ``weights`` (default all ones)
    weights each value, so tied independent-variable groups can be pooled beforehand and
    passed as a single weighted point each.
    """
    n = values.shape[0]
    w = np.ones(n, dtype=np.float64) if weights is None else weights
    # Each block stores (weighted sum, weight, length); blocks tile the sequence.
    block_sum = np.empty(n, dtype=np.float64)
    block_weight = np.empty(n, dtype=np.float64)
    block_mean = np.empty(n, dtype=np.float64)
    block_len = np.empty(n, dtype=np.int_)
    top = 0
    for value, weight in zip(values, w, strict=True):
        block_sum[top] = value * weight
        block_weight[top] = weight
        block_mean[top] = value
        block_len[top] = 1
        top += 1
        # Merge while the new block's mean is below its left neighbour's (a violation).
        while top > 1 and block_mean[top - 1] < block_mean[top - 2]:
            block_sum[top - 2] += block_sum[top - 1]
            block_weight[top - 2] += block_weight[top - 1]
            block_len[top - 2] += block_len[top - 1]
            block_mean[top - 2] = block_sum[top - 2] / block_weight[top - 2]
            top -= 1
    fitted = np.empty(n, dtype=np.float64)
    cursor = 0
    for b in range(top):
        length = int(block_len[b])
        fitted[cursor : cursor + length] = block_mean[b]
        cursor += length
    return fitted


def _validate_pairs(
    confidence: npt.ArrayLike, correct: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate and return the confidence and correctness arrays as 1-D float arrays."""
    conf = np.asarray(confidence, dtype=np.float64)
    truth = np.asarray(correct, dtype=np.float64)
    if conf.ndim != 1 or truth.ndim != 1:
        msg = f"confidence and correct must be 1-D, got {conf.shape} and {truth.shape}"
        raise ValueError(msg)
    if conf.shape != truth.shape:
        msg = (
            f"confidence and correct must have the same length, "
            f"got {conf.shape[0]} and {truth.shape[0]}"
        )
        raise ValueError(msg)
    if conf.size == 0:
        msg = "confidence and correct must be non-empty"
        raise ValueError(msg)
    if not np.all(np.isfinite(conf)):
        msg = "confidence must be finite (no NaN or inf)"
        raise ValueError(msg)
    if np.any(conf < 0.0) or np.any(conf > 1.0):
        msg = "confidence must lie in [0, 1]"
        raise ValueError(msg)
    if not np.all((truth == 0.0) | (truth == 1.0)):
        msg = "correct must be boolean or 0/1"
        raise ValueError(msg)
    return conf, truth


def calibrate_confidence(
    confidence: npt.ArrayLike,
    correct: npt.ArrayLike,
    *,
    n_bins: int = 10,
) -> ConfidenceCalibration:
    """Calibrate a confidence score against a ground-truth correctness label.

    Builds an equal-width reliability diagram over ``[0, 1]``, the expected (ECE) and
    maximum (MCE) calibration errors, the Brier score, and an isotonic (monotone)
    recalibration map ``confidence -> P(correct)`` fitted by pool-adjacent-violators.

    Parameters
    ----------
    confidence : npt.ArrayLike
        Per-sample confidence in ``[0, 1]`` (e.g. from
        :func:`vane.modal.unified_mode_confidence`).
    correct : npt.ArrayLike
        Per-sample correctness label, boolean or ``0``/``1``, aligned with
        ``confidence``.
    n_bins : int, optional
        Number of equal-width reliability bins over ``[0, 1]``. Default ``10``.

    Returns
    -------
    ConfidenceCalibration
        The reliability diagram, calibration-error metrics, Brier score, and isotonic
        recalibration map.

    Raises
    ------
    ValueError
        If the inputs are not equal-length non-empty 1-D arrays, ``confidence`` is
        non-finite or outside ``[0, 1]``, ``correct`` is not boolean/``0``/``1``, or
        ``n_bins`` is not a positive integer.
    """
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        msg = f"n_bins must be a positive integer, got {n_bins!r}"
        raise ValueError(msg)
    conf, truth = _validate_pairs(confidence, correct)
    n_samples = conf.size

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float64)
    # Bin index in [0, n_bins - 1]; the right edge (confidence == 1) joins the last bin.
    index = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, int(n_bins) - 1)

    bin_confidence = np.full(int(n_bins), np.nan, dtype=np.float64)
    bin_accuracy = np.full(int(n_bins), np.nan, dtype=np.float64)
    bin_count = np.zeros(int(n_bins), dtype=np.int_)
    ece = 0.0
    mce = 0.0
    for b in range(int(n_bins)):
        in_bin = index == b
        count = int(np.count_nonzero(in_bin))
        bin_count[b] = count
        if count == 0:
            continue
        mean_conf = float(conf[in_bin].mean())
        mean_acc = float(truth[in_bin].mean())
        bin_confidence[b] = mean_conf
        bin_accuracy[b] = mean_acc
        gap = abs(mean_acc - mean_conf)
        ece += (count / n_samples) * gap
        mce = max(mce, gap)

    brier = float(np.mean((conf - truth) ** 2))

    # Isotonic recalibration map. Tied confidences must share one calibrated value, so
    # pool them into weighted groups (mean label, count) *before* the isotonic fit — a
    # weighted pool-adjacent-violators over the unique confidences is then the proper
    # least-squares isotonic regression and is independent of the input order (fitting
    # the raw per-sample labels first lets ties' arbitrary order fabricate structure).
    unique_conf, inverse, counts = np.unique(
        conf, return_inverse=True, return_counts=True
    )
    group_sum = np.zeros(unique_conf.shape[0], dtype=np.float64)
    np.add.at(group_sum, inverse, truth)
    group_mean = group_sum / counts
    map_accuracy = _pool_adjacent_violators(group_mean, counts.astype(np.float64))

    return ConfidenceCalibration(
        bin_edges=edges,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_count=bin_count,
        expected_calibration_error=float(ece),
        max_calibration_error=float(mce),
        brier_score=brier,
        map_confidence=unique_conf.astype(np.float64),
        map_accuracy=map_accuracy.astype(np.float64),
        n_samples=int(n_samples),
    )


def recalibrate(
    calibration: ConfidenceCalibration, confidence: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Map raw confidence to a calibrated probability of being correct.

    Linearly interpolates ``confidence`` onto the isotonic recalibration map; values
    below or above the fitted range are clamped to the map's end values.

    Parameters
    ----------
    calibration : ConfidenceCalibration
        A calibration from :func:`calibrate_confidence`.
    confidence : npt.ArrayLike
        Raw confidence value(s) in ``[0, 1]``.

    Returns
    -------
    npt.NDArray[np.float64]
        The calibrated probability of being correct, same shape as ``confidence``.

    Raises
    ------
    ValueError
        If ``confidence`` is non-finite or outside ``[0, 1]``.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    if not np.all(np.isfinite(conf)):
        msg = "confidence must be finite (no NaN or inf)"
        raise ValueError(msg)
    if np.any(conf < 0.0) or np.any(conf > 1.0):
        msg = "confidence must lie in [0, 1]"
        raise ValueError(msg)
    # np.interp clamps to the end values outside the map's confidence span.
    calibrated = np.interp(conf, calibration.map_confidence, calibration.map_accuracy)
    return np.asarray(calibrated, dtype=np.float64)


def confidence_threshold(
    calibration: ConfidenceCalibration, target_accuracy: float
) -> float:
    """Return the smallest confidence whose calibrated ``P(correct)`` reaches a target.

    Inverts the same piecewise-linear curve :func:`recalibrate` evaluates, so the
    returned confidence ``c`` is the lowest value with ``recalibrate(c) >=
    target_accuracy``: when the target falls between two map points the crossing
    confidence is linearly interpolated, rather than rounded up to the next sampled
    point. The answer is reported on the observed confidence span; a target at or below
    the lowest mapped accuracy returns the lowest mapped confidence.

    Parameters
    ----------
    calibration : ConfidenceCalibration
        A calibration from :func:`calibrate_confidence`.
    target_accuracy : float
        The desired probability of being correct, in ``(0, 1]``.

    Returns
    -------
    float
        The smallest confidence whose calibrated probability meets the target.

    Raises
    ------
    ValueError
        If ``target_accuracy`` is not in ``(0, 1]``, or the calibrated probability never
        reaches it (the confidence is not discriminative enough for this target).
    """
    if not 0.0 < target_accuracy <= 1.0:
        msg = f"target_accuracy must be in (0, 1], got {target_accuracy}"
        raise ValueError(msg)
    accuracy = calibration.map_accuracy
    confidence = calibration.map_confidence
    upper = int(np.searchsorted(accuracy, target_accuracy, side="left"))
    if upper >= accuracy.shape[0]:
        msg = (
            f"calibrated accuracy never reaches {target_accuracy} "
            f"(max {float(accuracy.max()):.3g}); "
            f"the confidence is not discriminative enough for this target"
        )
        raise ValueError(msg)
    if upper == 0:
        # The lowest mapped confidence already meets the target.
        return float(confidence[0])
    # Linearly interpolate the crossing within (upper - 1, upper); accuracy is monotone
    # non-decreasing, and accuracy[upper] > accuracy[upper - 1] here (a flat block would
    # have placed the searchsorted result at its left edge), so the gap is non-zero.
    lower = upper - 1
    span = accuracy[upper] - accuracy[lower]
    fraction = (target_accuracy - accuracy[lower]) / span
    return float(confidence[lower] + fraction * (confidence[upper] - confidence[lower]))
