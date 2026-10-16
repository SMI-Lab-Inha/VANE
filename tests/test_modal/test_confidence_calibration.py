# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for :mod:`vane.modal.confidence_calibration`.

The ground truth is analytic: when the correctness label is drawn as
``correct ~ Bernoulli(g(confidence))`` for a known link ``g``, a large sample must
recover ``g`` as the recalibration map, drive the expected calibration error toward zero
when ``g`` is the identity, and never increase the Brier score after recalibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from vane.modal.confidence_calibration import (
    ConfidenceCalibration,
    _pool_adjacent_violators,
    calibrate_confidence,
    confidence_threshold,
    recalibrate,
)


class TestPoolAdjacentViolators:
    """The isotonic (pool-adjacent-violators) fit."""

    def test_already_monotone_is_unchanged(self) -> None:
        """A non-decreasing input is returned unchanged."""
        values = np.array([0.0, 0.2, 0.2, 0.7, 1.0])
        np.testing.assert_allclose(_pool_adjacent_violators(values), values)

    def test_single_violation_is_pooled(self) -> None:
        """A lone violator is pooled with its neighbour to their mean."""
        # [0, 1, 0, 1, 1] -> pool the middle {1, 0} to 0.5.
        fitted = _pool_adjacent_violators(np.array([0.0, 1.0, 0.0, 1.0, 1.0]))
        np.testing.assert_allclose(fitted, [0.0, 0.5, 0.5, 1.0, 1.0])

    def test_strictly_decreasing_collapses_to_mean(self) -> None:
        """A strictly decreasing input collapses to its overall mean."""
        fitted = _pool_adjacent_violators(np.array([1.0, 0.5, 0.0]))
        np.testing.assert_allclose(fitted, [0.5, 0.5, 0.5])

    def test_output_is_non_decreasing(self) -> None:
        """The fit is monotone non-decreasing for arbitrary input."""
        rng = np.random.default_rng(0)
        fitted = _pool_adjacent_violators(rng.random(200))
        assert np.all(np.diff(fitted) >= -1e-12)

    def test_weights_set_the_pooled_mean(self) -> None:
        """A violation pools to the weighted (not plain) mean of the block."""
        # values [1, 0] with weights [2, 1] -> pooled to (1*2 + 0*1)/3.
        fitted = _pool_adjacent_violators(np.array([1.0, 0.0]), np.array([2.0, 1.0]))
        np.testing.assert_allclose(fitted, [2.0 / 3.0, 2.0 / 3.0])


class TestCalibrateConfidence:
    """Reliability diagram, calibration errors, and the recalibration map."""

    def test_perfectly_calibrated_has_near_zero_ece(self) -> None:
        """A calibrated score has near-zero ECE and the analytic Brier 1/6."""
        rng = np.random.default_rng(1)
        conf = rng.uniform(0.0, 1.0, 40_000)
        correct = rng.uniform(0.0, 1.0, 40_000) < conf
        cal = calibrate_confidence(conf, correct)
        assert cal.expected_calibration_error < 0.02
        # Brier of a calibrated predictor: E[c - c^2] = 1/6.
        assert cal.brier_score == pytest.approx(1.0 / 6.0, abs=0.01)

    def test_recalibration_map_recovers_identity_when_calibrated(self) -> None:
        """For calibrated data the recalibration map is the identity."""
        rng = np.random.default_rng(2)
        conf = rng.uniform(0.0, 1.0, 40_000)
        correct = rng.uniform(0.0, 1.0, 40_000) < conf
        cal = calibrate_confidence(conf, correct)
        grid = np.linspace(0.1, 0.9, 9)
        np.testing.assert_allclose(recalibrate(cal, grid), grid, atol=0.05)

    def test_overconfident_is_detected_and_recalibrated(self) -> None:
        """An over-confident score shows a calibration gap and recovers g(c)=c^2."""
        rng = np.random.default_rng(3)
        conf = rng.uniform(0.0, 1.0, 60_000)
        # Truth is c^2: the score is systematically over-confident.
        correct = rng.uniform(0.0, 1.0, 60_000) < conf**2
        cal = calibrate_confidence(conf, correct)
        # Over-confidence: accuracy below confidence, so a real calibration gap.
        assert cal.expected_calibration_error > 0.05
        grid = np.linspace(0.1, 0.9, 9)
        np.testing.assert_allclose(recalibrate(cal, grid), grid**2, atol=0.06)

    def test_recalibration_never_worsens_brier(self) -> None:
        """Recalibration does not increase the Brier score."""
        rng = np.random.default_rng(4)
        conf = rng.uniform(0.0, 1.0, 40_000)
        correct = rng.uniform(0.0, 1.0, 40_000) < conf**2
        cal = calibrate_confidence(conf, correct)
        recal_brier = float(np.mean((recalibrate(cal, conf) - correct) ** 2))
        assert recal_brier <= cal.brier_score + 1e-9

    def test_perfect_separator_is_calibrated(self) -> None:
        """Confidence 1 exactly when correct (0 otherwise) is perfectly calibrated."""
        conf = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
        correct = np.array([1, 1, 0, 0, 1])
        cal = calibrate_confidence(conf, correct)
        assert cal.expected_calibration_error == pytest.approx(0.0)
        assert cal.max_calibration_error == pytest.approx(0.0)
        assert cal.brier_score == pytest.approx(0.0)

    def test_bin_structure_and_empty_bins(self) -> None:
        """Bin edges/counts are correct and empty bins are nan with count 0."""
        conf = np.array([0.05, 0.07, 0.95, 0.97])
        correct = np.array([0, 0, 1, 1])
        cal = calibrate_confidence(conf, correct, n_bins=10)
        assert cal.bin_edges.shape == (11,)
        assert cal.bin_count.sum() == 4
        assert cal.bin_count[0] == 2
        assert cal.bin_count[-1] == 2
        # Middle bins are empty: count 0 and nan diagram entries.
        assert cal.bin_count[5] == 0
        assert np.isnan(cal.bin_confidence[5])
        assert np.isnan(cal.bin_accuracy[5])
        assert cal.n_samples == 4

    def test_map_confidence_is_strictly_increasing(self) -> None:
        """The map confidence axis is strictly increasing, accuracy non-decreasing."""
        rng = np.random.default_rng(5)
        conf = rng.uniform(0.0, 1.0, 500)
        correct = rng.uniform(0.0, 1.0, 500) < conf
        cal = calibrate_confidence(conf, correct)
        assert np.all(np.diff(cal.map_confidence) > 0.0)
        assert np.all(np.diff(cal.map_accuracy) >= -1e-12)

    def test_duplicate_confidences_collapse_to_one_map_point(self) -> None:
        """Repeated confidence values collapse to a single map point."""
        conf = np.array([0.3, 0.3, 0.3, 0.8, 0.8])
        correct = np.array([0, 1, 0, 1, 1])
        cal = calibrate_confidence(conf, correct)
        # Two distinct confidence levels -> two map points.
        np.testing.assert_allclose(cal.map_confidence, [0.3, 0.8])

    def test_tied_confidences_pool_to_weighted_value(self) -> None:
        """Tied confidences are pooled before the fit, independent of input order."""
        # conf 0 has labels {0, 1} (mean 0.5, weight 2); conf 1 has {0} (mean 0,
        # weight 1). The proper isotonic fit pools both to (0.5*2 + 0*1)/3 = 1/3.
        conf = np.array([0.0, 0.0, 1.0])
        cal_a = calibrate_confidence(conf, np.array([0, 1, 0]))
        cal_b = calibrate_confidence(conf, np.array([1, 0, 0]))  # reordered ties
        np.testing.assert_allclose(cal_a.map_accuracy, [1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_allclose(cal_a.map_accuracy, cal_b.map_accuracy)


class TestRecalibrate:
    """Applying the learned map to new confidences."""

    def test_clamps_outside_the_fitted_range(self) -> None:
        """Confidences outside the fitted span clamp to the map's end values."""
        conf = np.array([0.2, 0.4, 0.6, 0.8])
        correct = np.array([0, 0, 1, 1])
        cal = calibrate_confidence(conf, correct)
        assert recalibrate(cal, np.array([0.0]))[0] == pytest.approx(
            cal.map_accuracy[0]
        )
        assert recalibrate(cal, np.array([1.0]))[0] == pytest.approx(
            cal.map_accuracy[-1]
        )

    def test_preserves_shape(self) -> None:
        """The calibrated output keeps the input shape."""
        cal = calibrate_confidence(np.array([0.1, 0.5, 0.9]), np.array([0, 1, 1]))
        out = recalibrate(cal, np.array([[0.2, 0.4], [0.6, 0.8]]))
        assert out.shape == (2, 2)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
    def test_rejects_invalid_confidence(self, bad: float) -> None:
        """Non-finite or out-of-range confidence is rejected."""
        cal = calibrate_confidence(np.array([0.1, 0.9]), np.array([0, 1]))
        with pytest.raises(ValueError, match="confidence must"):
            recalibrate(cal, np.array([bad]))


class TestConfidenceThreshold:
    """Reading the smallest confidence meeting a target calibrated accuracy."""

    def test_returns_smallest_confidence_meeting_target(self) -> None:
        """For calibrated data the 0.8 threshold sits near confidence 0.8."""
        rng = np.random.default_rng(6)
        conf = rng.uniform(0.0, 1.0, 40_000)
        correct = rng.uniform(0.0, 1.0, 40_000) < conf
        cal = calibrate_confidence(conf, correct)
        assert confidence_threshold(cal, 0.8) == pytest.approx(0.8, abs=0.05)

    def test_interpolates_between_map_points(self) -> None:
        """A target between two map points returns the interpolated crossing."""
        # Sparse map: confidences [0.2, 0.8] -> accuracies [0, 1].
        cal = calibrate_confidence(np.array([0.2, 0.8]), np.array([0, 1]))
        threshold = confidence_threshold(cal, 0.5)
        # Linear crossing of 0.5 is at confidence 0.5, consistent with recalibrate.
        assert threshold == pytest.approx(0.5)
        assert float(recalibrate(cal, np.array([threshold]))[0]) == pytest.approx(0.5)

    def test_low_target_returns_lowest_confidence(self) -> None:
        """A target at/below the lowest mapped accuracy returns the lowest conf."""
        # Both confidences map to accuracy 1.0, so any target <= 1 is met at the bottom.
        cal = calibrate_confidence(np.array([0.3, 0.8]), np.array([1, 1]))
        assert confidence_threshold(cal, 0.5) == pytest.approx(0.3)

    def test_is_monotone_in_target(self) -> None:
        """A higher target accuracy requires at least as high a confidence."""
        rng = np.random.default_rng(7)
        conf = rng.uniform(0.0, 1.0, 20_000)
        correct = rng.uniform(0.0, 1.0, 20_000) < conf
        cal = calibrate_confidence(conf, correct)
        assert confidence_threshold(cal, 0.6) <= confidence_threshold(cal, 0.9)

    def test_unreachable_target_raises(self) -> None:
        """A target above the map's maximum calibrated accuracy raises."""
        # Anti-correlated labels: the isotonic fit pools to a flat 0.5, so calibrated
        # accuracy never exceeds 0.5 and a 0.99 target is unreachable.
        conf = np.array([0.2, 0.4, 0.6, 0.8])
        correct = np.array([1, 1, 0, 0])
        cal = calibrate_confidence(conf, correct)
        assert cal.map_accuracy.max() == pytest.approx(0.5)
        with pytest.raises(ValueError, match="never reaches"):
            confidence_threshold(cal, 0.99)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_rejects_invalid_target(self, bad: float) -> None:
        """A target accuracy outside (0, 1] is rejected."""
        cal = calibrate_confidence(np.array([0.1, 0.9]), np.array([0, 1]))
        with pytest.raises(ValueError, match="target_accuracy must be in"):
            confidence_threshold(cal, bad)


class TestInputValidation:
    """Boundary validation of :func:`calibrate_confidence`."""

    def test_length_mismatch(self) -> None:
        """Mismatched confidence/correct lengths are rejected."""
        with pytest.raises(ValueError, match="same length"):
            calibrate_confidence(np.array([0.1, 0.2]), np.array([1]))

    def test_non_1d(self) -> None:
        """Non-1-D inputs are rejected."""
        with pytest.raises(ValueError, match="1-D"):
            calibrate_confidence(np.array([[0.1, 0.2]]), np.array([[1, 0]]))

    def test_empty(self) -> None:
        """Empty inputs are rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            calibrate_confidence(np.array([]), np.array([]))

    def test_non_finite_confidence(self) -> None:
        """A non-finite confidence is rejected."""
        with pytest.raises(ValueError, match="finite"):
            calibrate_confidence(np.array([0.1, np.nan]), np.array([0, 1]))

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_confidence_out_of_range(self, bad: float) -> None:
        """A confidence outside [0, 1] is rejected."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            calibrate_confidence(np.array([0.5, bad]), np.array([0, 1]))

    def test_correct_not_binary(self) -> None:
        """A non-boolean correctness label is rejected."""
        with pytest.raises(ValueError, match="boolean or 0/1"):
            calibrate_confidence(np.array([0.1, 0.9]), np.array([0.0, 0.5]))

    @pytest.mark.parametrize("bad", [0, -1, 2.5])
    def test_bad_n_bins(self, bad: float) -> None:
        """A non-positive or non-integer n_bins is rejected."""
        with pytest.raises(ValueError, match="n_bins must be a positive integer"):
            # Deliberately passing a non-positive / non-integer n_bins.
            calibrate_confidence(
                np.array([0.1, 0.9]),
                np.array([0, 1]),
                n_bins=bad,  # type: ignore[arg-type]
            )


def test_dataclass_fields_are_arrays() -> None:
    """The result is a ConfidenceCalibration with numpy-array fields."""
    cal = calibrate_confidence(np.array([0.1, 0.5, 0.9]), np.array([0, 1, 1]))
    assert isinstance(cal, ConfidenceCalibration)
    assert isinstance(cal.bin_edges, np.ndarray)
    assert isinstance(cal.map_confidence, np.ndarray)
    assert cal.map_accuracy.dtype == np.float64
