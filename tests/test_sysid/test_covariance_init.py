# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.covariance_init — Kalman covariance initialization."""

from __future__ import annotations

import numpy as np
import pytest

from vane.sysid.covariance_init import (
    covariances_from_confidence,
    initialize_covariances,
)


class TestInitializeCovariances:
    """Tests for initialize_covariances."""

    def test_diagonal_shapes_and_values(self) -> None:
        """Q, R, and P0 are diagonal with the requested values."""
        cov = initialize_covariances(
            3, 2, process_noise=1e-3, measurement_noise=1e-1, initial_variance=2.0
        )
        assert cov.q.shape == (3, 3)
        assert cov.r.shape == (2, 2)
        assert cov.p0.shape == (3, 3)
        assert cov.q[0, 0] == pytest.approx(1e-3)
        assert cov.r[1, 1] == pytest.approx(1e-1)
        assert cov.p0[2, 2] == pytest.approx(2.0)

    def test_negative_noise_raises(self) -> None:
        """A negative noise value raises ValueError."""
        with pytest.raises(ValueError, match="Noise"):
            initialize_covariances(2, 1, process_noise=-1.0)

    def test_non_finite_noise_raises(self) -> None:
        """A NaN/infinite noise value or initial_variance is rejected."""
        with pytest.raises(ValueError, match="finite"):
            initialize_covariances(2, 1, measurement_noise=np.nan)
        with pytest.raises(ValueError, match="finite"):
            initialize_covariances(2, 1, initial_variance=np.inf)


class TestCovariancesFromConfidence:
    """Tests for covariances_from_confidence."""

    def test_low_confidence_inflates_initial_variance(self) -> None:
        """A low-confidence mode gets a larger initial variance than a sure one."""
        cov = covariances_from_confidence(
            [1.0, 0.0], 2, base_variance=1.0, uncertainty_scale=10.0
        )
        # Two modes -> four states; first mode confident, second uncertain.
        assert cov.p0.shape == (4, 4)
        assert cov.p0[0, 0] == pytest.approx(1.0)
        assert cov.p0[2, 2] == pytest.approx(11.0)

    def test_confidence_is_clipped(self) -> None:
        """Out-of-range confidences are clipped to [0, 1]."""
        cov = covariances_from_confidence(
            [1.5, -0.5], 1, base_variance=1.0, uncertainty_scale=4.0
        )
        assert cov.p0[0, 0] == pytest.approx(1.0)  # clipped to 1.0
        assert cov.p0[2, 2] == pytest.approx(5.0)  # clipped to 0.0 -> 1 + 4

    def test_outputs_set_r_dimension(self) -> None:
        """R is sized by the number of outputs."""
        cov = covariances_from_confidence([0.5], 3)
        assert cov.r.shape == (3, 3)

    def test_negative_scale_raises(self) -> None:
        """A negative uncertainty scale raises ValueError."""
        with pytest.raises(ValueError, match="uncertainty_scale"):
            covariances_from_confidence([0.5], 1, uncertainty_scale=-1.0)

    def test_non_finite_confidence_raises(self) -> None:
        """A NaN confidence is rejected (np.clip would propagate it into P0)."""
        with pytest.raises(ValueError, match="confidences must be finite"):
            covariances_from_confidence([0.5, np.nan], 1)

    def test_non_finite_scale_raises(self) -> None:
        """A non-finite base_variance / uncertainty_scale is rejected."""
        with pytest.raises(ValueError, match="finite"):
            covariances_from_confidence([0.5], 1, base_variance=np.inf)
