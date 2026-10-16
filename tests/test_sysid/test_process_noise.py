# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.process_noise — discrete process/measurement covariances."""

from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import block_diag, expm

from vane.sysid import (
    discretize_process_noise,
    measurement_noise_from_specs,
    modal_process_noise,
)


def _block(freq: float, zeta: float) -> np.ndarray:
    """Return a stable continuous 2x2 modal block."""
    wn = 2.0 * np.pi * freq
    sigma = -zeta * wn
    wd = wn * np.sqrt(1.0 - zeta**2)
    return np.array([[sigma, -wd], [wd, sigma]])


class TestDiscretizeProcessNoise:
    """Tests for discretize_process_noise."""

    def test_matches_the_numerical_integral(self) -> None:
        """Qd equals the direct integral of e^{A s} Qc e^{A^T s} over the step."""
        a = np.asarray(block_diag(_block(0.3, 0.05), _block(1.5, 0.05)))
        rng = np.random.default_rng(1)
        root = rng.standard_normal((4, 4))
        qc = root @ root.T  # symmetric PSD
        dt = 0.05
        qd = discretize_process_noise(a, qc, dt)
        integral, _ = quad_vec(lambda s: expm(a * s) @ qc @ expm(a.T * s), 0.0, dt)
        np.testing.assert_allclose(qd, integral, atol=1e-10)
        np.testing.assert_allclose(qd, qd.T, atol=1e-12)  # symmetric

    def test_matches_the_scalar_closed_form(self) -> None:
        """For a 1-state system Qd = q (1 - e^{-2 a dt}) / (2 a)."""
        a = np.array([[-2.0]])
        qc = np.array([[3.0]])
        dt = 0.1
        qd = discretize_process_noise(a, qc, dt)
        expected = 3.0 * (1.0 - np.exp(-2.0 * 2.0 * dt)) / (2.0 * 2.0)
        np.testing.assert_allclose(qd, [[expected]], atol=1e-12)

    def test_is_positive_semidefinite(self) -> None:
        """The discrete covariance is symmetric PSD."""
        a = np.asarray(_block(0.5, 0.1))
        qc = np.diag([1.0, 4.0])
        qd = discretize_process_noise(a, qc, 0.02)
        assert np.all(np.linalg.eigvalsh(qd) >= -1e-12)

    def test_defined_for_unstable_system(self) -> None:
        """The step integral needs no stability; an unstable A is fine."""
        a = np.array([[0.5, -6.0], [6.0, 0.5]])  # unstable
        qd = discretize_process_noise(a, np.eye(2), 0.05)
        integral, _ = quad_vec(
            lambda s: expm(a * s) @ np.eye(2) @ expm(a.T * s), 0.0, 0.05
        )
        np.testing.assert_allclose(qd, integral, atol=1e-10)

    def test_stiff_stable_mode_stays_finite(self) -> None:
        """A stiff stable mode gives the finite covariance, not an overflow.

        For ``a = -1000`` the Van Loan block exponentiates ``-a`` and overflows to
        ``inf``/``nan`` before the cancellation; the stable Lyapunov path returns the
        true ``Qd = q / (2|a|) = 5e-4``.
        """
        qd = discretize_process_noise(np.array([[-1000.0]]), np.array([[1.0]]), 1.0)
        assert np.isfinite(qd).all()
        np.testing.assert_allclose(qd, [[1.0 / 2000.0]], rtol=1e-9)

    def test_near_marginal_mode_keeps_its_noise(self) -> None:
        """A near-marginal mode keeps the finite-step noise (no cancellation to zero).

        For ``a ≈ 0`` the one-step covariance is ``≈ q * dt``; a stationary-difference
        shortcut would cancel to zero, so the sub-stepped block must preserve it.
        """
        qd = discretize_process_noise(np.array([[-1e-16]]), np.array([[1.0]]), 0.1)
        np.testing.assert_allclose(qd, [[0.1]], rtol=1e-9)

    def test_mixed_stiff_and_marginal_modes(self) -> None:
        """A system with both a stiff and a near-marginal mode resolves both."""
        a = np.diag([-1000.0, -1e-16])
        qd = discretize_process_noise(a, np.eye(2), 1.0)
        assert np.isfinite(qd).all()
        np.testing.assert_allclose(np.diag(qd), [1.0 / 2000.0, 1.0], rtol=1e-9)

    def test_rejects_nonsquare_a(self) -> None:
        """A non-square system matrix is rejected."""
        with pytest.raises(ValueError, match="square"):
            discretize_process_noise(np.ones((2, 3)), np.eye(2), 0.1)

    def test_rejects_mismatched_spectral_density(self) -> None:
        """A spectral density that does not match A is rejected."""
        with pytest.raises(ValueError, match="spectral_density"):
            discretize_process_noise(np.eye(2), np.eye(3), 0.1)

    @pytest.mark.parametrize("bad_dt", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_invalid_dt(self, bad_dt: float) -> None:
        """A non-positive, infinite, or NaN sample step is rejected."""
        with pytest.raises(ValueError, match="dt must be a positive, finite"):
            discretize_process_noise(np.eye(2), np.eye(2), bad_dt)

    def test_rejects_non_psd_spectral_density(self) -> None:
        """A non-PSD spectral density is not a covariance and is rejected."""
        with pytest.raises(ValueError, match="positive semidefinite"):
            discretize_process_noise(np.array([[0.0]]), np.array([[-1.0]]), 1.0)

    def test_rejects_nonsymmetric_spectral_density(self) -> None:
        """A nonsymmetric spectral density is rejected."""
        with pytest.raises(ValueError, match="symmetric"):
            discretize_process_noise(np.eye(2), np.array([[1.0, 2.0], [0.0, 1.0]]), 0.1)


class TestModalProcessNoise:
    """Tests for modal_process_noise."""

    def test_places_each_intensity_on_two_states(self) -> None:
        """Each mode intensity lands on its two modal states."""
        qc = modal_process_noise([2.0, 5.0])
        assert qc.shape == (4, 4)
        np.testing.assert_allclose(np.diag(qc), [2.0, 2.0, 5.0, 5.0])
        np.testing.assert_allclose(qc - np.diag(np.diag(qc)), 0.0)

    def test_rejects_negative_intensity(self) -> None:
        """A negative process-noise intensity is rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            modal_process_noise([1.0, -0.5])

    def test_rejects_non_1d(self) -> None:
        """A non-1-D intensity array is rejected."""
        nested: Any = [[1.0, 2.0]]
        with pytest.raises(ValueError, match="1-D"):
            modal_process_noise(nested)

    def test_rejects_non_finite_intensity(self) -> None:
        """A NaN intensity is rejected, not propagated into Q."""
        with pytest.raises(ValueError, match="finite"):
            modal_process_noise([1.0, np.nan])


class TestMeasurementNoiseFromSpecs:
    """Tests for measurement_noise_from_specs."""

    def test_squares_the_standard_deviations(self) -> None:
        """R is the diagonal of the per-channel variances."""
        r = measurement_noise_from_specs([0.1, 2.0])
        np.testing.assert_allclose(r, np.diag([0.01, 4.0]))

    def test_rejects_negative_std(self) -> None:
        """A negative standard deviation is rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            measurement_noise_from_specs([0.1, -1.0])

    def test_rejects_non_finite_std(self) -> None:
        """A NaN or infinite standard deviation is rejected, not propagated into R."""
        with pytest.raises(ValueError, match="finite"):
            measurement_noise_from_specs([0.1, np.nan])
        with pytest.raises(ValueError, match="finite"):
            measurement_noise_from_specs([0.1, np.inf])

    def test_rejects_non_1d(self) -> None:
        """A non-1-D spec array is rejected."""
        nested: Any = [[0.1, 0.2]]
        with pytest.raises(ValueError, match="1-D"):
            measurement_noise_from_specs(nested)
