# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.benchmark — ground-truth accuracy and robustness."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from vane.benchmark import (
    robustness_curve,
    score_recovery,
    synthetic_system,
)
from vane.modal.eigensolver import compute_modes

_FREQS = [1.0, 2.0, 3.5]
_ZETAS = [0.01, 0.02, 0.05]


def _orthogonal_shapes() -> np.ndarray:
    """Return a fixed 3x3 orthogonal modal matrix."""
    basis = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0]])
    q, _ = np.linalg.qr(basis)
    return q


class TestSyntheticSystem:
    """Tests for synthetic_system construction and validation."""

    def test_state_matrix_realizes_prescribed_modes(self) -> None:
        """The built state matrix has exactly the prescribed frequencies/damping."""
        truth = synthetic_system(_FREQS, _ZETAS)
        solution = compute_modes(truth.a, ndof2=3, ndof1=0)
        recovered = np.sort(solution.natural_frequencies_hz)
        np.testing.assert_allclose(recovered, sorted(_FREQS), rtol=1e-9)

    def test_mismatched_lengths_raise(self) -> None:
        """Unequal frequency/damping lengths raise."""
        with pytest.raises(ValueError, match="equal length"):
            synthetic_system([1.0, 2.0], [0.01])

    def test_nonpositive_frequency_raises(self) -> None:
        """A non-positive frequency raises."""
        with pytest.raises(ValueError, match="strictly positive"):
            synthetic_system([0.0], [0.01])

    def test_damping_out_of_range_raises(self) -> None:
        """Damping outside [0, 1) raises."""
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            synthetic_system([1.0], [1.5])

    def test_non_finite_parameters_raise(self) -> None:
        """NaN or infinite frequency/damping is rejected."""
        with pytest.raises(ValueError, match="finite"):
            synthetic_system([1.0, np.nan], [0.01, 0.02])
        with pytest.raises(ValueError, match="finite"):
            synthetic_system([1.0, np.inf], [0.01, 0.02])
        with pytest.raises(ValueError, match="finite"):
            synthetic_system([1.0], [np.nan])

    def test_non_finite_mode_shapes_raise(self) -> None:
        """A non-finite modal matrix is rejected before inversion."""
        bad = np.array([[1.0, 0.0], [0.0, np.nan]])
        with pytest.raises(ValueError, match="finite"):
            synthetic_system([1.0, 2.0], [0.01, 0.02], mode_shapes=bad)

    def test_singular_mode_shapes_raise(self) -> None:
        """A singular modal matrix raises."""
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="invertible"):
            synthetic_system([1.0, 2.0], [0.01, 0.02], mode_shapes=singular)


class TestScoreRecovery:
    """Tests for score_recovery."""

    def test_clean_recovery_is_near_exact(self) -> None:
        """Extraction from a noise-free system recovers the modes near-exactly."""
        truth = synthetic_system(_FREQS, _ZETAS)
        score = score_recovery(compute_modes(truth.a, ndof2=3, ndof1=0), truth)
        assert score.n_matched == 3
        assert score.max_frequency_error < 1e-9
        assert score.max_damping_error < 1e-9
        assert score.min_mac > 1.0 - 1e-9

    def test_nontrivial_mode_shapes_recovered(self) -> None:
        """Coupled (non-identity) mode shapes are recovered with MAC ~ 1."""
        truth = synthetic_system(_FREQS, _ZETAS, mode_shapes=_orthogonal_shapes())
        score = score_recovery(compute_modes(truth.a, ndof2=3, ndof1=0), truth)
        assert score.min_mac > 0.999
        assert score.max_frequency_error < 1e-9

    def test_missing_recovered_mode_is_penalized(self) -> None:
        """A ground-truth mode that was not recovered drives min_mac to zero."""
        truth = synthetic_system([1.0, 2.0, 3.0], [0.02, 0.03, 0.04])
        # A 3-DOF system whose third mode is overdamped, so only two are recovered.
        n = 3
        omega = 2.0 * np.pi * np.array([1.0, 2.0, 3.0])
        stiffness = np.diag(omega**2)
        damping = np.diag([2 * 0.02 * omega[0], 2 * 0.03 * omega[1], 4.0 * omega[2]])
        a = np.block([[np.zeros((n, n)), np.eye(n)], [-stiffness, -damping]])
        solution = compute_modes(a, ndof2=3, ndof1=0)
        assert solution.n_modes == 2
        score = score_recovery(solution, truth)
        assert score.n_matched == 2
        assert score.min_mac == 0.0
        assert score.mean_mac < 0.7  # one of three truth modes contributes 0

    def test_empty_solution_raises(self) -> None:
        """Scoring a solution with no modes raises."""
        truth = synthetic_system([1.0], [0.01])
        overdamped = compute_modes(
            np.array([[0.0, 1.0], [-1.0, -5.0]]), ndof2=1, ndof1=0
        )
        assert overdamped.n_modes == 0
        with pytest.raises(ValueError, match="no modes"):
            score_recovery(overdamped, truth)


class TestRobustness:
    """Tests for robustness_curve."""

    def test_error_is_zero_without_noise_and_grows_monotonically(self) -> None:
        """No noise recovers exactly; a fixed-direction sweep grows monotonically."""
        truth = synthetic_system(_FREQS, _ZETAS, mode_shapes=_orthogonal_shapes())
        curve = robustness_curve(truth, [0.0, 1e-5, 1e-4, 1e-3], seed=1)
        errors = [score.max_frequency_error for _, score in curve]
        assert errors[0] < 1e-9
        # One fixed perturbation direction -> a pure amplitude sweep -> monotone.
        assert all(a <= b for a, b in pairwise(errors))
        assert errors[-1] < 0.1  # bounded for moderate noise

    def test_noise_amplitude_does_not_scale_with_system_size(self) -> None:
        """A normalized direction keeps sigma a true relative amplitude for any size."""
        truth = synthetic_system([float(i) for i in range(1, 13)], [0.02] * 12)
        error = robustness_curve(truth, [1e-3], seed=2)[0][1].max_frequency_error
        # ~0.1% relative noise -> small error; a raw (un-normalized) direction would
        # inject ~sqrt(2*12^2) times more and blow this past the bound.
        assert error < 0.02

    def test_negative_noise_level_raises(self) -> None:
        """A negative noise level is rejected."""
        truth = synthetic_system(_FREQS, _ZETAS)
        with pytest.raises(ValueError, match="non-negative"):
            robustness_curve(truth, [0.0, -1e-3])

    def test_is_deterministic(self) -> None:
        """The same seed yields the same robustness curve."""
        truth = synthetic_system(_FREQS, _ZETAS)
        first = robustness_curve(truth, [1e-3], seed=7)[0][1]
        second = robustness_curve(truth, [1e-3], seed=7)[0][1]
        assert first.max_frequency_error == second.max_frequency_error
