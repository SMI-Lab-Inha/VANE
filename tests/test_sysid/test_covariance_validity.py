# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid._covariance — scale-invariant, consistent covariance checks.

The same ``Q``/``R``/``P0`` must be accepted or rejected identically by every entry
point that consumes a covariance, whatever the physical units (and so the magnitude) of
the channels.
"""

import contextlib
import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from scipy.linalg import block_diag

from vane.sysid import (
    CovarianceSet,
    StateSpace,
    augment_with_disturbance,
    build_kalman_model,
    discretize_process_noise,
    observability_gramian,
    random_walk_disturbance,
    steady_state_estimator,
)
from vane.sysid._covariance import (
    is_positive_definite,
    is_positive_semidefinite,
    is_symmetric,
    validate_covariance,
)

Matrix = npt.NDArray[np.float64]


def _block(freq: float, zeta: float) -> Matrix:
    """Return a stable continuous 2x2 modal block."""
    wn = 2.0 * np.pi * freq
    sigma = -zeta * wn
    wd = wn * np.sqrt(1.0 - zeta**2)
    return np.array([[sigma, -wd], [wd, sigma]])


def _continuous_system() -> StateSpace:
    """Return a stable continuous 4-state, 2-output system."""
    a = np.asarray(block_diag(_block(0.3, 0.05), _block(1.5, 0.05)))
    c = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    return StateSpace(a=a, c=c, output_names=["y1", "y2"])


def _discrete_system() -> StateSpace:
    """Return the discretized system."""
    return _continuous_system().discretized(0.05)


_Q = 1e-3 * np.eye(4)
_R = 2.5e-3 * np.eye(2)
_SPD_2 = np.array([[2.0, 0.5], [0.5, 1.0]])
# Mixed units: a mooring-tension variance (N^2) next to an angle variance (rad^2).
_MIXED_DIAG = np.diag([1e8, 1e-6])
_MIXED_CORR = np.array([[1e8, 6.0], [6.0, 1e-6]])  # correlation 0.6

# --- entry points: each raises ValueError on an invalid matrix ---------------------


def _r_kalman(r: Matrix) -> None:
    build_kalman_model(_discrete_system(), CovarianceSet(q=_Q, r=r, p0=_Q))


def _r_assessment(r: Matrix) -> None:
    steady_state_estimator(_discrete_system(), process_noise=_Q, measurement_noise=r)


def _r_observability(r: Matrix) -> None:
    observability_gramian(_continuous_system(), measurement_noise=r)


def _q_kalman(q: Matrix) -> None:
    build_kalman_model(_discrete_system(), CovarianceSet(q=q, r=_R, p0=_Q))


def _p0_kalman(p0: Matrix) -> None:
    build_kalman_model(_discrete_system(), CovarianceSet(q=_Q, r=_R, p0=p0))


def _q_assessment(q: Matrix) -> None:
    steady_state_estimator(_discrete_system(), process_noise=q, measurement_noise=_R)


def _q_discretize(q: Matrix) -> None:
    discretize_process_noise(_continuous_system().a, q, 0.05)


def _q_augmentation(q: Matrix) -> None:
    augment_with_disturbance(
        _continuous_system(),
        random_walk_disturbance(1.0),
        entry=np.ones((4, 1)),
        process_noise=q,
    )


R_ENTRY_POINTS: list[Callable[[Matrix], None]] = [
    _r_kalman,
    _r_assessment,
    _r_observability,
]
Q_ENTRY_POINTS: list[Callable[[Matrix], None]] = [
    _q_kalman,
    _p0_kalman,
    _q_assessment,
    _q_discretize,
    _q_augmentation,
]


def _embed(block: Matrix) -> Matrix:
    """Embed a 2x2 block into a 4x4 matrix with a well-scaled positive complement."""
    scale = float(np.max(np.abs(block))) or 1.0
    out = scale * np.eye(4)
    out[:2, :2] = block
    return out


class TestMeasurementNoiseConsistency:
    """``R`` (positive definite) is judged identically by every entry point."""

    @pytest.mark.parametrize("entry", R_ENTRY_POINTS)
    @pytest.mark.parametrize(
        "r",
        [
            1e-12 * np.eye(2),
            1e-12 * _SPD_2,
            1e12 * _SPD_2,
            _MIXED_DIAG,
            _MIXED_CORR,
            # Asymmetric within tolerance and nearly singular: the upper triangle alone
            # is indefinite, so a factorization must use the validated symmetric part.
            np.array([[1.0, 1.0000000004], [0.999999999596, 1.0]]),
        ],
        ids=[
            "diag_1e-12",
            "tiny_spd",
            "huge_spd",
            "mixed_diag",
            "mixed_corr",
            "near_singular_asymmetric",
        ],
    )
    def test_small_scale_positive_definite_accepted(
        self, entry: Callable[[Matrix], None], r: Matrix
    ) -> None:
        """A well-conditioned R is valid whatever the units of each channel."""
        entry(r)

    @pytest.mark.parametrize("entry", R_ENTRY_POINTS)
    @pytest.mark.parametrize(
        "r",
        [
            1e-12 * np.diag([1.0, -0.5]),
            np.zeros((2, 2)),
            1e-6 * np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]]),
            np.array([[1e8, 10.0], [10.0, 1e-6]]),
            np.array([[1e8, 20.0], [20.0, 1e-6]]),
            np.array([[1.0, 0.3], [0.0, 1.0]]),
            np.array([[1e8, 6.0], [5.0, 1e-6]]),
        ],
        ids=[
            "tiny_indefinite",
            "zero",
            "numerically_singular",
            "mixed_perfectly_correlated",
            "mixed_indefinite",
            "non_symmetric",
            "mixed_non_symmetric",
        ],
    )
    def test_invalid_rejected(self, entry: Callable[[Matrix], None], r: Matrix) -> None:
        """Indefinite, singular, numerically singular, or asymmetric R is rejected."""
        with pytest.raises(ValueError, match=r"measurement_noise|R must"):
            entry(r)


class TestProcessNoiseConsistency:
    """``Q``/``P0`` (positive semidefinite) are judged identically everywhere."""

    @pytest.mark.parametrize("entry", Q_ENTRY_POINTS)
    @pytest.mark.parametrize(
        "q",
        [
            1e-12 * np.eye(4),
            np.zeros((4, 4)),
            _embed(1e-12 * _SPD_2),
            np.diag([1e-12, 0.0, 1e-12, 0.0]),
            np.diag([1e8, 1e-6, 1e8, 1e-6]),
            _embed(_MIXED_CORR),
            _embed(np.array([[1e8, 10.0], [10.0, 1e-6]])),
        ],
        ids=[
            "diag_1e-12",
            "zero",
            "tiny_spd",
            "tiny_rank_deficient",
            "mixed_diag",
            "mixed_corr",
            "mixed_perfectly_correlated",
        ],
    )
    def test_valid_accepted(self, entry: Callable[[Matrix], None], q: Matrix) -> None:
        """Small-scale and zero PSD matrices are valid covariances."""
        entry(q)

    @pytest.mark.parametrize("entry", Q_ENTRY_POINTS)
    def test_tiny_indefinite_rejected(self, entry: Callable[[Matrix], None]) -> None:
        """A clearly indefinite matrix is rejected however small its entries."""
        q = np.diag([1e-12, -0.5e-12, 1e-12, 1e-12])
        with pytest.raises(ValueError, match="positive semidefinite"):
            entry(q)

    @pytest.mark.parametrize("entry", Q_ENTRY_POINTS)
    def test_mixed_unit_indefinite_rejected(
        self, entry: Callable[[Matrix], None]
    ) -> None:
        """A correlation beyond one is rejected even across very different units."""
        with pytest.raises(ValueError, match="positive semidefinite"):
            entry(_embed(np.array([[1e8, 20.0], [20.0, 1e-6]])))

    @pytest.mark.parametrize("entry", Q_ENTRY_POINTS)
    def test_non_symmetric_rejected(self, entry: Callable[[Matrix], None]) -> None:
        """An asymmetric matrix is rejected."""
        q = 1e-12 * np.eye(4)
        q[0, 1] = 0.3e-12
        with pytest.raises(ValueError, match="symmetric"):
            entry(q)


# Matrices with a non-negative diagonal: every outcome is exactly invariant under any
# positive per-channel (Jacobi) rescaling S M S.
_JACOBI_CASES: list[Matrix] = [
    np.eye(3),
    np.diag([1.0, 1e-6, 1e-11]),
    np.diag([1.0, 1e-13, 1.0]),
    np.zeros((3, 3)),
    np.diag([1.0, 0.0, 2.0]),
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
    np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]]),
    np.array([[1.0, 1.0 - 1e-13], [1.0 - 1e-13, 1.0]]),
    np.array([[1.0, 1.0 - 1e-10], [1.0 - 1e-10, 1.0]]),
    np.array([[1.0, 2.0], [2.0, 1.0]]),
    np.array([[2.0, 0.5], [0.5, 1.0]]),
    np.array([[1.0, 0.5], [0.4, 1.0]]),
    np.ones((2, 2)),
    _MIXED_DIAG,
    _MIXED_CORR,
]
# A negative diagonal entry, and the row of a zero-variance channel, are judged against
# a round-off allowance relative to max|M_ii| (a zero variance carries no scale of its
# own), so those cases are invariant under global scaling only.
_CASES: list[Matrix] = [
    *_JACOBI_CASES,
    np.array([[0.0, 1.0], [1.0, 1.0]]),
    np.diag([1.0, -0.5, 2.0]),
    np.diag([1.0, -1e-12, 2.0]),
]


class TestPredicates:
    """Direct tests of the shared predicates."""

    @pytest.mark.parametrize("matrix", _CASES)
    @pytest.mark.parametrize("factor", [1e-15, 1e15])
    def test_scale_invariance(self, matrix: Matrix, factor: float) -> None:
        """Every outcome is unchanged when the matrix is scaled by 1e-15 or 1e15."""
        scaled = factor * matrix
        assert is_symmetric(scaled) == is_symmetric(matrix)
        assert is_positive_semidefinite(scaled) == is_positive_semidefinite(matrix)
        assert is_positive_definite(scaled) == is_positive_definite(matrix)

    def test_zero_variance_row_must_vanish_whatever_the_units(self) -> None:
        """A zero-variance channel with any coupling is indefinite in every unit."""
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0e-5], [0.0, 1.0e-5, 1.0]])
        s = np.array([1.0, 1.0e10, 1.0])
        rescaled = s[:, None] * matrix * s[None, :]
        assert not is_positive_semidefinite(matrix)
        assert not is_positive_semidefinite(rescaled)
        exact_zero_row = np.diag([1.0, 0.0, 1.0])
        assert is_positive_semidefinite(exact_zero_row)
        assert is_positive_semidefinite(s[:, None] * exact_zero_row * s[None, :])

    def test_zero_variance_bound_does_not_overflow(self) -> None:
        """Near-overflow variances cannot turn the row bound infinite."""
        matrix = np.array([[-8.0e298, 8.0e307], [8.0e307, 8.0e307]])
        assert not is_positive_semidefinite(matrix)

    def test_largest_finite_variances_are_accepted(self) -> None:
        """Symmetrization cannot overflow a finite covariance near the float64 limit."""
        largest = float(np.finfo(np.float64).max)
        matrix = np.diag([largest, largest])
        assert is_symmetric(matrix)
        assert is_positive_semidefinite(matrix)
        assert is_positive_definite(matrix)
        validate_covariance(matrix, "R", definite=True)

    def test_overflowing_correlation_is_rejected_cleanly(self) -> None:
        """A correlation that overflows is rejected without a numerical warning."""
        matrix = np.array([[1.0e-300, 1.0e300], [1.0e300, 1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert not is_positive_semidefinite(matrix)
            assert not is_positive_definite(matrix)

    def test_subnormal_variance_is_preserved(self) -> None:
        """The smallest subnormal variance is a valid, unit-normalizable covariance."""
        smallest = float(np.nextafter(0.0, 1.0))
        matrix = np.array([[smallest]])
        assert is_positive_semidefinite(matrix)
        assert is_positive_definite(matrix)

    def test_extreme_asymmetry_is_rejected_without_warning(self) -> None:
        """An overflowing asymmetry is reported as asymmetry, not as a warning."""
        largest = float(np.finfo(np.float64).max)
        matrix = np.array([[largest, largest], [-largest, largest]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert not is_symmetric(matrix)
            with pytest.raises(ValueError, match="symmetric"):
                validate_covariance(matrix, "R", definite=True)

    @pytest.mark.parametrize("seed", range(3))
    def test_extreme_values_never_warn_or_raise_unexpectedly(self, seed: int) -> None:
        """Matrices built from float64 extremes are judged without warnings."""
        smallest = float(np.nextafter(0.0, 1.0))
        largest = float(np.finfo(np.float64).max)
        magnitudes = [0.0, smallest, 1e-300, 1.0, 1e300, largest]
        values = np.array([*magnitudes, *(-v for v in magnitudes if v)])
        rng = np.random.default_rng(seed)
        for _ in range(500):
            size = int(rng.integers(1, 4))
            matrix = rng.choice(values, size=(size, size))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                is_symmetric(matrix)
                is_positive_semidefinite(matrix)
                is_positive_definite(matrix)
                for definite in (True, False):
                    with contextlib.suppress(ValueError):
                        validate_covariance(matrix, "M", definite=definite)

    def test_asymmetry_is_not_hidden_by_extreme_channel_units(self) -> None:
        """An asymmetric pair stays asymmetric however the channels are rescaled."""
        matrix = np.array([[1.0, 1.0e-6], [0.0, 1.0]])
        s = np.array([1.0e10, 1.0e-10])
        rescaled = s[:, None] * matrix * s[None, :]
        assert not is_symmetric(matrix)
        assert not is_symmetric(rescaled)
        with pytest.raises(ValueError, match="symmetric"):
            validate_covariance(rescaled, "R", definite=True)

    @pytest.mark.parametrize("matrix", _JACOBI_CASES)
    @pytest.mark.parametrize("seed", range(5))
    def test_per_channel_scaling_invariance(self, matrix: Matrix, seed: int) -> None:
        """Every outcome is unchanged under a random positive per-channel rescaling."""
        rng = np.random.default_rng(seed)
        s = 10.0 ** rng.uniform(-8.0, 8.0, size=matrix.shape[0])
        scaled = s[:, None] * matrix * s[None, :]
        assert is_symmetric(scaled) == is_symmetric(matrix)
        assert is_positive_semidefinite(scaled) == is_positive_semidefinite(matrix)
        assert is_positive_definite(scaled) == is_positive_definite(matrix)

    def test_expected_outcomes(self) -> None:
        """The outcomes match the definitions on representative matrices."""
        assert is_positive_definite(np.diag([1.0, 1e-11]))
        assert is_positive_definite(np.diag([1.0, 1e-13]))  # unit disparity only
        assert is_positive_definite(_MIXED_DIAG)
        assert is_positive_definite(_MIXED_CORR)
        near_one = 1.0 - 1e-13  # correlation structure numerically singular
        assert not is_positive_definite(np.array([[1.0, near_one], [near_one, 1.0]]))
        assert is_positive_semidefinite(np.array([[1.0, near_one], [near_one, 1.0]]))
        assert not is_positive_definite(np.diag([1.0, 0.0]))
        assert is_positive_semidefinite(np.diag([1.0, 0.0]))
        assert not is_positive_semidefinite(np.array([[0.0, 1.0], [1.0, 1.0]]))
        assert not is_positive_semidefinite(np.array([[1.0, 2.0], [2.0, 1.0]]))
        assert is_positive_semidefinite(np.zeros((2, 2)))
        assert not is_positive_definite(np.zeros((2, 2)))
        assert not is_positive_semidefinite(np.diag([1.0, -1e-6]))
        assert is_positive_semidefinite(np.diag([1.0, -1e-12]))
        assert not is_symmetric(np.array([[1.0, 0.5], [0.4, 1.0]]))
        assert is_symmetric(np.zeros((2, 2)))
        empty = np.zeros((0, 0))
        assert is_symmetric(empty)
        assert is_positive_semidefinite(empty)
        assert is_positive_definite(empty)

    def test_non_finite_is_invalid(self) -> None:
        """A NaN/inf matrix is neither PSD nor PD."""
        bad = np.array([[np.nan, 0.0], [0.0, 1.0]])
        assert not is_positive_semidefinite(bad)
        assert not is_positive_definite(bad)

    def test_validate_covariance_messages(self) -> None:
        """validate_covariance names the failing requirement."""
        bad = np.array([[np.inf, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="X must be finite"):
            validate_covariance(bad, "X", definite=False)
        with pytest.raises(ValueError, match="X must be symmetric"):
            validate_covariance(np.array([[1.0, 0.5], [0.0, 1.0]]), "X", definite=True)
        with pytest.raises(ValueError, match="X must be positive definite"):
            validate_covariance(np.zeros((2, 2)), "X", definite=True)
        with pytest.raises(ValueError, match="X must be positive semidefinite"):
            validate_covariance(-np.eye(2), "X", definite=False)
        validate_covariance(np.zeros((2, 2)), "X", definite=False)
