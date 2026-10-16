# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.uncertainty — azimuth spread and unified confidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.io.lin_reader import read_lin_file
from vane.io.mbc_transform import mbc3_transform
from vane.modal.eigensolver import compute_modes
from vane.modal.uncertainty import azimuth_spread, unified_mode_confidence

if TYPE_CHECKING:
    from tests.conftest import LinFileFactory
    from vane.io.mbc_transform import MBCResult

_NORMAL_A = np.array([[0.0, 2.0], [-2.0, 0.0]])  # skew-symmetric: condition number 1


def _rotor_states() -> list[tuple[str, bool, int]]:
    disp: list[tuple[str, bool, int]] = [
        (
            f"ED 1st flapwise bending-mode DOF of blade {b} "
            f"(internal DOF index = DOF_BF({b},1)), m",
            True,
            2,
        )
        for b in (1, 2, 3)
    ]
    return disp + [
        (f"ED First time derivative of {desc}", True, 2) for desc, _, _ in disp
    ]


def _rotor_a(frequencies: tuple[float, float, float]) -> np.ndarray:
    a = np.zeros((6, 6), dtype=np.float64)
    a[0:3, 3:6] = np.eye(3)
    a[3:6, 0:3] = -np.diag([w**2 for w in frequencies])
    a[3:6, 3:6] = -2.0 * 0.02 * np.diag(frequencies)
    return a


def _transform(
    make_lin_file: LinFileFactory,
    frequencies: tuple[float, float, float],
    *,
    retain: bool = True,
) -> MBCResult:
    states = _rotor_states()
    a = _rotor_a(frequencies)
    files = [
        read_lin_file(make_lin_file(states=states, a=a, rotor_speed=1.5, azimuth=psi))
        for psi in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    ]
    return mbc3_transform(files, retain_per_azimuth=retain)


class TestAzimuthSpread:
    """Tests for azimuth_spread."""

    def test_isotropic_rotor_has_near_zero_spread(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """An isotropic rotor is azimuth-invariant: the spread is ~0."""
        spread = azimuth_spread(_transform(make_lin_file, (2.0, 2.0, 2.0)))
        assert spread.n_azimuths == 6
        assert np.all(np.nan_to_num(spread.natural_frequency_std) < 1e-6)

    def test_anisotropic_rotor_has_nonzero_spread(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """An anisotropic rotor is azimuth-dependent: some spread is non-zero."""
        spread = azimuth_spread(_transform(make_lin_file, (1.8, 2.0, 2.2)))
        # ~1e-4 for this mild anisotropy, vs ~1e-16 for the isotropic rotor.
        assert float(spread.natural_frequency_std.max()) > 1e-5

    def test_requires_retained_per_azimuth(self, make_lin_file: LinFileFactory) -> None:
        """Spread on a result without retained matrices raises ValueError."""
        result = _transform(make_lin_file, (2.0, 2.0, 2.0), retain=False)
        with pytest.raises(ValueError, match="retain"):
            azimuth_spread(result)

    def test_no_triplet_path_honors_retain(self, make_lin_file: LinFileFactory) -> None:
        """A no-triplet (non-performed) result still retains per-azimuth matrices."""
        states: list[tuple[str, bool, int]] = [
            ("ED tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
            (
                "ED First time derivative of tower fore-aft "
                "(internal DOF index = DOF_TFA1), m/s",
                False,
                2,
            ),
        ]
        a = np.array([[0.0, 1.0], [-4.0, -0.2]])
        files = [
            read_lin_file(make_lin_file(states=states, a=a, azimuth=psi))
            for psi in (0.0, 0.5)
        ]
        result = mbc3_transform(files, retain_per_azimuth=True)
        assert result.performed_transformation is False
        assert result.per_azimuth_a is not None
        assert azimuth_spread(result).n_azimuths == 2

    def test_spread_is_aligned_with_the_modal_solution(self) -> None:
        """The spread has one entry per ModalSolution mode, in the same order.

        The eigensolver drops a numerically split rigid-body pair (the free rotor
        azimuth); the spread used to keep it, so its arrays were longer than the
        solution's and could not be passed to unified_mode_confidence.
        """
        from tests.test_modal.test_eigensolver import (
            _AZIMUTH_PAIR,
            _STIFF_PAIR,
            _SURGE_PAIR,
            _embedded_pairs,
            _mbc_result,
        )
        from vane.modal.eigensolver import modes_from_mbc

        a = _embedded_pairs((_AZIMUTH_PAIR, _SURGE_PAIR, _STIFF_PAIR), skew=0.01)
        result = _mbc_result(a, 0, 6)
        result.per_azimuth_a = [a, a * 1.001, a * 0.999]
        solution = modes_from_mbc(result)
        assert solution.n_split_rigid_body_pairs == 1
        spread = azimuth_spread(result)
        assert spread.natural_frequency_std.shape == (solution.n_modes,)
        assert spread.damping_ratio_std.shape == (solution.n_modes,)
        # Scaling A scales every eigenvalue, so each spread is ~0.1 % of its mode.
        np.testing.assert_allclose(
            spread.natural_frequency_std,
            solution.natural_frequencies_hz * 0.001 * np.sqrt(2.0 / 3.0),
            rtol=1e-3,
        )
        confidence = unified_mode_confidence(
            solution, frequency_spread=spread.natural_frequency_std
        )
        assert confidence.shape == (solution.n_modes,)


class TestUnifiedConfidence:
    """Tests for unified_mode_confidence."""

    def test_well_conditioned_mode_is_fully_confident(self) -> None:
        """A normal matrix with no spread or degeneracy has confidence 1."""
        solution = compute_modes(_NORMAL_A, ndof2=1, ndof1=0)
        confidence = unified_mode_confidence(solution)
        assert confidence.shape == (1,)
        assert confidence[0] == pytest.approx(1.0)

    def test_azimuth_spread_lowers_confidence(self) -> None:
        """A frequency spread reduces a mode's confidence."""
        solution = compute_modes(_NORMAL_A, ndof2=1, ndof1=0)
        base = unified_mode_confidence(solution)
        spread = 0.1 * solution.natural_frequencies_hz
        penalized = unified_mode_confidence(solution, frequency_spread=spread)
        assert penalized[0] < base[0]

    def test_tracking_confidence_multiplies(self) -> None:
        """A low tracking confidence reduces the unified confidence."""
        solution = compute_modes(_NORMAL_A, ndof2=1, ndof1=0)
        penalized = unified_mode_confidence(solution, track_confidence=np.array([0.5]))
        assert penalized[0] == pytest.approx(0.5)

    def test_confidence_feeds_kalman_covariance(self) -> None:
        """A low-confidence mode produces a larger initial covariance (loop closed)."""
        from vane.sysid.covariance_init import covariances_from_confidence

        covariances = covariances_from_confidence([0.9, 0.2], n_outputs=1)
        # Mode 0 (confident) -> states 0,1; mode 1 (uncertain) -> states 2,3.
        assert covariances.p0[2, 2] > covariances.p0[0, 0]
