# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.mac — MAC, MACXP, and mode matching."""

from __future__ import annotations

import numpy as np
import pytest

from vane.modal.mac import compute_mac, compute_macxp, match_modes


def _random_modes(n_dof: int, n_modes: int, seed: int) -> np.ndarray:
    """Return a column-wise complex mode-shape matrix from a fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_dof, n_modes)) + 1j * rng.standard_normal(
        (n_dof, n_modes)
    )


class TestComputeMac:
    """Tests for the standard MAC."""

    def test_identical_modes_give_identity_diagonal(self) -> None:
        """MAC of a mode set with itself has a unit diagonal."""
        phi = _random_modes(12, 4, seed=1)
        mac = compute_mac(phi, phi)
        np.testing.assert_allclose(np.diag(mac), 1.0, atol=1e-12)

    def test_orthogonal_modes_give_zero(self) -> None:
        """Orthogonal mode shapes have a MAC of zero."""
        phi_ref = np.array([[1.0], [0.0], [0.0]], dtype=np.complex128)
        phi_test = np.array([[0.0], [1.0], [0.0]], dtype=np.complex128)
        assert compute_mac(phi_ref, phi_test)[0, 0] == pytest.approx(0.0)

    def test_values_within_unit_interval(self) -> None:
        """All MAC values lie within [0, 1]."""
        mac = compute_mac(_random_modes(20, 5, 2), _random_modes(20, 6, 3))
        assert mac.shape == (5, 6)
        assert np.all(mac >= -1e-12)
        assert np.all(mac <= 1.0 + 1e-12)

    def test_invariant_to_scaling(self) -> None:
        """MAC is invariant to per-mode complex scaling."""
        phi = _random_modes(10, 3, seed=4)
        scaled = phi * np.array([2.0, 1j, -3.0 + 1j])
        np.testing.assert_allclose(compute_mac(phi, scaled), compute_mac(phi, phi))

    def test_incompatible_dof_raises(self) -> None:
        """Mismatched DOF dimensions raise ValueError."""
        with pytest.raises(ValueError, match="incompatible"):
            compute_mac(_random_modes(10, 2, 5), _random_modes(8, 2, 6))


class TestComputeMacxp:
    """Tests for the pole-weighted MACXP."""

    def test_self_correlation_is_one(self) -> None:
        """A complex mode correlated with itself has MACXP equal to 1."""
        phi = _random_modes(12, 4, seed=7)
        lam = np.array([-0.1 + 2j, -0.2 + 5j, -0.05 + 8j, -0.3 + 11j])
        macxp = compute_macxp(phi, lam, phi, lam)
        np.testing.assert_allclose(np.diag(macxp), 1.0, atol=1e-10)

    def test_distinct_modes_below_self(self) -> None:
        """Off-diagonal MACXP entries are smaller than the unit diagonal."""
        phi = _random_modes(15, 3, seed=8)
        lam = np.array([-0.1 + 2j, -0.2 + 5j, -0.05 + 8j])
        macxp = compute_macxp(phi, lam, phi, lam)
        assert np.all(np.diag(macxp) > macxp[~np.eye(3, dtype=bool)].max())

    def test_undamped_pole_masks_its_row_and_column(self) -> None:
        """Every entry involving an undamped pole is NaN, not a misleading 0.0."""
        phi = _random_modes(12, 3, seed=10)
        lam = np.array([-0.1 + 2j, 0.0 + 5j, -0.05 + 8j])
        macxp = compute_macxp(phi, lam, phi, lam)
        assert np.all(np.isnan(macxp[1, :]))
        assert np.all(np.isnan(macxp[:, 1]))
        damped = np.ix_([0, 2], [0, 2])
        assert np.all(np.isfinite(macxp[damped]))
        np.testing.assert_allclose(np.diag(macxp)[[0, 2]], 1.0, atol=1e-10)

    def test_pole_count_mismatch_raises(self) -> None:
        """A pole/mode count mismatch raises ValueError."""
        phi = _random_modes(10, 3, seed=9)
        with pytest.raises(ValueError, match="poles"):
            compute_macxp(phi, np.array([1j, 2j]), phi, np.array([1j, 2j, 3j]))


class TestMatchModes:
    """Tests for Hungarian mode matching."""

    def test_permuted_modes_are_recovered(self) -> None:
        """A permuted test set is matched back to the reference order."""
        phi = _random_modes(16, 5, seed=10)
        permutation = [2, 0, 4, 1, 3]
        mac = compute_mac(phi, phi[:, permutation])
        pairs = match_modes(mac)
        assert pairs == [(i, permutation.index(i)) for i in range(5)]

    def test_frequency_penalty_breaks_shape_ties(self) -> None:
        """Equal-MAC candidates are resolved by frequency proximity."""
        # Two test modes correlate equally with the reference; the closer
        # frequency should win once the penalty is applied.
        mac = np.array([[1.0, 1.0]])
        ref_freq = np.array([2.0])
        test_freq = np.array([5.0, 2.1])
        pairs = match_modes(mac, ref_freq, test_freq, frequency_weight=1.0)
        assert pairs == [(0, 1)]

    def test_rectangular_matches_minimum_count(self) -> None:
        """With more reference than test modes, only test-many pairs are returned."""
        mac = compute_mac(_random_modes(12, 4, 11), _random_modes(12, 2, 12))
        assert len(match_modes(mac)) == 2

    def test_forbidden_pair_is_a_hard_constraint(self) -> None:
        """A forbidden pair is avoided even when the frequency penalty zeroes it.

        With a full frequency penalty the (0, 0) pair's weight would be scaled to zero,
        so a soft negative sentinel on it vanishes; the constraint must still hold.
        """
        mac = np.array([[0.0, 0.2], [0.0, 0.9]])
        forbidden = np.array([[True, False], [False, False]])
        pairs = match_modes(
            mac,
            np.array([1.0, 2.0]),
            np.array([10.0, 2.0]),
            frequency_weight=1.0,
            forbidden=forbidden,
        )
        assert pairs == [(0, 1), (1, 0)]

    def test_forbidden_shape_and_feasibility_are_checked(self) -> None:
        """A mis-shaped mask or an unavoidable forbidden pair is rejected."""
        mac = np.eye(2)
        with pytest.raises(ValueError, match="forbidden must have shape"):
            match_modes(mac, forbidden=np.zeros((3, 2), dtype=bool))
        with pytest.raises(ValueError, match="avoids every forbidden pair"):
            match_modes(mac, forbidden=np.array([[True, True], [False, False]]))
