# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.eigensolver — eigenvalue analysis of the state matrix."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

from vane.io.mbc_transform import MBCResult
from vane.modal.eigensolver import ModalSolution, compute_modes, modes_from_mbc


def _second_order_a(
    frequencies: tuple[float, ...], dampings: tuple[float, ...]
) -> np.ndarray:
    """Return [[0, I], [-K, -C]] for uncoupled oscillators in [q2, q2dot] order."""
    stiffness = np.diag([w**2 for w in frequencies])
    damping = np.diag([2.0 * z * w for z, w in zip(dampings, frequencies, strict=True)])
    n = len(frequencies)
    a = np.zeros((2 * n, 2 * n), dtype=np.float64)
    a[0:n, n : 2 * n] = np.eye(n)
    a[n : 2 * n, 0:n] = -stiffness
    a[n : 2 * n, n : 2 * n] = -damping
    return a


def _mbc_result(a: np.ndarray | None, ndof2: int, ndof1: int) -> MBCResult:
    """Build a minimal MBCResult wrapping a state matrix for testing."""
    return MBCResult(
        avg_a=a,
        avg_b=None,
        avg_c=None,
        avg_d=None,
        state_descriptions=[f"state {i}" for i in range(2 * ndof2 + ndof1)],
        ndof2=ndof2,
        ndof1=ndof1,
        n_blades=3,
        performed_transformation=True,
        rotor_speed_rpm=0.0,
        wind_speed=0.0,
        azimuths_deg=np.zeros(1),
    )


class TestComputeModes:
    """Unit tests for compute_modes against analytical results."""

    def test_single_oscillator_matches_analytics(self) -> None:
        """A single 2nd-order DOF returns its analytical frequency and damping."""
        w0, zeta = 3.0, 0.05
        sol = compute_modes(_second_order_a((w0,), (zeta,)), ndof2=1, ndof1=0)
        assert sol.n_modes == 1
        assert sol.natural_frequencies_hz[0] == pytest.approx(w0 / (2 * np.pi))
        assert sol.damping_ratios[0] == pytest.approx(zeta)
        assert sol.damped_frequencies_hz[0] == pytest.approx(
            w0 * np.sqrt(1 - zeta**2) / (2 * np.pi)
        )
        assert sol.natural_frequencies_rad_s[0] == pytest.approx(w0)
        assert sol.n_rigid_body_modes == 0

    @pytest.mark.parametrize("name", ["descriptions", "mbc_coordinates"])
    @pytest.mark.parametrize("length", [3, 5])
    def test_per_state_lists_must_match_the_states(
        self, name: str, length: int
    ) -> None:
        """A description list of the wrong length is rejected, not misaligned.

        A longer list was accepted silently, attaching its entries (and the labels
        derived from them) to the wrong states.
        """
        a = _second_order_a((2.0, 5.0), (0.02, 0.03))
        with pytest.raises(ValueError, match=f"{name} has {length} entries"):
            compute_modes(a, ndof2=2, ndof1=0, **{name: ["x"] * length})

    def test_mode_shape_row_descriptions_are_accepted(self) -> None:
        """Descriptions of only the mode-shape rows work with first-order states too.

        Such a list was indexed as if it covered every state, which raised an
        IndexError as soon as the model had a first-order state.
        """
        a = np.zeros((3, 3))
        a[:2, :2] = _second_order_a((2.0,), (0.02,))
        a[2, 2] = -1.0
        sol = compute_modes(a, ndof2=1, ndof1=1, descriptions=["disp", "first"])
        assert sol.dof_descriptions == ["disp", "first"]
        assert sol.state_descriptions == []
        full = compute_modes(a, ndof2=1, ndof1=1, descriptions=["d", "v", "f"])
        assert full.dof_descriptions == ["d", "f"]
        assert full.state_descriptions == ["d", "v", "f"]

    def test_conjugate_pairs_collapse_to_one_per_mode(self) -> None:
        """Two DOFs (four eigenvalues) yield two under-damped modes."""
        sol = compute_modes(_second_order_a((2.0, 5.0), (0.02, 0.03)), ndof2=2, ndof1=0)
        assert sol.n_modes == 2

    def test_modes_sorted_by_ascending_frequency(self) -> None:
        """Returned modes are ordered by ascending natural frequency."""
        sol = compute_modes(_second_order_a((5.0, 2.0), (0.03, 0.02)), ndof2=2, ndof1=0)
        np.testing.assert_allclose(
            sol.natural_frequencies_hz,
            np.array([2.0, 5.0]) / (2 * np.pi),
            rtol=1e-9,
        )
        np.testing.assert_allclose(sol.damping_ratios, [0.02, 0.03], atol=1e-9)

    def test_mode_shape_dimensions(self) -> None:
        """Mode shapes drop velocity rows; full eigenvectors keep all states."""
        sol = compute_modes(_second_order_a((2.0, 5.0), (0.02, 0.03)), ndof2=2, ndof1=0)
        assert sol.mode_shapes.shape == (2, 2)
        assert sol.full_eigenvectors.shape == (4, 2)

    def test_mode_shape_dominant_component_is_real_positive(self) -> None:
        """Phase normalization makes each mode's dominant component real positive."""
        sol = compute_modes(_second_order_a((2.0, 5.0), (0.02, 0.03)), ndof2=2, ndof1=0)
        for col in range(sol.n_modes):
            shape = sol.mode_shapes[:, col]
            dominant = shape[np.argmax(np.abs(shape))]
            assert dominant.imag == pytest.approx(0.0, abs=1e-12)
            assert dominant.real > 0.0

    def test_rigid_body_mode_is_counted_not_returned(self) -> None:
        """A zero-stiffness DOF is counted as rigid-body, not a mode."""
        a = np.zeros((4, 4), dtype=np.float64)
        a[0:2, 2:4] = np.eye(2)
        a[2, 0], a[2, 2] = -4.0, -0.2  # DOF 1 is an oscillator; DOF 2 is rigid
        sol = compute_modes(a, ndof2=2, ndof1=0)
        assert sol.n_modes == 1
        assert sol.n_rigid_body_modes == 1

    def test_first_order_states_retained_in_mode_shapes(self) -> None:
        """First-order states appear in the reduced mode shapes after q2 rows."""
        a = _second_order_a((2.0,), (0.02,))  # 2 states (1 q2 DOF)
        big = np.zeros((3, 3), dtype=np.float64)
        big[0:2, 0:2] = a
        big[2, 2] = -1.0  # a stable first-order state
        sol = compute_modes(big, ndof2=1, ndof1=1)
        # reduced shape rows = [q2 displacement] + [q1] = 2 rows.
        assert sol.mode_shapes.shape[0] == 2

    def test_all_real_eigenvalues_yield_no_modes(self) -> None:
        """A matrix with no under-damped modes returns an empty, valid solution."""
        a = np.diag([-1.0, -2.0, -3.0])  # three stable first-order states
        sol = compute_modes(a, ndof2=0, ndof1=3)
        assert sol.n_modes == 0
        assert sol.n_rigid_body_modes == 3
        assert sol.mode_shapes.shape == (3, 0)
        assert sol.natural_frequencies_hz.shape == (0,)

    def test_non_square_raises(self) -> None:
        """A non-square matrix raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            compute_modes(np.zeros((2, 3)), ndof2=1, ndof1=0)

    def test_size_mismatch_raises(self) -> None:
        """A state-count mismatch raises ValueError."""
        with pytest.raises(ValueError, match="mismatch"):
            compute_modes(np.zeros((4, 4)), ndof2=1, ndof1=0)

    def test_non_finite_matrix_raises(self) -> None:
        """A state matrix with NaN/Inf is rejected instead of yielding NaN modes."""
        a = np.array([[0.0, 1.0], [-4.0, np.nan]])
        with pytest.raises(ValueError, match="non-finite"):
            compute_modes(a, ndof2=1, ndof1=0)

    def test_negative_partition_raises(self) -> None:
        """A negative DOF partition is rejected, even if the count equation holds.

        ndof2=-1, ndof1=3 satisfies 2*ndof2+ndof1 == 1 but is an impossible layout that
        would otherwise produce a degenerate (3, 0) mode-shape array from a 1x1 A.
        """
        with pytest.raises(ValueError, match="ndof2 must be a non-negative integer"):
            compute_modes(np.zeros((1, 1)), ndof2=-1, ndof1=3)

    def test_non_integer_partition_raises(self) -> None:
        """A non-integer DOF partition is rejected (untyped config guard)."""
        with pytest.raises(ValueError, match="non-negative integer"):
            compute_modes(np.zeros((1, 1)), ndof2=0.5, ndof1=0)  # type: ignore[arg-type]


class TestRobustEigenCore:
    """Tests for eigenvalue conditioning, degeneracy, and spectrum census."""

    def test_normal_matrix_is_well_conditioned(self) -> None:
        """A normal (skew-symmetric) matrix has condition number 1."""
        sol = compute_modes(np.array([[0.0, 2.0], [-2.0, 0.0]]), ndof2=1, ndof1=0)
        np.testing.assert_allclose(sol.condition_numbers, 1.0, atol=1e-9)
        assert sol.max_condition_number == pytest.approx(1.0)

    def test_non_normal_matrix_is_ill_conditioned(self) -> None:
        """A strongly non-normal matrix has a condition number well above 1."""
        sol = compute_modes(np.array([[-0.1, 10.0], [-0.1, -0.1]]), ndof2=1, ndof1=0)
        assert sol.condition_numbers[0] > 2.0

    def test_condition_number_at_least_one(self) -> None:
        """Every eigenvalue condition number is at least 1 (Cauchy-Schwarz)."""
        sol = compute_modes(_second_order_a((2.0, 5.0), (0.02, 0.03)), ndof2=2, ndof1=0)
        assert np.all(sol.condition_numbers >= 1.0 - 1e-9)

    def test_repeated_eigenvalues_flagged_degenerate(self) -> None:
        """Two identical oscillators yield repeated eigenvalues flagged degenerate."""
        sol = compute_modes(_second_order_a((3.0, 3.0), (0.02, 0.02)), ndof2=2, ndof1=0)
        assert sol.n_modes == 2
        assert np.all(sol.is_degenerate)

    def test_distinct_modes_not_degenerate(self) -> None:
        """Well-separated modes are not flagged degenerate."""
        sol = compute_modes(_second_order_a((2.0, 5.0), (0.02, 0.03)), ndof2=2, ndof1=0)
        assert not np.any(sol.is_degenerate)

    def test_overdamped_modes_counted(self) -> None:
        """Stable real eigenvalues are counted as over-damped, not returned."""
        sol = compute_modes(np.diag([-1.0, -2.0, -3.0]), ndof2=0, ndof1=3)
        assert sol.n_modes == 0
        assert sol.n_overdamped == 3
        assert sol.n_unstable == 0

    def test_unstable_eigenvalue_counted(self) -> None:
        """A positive-real-part eigenvalue is counted as unstable."""
        sol = compute_modes(np.array([[0.0, 1.0], [1.0, -0.1]]), ndof2=1, ndof1=0)
        assert sol.n_unstable == 1

    def test_slow_instability_in_stiff_spectrum_detected(self) -> None:
        """A small instability is not hidden by a large eigenvalue (stiff spectrum)."""
        # |lambda_max| ~ 1e6; a spectral-radius-scaled tolerance would hide the
        # genuine slow instability at Re = +1e-3, so it must still be detected.
        sol = compute_modes(np.diag([-1.0e6, 1.0e-3]), ndof2=0, ndof1=2)
        assert sol.n_unstable == 1


def _embedded_pairs(
    pairs: tuple[tuple[float, float], ...], *, skew: float
) -> np.ndarray:
    """Return a real first-order A with the eigenvalue pairs ``sigma +/- i omega``.

    ``A = V D V^-1`` with ``D`` block-diagonal and ``V`` a fixed random rotation.
    With ``skew > 0`` the first pair's eigenvectors are tilted to within ``skew`` of
    the last (stiff) pair's, which makes the first pair ill-conditioned the way the
    rotor-azimuth pair is coupled to the structural states of a real turbine.
    """
    blocks = [np.array([[s, w], [-w, s]]) for s, w in pairs]
    n = 2 * len(pairs)
    d = np.zeros((n, n))
    for i, block in enumerate(blocks):
        d[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block
    q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((n, n)))
    v = q.copy()
    if skew:
        v[:, 0:2] = q[:, -2:] + skew * q[:, 0:2]
    return np.asarray(v @ d @ np.linalg.inv(v), dtype=np.float64)


# A rotor-azimuth-like pair (0.00044 Hz), a floating-surge-like mode (0.008 Hz) and
# a stiff structural mode (2.9 Hz) that sets the spectral radius.
_AZIMUTH_PAIR = (-0.00175, 0.0021)
_SURGE_PAIR = (-0.0005, 2.0 * np.pi * 0.008)
_STIFF_PAIR = (-0.1, 18.0)


class TestSplitRigidBodyPairs:
    """Tests for classifying numerically split near-zero pairs as rigid-body."""

    def test_unresolved_slow_pair_is_rigid_body(self) -> None:
        """An ill-conditioned near-zero pair is dropped and counted as rigid-body."""
        a = _embedded_pairs((_AZIMUTH_PAIR, _SURGE_PAIR, _STIFF_PAIR), skew=0.01)
        sol = compute_modes(a, ndof2=0, ndof1=6)
        assert sol.n_split_rigid_body_pairs == 1
        assert sol.n_modes == 2
        assert sol.n_rigid_body_modes == 4  # 6 states - 2 oscillatory modes
        np.testing.assert_allclose(sol.natural_frequencies_hz, [0.008, 2.8648], 1e-4)

    def test_genuine_surge_mode_is_kept(self) -> None:
        """A slow but resolved platform mode (0.008 Hz) stays oscillatory."""
        a = _embedded_pairs((_AZIMUTH_PAIR, _SURGE_PAIR, _STIFF_PAIR), skew=0.01)
        sol = compute_modes(a, ndof2=0, ndof1=6)
        assert np.any(np.isclose(sol.damped_frequencies_hz, 0.008, rtol=1e-6))

    def test_resolved_pair_below_floor_is_kept(self) -> None:
        """A well-conditioned pair below the floor is resolved, so it is kept."""
        slow = (-0.0003, 2.0 * np.pi * 0.002)
        a = _embedded_pairs((slow, _SURGE_PAIR, _STIFF_PAIR), skew=0.0)
        sol = compute_modes(a, ndof2=0, ndof1=6)
        assert sol.n_split_rigid_body_pairs == 0
        assert sol.n_modes == 3

    def test_split_pair_is_not_counted_unstable(self) -> None:
        """A split pair's unresolved positive real part is not an instability."""
        unstable_split = (3e-4, 0.0021)
        a = _embedded_pairs((unstable_split, _SURGE_PAIR, _STIFF_PAIR), skew=0.01)
        sol = compute_modes(a, ndof2=0, ndof1=6)
        assert sol.n_split_rigid_body_pairs == 1
        assert sol.n_unstable == 0
        kept_all = compute_modes(a, ndof2=0, ndof1=6, rigid_body_frequency_hz=0.0)
        assert kept_all.n_unstable == 2

    @pytest.mark.parametrize(
        "kwargs", [{"rigid_body_frequency_hz": 0.0}, {"split_pair_rtol": 0.0}]
    )
    def test_zero_tolerance_disables_the_test(self, kwargs: dict[str, float]) -> None:
        """A zero floor or zero accuracy keeps every complex pair."""
        a = _embedded_pairs((_AZIMUTH_PAIR, _SURGE_PAIR, _STIFF_PAIR), skew=0.01)
        sol = compute_modes(a, ndof2=0, ndof1=6, **kwargs)
        assert sol.n_split_rigid_body_pairs == 0
        assert sol.n_modes == 3

    def test_floor_bounds_the_test(self) -> None:
        """A pair above the frequency floor is never reclassified."""
        a = _embedded_pairs((_AZIMUTH_PAIR, _SURGE_PAIR, _STIFF_PAIR), skew=0.01)
        sol = compute_modes(a, ndof2=0, ndof1=6, rigid_body_frequency_hz=1e-4)
        assert sol.n_split_rigid_body_pairs == 0

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("rigid_body_frequency_hz", -1.0),
            ("rigid_body_frequency_hz", float("nan")),
            ("split_pair_rtol", -1e-4),
            ("split_pair_rtol", float("inf")),
        ],
    )
    def test_invalid_tolerance_raises(self, name: str, value: float) -> None:
        """Negative or non-finite tolerances are rejected."""
        a = _second_order_a((2.0,), (0.02,))
        with pytest.raises(ValueError, match=name):
            compute_modes(a, ndof2=1, ndof1=0, **{name: value})

    def test_modes_from_mbc_forwards_the_tolerances(self) -> None:
        """modes_from_mbc passes the split-pair settings through."""
        a = _embedded_pairs((_AZIMUTH_PAIR, _SURGE_PAIR, _STIFF_PAIR), skew=0.01)
        result = _mbc_result(a, 0, 6)
        assert modes_from_mbc(result).n_split_rigid_body_pairs == 1
        disabled = modes_from_mbc(result, rigid_body_frequency_hz=0.0)
        assert disabled.n_split_rigid_body_pairs == 0


class TestModesFromMbc:
    """Tests for the MBCResult convenience wrapper."""

    def test_modes_from_mbc(self) -> None:
        """modes_from_mbc analyses the averaged A matrix."""
        result = _mbc_result(_second_order_a((2.0, 5.0), (0.02, 0.03)), 2, 0)
        sol = modes_from_mbc(result)
        assert sol.n_modes == 2

    def test_modes_from_mbc_without_a_raises(self) -> None:
        """modes_from_mbc raises when the averaged A matrix is absent."""
        with pytest.raises(ValueError, match="no averaged A"):
            modes_from_mbc(_mbc_result(None, 1, 0))


class TestEigensolverRTest:
    """Integration test on real CalcSteady MBC output (skipped if absent)."""

    def test_calcsteady_modes_are_physical(
        self, calcsteady_lin_files: list[Path]
    ) -> None:
        """Modes of the real 5MW land case are finite with sensible frequencies."""
        from vane.io.lin_reader import read_lin_file
        from vane.io.mbc_transform import mbc3_transform

        result = mbc3_transform([read_lin_file(p) for p in calcsteady_lin_files])
        sol = modes_from_mbc(result)
        assert sol.n_modes > 0
        assert np.all(sol.natural_frequencies_hz >= 0.0)
        assert np.all(np.isfinite(sol.natural_frequencies_hz))
        assert np.all(np.isfinite(sol.damping_ratios))
        assert sol.mode_shapes.shape[1] == sol.n_modes
        assert sol.full_eigenvectors.shape[1] == sol.n_modes


def _reference_runs_dir() -> Path:
    """Return the reference-turbine linearization folder, or skip when unavailable."""
    candidates = []
    env = os.environ.get("VANE_REFERENCE_RUNS_DIR")
    if env:
        candidates.append(Path(env))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("reference-turbine runs unavailable; set VANE_REFERENCE_RUNS_DIR")


def _reference_point(case_dir: Path, stem: str) -> ModalSolution:
    """Return the MBC modal solution of one azimuth sweep, or skip if absent."""
    from vane.io.lin_reader import read_lin_file
    from vane.io.mbc_transform import mbc3_transform

    pattern = re.compile(rf"{re.escape(stem)}\.(\d+)\.lin")
    matches = [
        (int(match.group(1)), path)
        for path in case_dir.glob(f"{stem}.*.lin")
        if (match := pattern.fullmatch(path.name)) is not None
    ]
    if len(matches) < 3:
        pytest.skip(f"reference-turbine sweep not found in {case_dir}")
    files = [path for _, path in sorted(matches)]
    return modes_from_mbc(mbc3_transform([read_lin_file(p) for p in files]))


class TestSplitRigidBodyPairsReferenceTurbine:
    """The rotor-azimuth pair of the NREL 5-MW land sweep (skipped if absent)."""

    _STEM = "5MW_Land_DLL_WTurb"

    def _solution(self, wind: int) -> ModalSolution:
        case = _reference_runs_dir() / "nrel5mw_land" / "cases" / f"ws{wind}"
        return _reference_point(case / self._STEM, self._STEM)

    def test_split_azimuth_pair_at_6_mps_is_rigid_body(self) -> None:
        """At 6 m/s the 0.0004 Hz azimuth pair is classified, not reported."""
        sol = self._solution(6)
        assert sol.n_split_rigid_body_pairs == 1
        assert sol.n_unstable == 0
        assert np.all(sol.natural_frequencies_hz > 0.01)

    def test_mode_count_matches_neighbouring_point(self) -> None:
        """6 m/s now has the same oscillatory modes as 8 m/s (real azimuth pair)."""
        at_6, at_8 = self._solution(6), self._solution(8)
        assert at_8.n_split_rigid_body_pairs == 0
        assert at_6.n_modes == at_8.n_modes
        assert at_6.n_rigid_body_modes == at_8.n_rigid_body_modes
