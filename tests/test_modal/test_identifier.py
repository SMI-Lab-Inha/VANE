# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.identifier — cross-operating-point mode tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.config.dof_map import DofCategory
from vane.modal.eigensolver import ModalSolution
from vane.modal.identifier import _max_weight_matching, identify_modes
from vane.modal.labeler import ModeLabel

if TYPE_CHECKING:
    from tests.conftest import LinFileFactory

_TOWER = "ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m"
_DRIVE = "ED drivetrain (internal DOF index = DOF_DrTr), rad"
_DESCS = [_TOWER, _DRIVE]


def _solution(frequencies: list[float], shapes: np.ndarray) -> ModalSolution:
    """Build a ModalSolution with given frequencies and mode-shape columns."""
    n = len(frequencies)
    freq = np.array(frequencies, dtype=np.float64)
    return ModalSolution(
        eigenvalues=(-0.01 + 1j * freq).astype(np.complex128),
        natural_frequencies_hz=freq,
        damping_ratios=np.full(n, 0.01),
        damped_frequencies_hz=freq,
        mode_shapes=shapes.astype(np.complex128),
        full_eigenvectors=shapes.astype(np.complex128),
        n_rigid_body_modes=0,
        dof_descriptions=_DESCS,
    )


def _solution_nd(frequencies: list[float], shapes: np.ndarray) -> ModalSolution:
    """Build a ModalSolution with arbitrary DOF count from explicit mode shapes.

    Descriptions are generic placeholders of the right length so the cross-point
    consistency check passes; only the mode shapes drive the tracking.
    """
    n = len(frequencies)
    freq = np.array(frequencies, dtype=np.float64)
    descriptions = [f"DOF {i}" for i in range(shapes.shape[0])]
    return ModalSolution(
        eigenvalues=(-0.01 + 1j * freq).astype(np.complex128),
        natural_frequencies_hz=freq,
        damping_ratios=np.full(n, 0.01),
        damped_frequencies_hz=freq,
        mode_shapes=shapes.astype(np.complex128),
        full_eigenvectors=shapes.astype(np.complex128),
        n_rigid_body_modes=0,
        dof_descriptions=descriptions,
    )


_TOWER_SHAPE = np.array([[1.0], [0.0]])
_DRIVE_SHAPE = np.array([[0.0], [1.0]])
_BOTH = np.array([[1.0, 0.0], [0.0, 1.0]])


class TestTuningValidation:
    """identify_modes validates its tuning parameters (the lower-level API)."""

    def test_frequency_weight_out_of_range_raises(self) -> None:
        """A frequency weight outside [0, 1] is rejected."""
        with pytest.raises(ValueError, match="frequency_weight"):
            identify_modes([], frequency_weight=1.5)

    def test_mac_threshold_negative_raises(self) -> None:
        """A negative MAC threshold is rejected."""
        with pytest.raises(ValueError, match="mac_threshold"):
            identify_modes([], mac_threshold=-0.1)

    def test_nan_threshold_raises(self) -> None:
        """A NaN threshold is rejected (NaN compares false to the bounds)."""
        with pytest.raises(ValueError, match="mac_threshold"):
            identify_modes([], mac_threshold=float("nan"))

    def test_ambiguity_margin_out_of_range_raises(self) -> None:
        """An ambiguity margin outside [0, 1] is rejected."""
        with pytest.raises(ValueError, match="ambiguity_margin"):
            identify_modes([], ambiguity_margin=2.0)

    def test_unknown_strategy_raises(self) -> None:
        """An unrecognized tracking strategy is rejected before any work is done."""
        with pytest.raises(ValueError, match="strategy"):
            identify_modes([], strategy="greedy")  # type: ignore[arg-type]


class TestIdentifyModes:
    """Unit tests for identify_modes."""

    def test_consistent_modes_form_single_tracks(self) -> None:
        """Two well-separated modes across three operating points give two tracks."""
        solutions = [
            _solution([1.0, 2.0], _BOTH),
            _solution([1.1, 2.1], _BOTH),
            _solution([1.2, 2.2], _BOTH),
        ]
        result = identify_modes(solutions)
        assert len(result.tracks) == 2
        assert all(len(track.operating_points) == 3 for track in result.tracks)
        np.testing.assert_allclose(
            result.tracks[0].natural_frequencies_hz, [1.0, 1.1, 1.2]
        )

    def test_frequency_crossing_tracked_by_shape(self) -> None:
        """When frequencies cross, modes are tracked by mode shape, not frequency."""
        # Tower shape mode rises 1.0 -> 1.5; drivetrain shape mode falls 2.0 -> 1.4.
        solutions = [
            _solution([1.0, 2.0], _BOTH),
            _solution([1.5, 1.4], _BOTH),
        ]
        result = identify_modes(solutions)
        by_first = {track.natural_frequencies_hz[0]: track for track in result.tracks}
        np.testing.assert_allclose(by_first[1.0].natural_frequencies_hz, [1.0, 1.5])
        np.testing.assert_allclose(by_first[2.0].natural_frequencies_hz, [2.0, 1.4])

    def test_appearing_mode_starts_new_track(self) -> None:
        """A mode that appears at a later operating point starts its own track."""
        solutions = [
            _solution([1.0], _TOWER_SHAPE),
            _solution([1.1, 2.0], _BOTH),
        ]
        result = identify_modes(solutions)
        spans = sorted(len(track.operating_points) for track in result.tracks)
        assert spans == [1, 2]

    def test_disappearing_mode_closes_track(self) -> None:
        """A mode absent from a later point leaves a shorter track."""
        solutions = [
            _solution([1.0, 2.0], _BOTH),
            _solution([1.1], _TOWER_SHAPE),
        ]
        result = identify_modes(solutions)
        spans = sorted(len(track.operating_points) for track in result.tracks)
        assert spans == [1, 2]

    def test_tracks_are_labeled(self) -> None:
        """Each track receives a physical label from its participation."""
        result = identify_modes([_solution([1.0, 2.0], _BOTH)])
        categories = {track.label.category for track in result.tracks}
        assert DofCategory.TOWER_FORE_AFT_1 in categories
        assert DofCategory.DRIVETRAIN_TORSION in categories

    def test_single_operating_point(self) -> None:
        """A single operating point yields one length-one track per mode."""
        result = identify_modes([_solution([1.0, 2.0], _BOTH)])
        assert result.n_operating_points == 1
        assert all(len(track.operating_points) == 1 for track in result.tracks)

    def test_tracks_sorted_by_first_frequency(self) -> None:
        """Tracks are returned in ascending order of first natural frequency."""
        result = identify_modes([_solution([2.0, 1.0], np.fliplr(_BOTH))])
        firsts = [track.natural_frequencies_hz[0] for track in result.tracks]
        assert firsts == sorted(firsts)

    def test_empty_solutions(self) -> None:
        """No operating points yields an empty result."""
        result = identify_modes([])
        assert result.tracks == []
        assert result.n_operating_points == 0

    def test_inconsistent_dof_count_raises(self) -> None:
        """Solutions with different mode-shape DOF counts raise ValueError."""
        big = ModalSolution(
            eigenvalues=np.array([1j], dtype=np.complex128),
            natural_frequencies_hz=np.array([1.0]),
            damping_ratios=np.array([0.01]),
            damped_frequencies_hz=np.array([1.0]),
            mode_shapes=np.ones((3, 1), dtype=np.complex128),
            full_eigenvectors=np.ones((3, 1), dtype=np.complex128),
            n_rigid_body_modes=0,
            dof_descriptions=["a", "b", "c"],
        )
        with pytest.raises(ValueError, match="mode-shape DOFs"):
            identify_modes([_solution([1.0], _TOWER_SHAPE), big])

    def test_inconsistent_dof_descriptions_raises(self) -> None:
        """Same DOF count but different descriptions across points is rejected."""
        first = _solution([1.0], _TOWER_SHAPE)
        relabeled = ModalSolution(
            eigenvalues=np.array([1j], dtype=np.complex128),
            natural_frequencies_hz=np.array([1.1]),
            damping_ratios=np.array([0.01]),
            damped_frequencies_hz=np.array([1.1]),
            mode_shapes=_TOWER_SHAPE.astype(np.complex128),
            full_eigenvectors=_TOWER_SHAPE.astype(np.complex128),
            n_rigid_body_modes=0,
            dof_descriptions=["a different DOF", "another DOF"],
        )
        with pytest.raises(ValueError, match="different DOF descriptions"):
            identify_modes([first, relabeled])


class TestGlobalTrackQuality:
    """Tests for the global tracker's confidence and ambiguity reporting."""

    def test_clean_modes_are_confident_and_unambiguous(self) -> None:
        """Well-separated, persistent modes track at full confidence, no ambiguity."""
        solutions = [
            _solution([1.0, 2.0], _BOTH),
            _solution([1.1, 2.1], _BOTH),
            _solution([1.2, 2.2], _BOTH),
        ]
        result = identify_modes(solutions)
        assert len(result.tracks) == 2
        assert all(track.confidence > 0.99 for track in result.tracks)
        assert not any(track.is_ambiguous for track in result.tracks)

    def test_near_tie_link_is_flagged_ambiguous(self) -> None:
        """A mode with two near-equally correlated continuations is flagged."""
        start = _solution([1.0], np.array([[1.0], [0.0]]))
        # Both candidates correlate equally with the start mode shape [1, 0].
        split = _solution([1.0, 1.0], np.array([[1.0, 1.0], [0.4, -0.4]]))
        result = identify_modes([start, split])
        linked = [t for t in result.tracks if len(t.operating_points) == 2]
        assert len(linked) == 1
        assert linked[0].is_ambiguous
        assert linked[0].confidence < 1.0

    def test_used_target_is_never_reused_across_tracks(self) -> None:
        """Two modes correlating with one next-point mode never share that target."""
        # OP0 has two modes both correlating with OP0+1's single mode [1, 0].
        start = _solution([1.0, 1.0], np.array([[1.0, 1.0], [0.0, 0.1]]))
        single = _solution([1.0], np.array([[1.0], [0.0]]))
        result = identify_modes([start, single])
        seen: set[tuple[int, int]] = set()
        for track in result.tracks:
            for op, mode in zip(
                track.operating_points, track.mode_indices, strict=True
            ):
                assert (op, mode) not in seen
                seen.add((op, mode))
        assert len(seen) == 3  # 2 modes at OP0 + 1 at OP1, each assigned once

    def test_threshold_passing_link_kept_at_full_frequency_weight(self) -> None:
        """A high-MAC link is kept even when frequency_weight=1 zeroes its weight."""
        # Same shape, far-apart frequencies: at frequency_weight=1 the penalty is 0
        # for this single pair, but the threshold-passing link must still link.
        start = _solution([1.0], np.array([[1.0], [0.0]]))
        far = _solution([2.0], np.array([[1.0], [0.0]]))
        result = identify_modes([start, far], frequency_weight=1.0)
        assert len(result.tracks) == 1
        assert result.tracks[0].operating_points == [0, 1]

    def test_weak_link_lowers_confidence(self) -> None:
        """A track's confidence is its weakest inter-point correlation."""
        # Mode shape drifts substantially between points (lower but dominant MAC).
        solutions = [
            _solution([1.0], np.array([[1.0], [0.0]])),
            _solution([1.1], np.array([[1.0], [0.6]])),
        ]
        result = identify_modes(solutions)
        assert len(result.tracks) == 1
        assert result.tracks[0].confidence < 0.95


class TestMaxWeightMatching:
    """Unit tests for the simultaneous-strategy assignment helper."""

    def test_matching_beats_greedy_best_link_first(self) -> None:
        """The assignment finds the global optimum a greedy match would miss.

        A greedy best-link-first match takes ``(0, 0)`` at 0.9 and is then forced
        onto ``(1, 1)`` at 0.01 (total 0.91); the globally optimal matching is
        ``(0, 1)`` + ``(1, 0)`` at 0.85 + 0.8 = 1.65. This is the per-transition
        gain of the simultaneous strategy over a per-line extraction, exercised here
        at the level where it is realizable (a free affinity matrix; the MAC
        completeness identity makes a single realizable transition harder to rig).
        """
        affinity = np.array([[0.9, 0.85], [0.8, 0.01]])
        assert sorted(_max_weight_matching(affinity)) == [(0, 1), (1, 0)]

    def test_forbidden_links_are_never_selected(self) -> None:
        """A sub-threshold (-inf) link is dropped; only the finite link is matched."""
        affinity = np.array([[0.9, -np.inf], [-np.inf, -np.inf]])
        assert _max_weight_matching(affinity) == [(0, 0)]

    def test_rectangular_leaves_extra_modes_unmatched(self) -> None:
        """With more columns than rows, only the best column is matched."""
        affinity = np.array([[0.8, 0.2, -np.inf]])
        assert _max_weight_matching(affinity) == [(0, 0)]

    def test_empty_dimension_returns_no_matches(self) -> None:
        """An operating point with no modes produces no links."""
        assert _max_weight_matching(np.empty((0, 3))) == []
        assert _max_weight_matching(np.empty((2, 0))) == []


class TestSimultaneousStrategy:
    """The simultaneous strategy agrees with single-path on well-separated spectra."""

    def test_consistent_modes_form_single_tracks(self) -> None:
        """Two well-separated modes across three operating points give two tracks."""
        solutions = [
            _solution([1.0, 2.0], _BOTH),
            _solution([1.1, 2.1], _BOTH),
            _solution([1.2, 2.2], _BOTH),
        ]
        result = identify_modes(solutions, strategy="simultaneous")
        assert len(result.tracks) == 2
        assert all(len(track.operating_points) == 3 for track in result.tracks)

    def test_frequency_crossing_tracked_by_shape(self) -> None:
        """When frequencies cross, modes are tracked by mode shape, not frequency."""
        solutions = [
            _solution([1.0, 2.0], _BOTH),
            _solution([1.5, 1.4], _BOTH),
        ]
        result = identify_modes(solutions, strategy="simultaneous")
        by_first = {track.natural_frequencies_hz[0]: track for track in result.tracks}
        np.testing.assert_allclose(by_first[1.0].natural_frequencies_hz, [1.0, 1.5])
        np.testing.assert_allclose(by_first[2.0].natural_frequencies_hz, [2.0, 1.4])

    def test_appearing_mode_starts_new_track(self) -> None:
        """A mode that appears at a later operating point starts its own track."""
        solutions = [
            _solution([1.0], _TOWER_SHAPE),
            _solution([1.1, 2.0], _BOTH),
        ]
        result = identify_modes(solutions, strategy="simultaneous")
        spans = sorted(len(track.operating_points) for track in result.tracks)
        assert spans == [1, 2]

    def test_disappearing_mode_closes_track(self) -> None:
        """A mode absent from a later point leaves a shorter track."""
        solutions = [
            _solution([1.0, 2.0], _BOTH),
            _solution([1.1], _TOWER_SHAPE),
        ]
        result = identify_modes(solutions, strategy="simultaneous")
        spans = sorted(len(track.operating_points) for track in result.tracks)
        assert spans == [1, 2]

    def test_assignment_partitions_every_mode_once(self) -> None:
        """Every (operating point, mode) is assigned to exactly one track.

        The per-transition matching makes this an invariant of the strategy: each
        next-point mode is either linked to one track or starts one, and each
        previous-point mode heads at most one link.
        """
        rng = np.random.default_rng(20240611)
        shapes = [rng.standard_normal((4, 3)) for _ in range(5)]
        solutions = [
            _solution_nd([1.0 + 0.1 * k, 2.0 + 0.1 * k, 3.0 - 0.1 * k], shape)
            for k, shape in enumerate(shapes)
        ]
        result = identify_modes(solutions, strategy="simultaneous", mac_threshold=0.1)
        seen: set[tuple[int, int]] = set()
        for track in result.tracks:
            for op, mode in zip(
                track.operating_points, track.mode_indices, strict=True
            ):
                assert (op, mode) not in seen
                seen.add((op, mode))
        assert len(seen) == sum(solution.n_modes for solution in solutions)

    def test_tracks_sorted_by_first_frequency(self) -> None:
        """Tracks are returned in ascending order of first natural frequency."""
        result = identify_modes(
            [_solution([2.0, 1.0], np.fliplr(_BOTH))], strategy="simultaneous"
        )
        firsts = [track.natural_frequencies_hz[0] for track in result.tracks]
        assert firsts == sorted(firsts)


class TestBranchReview:
    """The branch_review flag attaches hotspots without changing the default."""

    def test_default_has_no_branch_groups(self) -> None:
        """Without branch_review the result carries no branch groups."""
        result = identify_modes([_solution([1.0, 2.0], _BOTH)])
        assert result.branch_groups == []

    def test_branch_review_attaches_groups(self) -> None:
        """A crossing is flagged on branch_groups when branch_review is requested."""
        mixed = np.array([[1.0, 0.9], [0.9, 1.0]])
        solutions = [
            _solution([1.0, 2.0], _BOTH),
            _solution([1.5, 1.5], mixed),
            _solution([2.0, 1.0], _BOTH),
        ]
        result = identify_modes(solutions, strategy="simultaneous", branch_review=True)
        assert len(result.branch_groups) == 1


def _isotropic_rotor_solution(
    make_lin_file: LinFileFactory, omega: float
) -> ModalSolution:
    """Return the MBC modal solution of three identical flap oscillators at omega."""
    from vane.io.lin_reader import read_lin_file
    from vane.io.mbc_transform import mbc3_transform
    from vane.modal.eigensolver import modes_from_mbc

    disp: list[tuple[str, bool, int]] = [
        (
            f"ED 1st flapwise bending-mode DOF of blade {b} "
            f"(internal DOF index = DOF_BF({b},1)), m",
            True,
            2,
        )
        for b in (1, 2, 3)
    ]
    states = disp + [
        (f"ED First time derivative of {desc}", True, 2) for desc, _, _ in disp
    ]
    w0 = 3.0
    a = np.zeros((6, 6), dtype=np.float64)
    a[0:3, 3:6] = np.eye(3)
    a[3:6, 0:3] = -(w0**2) * np.eye(3)
    a[3:6, 3:6] = -2.0 * 0.02 * w0 * np.eye(3)
    files = [
        read_lin_file(make_lin_file(states=states, a=a, rotor_speed=omega, azimuth=psi))
        for psi in (0.0, 2.0, 4.0)
    ]
    return modes_from_mbc(mbc3_transform(files))


class TestTrackLabels:
    """Track labels carry the multi-blade type and follow the track's majority."""

    def test_tracks_carry_the_multiblade_type(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The regressive/collective/progressive split survives into the tracks."""
        solutions = [
            _isotropic_rotor_solution(make_lin_file, omega) for omega in (1.0, 1.5)
        ]
        result = identify_modes(solutions)
        assert len(result.tracks) == 3
        assert [track.label.multiblade for track in result.tracks] == [
            "regressive",
            "collective",
            "progressive",
        ]
        assert [track.label.label for track in result.tracks] == [
            "1st Blade Flap (Regressive)",
            "1st Blade Flap (Collective)",
            "1st Blade Flap (Progressive)",
        ]

    def test_majority_label_beats_one_confident_outlier(self) -> None:
        """The label held along most of the track names it, not the single best."""
        from vane.modal.identifier import _track_label

        majority = ModeLabel(DofCategory.BLADE_FLAP_1, "B (Collective)", 0.6, [])
        outlier = ModeLabel(DofCategory.BLADE_EDGE_1, "E (Collective)", 0.99, [])

        class _Labels:
            def has_descriptions(self, op: int) -> bool:
                return True

            def label(self, op: int, mode: int) -> ModeLabel:
                return outlier if op == 0 else majority

        entries = [(0, 0), (1, 0), (2, 0), (3, 0)]
        assert _track_label(entries, _Labels()) is majority  # type: ignore[arg-type]


def _tower_and_slow_mode(slow_hz: float | None) -> ModalSolution:
    """Return a tower mode, optionally with a near-zero-frequency second mode."""
    if slow_hz is None:
        return _solution([0.3], _TOWER_SHAPE)
    return _solution([slow_hz, 0.3], np.fliplr(_BOTH))


class TestRigidBodyModes:
    """Modes below the rigid-body frequency floor are left out of every track."""

    @pytest.mark.parametrize("strategy", ["single_path", "simultaneous"])
    def test_isolated_near_zero_mode_is_not_tracked(self, strategy: str) -> None:
        """A rotor-azimuth pair resolved as complex at one point forms no track."""
        solutions = [
            _tower_and_slow_mode(None),
            _tower_and_slow_mode(0.0005),
            _tower_and_slow_mode(None),
        ]
        result = identify_modes(solutions, strategy=strategy)  # type: ignore[arg-type]
        assert len(result.tracks) == 1
        assert result.tracks[0].operating_points == [0, 1, 2]
        assert result.tracks[0].mode_indices == [0, 1, 0]
        everything = identify_modes(
            solutions,
            strategy=strategy,  # type: ignore[arg-type]
            rigid_body_frequency_hz=0.0,
        )
        assert len(everything.tracks) == 2

    def test_slow_restrained_modes_are_kept(self) -> None:
        """A floating platform's surge near 0.008 Hz is above the floor."""
        result = identify_modes([_tower_and_slow_mode(0.008)] * 2)
        assert len(result.tracks) == 2

    @pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
    def test_invalid_floor_raises(self, value: float) -> None:
        """A negative or non-finite floor is rejected."""
        with pytest.raises(ValueError, match="rigid_body_frequency_hz"):
            identify_modes(
                [_solution([0.3], _TOWER_SHAPE)], rigid_body_frequency_hz=value
            )
