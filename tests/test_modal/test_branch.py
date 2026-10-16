# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.branch — crossing/veering branch-review detection."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from vane.config.dof_map import DofCategory
from vane.modal.branch import (
    BranchGroup,
    BranchKind,
    BranchSubWindow,
    _second_derivative,
    attach_branch_groups,
    detect_branch_groups,
    resolve_branch_groups,
)
from vane.modal.eigensolver import ModalSolution
from vane.modal.identifier import IdentificationResult, ModeTrack, identify_modes
from vane.modal.labeler import ModeLabel


def _sol(frequencies: list[float], shapes: list[list[float]]) -> ModalSolution:
    """Build a ModalSolution from frequencies and a (n_dof, n_modes) shape matrix."""
    freq = np.asarray(frequencies, dtype=np.float64)
    mode_shapes = np.asarray(shapes, dtype=np.float64)
    descriptions = [f"DOF {i}" for i in range(mode_shapes.shape[0])]
    return ModalSolution(
        eigenvalues=(-0.01 + 1j * freq).astype(np.complex128),
        natural_frequencies_hz=freq,
        damping_ratios=np.full(freq.size, 0.01),
        damped_frequencies_hz=freq,
        mode_shapes=mode_shapes.astype(np.complex128),
        full_eigenvectors=mode_shapes.astype(np.complex128),
        n_rigid_body_modes=0,
        dof_descriptions=descriptions,
    )


def _track(ops: list[int], modes: list[int], freqs: list[float]) -> ModeTrack:
    """Build a ModeTrack spanning the given operating points."""
    return ModeTrack(
        operating_points=list(ops),
        mode_indices=list(modes),
        natural_frequencies_hz=[float(f) for f in freqs],
        damping_ratios=[0.01] * len(ops),
        label=ModeLabel(DofCategory.UNKNOWN, "mode", 1.0, []),
    )


# Distinct, mixed-at-crossing, and blended shape matrices used by the scenarios.
_DISTINCT = [[1.0, 0.0], [0.0, 1.0]]
_MIXED = [[1.0, 0.9], [0.9, 1.0]]
_BLEND = [[1.0, 0.6], [0.6, 1.0]]


def _crossing_solutions() -> list[ModalSolution]:
    """Two modes whose frequencies swap, with shapes ambiguous at the crossing."""
    return [
        _sol([1.0, 2.0], _DISTINCT),
        _sol([1.5, 1.5], _MIXED),
        _sol([2.0, 1.0], _DISTINCT),
    ]


def _veering_solutions() -> list[ModalSolution]:
    """Two modes that approach (min gap 0.10) then separate without swapping."""
    return [
        _sol([1.0, 1.6], _DISTINCT),
        _sol([1.25, 1.35], _BLEND),
        _sol([1.5, 1.7], _DISTINCT),
    ]


def _bouncing_pair() -> list[ModalSolution]:
    """Return a contested pair whose shape-following base bounces (peak/valley).

    The frequencies touch at op 1 and the shapes compete, so the base tracks turn
    back (peak [1, 1.5, 1] and valley [2, 1.5, 2]); the smooth (minimum-curvature)
    alternative is the straight crossing [1, 1.5, 2] and [2, 1.5, 1].
    """
    return [
        _sol([1.0, 2.0], _DISTINCT),
        _sol([1.5, 1.5], [[1.0, 0.7], [0.7, 1.0]]),
        _sol([1.0, 2.0], _DISTINCT),
    ]


class TestThresholdValidation:
    """detect_branch_groups validates its thresholds."""

    def test_mac_competition_out_of_range_raises(self) -> None:
        """A cross-MAC threshold outside [0, 1] is rejected."""
        result = identify_modes(_crossing_solutions(), strategy="simultaneous")
        with pytest.raises(ValueError, match="mac_competition"):
            detect_branch_groups(result, _crossing_solutions(), mac_competition=1.5)

    def test_negative_tolerance_raises(self) -> None:
        """A negative veering tolerance is rejected."""
        result = identify_modes(_crossing_solutions(), strategy="simultaneous")
        with pytest.raises(ValueError, match="veering_tol_hz"):
            detect_branch_groups(result, _crossing_solutions(), veering_tol_hz=-0.1)


class TestDetectionScenarios:
    """The detector catches genuine hotspots and ignores coincidental near-misses."""

    def test_crossing_is_detected(self) -> None:
        """Swapping frequencies with ambiguous shapes form one crossing group."""
        solutions = _crossing_solutions()
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.2)
        assert len(groups) == 1
        group = groups[0]
        assert group.kind is BranchKind.CROSSING
        assert group.track_indices == (0, 1)
        assert group.op_lo == 0
        assert group.op_hi == 2
        assert group.min_separation_op == 1
        assert group.min_separation_hz == pytest.approx(0.0, abs=1e-9)
        assert not group.is_dense

    def test_veering_is_detected(self) -> None:
        """Approaching-then-separating modes form one veering group."""
        solutions = _veering_solutions()
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.15)
        assert len(groups) == 1
        group = groups[0]
        assert group.kind is BranchKind.VEERING
        assert group.min_separation_op == 1
        assert group.min_separation_hz == pytest.approx(0.10, abs=1e-9)
        assert all(not sub.truncated for sub in group.sub_windows)

    def test_orthogonal_close_pass_is_not_flagged(self) -> None:
        """Frequencies that pass close with orthogonal shapes are not contested."""
        solutions = [_sol([1.0, 1.1], _DISTINCT), _sol([1.1, 1.0], _DISTINCT)]
        result = identify_modes(solutions, strategy="simultaneous")
        assert detect_branch_groups(result, solutions, veering_tol_hz=0.2) == []


class TestContestedPredicate:
    """Both frequency proximity and mode-shape competition are required."""

    def test_competition_without_proximity_is_not_flagged(self) -> None:
        """A tolerance below the approach yields no group despite competition."""
        solutions = _veering_solutions()
        result = identify_modes(solutions, strategy="simultaneous")
        # Min approach is 0.10; a tighter tolerance fails the proximity test.
        assert detect_branch_groups(result, solutions, veering_tol_hz=0.05) == []

    def test_proximity_without_competition_is_not_flagged(self) -> None:
        """Close frequencies with orthogonal (non-competing) shapes are not flagged."""
        solutions = [_sol([1.0, 1.05], _DISTINCT), _sol([1.0, 1.05], _DISTINCT)]
        result = identify_modes(solutions, strategy="simultaneous")
        assert detect_branch_groups(result, solutions, veering_tol_hz=0.2) == []


class TestClassification:
    """Crossing versus veering classification and window truncation."""

    def test_truncated_veering_when_min_at_window_end(self) -> None:
        """A monotone approach whose closest point is at the window end is truncated."""
        solutions = [
            _sol([1.0, 1.3], _DISTINCT),
            _sol([1.0, 1.15], [[1.0, 0.5], [0.5, 1.0]]),
            _sol([1.0, 1.05], [[1.0, 0.7], [0.7, 1.0]]),
        ]
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.2)
        assert len(groups) == 1
        group = groups[0]
        assert group.kind is BranchKind.VEERING
        assert group.min_separation_op == 2
        assert any(sub.truncated for sub in group.sub_windows)

    def test_endpoint_tie_is_truncated_veering_not_crossing(self) -> None:
        """An exact frequency tie at a window endpoint is a truncated veering.

        The frequencies are equal at op 0 then separate without swapping, so the order
        never crosses within the window; it must not be reported as a crossing.
        """
        solutions = [
            _sol([1.0, 1.0], [[1.0, 0.6], [0.6, 1.0]]),
            _sol([1.0, 1.2], [[1.0, 0.3], [0.3, 1.0]]),
            _sol([1.0, 1.4], _DISTINCT),
        ]
        ops = [0, 1, 2]
        tracks = [
            _track(ops, [0, 0, 0], [1.0, 1.0, 1.0]),
            _track(ops, [1, 1, 1], [1.0, 1.2, 1.4]),
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=3)
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.25)
        assert len(groups) == 1
        assert groups[0].kind is BranchKind.VEERING
        assert any(sub.truncated for sub in groups[0].sub_windows)


class TestWindowingAndGrouping:
    """Window intersection, gaps, and connected-component grouping."""

    def test_gap_breaks_shared_window(self) -> None:
        """Tracks sharing only non-adjacent operating points have no transition."""
        tracks = [
            _track([0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0]),
            _track([0, 2], [1, 1], [1.02, 1.02]),
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=3)
        solutions = [_sol([1.0, 1.02], _MIXED) for _ in range(3)]
        assert detect_branch_groups(result, solutions, veering_tol_hz=0.2) == []

    def test_window_is_local_not_the_whole_coexistence(self) -> None:
        """A single approach in a long coexistence yields a window around it.

        The two tracks span all five operating points but only approach near the
        middle, so the contested window must be local (op 1..3), not the whole sweep.
        """
        near = [[1.0, 0.6], [0.6, 1.0]]
        solutions = [
            _sol([1.0, 2.0], _DISTINCT),
            _sol([1.0, 1.5], _DISTINCT),
            _sol([1.0, 1.05], near),
            _sol([1.0, 1.5], _DISTINCT),
            _sol([1.0, 2.0], _DISTINCT),
        ]
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.2)
        assert len(groups) == 1
        group = groups[0]
        assert group.op_lo == 1
        assert group.op_hi == 3
        assert group.min_separation_op == 2

    def test_adjacent_approaches_merge_into_one_window(self) -> None:
        """Two approaches separated by a single point merge into one window.

        Gaps of [0.05, 0.30, 0.05]: each close point's one-point shoulders overlap, so
        the two bands merge into the contiguous window [0, 2].
        """
        near = [[1.0, 0.6], [0.6, 1.0]]
        solutions = [
            _sol([1.0, 1.05], near),
            _sol([1.0, 1.30], _DISTINCT),
            _sol([1.0, 1.05], near),
        ]
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.2)
        assert len(groups) == 1
        assert groups[0].op_lo == 0
        assert groups[0].op_hi == 2

    def test_window_is_the_shared_intersection(self) -> None:
        """A shorter track limits the contested window to the shared range."""
        tracks = [
            _track([0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0]),
            _track([0, 1], [1, 1], [1.02, 1.02]),
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=3)
        solutions = [_sol([1.0, 1.02], _MIXED) for _ in range(3)]
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.2)
        assert len(groups) == 1
        assert groups[0].op_lo == 0
        assert groups[0].op_hi == 1

    def test_temporally_separate_hotspots_stay_separate(self) -> None:
        """A track in two distant hotspots gives two groups, not one dense span.

        Track B approaches A early (ops 0-2) and C late (ops 4-6); since the two
        contested windows do not overlap, they must not merge into one dense group.
        """
        ab = [[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]]
        bc = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.6], [0.0, 0.6, 1.0]]
        solutions = [
            _sol([1.0, 1.05, 4.0], ab),
            _sol([1.0, 1.05, 4.0], ab),
            _sol([1.0, 1.5, 4.0], bc),
            _sol([1.0, 2.0, 4.0], bc),
            _sol([1.0, 2.5, 3.5], bc),
            _sol([1.0, 2.95, 3.0], bc),
            _sol([1.0, 2.95, 3.0], bc),
        ]
        ops = list(range(7))
        tracks = [
            _track(ops, [0] * 7, [1.0] * 7),
            _track(ops, [1] * 7, [1.05, 1.05, 1.5, 2.0, 2.5, 2.95, 2.95]),
            _track(ops, [2] * 7, [4.0, 4.0, 4.0, 4.0, 3.5, 3.0, 3.0]),
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=7)
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.2)
        assert len(groups) == 2
        assert groups[0].track_indices == (0, 1)
        assert (groups[0].op_lo, groups[0].op_hi) == (0, 2)
        assert groups[1].track_indices == (1, 2)
        assert (groups[1].op_lo, groups[1].op_hi) == (4, 6)
        assert not any(group.is_dense for group in groups)

    def test_chained_pairs_form_one_dense_group(self) -> None:
        """A-B and B-C contested (A-C not) collapse into one dense group."""
        overlap = [[1.0, 0.6, 0.0], [0.6, 1.0, 0.6], [0.0, 0.6, 1.0]]
        solutions = [_sol([1.0, 1.05, 1.10], overlap) for _ in range(3)]
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.07)
        assert len(groups) == 1
        assert groups[0].track_indices == (0, 1, 2)
        assert groups[0].is_dense

    def test_disjoint_pairs_form_separate_groups(self) -> None:
        """Two crossings in different frequency bands are two distinct groups."""
        distinct = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        mixed = [
            [1.0, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.9],
            [0.0, 0.0, 0.9, 1.0],
        ]
        solutions = [
            _sol([1.0, 2.0, 5.0, 6.0], distinct),
            _sol([1.5, 1.5, 5.5, 5.5], mixed),
            _sol([2.0, 1.0, 6.0, 5.0], distinct),
        ]
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.3)
        assert len(groups) == 2
        # Canonically ordered by window then frequency: low band before high band.
        assert groups[0].track_indices == (0, 1)
        assert groups[1].track_indices == (2, 3)
        assert groups[0].frequency_band_hz[1] < groups[1].frequency_band_hz[0]


class TestAutoTolerance:
    """The auto-scaled default tolerance is deterministic and guarded."""

    def test_auto_tolerance_scales_with_spectrum(self) -> None:
        """The crossing is caught with the default (auto) tolerance."""
        solutions = _crossing_solutions()
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions)
        assert len(groups) == 1
        assert groups[0].kind is BranchKind.CROSSING

    def test_constant_frequencies_do_not_crash(self) -> None:
        """All-equal spectra (zero spacing) are handled without dividing by zero."""
        solutions = [_sol([1.0, 1.0], _MIXED), _sol([1.0, 1.0], _MIXED)]
        result = identify_modes(solutions, strategy="simultaneous")
        # Must not raise; equal frequencies are within a zero tolerance.
        detect_branch_groups(result, solutions)

    def test_single_mode_solutions_have_no_spacing(self) -> None:
        """With one mode per operating point the auto tolerance has no spacing."""
        solutions = [_sol([1.0], [[1.0], [0.0]]), _sol([1.0], [[0.0], [1.0]])]
        result = identify_modes(solutions, strategy="simultaneous")
        # Two unlinked single-mode tracks; auto-tolerance falls back without error.
        assert detect_branch_groups(result, solutions) == []


class TestDegenerateInputs:
    """Non-finite frequencies are skipped rather than producing garbage."""

    def test_nan_frequency_pair_is_skipped(self) -> None:
        """A NaN frequency in a window drops that pair without raising."""
        tracks = [
            _track([0, 1], [0, 0], [1.0, float("nan")]),
            _track([0, 1], [1, 1], [1.02, 1.02]),
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=2)
        solutions = [_sol([1.0, 1.02], _MIXED), _sol([1.0, 1.02], _MIXED)]
        assert detect_branch_groups(result, solutions, veering_tol_hz=0.2) == []

    def test_finite_approach_after_nan_is_still_detected(self) -> None:
        """A NaN at one operating point drops that sample, not the whole pair.

        The tracks have a NaN at op 1 but a genuine finite approach at ops 2-3, which
        must still be flagged from the finite segment.
        """
        nan = float("nan")
        near = [[1.0, 0.6], [0.6, 1.0]]
        solutions = [
            _sol([1.0, 2.0], _DISTINCT),
            _sol([1.0, 2.0], _DISTINCT),
            _sol([1.0, 1.05], near),
            _sol([1.0, 1.05], near),
        ]
        ops = [0, 1, 2, 3]
        tracks = [
            _track(ops, [0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0]),
            _track(ops, [1, 1, 1, 1], [2.0, nan, 1.05, 1.05]),
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=4)
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.2)
        assert len(groups) == 1
        assert (groups[0].op_lo, groups[0].op_hi) == (2, 3)

    def test_nan_frequency_does_not_poison_auto_tolerance(self) -> None:
        """A NaN in one mode must not disable detection of an unrelated crossing.

        With the default (auto) tolerance, a single non-finite frequency would make
        the median spacing NaN and silently suppress every group; it must be filtered.
        """
        nan = float("nan")
        ab = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ab_mixed = [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]]
        solutions = [
            _sol([1.0, 2.0, 5.0], ab),
            _sol([1.5, 1.5, nan], ab_mixed),
            _sol([2.0, 1.0, 5.0], ab),
        ]
        ops = [0, 1, 2]
        tracks = [
            _track(ops, [0, 0, 0], [1.0, 1.5, 2.0]),
            _track(ops, [1, 1, 1], [2.0, 1.5, 1.0]),
            _track(ops, [2, 2, 2], [5.0, nan, 5.0]),
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=3)
        groups = detect_branch_groups(result, solutions)  # default tolerance
        assert len(groups) == 1
        assert groups[0].kind is BranchKind.CROSSING
        assert groups[0].track_indices == (0, 1)


class TestEmptyAndSingle:
    """Degenerate sweep sizes produce no branch groups."""

    def test_empty_result(self) -> None:
        """No tracks yields no groups."""
        result = IdentificationResult(tracks=[], n_operating_points=0)
        assert detect_branch_groups(result, []) == []

    def test_single_operating_point(self) -> None:
        """A single operating point has no transition to contest."""
        solutions = [_sol([1.0, 2.0], _DISTINCT)]
        result = identify_modes(solutions, strategy="simultaneous")
        assert detect_branch_groups(result, solutions) == []

    def test_single_track(self) -> None:
        """One track cannot be contested."""
        solutions = [_sol([1.0], [[1.0]]), _sol([1.1], [[1.0]])]
        result = identify_modes(solutions, strategy="simultaneous")
        assert detect_branch_groups(result, solutions) == []


class TestDeterminismAndPurity:
    """Detection is pure and reproducible."""

    def test_repeated_calls_are_identical(self) -> None:
        """The same inputs always give the same groups."""
        solutions = _crossing_solutions()
        result = identify_modes(solutions, strategy="simultaneous")
        first = detect_branch_groups(result, solutions, veering_tol_hz=0.2)
        second = detect_branch_groups(result, solutions, veering_tol_hz=0.2)
        assert first == second

    def test_detection_does_not_mutate_the_result(self) -> None:
        """detect_branch_groups leaves the result's tracks and groups untouched."""
        solutions = _crossing_solutions()
        result = identify_modes(solutions, strategy="simultaneous")
        tracks_before = copy.deepcopy(result.tracks)
        detect_branch_groups(result, solutions, veering_tol_hz=0.2)
        assert result.branch_groups == []
        assert result.tracks == tracks_before


class TestAttachBranchGroups:
    """attach_branch_groups returns a populated copy without mutating the input."""

    def test_attach_populates_a_new_result(self) -> None:
        """The returned result carries the groups; the original stays empty."""
        solutions = _crossing_solutions()
        result = identify_modes(solutions, strategy="simultaneous")
        attached = attach_branch_groups(result, solutions, veering_tol_hz=0.2)
        assert attached is not result
        assert len(attached.branch_groups) == 1
        assert result.branch_groups == []
        assert attached.tracks is result.tracks


class TestResolution:
    """The smooth (minimum-curvature) alternative is opt-in and review-only."""

    def test_detection_alone_does_not_resolve(self) -> None:
        """Branch review without resolve leaves smooth_branches unset."""
        solutions = _bouncing_pair()
        result = identify_modes(solutions, strategy="simultaneous", branch_review=True)
        group = result.branch_groups[0]
        assert not group.resolved
        assert group.smooth_branches is None

    def test_resolve_attaches_the_smooth_alternative(self) -> None:
        """A bouncing base resolves to the straight crossing branches."""
        solutions = _bouncing_pair()
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.6)
        resolved = resolve_branch_groups(groups, result)
        assert resolved[0].resolved
        assert resolved[0].smooth_branches == ((1.0, 1.5, 2.0), (2.0, 1.5, 1.0))

    def test_smooth_differs_from_bouncing_base(self) -> None:
        """At a veering the smooth branches differ from the shape-following base."""
        solutions = _bouncing_pair()
        result = identify_modes(solutions, strategy="simultaneous", resolve=True)
        base = {tuple(track.natural_frequencies_hz) for track in result.tracks}
        assert base == {(1.0, 1.5, 1.0), (2.0, 1.5, 2.0)}  # peak and valley
        smooth = result.branch_groups[0].smooth_branches
        assert smooth is not None
        assert set(smooth) != base

    def test_smooth_matches_base_at_a_clean_crossing(self) -> None:
        """When the base already crosses straight, the smooth branches match it."""
        solutions = _crossing_solutions()
        result = identify_modes(solutions, strategy="simultaneous", resolve=True)
        base = {tuple(track.natural_frequencies_hz) for track in result.tracks}
        smooth = result.branch_groups[0].smooth_branches
        assert smooth is not None
        assert set(smooth) == base

    def test_dense_group_is_resolved_up_to_the_cap(self) -> None:
        """A three-track (dense) cluster is resolved into three smooth branches."""
        overlap = [[1.0, 0.6, 0.0], [0.6, 1.0, 0.6], [0.0, 0.6, 1.0]]
        solutions = [_sol([1.0, 1.05, 1.10], overlap) for _ in range(3)]
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.07)
        group = resolve_branch_groups(groups, result)[0]
        assert group.is_dense
        assert group.resolved
        assert group.smooth_branches is not None
        assert len(group.smooth_branches) == 3

    def test_dense_smooth_redraws_to_straight_branches(self) -> None:
        """A three-track cluster that bounces resolves to straight/flat branches."""
        tracks = [
            _track([0, 1, 2], [0, 0, 0], [0.0, 1.0, 0.0]),  # peak
            _track([0, 1, 2], [1, 1, 1], [2.0, 1.0, 2.0]),  # valley
            _track([0, 1, 2], [2, 2, 2], [1.0, 1.0, 1.0]),  # flat
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=3)
        group = BranchGroup(
            track_indices=(0, 1, 2),
            op_lo=0,
            op_hi=2,
            frequency_band_hz=(0.0, 2.0),
            kind=BranchKind.VEERING,
            min_separation_hz=0.0,
            min_separation_op=1,
            sub_windows=(BranchSubWindow(0, 2, BranchKind.VEERING, False),),
            is_dense=True,
        )
        resolved = resolve_branch_groups(
            [group], result, parameter_values=[0.0, 1.0, 2.0]
        )[0]
        assert resolved.smooth_branches == (
            (0.0, 1.0, 2.0),
            (1.0, 1.0, 1.0),
            (2.0, 1.0, 0.0),
        )

    def test_group_over_the_branch_cap_is_not_resolved(self) -> None:
        """A cluster of more than four tracks is surfaced but not resolved."""
        tracks = [_track([0, 1, 2], [i, i, i], [1.0 + 0.05 * i] * 3) for i in range(5)]
        result = IdentificationResult(tracks=tracks, n_operating_points=3)
        group = BranchGroup(
            track_indices=(0, 1, 2, 3, 4),
            op_lo=0,
            op_hi=2,
            frequency_band_hz=(1.0, 1.2),
            kind=BranchKind.VEERING,
            min_separation_hz=0.05,
            min_separation_op=0,
            sub_windows=(BranchSubWindow(0, 2, BranchKind.VEERING, False),),
            is_dense=True,
        )
        resolved = resolve_branch_groups([group], result)[0]
        assert not resolved.resolved
        assert resolved.smooth_branches is None

    def test_short_window_is_not_resolved(self) -> None:
        """A two-point window has no curvature, so it is not resolved."""
        solutions = [_sol([1.0, 2.0], _DISTINCT), _sol([1.1, 1.9], _MIXED)]
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=1.0)
        resolved = resolve_branch_groups(groups, result)
        assert resolved
        assert all(not group.resolved for group in resolved)

    def test_resolution_is_idempotent(self) -> None:
        """Resolving an already-resolved set of groups changes nothing."""
        solutions = _bouncing_pair()
        result = identify_modes(solutions, strategy="simultaneous")
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.6)
        once = resolve_branch_groups(groups, result)
        twice = resolve_branch_groups(once, result)
        assert once == twice

    def test_resolution_does_not_mutate_tracks(self) -> None:
        """Resolution attaches metadata only; the tracks are untouched."""
        solutions = _bouncing_pair()
        result = identify_modes(solutions, strategy="simultaneous")
        before = copy.deepcopy(result.tracks)
        groups = detect_branch_groups(result, solutions, veering_tol_hz=0.6)
        resolve_branch_groups(groups, result)
        assert result.tracks == before

    def test_ragged_membership_is_not_resolved(self) -> None:
        """A group whose hull includes an op with an absent member is not resolved."""
        tracks = [
            _track([0, 1, 2], [0, 0, 0], [1.0, 1.0, 1.0]),
            _track([0, 2], [1, 1], [1.05, 1.05]),  # absent at op 1
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=3)
        resolved = resolve_branch_groups([_two_track_group()], result)
        assert not resolved[0].resolved
        assert resolved[0].smooth_branches is None

    def test_non_finite_frequency_window_is_not_resolved(self) -> None:
        """A non-finite frequency inside the window leaves the group unresolved."""
        tracks = [
            _track([0, 1, 2], [0, 0, 0], [1.0, float("nan"), 1.0]),
            _track([0, 1, 2], [1, 1, 1], [2.0, 1.5, 2.0]),
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=3)
        resolved = resolve_branch_groups([_two_track_group()], result)
        assert not resolved[0].resolved


class TestResolutionCurvature:
    """The smooth assignment is parameter-aware and tolerates a bad axis."""

    def test_second_derivative_of_a_line_is_zero(self) -> None:
        """A straight line has zero curvature on any (non-uniform) grid."""
        # f(x) = 3x + 1 sampled at x = 0, 1, 3.
        value = _second_derivative(1.0, 4.0, 10.0, 1.0, 2.0)
        assert value == pytest.approx(0.0, abs=1e-9)

    def test_second_derivative_of_a_parabola_is_two(self) -> None:
        """f(x) = x^2 has constant second derivative 2 on a non-uniform grid."""
        # x = 0, 1, 3 -> f = 0, 1, 9.
        value = _second_derivative(0.0, 1.0, 9.0, 1.0, 2.0)
        assert value == pytest.approx(2.0, abs=1e-9)

    def test_uniform_parameter_matches_the_index_fallback(self) -> None:
        """Uniformly spaced parameter values reproduce the default (index) result."""
        result, group = _crossing_group()
        default = resolve_branch_groups([group], result)[0]
        uniform = resolve_branch_groups(
            [group], result, parameter_values=[0.0, 1.0, 2.0]
        )[0]
        assert default.smooth_branches == uniform.smooth_branches

    def test_non_monotone_parameter_falls_back_to_index(self) -> None:
        """A non-increasing parameter axis is ignored (would give zero/neg spacing)."""
        result, group = _crossing_group()
        default = resolve_branch_groups([group], result)[0]
        bad = resolve_branch_groups([group], result, parameter_values=[0.0, 0.0, 1.0])[
            0
        ]
        assert bad.smooth_branches == default.smooth_branches

    def test_wrong_length_parameter_falls_back_to_index(self) -> None:
        """A parameter axis of the wrong length is ignored."""
        result, group = _crossing_group()
        default = resolve_branch_groups([group], result)[0]
        bad = resolve_branch_groups([group], result, parameter_values=[0.0, 1.0])[0]
        assert bad.smooth_branches == default.smooth_branches

    def test_non_finite_parameter_falls_back_to_index(self) -> None:
        """A non-finite parameter axis is ignored (would give non-finite spacing)."""
        result, group = _crossing_group()
        default = resolve_branch_groups([group], result)[0]
        bad = resolve_branch_groups(
            [group], result, parameter_values=[0.0, 1.0, float("inf")]
        )[0]
        assert bad.smooth_branches == default.smooth_branches

    def test_parameter_spacing_changes_the_smooth_branches(self) -> None:
        """Non-uniform spacing can flip the minimum-curvature assignment.

        Two modes cross over four points; the straight and swapped assignments tie on
        a uniform grid (straight wins the tie) but the swapped one is far smoother when
        the two middle points are squeezed together.
        """
        tracks = [
            _track([0, 1, 2, 3], [0, 0, 0, 0], [0.0, 1.0, 3.0, 4.0]),
            _track([0, 1, 2, 3], [1, 1, 1, 1], [4.0, 3.0, 1.0, 0.0]),
        ]
        result = IdentificationResult(tracks=tracks, n_operating_points=4)
        group = BranchGroup(
            track_indices=(0, 1),
            op_lo=0,
            op_hi=3,
            frequency_band_hz=(0.0, 4.0),
            kind=BranchKind.CROSSING,
            min_separation_hz=0.0,
            min_separation_op=1,
            sub_windows=(BranchSubWindow(0, 3, BranchKind.CROSSING, False),),
            is_dense=False,
        )
        uniform = resolve_branch_groups(
            [group], result, parameter_values=[0.0, 1.0, 2.0, 3.0]
        )[0]
        squeezed = resolve_branch_groups(
            [group], result, parameter_values=[0.0, 1.0, 1.1, 2.1]
        )[0]
        assert uniform.smooth_branches == (
            (0.0, 1.0, 3.0, 4.0),
            (4.0, 3.0, 1.0, 0.0),
        )
        assert squeezed.smooth_branches == (
            (0.0, 1.0, 1.0, 0.0),
            (4.0, 3.0, 3.0, 4.0),
        )
        assert uniform.smooth_branches != squeezed.smooth_branches


def _two_track_group() -> BranchGroup:
    """Return a plain two-track group over the window [0, 2] for resolution tests."""
    return BranchGroup(
        track_indices=(0, 1),
        op_lo=0,
        op_hi=2,
        frequency_band_hz=(1.0, 2.0),
        kind=BranchKind.VEERING,
        min_separation_hz=0.5,
        min_separation_op=0,
        sub_windows=(BranchSubWindow(0, 2, BranchKind.VEERING, False),),
        is_dense=False,
    )


def _crossing_group() -> tuple[IdentificationResult, BranchGroup]:
    """Return a resolvable crossing result and its (unresolved) group."""
    solutions = _crossing_solutions()
    result = identify_modes(solutions, strategy="simultaneous")
    groups = detect_branch_groups(result, solutions, veering_tol_hz=0.6)
    return result, groups[0]
