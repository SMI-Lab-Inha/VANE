# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.campbell.excitation — nP lines and resonance detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vane.campbell.builder import CampbellDiagram
from vane.campbell.excitation import (
    ResonanceSeverity,
    excitation_frequencies,
    find_resonances,
)
from vane.config.dof_map import DofCategory
from vane.modal.identifier import ModeTrack
from vane.modal.labeler import ModeLabel


def _diagram(
    rpm: list[float],
    frequencies: list[float],
    *,
    parameter_name: str = "rotor_speed_rpm",
    damping: list[float] | None = None,
) -> CampbellDiagram:
    """Build a one-track diagram over the given rotor speeds and frequencies."""
    track = ModeTrack(
        operating_points=list(range(len(rpm))),
        mode_indices=[0] * len(rpm),
        natural_frequencies_hz=frequencies,
        damping_ratios=damping if damping is not None else [0.02] * len(rpm),
        label=ModeLabel(DofCategory.TOWER_FORE_AFT_1, "1st Tower Fore-Aft", 1.0, []),
    )
    return CampbellDiagram(
        parameter_values=np.array(rpm, dtype=np.float64),
        parameter_name=parameter_name,
        tracks=[track],
        n_operating_points=len(rpm),
    )


class TestExcitationFrequencies:
    """Tests for excitation_frequencies."""

    def test_np_line_is_n_times_per_rev(self) -> None:
        """The nP line frequency is n * rpm / 60 Hz."""
        lines = excitation_frequencies(np.array([0.0, 60.0]), [1, 3])
        np.testing.assert_allclose(lines[1], [0.0, 1.0])
        np.testing.assert_allclose(lines[3], [0.0, 3.0])


class TestHarmonicValidation:
    """Excitation harmonics must be positive, unique integers."""

    def test_non_positive_harmonic_raises(self) -> None:
        """A zero or negative harmonic is rejected."""
        with pytest.raises(ValueError, match="positive integers"):
            excitation_frequencies(np.array([10.0]), [0])

    def test_fractional_harmonic_raises(self) -> None:
        """A fractional harmonic is rejected."""
        fractional: Any = [1.5]  # invalid input under test, typed Any for both mypys
        with pytest.raises(ValueError, match="positive integers"):
            excitation_frequencies(np.array([10.0]), fractional)

    def test_duplicate_harmonics_raise(self) -> None:
        """Duplicate harmonics are rejected."""
        with pytest.raises(ValueError, match="unique"):
            excitation_frequencies(np.array([10.0]), [3, 3])

    def test_empty_harmonics_allowed(self) -> None:
        """An empty harmonic set is a valid no-op."""
        assert excitation_frequencies(np.array([10.0]), []) == {}

    def test_find_resonances_validates_harmonics(self) -> None:
        """find_resonances rejects invalid harmonics too."""
        diagram = _diagram([0.0, 60.0], [1.5, 1.5])
        with pytest.raises(ValueError, match="positive integers"):
            find_resonances(diagram, [-1])


class TestFindResonances:
    """Tests for find_resonances."""

    def test_detects_interpolated_crossing(self) -> None:
        """A 1.5 Hz mode crosses the 3P line at 30 rpm."""
        diagram = _diagram([0.0, 60.0], [1.5, 1.5])
        crossings = find_resonances(diagram, [3])
        assert len(crossings) == 1
        assert crossings[0].harmonic == 3
        assert crossings[0].rotor_speed_rpm == pytest.approx(30.0)
        assert crossings[0].frequency_hz == pytest.approx(1.5)

    def test_no_crossing_when_lines_do_not_meet(self) -> None:
        """A high mode that never meets the 1P line has no crossing."""
        diagram = _diagram([0.0, 12.0], [5.0, 5.1])
        assert find_resonances(diagram, [1]) == []

    def test_non_monotone_rpm_is_sorted_internally(self) -> None:
        """An unsorted rotor-speed sweep is handled and crossings still match."""
        # Same samples as the sorted [0, 60] -> [1.5, 1.5] case, supplied out of order.
        diagram = _diagram([60.0, 0.0], [1.5, 1.5])
        crossings = find_resonances(diagram, [3])
        assert len(crossings) == 1
        assert crossings[0].rotor_speed_rpm == pytest.approx(30.0)
        assert crossings[0].frequency_hz == pytest.approx(1.5)

    def test_duplicate_rpm_is_deduplicated(self) -> None:
        """A repeated rotor speed does not create a zero-width or spurious crossing."""
        # Two samples at 0 rpm (a tie) plus a sample at 60 rpm; one real 3P crossing.
        diagram = _diagram([0.0, 0.0, 60.0], [1.5, 1.5, 1.5])
        crossings = find_resonances(diagram, [3])
        assert len(crossings) == 1
        assert crossings[0].rotor_speed_rpm == pytest.approx(30.0)

    def test_all_tied_rpm_yields_no_crossing(self) -> None:
        """When every sample shares one rotor speed, the track collapses to a point."""
        diagram = _diagram([30.0, 30.0], [1.5, 1.5])
        assert find_resonances(diagram, [3]) == []

    def test_multiple_harmonics(self) -> None:
        """A rising mode crosses several nP lines."""
        diagram = _diagram([0.0, 120.0], [0.0, 4.0])
        crossings = find_resonances(diagram, [1, 3])
        harmonics = {crossing.harmonic for crossing in crossings}
        assert harmonics == {1, 3}

    def test_single_point_track_has_no_crossing(self) -> None:
        """A track with one operating point cannot produce a crossing."""
        diagram = _diagram([30.0], [1.5])
        assert find_resonances(diagram, [3]) == []

    def test_empty_harmonics(self) -> None:
        """No harmonics yields no crossings."""
        diagram = _diagram([0.0, 60.0], [1.5, 1.5])
        assert find_resonances(diagram, []) == []

    def test_requires_rpm_parameter(self) -> None:
        """A non-rotor-speed diagram raises ValueError."""
        diagram = _diagram([8.0, 10.0], [1.5, 1.6], parameter_name="wind_speed")
        with pytest.raises(ValueError, match="rotor-speed"):
            find_resonances(diagram, [3])


class TestResonanceSeverity:
    """Damping-aware severity and operating-window filtering."""

    def test_low_damping_is_high_severity(self) -> None:
        """A lightly damped mode at a crossing is high severity."""
        diagram = _diagram([0.0, 60.0], [1.5, 1.5], damping=[0.002, 0.002])
        crossing = find_resonances(diagram, [3])[0]
        assert crossing.severity is ResonanceSeverity.HIGH
        assert crossing.damping_ratio == pytest.approx(0.002)

    def test_high_damping_is_low_severity(self) -> None:
        """A well-damped mode at a crossing is low severity."""
        diagram = _diagram([0.0, 60.0], [1.5, 1.5], damping=[0.1, 0.1])
        crossing = find_resonances(diagram, [3])[0]
        assert crossing.severity is ResonanceSeverity.LOW

    def test_damping_is_interpolated_at_crossing(self) -> None:
        """The reported damping is interpolated to the crossing rotor speed."""
        # 1.5 Hz crosses the 3P line at 30 rpm, halfway along [0, 60].
        diagram = _diagram([0.0, 60.0], [1.5, 1.5], damping=[0.01, 0.03])
        crossing = find_resonances(diagram, [3])[0]
        assert crossing.rotor_speed_rpm == pytest.approx(30.0)
        assert crossing.damping_ratio == pytest.approx(0.02)  # midpoint
        assert crossing.severity is ResonanceSeverity.MEDIUM

    def test_operating_range_filters_crossings(self) -> None:
        """Crossings outside the operating window are excluded."""
        # A flat 3 Hz mode crosses 3P at 60 rpm and 1P at 180 rpm.
        diagram = _diagram([0.0, 240.0], [3.0, 3.0])
        all_crossings = find_resonances(diagram, [1, 3])
        within = find_resonances(diagram, [1, 3], operating_range=(0.0, 100.0))
        assert {round(c.rotor_speed_rpm) for c in all_crossings} == {60, 180}
        assert len(within) == 1
        assert within[0].rotor_speed_rpm == pytest.approx(60.0)

    def test_invalid_operating_range_raises(self) -> None:
        """An inverted operating window is rejected."""
        diagram = _diagram([0.0, 60.0], [1.5, 1.5])
        with pytest.raises(ValueError, match="min exceeds max"):
            find_resonances(diagram, [3], operating_range=(50.0, 10.0))

    def test_non_finite_operating_range_raises(self) -> None:
        """A NaN or infinite operating window is rejected."""
        diagram = _diagram([0.0, 60.0], [1.5, 1.5])
        with pytest.raises(ValueError, match="finite"):
            find_resonances(diagram, [3], operating_range=(0.0, float("nan")))
        with pytest.raises(ValueError, match="finite"):
            find_resonances(diagram, [3], operating_range=(float("-inf"), 60.0))
